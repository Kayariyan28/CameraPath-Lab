"""Geometry backend interface and the fallback ladder.

Every backend answers the same question — given frames, where was the camera? —
and is allowed to answer "I could not tell, and here is why". That second answer
is a first-class result, not an error: for a pure pan there IS no observable
translation, and a backend that always returns a confident pose is lying
(spec §22, §28).

The ladder (spec §28) degrades rather than fails:

    strong geometric reconstruction   (COLMAP)
      -> learned geometry             (VGGT, optional)
      -> 2D + 3D hybrid               (OpenCV essential matrix)
      -> perceptual motion match      (screen-space optimisation)
      -> 2D camera-motion proxy       (direct from the motion signature)

`solve_with_fallback` walks it, records a `SolverDecision` per attempt, and
returns the best honest result. It never raises (invariant I12).
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from app.core.logging import StageReporter, get_logger
from app.models.schemas.motion import CameraIntrinsics, MotionFrame
from app.models.schemas.trajectory import SolverDecision, SolverSource
from app.models.schemas.video import FrameMetadata, Shot, VideoInfo

log = get_logger("solvers.base")


@dataclass
class SolveContext:
    """Everything a backend may need. Passed by value so backends stay pure
    with respect to job state."""

    info: VideoInfo
    shot: Shot
    frames_meta: list[FrameMetadata]
    motion_frames: list[MotionFrame]
    intrinsics: CameraIntrinsics
    analysis_size: tuple[int, int]

    parallax_score: float = 0.0
    texture_score: float = 0.0
    work_dir: Path | None = None
    geometry_long_edge: int = 1600
    max_features: int = 8000
    keyframe_density: float = 1.0
    reporter: StageReporter | None = None

    def report(self, message: str) -> None:
        if self.reporter:
            self.reporter.info(message)
        log.info(message)

    def progress(self, fraction: float, message: str = "") -> None:
        if self.reporter:
            self.reporter.progress(fraction, message)


@dataclass
class GeometryResult:
    """Anchor poses from one backend.

    Anchors are sparse by design — a geometric solve runs on selected keyframes,
    and `trajectory/fusion.py` distributes the dense per-frame motion between
    them. `frame_indices` are absolute source frame indices.
    """

    source: SolverSource
    frame_indices: list[int] = field(default_factory=list)
    positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    quaternions: list[np.ndarray] = field(default_factory=list)
    """Camera-to-world, [w,x,y,z], in CameraPath convention."""

    per_pose_confidence: list[float] = field(default_factory=list)
    focal_pixels: float | None = None
    focal_confidence: float = 0.0

    #: Whether the data actually constrains focal length. Measured, not assumed:
    #: on a pure forward dolly, a 14 deg spread of FOV changed reprojection error
    #: by 0.001 px, and bundle adjustment "refined" the focal to wherever its prior
    #: pointed. When False, `focal_pixels` is the PRIOR the poses were solved
    #: under, and must be presented as such, never as a measurement (I7).
    focal_observable: bool = True
    #: Relative rise in reprojection error when focal is pinned +-15% away from
    #: the solution. The evidence behind `focal_observable`; None if not probed.
    focal_sensitivity: float | None = None
    #: Width in pixels of the images `focal_pixels` refers to. Focal in pixels is
    #: meaningless without it — the solve runs at a different resolution from the
    #: flow analysis, and mixing the two silently shifts the FOV.
    focal_image_width: int | None = None

    # --- quality evidence, consumed by validation/confidence.py ---
    registered_frames: int = 0
    total_frames: int = 0
    mean_reprojection_error: float | None = None
    median_reprojection_error: float | None = None
    track_count: int = 0
    mean_track_length: float = 0.0
    bundle_adjustment_residual: float | None = None

    #: Whether translation is geometrically observable in this result. False for
    #: a rotation-only solve, and it must stay False rather than being silently
    #: upgraded by a nonzero baseline that is actually noise.
    translation_observable: bool = True
    confidence: float = 0.0
    message: str = ""
    succeeded: bool = False

    @property
    def registered_ratio(self) -> float:
        return self.registered_frames / self.total_frames if self.total_frames else 0.0

    @property
    def pose_count(self) -> int:
        return len(self.frame_indices)

    def path_length(self) -> float:
        if len(self.positions) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.positions, axis=0), axis=1).sum())

    def to_decision(self, *, attempted: bool = True, selected: bool = False,
                    duration: float = 0.0) -> SolverDecision:
        return SolverDecision(
            solver=self.source,
            attempted=attempted,
            succeeded=self.succeeded,
            registered_frames=self.registered_frames,
            total_frames=self.total_frames,
            duration_seconds=duration,
            mean_reprojection_error=self.mean_reprojection_error,
            confidence=self.confidence,
            selected=selected,
            message=self.message,
        )


def failed_result(source: SolverSource, message: str, total_frames: int = 0) -> GeometryResult:
    return GeometryResult(
        source=source, succeeded=False, message=message,
        total_frames=total_frames, confidence=0.0,
    )


@runtime_checkable
class GeometryBackend(Protocol):
    """The single interface all backends implement (spec §9)."""

    @property
    def source(self) -> SolverSource: ...

    def available(self) -> tuple[bool, str]:
        """(usable on this machine, reason if not)."""
        ...

    def suitable_for(self, context: SolveContext) -> tuple[bool, str]:
        """(worth attempting on this shot, reason if not).

        Separate from `available` so the UI can distinguish "this backend is not
        installed" from "this backend cannot help with this particular shot".
        """
        ...

    def estimate(self, context: SolveContext) -> GeometryResult: ...


def select_keyframes(
    motion_frames: list[MotionFrame],
    shot: Shot,
    *,
    density: float = 1.0,
    min_count: int = 8,
    max_count: int = 240,
) -> list[int]:
    """Choose frames for the geometric solve, by accumulated motion.

    Uniform temporal sampling is wrong for this: it wastes keyframes on a static
    passage and under-samples a fast one, and the fast passages are exactly where
    the trajectory shape is decided. So keyframes are placed at roughly equal
    *accumulated image motion*, which puts them where the camera actually moved.

    A minimum spacing is still enforced, because two frames a few pixels apart
    have too small a baseline to triangulate and only add noise to the solve.
    """
    if not motion_frames:
        return [shot.start_frame, shot.end_frame]

    magnitudes = np.array([max(mf.flow_magnitude, 0.0) for mf in motion_frames])
    total_motion = float(magnitudes.sum())
    frame_count = shot.frame_count

    target = int(np.clip(
        round((12 + total_motion / 90.0) * max(density, 0.05)),
        min_count, min(max_count, frame_count),
    ))
    if frame_count <= target:
        return list(range(shot.start_frame, shot.end_frame + 1))

    min_gap = max(1, frame_count // (target * 3))

    chosen = [shot.start_frame]
    if total_motion < 1e-6:
        # Static shot: nothing to sample by motion, so fall back to uniform.
        step = max(1, frame_count // target)
        chosen = list(range(shot.start_frame, shot.end_frame + 1, step))
        if chosen[-1] != shot.end_frame:
            chosen.append(shot.end_frame)
        return chosen

    motion_step = total_motion / max(target - 1, 1)
    accumulated = 0.0
    for mf, mag in zip(motion_frames, magnitudes):
        accumulated += float(mag)
        if accumulated >= motion_step and (mf.frame_index - chosen[-1]) >= min_gap:
            chosen.append(mf.frame_index)
            accumulated = 0.0
    if chosen[-1] != shot.end_frame:
        if shot.end_frame - chosen[-1] < min_gap and len(chosen) > 1:
            chosen[-1] = shot.end_frame
        else:
            chosen.append(shot.end_frame)
    return chosen


def solve_with_fallback(
    backends: list[GeometryBackend],
    context: SolveContext,
    *,
    confidence_threshold: float = 0.35,
) -> tuple[GeometryResult, list[SolverDecision]]:
    """Walk the ladder in order; return the best honest result and the audit log.

    "Best" is by confidence, not by order: if a later, cheaper backend is more
    confident than an earlier one, it wins. Backends are still tried in ladder
    order so the expensive ones get first refusal, and the walk stops early once
    something clears the threshold.
    """
    decisions: list[SolverDecision] = []
    results: list[GeometryResult] = []

    for backend in backends:
        ok, why = backend.available()
        if not ok:
            decision = SolverDecision(
                solver=backend.source, attempted=False, succeeded=False,
                message=f"unavailable: {why}",
            )
            decisions.append(decision)
            if context.reporter:
                context.reporter.solver_decision(decision)
            continue

        suited, why_not = backend.suitable_for(context)
        if not suited:
            decision = SolverDecision(
                solver=backend.source, attempted=False, succeeded=False,
                message=f"skipped: {why_not}",
            )
            decisions.append(decision)
            if context.reporter:
                context.reporter.solver_decision(decision)
            continue

        started = time.time()
        try:
            result = backend.estimate(context)
        except Exception as exc:  # noqa: BLE001 - I12: a solver crash degrades, never fails
            log.exception("backend %s raised", backend.source.value)
            result = failed_result(
                backend.source,
                f"{type(exc).__name__}: {exc}",
                total_frames=context.shot.frame_count,
            )
            result.message += f" | {traceback.format_exc(limit=3).splitlines()[-1]}"
        elapsed = time.time() - started

        results.append(result)
        decision = result.to_decision(duration=elapsed)
        decisions.append(decision)
        if context.reporter:
            context.reporter.solver_decision(decision)

        if result.succeeded and result.confidence >= confidence_threshold:
            break

    successful = [r for r in results if r.succeeded and r.pose_count >= 2]
    if not successful:
        best = failed_result(
            SolverSource.MOTION_PROXY_2D,
            "no geometric backend produced a usable result; "
            "falling back to the 2D motion proxy",
            total_frames=context.shot.frame_count,
        )
        return best, decisions

    best = max(successful, key=lambda r: r.confidence)
    for decision in decisions:
        if decision.solver == best.source and decision.succeeded:
            decision.selected = True
            break
    if context.reporter:
        context.reporter.info(
            f"selected {best.source.value} (confidence {best.confidence:.2f}, "
            f"{best.registered_frames}/{best.total_frames} frames registered)"
        )
    return best, decisions
