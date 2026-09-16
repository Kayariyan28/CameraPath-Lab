"""Project a solved `Job` onto the agent-facing motion report.

This module is a lens, not a calculator. It selects fields, labels their units,
and aggregates with `min`/`max`/`mean`/`argmax` over values the pipeline already
produced. It never smooths, extrapolates, re-derives a pose, or substitutes a
plausible number for a missing one — a field the pipeline did not produce comes
out `None`, and `describe.py` will not turn a `None` into prose.

Two details are load-bearing:

  * Units come from `normalize.scale_units_label` via `exporters.resolve_scale`,
    the codebase's single source of the string "m", and never from
    `ShotTrajectory.scale_units`. A trajectory that claims metres without a
    surviving calibration is downgraded there, and this report inherits the
    downgrade instead of repeating the claim.
  * `translation.*` magnitudes are populated only when the solve reported
    translation as observable. A path length recovered from a pure pan is a
    number, but it is not travel, and publishing it as one is exactly the
    failure invariants I6/I7 exist to prevent.
"""

from __future__ import annotations

import math
from typing import Sequence

from app.agent.describe import describe_shot
from app.agent.models import (
    ConfidenceBlock,
    JobMotionReport,
    LensBlock,
    MoveOut,
    PoseOut,
    RotationBlock,
    ScaleBlock,
    ShotMotionReport,
    SolverAttempt,
    TranslationBlock,
    VideoFacts,
)
from app.models.schemas.jobs import Job
from app.models.schemas.trajectory import MotionLabel, ShotTrajectory
from app.models.schemas.video import Shot, VideoInfo
from app.trajectory.exporters import resolve_job_scale, resolve_scale


def video_facts(info: VideoInfo | None) -> VideoFacts | None:
    if info is None:
        return None
    return VideoFacts(
        filename=info.filename,
        width=info.width,
        height=info.height,
        duration_seconds=float(info.duration_seconds),
        frame_count=int(info.frame_count),
        frame_count_is_exact=bool(info.frame_count_is_exact),
        fps_average=float(info.fps_average),
        fps_nominal=float(info.fps_nominal),
        frame_rate_mode=info.frame_rate_mode.value,
        fps_jitter=float(info.fps_jitter),
        timing_source=info.timing_source.value,
        codec_name=info.codec_name,
        rotation_degrees=int(info.rotation_degrees),
    )


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _peak(values: Sequence[float], times: Sequence[float]) -> tuple[float | None, float | None]:
    """(max, time of max) over a measured series, or (None, None) if empty."""
    pairs = [
        (v, t)
        for v, t in zip(values, times)
        if v is not None and math.isfinite(float(v))
    ]
    if not pairs:
        return None, None
    value, when = max(pairs, key=lambda p: p[0])
    return float(value), float(when)


def _mean(values: Sequence[float]) -> float | None:
    usable = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not usable:
        return None
    return sum(usable) / len(usable)


def _focal_ratio(trajectory: ShotTrajectory) -> float | None:
    """tan(fov_start/2) / tan(fov_end/2).

    The same quantity `classify.py` thresholds against `ZOOM_RATIO_THRESHOLD`,
    computed the same way, so the structured number and the zoom label can never
    disagree about what the lens did.
    """
    fovs = [
        float(lf.fov_horizontal)
        for lf in trajectory.lens
        if lf.fov_horizontal is not None and math.isfinite(float(lf.fov_horizontal))
    ]
    if len(fovs) < 2:
        return None
    try:
        start = math.tan(math.radians(fovs[0]) / 2.0)
        end = math.tan(math.radians(fovs[-1]) / 2.0)
    except ValueError:
        return None
    if not (math.isfinite(start) and math.isfinite(end)) or end == 0.0:
        return None
    return start / end


def _solver_explanation(trajectory: ShotTrajectory) -> str:
    selected = next((d for d in trajectory.solver_decisions if d.selected), None)
    if selected is None:
        return "No solver was selected for this shot."
    others = [d for d in trajectory.solver_decisions if not d.selected and d.attempted]
    text = f"Selected {selected.solver.value}"
    if selected.message:
        text += f": {selected.message}"
    if others:
        reasons = "; ".join(
            f"{d.solver.value} {'succeeded but scored lower' if d.succeeded else 'did not succeed'}"
            + (f" ({d.message})" if d.message else "")
            for d in others
        )
        text += f". Other rungs — {reasons}"
    return text.rstrip(".") + "."


def _not_observable_reason(job: Job, shot_id: int, trajectory: ShotTrajectory) -> str | None:
    """Verbatim reason translation could not be measured, or None.

    Preference order matters: the routing decision explains *why the shot was
    sent down the perceptual path*, which is the answer an agent wants. The
    selected solver's own message is the fallback for a shot that was routed to
    a geometric solve and then found no translation in it.
    """
    if trajectory.confidence.translation_observable:
        return None
    if job.analysis is not None:
        for analysis in job.analysis.shot_analyses:
            if analysis.shot.id == shot_id and analysis.recommendation_reason:
                return analysis.recommendation_reason
    selected = next((d for d in trajectory.solver_decisions if d.selected), None)
    if selected is not None and selected.message:
        return selected.message
    return None


def _intrinsics_provenance(trajectory: ShotTrajectory) -> str:
    """What the lens numbers rest on.

    `initial_intrinsics` writes a full provenance sentence, but it goes to the
    job log rather than onto the trajectory, so the only persisted signal is
    `LensFrame.is_estimated`. This reports exactly that signal and nothing more.
    """
    if not trajectory.lens:
        return "no per-frame lens curve was produced"
    if any(lf.is_estimated for lf in trajectory.lens):
        return "estimated from image motion (LensFrame.is_estimated)"
    return "taken from a trusted prior — container metadata or a user override"


def _zoom(trajectory: ShotTrajectory) -> tuple[str, float]:
    labels = {m.label for m in trajectory.classified_moves}
    if MotionLabel.ZOOM_IN in labels:
        return "in", float(trajectory.confidence.zoom_confidence)
    if MotionLabel.ZOOM_OUT in labels:
        return "out", float(trajectory.confidence.zoom_confidence)
    return "none", float(trajectory.confidence.zoom_confidence)


def _primary_move(trajectory: ShotTrajectory) -> tuple[str | None, bool]:
    real = [m for m in trajectory.classified_moves if m.label is not MotionLabel.MIXED_6DOF]
    if not real:
        return None, False
    top = max(real, key=lambda m: m.strength)
    is_static = len(real) == 1 and real[0].label is MotionLabel.STATIC
    return top.label.value, is_static


def shot_report(
    job: Job,
    trajectory: ShotTrajectory,
    *,
    shot_index: int,
    shot: Shot | None,
    include_poses: bool = False,
    max_poses: int = 200,
) -> ShotMotionReport:
    """One shot's honest result, ready to hand to an agent."""
    scale = resolve_scale(trajectory)
    units = scale.units
    conf = trajectory.confidence
    info = job.video

    times = [k.timestamp for k in trajectory.kinematics]
    speeds = [k.speed for k in trajectory.kinematics]
    angular = [k.angular_speed for k in trajectory.kinematics]

    observable = bool(conf.translation_observable)
    peak_speed, peak_speed_time = _peak(speeds, times) if observable else (None, None)
    translation = TranslationBlock(
        observable=observable,
        confidence=float(conf.translation_confidence),
        units=units,
        total_path_length=_finite(trajectory.total_path_length) if observable else None,
        mean_speed=_mean(speeds) if observable else None,
        peak_speed=peak_speed,
        peak_speed_time=peak_speed_time,
        speed_units=f"{units}/s",
        not_observable_reason=_not_observable_reason(job, trajectory.shot_id, trajectory),
    )

    peak_ang, peak_ang_time = _peak(angular, times)
    rotation = RotationBlock(
        confidence=float(conf.rotation_confidence),
        total_rotation_deg=float(trajectory.total_rotation_deg),
        mean_angular_speed_deg_s=_mean(angular),
        peak_angular_speed_deg_s=peak_ang,
        peak_angular_speed_time=peak_ang_time,
    )

    fovs = [
        float(lf.fov_horizontal)
        for lf in trajectory.lens
        if lf.fov_horizontal is not None and math.isfinite(float(lf.fov_horizontal))
    ]
    zoom, zoom_confidence = _zoom(trajectory)
    lens = LensBlock(
        fov_horizontal_start_deg=fovs[0] if fovs else None,
        fov_horizontal_end_deg=fovs[-1] if fovs else None,
        fov_horizontal_min_deg=min(fovs) if fovs else None,
        fov_horizontal_max_deg=max(fovs) if fovs else None,
        focal_ratio=_focal_ratio(trajectory),
        zoom=zoom,
        zoom_confidence=zoom_confidence,
        lens_mode_used=job.settings.lens_mode.value,
        focal_is_estimated=bool(trajectory.lens and trajectory.lens[0].is_estimated),
        intrinsics_provenance=_intrinsics_provenance(trajectory),
    )

    primary, is_static = _primary_move(trajectory)

    poses: list[PoseOut] = []
    stride = 1
    if include_poses and trajectory.poses and max_poses > 0:
        # Stride, never resample: an interpolated pose is a pose nobody
        # measured, and the exporters make the same promise about their rows.
        stride = max(1, math.ceil(len(trajectory.poses) / max_poses))
        for pose in trajectory.poses[::stride]:
            poses.append(
                PoseOut(
                    frame_index=pose.frame_index,
                    timestamp=float(pose.timestamp),
                    position=[float(v) for v in pose.position],
                    quaternion=[float(v) for v in pose.quaternion],
                    fov_horizontal=float(pose.fov_horizontal),
                    focal_normalized=float(pose.focal_normalized),
                    confidence=float(pose.confidence),
                    solver_source=pose.solver_source.value,
                    is_anchor=bool(pose.is_anchor),
                )
            )

    report = ShotMotionReport(
        shot_id=trajectory.shot_id,
        shot_index=shot_index,
        start_time=float(shot.start_time) if shot else 0.0,
        end_time=float(shot.end_time) if shot else float(trajectory.duration),
        duration_seconds=float(trajectory.duration),
        frame_count=int(trajectory.frame_count),
        fps=float(trajectory.fps),
        timing_source=info.timing_source.value if info else None,
        frame_rate_mode=info.frame_rate_mode.value if info else None,
        fps_jitter=float(info.fps_jitter) if info else None,
        cut_confidence=float(shot.confidence) if shot else None,
        solver_used=(
            next((d.solver.value for d in trajectory.solver_decisions if d.selected), None)
            or (trajectory.poses[0].solver_source.value if trajectory.poses else "unknown")
        ),
        pipeline_mode_used=trajectory.pipeline_mode_used.value,
        motion_fidelity=trajectory.motion_fidelity.value,
        solver_attempts=[
            SolverAttempt(
                solver=d.solver.value,
                attempted=d.attempted,
                succeeded=d.succeeded,
                selected=d.selected,
                registered_frames=d.registered_frames,
                total_frames=d.total_frames,
                duration_seconds=float(d.duration_seconds),
                mean_reprojection_error=_finite(d.mean_reprojection_error),
                confidence=float(d.confidence),
                message=d.message,
            )
            for d in trajectory.solver_decisions
        ],
        solver_explanation=_solver_explanation(trajectory),
        confidence=ConfidenceBlock(
            level=conf.level.value,
            score=float(conf.score),
            headline=conf.headline,
            reasons=list(conf.reasons),
            registered_frame_ratio=float(conf.registered_frame_ratio),
            persistent_track_count=int(conf.persistent_track_count),
            mean_inlier_ratio=float(conf.mean_inlier_ratio),
            mean_reprojection_error=_finite(conf.mean_reprojection_error),
            median_reprojection_error=_finite(conf.median_reprojection_error),
            baseline_parallax_score=float(conf.baseline_parallax_score),
            bundle_adjustment_residual=_finite(conf.bundle_adjustment_residual),
            solver_agreement=_finite(conf.solver_agreement),
            focal_stability=float(conf.focal_stability),
            temporal_consistency=float(conf.temporal_consistency),
        ),
        translation=translation,
        rotation=rotation,
        lens=lens,
        moves=[
            MoveOut(
                label=m.label.value,
                strength=float(m.strength),
                start_time=float(m.start_time),
                end_time=float(m.end_time),
                description=m.description,
            )
            for m in trajectory.classified_moves
        ],
        primary_move=primary,
        is_static=is_static,
        pipeline_summary=trajectory.summary,
        poses=poses,
        pose_stride=stride,
        poses_are_strided_subset=bool(poses) and stride > 1,
        anchor_frame_count=sum(1 for p in trajectory.poses if p.is_anchor),
    )
    # The prose is written from the finished report, so it can only say things
    # the structured fields above already state.
    return report.model_copy(update={"prompt_description": describe_shot(report)})


def job_report(
    job: Job,
    *,
    ui_url: str,
    job_dir: str,
    shot_id: int | None = None,
    include_poses: bool = False,
    max_poses: int = 200,
) -> JobMotionReport:
    """Every solved shot, or just one."""
    trajectories = list(job.trajectories)
    shots_by_id = {s.id: s for s in (job.analysis.shots if job.analysis else [])}

    selected = [
        (index, t)
        for index, t in enumerate(trajectories)
        if shot_id is None or t.shot_id == shot_id
    ]

    scale = resolve_job_scale(trajectories)
    return JobMotionReport(
        job_id=job.id,
        ui_url=ui_url,
        job_dir=job_dir,
        shot_count=len(trajectories),
        video=video_facts(job.video),
        coordinate_system=(
            trajectories[0].coordinate_system.model_dump(mode="json") if trajectories else {}
        ),
        scale=ScaleBlock(
            mode=scale.mode.value,
            units=scale.units,
            metric_scale_factor=scale.metric_scale_factor,
            statement=" ".join(scale.notes) if scale.notes else "",
        ),
        warnings=list(job.analysis.warnings) if job.analysis else [],
        shots=[
            shot_report(
                job,
                trajectory,
                shot_index=index,
                shot=shots_by_id.get(trajectory.shot_id),
                include_poses=include_poses,
                max_poses=max_poses,
            )
            for index, trajectory in selected
        ],
    )
