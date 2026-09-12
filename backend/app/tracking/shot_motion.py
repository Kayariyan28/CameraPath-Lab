"""Per-shot dense motion analysis — the Phase-1 core.

Produces one `MotionFrame` for every frame transition in a shot, plus the
aggregate `MotionSignature` used to route the shot to a solver.

The two-pass structure per transition is deliberate. Background confidence is
needed to fit a clean model, but it can only be computed *from* a model fit. So:

    pass 1  fit a provisional similarity model on all tracks  -> residuals
            -> update each track's agreement history            (dynamic_rejection)
    pass 2  re-fit with low-confidence tracks excluded          (global_motion)

Doing it in one pass instead would let a large moving subject define the model it
is then judged against, which is exactly the failure this module exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.logging import StageReporter, get_logger
from app.models.schemas.motion import MotionFrame, MotionSignature
from app.models.schemas.video import FrameMetadata, Shot, VideoInfo
from app.tracking.dynamic_rejection import GeometricDynamicRejector
from app.tracking.features import blur_score, normalize_contrast, texture_score
from app.tracking.flow import LucasKanadeTracker
from app.tracking.global_motion import estimate_transition, summarise
from app.video.decoder import FrameDecoder, fit_long_edge

log = get_logger("tracking.shot_motion")

#: See the note at the use site before changing this.
DYNAMIC_CONTENT_WARNING_THRESHOLD = 0.35


@dataclass
class ShotMotionResult:
    shot_id: int
    motion_frames: list[MotionFrame]
    signature: MotionSignature
    analysis_size: tuple[int, int]
    """(width, height) the motion was measured at. Needed to convert pixel
    quantities to angles later, so it must travel with the data."""

    persistent_track_count: int = 0
    mean_track_age: float = 0.0
    total_tracks_spawned: int = 0
    dynamic_region_fraction: float = 0.0
    texture: float = 0.0
    blur: float = 1.0
    warnings: list[str] = field(default_factory=list)


def analyze_shot_motion(
    info: VideoInfo,
    shot: Shot,
    frames_meta: list[FrameMetadata],
    *,
    long_edge: int = 1080,
    max_features: int = 2000,
    dynamic_strength: float = 0.5,
    reporter: StageReporter | None = None,
) -> ShotMotionResult:
    """Analyse one shot. `frames_meta` is the whole video's frame list; only the
    shot's own range is read from it, so timestamps stay the measured ones."""
    width, height = fit_long_edge(info.width, info.height, long_edge)
    decoder = FrameDecoder(info, long_edge=long_edge, gray=True)
    tracker = LucasKanadeTracker(max_features=max_features)
    rejector = GeometricDynamicRejector()

    by_index = {fm.frame_index: fm for fm in frames_meta}
    start_time = by_index.get(shot.start_frame)
    start_seconds = start_time.time_seconds if start_time else shot.start_time

    motion_frames: list[MotionFrame] = []
    warnings: list[str] = []
    texture_samples: list[float] = []
    blur_samples: list[float] = []

    prev_time: float | None = None
    total = max(1, shot.frame_count)
    processed = 0

    if reporter:
        reporter.stage("Tracking features", f"shot {shot.id}: {total} frames at {width}x{height}")

    for frame_index, frame in decoder.iter_frames(
        start_frame=shot.start_frame,
        end_frame=shot.end_frame,
        start_time=start_seconds,
    ):
        meta = by_index.get(frame_index)
        timestamp = meta.time_seconds if meta else (
            shot.start_time + processed / max(info.fps_average or 30.0, 1e-6)
        )

        # Quality metrics are measured on the ORIGINAL frame: they describe the
        # source material, and CLAHE would flatter both of them.
        if processed % 8 == 0:
            texture_samples.append(texture_score(frame))
            blur_samples.append(blur_score(frame))

        flow_result = tracker.track(frame_index, normalize_contrast(frame))
        processed += 1

        if flow_result is None:
            prev_time = timestamp
            continue

        # Real elapsed time, from measured PTS. Never 1/fps (invariant I1/I2).
        dt = timestamp - prev_time if prev_time is not None else 0.0
        if dt <= 0:
            dt = 1.0 / max(info.fps_average or 30.0, 1e-6)
        prev_time = timestamp

        weights = rejector.update(
            flow_result, tracker.tracks, (width, height), strength=dynamic_strength
        )
        mf = estimate_transition(
            flow_result, timestamp, dt, (width, height), weights=weights
        )
        motion_frames.append(mf)

        if reporter and processed % 20 == 0:
            reporter.progress(processed / total, f"shot {shot.id}: frame {processed}/{total}")

    texture = float(np.mean(texture_samples)) if texture_samples else 0.0
    blur = float(np.mean(blur_samples)) if blur_samples else 1.0

    signature = summarise(
        motion_frames, duration=shot.duration, texture=texture, blur=blur
    )

    # ---- honest warnings, surfaced in the analysis panel ----
    if texture < 0.25:
        warnings.append(
            f"Low texture (score {texture:.2f}) — few reliable features. "
            "Geometric reconstruction may be unreliable."
        )
    if blur < 0.3:
        warnings.append(
            f"Heavy motion blur (sharpness {blur:.2f}) — feature localisation degraded."
        )
    if signature.mean_inlier_ratio < 0.45 and motion_frames:
        warnings.append(
            f"Only {signature.mean_inlier_ratio:.0%} of tracked features agree with a "
            "single motion model — the scene may contain large moving content."
        )
    dyn = rejector.dynamic_region_fraction()
    # Threshold set against measured headroom, not intuition. Across 15 shots of
    # synthetic footage containing no moving content at all, the highest value
    # observed is 0.249 — residual clustering from genuine tracking failures on
    # very fast motion. 0.35 leaves ~40% margin above that. A confidently wrong
    # "a moving subject fills the frame" warning is worse than no warning.
    if dyn > DYNAMIC_CONTENT_WARNING_THRESHOLD:
        warnings.append(
            f"Moving content detected across ~{dyn:.0%} of the frame. "
            "Camera estimation is using background features only."
        )
    if signature.parallax_score < 0.12 and signature.mean_flow_magnitude >= 0.5:
        warnings.append(
            "Very little parallax — a single homography explains the motion. "
            "Physical translation is not reliably observable in this shot."
        )

    result = ShotMotionResult(
        shot_id=shot.id,
        motion_frames=motion_frames,
        signature=signature,
        analysis_size=(width, height),
        persistent_track_count=tracker.persistent_track_count(),
        mean_track_age=tracker.mean_track_age(),
        total_tracks_spawned=tracker.total_spawned(),
        dynamic_region_fraction=dyn,
        texture=texture,
        blur=blur,
        warnings=warnings,
    )

    log.info(
        "shot %d motion: %d transitions, mean flow %.2f px, inliers %.0f%%, "
        "parallax %.2f, jitter %.2f, %d persistent tracks",
        shot.id, len(motion_frames), signature.mean_flow_magnitude,
        signature.mean_inlier_ratio * 100, signature.parallax_score,
        signature.jitter_score, result.persistent_track_count,
    )
    return result
