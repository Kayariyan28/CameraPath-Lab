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
from app.tracking.keyframe_geometry import KeyframeHomography, KeyframeTracker
from app.tracking.parallax import (
    ParallaxEvidence,
    TrackSnapshotRecorder,
    summarize as summarize_parallax,
)
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

    parallax: ParallaxEvidence = field(default_factory=ParallaxEvidence)
    """Wide-baseline parallax evidence. This decides whether translation is
    observable, and therefore which solver the shot is routed to."""

    keyframe_homographies: list[KeyframeHomography] = field(default_factory=list)
    """SIFT homographies between consecutive keyframes: the long-baseline
    reference for absolute rotation and zoom (see keyframe_geometry)."""

    persistent_track_count: int = 0
    mean_track_age: float = 0.0
    total_tracks_spawned: int = 0
    dynamic_region_fraction: float = 0.0
    texture: float = 0.0
    blur: float = 1.0
    warnings: list[str] = field(default_factory=list)


#: Feature budget for the parallax pre-pass. Deliberately far below the main
#: pass: parallax needs long-lived tracks spread across the frame, not density,
#: and this pass exists only to answer one yes/no question.
PREPASS_FEATURES = 700


def measure_parallax(
    info: VideoInfo,
    shot: Shot,
    frames_meta: list[FrameMetadata],
    *,
    long_edge: int = 1080,
) -> ParallaxEvidence:
    """Light pre-pass that establishes whether translation is observable.

    Why this runs before the main analysis rather than falling out of it: the
    parallax verdict decides two things the main pass needs up front — which
    static-scene model the dynamic-object rejector should judge against, and
    which solver the shot is routed to. Both are decided by whether the camera
    translated through depth, and that cannot be known until tracks have been
    followed across a real baseline.

    The alternative — measure parallax during the main pass and accept that the
    rejector spent that pass using the wrong model — produced a 48% false
    "moving content" reading on a synthetic dolly containing no moving object.
    A pre-pass at roughly a third of the feature budget is the cheaper mistake.
    """
    width, height = fit_long_edge(info.width, info.height, long_edge)
    decoder = FrameDecoder(info, long_edge=long_edge, gray=True)
    tracker = LucasKanadeTracker(max_features=PREPASS_FEATURES)
    snapshots = TrackSnapshotRecorder()

    by_index = {fm.frame_index: fm for fm in frames_meta}
    start = by_index.get(shot.start_frame)
    start_seconds = start.time_seconds if start else shot.start_time

    for frame_index, frame in decoder.iter_frames(
        start_frame=shot.start_frame,
        end_frame=shot.end_frame,
        start_time=start_seconds,
    ):
        flow_result = tracker.track(frame_index, normalize_contrast(frame))
        if flow_result is None or flow_result.count == 0:
            continue
        flow = flow_result.flow_vectors()
        magnitude = float(np.median(np.linalg.norm(flow, axis=1))) if len(flow) else 0.0
        snapshots.observe(
            frame_index, flow_result.track_ids, flow_result.curr_points, magnitude
        )

    evidence = summarize_parallax(snapshots.pairs())
    log.info(
        "shot %d parallax pre-pass: score=%.3f reliable=%s over %d baselines",
        shot.id, evidence.score, evidence.measurement_reliable, evidence.pair_count,
    )
    return evidence


def analyze_shot_motion(
    info: VideoInfo,
    shot: Shot,
    frames_meta: list[FrameMetadata],
    *,
    long_edge: int = 1080,
    max_features: int = 2000,
    dynamic_strength: float = 0.5,
    reporter: StageReporter | None = None,
    parallax: ParallaxEvidence | None = None,
) -> ShotMotionResult:
    """Analyse one shot. `frames_meta` is the whole video's frame list; only the
    shot's own range is read from it, so timestamps stay the measured ones.

    `parallax` may be supplied to skip the pre-pass when it has already been
    measured (the pipeline does this); otherwise it is measured here.
    """
    width, height = fit_long_edge(info.width, info.height, long_edge)

    # Parallax must be judged over a BASELINE, not between adjacent frames: a
    # slow dolly moves almost nothing per frame, so consecutive views really are
    # related by a homography to within noise. Measured on a synthetic clip with
    # 22 m of genuine translation, adjacent-frame analysis scored 0.008.
    if parallax is None:
        if reporter:
            reporter.progress(0.0, f"shot {shot.id}: measuring parallax")
        parallax = measure_parallax(info, shot, frames_meta, long_edge=long_edge)

    decoder = FrameDecoder(info, long_edge=long_edge, gray=True)
    tracker = LucasKanadeTracker(max_features=max_features)
    rejector = GeometricDynamicRejector()
    keyframes = KeyframeTracker(width)

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

        normalized = normalize_contrast(frame)
        flow_result = tracker.track(frame_index, normalized)
        processed += 1

        if flow_result is None:
            prev_time = timestamp
            keyframes.observe(frame_index, timestamp, normalized, None)
            continue

        # Real elapsed time, from measured PTS. Never 1/fps (invariant I1/I2).
        dt = timestamp - prev_time if prev_time is not None else 0.0
        if dt <= 0:
            dt = 1.0 / max(info.fps_average or 30.0, 1e-6)
        prev_time = timestamp

        weights = rejector.update(
            flow_result, tracker.tracks, (width, height),
            strength=dynamic_strength,
            parallax_present=parallax.translation_observable,
            dt=dt,
        )
        mf = estimate_transition(
            flow_result, timestamp, dt, (width, height), weights=weights
        )
        motion_frames.append(mf)
        keyframes.observe(frame_index, timestamp, normalized, mf)

        if reporter and processed % 20 == 0:
            reporter.progress(processed / total, f"shot {shot.id}: frame {processed}/{total}")

    keyframe_homographies = keyframes.finish()
    texture = float(np.mean(texture_samples)) if texture_samples else 0.0
    blur = float(np.mean(blur_samples)) if blur_samples else 1.0

    signature = summarise(
        motion_frames, duration=shot.duration, texture=texture, blur=blur
    )

    # The per-transition parallax estimate cannot see depth structure; the
    # wide-baseline measurement is authoritative.
    signature.parallax_score = parallax.score
    if parallax.pair_count:
        signature.homography_dominance = parallax.homography_inlier_ratio

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
    if not parallax.translation_observable and signature.mean_flow_magnitude >= 0.5:
        warnings.append(f"Translation is not reliably observable: {parallax.reason}")

    result = ShotMotionResult(
        shot_id=shot.id,
        motion_frames=motion_frames,
        signature=signature,
        analysis_size=(width, height),
        persistent_track_count=tracker.persistent_track_count(),
        mean_track_age=tracker.mean_track_age(),
        total_tracks_spawned=tracker.total_spawned(),
        dynamic_region_fraction=dyn,
        parallax=parallax,
        keyframe_homographies=keyframe_homographies,
        texture=texture,
        blur=blur,
        warnings=warnings,
    )

    log.info(
        "shot %d motion: %d transitions, mean flow %.2f px, inliers %.0f%%, "
        "parallax %.2f over %d baselines, jitter %.2f, %d persistent tracks",
        shot.id, len(motion_frames), signature.mean_flow_magnitude,
        signature.mean_inlier_ratio * 100, signature.parallax_score,
        parallax.pair_count, signature.jitter_score, result.persistent_track_count,
    )
    return result
