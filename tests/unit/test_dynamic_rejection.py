"""Dynamic-object rejection: catch real moving content, and do not invent it."""

from __future__ import annotations

import numpy as np
import pytest

from app.tracking.dynamic_rejection import (
    RESIDUAL_THRESHOLD,
    GeometricDynamicRejector,
    residual_threshold_for,
)
from app.tracking.flow import FlowResult, Track
from tests.conftest import requires_ffmpeg


class TestAdaptiveThreshold:
    """A fixed residual tolerance makes every fast pan look full of moving
    objects, because tracking error grows with displacement."""

    def test_floor_applies_to_slow_motion(self):
        assert residual_threshold_for(0.0) == RESIDUAL_THRESHOLD
        assert residual_threshold_for(5.0) == RESIDUAL_THRESHOLD

    def test_scales_with_fast_motion(self):
        assert residual_threshold_for(100.0) > RESIDUAL_THRESHOLD
        assert residual_threshold_for(100.0) == pytest.approx(8.0)

    def test_monotonic(self):
        values = [residual_threshold_for(f) for f in (0, 10, 25, 50, 100, 200)]
        assert values == sorted(values)


def _flow_result(src: np.ndarray, dst: np.ndarray) -> tuple[FlowResult, dict[int, Track]]:
    ids = np.arange(len(src), dtype=np.int64)
    tracks = {
        int(i): Track(id=int(i), x=float(src[i][0]), y=float(src[i][1]),
                      start_frame=0, last_frame=1, age=5)
        for i in ids
    }
    return FlowResult(
        frame_index=1,
        prev_points=src.astype(np.float32),
        curr_points=dst.astype(np.float32),
        track_ids=ids,
        fb_error=np.zeros(len(src), np.float32),
        tracks_in=len(src),
        tracks_survived=len(src),
    ), tracks


class TestRejectorBehaviour:
    def test_pure_global_motion_is_all_background(self):
        """Every point moving identically = a camera move, nothing dynamic."""
        rng = np.random.default_rng(0)
        src = rng.uniform(0, 960, size=(300, 2))
        dst = src + np.array([12.0, -3.0])
        fr, tracks = _flow_result(src, dst)

        rejector = GeometricDynamicRejector()
        weights = rejector.update(fr, tracks, (960, 540), strength=0.5)

        assert weights.min() > 0.5, "global motion must not be flagged as dynamic"
        assert rejector.dynamic_region_fraction() == 0.0

    def test_a_coherent_moving_patch_is_rejected(self):
        """A compact region moving against the scene is an object."""
        rng = np.random.default_rng(1)
        background = rng.uniform(0, 960, size=(400, 2))
        # A cluster in one corner moving the opposite way.
        subject = rng.uniform(0, 200, size=(120, 2)) + np.array([60.0, 60.0])
        src = np.vstack([background, subject])
        dst = np.vstack([
            background + np.array([10.0, 0.0]),
            subject + np.array([-22.0, 9.0]),
        ])
        fr, tracks = _flow_result(src, dst)

        rejector = GeometricDynamicRejector()
        weights = rejector.update(fr, tracks, (960, 540), strength=1.0)

        bg_w = weights[: len(background)].mean()
        subj_w = weights[len(background):].mean()
        assert subj_w < bg_w, f"subject {subj_w:.3f} should score below background {bg_w:.3f}"
        assert subj_w < 0.4, f"moving subject not rejected (weight {subj_w:.3f})"
        assert rejector.dynamic_region_fraction() > 0.0

    def test_scattered_noise_is_not_an_object(self):
        """Random residuals are bad tracks, not a rigid moving body."""
        rng = np.random.default_rng(2)
        src = rng.uniform(0, 960, size=(400, 2))
        dst = src + np.array([8.0, 0.0])
        # Perturb a random scattered 15% with random directions.
        idx = rng.choice(len(src), 60, replace=False)
        dst[idx] += rng.normal(0, 9, size=(60, 2))
        fr, tracks = _flow_result(src, dst)

        rejector = GeometricDynamicRejector()
        rejector.update(fr, tracks, (960, 540), strength=0.5)
        assert rejector.dynamic_region_fraction() < 0.12, (
            "scattered tracking noise must not be reported as a moving object"
        )

    def test_handles_too_few_points(self):
        src = np.array([[10.0, 10.0], [20.0, 20.0]])
        fr, tracks = _flow_result(src, src + 1.0)
        weights = GeometricDynamicRejector().update(fr, tracks, (960, 540))
        assert len(weights) == 2

    def test_handles_empty(self):
        fr, tracks = _flow_result(np.empty((0, 2)), np.empty((0, 2)))
        weights = GeometricDynamicRejector().update(fr, tracks, (960, 540))
        assert len(weights) == 0


@requires_ffmpeg
class TestNoFalsePositivesOnRealClips:
    """None of the synthetic clips contain a moving object. Any dynamic-content
    warning on them is a false positive, and a confident false warning is worse
    than no warning at all."""

    CLIPS = ["pan_constant.mp4", "pan_accelerating.mp4", "zoom_only.mp4",
             "roll.mp4", "static.mp4", "handheld_jitter.mp4",
             "fast_motion_no_cut.mp4", "hard_cut.mp4"]

    @pytest.mark.parametrize("name", CLIPS)
    def test_no_moving_content_warning(self, analyzed, name):
        _, _, _, _, motion = analyzed(name)
        offenders = [w for w in motion.warnings if "Moving content" in w]
        assert not offenders, f"{name}: false positive -> {offenders}"


class TestStuckTrackDiscrimination:
    """The cue that makes coherence usable.

    A Lucas-Kanade track that sticks reports zero motion, so its residual equals
    the negative of the true motion. Every stuck track therefore shares one
    residual direction, and a cluster of them looks exactly like a rigid object
    by direction alone — while also passing forward-backward validation, since a
    stuck track round-trips to itself.
    """

    def test_anti_parallel_same_magnitude_is_stuck(self):
        from app.tracking.dynamic_rejection import GeometricDynamicRejector as R
        global_motion = np.array([20.0, 0.0])
        # A stuck track's residual is exactly -global_motion.
        assert R._is_stuck_track_cluster(np.array([-20.0, 0.0]), global_motion)
        assert R._is_stuck_track_cluster(np.array([-19.0, 2.0]), global_motion)

    def test_unrelated_direction_is_an_object(self):
        from app.tracking.dynamic_rejection import GeometricDynamicRejector as R
        global_motion = np.array([20.0, 0.0])
        assert not R._is_stuck_track_cluster(np.array([0.0, 18.0]), global_motion)
        assert not R._is_stuck_track_cluster(np.array([25.0, 0.0]), global_motion)

    def test_wrong_magnitude_is_an_object(self):
        """Anti-parallel but much faster than the camera motion is a real object
        moving against the camera, not a failed track."""
        from app.tracking.dynamic_rejection import GeometricDynamicRejector as R
        global_motion = np.array([20.0, 0.0])
        assert not R._is_stuck_track_cluster(np.array([-70.0, 0.0]), global_motion)
        assert not R._is_stuck_track_cluster(np.array([-3.0, 0.0]), global_motion)

    def test_degenerate_inputs(self):
        from app.tracking.dynamic_rejection import GeometricDynamicRejector as R
        assert not R._is_stuck_track_cluster(np.zeros(2), np.array([10.0, 0.0]))
        assert not R._is_stuck_track_cluster(np.array([10.0, 0.0]), np.zeros(2))

    def test_stuck_tracks_do_not_read_as_an_object(self):
        """End-to-end: a pan where 12% of tracks stick must not be reported as
        containing moving content."""
        rng = np.random.default_rng(7)
        src = rng.uniform(0, 960, size=(500, 2))
        motion = np.array([22.0, 0.0])
        dst = src + motion
        stuck = rng.choice(len(src), 60, replace=False)
        dst[stuck] = src[stuck]  # these tracks report no motion at all
        fr, tracks = _flow_result(src, dst)

        rejector = GeometricDynamicRejector()
        rejector.update(fr, tracks, (960, 540), strength=0.5)
        assert rejector.dynamic_region_fraction() < 0.1, (
            "clustered stuck tracks were mistaken for a moving object"
        )


@requires_ffmpeg
class TestMeasuredHeadroom:
    """Records the actual margin the warning threshold relies on, so a future
    change that erodes it fails here rather than in the field."""

    def test_clean_footage_stays_below_the_warning_threshold(self, clips_dir):
        from app.tracking.shot_motion import (
            DYNAMIC_CONTENT_WARNING_THRESHOLD,
            analyze_shot_motion,
        )
        from app.video.ffprobe import probe_frames, probe_video
        from app.video.shots import detect_shots

        worst = 0.0
        worst_name = ""
        for path in sorted(clips_dir.glob("*.mp4")):
            info = probe_video(path)
            frames, info = probe_frames(info)
            shots, _ = detect_shots(info, frames)
            for shot in shots:
                result = analyze_shot_motion(
                    info, shot, frames, long_edge=1080, max_features=2400
                )
                if result.dynamic_region_fraction > worst:
                    worst = result.dynamic_region_fraction
                    worst_name = f"{path.name} shot {shot.id}"
        assert worst < DYNAMIC_CONTENT_WARNING_THRESHOLD, (
            f"{worst_name} reached {worst:.3f}, at or above the "
            f"{DYNAMIC_CONTENT_WARNING_THRESHOLD} warning threshold — no synthetic "
            "clip contains moving content, so this would be a false warning"
        )
