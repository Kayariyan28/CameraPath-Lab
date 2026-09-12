"""Motion signatures, checked against clips whose true motion is known.

Each clip is an exact 2D image transform, so the expected image-space quantities
are computable in closed form. The analysis runs at 1080 px long edge on a
1280 px source, so pixel quantities scale by 1080/1280 = 0.84375.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.models.schemas.motion import MotionModel
from tests.conftest import requires_ffmpeg

SCALE = 1080 / 1280


@requires_ffmpeg
class TestTranslation:
    def test_constant_pan_total_displacement(self, analyzed):
        """Crop travels 3840-1280 = 2560 source px across the clip."""
        _, _, _, _, motion = analyzed("pan_constant.mp4")
        expected = -2560 * SCALE
        assert motion.signature.net_dx_pixels == pytest.approx(expected, rel=0.04)

    def test_constant_pan_has_uniform_velocity(self, analyzed):
        _, _, _, _, motion = analyzed("pan_constant.mp4")
        dxs = np.array([m.dx_pixels for m in motion.motion_frames])
        # Constant velocity: per-frame displacement should barely vary.
        assert np.std(dxs) / abs(np.mean(dxs)) < 0.10

    def test_accelerating_pan_covers_the_same_ground(self, analyzed):
        _, _, _, _, motion = analyzed("pan_accelerating.mp4")
        assert motion.signature.net_dx_pixels == pytest.approx(-2560 * SCALE, rel=0.05)

    def test_acceleration_is_visible_in_the_velocity_profile(self, analyzed):
        """I1 in spirit: a t^2 ramp must show a rising speed profile, not an
        averaged-out constant one."""
        _, _, _, _, motion = analyzed("pan_accelerating.mp4")
        mags = np.abs([m.dx_pixels for m in motion.motion_frames])
        first_third = mags[: len(mags) // 3].mean()
        last_third = mags[-len(mags) // 3:].mean()
        assert last_third > first_third * 3, (
            f"acceleration lost: {first_third:.1f} -> {last_third:.1f} px/frame"
        )

    def test_static_clip_invents_no_motion(self, analyzed):
        _, _, _, _, motion = analyzed("static.mp4")
        sig = motion.signature
        assert abs(sig.net_dx_pixels) < 1.0
        assert abs(sig.net_dy_pixels) < 1.0
        assert sig.mean_flow_magnitude < 0.4
        assert abs(sig.total_image_rotation_deg) < 0.5


@requires_ffmpeg
class TestRotationAndZoom:
    def test_roll_angle(self, analyzed):
        """rotate='0.35*t' over 3 s = 1.05 rad = 60.16 deg."""
        _, _, _, _, motion = analyzed("roll.mp4")
        expected = np.degrees(0.35 * 3.0)
        assert abs(motion.signature.total_image_rotation_deg) == pytest.approx(
            expected, rel=0.05
        )

    def test_roll_has_no_translation(self, analyzed):
        _, _, _, _, motion = analyzed("roll.mp4")
        assert abs(motion.signature.net_dx_pixels) < 10
        assert abs(motion.signature.net_dy_pixels) < 10

    def test_roll_is_rotation_dominant(self, analyzed):
        _, _, _, _, motion = analyzed("roll.mp4")
        assert motion.signature.rotation_dominance > 0.5

    def test_zoom_scale_factor(self, analyzed):
        """zoompan z: 1.0 -> 1.6 across the clip."""
        _, _, _, _, motion = analyzed("zoom_only.mp4")
        assert motion.signature.net_scale_change == pytest.approx(1.6, rel=0.05)

    def test_zoom_expands_radially(self, analyzed):
        _, _, _, _, motion = analyzed("zoom_only.mp4")
        assert motion.signature.mean_radial_flow > 0.3, "zoom-in must expand"

    def test_zoom_has_no_net_translation_or_rotation(self, analyzed):
        _, _, _, _, motion = analyzed("zoom_only.mp4")
        assert abs(motion.signature.net_dx_pixels) < 12
        assert abs(motion.signature.total_image_rotation_deg) < 1.0


@requires_ffmpeg
class TestJitter:
    """I10: handheld jitter must be detected, and smooth motion must not be
    mislabelled as jitter."""

    SMOOTH = ["static.mp4", "roll.mp4", "zoom_only.mp4",
              "pan_constant.mp4", "pan_accelerating.mp4"]

    @pytest.mark.parametrize("name", SMOOTH)
    def test_smooth_motion_scores_no_jitter(self, analyzed, name):
        _, _, _, _, motion = analyzed(name)
        assert motion.signature.jitter_score < 0.05, (
            f"{name} wrongly reported jitter {motion.signature.jitter_score:.3f}"
        )

    def test_handheld_scores_jitter(self, analyzed):
        _, _, _, _, motion = analyzed("handheld_jitter.mp4")
        assert motion.signature.jitter_score > 0.25

    def test_handheld_is_the_most_jittery(self, analyzed):
        handheld = analyzed("handheld_jitter.mp4")[4].signature.jitter_score
        for name in self.SMOOTH:
            assert handheld > analyzed(name)[4].signature.jitter_score


@requires_ffmpeg
class TestParallaxHonesty:
    """I5/I6 + spec §25: these clips are 2D image transforms with zero parallax
    by construction. The system must report that translation is unobservable
    rather than inventing a baseline."""

    @pytest.mark.parametrize(
        "name", ["pan_constant.mp4", "zoom_only.mp4", "roll.mp4", "pan_accelerating.mp4"]
    )
    def test_no_parallax_is_reported(self, analyzed, name):
        _, _, _, _, motion = analyzed(name)
        assert motion.signature.parallax_score < 0.15, (
            f"{name} claims parallax {motion.signature.parallax_score:.3f} "
            "but was generated as a pure 2D transform"
        )

    @pytest.mark.parametrize("name", ["pan_constant.mp4", "zoom_only.mp4", "roll.mp4"])
    def test_homography_explains_these_clips(self, analyzed, name):
        _, _, _, _, motion = analyzed(name)
        assert motion.signature.homography_dominance > 0.8

    def test_degeneracy_is_surfaced_as_a_warning(self, analyzed):
        _, _, _, _, motion = analyzed("pan_constant.mp4")
        assert any("parallax" in w.lower() for w in motion.warnings), motion.warnings


@requires_ffmpeg
class TestMotionFrameIntegrity:
    def test_one_motion_frame_per_transition(self, analyzed):
        info, frames, shots, _, motion = analyzed("pan_constant.mp4")
        assert len(motion.motion_frames) == shots[0].frame_count - 1

    def test_dt_comes_from_measured_timestamps(self, analyzed):
        """I1/I2: dt must be real elapsed time, never a hard-coded 1/fps."""
        info, frames, _, _, motion = analyzed("portrait_23976.mp4")
        expected = 1001 / 24000
        for mf in motion.motion_frames[1:10]:
            assert mf.dt == pytest.approx(expected, abs=1e-4)

    def test_timestamps_are_monotonic(self, analyzed):
        _, _, _, _, motion = analyzed("pan_constant.mp4")
        times = [m.timestamp for m in motion.motion_frames]
        assert times == sorted(times)

    def test_a_model_was_found_for_textured_clips(self, analyzed):
        _, _, _, _, motion = analyzed("pan_constant.mp4")
        models = {m.model_used for m in motion.motion_frames}
        assert models != {MotionModel.NONE}

    def test_confidence_is_bounded(self, analyzed):
        for name in ("pan_constant.mp4", "static.mp4", "handheld_jitter.mp4"):
            _, _, _, _, motion = analyzed(name)
            for m in motion.motion_frames:
                assert 0.0 <= m.confidence <= 1.0
                assert 0.0 <= m.inlier_ratio <= 1.0
