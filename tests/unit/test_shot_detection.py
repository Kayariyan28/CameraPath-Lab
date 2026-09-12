"""Cut detection: find real edits, never split on fast camera motion (I4)."""

from __future__ import annotations

import pytest

from app.models.schemas.video import ShotComplexity
from app.video.shots import (
    CutThresholds,
    _neighbour_baseline,
    detect_shots,
    estimate_complexity,
)
from tests.conftest import requires_ffmpeg


class TestNeighbourBaseline:
    def test_excludes_the_sample_itself(self):
        import numpy as np
        values = np.array([0.0, 0.0, 9.0, 0.0, 0.0])
        assert _neighbour_baseline(values, 2, 2) == 0.0

    def test_takes_the_worst_neighbour(self):
        import numpy as np
        values = np.array([0.1, 0.5, 9.0, 0.2, 0.05])
        assert _neighbour_baseline(values, 2, 2) == 0.5

    def test_handles_edges(self):
        import numpy as np
        values = np.array([5.0, 1.0, 1.0])
        assert _neighbour_baseline(values, 0, 2) == 1.0
        assert _neighbour_baseline(values, 2, 2) == 5.0

    def test_single_element(self):
        import numpy as np
        assert _neighbour_baseline(np.array([3.0]), 0, 2) == 0.0


@requires_ffmpeg
class TestRealCuts:
    def test_finds_a_single_cut(self, clips_dir):
        shots, candidates = _detect(clips_dir, "hard_cut.mp4")
        assert len(shots) == 2
        assert abs(shots[1].start_time - 1.5) < 0.07

    def test_cut_between_similar_looking_shots(self, clips_dir):
        """Both halves of hard_cut.mp4 are built from the same texture recipe,
        so their histograms barely differ — the case a histogram-based detector
        misses. Motion-compensated residual still catches it."""
        _, candidates = _detect(clips_dir, "hard_cut.mp4")
        accepted = [c for c in candidates if c.accepted]
        assert len(accepted) == 1
        # Confirm the appearance cue really was weak, i.e. this test is testing
        # what it claims to test.
        assert accepted[0].histogram_score < 0.5

    def test_finds_two_cuts(self, clips_dir):
        shots, candidates = _detect(clips_dir, "three_shots.mp4")
        assert len(shots) == 3
        assert abs(shots[1].start_time - 1.0) < 0.07
        assert abs(shots[2].start_time - 2.0) < 0.07
        assert sum(1 for c in candidates if c.accepted) == 2

    def test_cut_immediately_after_a_whip_pan(self, clips_dir):
        """Adversarial: the whip must be rejected and the cut right after it
        still found. A detector that suppresses everything near fast motion
        fails this."""
        shots, candidates = _detect(clips_dir, "cut_after_whip.mp4")
        assert len(shots) == 2
        assert abs(shots[1].start_time - 1.2) < 0.08
        assert sum(1 for c in candidates if c.accepted) == 1

    def test_accepted_cuts_are_explained(self, clips_dir):
        _, candidates = _detect(clips_dir, "hard_cut.mp4")
        for c in candidates:
            assert c.reason
            if c.accepted:
                assert "residual" in c.reason


@requires_ffmpeg
class TestNoFalseCuts:
    CONTINUOUS = [
        "fast_motion_no_cut.mp4", "pan_constant.mp4", "pan_accelerating.mp4",
        "zoom_only.mp4", "roll.mp4", "static.mp4", "handheld_jitter.mp4",
        "portrait_23976.mp4",
    ]

    @pytest.mark.parametrize("name", CONTINUOUS)
    def test_continuous_clips_stay_one_shot(self, clips_dir, name):
        shots, _ = _detect(clips_dir, name)
        assert len(shots) == 1, f"{name} wrongly split into {len(shots)} shots"

    def test_whip_pan_rejection_is_explained(self, clips_dir):
        """The rejection must state that the neighbours are equally bad, since
        that is the actual reasoning."""
        _, candidates = _detect(clips_dir, "fast_motion_no_cut.mp4")
        rejected = [c for c in candidates if not c.accepted]
        assert rejected, "the whip pan should produce examined-but-rejected candidates"
        assert any("neighbours" in c.reason for c in rejected)


@requires_ffmpeg
class TestShotStructure:
    def test_shots_tile_the_timeline(self, clips_dir):
        from app.video.ffprobe import probe_frames, probe_video
        info = probe_video(clips_dir / "three_shots.mp4")
        frames, info = probe_frames(info)
        shots, _ = detect_shots(info, frames)
        assert shots[0].start_frame == 0
        assert shots[-1].end_frame == info.frame_count - 1
        for a, b in zip(shots, shots[1:]):
            assert b.start_frame == a.end_frame + 1

    def test_shot_ids_are_contiguous(self, clips_dir):
        shots, _ = _detect(clips_dir, "three_shots.mp4")
        assert [s.id for s in shots] == list(range(len(shots)))

    def test_no_shot_shorter_than_the_minimum(self, clips_dir):
        th = CutThresholds()
        shots, _ = _detect(clips_dir, "three_shots.mp4")
        for s in shots:
            assert s.frame_count >= th.min_shot_frames

    def test_short_input(self, clips_dir):
        from app.video.ffprobe import probe_frames, probe_video
        info = probe_video(clips_dir / "static.mp4")
        frames, info = probe_frames(info)
        shots, _ = detect_shots(info, frames[:2])
        assert len(shots) == 1

    def test_empty_input(self, clips_dir):
        from app.video.ffprobe import probe_video
        info = probe_video(clips_dir / "static.mp4")
        shots, candidates = detect_shots(info, [])
        assert shots == [] and candidates == []


class TestComplexity:
    def test_clean_shot_is_simple(self):
        level, reasons = estimate_complexity(
            frame_count=120, duration=4.0, mean_flow=6.0,
            inlier_ratio=0.95, texture_score=0.8, parallax_score=0.6,
        )
        assert level in (ShotComplexity.TRIVIAL, ShotComplexity.LOW)
        assert reasons

    def test_low_texture_fast_motion_is_hard(self):
        level, reasons = estimate_complexity(
            frame_count=60, duration=2.0, mean_flow=40.0,
            inlier_ratio=0.3, texture_score=0.1, parallax_score=0.02,
        )
        assert level in (ShotComplexity.HIGH, ShotComplexity.EXTREME)
        assert any("texture" in r for r in reasons)
        assert any("parallax" in r for r in reasons)

    def test_always_gives_a_reason(self):
        for args in [(120, 4.0, 6.0, 0.95, 0.8, 0.6), (10, 0.3, 50.0, 0.2, 0.05, 0.0)]:
            _, reasons = estimate_complexity(*args)
            assert reasons


def _detect(clips_dir, name):
    from app.video.ffprobe import probe_frames, probe_video
    info = probe_video(clips_dir / name)
    frames, info = probe_frames(info)
    return detect_shots(info, frames)
