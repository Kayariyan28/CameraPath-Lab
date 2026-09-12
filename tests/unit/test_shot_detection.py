"""Cut detection: find real edits, and never split on fast camera motion (I4)."""

from __future__ import annotations

from app.video.shots import CutThresholds, detect_shots, estimate_complexity
from app.models.schemas.video import ShotComplexity
from tests.conftest import requires_ffmpeg


@requires_ffmpeg
class TestCutDetection:
    def test_finds_the_real_cut(self, analyzed):
        info, frames, shots, candidates, _ = analyzed("hard_cut.mp4")
        assert len(shots) == 2, f"expected 2 shots, got {len(shots)}"
        # The clip is two 1.5 s halves concatenated.
        boundary = shots[1].start_time
        assert abs(boundary - 1.5) < 0.1, f"cut at {boundary:.3f}s, expected ~1.5s"

    def test_cut_is_explained(self, analyzed):
        _, _, _, candidates, _ = analyzed("hard_cut.mp4")
        accepted = [c for c in candidates if c.accepted]
        assert accepted
        assert "track survival" in accepted[0].reason

    def test_fast_motion_is_not_a_cut(self, analyzed):
        """The classic false positive: a whip pan changes appearance as much as
        an edit does, but tracks survive and one model still explains them."""
        _, _, shots, _, _ = analyzed("fast_motion_no_cut.mp4")
        assert len(shots) == 1, f"whip pan split into {len(shots)} shots"

    def test_smooth_clips_are_single_shots(self, analyzed):
        for name in ("pan_constant.mp4", "pan_accelerating.mp4", "zoom_only.mp4",
                     "roll.mp4", "static.mp4", "handheld_jitter.mp4"):
            _, _, shots, _, _ = analyzed(name)
            assert len(shots) == 1, f"{name} split into {len(shots)} shots"

    def test_shots_tile_the_timeline_without_gaps_or_overlap(self, analyzed):
        info, frames, shots, _, _ = analyzed("hard_cut.mp4")
        assert shots[0].start_frame == 0
        assert shots[-1].end_frame == info.frame_count - 1
        for a, b in zip(shots, shots[1:]):
            assert b.start_frame == a.end_frame + 1, "shots must be contiguous"

    def test_shot_ids_are_contiguous(self, analyzed):
        _, _, shots, _, _ = analyzed("hard_cut.mp4")
        assert [s.id for s in shots] == list(range(len(shots)))

    def test_very_short_input_yields_one_shot(self, clips_dir):
        from app.video.ffprobe import probe_frames, probe_video
        info = probe_video(clips_dir / "static.mp4")
        frames, info = probe_frames(info)
        shots, _ = detect_shots(info, frames[:2])
        assert len(shots) == 1

    def test_empty_input_is_handled(self, clips_dir):
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
            assert reasons, "complexity must always be explained"
