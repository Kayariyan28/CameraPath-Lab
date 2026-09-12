"""Ingestion must report measured facts, not assumptions (invariants I1, I2)."""

from __future__ import annotations

import pytest

from app.models.schemas.video import FrameRateMode, TimingSource
from app.video.decoder import FrameDecoder, fit_long_edge
from app.video.ffprobe import probe_frames, probe_video
from tests.conftest import requires_ffmpeg


class TestFitLongEdge:
    def test_no_upscale(self):
        assert fit_long_edge(640, 480, 1080) == (640, 480)

    def test_downscale_preserves_aspect(self):
        w, h = fit_long_edge(3840, 2160, 1080)
        assert w == 1080
        assert abs((w / h) - (3840 / 2160)) < 0.02

    def test_portrait_caps_the_long_edge(self):
        w, h = fit_long_edge(720, 1280, 960)
        assert h == 960 and w < h

    def test_dimensions_are_even(self):
        for src in [(1920, 1080), (1279, 721), (3840, 2160), (1001, 999)]:
            w, h = fit_long_edge(*src, 501)
            assert w % 2 == 0 and h % 2 == 0, src

    def test_none_is_passthrough(self):
        assert fit_long_edge(1920, 1080, None) == (1920, 1080)


@requires_ffmpeg
class TestProbe:
    def test_timestamps_come_from_the_container(self, clips_dir):
        """I2: timing must be PTS-derived, not index/fps."""
        info = probe_video(clips_dir / "pan_constant.mp4")
        frames, info = probe_frames(info)
        assert info.timing_source is TimingSource.CONTAINER_PTS
        assert info.frame_count_is_exact

    def test_non_integer_frame_rate_is_measured_exactly(self, clips_dir):
        info = probe_video(clips_dir / "portrait_23976.mp4")
        frames, info = probe_frames(info)
        expected_dt = 1001 / 24000
        deltas = [
            frames[i + 1].time_seconds - frames[i].time_seconds
            for i in range(min(10, len(frames) - 1))
        ]
        for d in deltas:
            assert abs(d - expected_dt) < 1e-4, f"{d} vs {expected_dt}"

    def test_constant_frame_rate_is_not_called_variable(self, clips_dir):
        info = probe_video(clips_dir / "pan_constant.mp4")
        _, info = probe_frames(info)
        assert info.frame_rate_mode is FrameRateMode.CONSTANT
        assert info.fps_jitter < 0.02

    def test_first_frame_is_time_zero(self, clips_dir):
        """Shot-relative clock: downstream kinematics assume it."""
        info = probe_video(clips_dir / "pan_constant.mp4")
        frames, _ = probe_frames(info)
        assert frames[0].time_seconds == pytest.approx(0.0, abs=1e-9)

    def test_rotation_gives_display_oriented_dimensions(self, clips_dir):
        info = probe_video(clips_dir / "portrait_23976.mp4")
        assert info.is_portrait
        assert info.height > info.width

    def test_missing_file_raises(self, clips_dir):
        from app.video.ffprobe import FFprobeError
        with pytest.raises(FFprobeError):
            probe_video(clips_dir / "does_not_exist.mp4")


@requires_ffmpeg
class TestDecoder:
    def test_decodes_every_frame_contiguously(self, clips_dir):
        info = probe_video(clips_dir / "pan_constant.mp4")
        frames, info = probe_frames(info)
        dec = FrameDecoder(info, long_edge=480, gray=True)
        got = [i for i, _ in dec.iter_frames(start_frame=0, end_frame=info.frame_count - 1)]
        assert got == list(range(info.frame_count))

    def test_seek_is_frame_exact(self, clips_dir):
        """A seeked frame must be byte-identical to the sequentially decoded one.

        This is the guard against silently analysing the wrong frames, which
        would corrupt every timestamp downstream.
        """
        import numpy as np
        info = probe_video(clips_dir / "pan_constant.mp4")
        frames, info = probe_frames(info)
        dec = FrameDecoder(info, long_edge=480, gray=True)
        sequential = [f for _, f in dec.iter_frames(start_frame=0, end_frame=info.frame_count - 1)]
        for target in (7, info.frame_count // 2, info.frame_count - 1):
            seeked = dec.read_frame(target, frames[target].time_seconds)
            assert seeked is not None
            assert np.array_equal(seeked, sequential[target]), f"frame {target} mismatch"

    def test_frames_are_writable(self, clips_dir):
        """cv2 operates in place; a read-only view over the pipe buffer crashes."""
        info = probe_video(clips_dir / "static.mp4")
        frames, info = probe_frames(info)
        dec = FrameDecoder(info, long_edge=320, gray=True)
        _, frame = next(iter(dec.iter_frames(start_frame=0, end_frame=0)))
        frame[0, 0] = 42  # must not raise
        assert frame[0, 0] == 42

    def test_chunking_covers_everything_exactly_once(self, clips_dir):
        info = probe_video(clips_dir / "pan_constant.mp4")
        frames, info = probe_frames(info)
        dec = FrameDecoder(info, long_edge=320, gray=True)
        seen = [i for chunk in dec.iter_chunks(chunk_frames=17) for i, _ in chunk]
        assert seen == list(range(info.frame_count))

    def test_materialize_writes_only_requested_frames(self, clips_dir, tmp_path):
        from app.video.decoder import materialize_frames
        info = probe_video(clips_dir / "pan_constant.mp4")
        frames, info = probe_frames(info)
        wanted = [0, 10, 11, 12, 40, 89]
        out = materialize_frames(
            info, wanted, [frames[i].time_seconds for i in wanted], tmp_path, long_edge=640
        )
        assert len(out) == len(wanted)
        assert len(list(tmp_path.glob("*.jpg"))) == len(wanted)
