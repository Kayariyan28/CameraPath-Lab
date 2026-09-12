"""Frame decoding via an ffmpeg rawvideo pipe.

Deliberately does NOT dump the whole video to lossless images (spec §3). Frames
are streamed out of ffmpeg at analysis resolution straight into numpy, in bounded
chunks sized by the adaptive resource policy. Only the handful of keyframes that
SfM actually needs are ever written to disk, by `materialize_frames`.

Orientation note: ffmpeg auto-applies container rotation on decode, so what
arrives here is already display-oriented and matches `VideoInfo.width/height`.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.models.schemas.video import VideoInfo

log = get_logger("video.decoder")


class DecodeError(RuntimeError):
    pass


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise DecodeError("ffmpeg not found on PATH. Install with `brew install ffmpeg`.")
    return path


def _read_exact(stream, nbytes: int) -> bytes:
    """Read exactly `nbytes`, or fewer only at genuine EOF.

    A raw pipe read returns at most one pipe-buffer's worth (64 KiB on macOS),
    which is far smaller than one frame. Without this loop every frame past the
    first 64 KiB is silently truncated.
    """
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break  # EOF
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def fit_long_edge(width: int, height: int, long_edge: int | None) -> tuple[int, int]:
    """Scale (w,h) so the longer side is at most `long_edge`, preserving aspect.

    Rounded to even dimensions: some scalers and every yuv420 encoder require it,
    and keeping analysis frames even avoids a class of off-by-one chroma issues
    if the same size is reused for encoding a diagnostic overlay.
    """
    if not long_edge or long_edge <= 0:
        return width, height
    longest = max(width, height)
    if longest <= long_edge:
        return width, height
    scale = long_edge / longest
    w = max(2, int(round(width * scale)) // 2 * 2)
    h = max(2, int(round(height * scale)) // 2 * 2)
    return w, h


@dataclass
class DecodeSpec:
    """Resolved decode parameters — what the pipe will actually produce."""

    width: int
    height: int
    channels: int
    pix_fmt: str

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * self.channels

    def shape(self) -> tuple[int, ...]:
        return (self.height, self.width) if self.channels == 1 else (self.height, self.width, self.channels)


class FrameDecoder:
    """Streams frames for a video, optionally limited to a frame range.

    Usage:
        dec = FrameDecoder(info, long_edge=1080, gray=True)
        for idx, frame in dec.iter_frames(start_frame=0, end_frame=89):
            ...
    """

    def __init__(
        self,
        info: VideoInfo,
        *,
        long_edge: int | None = None,
        gray: bool = True,
    ):
        self.info = info
        self.gray = gray
        w, h = fit_long_edge(info.width, info.height, long_edge)
        self.spec = DecodeSpec(
            width=w,
            height=h,
            channels=1 if gray else 3,
            pix_fmt="gray" if gray else "bgr24",
        )
        log.info(
            "decoder ready: %dx%d -> %dx%d %s",
            info.width, info.height, w, h, self.spec.pix_fmt,
        )

    # ------------------------------------------------------------------ pipe

    def _build_command(self, start_time: float | None, frame_count: int | None) -> list[str]:
        cmd = [_ffmpeg(), "-v", "error", "-nostdin"]
        if start_time and start_time > 0:
            # Input-side seek with accurate_seek (ffmpeg default) decodes from the
            # preceding keyframe and discards, so it is both fast and frame-exact.
            cmd += ["-accurate_seek", "-ss", f"{start_time:.6f}"]
        cmd += ["-i", str(self.info.path)]
        if frame_count is not None and frame_count > 0:
            cmd += ["-frames:v", str(frame_count)]
        vf = f"scale={self.spec.width}:{self.spec.height}:flags=area"
        cmd += [
            "-vf", vf,
            "-an", "-sn", "-dn",
            "-f", "rawvideo",
            "-pix_fmt", self.spec.pix_fmt,
            "-",
        ]
        return cmd

    def iter_frames(
        self,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        start_time: float | None = None,
        step: int = 1,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (absolute_frame_index, frame) pairs.

        `start_time` should be the *measured* PTS of `start_frame` when available
        — passing it lets ffmpeg seek instead of decoding from zero, which is the
        difference between instant and minutes on a long source.
        """
        if end_frame is None:
            end_frame = max(0, self.info.frame_count - 1)
        if end_frame < start_frame:
            return
        count = end_frame - start_frame + 1

        cmd = self._build_command(start_time, count)
        log.debug("decode: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        assert proc.stdout is not None

        nbytes = self.spec.frame_bytes
        shape = self.spec.shape()
        index = start_frame
        emitted = 0
        try:
            while True:
                buf = _read_exact(proc.stdout, nbytes)
                if not buf:
                    break
                if len(buf) < nbytes:
                    # Truncated tail — a partial frame is never usable.
                    log.warning(
                        "truncated frame at index %d (%d/%d bytes)", index, len(buf), nbytes
                    )
                    break
                if (index - start_frame) % step == 0:
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
                    # frombuffer gives a read-only view over the pipe buffer;
                    # callers (cv2 in-place ops) need a writable array.
                    yield index, frame.copy()
                    emitted += 1
                index += 1
                if index > end_frame:
                    break
        finally:
            if proc.stdout:
                proc.stdout.close()
            stderr = b""
            try:
                stderr = proc.stderr.read() if proc.stderr else b""
            except OSError:
                pass
            if proc.stderr:
                proc.stderr.close()
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            # A non-zero rc with frames already emitted means we closed the pipe
            # early on purpose (requested range satisfied) — ffmpeg reports
            # EPIPE. Only a failure that produced nothing is a real error.
            if proc.returncode not in (0, None) and emitted == 0:
                raise DecodeError(
                    f"ffmpeg decode produced no frames (rc={proc.returncode}): "
                    f"{stderr.decode('utf-8', 'replace')[:400]}"
                )

    def iter_chunks(
        self,
        *,
        start_frame: int = 0,
        end_frame: int | None = None,
        start_time: float | None = None,
        chunk_frames: int = 128,
    ) -> Iterator[list[tuple[int, np.ndarray]]]:
        """Same as `iter_frames` but batched, so memory stays bounded."""
        chunk: list[tuple[int, np.ndarray]] = []
        for item in self.iter_frames(
            start_frame=start_frame, end_frame=end_frame, start_time=start_time
        ):
            chunk.append(item)
            if len(chunk) >= chunk_frames:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def read_frame(self, frame_index: int, time_seconds: float | None = None) -> np.ndarray | None:
        """Single frame, for previews and diagnostics."""
        for _, frame in self.iter_frames(
            start_frame=frame_index, end_frame=frame_index, start_time=time_seconds
        ):
            return frame
        return None


def materialize_frames(
    info: VideoInfo,
    frame_indices: list[int],
    frame_times: list[float],
    out_dir: Path,
    *,
    long_edge: int | None = None,
    quality: int = 2,
    prefix: str = "f",
) -> list[Path]:
    """Write only the requested frames to disk as JPEGs.

    Needed because COLMAP's feature extractor reads image *files*. We write the
    minimum set — the selected SfM keyframes — rather than the whole video
    (spec §3). JPEG at q=2 is visually lossless for feature detection and about
    an order of magnitude smaller than PNG, which matters on a full disk.

    `frame_times` must be the measured PTS for each index, so each extraction
    seeks rather than scanning.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = fit_long_edge(info.width, info.height, long_edge)
    written: list[Path] = []

    if len(frame_times) != len(frame_indices):
        raise DecodeError("frame_indices and frame_times must be the same length")

    # Grouping consecutive runs into a single ffmpeg invocation is a large win:
    # keyframe selection often produces runs, and each invocation costs a process
    # spawn plus a seek.
    runs: list[list[int]] = []
    for pos in range(len(frame_indices)):
        if runs and frame_indices[pos] == frame_indices[pos - 1] + 1:
            runs[-1].append(pos)
        else:
            runs.append([pos])

    for run in runs:
        first, last = run[0], run[-1]
        cmd = [
            _ffmpeg(), "-v", "error", "-nostdin",
            "-accurate_seek", "-ss", f"{frame_times[first]:.6f}",
            "-i", str(info.path),
            "-frames:v", str(len(run)),
            "-vf", f"scale={w}:{h}:flags=area",
            "-q:v", str(quality),
            "-an", "-sn", "-dn",
            "-start_number", str(frame_indices[first]),
            str(out_dir / f"{prefix}%06d.jpg"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise DecodeError(
                f"frame extraction failed for run {frame_indices[first]}-"
                f"{frame_indices[last]}: {(proc.stderr or '').strip()[:300]}"
            )

    for idx in frame_indices:
        p = out_dir / f"{prefix}{idx:06d}.jpg"
        if p.is_file():
            written.append(p)
        else:
            log.warning("expected extracted frame missing: %s", p.name)

    log.info("materialized %d/%d frames at %dx%d into %s",
             len(written), len(frame_indices), w, h, out_dir.name)
    return written
