"""Authoritative container inspection via ffprobe.

Why this module is fussy about timestamps: the whole product promises that a
2-second deceleration in the source stays 2 seconds in the output (invariant I1).
That promise is only keepable if frame times come from the container's
presentation timestamps. `frame_index / nominal_fps` silently destroys it on any
VFR source — screen recordings, phone footage with adaptive frame rate, and most
drone footage shot in "auto" mode. So we read real PTS, and when we genuinely
cannot, we say so via `TimingSource.NOMINAL_FPS` rather than pretending.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from app.core.logging import get_logger
from app.models.schemas.video import (
    FrameMetadata,
    FrameRateMode,
    TimingSource,
    VideoInfo,
)

log = get_logger("video.ffprobe")

#: Above this relative std-dev of inter-frame deltas we call a stream VFR.
#: 2% is comfortably above container rounding noise (a 30000/1001 stream shows
#: ~0.3%) and below anything a genuinely variable stream produces.
VFR_JITTER_THRESHOLD = 0.02

#: Reading per-frame metadata for a very long video is slow; above this we probe
#: packets (no decode) instead of frames.
PACKET_MODE_FRAME_LIMIT = 20_000


class FFprobeError(RuntimeError):
    pass


def _binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFprobeError(
            f"{name} not found on PATH. Install with `brew install ffmpeg`."
        )
    return path


def _run_json(args: list[str], timeout: float = 240.0) -> dict:
    cmd = [_binary("ffprobe"), "-v", "error", "-print_format", "json", *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise FFprobeError(f"ffprobe timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise FFprobeError(f"ffprobe failed: {(proc.stderr or '').strip()[:500]}")
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFprobeError(f"ffprobe returned unparseable JSON: {exc}") from exc


def _parse_rate(value: str | None) -> float:
    """Parse ffprobe's 'num/den' rate strings. Returns 0.0 when unusable."""
    if not value or value in ("0/0", "N/A"):
        return 0.0
    try:
        frac = Fraction(value)
        return float(frac) if frac.denominator else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_rotation(stream: dict) -> int:
    """Recover display rotation from either the modern displaymatrix side data
    or the legacy `rotate` tag. Normalised to {0, 90, 180, 270}.

    This matters because ffmpeg *auto-applies* rotation when decoding, so the
    frames we analyse are already upright — but the stream's own width/height
    are not. Reporting the unrotated size would make every portrait iPhone clip
    analyse at the wrong aspect ratio.
    """
    rot = 0
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rot = int(round(float(side["rotation"])))
                break
            except (TypeError, ValueError):
                pass
    if rot == 0:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag:
            try:
                rot = int(round(float(tag)))
            except (TypeError, ValueError):
                rot = 0
    # ffprobe reports displaymatrix rotation as negative-clockwise; we only need
    # the axis-swap question, so fold into [0, 360).
    return int(rot) % 360


def _video_stream(data: dict) -> dict:
    streams = data.get("streams", []) or []
    for s in streams:
        if s.get("codec_type") == "video":
            return s
    raise FFprobeError("no video stream found in file")


def probe_video(path: str | Path) -> VideoInfo:
    """Container + stream facts. Cheap: no decoding."""
    p = Path(path)
    if not p.is_file():
        raise FFprobeError(f"file does not exist: {p}")

    data = _run_json(["-show_format", "-show_streams", str(p)])
    stream = _video_stream(data)
    fmt = data.get("format", {}) or {}

    codec_width = int(stream.get("width") or 0)
    codec_height = int(stream.get("height") or 0)
    if not codec_width or not codec_height:
        raise FFprobeError("video stream reports no dimensions")

    rotation = _parse_rotation(stream)
    # Decoded frames arrive already rotated, so display dimensions are what the
    # rest of the pipeline sees.
    if rotation in (90, 270):
        width, height = codec_height, codec_width
    else:
        width, height = codec_width, codec_height

    fps_nominal = _parse_rate(stream.get("r_frame_rate"))
    fps_average = _parse_rate(stream.get("avg_frame_rate")) or fps_nominal

    duration = 0.0
    for candidate in (stream.get("duration"), fmt.get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    frame_count = 0
    exact = False
    for key in ("nb_frames", "nb_read_frames"):
        try:
            frame_count = int(stream.get(key) or 0)
        except (TypeError, ValueError):
            frame_count = 0
        if frame_count > 0:
            exact = True
            break
    if frame_count <= 0 and duration > 0 and fps_average > 0:
        frame_count = int(round(duration * fps_average))
        exact = False

    has_audio = any(s.get("codec_type") == "audio" for s in data.get("streams", []) or [])

    tags = {}
    for source in (fmt.get("tags") or {}, stream.get("tags") or {}):
        for k, v in source.items():
            if isinstance(v, (str, int, float)):
                tags[str(k)] = str(v)

    bit_rate = None
    for candidate in (stream.get("bit_rate"), fmt.get("bit_rate")):
        try:
            bit_rate = int(candidate)
            break
        except (TypeError, ValueError):
            continue

    info = VideoInfo(
        path=str(p.resolve()),
        filename=p.name,
        size_bytes=p.stat().st_size,
        duration_seconds=duration,
        frame_count=frame_count,
        frame_count_is_exact=exact,
        width=width,
        height=height,
        sample_aspect_ratio=stream.get("sample_aspect_ratio"),
        display_aspect_ratio=stream.get("display_aspect_ratio"),
        rotation_degrees=rotation,
        fps_nominal=fps_nominal,
        fps_average=fps_average,
        frame_rate_mode=FrameRateMode.UNKNOWN,
        fps_jitter=0.0,
        codec_name=str(stream.get("codec_name") or "unknown"),
        codec_long_name=stream.get("codec_long_name"),
        pix_fmt=stream.get("pix_fmt"),
        bit_rate=bit_rate,
        profile=str(stream.get("profile")) if stream.get("profile") is not None else None,
        time_base=stream.get("time_base"),
        timing_source=TimingSource.NOMINAL_FPS,  # upgraded by probe_frames()
        has_audio=has_audio,
        metadata_tags=tags,
    )
    log.info(
        "probed %s: %dx%d %s %.3fs %.4f fps (nominal) rot=%d",
        p.name, info.width, info.height, info.codec_name,
        info.duration_seconds, info.fps_nominal, rotation,
    )
    return info


def _probe_packet_times(path: Path, timeout: float) -> list[tuple[float, bool]]:
    """(pts_time, is_keyframe) per video packet — fast, no decode.

    For H.264/HEVC in MP4/MOV one packet is one frame, and packet PTS *is* the
    presentation timestamp, so sorting by PTS yields presentation order even
    with B-frames present.
    """
    data = _run_json(
        [
            "-select_streams", "v:0",
            "-show_entries", "packet=pts_time,dts_time,flags",
            str(path),
        ],
        timeout=timeout,
    )
    out: list[tuple[float, bool]] = []
    for pkt in data.get("packets", []) or []:
        raw = pkt.get("pts_time")
        if raw in (None, "N/A"):
            raw = pkt.get("dts_time")
        try:
            t = float(raw)
        except (TypeError, ValueError):
            continue
        out.append((t, "K" in str(pkt.get("flags") or "")))
    out.sort(key=lambda item: item[0])
    return out


def _probe_frame_times(path: Path, timeout: float) -> list[tuple[float, bool]]:
    """(best_effort_timestamp_time, is_keyframe) per decoded frame.

    Slower (requires decode) but authoritative, and correct even for containers
    whose packet count differs from the frame count.
    """
    data = _run_json(
        [
            "-select_streams", "v:0",
            "-show_entries", "frame=best_effort_timestamp_time,pts_time,key_frame",
            str(path),
        ],
        timeout=timeout,
    )
    out: list[tuple[float, bool]] = []
    for fr in data.get("frames", []) or []:
        raw = fr.get("best_effort_timestamp_time")
        if raw in (None, "N/A"):
            raw = fr.get("pts_time")
        try:
            t = float(raw)
        except (TypeError, ValueError):
            continue
        out.append((t, bool(int(fr.get("key_frame") or 0))))
    out.sort(key=lambda item: item[0])
    return out


def probe_frames(
    info: VideoInfo, *, force_frame_mode: bool = False, timeout: float = 600.0
) -> tuple[list[FrameMetadata], VideoInfo]:
    """Per-frame timing. Returns the frame list plus an updated VideoInfo whose
    `timing_source`, `frame_rate_mode`, `fps_jitter` and `frame_count` reflect
    what was actually measured.
    """
    path = Path(info.path)

    use_packets = not force_frame_mode and (
        info.frame_count > PACKET_MODE_FRAME_LIMIT or info.frame_count == 0
    )
    times: list[tuple[float, bool]] = []

    if use_packets:
        try:
            times = _probe_packet_times(path, timeout)
            log.info("packet-mode timing: %d packets", len(times))
        except FFprobeError as exc:
            log.warning("packet probe failed (%s); falling back to frame probe", exc)

    if not times:
        try:
            times = _probe_frame_times(path, timeout)
            log.info("frame-mode timing: %d frames", len(times))
        except FFprobeError as exc:
            log.warning("frame probe failed (%s)", exc)

    updated = info.model_copy(deep=True)

    if len(times) >= 2:
        base = times[0][0]
        # Normalise so the first frame sits at t=0. Some containers start at a
        # non-zero PTS (edit lists, trimmed exports) and downstream kinematics
        # assume a shot-relative clock.
        deltas = [times[i + 1][0] - times[i][0] for i in range(len(times) - 1)]
        positive = [d for d in deltas if d > 1e-9]
        mean_dt = sum(positive) / len(positive) if positive else 0.0
        if mean_dt > 0:
            var = sum((d - mean_dt) ** 2 for d in positive) / len(positive)
            jitter = (var ** 0.5) / mean_dt
        else:
            jitter = 0.0

        updated.fps_jitter = jitter
        updated.frame_rate_mode = (
            FrameRateMode.VARIABLE if jitter > VFR_JITTER_THRESHOLD else FrameRateMode.CONSTANT
        )
        updated.timing_source = TimingSource.CONTAINER_PTS
        updated.frame_count = len(times)
        updated.frame_count_is_exact = True
        if mean_dt > 0:
            measured_fps = 1.0 / mean_dt
            # Keep the declared rate if it agrees; it is usually the exact
            # rational (e.g. 30000/1001) while ours is a float approximation.
            if info.fps_nominal <= 0 or abs(measured_fps - info.fps_nominal) > 0.5:
                updated.fps_average = measured_fps
        measured_duration = times[-1][0] - base + (mean_dt or 0.0)
        if measured_duration > 0:
            updated.duration_seconds = measured_duration

        frames = [
            FrameMetadata(
                frame_index=i,
                pts=None,
                time_seconds=max(0.0, t - base),
                width=info.width,
                height=info.height,
                keyframe=kf,
            )
            for i, (t, kf) in enumerate(times)
        ]
        return frames, updated

    # ---- Fallback: no usable timestamps at all. Say so (I2). ----
    fps = info.fps_average or info.fps_nominal or 30.0
    count = info.frame_count or (int(round(info.duration_seconds * fps)) if fps else 0)
    log.warning(
        "no usable container timestamps; synthesising %d frames at %.4f fps", count, fps
    )
    updated.timing_source = TimingSource.NOMINAL_FPS
    updated.frame_rate_mode = FrameRateMode.UNKNOWN
    frames = [
        FrameMetadata(
            frame_index=i,
            pts=None,
            time_seconds=i / fps,
            width=info.width,
            height=info.height,
            keyframe=(i == 0),
        )
        for i in range(max(0, count))
    ]
    return frames, updated
