"""Container / frame level facts about the source video.

Everything here is *measured*, never assumed. In particular `time_seconds` is
derived from container presentation timestamps whenever ffprobe supplies them
(invariant I2) — `frame_index / nominal_fps` is a labelled fallback, not the
default.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TimingSource(str, Enum):
    """Where a frame's timestamp came from. Surfaced to the UI so the user can
    see whether timing is authoritative or reconstructed."""

    CONTAINER_PTS = "container_pts"
    """Read from packet/frame PTS. Authoritative."""

    NOMINAL_FPS = "nominal_fps"
    """Synthesised as index / fps because the container gave no usable PTS."""


class FrameRateMode(str, Enum):
    CONSTANT = "constant"
    VARIABLE = "variable"
    UNKNOWN = "unknown"


class FrameMetadata(BaseModel):
    """One decoded source frame."""

    frame_index: int
    pts: int | None = Field(None, description="Raw container PTS in time_base units.")
    time_seconds: float = Field(..., description="Presentation time from shot start of stream.")
    width: int
    height: int
    keyframe: bool = False
    source_path: str | None = Field(
        None, description="Set only when the frame was materialised to disk."
    )


class VideoInfo(BaseModel):
    """Authoritative container description, from ffprobe."""

    path: str
    filename: str
    size_bytes: int

    duration_seconds: float
    frame_count: int
    frame_count_is_exact: bool = Field(
        ...,
        description="True when counted from packets; False when estimated from duration*fps.",
    )

    width: int
    height: int
    sample_aspect_ratio: str | None = None
    display_aspect_ratio: str | None = None
    rotation_degrees: int = 0

    fps_nominal: float = Field(..., description="r_frame_rate — the stream's declared rate.")
    fps_average: float = Field(..., description="avg_frame_rate — total frames / duration.")
    frame_rate_mode: FrameRateMode = FrameRateMode.UNKNOWN
    fps_jitter: float = Field(
        0.0, description="Std-dev of inter-frame delta / mean delta. >~0.02 implies VFR."
    )

    codec_name: str
    codec_long_name: str | None = None
    pix_fmt: str | None = None
    bit_rate: int | None = None
    profile: str | None = None
    time_base: str | None = None

    timing_source: TimingSource
    has_audio: bool = False

    # Camera / lens metadata, when the container carries any. Treated as a PRIOR,
    # never as truth (spec §7).
    metadata_tags: dict[str, str] = Field(default_factory=dict)

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    @property
    def is_portrait(self) -> bool:
        return self.height > self.width


class ShotComplexity(str, Enum):
    TRIVIAL = "trivial"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class Shot(BaseModel):
    """A continuous run of frames with no hard edit inside it.

    Each shot receives an independent coordinate system (invariant I4).
    """

    id: int
    start_frame: int
    end_frame: int = Field(..., description="Inclusive.")
    start_time: float
    end_time: float
    confidence: float = Field(
        1.0, description="Confidence that the cut *preceding* this shot is a real cut."
    )

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class CutCandidate(BaseModel):
    """Diagnostic record for one detected (or rejected) cut. Kept so the UI can
    explain why a boundary was or was not accepted."""

    frame_index: int
    time_seconds: float
    histogram_score: float
    structural_score: float
    match_collapse_score: float
    flow_coherence_score: float
    combined_score: float
    accepted: bool
    reason: str
