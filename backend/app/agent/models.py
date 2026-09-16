"""Response models for the agent surface, and the one error type it raises.

These models are the contract. The MCP server declares them as tool return types
so FastMCP emits `structuredContent`, and the CLI dumps the same objects in JSON
mode — which is why a tool result and a `cpl` invocation agree field for field.

Everything here is a *projection*. No model computes a number: each field is
either copied verbatim from the pipeline or aggregated from pipeline values with
min/max/mean. A quantity the pipeline did not produce is `None`, never a
plausible substitute — an agent that cannot tell "we did not measure this" from
"this is zero" will state the second as fact (I5/I6/I7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

RunnerMode = Literal["delegated", "local"]
SourceMode = Literal["link", "copy"]
DescriptionStyle = Literal["prompt", "technical", "brief"]

#: Every value `AgentError.code` can take. The CLI maps these onto exit codes
#: (§7 of the spec) and the MCP layer puts them in the tool error text, so an
#: agent can branch on a stable string rather than parsing prose.
ERROR_CODES = (
    "invalid_input_path",
    "unsupported_media_type",
    "file_too_large",
    "job_not_found",
    "job_not_ready",
    "job_failed",
    "job_cancelled",
    "job_busy",
    "environment_unavailable",
    "render_unavailable",
    "backend_unreachable",
    "timeout",
    "internal",
)


class AgentError(Exception):
    """The only exception any public `AgentService` method raises.

    Transports must never leak a traceback: an MCP client sees an unhandled
    exception as a broken server, and a shell agent sees it as unparseable
    output. Both front ends catch exactly this type and render it — a tool error
    with `isError: true`, or a JSON failure envelope with an exit code.
    """

    def __init__(self, code: str, message: str, *, detail: str = "", hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail
        self.hint = hint

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "hint": self.hint,
        }

    def __str__(self) -> str:  # what an MCP client shows the model
        parts = [f"[{self.code}] {self.message}"]
        if self.detail:
            parts.append(self.detail)
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return " ".join(parts)


@dataclass(frozen=True)
class Stages:
    """Which parts of the chain one `start` call should run.

    A frozen triple rather than three booleans threaded through every signature:
    "analysis only", "solve without render" and the full chain are the three
    things callers actually ask for, and one object keeps them expressible in a
    single argument that is also echoed back in `JobStatus.stages_requested`.
    """

    analyze: bool = True
    solve: bool = True
    render: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {"analyze": self.analyze, "solve": self.solve, "render": self.render}


ANALYSIS_ONLY = Stages(analyze=True, solve=False, render=False)
FULL_CHAIN = Stages(analyze=True, solve=True, render=True)


# --------------------------------------------------------------------- video


class VideoFacts(BaseModel):
    """Measured container facts, straight from ffprobe via `VideoInfo`."""

    filename: str
    width: int
    height: int
    duration_seconds: float
    frame_count: int
    frame_count_is_exact: bool
    fps_average: float
    fps_nominal: float
    frame_rate_mode: str
    fps_jitter: float
    timing_source: str
    codec_name: str
    rotation_degrees: int = 0


# --------------------------------------------------------------- environment


class HostFacts(BaseModel):
    chip: str
    platform: str
    total_memory_gb: float
    available_memory_gb: float
    cpu_cores_total: int
    disk_free_gb: float


class ResourcePolicyFacts(BaseModel):
    analysis_long_edge: int
    geometry_long_edge: int
    colmap_max_num_features: int
    worker_threads: int
    reason: str


class EnvironmentReport(BaseModel):
    ready: bool = Field(..., description="ffmpeg and ffprobe are both present.")
    runner_mode: RunnerMode
    backend_url: str | None = None
    ui_base_url: str
    workspace_dir: str
    can_ingest: bool
    can_reconstruct_3d: bool
    can_render: bool
    can_use_vggt: bool
    blender_version: str | None = None
    ffmpeg_version: str | None = None
    pycolmap_version: str | None = None
    host: HostFacts
    resource_policy: ResourcePolicyFacts
    warnings: list[str] = Field(default_factory=list)
    supported_video_extensions: list[str] = Field(default_factory=list)
    max_input_bytes: int = 0
    input_roots: list[str] = Field(
        default_factory=list,
        description="Allowlisted input directories. Empty = any readable path is accepted.",
    )


# ------------------------------------------------------------------ job refs


class JobRef(BaseModel):
    job_id: str
    state: str
    runner_mode: RunnerMode
    ui_url: str
    job_dir: str
    source_video: str
    source_attached_by: Literal["link", "copy"]
    video: VideoFacts | None = None
    stages_requested: dict[str, bool]
    poll_with: str = "wait_for_job"
    estimated_runtime_hint: str = ""


class StageFacts(BaseModel):
    name: str | None = None
    message: str = ""
    shot_id: int | None = None
    started_at: str | None = None


class JobStatus(BaseModel):
    job_id: str
    state: str
    terminal: bool
    runner_mode: RunnerMode
    stage: StageFacts | None = None
    progress: float | None = None
    stages_requested: dict[str, bool] = Field(default_factory=dict)
    stages_done: dict[str, bool] = Field(default_factory=dict)
    shot_count: int | None = None
    elapsed_seconds: float | None = None
    error: str | None = None
    error_detail: str | None = None
    render_skipped_reason: str | None = None
    ui_url: str
    job_dir: str
    outputs_ready: bool = False
    log_tail: list[str] = Field(default_factory=list)
    next_step: str = ""
    cancel_requested: bool = False

    # Only set by `wait_for_job` / `cpl wait`.
    waited_seconds: float | None = None
    timed_out: bool | None = None
    timeout_clamped_to: float | None = None


class JobSummaryOut(BaseModel):
    job_id: str
    state: str
    created_at: str
    filename: str | None = None
    duration_seconds: float | None = None
    shot_count: int | None = None
    ui_url: str
    job_dir: str


class JobListResult(BaseModel):
    jobs: list[JobSummaryOut] = Field(default_factory=list)
    workspace_dir: str
    runner_mode: RunnerMode


class DeleteResult(BaseModel):
    job_id: str
    deleted: bool
    freed_bytes: int = 0


# ------------------------------------------------------------- motion report


class SolverAttempt(BaseModel):
    solver: str
    attempted: bool
    succeeded: bool
    selected: bool
    registered_frames: int
    total_frames: int
    duration_seconds: float
    mean_reprojection_error: float | None = None
    confidence: float
    message: str = ""


class ConfidenceBlock(BaseModel):
    level: Literal["high", "medium", "low"]
    score: float
    headline: str
    reasons: list[str] = Field(default_factory=list)
    registered_frame_ratio: float
    persistent_track_count: int
    mean_inlier_ratio: float
    mean_reprojection_error: float | None = None
    median_reprojection_error: float | None = None
    baseline_parallax_score: float
    bundle_adjustment_residual: float | None = None
    solver_agreement: float | None = None
    focal_stability: float
    temporal_consistency: float


class TranslationBlock(BaseModel):
    observable: bool
    confidence: float
    units: str
    total_path_length: float | None = None
    mean_speed: float | None = None
    peak_speed: float | None = None
    peak_speed_time: float | None = None
    speed_units: str
    not_observable_reason: str | None = None


class RotationBlock(BaseModel):
    confidence: float
    total_rotation_deg: float
    mean_angular_speed_deg_s: float | None = None
    peak_angular_speed_deg_s: float | None = None
    peak_angular_speed_time: float | None = None
    units: str = "deg"


class LensBlock(BaseModel):
    fov_horizontal_start_deg: float | None = None
    fov_horizontal_end_deg: float | None = None
    fov_horizontal_min_deg: float | None = None
    fov_horizontal_max_deg: float | None = None
    focal_ratio: float | None = Field(
        None,
        description="tan(fov_start/2) / tan(fov_end/2) — the quantity classify.py thresholds.",
    )
    zoom: Literal["none", "in", "out"] = "none"
    zoom_confidence: float = 0.0
    lens_mode_used: str = "auto"
    focal_is_estimated: bool = True
    intrinsics_provenance: str = ""


class MoveOut(BaseModel):
    label: str
    strength: float
    start_time: float
    end_time: float
    description: str = ""


class PoseOut(BaseModel):
    frame_index: int
    timestamp: float
    position: list[float]
    quaternion: list[float] = Field(..., description="[w, x, y, z], camera_to_world.")
    fov_horizontal: float
    focal_normalized: float
    confidence: float
    solver_source: str
    is_anchor: bool


class ShotMotionReport(BaseModel):
    # --- identity & timing ---
    shot_id: int
    shot_index: int
    start_time: float
    end_time: float
    duration_seconds: float
    frame_count: int
    fps: float
    timing_source: str | None = None
    frame_rate_mode: str | None = None
    fps_jitter: float | None = None
    cut_confidence: float | None = Field(
        None, description="Confidence that the cut *preceding* this shot is real."
    )

    # --- solver ---
    solver_used: str
    pipeline_mode_used: str
    motion_fidelity: str
    solver_attempts: list[SolverAttempt] = Field(default_factory=list)
    solver_explanation: str = ""

    # --- measurement quality ---
    confidence: ConfidenceBlock
    translation: TranslationBlock
    rotation: RotationBlock
    lens: LensBlock

    # --- post-hoc description ---
    moves: list[MoveOut] = Field(default_factory=list)
    primary_move: str | None = None
    is_static: bool = False
    pipeline_summary: str = ""
    prompt_description: str = ""

    # --- optional raw poses ---
    poses: list[PoseOut] = Field(default_factory=list)
    pose_stride: int = 1
    poses_are_strided_subset: bool = False
    anchor_frame_count: int = 0


class ScaleBlock(BaseModel):
    mode: Literal["normalized", "metric"]
    units: str
    metric_scale_factor: float | None = None
    statement: str


class JobMotionReport(BaseModel):
    job_id: str
    ui_url: str
    job_dir: str
    shot_count: int
    video: VideoFacts | None = None
    coordinate_system: dict[str, Any] = Field(default_factory=dict)
    scale: ScaleBlock
    warnings: list[str] = Field(default_factory=list)
    shots: list[ShotMotionReport] = Field(default_factory=list)


# ------------------------------------------------------------- description


class ShotDescription(BaseModel):
    shot_id: int
    start_time: float
    end_time: float
    text: str


class DescriptionResult(BaseModel):
    text: str
    style: DescriptionStyle
    shot_count: int
    per_shot: list[ShotDescription] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    translation_observable_all: bool = True
    confidence_level_min: str = "high"
    scale_units: str = "normalized"
    derived_from: list[str] = Field(
        default_factory=list,
        description="Measured fields the text used, so every clause can be audited.",
    )


# ----------------------------------------------------------------- outputs


class FileFacts(BaseModel):
    path: str | None = None
    size_bytes: int | None = None
    exists: bool = False


class VerifiedMp4(BaseModel):
    path: str | None = None
    size_bytes: int | None = None
    exists: bool = False
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    timing_matches_source: bool | None = None
    verify_reason: str | None = Field(
        None, description="Why the numbers above are null, when they are."
    )


class ShotOutputs(BaseModel):
    shot_id: int
    trajectory_json: FileFacts
    trajectory_csv: FileFacts
    chan: FileFacts
    chan_meta_json: FileFacts
    motion_proxy_mp4: FileFacts


class OutputFile(BaseModel):
    name: str
    path: str
    size_bytes: int
    kind: str


class OutputsReport(BaseModel):
    job_id: str
    job_dir: str
    outputs_dir: str
    ui_url: str
    motion_proxy_mp4: VerifiedMp4
    trajectory_json: FileFacts
    trajectory_csv: FileFacts
    trajectory_index_json: FileFacts
    analysis_json: FileFacts
    camera_scene_blend: FileFacts
    trajectory_preview_mp4: FileFacts
    per_shot: list[ShotOutputs] = Field(default_factory=list)
    files: list[OutputFile] = Field(default_factory=list)
    render_skipped_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- samples


class TrajectorySamples(BaseModel):
    job_id: str
    shot_id: int
    columns: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    stride: int = 1
    is_subsample: bool = False
    truncated: bool = False
    coordinate_system: dict[str, Any] = Field(default_factory=dict)
    source_file: str | None = None


class ExportText(BaseModel):
    job_id: str
    filename: str
    path: str
    mime_type: str
    size_bytes: int
    truncated: bool
    text: str
