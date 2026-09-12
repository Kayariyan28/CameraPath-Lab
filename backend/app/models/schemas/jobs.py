"""Job lifecycle, settings and progress."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field

from app.models.schemas.motion import MotionSignature
from app.models.schemas.trajectory import (
    MotionFidelity,
    PipelineMode,
    ScaleMode,
    ShotTrajectory,
)
from app.models.schemas.video import CutCandidate, Shot, ShotComplexity, VideoInfo


class JobState(str, Enum):
    CREATED = "created"
    UPLOADED = "uploaded"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    SOLVING = "solving"
    SOLVED = "solved"
    RENDERING = "rendering"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Stage(str, Enum):
    """User-facing pipeline stages. The strings are exactly what the UI shows,
    matching spec §2.6."""

    PREPARING_FRAMES = "Preparing frames"
    TRACKING_FEATURES = "Tracking features"
    ESTIMATING_2D_MOTION = "Estimating 2D motion"
    ESTIMATING_CAMERA_GEOMETRY = "Estimating camera geometry"
    OPTIMIZING_CAMERA_POSES = "Optimizing camera poses"
    RECOVERING_LENS_MOTION = "Recovering lens motion"
    VALIDATING_TRAJECTORY = "Validating trajectory"
    CREATING_BLENDER_SCENE = "Creating Blender scene"
    RENDERING_MP4 = "Rendering MP4"
    COMPLETE = "Complete"


class ProxyStyle(str, Enum):
    MOTION_CAGE = "motion_cage"
    DEPTH_POLES = "depth_poles"
    GROUND_GRID = "ground_grid"
    MINIMAL = "minimal"


class LensMode(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"
    VARIABLE = "variable"


class ScaleCalibrationKind(str, Enum):
    CAMERA_HEIGHT = "camera_height"
    POINT_DISTANCE = "point_distance"
    TRAVEL_DISTANCE = "travel_distance"


class ScaleCalibration(BaseModel):
    """User-supplied real-world reference that upgrades NORMALIZED → METRIC."""

    kind: ScaleCalibrationKind
    value_meters: float = Field(..., gt=0)
    note: str = ""


class SolveSettings(BaseModel):
    """Everything the user can tune. Advanced fields have safe defaults so basic
    mode never has to show them (spec §32)."""

    mode: PipelineMode = PipelineMode.AUTO
    motion_fidelity: MotionFidelity = MotionFidelity.EXACT
    scale_mode: ScaleMode = ScaleMode.NORMALIZED
    scale_calibration: ScaleCalibration | None = None

    max_analysis_resolution: int | None = Field(
        None, description="Long-edge cap for analysis frames. None = derived from memory."
    )
    keyframe_density: float = Field(
        1.0, ge=0.25, le=4.0, description="Multiplier on automatic keyframe selection."
    )
    dynamic_rejection_strength: float = Field(0.5, ge=0.0, le=1.0)
    confidence_threshold: float = Field(
        0.35, ge=0.0, le=1.0, description="Below this, AUTO falls back down the ladder."
    )

    lens_mode: LensMode = LensMode.AUTO
    fov_override_degrees: float | None = Field(None, gt=1.0, lt=179.0)

    enable_vggt: bool = False
    enable_colmap: bool = True

    # --- output ---
    proxy_style: ProxyStyle = ProxyStyle.MOTION_CAGE
    output_width: int = 1920
    output_height: int = 1080
    match_source_aspect: bool = False
    output_fps: float | None = Field(None, description="None = source fps.")
    render_trajectory_preview: bool = True


class StageProgress(BaseModel):
    stage: Stage
    progress: float = Field(0.0, ge=0.0, le=1.0)
    message: str = ""
    shot_id: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ShotAnalysis(BaseModel):
    """Cheap analysis result for one shot, available before any solve."""

    shot: Shot
    signature: MotionSignature
    complexity: ShotComplexity
    complexity_reasons: list[str] = Field(default_factory=list)
    recommended_mode: PipelineMode
    recommendation_reason: str = ""


class AnalysisResult(BaseModel):
    video: VideoInfo
    shots: list[Shot]
    cut_candidates: list[CutCandidate] = Field(default_factory=list)
    shot_analyses: list[ShotAnalysis] = Field(default_factory=list)
    analysis_resolution: list[int] = Field(
        default_factory=list, description="[width, height] actually used for analysis."
    )
    warnings: list[str] = Field(default_factory=list)


class JobOutputs(BaseModel):
    motion_proxy_mp4: str | None = None
    trajectory_preview_mp4: str | None = None
    trajectory_json: str | None = None
    trajectory_csv: str | None = None
    analysis_json: str | None = None
    camera_scene_blend: str | None = None


class Job(BaseModel):
    id: str
    state: JobState = JobState.CREATED
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    video: VideoInfo | None = None
    settings: SolveSettings = Field(default_factory=SolveSettings)

    analysis: AnalysisResult | None = None
    trajectories: list[ShotTrajectory] = Field(default_factory=list)
    outputs: JobOutputs = Field(default_factory=JobOutputs)

    current_stage: StageProgress | None = None
    stage_history: list[StageProgress] = Field(default_factory=list)

    error: str | None = None
    error_detail: str | None = None
    log: list[str] = Field(default_factory=list)


class JobSummary(BaseModel):
    """Light listing projection — avoids shipping every pose to the job list."""

    id: str
    state: JobState
    created_at: datetime
    filename: str | None = None
    duration_seconds: float | None = None
    shot_count: int | None = None
