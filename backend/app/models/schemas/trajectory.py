"""Recovered camera trajectory.

Naming note (invariant I11): every pose here is the **optical camera** pose. For
drone footage with an independently articulated gimbal this is NOT the aircraft
body pose, and the schema deliberately does not offer a field that could be
mistaken for one.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.schemas.motion import LensFrame


class ScaleMode(str, Enum):
    NORMALIZED = "normalized"
    """Relative translation only. Units are Blender units. NOT metres (I5)."""

    METRIC = "metric"
    """Absolute, only after explicit user calibration."""


class SolverSource(str, Enum):
    OPENCV = "opencv"
    COLMAP = "colmap"
    VGGT = "vggt"
    PERCEPTUAL = "perceptual"
    MOTION_PROXY_2D = "motion_proxy_2d"
    FUSED = "fused"
    INTERPOLATED = "interpolated"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MotionFidelity(str, Enum):
    EXACT = "exact"
    """Default. Preserve intentional micro-movement and jitter (I10)."""

    CLEAN = "clean"
    """Remove obvious solver noise only."""

    SMOOTH = "smooth"
    """Deliberate cinematic smoothing. Changes the move; opt-in only."""


class PipelineMode(str, Enum):
    AUTO = "auto"
    PHYSICAL_3D = "physical_3d"
    PERCEPTUAL_MATCH = "perceptual_match"
    FAST = "fast"
    HIGH_ACCURACY = "high_accuracy"


class CameraPose(BaseModel):
    """One camera pose in CameraPath world coordinates (right-handed, Z-up).

    `position` is the camera *centre* in world space.
    `quaternion` is [w, x, y, z], camera→world, i.e. it rotates a direction
    expressed in the camera frame into world space.
    """

    frame_index: int
    timestamp: float

    position: list[float] = Field(..., min_length=3, max_length=3)
    quaternion: list[float] = Field(..., min_length=4, max_length=4, description="[w,x,y,z]")

    fov_horizontal: float
    focal_normalized: float

    confidence: float = 0.0
    solver_source: SolverSource = SolverSource.INTERPOLATED
    is_anchor: bool = Field(
        False, description="True when this pose came from a geometric solve, not interpolation."
    )


class Kinematics(BaseModel):
    """Derived temporal quantities for one frame. Speeds are in *scale-mode units*
    per second — normalized units unless scale_mode is metric (I5)."""

    frame_index: int
    timestamp: float

    linear_velocity: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    speed: float = 0.0
    speed_normalized: float = Field(0.0, description="speed / max_speed over the shot, 0-1.")

    linear_acceleration: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    acceleration_magnitude: float = 0.0
    jerk_magnitude: float = 0.0

    angular_velocity: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Body-frame rates [yaw_rate, pitch_rate, roll_rate] in deg/s.",
    )
    angular_speed: float = Field(0.0, description="Total angular rate magnitude, deg/s.")
    angular_acceleration: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    curvature: float = Field(0.0, description="Path curvature, 1/unit.")


class MotionLabel(str, Enum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    ROLL = "roll"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRUCK_LEFT = "truck_left"
    TRUCK_RIGHT = "truck_right"
    PEDESTAL_UP = "pedestal_up"
    PEDESTAL_DOWN = "pedestal_down"
    CRANE = "crane"
    ORBIT = "orbit"
    ARC = "arc"
    PUSH_IN = "push_in"
    PULL_OUT = "pull_out"
    HANDHELD = "handheld"
    FPV = "fpv"
    DRONE_FLY_THROUGH = "drone_fly_through"
    DRONE_REVEAL = "drone_reveal"
    RISE_AND_TILT = "rise_and_tilt"
    SPIRAL = "spiral"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    MIXED_6DOF = "mixed_6dof"


class ClassifiedMove(BaseModel):
    """Post-hoc description. NEVER used to drive reconstruction (spec §13)."""

    label: MotionLabel
    strength: float = Field(..., description="0-1 how strongly this label applies.")
    start_time: float
    end_time: float
    description: str = ""


class ConfidenceReport(BaseModel):
    level: ConfidenceLevel
    score: float = Field(..., description="0-1 aggregate.")
    headline: str
    reasons: list[str] = Field(default_factory=list)

    registered_frame_ratio: float = 0.0
    persistent_track_count: int = 0
    mean_inlier_ratio: float = 0.0
    mean_reprojection_error: float | None = None
    median_reprojection_error: float | None = None
    baseline_parallax_score: float = 0.0
    bundle_adjustment_residual: float | None = None
    solver_agreement: float | None = None
    focal_stability: float = 0.0
    temporal_consistency: float = 0.0

    translation_observable: bool = True
    translation_confidence: float = 0.0
    rotation_confidence: float = 0.0
    zoom_confidence: float = 0.0


class SolverDecision(BaseModel):
    """Audit record: which backend ran, what it produced, why it was kept or
    dropped. Returned to the UI so failures are legible (invariant I9)."""

    solver: SolverSource
    attempted: bool
    succeeded: bool
    registered_frames: int = 0
    total_frames: int = 0
    duration_seconds: float = 0.0
    mean_reprojection_error: float | None = None
    confidence: float = 0.0
    selected: bool = False
    message: str = ""


class CoordinateSystem(BaseModel):
    """Explicit statement of the frame the exported poses live in. Written into
    every export so a consumer never has to guess (invariant I8)."""

    name: str = "camerapath_world"
    handedness: str = "right"
    up_axis: str = "+Z"
    forward_axis: str = "+Y"
    camera_looks_along: str = "+Y"
    camera_up: str = "+Z"
    quaternion_order: str = "wxyz"
    quaternion_maps: str = "camera_to_world"
    notes: str = (
        "Camera centre in world space. Blender export applies "
        "geometry.conventions.world_to_blender_camera()."
    )


class ShotTrajectory(BaseModel):
    """Full result for one shot."""

    shot_id: int
    frame_count: int
    duration: float
    fps: float

    scale_mode: ScaleMode = ScaleMode.NORMALIZED
    scale_units: str = Field("normalized", description="'normalized' or 'm'.")
    metric_scale_factor: float | None = None

    coordinate_system: CoordinateSystem = Field(default_factory=CoordinateSystem)

    poses: list[CameraPose] = Field(default_factory=list)
    kinematics: list[Kinematics] = Field(default_factory=list)
    lens: list[LensFrame] = Field(default_factory=list)

    confidence: ConfidenceReport
    classified_moves: list[ClassifiedMove] = Field(default_factory=list)
    summary: str = ""

    solver_decisions: list[SolverDecision] = Field(default_factory=list)
    pipeline_mode_used: PipelineMode = PipelineMode.AUTO
    motion_fidelity: MotionFidelity = MotionFidelity.EXACT

    total_path_length: float = 0.0
    total_rotation_deg: float = 0.0
