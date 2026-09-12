"""Image-space motion description.

CRITICAL (invariant I6): everything in this module is measured in *pixels* and
*image-space degrees*. None of it is physical camera translation. `dx_pixels` of
+40 means the image content shifted 40 px; it does not mean the camera trucked
right, and it must never be reported as such. The conversion from these
quantities to a physical pose only ever happens in `solvers/` and `trajectory/`,
where a camera model and a geometric solve are available.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MotionModel(str, Enum):
    """Which 2D model successfully explained the transition."""

    NONE = "none"
    TRANSLATION = "translation"
    EUCLIDEAN = "euclidean"
    AFFINE = "affine"
    HOMOGRAPHY = "homography"


class MotionFrame(BaseModel):
    """Motion of frame N relative to frame N-1, in image space.

    One of these exists for every frame transition in a shot — this is the dense
    temporal signal that survives even when 3D reconstruction only runs on
    keyframes. It is what preserves acceleration, jitter and direction changes.
    """

    frame_index: int = Field(..., description="Index of the *target* frame (N).")
    timestamp: float = Field(..., description="Presentation time of frame N.")
    dt: float = Field(..., description="Seconds since frame N-1. Real, not 1/fps.")

    # --- dominant rigid-ish model, in pixels at analysis resolution ---
    dx_pixels: float = 0.0
    dy_pixels: float = 0.0
    rotation_deg: float = Field(0.0, description="In-plane (roll-like) image rotation.")
    scale: float = Field(1.0, description="Isotropic image scale. >1 = content grew.")

    model_used: MotionModel = MotionModel.NONE
    affine: list[float] | None = Field(None, description="Row-major 2x3, if estimated.")
    homography: list[float] | None = Field(None, description="Row-major 3x3, if estimated.")

    # --- flow field statistics ---
    flow_magnitude: float = Field(0.0, description="Median flow magnitude, px.")
    flow_magnitude_p90: float = 0.0
    median_flow: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    radial_flow: float = Field(
        0.0,
        description="Mean outward flow component about the image centre, px. "
        "Positive = content expanding. Caused by dolly-in OR zoom-in; "
        "the two are only separable with parallax evidence.",
    )
    flow_divergence: float = Field(0.0, description="Mean ∂u/∂x + ∂v/∂y, 1/px.")
    flow_curl: float = Field(0.0, description="Mean ∂v/∂x - ∂u/∂y, 1/px.")

    # --- quality ---
    tracks_in: int = 0
    tracks_survived: int = 0
    inlier_ratio: float = 0.0
    confidence: float = Field(0.0, description="0-1 trust in this transition's model.")

    # --- dynamic-content diagnostics (spec §6) ---
    background_track_count: int = 0
    rejected_track_count: int = 0
    dynamic_area_fraction: float = Field(
        0.0, description="Fraction of tracked area disagreeing with the dominant model."
    )

    # Guard against pydantic treating `model_used` as a protected namespace.
    model_config = {"protected_namespaces": ()}


class LensFrame(BaseModel):
    """Per-frame lens state. Focal length is NOT assumed constant (spec §7)."""

    frame_index: int
    timestamp: float
    focal_normalized: float = Field(
        ..., description="focal_px / image_long_edge_px. Resolution independent."
    )
    fov_horizontal: float = Field(..., description="Degrees.")
    fov_vertical: float = Field(..., description="Degrees.")
    confidence: float = 0.0
    is_estimated: bool = Field(
        True, description="False when taken from a trusted metadata prior."
    )


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics at a stated resolution, plus optional distortion."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    focal_normalized: float
    fov_horizontal: float
    fov_vertical: float

    distortion: list[float] = Field(
        default_factory=list, description="OpenCV order k1,k2,p1,p2,k3 — empty if unmodelled."
    )

    confidence: float = 0.0
    source: str = Field("estimated", description="metadata_prior | estimated | user_override | default_prior")

    def scaled_to(self, width: int, height: int) -> CameraIntrinsics:
        """Rescale intrinsics to a different image resolution."""
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            focal_normalized=self.focal_normalized,
            fov_horizontal=self.fov_horizontal,
            fov_vertical=self.fov_vertical,
            distortion=list(self.distortion),
            confidence=self.confidence,
            source=self.source,
        )


class MotionSignature(BaseModel):
    """Aggregate description of a whole shot's image-space motion. Used for
    pipeline routing (is this shot degenerate?) and shot-complexity display."""

    frame_count: int
    duration: float

    mean_flow_magnitude: float = 0.0
    peak_flow_magnitude: float = 0.0
    mean_inlier_ratio: float = 0.0

    total_image_rotation_deg: float = 0.0
    net_dx_pixels: float = 0.0
    net_dy_pixels: float = 0.0
    cumulative_path_pixels: float = 0.0

    mean_radial_flow: float = 0.0
    net_scale_change: float = 1.0

    # --- degeneracy indicators. These decide PHYSICAL vs PERCEPTUAL routing. ---
    parallax_score: float = Field(
        0.0,
        description="0-1. Evidence of depth-dependent flow, i.e. real translation. "
        "Low = homography explains everything = rotation/zoom only.",
    )
    homography_dominance: float = Field(
        0.0, description="0-1. Fraction of transitions fully explained by a homography."
    )
    rotation_dominance: float = Field(0.0, description="0-1. Motion is mostly rotational.")
    texture_score: float = Field(0.0, description="0-1. Trackable-feature density.")
    blur_score: float = Field(0.0, description="0-1. 1 = sharp, 0 = heavy motion blur.")
    jitter_score: float = Field(
        0.0, description="0-1. High-frequency motion energy — handheld indicator."
    )
