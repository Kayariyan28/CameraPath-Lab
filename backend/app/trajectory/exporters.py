"""Trajectory exports: `trajectory.json`, `trajectory.csv`, `analysis.json`, `.chan`.

An export is the only part of this system a *third party* ever reads. Everything
else can rely on a shared understanding of the conventions; a file on disk
cannot. So every document written here is self-describing to the point of
redundancy, and three of those redundancies are load-bearing invariants:

  * **Scale (I5).** `scale_mode` *and* an explicit `units` field appear on every
    export, including the CSV (as columns, so a spreadsheet cannot separate the
    numbers from their label) and the `.chan` sidecar. Units are *derived* from
    the scale mode through `normalize.scale_units_label`, the codebase's single
    source of "m", never copied from the trajectory's `scale_units` field: a
    caller that sets `scale_mode=NORMALIZED` and `scale_units="m"` gets
    "normalized" plus a note, and a `METRIC` claim with no calibration factor
    behind it is downgraded rather than published.

  * **Frame (I8).** The `coordinate_system` block states handedness, up axis,
    camera forward axis, camera up axis, the camera's local axes, quaternion
    component order, what the quaternion maps, what `position` means, and which
    shot the frame belongs to. A consumer never has to infer a convention from
    the data. The one conversion this module performs (to Nuke's Y-up world) is
    a named, unit-tested function.

  * **Provenance (I7).** Every frame carries its own `confidence`,
    `solver_source` and `is_anchor`, so a consumer can see exactly which frames
    were geometrically solved and which were interpolated between anchors. A
    document with uniformly low confidence is a valid document; it is not
    smoothed over here.

Three things this module deliberately does not do:

  * **No resampling.** Poses are written on the timestamps they were solved on,
    straight from container PTS (I1, I2). Resampling onto a uniform frame grid
    would mean interpolating rotations at export time, outside the one module
    allowed to do that (I3).
  * **No rounding.** Floats are emitted at full round-trip precision (`repr`,
    which is JSON's own float format). Rounding a normalized position to three
    decimals quantises a trajectory whose whole extent is ~10 units into visible
    steps; the export would be measurably coarser than the solve.
  * **No invented values.** A number the pipeline did not produce is written as
    `null` / an empty cell, never as a plausible default. That covers missing
    kinematics (a zero velocity is a measurement), a vertical FOV with no aspect
    evidence, and non-finite solver output, which strict JSON cannot carry and
    which a consumer would otherwise receive as a crash or a silent zero (I12).

Multi-shot jobs get one document per shot, because a shot boundary is a change
of coordinate system and a single table spanning a cut would imply the two sides
are comparable (I4). A `trajectory_index.json` manifest names the per-shot files
and records the cut times.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np

from app.core.logging import get_logger
from app.core.paths import JobPaths, atomic_write_json, atomic_write_text
from app.geometry.rotations import quat_to_matrix
from app.models.schemas.jobs import Job, JobOutputs
from app.models.schemas.motion import LensFrame
from app.models.schemas.trajectory import (
    CameraPose,
    CoordinateSystem,
    Kinematics,
    ScaleMode,
    ShotTrajectory,
)
from app.models.schemas.video import Shot, VideoInfo
from app.trajectory.normalize import scale_units_label

log = get_logger("trajectory.exporters")

#: Bumped when the document shape changes incompatibly. Consumers pin on this.
DOCUMENT_VERSION = "1.0"

GENERATOR = "CameraPath Lab"

TRAJECTORY_STEM = "trajectory"
TRAJECTORY_INDEX_NAME = "trajectory_index.json"
ANALYSIS_JSON_NAME = "analysis.json"
CHAN_META_SUFFIX = ".meta.json"

#: Every file name `export_all` can produce for a trajectory. Used to find
#: exports left behind by an earlier run of this re-runnable stage: if shot
#: detection changed, a stale `trajectory_shot_002.json` from the old cut list
#: would otherwise sit next to the new documents looking current (I4).
GENERATED_TRAJECTORY_FILE = re.compile(
    r"^trajectory(?:_shot_\d{3,})?\.(?:json|csv|chan|chan\.meta\.json)$"
)

#: Angular rates come from `Kinematics.angular_velocity`, which the schema
#: defines as body-frame deg/s. Stated in the document so nobody has to assume.
ANGULAR_VELOCITY_UNITS = "deg/s"

#: `normalize.normalize_trajectory` subtracts the first position exactly, so a
#: re-origined shot has a first camera of exactly 0.0. Anything above this in a
#: ~10-25 unit path means the origin is somewhere else, and the document must
#: not claim otherwise.
ORIGIN_TOLERANCE = 1e-9

#: Lens and pose horizontal FOVs closer than this are the same measurement, so
#: the lens's own vertical FOV can be used verbatim. Far below any real zoom
#: step (a 1 mm change on a 50 mm lens moves the FOV by ~0.8 deg).
FOV_AGREEMENT_DEG = 1e-6


def shot_stem(shot_id: int) -> str:
    return f"{TRAJECTORY_STEM}_shot_{shot_id:03d}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Strict-JSON values
# ---------------------------------------------------------------------------


def _is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def json_safe(value: Any) -> Any:
    """Recursively convert to values strict JSON can represent, losslessly.

    `json.dumps` writes NaN and Infinity as bare tokens that are not JSON:
    `JSON.parse` in the viewer, and most other parsers, reject the entire file.
    A non-finite float is the solver saying "no value", so it becomes `null`.
    Finite floats pass through untouched, so precision is unaffected. Numpy
    scalars become Python scalars; `bool` is checked before `int` because it is
    an `int` subclass and would otherwise be written as 1.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _non_finite_frames(trajectory: ShotTrajectory) -> list[int]:
    """Frame indices whose pose or kinematics carry a NaN/Inf anywhere."""
    kinematics = {k.frame_index: k for k in trajectory.kinematics}
    bad: list[int] = []
    for pose in trajectory.poses:
        numbers: list[float] = [
            pose.timestamp,
            *pose.position,
            *pose.quaternion,
            pose.fov_horizontal,
            pose.focal_normalized,
            pose.confidence,
        ]
        kin = kinematics.get(pose.frame_index)
        if kin is not None:
            numbers.extend([*kin.linear_velocity, kin.speed, *kin.angular_velocity, kin.angular_speed])
        if not all(_is_finite(n) for n in numbers):
            bad.append(pose.frame_index)
    return bad


# ---------------------------------------------------------------------------
# Scale labelling (I5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedScale:
    """The scale label this export is allowed to make, plus why."""

    mode: ScaleMode
    units: str
    metric_scale_factor: float | None
    notes: tuple[str, ...] = ()

    def block(self) -> dict[str, Any]:
        return {
            "scale_mode": self.mode.value,
            "units": self.units,
            "metric_scale_factor": self.metric_scale_factor,
            "scale_notes": list(self.notes),
        }


def resolve_scale(trajectory: ShotTrajectory) -> ResolvedScale:
    """Decide the scale label for one trajectory, defensively.

    Monocular video does not determine absolute scale, so a metric claim has to
    be backed by a user calibration that survived into `metric_scale_factor`. A
    `METRIC` mode with no valid factor is an unsubstantiated claim and is
    downgraded to `normalized` here rather than published as metres. The export
    is the last place this can be caught, and a wrong unit label is the kind of
    error that propagates silently into someone else's shot. The downgrade only
    ever errs towards "relative": metric numbers labelled normalized are still
    correct relative numbers, whereas the reverse is a false claim.
    """
    notes: list[str] = []
    mode = trajectory.scale_mode
    factor = trajectory.metric_scale_factor

    if mode is ScaleMode.METRIC and (factor is None or not _is_finite(factor) or factor <= 0.0):
        notes.append(
            "scale_mode was 'metric' but no valid metric_scale_factor was present; "
            "exported as 'normalized' because monocular scale is unobservable "
            "without a user calibration."
        )
        mode = ScaleMode.NORMALIZED
        factor = None

    units = scale_units_label(mode)

    declared = (trajectory.scale_units or "").strip()
    if declared and declared != units:
        notes.append(
            f"trajectory.scale_units was {declared!r}, which contradicts "
            f"scale_mode={mode.value!r}; exported units are {units!r}."
        )

    if mode is ScaleMode.NORMALIZED:
        notes.append(
            "Translation is relative only: Blender units, NOT metres. "
            "Ratios, timing and rotations are exact; absolute distance is not recoverable."
        )

    return ResolvedScale(
        mode=mode,
        units=units,
        metric_scale_factor=float(factor) if factor is not None else None,
        notes=tuple(notes),
    )


def resolve_job_scale(trajectories: Sequence[ShotTrajectory]) -> ResolvedScale:
    """Job-level scale label across several shots: the most conservative one.

    Shots are solved independently, so one shot can be calibrated while another
    is not. A job-level label of "m" would then be wrong for part of the job, so
    a single normalized shot makes the whole job normalized.
    """
    normalized_units = scale_units_label(ScaleMode.NORMALIZED)
    if not trajectories:
        return ResolvedScale(
            mode=ScaleMode.NORMALIZED,
            units=normalized_units,
            metric_scale_factor=None,
            notes=("No trajectory was solved; nothing is scaled.",),
        )

    resolved = [resolve_scale(t) for t in trajectories]
    if all(r.mode is ScaleMode.METRIC for r in resolved):
        factors = {r.metric_scale_factor for r in resolved}
        notes: tuple[str, ...] = ()
        if len(factors) > 1:
            notes = ("Shots carry different metric scale factors; see each shot document.",)
        return ResolvedScale(
            mode=ScaleMode.METRIC,
            units=scale_units_label(ScaleMode.METRIC),
            metric_scale_factor=resolved[0].metric_scale_factor if len(factors) == 1 else None,
            notes=notes,
        )

    if any(r.mode is ScaleMode.METRIC for r in resolved):
        notes = (
            "At least one shot is normalized, so the job is labelled normalized; "
            "per-shot documents carry their own scale_mode.",
        )
    else:
        notes = ("Translation is relative only: Blender units, NOT metres.",)
    return ResolvedScale(
        mode=ScaleMode.NORMALIZED,
        units=normalized_units,
        metric_scale_factor=None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Self-describing blocks
# ---------------------------------------------------------------------------


def _origin_is_first_camera(trajectory: ShotTrajectory) -> bool:
    if not trajectory.poses:
        return False
    first = trajectory.poses[0].position
    return all(_is_finite(v) and abs(float(v)) <= ORIGIN_TOLERANCE for v in first)


def coordinate_system_block(
    system: CoordinateSystem,
    *,
    position_units: str,
    shot_id: int | None = None,
    origin_is_first_camera: bool | None = None,
) -> dict[str, Any]:
    """Full statement of the frame the poses live in (I8).

    Everything a consumer needs to rebuild a pose without guessing: handedness,
    which axis is up, where the camera looks, the camera's local axes, the
    quaternion component order, which direction the quaternion maps, and what
    `position` refers to. The last one matters more than it looks: a camera
    *centre* and a COLMAP translation vector are both three numbers, and
    swapping them yields a mirrored, inverted path that still looks like camera
    motion.

    `shot_id` scopes the frame to one shot. The schema's default name is the
    same for every shot, so without it two per-shot documents would claim the
    same "camerapath_world" and invite a consumer to compare coordinates across
    a cut (I4). The origin is stated only when the data shows it; otherwise it
    is "unspecified", never assumed.
    """
    block = system.model_dump(mode="json")
    block.update(
        {
            "camera_local_axes": {
                "+X": "right",
                "+Y": "forward (view direction)",
                "+Z": "up",
            },
            "rotation_matrix_columns": (
                "camera right, forward and up axes, expressed in world coordinates"
            ),
            "position_is": "camera_centre_in_world",
            "position_units": position_units,
            "angle_units": "degrees",
            "quaternion_convention": (
                f"{system.quaternion_order} order, Hamilton product, "
                f"{system.quaternion_maps}: rotates a camera-local direction into "
                "world space. Unit quaternion; q and -q denote the same rotation."
            ),
            "blender_conversion": (
                "location = position (both worlds are right-handed Z-up); "
                "rotation_quaternion = app.geometry.conventions.camerapath_quat_to_blender(q)"
            ),
            "pose_is": "optical_camera",
            "pose_note": (
                "The recovered pose is the optical camera, not a vehicle or "
                "gimbal body pose."
            ),
        }
    )
    if shot_id is not None:
        block["shot_id"] = int(shot_id)
        block["scope"] = "shot"
        block["scope_note"] = (
            "This coordinate frame belongs to one shot only. Coordinates are not "
            "comparable with any other shot's document: each shot is solved "
            "independently and a hard cut is a change of coordinate system."
        )
    if origin_is_first_camera is not None:
        block["origin"] = "first_camera_centre" if origin_is_first_camera else "unspecified"
    return block


def _trajectory_coordinate_block(trajectory: ShotTrajectory, units: str) -> dict[str, Any]:
    return coordinate_system_block(
        trajectory.coordinate_system,
        position_units=units,
        shot_id=trajectory.shot_id,
        origin_is_first_camera=_origin_is_first_camera(trajectory),
    )


def video_block(info: VideoInfo | None) -> dict[str, Any]:
    """Source facts a consumer needs to interpret the timing.

    `timing_source` is here because it is the difference between authoritative
    timestamps and reconstructed ones (I2): with `nominal_fps` the per-frame
    times were synthesised from an index, and anything measuring source timing
    off this document needs to know that.
    """
    if info is None:
        return {
            "available": False,
            "note": "No probe result on the job; video facts are unknown.",
        }
    return {
        "available": True,
        "filename": info.filename,
        "width": info.width,
        "height": info.height,
        "dimensions_are": "display (container rotation already applied)",
        "rotation_degrees": info.rotation_degrees,
        "display_aspect_ratio": info.display_aspect_ratio,
        "duration_seconds": info.duration_seconds,
        "frame_count": info.frame_count,
        "frame_count_is_exact": info.frame_count_is_exact,
        "fps_nominal": info.fps_nominal,
        "fps_average": info.fps_average,
        "frame_rate_mode": info.frame_rate_mode.value,
        "fps_jitter": info.fps_jitter,
        "timing_source": info.timing_source.value,
        "codec_name": info.codec_name,
        "pix_fmt": info.pix_fmt,
        "has_audio": info.has_audio,
    }


def _shot_block(shot: Shot | None, trajectory: ShotTrajectory) -> dict[str, Any]:
    if shot is None:
        return {
            "id": trajectory.shot_id,
            "frame_count": trajectory.frame_count,
            "duration": trajectory.duration,
            "note": "Shot boundaries unavailable; taken from the trajectory itself.",
        }
    return {
        "id": shot.id,
        "start_frame": shot.start_frame,
        "end_frame": shot.end_frame,
        "start_time": shot.start_time,
        "end_time": shot.end_time,
        "frame_count": shot.frame_count,
        "duration": shot.duration,
        "preceding_cut_confidence": shot.confidence,
    }


def _find_shot(job: Job, shot_id: int) -> Shot | None:
    if job.analysis is None:
        return None
    for shot in job.analysis.shots:
        if shot.id == shot_id:
            return shot
    return None


def _poses_outside_shot(trajectory: ShotTrajectory, shot: Shot | None) -> list[int]:
    """Frame indices that lie outside the shot's own inclusive frame range.

    A non-empty result means the trajectory reaches across a shot boundary,
    which is an upstream I4 violation. The exporter cannot repair it, but it
    must not publish it silently either.
    """
    if shot is None:
        return []
    return [
        p.frame_index
        for p in trajectory.poses
        if not (shot.start_frame <= p.frame_index <= shot.end_frame)
    ]


# ---------------------------------------------------------------------------
# Lens / FOV
# ---------------------------------------------------------------------------


def _valid_fov(value: object) -> bool:
    return _is_finite(value) and 0.0 < float(value) < 180.0  # type: ignore[arg-type]


def horizontal_fov_to_vertical(fov_horizontal_deg: float, aspect_ratio: float) -> float:
    """Vertical FOV from a horizontal one, for square pixels.

    Pinhole: `f = (w/2) / tan(hfov/2)` and `vfov = 2 atan((h/2) / f)`, so with
    `aspect = w/h` the pixel counts cancel. Equivalent to composing
    `geometry.intrinsics.fov_to_focal_pixels` with `focal_pixels_to_fov`, done
    in closed form here to avoid inventing an integer sensor size.

    Out-of-range input raises rather than being clamped: a clamped FOV is a
    different lens, and it would arrive in a compositing package looking real.
    """
    if not (_is_finite(aspect_ratio) and aspect_ratio > 0.0):
        raise ValueError(f"aspect_ratio must be positive and finite, got {aspect_ratio!r}")
    if not _valid_fov(fov_horizontal_deg):
        raise ValueError(f"fov_horizontal_deg must be in (0, 180), got {fov_horizontal_deg!r}")
    half_h = math.radians(float(fov_horizontal_deg)) / 2.0
    return math.degrees(2.0 * math.atan(math.tan(half_h) / float(aspect_ratio)))


def _lens_by_frame(trajectory: ShotTrajectory) -> dict[int, LensFrame]:
    return {lens.frame_index: lens for lens in trajectory.lens}


def _vertical_fov(
    pose: CameraPose,
    lens: LensFrame | None,
    aspect_ratio: float | None,
) -> float | None:
    """Vertical FOV for one pose, or None when there is no evidence for one.

    The pose's horizontal FOV is authoritative. When the lens frame measured the
    same horizontal FOV, its vertical FOV is used verbatim. When they disagree
    (a later stage refined the pose's FOV), the lens frame still carries the
    frame aspect, `tan(h/2) / tan(v/2)`, and the pose's own FOV is converted with
    it. Without a lens frame the video aspect is used. With neither, None: a
    fabricated vertical FOV would land in a Nuke camera as a real lens.
    """
    hfov = pose.fov_horizontal
    if not _valid_fov(hfov):
        return None
    if lens is not None and _valid_fov(lens.fov_vertical) and _valid_fov(lens.fov_horizontal):
        if abs(float(lens.fov_horizontal) - float(hfov)) <= FOV_AGREEMENT_DEG:
            return float(lens.fov_vertical)
        lens_aspect = math.tan(math.radians(lens.fov_horizontal) / 2.0) / math.tan(
            math.radians(lens.fov_vertical) / 2.0
        )
        return horizontal_fov_to_vertical(hfov, lens_aspect)
    if aspect_ratio is not None and _is_finite(aspect_ratio) and aspect_ratio > 0.0:
        return horizontal_fov_to_vertical(hfov, aspect_ratio)
    return None


def _aspect_ratio(info: VideoInfo | None) -> float | None:
    if info is None or info.height <= 0 or info.width <= 0:
        return None
    return info.width / info.height


# ---------------------------------------------------------------------------
# trajectory.json
# ---------------------------------------------------------------------------


def _vec(values: Iterable[float]) -> list[float]:
    """Plain Python floats; numpy scalars would be stringified by the JSON writer."""
    return [float(v) for v in values]


def _frame_entry(
    pose: CameraPose,
    kinematics: Kinematics | None,
    lens: LensFrame | None,
    aspect_ratio: float | None,
) -> dict[str, Any]:
    """One frame of the documented `frames` array.

    `linear_velocity` / `angular_velocity` are null rather than zero when no
    kinematics were computed for the frame: a zero vector is a measurement
    ("the camera was stationary") and claiming one we did not make would be a
    fabrication (I7). `fov_confidence` is null without a lens frame for the same
    reason; a low value there says the FOV is a prior, not a measurement.
    """
    return {
        "frame": int(pose.frame_index),
        "time": float(pose.timestamp),
        "position": _vec(pose.position),
        "quaternion": _vec(pose.quaternion),
        "fov": float(pose.fov_horizontal),
        "fov_vertical": _vertical_fov(pose, lens, aspect_ratio),
        "fov_confidence": float(lens.confidence) if lens is not None else None,
        "focal_normalized": float(pose.focal_normalized),
        "linear_velocity": _vec(kinematics.linear_velocity) if kinematics else None,
        "angular_velocity": _vec(kinematics.angular_velocity) if kinematics else None,
        "confidence": float(pose.confidence),
        "solver_source": pose.solver_source.value,
        "is_anchor": bool(pose.is_anchor),
    }


def build_trajectory_document(
    job: Job,
    trajectory: ShotTrajectory,
    *,
    shot: Shot | None = None,
    aspect_ratio: float | None = None,
) -> dict[str, Any]:
    """Build the documented trajectory document for one shot.

    `shot` and `aspect_ratio` default to whatever the job knows; they are
    parameters so the function stays usable on a partially-populated job, which
    is exactly the state a degraded run leaves it in (I12). The result is
    already strict-JSON safe.
    """
    shot = shot if shot is not None else _find_shot(job, trajectory.shot_id)
    aspect = aspect_ratio if aspect_ratio is not None else _aspect_ratio(job.video)
    scale = resolve_scale(trajectory)

    kinematics = {k.frame_index: k for k in trajectory.kinematics}
    lenses = _lens_by_frame(trajectory)

    warnings: list[str] = []
    if not trajectory.poses:
        warnings.append("No poses were solved for this shot; 'frames' is empty.")
    if not trajectory.kinematics and trajectory.poses:
        warnings.append(
            "No kinematics were computed; per-frame velocities are null rather than zero."
        )
    if trajectory.poses and len(trajectory.poses) != trajectory.frame_count:
        warnings.append(
            f"{len(trajectory.poses)} poses for a {trajectory.frame_count}-frame shot; "
            "frames are reported as solved, not padded."
        )
    outside = _poses_outside_shot(trajectory, shot)
    if outside and shot is not None:
        warnings.append(
            f"{len(outside)} pose(s) lie outside shot {shot.id}'s frame range "
            f"[{shot.start_frame}, {shot.end_frame}] (first: frame {outside[0]}); "
            "this trajectory may span a hard cut and its coordinates may not be "
            "one continuous camera (I4)."
        )
    non_finite = _non_finite_frames(trajectory)
    if non_finite:
        warnings.append(
            f"{len(non_finite)} frame(s) carried non-finite solver values "
            f"(first: frame {non_finite[0]}); those values are written as null."
        )

    first_time = float(trajectory.poses[0].timestamp) if trajectory.poses else None
    last_time = float(trajectory.poses[-1].timestamp) if trajectory.poses else None

    doc: dict[str, Any] = {
        "schema_version": DOCUMENT_VERSION,
        "generator": GENERATOR,
        "generated_at": _now_iso(),
        "job_id": job.id,
        "video": video_block(job.video),
        "shot": _shot_block(shot, trajectory),
        "coordinate_system": _trajectory_coordinate_block(trajectory, scale.units),
        "fps": float(trajectory.fps),
        "frame_count": int(trajectory.frame_count),
        "duration": float(trajectory.duration),
        "solved_frame_count": len(trajectory.poses),
        "first_time": first_time,
        "last_time": last_time,
        "time_note": (
            "'time' is the presentation time of each solved frame, taken from the "
            "container timestamps (or index/fps when video.timing_source says so). "
            "Frames are not resampled onto a uniform grid."
        ),
        "motion_fidelity": trajectory.motion_fidelity.value,
        "pipeline_mode_used": trajectory.pipeline_mode_used.value,
        "fov_axis": "horizontal",
        "fov_units": "degrees",
        "linear_velocity_frame": "world",
        "linear_velocity_units": f"{scale.units}/s",
        "angular_velocity_frame": "camera body",
        "angular_velocity_components": [
            "yaw_rate: about camera +Z (up)",
            "pitch_rate: about camera +X (right)",
            "roll_rate: about camera +Y (forward)",
        ],
        "angular_velocity_units": ANGULAR_VELOCITY_UNITS,
        "provenance_note": (
            "Per-frame 'solver_source' and 'is_anchor' say whether a pose was "
            "geometrically solved or interpolated between anchors; 'confidence' "
            "is computed from evidence and is allowed to be low."
        ),
        "confidence": trajectory.confidence.model_dump(mode="json"),
        "classified_moves": [m.model_dump(mode="json") for m in trajectory.classified_moves],
        "solver_decisions": [d.model_dump(mode="json") for d in trajectory.solver_decisions],
        "totals": {
            "path_length": float(trajectory.total_path_length),
            "path_length_units": scale.units,
            "rotation_deg": float(trajectory.total_rotation_deg),
        },
        "summary": trajectory.summary,
        "warnings": warnings,
        "frames": [
            _frame_entry(
                pose,
                kinematics.get(pose.frame_index),
                lenses.get(pose.frame_index),
                aspect,
            )
            for pose in trajectory.poses
        ],
    }
    # Scale keys last so no earlier literal in this function can shadow them.
    doc.update(scale.block())
    return json_safe(doc)


def write_trajectory_json(path: Path, doc: dict[str, Any]) -> Path:
    """Write a trajectory document atomically as strict JSON. Returns the path."""
    path = Path(path)
    atomic_write_json(path, json_safe(doc))
    return path


# ---------------------------------------------------------------------------
# trajectory.csv
# ---------------------------------------------------------------------------

#: Flat per-frame table. `scale_mode` and `units` are columns, not a header
#: comment: a comment line breaks naive parsers, and a separate metadata file
#: gets separated. Repeating the label on every row makes it impossible to read
#: the numbers without it (I5). Conventions match `trajectory.json` exactly:
#: position is the camera centre, quaternion is wxyz camera-to-world, CameraPath
#: world (right-handed, Z-up, camera looks +Y).
CSV_COLUMNS = (
    "frame",
    "time",
    "pos_x",
    "pos_y",
    "pos_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "fov_horizontal_deg",
    "fov_vertical_deg",
    "fov_confidence",
    "focal_normalized",
    "vel_x",
    "vel_y",
    "vel_z",
    "speed",
    "yaw_rate_deg_s",
    "pitch_rate_deg_s",
    "roll_rate_deg_s",
    "angular_speed_deg_s",
    "confidence",
    "solver_source",
    "is_anchor",
    "scale_mode",
    "units",
)


def _num(value: float | None) -> str:
    """Shortest representation that reads back as the identical float.

    `repr` is exact for IEEE-754 doubles in Python 3 and is what `json.dumps`
    uses, so the CSV and the JSON carry the same values bit for bit. Fixed-point
    formatting would not: `f"{x:.3f}"` on a normalized position quantises the
    trajectory. Missing and non-finite values are an empty cell, the CSV
    equivalent of the JSON document's `null`.
    """
    if value is None or not _is_finite(value):
        return ""
    return repr(float(value))


def _bool(value: bool) -> str:
    return "true" if value else "false"


def trajectory_csv_rows(
    trajectory: ShotTrajectory, *, aspect_ratio: float | None = None
) -> list[list[str]]:
    """Header row plus one row per solved pose."""
    scale = resolve_scale(trajectory)
    kinematics = {k.frame_index: k for k in trajectory.kinematics}
    lenses = _lens_by_frame(trajectory)

    rows: list[list[str]] = [list(CSV_COLUMNS)]
    for pose in trajectory.poses:
        kin = kinematics.get(pose.frame_index)
        lens = lenses.get(pose.frame_index)
        vel: Sequence[float | None] = kin.linear_velocity if kin else (None, None, None)
        ang: Sequence[float | None] = kin.angular_velocity if kin else (None, None, None)
        rows.append(
            [
                str(int(pose.frame_index)),
                _num(pose.timestamp),
                _num(pose.position[0]),
                _num(pose.position[1]),
                _num(pose.position[2]),
                _num(pose.quaternion[0]),
                _num(pose.quaternion[1]),
                _num(pose.quaternion[2]),
                _num(pose.quaternion[3]),
                _num(pose.fov_horizontal),
                _num(_vertical_fov(pose, lens, aspect_ratio)),
                _num(lens.confidence) if lens is not None else "",
                _num(pose.focal_normalized),
                _num(vel[0]),
                _num(vel[1]),
                _num(vel[2]),
                _num(kin.speed) if kin else "",
                _num(ang[0]),
                _num(ang[1]),
                _num(ang[2]),
                _num(kin.angular_speed) if kin else "",
                _num(pose.confidence),
                pose.solver_source.value,
                _bool(pose.is_anchor),
                scale.mode.value,
                scale.units,
            ]
        )
    return rows


def write_trajectory_csv(
    path: Path, trajectory: ShotTrajectory, *, aspect_ratio: float | None = None
) -> Path:
    """Write the flat per-frame table atomically. Returns the path written."""
    path = Path(path)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(trajectory_csv_rows(trajectory, aspect_ratio=aspect_ratio))
    atomic_write_text(path, buffer.getvalue())
    return path


# ---------------------------------------------------------------------------
# Nuke .chan
# ---------------------------------------------------------------------------
#
# Nuke's world is right-handed and **Y-up**, and its camera looks along its own
# -Z with +Y up: the same local axes as a Blender camera, in a differently
# oriented world. So the conversion is two independent changes, and conflating
# them is the failure mode this module's tests exist to catch:
#
#   * a change of WORLD basis, applied on the left: CameraPath +X -> Nuke +X,
#     CameraPath +Y -> Nuke -Z, CameraPath +Z (up) -> Nuke +Y. This moves both
#     positions and orientations. It is the standard Z-up -> Y-up change, a
#     -90 deg rotation about X.
#   * a relabelling of the camera's LOCAL axes, applied on the right, exactly as
#     `geometry.conventions` does for Blender.
#
# The two happen to cancel for an identity orientation (a CameraPath camera at
# rest, looking along +Y with up +Z, is a Nuke camera at rest, looking along -Z
# with up +Y), which makes `rotate = (0, 0, 0)` a hand-checkable fixture.
#
# `.chan` stores Euler angles because that is the format. They are computed per
# frame from the solved quaternion and never interpolated here (I3). What Nuke
# does between keys is outside this file, so the angles are made continuous
# frame to frame: without that a pan through +/-180 deg writes a 360 deg jump
# and Nuke's sub-frame interpolation spins the camera the long way round.

#: Left-multiplied world change of basis: `v_nuke = NUKE_WORLD_FROM_CPL @ v_cpl`.
NUKE_WORLD_FROM_CPL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])

#: Columns: the Nuke camera's local axes (right, up, backward) written in
#: CameraPath-local coordinates. Right-multiplied on a camera-to-world rotation.
NUKE_CAMERA_AXES_IN_CPL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])

#: Nuke's default Axis/Camera rotation order. "ZXY" names the order the
#: rotations are *applied* in (roll about Z, then tilt about X, then pan about
#: Y), which as a column-vector matrix product is `R = Ry(ry) . Rx(rx) . Rz(rz)`.
#: That is the standard camera gimbal: the outermost rotation is the pan about
#: world up, the innermost is the roll, and gimbal lock needs the camera to look
#: straight up or down, the rarest orientation in real footage.
NUKE_ROTATION_ORDER = "ZXY"

#: Below this |cos(rx)| the ry/rz split is degenerate (camera pointing straight
#: up or down): rx is within ~6e-6 deg of +/-90 and only one combination of the
#: remaining two angles is recoverable.
_GIMBAL_EPS = 1e-7

CHAN_COLUMNS = ("frame", "tx", "ty", "tz", "rx", "ry", "rz", "vfov")

#: Relative deviation of the per-frame interval above which the source frames
#: are not evenly spaced in time. Matches the ~0.02 jitter `VideoInfo.fps_jitter`
#: treats as variable frame rate. A `.chan` file has integer frames only, so
#: on such a source Nuke's uniform frame grid misstates timing and the sidecar
#: has to say so (I1).
CHAN_TIMING_TOLERANCE = 0.02


def _assert_rotation(matrix: np.ndarray, name: str) -> None:
    det = float(np.linalg.det(matrix))
    if not np.isclose(det, 1.0, atol=1e-9):
        raise ValueError(f"{name} is not a proper rotation (det={det:.9f})")


_assert_rotation(NUKE_WORLD_FROM_CPL, "NUKE_WORLD_FROM_CPL")
_assert_rotation(NUKE_CAMERA_AXES_IN_CPL, "NUKE_CAMERA_AXES_IN_CPL")


def _clean_zero(value: float) -> float:
    """Turn -0.0 into 0.0; every other value is returned unchanged.

    `asin(-0.0)` is -0.0, so an untilted camera would otherwise be written as a
    "-0.0" tilt. Adding +0.0 is exact for every non-zero double.
    """
    return value + 0.0


def euler_zxy_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a rotation into Nuke's ZXY angles `(rx, ry, rz)`, in degrees.

    Solves `R = Ry(ry) . Rx(rx) . Rz(rz)` (see `NUKE_ROTATION_ORDER`), whose
    middle row is `[cos(rx)sin(rz), cos(rx)cos(rz), -sin(rx)]`: that row alone
    gives rx and rz, and the third column's outer entries give ry. This is the
    principal branch, rx in [-90, 90].
    """
    m = np.asarray(matrix, dtype=np.float64)
    sin_rx = float(np.clip(-m[1, 2], -1.0, 1.0))
    rx = math.asin(sin_rx)
    cos_rx = math.cos(rx)

    if abs(cos_rx) > _GIMBAL_EPS:
        rz = math.atan2(float(m[1, 0]), float(m[1, 1]))
        ry = math.atan2(float(m[0, 2]), float(m[2, 2]))
    else:
        # Gimbal lock: only (ry - sign(rx) * rz) is determined. Attribute it all
        # to the pan, which is the angle a compositor expects to keyframe.
        rz = 0.0
        ry = math.atan2(-float(m[2, 0]), float(m[0, 0]))

    return (
        _clean_zero(math.degrees(rx)),
        _clean_zero(math.degrees(ry)),
        _clean_zero(math.degrees(rz)),
    )


def _wrap_near(angle: float, reference: float) -> float:
    """`angle` plus the multiple of 360 that lands closest to `reference`."""
    return angle + 360.0 * round((reference - angle) / 360.0)


def euler_zxy_degrees_near(
    matrix: np.ndarray, previous: tuple[float, float, float]
) -> tuple[float, float, float]:
    """The ZXY Euler triple for `matrix` closest to the previous frame's triple.

    Every rotation has two ZXY branches, `(rx, ry, rz)` and
    `(180 - rx, ry + 180, rz + 180)`, and each angle is only defined modulo 360.
    All of those are the *same* rotation, so choosing among them changes no
    pose; it only picks the representation that moves least from the previous
    frame. That is the standard Euler filter: a pan through 180 deg keeps
    counting instead of jumping, and a tilt over the vertical continues past 90
    instead of flipping pan and roll by 180.

    At gimbal lock only one combination of pan and roll is determined, so the
    previous roll is kept and the pan absorbs the rest.
    """
    rx, ry, rz = euler_zxy_degrees(matrix)
    prx, pry, prz = previous

    if abs(math.cos(math.radians(rx))) <= _GIMBAL_EPS:
        sign = 1.0 if rx > 0.0 else -1.0
        candidates = [(rx, ry + sign * prz, prz)]
    else:
        candidates = [(rx, ry, rz), (180.0 - rx, ry + 180.0, rz + 180.0)]

    best: tuple[float, float, float] | None = None
    best_cost = math.inf
    for crx, cry, crz in candidates:
        wrapped = (_wrap_near(crx, prx), _wrap_near(cry, pry), _wrap_near(crz, prz))
        cost = abs(wrapped[0] - prx) + abs(wrapped[1] - pry) + abs(wrapped[2] - prz)
        if cost < best_cost:
            best, best_cost = wrapped, cost
    assert best is not None
    return (_clean_zero(best[0]), _clean_zero(best[1]), _clean_zero(best[2]))


def camerapath_rotation_to_nuke(r_cpl: np.ndarray) -> np.ndarray:
    """CameraPath camera-to-world rotation -> Nuke camera-to-world rotation.

    `R_nuke = NUKE_WORLD_FROM_CPL . R_cpl . NUKE_CAMERA_AXES_IN_CPL`: the world
    basis change on the left, the camera's local-axis relabelling on the right.
    """
    return NUKE_WORLD_FROM_CPL @ np.asarray(r_cpl, dtype=np.float64) @ NUKE_CAMERA_AXES_IN_CPL


def camerapath_position_to_nuke(position: Sequence[float]) -> np.ndarray:
    """CameraPath world point -> Nuke world point. Scale is untouched."""
    return NUKE_WORLD_FROM_CPL @ np.asarray(position, dtype=np.float64).reshape(3)


def camerapath_pose_to_nuke(
    position: Sequence[float],
    quaternion: Sequence[float],
    *,
    previous_rotate: tuple[float, float, float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """CameraPath camera pose -> Nuke `(translate, rotate)`, rotate in degrees.

    A named, unit-tested convention change (I8). Positions are rotated into
    Nuke's Y-up world; orientations additionally have the camera's local axes
    relabelled from CameraPath's (+X right, +Y forward, +Z up) to Nuke's
    (+X right, +Y up, +Z backward). See `camerapath_rotation_to_nuke`.

    The returned angles are Nuke's ZXY Euler triple for that matrix: the
    principal branch, or with `previous_rotate` the equivalent triple closest to
    it. Scale is untouched: normalized positions stay normalized, and the
    `.chan` sidecar says so.
    """
    r_nuke = camerapath_rotation_to_nuke(quat_to_matrix(np.asarray(quaternion, dtype=np.float64)))
    t_nuke = camerapath_position_to_nuke(position)
    if previous_rotate is None:
        rotate = euler_zxy_degrees(r_nuke)
    else:
        rotate = euler_zxy_degrees_near(r_nuke, previous_rotate)
    translate = (
        _clean_zero(float(t_nuke[0])),
        _clean_zero(float(t_nuke[1])),
        _clean_zero(float(t_nuke[2])),
    )
    return translate, rotate


def chan_meta_path(path: Path) -> Path:
    return Path(str(path) + CHAN_META_SUFFIX)


def _chan_exportable(pose: CameraPose) -> bool:
    """A pose Nuke can be given: finite numbers and a non-degenerate quaternion.

    `quat_normalize` maps a zero quaternion to identity, which here would turn a
    failed solve into a confident "camera at rest". Such poses are left out of
    the file (Nuke interpolates across the gap) and named in the sidecar.
    """
    numbers = [*pose.position, *pose.quaternion]
    if not all(_is_finite(v) for v in numbers):
        return False
    return float(np.linalg.norm(np.asarray(pose.quaternion, dtype=np.float64))) > 1e-9


def _frame_spacing_is_uniform(frames: Sequence[int], times: Sequence[float]) -> bool | None:
    """Whether the per-frame interval is constant, from the poses themselves.

    None when there is no interval to measure. Gaps in the frame numbering are
    divided out, so a dropped solve does not read as variable frame rate.
    """
    per_frame: list[float] = []
    for (f0, t0), (f1, t1) in zip(zip(frames, times), zip(frames[1:], times[1:])):
        if f1 > f0 and _is_finite(t0) and _is_finite(t1):
            per_frame.append((float(t1) - float(t0)) / (f1 - f0))
    if not per_frame:
        return None
    typical = median(per_frame)
    if typical <= 0.0:
        return False
    return max(abs(d - typical) for d in per_frame) / typical <= CHAN_TIMING_TOLERANCE


def export_for_nuke_chan(
    path: Path,
    trajectory: ShotTrajectory,
    *,
    aspect_ratio: float | None = None,
    start_frame: int = 1,
) -> Path:
    """Write a Nuke `.chan` file: `frame tx ty tz rx ry rz [vfov]` per line.

    Strictly numeric lines, whitespace separated, one per exportable pose.
    Nuke's chan reader is a numeric parser and a comment line can abort the
    import. The mandatory `scale_mode` / `units` statement (I5) therefore goes in
    a sidecar `<name>.chan.meta.json`, written alongside, which also records the
    axis conversion, rotation order, and the source time of every line.

    `start_frame` is the Nuke frame number of the shot's first pose; gaps in the
    source frame indices are preserved relative to it, so a shot exports as a
    self-contained clip starting at frame 1 by default (I4).

    The `vfov` column is written only when a vertical FOV is known for every
    pose: from lens data, or derived from the horizontal FOV when `aspect_ratio`
    is given. The column is optional in the format, and an invented FOV would
    arrive in Nuke looking like a measured lens.
    """
    path = Path(path)
    lenses = _lens_by_frame(trajectory)
    poses = [p for p in trajectory.poses if _chan_exportable(p)]
    skipped = [p.frame_index for p in trajectory.poses if not _chan_exportable(p)]

    vfovs = [_vertical_fov(p, lenses.get(p.frame_index), aspect_ratio) for p in poses]
    include_vfov = bool(poses) and all(v is not None for v in vfovs)

    first_index = trajectory.poses[0].frame_index if trajectory.poses else 0
    lines: list[str] = []
    nuke_frames: list[int] = []
    previous: tuple[float, float, float] | None = None
    for pose, vfov in zip(poses, vfovs):
        (tx, ty, tz), rotate = camerapath_pose_to_nuke(
            pose.position, pose.quaternion, previous_rotate=previous
        )
        previous = rotate
        frame = int(start_frame) + (int(pose.frame_index) - int(first_index))
        nuke_frames.append(frame)
        fields = [str(frame), _num(tx), _num(ty), _num(tz), *(_num(a) for a in rotate)]
        if include_vfov:
            fields.append(_num(vfov))
        lines.append(" ".join(fields))

    atomic_write_text(path, "".join(line + "\n" for line in lines))

    scale = resolve_scale(trajectory)
    times = [float(p.timestamp) for p in poses]
    uniform = _frame_spacing_is_uniform([p.frame_index for p in poses], times)
    meta: dict[str, Any] = {
        "schema_version": DOCUMENT_VERSION,
        "generator": GENERATOR,
        "generated_at": _now_iso(),
        "describes": path.name,
        "format": "nuke_chan",
        "columns": list(CHAN_COLUMNS if include_vfov else CHAN_COLUMNS[:-1]),
        "rotation_order": NUKE_ROTATION_ORDER,
        "rotation_order_note": (
            "Set the Nuke Camera's rot order to ZXY (its default). Applied Z, then X, "
            "then Y; as a matrix product R = Ry(ry) . Rx(rx) . Rz(rz). Angles are made "
            "continuous frame to frame, so they can exceed +/-180 deg, and rx can pass "
            "+/-90 deg when the camera tilts over the vertical."
        ),
        "angle_units": "degrees",
        "vfov_units": "degrees" if include_vfov else None,
        "vfov_included": include_vfov,
        "coordinate_system": {
            "name": "nuke_world",
            "handedness": "right",
            "up_axis": "+Y",
            "camera_looks_along": "-Z",
            "camera_up": "+Y",
            "position_is": "camera_centre_in_world",
            "position_units": scale.units,
            "converted_from": trajectory.coordinate_system.name,
            "shot_id": trajectory.shot_id,
            "conversion": (
                "t_nuke = NUKE_WORLD_FROM_CPL @ t_cpl; "
                "R_nuke = NUKE_WORLD_FROM_CPL @ R_cpl @ NUKE_CAMERA_AXES_IN_CPL, "
                "NUKE_WORLD_FROM_CPL = [[1,0,0],[0,0,1],[0,-1,0]] (CameraPath +Y -> "
                "Nuke -Z, CameraPath +Z -> Nuke +Y) "
                "(app.trajectory.exporters.camerapath_pose_to_nuke)"
            ),
        },
        "shot_id": trajectory.shot_id,
        "fps": float(trajectory.fps),
        "frame_count": len(poses),
        "start_frame": int(start_frame),
        "source_first_frame": int(first_index),
        "nuke_frames": nuke_frames,
        "source_frames": [int(p.frame_index) for p in poses],
        "times": times,
        "uniform_frame_spacing": uniform,
        "skipped_source_frames": skipped,
        "pose_is": "optical_camera",
    }
    if uniform is False:
        meta["timing_note"] = (
            "Source frames are not evenly spaced in time (variable frame rate). A "
            ".chan file can only address whole frames, so Nuke will play these "
            "poses on a uniform grid; 'times' gives each line's true source time."
        )
    if skipped:
        meta["skipped_note"] = (
            "These solved frames carried non-finite or degenerate pose values and "
            "were left out rather than written as a fabricated pose."
        )
    if include_vfov:
        meta["vfov_note"] = (
            "Nuke derives focal length from vfov and the camera's vertical aperture; "
            "set the aperture aspect to the plate aspect so horizontal FOV matches."
        )
    else:
        meta["vfov_note"] = (
            "Vertical FOV omitted: no lens data and no frame aspect ratio for every "
            "pose, so no honest value was available."
        )
    meta.update(scale.block())
    atomic_write_json(chan_meta_path(path), json_safe(meta))
    return path


# ---------------------------------------------------------------------------
# analysis.json
# ---------------------------------------------------------------------------


def _shot_analysis_blocks(job: Job) -> list[dict[str, Any]]:
    analysis = job.analysis
    if analysis is None:
        return []

    by_id = {sa.shot.id: sa for sa in analysis.shot_analyses}
    blocks: list[dict[str, Any]] = []
    for shot in analysis.shots:
        block: dict[str, Any] = {"shot": shot.model_dump(mode="json")}
        sa = by_id.get(shot.id)
        if sa is not None:
            block.update(
                {
                    "signature": sa.signature.model_dump(mode="json"),
                    "signature_units": (
                        "image space (pixels at analysis resolution, image-space "
                        "degrees); not physical camera translation"
                    ),
                    "complexity": sa.complexity.value,
                    "complexity_reasons": list(sa.complexity_reasons),
                    "recommended_mode": sa.recommended_mode.value,
                    "recommendation_reason": sa.recommendation_reason,
                }
            )
        else:
            block["note"] = "No motion analysis for this shot."
        blocks.append(block)
    return blocks


def cut_records(job: Job, trajectories: Sequence[ShotTrajectory]) -> list[dict[str, Any]]:
    """The accepted hard cuts, i.e. the shot boundaries, one record each.

    A cut is the boundary *before* a shot, so the first shot contributes none;
    `time` and `frame` are those of the first frame after the cut. Falls back to
    each trajectory's first pose when shot boundaries are unavailable, which is
    still a true statement about where one coordinate system ends and the next
    begins (I4).
    """
    if job.analysis is not None and len(job.analysis.shots) > 1:
        shots = job.analysis.shots
        return [
            {
                "time": float(after.start_time),
                "frame": int(after.start_frame),
                "shot_before": before.id,
                "shot_after": after.id,
                "confidence": float(after.confidence),
                "source": "shot_detection",
            }
            for before, after in zip(shots, shots[1:])
        ]

    records: list[dict[str, Any]] = []
    trajs = list(trajectories)
    for before, after in zip(trajs, trajs[1:]):
        if after.poses:
            records.append(
                {
                    "time": float(after.poses[0].timestamp),
                    "frame": int(after.poses[0].frame_index),
                    "shot_before": before.shot_id,
                    "shot_after": after.shot_id,
                    "confidence": None,
                    "source": "first_pose_of_following_shot",
                }
            )
    return records


def cut_times(job: Job, trajectories: Sequence[ShotTrajectory]) -> list[float]:
    """Times of the accepted hard cuts. See `cut_records`."""
    return [record["time"] for record in cut_records(job, trajectories)]


def _overlapping_shots(trajectories: Sequence[ShotTrajectory]) -> list[tuple[int, int]]:
    """Pairs of trajectories whose solved frame ranges overlap (an I4 breach)."""
    spans = sorted(
        (min(p.frame_index for p in t.poses), max(p.frame_index for p in t.poses), t.shot_id)
        for t in trajectories
        if t.poses
    )
    return [
        (a_id, b_id)
        for (_, a_end, a_id), (b_start, _, b_id) in zip(spans, spans[1:])
        if b_start <= a_end
    ]


def build_analysis_document(
    job: Job, *, trajectories: Sequence[ShotTrajectory] | None = None
) -> dict[str, Any]:
    """Everything the run *decided*, in one auditable document.

    Video facts, shot boundaries with the cut scores behind them (accepted and
    rejected), per-shot motion signatures, every solver attempt with its
    outcome (I9), the confidence report with its reasons (I7), and the
    warnings. A LOW-confidence run produces a complete document saying so.
    """
    trajs = list(job.trajectories if trajectories is None else trajectories)
    analysis = job.analysis
    scale = resolve_job_scale(trajs)

    warnings: list[str] = list(analysis.warnings) if analysis is not None else []
    if analysis is None:
        warnings.append("Analysis stage did not complete; shot and signature data are absent.")
    if not trajs:
        warnings.append("No trajectory was solved.")
    for a_id, b_id in _overlapping_shots(trajs):
        warnings.append(
            f"Shots {a_id} and {b_id} have overlapping solved frame ranges; a trajectory "
            "may span a hard cut (I4)."
        )
    if job.error:
        warnings.append(f"Job error: {job.error}")

    traj_blocks: list[dict[str, Any]] = []
    for traj in trajs:
        traj_scale = resolve_scale(traj)
        block = {
            "shot_id": traj.shot_id,
            "frame_count": traj.frame_count,
            "solved_frame_count": len(traj.poses),
            "anchor_frame_count": sum(1 for p in traj.poses if p.is_anchor),
            "duration": float(traj.duration),
            "fps": float(traj.fps),
            "motion_fidelity": traj.motion_fidelity.value,
            "pipeline_mode_used": traj.pipeline_mode_used.value,
            "coordinate_system": _trajectory_coordinate_block(traj, traj_scale.units),
            "confidence": traj.confidence.model_dump(mode="json"),
            "solver_decisions": [d.model_dump(mode="json") for d in traj.solver_decisions],
            "classified_moves": [m.model_dump(mode="json") for m in traj.classified_moves],
            "totals": {
                "path_length": float(traj.total_path_length),
                "path_length_units": traj_scale.units,
                "rotation_deg": float(traj.total_rotation_deg),
            },
            "summary": traj.summary,
        }
        block.update(traj_scale.block())
        traj_blocks.append(block)

    doc: dict[str, Any] = {
        "schema_version": DOCUMENT_VERSION,
        "generator": GENERATOR,
        "generated_at": _now_iso(),
        "job": {
            "id": job.id,
            "state": job.state.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "error": job.error,
            "error_detail": job.error_detail,
        },
        "settings": job.settings.model_dump(mode="json"),
        "video": video_block(job.video),
        "analysis_resolution": list(analysis.analysis_resolution) if analysis else [],
        "shot_count": len(analysis.shots) if analysis else len(trajs),
        "cut_times": cut_times(job, trajs),
        "cuts": cut_records(job, trajs),
        "shots": _shot_analysis_blocks(job),
        "cut_decisions": (
            [c.model_dump(mode="json") for c in analysis.cut_candidates] if analysis else []
        ),
        "confidence_by_shot": [
            {
                "shot_id": t.shot_id,
                "level": t.confidence.level.value,
                "score": float(t.confidence.score),
                "headline": t.confidence.headline,
                "translation_observable": t.confidence.translation_observable,
            }
            for t in trajs
        ],
        "trajectories": traj_blocks,
        "stage_history": [s.model_dump(mode="json") for s in job.stage_history],
        "warnings": warnings,
    }
    doc.update(scale.block())
    return json_safe(doc)


def write_analysis_json(path: Path, doc: dict[str, Any]) -> Path:
    """Write the analysis document atomically as strict JSON. Returns the path."""
    path = Path(path)
    atomic_write_json(path, json_safe(doc))
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class _WrittenShot:
    trajectory: ShotTrajectory
    json_path: Path
    csv_path: Path
    chan_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def paths(self) -> list[Path]:
        written = [self.json_path, self.csv_path]
        if self.chan_path is not None:
            written += [self.chan_path, chan_meta_path(self.chan_path)]
        return written


def _index_document(
    job: Job,
    written: Sequence[_WrittenShot],
    trajectories: Sequence[ShotTrajectory],
    *,
    warnings: Sequence[str],
    removed_stale: Sequence[str],
) -> dict[str, Any]:
    """Manifest naming every per-shot document and recording the cut times.

    Paths are file names relative to the index, so the outputs directory can be
    moved or handed to someone else without rewriting them.
    """
    scale = resolve_job_scale([w.trajectory for w in written] or list(trajectories))
    shots: list[dict[str, Any]] = []
    for entry in written:
        traj = entry.trajectory
        shot = _find_shot(job, traj.shot_id)
        shot_scale = resolve_scale(traj)
        block: dict[str, Any] = {
            "shot_id": traj.shot_id,
            "shot": _shot_block(shot, traj),
            "fps": float(traj.fps),
            "frame_count": traj.frame_count,
            "solved_frame_count": len(traj.poses),
            "duration": float(traj.duration),
            "confidence_level": traj.confidence.level.value,
            "confidence_score": float(traj.confidence.score),
            "coordinate_system": _trajectory_coordinate_block(traj, shot_scale.units),
            "trajectory_json": entry.json_path.name,
            "trajectory_csv": entry.csv_path.name,
            "trajectory_chan": entry.chan_path.name if entry.chan_path else None,
            "trajectory_chan_meta": (
                chan_meta_path(entry.chan_path).name if entry.chan_path else None
            ),
        }
        block.update(shot_scale.block())
        if entry.notes:
            block["notes"] = list(entry.notes)
        shots.append(block)

    doc: dict[str, Any] = {
        "schema_version": DOCUMENT_VERSION,
        "generator": GENERATOR,
        "generated_at": _now_iso(),
        "job_id": job.id,
        "video": video_block(job.video),
        "shot_count": len(shots),
        "cut_times": cut_times(job, trajectories),
        "cuts": cut_records(job, trajectories),
        "cut_time_note": (
            "Time of the first frame after each accepted hard cut. Each shot is an "
            "independent coordinate system and its own document; a trajectory never "
            "spans a cut."
        ),
        "shots": shots,
        "removed_stale_exports": list(removed_stale),
        "warnings": list(warnings),
    }
    doc.update(scale.block())
    return json_safe(doc)


def _remove_stale_exports(outputs_dir: Path, keep: set[Path]) -> list[str]:
    """Delete trajectory exports from an earlier run that this run did not write.

    Only names matching `GENERATED_TRAJECTORY_FILE` directly inside the outputs
    directory are candidates, so renders, the .blend and anything a user put
    there are never touched.
    """
    keep_names = {p.name for p in keep}
    removed: list[str] = []
    try:
        candidates = sorted(outputs_dir.iterdir())
    except OSError:
        log.exception("could not list %s for stale exports", outputs_dir)
        return removed
    for candidate in candidates:
        if (
            candidate.is_file()
            and GENERATED_TRAJECTORY_FILE.match(candidate.name)
            and candidate.name not in keep_names
        ):
            try:
                candidate.unlink()
                removed.append(candidate.name)
            except OSError:
                log.exception("could not remove stale export %s", candidate)
    return removed


def export_all(
    job: Job, trajectories: Sequence[ShotTrajectory], paths: JobPaths
) -> JobOutputs:
    """Write every trajectory export plus `analysis.json`, and return the paths.

    One document set per shot (`trajectory.json/.csv/.chan` for a single-shot
    job, `trajectory_shot_000.*` and friends for several) plus a
    `trajectory_index.json` manifest. Existing output paths on the job owned by
    other stages (the rendered MP4s, the .blend) are carried through untouched;
    the data-export slots are set from what this run actually wrote, and cleared
    when nothing was, so a slot never points at a previous run's file.

    A failure writing one shot is recorded and the rest still export (I12): a
    half-exported job is more useful than an exception, and every skipped shot
    is named in the index's warnings.
    """
    trajectories = list(trajectories)
    outputs_dir = paths.outputs_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)

    outputs = job.outputs.model_copy(deep=True)
    aspect = _aspect_ratio(job.video)
    multi = len(trajectories) > 1
    warnings: list[str] = []
    written: list[_WrittenShot] = []

    stems = [shot_stem(t.shot_id) if multi else TRAJECTORY_STEM for t in trajectories]
    if len(set(stems)) != len(stems):
        warnings.append(
            "Several trajectories share a shot id; only the first of each is exported "
            "so that no document silently overwrites another shot."
        )

    seen_stems: set[str] = set()
    for traj, stem in zip(trajectories, stems):
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        try:
            json_path = write_trajectory_json(
                outputs_dir / f"{stem}.json",
                build_trajectory_document(job, traj, aspect_ratio=aspect),
            )
            csv_path = write_trajectory_csv(
                outputs_dir / f"{stem}.csv", traj, aspect_ratio=aspect
            )
        except Exception as exc:  # noqa: BLE001 - an export failure must not kill the job
            log.exception("trajectory export failed for shot %s", traj.shot_id)
            warnings.append(f"Shot {traj.shot_id}: export failed ({exc!r}).")
            continue

        entry = _WrittenShot(trajectory=traj, json_path=json_path, csv_path=csv_path)
        try:
            entry.chan_path = export_for_nuke_chan(
                outputs_dir / f"{stem}.chan", traj, aspect_ratio=aspect
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("chan export failed for shot %s", traj.shot_id)
            note = f"Nuke .chan export failed ({exc!r}); JSON and CSV were written."
            entry.notes.append(note)
            warnings.append(f"Shot {traj.shot_id}: {note}")
        written.append(entry)

    keep = {p for entry in written for p in entry.paths()}
    removed_stale = _remove_stale_exports(outputs_dir, keep)

    index_path: Path | None = outputs_dir / TRAJECTORY_INDEX_NAME
    try:
        atomic_write_json(
            index_path,
            _index_document(
                job, written, trajectories, warnings=warnings, removed_stale=removed_stale
            ),
        )
    except Exception:  # noqa: BLE001
        log.exception("trajectory index write failed")
        index_path = None

    # JobOutputs has one slot per kind, so a multi-shot job points its JSON slot
    # at the manifest: it names every per-shot document, including the .chan
    # files, which the schema has no field for. There is no manifest equivalent
    # for CSV, so that slot names the first shot and the manifest names the rest,
    # rather than concatenating shots into one table, which would imply their
    # coordinate systems were comparable (I4).
    if written:
        outputs.trajectory_json = str(
            index_path if (multi and index_path is not None) else written[0].json_path
        )
        outputs.trajectory_csv = str(written[0].csv_path)
    else:
        outputs.trajectory_json = None
        outputs.trajectory_csv = None

    try:
        outputs.analysis_json = str(
            write_analysis_json(
                outputs_dir / ANALYSIS_JSON_NAME,
                build_analysis_document(job, trajectories=trajectories),
            )
        )
    except Exception:  # noqa: BLE001
        log.exception("analysis export failed")
        outputs.analysis_json = None

    return outputs
