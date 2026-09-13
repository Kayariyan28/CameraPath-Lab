"""Scale modes: `normalized` by default, `metric` only after a user calibration.

Invariant I5, architecture section 7. A single monocular camera cannot determine
absolute scale: the same image sequence is produced by a small camera move in a
small scene and a large move in a large scene. Every monocular translation is
therefore *relative*, and this module's job is to be honest about that while
still handing Blender numbers of a convenient size.

Two things follow, and they are the whole design:

  * Normalization is a **similarity transform only** — a translation of the
    origin and one global scale factor. Segment-length ratios, all timing and
    all orientations come through untouched, so the shape of the move and its
    speed profile survive exactly. Nothing here may touch a quaternion.
  * The unit label is derived from the scale *mode*, never from the numbers.
    `scale_units_label` is the only place a "m" can come from, and it only ever
    returns it for `ScaleMode.METRIC`.

The subtle failure this module has to avoid is amplifying noise. "Scale the path
length into the 10-25 unit range" applied blindly to a locked-off tripod shot
takes a few thousandths of a unit of solver jitter and turns it into a twenty-unit
random walk — a large fabricated camera move, presented with the same confidence
as a real one. So the path length is only trusted as a measure of *travel* when
the path has temporal coherence (see `_coherent_path_ratio`); otherwise the shot
is treated as near-static and is never amplified.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.models.schemas.jobs import ScaleCalibration, ScaleCalibrationKind
from app.models.schemas.trajectory import CameraPose, ScaleMode

log = get_logger("trajectory.normalize")

#: Blender-friendly window for a shot's total path length, in Blender units.
#: Wide enough that a 10 m dolly and a 400 m drone flight both land somewhere
#: sane, small enough that the default camera clipping range still contains the
#: whole move.
DEFAULT_TARGET_PATH_UNITS = (10.0, 25.0)

#: Below this a path length is numerically zero and no ratio can be formed.
PATH_LENGTH_EPSILON = 1e-12

#: Window used to separate sustained travel from frame-to-frame reversal.
COHERENCE_WINDOW_FRAMES = 9

#: Fewer samples than this and the coherence statistic is meaningless (the
#: window collapses to three frames), so the static test is skipped and the
#: trajectory is scaled on its raw path length. Documented limitation: a
#: sub-quarter-second static shot may be scaled as if it were a move.
MIN_FRAMES_FOR_STATIC_TEST = 6

#: A moving average of width w applied to a random walk shrinks its path length
#: by roughly 1/sqrt(w): the smoothed increments have standard deviation
#: sigma/sqrt(w). Real travel survives smoothing almost intact (ratio ~0.9), so
#: the window-aware threshold `HEADROOM / sqrt(w)` separates the two with 25%
#: margin instead of relying on one hand-tuned number.
RANDOM_WALK_HEADROOM = 1.25

#: Cap for the above: with a three-frame window the prediction (0.72) would
#: start catching genuinely jittery real moves.
MAX_COHERENCE_THRESHOLD = 0.6

#: Ceiling on the spatial extent kept for a near-static trajectory, in Blender
#: units. A quarter unit of residual wobble reads as a locked-off camera beside
#: a 10-25 unit move. This is only ever used to *shrink*: a camera that did not
#: travel must not travel in the output, so the static branch never amplifies.
STATIC_MAX_EXTENT_UNITS = 0.25

#: A metric calibration reference shorter than this fraction of the total path
#: is dominated by solver noise; scaling the whole shot by it would hang its
#: metric scale on a rounding error.
MIN_CALIBRATION_REFERENCE_FRACTION = 0.01


def path_length(positions: np.ndarray) -> float:
    """Total distance travelled along a polyline of camera centres."""
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with clamped edges, one row per input row."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    radius = window // 2
    idx = np.arange(n)
    out = np.zeros_like(v)
    for offset in range(-radius, radius + 1):
        out += v[np.clip(idx + offset, 0, n - 1)]
    return out / float(2 * radius + 1)


def _coherent_path_ratio(positions: np.ndarray) -> tuple[float, float]:
    """(ratio, threshold) for "is this path travel, or accumulated jitter?".

    The ratio is the path length of the smoothed trajectory over the path length
    of the raw one. Sustained travel is barely shortened by smoothing; a
    stationary camera whose recovered position is a random walk loses most of
    its path length, because the reversals cancel. Returns a ratio of 1.0 (and a
    threshold of 0.0, i.e. "not static") when there are too few samples to tell.
    """
    p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(p)
    raw = path_length(p)
    if n < MIN_FRAMES_FOR_STATIC_TEST or raw <= PATH_LENGTH_EPSILON:
        return 1.0, 0.0

    # Odd window, at most half the shot: a window comparable to the whole
    # sequence averages a genuine ramp down to its mean and would report a
    # real short move as static.
    window = min(COHERENCE_WINDOW_FRAMES, n // 2)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return 1.0, 0.0

    coherent = path_length(_moving_average(p, window))
    threshold = min(MAX_COHERENCE_THRESHOLD, RANDOM_WALK_HEADROOM / np.sqrt(window))
    return coherent / raw, float(threshold)


def _rescaled(poses: list[CameraPose], origin: np.ndarray, scale: float) -> list[CameraPose]:
    """Apply `(p - origin) * scale` to positions and change nothing else.

    Orientations, timestamps, frame indices, lens values, confidences and solver
    provenance are copied through untouched — this is a change of units, not a
    modification of the motion.
    """
    out: list[CameraPose] = []
    for pose in poses:
        p = (np.asarray(pose.position, dtype=np.float64) - origin) * scale
        out.append(pose.model_copy(update={"position": [float(x) for x in p]}))
    return out


def normalize_trajectory(
    poses: list[CameraPose],
    target_path_units: tuple[float, float] = DEFAULT_TARGET_PATH_UNITS,
) -> tuple[list[CameraPose], float]:
    """Put the first camera at the origin and scale the path into Blender range.

    Returns `(poses, scale_factor)` where `scale_factor` is the single number
    every position was multiplied by after re-origining. Callers must keep
    `ScaleMode.NORMALIZED` for the result: nothing here makes the numbers metres
    (I5).

    Near-static trajectories are handled explicitly and differently. When the
    recovered path has no temporal coherence — reversals cancelling out, i.e.
    solver jitter around a stationary camera — its path length is not travel and
    must not be stretched to fill the target range. Such a trajectory is only
    ever *shrunk*, so that a camera which did not move does not move in the
    output.
    """
    low, high = float(target_path_units[0]), float(target_path_units[1])
    if not (0.0 < low <= high):
        raise ValueError(f"target_path_units must satisfy 0 < low <= high, got {target_path_units}")

    if not poses:
        return [], 1.0

    positions = np.array([pose.position for pose in poses], dtype=np.float64).reshape(-1, 3)
    origin = positions[0].copy()
    centred = positions - origin

    raw = path_length(centred)
    if raw <= PATH_LENGTH_EPSILON:
        # A single pose, or every pose at the same point. There is no length to
        # scale; re-origin only, and report a scale of exactly 1.
        return _rescaled(poses, origin, 1.0), 1.0

    ratio, threshold = _coherent_path_ratio(centred)
    if ratio < threshold:
        extent = float(np.linalg.norm(centred, axis=1).max())
        scale = 1.0 if extent <= PATH_LENGTH_EPSILON else min(1.0, STATIC_MAX_EXTENT_UNITS / extent)
        log.info(
            "near-static trajectory (coherence %.3f < %.3f): keeping it static, "
            "scale=%.6g, extent %.4g -> %.4g units",
            ratio, threshold, scale, extent, extent * scale,
        )
        return _rescaled(poses, origin, scale), scale

    # Aim at the middle of the range rather than an edge, so a later per-shot
    # adjustment has headroom on both sides.
    target = 0.5 * (low + high)
    scale = target / raw
    log.info("normalized path length %.4g -> %.4g units (scale %.6g)", raw, raw * scale, scale)
    return _rescaled(poses, origin, scale), float(scale)


def _calibration_reference(
    positions: np.ndarray, kind: ScaleCalibrationKind
) -> tuple[float, str]:
    """(normalized reference length, statement of what was assumed about it).

    Each calibration kind names a real-world length; this picks the normalized
    length it corresponds to, and says out loud what had to be assumed to pair
    them up. The assumptions are the honest part: without scene structure at
    this stage, "known height" and "known distance between two points" can only
    be read against the camera path itself.
    """
    if kind is ScaleCalibrationKind.TRAVEL_DISTANCE:
        return path_length(positions), "the camera's total travelled path length"

    if kind is ScaleCalibrationKind.POINT_DISTANCE:
        chord = float(np.linalg.norm(positions[-1] - positions[0]))
        return chord, (
            "the straight-line distance between the first and last camera position "
            "(the two points are read off the camera path; a distance between two "
            "scene features would need the reconstructed point cloud)"
        )

    if kind is ScaleCalibrationKind.CAMERA_HEIGHT:
        span = float(positions[:, 2].max() - positions[:, 2].min())
        return span, (
            "the vertical span of the camera path, assuming its lowest point sits "
            "at ground level and its highest point is the stated height above it"
        )

    raise ValueError(f"unsupported scale calibration kind: {kind}")


def apply_metric_calibration(
    poses: list[CameraPose],
    calibration: ScaleCalibration,
    normalized_scale: float,
) -> tuple[list[CameraPose], float, str]:
    """Convert a normalized trajectory to metres using one user measurement.

    Returns `(poses_in_metres, metres_per_normalized_unit, statement)`. The
    statement is written for a human and names the assumption that was made,
    because a metric number with an unstated assumption behind it is worse than
    an honestly relative one (I5, I7).

    Raises `ValueError` when the calibration cannot be applied — an empty
    trajectory, a non-positive `normalized_scale`, or a reference length that is
    too short to carry the whole shot's scale (a camera that returned to its
    start cannot be calibrated by first-to-last distance, for instance). The
    caller must then leave the trajectory in `ScaleMode.NORMALIZED` rather than
    inventing a scale.
    """
    if not poses:
        raise ValueError("cannot apply a metric calibration to an empty trajectory")
    if normalized_scale <= 0.0:
        raise ValueError(f"normalized_scale must be positive, got {normalized_scale}")

    positions = np.array([pose.position for pose in poses], dtype=np.float64).reshape(-1, 3)
    kind = ScaleCalibrationKind(calibration.kind)
    reference, assumption = _calibration_reference(positions, kind)

    total = path_length(positions)
    floor = MIN_CALIBRATION_REFERENCE_FRACTION * total
    if total <= PATH_LENGTH_EPSILON or reference <= max(floor, PATH_LENGTH_EPSILON):
        raise ValueError(
            f"{kind.value} calibration is not usable for this shot: "
            f"{assumption} measures {reference:.6g} normalized units, which is under "
            f"{MIN_CALIBRATION_REFERENCE_FRACTION:.0%} of the {total:.6g}-unit path. "
            "The shot has to be left normalized, or calibrated with a different measurement."
        )

    metres_per_unit = float(calibration.value_meters) / float(reference)
    out = _rescaled(poses, np.zeros(3), metres_per_unit)

    metres_per_solver_unit = metres_per_unit * float(normalized_scale)
    statement = (
        f"Metric scale from a user calibration ({kind.value}): {assumption} measures "
        f"{reference:.4g} normalized units and was stated to be {calibration.value_meters:.4g} m, "
        f"so 1 normalized unit = {metres_per_unit:.6g} m "
        f"(1 solver unit = {metres_per_solver_unit:.6g} m at the normalization scale of "
        f"{normalized_scale:.6g}). Positions are now in metres; timing and all rotations are "
        "unchanged. Accuracy of the absolute scale is exactly the accuracy of that one "
        "measurement plus the stated assumption."
    )
    if calibration.note:
        statement = f"{statement} User note: {calibration.note}"

    log.info("metric calibration applied: %.6g m per normalized unit", metres_per_unit)
    return out, metres_per_unit, statement


def scale_units_label(mode: ScaleMode | str) -> str:
    """Unit string for an export. The only source of "m" in the codebase.

    `ScaleMode.NORMALIZED` can never produce "m" — that is invariant I5 reduced
    to a single branch, so there is exactly one line of code to audit.
    """
    resolved = ScaleMode(mode)
    if resolved is ScaleMode.NORMALIZED:
        return "normalized"
    if resolved is ScaleMode.METRIC:
        return "m"
    raise ValueError(f"unhandled scale mode: {resolved}")
