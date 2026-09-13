"""Derived temporal quantities: velocity, acceleration, jerk, angular rates, curvature.

Every derivative here is taken against the *real* time between poses, read from
the timestamps the poses carry (which come from container PTS, I2). Using
`1 / fps` instead is the classic way to lose a variable-frame-rate source's
actual motion: on a clip whose intervals swing by 20% — routine for phone
footage — a nominal-fps derivative reports a smooth speed curve for a camera
that visibly surged, and that error compounds through acceleration and jerk.

Two other decisions worth stating:

  * **Endpoints are one-sided, never zero.** Padding the first and last frames
    with zero velocity puts a stop at the start and end of every shot. A shot
    that was cut out of the middle of a moving camera pass must come back out
    moving at the cut.
  * **Angular rates are per-interval first, then averaged.** Rotation does not
    subtract, so the "central difference" of an orientation sequence is built
    from the two adjacent relative rotations, each divided by its own real
    interval, then combined with duration weights. That is the exact
    non-uniform centred estimate, and it never differences quaternion
    components (I3).

Units: linear quantities are in *scale-mode units* — normalized Blender units
unless the trajectory was explicitly calibrated (I5). Angular quantities are
always degrees per second, because that is what a camera operator and a
generative-video model both read.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.geometry.rotations import (
    angular_velocity,
    quat_angular_distance,
    unroll_quaternions,
)
from app.models.schemas.trajectory import CameraPose, Kinematics, ScaleMode
from app.models.schemas.video import Shot

log = get_logger("trajectory.kinematics")

#: Below this speed the velocity direction is noise, so curvature — which
#: divides by speed cubed — is reported as zero rather than as a huge number.
#: In normalized units a whole trajectory spans ~10-25 units, so 1e-6 units/s
#: is many orders of magnitude below any real camera move.
CURVATURE_SPEED_FLOOR = 1e-6

#: Guard for degenerate timestamps (duplicate PTS, which some containers do
#: emit). A zero interval cannot produce a derivative, so it is skipped rather
#: than dividing by it.
MIN_DT = 1e-9


def _finite_difference(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """d(values)/d(time) with a centred interior and one-sided endpoints.

    The interior form `(v[i+1] - v[i-1]) / (t[i+1] - t[i-1])` is the correct
    non-uniform centred estimate — it is the duration-weighted mean of the two
    adjacent secants, which is why it stays first-order-exact when the intervals
    differ. Endpoints use the single interval that exists.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
        flat = True
    else:
        flat = False
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    n = len(t)
    out = np.zeros_like(v)
    if n < 2:
        return out[:, 0] if flat else out

    for i in range(n):
        lo = max(i - 1, 0)
        hi = min(i + 1, n - 1)
        span = t[hi] - t[lo]
        if span > MIN_DT:
            out[i] = (v[hi] - v[lo]) / span
    return out[:, 0] if flat else out


def body_rates_to_yaw_pitch_roll(omega_body: np.ndarray) -> np.ndarray:
    """Relabel a body-frame rotation vector as [yaw, pitch, roll].

    This is a *labelling*, not a change of coordinate frame — no axis moves, the
    three components are only reordered into the names the schema uses. In the
    CameraPath camera frame the local axes are +X right, +Y forward, +Z up, so a
    rotation about +Z is a pan (yaw), about +X a tilt (pitch), and about +Y a
    roll. Sign follows the right-hand rule about each camera axis: positive yaw
    turns the camera to its left, positive pitch tilts it up, positive roll
    tips its top to the right.

    Named and tested rather than written inline as `w[[2, 0, 1]]` because a
    silent component swap is indistinguishable from correct output until
    someone watches a pan come out as a tilt (I8 in spirit).
    """
    w = np.asarray(omega_body, dtype=np.float64).reshape(-1, 3)
    relabelled = np.stack([w[:, 2], w[:, 0], w[:, 1]], axis=1)
    return relabelled.reshape(np.shape(omega_body))


def _body_angular_rates(quats: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Per-frame body-frame angular velocity in rad/s, (N, 3) about +X/+Y/+Z.

    Each interval's rate is exact for that interval: the relative rotation
    `q[i]^-1 q[i+1]` expressed as a rotation vector, divided by the interval's
    own duration. Interior frames take the duration-weighted mean of the two
    intervals they sit between, which is the rotational analogue of a centred
    difference; endpoints take their single interval.
    """
    n = len(times)
    rates = np.zeros((n, 3))
    if n < 2:
        return rates

    dt = np.diff(times)
    interval = np.zeros((n - 1, 3))
    for i in range(n - 1):
        if dt[i] > MIN_DT:
            interval[i] = angular_velocity(quats[i], quats[i + 1], float(dt[i]))

    rates[0] = interval[0]
    rates[-1] = interval[-1]
    for i in range(1, n - 1):
        weight = dt[i - 1] + dt[i]
        if weight > MIN_DT:
            rates[i] = (interval[i - 1] * dt[i - 1] + interval[i] * dt[i]) / weight
        else:
            rates[i] = interval[i]
    return rates


def compute_kinematics(
    poses: list[CameraPose], scale_mode: ScaleMode = ScaleMode.NORMALIZED
) -> list[Kinematics]:
    """Per-frame kinematics for a shot's poses.

    `scale_mode` does not change any number here — it declares what the linear
    units mean. Normalized and metric trajectories are differentiated
    identically; only the label differs, and that label is what stops a
    normalized speed from being read as m/s (I5).
    """
    if not poses:
        return []

    times = np.array([p.timestamp for p in poses], dtype=np.float64)
    positions = np.array([p.position for p in poses], dtype=np.float64).reshape(-1, 3)
    # Unroll before touching the orientation sequence: a hemisphere flip is not
    # a rotation, and anything that reads the components as a signal — plotting,
    # filtering, export — sees a spurious 360-degree jump. The relative-rotation
    # path below is sign-safe on its own, but relying on that is fragile.
    quats = unroll_quaternions(
        np.array([p.quaternion for p in poses], dtype=np.float64).reshape(-1, 4)
    )

    velocity = _finite_difference(positions, times)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = _finite_difference(velocity, times)
    jerk = _finite_difference(acceleration, times)

    rates_rad = _body_angular_rates(quats, times)
    rates_deg = np.degrees(rates_rad)
    angular_yaw_pitch_roll = body_rates_to_yaw_pitch_roll(rates_deg)
    angular_speed = np.linalg.norm(rates_deg, axis=1)
    angular_acceleration = _finite_difference(angular_yaw_pitch_roll, times)

    max_speed = float(speed.max()) if len(speed) else 0.0

    results: list[Kinematics] = []
    for i, pose in enumerate(poses):
        v = velocity[i]
        a = acceleration[i]
        s = float(speed[i])
        if s > CURVATURE_SPEED_FLOOR:
            # kappa = |v x a| / |v|^3, the standard curvature of a
            # parameterised space curve. Only the component of acceleration
            # perpendicular to velocity bends the path; the cross product
            # extracts exactly that.
            curvature = float(np.linalg.norm(np.cross(v, a)) / (s ** 3))
        else:
            curvature = 0.0

        results.append(
            Kinematics(
                frame_index=pose.frame_index,
                timestamp=pose.timestamp,
                linear_velocity=[float(x) for x in v],
                speed=s,
                speed_normalized=float(s / max_speed) if max_speed > 0 else 0.0,
                linear_acceleration=[float(x) for x in a],
                acceleration_magnitude=float(np.linalg.norm(a)),
                jerk_magnitude=float(np.linalg.norm(jerk[i])),
                angular_velocity=[float(x) for x in angular_yaw_pitch_roll[i]],
                angular_speed=float(angular_speed[i]),
                angular_acceleration=[float(x) for x in angular_acceleration[i]],
                curvature=curvature,
            )
        )

    log.debug(
        "kinematics for %d poses: peak speed %.4f %s/s, peak angular speed %.2f deg/s",
        len(poses), max_speed,
        "unit" if scale_mode == ScaleMode.NORMALIZED else "m",
        float(angular_speed.max()) if len(angular_speed) else 0.0,
    )
    return results


def path_length(poses: list[CameraPose]) -> float:
    """Total distance travelled, in scale-mode units.

    The straight-line sum between consecutive poses, not the spline's arc
    length: the poses are the trajectory as exported, so this is the number that
    matches what a consumer measures.
    """
    if len(poses) < 2:
        return 0.0
    positions = np.array([p.position for p in poses], dtype=np.float64).reshape(-1, 3)
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def total_rotation_degrees(poses: list[CameraPose]) -> float:
    """Accumulated geodesic rotation, in degrees.

    Summed per transition rather than measured start-to-end, so a pan that goes
    out and comes back reports the distance the camera actually turned rather
    than zero.
    """
    if len(poses) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(poses[:-1], poses[1:]):
        total += np.degrees(quat_angular_distance(np.array(a.quaternion), np.array(b.quaternion)))
    return float(total)


def duration_matches_source(
    poses: list[CameraPose], shot: Shot, fps: float
) -> tuple[bool, float]:
    """Invariant I1 check: (within one output frame, signed delta in seconds).

    Both durations are measured the same way — the span from the first pose's
    timestamp to the last — so this compares like with like and does not trip
    over whether the final frame's own display interval counts. What it catches
    is a trajectory that was retimed: resampled onto a synthetic uniform grid,
    truncated, or padded. Any of those changes how long the move takes, which is
    a change to the move.

    A non-positive `fps` falls back to the shot's own measured mean rate, so the
    tolerance is always one real output frame rather than an assumed one.
    """
    if len(poses) < 2:
        # A single pose has no duration to compare; only a single-frame shot can
        # honestly match.
        delta = -float(shot.duration)
        return abs(delta) <= MIN_DT, delta

    output_duration = float(poses[-1].timestamp - poses[0].timestamp)
    delta = output_duration - float(shot.duration)

    if fps > 0:
        frame_time = 1.0 / float(fps)
    elif shot.duration > 0 and shot.frame_count > 1:
        frame_time = float(shot.duration) / float(shot.frame_count - 1)
    else:
        # No usable rate at all: the only defensible tolerance is exactness.
        frame_time = MIN_DT

    # Float epsilon, not slack: timestamps arrive as float seconds converted
    # from integer PTS, so an exact match can miss by ~1e-15.
    return abs(delta) <= frame_time + 1e-9, delta
