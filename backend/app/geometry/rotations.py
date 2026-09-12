"""Quaternion algebra and correct rotation interpolation.

Invariant I3: rotations are never interpolated as Euler angles. Euler
interpolation is wrong in three separate ways that all matter here — it depends
on the arbitrary axis order, it does not follow the shortest path, and it
produces speed variation and gimbal artefacts that would be indistinguishable
from real camera motion in the output. Since reproducing real camera motion
faithfully is the entire product, that is disqualifying.

Quaternion layout is [w, x, y, z] throughout, matching Blender's
`rotation_quaternion`. Every function here assumes unit quaternions and
normalises defensively, because repeated composition drifts off the unit sphere.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def quat_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < EPS:
        return quat_identity()
    return q / n


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product. `quat_multiply(a, b)` applies b first, then a."""
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by a unit quaternion."""
    q = quat_normalize(q)
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=np.float64)
    # Rodrigues form: v' = v + 2w(u x v) + 2u x (u x v)
    uv = np.cross(u, v)
    return v + 2.0 * w * uv + 2.0 * np.cross(u, uv)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """Rotation matrix to [w,x,y,z].

    Uses Shepperd's branch selection: pick the largest diagonal term so the
    divisor is never near zero. The naive single-branch formula loses all
    precision near 180 degrees, which is reachable in a shot that reverses
    direction.
    """
    m = np.asarray(m, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return quat_canonical(quat_normalize(np.array([w, x, y, z])))


def quat_canonical(q: np.ndarray) -> np.ndarray:
    """Force w >= 0.

    q and -q are the same rotation. Without canonicalisation a pose sequence can
    flip sign between frames, and anything that treats the components as a signal
    — plotting, smoothing, finite differencing — sees a spurious 360-degree jump.
    """
    q = np.asarray(q, dtype=np.float64)
    return -q if q[0] < 0 else q


def quat_angle(q: np.ndarray) -> float:
    """Rotation magnitude in radians, in [0, pi]."""
    q = quat_canonical(quat_normalize(q))
    return float(2.0 * np.arccos(np.clip(q[0], -1.0, 1.0)))


def quat_relative(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation taking `a` to `b`: r such that b = a * r."""
    return quat_multiply(quat_conjugate(quat_normalize(a)), quat_normalize(b))


def quat_angular_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle between two orientations, radians."""
    return quat_angle(quat_relative(a, b))


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(axis))
    if n < EPS or abs(angle) < EPS:
        return quat_identity()
    axis = axis / n
    h = angle * 0.5
    s = np.sin(h)
    return np.array([np.cos(h), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_log(q: np.ndarray) -> np.ndarray:
    """Log map to a rotation vector (axis * angle), radians."""
    q = quat_canonical(quat_normalize(q))
    v = q[1:]
    vn = float(np.linalg.norm(v))
    if vn < EPS:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vn, q[0])
    return v / vn * angle


def quat_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Exp map from a rotation vector back to a quaternion."""
    r = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(r))
    if angle < EPS:
        return quat_identity()
    return quat_from_axis_angle(r / angle, angle)


def quat_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Shortest-path spherical linear interpolation."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))

    # q and -q are the same rotation; flip so we take the short way round.
    if dot < 0.0:
        b = -b
        dot = -dot

    if dot > 0.9995:
        # Nearly parallel: slerp's sin(theta) divisor degenerates, and lerp is
        # accurate to well below single precision at this angle.
        return quat_normalize(a + t * (b - a))

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s_a = np.sin(theta_0 - theta) / sin_theta_0
    s_b = np.sin(theta) / sin_theta_0
    return quat_normalize(s_a * a + s_b * b)


def _squad_tangent(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """Inner control quaternion for SQUAD at `cur`."""
    cur = quat_normalize(cur)
    inv = quat_conjugate(cur)
    l1 = quat_log(quat_multiply(inv, quat_normalize(prev)))
    l2 = quat_log(quat_multiply(inv, quat_normalize(nxt)))
    return quat_normalize(quat_multiply(cur, quat_exp(-(l1 + l2) / 4.0)))


def quat_squad(
    q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray, t: float
) -> np.ndarray:
    """Spherical cubic interpolation between q1 and q2, using q0/q3 as tangents.

    SQUAD rather than plain SLERP because a chain of SLERP segments is only C0:
    angular velocity jumps at every control point. Those jumps would appear in
    the exported camera as a per-keyframe stutter that is not in the source —
    the output must carry the source's angular acceleration, not artefacts of
    its own keyframe spacing.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    s1 = _squad_tangent(q0, q1, q2)
    s2 = _squad_tangent(q1, q2, q3)
    a = quat_slerp(q1, q2, t)
    b = quat_slerp(s1, s2, t)
    return quat_canonical(quat_slerp(a, b, 2.0 * t * (1.0 - t)))


def quat_sequence_squad(quats: list[np.ndarray], times: np.ndarray,
                        sample_times: np.ndarray) -> np.ndarray:
    """Resample an orientation sequence onto `sample_times` with SQUAD.

    Control orientations are hit exactly at their own times; between them the
    curve is C1. Endpoints are extended by duplication rather than extrapolated,
    because extrapolating rotation past the measured range invents motion.
    """
    if not quats:
        return np.tile(quat_identity(), (len(sample_times), 1))
    if len(quats) == 1:
        return np.tile(quat_normalize(quats[0]), (len(sample_times), 1))

    times = np.asarray(times, dtype=np.float64)
    q = [quat_normalize(x) for x in quats]
    out = np.zeros((len(sample_times), 4))

    for i, t in enumerate(np.asarray(sample_times, dtype=np.float64)):
        if t <= times[0]:
            out[i] = q[0]
            continue
        if t >= times[-1]:
            out[i] = q[-1]
            continue
        k = int(np.searchsorted(times, t, side="right") - 1)
        k = max(0, min(k, len(q) - 2))
        span = times[k + 1] - times[k]
        local = 0.0 if span <= EPS else (t - times[k]) / span
        q0 = q[k - 1] if k - 1 >= 0 else q[k]
        q3 = q[k + 2] if k + 2 < len(q) else q[k + 1]
        out[i] = quat_squad(q0, q[k], q[k + 1], q3, float(local))
    return out


def angular_velocity(q_a: np.ndarray, q_b: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity taking q_a to q_b over dt, radians/second.

    Body frame, not world: the reported rates are yaw/pitch/roll *of the camera*,
    which is what a camera operator and a generative-video model both mean by
    "pan rate" and "tilt rate".
    """
    if dt <= EPS:
        return np.zeros(3)
    return quat_log(quat_relative(q_a, q_b)) / dt


def unroll_quaternions(quats: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions lie in the same hemisphere.

    Required before any per-component differencing or filtering: an unflipped
    sequence contains sign discontinuities that are not rotations, and
    differencing them produces enormous spurious angular velocities.
    """
    q = np.array(quats, dtype=np.float64).reshape(-1, 4)
    for i in range(1, len(q)):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] = -q[i]
    return q


def look_at_quaternion(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world quaternion for a camera looking along `forward`.

    Returns a CameraPath-convention orientation: camera +Y along `forward`,
    camera +Z along the up-ward component of `up`.
    """
    f = np.asarray(forward, dtype=np.float64)
    fn = float(np.linalg.norm(f))
    if fn < EPS:
        return quat_identity()
    f = f / fn

    u = np.asarray(up, dtype=np.float64)
    if float(np.linalg.norm(u)) < EPS:
        u = np.array([0.0, 0.0, 1.0])
    # Gram-Schmidt; fall back to another axis if up is parallel to forward.
    u_perp = u - f * float(np.dot(u, f))
    if float(np.linalg.norm(u_perp)) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u_perp = alt - f * float(np.dot(alt, f))
    u_perp /= float(np.linalg.norm(u_perp))

    right = np.cross(f, u_perp)
    right /= max(float(np.linalg.norm(right)), EPS)
    # Columns are the camera's [right, forward, up] axes in world space.
    m = np.column_stack([right, f, u_perp])
    return matrix_to_quat(m)
