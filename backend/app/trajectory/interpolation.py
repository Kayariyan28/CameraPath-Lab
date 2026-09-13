"""Resampling that preserves motion character.

Interpolation is where a camera-motion pipeline most easily destroys the thing
it was built to recover. Three specific failure modes are designed out here:

  * **Uniform-parameter splines.** A Catmull-Rom spline that assumes evenly
    spaced control points computes its tangents in units of "per sample" rather
    than "per second". With variable frame intervals — VFR phone footage, which
    is extremely common, or anchor keyframes placed by accumulated motion rather
    than by time — that silently rescales velocity inside every segment. A
    2-second deceleration comes back out at the wrong rate, which violates I1.
    So every tangent here is a real d(value)/d(time) derivative estimate over
    the actual intervals.
  * **Overshoot on scalar curves.** A cubic through a zoom ramp that flattens
    into a hold will bulge past the hold value and come back. In the output that
    reads as a zoom that went slightly too far and reversed — a move that is not
    in the source. Focal length therefore gets a monotone (Fritsch-Carlson)
    cubic, which cannot overshoot by construction.
  * **Extrapolation.** Past the measured range there is no evidence, so the
    endpoint is held. Letting a cubic run off the end of the data invents motion
    that was never observed, and it diverges fastest exactly where the solve was
    least constrained.

Rotations are not resampled here at all beyond a thin wrapper: they go through
`geometry.rotations.quat_sequence_squad` (I3). There is deliberately no Euler
path in this module.
"""

from __future__ import annotations

import numpy as np

from app.geometry.rotations import quat_identity, quat_normalize, quat_sequence_squad

#: Below this, two control times are treated as coincident and the spline holds
#: rather than dividing by the gap. 1 ns is far below any real frame interval.
EPS = 1e-9

#: Fraction of a speed-warped parameterisation that is allocated by time rather
#: than by measured motion. A genuinely static passage produces zero measured
#: arc length, which would collapse a spline segment to zero width and make the
#: interpolation ill-posed. Reserving a small share for time keeps the parameter
#: strictly increasing everywhere while perturbing the measured velocity profile
#: by at most this fraction.
DEFAULT_TIME_FLOOR = 0.02


def _as_series(values: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a (M, D) view of control values, plus whether the input was 1-D."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    return arr, False


def _check_times(t: np.ndarray) -> None:
    if len(t) > 1 and bool(np.any(np.diff(t) < 0.0)):
        # Unsorted control times have no single interpolant; silently sorting
        # would pair values with the wrong instants.
        raise ValueError("control times must be non-decreasing")


def catmull_rom_tangents(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Finite-difference tangents on NON-UNIFORM time spacing, d(value)/d(time).

    Interior tangent at k is the derivative of the parabola through
    ``(t[k-1], t[k], t[k+1])``::

        m_k = (h_k * s_{k-1} + h_{k-1} * s_k) / (h_{k-1} + h_k)

    where ``h`` are the real intervals and ``s`` the secant slopes on either
    side. Each secant is weighted by the *opposite* interval, which is what makes
    the estimate exact for a constant acceleration however unequal the intervals
    are — so a steady deceleration keeps its rate through the spline. When the
    spacing is uniform this reduces exactly to the classic Catmull-Rom central
    difference ``(p[k+1] - p[k-1]) / 2h``; the uniform form is the special case,
    and using it on unequal intervals is wrong in proportion to how unequal they
    are.

    Endpoints use the matching one-sided three-point derivative (two-point secant
    when only two controls exist) — never zero, which would put a stop at the
    ends of the curve that the source never contained.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    v, _ = _as_series(values)
    n = len(t)
    m = np.zeros_like(v)
    if n < 2:
        return m

    h = np.diff(t)
    secant = np.zeros((n - 1, v.shape[1]))
    ok = h > EPS
    secant[ok] = (v[1:][ok] - v[:-1][ok]) / h[ok, None]

    if n == 2:
        m[0] = secant[0]
        m[1] = secant[0]
        return m

    for k in range(1, n - 1):
        h0, h1 = h[k - 1], h[k]
        if h0 <= EPS and h1 <= EPS:
            continue
        if h0 <= EPS:
            m[k] = secant[k]
        elif h1 <= EPS:
            m[k] = secant[k - 1]
        else:
            m[k] = (h1 * secant[k - 1] + h0 * secant[k]) / (h0 + h1)

    # One-sided three-point derivatives: the slope of the same parabola, read at
    # its first and last node.
    h0, h1 = h[0], h[1]
    if h0 > EPS and h1 > EPS:
        m[0] = secant[0] - h0 * (secant[1] - secant[0]) / (h0 + h1)
    else:
        m[0] = secant[0] if h0 > EPS else secant[1]
    h0, h1 = h[-2], h[-1]
    if h0 > EPS and h1 > EPS:
        m[-1] = secant[-1] + h1 * (secant[-1] - secant[-2]) / (h0 + h1)
    else:
        m[-1] = secant[-1] if h1 > EPS else secant[-2]
    return m


def hermite_resample(
    times: np.ndarray,
    positions: np.ndarray,
    sample_times: np.ndarray,
    tangents: np.ndarray | None = None,
) -> np.ndarray:
    """Cubic Hermite resampling of a vector series onto `sample_times`.

    `positions` is (M, D) — normally (M, 3) for camera centres; 1-D input gives
    1-D output. Control points are reproduced exactly at their own times, and
    samples outside `[times[0], times[-1]]` hold the nearest endpoint rather than
    extrapolating.

    Supply `tangents` (per second) when a better velocity estimate exists;
    otherwise non-uniform finite-difference tangents are used
    (`catmull_rom_tangents`).
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    p, was_1d = _as_series(positions)
    st = np.asarray(sample_times, dtype=np.float64).reshape(-1)

    if len(t) != len(p):
        raise ValueError(f"{len(t)} control times but {len(p)} control values")
    _check_times(t)

    dim = p.shape[1]
    if len(t) == 0:
        out = np.zeros((len(st), dim))
        return out[:, 0] if was_1d else out
    if len(t) == 1:
        out = np.tile(p[0], (len(st), 1))
        return out[:, 0] if was_1d else out

    if tangents is None:
        m = catmull_rom_tangents(t, p)
    else:
        m, _ = _as_series(tangents)
        if m.shape != p.shape:
            raise ValueError(f"tangents {m.shape} do not match values {p.shape}")

    k = np.clip(np.searchsorted(t, st, side="right") - 1, 0, len(t) - 2)
    h = t[k + 1] - t[k]
    safe_h = np.where(h > EPS, h, 1.0)
    u = np.where(h > EPS, (st - t[k]) / safe_h, 0.0)
    u2 = u * u
    u3 = u2 * u
    # Hermite basis. Tangents are per-second, so they scale by the segment
    # width h — this factor is exactly what the uniform-spacing form drops.
    h00 = (2.0 * u3 - 3.0 * u2 + 1.0)[:, None]
    h10 = (u3 - 2.0 * u2 + u)[:, None]
    h01 = (-2.0 * u3 + 3.0 * u2)[:, None]
    h11 = (u3 - u2)[:, None]
    hw = np.where(h > EPS, h, 0.0)[:, None]
    out = h00 * p[k] + h10 * hw * m[k] + h01 * p[k + 1] + h11 * hw * m[k + 1]

    # Exact reproduction and no extrapolation. Assigned rather than trusted to
    # the basis: at u == 0 the basis is exact in IEEE arithmetic, but holding the
    # ends must not depend on that.
    out[st <= t[0]] = p[0]
    out[st >= t[-1]] = p[-1]
    hits = np.searchsorted(t, st, side="left")
    in_range = hits < len(t)
    exact = np.zeros(len(st), dtype=bool)
    exact[in_range] = t[hits[in_range]] == st[in_range]
    out[exact] = p[hits[exact]]

    return out[:, 0] if was_1d else out


def monotone_resample(
    times: np.ndarray, values: np.ndarray, sample_times: np.ndarray
) -> np.ndarray:
    """1-D Fritsch-Carlson monotone cubic resampling.

    Used for scalar curves where an overshoot would be read as a real event:
    focal length above all. A plain cubic through a zoom that ramps and then
    holds bulges past the hold value and comes back, which in the output is a
    zoom reversal that is not in the source.

    Fritsch & Carlson (1980): zero the tangent wherever the data has a local
    extremum or a flat segment, then scale tangents so each segment stays inside
    the monotonicity region ``alpha^2 + beta^2 <= 9``. Non-monotone input is
    still interpolated — the guarantee is that the interpolant adds no extrema
    of its own.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    st = np.asarray(sample_times, dtype=np.float64).reshape(-1)

    if len(t) != len(v):
        raise ValueError(f"{len(t)} control times but {len(v)} control values")
    _check_times(t)
    if len(t) == 0:
        return np.zeros(len(st))
    if len(t) == 1:
        return np.full(len(st), v[0])

    n = len(t)
    dt = np.diff(t)
    secant = np.zeros(n - 1)
    nonzero = dt > EPS
    secant[nonzero] = np.diff(v)[nonzero] / dt[nonzero]

    m = np.zeros(n)
    m[0] = secant[0]
    m[-1] = secant[-1]
    for k in range(1, n - 1):
        # Opposite signs means a local extremum in the data: the curve must be
        # flat there or it overshoots the extremum.
        if secant[k - 1] * secant[k] > 0.0:
            m[k] = (secant[k - 1] + secant[k]) / 2.0

    # A flat segment must stay flat at both ends, otherwise the cubic dips out
    # of the segment's own value range.
    for k in range(n - 1):
        if abs(secant[k]) <= EPS:
            m[k] = 0.0
            m[k + 1] = 0.0

    for k in range(n - 1):
        if abs(secant[k]) <= EPS:
            continue
        alpha = m[k] / secant[k]
        beta = m[k + 1] / secant[k]
        # A tangent pointing against the segment's own direction is an
        # overshoot in the making; clamp it out.
        if alpha < 0.0:
            m[k] = 0.0
            alpha = 0.0
        if beta < 0.0:
            m[k + 1] = 0.0
            beta = 0.0
        radius = alpha * alpha + beta * beta
        if radius > 9.0:
            tau = 3.0 / np.sqrt(radius)
            m[k] = tau * alpha * secant[k]
            m[k + 1] = tau * beta * secant[k]

    return hermite_resample(t, v, st, tangents=m)


def resample_rotations(
    times: np.ndarray, quats: np.ndarray, sample_times: np.ndarray
) -> np.ndarray:
    """Resample an orientation sequence onto `sample_times`. Returns (N, 4).

    Thin wrapper over `geometry.rotations.quat_sequence_squad` so that callers in
    `trajectory/` never have a reason to reach for an Euler representation (I3).
    Control orientations are hit exactly; endpoints are held, not extrapolated.
    """
    st = np.asarray(sample_times, dtype=np.float64).reshape(-1)
    arr = np.asarray(quats, dtype=np.float64)
    if arr.size == 0:
        return np.tile(quat_identity(), (len(st), 1))
    arr = arr.reshape(-1, 4)

    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(t) != len(arr):
        raise ValueError(f"{len(t)} control times but {len(arr)} control quaternions")
    _check_times(t)

    return quat_sequence_squad([quat_normalize(q) for q in arr], t, st)


def speed_warped_parameterization(
    times: np.ndarray,
    speed_profile: np.ndarray,
    *,
    time_floor: float = DEFAULT_TIME_FLOOR,
) -> np.ndarray:
    """Cumulative normalised arc-length parameters from a measured speed profile.

    This is what makes interpolation follow the *measured* velocity profile
    instead of uniform time. Interpolating anchor positions against time assumes
    the camera covered equal ground in equal time between anchors; interpolating
    them against accumulated measured motion does not. A source that decelerates
    for two seconds therefore comes back out decelerating for two seconds (I1),
    even when the anchors are far apart.

    `speed_profile` is a per-interval rate (e.g. flow magnitude per second).
    Two lengths are accepted:

      * ``len(times) - 1`` — entry *i* is the rate over ``[times[i], times[i+1]]``.
      * ``len(times)``     — entry *i* is the rate over ``[times[i-1], times[i]]``,
        matching the per-frame arrays in this codebase where a value describes
        the transition *into* its own frame. Entry 0 has no preceding interval
        and is ignored.

    Returns a non-decreasing array of length ``len(times)``, starting at 0 and
    ending at 1: ``(1 - time_floor) * arc_fraction + time_floor * time_fraction``.
    A completely static input degrades to normalised time rather than to a
    zero-width, ill-posed parameterisation.
    """
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if len(t) == 0:
        return np.zeros(0)
    if len(t) == 1:
        return np.zeros(1)
    _check_times(t)

    speed = np.asarray(speed_profile, dtype=np.float64).reshape(-1)
    if len(speed) == len(t):
        speed = speed[1:]
    elif len(speed) != len(t) - 1:
        raise ValueError(
            f"speed_profile must have {len(t) - 1} or {len(t)} entries, got {len(speed)}"
        )

    dt = np.diff(t)
    # Negative or non-finite measured speed is meaningless; it would run the
    # parameter backwards and reorder the trajectory.
    speed = np.where(np.isfinite(speed), np.clip(speed, 0.0, None), 0.0)
    arc = speed * dt

    total_arc = float(arc.sum())
    total_time = float(dt.sum())
    if total_time <= EPS:
        return np.linspace(0.0, 1.0, len(t))

    floor = float(np.clip(time_floor, 0.0, 1.0))
    if total_arc <= EPS:
        # No measured motion at all: the only honest parameterisation left is
        # time. Saying otherwise would invent a velocity profile.
        step = dt / total_time
    else:
        step = (1.0 - floor) * (arc / total_arc) + floor * (dt / total_time)

    param = np.concatenate([[0.0], np.cumsum(step)])
    # Guard against float drift so the last control point lands exactly on 1.
    param[-1] = 1.0
    return param
