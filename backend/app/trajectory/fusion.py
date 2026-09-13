"""Fusion of sparse geometric anchors with the dense per-frame motion signature.

This is the heart of the product (architecture §5). The geometric solve gives
globally consistent poses, but only on selected keyframes; the dense motion
signature gives every frame transition, but only in image space. Neither alone
is the answer:

  * Interpolating between keyframes with anything smooth destroys exactly what
    this tool exists to preserve — acceleration, jitter, micro-movement, the
    moment a handheld operator catches themselves. The result looks like stock
    camera-move footage rather than like the reference.
  * Integrating the dense signal alone drifts without bound and cannot know
    where the camera actually was, because image motion has no depth.

So anchors own the global path and the dense signal owns the shape between
anchors.

**Rotation.** Anchors own the orientation path: SQUAD through them (I3,
quaternions throughout), extrapolated past the first and last anchor at the end
segments' angular rate, decaying with EXTRAPOLATION_DECAY_SECONDS.

**Residual, and why its gain is measured rather than fixed.** The dense
per-frame increments are integrated into an orientation path; its own
anchor-spaced trend is divided out, high-passed, and the leftover is what
keyframes are too sparse to see. The draft of this module composed that
leftover onto the anchor path at gain 1.0 for EXACT fidelity, reasoning that the
residual IS the source's micro-motion. Ground truth says otherwise. The
flow-to-rotation conversion (`yaw = atan(dx / fx)`) cannot tell rotation from
translation, so on a moving camera the residual is mostly parallax:

    scene          anchors only   residual gain 1   true shake preserved?
    handheld         0.845 deg        0.403 deg        corr 0.47 -> 0.96
    fpv_curve        0.072 deg        0.409 deg        -
    accelerating     0.023 deg        0.332 deg        -
    orbit            0.051 deg        2.970 deg        -

A fixed gain is wrong on one side or the other. The gain is therefore chosen per
shot by holding out anchors: each held-out anchor is predicted from its
neighbours with and without the residual, and the residual is applied at the
gain that predicts held-out anchors best — only if it beats no residual by a
clear margin. That picked 1.0 on handheld and 0.0 on every smooth move above,
matching or beating the best fixed choice on all seven synthetic scenes. It uses
nothing but the solve's own anchors, so it runs on real footage.

Motion Fidelity scales the selected gain: EXACT applies it in full (the most
faithful estimate of the source's motion, shake included where the data
supports it, I10); CLEAN halves it; SMOOTH drops it.

**Translation.** Anchor centres are interpolated with a Hermite spline
parameterised by the *measured* flow-magnitude profile rather than by time, so
the velocity profile between two anchors is the measured one (I1). Past the end
anchors, position continues at the end segments' velocity, decaying like the
rotation, rather than freezing.

What this module deliberately does NOT do is turn lateral flow into positional
shake. The same pixels of image motion are already spent on rotation via the
camera model, and without depth a pixel of flow cannot be split between "the
camera turned" and "the camera moved" — converting it a second time would both
double the shake and fabricate a baseline (I6).

Two further honesty rules:

  * When translation is not observable, positions are held constant rather than
    integrated out of flow (I6, I7).
  * With fewer than two anchors there is no geometric path; the result degrades
    to a rotation-only 2D motion proxy, labelled as such, with LOW confidence,
    and never raises (I7, I12).
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.geometry.intrinsics import (
    MAX_HORIZONTAL_FOV,
    MIN_HORIZONTAL_FOV,
    focal_pixels_to_fov,
    fov_to_focal_pixels,
)
from app.geometry.rotations import (
    quat_angular_distance,
    quat_exp,
    quat_log,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_relative,
    unroll_quaternions,
)
from app.models.schemas.motion import CameraIntrinsics, LensFrame, MotionFrame
from app.models.schemas.trajectory import CameraPose, MotionFidelity, SolverSource
from app.models.schemas.video import FrameMetadata, Shot
from app.solvers.base import GeometryResult
from app.trajectory.interpolation import (
    hermite_resample,
    monotone_resample,
    resample_rotations,
    speed_warped_parameterization,
)

log = get_logger("trajectory.fusion")

#: Gain applied to the measured high-frequency residual, per fidelity mode.
#: EXACT is 1.0 by definition (I10): the residual IS the source's micro-motion,
#: and attenuating it would be a silent edit of the camera move. CLEAN halves it
#: — enough to take the edge off tracking noise while the move stays
#: recognisable. SMOOTH drops it entirely and is opt-in, because it changes the
#: move.
RESIDUAL_GAIN: dict[MotionFidelity, float] = {
    MotionFidelity.EXACT: 1.0,
    MotionFidelity.CLEAN: 0.5,
    MotionFidelity.SMOOTH: 0.0,
}

#: Time constant separating "deliberate move" from "shake", in seconds. A
#: centred moving average of this width passes content below roughly
#: 0.44 / 0.12 s ~= 3.7 Hz and has its first null at 1 / 0.12 s ~= 8 Hz.
#: Deliberate camera moves live below ~2 Hz (even a fast whip pan takes a
#: quarter second); hand tremor and rig shake live at 4-12 Hz. Defining the
#: split in seconds rather than in frames keeps it frame-rate independent.
JITTER_TIME_CONSTANT = 0.12

#: How much a poor dense transition is allowed to pull down an interpolated
#: pose's confidence. A frame sitting between two good anchors is still mostly
#: trustworthy even if its own flow estimate was weak, so the dense term scales
#: confidence between this floor and 1.0 rather than down to zero.
DENSE_CONFIDENCE_FLOOR = 0.5

#: Ceiling on per-pose confidence in the degraded, rotation-only path. A 2D
#: motion proxy has no geometric evidence for anything, so it must never report
#: better than LOW. 0.2 sits well below both `validation/confidence.py`'s
#: MEDIUM threshold (0.45) and its no-geometry cap (0.35) (I7).
DEGRADED_MAX_CONFIDENCE = 0.2

#: Ceiling on lens confidence when the solve reported focal length as
#: unobservable. The value emitted is then the prior, and nothing in the footage
#: tested it, so its confidence must read LOW whatever the prior's own
#: provenance — a user override is honoured as the value, but it is still not
#: evidence (I7).
UNOBSERVABLE_FOCAL_CONFIDENCE = 0.2

#: Relative deviation of the prior lens curve from its level at the anchors
#: below which the curve is treated as flat. Mirrors the 2% focal-noise
#: tolerance in `validation/confidence.py`: below it, the curve's "shape" is
#: tracking noise and the measured focal speaks for the whole frame.
FOCAL_SHAPE_FLAT_TOLERANCE = 0.02

#: Residual gains considered by the held-out-anchor test. A coarse grid is enough:
#: on every synthetic scene the error was monotone in gain.
HOLDOUT_GAINS = (0.0, 0.25, 0.5, 0.75, 1.0)

#: A nonzero gain must predict held-out anchors at least this much better than no
#: residual (ratio of mean errors) before it is used, so the residual is never
#: applied on the strength of noise.
HOLDOUT_REQUIRED_IMPROVEMENT = 0.9

#: Held-out anchors examined per shot. Evenly spaced; bounds cost on long shots.
MAX_HOLDOUT_ANCHORS = 40

#: Anchors needed before holding one out means anything (the end anchors are
#: never held out, and each prediction needs a neighbour on both sides).
MIN_ANCHORS_FOR_HOLDOUT = 4

#: Anchors either side of a held-out one used to rebuild SQUAD locally. SQUAD's
#: control points depend only on immediate neighbours, so 3 reproduces the global
#: curve between them exactly.
HOLDOUT_LOCAL_ANCHORS = 3

#: Past the first or last anchor, motion continues at the end segment's rate:
#: constant for this long, then decaying with EXTRAPOLATION_DECAY_SECONDS.
#: Freezing is certainly wrong (the orbit's untracked 0.27 s tail froze into 7.9
#: deg of error), and a constant rate measured best over that gap (0.057 deg;
#: decaying from the first instant under-rotated by 12% and cost 0.65 deg). The
#: decay beyond it is an assumption, not a measurement: without evidence, a
#: camera is not presumed to keep turning indefinitely. Confidence decays with it.
EXTRAPOLATION_CONSTANT_SECONDS = 0.5
EXTRAPOLATION_DECAY_SECONDS = 1.0

#: Two anchors are the minimum for an interpolated path; with fewer there is no
#: global trajectory to interpolate and the result degrades to a proxy.
MIN_ANCHORS_FOR_GEOMETRY = 2

#: Guard for zero or negative time intervals (duplicate PTS).
MIN_DT = 1e-9


# ---------------------------------------------------------------------------
# Shot frames and time base
# ---------------------------------------------------------------------------


def _shot_frames(frames_meta: list[FrameMetadata], shot: Shot) -> list[FrameMetadata]:
    """Source frames belonging to this shot, in order, one per frame index.

    A shot is an independent coordinate system (I4), so anything outside its
    frame range is not merely unhelpful — it belongs to a different trajectory.
    """
    by_index: dict[int, FrameMetadata] = {}
    for fm in frames_meta:
        if shot.start_frame <= fm.frame_index <= shot.end_frame:
            by_index.setdefault(fm.frame_index, fm)
    return [by_index[k] for k in sorted(by_index)]


def _spline_times(times: np.ndarray, shot_id: int) -> np.ndarray:
    """Non-decreasing copy of the PTS-derived times, for spline parameters only.

    Poses are always stamped with the container times as given (I2). A container
    that emits out-of-order presentation times would otherwise make every
    interpolant ill-posed, so only the *parameterisation* is made monotone, and
    the repair is logged rather than hidden.
    """
    if len(times) > 1 and bool(np.any(np.diff(times) < 0.0)):
        log.warning(
            "shot %d: frame timestamps are not monotone in frame order; "
            "interpolating against their running maximum", shot_id,
        )
        return np.maximum.accumulate(times)
    return times


# ---------------------------------------------------------------------------
# Dense signal preparation
# ---------------------------------------------------------------------------


def _jitter_window(times: np.ndarray) -> int:
    """Odd moving-average width in frames that realises JITTER_TIME_CONSTANT."""
    if len(times) < 3:
        return 3
    dt = np.diff(times)
    dt = dt[dt > MIN_DT]
    median_dt = float(np.median(dt)) if len(dt) else 0.0
    if median_dt <= 0:
        return 3
    width = int(round(JITTER_TIME_CONSTANT / median_dt))
    if width % 2 == 0:
        width += 1
    return max(3, width)


def _moving_average(series: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over axis 0, padded at the edges by repetition.

    Edge repetition rather than zero padding: zeros would drag the first and
    last few frames toward no motion, which reads as the camera pausing at the
    start and end of every shot.
    """
    arr = np.asarray(series, dtype=np.float64)
    flat = arr.ndim == 1
    if flat:
        arr = arr.reshape(-1, 1)
    n = len(arr)
    if n < 3 or window < 3:
        return arr[:, 0].copy() if flat else arr.copy()

    half = min(window // 2, n - 1)
    width = 2 * half + 1
    padded = np.pad(arr, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(width) / width
    out = np.empty_like(arr)
    for d in range(arr.shape[1]):
        out[:, d] = np.convolve(padded[:, d], kernel, mode="valid")
    return out[:, 0] if flat else out


def _apply_residual_gain(series: np.ndarray, window: int, gain: float) -> np.ndarray:
    """Keep the low-frequency part of a per-transition series, scale the rest.

    Row 0 is the shot's first frame, which has no incoming transition; it is
    excluded from the filter so that its structural zero is not averaged into
    the first real transitions as a fake slowdown.

    At gain 1.0 the series is returned untouched — bit-for-bit, not
    "reconstructed from the two halves" — so EXACT fidelity cannot introduce
    filtering artefacts of its own.
    """
    arr = np.asarray(series, dtype=np.float64)
    out = arr.copy()
    if abs(gain - 1.0) < 1e-12 or len(arr) < 2:
        return out
    transitions = arr[1:]
    low = _moving_average(transitions, window)
    out[1:] = low + gain * (transitions - low)
    return out


def _dense_rotation_increments(
    motion_by_index: dict[int, MotionFrame],
    frame_indices: list[int],
    fx_series: np.ndarray,
    fy_series: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Per-frame body-frame rotation increments as rotation vectors, radians.

    I6: this is the one legitimate place where image-space flow becomes camera
    rotation, and it is legitimate *only* because a camera model is available:
    a horizontal image shift of `dx` pixels corresponds to a yaw of
    `atan(dx / fx)`. Without intrinsics the same `dx` is consistent with any
    angle at all, which is precisely why `models/schemas/motion.py` forbids
    treating these quantities as camera motion anywhere upstream. The focal used
    is the fused per-frame lens curve at *analysis* resolution — the resolution
    the flow was measured at — so a zoom changes the conversion as it should.

    The conversion is exact for a rotating camera. Image translation caused by
    real camera translation is indistinguishable here and is absorbed as
    rotation; the fusion bounds that error by dividing out the anchor-spaced
    trend, so it can only shape the path *between* anchors, never move one.

    Signs, derived from the CameraPath convention (camera +X right, +Y forward,
    +Z up) and image axes (x right, y down):

      * the camera yawing left (positive rotation about its +Z) sweeps world
        content toward image +x, so `+dx` implies `+yaw` about +Z;
      * the camera tilting up (positive rotation about its +X) sweeps content
        toward image +y, so `+dy` implies `+pitch` about +X;
      * `rotation_deg` is measured in image coordinates, where a positive angle
        turns content clockwise on screen; a camera rolling its top to the
        right (positive rotation about +Y) turns content counter-clockwise.
        Hence roll about +Y is the negation of `rotation_deg`.

    Returns (increments with a zero first row, number of frames matched to a
    measured transition).
    """
    n = len(frame_indices)
    increments = np.zeros((n, 3))
    matched = 0

    for i, frame_index in enumerate(frame_indices):
        mf = motion_by_index.get(frame_index)
        # The first frame of the shot has no preceding transition inside the shot
        # (the one before it crosses the cut, I4), and a frame whose transition
        # was never measured contributes no increment rather than a guessed one.
        if i == 0 or mf is None:
            continue
        raw = (mf.dx_pixels, mf.dy_pixels, mf.rotation_deg)
        if not all(np.isfinite(v) for v in raw):
            continue
        # The transition spans frames i-1 and i; during a zoom their focal
        # lengths differ, and the mean is the second-order midpoint estimate.
        fx = max(0.5 * float(fx_series[i - 1] + fx_series[i]), 1e-6)
        fy = max(0.5 * float(fy_series[i - 1] + fy_series[i]), 1e-6)
        yaw = float(np.arctan2(mf.dx_pixels, fx))
        pitch = float(np.arctan2(mf.dy_pixels, fy))
        roll = -float(np.radians(mf.rotation_deg))
        increments[i] = (pitch, roll, yaw)  # about camera +X, +Y, +Z
        matched += 1

    return increments, matched


def _integrate_rotation(increments: np.ndarray) -> np.ndarray:
    """Compose body-frame increments into an orientation path from identity.

    Right-multiplication because the increments are expressed in the camera's
    own frame at the previous instant: `q[i] = q[i-1] * exp(w[i])`.
    """
    n = len(increments)
    path = np.zeros((n, 4))
    current = quat_identity()
    for i in range(n):
        if i > 0:
            current = quat_normalize(quat_multiply(current, quat_exp(increments[i])))
        path[i] = current
    return path


def _dense_speed_profile(
    motion_by_index: dict[int, MotionFrame],
    frame_indices: list[int],
    times: np.ndarray,
) -> np.ndarray:
    """Measured image speed into each frame, pixels per second.

    Divided by the PTS-derived interval, not by `MotionFrame.dt` or `1 / fps`, so
    that `speed * dt` in the parameterisation recovers exactly the measured flow
    of each transition (I2).
    """
    n = len(frame_indices)
    speed = np.zeros(n)
    for i, frame_index in enumerate(frame_indices):
        mf = motion_by_index.get(frame_index)
        if i == 0 or mf is None:
            continue
        dt = float(times[i] - times[i - 1])
        magnitude = float(mf.flow_magnitude)
        if dt <= MIN_DT or not np.isfinite(magnitude):
            continue
        speed[i] = max(magnitude, 0.0) / dt
    return speed


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


class _Anchors:
    """Anchor poses reduced to the frames of this shot that actually exist."""

    def __init__(self) -> None:
        self.rows: list[int] = []  # index into the shot's frame list
        self.positions: list[np.ndarray] = []
        self.quats: list[np.ndarray] = []
        self.confidence: list[float] = []

    def __len__(self) -> int:
        return len(self.rows)


def _collect_anchors(anchors: GeometryResult, row_of_frame: dict[int, int]) -> _Anchors:
    """Map a GeometryResult onto shot-local rows, dropping anything unusable.

    Non-finite poses are dropped rather than propagated: a single NaN anchor
    poisons every spline segment that touches it, and silently turns the whole
    shot into NaN. An anchor outside the shot belongs to another coordinate
    system (I4) and is dropped too.
    """
    collected = _Anchors()
    raw_positions = np.asarray(anchors.positions, dtype=np.float64)
    positions = raw_positions.reshape(-1, 3) if raw_positions.size else np.zeros((0, 3))
    quats = list(anchors.quaternions or [])
    confidences = list(anchors.per_pose_confidence or [])

    chosen: dict[int, int] = {}
    for slot, frame_index in enumerate(anchors.frame_indices):
        row = row_of_frame.get(int(frame_index))
        if row is None or row in chosen:
            continue
        if slot >= len(positions) or slot >= len(quats):
            continue
        quat = np.asarray(quats[slot], dtype=np.float64).reshape(-1)
        if quat.size != 4 or not np.isfinite(positions[slot]).all() or not np.isfinite(quat).all():
            continue
        if float(np.linalg.norm(quat)) < 1e-9:
            continue
        chosen[row] = slot

    for row in sorted(chosen):
        slot = chosen[row]
        collected.rows.append(row)
        collected.positions.append(positions[slot].copy())
        collected.quats.append(quat_normalize(np.asarray(quats[slot], dtype=np.float64)))
        confidence = (
            float(confidences[slot]) if slot < len(confidences) else float(anchors.confidence)
        )
        if not np.isfinite(confidence):
            confidence = 0.0
        collected.confidence.append(float(np.clip(confidence, 0.0, 1.0)))

    return collected


# ---------------------------------------------------------------------------
# Lens
# ---------------------------------------------------------------------------


def _prior_lens_series(
    lens: list[LensFrame],
    shot: Shot,
    times: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """The pre-solve lens curve resampled onto the shot's frame times.

    Returns (focal_normalized, confidence, is_estimated). Monotone cubic, not a
    plain spline: a zoom that ramps and then holds would otherwise overshoot at
    the junction, which reads as the lens going too far and coming back — a
    move that is not in the source.
    """
    usable = [
        lf for lf in lens
        if shot.start_frame <= lf.frame_index <= shot.end_frame
        and np.isfinite(lf.timestamp)
        and np.isfinite(lf.focal_normalized)
        and lf.focal_normalized > 0
    ]
    if not usable:
        return (
            np.full(len(times), float(intrinsics.focal_normalized)),
            np.full(len(times), float(np.clip(intrinsics.confidence, 0.0, 1.0))),
            intrinsics.source != "user_override",
        )

    usable.sort(key=lambda lf: (lf.timestamp, lf.frame_index))
    control_t: list[float] = []
    control_f: list[float] = []
    control_c: list[float] = []
    for lf in usable:
        # Duplicate timestamps would make a zero-width spline segment; the first
        # value at a given instant wins.
        if control_t and lf.timestamp - control_t[-1] <= MIN_DT:
            continue
        control_t.append(float(lf.timestamp))
        control_f.append(float(lf.focal_normalized))
        control_c.append(float(np.clip(lf.confidence, 0.0, 1.0)))

    t = np.array(control_t)
    focal = monotone_resample(t, np.array(control_f), times)
    confidence = np.interp(times, t, np.array(control_c))
    is_estimated = any(lf.is_estimated for lf in usable)
    return focal, confidence, is_estimated


def _measured_focal_normalized(
    anchors: GeometryResult, intrinsics: CameraIntrinsics
) -> tuple[float | None, str]:
    """The solve's focal as `focal_px / long_edge` of the lens frame, or None.

    `anchors.focal_pixels` is in pixels of `focal_image_width`-wide images — the
    geometry resolution, which is not the flow-analysis resolution. It is
    converted through the horizontal FOV, which is resolution independent, and
    only then expressed against the intrinsics' own frame. Reading it against
    any other width silently shifts the FOV by the resolution ratio.
    """
    focal_px = anchors.focal_pixels
    if focal_px is None or not np.isfinite(focal_px) or focal_px <= 0:
        return None, "solver reported no focal length"
    if not anchors.focal_observable:
        return None, (
            "the solve could not constrain focal length, so its focal is the prior "
            "it was seeded with, not a measurement"
        )
    width_px = anchors.focal_image_width
    if width_px is None or width_px <= 0:
        return None, (
            "solver reported a focal length without the image width it refers to; "
            "it cannot be interpreted and was ignored"
        )
    fov = focal_pixels_to_fov(float(focal_px), int(width_px))
    if not (MIN_HORIZONTAL_FOV <= fov <= MAX_HORIZONTAL_FOV):
        return None, f"solved focal implies an unphysical {fov:.1f} degree FOV; ignored"
    long_edge = max(intrinsics.width, intrinsics.height, 1)
    focal_normalized = fov_to_focal_pixels(fov, intrinsics.width) / long_edge
    return focal_normalized, f"focal length measured by the solve ({fov:.1f} degrees horizontal)"


def fuse_lens_curve(
    anchors: GeometryResult,
    lens: list[LensFrame],
    frames_meta: list[FrameMetadata],
    shot: Shot,
    intrinsics: CameraIntrinsics,
) -> tuple[list[LensFrame], str]:
    """Per-source-frame lens curve for the shot, plus a provenance note.

    Two cases, decided by the solve's own evidence:

      * **Focal observable** and interpretable: the measured focal sets the
        curve's absolute level at the anchors, and the prior curve contributes
        only its relative shape (a zoom). Confidence is the solve's focal
        confidence, limited by the prior curve's own confidence wherever the
        curve departs from flat — the shape was never measured by the solve.
      * **Focal unobservable**: the curve is the prior, the solve's focal is not
        used and not presented as measured, and confidence is capped LOW. On a
        pure dolly a 14 degree FOV spread changed reprojection error by 0.001 px;
        reporting that "refined" value as a measurement would be fabricated
        certainty (I7).

    With no focal reported at all, the prior curve passes through unchanged.
    """
    shot_frames = _shot_frames(frames_meta, shot)
    if not shot_frames:
        return [], "no frames in shot"

    times = np.array([fm.time_seconds for fm in shot_frames], dtype=np.float64)
    spline_t = _spline_times(times, shot.id)
    prior_fn, prior_conf, prior_estimated = _prior_lens_series(lens, shot, spline_t, intrinsics)

    width, height = max(intrinsics.width, 1), max(intrinsics.height, 1)
    long_edge = max(width, height)
    aspect_fy = intrinsics.fy / intrinsics.fx if intrinsics.fx > 0 else 1.0
    lo_fn = fov_to_focal_pixels(MAX_HORIZONTAL_FOV, width) / long_edge
    hi_fn = fov_to_focal_pixels(MIN_HORIZONTAL_FOV, width) / long_edge

    measured, note = _measured_focal_normalized(anchors, intrinsics)
    row_of_frame = {fm.frame_index: row for row, fm in enumerate(shot_frames)}
    anchor_rows = sorted({
        row_of_frame[int(fi)] for fi in anchors.frame_indices if int(fi) in row_of_frame
    })

    if measured is not None:
        reference_rows = anchor_rows if anchor_rows else list(range(len(shot_frames)))
        reference = float(np.median(prior_fn[reference_rows]))
        ratio = measured / reference if reference > 0 else 1.0
        focal = np.clip(prior_fn * ratio, lo_fn, hi_fn)
        base = float(np.clip(anchors.focal_confidence, 0.0, 1.0))
        shape = np.abs(prior_fn / reference - 1.0) if reference > 0 else np.zeros(len(prior_fn))
        confidence = np.where(
            shape <= FOCAL_SHAPE_FLAT_TOLERANCE, base, np.minimum(base, prior_conf)
        )
        is_estimated = True
    else:
        focal = prior_fn
        confidence = prior_conf.copy()
        is_estimated = prior_estimated
        if not anchors.focal_observable:
            confidence = np.minimum(confidence, UNOBSERVABLE_FOCAL_CONFIDENCE)
            note += "; lens curve uses the prior with low confidence"

    curve = [
        LensFrame(
            frame_index=fm.frame_index,
            timestamp=float(fm.time_seconds),
            focal_normalized=float(focal[row]),
            fov_horizontal=focal_pixels_to_fov(float(focal[row]) * long_edge, width),
            fov_vertical=focal_pixels_to_fov(float(focal[row]) * long_edge * aspect_fy, height),
            confidence=float(confidence[row]),
            is_estimated=bool(is_estimated),
        )
        for row, fm in enumerate(shot_frames)
    ]
    return curve, note


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fuse_trajectory(
    anchors: GeometryResult,
    motion_frames: list[MotionFrame],
    frames_meta: list[FrameMetadata],
    shot: Shot,
    lens: list[LensFrame],
    intrinsics: CameraIntrinsics,
    analysis_size: tuple[int, int],
    fidelity: MotionFidelity = MotionFidelity.EXACT,
    translation_observable: bool = True,
) -> list[CameraPose]:
    """One `CameraPose` per source frame of the shot, anchors hit exactly.

    Timestamps come from `frames_meta`, i.e. from container PTS (I2) — never
    from `index / fps`, so a variable-rate source keeps its real timing.
    `analysis_size` is (width, height) of the images the motion signature was
    measured on.

    `translation_observable=False` (or a solve that itself reports translation
    as unobservable) produces a rotation-only trajectory whose position is the
    first anchor's centre at every frame. That is the honest answer for a pure
    pan, a zoom, or a distant subject: there is no measurable baseline, and
    integrating flow into one would invent a dolly that never happened (I6,
    I7). Anchor orientations are still hit exactly in that mode.

    Fewer than two usable anchors degrades to a 2D motion proxy — rotation
    integrated from the dense signal, constant position, every pose labelled
    `MOTION_PROXY_2D` with LOW confidence. It never raises on degenerate input
    (I12).
    """
    shot_frames = _shot_frames(frames_meta, shot)
    if not shot_frames:
        log.warning(
            "shot %d has no frame metadata in range %d-%d; no poses emitted",
            shot.id, shot.start_frame, shot.end_frame,
        )
        return []

    frame_indices = [fm.frame_index for fm in shot_frames]
    times = np.array([fm.time_seconds for fm in shot_frames], dtype=np.float64)
    spline_t = _spline_times(times, shot.id)
    row_of_frame = {fi: row for row, fi in enumerate(frame_indices)}
    # The transition INTO the shot's first frame crosses the cut (I4).
    motion_by_index = {
        mf.frame_index: mf for mf in motion_frames
        if shot.start_frame < mf.frame_index <= shot.end_frame
    }

    gain = RESIDUAL_GAIN.get(fidelity, 1.0)
    window = _jitter_window(spline_t)

    lens_curve, lens_note = fuse_lens_curve(anchors, lens, shot_frames, shot, intrinsics)
    focal_series = np.array([lf.focal_normalized for lf in lens_curve], dtype=np.float64)
    fov_series = np.array([lf.fov_horizontal for lf in lens_curve], dtype=np.float64)
    log.info("shot %d lens: %s", shot.id, lens_note)

    analysis_long = float(max(int(analysis_size[0]), int(analysis_size[1]), 1))
    aspect_fy = intrinsics.fy / intrinsics.fx if intrinsics.fx > 0 else 1.0
    fx_series = focal_series * analysis_long
    increments, matched = _dense_rotation_increments(
        motion_by_index, frame_indices, fx_series, fx_series * aspect_fy
    )
    has_dense = matched > 0
    # Raw: the residual gain is chosen later from held-out anchors, then scaled by
    # fidelity. Filtering here would hide from that test what it is testing.
    dense_path = _integrate_rotation(increments)

    collected = _collect_anchors(anchors, row_of_frame)
    geometric = len(collected) >= MIN_ANCHORS_FOR_GEOMETRY

    result = None
    if geometric:
        try:
            result = _fuse_against_anchors(
                collected=collected,
                times=spline_t,
                dense_path=dense_path,
                speed=_apply_residual_gain(
                    _dense_speed_profile(motion_by_index, frame_indices, spline_t), window, gain
                ),
                motion_by_index=motion_by_index,
                frame_indices=frame_indices,
                anchor_source=anchors.source,
                has_dense=has_dense,
                translation_observable=translation_observable and anchors.translation_observable,
                window=window,
                fidelity_gain=gain,
                shot_id=shot.id,
            )
            if not (np.isfinite(result[0]).all() and np.isfinite(result[1]).all()):
                log.error("shot %d: fusion produced non-finite poses; degrading", shot.id)
                result = None
        except Exception:  # noqa: BLE001 - I12: a fusion failure degrades, never crashes
            log.exception("shot %d: anchor fusion failed; degrading to motion proxy", shot.id)
            result = None

    if result is None:
        geometric = False
        result = _degrade_to_motion_proxy(
            collected=collected,
            dense_path=dense_path,
            motion_by_index=motion_by_index,
            frame_indices=frame_indices,
            has_dense=has_dense,
            shot_id=shot.id,
        )

    quats, positions, confidences, sources = result
    anchor_rows = set(collected.rows) if geometric else set()

    poses: list[CameraPose] = []
    for row, fm in enumerate(shot_frames):
        poses.append(
            CameraPose(
                frame_index=fm.frame_index,
                timestamp=float(fm.time_seconds),
                position=[float(v) for v in positions[row]],
                quaternion=[float(v) for v in quats[row]],
                fov_horizontal=float(fov_series[row]),
                focal_normalized=float(focal_series[row]),
                confidence=float(np.clip(confidences[row], 0.0, 1.0)),
                solver_source=sources[row],
                is_anchor=row in anchor_rows,
            )
        )
    return poses


def _fuse_against_anchors(
    *,
    collected: _Anchors,
    times: np.ndarray,
    dense_path: np.ndarray,
    speed: np.ndarray,
    motion_by_index: dict[int, MotionFrame],
    frame_indices: list[int],
    anchor_source: SolverSource,
    has_dense: bool,
    translation_observable: bool,
    window: int = 5,
    fidelity_gain: float = 1.0,
    shot_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SolverSource]]:
    """The full fusion path: >= 2 anchors plus (optionally) a dense signal."""
    n = len(times)
    rows = np.array(collected.rows, dtype=int)
    anchor_times = times[rows]
    anchor_quats = np.array(collected.quats, dtype=np.float64)
    anchor_positions = np.array(collected.positions, dtype=np.float64)

    # --- rotation -----------------------------------------------------------
    order = np.argsort(rows)
    rows, anchor_times = rows[order], anchor_times[order]
    anchor_quats, anchor_positions = anchor_quats[order], anchor_positions[order]
    anchor_path = resample_rotations(anchor_times, anchor_quats, times)
    _extrapolate_rotation(anchor_path, rows, times)

    data_gain = 0.0
    if has_dense and len(rows) >= MIN_ANCHORS_FOR_HOLDOUT and fidelity_gain > 0:
        data_gain, scores = _holdout_residual_gain(rows, anchor_quats, times, dense_path, window)
        log.info(
            "shot %d residual gain %.2f from held-out anchors (mean error by gain: %s)",
            shot_id, data_gain, ", ".join(f"{g:g}:{e:.3f}deg" for g, e in scores.items()),
        )
    applied = data_gain * fidelity_gain

    quats = anchor_path.copy()
    if applied > 0:
        residual = _corrected_residual(
            np.arange(n), rows, times, dense_path, window
        )
        for i in range(n):
            quats[i] = quat_normalize(quat_multiply(anchor_path[i], quat_exp(applied * residual[i])))
    quats[rows] = anchor_quats
    # Consecutive quaternions in one hemisphere, so a per-component consumer
    # (Blender F-curves) never sees a sign flip as a 360 degree spin.
    quats = unroll_quaternions(quats)

    # --- translation --------------------------------------------------------
    if not translation_observable:
        # Rotation-only trajectory. Holding the first anchor's centre is the
        # only honest position when no baseline was measured (I6, I7).
        positions = np.tile(anchor_positions[0], (n, 1))
    else:
        # Parameterise by measured motion, not by time: two anchors 30 frames
        # apart say nothing about *when* the camera covered the ground between
        # them, but the dense flow profile does (I1). Outside the anchor range
        # Hermite holds the end anchor: flow cannot say how far the camera went.
        param = speed_warped_parameterization(times, speed)
        positions = hermite_resample(param[rows], anchor_positions, param)
        positions[rows] = anchor_positions
        _extrapolate_positions(positions, rows, times)

    # --- confidence and provenance -----------------------------------------
    anchor_confidence = np.array(collected.confidence, dtype=np.float64)
    base = np.interp(times, anchor_times, anchor_confidence)
    confidences = np.zeros(n)
    interpolated_source = SolverSource.FUSED if has_dense else SolverSource.INTERPOLATED
    sources = [interpolated_source] * n

    for i, frame_index in enumerate(frame_indices):
        mf = motion_by_index.get(frame_index)
        dense_term = 1.0
        if mf is not None and np.isfinite(mf.confidence):
            dense_term = DENSE_CONFIDENCE_FLOOR + (1.0 - DENSE_CONFIDENCE_FLOOR) * float(
                np.clip(mf.confidence, 0.0, 1.0)
            )
        confidences[i] = base[i] * dense_term
        gap = max(anchor_times[0] - times[i], times[i] - anchor_times[-1], 0.0)
        if gap > 0:
            # Extrapolated, not interpolated: evidence thins with distance from the
            # last anchor even while the rate is still held.
            confidences[i] *= float(np.exp(-gap / EXTRAPOLATION_DECAY_SECONDS))

    for slot, row in enumerate(collected.rows):
        confidences[row] = collected.confidence[slot]
        sources[row] = anchor_source

    return quats, positions, confidences, sources


def _decayed_elapsed(dt: float) -> float:
    """Distance covered in `dt` at a unit rate that holds for
    EXTRAPOLATION_CONSTANT_SECONDS and then decays with time constant
    EXTRAPOLATION_DECAY_SECONDS."""
    dt = abs(dt)
    hold = EXTRAPOLATION_CONSTANT_SECONDS
    if dt <= hold:
        return dt
    tau = EXTRAPOLATION_DECAY_SECONDS
    return hold + tau * (1.0 - float(np.exp(-(dt - hold) / tau)))


def _extrapolate_rotation(path: np.ndarray, rows: np.ndarray, times: np.ndarray) -> None:
    """Continue rotation past the end anchors at the end segments' decaying rate.
    In place. SQUAD holds its endpoints, which freezes an untracked tail."""
    if len(rows) < 2:
        return
    first, second, last, before = rows[0], rows[1], rows[-1], rows[-2]
    rate = quat_log(quat_relative(path[before], path[last])) / max(times[last] - times[before], MIN_DT)
    for i in range(last + 1, len(times)):
        path[i] = quat_normalize(quat_multiply(path[last], quat_exp(rate * _decayed_elapsed(times[i] - times[last]))))
    rate = quat_log(quat_relative(path[second], path[first])) / max(times[second] - times[first], MIN_DT)
    for i in range(0, first):
        path[i] = quat_normalize(quat_multiply(path[first], quat_exp(rate * _decayed_elapsed(times[first] - times[i]))))


def _extrapolate_positions(positions: np.ndarray, rows: np.ndarray, times: np.ndarray) -> None:
    """Continue position past the end anchors at the end segments' decaying
    velocity. In place."""
    if len(rows) < 2:
        return
    first, second, last, before = rows[0], rows[1], rows[-1], rows[-2]
    velocity = (positions[last] - positions[before]) / max(times[last] - times[before], MIN_DT)
    for i in range(last + 1, len(times)):
        positions[i] = positions[last] + velocity * _decayed_elapsed(times[i] - times[last])
    velocity = (positions[first] - positions[second]) / max(times[second] - times[first], MIN_DT)
    for i in range(0, first):
        positions[i] = positions[first] + velocity * _decayed_elapsed(times[first] - times[i])


def _corrected_residual(
    seg_rows: np.ndarray,
    ctrl_rows: np.ndarray,
    times: np.ndarray,
    dense_path: np.ndarray,
    window: int,
) -> np.ndarray:
    """High-passed rotation residual of the dense path over `seg_rows`, relative to
    its own SQUAD trend through `ctrl_rows`, forced to zero at the control rows so
    composing it never moves an anchor. Rotation vectors, radians, (len(seg), 3).

    Control rows outside the segment still shape the trend; only rows inside it
    are used for the zero-at-anchor correction.
    """
    seg_times = times[seg_rows]
    trend = resample_rotations(times[ctrl_rows], dense_path[ctrl_rows], seg_times)
    residual = np.array([quat_log(quat_relative(trend[j], dense_path[r])) for j, r in enumerate(seg_rows)])
    residual = residual - _moving_average(residual, window)
    inside = [k for k, r in enumerate(seg_rows) if r in set(ctrl_rows.tolist())]
    if inside:
        at = seg_times[inside]
        for d in range(3):
            residual[:, d] -= np.interp(seg_times, at, residual[inside, d])
    return residual


def _holdout_residual_gain(
    rows: np.ndarray,
    anchor_quats: np.ndarray,
    times: np.ndarray,
    dense_path: np.ndarray,
    window: int,
) -> tuple[float, dict[float, float]]:
    """Pick the residual gain that best predicts anchors the fill has not seen.

    For each held-out interior anchor, SQUAD and the residual are rebuilt from its
    neighbours alone (locally — see HOLDOUT_LOCAL_ANCHORS) and the prediction at
    its time is compared with the anchor. Returns (gain, mean error in degrees per
    gain). Gain 0 wins unless a nonzero gain beats it by
    HOLDOUT_REQUIRED_IMPROVEMENT.
    """
    interior = np.arange(1, len(rows) - 1)
    if len(interior) > MAX_HOLDOUT_ANCHORS:
        interior = np.unique(np.linspace(1, len(rows) - 2, MAX_HOLDOUT_ANCHORS).round().astype(int))

    errors: dict[float, list[float]] = {g: [] for g in HOLDOUT_GAINS}
    half = max(1, window // 2)
    for k in interior:
        lo = max(0, k - HOLDOUT_LOCAL_ANCHORS)
        hi = min(len(rows), k + HOLDOUT_LOCAL_ANCHORS + 1)
        keep = [j for j in range(lo, hi) if j != k]
        ctrl = rows[keep]
        predicted_base = resample_rotations(times[ctrl], anchor_quats[keep], times[[rows[k]]])[0]
        seg_lo = max(0, rows[k - 1] - half)
        seg_hi = min(len(times), rows[k + 1] + half + 1)
        seg = np.arange(seg_lo, seg_hi)
        residual = _corrected_residual(seg, ctrl, times, dense_path, window)[rows[k] - seg_lo]
        for g in HOLDOUT_GAINS:
            prediction = predicted_base if g == 0 else quat_multiply(predicted_base, quat_exp(g * residual))
            errors[g].append(float(np.degrees(quat_angular_distance(prediction, anchor_quats[k]))))

    scores = {g: float(np.mean(v)) for g, v in errors.items() if v}
    if not scores:
        return 0.0, {}
    best = min(scores, key=scores.get)
    if best > 0 and scores[best] > HOLDOUT_REQUIRED_IMPROVEMENT * scores[0.0]:
        best = 0.0
    return float(best), scores


def _degrade_to_motion_proxy(
    *,
    collected: _Anchors,
    dense_path: np.ndarray,
    motion_by_index: dict[int, MotionFrame],
    frame_indices: list[int],
    has_dense: bool,
    shot_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SolverSource]]:
    """Rotation-only fallback for 0 or 1 anchors. Honest, LOW confidence, no raise.

    With one anchor the dense path is re-based so that anchor's orientation and
    centre define the frame of reference; with none the path starts from the
    identity at the origin. Position is constant either way: a single pose
    defines no baseline, and integrating flow into one would invent a dolly (I6).
    Every pose is labelled `MOTION_PROXY_2D` and none is flagged as an anchor,
    so nothing downstream can mistake this for a geometric solve (I7).
    """
    n = len(frame_indices)
    log.warning(
        "shot %d: %d usable anchors, below the %d needed for a geometric path; "
        "degrading to a rotation-only 2D motion proxy",
        shot_id, len(collected), MIN_ANCHORS_FOR_GEOMETRY,
    )

    quats = np.zeros((n, 4))
    if len(collected) >= 1:
        anchor_row = collected.rows[0]
        anchor_quat = collected.quats[0]
        for i in range(n):
            quats[i] = quat_normalize(
                quat_multiply(anchor_quat, quat_relative(dense_path[anchor_row], dense_path[i]))
            )
        positions = np.tile(collected.positions[0], (n, 1))
    else:
        quats = np.array([quat_normalize(q) for q in dense_path])
        positions = np.zeros((n, 3))
    quats = unroll_quaternions(quats)

    evidence = np.zeros(n)
    if has_dense:
        for i, frame_index in enumerate(frame_indices):
            mf = motion_by_index.get(frame_index)
            if mf is not None and np.isfinite(mf.confidence):
                evidence[i] = float(np.clip(mf.confidence, 0.0, 1.0))
        if n > 1:
            # The first frame is the reference orientation; it is as trustworthy
            # as the first transition that is measured against it.
            evidence[0] = evidence[1]
    # The proxy's ceiling dominates: whatever the dense signal's internal
    # agreement, there is no geometric evidence behind these poses.
    confidences = DEGRADED_MAX_CONFIDENCE * evidence

    sources = [SolverSource.MOTION_PROXY_2D] * n
    return quats, positions, confidences, sources
