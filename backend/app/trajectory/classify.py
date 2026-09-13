"""Post-hoc description of the recovered move (spec section 13).

This module only DESCRIBES a trajectory that has already been solved. Nothing here
feeds back into reconstruction: a label is a summary for a human, never a prior.

Method: translation is decomposed into the camera's own axes at every frame
(forward -> dolly, right -> truck, up -> pedestal). Rotation is described in the
gravity-aligned world (the solve orients it Z-up): pan is change of HEADING, tilt
of ELEVATION, roll of BANK — how a camera operator describes a move. Body-frame
rates are the wrong basis for labels: a camera pitched 6 deg down that orbits 104
deg picks up 11 deg of body-frame roll (104 x sin 6.3), which is physically true
and would be reported as a roll nobody performed. Each component's rate is segmented into
sustained intervals, so a shot that pans and then tilts gets two labels with their
own time ranges rather than one blurred verdict. Compound moves (orbit/arc, crane,
rise-and-tilt, spiral, FPV) are recognised from co-occurring components.

Honesty rule (I6/I7): when translation was not observable, no translation-based
label is emitted at all. A pan must never be described as a truck because its
positions happened to be held at a point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.geometry.conventions import camera_axes
from app.models.schemas.motion import LensFrame
from app.models.schemas.trajectory import CameraPose, ClassifiedMove, Kinematics, MotionLabel

#: A rotation component counts as moving above this body rate, deg/s.
ROTATION_RATE_THRESHOLD = 3.0
#: ...and a labelled rotation segment must accumulate at least this much, deg.
ROTATION_MIN_ANGLE = 4.0
#: A translation component must carry this share of the frame's speed to count.
TRANSLATION_SHARE_THRESHOLD = 0.45
#: ...and a labelled translation segment must cover this share of the path.
TRANSLATION_MIN_PATH_SHARE = 0.15
#: Gaps shorter than this merge neighbouring segments; shorter segments are dropped.
MERGE_GAP_SECONDS = 0.25
MIN_SEGMENT_SECONDS = 0.3
#: Focal change needed to call a zoom (ratio of focal lengths).
ZOOM_RATIO_THRESHOLD = 1.08
#: Signature jitter above which the move is described as handheld.
HANDHELD_JITTER = 0.25
#: Orbit/arc: yaw while trucking the opposite way (keeps a subject centred).
ARC_MIN_YAW = 15.0
ORBIT_MIN_YAW = 90.0
#: Banking roll amplitude that, with forward travel, reads as FPV.
FPV_ROLL_AMPLITUDE = 8.0


@dataclass
class _Segment:
    start: int
    end: int  # inclusive row
    sign: int
    magnitude: float


def _segments(rate: np.ndarray, times: np.ndarray, threshold: float) -> list[_Segment]:
    """Sustained same-sign intervals where |rate| exceeds `threshold`."""
    n = len(rate)
    if n < 2:
        return []
    active = np.abs(rate) > threshold
    sign = np.sign(rate).astype(int)
    raw: list[list[int]] = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and active[j + 1] and sign[j + 1] == sign[i]:
            j += 1
        raw.append([i, j, int(sign[i])])
        i = j + 1
    merged: list[list[int]] = []
    for seg in raw:
        if merged and seg[2] == merged[-1][2] and times[seg[0]] - times[merged[-1][1]] <= MERGE_GAP_SECONDS:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    out = []
    for s, e, sg in merged:
        if times[e] - times[s] < MIN_SEGMENT_SECONDS:
            continue
        dt = np.diff(times[s:e + 2]) if e + 1 < n else np.diff(times[s:e + 1])
        vals = rate[s:s + len(dt)]
        out.append(_Segment(s, e, sg, float(np.abs(np.sum(vals * dt)))))
    return out


def _operator_rates(poses: list[CameraPose], times: np.ndarray) -> np.ndarray:
    """Per-frame [heading, elevation, bank] rates in deg/s, world Z up.

    heading   - azimuth of the view direction (+ = turning left, matching +yaw)
    elevation - angle of the view direction above the horizon (+ = tilting up)
    bank      - angle of the right axis out of the horizontal (+ = rolling right)
    Heading is unwrapped so a full orbit does not jump at +-180 deg.
    """
    view = np.array([camera_axes(np.array(p.quaternion))[1] for p in poses])
    right = np.array([camera_axes(np.array(p.quaternion))[0] for p in poses])
    heading = np.unwrap(np.arctan2(-view[:, 0], view[:, 1]))
    elevation = np.arcsin(np.clip(view[:, 2], -1.0, 1.0))
    bank = -np.arcsin(np.clip(right[:, 2], -1.0, 1.0))
    angles = np.degrees(np.stack([heading, elevation, bank], axis=1))
    return np.gradient(angles, times, axis=0)


def classify_motion(
    poses: list[CameraPose],
    kinematics: list[Kinematics],
    lens: list[LensFrame],
    *,
    jitter_score: float,
    translation_observable: bool,
    units: str = "normalized",
) -> tuple[list[ClassifiedMove], str]:
    """Labels with time ranges and strengths, plus a one-paragraph summary."""
    if len(poses) < 2 or len(kinematics) != len(poses):
        return [], "Too few frames to describe a camera move."

    times = np.array([p.timestamp for p in poses])
    rates = _operator_rates(poses, times)  # heading, elevation, bank deg/s
    moves: list[ClassifiedMove] = []

    def add(label: MotionLabel, seg: _Segment, strength: float, description: str) -> None:
        moves.append(ClassifiedMove(
            label=label, strength=float(np.clip(strength, 0.0, 1.0)),
            start_time=float(times[seg.start]), end_time=float(times[seg.end]),
            description=description,
        ))

    whole = _Segment(0, len(poses) - 1, 0, 0.0)

    # --- rotation ------------------------------------------------------------
    yaw_segments = [s for s in _segments(rates[:, 0], times, ROTATION_RATE_THRESHOLD) if s.magnitude >= ROTATION_MIN_ANGLE]
    pitch_segments = [s for s in _segments(rates[:, 1], times, ROTATION_RATE_THRESHOLD) if s.magnitude >= ROTATION_MIN_ANGLE]
    roll_segments = [s for s in _segments(rates[:, 2], times, ROTATION_RATE_THRESHOLD) if s.magnitude >= ROTATION_MIN_ANGLE]

    # --- translation in the camera's own axes -------------------------------
    trans_segments: dict[str, list[_Segment]] = {"forward": [], "right": [], "up": []}
    path = 0.0
    if translation_observable:
        velocity = np.array([k.linear_velocity for k in kinematics])
        speed = np.linalg.norm(velocity, axis=1)
        path = float(np.sum(speed[:-1] * np.diff(times)))
        if path > 1e-9:
            axes = [camera_axes(np.array(p.quaternion)) for p in poses]
            comps = {
                "right": np.array([v @ a[0] for v, a in zip(velocity, axes)]),
                "forward": np.array([v @ a[1] for v, a in zip(velocity, axes)]),
                "up": np.array([v @ a[2] for v, a in zip(velocity, axes)]),
            }
            for name, comp in comps.items():
                share = np.where(speed > 1e-9, comp / np.maximum(speed, 1e-9), 0.0)
                gated = np.where(np.abs(share) >= TRANSLATION_SHARE_THRESHOLD, comp, 0.0)
                segs = _segments(gated, times, 1e-9)
                trans_segments[name] = [s for s in segs if s.magnitude >= TRANSLATION_MIN_PATH_SHARE * path]

    # --- compound moves (claimed before their components are listed) --------
    claimed: set[int] = set()
    if translation_observable:
        for ys in yaw_segments:
            for ts in trans_segments["right"]:
                overlap = min(ys.end, ts.end) - max(ys.start, ts.start)
                # Yawing left while trucking right (or vice versa) keeps a subject centred.
                # Trucking right while panning LEFT keeps a subject centred ahead:
                # in this convention (+yaw = left, +right = right) the signs match.
                if overlap > 0.5 * (ys.end - ys.start) and ys.sign == ts.sign:
                    label = MotionLabel.ORBIT if ys.magnitude >= ORBIT_MIN_YAW else (
                        MotionLabel.ARC if ys.magnitude >= ARC_MIN_YAW else None)
                    if label is not None:
                        add(label, _Segment(max(ys.start, ts.start), min(ys.end, ts.end), 0, 0.0),
                            ys.magnitude / 180.0,
                            f"{ys.magnitude:.0f} deg of yaw while travelling sideways about a centre")
                        claimed.update({id(ys), id(ts)})
        for us in trans_segments["up"]:
            for ps in pitch_segments:
                if min(us.end, ps.end) - max(us.start, ps.start) > 0.5 * (us.end - us.start):
                    rising = us.sign > 0
                    tilting_down = ps.sign < 0
                    label = MotionLabel.RISE_AND_TILT if rising and tilting_down else MotionLabel.CRANE
                    add(label, _Segment(max(us.start, ps.start), min(us.end, ps.end), 0, 0.0),
                        min(1.0, ps.magnitude / 30.0 + 0.3),
                        f"vertical travel with {ps.magnitude:.0f} deg of tilt")
                    claimed.update({id(us), id(ps)})
        if trans_segments["forward"] and roll_segments:
            amplitude = float(np.max(np.abs(np.cumsum(rates[:-1, 2] * np.diff(times)))))
            if amplitude >= FPV_ROLL_AMPLITUDE:
                add(MotionLabel.FPV, whole, amplitude / 30.0,
                    f"forward flight banking up to {amplitude:.0f} deg")
                # FPV subsumes its own banking and forward travel.
                claimed.update(id(x) for x in roll_segments + trans_segments["forward"])
        total_yaw = sum(s.magnitude for s in yaw_segments)
        if total_yaw >= 270.0 and trans_segments["up"]:
            add(MotionLabel.SPIRAL, whole, total_yaw / 720.0, f"{total_yaw:.0f} deg of yaw while rising/falling")

    # --- components -----------------------------------------------------------
    for s in yaw_segments:
        if id(s) in claimed:
            continue
        add(MotionLabel.PAN_LEFT if s.sign > 0 else MotionLabel.PAN_RIGHT, s, s.magnitude / 45.0,
            f"{s.magnitude:.0f} deg pan {'left' if s.sign > 0 else 'right'}")
    for s in pitch_segments:
        if id(s) in claimed:
            continue
        add(MotionLabel.TILT_UP if s.sign > 0 else MotionLabel.TILT_DOWN, s, s.magnitude / 30.0,
            f"{s.magnitude:.0f} deg tilt {'up' if s.sign > 0 else 'down'}")
    for s in roll_segments:
        if id(s) in claimed:
            continue
        add(MotionLabel.ROLL, s, s.magnitude / 30.0, f"{s.magnitude:.0f} deg roll")
    names = {"forward": (MotionLabel.DOLLY_IN, MotionLabel.DOLLY_OUT, "dolly"),
             "right": (MotionLabel.TRUCK_RIGHT, MotionLabel.TRUCK_LEFT, "truck"),
             "up": (MotionLabel.PEDESTAL_UP, MotionLabel.PEDESTAL_DOWN, "pedestal")}
    for key, segs in trans_segments.items():
        positive, negative, word = names[key]
        for s in segs:
            if id(s) in claimed:
                continue
            share = s.magnitude / max(path, 1e-9)
            add(positive if s.sign > 0 else negative, s, share,
                f"{word} covering {share:.0%} of the path")

    # --- lens and texture -----------------------------------------------------
    fovs = [lf.fov_horizontal for lf in lens if np.isfinite(lf.fov_horizontal)]
    if len(fovs) >= 2:
        tan = lambda f: np.tan(np.radians(f) / 2.0)  # noqa: E731
        ratio = float(tan(fovs[0]) / tan(fovs[-1]))
        if ratio >= ZOOM_RATIO_THRESHOLD or ratio <= 1.0 / ZOOM_RATIO_THRESHOLD:
            add(MotionLabel.ZOOM_IN if ratio > 1 else MotionLabel.ZOOM_OUT, whole,
                abs(np.log(ratio)) / np.log(2.0), f"focal length x{ratio:.2f}")
    if jitter_score >= HANDHELD_JITTER:
        add(MotionLabel.HANDHELD, whole, jitter_score, f"high-frequency shake (jitter {jitter_score:.2f})")

    components = {m.label for m in moves} - {MotionLabel.HANDHELD, MotionLabel.ZOOM_IN, MotionLabel.ZOOM_OUT}
    if len(components) >= 3:
        add(MotionLabel.MIXED_6DOF, whole, min(1.0, len(components) / 5.0),
            f"{len(components)} simultaneous or sequential motion components")
    if not moves:
        add(MotionLabel.STATIC, whole, 1.0, "no sustained rotation, translation or zoom")

    moves.sort(key=lambda m: (m.start_time, -m.strength))
    return moves, _summary(moves, translation_observable, poses)


def _summary(moves: list[ClassifiedMove], translation_observable: bool, poses: list[CameraPose]) -> str:
    duration = poses[-1].timestamp - poses[0].timestamp
    if len(moves) == 1 and moves[0].label is MotionLabel.STATIC:
        text = f"A locked-off, static camera for {duration:.1f} s."
    else:
        parts = []
        for m in moves:
            if m.label is MotionLabel.MIXED_6DOF:
                continue
            span = "" if m.end_time - m.start_time >= 0.95 * duration else f" ({m.start_time:.1f}-{m.end_time:.1f} s)"
            parts.append(f"{m.description}{span}")
        text = "; ".join(parts[:6]).capitalize() + f", over {duration:.1f} s."
    if not translation_observable:
        text += " Translation was not observable, so only rotation and lens are described."
    return text
