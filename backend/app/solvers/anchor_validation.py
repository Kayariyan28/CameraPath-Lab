"""Cross-check geometric anchor poses against the dense per-frame motion.

Why this exists: a structure-from-motion solver can register a frame into a
confidently wrong pose. On a synthetic orbit, COLMAP matched truth to 0.015 deg
in relative rotation for 20 keyframes, then — where the camera grazes a wall and
the frame fills with one flat, blurred surface — placed the next two keyframes
51 deg and 7 deg off, while reporting 0.77-0.84 px reprojection error. Nothing
inside the reconstruction flags it. The dense optical flow, measured
independently on every frame, does.

The test is an upper bound, so it can only reject, never invent. For a pinhole
camera rotating by theta, content at the principal point moves f*tan(theta)
pixels; roll shows up as in-plane image rotation. Translation and parallax only
ADD flow. So between two anchors

    rotation(a -> b)  <=  sum over frames of [ atan(centre_displacement / f) + |roll| ]

and an anchor pair that claims more rotation than the image motion allows is
inconsistent. The bound is loose when translation dominates the flow, which
means small registration errors can slip through. It is designed to catch the
gross ones — the kind that wreck a trajectory.

The decision is LOCAL, and that is deliberate. A first version kept the longest
globally consistent chain. On the orbit, where the bound is loose because lateral
motion floods the image with translational flow, the two bad anchors fitted under
it — and the search dropped a GOOD anchor, because skipping it made the chain
longer. An anchor is now rejected only as a spike: inconsistent with its
neighbour(s) while those neighbours are consistent with each other without it.
Under that rule a good anchor cannot be sacrificed to make room for bad ones; the
cost is that a run of several consecutive bad anchors is not caught here, which is
why this check backs up, rather than replaces, the solver's own registration
evidence (see colmap_solver.MISREGISTERED_*).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from app.core.logging import get_logger
from app.geometry.rotations import quat_angular_distance
from app.models.schemas.motion import MotionFrame
from app.solvers.base import GeometryResult

log = get_logger("solvers.anchor_validation")

#: Multiplicative slack on the flow bound. Absorbs focal uncertainty: the bound
#: scales with 1/f, and a 1.5x margin tolerates a focal prior that is 33% too
#: long before a genuine anchor could be wrongly rejected.
BOUND_MARGIN = 1.5

#: Additive slack, degrees — rotation below this is never grounds for rejection,
#: since it is within what sub-pixel flow noise and model error can hide.
BOUND_TOLERANCE_DEG = 3.0

#: Transitions whose dominant-model fit is this weak report an unreliable centre
#: displacement; their 90th-percentile flow magnitude is used instead, which
#: loosens the bound rather than risk understating it.
LOW_CONFIDENCE = 0.35


@dataclass
class RejectedAnchor:
    frame_index: int
    claimed_rotation_deg: float
    flow_bound_deg: float
    neighbour_frame: int

    def describe(self) -> str:
        return (
            f"frame {self.frame_index}: solver claims {self.claimed_rotation_deg:.1f} deg of "
            f"rotation from frame {self.neighbour_frame}, but the image motion allows at "
            f"most {self.flow_bound_deg:.1f} deg"
        )


def per_transition_rotation_bound(
    motion_frames: list[MotionFrame], focal_px: float
) -> dict[int, float]:
    """Upper bound on camera rotation (deg) for each transition, keyed by the
    TARGET frame index (the transition into that frame)."""
    bounds: dict[int, float] = {}
    for mf in motion_frames:
        displacement = float(np.hypot(mf.dx_pixels, mf.dy_pixels))
        if mf.confidence < LOW_CONFIDENCE or mf.inlier_ratio < LOW_CONFIDENCE:
            displacement = max(displacement, float(mf.flow_magnitude_p90))
        bounds[mf.frame_index] = float(
            np.degrees(np.arctan(displacement / max(focal_px, 1e-6))) + abs(mf.rotation_deg)
        )
    return bounds


def validate_anchor_rotations(
    result: GeometryResult,
    motion_frames: list[MotionFrame],
    focal_px: float,
    *,
    margin: float = BOUND_MARGIN,
    tolerance_deg: float = BOUND_TOLERANCE_DEG,
) -> tuple[GeometryResult, list[RejectedAnchor]]:
    """Drop anchors whose rotation the dense flow cannot explain.

    `focal_px` must be in the pixels of the images the flow was MEASURED at (the
    analysis resolution), not the solver's image width. Pass the smaller of the
    solved and prior focal: a smaller focal widens the bound, trading missed
    detections for no false rejections.

    A missing transition between two anchors makes their bound infinite: absence
    of evidence is never evidence of inconsistency.
    """
    n = result.pose_count
    if not result.succeeded or n < 3 or not motion_frames:
        return result, []

    order = np.argsort(result.frame_indices)
    frames = [int(result.frame_indices[i]) for i in order]
    quats = [np.asarray(result.quaternions[i]) for i in order]

    per_step = per_transition_rotation_bound(motion_frames, focal_px)
    lo, hi = frames[0], frames[-1]
    # prefix[k] = summed bound for transitions into frames lo+1 .. k; a gap makes
    # every interval spanning it unbounded (no evidence, no rejection).
    prefix: dict[int, float] = {lo: 0.0}
    running = 0.0
    for f in range(lo + 1, hi + 1):
        running += per_step.get(f, float("inf"))
        prefix[f] = running

    def check(i: int, j: int) -> tuple[bool, float, float]:
        a, b = min(i, j), max(i, j)
        claimed = float(np.degrees(quat_angular_distance(quats[a], quats[b])))
        total = prefix[frames[b]] - prefix[frames[a]]
        limit = float("inf") if not np.isfinite(total) else float(total)
        return claimed <= margin * limit + tolerance_deg, claimed, limit

    rejected: list[RejectedAnchor] = []
    drop: set[int] = set()
    for k in range(n):
        if k == 0:
            # Endpoint: spike if it disagrees with its neighbour while that
            # neighbour agrees with the next one.
            ok_prev, claimed, limit = check(0, 1)
            if not ok_prev and check(1, 2)[0]:
                drop.add(0)
                rejected.append(RejectedAnchor(frames[0], claimed, limit, frames[1]))
        elif k == n - 1:
            ok_prev, claimed, limit = check(k - 1, k)
            if not ok_prev and (k - 1) not in drop and check(k - 2, k - 1)[0]:
                drop.add(k)
                rejected.append(RejectedAnchor(frames[k], claimed, limit, frames[k - 1]))
        else:
            ok_prev, claimed, limit = check(k - 1, k)
            ok_next = check(k, k + 1)[0]
            if not ok_prev and not ok_next and check(k - 1, k + 1)[0]:
                drop.add(k)
                rejected.append(RejectedAnchor(frames[k], claimed, limit, frames[k - 1]))

    if not drop:
        return result, []
    kept_sorted = [k for k in range(n) if k not in drop]
    kept = set(kept_sorted)

    positions = np.asarray(result.positions)[order]
    per_pose = (
        [result.per_pose_confidence[i] for i in order]
        if len(result.per_pose_confidence) == n else []
    )
    note = (
        f"rejected {len(rejected)} of {n} anchors whose rotation the image motion cannot "
        f"explain ({'; '.join(r.describe() for r in rejected[:3])}"
        f"{'; ...' if len(rejected) > 3 else ''})"
    )
    for r in rejected:
        log.warning("anchor rejected: %s", r.describe())

    validated = replace(
        result,
        frame_indices=[frames[k] for k in kept_sorted],
        positions=positions[kept_sorted],
        quaternions=[quats[k] for k in kept_sorted],
        per_pose_confidence=[per_pose[k] for k in kept_sorted] if per_pose else [],
        # Inconsistent registrations are evidence the reconstruction is less
        # reliable overall, not just at the dropped frames.
        confidence=float(result.confidence * len(kept) / n),
        message=f"{result.message}; {note}" if result.message else note,
        succeeded=len(kept) >= 2,
    )
    return validated, rejected
