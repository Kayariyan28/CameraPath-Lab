"""Perceptual Motion Match — screen-space motion matching (architecture §6, spec §12).

This is the last geometric rung of the ladder and the most important fallback,
because the cases it handles are not rare: pure rotation, pure zoom, distant
scenery, low texture, near-zero parallax. In every one of those, translation is
genuinely *unobservable*, structure-from-motion is degenerate rather than merely
noisy, and a physical solver will happily return a confident, arbitrary baseline.

So this backend answers a different question. Not "where was the camera?" but
"what camera would produce this motion on screen?":

    minimise  Σ_t ‖ signature(canonical_scene seen by camera_θ, t)
                   − signature(source, t) ‖²

The signature is the same `MotionFrame` feature vector used everywhere else
(`dx_pixels`, `dy_pixels`, `rotation_deg`, `radial_flow`), so source and
candidate are compared in identical units. The result is explicitly *not* a
claim about the physical camera path — it is a camera whose screen-space motion
matches the reference, which is exactly what a downstream generative-video model
consumes.

Three design decisions worth reading before changing anything here:

**Closed-form initialisation.** For a rotating pinhole camera the image motion
is analytic: `dx = fx·tan(Δyaw)`, `dy = fy·tan(Δpitch)`, image rotation `= −Δroll`
(see `signature_to_angles` for the derivation and the sign conventions). That
inversion is exact for single-axis motion, so the optimiser starts at or very
near the answer and only has to reconcile the four measured channels against
each other. Starting from zero instead would make a 900-transition shot a
genuine optimisation problem rather than a refinement.

**Translation is held at zero unless parallax says otherwise.** Image-space
motion alone cannot fix a baseline — that is the whole reason this rung exists.
Where `parallax_score` shows no depth-dependent flow, position stays exactly at
the origin and `translation_observable` is False. Where parallax *does* support
translation, a view-axis dolly is recovered against the canonical scene's
reference depth, so the units are "fractions of the distance to the scene" —
normalized, never metres (I5) — and the message says so. Lateral truck versus
pan remains unresolvable here by construction; that ambiguity is what the
geometric backends are for.

**Temporal regularisation is weighted by measured confidence.** A smoothness
prior that acts everywhere would erase handheld jitter, and preserving that
jitter is the product (I10). So the smoothness residual for each link is scaled
by `1 − confidence` of the transitions it joins: where the measurement is
trusted the prior is switched off and the solution follows the data even when
the data is high-frequency; where tracking was poor the prior is allowed to pull
the estimate toward its neighbours rather than let a bad transition inject a
spike. The trade-off is deliberate and asymmetric — it degrades unmeasured
motion, never measured motion.

This backend always succeeds when there is any motion signature at all: a static
signature yields a static trajectory, which is the honest answer, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from app.core.logging import get_logger
from app.geometry.intrinsics import build_lens_curve, estimate_focal_from_homographies
from app.tracking.global_motion import decompose_similarity, local_affine_of_homography
from app.geometry.rotations import (
    quat_exp,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_to_matrix,
)
from app.models.schemas.motion import CameraIntrinsics, MotionFrame
from app.models.schemas.trajectory import SolverSource
from app.solvers.base import GeometryResult, SolveContext, failed_result

log = get_logger("solvers.perceptual")

#: Sample grid across the frame for the canonical scene. 7x7 = 49 points
#: over-determines the 4-DoF similarity plus the radial term comfortably, and a
#: full prediction still costs microseconds.
CANONICAL_GRID = 7

#: Grid inset from the frame edge, as a fraction of width/height. Points sitting
#: exactly on the border leave the frame under the slightest rotation, and a
#: sample that has left the frame is not something the measured signature saw
#: either.
CANONICAL_INSET = 0.10

#: Depth of the canonical scene, in its own units. Every sample sits at this
#: depth, which is precisely what makes any recovered translation *normalized*
#: (I5): it is measured in units of "distance to the canonical scene".
REFERENCE_DEPTH = 1.0

#: Depth below which a sample has swung behind the camera. Clamped rather than
#: dropped, so the residual stays finite and the optimiser is penalised for the
#: candidate instead of the problem silently changing size.
MIN_PROJECTION_DEPTH = 0.05

#: Largest per-transition rotation the inversion will accept, degrees. A larger
#: inter-frame angle means a missed cut or a tracking failure, not a camera
#: move — and `tan` near 90 degrees turns measurement noise into infinities.
MAX_INCREMENT_DEG = 30.0

#: Measurement noise floor for the pixel channels, px at analysis resolution.
#: Same justification as `tracking.global_motion.STATIC_FLOW_FLOOR`: a single LK
#: correspondence localises to roughly 0.3 px, and a model fitted over hundreds
#: of them is better but not by orders of magnitude. Used to whiten the
#: residual so pixels, degrees and depth-fractions are commensurable.
PIXEL_NOISE_FLOOR = 0.35

#: Noise floor for the image-rotation channel, degrees. `intrinsics.py` treats
#: 0.05 degrees as the threshold below which an image rotation carries no
#: information, so that is the floor here too.
ROTATION_NOISE_FLOOR_DEG = 0.05

#: Smoothness weight at *zero* measurement confidence. At 0.5 the prior is worth
#: half a noise-floor of data disagreement, so it can only move the solution
#: where the data is effectively silent. At full confidence the weight is zero
#: and the prior is switched off entirely (I10).
SMOOTHNESS_WEIGHT = 0.5

#: Parallax below which translation is held at exactly zero. Above it, the
#: radial-flow channel is attributed to a view-axis dolly rather than to the
#: lens, matching `geometry.intrinsics.build_lens_curve`'s AUTO reasoning from
#: the other side.
PARALLAX_SUPPORTS_TRANSLATION = 0.25

#: Net radial flow needed before a dolly is worth solving for at all, px.
MIN_RADIAL_FOR_DOLLY = 0.5

#: Largest per-transition forward step, as a fraction of the reference depth.
#: Beyond a third of the way to the scene in one frame the single-depth
#: canonical scene stops being a usable model of what the viewer saw.
MAX_FORWARD_STEP = 0.35

#: Confidence ceiling. A screen-space match is not a measurement of the physical
#: path, so it can never be as trustworthy as a successful geometric solve no
#: matter how well it fits — the fit being perfect only means the *look* was
#: reproduced. COLMAP's score is uncapped up to 1.0 for comparison.
PERCEPTUAL_MAX_CONFIDENCE = 0.55

#: Transitions per refinement block. The only coupling between transitions is
#: the nearest-neighbour smoothness term, so refining long shots in overlapping
#: blocks is near-exact and keeps cost linear instead of quadratic in the
#: trust-region solve.
MAX_BLOCK_TRANSITIONS = 512
BLOCK_OVERLAP = 8


# ---------------------------------------------------------------------------
# Closed-form image-motion model for a rotating pinhole camera.
#
# Derivation, in the CameraPath camera frame (+X right, +Y forward/view,
# +Z up) with the pixel axes u right and v down:
#
#     u − cx = fx · px / py          v − cy = −fy · pz / py
#
# Yaw is a right-handed rotation about the camera's own +Z (up), i.e. positive
# yaw turns the camera to the LEFT. A world point straight ahead at distance d
# has coordinates R_z(ψ)^T · (0, d, 0) = (d·sinψ, d·cosψ, 0) in the rotated
# frame, so it projects to u − cx = fx·tanψ: the content moves right when the
# camera turns left. Hence dx = +fx·tan(Δyaw).
#
# Pitch is right-handed about +X (right), so positive pitch tilts the camera UP.
# The same point becomes (0, d·cosθ, −d·sinθ), giving v − cy = +fy·tanθ: content
# moves down when the camera tilts up. Hence dy = +fy·tan(Δpitch).
#
# Roll is right-handed about +Y (forward). It maps image offsets by
# [[cosφ, sinφ], [−sinφ, cosφ]], and the image-rotation convention used by
# `tracking.global_motion` — atan2(a10 − a01, a00 + a11) — reads that as −φ.
# Hence rotation_deg = −degrees(Δroll). The sign is not cosmetic: getting it
# backwards produces a trajectory that rolls the wrong way while fitting
# everything else perfectly.
# ---------------------------------------------------------------------------


def signature_to_angles(
    dx_pixels: float, dy_pixels: float, rotation_deg: float, fx: float, fy: float
) -> tuple[float, float, float]:
    """Invert the closed-form relation: image signature -> (yaw, pitch, roll).

    Radians, right-handed about the camera's own axes. Exact for single-axis
    motion; first-order accurate for mixed motion, which is what makes it a good
    starting point rather than a final answer.
    """
    dyaw = float(np.arctan2(float(dx_pixels), max(float(fx), 1e-6)))
    dpitch = float(np.arctan2(float(dy_pixels), max(float(fy), 1e-6)))
    droll = float(-np.radians(float(rotation_deg)))
    return dyaw, dpitch, droll


def angles_to_signature(
    dyaw: float, dpitch: float, droll: float, fx: float, fy: float
) -> tuple[float, float, float]:
    """Forward direction of `signature_to_angles`. Exact inverse of it."""
    limit = np.radians(89.0)
    dx = float(fx) * float(np.tan(np.clip(dyaw, -limit, limit)))
    dy = float(fy) * float(np.tan(np.clip(dpitch, -limit, limit)))
    return dx, dy, float(-np.degrees(droll))


def angles_to_rotation_vector(dyaw: float, dpitch: float, droll: float) -> np.ndarray:
    """(yaw, pitch, roll) -> a rotation vector about the camera's own axes.

    Named rather than inlined because the axis assignment is a convention and
    conventions are where silent errors live (I8): pitch is about the camera's
    +X (right), roll about +Y (forward), yaw about +Z (up).

    The three angles are *not* Euler angles being interpolated — nothing here
    ever interpolates them (I3). They parameterise one rotation vector, which is
    turned into a quaternion by `quat_exp` and composed as a quaternion.
    """
    return np.array([dpitch, droll, dyaw], dtype=np.float64)


def rotation_vector_to_angles(rotvec: np.ndarray) -> tuple[float, float, float]:
    """Inverse of `angles_to_rotation_vector`."""
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    return float(r[2]), float(r[0]), float(r[1])


# ---------------------------------------------------------------------------
# Canonical scene + prediction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalScene:
    """A neutral scene to film, standing in for the source content.

    Points are a grid of image samples back-projected to a single depth. A
    single depth is the honest choice: with no depth variation the scene carries
    no parallax, so nothing in the prediction can pretend to resolve a dolly
    from a zoom. Recovered forward translation is therefore expressed in units
    of this depth, and is meaningless as a metric distance (I5).
    """

    points: np.ndarray
    """(P, 3) camera-frame positions at `REFERENCE_DEPTH`."""
    centre: np.ndarray
    """(2,) principal point, the origin for `dx`/`dy` and the radial term."""
    mean_radius: float
    """Mean sample distance from the centre, px. Converts a scale change into
    the radial-flow units the measured signature reports."""


def build_canonical_scene(intr: CameraIntrinsics) -> CanonicalScene:
    """Back-project an inset grid of image samples to the reference depth."""
    fx = max(float(intr.fx), 1e-6)
    fy = max(float(intr.fy), 1e-6)
    cx, cy = float(intr.cx), float(intr.cy)

    lo, hi = CANONICAL_INSET, 1.0 - CANONICAL_INSET
    us = np.linspace(lo * intr.width, hi * intr.width, CANONICAL_GRID)
    vs = np.linspace(lo * intr.height, hi * intr.height, CANONICAL_GRID)
    grid_u, grid_v = np.meshgrid(us, vs)
    u = grid_u.ravel()
    v = grid_v.ravel()

    # Image sample -> camera-frame ray, inverting the projection in the module
    # header: px = (u - cx)/fx * py, pz = -(v - cy)/fy * py.
    px = (u - cx) / fx * REFERENCE_DEPTH
    pz = -(v - cy) / fy * REFERENCE_DEPTH
    py = np.full_like(px, REFERENCE_DEPTH)
    points = np.column_stack([px, py, pz])

    radii = np.hypot(u - cx, v - cy)
    return CanonicalScene(
        points=points,
        centre=np.array([cx, cy], dtype=np.float64),
        mean_radius=float(radii.mean()),
    )


def project_points(
    points: np.ndarray, fx: float, fy: float, centre: np.ndarray
) -> np.ndarray:
    """Pinhole projection in the CameraPath camera frame (view along +Y)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    depth = np.maximum(p[:, 1], MIN_PROJECTION_DEPTH)
    u = centre[0] + fx * p[:, 0] / depth
    v = centre[1] - fy * p[:, 2] / depth
    return np.column_stack([u, v])


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares 4-DoF similarity (rotation + uniform scale + translation).

    Closed form, so the prediction has no inner optimisation. Mirrors what
    `tracking.global_motion` fits to real correspondences with RANSAC; here the
    correspondences are exact by construction, so plain least squares is the
    same estimator without the robustness machinery.
    """
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    p = src - mu_s
    q = dst - mu_d
    denom = float((p * p).sum())
    if denom < 1e-12:
        return np.eye(2), mu_d - mu_s
    cos_term = float((p * q).sum()) / denom
    sin_term = float((p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0]).sum()) / denom
    a = np.array([[cos_term, -sin_term], [sin_term, cos_term]])
    return a, mu_d - a @ mu_s


def _decompose_similarity(
    a: np.ndarray, t: np.ndarray, centre: np.ndarray
) -> tuple[float, float, float]:
    """(dx, dy, rotation_deg) from a similarity, matching the measured signature.

    Deliberately identical in definition to
    `tracking.global_motion._decompose_similarity`: translation is the
    displacement of the *image centre* under the model, and rotation is
    atan2(a10 − a01, a00 + a11). Comparing a differently-defined prediction
    against that measurement would be comparing two different quantities.
    """
    moved = a @ centre + t
    rotation = float(np.degrees(np.arctan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1])))
    return float(moved[0] - centre[0]), float(moved[1] - centre[1]), rotation


def _radial_flow(src: np.ndarray, dst: np.ndarray, centre: np.ndarray) -> float:
    """Mean outward flow component about the centre, px.

    Same definition as `tracking.global_motion._radial_flow`, for the same
    reason as `_decompose_similarity`.
    """
    rel = src - centre
    norms = np.linalg.norm(rel, axis=1)
    valid = norms > 1e-3
    if not valid.any():
        return 0.0
    unit = rel[valid] / norms[valid, None]
    return float(np.sum(unit * (dst - src)[valid], axis=1).mean())


def predict_signature(
    rotvec: np.ndarray,
    forward: float,
    fx_old: float,
    fy_old: float,
    fx_new: float,
    fy_new: float,
    scene: CanonicalScene,
) -> np.ndarray:
    """Image-space signature of the canonical scene under one camera increment.

    Returns [dx_pixels, dy_pixels, rotation_deg, radial_flow] — the four
    channels of the measured `MotionFrame` signature, computed by actually
    filming the canonical scene rather than by the small-angle approximation,
    so the prediction stays valid for a fast whip pan.

    `forward` is a step along the camera's own view axis in units of
    `REFERENCE_DEPTH`.
    """
    rot = quat_to_matrix(quat_exp(np.asarray(rotvec, dtype=np.float64)))
    # The increment rotates the camera's axes; a point's coordinates in the new
    # frame are therefore R^T applied to its coordinates relative to the new
    # camera origin. `points @ rot` is `(rot.T @ points.T).T`.
    translated = scene.points - np.array([0.0, forward * REFERENCE_DEPTH, 0.0])
    moved = translated @ rot

    src = project_points(scene.points, fx_old, fy_old, scene.centre)
    dst = project_points(moved, fx_new, fy_new, scene.centre)

    # The canonical points share one depth, so their motion is exactly a
    # homography even with a forward step. Reading the signature from that
    # homography's local affine at the centre is the definition the measurement
    # uses (tracking.global_motion). Fitting a similarity here instead biased
    # every recovered pan: on a synthetic 1.153 deg/frame pan with exact measured
    # motion it returned 1.054 deg/frame, 8.6% short.
    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is not None and np.isfinite(homography).all():
        centre = (float(scene.centre[0]), float(scene.centre[1]))
        dx, dy, rotation_deg, _ = decompose_similarity(
            local_affine_of_homography(np.asarray(homography, dtype=np.float64), centre), centre
        )
    else:
        a, t = _fit_similarity(src, dst)
        dx, dy, rotation_deg = _decompose_similarity(a, t, scene.centre)
    return np.array([dx, dy, rotation_deg, _radial_flow(src, dst, scene.centre)])


# ---------------------------------------------------------------------------
# The optimisation problem
# ---------------------------------------------------------------------------


@dataclass
class _MatchProblem:
    """One contiguous run of transitions to fit.

    Parameters are per-transition *increments* — three rotation-vector
    components, plus a forward step when translation is being solved. Increments
    rather than absolute orientations because the measurement is itself an
    increment: fitting absolutes would couple every transition to every other
    one through the chain and make a single bad frame shift the whole tail.
    """

    measured: np.ndarray          # (n, 4) dx, dy, rotation_deg, radial_flow
    frame_weights: np.ndarray     # (n,) sqrt of measured confidence
    fx_old: np.ndarray            # (n,)
    fy_old: np.ndarray            # (n,)
    fx_new: np.ndarray            # (n,)
    fy_new: np.ndarray            # (n,)
    scene: CanonicalScene
    solve_translation: bool

    @property
    def n(self) -> int:
        return len(self.measured)

    @property
    def block(self) -> int:
        return 4 if self.solve_translation else 3

    @property
    def channel_weights(self) -> np.ndarray:
        return np.array([
            1.0 / PIXEL_NOISE_FLOOR,
            1.0 / PIXEL_NOISE_FLOOR,
            1.0 / ROTATION_NOISE_FLOOR_DEG,
            1.0 / PIXEL_NOISE_FLOOR,
        ])

    def smoothness_scale(self) -> np.ndarray:
        """Per-parameter conversion from an increment difference to noise units.

        An angle difference becomes comparable to a pixel disagreement through
        the focal length; a forward-step difference through the mean sample
        radius, which is what turns a depth fraction into radial pixels.
        """
        fx = float(np.median(self.fx_old)) if self.n else 1.0
        angle_noise = PIXEL_NOISE_FLOOR / max(fx, 1e-6)
        scale = [angle_noise, angle_noise, angle_noise]
        if self.solve_translation:
            scale.append(PIXEL_NOISE_FLOOR / max(self.scene.mean_radius, 1e-6))
        return np.array(scale)

    def smoothness_weights(self) -> np.ndarray:
        """(n-1,) prior strength per link, from measured confidence (I10).

        Confidence enters as `1 − min(conf_a, conf_b)`: the link is only
        smoothed as much as its *worse* end justifies, so one well-measured
        neighbour is never enough to smooth away a badly-measured spike, and one
        badly-measured neighbour is never enough to smooth a genuine one.
        """
        if self.n < 2:
            return np.zeros(0)
        conf = np.clip(self.frame_weights ** 2, 0.0, 1.0)
        pair_conf = np.minimum(conf[:-1], conf[1:])
        return SMOOTHNESS_WEIGHT * (1.0 - pair_conf)

    # ------------------------------------------------------------- residuals

    def predict(self, params: np.ndarray) -> np.ndarray:
        p = params.reshape(self.n, self.block)
        out = np.empty((self.n, 4))
        for i in range(self.n):
            forward = float(p[i, 3]) if self.solve_translation else 0.0
            out[i] = predict_signature(
                p[i, :3], forward,
                float(self.fx_old[i]), float(self.fy_old[i]),
                float(self.fx_new[i]), float(self.fy_new[i]),
                self.scene,
            )
        return out

    def residuals(self, params: np.ndarray) -> np.ndarray:
        p = params.reshape(self.n, self.block)
        data = (self.predict(params) - self.measured) * self.channel_weights
        data *= self.frame_weights[:, None]
        if self.n < 2:
            return data.ravel()
        diff = (p[1:] - p[:-1]) / self.smoothness_scale()
        smooth = diff * self.smoothness_weights()[:, None]
        return np.concatenate([data.ravel(), smooth.ravel()])

    def sparsity(self) -> lil_matrix:
        """Block-banded structure of the Jacobian.

        Each transition's data residual depends only on its own parameters, and
        each smoothness residual only on the two blocks it joins. Handing that
        to `least_squares` lets it finite-difference a dozen column groups
        instead of one column per parameter — the difference between seconds and
        minutes on a long shot.
        """
        rows = 4 * self.n + (self.block * max(self.n - 1, 0))
        cols = self.block * self.n
        m = lil_matrix((rows, cols), dtype=int)
        for i in range(self.n):
            m[4 * i:4 * i + 4, self.block * i:self.block * (i + 1)] = 1
        base = 4 * self.n
        for i in range(max(self.n - 1, 0)):
            r0 = base + self.block * i
            m[r0:r0 + self.block, self.block * i:self.block * (i + 1)] = 1
            m[r0:r0 + self.block, self.block * (i + 1):self.block * (i + 2)] = 1
        return m

    def slice(self, start: int, stop: int) -> _MatchProblem:
        return _MatchProblem(
            measured=self.measured[start:stop],
            frame_weights=self.frame_weights[start:stop],
            fx_old=self.fx_old[start:stop],
            fy_old=self.fy_old[start:stop],
            fx_new=self.fx_new[start:stop],
            fy_new=self.fy_new[start:stop],
            scene=self.scene,
            solve_translation=self.solve_translation,
        )


def _refine(problem: _MatchProblem, initial: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run the trust-region solve, in overlapping blocks for long shots.

    Returns (parameters, converged). A failed or non-converging solve returns
    the initial closed-form estimate rather than raising: a worse answer is
    still an answer, and this rung is not allowed to fail (I12).
    """
    n = problem.n
    if n == 0:
        return initial, True

    if n <= MAX_BLOCK_TRANSITIONS:
        return _refine_block(problem, initial)

    out = initial.copy()
    ok = True
    step = MAX_BLOCK_TRANSITIONS - BLOCK_OVERLAP
    for start in range(0, n, step):
        stop = min(start + MAX_BLOCK_TRANSITIONS, n)
        block = problem.slice(start, stop)
        b = problem.block
        guess = initial[start * b:stop * b]
        refined, block_ok = _refine_block(block, guess)
        ok = ok and block_ok
        # Keep the interior only: the overlap exists so a block boundary does
        # not sit where the smoothness prior has nothing on one side.
        keep_from = 0 if start == 0 else BLOCK_OVERLAP // 2
        keep_to = stop - start
        out[(start + keep_from) * b:(start + keep_to) * b] = \
            refined[keep_from * b:keep_to * b]
        if stop >= n:
            break
    return out, ok


def _refine_block(problem: _MatchProblem, initial: np.ndarray) -> tuple[np.ndarray, bool]:
    try:
        result = least_squares(
            problem.residuals,
            initial,
            jac_sparsity=problem.sparsity(),
            method="trf",
            # The closed-form start is already close, so a modest iteration
            # budget is enough and keeps a pathological shot from stalling the
            # pipeline.
            max_nfev=60,
            xtol=1e-10,
            ftol=1e-10,
            verbose=0,
        )
    except Exception as exc:  # noqa: BLE001 - I12
        log.warning("perceptual refinement failed (%s); keeping the closed-form estimate", exc)
        return initial, False
    if not np.isfinite(result.x).all():
        log.warning("perceptual refinement produced non-finite parameters; keeping estimate")
        return initial, False
    return result.x, bool(result.status > 0)


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class PerceptualMatchBackend:
    """Screen-space motion matching. Pure numpy/scipy, always available."""

    @property
    def source(self) -> SolverSource:
        return SolverSource.PERCEPTUAL

    def available(self) -> tuple[bool, str]:
        # numpy and scipy are hard dependencies of the backend package, so this
        # rung is the one thing on the ladder that cannot be missing.
        return True, ""

    def suitable_for(self, context: SolveContext) -> tuple[bool, str]:
        if not context.motion_frames:
            return False, "no motion signature was measured for this shot"
        return True, ""

    # ------------------------------------------------------------------ solve

    def estimate(self, context: SolveContext) -> GeometryResult:
        total = context.shot.frame_count
        frames = [mf for mf in context.motion_frames if np.isfinite(mf.dt)]
        if not frames:
            return failed_result(
                self.source, "no motion signature to match against", total
            )

        width, height = context.analysis_size
        if width <= 0 or height <= 0:
            return failed_result(
                self.source, f"invalid analysis resolution {width}x{height}", total
            )
        # The signature is measured in analysis-resolution pixels, so the
        # intrinsics must be too. Skipping this rescale is a silent factor-of-two
        # error in every recovered angle.
        intr = context.intrinsics.scaled_to(width, height)
        scene = build_canonical_scene(intr)

        solve_translation, translation_note = self._translation_decision(context, frames)
        # With translation held at zero the only thing left that can explain a
        # radial flow is the lens, and vice versa. Stating that as an explicit
        # lens mode keeps the two halves of the model consistent instead of
        # letting both try to explain the same pixels.
        lens_mode = "fixed" if solve_translation else "variable"
        lens_curve, lens_note = build_lens_curve(
            frames, intr, parallax_score=context.parallax_score, lens_mode=lens_mode
        )
        # Report focal as measured only when the footage measured it: the same
        # test build_lens_curve applies (a rotating camera below the parallax
        # gate). A prior or a user override may be the right value, but nothing
        # in this shot confirmed it (I7).
        measured_focal, measured_confidence = (None, 0.0)
        if context.parallax_score < 0.15 and intr.source != "user_override":
            measured_focal, measured_confidence = estimate_focal_from_homographies(frames, width, height)

        long_edge = max(intr.width, intr.height)
        aspect = intr.fy / intr.fx if intr.fx > 0 else 1.0
        focals = np.array(
            [lf.focal_normalized * long_edge for lf in lens_curve], dtype=np.float64
        )
        if len(focals) != len(frames) + 1:
            # build_lens_curve emits one entry per frame (transitions + 1). A
            # mismatch means the curve and the signature disagree about the
            # timeline, which would silently misalign every focal.
            return failed_result(
                self.source,
                f"lens curve has {len(focals)} entries for {len(frames)} transitions",
                total,
            )

        problem = _MatchProblem(
            measured=np.array([
                [mf.dx_pixels, mf.dy_pixels, mf.rotation_deg, mf.radial_flow]
                for mf in frames
            ], dtype=np.float64),
            frame_weights=np.sqrt(np.clip(
                np.array([mf.confidence for mf in frames], dtype=np.float64), 0.05, 1.0
            )),
            fx_old=focals[:-1],
            fy_old=focals[:-1] * aspect,
            fx_new=focals[1:],
            fy_new=focals[1:] * aspect,
            scene=scene,
            solve_translation=solve_translation,
        )

        initial, clamped = self._initial_parameters(problem, frames)
        context.progress(0.4, f"matching {problem.n} transitions in screen space")
        params, converged = _refine(problem, initial)

        # --- evidence, measured on the fit rather than assumed ---------------
        residual = problem.predict(params) - problem.measured
        pixel_rms = float(np.sqrt((residual[:, [0, 1, 3]] ** 2).mean()))
        rotation_rms = float(np.sqrt((residual[:, 2] ** 2).mean()))
        motion_scale = float(np.abs(problem.measured[:, [0, 1, 3]]).mean())

        poses = self._integrate(problem, params, frames, context)
        frame_indices, positions, quaternions = poses

        mean_conf = float(np.mean([mf.confidence for mf in frames]))
        mean_inlier = float(np.mean([mf.inlier_ratio for mf in frames]))
        confidence = self._score(
            mean_confidence=mean_conf,
            mean_inlier_ratio=mean_inlier,
            pixel_rms=pixel_rms,
            motion_scale=motion_scale,
            converged=converged,
        )

        notes = [
            f"screen-space match over {problem.n} transitions",
            f"signature residual {pixel_rms:.2f} px and {rotation_rms:.3f} deg "
            "(reported as the reprojection error; no 3D points are triangulated "
            "by this backend)",
            translation_note,
            lens_note,
        ]
        if clamped:
            notes.append(
                f"{clamped} transition(s) exceeded {MAX_INCREMENT_DEG:.0f} deg of "
                "inter-frame rotation and were clamped — likely a missed cut or a "
                "tracking failure rather than a camera move"
            )
        if not converged:
            notes.append(
                "the refinement did not converge; the closed-form estimate was kept"
            )
        notes.append(
            "this is a camera whose screen-space motion matches the reference, "
            "not a measurement of the physical camera path"
        )

        result = GeometryResult(
            source=self.source,
            frame_indices=frame_indices,
            positions=positions,
            quaternions=quaternions,
            per_pose_confidence=[confidence] * len(frame_indices),
            focal_pixels=float(np.median(focals)),
            focal_confidence=(
                float(measured_confidence) if measured_focal is not None
                else float(min(intr.confidence, 0.2))
            ),
            focal_observable=measured_focal is not None,
            # `focals` come from the lens curve at the analysis resolution.
            focal_image_width=int(width),
            registered_frames=len(frame_indices),
            total_frames=total,
            mean_reprojection_error=pixel_rms,
            median_reprojection_error=float(
                np.median(np.abs(residual[:, [0, 1, 3]]))
            ),
            track_count=0,
            mean_track_length=0.0,
            translation_observable=solve_translation,
            confidence=confidence,
            message="; ".join(n for n in notes if n),
            succeeded=True,
        )
        log.info("perceptual result: %s (confidence %.2f)", result.message, confidence)
        return result

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _translation_decision(
        context: SolveContext, frames: list[MotionFrame]
    ) -> tuple[bool, str]:
        """Decide whether to solve for a dolly at all, and say why.

        The refusal is the default and the important case: with no depth-dependent
        flow there is nothing in the data that fixes a baseline, so any position
        this backend emitted would be invented.
        """
        net_radial = float(np.mean([mf.radial_flow for mf in frames]))
        if context.parallax_score < PARALLAX_SUPPORTS_TRANSLATION:
            return False, (
                f"translation held at zero: parallax {context.parallax_score:.2f} is "
                f"below {PARALLAX_SUPPORTS_TRANSLATION:.2f}, so image motion alone "
                "cannot fix a baseline and any path would be invented"
            )
        if abs(net_radial) < MIN_RADIAL_FOR_DOLLY:
            return False, (
                f"translation held at zero: parallax {context.parallax_score:.2f} "
                f"supports translation but the mean radial flow is only "
                f"{net_radial:.2f} px, which is at the measurement floor"
            )
        return True, (
            f"a view-axis dolly was solved for: parallax {context.parallax_score:.2f} "
            f"attributes {net_radial:.2f} px of mean radial flow to forward motion "
            "rather than to the lens. Distances are fractions of the distance to the "
            "canonical scene — normalized, never metres — and lateral truck remains "
            "indistinguishable from pan in a screen-space match"
        )

    @staticmethod
    def _initial_parameters(
        problem: _MatchProblem, frames: list[MotionFrame]
    ) -> tuple[np.ndarray, int]:
        """Closed-form starting point, plus how many transitions were clamped."""
        limit = np.radians(MAX_INCREMENT_DEG)
        rows: list[np.ndarray] = []
        clamped = 0
        for i, mf in enumerate(frames):
            dyaw, dpitch, droll = signature_to_angles(
                mf.dx_pixels, mf.dy_pixels, mf.rotation_deg,
                float(problem.fx_old[i]), float(problem.fy_old[i]),
            )
            if max(abs(dyaw), abs(dpitch), abs(droll)) > limit:
                clamped += 1
            dyaw = float(np.clip(dyaw, -limit, limit))
            dpitch = float(np.clip(dpitch, -limit, limit))
            droll = float(np.clip(droll, -limit, limit))
            row = list(angles_to_rotation_vector(dyaw, dpitch, droll))
            if problem.solve_translation:
                # Content growing by factor s is what a step of (1 - 1/s) of the
                # way to the scene produces, at the reference depth.
                scale = max(float(mf.scale), 1e-3)
                row.append(float(np.clip(
                    1.0 - 1.0 / scale, -MAX_FORWARD_STEP, MAX_FORWARD_STEP
                )))
            rows.append(np.array(row, dtype=np.float64))
        return np.concatenate(rows) if rows else np.zeros(0), clamped

    @staticmethod
    def _integrate(
        problem: _MatchProblem,
        params: np.ndarray,
        frames: list[MotionFrame],
        context: SolveContext,
    ) -> tuple[list[int], np.ndarray, list[np.ndarray]]:
        """Compose per-transition increments into absolute poses.

        Quaternion composition throughout — the increments are never blended,
        summed or converted to Euler angles anywhere on this path (I3). The first
        pose is the identity at the origin, because each shot is an independent
        coordinate system (I4).
        """
        p = params.reshape(problem.n, problem.block)
        first_index = frames[0].frame_index - 1
        if first_index < context.shot.start_frame:
            first_index = context.shot.start_frame

        indices = [first_index] + [mf.frame_index for mf in frames]
        quats = [quat_identity()]
        centres = [np.zeros(3)]

        for i in range(problem.n):
            previous = quats[-1]
            increment = quat_exp(p[i, :3])
            quats.append(quat_normalize(quat_multiply(previous, increment)))
            if problem.solve_translation:
                # Step along the camera's own view axis, which is column 1 of the
                # camera-to-world matrix in the CameraPath convention.
                forward_world = quat_to_matrix(previous)[:, 1]
                centres.append(centres[-1] + forward_world * float(p[i, 3]) * REFERENCE_DEPTH)
            else:
                centres.append(np.zeros(3))

        return indices, np.array(centres), quats

    @staticmethod
    def _score(
        *,
        mean_confidence: float,
        mean_inlier_ratio: float,
        pixel_rms: float,
        motion_scale: float,
        converged: bool,
    ) -> float:
        """Confidence from evidence (I7), capped below a geometric solve.

        The fit term is scale-free on purpose: half a pixel of residual is
        excellent against 40 px of measured motion and hopeless against 1 px of
        it, and an absolute threshold would call a static shot a perfect match.
        """
        measurement = float(np.clip(mean_confidence, 0.0, 1.0))
        inliers = float(np.clip(mean_inlier_ratio, 0.0, 1.0))

        if motion_scale < PIXEL_NOISE_FLOOR:
            # Nothing measurable moved. The static answer is right, but it is
            # not evidence that the *method* worked, so the fit term stays
            # neutral rather than perfect.
            fit = 0.5
        else:
            relative = pixel_rms / motion_scale
            fit = float(np.clip(1.0 - (relative - 0.02) / 0.20, 0.0, 1.0))

        score = 0.32 * measurement + 0.23 * inliers + 0.45 * fit
        if not converged:
            score *= 0.8
        return float(np.clip(score, 0.0, 1.0) * PERCEPTUAL_MAX_CONFIDENCE)
