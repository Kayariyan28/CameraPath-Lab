"""Dominant image-space motion per frame transition.

This is the dense temporal backbone of the whole system. It runs on EVERY frame
transition, unlike the geometric solve which runs only on keyframes, and it is
what preserves acceleration, jitter, micro-movement and direction changes
(spec §5).

Read the module docstring of `models/schemas/motion.py` before using the output:
every quantity here is image-space. `dx_pixels` is how far the picture moved, and
turning that into a camera pose requires a camera model and a geometric solve.
Conflating the two is invariant I6, and it is the single easiest way to produce
confident nonsense.

Parallax is estimated here too, because it is the deciding evidence for whether
translation is observable at all. If one homography explains every
correspondence, the shot is either planar, distant, or pure rotation — and in all
three cases a physical translation estimate would be invented rather than
measured (spec §12, §25).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.logging import get_logger
from app.models.schemas.motion import MotionFrame, MotionModel, MotionSignature
from app.tracking.flow import FlowResult
from app.tracking.robust import robust

log = get_logger("tracking.global_motion")

#: RANSAC reprojection threshold in pixels, at analysis resolution.
RANSAC_THRESHOLD = 2.2

#: Below this median flow, motion is at or under the noise floor of sub-pixel
#: corner localisation and nothing about it should be trusted.
STATIC_FLOW_FLOOR = 0.35

#: A homography within this inlier ratio of the selected simpler model is used to
#: read dx/dy/rotation/scale (see estimate_transition).
HOMOGRAPHY_DECOMPOSITION_TOLERANCE = 0.05

#: A homography this good means no measurable parallax.
HOMOGRAPHY_EXPLAINS_ALL_INLIERS = 0.93
HOMOGRAPHY_EXPLAINS_ALL_RESIDUAL = 1.5


def _decompose_similarity(matrix: np.ndarray, centre: tuple[float, float]
                          ) -> tuple[float, float, float, float]:
    """(dx, dy, rotation_deg, scale) from a 2x3 similarity/affine matrix.

    Translation is reported as the displacement of the *image centre* under the
    model, not the raw `tx`/`ty`. Raw translation terms are measured from the
    origin at the top-left corner, so any rotation or scale contaminates them
    wildly — a pure 5° roll about the centre has a large raw `tx` and zero actual
    centre displacement.
    """
    a = matrix[:2, :2]
    cx, cy = centre
    moved = a @ np.array([cx, cy]) + matrix[:2, 2]
    dx, dy = float(moved[0] - cx), float(moved[1] - cy)

    # Closest similarity to the linear part: average the two rotation estimates
    # the matrix implies, which is stable for a near-similarity affine.
    rotation = float(np.degrees(np.arctan2(a[1, 0] - a[0, 1], a[0, 0] + a[1, 1])))
    scale = float(np.sqrt(max(abs(np.linalg.det(a)), 1e-12)))
    return dx, dy, rotation, scale


def _local_affine_of_homography(h: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """2x3 affine that agrees with homography `h` at the image centre: same
    image of the centre, and the same Jacobian there.

    Taking hom[:2, :3] instead — dropping the perspective row — is only valid for
    a homography that is already nearly affine, which is exactly what a panning
    or tilting wide lens is not. On a synthetic 1.15 deg/frame pan with a 65 deg
    lens it reported dx -10.0 px, dy +4.0 px and 0.21 deg of roll, where the true
    image motion is about -17 px of dx and neither of the others; the same error
    produced a steady, spurious 2% per-frame "scale" that read as a zoom. The
    synthetic 2D test clips never exposed it because their homographies are
    affine to begin with.

    For x' = (a.p)/(c.p), y' = (b.p)/(c.p), the Jacobian at p is
    d(x',y')/d(x,y) = ([a0 a1; b0 b1] - [x'; y'] [c0 c1]) / (c.p).
    """
    cx, cy = centre
    p = np.array([cx, cy, 1.0])
    w = float(h[2] @ p)
    if abs(w) < 1e-12 or not np.isfinite(w):
        return np.asarray(h[:2, :3], dtype=np.float64)
    mapped = (h[:2] @ p) / w
    jacobian = (h[:2, :2] - np.outer(mapped, h[2, :2])) / w
    translation = mapped - jacobian @ np.array([cx, cy])
    return np.hstack([jacobian, translation.reshape(2, 1)])


# Public names: the Perceptual Match predictor must compute the signature with
# exactly the same definitions as the measurement, or it fits a bias.
local_affine_of_homography = _local_affine_of_homography


def _first_order_flow_field(points: np.ndarray, flow: np.ndarray,
                            centre: tuple[float, float]
                            ) -> tuple[float, float, np.ndarray]:
    """Least-squares first-order fit of the flow field: u(p) = J·(p-c) + t.

    Returns (divergence, curl, J). Both are spatial derivatives of a displacement
    field, so they are dimensionless (pixels of flow per pixel of position).

    Fitting the actual flow rather than reading the derivatives off the fitted
    homography matters: where parallax exists the true field is *not* a
    homography, and the least-squares fit reports the average expansion the
    viewer actually perceives.
    """
    if len(points) < 6:
        return 0.0, 0.0, np.zeros((2, 2), np.float64)
    cx, cy = centre
    rel = points.astype(np.float64) - np.array([cx, cy])
    design = np.hstack([rel, np.ones((len(rel), 1))])  # (N,3)
    try:
        sol, *_ = np.linalg.lstsq(design, flow.astype(np.float64), rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, np.zeros((2, 2), np.float64)
    # sol is (3,2): rows = [d/dx, d/dy, const], cols = [u, v]
    jac = sol[:2, :].T  # J[i,j] = d(flow_i)/d(p_j)
    divergence = float(jac[0, 0] + jac[1, 1])
    curl = float(jac[1, 0] - jac[0, 1])
    return divergence, curl, jac


def _radial_flow(points: np.ndarray, flow: np.ndarray,
                 centre: tuple[float, float]) -> float:
    """Mean outward flow component about the image centre, in pixels.

    Positive means content is expanding. Crucially this is caused by dolly-in
    *or* zoom-in, and those are indistinguishable from this signal alone — the
    distinction needs parallax (a dolly changes relative depth spacing, a zoom
    does not). Reported here, disambiguated in `geometry/intrinsics.py`.
    """
    if len(points) == 0:
        return 0.0
    cx, cy = centre
    rel = points.astype(np.float64) - np.array([cx, cy])
    norms = np.linalg.norm(rel, axis=1)
    valid = norms > 1e-3
    if not valid.any():
        return 0.0
    unit = rel[valid] / norms[valid, None]
    return float(np.sum(unit * flow[valid], axis=1).mean())


def _symmetric_transfer_error(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point symmetric transfer error for a homography, in pixels."""
    def apply(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        out = homo @ mat.T
        w = out[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return out[:, :2] / w

    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.full(len(src), 1e6)
    fwd = np.linalg.norm(apply(h, src) - dst, axis=1)
    bwd = np.linalg.norm(apply(h_inv, dst) - src, axis=1)
    return (fwd + bwd) * 0.5


def estimate_transition(
    flow_result: FlowResult,
    timestamp: float,
    dt: float,
    image_size: tuple[int, int],
    *,
    weights: np.ndarray | None = None,
) -> MotionFrame:
    """Build one MotionFrame from one transition's correspondences.

    `weights` is the optional per-track background confidence from
    `dynamic_rejection`; when supplied, low-confidence (likely moving-object)
    tracks are excluded from the model fit entirely rather than merely
    down-weighted, since RANSAC already tolerates outliers but not a *majority*
    of coherent wrong ones.
    """
    w, h = image_size
    centre = (w / 2.0, h / 2.0)
    mf = MotionFrame(
        frame_index=flow_result.frame_index,
        timestamp=timestamp,
        dt=dt,
        tracks_in=flow_result.tracks_in,
        tracks_survived=flow_result.tracks_survived,
    )

    src = flow_result.prev_points
    dst = flow_result.curr_points
    if weights is not None and len(weights) == len(src):
        trusted = weights >= 0.35
        mf.rejected_track_count = int((~trusted).sum())
        if trusted.sum() >= 12:
            src, dst = src[trusted], dst[trusted]
    mf.background_track_count = len(src)

    if len(src) < 6:
        mf.confidence = 0.0
        mf.model_used = MotionModel.NONE
        return mf

    flow = dst - src
    magnitudes = np.linalg.norm(flow, axis=1)
    mf.flow_magnitude = float(np.median(magnitudes))
    mf.flow_magnitude_p90 = float(np.percentile(magnitudes, 90))
    mf.median_flow = [float(np.median(flow[:, 0])), float(np.median(flow[:, 1]))]

    # ---- dominant similarity model (4 DoF) — the most stable choice ----
    sim, sim_inliers = robust(cv2.estimateAffinePartial2D, 
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=3000, confidence=0.995, refineIters=24,
    )
    sim_ratio = float(sim_inliers.ravel().mean()) if sim_inliers is not None else 0.0

    # ---- full affine (6 DoF) ----
    aff, aff_inliers = robust(cv2.estimateAffine2D, 
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=3000, confidence=0.995, refineIters=24,
    )
    aff_ratio = float(aff_inliers.ravel().mean()) if aff_inliers is not None else 0.0

    # ---- homography (8 DoF) — needs more points to be meaningful ----
    hom, hom_inliers, hom_ratio, hom_residual = None, None, 0.0, float("inf")
    if len(src) >= 14:
        hom, hom_inliers = robust(cv2.findHomography, 
            src, dst, method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=4000, confidence=0.995,
        )
        if hom is not None and hom_inliers is not None:
            hom_ratio = float(hom_inliers.ravel().mean())
            errs = _symmetric_transfer_error(hom, src, dst)
            hom_residual = float(np.median(errs))

    # Model selection: prefer the simplest model that is not clearly beaten.
    # Each added degree of freedom must buy a real improvement in inliers, or it
    # is just fitting noise — a homography fitted to a pure translation will
    # happily invent perspective that is not there.
    chosen_matrix, chosen_inliers, chosen_ratio, model = None, None, 0.0, MotionModel.NONE
    if sim is not None and sim_ratio > 0:
        chosen_matrix, chosen_inliers, chosen_ratio, model = sim, sim_inliers, sim_ratio, MotionModel.EUCLIDEAN
    if aff is not None and aff_ratio > chosen_ratio + 0.04:
        chosen_matrix, chosen_inliers, chosen_ratio, model = aff, aff_inliers, aff_ratio, MotionModel.AFFINE
    if hom is not None and hom_ratio > chosen_ratio + 0.06:
        chosen_inliers, chosen_ratio, model = hom_inliers, hom_ratio, MotionModel.HOMOGRAPHY
        chosen_matrix = _local_affine_of_homography(np.asarray(hom, dtype=np.float64), centre)
    elif hom is not None and hom_ratio >= chosen_ratio - HOMOGRAPHY_DECOMPOSITION_TOLERANCE:
        # The simpler model won on parsimony, but its centre displacement,
        # rotation and scale are still wrong under perspective: a least-squares
        # similarity over a panning wide lens over-reads the centre shift and
        # invents a scale. When the homography explains the data about as well,
        # the signature is read from its local affine instead, so every frame's
        # dx/dy/rotation/scale has ONE definition — the same one the Perceptual
        # Match predictor uses. `model_used` keeps the parsimonious label.
        chosen_matrix = _local_affine_of_homography(np.asarray(hom, dtype=np.float64), centre)

    if chosen_matrix is None:
        mf.confidence = 0.0
        return mf

    mf.model_used = model
    mf.inlier_ratio = chosen_ratio
    if aff is not None:
        mf.affine = [float(v) for v in np.asarray(aff).ravel()]
    if hom is not None:
        mf.homography = [float(v) for v in np.asarray(hom).ravel()]

    dx, dy, rotation, scale = _decompose_similarity(np.asarray(chosen_matrix), centre)
    mf.dx_pixels, mf.dy_pixels = dx, dy
    mf.rotation_deg, mf.scale = rotation, scale

    inlier_mask = (chosen_inliers.ravel() == 1) if chosen_inliers is not None else np.ones(len(src), bool)
    if inlier_mask.sum() >= 6:
        in_pts, in_flow = src[inlier_mask], flow[inlier_mask]
    else:
        in_pts, in_flow = src, flow

    mf.radial_flow = _radial_flow(in_pts, in_flow, centre)
    divergence, curl, _ = _first_order_flow_field(in_pts, in_flow, centre)
    mf.flow_divergence, mf.flow_curl = divergence, curl

    # ---- dynamic content: inliers that the dominant model failed to explain ----
    if chosen_inliers is not None and len(src) > 0:
        outliers = ~inlier_mask
        if outliers.any():
            # Area fraction, approximated by the convex-hull-free bounding spread
            # of disagreeing points — cheap and good enough to warn the user that
            # a large moving subject is present.
            out_pts = src[outliers]
            spread_w = float(np.ptp(out_pts[:, 0])) / max(w, 1)
            spread_h = float(np.ptp(out_pts[:, 1])) / max(h, 1)
            mf.dynamic_area_fraction = float(
                np.clip(spread_w * spread_h * (outliers.sum() / len(src)), 0.0, 1.0)
            )

    # ---- confidence ----
    survival = (
        flow_result.tracks_survived / flow_result.tracks_in
        if flow_result.tracks_in else 0.0
    )
    track_support = min(1.0, len(src) / 180.0)
    confidence = float(
        np.clip(0.5 * chosen_ratio + 0.3 * track_support + 0.2 * survival, 0.0, 1.0)
    )
    if mf.flow_magnitude < STATIC_FLOW_FLOOR:
        # Genuinely static content. The *model* is trustworthy (it is ~identity)
        # but any direction we report is noise, so say the direction is unreliable
        # while keeping the "nothing moved" conclusion itself confident.
        confidence *= 0.85
    mf.confidence = confidence

    # Stash parallax evidence for the shot-level aggregate.
    mf_extra_h_ratio = hom_ratio
    mf_extra_h_resid = hom_residual
    setattr(mf, "_h_inlier_ratio", mf_extra_h_ratio)
    setattr(mf, "_h_residual", mf_extra_h_resid)
    return mf


def parallax_from_transition(mf: MotionFrame) -> float:
    """0-1 evidence that this transition contains depth-dependent flow.

    Logic: if a single homography explains the correspondences to sub-pixel
    accuracy, the scene is effectively planar or the motion is pure rotation, and
    translation is NOT observable. Structured residual beyond a homography is the
    signature of parallax.

    Gated on there being enough motion to measure at all — a static frame has no
    parallax evidence either way, and must not be scored as "no parallax", which
    would read as positive evidence against translation.
    """
    h_ratio = getattr(mf, "_h_inlier_ratio", 0.0)
    h_resid = getattr(mf, "_h_residual", float("inf"))

    if mf.flow_magnitude < STATIC_FLOW_FLOOR:
        return 0.0
    if not np.isfinite(h_resid):
        # No homography was fitted (too few points) — no evidence.
        return 0.0
    if h_ratio >= HOMOGRAPHY_EXPLAINS_ALL_INLIERS and h_resid <= HOMOGRAPHY_EXPLAINS_ALL_RESIDUAL:
        return 0.0
    # Residual grows with parallax; 3 px of unexplained median error at analysis
    # resolution is already strong depth structure.
    residual_term = float(np.clip((h_resid - HOMOGRAPHY_EXPLAINS_ALL_RESIDUAL) / 3.0, 0.0, 1.0))
    outlier_term = float(np.clip((HOMOGRAPHY_EXPLAINS_ALL_INLIERS - h_ratio) / 0.35, 0.0, 1.0))
    return float(np.clip(0.65 * residual_term + 0.35 * outlier_term, 0.0, 1.0))


decompose_similarity = _decompose_similarity


def summarise(
    motion_frames: list[MotionFrame],
    *,
    duration: float,
    texture: float = 0.0,
    blur: float = 1.0,
) -> MotionSignature:
    """Aggregate a shot's transitions into a routing-relevant signature."""
    sig = MotionSignature(frame_count=len(motion_frames) + 1, duration=duration)
    if not motion_frames:
        return sig

    mags = np.array([m.flow_magnitude for m in motion_frames])
    inliers = np.array([m.inlier_ratio for m in motion_frames])
    rotations = np.array([m.rotation_deg for m in motion_frames])
    radials = np.array([m.radial_flow for m in motion_frames])
    scales = np.array([m.scale for m in motion_frames])

    sig.mean_flow_magnitude = float(mags.mean())
    sig.peak_flow_magnitude = float(mags.max())
    sig.mean_inlier_ratio = float(inliers.mean())
    sig.total_image_rotation_deg = float(rotations.sum())
    sig.net_dx_pixels = float(sum(m.dx_pixels for m in motion_frames))
    sig.net_dy_pixels = float(sum(m.dy_pixels for m in motion_frames))
    sig.cumulative_path_pixels = float(
        sum(float(np.hypot(m.dx_pixels, m.dy_pixels)) for m in motion_frames)
    )
    sig.mean_radial_flow = float(radials.mean())
    sig.net_scale_change = float(np.prod(np.clip(scales, 1e-3, 1e3)))

    parallaxes = np.array([parallax_from_transition(m) for m in motion_frames])
    moving = mags >= STATIC_FLOW_FLOOR
    # Average parallax over transitions where motion was actually measurable.
    sig.parallax_score = float(parallaxes[moving].mean()) if moving.any() else 0.0

    h_ratios = np.array([getattr(m, "_h_inlier_ratio", 0.0) for m in motion_frames])
    sig.homography_dominance = float((h_ratios >= HOMOGRAPHY_EXPLAINS_ALL_INLIERS).mean())

    # Rotational vs translational share of the perceived motion. Image rotation
    # of 1 deg is comparable to roughly long_edge/57 px of translation at the
    # frame edge, so compare in those terms.
    rot_energy = float(np.abs(rotations).sum())
    trans_energy = float(np.abs(mags).sum())
    total = rot_energy * 8.0 + trans_energy
    # Gate on there being real motion at all. Without the floor, a locked-off
    # shot divides sub-pixel noise by sub-pixel noise and reports a confident
    # "mostly rotational", which is meaningless.
    if trans_energy + rot_energy * 8.0 < STATIC_FLOW_FLOOR * len(motion_frames):
        sig.rotation_dominance = 0.0
    else:
        sig.rotation_dominance = float(rot_energy * 8.0 / total) if total > 1e-9 else 0.0

    sig.texture_score = texture
    sig.blur_score = blur

    # Jitter: high-frequency energy in the per-frame translation, as a share of
    # total. A gimbal produces a smooth ramp (low); handheld produces a ramp plus
    # broadband content (high). This is what EXACT fidelity must preserve (I10).
    dxs = np.array([m.dx_pixels for m in motion_frames])
    dys = np.array([m.dy_pixels for m in motion_frames])
    sig.jitter_score = _jitter_score(dxs, dys)
    return sig


#: Noise floor for the second difference of per-frame translation, in pixels at
#: analysis resolution. Justification: a single LK correspondence localises to
#: ~0.3 px; a model fitted over hundreds of them is better, but the second
#: difference combines three independent measurements and so carries roughly
#: 2.4x the per-measurement noise. Measured on synthetic constant-velocity and
#: pure-zoom clips (which contain no shake by construction) the observed value is
#: 0.8 px, so 1.2 px sits above the noise without swallowing real shake — the
#: handheld reference clip measures 4.3 px.
JITTER_NOISE_FLOOR = 1.2


def _high_freq_energy(series: np.ndarray) -> float:
    """Mean absolute second difference — energy above the smooth trend."""
    if len(series) < 3:
        return 0.0
    return float(np.abs(np.diff(series, n=2)).mean())


def _jitter_score(dxs: np.ndarray, dys: np.ndarray) -> float:
    """0-1 share of translation energy that is shake rather than intent.

    `excess / (excess + motion)`, where `excess` is high-frequency energy above
    the measurement noise floor. Bounded by construction, with no tuned gain.

    Two traps this avoids, both found by testing against synthetic clips:

      * Dividing by *mean motion* alone makes a locked-off tripod shot report
        maximum jitter — numerator and denominator are both sub-pixel noise.
        Hence the absolute floor, applied before any ratio.
      * Dividing by the *first* difference makes a constant-velocity pan report
        maximum jitter, because a constant velocity has zero first difference
        too. Hence motion amplitude, not motion variation, in the denominator.

    Reference values at 1080 px analysis resolution: static 0.00, pure roll 0.00,
    zoom-only 0.00, constant pan 0.00, accelerating pan 0.00, fast sweep 0.16,
    handheld 0.37.
    """
    if len(dxs) < 5:
        return 0.0
    hf = (_high_freq_energy(dxs) + _high_freq_energy(dys)) / 2.0
    excess = hf - JITTER_NOISE_FLOOR
    if excess <= 0.0:
        return 0.0
    motion = (float(np.abs(dxs).mean()) + float(np.abs(dys).mean())) / 2.0
    return float(np.clip(excess / (excess + motion + 1e-9), 0.0, 1.0))
