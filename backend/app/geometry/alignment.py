"""Sim(3) and SE(3) alignment via the Umeyama method.

Needed in two places:

  * merging overlapping temporal windows from a learned backend, where each
    window has its own arbitrary scale (spec §9)
  * comparing an estimate against Blender ground truth, where monocular scale
    is unrecoverable, so the comparison MUST allow a similarity transform or it
    measures nothing but the scale ambiguity (spec §24)

The scale term is what makes this Sim(3) rather than rigid alignment, and using
it correctly is the difference between an honest error metric and a meaningless
one. Equally, applying scale alignment and then reporting the result in metres
would launder an unknown scale into a false measurement — hence `allow_scale`
being explicit at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.geometry.rotations import (
    matrix_to_quat,
    quat_angular_distance,
    quat_to_matrix,
)


@dataclass
class Similarity:
    """y ~= scale * R @ x + t."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return (self.scale * (self.rotation @ p.T)).T + self.translation

    def apply_quaternion(self, quat_wxyz: np.ndarray) -> np.ndarray:
        """Orientations take the rotation only — scale and translation do not
        affect them."""
        return matrix_to_quat(self.rotation @ quat_to_matrix(quat_wxyz))

    def inverse(self) -> Similarity:
        inv_scale = 1.0 / self.scale if abs(self.scale) > 1e-12 else 1.0
        inv_rot = self.rotation.T
        return Similarity(
            scale=inv_scale,
            rotation=inv_rot,
            translation=-inv_scale * (inv_rot @ self.translation),
        )


def umeyama(
    source: np.ndarray, target: np.ndarray, *, allow_scale: bool = True
) -> Similarity:
    """Least-squares similarity taking `source` onto `target`.

    Umeyama (1991). Both inputs are (N, 3) and correspond row-by-row.

    The reflection guard matters: the naive SVD solution can return a
    determinant of -1, which is a reflection, not a rotation. Accepting it
    produces a mirrored trajectory that fits the points *better* than the
    correct answer — a silently wrong result with a flattering error score.
    """
    src = np.asarray(source, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError(f"shape mismatch: {src.shape} vs {dst.shape}")
    n = len(src)
    if n == 0:
        return Similarity(1.0, np.eye(3), np.zeros(3))
    if n == 1:
        return Similarity(1.0, np.eye(3), dst[0] - src[0])

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance, then its SVD.
    cov = (dst_c.T @ src_c) / n
    u, sigma, vt = np.linalg.svd(cov)

    # Guard against a reflection.
    s = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[-1] = -1.0
    rotation = u @ np.diag(s) @ vt

    if allow_scale:
        var_src = float((src_c ** 2).sum() / n)
        if var_src < 1e-12:
            # Degenerate: every source point is the same. No scale is
            # recoverable, so decline to invent one.
            scale = 1.0
        else:
            scale = float((sigma * s).sum() / var_src)
        if not np.isfinite(scale) or abs(scale) < 1e-12:
            scale = 1.0
    else:
        scale = 1.0

    translation = mu_dst - scale * (rotation @ mu_src)
    return Similarity(scale=scale, rotation=rotation, translation=translation)


#: Positions constrain the alignment rotation only when they spread in at least
#: two directions. Below this ratio of second to first singular value, the
#: perpendicular spread is under 5% of the path extent, so the roll about the
#: path axis is fixed by a few percent of the signal and is dominated by noise.
#: A straight COLMAP dolly measured 1.1e-4.
POSITION_RANK_RATIO = 0.05


def rotation_from_orientations(
    source_quats: list[np.ndarray], target_quats: list[np.ndarray]
) -> np.ndarray:
    """Global rotation R minimising sum ||R_target_i - R . R_source_i||_F.

    Chordal L2 mean of the per-frame rotation offsets, projected back onto SO(3)
    with the same reflection guard as Umeyama. Every orientation constrains all
    three axes, so unlike a position set this is never degenerate while at least
    one pose is available.
    """
    acc = np.zeros((3, 3))
    for sq, tq in zip(source_quats, target_quats):
        acc += quat_to_matrix(np.asarray(tq)) @ quat_to_matrix(np.asarray(sq)).T
    u, _, vt = np.linalg.svd(acc)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    return u @ np.diag(d) @ vt


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.clip(a @ b, -1.0, 1.0))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate 180 deg about any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _travel_axis(points: np.ndarray) -> np.ndarray:
    """Principal direction of a point set, signed along the direction of travel."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    if axis @ (points[-1] - points[0]) < 0:
        axis = -axis
    return axis


def align_poses(
    source_positions: np.ndarray,
    source_quats: list[np.ndarray],
    target_positions: np.ndarray,
    target_quats: list[np.ndarray],
    *,
    allow_scale: bool = True,
) -> tuple[Similarity, str]:
    """Similarity taking a trajectory onto another, using orientations for
    exactly the rotation the positions cannot determine — and no more.

    Umeyama on positions alone is the standard trajectory alignment, and it is
    silently wrong for the most common camera moves. A straight dolly, truck or
    pedestal has collinear camera centres, so the position cross-covariance has
    rank one and the TWIST about the path axis is arbitrary — the SVD returns
    whatever its null space happens to contain. Measured on a COLMAP
    reconstruction of a synthetic dolly whose relative rotations were accurate to
    0.011 deg, position-only alignment reported 179.7 deg of rotation error: a
    perfect solve scored as a total failure.

    The fix must be exactly as wide as the degeneracy. Collinear positions still
    fix the path DIRECTION (two of three rotational degrees of freedom), so an
    estimate whose cameras are yawed off the path must still show that error.
    An earlier version that took the whole rotation from orientations absorbed a
    genuine 3 deg yaw into the alignment and reported zero. So, by rank:

      * spread in >= 2 directions -> Umeyama on positions ("positions")
      * collinear                 -> path direction from positions, twist about
                                     it from orientations ("path_axis+orientations")
      * static (no spread)        -> whole rotation from orientations ("orientations")

    Returns the transform and the method, which callers should surface: anything
    other than "positions" means position data alone could not validate the
    rotation.
    """
    src = np.asarray(source_positions, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(target_positions, dtype=np.float64).reshape(-1, 3)
    n = min(len(src), len(dst))
    src, dst = src[:n], dst[:n]
    m = min(len(source_quats), len(target_quats), n)

    def spread_of(points: np.ndarray) -> np.ndarray:
        if len(points) < 2:
            return np.zeros(3)
        return np.linalg.svd(points - points.mean(axis=0), compute_uv=False)

    # Rank is judged on BOTH trajectories. The cross-covariance can be no better
    # conditioned than the worse of the two, and a noise cloud is well spread by
    # any ratio test while carrying no directional information at all: a pan
    # whose reference cameras sit at one point was aligned on the estimate's
    # position noise and scored 99.5 deg of rotation error.
    src_spread, dst_spread = spread_of(src), spread_of(dst)
    has_extent = n >= 2 and src_spread[0] > 1e-9 and dst_spread[0] > 1e-9
    well_conditioned = (
        has_extent and n >= 3
        and src_spread[1] / src_spread[0] >= POSITION_RANK_RATIO
        and dst_spread[1] / dst_spread[0] >= POSITION_RANK_RATIO
    )

    if well_conditioned:
        return umeyama(src, dst, allow_scale=allow_scale), "positions"
    if m == 0:
        return umeyama(src, dst, allow_scale=allow_scale), "positions_degenerate"

    src_q = [np.asarray(q) for q in source_quats[:m]]
    dst_q = [np.asarray(q) for q in target_quats[:m]]

    if has_extent and np.linalg.norm(dst[-1] - dst[0]) > 1e-9:
        # Two DoF from positions: bring the direction of travel onto the target's.
        axis_dst = _travel_axis(dst)
        base = _rotation_between(_travel_axis(src), axis_dst)

        # One DoF from orientations: the twist theta about axis_dst maximising
        # trace(R(theta)^T M), with M the summed residual rotation offsets. With
        # R(theta) = cos I + sin [a]x + (1 - cos) a a^T this is
        # A cos + B sin + const, maximised at theta = atan2(B, A).
        acc = np.zeros((3, 3))
        for sq, tq in zip(src_q, dst_q):
            acc += quat_to_matrix(tq) @ (base @ quat_to_matrix(sq)).T
        a = axis_dst
        ax = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        coeff_cos = float(np.trace(acc) - a @ acc @ a)
        coeff_sin = float(np.trace(ax.T @ acc))
        theta = float(np.arctan2(coeff_sin, coeff_cos))
        twist = (
            np.cos(theta) * np.eye(3) + np.sin(theta) * ax
            + (1.0 - np.cos(theta)) * np.outer(a, a)
        )
        rotation = twist @ base
        method = "path_axis+orientations"
    else:
        rotation = rotation_from_orientations(src_q, dst_q)
        method = "orientations"

    mu_src = src.mean(axis=0) if n else np.zeros(3)
    mu_dst = dst.mean(axis=0) if n else np.zeros(3)
    scale = 1.0
    if allow_scale and n >= 2:
        src_c = (src - mu_src) @ rotation.T
        dst_c = dst - mu_dst
        denom = float((src_c ** 2).sum())
        if denom > 1e-12:
            candidate = float((src_c * dst_c).sum() / denom)
            # A non-positive least-squares scale means the paths disagree in
            # direction under this rotation; keeping the sign would present that
            # as a good fit, so it is left to show up in the shape error.
            if np.isfinite(candidate) and candidate > 1e-12:
                scale = candidate
    translation = mu_dst - scale * (rotation @ mu_src)
    return Similarity(scale=scale, rotation=rotation, translation=translation), method

@dataclass
class TrajectoryError:
    """Alignment-invariant error metrics between an estimate and a reference."""

    absolute_translation_rmse: float
    """RMSE of positions after alignment, in reference units."""

    normalized_shape_error: float
    """Position RMSE as a fraction of the reference path length. This is the
    scale-free "is the shape right?" number the acceptance targets use."""

    rotation_mae_degrees: float
    rotation_max_degrees: float

    scale_factor: float
    """Similarity scale the alignment had to apply. Not an error — monocular
    scale is unrecoverable — but a large value flags a suspicious solve."""

    path_length_ratio: float
    reference_path_length: float
    estimated_path_length: float
    sample_count: int

    alignment_method: str = "positions"
    """What determined the alignment rotation: "positions" (standard),
    "path_axis+orientations" (collinear path), "orientations" (static camera
    centres), or "positions_degenerate" (degenerate AND no orientations — treat
    the rotation error as unreliable). See `align_poses`."""


def align_and_measure(
    estimated_positions: np.ndarray,
    estimated_quats: list[np.ndarray],
    reference_positions: np.ndarray,
    reference_quats: list[np.ndarray],
    *,
    allow_scale: bool = True,
) -> tuple[Similarity, TrajectoryError]:
    """Align an estimate to a reference, then measure what remains.

    `allow_scale` must be True for monocular estimates. With it False the
    reported error is dominated by the scale ambiguity and says nothing about
    whether the recovered motion is correct.
    """
    est_p = np.asarray(estimated_positions, dtype=np.float64).reshape(-1, 3)
    ref_p = np.asarray(reference_positions, dtype=np.float64).reshape(-1, 3)
    n = min(len(est_p), len(ref_p))
    est_p, ref_p = est_p[:n], ref_p[:n]

    transform, method = align_poses(
        est_p, list(estimated_quats), ref_p, list(reference_quats), allow_scale=allow_scale
    )
    aligned = transform.apply(est_p)

    residuals = np.linalg.norm(aligned - ref_p, axis=1)
    rmse = float(np.sqrt((residuals ** 2).mean())) if n else 0.0

    ref_path = float(np.linalg.norm(np.diff(ref_p, axis=0), axis=1).sum()) if n > 1 else 0.0
    est_path = float(np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum()) if n > 1 else 0.0

    # Normalise by path length, falling back to the reference's spatial extent
    # for a near-static reference where path length is ~0 and would explode the
    # ratio.
    denom = ref_path
    if denom < 1e-9 and n > 1:
        denom = float(np.linalg.norm(ref_p.max(axis=0) - ref_p.min(axis=0)))
    shape_error = float(rmse / denom) if denom > 1e-9 else 0.0

    rot_errors: list[float] = []
    m = min(len(estimated_quats), len(reference_quats), n)
    for i in range(m):
        aligned_q = transform.apply_quaternion(estimated_quats[i])
        rot_errors.append(np.degrees(quat_angular_distance(aligned_q, reference_quats[i])))

    error = TrajectoryError(
        absolute_translation_rmse=rmse,
        normalized_shape_error=shape_error,
        rotation_mae_degrees=float(np.mean(rot_errors)) if rot_errors else 0.0,
        rotation_max_degrees=float(np.max(rot_errors)) if rot_errors else 0.0,
        scale_factor=transform.scale,
        path_length_ratio=float(est_path / ref_path) if ref_path > 1e-9 else 1.0,
        reference_path_length=ref_path,
        estimated_path_length=est_path,
        sample_count=n,
        alignment_method=method,
    )
    return transform, error


def align_overlapping_windows(
    window_a_positions: np.ndarray,
    window_a_quats: list[np.ndarray],
    window_b_positions: np.ndarray,
    window_b_quats: list[np.ndarray],
    overlap_a: slice,
    overlap_b: slice,
) -> tuple[Similarity, np.ndarray, list[np.ndarray]]:
    """Bring window B into window A's frame using their shared frames.

    Used to stitch a long sequence out of bounded windows (spec §9). Each window
    carries its own arbitrary scale, so the overlap is solved as a full Sim(3).
    Returns the transform plus B's transformed poses.
    """
    a_overlap = np.asarray(window_a_positions)[overlap_a]
    b_overlap = np.asarray(window_b_positions)[overlap_b]
    if len(a_overlap) != len(b_overlap) or len(a_overlap) < 2:
        raise ValueError(
            f"overlap must match and contain at least 2 frames "
            f"(got {len(a_overlap)} and {len(b_overlap)})"
        )

    # A short overlap is nearly always close to collinear, so position-only
    # Umeyama would hand every stitched window an arbitrary roll. See align_poses.
    b_quats_overlap = list(window_b_quats)[overlap_b]
    a_quats_overlap = list(window_a_quats)[overlap_a]
    transform, _ = align_poses(
        b_overlap, b_quats_overlap, a_overlap, a_quats_overlap, allow_scale=True
    )
    positions = transform.apply(np.asarray(window_b_positions))
    quats = [transform.apply_quaternion(q) for q in window_b_quats]
    return transform, positions, quats
