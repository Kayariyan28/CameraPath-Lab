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

    transform = umeyama(est_p, ref_p, allow_scale=allow_scale)
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

    transform = umeyama(b_overlap, a_overlap, allow_scale=True)
    positions = transform.apply(np.asarray(window_b_positions))
    quats = [transform.apply_quaternion(q) for q in window_b_quats]
    return transform, positions, quats
