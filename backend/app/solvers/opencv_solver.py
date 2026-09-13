"""Rotation from long-baseline keyframe homographies — the OpenCV rung.

For a camera that rotates (and zooms) without translating through depth — a pan,
tilt, roll, or a move over distant scenery — consecutive keyframes are related
exactly by H = K_b R K_a^-1, so the relative rotation is recoverable in closed
form: R = K_b^-1 H K_a, projected onto SO(3). Chaining ~one factor per second of
footage keeps bias from compounding, which integrating per-frame image motion
does not: on a real locked-off 59.94 fps shot that integration leaked 4 deg of
phantom rotation while the keyframe homographies measured an identity.

It sits between COLMAP and Perceptual Match on the ladder. COLMAP declines shots
with no parallax (translation is unobservable there); this backend accepts exactly
those, measures their rotation geometrically, and reports translation as
unobservable. Perceptual Match remains the fallback where keyframe matching fails
(low texture, extreme blur). Fusion then fills between the keyframe anchors, and
its held-out-anchor test decides how much per-frame detail to add back.

Convention: H maps analysis-resolution pixels of keyframe a to keyframe b, in
OpenCV camera axes (x right, y down, z forward), so X_b = R_ba X_a. The first
keyframe defines a level world (CameraPath identity). With A = CPL_AXES_IN_CV,
camera-to-world orientations chain as
    R_cw_cpl(b) = R_cw_cpl(a) . A^T R_ba^T A
which `tests/unit/test_opencv_solver.py` checks against known rotations.
"""

from __future__ import annotations

import numpy as np

from app.core.logging import get_logger
from app.geometry.conventions import CPL_AXES_IN_CV
from app.geometry.intrinsics import fov_to_focal_pixels
from app.geometry.rotations import matrix_to_quat, quat_identity, quat_multiply, quat_normalize
from app.models.schemas.trajectory import SolverSource
from app.solvers.base import GeometryResult, SolveContext, failed_result

log = get_logger("solvers.opencv")

#: Above this parallax score translation is observable and COLMAP is the right
#: backend; the homography model would absorb depth structure as fake rotation.
MAX_PARALLAX = 0.15

#: The keyframe chain must span at least this fraction of the shot's duration,
#: or too much of the rotation would rest on extrapolation.
MIN_CHAIN_COVERAGE = 0.6

#: Confidence ceiling: an exact model when its assumption holds, but the
#: assumption (no translation through depth) is inferred, not measured.
MAX_CONFIDENCE = 0.85


def rotation_from_homography(h: np.ndarray, focal_a: float, focal_b: float,
                             width: int, height: int) -> np.ndarray:
    """R_ba (OpenCV camera axes) from a rotation/zoom homography a -> b."""
    cx, cy = width / 2.0, height / 2.0
    k_a = np.array([[focal_a, 0.0, cx], [0.0, focal_a, cy], [0.0, 0.0, 1.0]])
    k_b = np.array([[focal_b, 0.0, cx], [0.0, focal_b, cy], [0.0, 0.0, 1.0]])
    m = np.linalg.inv(k_b) @ np.asarray(h, dtype=np.float64) @ k_a
    # Homographies are defined up to scale; the nearest rotation absorbs it.
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return r


def relative_cpl_rotation(r_ba_cv: np.ndarray) -> np.ndarray:
    """Body-frame increment in CameraPath axes: R_cw(b) = R_cw(a) . this."""
    a = CPL_AXES_IN_CV
    return a.T @ np.asarray(r_ba_cv).T @ a


class OpenCVPoseBackend:
    @property
    def source(self) -> SolverSource:
        return SolverSource.OPENCV

    def available(self) -> tuple[bool, str]:
        return True, ""

    def suitable_for(self, context: SolveContext) -> tuple[bool, str]:
        if context.parallax_score >= MAX_PARALLAX:
            return False, (
                f"parallax {context.parallax_score:.2f} — the camera translated through "
                "depth, which a homography cannot represent; use the SfM solve"
            )
        if not context.keyframe_homographies or len(context.keyframe_homographies) < 2:
            return False, "fewer than two keyframe homographies were measured (low texture or blur)"
        return True, ""

    def estimate(self, context: SolveContext) -> GeometryResult:
        total = context.shot.frame_count
        pairs = sorted(context.keyframe_homographies or [], key=lambda p: p.frame_a)
        if len(pairs) < 2:
            return failed_result(self.source, "no keyframe homographies", total)

        # Longest contiguous chain.
        chains: list[list] = [[pairs[0]]]
        for pair in pairs[1:]:
            if pair.frame_a == chains[-1][-1].frame_b:
                chains[-1].append(pair)
            else:
                chains.append([pair])
        chain = max(chains, key=lambda c: c[-1].time_b - c[0].time_a)
        span = chain[-1].time_b - chain[0].time_a
        coverage = span / max(context.shot.duration, 1e-6)
        if coverage < MIN_CHAIN_COVERAGE:
            return failed_result(
                self.source,
                f"keyframe chain covers only {coverage:.0%} of the shot; matching broke down",
                total,
            )

        width, height = context.analysis_size
        prior_focal = float(context.intrinsics.scaled_to(width, height).fx)
        lens_by_frame = {lf.frame_index: lf for lf in (context.lens or [])}

        def focal_at(frame: int) -> float:
            lf = lens_by_frame.get(frame)
            if lf is None or not np.isfinite(lf.fov_horizontal):
                return prior_focal
            return float(fov_to_focal_pixels(lf.fov_horizontal, width))

        frames = [chain[0].frame_a]
        quats = [quat_identity()]
        inlier_ratios = [1.0]
        for pair in chain:
            r_ba = rotation_from_homography(pair.matrix(), focal_at(pair.frame_a),
                                            focal_at(pair.frame_b), width, height)
            increment = matrix_to_quat(relative_cpl_rotation(r_ba))
            quats.append(quat_normalize(quat_multiply(quats[-1], increment)))
            frames.append(pair.frame_b)
            inlier_ratios.append(pair.inliers / max(pair.matches, 1))

        support = float(np.clip(np.mean(inlier_ratios[1:]), 0.0, 1.0))
        confidence = float(min(MAX_CONFIDENCE, MAX_CONFIDENCE * support * min(1.0, coverage / 0.9)))
        lens_measured = any(lf.confidence >= 0.5 for lf in (context.lens or []))
        return GeometryResult(
            source=self.source,
            frame_indices=frames,
            positions=np.zeros((len(frames), 3)),
            quaternions=quats,
            per_pose_confidence=[confidence] * len(frames),
            focal_pixels=float(np.median([focal_at(f) for f in frames])),
            focal_confidence=0.7 if lens_measured else 0.1,
            focal_observable=lens_measured,
            focal_image_width=int(width),
            registered_frames=len(frames),
            total_frames=len(frames),
            track_count=int(sum(p.inliers for p in chain)),
            mean_track_length=0.0,
            translation_observable=False,
            confidence=confidence,
            message=(
                f"rotation from {len(chain)} long-baseline keyframe homographies covering "
                f"{coverage:.0%} of the shot ({support:.0%} mean inlier ratio); translation "
                "is not observable without parallax and is held fixed"
            ),
            succeeded=True,
        )
