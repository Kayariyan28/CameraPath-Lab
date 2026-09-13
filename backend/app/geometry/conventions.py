"""Coordinate-convention conversions — every one explicit, named and tested.

Invariant I8. This module exists because getting a camera convention backwards
produces a trajectory that looks entirely plausible and is completely wrong: a
mirrored, inverted path that still moves smoothly and still reprojects. There is
no runtime symptom. The only defence is to write each conversion once, name it
after both endpoints, and round-trip it in a test.

Three conventions are in play. Each is defined by where the camera's local axes
point, in terms of the semantic directions right / forward(view) / up:

| Convention          | +X    | +Y      | +Z       | view dir | world up |
|---------------------|-------|---------|----------|----------|----------|
| OpenCV / COLMAP cam | right | down    | forward  | +Z       | (none)   |
| CameraPath world    | right | forward | up       | +Y       | +Z       |
| Blender camera      | right | up      | backward | -Z       | +Z       |

A camera-to-world rotation matrix has the camera's local axes as its columns,
each expressed in world coordinates. So converting between conventions is a
fixed permutation/sign change applied on the *right* of that matrix — it
re-labels which local axis is which, and leaves the world frame untouched.

COLMAP additionally stores **world-to-camera**, not camera-to-world, and stores
translation in the camera frame. The camera centre in world space is
`C = -R_wc^T . t_wc`. Using `t_wc` directly as a position is the single most
common way to get a mirrored trajectory.
"""

from __future__ import annotations

import numpy as np

from app.geometry.rotations import matrix_to_quat, quat_normalize, quat_to_matrix

# ---------------------------------------------------------------------------
# Basis permutations. Each maps CameraPath-local axis indices onto the other
# convention's local axis indices, as a right-multiplied matrix.
# ---------------------------------------------------------------------------

# Naming is deliberate and worth reading once. Each matrix's COLUMNS are the
# other convention's local axes, written in CameraPath-local coordinates. That
# gives it two equivalent readings, and conflating them is the mistake this
# module exists to prevent:
#
#   * right-multiplied on a camera-to-world rotation, it relabels local axes:
#         R_cv_cam_to_world = R_cpl_cam_to_world @ CV_AXES_IN_CPL
#   * applied to a vector's components, it converts the OTHER convention's
#     local components into CameraPath-local components:
#         v_cpl_local = CV_AXES_IN_CPL @ v_cv_local
#
# The names say "<other> axes, expressed in CPL" so neither reading can be
# mistaken for its inverse. A name like `CV_AXES_IN_CPL` reads correctly for the
# first usage and exactly backwards for the second.

#: Columns: the OpenCV/COLMAP camera's local axes in CPL-local coordinates.
#: OpenCV is [right, down, forward] = [+x_cpl, -z_cpl, +y_cpl].
CV_AXES_IN_CPL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])

#: Columns: the Blender camera's local axes in CPL-local coordinates.
#: Blender is [right, up, backward] = [+x_cpl, +z_cpl, -y_cpl].
BLENDER_AXES_IN_CPL = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])

#: Inverses, for the reverse component transform.
CPL_AXES_IN_CV = CV_AXES_IN_CPL.T
CPL_AXES_IN_BLENDER = BLENDER_AXES_IN_CPL.T


def _assert_rotation(m: np.ndarray, name: str) -> None:
    det = float(np.linalg.det(m))
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"{name} is not a proper rotation (det={det:.6f})")


_assert_rotation(CV_AXES_IN_CPL, "CV_AXES_IN_CPL")
_assert_rotation(BLENDER_AXES_IN_CPL, "BLENDER_AXES_IN_CPL")


# ---------------------------------------------------------------------------
# COLMAP
# ---------------------------------------------------------------------------


def colmap_world_to_camera_to_pose(
    quat_wc_wxyz: np.ndarray, t_wc: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """COLMAP `(qvec, tvec)` -> (camera centre in world, camera-to-world rotation).

    COLMAP's stored pose maps world points into the camera frame:
        x_cam = R_wc . x_world + t_wc

    Therefore the camera centre, which is the world point mapping to the camera
    origin, is:
        C = -R_wc^T . t_wc

    Both returned values are still in OpenCV *camera* axis convention; call
    `cv_rotation_to_camerapath` to re-label the axes.
    """
    r_wc = quat_to_matrix(quat_wc_wxyz)
    t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)
    r_cw = r_wc.T
    centre = -r_cw @ t_wc
    return centre, r_cw


def pose_to_colmap_world_to_camera(
    centre: np.ndarray, r_cw: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `colmap_world_to_camera_to_pose`."""
    r_cw = np.asarray(r_cw, dtype=np.float64)
    centre = np.asarray(centre, dtype=np.float64).reshape(3)
    r_wc = r_cw.T
    t_wc = -r_wc @ centre
    return matrix_to_quat(r_wc), t_wc


# ---------------------------------------------------------------------------
# Axis relabelling
# ---------------------------------------------------------------------------


def cv_rotation_to_camerapath(r_cw_cv: np.ndarray) -> np.ndarray:
    """OpenCV camera-to-world rotation -> CameraPath camera-to-world rotation."""
    return np.asarray(r_cw_cv, dtype=np.float64) @ CPL_AXES_IN_CV


def camerapath_rotation_to_cv(r_cw_cpl: np.ndarray) -> np.ndarray:
    return np.asarray(r_cw_cpl, dtype=np.float64) @ CV_AXES_IN_CPL


def camerapath_rotation_to_blender(r_cw_cpl: np.ndarray) -> np.ndarray:
    """CameraPath camera-to-world rotation -> Blender camera-to-world rotation."""
    return np.asarray(r_cw_cpl, dtype=np.float64) @ BLENDER_AXES_IN_CPL


def blender_rotation_to_camerapath(r_cw_bl: np.ndarray) -> np.ndarray:
    return np.asarray(r_cw_bl, dtype=np.float64) @ CPL_AXES_IN_BLENDER


def camerapath_quat_to_blender(quat_cpl_wxyz: np.ndarray) -> np.ndarray:
    """Quaternion for Blender's `object.rotation_quaternion`, as [w,x,y,z].

    Positions need no conversion: both worlds are right-handed and Z-up, so a
    CameraPath position is already a Blender location. Only the camera's local
    axis labelling differs.
    """
    return matrix_to_quat(
        camerapath_rotation_to_blender(quat_to_matrix(quat_cpl_wxyz))
    )


def blender_quat_to_camerapath(quat_bl_wxyz: np.ndarray) -> np.ndarray:
    return matrix_to_quat(
        blender_rotation_to_camerapath(quat_to_matrix(quat_bl_wxyz))
    )


# ---------------------------------------------------------------------------
# Semantic accessors
# ---------------------------------------------------------------------------


def camera_axes(quat_cpl_wxyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, forward, up) unit vectors in world space, from a CPL orientation."""
    m = quat_to_matrix(quat_cpl_wxyz)
    return m[:, 0].copy(), m[:, 1].copy(), m[:, 2].copy()


def view_direction(quat_cpl_wxyz: np.ndarray) -> np.ndarray:
    """Where the camera is looking, in world space."""
    return quat_to_matrix(quat_cpl_wxyz)[:, 1].copy()


def camera_up(quat_cpl_wxyz: np.ndarray) -> np.ndarray:
    return quat_to_matrix(quat_cpl_wxyz)[:, 2].copy()


# ---------------------------------------------------------------------------
# World orientation
# ---------------------------------------------------------------------------


#: Right vectors must spread at least this much (second/first singular value)
#: before they define the up direction. Below it the shot has too little yaw
#: variety for the horizontal plane to be pinned down.
RIGHT_VECTOR_SPREAD_MIN = 0.1

#: The right-vector estimate is used only when it is self-consistent: the roll it
#: implies for each camera (angle of the right axis out of the estimated
#: horizontal plane), as an RMS, divided by the right-vector spread, must stay
#: under this many degrees. The ratio behaves like an error bound: little spread
#: amplifies any roll into a badly tilted plane. Measured on solved cameras: an
#: orbit 0.004 deg / 0.56 spread -> 0.007 (estimate exact); an FPV curve banking
#: +-13 deg 3.83 deg / 0.195 -> 19.6, where the right-vector plane was 51.5 deg off
#: gravity and the mean-up fallback was 3 deg off.
RIGHT_VECTOR_ERROR_BOUND_DEG = 2.0


def estimate_world_up(quats_cpl: list[np.ndarray]) -> np.ndarray:
    """Estimate which world direction is "up" from the cameras' orientations.

    A structure-from-motion world frame has no inherent orientation — COLMAP's
    axes are whatever the initial image pair implied (in practice the first
    camera's OpenCV frame, y down). A Blender scene with a ground plane needs an
    up direction, and without one the proxy films the cage with gravity sideways:
    exported COLMAP orbits had camera up (0, -1, 0) while claiming +Z up.

    Two cues, both heuristics and labelled as such:

      * Camera RIGHT vectors (preferred). Operators rarely roll, so a camera's
        right axis stays horizontal even when it pitches down — which is exactly
        when the up-vector cue fails (a drone looking 60 deg down has its up
        vector 60 deg off gravity). When the shot has yaw variety the right
        vectors span the horizontal plane, and up is its normal. Used only when
        that no-roll hypothesis checks out (see RIGHT_VECTOR_ERROR_BOUND_DEG):
        a banking FPV move otherwise tilts the estimate by tens of degrees.
      * Mean camera UP vector (fallback, when right vectors do not spread — no
        yaw variety). Assumes the camera was kept roughly level on average, so a
        steadily pitched-down shot with no yaw is re-levelled incorrectly.

    Never changes relative motion — only the frame it is expressed in.
    """
    if not quats_cpl:
        return np.array([0.0, 0.0, 1.0])
    ups = np.array([camera_up(q) for q in quats_cpl])
    mean_up = ups.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_up))

    rights = np.array([camera_axes(q)[0] for q in quats_cpl])
    if len(rights) >= 3:
        # Rows are unit vectors; their scatter matrix's weakest direction is the
        # one they are all perpendicular to.
        _, sv, vt = np.linalg.svd(rights, full_matrices=False)
        spread = float(sv[1] / sv[0]) if sv[0] > 1e-9 else 0.0
        implied_roll_rms = float(np.degrees(np.sqrt(np.mean(
            np.arcsin(np.clip(rights @ vt[-1], -1.0, 1.0)) ** 2
        ))))
        if spread >= RIGHT_VECTOR_SPREAD_MIN and implied_roll_rms / spread <= RIGHT_VECTOR_ERROR_BOUND_DEG:
            normal = vt[-1]
            if mean_norm > 1e-6 and float(normal @ mean_up) < 0:
                normal = -normal
            n = float(np.linalg.norm(normal))
            if n > 1e-9:
                return normal / n

    if mean_norm < 1e-6:
        # Up vectors cancelled out, e.g. a full roll. Keep the existing frame.
        return np.array([0.0, 0.0, 1.0])
    return mean_up / mean_norm


def rotation_aligning(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Shortest rotation matrix taking unit vector `source` onto `target`."""
    a = np.asarray(source, dtype=np.float64)
    b = np.asarray(target, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Anti-parallel: rotate 180 degrees about any axis perpendicular to a.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        axis /= float(np.linalg.norm(axis))
        k = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ])
        return np.eye(3) + 2.0 * (k @ k)
    k = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def apply_world_rotation(
    positions: np.ndarray, quats_cpl: list[np.ndarray], r_world: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rotate a whole trajectory into a new world frame.

    Positions rotate; orientations are pre-multiplied. Relative geometry, path
    length, timing and all angular rates are unchanged — this is a change of
    basis, not a modification of the motion.
    """
    r_world = np.asarray(r_world, dtype=np.float64)
    new_positions = (r_world @ np.asarray(positions, dtype=np.float64).T).T
    new_quats = [
        matrix_to_quat(r_world @ quat_to_matrix(q)) for q in quats_cpl
    ]
    return new_positions, new_quats


def orient_trajectory_z_up(
    positions: np.ndarray, quats_cpl: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Rotate the trajectory so the estimated up direction becomes +Z.

    Returns (positions, quats, applied_rotation). See `estimate_world_up` for
    the assumption this rests on.
    """
    up = estimate_world_up(quats_cpl)
    r = rotation_aligning(up, np.array([0.0, 0.0, 1.0]))
    p, q = apply_world_rotation(positions, quats_cpl, r)
    return p, q, r


def normalize_quaternion_list(quats: list[np.ndarray]) -> list[np.ndarray]:
    return [quat_normalize(q) for q in quats]
