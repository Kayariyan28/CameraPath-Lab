"""Coordinate-convention tests (invariant I8, spec §8/§34).

A convention error produces a mirrored or inverted trajectory that still looks
smooth, still reprojects, and has no runtime symptom. These tests are the only
thing standing between that bug and a confidently wrong product, so they check
semantics (where is the camera actually looking?) rather than only round-tripping
matrices — a self-consistent pair of wrong conversions would round-trip fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.conventions import (
    BLENDER_AXES_IN_CPL,
    CV_AXES_IN_CPL,
    apply_world_rotation,
    blender_quat_to_camerapath,
    blender_rotation_to_camerapath,
    camera_axes,
    camera_up,
    camerapath_quat_to_blender,
    camerapath_rotation_to_blender,
    camerapath_rotation_to_cv,
    colmap_world_to_camera_to_pose,
    cv_rotation_to_camerapath,
    estimate_world_up,
    orient_trajectory_z_up,
    pose_to_colmap_world_to_camera,
    rotation_aligning,
    view_direction,
)
from app.geometry.rotations import (
    look_at_quaternion,
    matrix_to_quat,
    quat_from_axis_angle,
    quat_identity,
    quat_to_matrix,
)

X, Y, Z = np.eye(3)


def random_rotations(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        axis = rng.normal(size=3)
        angle = rng.uniform(-np.pi, np.pi)
        out.append(quat_to_matrix(quat_from_axis_angle(axis, angle)))
    return out


class TestBasisMatrices:
    def test_are_proper_rotations(self):
        for name, m in [("CV_AXES_IN_CPL", CV_AXES_IN_CPL),
                        ("BLENDER_AXES_IN_CPL", BLENDER_AXES_IN_CPL)]:
            assert np.isclose(np.linalg.det(m), 1.0), f"{name} det != 1"
            assert np.allclose(m @ m.T, np.eye(3)), f"{name} not orthonormal"

    def test_cv_axis_semantics(self):
        """Columns are OpenCV's local axes written in CPL-local coordinates, so
        column j answers "what is OpenCV's axis j, in CameraPath terms?"."""
        assert np.allclose(CV_AXES_IN_CPL @ X, X)     # cv x (right)   = cpl +x
        assert np.allclose(CV_AXES_IN_CPL @ Y, -Z)    # cv y (down)    = cpl -z
        assert np.allclose(CV_AXES_IN_CPL @ Z, Y)     # cv z (forward) = cpl +y

    def test_blender_axis_semantics(self):
        """Same reading for Blender: x=right, y=up, z=backward."""
        assert np.allclose(BLENDER_AXES_IN_CPL @ X, X)   # bl x (right)    = cpl +x
        assert np.allclose(BLENDER_AXES_IN_CPL @ Y, Z)   # bl y (up)       = cpl +z
        assert np.allclose(BLENDER_AXES_IN_CPL @ Z, -Y)  # bl z (backward) = cpl -y


class TestSemantics:
    """Where the camera actually points. These catch a self-consistent pair of
    wrong conversions, which round-trip tests cannot."""

    def test_identity_camera_looks_along_plus_y(self):
        assert np.allclose(view_direction(quat_identity()), Y)
        assert np.allclose(camera_up(quat_identity()), Z)

    def test_blender_identity_camera_looks_down_minus_z(self):
        """Blender's convention: an unrotated camera looks along -Z."""
        r_bl = camerapath_rotation_to_blender(np.eye(3))
        # The camera's local -Z axis, in world space, must be the view direction.
        assert np.allclose(r_bl @ (-Z), Y), "Blender view axis mismatch"
        assert np.allclose(r_bl @ Y, Z), "Blender up axis mismatch"

    def test_cv_identity_camera_looks_along_plus_z(self):
        r_cv = camerapath_rotation_to_cv(np.eye(3))
        assert np.allclose(r_cv @ Z, Y), "OpenCV view axis mismatch"
        assert np.allclose(r_cv @ Y, -Z), "OpenCV y should be down"

    def test_look_at_points_where_asked(self):
        for target in [X, -X, Y, -Y, np.array([1.0, 2.0, -0.5])]:
            q = look_at_quaternion(target, Z)
            expected = target / np.linalg.norm(target)
            assert np.allclose(view_direction(q), expected, atol=1e-9), target

    def test_look_at_keeps_up_upright(self):
        q = look_at_quaternion(Y, Z)
        assert camera_up(q)[2] > 0.99

    def test_look_at_handles_up_parallel_to_forward(self):
        """Looking straight down: `up` is degenerate and must not produce NaN."""
        q = look_at_quaternion(-Z, Z)
        d = view_direction(q)
        assert np.all(np.isfinite(d))
        assert np.allclose(d, -Z, atol=1e-9)

    def test_camera_axes_are_orthonormal_right_handed(self):
        for m in random_rotations(20, seed=3):
            right, forward, up = camera_axes(matrix_to_quat(m))
            assert np.isclose(np.dot(right, forward), 0, atol=1e-9)
            assert np.isclose(np.dot(forward, up), 0, atol=1e-9)
            assert np.allclose(np.cross(right, forward), up, atol=1e-9)


class TestRoundTrips:
    def test_cv_round_trip(self):
        for m in random_rotations(30, seed=1):
            assert np.allclose(cv_rotation_to_camerapath(camerapath_rotation_to_cv(m)), m)

    def test_blender_round_trip(self):
        for m in random_rotations(30, seed=2):
            assert np.allclose(
                blender_rotation_to_camerapath(camerapath_rotation_to_blender(m)), m
            )

    def test_blender_quaternion_round_trip(self):
        for m in random_rotations(30, seed=4):
            q = matrix_to_quat(m)
            back = blender_quat_to_camerapath(camerapath_quat_to_blender(q))
            # Compare as rotations, not componentwise: q and -q are equal.
            assert np.allclose(quat_to_matrix(back), quat_to_matrix(q), atol=1e-9)


class TestColmapPose:
    def test_camera_centre_is_negated_and_rotated(self):
        """C = -R_wc^T . t_wc. Using t_wc directly gives a mirrored path, which
        is the classic COLMAP integration bug."""
        r_wc = quat_to_matrix(quat_from_axis_angle(Z, 0.7))
        centre_true = np.array([3.0, -1.0, 2.0])
        t_wc = -r_wc @ centre_true

        centre, r_cw = colmap_world_to_camera_to_pose(matrix_to_quat(r_wc), t_wc)
        assert np.allclose(centre, centre_true, atol=1e-9)
        assert np.allclose(r_cw, r_wc.T, atol=1e-9)

    def test_centre_differs_from_raw_translation(self):
        """Guard against the two being accidentally equal in the fixture, which
        would make the test above vacuous."""
        r_wc = quat_to_matrix(quat_from_axis_angle(np.array([0.3, 1.0, 0.2]), 1.1))
        centre_true = np.array([5.0, 2.0, -3.0])
        t_wc = -r_wc @ centre_true
        assert not np.allclose(t_wc, centre_true, atol=0.5)

    def test_colmap_round_trip(self):
        rng = np.random.default_rng(11)
        for _ in range(25):
            r_cw = quat_to_matrix(quat_from_axis_angle(rng.normal(size=3), rng.uniform(-3, 3)))
            centre = rng.normal(scale=4.0, size=3)
            q_wc, t_wc = pose_to_colmap_world_to_camera(centre, r_cw)
            centre_back, r_cw_back = colmap_world_to_camera_to_pose(q_wc, t_wc)
            assert np.allclose(centre_back, centre, atol=1e-8)
            assert np.allclose(r_cw_back, r_cw, atol=1e-8)

    def test_identity_pose_is_origin(self):
        centre, r_cw = colmap_world_to_camera_to_pose(quat_identity(), np.zeros(3))
        assert np.allclose(centre, np.zeros(3))
        assert np.allclose(r_cw, np.eye(3))

    def test_full_colmap_to_blender_chain(self):
        """The complete path a real pose takes: COLMAP -> CameraPath -> Blender.

        A camera at (0,-5,0) in COLMAP world looking towards the origin must end
        up in Blender at the same location, with its -Z axis pointing at the
        origin.
        """
        # Build a COLMAP pose for a camera at (0,-5,0) looking along +Y (world),
        # expressed in OpenCV axes (view along local +Z, local +Y down).
        r_cw_cv = np.column_stack([X, -Z, Y])  # right=+X, down=-Z, forward=+Y
        centre = np.array([0.0, -5.0, 0.0])
        q_wc, t_wc = pose_to_colmap_world_to_camera(centre, r_cw_cv)

        centre_out, r_cw_out = colmap_world_to_camera_to_pose(q_wc, t_wc)
        r_cpl = cv_rotation_to_camerapath(r_cw_out)
        q_cpl = matrix_to_quat(r_cpl)

        assert np.allclose(centre_out, centre, atol=1e-9)
        assert np.allclose(view_direction(q_cpl), Y, atol=1e-9)
        assert np.allclose(camera_up(q_cpl), Z, atol=1e-9)

        r_bl = quat_to_matrix(camerapath_quat_to_blender(q_cpl))
        assert np.allclose(r_bl @ (-Z), Y, atol=1e-9), "Blender camera must look along +Y world"
        assert np.allclose(r_bl @ Y, Z, atol=1e-9), "Blender camera up must be +Z world"


class TestWorldOrientation:
    def test_rotation_aligning_basic(self):
        for a, b in [(X, Y), (Y, Z), (Z, X), (X, -X), (Z, -Z)]:
            r = rotation_aligning(a, b)
            assert np.isclose(np.linalg.det(r), 1.0, atol=1e-9)
            assert np.allclose(r @ a, b, atol=1e-9), f"{a} -> {b}"

    def test_rotation_aligning_identity(self):
        assert np.allclose(rotation_aligning(Z, Z), np.eye(3))

    def test_estimate_up_from_upright_cameras(self):
        quats = [look_at_quaternion(np.array([np.cos(a), np.sin(a), 0.0]), Z)
                 for a in np.linspace(0, 1.2, 8)]
        assert np.allclose(estimate_world_up(quats), Z, atol=1e-6)

    def test_orient_z_up_makes_up_vertical(self):
        tilt = rotation_aligning(Z, np.array([0.3, 0.2, 0.93]))
        quats = [matrix_to_quat(tilt @ quat_to_matrix(look_at_quaternion(Y, Z)))
                 for _ in range(5)]
        positions = np.array([[0, i * 1.0, 0] for i in range(5)], dtype=float)
        _, q_out, _ = orient_trajectory_z_up(positions, quats)
        assert np.allclose(camera_up(q_out[0]), Z, atol=1e-6)

    def test_world_rotation_preserves_relative_geometry(self):
        """A change of basis must not alter path length or inter-camera angles."""
        rng = np.random.default_rng(5)
        positions = rng.normal(size=(12, 3))
        quats = [matrix_to_quat(m) for m in random_rotations(12, seed=6)]
        r = quat_to_matrix(quat_from_axis_angle(np.array([0.4, -1.0, 0.3]), 0.9))

        p2, q2 = apply_world_rotation(positions, quats, r)

        len_before = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        len_after = float(np.linalg.norm(np.diff(p2, axis=0), axis=1).sum())
        assert np.isclose(len_before, len_after, rtol=1e-9)

        for i in range(len(quats) - 1):
            from app.geometry.rotations import quat_angular_distance
            before = quat_angular_distance(quats[i], quats[i + 1])
            after = quat_angular_distance(q2[i], q2[i + 1])
            assert np.isclose(before, after, atol=1e-9)

    def test_degenerate_up_estimate_does_not_crash(self):
        """A full roll makes the camera up vectors cancel; no usable estimate."""
        quats = [matrix_to_quat(quat_to_matrix(quat_from_axis_angle(Y, a)))
                 for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        up = estimate_world_up(quats)
        assert np.all(np.isfinite(up))
        assert np.isclose(np.linalg.norm(up), 1.0, atol=1e-6)


class TestNoSilentHandednessFlip:
    @pytest.mark.parametrize("fn", [
        camerapath_rotation_to_cv, cv_rotation_to_camerapath,
        camerapath_rotation_to_blender, blender_rotation_to_camerapath,
    ])
    def test_conversions_preserve_determinant(self, fn):
        """A sign error anywhere here flips handedness, mirroring the whole
        trajectory while leaving it smooth and plausible."""
        for m in random_rotations(15, seed=7):
            out = fn(m)
            assert np.isclose(np.linalg.det(out), 1.0, atol=1e-9)
            assert np.allclose(out @ out.T, np.eye(3), atol=1e-9)
