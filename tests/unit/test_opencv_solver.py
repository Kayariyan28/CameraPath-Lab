"""Rotation from keyframe homographies: the convention must be exactly right."""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.conventions import camerapath_rotation_to_cv, view_direction
from app.geometry.rotations import (
    matrix_to_quat, quat_angular_distance, quat_from_axis_angle, quat_multiply, quat_to_matrix,
)
from app.solvers.opencv_solver import relative_cpl_rotation, rotation_from_homography

W, H, F = 1080, 608, 840.0


def _homography_between(q_a: np.ndarray, q_b: np.ndarray, f_a: float, f_b: float) -> np.ndarray:
    """Ground-truth homography for two CameraPath camera-to-world orientations."""
    r_cw_a = camerapath_rotation_to_cv(quat_to_matrix(q_a))
    r_cw_b = camerapath_rotation_to_cv(quat_to_matrix(q_b))
    r_ba = r_cw_b.T @ r_cw_a  # X_b = R_ba X_a in OpenCV camera axes
    k = lambda f: np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])  # noqa: E731
    return k(f_b) @ r_ba @ np.linalg.inv(k(f_a))


@pytest.mark.parametrize("axis,label", [
    (np.array([0, 0, 1.0]), "pan (yaw about +Z)"),
    (np.array([1.0, 0, 0]), "tilt (pitch about +X)"),
    (np.array([0, 1.0, 0]), "roll (about the view axis +Y)"),
    (np.array([0.4, 0.3, 0.8]) / np.linalg.norm([0.4, 0.3, 0.8]), "compound"),
])
def test_known_rotation_is_recovered_in_camerapath_convention(axis, label):
    q_a = quat_from_axis_angle(np.array([0.2, 0.1, 1.0]) / np.linalg.norm([0.2, 0.1, 1.0]), 0.4)
    increment = quat_from_axis_angle(axis, np.radians(9.0))
    q_b = quat_multiply(q_a, increment)  # body-frame increment
    h = _homography_between(q_a, q_b, F, F)
    recovered = quat_multiply(q_a, matrix_to_quat(relative_cpl_rotation(rotation_from_homography(h, F, F, W, H))))
    assert np.degrees(quat_angular_distance(recovered, q_b)) < 1e-6, label


def test_zoom_between_keyframes_does_not_leak_into_rotation():
    q_a = quat_from_axis_angle(np.array([0, 0, 1.0]), 0.0)
    q_b = quat_from_axis_angle(np.array([0, 0, 1.0]), np.radians(6.0))
    h = _homography_between(q_a, q_b, F, F * 1.3)
    r = rotation_from_homography(h, F, F * 1.3, W, H)
    recovered = matrix_to_quat(relative_cpl_rotation(r))
    assert np.degrees(quat_angular_distance(recovered, q_b)) < 1e-6


def test_identity_homography_is_no_rotation():
    r = rotation_from_homography(np.eye(3), F, F, W, H)
    assert np.allclose(r, np.eye(3), atol=1e-12)


def test_pan_direction_sign():
    """Yawing left (+Z) moves world content toward image +x; the recovered camera
    must look further left, not right."""
    q_b = quat_from_axis_angle(np.array([0, 0, 1.0]), np.radians(10.0))
    h = _homography_between(quat_from_axis_angle(np.array([0, 0, 1.0]), 0.0), q_b, F, F)
    q = matrix_to_quat(relative_cpl_rotation(rotation_from_homography(h, F, F, W, H)))
    assert view_direction(q)[0] < 0  # CameraPath +Y forward, +X right: left is -X
