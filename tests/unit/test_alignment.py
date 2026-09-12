"""Sim(3) alignment (spec §9, §24)."""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.alignment import (
    Similarity,
    align_and_measure,
    align_overlapping_windows,
    umeyama,
)
from app.geometry.rotations import (
    matrix_to_quat,
    quat_from_axis_angle,
    quat_identity,
    quat_to_matrix,
)


def rand_rotation(rng) -> np.ndarray:
    return quat_to_matrix(quat_from_axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi)))


class TestUmeyama:
    def test_recovers_a_known_similarity(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(40, 3))
        R = rand_rotation(rng)
        scale, t = 2.7, np.array([4.0, -1.0, 0.5])
        dst = (scale * (R @ src.T)).T + t

        fit = umeyama(src, dst)
        assert np.isclose(fit.scale, scale, rtol=1e-9)
        assert np.allclose(fit.rotation, R, atol=1e-9)
        assert np.allclose(fit.translation, t, atol=1e-9)

    def test_rigid_mode_ignores_scale(self):
        rng = np.random.default_rng(1)
        src = rng.normal(size=(30, 3))
        dst = (3.0 * src.T).T
        fit = umeyama(src, dst, allow_scale=False)
        assert fit.scale == 1.0

    def test_never_returns_a_reflection(self):
        """A reflection fits mirrored data BETTER than the correct rotation, so
        accepting one produces a silently mirrored trajectory with a flattering
        error score."""
        rng = np.random.default_rng(2)
        src = rng.normal(size=(50, 3))
        mirrored = src.copy()
        mirrored[:, 0] *= -1.0  # a reflection, not a rotation
        fit = umeyama(src, mirrored)
        assert np.isclose(np.linalg.det(fit.rotation), 1.0, atol=1e-9)

    def test_round_trip_through_inverse(self):
        rng = np.random.default_rng(3)
        src = rng.normal(size=(25, 3))
        fit = Similarity(1.8, rand_rotation(rng), np.array([1.0, 2.0, 3.0]))
        assert np.allclose(fit.inverse().apply(fit.apply(src)), src, atol=1e-9)

    def test_degenerate_inputs(self):
        assert umeyama(np.empty((0, 3)), np.empty((0, 3))).scale == 1.0
        one = umeyama(np.array([[1.0, 1.0, 1.0]]), np.array([[2.0, 2.0, 2.0]]))
        assert np.allclose(one.translation, np.ones(3))
        # All source points identical: no scale is recoverable.
        same = umeyama(np.ones((5, 3)), np.random.default_rng(4).normal(size=(5, 3)))
        assert same.scale == 1.0 and np.isfinite(same.scale)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            umeyama(np.zeros((3, 3)), np.zeros((4, 3)))

    def test_quaternion_transform_uses_rotation_only(self):
        rng = np.random.default_rng(5)
        fit = Similarity(4.2, rand_rotation(rng), np.array([9.0, 9.0, 9.0]))
        q = quat_from_axis_angle(np.array([0.2, 1.0, -0.4]), 0.8)
        out = fit.apply_quaternion(q)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-9)
        assert np.allclose(quat_to_matrix(out), fit.rotation @ quat_to_matrix(q), atol=1e-9)


class TestAlignAndMeasure:
    def test_perfect_estimate_scores_zero(self):
        rng = np.random.default_rng(6)
        pos = np.cumsum(rng.normal(size=(30, 3)), axis=0)
        quats = [quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), i * 0.05) for i in range(30)]
        _, err = align_and_measure(pos, quats, pos, quats)
        assert err.absolute_translation_rmse < 1e-9
        assert err.normalized_shape_error < 1e-9
        assert err.rotation_mae_degrees < 1e-9

    def test_scale_only_difference_scores_zero_shape_error(self):
        """The point of Sim(3): a correctly-shaped monocular estimate at the
        wrong scale must score as correct, because scale is unrecoverable."""
        rng = np.random.default_rng(7)
        ref = np.cumsum(rng.normal(size=(40, 3)), axis=0)
        est = ref * 0.13
        quats = [quat_identity()] * 40
        _, err = align_and_measure(est, quats, ref, quats)
        assert err.normalized_shape_error < 1e-9
        assert np.isclose(err.scale_factor, 1 / 0.13, rtol=1e-6)

    def test_rigid_comparison_exposes_the_scale_error(self):
        """The same data compared rigidly must score far worse, confirming
        `allow_scale` does real work rather than being decorative.

        Compared against the Sim(3) result rather than an absolute threshold:
        the rigid error depends on the trajectory's shape, so any fixed number
        would be a fixture-specific magic value.
        """
        rng = np.random.default_rng(8)
        ref = np.cumsum(rng.normal(size=(40, 3)), axis=0)
        est = ref * 0.13
        quats = [quat_identity()] * 40

        _, similar = align_and_measure(est, quats, ref, quats, allow_scale=True)
        _, rigid = align_and_measure(est, quats, ref, quats, allow_scale=False)

        assert similar.normalized_shape_error < 1e-9
        assert rigid.normalized_shape_error > 1e4 * max(similar.normalized_shape_error, 1e-12)
        assert rigid.normalized_shape_error > 0.01
        # And the rigid fit must not have smuggled a scale in.
        assert rigid.scale_factor == 1.0
        assert np.isclose(similar.scale_factor, 1 / 0.13, rtol=1e-6)

    def test_rotation_error_is_measured(self):
        pos = np.array([[0.0, float(i), 0.0] for i in range(10)])
        ref_q = [quat_identity()] * 10
        est_q = [quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), np.radians(3.0))] * 10
        _, err = align_and_measure(pos, est_q, pos, ref_q, allow_scale=False)
        assert np.isclose(err.rotation_mae_degrees, 3.0, atol=1e-6)

    def test_shape_error_grows_with_distortion(self):
        rng = np.random.default_rng(9)
        ref = np.cumsum(rng.normal(size=(50, 3)), axis=0)
        quats = [quat_identity()] * 50
        errs = []
        for noise in (0.0, 0.05, 0.2):
            est = ref + rng.normal(scale=noise, size=ref.shape)
            _, e = align_and_measure(est, quats, ref, quats)
            errs.append(e.normalized_shape_error)
        assert errs[0] < errs[1] < errs[2]

    def test_near_static_reference_does_not_explode(self):
        """Path length ~0 would make the normalised error divide by zero."""
        ref = np.zeros((20, 3)) + np.array([1.0, 2.0, 3.0])
        est = ref + 1e-4
        quats = [quat_identity()] * 20
        _, err = align_and_measure(est, quats, ref, quats)
        assert np.isfinite(err.normalized_shape_error)

    def test_handles_length_mismatch(self):
        ref = np.cumsum(np.ones((20, 3)), axis=0)
        est = ref[:12]
        q = [quat_identity()] * 20
        _, err = align_and_measure(est, q[:12], ref, q)
        assert err.sample_count == 12


class TestWindowMerge:
    def test_stitches_windows_with_different_scales(self):
        """Each learned-geometry window has its own arbitrary scale; the overlap
        is what ties them together (spec §9)."""
        rng = np.random.default_rng(10)
        truth = np.cumsum(rng.normal(size=(60, 3)), axis=0)
        truth_q = [quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), i * 0.02) for i in range(60)]

        # Window A: frames 0-39 in the true frame.
        a_pos, a_q = truth[:40], truth_q[:40]
        # Window B: frames 24-59, in its own arbitrary similarity frame.
        R = rand_rotation(rng)
        scale, t = 0.37, np.array([-2.0, 5.0, 1.0])
        b_pos = (scale * (R @ truth[24:].T)).T + t
        b_q = [matrix_to_quat(R @ quat_to_matrix(q)) for q in truth_q[24:]]

        _, merged_pos, merged_q = align_overlapping_windows(
            a_pos, a_q, b_pos, b_q, slice(24, 40), slice(0, 16)
        )
        assert np.allclose(merged_pos, truth[24:], atol=1e-7)
        for i in (0, 8, 30):
            assert np.allclose(
                quat_to_matrix(merged_q[i]), quat_to_matrix(truth_q[24 + i]), atol=1e-7
            )

    def test_rejects_insufficient_overlap(self):
        pos = np.zeros((10, 3))
        q = [quat_identity()] * 10
        with pytest.raises(ValueError):
            align_overlapping_windows(pos, q, pos, q, slice(0, 1), slice(0, 1))
        with pytest.raises(ValueError):
            align_overlapping_windows(pos, q, pos, q, slice(0, 5), slice(0, 3))
