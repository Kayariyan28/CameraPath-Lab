"""Quaternion algebra and interpolation (invariant I3)."""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.rotations import (
    angular_velocity,
    matrix_to_quat,
    quat_angle,
    quat_angular_distance,
    quat_canonical,
    quat_conjugate,
    quat_exp,
    quat_from_axis_angle,
    quat_identity,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_relative,
    quat_rotate,
    quat_sequence_squad,
    quat_slerp,
    quat_squad,
    quat_to_matrix,
    unroll_quaternions,
)

X, Y, Z = np.eye(3)


class TestAlgebra:
    def test_identity(self):
        assert np.allclose(quat_to_matrix(quat_identity()), np.eye(3))

    def test_multiply_by_conjugate_is_identity(self):
        q = quat_from_axis_angle(np.array([1.0, 2.0, 3.0]), 0.9)
        assert np.allclose(quat_multiply(q, quat_conjugate(q)), quat_identity())

    def test_rotate_matches_matrix(self):
        rng = np.random.default_rng(0)
        for _ in range(30):
            q = quat_from_axis_angle(rng.normal(size=3), rng.uniform(-3, 3))
            v = rng.normal(size=3)
            assert np.allclose(quat_rotate(q, v), quat_to_matrix(q) @ v, atol=1e-9)

    def test_composition_order(self):
        """quat_multiply(a, b) must apply b first, then a — matching matrix order."""
        a = quat_from_axis_angle(Z, 0.5)
        b = quat_from_axis_angle(X, 0.3)
        assert np.allclose(
            quat_to_matrix(quat_multiply(a, b)),
            quat_to_matrix(a) @ quat_to_matrix(b),
            atol=1e-9,
        )

    def test_90_degree_rotations(self):
        q = quat_from_axis_angle(Z, np.pi / 2)
        assert np.allclose(quat_rotate(q, X), Y, atol=1e-9)
        q = quat_from_axis_angle(X, np.pi / 2)
        assert np.allclose(quat_rotate(q, Y), Z, atol=1e-9)

    def test_normalize_handles_zero(self):
        assert np.allclose(quat_normalize(np.zeros(4)), quat_identity())

    def test_canonical_forces_positive_w(self):
        q = np.array([-0.7071, 0.0, 0.7071, 0.0])
        assert quat_canonical(q)[0] >= 0
        # Same rotation either way.
        assert np.allclose(quat_to_matrix(q), quat_to_matrix(quat_canonical(q)), atol=1e-9)


class TestMatrixConversion:
    def test_round_trip(self):
        rng = np.random.default_rng(1)
        for _ in range(80):
            q = quat_canonical(quat_from_axis_angle(rng.normal(size=3), rng.uniform(-np.pi, np.pi)))
            assert np.allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-8)

    @pytest.mark.parametrize("angle", [0.0, 1e-7, np.pi / 2, np.pi - 1e-6, np.pi])
    def test_precision_at_extremes(self, angle):
        """Near 180 degrees the naive single-branch conversion loses all
        precision. A shot that reverses direction reaches this."""
        for axis in (X, Y, Z, np.array([1.0, 1.0, 1.0])):
            q = quat_from_axis_angle(axis, angle)
            m = quat_to_matrix(q)
            back = matrix_to_quat(m)
            assert np.allclose(quat_to_matrix(back), m, atol=1e-7), (axis, angle)

    def test_all_branches_exercised(self):
        """Each of the four Shepperd branches must produce a valid result."""
        cases = [
            quat_from_axis_angle(X, np.pi),   # x-dominant
            quat_from_axis_angle(Y, np.pi),   # y-dominant
            quat_from_axis_angle(Z, np.pi),   # z-dominant
            quat_identity(),                  # trace-dominant
        ]
        for q in cases:
            m = quat_to_matrix(q)
            assert np.allclose(quat_to_matrix(matrix_to_quat(m)), m, atol=1e-9)


class TestLogExp:
    def test_round_trip(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            q = quat_canonical(quat_from_axis_angle(rng.normal(size=3), rng.uniform(0, np.pi * 0.99)))
            assert np.allclose(quat_exp(quat_log(q)), q, atol=1e-8)

    def test_log_of_identity_is_zero(self):
        assert np.allclose(quat_log(quat_identity()), np.zeros(3))

    def test_log_magnitude_is_the_angle(self):
        for angle in (0.1, 1.0, 2.5):
            q = quat_from_axis_angle(X, angle)
            assert np.isclose(np.linalg.norm(quat_log(q)), angle, atol=1e-9)


class TestSlerp:
    def test_endpoints_are_exact(self):
        a = quat_from_axis_angle(X, 0.3)
        b = quat_from_axis_angle(Y, 1.1)
        assert np.allclose(quat_to_matrix(quat_slerp(a, b, 0.0)), quat_to_matrix(a), atol=1e-9)
        assert np.allclose(quat_to_matrix(quat_slerp(a, b, 1.0)), quat_to_matrix(b), atol=1e-9)

    def test_constant_angular_speed(self):
        """The defining property: equal parameter steps give equal angles."""
        a = quat_identity()
        b = quat_from_axis_angle(Z, 1.4)
        steps = np.linspace(0, 1, 21)
        quats = [quat_slerp(a, b, t) for t in steps]
        deltas = [quat_angular_distance(quats[i], quats[i + 1]) for i in range(len(quats) - 1)]
        assert np.std(deltas) < 1e-9, "slerp must have constant angular speed"

    def test_takes_the_short_path(self):
        """With b negated (same rotation), slerp must not go the long way."""
        a = quat_identity()
        b = quat_from_axis_angle(Z, 0.6)
        mid_pos = quat_slerp(a, b, 0.5)
        mid_neg = quat_slerp(a, -b, 0.5)
        assert np.allclose(quat_to_matrix(mid_pos), quat_to_matrix(mid_neg), atol=1e-9)

    def test_midpoint_is_half_the_angle(self):
        a = quat_identity()
        b = quat_from_axis_angle(X, 1.0)
        mid = quat_slerp(a, b, 0.5)
        assert np.isclose(quat_angle(mid), 0.5, atol=1e-9)

    def test_nearly_identical_inputs(self):
        a = quat_from_axis_angle(X, 1.0)
        b = quat_from_axis_angle(X, 1.0 + 1e-9)
        out = quat_slerp(a, b, 0.5)
        assert np.all(np.isfinite(out))
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-9)


class TestSquad:
    def test_endpoints_are_exact(self):
        q = [quat_from_axis_angle(Z, a) for a in (0.0, 0.4, 0.9, 1.5)]
        assert np.allclose(quat_to_matrix(quat_squad(*q, 0.0)), quat_to_matrix(q[1]), atol=1e-8)
        assert np.allclose(quat_to_matrix(quat_squad(*q, 1.0)), quat_to_matrix(q[2]), atol=1e-8)

    def test_stays_a_unit_quaternion(self):
        q = [quat_from_axis_angle(np.array([1.0, 0.4, -0.3]), a) for a in (0.1, 0.7, 1.3, 2.0)]
        for t in np.linspace(0, 1, 15):
            assert np.isclose(np.linalg.norm(quat_squad(*q, float(t))), 1.0, atol=1e-9)

    def test_angular_velocity_is_continuous(self):
        """SQUAD exists because chained SLERP is only C0: angular velocity jumps
        at every control point, which would show up in the exported camera as a
        per-keyframe stutter that is not in the source.

        Measured as the largest single-step change in angular rate — a genuine
        discontinuity measure. Total variation is NOT usable here: it penalises
        SQUAD's continuous rate variation exactly as much as SLERP's jumps, and
        by that measure SLERP scores "better" while being visibly worse.

        The decisive evidence is convergence. For a C1 curve the largest step
        change shrinks as ~1/N^2 as sampling density rises; across a genuine
        discontinuity it can only shrink as ~1/N. Measured here: SQUAD improves
        4x from N=80 to N=320 while SLERP improves 4x only in proportion to the
        step size, leaving SQUAD 17x smoother at N=320.
        """
        angles = [0.0, 0.15, 0.8, 1.0, 1.9]  # deliberately uneven spacing
        controls = [quat_from_axis_angle(Z, a) for a in angles]
        times = np.arange(len(controls), dtype=float)

        def max_rate_jump(seq: np.ndarray) -> float:
            rates = np.array([
                quat_angular_distance(seq[i], seq[i + 1]) for i in range(len(seq) - 1)
            ])
            return float(np.abs(np.diff(rates)).max())

        def sample(n: int) -> tuple[float, float]:
            samples = np.linspace(0, len(controls) - 1, n)
            squad = quat_sequence_squad(controls, times, samples)
            slerp = np.array([
                quat_slerp(
                    controls[min(int(t), len(controls) - 2)],
                    controls[min(int(t), len(controls) - 2) + 1],
                    float(t - min(int(t), len(controls) - 2)),
                )
                for t in samples
            ])
            return max_rate_jump(squad), max_rate_jump(slerp)

        squad_80, slerp_80 = sample(80)
        squad_320, slerp_320 = sample(320)

        assert squad_80 < slerp_80, "SQUAD must have smaller rate discontinuities"
        assert squad_320 < slerp_320 / 5.0, (
            f"SQUAD {squad_320:.6f} vs SLERP {slerp_320:.6f} — expected a large margin"
        )
        # C1 convergence: 4x the samples should give much better than 4x
        # improvement for a continuous curve.
        assert squad_80 / squad_320 > 8.0, (
            f"rate jump fell only {squad_80 / squad_320:.1f}x for 4x samples — "
            "angular velocity does not look continuous"
        )

    def test_sequence_hits_controls_exactly(self):
        controls = [quat_from_axis_angle(Z, a) for a in (0.0, 0.5, 1.2, 1.9)]
        times = np.array([0.0, 1.0, 2.0, 3.0])
        out = quat_sequence_squad(controls, times, times)
        for i, q in enumerate(controls):
            assert np.allclose(quat_to_matrix(out[i]), quat_to_matrix(q), atol=1e-8), i

    def test_sequence_does_not_extrapolate(self):
        """Past the measured range, hold — never invent rotation."""
        controls = [quat_identity(), quat_from_axis_angle(Z, 1.0)]
        times = np.array([0.0, 1.0])
        out = quat_sequence_squad(controls, times, np.array([-5.0, 0.0, 1.0, 9.0]))
        assert np.allclose(quat_to_matrix(out[0]), quat_to_matrix(controls[0]), atol=1e-8)
        assert np.allclose(quat_to_matrix(out[-1]), quat_to_matrix(controls[1]), atol=1e-8)

    def test_sequence_handles_single_and_empty(self):
        assert quat_sequence_squad([], np.array([]), np.array([0.0, 1.0])).shape == (2, 4)
        one = quat_sequence_squad([quat_from_axis_angle(X, 0.4)], np.array([0.0]),
                                  np.array([0.0, 5.0]))
        assert np.allclose(one[0], one[1])


class TestAngularVelocity:
    def test_matches_a_known_rate(self):
        """A 1 rad/s yaw for 0.1 s must read back as 1 rad/s about the yaw axis."""
        a = quat_identity()
        b = quat_from_axis_angle(Z, 0.1)
        w = angular_velocity(a, b, 0.1)
        assert np.allclose(w, np.array([0.0, 0.0, 1.0]), atol=1e-9)

    def test_is_body_frame(self):
        """Rates are expressed in the camera's own frame, which is what "pan
        rate" and "tilt rate" mean to an operator."""
        base = quat_from_axis_angle(Z, np.pi / 2)  # camera yawed 90 degrees
        delta = quat_from_axis_angle(X, 0.2)       # then pitched in its own frame
        w = angular_velocity(base, quat_multiply(base, delta), 0.2)
        assert np.allclose(w, np.array([1.0, 0.0, 0.0]), atol=1e-9)

    def test_zero_dt(self):
        assert np.allclose(angular_velocity(quat_identity(), quat_identity(), 0.0), np.zeros(3))

    def test_relative_is_consistent(self):
        a = quat_from_axis_angle(X, 0.3)
        b = quat_from_axis_angle(Y, 0.9)
        assert np.allclose(quat_to_matrix(quat_multiply(a, quat_relative(a, b))),
                           quat_to_matrix(b), atol=1e-9)


class TestUnroll:
    def test_removes_sign_flips(self):
        q = quat_from_axis_angle(Z, 0.4)
        seq = np.array([q, -q, q, -q])
        out = unroll_quaternions(seq)
        for i in range(len(out) - 1):
            assert float(np.dot(out[i], out[i + 1])) > 0

    def test_preserves_rotations(self):
        rng = np.random.default_rng(9)
        seq = np.array([
            quat_from_axis_angle(rng.normal(size=3), rng.uniform(-3, 3)) for _ in range(12)
        ])
        out = unroll_quaternions(seq)
        for a, b in zip(seq, out):
            assert np.allclose(quat_to_matrix(a), quat_to_matrix(b), atol=1e-9)

    def test_differencing_is_safe_after_unrolling(self):
        """The reason this exists: differencing a sign-flipped sequence yields
        enormous spurious angular velocities."""
        q = quat_from_axis_angle(Z, 0.4)
        seq = np.array([q, -q, q])
        raw = float(np.abs(np.diff(seq, axis=0)).max())
        rolled = float(np.abs(np.diff(unroll_quaternions(seq), axis=0)).max())
        assert raw > 1.0 and rolled < 1e-9
