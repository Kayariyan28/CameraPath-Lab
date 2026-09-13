"""Anchor cross-checks: gross mis-registrations must be caught, and a good anchor
must never be sacrificed. Both failure directions were observed on real solves."""

from __future__ import annotations

import numpy as np

from app.geometry.rotations import quat_from_axis_angle
from app.models.schemas.motion import MotionFrame
from app.models.schemas.trajectory import SolverSource
from app.solvers.anchor_validation import (
    per_transition_rotation_bound,
    validate_anchor_rotations,
)
from app.solvers.base import GeometryResult

FOCAL = 900.0
YAW_DEG_PER_FRAME = 1.0


def _motion(frames: range, yaw_deg_per_frame: float = YAW_DEG_PER_FRAME,
            extra_flow_px: float = 0.0) -> list[MotionFrame]:
    """Flow a camera yawing at a constant rate would produce, plus optional
    translational flow (which only loosens the bound)."""
    dx = FOCAL * np.tan(np.radians(yaw_deg_per_frame)) + extra_flow_px
    return [
        MotionFrame(frame_index=f, timestamp=f / 30, dt=1 / 30, dx_pixels=dx,
                    dy_pixels=0.0, rotation_deg=0.0, confidence=0.9, inlier_ratio=0.9,
                    flow_magnitude=abs(dx), flow_magnitude_p90=abs(dx))
        for f in frames
    ]


def _result(frames: list[int], yaw_deg: dict[int, float] | None = None) -> GeometryResult:
    yaw_deg = yaw_deg or {}
    quats = [
        quat_from_axis_angle(np.array([0, 0, 1.0]),
                             np.radians(yaw_deg.get(f, f * YAW_DEG_PER_FRAME)))
        for f in frames
    ]
    return GeometryResult(
        source=SolverSource.COLMAP, frame_indices=list(frames),
        positions=np.zeros((len(frames), 3)), quaternions=quats,
        per_pose_confidence=[0.9] * len(frames), confidence=0.9,
        registered_frames=len(frames), total_frames=len(frames), succeeded=True,
    )


class TestBound:
    def test_pure_rotation_bound_matches_closed_form(self):
        bounds = per_transition_rotation_bound(_motion(range(1, 4)), FOCAL)
        for b in bounds.values():
            assert abs(b - YAW_DEG_PER_FRAME) < 1e-9

    def test_low_confidence_transitions_never_tighten_the_bound(self):
        weak = _motion(range(1, 2))[0].model_copy(
            update={"confidence": 0.1, "inlier_ratio": 0.1, "flow_magnitude_p90": 80.0}
        )
        assert per_transition_rotation_bound([weak], FOCAL)[1] > YAW_DEG_PER_FRAME


class TestValidation:
    FRAMES = [0, 5, 10, 15, 20, 25, 30]

    def test_consistent_anchors_are_all_kept(self):
        out, rejected = validate_anchor_rotations(_result(self.FRAMES), _motion(range(1, 31)), FOCAL)
        assert rejected == [] and out.pose_count == len(self.FRAMES)

    def test_mid_sequence_spike_is_rejected(self):
        bad = {15: 15 * YAW_DEG_PER_FRAME + 50.0}
        out, rejected = validate_anchor_rotations(_result(self.FRAMES, bad), _motion(range(1, 31)), FOCAL)
        assert [r.frame_index for r in rejected] == [15]
        assert 15 not in out.frame_indices and out.pose_count == len(self.FRAMES) - 1

    def test_tail_spike_is_rejected(self):
        """The orbit failure was at the END of the shot."""
        bad = {30: 30 * YAW_DEG_PER_FRAME + 50.0}
        out, rejected = validate_anchor_rotations(_result(self.FRAMES, bad), _motion(range(1, 31)), FOCAL)
        assert [r.frame_index for r in rejected] == [30]

    def test_head_spike_is_rejected(self):
        bad = {0: -50.0}
        _, rejected = validate_anchor_rotations(_result(self.FRAMES, bad), _motion(range(1, 31)), FOCAL)
        assert [r.frame_index for r in rejected] == [0]

    @staticmethod
    def _uneven_motion() -> list[MotionFrame]:
        """True yaw 1 deg/frame everywhere, but frames 16-20 carry heavy
        translational flow (bound 5 deg/frame) and the rest carry none."""
        motion = _motion(range(1, 31))
        heavy = FOCAL * np.tan(np.radians(5.0))
        return [
            m.model_copy(update={"dx_pixels": heavy, "flow_magnitude": heavy,
                                 "flow_magnitude_p90": heavy})
            if 16 <= m.frame_index <= 20 else m
            for m in motion
        ]

    def test_good_anchor_is_never_sacrificed_under_a_loose_bound(self):
        """Regression for the longest-chain search this replaced. Two bad anchors
        (25, 30, both +20 deg) agree with each other and, across the loose
        stretch, with anchor 15 — but not with the GOOD anchor 20 over the tight
        stretch. Skipping 20 made a longer chain, so the search dropped it and
        kept the bad pair. Verified that the old search fails this test."""
        bad = {25: 25 * YAW_DEG_PER_FRAME + 20.0, 30: 30 * YAW_DEG_PER_FRAME + 20.0}
        out, _ = validate_anchor_rotations(_result(self.FRAMES, bad), self._uneven_motion(), FOCAL)
        assert 20 in out.frame_indices

    def test_missing_flow_is_not_evidence_of_inconsistency(self):
        bad = {15: 15 * YAW_DEG_PER_FRAME + 50.0}
        motion = [m for m in _motion(range(1, 31)) if not 11 <= m.frame_index <= 19]
        _, rejected = validate_anchor_rotations(_result(self.FRAMES, bad), motion, FOCAL)
        assert rejected == []

    def test_rejection_lowers_confidence_and_explains_itself(self):
        bad = {15: 15 * YAW_DEG_PER_FRAME + 50.0}
        out, _ = validate_anchor_rotations(_result(self.FRAMES, bad), _motion(range(1, 31)), FOCAL)
        assert out.confidence < 0.9
        assert "rejected 1 of 7" in out.message

    def test_degenerate_input_passes_through(self):
        r = _result([0, 5])
        out, rejected = validate_anchor_rotations(r, _motion(range(1, 6)), FOCAL)
        assert out is r and rejected == []


class TestMisregistrationEvidence:
    """colmap_solver._misregistered_images needs BOTH weak observations and high
    error; a distant-but-correct frame has only the first."""

    class _Pt:
        def __init__(self, pid): self.point3D_id, self._has = pid, True
        def has_point3D(self): return self._has

    class _P3:
        def __init__(self, err): self.error = err

    class _Img:
        def __init__(self, name, pids): self.name, self.points2D = name, [TestMisregistrationEvidence._Pt(p) for p in pids]

    class _Rec:
        def __init__(self, spec):
            self.points3D, self._images, pid = {}, {}, 0
            for i, (name, count, err) in enumerate(spec):
                ids = list(range(pid, pid + count)); pid += count
                for q in ids:
                    self.points3D[q] = TestMisregistrationEvidence._P3(err)
                self._images[i] = TestMisregistrationEvidence._Img(name, ids)
        def reg_image_ids(self): return list(self._images)
        def image(self, i): return self._images[i]

    def _flagged(self, spec):
        from app.solvers.colmap_solver import PyColmapBackend
        return PyColmapBackend._misregistered_images(self._Rec(spec))

    def test_weak_and_inaccurate_is_flagged(self):
        spec = [(f"k{i}", 2000, 0.34) for i in range(8)] + [("bad", 400, 0.80)]
        assert self._flagged(spec) == {"bad"}

    def test_few_observations_alone_is_not_enough(self):
        spec = [(f"k{i}", 3700, 0.30) for i in range(8)] + [("distant_start", 950, 0.30)]
        assert self._flagged(spec) == set()

    def test_high_error_alone_is_not_enough(self):
        spec = [(f"k{i}", 2000, 0.34) for i in range(8)] + [("blurry", 1900, 0.80)]
        assert self._flagged(spec) == set()

    def test_too_few_peers_flags_nothing(self):
        assert self._flagged([("a", 2000, 0.3), ("b", 100, 2.0)]) == set()
