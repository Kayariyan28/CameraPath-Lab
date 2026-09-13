"""Confidence must be earned by evidence and allowed to be LOW (I7)."""

from __future__ import annotations

import numpy as np
import pytest

from app.geometry.rotations import quat_from_axis_angle
from app.models.schemas.motion import LensFrame, MotionFrame, MotionSignature
from app.models.schemas.trajectory import (
    CameraPose, ConfidenceLevel, PipelineMode, SolverDecision, SolverSource,
)
from app.solvers.base import GeometryResult
from app.validation.confidence import _radial_significance, assess_confidence

N = 60


def _poses(translate: bool, source: SolverSource) -> list[CameraPose]:
    return [
        CameraPose(
            frame_index=i, timestamp=i / 30,
            position=[0.0, 0.3 * i if translate else 0.0, 1.5],
            quaternion=[float(v) for v in quat_from_axis_angle(np.array([0, 0, 1.0]), 0.01 * i)],
            fov_horizontal=65.0, focal_normalized=0.78, confidence=0.9,
            solver_source=source, is_anchor=i % 5 == 0,
        )
        for i in range(N)
    ]


def _motion(inliers: float) -> list[MotionFrame]:
    return [
        MotionFrame(frame_index=i, timestamp=i / 30, dt=1 / 30, dx_pixels=8.0, flow_magnitude=8.0,
                    flow_magnitude_p90=12.0, inlier_ratio=inliers, confidence=0.9,
                    tracks_in=1500, tracks_survived=1400)
        for i in range(1, N)
    ]


def _lens(zoom: float = 1.0) -> list[LensFrame]:
    return [LensFrame(frame_index=i, timestamp=i / 30, focal_normalized=0.78 * (1 + (zoom - 1) * i / (N - 1)),
                      fov_horizontal=65.0, fov_vertical=40.0, confidence=0.8) for i in range(N)]


def _signature(parallax: float, inliers: float, scale_change: float = 1.0) -> MotionSignature:
    return MotionSignature(frame_count=N, duration=2.0, mean_flow_magnitude=8.0, mean_inlier_ratio=inliers,
                           parallax_score=parallax, texture_score=0.9, blur_score=0.9,
                           net_scale_change=scale_change, rotation_dominance=0.3)


def _colmap(translation_observable: bool = True) -> GeometryResult:
    frames = list(range(0, N, 5))
    return GeometryResult(
        source=SolverSource.COLMAP, frame_indices=frames,
        positions=np.array([[0.0, 0.3 * f, 1.5] for f in frames]),
        quaternions=[quat_from_axis_angle(np.array([0, 0, 1.0]), 0.01 * f) for f in frames],
        per_pose_confidence=[0.95] * len(frames), focal_pixels=998.0, focal_confidence=0.8,
        focal_observable=True, focal_sensitivity=0.45, focal_image_width=1280,
        registered_frames=len(frames), total_frames=len(frames), mean_reprojection_error=0.34,
        median_reprojection_error=0.26, track_count=11000, mean_track_length=4.3,
        translation_observable=translation_observable, confidence=0.95, message="ok", succeeded=True,
    )


def _decision(result: GeometryResult) -> list[SolverDecision]:
    d = result.to_decision(selected=True)
    return [d]


def test_strong_geometric_solve_with_parallax_reaches_high():
    """Regression: low 2D-model agreement is what parallax looks like, and must not
    hold a sub-pixel COLMAP solve at MEDIUM (it did, at 49% 2D inliers)."""
    geo = _colmap()
    report = assess_confidence(geo, _signature(0.47, 0.49), _motion(0.49), _poses(True, SolverSource.COLMAP),
                               _lens(), _decision(geo), 0.47, 0.9, 0.9, PipelineMode.PHYSICAL_3D)
    assert report.level is ConfidenceLevel.HIGH, report.headline
    assert report.reasons


def test_rotation_only_can_never_be_high():
    geo = _colmap(translation_observable=False)
    report = assess_confidence(geo, _signature(0.02, 0.98), _motion(0.98), _poses(False, SolverSource.COLMAP),
                               _lens(), _decision(geo), 0.02, 0.9, 0.9, PipelineMode.PHYSICAL_3D)
    assert report.level is not ConfidenceLevel.HIGH
    assert report.translation_confidence <= 0.05 + 1e-9


def test_perceptual_match_is_never_high_for_the_physical_path():
    geo = GeometryResult(
        source=SolverSource.PERCEPTUAL, frame_indices=list(range(N)), positions=np.zeros((N, 3)),
        quaternions=[quat_from_axis_angle(np.array([0, 0, 1.0]), 0.01 * i) for i in range(N)],
        per_pose_confidence=[0.9] * N, focal_pixels=840.0, focal_observable=True, focal_image_width=1080,
        registered_frames=N, total_frames=N, mean_reprojection_error=0.2,
        translation_observable=False, confidence=0.9, message="screen-space match", succeeded=True,
    )
    report = assess_confidence(geo, _signature(0.0, 0.99), _motion(0.99), _poses(False, SolverSource.PERCEPTUAL),
                               _lens(), _decision(geo), 0.0, 0.9, 0.9, PipelineMode.PERCEPTUAL_MATCH)
    assert report.level is not ConfidenceLevel.HIGH
    assert any("Perceptual Match" in r for r in report.reasons)


def test_zoom_significance_comes_from_the_lens_curve_not_the_similarity_scale():
    """A pan's similarity fit reports a large scale change that is not a zoom."""
    fake_pan_scale = _signature(0.0, 0.99, scale_change=2.12)
    assert _radial_significance(fake_pan_scale, _lens(zoom=1.0)) == 0.0
    assert _radial_significance(_signature(0.0, 0.99), _lens(zoom=2.25)) == pytest.approx(1.0)


def test_reasons_are_never_empty_even_for_a_failed_solve():
    failed = GeometryResult(source=SolverSource.MOTION_PROXY_2D, succeeded=False, message="nothing registered")
    report = assess_confidence(failed, _signature(0.0, 0.2), _motion(0.2), _poses(False, SolverSource.MOTION_PROXY_2D),
                               _lens(), [], 0.0, 0.1, 0.1, PipelineMode.PERCEPTUAL_MATCH)
    assert report.reasons and report.level is ConfidenceLevel.LOW


def test_solver_agreement_is_not_fabricated_from_one_solver():
    geo = _colmap()
    report = assess_confidence(geo, _signature(0.47, 0.49), _motion(0.49), _poses(True, SolverSource.COLMAP),
                               _lens(), _decision(geo), 0.47, 0.9, 0.9, PipelineMode.PHYSICAL_3D)
    assert report.solver_agreement is None
