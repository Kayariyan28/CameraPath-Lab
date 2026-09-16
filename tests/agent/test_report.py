"""The projection from a solved job onto the agent-facing motion report.

These tests are about what the report is *not allowed* to say: no invented
units, no travel numbers from a shot where translation was not observable, no
solver attributed to a rung that was not selected.
"""

from __future__ import annotations

import math

import pytest

from app.agent import report as report_mod
from app.models.schemas.trajectory import (
    ClassifiedMove,
    ConfidenceLevel,
    MotionLabel,
    ScaleMode,
    SolverSource,
)
from app.trajectory.classify import ZOOM_RATIO_THRESHOLD
from tests.agent.conftest import make_job, make_trajectory


def _report(**kwargs):
    trajectory = make_trajectory(**kwargs)
    job = make_job(trajectories=[trajectory])
    return report_mod.job_report(
        job, ui_url="http://x/?job=1", job_dir="/tmp/j"
    ).shots[0], job, trajectory


def test_units_come_from_the_scale_mode_not_the_trajectorys_own_label():
    # A trajectory that *claims* metres without a surviving calibration must not
    # be published as metres: resolve_scale downgrades it, and the report has to
    # inherit the downgrade rather than repeat the claim.
    shot, _job, _traj = _report(scale_mode=ScaleMode.NORMALIZED, scale_units="m")
    assert shot.translation.units == "normalized"
    assert shot.translation.speed_units == "normalized/s"


def test_metric_units_require_a_surviving_calibration():
    shot, _job, _t = _report(
        scale_mode=ScaleMode.METRIC, metric_scale_factor=2.5, scale_units="m"
    )
    assert shot.translation.units == "m"

    downgraded, _job2, _t2 = _report(scale_mode=ScaleMode.METRIC, metric_scale_factor=None)
    assert downgraded.translation.units == "normalized"


def test_translation_magnitudes_are_null_when_translation_is_not_observable():
    shot, _job, _t = _report(translation_observable=False)
    assert shot.translation.observable is False
    assert shot.translation.total_path_length is None
    assert shot.translation.mean_speed is None
    assert shot.translation.peak_speed is None
    assert shot.translation.peak_speed_time is None


def test_translation_magnitudes_are_populated_when_it_is_observable():
    shot, _job, traj = _report(translation_observable=True)
    assert shot.translation.total_path_length == pytest.approx(traj.total_path_length)
    assert shot.translation.mean_speed == pytest.approx(1.0)


def test_not_observable_reason_is_the_routing_reason_verbatim():
    reason = "a single homography explains 98% of the motion and parallax is 0.02"
    trajectory = make_trajectory(translation_observable=False)
    job = make_job(trajectories=[trajectory], recommendation_reason=reason)
    shot = report_mod.job_report(job, ui_url="u", job_dir="d").shots[0]
    assert shot.translation.not_observable_reason == reason


def test_solver_used_matches_the_selected_decision():
    shot, _job, _t = _report(solver=SolverSource.PERCEPTUAL)
    assert shot.solver_used == "perceptual"
    assert [a.solver for a in shot.solver_attempts if a.selected] == ["perceptual"]
    assert "perceptual" in shot.solver_explanation


def test_confidence_reasons_are_verbatim():
    reasons = ["only 41% of frames registered", "median reprojection error 2.8 px"]
    shot, _job, _t = _report(level=ConfidenceLevel.LOW, reasons=reasons)
    assert shot.confidence.reasons == reasons
    assert shot.confidence.level == "low"


def test_focal_ratio_uses_the_same_tan_formula_as_the_classifier():
    shot, _job, _t = _report(fov_start=60.0, fov_end=45.0)
    expected = math.tan(math.radians(60.0) / 2) / math.tan(math.radians(45.0) / 2)
    assert shot.lens.focal_ratio == pytest.approx(expected)
    assert shot.lens.fov_horizontal_start_deg == pytest.approx(60.0)
    assert shot.lens.fov_horizontal_end_deg == pytest.approx(45.0)


def test_zoom_is_none_unless_the_classifier_emitted_a_zoom_label():
    # A FOV drift below the classifier's threshold is reported as a ratio and is
    # explicitly not a zoom.
    drift = 60.0 / (ZOOM_RATIO_THRESHOLD ** 0.5)
    shot, _job, _t = _report(fov_start=60.0, fov_end=drift)
    assert shot.lens.focal_ratio is not None
    assert shot.lens.zoom == "none"

    zoomed, _j, _t2 = _report(
        moves=[ClassifiedMove(label=MotionLabel.ZOOM_IN, strength=0.4, start_time=0.0,
                              end_time=1.0, description="focal length x1.34")]
    )
    assert zoomed.lens.zoom == "in"


def test_moves_round_trip_every_field():
    move = ClassifiedMove(
        label=MotionLabel.ORBIT, strength=0.58, start_time=0.4, end_time=2.9,
        description="104 deg of yaw while travelling sideways about a centre",
    )
    shot, _job, _t = _report(moves=[move])
    assert shot.moves[0].label == "orbit"
    assert shot.moves[0].strength == pytest.approx(0.58)
    assert shot.moves[0].start_time == pytest.approx(0.4)
    assert shot.moves[0].end_time == pytest.approx(2.9)
    assert shot.moves[0].description == move.description
    assert shot.primary_move == "orbit"
    assert shot.is_static is False


def test_static_is_flagged_only_for_a_lone_static_label():
    shot, _job, _t = _report(
        moves=[ClassifiedMove(label=MotionLabel.STATIC, strength=1.0, start_time=0.0,
                              end_time=1.0, description="no sustained motion")]
    )
    assert shot.is_static is True


def test_poses_are_strided_never_resampled():
    trajectory = make_trajectory(frames=100)
    job = make_job(trajectories=[trajectory])
    report = report_mod.job_report(
        job, ui_url="u", job_dir="d", include_poses=True, max_poses=10
    )
    shot = report.shots[0]
    assert shot.pose_stride == 10
    assert shot.poses_are_strided_subset is True
    assert len(shot.poses) == 10
    # Every returned row is a real solved pose, unchanged.
    for i, pose in enumerate(shot.poses):
        original = trajectory.poses[i * 10]
        assert pose.frame_index == original.frame_index
        assert pose.timestamp == pytest.approx(original.timestamp)
        assert pose.position == [pytest.approx(v) for v in original.position]


def test_poses_are_omitted_unless_requested():
    shot, _job, _t = _report()
    assert shot.poses == []
    assert shot.poses_are_strided_subset is False


def test_anchor_count_is_counted_over_all_poses_not_the_subset():
    trajectory = make_trajectory(frames=100)
    job = make_job(trajectories=[trajectory])
    shot = report_mod.job_report(
        job, ui_url="u", job_dir="d", include_poses=True, max_poses=5
    ).shots[0]
    assert shot.anchor_frame_count == sum(1 for p in trajectory.poses if p.is_anchor)


def test_job_scale_is_the_most_conservative_across_shots():
    metric = make_trajectory(0, scale_mode=ScaleMode.METRIC, metric_scale_factor=2.0)
    normalized = make_trajectory(1, start=2.0)
    job = make_job(shots=2, trajectories=[metric, normalized])
    report = report_mod.job_report(job, ui_url="u", job_dir="d")
    assert report.scale.units == "normalized"
    assert report.shot_count == 2


def test_shot_identity_and_timing_come_from_the_analysis_and_video():
    job = make_job(shots=2)
    report = report_mod.job_report(job, ui_url="u", job_dir="d")
    assert [s.shot_id for s in report.shots] == [0, 1]
    assert report.shots[1].start_time == pytest.approx(2.0)
    assert report.shots[0].timing_source == "container_pts"
    assert report.shots[0].frame_rate_mode == "constant"
    assert report.shots[0].cut_confidence == pytest.approx(1.0)


def test_selecting_one_shot_returns_only_that_shot():
    job = make_job(shots=2)
    report = report_mod.job_report(job, ui_url="u", job_dir="d", shot_id=1)
    assert [s.shot_id for s in report.shots] == [1]
    assert report.shot_count == 2
