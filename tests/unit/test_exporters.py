"""Trajectory exports (spec §19).

These tests are written to fail if an invariant slips, not merely if the code
raises:

  * I5 — a normalized trajectory must never be labelled metres, in any of the
    four export formats, including when the input contradicts itself.
  * I8 — the `.chan` axis conversion is checked against poses whose Nuke
    rotation can be worked out by hand, and against a recomposition of the
    stated Euler order.
  * I4 — a multi-shot job gets one document per shot and an index recording the
    cut times; no document spans a cut.
  * I1/I2 — timestamps are written through unmodified, at full precision.
  * I12 — an empty or partially populated job exports rather than raising.
"""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import numpy as np
import pytest

from app.core.paths import JobPaths
from app.geometry.rotations import quat_from_axis_angle, quat_identity, quat_to_matrix
from app.models.schemas.jobs import AnalysisResult, Job, JobOutputs, ShotAnalysis
from app.models.schemas.motion import LensFrame, MotionSignature
from app.models.schemas.trajectory import (
    CameraPose,
    ClassifiedMove,
    ConfidenceLevel,
    ConfidenceReport,
    CoordinateSystem,
    Kinematics,
    MotionLabel,
    PipelineMode,
    ScaleMode,
    ShotTrajectory,
    SolverDecision,
    SolverSource,
)
from app.models.schemas.video import (
    CutCandidate,
    FrameRateMode,
    Shot,
    ShotComplexity,
    TimingSource,
    VideoInfo,
)
from app.trajectory.exporters import (
    CSV_COLUMNS,
    NUKE_CAMERA_AXES_IN_CPL,
    NUKE_WORLD_FROM_CPL,
    build_analysis_document,
    build_trajectory_document,
    camerapath_pose_to_nuke,
    chan_meta_path,
    cut_times,
    euler_zxy_degrees,
    export_all,
    export_for_nuke_chan,
    horizontal_fov_to_vertical,
    resolve_scale,
    write_analysis_json,
    write_trajectory_csv,
    write_trajectory_json,
)

# Positions chosen so that rounding to three decimals would change the value:
# the third component's tail and the tiny second component both vanish at 3 dp.
FINE_POSITION = [0.12345678901234567, -1.2345678901234567e-05, 9.876543210987654]


# --------------------------------------------------------------------------- fixtures


def make_video_info(**overrides) -> VideoInfo:
    fields = dict(
        path="/tmp/jobs/x/source/clip.mp4",
        filename="clip.mp4",
        size_bytes=1234567,
        duration_seconds=3.0,
        frame_count=72,
        frame_count_is_exact=True,
        width=1920,
        height=1080,
        fps_nominal=24.0,
        fps_average=24.0,
        frame_rate_mode=FrameRateMode.CONSTANT,
        codec_name="h264",
        timing_source=TimingSource.CONTAINER_PTS,
    )
    fields.update(overrides)
    return VideoInfo(**fields)


def make_confidence(level: ConfidenceLevel = ConfidenceLevel.MEDIUM) -> ConfidenceReport:
    return ConfidenceReport(
        level=level,
        score=0.55,
        headline="Rotation is solid; translation is weakly observable.",
        reasons=["mean inlier ratio 0.71", "parallax score 0.22"],
        registered_frame_ratio=0.9,
        mean_inlier_ratio=0.71,
        baseline_parallax_score=0.22,
        translation_observable=True,
        translation_confidence=0.3,
        rotation_confidence=0.8,
    )


def make_trajectory(
    *,
    shot_id: int = 0,
    frame_offset: int = 0,
    time_offset: float = 0.0,
    count: int = 5,
    fps: float = 24.0,
    scale_mode: ScaleMode = ScaleMode.NORMALIZED,
    scale_units: str = "normalized",
    metric_scale_factor: float | None = None,
    with_kinematics: bool = True,
    with_lens: bool = True,
    coordinate_name: str | None = None,
) -> ShotTrajectory:
    """A small but fully populated trajectory.

    Frame 0 is a solved anchor and the rest are interpolated, so tests can check
    that per-frame provenance survives the export (I7).
    """
    dt = 1.0 / fps
    poses: list[CameraPose] = []
    kinematics: list[Kinematics] = []
    lens: list[LensFrame] = []

    for i in range(count):
        index = frame_offset + i
        timestamp = time_offset + i * dt
        yaw = math.radians(3.0 * i)
        poses.append(
            CameraPose(
                frame_index=index,
                timestamp=timestamp,
                position=[FINE_POSITION[0] + 0.25 * i, FINE_POSITION[1], FINE_POSITION[2]],
                quaternion=list(quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), yaw)),
                fov_horizontal=55.0 + 0.5 * i,
                focal_normalized=0.96,
                confidence=0.8 if i == 0 else 0.4,
                solver_source=SolverSource.COLMAP if i == 0 else SolverSource.INTERPOLATED,
                is_anchor=(i == 0),
            )
        )
        if with_kinematics:
            kinematics.append(
                Kinematics(
                    frame_index=index,
                    timestamp=timestamp,
                    linear_velocity=[6.000000000000001, -0.5, 0.125],
                    speed=6.020797289396148,
                    linear_acceleration=[0.1, 0.0, 0.0],
                    angular_velocity=[72.0, -1.5, 0.25],
                    angular_speed=72.016,
                    curvature=0.031,
                )
            )
        if with_lens:
            lens.append(
                LensFrame(
                    frame_index=index,
                    timestamp=timestamp,
                    focal_normalized=0.96,
                    fov_horizontal=55.0 + 0.5 * i,
                    fov_vertical=32.5 + 0.25 * i,
                    confidence=0.5,
                )
            )

    system = CoordinateSystem()
    if coordinate_name is not None:
        system = CoordinateSystem(
            name=coordinate_name,
            notes=f"Origin is the first camera of shot {shot_id}.",
        )

    return ShotTrajectory(
        shot_id=shot_id,
        frame_count=count,
        duration=(count - 1) * dt,
        fps=fps,
        scale_mode=scale_mode,
        scale_units=scale_units,
        metric_scale_factor=metric_scale_factor,
        coordinate_system=system,
        poses=poses,
        kinematics=kinematics,
        lens=lens,
        confidence=make_confidence(),
        classified_moves=[
            ClassifiedMove(
                label=MotionLabel.PAN_RIGHT,
                strength=0.8,
                start_time=time_offset,
                end_time=time_offset + (count - 1) * dt,
                description="Steady pan right.",
            )
        ],
        solver_decisions=[
            SolverDecision(
                solver=SolverSource.COLMAP,
                attempted=True,
                succeeded=True,
                registered_frames=count,
                total_frames=count,
                confidence=0.55,
                selected=True,
                message="registered every keyframe",
            ),
            SolverDecision(
                solver=SolverSource.VGGT,
                attempted=False,
                succeeded=False,
                message="disabled in settings",
            ),
        ],
        pipeline_mode_used=PipelineMode.PHYSICAL_3D,
        total_path_length=1.0,
        total_rotation_deg=12.0,
        summary="Pan right with a slight drift.",
    )


def make_shot(shot_id: int, start_frame: int, end_frame: int, start_time: float,
              end_time: float, confidence: float = 1.0) -> Shot:
    return Shot(
        id=shot_id,
        start_frame=start_frame,
        end_frame=end_frame,
        start_time=start_time,
        end_time=end_time,
        confidence=confidence,
    )


def make_signature() -> MotionSignature:
    return MotionSignature(
        frame_count=5,
        duration=0.1667,
        mean_flow_magnitude=4.2,
        peak_flow_magnitude=9.1,
        mean_inlier_ratio=0.71,
        parallax_score=0.22,
        homography_dominance=0.6,
        rotation_dominance=0.7,
        texture_score=0.5,
        blur_score=0.8,
        jitter_score=0.3,
    )


def make_job(
    trajectories: list[ShotTrajectory],
    *,
    shots: list[Shot] | None = None,
    video: VideoInfo | None = None,
    with_analysis: bool = True,
) -> Job:
    job = Job(id="00000000-0000-4000-8000-000000000000")
    job.video = video if video is not None else make_video_info()
    job.trajectories = list(trajectories)
    job.outputs = JobOutputs(motion_proxy_mp4="/tmp/jobs/x/outputs/motion_proxy.mp4")

    if with_analysis:
        shots = shots if shots is not None else [
            make_shot(t.shot_id, t.poses[0].frame_index if t.poses else 0,
                      t.poses[-1].frame_index if t.poses else 0,
                      t.poses[0].timestamp if t.poses else 0.0,
                      t.poses[-1].timestamp if t.poses else 0.0)
            for t in trajectories
        ]
        job.analysis = AnalysisResult(
            video=job.video,
            shots=shots,
            cut_candidates=[
                CutCandidate(
                    frame_index=s.start_frame,
                    time_seconds=s.start_time,
                    histogram_score=0.42,
                    structural_score=0.61,
                    match_collapse_score=0.88,
                    flow_coherence_score=0.91,
                    combined_score=0.79,
                    accepted=i > 0,
                    reason="cue agreement" if i > 0 else "start of stream",
                )
                for i, s in enumerate(shots)
            ],
            shot_analyses=[
                ShotAnalysis(
                    shot=s,
                    signature=make_signature(),
                    complexity=ShotComplexity.MODERATE,
                    complexity_reasons=["low parallax"],
                    recommended_mode=PipelineMode.PHYSICAL_3D,
                    recommendation_reason="enough texture for SfM",
                )
                for s in shots
            ],
            analysis_resolution=[1280, 720],
            warnings=["Container reported VFR jitter of 0.03."],
        )
    return job


def rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def read_chan(path: Path) -> list[list[str]]:
    return [line.split() for line in path.read_text().splitlines()]


# --------------------------------------------------------------------------- I5


class TestScaleLabelling:
    def test_normalized_is_never_labelled_metres(self, tmp_path):
        traj = make_trajectory()
        job = make_job([traj])

        doc = build_trajectory_document(job, traj)
        assert doc["scale_mode"] == "normalized"
        assert doc["units"] == "normalized"
        assert doc["metric_scale_factor"] is None
        assert doc["coordinate_system"]["position_units"] == "normalized"
        assert doc["linear_velocity_units"] == "normalized/s"
        assert doc["totals"]["path_length_units"] == "normalized"

    def test_scale_mode_round_trips_through_json(self, tmp_path):
        traj = make_trajectory()
        job = make_job([traj])
        path = tmp_path / "trajectory.json"
        write_trajectory_json(path, build_trajectory_document(job, traj))

        reloaded = json.loads(path.read_text())
        assert reloaded["scale_mode"] == traj.scale_mode.value
        assert reloaded["units"] == "normalized"

    def test_metric_needs_a_calibration_factor(self):
        """A metric claim with nothing behind it is downgraded, not published."""
        traj = make_trajectory(scale_mode=ScaleMode.METRIC, scale_units="m")
        resolved = resolve_scale(traj)
        assert resolved.mode is ScaleMode.NORMALIZED
        assert resolved.units == "normalized"
        assert any("metric_scale_factor" in note for note in resolved.notes)

    def test_calibrated_metric_is_labelled_metres(self):
        traj = make_trajectory(
            scale_mode=ScaleMode.METRIC, scale_units="m", metric_scale_factor=1.83
        )
        resolved = resolve_scale(traj)
        assert resolved.mode is ScaleMode.METRIC
        assert resolved.units == "m"
        assert resolved.metric_scale_factor == 1.83

    def test_a_contradictory_units_field_does_not_win(self):
        """The units label is derived from the mode, never copied from a field
        that a caller could have set inconsistently."""
        traj = make_trajectory(scale_mode=ScaleMode.NORMALIZED, scale_units="m")
        resolved = resolve_scale(traj)
        assert resolved.units == "normalized"
        assert any("contradicts" in note for note in resolved.notes)

    def test_every_export_format_carries_the_label(self, tmp_path):
        traj = make_trajectory(scale_units="m")  # deliberately mislabelled input
        job = make_job([traj])
        outputs = export_all(job, [traj], JobPaths(tmp_path).ensure())

        json_doc = json.loads(Path(outputs.trajectory_json).read_text())
        assert (json_doc["scale_mode"], json_doc["units"]) == ("normalized", "normalized")

        rows = list(csv.DictReader(io.StringIO(Path(outputs.trajectory_csv).read_text())))
        assert rows and all(r["units"] == "normalized" for r in rows)
        assert all(r["scale_mode"] == "normalized" for r in rows)

        analysis = json.loads(Path(outputs.analysis_json).read_text())
        assert (analysis["scale_mode"], analysis["units"]) == ("normalized", "normalized")

        index = json.loads((tmp_path / "outputs" / "trajectory_index.json").read_text())
        assert (index["scale_mode"], index["units"]) == ("normalized", "normalized")

        chan_meta = json.loads(
            (tmp_path / "outputs" / "trajectory.chan.meta.json").read_text()
        )
        assert (chan_meta["scale_mode"], chan_meta["units"]) == ("normalized", "normalized")
        assert chan_meta["coordinate_system"]["position_units"] == "normalized"

    def test_job_scale_is_the_most_conservative_across_shots(self):
        metric = make_trajectory(
            shot_id=0, scale_mode=ScaleMode.METRIC, scale_units="m", metric_scale_factor=2.0
        )
        normalized = make_trajectory(shot_id=1)
        job = make_job([metric, normalized])
        doc = build_analysis_document(job)
        assert doc["scale_mode"] == "normalized"
        assert doc["trajectories"][0]["units"] == "m"
        assert doc["trajectories"][1]["units"] == "normalized"


# --------------------------------------------------------------------------- json


class TestTrajectoryDocument:
    def test_contains_every_documented_key(self):
        traj = make_trajectory()
        doc = build_trajectory_document(make_job([traj]), traj)
        for key in ("video", "coordinate_system", "scale_mode", "units", "fps", "frames"):
            assert key in doc, key

        frame = doc["frames"][0]
        for key in (
            "frame",
            "time",
            "position",
            "quaternion",
            "fov",
            "linear_velocity",
            "angular_velocity",
            "confidence",
            "solver_source",
        ):
            assert key in frame, key

    def test_frame_count_matches_pose_count(self):
        traj = make_trajectory(count=9)
        doc = build_trajectory_document(make_job([traj]), traj)
        assert len(doc["frames"]) == len(traj.poses) == 9
        assert doc["solved_frame_count"] == 9

    def test_coordinate_system_fully_describes_the_frame(self):
        """A consumer must be able to rebuild a pose without guessing (I8)."""
        traj = make_trajectory()
        block = build_trajectory_document(make_job([traj]), traj)["coordinate_system"]
        assert block["handedness"] == "right"
        assert block["up_axis"] == "+Z"
        assert block["camera_looks_along"] == "+Y"
        assert block["camera_up"] == "+Z"
        assert block["quaternion_order"] == "wxyz"
        assert block["quaternion_maps"] == "camera_to_world"
        assert block["position_is"] == "camera_centre_in_world"
        assert block["pose_is"] == "optical_camera"
        assert block["angle_units"] == "degrees"

    def test_per_frame_provenance_distinguishes_solved_from_interpolated(self):
        traj = make_trajectory(count=4)
        frames = build_trajectory_document(make_job([traj]), traj)["frames"]
        assert frames[0]["solver_source"] == "colmap"
        assert frames[0]["is_anchor"] is True
        assert frames[0]["confidence"] == pytest.approx(0.8)
        assert [f["solver_source"] for f in frames[1:]] == ["interpolated"] * 3
        assert all(f["is_anchor"] is False for f in frames[1:])

    def test_low_confidence_is_reported_not_smoothed(self):
        traj = make_trajectory()
        traj.confidence = make_confidence(ConfidenceLevel.LOW)
        traj.confidence.score = 0.11
        doc = build_trajectory_document(make_job([traj]), traj)
        assert doc["confidence"]["level"] == "low"
        assert doc["confidence"]["score"] == pytest.approx(0.11)
        assert doc["confidence"]["reasons"]

    def test_timestamps_are_written_through_untouched(self, tmp_path):
        """I1/I2: exported times are the container-derived solve times, exactly."""
        traj = make_trajectory(count=6, fps=23.976023976023978)
        path = tmp_path / "t.json"
        write_trajectory_json(path, build_trajectory_document(make_job([traj]), traj))
        frames = json.loads(path.read_text())["frames"]

        assert [f["time"] for f in frames] == [p.timestamp for p in traj.poses]
        span = frames[-1]["time"] - frames[0]["time"]
        assert abs(span - traj.duration) <= 1.0 / traj.fps

    def test_floats_are_not_quantised(self, tmp_path):
        """Rounding a normalized position to 3 dp is visible in the output."""
        traj = make_trajectory()
        path = tmp_path / "t.json"
        write_trajectory_json(path, build_trajectory_document(make_job([traj]), traj))
        position = json.loads(path.read_text())["frames"][0]["position"]

        assert position[0] == traj.poses[0].position[0]
        assert position[1] == traj.poses[0].position[1]
        assert position[2] == traj.poses[0].position[2]
        # The tiny component would have collapsed to 0.0 under 3-dp rounding.
        assert position[1] != 0.0
        assert repr(position[0]) == repr(FINE_POSITION[0])

    def test_kinematics_absence_is_null_not_a_fabricated_zero(self):
        traj = make_trajectory(with_kinematics=False)
        doc = build_trajectory_document(make_job([traj]), traj)
        assert doc["frames"][0]["linear_velocity"] is None
        assert doc["frames"][0]["angular_velocity"] is None
        assert any("kinematics" in w for w in doc["warnings"])

    def test_angular_velocity_units_are_stated(self):
        traj = make_trajectory()
        doc = build_trajectory_document(make_job([traj]), traj)
        assert doc["angular_velocity_units"] == "deg/s"
        assert doc["frames"][0]["angular_velocity"] == pytest.approx([72.0, -1.5, 0.25])

    def test_video_block_states_the_timing_source(self):
        traj = make_trajectory()
        nominal = make_video_info(timing_source=TimingSource.NOMINAL_FPS)
        doc = build_trajectory_document(make_job([traj], video=nominal), traj)
        assert doc["video"]["timing_source"] == "nominal_fps"

    def test_shot_bounds_come_from_the_job_when_available(self):
        traj = make_trajectory(shot_id=1, frame_offset=36, time_offset=1.5)
        job = make_job([traj], shots=[make_shot(1, 36, 71, 1.5, 3.0, confidence=0.93)])
        doc = build_trajectory_document(job, traj)
        assert doc["shot"]["start_frame"] == 36
        assert doc["shot"]["start_time"] == pytest.approx(1.5)
        assert doc["shot"]["preceding_cut_confidence"] == pytest.approx(0.93)


# --------------------------------------------------------------------------- csv


class TestTrajectoryCsv:
    def test_row_count_is_poses_plus_one_header(self, tmp_path):
        traj = make_trajectory(count=7)
        path = write_trajectory_csv(tmp_path / "t.csv", traj)
        rows = list(csv.reader(io.StringIO(path.read_text())))
        assert len(rows) == len(traj.poses) + 1
        assert rows[0] == list(CSV_COLUMNS)

    def test_values_parse_back_to_the_originals(self, tmp_path):
        traj = make_trajectory(count=5)
        path = write_trajectory_csv(tmp_path / "t.csv", traj)
        rows = list(csv.DictReader(io.StringIO(path.read_text())))

        for row, pose in zip(rows, traj.poses):
            assert int(row["frame"]) == pose.frame_index
            assert float(row["time"]) == pytest.approx(pose.timestamp, abs=1e-9)
            for column, value in zip(("pos_x", "pos_y", "pos_z"), pose.position):
                assert float(row[column]) == pytest.approx(value, abs=1e-9)
                assert float(row[column]) == value  # exact, not merely close
            for column, value in zip(
                ("quat_w", "quat_x", "quat_y", "quat_z"), pose.quaternion
            ):
                assert float(row[column]) == pytest.approx(value, abs=1e-9)
            assert float(row["fov_horizontal_deg"]) == pytest.approx(pose.fov_horizontal)
            assert float(row["confidence"]) == pytest.approx(pose.confidence)
            assert row["solver_source"] == pose.solver_source.value

    def test_kinematics_columns_round_trip(self, tmp_path):
        traj = make_trajectory(count=3)
        path = write_trajectory_csv(tmp_path / "t.csv", traj)
        row = next(csv.DictReader(io.StringIO(path.read_text())))
        kin = traj.kinematics[0]
        assert float(row["vel_x"]) == kin.linear_velocity[0]
        assert float(row["yaw_rate_deg_s"]) == kin.angular_velocity[0]
        assert float(row["speed"]) == kin.speed

    def test_missing_kinematics_leaves_cells_empty(self, tmp_path):
        traj = make_trajectory(count=3, with_kinematics=False)
        path = write_trajectory_csv(tmp_path / "t.csv", traj)
        row = next(csv.DictReader(io.StringIO(path.read_text())))
        assert row["vel_x"] == ""
        assert row["speed"] == ""

    def test_empty_trajectory_writes_only_a_header(self, tmp_path):
        traj = make_trajectory(count=0)
        path = write_trajectory_csv(tmp_path / "t.csv", traj)
        rows = list(csv.reader(io.StringIO(path.read_text())))
        assert rows == [list(CSV_COLUMNS)]


# --------------------------------------------------------------------------- chan


class TestNukeChanConversion:
    def test_identity_pose_is_the_nuke_rest_pose(self):
        """A CameraPath camera at rest (looking +Y, up +Z) is a Nuke camera at
        rest (looking -Z, up +Y), so only the world axis swap shows up."""
        translate, rotate = camerapath_pose_to_nuke([1.0, 2.0, 3.0], quat_identity())
        assert translate == pytest.approx((1.0, 3.0, -2.0))
        assert rotate == pytest.approx((0.0, 0.0, 0.0))

    def test_pan_about_camerapath_up_becomes_nuke_ry(self):
        quat = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(30.0))
        _, (rx, ry, rz) = camerapath_pose_to_nuke([0.0, 0.0, 0.0], quat)
        assert (rx, ry, rz) == pytest.approx((0.0, 30.0, 0.0), abs=1e-9)

    def test_tilt_about_camera_right_becomes_nuke_rx(self):
        quat = quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(30.0))
        _, (rx, ry, rz) = camerapath_pose_to_nuke([0.0, 0.0, 0.0], quat)
        assert (rx, ry, rz) == pytest.approx((30.0, 0.0, 0.0), abs=1e-9)

    def test_roll_about_the_view_axis_flips_sign(self):
        """CameraPath rolls about +Y (the view direction); Nuke rolls about its
        own +Z, which is the *backward* axis — hence the sign change."""
        quat = quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(30.0))
        _, (rx, ry, rz) = camerapath_pose_to_nuke([0.0, 0.0, 0.0], quat)
        assert (rx, ry, rz) == pytest.approx((0.0, 0.0, -30.0), abs=1e-9)

    def test_camera_looking_along_world_x(self):
        """Hand-computed: turning to look along CameraPath +X is a -90 deg pan,
        and in Nuke the camera's -Z must then point along Nuke +X."""
        quat = quat_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(-90.0))
        _, (rx, ry, rz) = camerapath_pose_to_nuke([0.0, 0.0, 0.0], quat)
        assert (rx, ry, rz) == pytest.approx((0.0, -90.0, 0.0), abs=1e-9)

        r_nuke = rot_y(ry) @ rot_x(rx) @ rot_z(rz)
        assert r_nuke @ np.array([0.0, 0.0, -1.0]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
        assert r_nuke @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)

    def test_euler_extraction_matches_the_stated_order(self):
        """Recomposing R = Ry.Rx.Rz from the extracted angles must reproduce the
        Nuke matrix, for arbitrary orientations."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            quat = quat_from_axis_angle(rng.normal(size=3), rng.uniform(-math.pi, math.pi))
            expected = NUKE_WORLD_FROM_CPL @ quat_to_matrix(quat) @ NUKE_CAMERA_AXES_IN_CPL
            rx, ry, rz = euler_zxy_degrees(expected)
            assert rot_y(ry) @ rot_x(rx) @ rot_z(rz) == pytest.approx(expected, abs=1e-9)

    def test_gimbal_lock_stays_finite_and_consistent(self):
        locked = rot_y(40.0) @ rot_x(90.0) @ rot_z(10.0)
        rx, ry, rz = euler_zxy_degrees(locked)
        assert rx == pytest.approx(90.0)
        assert rz == 0.0
        assert ry == pytest.approx(30.0)  # only (ry - rz) is observable
        assert rot_y(ry) @ rot_x(rx) @ rot_z(rz) == pytest.approx(locked, abs=1e-9)

    def test_conversion_matrices_are_proper_rotations(self):
        for matrix in (NUKE_WORLD_FROM_CPL, NUKE_CAMERA_AXES_IN_CPL):
            assert np.linalg.det(matrix) == pytest.approx(1.0)
            assert matrix @ matrix.T == pytest.approx(np.eye(3))


class TestNukeChanFile:
    def test_lines_are_frame_plus_seven_numbers(self, tmp_path):
        traj = make_trajectory(count=4)
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        rows = read_chan(path)
        assert len(rows) == 4
        assert all(len(row) == 8 for row in rows)
        for row in rows:  # strictly numeric: Nuke's reader is not a text parser
            for field in row:
                float(field)

    def test_hand_computed_line(self, tmp_path):
        traj = make_trajectory(count=1)
        traj.poses[0].position = [1.0, 2.0, 3.0]
        traj.poses[0].quaternion = [1.0, 0.0, 0.0, 0.0]
        traj.lens[0].fov_vertical = 31.0
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        assert read_chan(path)[0] == ["1", "1.0", "3.0", "-2.0", "0.0", "0.0", "0.0", "31.0"]

    def test_frame_numbering_starts_at_one_and_keeps_spacing(self, tmp_path):
        traj = make_trajectory(shot_id=1, frame_offset=36, time_offset=1.5, count=3)
        traj.poses[2].frame_index = 40  # a gap, as a dropped-frame solve would give
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        assert [row[0] for row in read_chan(path)] == ["1", "2", "5"]

    def test_vfov_column_uses_measured_vertical_fov(self, tmp_path):
        traj = make_trajectory(count=3)
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        vfovs = [float(row[7]) for row in read_chan(path)]
        assert vfovs == pytest.approx([lens.fov_vertical for lens in traj.lens])

    def test_vfov_is_derived_from_aspect_when_lens_data_is_absent(self, tmp_path):
        traj = make_trajectory(count=2, with_lens=False)
        path = export_for_nuke_chan(tmp_path / "t.chan", traj, aspect_ratio=16 / 9)
        expected = horizontal_fov_to_vertical(traj.poses[0].fov_horizontal, 16 / 9)
        assert float(read_chan(path)[0][7]) == pytest.approx(expected)
        assert expected < traj.poses[0].fov_horizontal

    def test_vfov_column_is_omitted_rather_than_invented(self, tmp_path):
        traj = make_trajectory(count=2, with_lens=False)
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        assert all(len(row) == 7 for row in read_chan(path))
        meta = json.loads(chan_meta_path(path).read_text())
        assert meta["vfov_included"] is False
        assert "vfov_note" in meta

    def test_sidecar_documents_the_conversion(self, tmp_path):
        traj = make_trajectory(count=2)
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        meta = json.loads(chan_meta_path(path).read_text())
        assert meta["rotation_order"] == "ZXY"
        assert meta["columns"][:4] == ["frame", "tx", "ty", "tz"]
        assert meta["coordinate_system"]["up_axis"] == "+Y"
        assert meta["coordinate_system"]["camera_looks_along"] == "-Z"
        assert meta["units"] == "normalized"
        assert meta["angle_units"] == "degrees"

    def test_empty_trajectory_writes_an_empty_file(self, tmp_path):
        traj = make_trajectory(count=0)
        path = export_for_nuke_chan(tmp_path / "t.chan", traj)
        assert path.read_text() == ""
        assert chan_meta_path(path).is_file()


class TestVerticalFov:
    def test_square_frame_is_symmetric(self):
        assert horizontal_fov_to_vertical(60.0, 1.0) == pytest.approx(60.0)

    def test_wider_frame_means_narrower_vertical_fov(self):
        assert horizontal_fov_to_vertical(60.0, 16 / 9) < 60.0

    def test_rejects_a_nonsense_aspect(self):
        with pytest.raises(ValueError):
            horizontal_fov_to_vertical(60.0, 0.0)


# --------------------------------------------------------------------------- analysis


class TestAnalysisDocument:
    def test_records_video_shots_cuts_signatures_and_decisions(self):
        traj = make_trajectory()
        doc = build_analysis_document(make_job([traj]))

        assert doc["video"]["filename"] == "clip.mp4"
        assert doc["video"]["timing_source"] == "container_pts"
        assert doc["analysis_resolution"] == [1280, 720]
        assert doc["shots"][0]["shot"]["id"] == 0
        assert doc["shots"][0]["signature"]["parallax_score"] == pytest.approx(0.22)
        assert doc["shots"][0]["complexity"] == "moderate"
        assert doc["cut_decisions"][0]["combined_score"] == pytest.approx(0.79)
        assert doc["trajectories"][0]["solver_decisions"][0]["solver"] == "colmap"
        assert doc["trajectories"][0]["solver_decisions"][1]["attempted"] is False
        assert doc["trajectories"][0]["confidence"]["level"] == "medium"
        assert "Container reported VFR jitter of 0.03." in doc["warnings"]

    def test_survives_a_job_with_nothing_solved(self):
        job = make_job([], with_analysis=False)
        job.video = None
        doc = build_analysis_document(job)
        assert doc["video"]["available"] is False
        assert doc["shots"] == []
        assert doc["units"] == "normalized"
        assert any("No trajectory" in w for w in doc["warnings"])

    def test_write_is_atomic_and_reloadable(self, tmp_path):
        traj = make_trajectory()
        path = write_analysis_json(tmp_path / "analysis.json", build_analysis_document(make_job([traj])))
        assert json.loads(path.read_text())["schema_version"]
        assert not list(tmp_path.glob(".tmp-*"))


# --------------------------------------------------------------------------- multi-shot


class TestMultiShotExport:
    def build(self, tmp_path):
        first = make_trajectory(
            shot_id=0, frame_offset=0, time_offset=0.0, count=4,
            coordinate_name="camerapath_world_shot_000",
        )
        second = make_trajectory(
            shot_id=1, frame_offset=36, time_offset=1.5, count=5,
            coordinate_name="camerapath_world_shot_001",
        )
        job = make_job(
            [first, second],
            shots=[make_shot(0, 0, 35, 0.0, 1.5), make_shot(1, 36, 71, 1.5, 3.0, 0.93)],
        )
        outputs = export_all(job, [first, second], JobPaths(tmp_path).ensure())
        return job, [first, second], outputs

    def test_one_document_per_shot(self, tmp_path):
        _, trajs, _ = self.build(tmp_path)
        out = tmp_path / "outputs"
        for traj in trajs:
            stem = f"trajectory_shot_{traj.shot_id:03d}"
            assert (out / f"{stem}.json").is_file()
            assert (out / f"{stem}.csv").is_file()
            assert (out / f"{stem}.chan").is_file()
        # No single document pretends to cover both shots.
        assert not (out / "trajectory.json").exists()

    def test_each_document_has_its_own_coordinate_system(self, tmp_path):
        _, _, _ = self.build(tmp_path)
        out = tmp_path / "outputs"
        first = json.loads((out / "trajectory_shot_000.json").read_text())
        second = json.loads((out / "trajectory_shot_001.json").read_text())
        assert first["coordinate_system"]["name"] == "camerapath_world_shot_000"
        assert second["coordinate_system"]["name"] == "camerapath_world_shot_001"
        assert "shot 1" in second["coordinate_system"]["notes"]

    def test_no_document_spans_a_cut(self, tmp_path):
        _, _, _ = self.build(tmp_path)
        out = tmp_path / "outputs"
        first = json.loads((out / "trajectory_shot_000.json").read_text())
        second = json.loads((out / "trajectory_shot_001.json").read_text())
        assert max(f["frame"] for f in first["frames"]) < 36
        assert min(f["frame"] for f in second["frames"]) >= 36
        assert max(f["time"] for f in first["frames"]) < 1.5

    def test_index_records_the_cut_times(self, tmp_path):
        _, _, outputs = self.build(tmp_path)
        index = json.loads(Path(outputs.trajectory_json).read_text())
        assert index["shot_count"] == 2
        assert index["cut_times"] == pytest.approx([1.5])
        assert [s["trajectory_json"] for s in index["shots"]] == [
            "trajectory_shot_000.json",
            "trajectory_shot_001.json",
        ]
        assert [s["trajectory_chan"] for s in index["shots"]] == [
            "trajectory_shot_000.chan",
            "trajectory_shot_001.chan",
        ]
        assert index["shots"][1]["shot"]["start_time"] == pytest.approx(1.5)

    def test_job_outputs_point_at_the_manifest(self, tmp_path):
        _, _, outputs = self.build(tmp_path)
        assert Path(outputs.trajectory_json).name == "trajectory_index.json"
        assert Path(outputs.trajectory_csv).name == "trajectory_shot_000.csv"
        assert Path(outputs.analysis_json).name == "analysis.json"
        # Outputs owned by other stages are carried through untouched.
        assert outputs.motion_proxy_mp4 == "/tmp/jobs/x/outputs/motion_proxy.mp4"

    def test_cut_times_fall_back_to_pose_timestamps(self):
        first = make_trajectory(shot_id=0, count=3)
        second = make_trajectory(shot_id=1, frame_offset=36, time_offset=1.5, count=3)
        job = make_job([first, second], with_analysis=False)
        assert cut_times(job, [first, second]) == pytest.approx([1.5])


class TestSingleShotExport:
    def test_uses_the_documented_file_names(self, tmp_path):
        traj = make_trajectory()
        job = make_job([traj])
        outputs = export_all(job, [traj], JobPaths(tmp_path).ensure())
        out = tmp_path / "outputs"

        assert Path(outputs.trajectory_json) == out / "trajectory.json"
        assert Path(outputs.trajectory_csv) == out / "trajectory.csv"
        assert (out / "trajectory.chan").is_file()
        assert (out / "trajectory.chan.meta.json").is_file()
        index = json.loads((out / "trajectory_index.json").read_text())
        assert index["cut_times"] == []
        assert index["shots"][0]["trajectory_json"] == "trajectory.json"

    def test_documents_and_csv_agree_frame_for_frame(self, tmp_path):
        traj = make_trajectory(count=6)
        job = make_job([traj])
        outputs = export_all(job, [traj], JobPaths(tmp_path).ensure())

        frames = json.loads(Path(outputs.trajectory_json).read_text())["frames"]
        rows = list(csv.DictReader(io.StringIO(Path(outputs.trajectory_csv).read_text())))
        assert len(frames) == len(rows)
        for frame, row in zip(frames, rows):
            assert int(row["frame"]) == frame["frame"]
            assert float(row["pos_x"]) == frame["position"][0]
            assert float(row["time"]) == frame["time"]

    def test_leaves_no_temporary_files_behind(self, tmp_path):
        traj = make_trajectory()
        export_all(make_job([traj]), [traj], JobPaths(tmp_path).ensure())
        assert not list((tmp_path / "outputs").glob(".tmp-*"))


class TestDegradedJobs:
    def test_export_all_with_no_trajectories_does_not_raise(self, tmp_path):
        job = make_job([], with_analysis=False)
        outputs = export_all(job, [], JobPaths(tmp_path).ensure())
        assert outputs.trajectory_json is None
        assert outputs.trajectory_csv is None
        assert Path(outputs.analysis_json).is_file()
        index = json.loads((tmp_path / "outputs" / "trajectory_index.json").read_text())
        assert index["shot_count"] == 0

    def test_export_all_without_a_probe_result(self, tmp_path):
        traj = make_trajectory(with_lens=False)
        job = make_job([traj], with_analysis=False)
        job.video = None
        outputs = export_all(job, [traj], JobPaths(tmp_path).ensure())

        doc = json.loads(Path(outputs.trajectory_json).read_text())
        assert doc["video"]["available"] is False
        assert len(doc["frames"]) == len(traj.poses)
        # No aspect ratio and no lens data: the chan file keeps seven columns.
        rows = read_chan(tmp_path / "outputs" / "trajectory.chan")
        assert all(len(row) == 7 for row in rows)

    def test_empty_trajectory_exports_an_empty_but_valid_document(self, tmp_path):
        traj = make_trajectory(count=0)
        job = make_job([traj], with_analysis=False)
        outputs = export_all(job, [traj], JobPaths(tmp_path).ensure())
        doc = json.loads(Path(outputs.trajectory_json).read_text())
        assert doc["frames"] == []
        assert doc["units"] == "normalized"
        assert any("No poses" in w for w in doc["warnings"])
