"""Fixtures for the agent surface.

The unit tests here never run a pipeline. They build `Job` and `ShotTrajectory`
objects directly and hand them to a `FakeRunner`, so the honesty rules — units,
observability, confidence wording — can be tested against inputs that would be
hard or slow to provoke through a real solve, including inputs a real solve
should never produce (a trajectory that claims metres without a calibration).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.agent.models import Stages  # noqa: E402
from app.agent.service import AgentService  # noqa: E402
from app.config import Settings  # noqa: E402
from app.core.environment import detect_environment  # noqa: E402
from app.core.paths import Workspace, new_job_id  # noqa: E402
from app.models.schemas.jobs import (  # noqa: E402
    AnalysisResult,
    Job,
    JobOutputs,
    JobState,
    JobSummary,
    ShotAnalysis,
    SolveSettings,
)
from app.models.schemas.motion import LensFrame, MotionSignature  # noqa: E402
from app.models.schemas.trajectory import (  # noqa: E402
    CameraPose,
    ClassifiedMove,
    ConfidenceLevel,
    ConfidenceReport,
    Kinematics,
    MotionLabel,
    PipelineMode,
    ScaleMode,
    ShotTrajectory,
    SolverDecision,
    SolverSource,
)
from app.models.schemas.video import (  # noqa: E402
    FrameRateMode,
    Shot,
    ShotComplexity,
    TimingSource,
    VideoInfo,
)


def requires_blender() -> bool:
    from app.blender.runner import find_blender

    return find_blender() is not None


# --------------------------------------------------------------------- builders


def make_video_info(*, filename: str = "clip.mp4", frames: int = 60, fps: float = 30.0) -> VideoInfo:
    return VideoInfo(
        path=f"/tmp/{filename}",
        filename=filename,
        size_bytes=1234,
        duration_seconds=frames / fps,
        frame_count=frames,
        frame_count_is_exact=True,
        width=1280,
        height=720,
        fps_nominal=fps,
        fps_average=fps,
        frame_rate_mode=FrameRateMode.CONSTANT,
        fps_jitter=0.0,
        codec_name="h264",
        timing_source=TimingSource.CONTAINER_PTS,
    )


def make_shot(shot_id: int = 0, *, start: float = 0.0, frames: int = 60,
              fps: float = 30.0, confidence: float = 1.0) -> Shot:
    return Shot(
        id=shot_id,
        start_frame=int(start * fps),
        end_frame=int(start * fps) + frames - 1,
        start_time=start,
        end_time=start + frames / fps,
        confidence=confidence,
    )


def make_trajectory(
    shot_id: int = 0,
    *,
    frames: int = 30,
    fps: float = 30.0,
    start: float = 0.0,
    translation_observable: bool = True,
    scale_mode: ScaleMode = ScaleMode.NORMALIZED,
    scale_units: str | None = None,
    metric_scale_factor: float | None = None,
    solver: SolverSource = SolverSource.COLMAP,
    level: ConfidenceLevel = ConfidenceLevel.HIGH,
    moves: list[ClassifiedMove] | None = None,
    fov_start: float = 60.0,
    fov_end: float = 60.0,
    speeds: list[float] | None = None,
    reasons: list[str] | None = None,
) -> ShotTrajectory:
    times = [start + i / fps for i in range(frames)]
    poses = [
        CameraPose(
            frame_index=i,
            timestamp=t,
            position=[float(i) * 0.1, 0.0, 0.0],
            quaternion=[1.0, 0.0, 0.0, 0.0],
            fov_horizontal=fov_start + (fov_end - fov_start) * (i / max(1, frames - 1)),
            focal_normalized=0.8,
            confidence=0.9,
            solver_source=solver,
            is_anchor=(i % 5 == 0),
        )
        for i, t in enumerate(times)
    ]
    if speeds is None:
        speeds = [1.0] * frames
    kinematics = [
        Kinematics(
            frame_index=i,
            timestamp=t,
            linear_velocity=[speeds[i], 0.0, 0.0],
            speed=speeds[i],
            angular_speed=10.0 + i * 0.1,
            curvature=0.01,
        )
        for i, t in enumerate(times)
    ]
    lens = [
        LensFrame(
            frame_index=i,
            timestamp=t,
            focal_normalized=0.8,
            fov_horizontal=poses[i].fov_horizontal,
            fov_vertical=poses[i].fov_horizontal * 0.56,
            confidence=0.8,
            is_estimated=True,
        )
        for i, t in enumerate(times)
    ]
    if moves is None:
        moves = [
            ClassifiedMove(
                label=MotionLabel.PAN_LEFT, strength=0.7,
                start_time=times[0], end_time=times[-1], description="104 deg pan left",
            )
        ]
    return ShotTrajectory(
        shot_id=shot_id,
        frame_count=frames,
        duration=frames / fps,
        fps=fps,
        scale_mode=scale_mode,
        scale_units=scale_units if scale_units is not None else (
            "m" if scale_mode is ScaleMode.METRIC else "normalized"
        ),
        metric_scale_factor=metric_scale_factor,
        poses=poses,
        kinematics=kinematics,
        lens=lens,
        confidence=ConfidenceReport(
            level=level,
            score={"high": 0.85, "medium": 0.6, "low": 0.25}[level.value],
            headline=f"{level.value.upper()} CONFIDENCE",
            reasons=reasons if reasons is not None else ["strong parallax", "many persistent tracks"],
            registered_frame_ratio=0.95,
            persistent_track_count=400,
            mean_inlier_ratio=0.8,
            mean_reprojection_error=0.6,
            baseline_parallax_score=0.5,
            focal_stability=0.9,
            temporal_consistency=0.9,
            translation_observable=translation_observable,
            translation_confidence=0.8 if translation_observable else 0.0,
            rotation_confidence=0.9,
            zoom_confidence=0.5,
        ),
        classified_moves=moves,
        summary="a synthetic trajectory used only by the tests",
        solver_decisions=[
            SolverDecision(
                solver=solver, attempted=True, succeeded=True, selected=True,
                registered_frames=frames, total_frames=frames, duration_seconds=1.0,
                mean_reprojection_error=0.6, confidence=0.85, message="ok",
            ),
            SolverDecision(
                solver=SolverSource.PERCEPTUAL, attempted=False, succeeded=False,
                selected=False, message="not needed",
            ),
        ],
        pipeline_mode_used=PipelineMode.PHYSICAL_3D,
        total_path_length=3.0,
        total_rotation_deg=104.0,
    )


def make_job(
    *,
    job_id: str | None = None,
    state: JobState = JobState.COMPLETE,
    shots: int = 1,
    trajectories: list[ShotTrajectory] | None = None,
    analysis: bool = True,
    recommendation_reason: str = "",
) -> Job:
    info = make_video_info(frames=60 * shots)
    shot_list = [make_shot(i, start=i * 2.0, frames=60) for i in range(shots)]
    trajectories = trajectories if trajectories is not None else [
        make_trajectory(i, start=i * 2.0) for i in range(shots)
    ]
    result = None
    if analysis:
        result = AnalysisResult(
            video=info,
            shots=shot_list,
            shot_analyses=[
                ShotAnalysis(
                    shot=s,
                    signature=MotionSignature(frame_count=60, duration=2.0),
                    complexity=ShotComplexity.MODERATE,
                    recommended_mode=PipelineMode.PHYSICAL_3D,
                    recommendation_reason=recommendation_reason,
                )
                for s in shot_list
            ],
            analysis_resolution=[1280, 720],
            warnings=[],
        )
    return Job(
        id=job_id or new_job_id(),
        state=state,
        video=info,
        settings=SolveSettings(),
        analysis=result,
        trajectories=trajectories,
        outputs=JobOutputs(),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


# -------------------------------------------------------------------- fake runner


class FakeRunner:
    """A `JobRunner` that records calls and serves canned jobs."""

    mode: Literal["delegated", "local"] = "local"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.jobs: dict[str, Job] = {}
        self.calls: list[tuple] = []
        self.running: set[str] = set()
        self.cancelled: set[str] = set()

    def add(self, job: Job) -> Job:
        self.workspace.job(job.id).ensure()
        self.jobs[job.id] = job
        return job

    # --- protocol ---------------------------------------------------------

    def create(self) -> str:
        job_id = new_job_id()
        self.workspace.create(job_id)
        self.jobs[job_id] = Job(id=job_id, state=JobState.CREATED)
        self.calls.append(("create", job_id))
        return job_id

    def attach_source(self, job_id: str, src: Path, *, mode: str = "link"):
        from app.agent.paths import attach_source as do_attach

        target, how = do_attach(src, self.workspace.job(job_id).source_dir, mode=mode)
        job = self.jobs[job_id]
        job.video = make_video_info(filename=target.name)
        job.state = JobState.UPLOADED
        self.calls.append(("attach", job_id, str(target), how))
        return job, how

    def start(self, job_id: str, stages: Stages, settings: SolveSettings) -> None:
        self.calls.append(("start", job_id, stages.to_dict()))
        self.running.add(job_id)

    def render_only(self, job_id: str, settings) -> None:  # noqa: ANN001
        self.calls.append(("render_only", job_id))

    def get(self, job_id: str) -> Job:
        from app.agent.models import AgentError

        job = self.jobs.get(job_id)
        if job is None:
            raise AgentError("job_not_found", f"no job {job_id}")
        return job

    def list(self, limit: int) -> list[JobSummary]:
        return [
            JobSummary(
                id=j.id, state=j.state, created_at=j.created_at,
                filename=j.video.filename if j.video else None,
                duration_seconds=j.video.duration_seconds if j.video else None,
                shot_count=len(j.analysis.shots) if j.analysis else None,
            )
            for j in list(self.jobs.values())[:limit]
        ]

    def cancel(self, job_id: str) -> None:
        self.get(job_id)
        self.cancelled.add(job_id)
        self.calls.append(("cancel", job_id))

    def delete(self, job_id: str) -> bool:
        self.jobs.pop(job_id, None)
        return self.workspace.delete(job_id)

    def environment(self):
        return detect_environment(self.workspace.root)

    def job_dir(self, job_id: str) -> Path:
        return self.workspace.job(job_id).root

    def is_running(self, job_id: str) -> bool:
        return job_id in self.running


# --------------------------------------------------------------------- fixtures


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    return Settings(workspace_dir=tmp_path / "workspace")  # type: ignore[call-arg]


@pytest.fixture()
def tmp_workspace(tmp_settings: Settings) -> Workspace:
    return Workspace(tmp_settings.workspace_dir)


@pytest.fixture()
def fake_runner(tmp_workspace: Workspace) -> FakeRunner:
    return FakeRunner(tmp_workspace)


@pytest.fixture()
def service(fake_runner: FakeRunner, tmp_settings: Settings) -> AgentService:
    return AgentService(fake_runner, tmp_settings, ui_base_url="http://localhost:5173")


@pytest.fixture()
def solved_job(fake_runner: FakeRunner) -> Job:
    return fake_runner.add(make_job())
