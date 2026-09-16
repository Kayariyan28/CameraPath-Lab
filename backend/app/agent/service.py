"""`AgentService` — the single API the MCP server and the CLI both call.

Transport-agnostic and synchronous. It owns a `JobRunner`, never imports `mcp`
or `argparse`, and raises exactly one exception type. That last part is the
reason this layer exists at all: an MCP client reads an unhandled traceback as a
broken server and a shell agent reads it as unparseable output, so every failure
has to arrive as a typed, coded value that both front ends can render.

A pipeline failure is *not* one of those. `job.state == FAILED` with `job.error`
set is a result, and it is returned as data in `JobStatus`. Only the two calls
that promise an answer — `motion` and `description` — turn a failed job into an
error, because there is nothing honest for them to return.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from app.agent import outputs as outputs_mod
from app.agent import report as report_mod
from app.agent.describe import describe_job, describe_shot
from app.agent.models import (
    ANALYSIS_ONLY,
    AgentError,
    DeleteResult,
    DescriptionResult,
    DescriptionStyle,
    EnvironmentReport,
    ExportText,
    HostFacts,
    JobListResult,
    JobMotionReport,
    JobRef,
    JobStatus,
    JobSummaryOut,
    OutputsReport,
    ResourcePolicyFacts,
    ShotDescription,
    SourceMode,
    Stages,
    StageFacts,
    TrajectorySamples,
)
from app.agent.paths import allowed_input_roots, resolve_input_video
from app.agent.runner import JobRunner
from app.config import Settings, get_settings
from app.core.logging import EVENT_BUS, StageReporter, get_logger
from app.core.paths import atomic_write_json, read_json
from app.models.schemas.jobs import Job, JobState, SolveSettings
from app.trajectory.exporters import TRAJECTORY_STEM, shot_stem

log = get_logger("agent.service")

ENV_UI_BASE_URL = "CPL_UI_BASE_URL"
DEFAULT_UI_BASE_URL = "http://localhost:5173"

TERMINAL_STATES = {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}

#: States in which a pipeline is actively working. Everything else is a resting
#: state, which is the distinction `terminal` turns on.
BUSY_STATES = {JobState.ANALYZING, JobState.SOLVING, JobState.RENDERING}

#: Where a job records which stages the agent asked for. It lives in the job
#: directory rather than on the `Job` model because the workspace is shared and
#: local in both runner modes, so a later CLI process — or the other runner —
#: can read it without a schema change to a record the web app also writes.
STAGES_FILE = "agent_stages.json"

#: Columns `get_trajectory_samples` can return, and the default selection.
SAMPLE_FIELDS = (
    "time", "position", "quaternion", "fov_horizontal", "focal_normalized",
    "speed", "angular_speed", "curvature", "confidence", "solver_source", "is_anchor",
)
DEFAULT_SAMPLE_FIELDS = (
    "time", "position", "quaternion", "fov_horizontal",
    "speed", "angular_speed", "confidence", "is_anchor",
)

#: Very rough seconds-per-frame for the runtime *hint*. Measured on a mid-tier
#: Apple Silicon machine and labelled as a hint everywhere it surfaces — it is a
#: planning aid for an agent choosing a poll interval, never a measurement.
HINT_SECONDS_PER_FRAME = {"very_large": 0.35, "large": 0.45, "medium": 0.7, "small": 1.1}


def default_ui_base_url() -> str:
    return (os.environ.get(ENV_UI_BASE_URL, "").strip() or DEFAULT_UI_BASE_URL).rstrip("/")


class AgentService:
    def __init__(
        self,
        runner: JobRunner,
        settings: Settings | None = None,
        ui_base_url: str | None = None,
    ) -> None:
        self.runner = runner
        self.settings = settings or get_settings()
        self.ui_base_url = (ui_base_url or default_ui_base_url()).rstrip("/")

    # ------------------------------------------------------------- helpers

    def ui_url(self, job_id: str) -> str:
        return f"{self.ui_base_url}/?job={job_id}"

    def job_dir(self, job_id: str) -> Path:
        return self.runner.job_dir(job_id)

    @property
    def runner_mode(self) -> str:
        return self.runner.mode

    def _record_stages(self, job_id: str, stages: Stages) -> None:
        try:
            atomic_write_json(self.job_dir(job_id) / "cache" / STAGES_FILE, stages.to_dict())
        except Exception as exc:  # noqa: BLE001 - bookkeeping never blocks a start
            log.warning("could not record requested stages for %s: %s", job_id, exc)

    def _requested_stages(self, job_id: str) -> Stages | None:
        """What this job was asked to do, or None for a job the agent did not start.

        None matters: a job created through the web UI has no record here, and
        guessing one would make `terminal` claim a half-driven job had finished.
        """
        raw = read_json(self.job_dir(job_id) / "cache" / STAGES_FILE)
        if not isinstance(raw, dict):
            return None
        try:
            return Stages(
                analyze=bool(raw["analyze"]), solve=bool(raw["solve"]),
                render=bool(raw["render"]),
            )
        except (KeyError, TypeError):
            return None

    @staticmethod
    def _is_terminal(job: Job, stages: Stages | None) -> bool:
        """Has this job stopped for good?

        `complete`, `failed` and `cancelled` always qualify. The subtle cases are
        `analyzed` and `solved`: they are resting states, not end states, and
        whether resting there counts as finished depends entirely on what was
        asked for. A solve run with `render=false` ends at `solved` and never
        reaches `complete`, so without this a caller would poll it forever.
        """
        if job.state in TERMINAL_STATES:
            return True
        if job.state in BUSY_STATES or stages is None:
            return False
        if job.state is JobState.ANALYZED:
            return not (stages.solve or stages.render)
        if job.state is JobState.SOLVED:
            return not stages.render
        return False

    # ----------------------------------------------------------- discovery

    def environment(self) -> EnvironmentReport:
        env = self.runner.environment()
        policy = env.resource_policy()
        backend_url = getattr(self.runner, "base_url", None)
        pycolmap = env.packages.get("pycolmap")
        ffmpeg = env.tools.get("ffmpeg")
        blender = env.tools.get("blender")
        return EnvironmentReport(
            ready=env.ffmpeg_ok and env.ffprobe_ok,
            runner_mode=self.runner.mode,
            backend_url=backend_url,
            ui_base_url=self.ui_base_url,
            workspace_dir=str(Path(self.settings.workspace_dir).resolve()),
            can_ingest=env.ffmpeg_ok and env.ffprobe_ok,
            can_reconstruct_3d=env.colmap_ok,
            can_render=env.blender_ok,
            can_use_vggt=env.torch_ok and env.mps_available,
            blender_version=blender.version if blender and blender.available else None,
            ffmpeg_version=ffmpeg.version if ffmpeg and ffmpeg.available else None,
            pycolmap_version=pycolmap.version if pycolmap and pycolmap.available else None,
            host=HostFacts(
                chip=env.chip,
                platform=env.platform,
                total_memory_gb=round(env.total_memory_gb, 1),
                available_memory_gb=round(env.available_memory_gb, 1),
                cpu_cores_total=env.cpu_cores_total,
                disk_free_gb=round(env.disk_free_gb, 1),
            ),
            resource_policy=ResourcePolicyFacts(
                analysis_long_edge=policy.analysis_long_edge,
                geometry_long_edge=policy.geometry_long_edge,
                colmap_max_num_features=policy.colmap_max_num_features,
                worker_threads=policy.worker_threads,
                reason=policy.reason,
            ),
            warnings=list(env.warnings),
            supported_video_extensions=list(self.settings.allowed_extensions),
            max_input_bytes=int(self.settings.max_upload_bytes),
            input_roots=[str(p) for p in allowed_input_roots()],
        )

    def list_jobs(self, limit: int = 20) -> JobListResult:
        limit = max(1, min(100, int(limit)))
        summaries = self.runner.list(limit)
        return JobListResult(
            jobs=[
                JobSummaryOut(
                    job_id=s.id,
                    state=s.state.value,
                    created_at=s.created_at.isoformat(),
                    filename=s.filename,
                    duration_seconds=s.duration_seconds,
                    shot_count=s.shot_count,
                    ui_url=self.ui_url(s.id),
                    job_dir=str(self.job_dir(s.id)),
                )
                for s in summaries
            ],
            workspace_dir=str(Path(self.settings.workspace_dir).resolve()),
            runner_mode=self.runner.mode,
        )

    # ---------------------------------------------------------------- work

    def create_job_from_video(
        self, video_path: str, *, source_mode: SourceMode = "link"
    ) -> JobRef:
        """Validate, create the job, attach the source. Nothing is started.

        Validation happens before `create()` on purpose: a bad path must not
        leave an orphan job directory for the GC to find later.
        """
        src = resolve_input_video(video_path, self.settings)
        job_id = self.runner.create()
        try:
            job, how = self.runner.attach_source(job_id, src, mode=source_mode)
        except AgentError:
            try:
                self.runner.delete(job_id)
            except Exception:  # noqa: BLE001 - the original error is the useful one
                log.warning("could not clean up job %s after a failed attach", job_id)
            raise

        paths_root = self.job_dir(job_id)
        source = paths_root / "source"
        # Defensive: a runner that reported success without leaving a readable
        # source directory must not surface as a bare OSError from iterdir().
        # Every escape from this layer is an AgentError (see the class docstring).
        attached = None
        if source.is_dir():
            attached = next((p for p in sorted(source.iterdir()) if p.is_file()), None)
        return JobRef(
            job_id=job_id,
            state=job.state.value,
            runner_mode=self.runner.mode,
            ui_url=self.ui_url(job_id),
            job_dir=str(paths_root),
            source_video=str(attached) if attached else str(source / src.name),
            source_attached_by=how,  # type: ignore[arg-type]
            video=report_mod.video_facts(job.video),
            stages_requested={"analyze": False, "solve": False, "render": False},
            estimated_runtime_hint="nothing has been started yet",
        )

    def start(self, job_id: str, *, stages: Stages, settings: SolveSettings) -> JobStatus:
        self._record_stages(job_id, stages)
        self.runner.start(job_id, stages, settings)
        return self.status(job_id)

    def start_recovery(
        self,
        video_path: str,
        *,
        settings: SolveSettings,
        stages: Stages,
        source_mode: SourceMode = "link",
    ) -> JobRef:
        """Create, attach and start in one call — the main agent entrypoint."""
        ref = self.create_job_from_video(video_path, source_mode=source_mode)
        self._record_stages(ref.job_id, stages)
        self.runner.start(ref.job_id, stages, settings)
        job = self.runner.get(ref.job_id)
        return ref.model_copy(
            update={
                "state": job.state.value,
                "stages_requested": stages.to_dict(),
                "estimated_runtime_hint": self._runtime_hint(job, stages),
            }
        )

    def _runtime_hint(self, job: Job, stages: Stages) -> str:
        if job.video is None:
            return "unknown — the source has not been probed"
        if not stages.solve and not stages.render:
            frames = job.video.frame_count
            return f"analysis only: seconds to a minute for {frames} frames (a hint, not a measurement)"
        env = self.runner.environment()
        reason = env.resource_policy().reason
        tier = "medium"
        for name in HINT_SECONDS_PER_FRAME:
            if f"tier={name}" in reason:
                tier = name
                break
        seconds = job.video.frame_count * HINT_SECONDS_PER_FRAME[tier]
        if stages.render:
            seconds *= 1.35
        low, high = seconds * 0.5, seconds * 2.0
        return (
            f"a hint, not a measurement: roughly {low:.0f}-{high:.0f} s for "
            f"{job.video.frame_count} frames on this machine ({reason})"
        )

    # -------------------------------------------------------------- status

    def _job(self, job_id: str) -> Job:
        return self.runner.get(job_id)

    def _progress(self, job: Job) -> float | None:
        """Overall fraction, from the most authoritative source available.

        In local mode the pipeline publishes to this process's `EVENT_BUS`, so
        the exact monotonic fraction `StageReporter._overall_progress` produced
        is available. In delegated mode that bus lives in the other process, and
        the honest fallback is the *floor* of the current stage computed from the
        very same weight table — a lower bound that advances stage by stage,
        never a guess at where inside the stage the work is.
        """
        if job.state in (JobState.COMPLETE, JobState.SOLVED, JobState.ANALYZED):
            # A job resting where it was asked to stop is finished, whatever the
            # stage weights say about the stages it was never asked to run.
            if self._is_terminal(job, self._requested_stages(job.id)):
                return 1.0
        for event in reversed(EVENT_BUS.replay(job.id)):
            if event.progress is not None:
                return float(event.progress)
        if job.current_stage is None:
            return None
        name = job.current_stage.stage.value
        weights = StageReporter.STAGE_WEIGHTS
        if name not in weights:
            return None
        total = sum(weights.values())
        done = 0
        for stage_name, weight in weights.items():
            if stage_name == name:
                break
            done += weight
        return done / total

    def _render_skipped_reason(self, job: Job) -> str | None:
        if job.outputs.motion_proxy_mp4 and Path(job.outputs.motion_proxy_mp4).is_file():
            return None
        if not job.trajectories:
            return None
        env = self.runner.environment()
        if not env.blender_ok:
            blender = env.tools.get("blender")
            return (
                "Blender was not found, so no MP4 was rendered. The trajectory exports "
                "are complete. " + (blender.detail if blender else "")
            ).strip()
        if not job.settings.render_trajectory_preview and job.state is JobState.SOLVED:
            return None
        if job.state is JobState.SOLVED:
            return "the solve was run with render disabled; call render_motion_proxy to produce the MP4"
        return None

    def _next_step(self, job: Job, stages_done: dict[str, bool],
                   requested: Stages | None = None) -> str:
        state = job.state
        if state is JobState.FAILED:
            return "the job failed; read `error`, then start a new job (the partial outputs are still on disk)"
        if state is JobState.CANCELLED:
            return "the job was cancelled; exports written before the cancel are still on disk"
        if state is JobState.COMPLETE:
            return "call describe_camera_motion for prompt text and list_outputs for the MP4 and data files"
        if state is JobState.SOLVED:
            if requested is not None and requested.render:
                return "the render is starting; call wait_for_job again with the same job_id"
            return "call get_camera_motion, or render_motion_proxy to produce the MP4"
        if state is JobState.ANALYZED and not stages_done.get("solve"):
            if requested is not None and not (requested.solve or requested.render):
                return "analysis is done; call get_job_status for shot_count, or start a full solve"
            return "analysis is done; the solve runs next, or call get_job_status again"
        if state in (JobState.ANALYZING, JobState.SOLVING, JobState.RENDERING):
            return "call wait_for_job again with the same job_id"
        if state is JobState.UPLOADED:
            return "call start_camera_motion_recovery, or start_video_analysis, on this job's video"
        return "call get_job_status"

    def _log_tail(self, job_id: str, tail: int) -> list[str]:
        if tail <= 0:
            return []
        path = self.job_dir(job_id) / "job.log"
        if not path.is_file():
            return []
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-min(int(tail), 200):]

    def status(self, job_id: str, *, log_tail: int = 0) -> JobStatus:
        job = self._job(job_id)
        job_dir = self.job_dir(job_id)
        proxy_exists = bool(
            job.outputs.motion_proxy_mp4 and Path(job.outputs.motion_proxy_mp4).is_file()
        )
        stages_done = {
            "analyze": job.analysis is not None,
            "solve": bool(job.trajectories),
            "render": proxy_exists,
        }
        requested = self._requested_stages(job_id)
        stage = None
        if job.current_stage is not None:
            stage = StageFacts(
                name=job.current_stage.stage.value,
                message=job.current_stage.message,
                shot_id=job.current_stage.shot_id,
                started_at=(
                    job.current_stage.started_at.isoformat()
                    if job.current_stage.started_at
                    else None
                ),
            )
        return JobStatus(
            job_id=job.id,
            state=job.state.value,
            terminal=self._is_terminal(job, requested),
            runner_mode=self.runner.mode,
            stage=stage,
            progress=self._progress(job),
            stages_requested=requested.to_dict() if requested else {},
            stages_done=stages_done,
            shot_count=len(job.analysis.shots) if job.analysis else None,
            elapsed_seconds=max(0.0, (job.updated_at - job.created_at).total_seconds()),
            error=job.error,
            error_detail=job.error_detail,
            render_skipped_reason=self._render_skipped_reason(job),
            ui_url=self.ui_url(job.id),
            job_dir=str(job_dir),
            outputs_ready=bool(job.trajectories),
            log_tail=self._log_tail(job_id, log_tail),
            next_step=self._next_step(job, stages_done, requested),
        )

    def wait(
        self,
        job_id: str,
        *,
        timeout_s: float,
        on_progress: Callable[[JobStatus], None] | None = None,
        poll_interval_s: float = 1.0,
    ) -> JobStatus:
        """Block until the job is terminal or the timeout expires.

        A timeout is a normal result, not an error: the caller gets
        `timed_out=True`, `terminal=False` and a `next_step` telling it to call
        again. That is what keeps a minutes-long solve inside the per-call
        timeouts MCP clients impose.
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        started = time.monotonic()
        interval = max(0.25, min(10.0, float(poll_interval_s)))

        status = self.status(job_id)
        last_key = (status.progress, status.stage.name if status.stage else None,
                    status.stage.message if status.stage else "")
        if on_progress is not None:
            on_progress(status)

        while not status.terminal and time.monotonic() < deadline:
            time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
            status = self.status(job_id)
            key = (status.progress, status.stage.name if status.stage else None,
                   status.stage.message if status.stage else "")
            if on_progress is not None and key != last_key:
                on_progress(status)
                last_key = key

        waited = time.monotonic() - started
        timed_out = not status.terminal
        update = {"waited_seconds": round(waited, 3), "timed_out": timed_out}
        if timed_out:
            update["next_step"] = "call wait_for_job again with the same job_id"
        return status.model_copy(update=update)

    def cancel(self, job_id: str) -> JobStatus:
        self.runner.cancel(job_id)
        status = self.status(job_id)
        return status.model_copy(
            update={
                "cancel_requested": True,
                "next_step": (
                    "cancellation takes effect at the next stage or shot boundary; "
                    "exports already written stay on disk. Poll with wait_for_job."
                ),
            }
        )

    def rerender(self, job_id: str, *, output: dict | None = None) -> JobStatus:
        """Re-render from the exported trajectory JSON. No geometry is recomputed.

        `output` carries only output-side overrides; anything absent keeps the
        job's current value, which is why the merge happens against the job's own
        settings rather than against a fresh default object.
        """
        job = self._job(job_id)
        if not job.trajectories:
            raise AgentError(
                "job_not_ready",
                f"job {job_id} has no solved trajectory to render.",
                detail=f"state={job.state.value}",
                hint="Run start_camera_motion_recovery first.",
            )
        env = self.runner.environment()
        if not env.blender_ok:
            blender = env.tools.get("blender")
            raise AgentError(
                "render_unavailable",
                "Blender was not found, so no MP4 can be rendered.",
                detail=blender.detail if blender else "",
                hint="The trajectory exports are complete and usable without it.",
            )
        merged = job.settings.model_copy(deep=True)
        for name, value in (output or {}).items():
            if value is None:
                continue
            if name not in OUTPUT_SETTING_FIELDS:
                raise AgentError(
                    "invalid_input_path",
                    f"{name} is not an output setting.",
                    detail=f"Re-render accepts only: {', '.join(OUTPUT_SETTING_FIELDS)}",
                )
            setattr(merged, name, value)
        self._record_stages(job_id, Stages(analyze=False, solve=False, render=True))
        self.runner.render_only(job_id, merged)
        status = self.status(job_id)
        return status.model_copy(
            update={"next_step": "call wait_for_job again with the same job_id"}
        )

    def delete_job(self, job_id: str) -> DeleteResult:
        job = self._job(job_id)
        if self.runner.is_running(job_id) or job.state in (
            JobState.ANALYZING, JobState.SOLVING, JobState.RENDERING
        ):
            raise AgentError(
                "job_busy",
                f"job {job_id} is running; cancel it before deleting.",
                detail=f"state={job.state.value}",
            )
        root = self.job_dir(job_id)
        freed = 0
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                try:
                    freed += os.path.getsize(os.path.join(dirpath, name))
                except OSError:
                    pass
        deleted = self.runner.delete(job_id)
        return DeleteResult(job_id=job_id, deleted=deleted, freed_bytes=freed if deleted else 0)

    # ------------------------------------------------------------- results

    def _solved_job(self, job_id: str) -> Job:
        """A job with trajectories, or the most specific error explaining why not."""
        job = self._job(job_id)
        if job.trajectories:
            return job
        if job.state is JobState.FAILED:
            raise AgentError(
                "job_failed",
                job.error or "the job failed before producing a trajectory.",
                detail=job.error_detail or "",
                hint="Read get_job_status(log_tail=50) for the stage log.",
            )
        if job.state is JobState.CANCELLED:
            raise AgentError("job_cancelled", f"job {job_id} was cancelled before it solved.")
        raise AgentError(
            "job_not_ready",
            f"job {job_id} has not solved yet (state: {job.state.value}).",
            hint="Poll with wait_for_job until `terminal` is true.",
        )

    def motion(
        self,
        job_id: str,
        *,
        shot_id: int | None = None,
        include_poses: bool = False,
        max_poses: int = 200,
    ) -> JobMotionReport:
        job = self._solved_job(job_id)
        report = report_mod.job_report(
            job,
            ui_url=self.ui_url(job_id),
            job_dir=str(self.job_dir(job_id)),
            shot_id=shot_id,
            include_poses=include_poses,
            max_poses=max(0, min(2000, int(max_poses))),
        )
        if shot_id is not None and not report.shots:
            raise AgentError(
                "job_not_ready",
                f"job {job_id} has no shot {shot_id}.",
                detail=f"solved shots: {[t.shot_id for t in job.trajectories]}",
            )
        return report

    def description(
        self,
        job_id: str,
        *,
        shot_id: int | None = None,
        style: DescriptionStyle = "prompt",
        max_chars: int = 600,
    ) -> DescriptionResult:
        report = self.motion(job_id, shot_id=shot_id)
        max_chars = max(80, min(2000, int(max_chars)))
        text, caveats, derived = describe_job(
            report.shots, style=style, max_chars=max_chars, scale_units=report.scale.units
        )
        levels = [s.confidence.level for s in report.shots]
        order = {"high": 0, "medium": 1, "low": 2}
        worst = max(levels, key=lambda lv: order.get(lv, 0)) if levels else "high"
        return DescriptionResult(
            text=text,
            style=style,
            shot_count=len(report.shots),
            per_shot=[
                ShotDescription(
                    shot_id=s.shot_id,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    text=describe_shot(s, style=style),
                )
                for s in report.shots
            ],
            caveats=caveats,
            translation_observable_all=all(s.translation.observable for s in report.shots),
            confidence_level_min=worst,
            scale_units=report.scale.units,
            derived_from=derived,
        )

    def samples(
        self,
        job_id: str,
        *,
        shot_id: int,
        stride: int = 1,
        max_rows: int = 500,
        fields: Sequence[str] | None = None,
    ) -> TrajectorySamples:
        job = self._solved_job(job_id)
        trajectory = next((t for t in job.trajectories if t.shot_id == shot_id), None)
        if trajectory is None:
            raise AgentError(
                "job_not_ready",
                f"job {job_id} has no shot {shot_id}.",
                detail=f"solved shots: {[t.shot_id for t in job.trajectories]}",
            )

        chosen = list(fields) if fields else list(DEFAULT_SAMPLE_FIELDS)
        unknown = [f for f in chosen if f not in SAMPLE_FIELDS]
        if unknown:
            raise AgentError(
                "invalid_input_path",
                f"unknown sample field(s): {', '.join(unknown)}",
                detail=f"Available: {', '.join(SAMPLE_FIELDS)}",
            )

        stride = max(1, min(100, int(stride)))
        max_rows = max(1, min(2000, int(max_rows)))
        from app.trajectory.exporters import resolve_scale

        units_label = resolve_scale(trajectory).units
        kinematics = {k.frame_index: k for k in trajectory.kinematics}

        rows: list[list] = []
        selected = trajectory.poses[::stride]
        truncated = len(selected) > max_rows
        for pose in selected[:max_rows]:
            kin = kinematics.get(pose.frame_index)
            row: list = []
            for name in chosen:
                if name == "time":
                    row.append(float(pose.timestamp))
                elif name == "position":
                    row.append([float(v) for v in pose.position])
                elif name == "quaternion":
                    row.append([float(v) for v in pose.quaternion])
                elif name == "fov_horizontal":
                    row.append(float(pose.fov_horizontal))
                elif name == "focal_normalized":
                    row.append(float(pose.focal_normalized))
                elif name == "speed":
                    row.append(float(kin.speed) if kin else None)
                elif name == "angular_speed":
                    row.append(float(kin.angular_speed) if kin else None)
                elif name == "curvature":
                    row.append(float(kin.curvature) if kin else None)
                elif name == "confidence":
                    row.append(float(pose.confidence))
                elif name == "solver_source":
                    row.append(pose.solver_source.value)
                elif name == "is_anchor":
                    row.append(bool(pose.is_anchor))
            rows.append(row)

        units = {
            "time": "s",
            "position": units_label,
            "quaternion": "wxyz, camera_to_world",
            "fov_horizontal": "deg",
            "focal_normalized": "focal_px / image_long_edge_px",
            "speed": f"{units_label}/s",
            "angular_speed": "deg/s",
            "curvature": f"1/{units_label}",
            "confidence": "0-1",
            "solver_source": "enum",
            "is_anchor": "bool",
        }

        multi = len(job.trajectories) > 1
        stem = shot_stem(shot_id) if multi else TRAJECTORY_STEM
        source = self.job_dir(job_id) / "outputs" / f"{stem}.json"
        return TrajectorySamples(
            job_id=job_id,
            shot_id=shot_id,
            columns=chosen,
            units={k: v for k, v in units.items() if k in chosen},
            rows=rows,
            row_count=len(rows),
            stride=stride,
            # Strided subsampling only, never resampled onto a uniform grid —
            # the same promise the exporters make about their rows.
            is_subsample=stride > 1 or truncated,
            truncated=truncated,
            coordinate_system=trajectory.coordinate_system.model_dump(mode="json"),
            source_file=str(source) if source.is_file() else None,
        )

    def outputs(self, job_id: str, *, verify_mp4: bool = True) -> OutputsReport:
        job = self._job(job_id)
        paths = _job_paths(self.runner, job_id)
        return outputs_mod.build_outputs_report(
            job,
            paths,
            ui_url=self.ui_url(job_id),
            verify=verify_mp4,
            render_skipped_reason=self._render_skipped_reason(job),
        )

    def read_export(self, job_id: str, filename: str, *, max_bytes: int) -> ExportText:
        self._job(job_id)
        return outputs_mod.read_export(
            job_id, _job_paths(self.runner, job_id), filename, max_bytes=max_bytes
        )


def _job_paths(runner: JobRunner, job_id: str):
    from app.core.paths import JobPaths

    return JobPaths(runner.job_dir(job_id))


# ------------------------------------------------------------------ settings


def build_solve_settings(
    *,
    mode: str = "auto",
    motion_fidelity: str = "exact",
    proxy_style: str = "motion_cage",
    output_width: int = 1920,
    output_height: int = 1080,
    match_source_aspect: bool = False,
    output_fps: float | None = None,
    lens_mode: str = "auto",
    fov_override_degrees: float | None = None,
    keyframe_density: float = 1.0,
    dynamic_rejection_strength: float = 0.5,
    confidence_threshold: float = 0.35,
    scale_calibration: dict | None = None,
) -> SolveSettings:
    """Validated `SolveSettings` from flat tool/CLI parameters.

    Shared so that an MCP tool call and a `cpl` invocation with the same flags
    produce the same settings object — and therefore the same job.
    """
    from app.models.schemas.jobs import (
        LensMode,
        ProxyStyle,
        ScaleCalibration,
        ScaleMode,
    )
    from app.models.schemas.trajectory import MotionFidelity, PipelineMode

    def _enum(enum_cls, value, name: str):
        try:
            return enum_cls(value)
        except ValueError as exc:
            allowed = ", ".join(m.value for m in enum_cls)
            raise AgentError(
                "invalid_input_path",
                f"invalid {name}: {value!r}",
                detail=f"Expected one of: {allowed}",
            ) from exc

    def _even(value: int, name: str, low: int, high: int) -> int:
        value = int(value)
        if not (low <= value <= high):
            raise AgentError(
                "invalid_input_path", f"{name} must be between {low} and {high}, got {value}"
            )
        if value % 2:
            raise AgentError(
                "invalid_input_path",
                f"{name} must be even (H.264 chroma subsampling), got {value}",
            )
        return value

    def _range(value: float, name: str, low: float, high: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not (low <= value <= high):
            raise AgentError(
                "invalid_input_path", f"{name} must be between {low} and {high}, got {value}"
            )
        return value

    calibration = None
    scale_mode = ScaleMode.NORMALIZED
    if scale_calibration:
        try:
            calibration = ScaleCalibration.model_validate(scale_calibration)
        except Exception as exc:  # noqa: BLE001
            raise AgentError(
                "invalid_input_path",
                f"invalid scale_calibration: {exc}",
                detail="Expected {kind: camera_height|point_distance|travel_distance, "
                       "value_meters: > 0, note?: str}",
            ) from exc
        scale_mode = ScaleMode.METRIC

    if output_fps is not None:
        output_fps = _range(output_fps, "output_fps", 1.0, 240.0)
    if fov_override_degrees is not None:
        fov_override_degrees = float(fov_override_degrees)
        if not (1.0 < fov_override_degrees < 179.0):
            raise AgentError(
                "invalid_input_path",
                f"fov_override_degrees must be strictly between 1 and 179, got "
                f"{fov_override_degrees}",
            )

    return SolveSettings(
        mode=_enum(PipelineMode, mode, "mode"),
        motion_fidelity=_enum(MotionFidelity, motion_fidelity, "motion_fidelity"),
        scale_mode=scale_mode,
        scale_calibration=calibration,
        keyframe_density=_range(keyframe_density, "keyframe_density", 0.25, 4.0),
        dynamic_rejection_strength=_range(
            dynamic_rejection_strength, "dynamic_rejection_strength", 0.0, 1.0
        ),
        confidence_threshold=_range(confidence_threshold, "confidence_threshold", 0.0, 1.0),
        lens_mode=_enum(LensMode, lens_mode, "lens_mode"),
        fov_override_degrees=fov_override_degrees,
        proxy_style=_enum(ProxyStyle, proxy_style, "proxy_style"),
        output_width=_even(output_width, "output_width", 160, 7680),
        output_height=_even(output_height, "output_height", 160, 4320),
        match_source_aspect=bool(match_source_aspect),
        output_fps=output_fps,
    )


ANALYSIS_STAGES = ANALYSIS_ONLY

#: The only settings a re-render may change. Geometry settings are excluded by
#: construction: `render_motion_proxy` reads the exported trajectory and must not
#: imply that changing `mode` would re-solve anything.
OUTPUT_SETTING_FIELDS = (
    "proxy_style",
    "output_width",
    "output_height",
    "match_source_aspect",
    "output_fps",
    "render_trajectory_preview",
)
