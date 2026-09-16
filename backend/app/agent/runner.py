"""Two ways to run a job, behind one protocol.

**Delegated** hands the work to the backend the user is already running on
:8848. **Local** runs it in this process. Both exist because neither is right on
its own, and the difference is visible in the code this repo already has:

  * `Settings.max_concurrent_jobs = 1` is enforced by one `ThreadPoolExecutor`
    and one `Services.running` set *per process*. A second process means two
    concurrent COLMAP runs on one machine — exactly what that setting prevents.
  * `JobStore` is memory-authoritative and only disk-backed as a fallback, so a
    job mutated here is stale in an already-warm web backend and the UI link
    would show an out-of-date job.
  * `EVENT_BUS` is process-local, so a job run here emits no SSE and
    `localhost:5173/?job=<id>` shows no live progress.

Delegating whenever the app is up gives the right behaviour in the common case
(the UI link is a stated product goal). Running locally when it is not keeps the
tool usable for an agent that never opens a browser. The cost is one small HTTP
client and a mode flag that is reported on every status, so an agent is never
guessing which regime it is in.
"""

from __future__ import annotations

import atexit
import os
import signal
import threading
import time
from pathlib import Path
from typing import Literal, Protocol

from app.agent.models import AgentError, SourceMode, Stages
from app.agent.paths import attach_source
from app.config import Settings, get_settings
from app.core.environment import Environment, detect_environment
from app.core.logging import get_logger
from app.core.paths import InvalidJobId, Workspace, validate_job_id
from app.models.schemas.jobs import Job, JobState, JobSummary, SolveSettings

log = get_logger("agent.runner")

ENV_RUNNER = "CPL_AGENT_RUNNER"
ENV_BACKEND_URL = "CPL_AGENT_BACKEND_URL"
ENV_SHUTDOWN_GRACE = "CPL_AGENT_SHUTDOWN_GRACE"

#: Probe budget for "is the dev backend up?". Short on purpose: this runs on
#: every process start, and a slow answer is the same as "no" for a tool the
#: user is waiting on.
HEALTH_TIMEOUT_S = 1.5

TERMINAL_STATES = {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}


def default_backend_url() -> str:
    explicit = os.environ.get(ENV_BACKEND_URL, "").strip()
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get("CPL_PORT", "").strip() or "8848"
    return f"http://127.0.0.1:{port}"


class JobRunner(Protocol):
    def create(self) -> str: ...
    def attach_source(self, job_id: str, src: Path, *, mode: SourceMode) -> tuple[Job, str]: ...
    def start(self, job_id: str, stages: Stages, settings: SolveSettings) -> None: ...
    def render_only(self, job_id: str, settings: SolveSettings | None) -> None: ...
    def get(self, job_id: str) -> Job: ...
    def list(self, limit: int) -> list[JobSummary]: ...
    def cancel(self, job_id: str) -> None: ...
    def delete(self, job_id: str) -> bool: ...
    def environment(self) -> Environment: ...
    def job_dir(self, job_id: str) -> Path: ...
    def is_running(self, job_id: str) -> bool: ...
    @property
    def mode(self) -> Literal["delegated", "local"]: ...


def _checked_job_id(job_id: str) -> str:
    try:
        return validate_job_id(job_id)
    except InvalidJobId as exc:
        raise AgentError(
            "job_not_found",
            f"malformed job id: {job_id!r}",
            detail="A job id is a canonical UUID4 string.",
        ) from exc


# --------------------------------------------------------------------- local


class LocalRunner:
    """In-process execution on the same `Services` the web app would build."""

    mode: Literal["delegated", "local"] = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        from app.api.deps import build_services

        self.settings = settings or get_settings()
        self.services = build_services()
        self._chain_threads: list[threading.Thread] = []
        self._shutdown_done = threading.Event()
        self._install_shutdown_hooks()

    # ----------------------------------------------------------- lifecycle

    def create(self) -> str:
        try:
            self.services.store.gc()
        except Exception as exc:  # noqa: BLE001 - housekeeping never blocks work
            log.warning("workspace gc failed: %s", exc)
        return self.services.store.create().id

    def attach_source(self, job_id: str, src: Path, *, mode: SourceMode) -> tuple[Job, str]:
        job_id = _checked_job_id(job_id)
        paths = self.services.workspace.job(job_id).ensure()
        target, how = attach_source(src, paths.source_dir, mode=mode)

        from app.video.ffprobe import FFprobeError, probe_frames, probe_video

        try:
            info = probe_video(target)
            # Read per-frame timestamps now, the same way the upload endpoint
            # does: this is what upgrades timing to CONTAINER_PTS, makes the
            # frame count exact, and surfaces VFR before a solve is committed.
            _frames, info = probe_frames(info)
        except FFprobeError as exc:
            target.unlink(missing_ok=True)
            raise AgentError(
                "unsupported_media_type",
                f"could not read this file as video: {exc}",
                detail=str(src),
                hint="The extension is supported but the bytes are not decodable video.",
            ) from exc

        def mutate(job: Job) -> None:
            job.video = info
            job.state = JobState.UPLOADED
            job.error = None
            job.error_detail = None
            job.analysis = None
            job.trajectories = []

        return self.services.store.update(job_id, mutate), how

    def start(self, job_id: str, stages: Stages, settings: SolveSettings) -> None:
        job_id = _checked_job_id(job_id)
        if not self.services.mark_running(job_id):
            raise AgentError(
                "job_busy",
                f"job {job_id} is already running.",
                hint="Poll it with wait_for_job, or cancel it first.",
            )

        analysis = self.services.analysis_pipeline()

        def work() -> None:
            try:
                if stages.analyze:
                    analysis.run(job_id, settings)
                    job = self.services.store.get(job_id)
                    if job.state is not JobState.ANALYZED:
                        return  # failed or cancelled; the record already says so
                if stages.solve:
                    solve = self.services.solve_pipeline()
                    solve.run(job_id, settings, render=stages.render)
                elif stages.render:
                    self.services.solve_pipeline().render(job_id)
            except Exception:  # noqa: BLE001 - the pipelines record their own failures
                log.exception("agent job %s failed outside the pipeline's own handling", job_id)
            finally:
                self.services.mark_done(job_id)

        self.services.executor.submit(work)

    def render_only(self, job_id: str, settings: SolveSettings | None) -> None:
        job_id = _checked_job_id(job_id)
        if settings is not None:
            output_fields = (
                "proxy_style", "output_width", "output_height",
                "match_source_aspect", "output_fps", "render_trajectory_preview",
            )

            def apply(job: Job) -> None:
                for name in output_fields:
                    setattr(job.settings, name, getattr(settings, name))

            self.services.store.update(job_id, apply)
        self.start(job_id, Stages(analyze=False, solve=False, render=True),
                   self.services.store.get(job_id).settings)

    # -------------------------------------------------------------- reads

    def get(self, job_id: str) -> Job:
        from app.workers.jobstore import JobNotFound

        job_id = _checked_job_id(job_id)
        try:
            return self.services.store.get(job_id)
        except (JobNotFound, KeyError) as exc:
            raise AgentError("job_not_found", f"no job {job_id}") from exc

    def list(self, limit: int) -> list[JobSummary]:
        return self.services.store.list_summaries(limit=limit)

    def cancel(self, job_id: str) -> None:
        self.get(job_id)
        self.services.store.request_cancel(job_id)

    def delete(self, job_id: str) -> bool:
        self.get(job_id)
        return self.services.store.delete(job_id)

    def environment(self) -> Environment:
        env = detect_environment(self.services.workspace.root)
        self.services.environment = env
        return env

    def job_dir(self, job_id: str) -> Path:
        return self.services.workspace.job(_checked_job_id(job_id)).root

    def is_running(self, job_id: str) -> bool:
        return self.services.is_running(job_id)

    # ----------------------------------------------------------- shutdown

    def _install_shutdown_hooks(self) -> None:
        atexit.register(self.shutdown)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                previous = signal.getsignal(sig)

                def handler(signum, frame, _previous=previous):  # noqa: ANN001
                    self.shutdown()
                    if callable(_previous) and _previous not in (
                        signal.SIG_DFL, signal.SIG_IGN
                    ):
                        _previous(signum, frame)
                    else:
                        raise SystemExit(128 + signum)

                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not the main thread (a test harness, an embedded runner).
                # atexit still covers the ordinary exit path.
                pass

    def shutdown(self) -> None:
        """Leave no job claiming to be running after this process is gone.

        A local solve does not survive the process. The important part is step
        three: without it a job's last persisted state is `solving`, and a later
        poll — from a new session or the CLI — reports a job that runs forever.
        An honest FAILED record naming the stage, with the partial job directory
        intact, is what the agent can actually act on.
        """
        if self._shutdown_done.is_set():
            return
        self._shutdown_done.set()

        try:
            grace = float(os.environ.get(ENV_SHUTDOWN_GRACE, "3.0"))
        except ValueError:
            grace = 3.0

        with self.services.lock:
            running = sorted(self.services.running)
        for job_id in running:
            self.services.store.request_cancel(job_id)

        deadline = time.monotonic() + max(0.0, grace)
        while running and time.monotonic() < deadline:
            time.sleep(0.1)
            still: list[str] = []
            for job_id in running:
                try:
                    if self.services.store.get(job_id).state not in TERMINAL_STATES:
                        still.append(job_id)
                except Exception:  # noqa: BLE001
                    pass
            running = still

        for job_id in running:
            try:
                job = self.services.store.get(job_id)
                stage = job.current_stage.stage.value if job.current_stage else "unknown"

                def fail(j: Job, _stage: str = stage) -> None:
                    j.state = JobState.FAILED
                    j.error = (
                        "the CameraPath Lab agent process exited while this job was "
                        f"running (stage: {_stage}); the partial job directory is intact "
                        "— re-run with cpl_start_camera_motion_recovery on this job id"
                    )
                    j.error_detail = f"interrupted during stage: {_stage}"

                self.services.store.update(job_id, fail)
            except Exception:  # noqa: BLE001 - best effort by definition
                log.warning("could not record the interrupted state of job %s", job_id)

        # A Blender or COLMAP child that outlived the grace window is left
        # alone: killing a process we may not own is worse than letting it
        # finish into a job directory nobody will read.
        try:
            self.services.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------- delegated


class HttpRunner:
    """Delegate to the backend the user already has running."""

    mode: Literal["delegated", "local"] = "delegated"

    def __init__(self, base_url: str | None = None, settings: Settings | None = None) -> None:
        import httpx

        self.settings = settings or get_settings()
        self.base_url = (base_url or default_backend_url()).rstrip("/")
        # Long read timeout: the backend's stage endpoints return immediately,
        # but a multipart upload of a large source does not.
        self._client = httpx.Client(base_url=self.base_url, timeout=httpx.Timeout(600.0, connect=5.0))
        self.workspace = Workspace(self.settings.workspace_dir)
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------ plumbing

    def _request(self, method: str, path: str, **kwargs):
        import httpx

        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise AgentError(
                "backend_unreachable",
                f"could not reach the CameraPath Lab backend at {self.base_url}.",
                detail=f"{type(exc).__name__}: {exc}",
                hint="Start it with ./scripts/dev.sh, or set CPL_AGENT_RUNNER=local.",
            ) from exc
        if response.status_code == 404:
            raise AgentError("job_not_found", _detail(response) or f"not found: {path}")
        if response.status_code == 409:
            raise AgentError("job_not_ready", _detail(response) or "the job is not in a state for that")
        if response.status_code in (413, 415, 422):
            raise AgentError("unsupported_media_type", _detail(response) or "the backend rejected that input")
        if response.status_code >= 400:
            raise AgentError(
                "internal",
                f"backend returned HTTP {response.status_code}",
                detail=_detail(response),
            )
        return response

    # ----------------------------------------------------------- lifecycle

    def create(self) -> str:
        return str(self._request("POST", "/api/jobs").json()["id"])

    def attach_source(self, job_id: str, src: Path, *, mode: SourceMode) -> tuple[Job, str]:
        """Upload through the backend's own ingestion endpoint.

        `source_mode` cannot be honoured here: the backend owns the write, so the
        bytes are streamed and land as a copy. Reported as `"copy"` rather than
        echoing what was asked for — saying "link" when a copy happened would
        make the only observable difference (disk used) a surprise.
        """
        job_id = _checked_job_id(job_id)
        with open(src, "rb") as fh:
            response = self._request(
                "POST",
                f"/api/jobs/{job_id}/video",
                files={"file": (src.name, fh, "application/octet-stream")},
            )
        job = Job.model_validate(response.json())
        self._verify_shared_workspace(job_id, job)
        return job, "copy"

    def _verify_shared_workspace(self, job_id: str, job: Job) -> None:
        """Confirm the backend on that port writes into *this* workspace.

        Delegation reads results straight off disk, so it is only correct when
        the backend is the same checkout. It need not be: anything listening on
        the port answers /api/system/health the same way, and a second clone —
        or a worktree started by `scripts/dev.sh`, which frees the port by
        killing whatever held it — will happily accept the upload and write it
        into a workspace we never look at. Caught here, right after the first
        write, rather than several stages later as a bare ENOENT on a path the
        caller has no reason to recognise.
        """
        local_root = self.job_dir(job_id)
        if local_root.is_dir():
            return
        remote = getattr(job.video, "path", None)
        raise AgentError(
            "backend_unreachable",
            f"the backend at {self.base_url} does not share this workspace.",
            detail=(
                f"it stored the upload at {remote}, "
                f"but this checkout expects {local_root}"
                if remote
                else f"expected the job directory {local_root} to exist after the upload"
            ),
            hint=(
                "Another copy of CameraPath Lab is serving that port. Stop it and start "
                "this one with ./scripts/dev.sh, point CPL_AGENT_BACKEND_URL at the right "
                "port, or set CPL_AGENT_RUNNER=local to run the job in this process."
            ),
        )

    def start(self, job_id: str, stages: Stages, settings: SolveSettings) -> None:
        job_id = _checked_job_id(job_id)
        body = settings.model_dump(mode="json")

        if stages.analyze:
            self._request("POST", f"/api/jobs/{job_id}/analyze", json=body)
            if stages.solve or stages.render:
                self._chain_after_analysis(job_id, stages, body)
            return
        if stages.solve:
            self._request(
                "POST", f"/api/jobs/{job_id}/solve",
                params={"render": str(bool(stages.render)).lower()}, json=body,
            )
            return
        if stages.render:
            self._request("POST", f"/api/jobs/{job_id}/render", json=body)

    def _chain_after_analysis(self, job_id: str, stages: Stages, body: dict) -> None:
        """Start the solve once analysis lands, from one daemon thread.

        The backend has no "analyze then solve" endpoint — the UI drives the two
        stages itself — so the chaining lives here, as a poll on the job's state.
        The thread is a daemon: if this process dies the *backend's* job keeps
        running, and only the automatic hand-off to the solve is lost, which the
        agent can redo with one call.
        """

        def chain() -> None:
            deadline = time.monotonic() + 3600.0
            while time.monotonic() < deadline:
                time.sleep(1.0)
                try:
                    job = self.get(job_id)
                except AgentError:
                    return
                if job.state is JobState.ANALYZED:
                    try:
                        self._request(
                            "POST", f"/api/jobs/{job_id}/solve",
                            params={"render": str(bool(stages.render)).lower()},
                            json=body,
                        )
                    except AgentError as exc:
                        log.warning("delegated solve for %s could not start: %s", job_id, exc)
                    return
                if job.state in TERMINAL_STATES:
                    return

        thread = threading.Thread(target=chain, name=f"cpl-chain-{job_id[:8]}", daemon=True)
        thread.start()
        self._threads.append(thread)

    def render_only(self, job_id: str, settings: SolveSettings | None) -> None:
        job_id = _checked_job_id(job_id)
        body = settings.model_dump(mode="json") if settings is not None else None
        self._request("POST", f"/api/jobs/{job_id}/render", json=body)

    # -------------------------------------------------------------- reads

    def get(self, job_id: str) -> Job:
        job_id = _checked_job_id(job_id)
        return Job.model_validate(self._request("GET", f"/api/jobs/{job_id}").json())

    def list(self, limit: int) -> list[JobSummary]:
        data = self._request("GET", "/api/jobs", params={"limit": limit}).json()
        return [JobSummary.model_validate(row) for row in data]

    def cancel(self, job_id: str) -> None:
        self._request("POST", f"/api/jobs/{_checked_job_id(job_id)}/cancel")

    def delete(self, job_id: str) -> bool:
        data = self._request("DELETE", f"/api/jobs/{_checked_job_id(job_id)}").json()
        return bool(data.get("deleted"))

    def environment(self) -> Environment:
        """Probe this host directly.

        The backend's `/api/system/environment` returns the same facts, but as a
        JSON projection of a dataclass that would have to be reassembled field by
        field. The MCP process runs on the same machine as the backend — that is
        the premise of delegated mode — so probing here is the same measurement
        with none of the reassembly.
        """
        return detect_environment(self.workspace.root)

    def job_dir(self, job_id: str) -> Path:
        return self.workspace.job(_checked_job_id(job_id)).root

    def is_running(self, job_id: str) -> bool:
        state = self.get(job_id).state
        return state in (
            JobState.ANALYZING, JobState.SOLVING, JobState.RENDERING,
        )

    def shutdown(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


def _detail(response) -> str:  # noqa: ANN001
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return (response.text or "")[:300]
    if isinstance(payload, dict) and "detail" in payload:
        return str(payload["detail"])
    return str(payload)[:300]


# ----------------------------------------------------------------- selection


def backend_health(base_url: str, timeout: float = HEALTH_TIMEOUT_S) -> dict | None:
    """`/api/system/health` payload, or None if the backend is not answering."""
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/system/health", timeout=timeout)
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def pick_runner(
    requested: str | None = None,
    *,
    backend_url: str | None = None,
    settings: Settings | None = None,
) -> JobRunner:
    """Choose a runner. `auto` delegates when the app is up, else runs locally."""
    mode = (requested or os.environ.get(ENV_RUNNER, "auto")).strip().lower() or "auto"
    url = backend_url or default_backend_url()

    if mode == "local":
        return LocalRunner(settings)
    if mode == "delegate":
        health = backend_health(url)
        if health is None or not health.get("can_ingest"):
            raise AgentError(
                "backend_unreachable",
                f"CPL_AGENT_RUNNER=delegate but {url} is not answering as a healthy backend.",
                detail=str(health) if health is not None else "no response to /api/system/health",
                hint="Start it with ./scripts/dev.sh, or use CPL_AGENT_RUNNER=auto.",
            )
        return HttpRunner(url, settings)
    if mode != "auto":
        raise AgentError(
            "invalid_input_path",
            f"unknown runner mode {mode!r}.",
            detail="Expected one of: auto, local, delegate.",
        )

    health = backend_health(url)
    if health is not None and health.get("can_ingest"):
        log.info("delegating to the backend at %s", url)
        return HttpRunner(url, settings)
    log.info("no healthy backend at %s; running jobs in this process", url)
    return LocalRunner(settings)
