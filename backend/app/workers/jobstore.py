"""Job registry: in-memory state of record, mirrored to disk.

Jobs live in memory for speed and are written to `job.json` after every
mutation so a backend restart does not lose completed work. Disk is the
fallback, memory is authoritative while the process lives.

Mutation goes through `update()` rather than letting callers touch the model, so
there is exactly one place that bumps `updated_at`, persists, and publishes the
state change to subscribers.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone

from app.core.logging import EVENT_BUS, JobEvent, get_logger
from app.core.paths import Workspace, atomic_write_text, read_json
from app.models.schemas.jobs import Job, JobState, JobSummary

log = get_logger("workers.jobstore")


class JobNotFound(KeyError):
    pass


class JobStore:
    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        self._cancelled: set[str] = set()

    # ------------------------------------------------------------- lifecycle

    def create(self) -> Job:
        job_id, _ = self.workspace.create()
        job = Job(id=job_id, state=JobState.CREATED)
        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)
        log.info("created job %s", job_id)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job
        job = self._load_from_disk(job_id)
        if job is None:
            raise JobNotFound(job_id)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def exists(self, job_id: str) -> bool:
        try:
            self.get(job_id)
            return True
        except (JobNotFound, Exception):
            return False

    def update(self, job_id: str, mutate: Callable[[Job], None]) -> Job:
        """Apply `mutate` under the lock, then persist and announce."""
        with self._lock:
            job = self.get(job_id)
            previous_state = job.state
            mutate(job)
            job.updated_at = datetime.now(timezone.utc)
            self._jobs[job_id] = job
            state_changed = job.state != previous_state
            snapshot = job
        self._persist(snapshot)
        if state_changed:
            EVENT_BUS.publish(
                JobEvent(job_id, "state", message=snapshot.state.value,
                         data={"state": snapshot.state.value})
            )
        return snapshot

    def set_state(self, job_id: str, state: JobState, error: str | None = None) -> Job:
        def mutate(job: Job) -> None:
            job.state = state
            if error is not None:
                job.error = error
        return self.update(job_id, mutate)

    def append_log(self, job_id: str, line: str) -> None:
        def mutate(job: Job) -> None:
            job.log.append(line)
            # Keep the in-model log bounded; job.log on disk is the full record.
            if len(job.log) > 400:
                del job.log[:-400]
        self.update(job_id, mutate)

    def list_summaries(self, limit: int = 50) -> list[JobSummary]:
        out: list[JobSummary] = []
        for job_id in self.workspace.list_job_ids()[:limit]:
            try:
                job = self.get(job_id)
            except JobNotFound:
                continue
            out.append(
                JobSummary(
                    id=job.id,
                    state=job.state,
                    created_at=job.created_at,
                    filename=job.video.filename if job.video else None,
                    duration_seconds=job.video.duration_seconds if job.video else None,
                    shot_count=len(job.analysis.shots) if job.analysis else None,
                )
            )
        return out

    def delete(self, job_id: str) -> bool:
        with self._lock:
            self._jobs.pop(job_id, None)
            self._cancelled.discard(job_id)
        EVENT_BUS.clear(job_id)
        return self.workspace.delete(job_id)

    # ----------------------------------------------------------- cancellation

    def request_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancelled.add(job_id)
        log.info("cancellation requested for %s", job_id)

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._cancelled

    def clear_cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancelled.discard(job_id)

    # ------------------------------------------------------------ persistence

    def _persist(self, job: Job) -> None:
        try:
            paths = self.workspace.job(job.id)
            paths.root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(paths.job_file, job.model_dump_json(indent=2))
        except Exception as exc:  # noqa: BLE001 - persistence must never kill a job
            log.warning("could not persist job %s: %s", job.id, exc)

    def _load_from_disk(self, job_id: str) -> Job | None:
        try:
            paths = self.workspace.job(job_id)
        except Exception:
            return None
        if not paths.job_file.is_file():
            return None
        raw = read_json(paths.job_file)
        if raw is None:
            log.warning("job %s has an unreadable job.json", job_id)
            return None
        try:
            return Job.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - a stale schema is not fatal
            log.warning("job %s failed validation (schema drift?): %s", job_id, exc)
            return None

    def gc(self) -> list[str]:
        from app.config import get_settings
        s = get_settings()
        removed = self.workspace.gc(
            max_age_hours=s.job_retention_hours,
            max_total_gb=s.job_workspace_budget_gb,
            keep_min=s.keep_min_jobs,
        )
        with self._lock:
            for job_id in removed:
                self._jobs.pop(job_id, None)
        if removed:
            log.info("garbage-collected %d job(s)", len(removed))
        return removed
