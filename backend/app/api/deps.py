"""Shared application services, constructed once at startup."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.core.environment import Environment, detect_environment
from app.core.logging import get_logger
from app.core.paths import Workspace
from app.workers.jobstore import JobStore
from app.workers.pipeline import AnalysisPipeline

log = get_logger("api.deps")


@dataclass
class Services:
    settings: Settings
    environment: Environment
    workspace: Workspace
    store: JobStore
    executor: ThreadPoolExecutor
    running: set[str]
    lock: threading.Lock

    def is_running(self, job_id: str) -> bool:
        with self.lock:
            return job_id in self.running

    def mark_running(self, job_id: str) -> bool:
        """Claim the job. False if it is already in flight."""
        with self.lock:
            if job_id in self.running:
                return False
            self.running.add(job_id)
            return True

    def mark_done(self, job_id: str) -> None:
        with self.lock:
            self.running.discard(job_id)

    def analysis_pipeline(self) -> AnalysisPipeline:
        return AnalysisPipeline(self.store, self.workspace, self.environment)

    def solve_pipeline(self):
        # Imported lazily: the solve stack pulls in pycolmap, which the analysis
        # endpoints do not need.
        from app.workers.solve import SolvePipeline
        return SolvePipeline(self.store, self.workspace, self.environment)


_services: Services | None = None


def build_services() -> Services:
    global _services
    settings = get_settings()
    workspace = Workspace(settings.workspace_dir)
    environment = detect_environment(workspace.root)
    store = JobStore(workspace)
    executor = ThreadPoolExecutor(
        max_workers=max(1, settings.max_concurrent_jobs),
        thread_name_prefix="cpl-job",
    )
    _services = Services(
        settings=settings,
        environment=environment,
        workspace=workspace,
        store=store,
        executor=executor,
        running=set(),
        lock=threading.Lock(),
    )
    return _services


def get_services() -> Services:
    if _services is None:
        return build_services()
    return _services


def shutdown_services() -> None:
    global _services
    if _services is not None:
        _services.executor.shutdown(wait=False, cancel_futures=True)
        _services = None
