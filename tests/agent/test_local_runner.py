"""LocalRunner's shutdown contract.

A local solve does not survive the MCP process. The part that matters is what
the job record says afterwards: without an explicit terminal write, the last
persisted state is `solving`, and a later poll — from a new session or the CLI —
reports a job that runs forever.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.models.schemas.jobs import Job, JobState, Stage, StageProgress


@pytest.fixture()
def local_runner(tmp_path, monkeypatch):
    from app.agent.runner import LocalRunner

    monkeypatch.setenv("CPL_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("CPL_AGENT_SHUTDOWN_GRACE", "0.2")
    get_settings.cache_clear()
    try:
        runner = LocalRunner()
        yield runner
    finally:
        get_settings.cache_clear()


def test_shutdown_records_an_interrupted_job_as_failed(local_runner):
    job_id = local_runner.create()

    def mark(job: Job) -> None:
        job.state = JobState.SOLVING
        job.current_stage = StageProgress(stage=Stage.ESTIMATING_CAMERA_GEOMETRY)

    local_runner.services.store.update(job_id, mark)
    local_runner.services.mark_running(job_id)

    local_runner.shutdown()

    job = local_runner.services.store.get(job_id)
    assert job.state is JobState.FAILED
    assert "exited while this job was running" in job.error
    assert Stage.ESTIMATING_CAMERA_GEOMETRY.value in job.error
    assert Stage.ESTIMATING_CAMERA_GEOMETRY.value in job.error_detail
    # The partial directory is intact, which is what the message promises.
    assert local_runner.job_dir(job_id).is_dir()


def test_shutdown_requests_cancellation_first(local_runner):
    job_id = local_runner.create()
    local_runner.services.mark_running(job_id)
    local_runner.shutdown()
    assert local_runner.services.store.is_cancelled(job_id)


def test_shutdown_leaves_a_finished_job_alone(local_runner):
    job_id = local_runner.create()
    local_runner.services.store.set_state(job_id, JobState.COMPLETE)
    local_runner.shutdown()
    assert local_runner.services.store.get(job_id).state is JobState.COMPLETE


def test_shutdown_is_idempotent(local_runner):
    local_runner.shutdown()
    local_runner.shutdown()


def test_malformed_job_id_is_job_not_found(local_runner):
    from app.agent.models import AgentError

    with pytest.raises(AgentError) as exc:
        local_runner.get("not-a-uuid")
    assert exc.value.code == "job_not_found"
