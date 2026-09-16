"""HttpRunner's one unstated assumption: the backend shares this workspace.

Delegation reads results off local disk, so it is only correct when the backend
answering the port is this checkout. Anything else — a second clone, a worktree
whose `scripts/dev.sh` freed the port by killing ours — accepts the upload and
writes it into a workspace we never read, and every later stage then fails on a
path the caller has no reason to recognise. That mismatch is caught at the first
write instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.models import AgentError
from app.config import get_settings


class _Response:
    """The single httpx response shape HttpRunner.attach_source consumes."""

    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def _job_payload(job_id: str, source_path: str) -> dict:
    """What `POST /api/jobs/{id}/video` returns, built from the real schemas so
    this test cannot drift from the shape the runner actually parses."""
    from app.models.schemas.jobs import Job, JobState

    from .conftest import make_video_info

    video = make_video_info(filename=Path(source_path).name)
    video.path = source_path
    job = Job(id=job_id, state=JobState.UPLOADED, video=video)
    return job.model_dump(mode="json")


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    from app.agent.runner import HttpRunner

    monkeypatch.setenv("CPL_WORKSPACE_DIR", str(tmp_path / "workspace"))
    get_settings.cache_clear()
    try:
        yield HttpRunner("http://127.0.0.1:8848")
    finally:
        get_settings.cache_clear()


def test_attach_rejects_a_backend_writing_to_another_workspace(runner, tmp_path, monkeypatch):
    job_id = "11111111-2222-4333-8444-555555555555"
    foreign = tmp_path / "other-checkout" / "workspace" / "jobs" / job_id / "source" / "clip.mp4"
    monkeypatch.setattr(
        runner, "_request",
        lambda *a, **k: _Response(_job_payload(job_id, str(foreign))),
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\0" * 16)

    with pytest.raises(AgentError) as excinfo:
        runner.attach_source(job_id, source, mode="copy")

    err = excinfo.value
    assert err.code == "backend_unreachable"
    # The message must name both paths: the mismatch is invisible otherwise,
    # since both backends answer /api/system/health identically.
    assert str(foreign) in err.detail
    assert "CPL_AGENT_RUNNER=local" in err.hint


def test_attach_accepts_a_backend_sharing_this_workspace(runner, tmp_path, monkeypatch):
    job_id = "66666666-7777-4888-8999-aaaaaaaaaaaa"
    local_source = runner.job_dir(job_id) / "source"
    local_source.mkdir(parents=True)
    (local_source / "clip.mp4").write_bytes(b"\0" * 16)
    monkeypatch.setattr(
        runner, "_request",
        lambda *a, **k: _Response(_job_payload(job_id, str(local_source / "clip.mp4"))),
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"\0" * 16)

    job, how = runner.attach_source(job_id, source, mode="link")

    assert job.state.value == "uploaded"
    # "link" cannot be honoured over HTTP — the backend owns the write — and the
    # runner must say what actually happened rather than echo the request.
    assert how == "copy"
