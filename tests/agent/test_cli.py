"""The JSON CLI: one document on stdout, and an exit code you can branch on."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from app.agent import cli as cli_mod
from app.agent.models import AgentError
from app.models.schemas.jobs import JobState
from tests.agent.conftest import make_job, make_trajectory


@pytest.fixture()
def run_cli(service, fake_runner, monkeypatch):
    """Run `cpl` in-process against the FakeRunner-backed service."""
    import app.agent.runner as runner_mod

    monkeypatch.setattr(runner_mod, "pick_runner", lambda *a, **k: fake_runner)
    monkeypatch.setattr(
        cli_mod, "AgentService", lambda runner, **kwargs: service
    )

    def run(*argv: str) -> tuple[int, str, str]:
        import sys

        out, err = io.StringIO(), io.StringIO()
        real_out, real_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            code = cli_mod.main(list(argv))
        finally:
            sys.stdout, sys.stderr = real_out, real_err
        return code, out.getvalue(), err.getvalue()

    return run


def _only_json(stdout: str) -> dict:
    """Parse stdout and assert it is exactly one JSON document, nothing else."""
    text = stdout.strip()
    assert text, "nothing was written to stdout"
    document = json.loads(text)  # raises if there is trailing prose
    assert stdout.count("\n") == 1, f"expected one line on stdout, got:\n{stdout}"
    return document


# ------------------------------------------------------------------ envelopes


def test_env_emits_one_json_document(run_cli):
    code, out, _err = run_cli("env")
    payload = _only_json(out)
    assert code == 0
    assert payload["ok"] is True
    assert payload["command"] == "env"
    assert payload["result"]["runner_mode"] == "local"


def test_pretty_is_still_valid_json(run_cli):
    code, out, _err = run_cli("--pretty", "env")
    assert code == 0
    assert json.loads(out)["ok"] is True
    assert "\n  " in out  # actually indented


def test_jobs_lists_summaries(run_cli, fake_runner):
    fake_runner.add(make_job())
    code, out, _err = run_cli("jobs", "--limit", "5")
    payload = _only_json(out)
    assert code == 0
    assert len(payload["result"]["jobs"]) == 1


def test_status_motion_describe_and_outputs_all_emit_one_document(run_cli, fake_runner):
    job = fake_runner.add(make_job())
    for argv in (
        ("status", job.id),
        ("motion", job.id),
        ("describe", job.id),
        ("outputs", job.id, "--no-verify"),
        ("samples", job.id, "--shot", "0"),
    ):
        code, out, _err = run_cli(*argv)
        payload = _only_json(out)
        assert code == 0, argv
        assert payload["ok"] is True, argv
        assert payload["job_id"] == job.id, argv


def test_progress_goes_to_stderr_not_stdout(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.COMPLETE))
    code, out, err = run_cli("wait", job.id, "--timeout", "2")
    _only_json(out)
    assert code == 0
    assert "complete" in err


def test_quiet_suppresses_the_stderr_chatter(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.COMPLETE))
    code, out, err = run_cli("--quiet", "wait", job.id, "--timeout", "2")
    _only_json(out)
    assert code == 0
    assert err.strip() == ""


# ----------------------------------------------------------------- exit codes


def test_a_bad_path_exits_2_and_still_prints_json(run_cli):
    code, out, _err = run_cli("run", "/no/such/clip.mp4")
    payload = _only_json(out)
    assert code == 2
    assert payload["ok"] is False
    assert payload["error"]["code"] == "invalid_input_path"


def test_an_unsupported_extension_exits_2(run_cli, tmp_path: Path):
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    code, out, _err = run_cli("run", str(notes))
    assert code == 2
    assert _only_json(out)["error"]["code"] == "unsupported_media_type"


def test_an_unknown_job_exits_4(run_cli):
    code, out, _err = run_cli("status", "11111111-2222-3333-4444-555555555555")
    assert code == 4
    assert _only_json(out)["error"]["code"] == "job_not_found"


def test_an_unsolved_job_exits_9(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.ANALYZING, trajectories=[]))
    code, out, _err = run_cli("motion", job.id)
    assert code == 9
    assert _only_json(out)["error"]["code"] == "job_not_ready"


def test_a_failed_job_exits_3_from_describe(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.FAILED, trajectories=[]))
    job.error = "solve failed"
    code, out, _err = run_cli("describe", job.id)
    assert code == 3
    assert _only_json(out)["error"]["code"] == "job_failed"


def test_a_cancelled_job_exits_6(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.CANCELLED, trajectories=[]))
    code, out, _err = run_cli("motion", job.id)
    assert code == 6
    assert _only_json(out)["error"]["code"] == "job_cancelled"


def test_a_wait_that_times_out_exits_5(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    code, out, _err = run_cli("--quiet", "wait", job.id, "--timeout", "0.5", "--poll", "0.25")
    payload = _only_json(out)
    assert code == 5
    assert payload["ok"] is True  # the wait itself succeeded; the job did not finish
    assert payload["result"]["timed_out"] is True


def test_deleting_a_running_job_exits_9(run_cli, fake_runner):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    code, out, _err = run_cli("rm", job.id)
    assert code == 9
    assert _only_json(out)["error"]["code"] == "job_busy"


def test_an_unexpected_exception_still_prints_json_and_exits_1(run_cli, fake_runner, monkeypatch):
    job = fake_runner.add(make_job())
    from app.agent.service import AgentService

    def boom(*_a, **_k):
        raise ZeroDivisionError("something nobody planned for")

    monkeypatch.setattr(AgentService, "motion", boom)
    code, out, _err = run_cli("motion", job.id)
    payload = _only_json(out)
    assert code == 1
    assert payload["error"]["code"] == "internal"
    assert "ZeroDivisionError" in payload["error"]["message"]


def test_the_exit_code_table_covers_every_error_code():
    from app.agent.models import ERROR_CODES

    assert set(cli_mod.EXIT_CODES) == set(ERROR_CODES)


# ------------------------------------------------------------- shape opt-ins


def test_describe_text_prints_the_bare_text(run_cli, fake_runner):
    job = fake_runner.add(make_job())
    code, out, _err = run_cli("describe", job.id, "--text")
    assert code == 0
    assert out.startswith("Camera ")
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_samples_csv_carries_a_unit_bearing_header(run_cli, fake_runner):
    job = fake_runner.add(make_job(trajectories=[make_trajectory(0, frames=10)]))
    code, out, _err = run_cli("samples", job.id, "--shot", "0", "--csv")
    assert code == 0
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0][0] == "time [s]"
    assert "position [normalized]" in rows[0]
    assert len(rows) == 11  # header + 10 samples


def test_samples_fields_select_columns(run_cli, fake_runner):
    job = fake_runner.add(make_job(trajectories=[make_trajectory(0, frames=6)]))
    code, out, _err = run_cli("samples", job.id, "--shot", "0", "--fields", "time,speed")
    payload = _only_json(out)
    assert code == 0
    assert payload["result"]["columns"] == ["time", "speed"]


# ------------------------------------------------------------- equivalence


@pytest.mark.asyncio
async def test_cli_motion_matches_the_mcp_tools_structured_content(
    run_cli, service, fake_runner
):
    """One service layer means one payload. This is the assertion that keeps it
    that way if either front end grows its own formatting."""
    from app.agent import mcp_server

    job = fake_runner.add(make_job(shots=2))
    code, out, _err = run_cli("motion", job.id)
    assert code == 0
    from_cli = _only_json(out)["result"]

    mcp_server.set_service(service)
    try:
        async with create_connected_server_and_client_session(mcp_server.mcp) as session:
            result = await session.call_tool("get_camera_motion", {"job_id": job.id})
    finally:
        mcp_server.set_service(None)

    assert result.isError is False
    assert from_cli == result.structuredContent


def test_run_requires_a_video_argument(run_cli):
    with pytest.raises(SystemExit) as exc:
        run_cli("run")
    assert exc.value.code == 2
