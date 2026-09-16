"""AgentService behaviour, against a FakeRunner — no pipeline, no I/O."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from app.agent.models import AgentError, Stages
from app.agent.service import AgentService, build_solve_settings
from app.models.schemas.jobs import JobState
from tests.agent.conftest import FakeRunner, make_job, make_trajectory


def _video(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\0" * 4096)
    return path


# ----------------------------------------------------------------- lifecycle


def test_create_validates_before_creating_a_job(service: AgentService, fake_runner: FakeRunner):
    with pytest.raises(AgentError) as exc:
        service.create_job_from_video("/no/such/file.mp4")
    assert exc.value.code == "invalid_input_path"
    # No orphan job directory was left behind for the GC to find.
    assert fake_runner.calls == []
    assert fake_runner.workspace.list_job_ids() == []


def test_create_attaches_the_source_inside_the_job_directory(
    service: AgentService, fake_runner: FakeRunner, tmp_path: Path
):
    ref = service.create_job_from_video(str(_video(tmp_path)))
    assert ref.source_attached_by in ("link", "copy")
    assert Path(ref.source_video).is_file()
    assert Path(ref.source_video).parent == Path(ref.job_dir) / "source"
    assert ref.ui_url == f"http://localhost:5173/?job={ref.job_id}"
    assert ref.stages_requested == {"analyze": False, "solve": False, "render": False}


def test_start_recovery_records_the_requested_stages(
    service: AgentService, fake_runner: FakeRunner, tmp_path: Path
):
    stages = Stages(analyze=True, solve=True, render=False)
    ref = service.start_recovery(
        str(_video(tmp_path)), settings=build_solve_settings(), stages=stages
    )
    assert ref.stages_requested == stages.to_dict()
    assert ("start", ref.job_id, stages.to_dict()) in fake_runner.calls
    assert service.status(ref.job_id).stages_requested == stages.to_dict()


# -------------------------------------------------------------------- status


def test_terminal_respects_the_stages_that_were_asked_for(
    service: AgentService, fake_runner: FakeRunner, tmp_path: Path
):
    ref = service.start_recovery(
        str(_video(tmp_path)),
        settings=build_solve_settings(),
        stages=Stages(analyze=True, solve=True, render=False),
    )
    job = fake_runner.jobs[ref.job_id]

    job.state = JobState.SOLVING
    assert service.status(ref.job_id).terminal is False

    # `solved` is where a render-less solve rests. Without this it would never
    # be terminal and a caller would poll it forever.
    job.state = JobState.SOLVED
    status = service.status(ref.job_id)
    assert status.terminal is True
    assert status.progress == 1.0


def test_a_render_job_is_not_terminal_at_solved(
    service: AgentService, fake_runner: FakeRunner, tmp_path: Path
):
    ref = service.start_recovery(
        str(_video(tmp_path)),
        settings=build_solve_settings(),
        stages=Stages(analyze=True, solve=True, render=True),
    )
    fake_runner.jobs[ref.job_id].state = JobState.SOLVED
    assert service.status(ref.job_id).terminal is False
    fake_runner.jobs[ref.job_id].state = JobState.COMPLETE
    assert service.status(ref.job_id).terminal is True


def test_a_job_the_agent_did_not_start_is_only_terminal_at_an_end_state(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.SOLVED))
    assert service.status(job.id).terminal is False
    job.state = JobState.COMPLETE
    assert service.status(job.id).terminal is True


def test_a_failed_job_is_returned_as_data_not_as_an_error(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.FAILED, trajectories=[]))
    job.error = "RuntimeError: no frames could be read"
    status = service.status(job.id)
    assert status.terminal is True
    assert status.error == "RuntimeError: no frames could be read"
    assert "failed" in status.next_step


def test_unknown_job_id_raises_job_not_found(service: AgentService):
    with pytest.raises(AgentError) as exc:
        service.status("11111111-2222-3333-4444-555555555555")
    assert exc.value.code == "job_not_found"


# ---------------------------------------------------------------------- wait


def test_wait_returns_as_soon_as_the_job_is_terminal(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))

    def finish() -> None:
        time.sleep(0.3)
        job.state = JobState.COMPLETE

    threading.Thread(target=finish, daemon=True).start()
    status = service.wait(job.id, timeout_s=10.0, poll_interval_s=0.25)
    assert status.terminal is True
    assert status.timed_out is False
    assert status.waited_seconds < 5.0


def test_wait_honours_the_timeout_without_raising(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    status = service.wait(job.id, timeout_s=0.5, poll_interval_s=0.25)
    assert status.timed_out is True
    assert status.terminal is False
    assert "wait_for_job again" in status.next_step


def test_wait_calls_on_progress_only_when_something_changed(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    seen: list[str] = []

    def record(status) -> None:
        seen.append(status.state)

    def advance() -> None:
        time.sleep(0.3)
        job.state = JobState.COMPLETE

    threading.Thread(target=advance, daemon=True).start()
    service.wait(job.id, timeout_s=5.0, on_progress=record, poll_interval_s=0.1)
    # The first call is the initial snapshot; the rest are changes only.
    assert seen[0] == "solving"
    assert seen.count("solving") == 1


# ------------------------------------------------------------------- results


def test_motion_is_not_ready_before_a_solve(service: AgentService, fake_runner: FakeRunner):
    job = fake_runner.add(make_job(state=JobState.ANALYZING, trajectories=[]))
    with pytest.raises(AgentError) as exc:
        service.motion(job.id)
    assert exc.value.code == "job_not_ready"
    assert "wait_for_job" in exc.value.hint


def test_motion_on_a_failed_job_without_trajectories_raises_job_failed(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.FAILED, trajectories=[]))
    job.error = "solve failed"
    with pytest.raises(AgentError) as exc:
        service.description(job.id)
    assert exc.value.code == "job_failed"
    assert exc.value.message == "solve failed"


def test_motion_on_a_failed_job_that_still_has_trajectories_is_served(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.FAILED))
    job.error = "render failed (trajectory exports are still available)"
    assert service.motion(job.id).shot_count == 1


def test_motion_on_a_cancelled_job_raises_job_cancelled(
    service: AgentService, fake_runner: FakeRunner
):
    job = fake_runner.add(make_job(state=JobState.CANCELLED, trajectories=[]))
    with pytest.raises(AgentError) as exc:
        service.motion(job.id)
    assert exc.value.code == "job_cancelled"


def test_reads_do_not_mutate_the_job(service: AgentService, solved_job):
    before = solved_job.model_dump_json()
    service.motion(solved_job.id)
    service.description(solved_job.id)
    service.outputs(solved_job.id)
    service.samples(solved_job.id, shot_id=0)
    assert solved_job.model_dump_json() == before


def test_samples_are_strided_and_labelled(service: AgentService, fake_runner: FakeRunner):
    job = fake_runner.add(make_job(trajectories=[make_trajectory(0, frames=100)]))
    samples = service.samples(job.id, shot_id=0, stride=4, max_rows=10)
    assert samples.stride == 4
    assert samples.row_count == 10
    assert samples.truncated is True
    assert samples.is_subsample is True
    assert samples.units["time"] == "s"
    assert samples.units["position"] == "normalized"
    assert samples.units["speed"] == "normalized/s"
    assert samples.units["quaternion"] == "wxyz, camera_to_world"


def test_samples_reject_an_unknown_field(service: AgentService, solved_job):
    with pytest.raises(AgentError) as exc:
        service.samples(solved_job.id, shot_id=0, fields=["time", "altitude"])
    assert exc.value.code == "invalid_input_path"
    assert "altitude" in exc.value.message


def test_samples_reject_an_unknown_shot(service: AgentService, solved_job):
    with pytest.raises(AgentError) as exc:
        service.samples(solved_job.id, shot_id=7)
    assert exc.value.code == "job_not_ready"


# ------------------------------------------------------------------ control


def test_cancel_is_idempotent(service: AgentService, fake_runner: FakeRunner):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    first = service.cancel(job.id)
    second = service.cancel(job.id)
    assert first.cancel_requested and second.cancel_requested
    assert fake_runner.cancelled == {job.id}


def test_rerender_requires_a_solved_trajectory(service: AgentService, fake_runner: FakeRunner):
    job = fake_runner.add(make_job(state=JobState.ANALYZED, trajectories=[]))
    with pytest.raises(AgentError) as exc:
        service.rerender(job.id)
    assert exc.value.code == "job_not_ready"


def test_rerender_rejects_a_non_output_setting(service: AgentService, solved_job):
    env = service.runner.environment()
    if not env.blender_ok:
        pytest.skip("Blender not installed; render_unavailable is raised first")
    with pytest.raises(AgentError) as exc:
        service.rerender(solved_job.id, output={"mode": "fast"})
    assert exc.value.code == "invalid_input_path"


def test_delete_refuses_a_running_job(service: AgentService, fake_runner: FakeRunner):
    job = fake_runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    with pytest.raises(AgentError) as exc:
        service.delete_job(job.id)
    assert exc.value.code == "job_busy"


def test_delete_removes_the_directory(service: AgentService, solved_job, fake_runner):
    result = service.delete_job(solved_job.id)
    assert result.deleted is True
    assert not (fake_runner.workspace.jobs_root / solved_job.id).exists()


# ------------------------------------------------------------------ settings


def test_build_solve_settings_rejects_an_odd_output_width():
    with pytest.raises(AgentError) as exc:
        build_solve_settings(output_width=1921)
    assert "even" in exc.value.message


def test_build_solve_settings_rejects_an_unknown_mode():
    with pytest.raises(AgentError) as exc:
        build_solve_settings(mode="telepathy")
    assert exc.value.code == "invalid_input_path"
    assert "physical_3d" in exc.value.detail


def test_a_scale_calibration_switches_the_job_to_metric_mode():
    settings = build_solve_settings(
        scale_calibration={"kind": "travel_distance", "value_meters": 3.5}
    )
    assert settings.scale_mode.value == "metric"
    assert settings.scale_calibration.value_meters == pytest.approx(3.5)


def test_no_calibration_leaves_the_job_normalized():
    assert build_solve_settings().scale_mode.value == "normalized"
