"""The MCP surface, driven through the SDK's own in-memory transport.

No subprocess and no sockets: `create_connected_server_and_client_session` wires
a real `ClientSession` to the real `FastMCP` instance, so these tests exercise
the actual protocol — schemas, structured content, error flags — rather than
calling the tool functions directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import AnyUrl

from app.agent import mcp_server
from app.agent.models import (
    DescriptionResult,
    EnvironmentReport,
    JobListResult,
    JobMotionReport,
    JobRef,
    JobStatus,
    OutputsReport,
    TrajectorySamples,
)
from app.models.schemas.jobs import JobState
from tests.agent.conftest import make_job, make_trajectory

pytestmark = pytest.mark.mcp

EXPECTED_TOOLS = {
    "check_environment",
    "start_camera_motion_recovery",
    "start_video_analysis",
    "get_job_status",
    "wait_for_job",
    "cancel_job",
    "get_camera_motion",
    "describe_camera_motion",
    "list_outputs",
    "get_trajectory_samples",
    "render_motion_proxy",
    "list_jobs",
    "delete_job",
}

#: (tool, required params, {param: default}) straight from the spec.
TOOL_CONTRACT = [
    ("check_environment", set(), {}),
    (
        "start_camera_motion_recovery",
        {"video_path"},
        {
            "mode": "auto", "motion_fidelity": "exact", "render": True,
            "proxy_style": "motion_cage", "output_width": 1920, "output_height": 1080,
            "match_source_aspect": False, "output_fps": None, "lens_mode": "auto",
            "fov_override_degrees": None, "keyframe_density": 1.0,
            "dynamic_rejection_strength": 0.5, "confidence_threshold": 0.35,
            "scale_calibration": None, "source_mode": "link",
        },
    ),
    (
        "start_video_analysis",
        {"video_path"},
        {"mode": "auto", "dynamic_rejection_strength": 0.5, "source_mode": "link"},
    ),
    ("get_job_status", {"job_id"}, {"log_tail": 0}),
    ("wait_for_job", {"job_id"}, {"timeout_seconds": 60.0, "poll_interval_seconds": 1.0}),
    ("cancel_job", {"job_id"}, {}),
    (
        "get_camera_motion",
        {"job_id"},
        {"shot_id": None, "include_poses": False, "max_poses": 200},
    ),
    (
        "describe_camera_motion",
        {"job_id"},
        {"shot_id": None, "style": "prompt", "max_chars": 600},
    ),
    ("list_outputs", {"job_id"}, {"verify_mp4": True}),
    (
        "get_trajectory_samples",
        {"job_id", "shot_id"},
        {"stride": 1, "max_rows": 500, "fields": None},
    ),
    (
        "render_motion_proxy",
        {"job_id"},
        {
            "proxy_style": None, "output_width": None, "output_height": None,
            "match_source_aspect": None, "output_fps": None,
            "render_trajectory_preview": None,
        },
    ),
    ("list_jobs", set(), {"limit": 20}),
    ("delete_job", {"job_id"}, {}),
]


@pytest.fixture()
def wired(service, fake_runner):
    """Point the module-level server at a FakeRunner-backed service."""
    mcp_server.set_service(service)
    try:
        yield service, fake_runner
    finally:
        mcp_server.set_service(None)


def _session():
    return create_connected_server_and_client_session(mcp_server.mcp)


def _structured(result):
    assert result.isError is False, result.content
    assert result.structuredContent is not None
    return result.structuredContent


def _default(schema: dict, name: str):
    return schema["properties"][name].get("default")


# ------------------------------------------------------------------ discovery


@pytest.mark.asyncio
async def test_tool_names_are_exactly_the_documented_set(wired):
    async with _session() as session:
        tools = (await session.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_every_tool_has_a_description_and_an_output_schema(wired):
    async with _session() as session:
        tools = (await session.list_tools()).tools
    for tool in tools:
        assert tool.description and tool.description.strip(), tool.name
        assert tool.inputSchema["type"] == "object", tool.name
        # A declared return model is what makes FastMCP emit structuredContent.
        assert tool.outputSchema is not None, tool.name


@pytest.mark.asyncio
@pytest.mark.parametrize("name,required,defaults", TOOL_CONTRACT, ids=[t[0] for t in TOOL_CONTRACT])
async def test_tool_schema_matches_the_spec(wired, name, required, defaults):
    async with _session() as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
    schema = tools[name].inputSchema
    assert set(schema.get("required", [])) == required, name
    for param, expected in defaults.items():
        assert param in schema["properties"], f"{name}.{param}"
        assert _default(schema, param) == expected, f"{name}.{param}"
    # `ctx` is injected by the framework and must never reach the wire.
    assert "ctx" not in schema["properties"], name


@pytest.mark.asyncio
async def test_resources_and_templates_are_the_documented_uris(wired):
    async with _session() as session:
        resources = {str(r.uri) for r in (await session.list_resources()).resources}
        templates = {
            t.uriTemplate for t in (await session.list_resource_templates()).resourceTemplates
        }
    assert resources == {
        "camerapath://conventions", "camerapath://environment", "camerapath://jobs",
    }
    assert templates == {
        "camerapath://jobs/{job_id}/motion",
        "camerapath://jobs/{job_id}/description",
        "camerapath://jobs/{job_id}/outputs/{filename}",
    }


@pytest.mark.asyncio
async def test_conventions_resource_states_the_coordinate_contract(wired):
    async with _session() as session:
        result = await session.read_resource(AnyUrl("camerapath://conventions"))
    text = result.contents[0].text
    assert result.contents[0].mimeType == "text/markdown"
    for token in ("+Z", "wxyz", "camera_to_world", "normalized", "observable"):
        assert token in text, token


@pytest.mark.asyncio
async def test_server_instructions_state_the_units_rule(wired):
    async with _session() as session:
        init = session  # instructions are captured at initialize
        assert init is not None
    assert "normalized units, not metres" in mcp_server.INSTRUCTIONS
    assert "translation.observable" in mcp_server.INSTRUCTIONS


# ---------------------------------------------------------------------- calls


@pytest.mark.asyncio
async def test_check_environment_returns_a_valid_report(wired):
    async with _session() as session:
        payload = _structured(await session.call_tool("check_environment", {}))
    report = EnvironmentReport.model_validate(payload)
    assert report.runner_mode in ("local", "delegated")
    assert Path(report.workspace_dir).is_absolute()


@pytest.mark.asyncio
async def test_get_camera_motion_round_trips_and_keeps_paths_inside_the_job(wired):
    service, runner = wired
    job = runner.add(make_job())
    async with _session() as session:
        payload = _structured(
            await session.call_tool("get_camera_motion", {"job_id": job.id})
        )
    report = JobMotionReport.model_validate(payload)
    assert report.job_id == job.id
    assert report.ui_url == f"http://localhost:5173/?job={job.id}"
    assert Path(report.job_dir).is_absolute()
    assert Path(report.job_dir) == runner.job_dir(job.id)
    assert report.scale.units == "normalized"
    assert report.shots[0].solver_used == "colmap"


@pytest.mark.asyncio
async def test_describe_camera_motion_returns_text_and_caveats(wired):
    service, runner = wired
    job = runner.add(make_job())
    async with _session() as session:
        payload = _structured(
            await session.call_tool("describe_camera_motion", {"job_id": job.id})
        )
    result = DescriptionResult.model_validate(payload)
    assert result.text
    assert result.style == "prompt"
    assert result.scale_units == "normalized"
    assert result.derived_from


@pytest.mark.asyncio
async def test_list_outputs_paths_are_absolute_and_contained(wired):
    service, runner = wired
    job = runner.add(make_job())
    outputs_dir = runner.job_dir(job.id) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "trajectory.json").write_text("{}")
    async with _session() as session:
        payload = _structured(
            await session.call_tool("list_outputs", {"job_id": job.id, "verify_mp4": False})
        )
    report = OutputsReport.model_validate(payload)
    assert Path(report.outputs_dir) == outputs_dir
    assert report.trajectory_json.exists is True
    assert Path(report.trajectory_json.path).parent == outputs_dir
    assert report.motion_proxy_mp4.exists is False
    assert report.motion_proxy_mp4.frame_count is None


@pytest.mark.asyncio
async def test_get_trajectory_samples_labels_every_column(wired):
    service, runner = wired
    job = runner.add(make_job(trajectories=[make_trajectory(0, frames=50)]))
    async with _session() as session:
        payload = _structured(
            await session.call_tool(
                "get_trajectory_samples",
                {"job_id": job.id, "shot_id": 0, "stride": 5, "max_rows": 5},
            )
        )
    samples = TrajectorySamples.model_validate(payload)
    assert samples.row_count == 5
    assert set(samples.units) == set(samples.columns)
    assert samples.units["position"] == "normalized"


@pytest.mark.asyncio
async def test_list_jobs_returns_summaries(wired):
    service, runner = wired
    runner.add(make_job())
    async with _session() as session:
        payload = _structured(await session.call_tool("list_jobs", {"limit": 5}))
    listing = JobListResult.model_validate(payload)
    assert len(listing.jobs) == 1
    assert listing.jobs[0].ui_url.endswith(listing.jobs[0].job_id)


@pytest.mark.asyncio
async def test_start_camera_motion_recovery_returns_a_job_ref(wired, tmp_path):
    service, runner = wired
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\0" * 2048)
    async with _session() as session:
        payload = _structured(
            await session.call_tool(
                "start_camera_motion_recovery",
                {"video_path": str(video), "render": False},
            )
        )
    ref = JobRef.model_validate(payload)
    assert ref.stages_requested == {"analyze": True, "solve": True, "render": False}
    assert Path(ref.source_video).is_file()
    assert ref.poll_with == "wait_for_job"
    assert "hint" in ref.estimated_runtime_hint


# --------------------------------------------------------------------- errors


@pytest.mark.asyncio
async def test_unknown_job_id_is_a_tool_error_not_a_traceback(wired):
    async with _session() as session:
        result = await session.call_tool(
            "get_job_status", {"job_id": "11111111-2222-3333-4444-555555555555"}
        )
    assert result.isError is True
    text = result.content[0].text
    assert "job_not_found" in text
    assert "Traceback" not in text


@pytest.mark.asyncio
async def test_malformed_job_id_is_rejected_before_the_filesystem(wired):
    async with _session() as session:
        result = await session.call_tool("get_job_status", {"job_id": "../../etc/passwd"})
    assert result.isError is True
    assert "job_not_found" in result.content[0].text


@pytest.mark.asyncio
async def test_a_bad_video_path_is_a_tool_error(wired):
    async with _session() as session:
        result = await session.call_tool(
            "start_camera_motion_recovery", {"video_path": "/no/such/clip.mp4"}
        )
    assert result.isError is True
    assert "invalid_input_path" in result.content[0].text


@pytest.mark.asyncio
async def test_a_job_that_has_not_solved_is_job_not_ready(wired):
    service, runner = wired
    job = runner.add(make_job(state=JobState.ANALYZING, trajectories=[]))
    async with _session() as session:
        result = await session.call_tool("get_camera_motion", {"job_id": job.id})
    assert result.isError is True
    assert "job_not_ready" in result.content[0].text


# ----------------------------------------------------------------------- wait


@pytest.mark.asyncio
async def test_wait_for_job_reports_a_clamped_timeout(wired):
    service, runner = wired
    job = runner.add(make_job(state=JobState.COMPLETE))
    async with _session() as session:
        payload = _structured(
            await session.call_tool(
                "wait_for_job", {"job_id": job.id, "timeout_seconds": 9000.0}
            )
        )
    status = JobStatus.model_validate(payload)
    assert status.timeout_clamped_to == 600.0
    assert status.terminal is True
    assert status.timed_out is False


@pytest.mark.asyncio
async def test_a_timed_out_wait_is_a_normal_result(wired):
    service, runner = wired
    job = runner.add(make_job(state=JobState.SOLVING, trajectories=[]))
    async with _session() as session:
        result = await session.call_tool(
            "wait_for_job",
            {"job_id": job.id, "timeout_seconds": 1.0, "poll_interval_seconds": 0.25},
        )
    assert result.isError is False
    status = JobStatus.model_validate(result.structuredContent)
    assert status.timed_out is True
    assert status.terminal is False
    assert "wait_for_job again" in status.next_step


@pytest.mark.asyncio
async def test_wait_for_job_emits_progress_notifications(wired):
    service, runner = wired
    job = runner.add(make_job(state=JobState.COMPLETE))
    seen: list[tuple[float, float | None, str | None]] = []

    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        seen.append((progress, total, message))

    async with _session() as session:
        await session.call_tool(
            "wait_for_job",
            {"job_id": job.id, "timeout_seconds": 2.0},
            progress_callback=on_progress,
        )
    assert seen, "no notifications/progress was sent"
    assert seen[0][1] == 1.0


# ------------------------------------------------------------------- SDK guard


def test_the_private_server_attribute_the_stdio_runner_uses_still_exists():
    # `_run_stdio` drives `mcp._mcp_server` directly because the fd swap makes
    # `run_stdio_async`'s implicit stdout wrong. If an SDK bump moves this, fail
    # here rather than at the first client connection.
    assert hasattr(mcp_server.mcp, "_mcp_server")
    assert hasattr(mcp_server.mcp._mcp_server, "create_initialization_options")
    assert hasattr(mcp_server.mcp._mcp_server, "run")


# ---------------------------------------------------------------- resources II


@pytest.mark.asyncio
async def test_job_motion_resource_matches_the_tool(wired):
    service, runner = wired
    job = runner.add(make_job())
    async with _session() as session:
        resource = await session.read_resource(AnyUrl(f"camerapath://jobs/{job.id}/motion"))
        payload = _structured(await session.call_tool("get_camera_motion", {"job_id": job.id}))
    assert json.loads(resource.contents[0].text) == payload


@pytest.mark.asyncio
async def test_job_output_resource_refuses_a_binary_file(wired):
    service, runner = wired
    job = runner.add(make_job())
    outputs = runner.job_dir(job.id) / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "motion_proxy.mp4").write_bytes(b"\0" * 16)
    with pytest.raises(Exception) as exc:
        async with _session() as session:
            await session.read_resource(
                AnyUrl(f"camerapath://jobs/{job.id}/outputs/motion_proxy.mp4")
            )
    # read_resource surfaces server errors through an anyio ExceptionGroup, so the
    # message is in the group's repr rather than its str.
    assert "binary output" in repr(exc.value)


@pytest.mark.asyncio
async def test_job_output_resource_serves_a_text_export(wired):
    service, runner = wired
    job = runner.add(make_job())
    outputs = runner.job_dir(job.id) / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "trajectory.json").write_text('{"schema_version": "1.0"}')
    async with _session() as session:
        result = await session.read_resource(
            AnyUrl(f"camerapath://jobs/{job.id}/outputs/trajectory.json")
        )
    assert json.loads(result.contents[0].text)["schema_version"] == "1.0"
