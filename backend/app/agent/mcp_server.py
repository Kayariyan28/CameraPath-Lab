"""The CameraPath Lab MCP server.

Thin by design: every tool validates its arguments, calls one `AgentService`
method, and returns a pydantic model. The models are declared as return types so
FastMCP emits `structuredContent` alongside the text block, which is what lets an
agent read a field rather than parse a sentence.

Two contracts matter more than the individual tools.

**Nothing blocks.** Every tool that starts work returns in well under a second
and hands back a job id. The work runs in a background thread (local mode) or in
the backend's executor (delegated mode). An agent polls with `get_job_status` or
blocks — bounded — with `wait_for_job`, which reports progress notifications
while it waits so a client with a progress token does not time the call out. A
timed-out wait is a normal result, never an error.

**Nothing writes to stdout.** Under stdio, fd 1 is the protocol. `stdio_guard`
swaps it before anything else is imported; this module never prints.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Annotated, Any, Literal

import anyio
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import Field

from app.agent.models import (
    AgentError,
    DeleteResult,
    DescriptionResult,
    EnvironmentReport,
    JobListResult,
    JobMotionReport,
    JobRef,
    JobStatus,
    OutputsReport,
    Stages,
    TrajectorySamples,
)
from app.agent.service import AgentService, build_solve_settings
from app.core.logging import get_logger
from app.core.paths import JOB_ID_RE
from app.models.schemas.trajectory import CoordinateSystem

log = get_logger("agent.mcp")

INSTRUCTIONS = """\
CameraPath Lab recovers real camera motion from a reference video and renders a \
neutral "motion proxy" MP4 that a generative video model can be conditioned on.

Typical flow: `check_environment` once, then \
`start_camera_motion_recovery(video_path=...)`, then `wait_for_job` in a loop \
until `terminal` is true, then `describe_camera_motion` for prompt text and \
`list_outputs` for the MP4 and data files.

Every shot is solved in its own coordinate system; a video with a hard cut \
returns several shots. Translation is in **normalized units, not metres**, \
unless `scale_units` says `m`. When `translation.observable` is false the tool \
measured rotation and lens only — do not describe the camera as moving through \
space.\
"""

WEBSITE_URL = "https://github.com/Kayariyan28/CameraPath-Lab"

#: Kept in step with the FastAPI app's version in app/main.py.
SERVER_VERSION = "0.1.0"

mcp: FastMCP = FastMCP(
    name="camerapath-lab",
    instructions=INSTRUCTIONS,
    website_url=WEBSITE_URL,
    host=os.environ.get("CPL_MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("CPL_MCP_PORT", "8849")),
)

# FastMCP takes no `version`, and the server it wraps then falls back to the SDK's
# own version — so a client's initialize response advertises the protocol library's
# number as this tool's. The wrapped server does accept one, so set it there.
# Guarded: a future SDK that exposes `version` on FastMCP must not break startup.
if hasattr(mcp, "_mcp_server"):
    mcp._mcp_server.version = SERVER_VERSION

_service: AgentService | None = None


def set_service(service: AgentService | None) -> None:
    """Install the service this server calls. Used by the entrypoint and tests."""
    global _service
    _service = service


def get_service() -> AgentService:
    global _service
    if _service is None:
        from app.agent.runner import pick_runner

        _service = AgentService(pick_runner())
    return _service


# --------------------------------------------------------------------- guards


def _fail(exc: AgentError) -> ToolError:
    """Render an AgentError as the tool error an MCP client will show."""
    return ToolError(str(exc))


def _job_id(value: str) -> str:
    """Validate before the string can touch the filesystem."""
    if not isinstance(value, str) or not JOB_ID_RE.match(value):
        raise ToolError(
            "[job_not_found] malformed job id. A job id is the canonical UUID4 "
            "string returned by start_camera_motion_recovery."
        )
    return value


def _clamp(value: float, low: float, high: float) -> tuple[float, bool]:
    clamped = max(low, min(high, float(value)))
    return clamped, clamped != float(value)


# ----------------------------------------------------------------------- tools


@mcp.tool()
def check_environment() -> EnvironmentReport:
    """What this machine can actually do, and which execution mode is in force.

    Call once at the start of a session. `ready` means video ingestion works
    (ffmpeg + ffprobe). `can_render` false means trajectory exports still work
    but no MP4 can be produced. `runner_mode` says whether jobs run inside the
    already-running CameraPath Lab backend ("delegated" — they survive this
    process and show live progress in the UI) or inside this process ("local" —
    they do not survive it).
    """
    try:
        return get_service().environment()
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def start_camera_motion_recovery(
    video_path: Annotated[str, Field(description="Absolute path to a video file on this machine.")],
    mode: Literal["auto", "physical_3d", "perceptual_match", "fast", "high_accuracy"] = "auto",
    motion_fidelity: Literal["exact", "clean", "smooth"] = "exact",
    render: Annotated[bool, Field(description="Render the motion-proxy MP4 after solving.")] = True,
    proxy_style: Literal["motion_cage", "depth_poles", "ground_grid", "minimal"] = "motion_cage",
    output_width: Annotated[int, Field(ge=160, le=7680)] = 1920,
    output_height: Annotated[int, Field(ge=160, le=4320)] = 1080,
    match_source_aspect: bool = False,
    output_fps: Annotated[float | None, Field(description="None = the source's own fps.")] = None,
    lens_mode: Literal["auto", "fixed", "variable"] = "auto",
    fov_override_degrees: float | None = None,
    keyframe_density: Annotated[float, Field(ge=0.25, le=4.0)] = 1.0,
    dynamic_rejection_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.35,
    scale_calibration: Annotated[
        dict[str, Any] | None,
        Field(
            description=(
                "One real-world measurement that upgrades the result from relative to "
                "metric: {kind: camera_height|point_distance|travel_distance, "
                "value_meters: > 0, note?: str}. Without it, distances stay normalized "
                "— monocular video carries no absolute scale."
            )
        ),
    ] = None,
    source_mode: Literal["link", "copy"] = "link",
) -> JobRef:
    """Recover camera motion from a video: analyse, solve every shot, render the proxy.

    Returns immediately with a job id. The work runs in the background — poll
    with `wait_for_job` until `terminal` is true, then call
    `describe_camera_motion` and `list_outputs`.

    A video with hard cuts is split into shots and each is solved in its own
    coordinate system; no trajectory ever spans a cut. If Blender is missing the
    job still completes with all the trajectory exports and
    `render_skipped_reason` set.
    """
    service = get_service()
    try:
        settings = build_solve_settings(
            mode=mode,
            motion_fidelity=motion_fidelity,
            proxy_style=proxy_style,
            output_width=output_width,
            output_height=output_height,
            match_source_aspect=match_source_aspect,
            output_fps=output_fps,
            lens_mode=lens_mode,
            fov_override_degrees=fov_override_degrees,
            keyframe_density=keyframe_density,
            dynamic_rejection_strength=dynamic_rejection_strength,
            confidence_threshold=confidence_threshold,
            scale_calibration=scale_calibration,
        )
        return service.start_recovery(
            video_path,
            settings=settings,
            stages=Stages(analyze=True, solve=True, render=bool(render)),
            source_mode=source_mode,
        )
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def start_video_analysis(
    video_path: str,
    mode: Literal["auto", "physical_3d", "perceptual_match", "fast", "high_accuracy"] = "auto",
    dynamic_rejection_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
    source_mode: Literal["link", "copy"] = "link",
) -> JobRef:
    """The cheap look: shot cuts, dense 2D motion, complexity and routing only.

    No geometry and no render, so it finishes in seconds rather than minutes.
    Use it to find the cuts and to see whether translation is likely to be
    observable before committing to a full solve. Poll with `wait_for_job`, then
    read `shot_count` from `get_job_status`.
    """
    service = get_service()
    try:
        settings = build_solve_settings(
            mode=mode, dynamic_rejection_strength=dynamic_rejection_strength
        )
        return service.start_recovery(
            video_path,
            settings=settings,
            stages=Stages(analyze=True, solve=False, render=False),
            source_mode=source_mode,
        )
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def get_job_status(
    job_id: str,
    log_tail: Annotated[int, Field(ge=0, le=200, description="Trailing stage-log lines. 0 = none.")] = 0,
) -> JobStatus:
    """Where a job is right now. Instant, and safe to call as often as you like.

    `terminal` is true for complete, failed and cancelled. A failed job is
    returned as data, not as an error: read `error` and `error_detail`.
    `next_step` names the call to make next.
    """
    service = get_service()
    try:
        return service.status(_job_id(job_id), log_tail=max(0, min(200, int(log_tail))))
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
async def wait_for_job(
    job_id: str,
    ctx: Context,
    timeout_seconds: Annotated[
        float,
        Field(
            gt=0.0,
            description=(
                "Seconds to block, 1-600. Values outside that range are clamped "
                "rather than rejected, and the clamp is reported as "
                "timeout_clamped_to — a wait must never fail the caller."
            ),
        ),
    ] = 60.0,
    poll_interval_seconds: Annotated[
        float, Field(gt=0.0, description="Seconds between polls, 0.25-10. Clamped.")
    ] = 1.0,
) -> JobStatus:
    """Block until the job finishes, or until the timeout — whichever is first.

    A timeout is a normal result, not an error: you get `timed_out: true`,
    `terminal: false` and a `next_step` telling you to call again with the same
    job id. Looping is the right way to wait out a long solve; raising your
    client's per-call timeout is not.

    Progress notifications are sent on every change while waiting. Cancelling
    this *request* abandons the wait only — it does not stop the job. Use
    `cancel_job` for that.
    """
    service = get_service()
    job_id = _job_id(job_id)
    timeout, clamped = _clamp(timeout_seconds, 1.0, 600.0)
    interval, _ = _clamp(poll_interval_seconds, 0.25, 10.0)

    try:
        status = await anyio.to_thread.run_sync(lambda: service.status(job_id))
    except AgentError as exc:
        raise _fail(exc) from exc

    started = anyio.current_time()
    deadline = started + timeout
    last = (status.progress, status.stage.name if status.stage else None,
            status.stage.message if status.stage else "")
    await _report(ctx, status)

    while not status.terminal and anyio.current_time() < deadline:
        await anyio.sleep(min(interval, max(0.0, deadline - anyio.current_time())))
        try:
            status = await anyio.to_thread.run_sync(lambda: service.status(job_id))
        except AgentError as exc:
            raise _fail(exc) from exc
        key = (status.progress, status.stage.name if status.stage else None,
               status.stage.message if status.stage else "")
        if key != last:
            await _report(ctx, status)
            last = key

    waited = anyio.current_time() - started
    update: dict[str, Any] = {
        "waited_seconds": round(waited, 3),
        "timed_out": not status.terminal,
    }
    if clamped:
        update["timeout_clamped_to"] = timeout
    if not status.terminal:
        update["next_step"] = "call wait_for_job again with the same job_id"
    return status.model_copy(update=update)


async def _report(ctx: Context, status: JobStatus) -> None:
    """Mirror one status change into MCP progress and logging notifications."""
    message = status.stage.message if status.stage and status.stage.message else status.state
    try:
        if status.progress is not None:
            await ctx.report_progress(status.progress, 1.0, message=message)
        await ctx.info(f"{status.state}: {message}")
    except Exception:  # noqa: BLE001 - a client that cannot take notifications still gets the result
        pass


@mcp.tool()
def cancel_job(job_id: str) -> JobStatus:
    """Ask a running job to stop.

    Cancellation is cooperative: it takes effect at the next stage or shot
    boundary, so the state may still read `solving` for a few seconds. Exports
    already written stay on disk.
    """
    service = get_service()
    try:
        return service.cancel(_job_id(job_id))
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def get_camera_motion(
    job_id: str,
    shot_id: Annotated[int | None, Field(ge=0, description="One shot, or None for all.")] = None,
    include_poses: bool = False,
    max_poses: Annotated[int, Field(ge=0, le=2000)] = 200,
) -> JobMotionReport:
    """The full measured result: solver, confidence, observability, rotation, lens, moves.

    Read `translation.observable` before you read anything about travel. When it
    is false the pipeline measured rotation and lens only, and the camera must
    not be described as moving through space. Distances are in
    `scale.units` — `normalized` unless a calibration was supplied, and
    normalized units are relative, not metres.

    Poses, when requested, are a strided subset of the real solved poses —
    never resampled or interpolated. `pose_stride` says how they were thinned.
    """
    service = get_service()
    try:
        return service.motion(
            _job_id(job_id),
            shot_id=shot_id,
            include_poses=bool(include_poses),
            max_poses=max_poses,
        )
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def describe_camera_motion(
    job_id: str,
    shot_id: Annotated[int | None, Field(ge=0)] = None,
    style: Literal["prompt", "technical", "brief"] = "prompt",
    max_chars: Annotated[int, Field(ge=80, le=2000)] = 600,
) -> DescriptionResult:
    """Prompt-ready text describing the recovered camera move.

    Every magnitude in the text is a number the pipeline measured — the
    classifier's own phrase, copied verbatim — so each clause can be traced back
    to a field in `get_camera_motion`. `derived_from` lists which fields were
    used. `caveats` is never truncated: read it before passing the text to a
    video model.
    """
    service = get_service()
    try:
        return service.description(
            _job_id(job_id), shot_id=shot_id, style=style, max_chars=max_chars
        )
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def list_outputs(job_id: str, verify_mp4: bool = True) -> OutputsReport:
    """Every file the job produced, with absolute paths.

    `motion_proxy_mp4` is the motion reference to feed a generative video model.
    Its frame count, fps and duration are read back off the encoded file with
    ffprobe — not what Blender was asked to do — and `timing_matches_source`
    says whether the proxy's timing matches the source video.
    """
    service = get_service()
    try:
        return service.outputs(_job_id(job_id), verify_mp4=bool(verify_mp4))
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def get_trajectory_samples(
    job_id: str,
    shot_id: Annotated[int, Field(ge=0)],
    stride: Annotated[int, Field(ge=1, le=100)] = 1,
    max_rows: Annotated[int, Field(ge=1, le=2000)] = 500,
    fields: Annotated[
        list[str] | None,
        Field(
            description=(
                "Subset of: time, position, quaternion, fov_horizontal, "
                "focal_normalized, speed, angular_speed, curvature, confidence, "
                "solver_source, is_anchor."
            )
        ),
    ] = None,
) -> TrajectorySamples:
    """Numeric trajectory rows inline, for an agent without shell access.

    Strided subsampling only — rows are real solved samples, never resampled
    onto a uniform grid. `units` gives the unit of every column; positions are
    in the shot's own scale units, which are relative unless they say `m`.
    """
    service = get_service()
    try:
        return service.samples(
            _job_id(job_id),
            shot_id=int(shot_id),
            stride=stride,
            max_rows=max_rows,
            fields=fields,
        )
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def render_motion_proxy(
    job_id: str,
    proxy_style: Literal["motion_cage", "depth_poles", "ground_grid", "minimal"] | None = None,
    output_width: Annotated[int | None, Field(ge=160, le=7680)] = None,
    output_height: Annotated[int | None, Field(ge=160, le=4320)] = None,
    match_source_aspect: bool | None = None,
    output_fps: float | None = None,
    render_trajectory_preview: bool | None = None,
) -> JobStatus:
    """Re-render the proxy MP4 from the already-exported trajectory.

    No geometry is recomputed, so a new style or output size costs a render
    rather than a solve. `None` for a parameter keeps the job's current value.
    Returns immediately — poll with `wait_for_job`.
    """
    service = get_service()
    try:
        return service.rerender(
            _job_id(job_id),
            output={
                "proxy_style": proxy_style,
                "output_width": output_width,
                "output_height": output_height,
                "match_source_aspect": match_source_aspect,
                "output_fps": output_fps,
                "render_trajectory_preview": render_trajectory_preview,
            },
        )
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def list_jobs(limit: Annotated[int, Field(ge=1, le=100)] = 20) -> JobListResult:
    """Recent jobs in this workspace, newest first."""
    service = get_service()
    try:
        return service.list_jobs(limit=limit)
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.tool()
def delete_job(job_id: str) -> DeleteResult:
    """Delete a job directory and everything in it. Not reversible.

    Refused while the job is running. Collect any output paths you still need
    first — deleting removes the MP4 and the exports too.
    """
    service = get_service()
    try:
        return service.delete_job(_job_id(job_id))
    except AgentError as exc:
        raise _fail(exc) from exc


# ------------------------------------------------------------------- resources


def _conventions_markdown() -> str:
    """Generated from the `CoordinateSystem` defaults so it cannot drift."""
    cs = CoordinateSystem()
    return f"""\
# CameraPath Lab coordinate and scale conventions

## World frame

* Name: `{cs.name}`
* Handedness: **{cs.handedness}-handed**
* Up axis: **{cs.up_axis}**
* Forward axis: **{cs.forward_axis}**
* The camera looks along **{cs.camera_looks_along}**, with its up along **{cs.camera_up}**
* `position` is the camera **centre** in world space
* `quaternion` is **`{cs.quaternion_order}`** and maps **{cs.quaternion_maps}** — it rotates a
  direction expressed in the camera frame into world space
* {cs.notes}

## One coordinate system per shot

Each shot is solved independently and has its own origin, orientation and scale.
Positions from two shots are not comparable, and no trajectory ever spans a hard
cut.

## Scale: normalized vs metric

Monocular video does not determine absolute scale — the same images come from a
small move in a small scene and a large move in a large scene. So translation is
**relative by default**, reported in normalized units, and the string `m` appears
only when the user supplied a real-world calibration that survived validation.
Ratios, timing and all rotations are exact either way; absolute distance is not.

## Optical camera pose, not a body pose

Every pose is the **optical camera**. For drone footage with an independently
articulated gimbal this is not the aircraft's body pose, and there is
deliberately no field that could be mistaken for one.

## `translation.observable = false`

The solve measured rotation and lens only. A pure pan, a planar or distant
scene, or a rotation-and-zoom move gives no parallax, so there is no geometric
evidence of the camera moving through space. When this flag is false:

* no dolly, truck, pedestal, orbit, crane or fly-through wording is justified;
* the positions in the export are held, not measured, and their path length is
  not travel;
* `not_observable_reason` carries the pipeline's own verbatim explanation.
"""


@mcp.resource(
    "camerapath://conventions",
    name="conventions",
    title="Coordinate, scale and honesty conventions",
    mime_type="text/markdown",
)
def conventions_resource() -> str:
    """The coordinate system, the normalized-vs-metric rule, and what
    `translation.observable = false` means. Read this once before interpreting
    any trajectory."""
    return _conventions_markdown()


@mcp.resource(
    "camerapath://environment",
    name="environment",
    title="Host capabilities",
    mime_type="application/json",
)
def environment_resource() -> str:
    """Same payload as `check_environment()`."""
    try:
        return get_service().environment().model_dump_json(indent=2)
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.resource(
    "camerapath://jobs",
    name="jobs",
    title="Recent jobs",
    mime_type="application/json",
)
def jobs_resource() -> str:
    """Same payload as `list_jobs(limit=20)`."""
    try:
        return get_service().list_jobs(limit=20).model_dump_json(indent=2)
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.resource(
    "camerapath://jobs/{job_id}/motion",
    name="job_motion",
    title="Recovered camera motion for one job",
    mime_type="application/json",
)
def job_motion_resource(job_id: str) -> str:
    """`get_camera_motion(job_id)` without poses."""
    try:
        return get_service().motion(_job_id(job_id)).model_dump_json(indent=2)
    except AgentError as exc:
        raise _fail(exc) from exc


@mcp.resource(
    "camerapath://jobs/{job_id}/description",
    name="job_description",
    title="Prompt-ready camera-motion description",
    mime_type="text/plain",
)
def job_description_resource(job_id: str) -> str:
    """`describe_camera_motion(job_id, style="prompt")`, as plain text."""
    try:
        result = get_service().description(_job_id(job_id), style="prompt")
    except AgentError as exc:
        raise _fail(exc) from exc
    if result.caveats:
        return result.text + "\n\nCaveats:\n" + "\n".join(f"- {c}" for c in result.caveats)
    return result.text


@mcp.resource(
    "camerapath://jobs/{job_id}/outputs/{filename}",
    name="job_output_file",
    title="One text export from a job",
)
def job_output_resource(job_id: str, filename: str) -> str:
    """A `.json`, `.csv` or `.chan` export, read from that job's outputs
    directory. Binary outputs (`.mp4`, `.blend`) are refused — use the absolute
    path from `list_outputs` instead."""
    try:
        return get_service().read_export(
            _job_id(job_id), filename, max_bytes=4 * 1024 * 1024
        ).text
    except AgentError as exc:
        raise _fail(exc) from exc


# ------------------------------------------------------------------ entrypoint


async def _run_stdio() -> None:
    """Run the stdio loop against the private protocol descriptor.

    `FastMCP.run_stdio_async()` calls `stdio_server()` with no arguments, which
    would take `sys.stdout.buffer` — and `stdio_guard` has pointed that at
    stderr. So the loop is driven here with the real descriptor instead.
    `_mcp_server` is private, but it is the same attribute the SDK's own
    `mcp/shared/memory.py` reaches for; a test asserts it still exists so an SDK
    bump fails loudly rather than silently.
    """
    from mcp.server.stdio import stdio_server

    from app.agent import stdio_guard

    stream = stdio_guard.protocol_text_stream()
    async with stdio_server(stdout=stream) as (read, write):
        await mcp._mcp_server.run(  # noqa: SLF001 - see docstring
            read, write, mcp._mcp_server.create_initialization_options()
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpl-mcp",
        description="Run the CameraPath Lab MCP server.",
    )
    parser.add_argument(
        "--transport", choices=("stdio", "streamable-http"), default="stdio",
        help="Default stdio: the transport an MCP client launches a local server with.",
    )
    parser.add_argument(
        "--http", action="store_true",
        help="Alias for --transport streamable-http.",
    )
    parser.add_argument("--host", default=None, help="Bind address for streamable-http.")
    parser.add_argument("--port", type=int, default=None, help="Bind port for streamable-http.")
    parser.add_argument(
        "--runner", choices=("auto", "local", "delegate"), default=None,
        help="auto (default) delegates to a running backend and otherwise runs jobs here.",
    )
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--ui-base-url", default=None)
    parser.add_argument("--workspace", default=None, help="Override CPL_WORKSPACE_DIR.")
    parser.add_argument(
        "--log-level", default=os.environ.get("CPL_LOG_LEVEL", "INFO"),
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The fd swap happens before the runner is built, because building it
    # imports the pipeline stack and probes the toolchain with subprocesses.
    from app.agent import stdio_guard
    from app.agent.solver_logs import quiet_solver_logs

    level = getattr(logging, args.log_level, logging.INFO)
    stdio_guard.install(level=level)
    quiet_solver_logs(level)

    if args.workspace:
        os.environ["CPL_WORKSPACE_DIR"] = args.workspace
    if args.host:
        os.environ["CPL_MCP_HOST"] = args.host
        mcp.settings.host = args.host
    if args.port:
        os.environ["CPL_MCP_PORT"] = str(args.port)
        mcp.settings.port = int(args.port)

    from app.agent.runner import pick_runner

    try:
        runner = pick_runner(args.runner, backend_url=args.backend_url)
    except AgentError as exc:
        log.error("%s", exc)
        return 8
    set_service(AgentService(runner, ui_base_url=args.ui_base_url))
    log.info("camerapath-lab MCP server ready (runner=%s)", runner.mode)

    transport = "streamable-http" if args.http else args.transport
    try:
        if transport == "stdio":
            anyio.run(_run_stdio)
        else:
            # The fd swap is applied here too, even though HTTP does not need
            # it, so the two transports behave identically.
            anyio.run(mcp.run_streamable_http_async)
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
