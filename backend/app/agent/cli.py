"""JSON CLI for agents with shell access.

Every command prints exactly one JSON document to stdout and nothing else.
Progress, warnings and logs go to stderr, so `cpl run ... | jq` works while the
user still sees what is happening. The two exceptions are explicit opt-ins:
`describe --text` prints the bare description, and `samples --csv` prints CSV —
both because the caller asked for that shape instead of the envelope.

`result` is the same pydantic model the MCP tool returns, dumped in JSON mode,
so `cpl motion <id>` and the `get_camera_motion` tool's `structuredContent`
agree field for field. A test asserts that equivalence.

Exit codes carry the error class (see `EXIT_CODES`), so a shell agent can branch
without parsing prose — and a non-zero exit still prints the failure JSON.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from typing import Any

from app.agent.models import AgentError, Stages
from app.agent.service import AgentService, build_solve_settings

#: error code -> process exit status.
EXIT_CODES = {
    "internal": 1,
    "invalid_input_path": 2,
    "unsupported_media_type": 2,
    "file_too_large": 2,
    "job_failed": 3,
    "job_not_found": 4,
    "timeout": 5,
    "job_cancelled": 6,
    "environment_unavailable": 7,
    "render_unavailable": 7,
    "backend_unreachable": 8,
    "job_not_ready": 9,
    "job_busy": 9,
}

_STYLES = ("prompt", "technical", "brief")
_MODES = ("auto", "physical_3d", "perceptual_match", "fast", "high_accuracy")
_PROXY_STYLES = ("motion_cage", "depth_poles", "ground_grid", "minimal")


def _emit(payload: dict[str, Any], *, pretty: bool) -> None:
    text = json.dumps(payload, indent=2 if pretty else None, default=str)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def _note(message: str, *, quiet: bool) -> None:
    if not quiet:
        sys.stderr.write(message.rstrip() + "\n")
        sys.stderr.flush()


def _ok(command: str, result: Any, *, job_id: str | None, elapsed: float, pretty: bool) -> int:
    _emit(
        {
            "ok": True,
            "command": command,
            "job_id": job_id,
            "elapsed_seconds": round(elapsed, 3),
            "result": result.model_dump(mode="json") if hasattr(result, "model_dump") else result,
        },
        pretty=pretty,
    )
    return 0


def _err(command: str, exc: AgentError, *, job_id: str | None, pretty: bool) -> int:
    _emit(
        {"ok": False, "command": command, "job_id": job_id, "error": exc.to_dict()},
        pretty=pretty,
    )
    return EXIT_CODES.get(exc.code, 1)


# --------------------------------------------------------------------- parser


def _add_output_flags(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    """Output-side flags.

    `defaults=False` is the re-render case, where every unset flag has to stay
    `None` so it can mean "keep the job's current value" rather than silently
    resetting a size the user chose when the job was created.
    """
    parser.add_argument("--proxy-style", choices=_PROXY_STYLES,
                        default="motion_cage" if defaults else None)
    parser.add_argument("--output-width", type=int, default=1920 if defaults else None)
    parser.add_argument("--output-height", type=int, default=1080 if defaults else None)
    parser.add_argument("--match-source-aspect", action="store_true", default=None)
    parser.add_argument("--output-fps", type=float, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cpl",
        description="CameraPath Lab — recover camera motion and emit JSON for agents.",
    )
    parser.add_argument("--runner", choices=("auto", "local", "delegate"), default=None)
    parser.add_argument("--backend-url", default=None)
    parser.add_argument("--ui-base-url", default=None)
    parser.add_argument("--workspace", default=None, help="Override CPL_WORKSPACE_DIR.")
    parser.add_argument("--log-level", default=os.environ.get("CPL_LOG_LEVEL", "WARNING"))
    parser.add_argument("--pretty", action="store_true", help="Indent the JSON document.")
    parser.add_argument("--quiet", action="store_true", help="Suppress stderr progress lines.")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="What this machine can do, and which runner is in force.")

    run = sub.add_parser("run", help="Create a job, solve every shot, render the proxy.")
    run.add_argument("video")
    run.add_argument("--mode", choices=_MODES, default="auto")
    run.add_argument("--motion-fidelity", choices=("exact", "clean", "smooth"), default="exact")
    run.add_argument("--no-render", dest="render", action="store_false", default=True)
    run.add_argument("--lens-mode", choices=("auto", "fixed", "variable"), default="auto")
    run.add_argument("--fov-override-degrees", type=float, default=None)
    run.add_argument("--keyframe-density", type=float, default=1.0)
    run.add_argument("--dynamic-rejection-strength", type=float, default=0.5)
    run.add_argument("--confidence-threshold", type=float, default=0.35)
    run.add_argument("--scale-calibration", default=None,
                     help='JSON: {"kind":"travel_distance","value_meters":3.5}')
    run.add_argument("--source-mode", choices=("link", "copy"), default="link")
    _add_output_flags(run, defaults=True)
    run.add_argument("--wait", dest="wait", action="store_true", default=True)
    run.add_argument("--no-wait", dest="wait", action="store_false")
    run.add_argument("--timeout", type=float, default=900.0, help="0 = wait forever.")
    run.add_argument("--describe", action="store_true",
                     help="Include the prompt-ready description in the result.")

    analyze = sub.add_parser("analyze", help="Shot cuts and 2D motion only — no geometry.")
    analyze.add_argument("video")
    analyze.add_argument("--mode", choices=_MODES, default="auto")
    analyze.add_argument("--dynamic-rejection-strength", type=float, default=0.5)
    analyze.add_argument("--source-mode", choices=("link", "copy"), default="link")
    analyze.add_argument("--wait", dest="wait", action="store_true", default=True)
    analyze.add_argument("--no-wait", dest="wait", action="store_false")
    analyze.add_argument("--timeout", type=float, default=600.0)

    status = sub.add_parser("status", help="Current state of a job.")
    status.add_argument("job_id")
    status.add_argument("--log-tail", type=int, default=0)

    wait = sub.add_parser("wait", help="Block until a job is terminal, or the timeout.")
    wait.add_argument("job_id")
    wait.add_argument("--timeout", type=float, default=900.0)
    wait.add_argument("--poll", type=float, default=1.0)

    motion = sub.add_parser("motion", help="The full measured result.")
    motion.add_argument("job_id")
    motion.add_argument("--shot", type=int, default=None)
    motion.add_argument("--poses", action="store_true")
    motion.add_argument("--max-poses", type=int, default=200)

    describe = sub.add_parser("describe", help="Prompt-ready camera-motion text.")
    describe.add_argument("job_id")
    describe.add_argument("--shot", type=int, default=None)
    describe.add_argument("--style", choices=_STYLES, default="prompt")
    describe.add_argument("--max-chars", type=int, default=600)
    describe.add_argument("--text", action="store_true",
                          help="Print the bare text instead of the JSON envelope.")

    samples = sub.add_parser("samples", help="Numeric trajectory rows.")
    samples.add_argument("job_id")
    samples.add_argument("--shot", type=int, required=True)
    samples.add_argument("--stride", type=int, default=1)
    samples.add_argument("--max-rows", type=int, default=500)
    samples.add_argument("--fields", default=None, help="Comma-separated column names.")
    samples.add_argument("--csv", action="store_true",
                         help="Print CSV instead of the JSON envelope.")

    outputs = sub.add_parser("outputs", help="Every file the job produced.")
    outputs.add_argument("job_id")
    outputs.add_argument("--no-verify", dest="verify", action="store_false", default=True)

    render = sub.add_parser("render", help="Re-render the proxy from the exported trajectory.")
    render.add_argument("job_id")
    _add_output_flags(render, defaults=False)
    render.add_argument("--render-trajectory-preview", action="store_true", default=None)
    render.add_argument("--wait", dest="wait", action="store_true", default=False)
    render.add_argument("--timeout", type=float, default=900.0)

    cancel = sub.add_parser("cancel", help="Ask a running job to stop.")
    cancel.add_argument("job_id")

    jobs = sub.add_parser("jobs", help="Recent jobs, newest first.")
    jobs.add_argument("--limit", type=int, default=20)

    rm = sub.add_parser("rm", help="Delete a job directory.")
    rm.add_argument("job_id")
    rm.add_argument("--force", action="store_true", help="Cancel it first if it is running.")

    serve = sub.add_parser("serve-mcp", help="Run the MCP server (same entrypoint as cpl-mcp).")
    serve.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    return parser


# -------------------------------------------------------------------- helpers


def _wait_and_report(service: AgentService, job_id: str, timeout: float, *,
                     quiet: bool, poll: float = 1.0):
    """Block to a terminal state, echoing stage lines to stderr."""
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    last = ""

    def on_progress(status) -> None:  # noqa: ANN001
        nonlocal last
        stage = status.stage.message if status.stage and status.stage.message else status.state
        pct = f"{status.progress * 100:5.1f}%" if status.progress is not None else "   -- "
        line = f"[{pct}] {status.state}: {stage}"
        if line != last:
            _note(line, quiet=quiet)
            last = line

    while True:
        remaining = 60.0 if deadline is None else max(0.0, deadline - time.monotonic())
        status = service.wait(
            job_id, timeout_s=min(60.0, remaining) if deadline else 60.0,
            on_progress=on_progress, poll_interval_s=poll,
        )
        if status.terminal:
            return status
        if deadline is not None and time.monotonic() >= deadline:
            return status


def _terminal_exit(status) -> int:  # noqa: ANN001
    """Exit status for a command that waited.

    Success is "the job reached the end of what it was asked to do", not
    literally `complete`: an analysis-only run rests at `analyzed` and a solve
    with `--no-render` rests at `solved`, and both are finished. `terminal`
    already encodes that, so a non-terminal status here means the wait ran out.
    """
    if status.state == "failed":
        return EXIT_CODES["job_failed"]
    if status.state == "cancelled":
        return EXIT_CODES["job_cancelled"]
    return 0 if status.terminal else EXIT_CODES["timeout"]


def _output_overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "proxy_style": getattr(args, "proxy_style", None),
        "output_width": getattr(args, "output_width", None),
        "output_height": getattr(args, "output_height", None),
        "match_source_aspect": getattr(args, "match_source_aspect", None),
        "output_fps": getattr(args, "output_fps", None),
        "render_trajectory_preview": getattr(args, "render_trajectory_preview", None),
    }


def _parse_calibration(raw: str | None) -> dict | None:
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AgentError(
            "invalid_input_path",
            f"--scale-calibration is not valid JSON: {exc}",
            detail='Expected {"kind": "travel_distance", "value_meters": 3.5}',
        ) from exc
    if not isinstance(value, dict):
        raise AgentError("invalid_input_path", "--scale-calibration must be a JSON object")
    return value


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from app.core.logging import configure_agent_logging

    configure_agent_logging(
        sys.stderr, level=getattr(logging, str(args.log_level).upper(), logging.WARNING)
    )

    if args.workspace:
        os.environ["CPL_WORKSPACE_DIR"] = args.workspace

    if args.command == "serve-mcp":
        from app.agent.mcp_server import main as mcp_main

        forwarded = ["--transport", args.transport, "--log-level", str(args.log_level).upper()]
        if args.host:
            forwarded += ["--host", args.host]
        if args.port:
            forwarded += ["--port", str(args.port)]
        if args.runner:
            forwarded += ["--runner", args.runner]
        if args.backend_url:
            forwarded += ["--backend-url", args.backend_url]
        if args.ui_base_url:
            forwarded += ["--ui-base-url", args.ui_base_url]
        return mcp_main(forwarded)

    pretty = bool(args.pretty)
    quiet = bool(args.quiet)
    job_id = getattr(args, "job_id", None)
    started = time.monotonic()

    try:
        from app.agent.runner import pick_runner

        service = AgentService(
            pick_runner(args.runner, backend_url=args.backend_url),
            ui_base_url=args.ui_base_url,
        )
        return _dispatch(service, args, job_id=job_id, pretty=pretty, quiet=quiet,
                         started=started)
    except AgentError as exc:
        return _err(args.command, exc, job_id=job_id, pretty=pretty)
    except KeyboardInterrupt:
        return _err(
            args.command,
            AgentError("internal", "interrupted"),
            job_id=job_id,
            pretty=pretty,
        )
    except Exception as exc:  # noqa: BLE001 - stdout must stay parseable whatever happens
        return _err(
            args.command,
            AgentError("internal", f"{type(exc).__name__}: {exc}"),
            job_id=job_id,
            pretty=pretty,
        )


def _dispatch(service: AgentService, args: argparse.Namespace, *, job_id: str | None,
              pretty: bool, quiet: bool, started: float) -> int:
    command = args.command

    if command == "env":
        return _ok(command, service.environment(), job_id=None,
                   elapsed=time.monotonic() - started, pretty=pretty)

    if command == "jobs":
        return _ok(command, service.list_jobs(limit=args.limit), job_id=None,
                   elapsed=time.monotonic() - started, pretty=pretty)

    if command in ("run", "analyze"):
        if command == "run":
            settings = build_solve_settings(
                mode=args.mode,
                motion_fidelity=args.motion_fidelity,
                proxy_style=args.proxy_style,
                output_width=args.output_width,
                output_height=args.output_height,
                match_source_aspect=bool(args.match_source_aspect),
                output_fps=args.output_fps,
                lens_mode=args.lens_mode,
                fov_override_degrees=args.fov_override_degrees,
                keyframe_density=args.keyframe_density,
                dynamic_rejection_strength=args.dynamic_rejection_strength,
                confidence_threshold=args.confidence_threshold,
                scale_calibration=_parse_calibration(args.scale_calibration),
            )
            stages = Stages(analyze=True, solve=True, render=bool(args.render))
        else:
            settings = build_solve_settings(
                mode=args.mode,
                dynamic_rejection_strength=args.dynamic_rejection_strength,
            )
            stages = Stages(analyze=True, solve=False, render=False)

        ref = service.start_recovery(
            args.video, settings=settings, stages=stages, source_mode=args.source_mode
        )
        job_id = ref.job_id
        _note(f"job {job_id} — {ref.ui_url}", quiet=quiet)
        result: dict[str, Any] = {"job": ref.model_dump(mode="json")}

        if not args.wait:
            result["status"] = service.status(job_id).model_dump(mode="json")
            return _ok(command, result, job_id=job_id,
                       elapsed=time.monotonic() - started, pretty=pretty)

        status = _wait_and_report(service, job_id, args.timeout, quiet=quiet)
        result["status"] = status.model_dump(mode="json")
        if status.state in ("solved", "complete") or status.outputs_ready:
            try:
                motion = service.motion(job_id)
                result["motion_summary"] = {
                    "shot_count": motion.shot_count,
                    "scale": motion.scale.model_dump(mode="json"),
                    "shots": [
                        {
                            "shot_id": s.shot_id,
                            "start_time": s.start_time,
                            "end_time": s.end_time,
                            "solver_used": s.solver_used,
                            "confidence": s.confidence.level,
                            "translation_observable": s.translation.observable,
                            "primary_move": s.primary_move,
                            "pipeline_summary": s.pipeline_summary,
                        }
                        for s in motion.shots
                    ],
                }
                result["outputs"] = service.outputs(job_id).model_dump(mode="json")
                if getattr(args, "describe", False):
                    result["description"] = service.description(job_id).model_dump(mode="json")
            except AgentError as exc:
                result["motion_summary"] = {"unavailable": exc.to_dict()}

        _ok(command, result, job_id=job_id, elapsed=time.monotonic() - started, pretty=pretty)
        return _terminal_exit(status)

    if command == "status":
        return _ok(command, service.status(args.job_id, log_tail=args.log_tail),
                   job_id=args.job_id, elapsed=time.monotonic() - started, pretty=pretty)

    if command == "wait":
        status = _wait_and_report(service, args.job_id, args.timeout, quiet=quiet,
                                  poll=args.poll)
        _ok(command, status, job_id=args.job_id, elapsed=time.monotonic() - started,
            pretty=pretty)
        return _terminal_exit(status)

    if command == "motion":
        return _ok(
            command,
            service.motion(args.job_id, shot_id=args.shot, include_poses=args.poses,
                           max_poses=args.max_poses),
            job_id=args.job_id, elapsed=time.monotonic() - started, pretty=pretty,
        )

    if command == "describe":
        result = service.description(args.job_id, shot_id=args.shot, style=args.style,
                                     max_chars=args.max_chars)
        if args.text:
            sys.stdout.write(result.text + "\n")
            sys.stdout.flush()
            return 0
        return _ok(command, result, job_id=args.job_id,
                   elapsed=time.monotonic() - started, pretty=pretty)

    if command == "samples":
        fields = [f.strip() for f in args.fields.split(",")] if args.fields else None
        result = service.samples(args.job_id, shot_id=args.shot, stride=args.stride,
                                 max_rows=args.max_rows, fields=fields)
        if args.csv:
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow([f"{c} [{result.units.get(c, '')}]" for c in result.columns])
            for row in result.rows:
                writer.writerow([
                    json.dumps(v) if isinstance(v, list) else v for v in row
                ])
            sys.stdout.write(buffer.getvalue())
            sys.stdout.flush()
            return 0
        return _ok(command, result, job_id=args.job_id,
                   elapsed=time.monotonic() - started, pretty=pretty)

    if command == "outputs":
        return _ok(command, service.outputs(args.job_id, verify_mp4=args.verify),
                   job_id=args.job_id, elapsed=time.monotonic() - started, pretty=pretty)

    if command == "render":
        status = service.rerender(args.job_id, output=_output_overrides(args))
        if args.wait:
            status = _wait_and_report(service, args.job_id, args.timeout, quiet=quiet)
            _ok(command, status, job_id=args.job_id, elapsed=time.monotonic() - started,
                pretty=pretty)
            return _terminal_exit(status)
        return _ok(command, status, job_id=args.job_id,
                   elapsed=time.monotonic() - started, pretty=pretty)

    if command == "cancel":
        return _ok(command, service.cancel(args.job_id), job_id=args.job_id,
                   elapsed=time.monotonic() - started, pretty=pretty)

    if command == "rm":
        if args.force:
            try:
                service.cancel(args.job_id)
            except AgentError:
                pass
        return _ok(command, service.delete_job(args.job_id), job_id=args.job_id,
                   elapsed=time.monotonic() - started, pretty=pretty)

    raise AgentError("internal", f"unhandled command: {command}")


if __name__ == "__main__":
    sys.exit(main())
