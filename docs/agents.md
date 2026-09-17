# Using CameraPath Lab from an AI agent

CameraPath Lab is usable as a tool, not just as an app. Two front ends sit on one
service layer:

- an **MCP server** (`scripts/cpl-mcp`) for any MCP-capable client or agent framework
- a **JSON CLI** (`scripts/cpl`) for any agent that can run a shell command

Both give an agent the same thing the UI gives a person: a recovered camera
trajectory, a rendered motion-reference MP4, and a plain-language description of the
camera move that is safe to paste into a video-generation prompt.

Neither invents numbers. Every value is what the pipeline measured, with its
confidence, its units, and whether translation was observable at all.

## Setup

```bash
./scripts/bootstrap_macos.sh      # installs the venv, including the mcp dependency
./scripts/cpl env                 # what this machine can do, as JSON
```

`scripts/cpl` and `scripts/cpl-mcp` resolve the repository themselves, so they work
from any directory and from a path containing spaces.

### MCP client configuration

Most clients take a JSON block like this. Use the **absolute path** to the wrapper:

```json
{
  "mcpServers": {
    "camerapath-lab": {
      "command": "/absolute/path/to/CameraPath Lab/scripts/cpl-mcp"
    }
  }
}
```

No `args`, `cwd` or `env` are required. To pin the execution mode (see below), add:

```json
{
  "mcpServers": {
    "camerapath-lab": {
      "command": "/absolute/path/to/CameraPath Lab/scripts/cpl-mcp",
      "env": { "CPL_AGENT_RUNNER": "local" }
    }
  }
}
```

For a client that connects over HTTP instead of spawning a process:

```bash
./scripts/cpl-mcp --http --port 8899      # streamable-http on 127.0.0.1:8899
```

## Execution modes

The server runs jobs one of two ways, chosen by `CPL_AGENT_RUNNER` (or `--runner`):

| mode | behaviour |
|------|-----------|
| `auto` (default) | Delegates to the CameraPath Lab backend if one is already running on `http://127.0.0.1:8848`; otherwise runs the job inside the MCP process. |
| `local` | Always runs the job in this process. No backend needed. |
| `delegate` | Requires the running backend, and fails with a clear error if it is absent. |

Delegation reads results off local disk, so it is only valid when the backend is
**this** checkout. If another copy of the project is serving that port, the first
upload fails with `backend_unreachable` naming both paths, rather than failing later
on a mysterious missing file. Use `CPL_AGENT_RUNNER=local` in that case.

Jobs land in the same workspace as the web app either way, so any job can be opened
in the UI at `http://localhost:5173/?job=<job_id>`. Every result carries that link as
`ui_url`.

A job started in `local` mode belongs to the MCP process: if the client shuts the
server down mid-solve, the job is recorded as interrupted rather than left claiming
to be running forever. Long work therefore survives better in `delegate` mode.

## The typical agent flow

1. `check_environment` — confirm `can_reconstruct_3d` and `can_render`.
2. `start_camera_motion_recovery(video_path=...)` — returns a `job_id` immediately.
3. `wait_for_job(job_id, timeout_seconds=...)` — call again if it returns still-running.
4. `describe_camera_motion(job_id)` — prompt-ready text.
5. `list_outputs(job_id)` — absolute path of the motion-proxy MP4 and the exports.

A solve takes tens of seconds to minutes, which is longer than many MCP clients allow
for a single tool call. That is why starting and waiting are separate: `wait_for_job`
is bounded by its own timeout and returns the current state rather than hanging.

## MCP tools

| tool | what it does |
|------|--------------|
| `check_environment` | What this machine can do, and which execution mode is in force. |
| `start_camera_motion_recovery` | Analyse, solve every shot and render the proxy. Returns a job id at once. |
| `start_video_analysis` | The cheap look only: shot cuts, 2D motion, complexity, routing. |
| `get_job_status` | Where a job is now. Instant; safe to poll. |
| `wait_for_job` | Block until the job finishes or the timeout expires. |
| `cancel_job` | Ask a running job to stop. |
| `get_camera_motion` | The full measured result: solver, confidence, observability, rotation, lens, moves, optional poses. |
| `describe_camera_motion` | Prompt-ready text (`prompt`, `technical` or `brief`). |
| `list_outputs` | Every produced file with absolute paths; verifies the MP4 with ffprobe. |
| `get_trajectory_samples` | Numeric per-frame rows inline, for agents without shell access. |
| `render_motion_proxy` | Re-render the MP4 in another style or size. Reuses the solved trajectory. |
| `list_jobs` | Recent jobs in this workspace. |
| `delete_job` | Delete a job directory. Not reversible. |

Resources: `camerapath://conventions` (coordinate system, units and what the numbers
mean — read this before interpreting poses), `camerapath://environment`, and
`camerapath://jobs`.

### Proxy styles

`proxy_style` accepts `motion_cage` (default, densest), `depth_poles`, `ground_grid`
and `minimal`. Denser scenes make the motion easier to read back; sparser ones are
less visually distracting as a reference. `ground_grid` measured best on a fast FPV
shot — a re-solve of the rendered proxy recovered the path shape to 8%, against
18–30% for the others.

For a vertical source, pass `match_source_aspect: true` so the proxy matches the
input's aspect instead of being letterboxed into 16:9.

## CLI

One JSON document on stdout per command; progress and logs on stderr.

```bash
export CPL_AGENT_RUNNER=local

./scripts/cpl env
./scripts/cpl run clip.mp4 --wait --describe          # the whole pipeline
./scripts/cpl run clip.mp4 --no-render --wait         # skip the Blender render
./scripts/cpl analyze clip.mp4 --wait                 # cuts and motion only
./scripts/cpl status <job_id>
./scripts/cpl wait <job_id> --timeout 600
./scripts/cpl motion <job_id>
./scripts/cpl describe <job_id> --style prompt
./scripts/cpl samples <job_id> --shot 0 --stride 5
./scripts/cpl outputs <job_id>
./scripts/cpl render <job_id> --proxy-style ground_grid --match-source-aspect
./scripts/cpl jobs --limit 10
./scripts/cpl rm <job_id>
```

`--pretty` (indented JSON) and `--quiet` (no stderr progress) are global flags, so
they go **before** the subcommand: `./scripts/cpl --pretty motion <job_id>`.

COLMAP's own per-image logging is turned down to warnings so it cannot bury the
progress lines (about 300 lines to six on a typical shot). `--log-level DEBUG`, or
setting `GLOG_minloglevel` yourself, keeps all of it.

### Exit codes

| code | meaning |
|-----:|---------|
| 0 | success |
| 1 | internal error |
| 2 | bad input path, unsupported media type, file too large |
| 3 | the job failed |
| 4 | no such job |
| 5 | timed out while waiting |
| 6 | the job was cancelled |
| 7 | environment cannot do it (e.g. no Blender for a render) |
| 8 | the backend was unreachable or is a different checkout |
| 9 | the job is busy or not in a valid state for that call |

## Reading the results honestly

- **Units.** Translation is `normalized` unless a scale is calibrated. It is never
  metres by default, and `scale.units` always says which.
- **Observability.** `translation.observable` can be `false` — on a pure pan there is
  no recoverable camera path, and the tool says so instead of inventing one. When it
  is false, use only the rotation and lens fields.
- **Confidence.** `confidence.level` is `high`, `medium` or `low` with `reasons`
  explaining the score. A low-confidence result is still returned; treat it as a hint.
- **Shots.** A multi-shot video yields one trajectory per shot, each with its own
  coordinate system. Positions from different shots are not comparable.
- **Timing.** Frame count and duration always match the source exactly.
- **Pose.** The recovered pose is the optical camera, not a drone body.

## Input path rules

Tools take a path to a video already on this machine; nothing is downloaded. Paths
resolve against the current directory (override with `CPL_AGENT_INPUT_CWD`) and must
carry a video extension. Set `CPL_AGENT_ALLOWED_INPUT_ROOTS` (a path-separated list)
to restrict input to specific directories — worth doing when the agent picking the
path is not the person running the server. Output files are only ever served from
inside the job directory.

## Limitations

- macOS on Apple Silicon only, and the render needs Blender installed.
- A dolly zoom still fails to solve; the tool reports low confidence rather than
  guessing a path.
- Handheld positional shake is only partly reproduced.
- The full list is in the [README](../README.md#known-limitations).
