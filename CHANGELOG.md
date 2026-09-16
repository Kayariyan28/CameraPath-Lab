# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Agent integration.** The pipeline is now usable as a tool, not just through the
  UI, via one shared service layer (`backend/app/agent/`):
  - an **MCP server** (`scripts/cpl-mcp`, stdio by default, optional streamable-HTTP)
    with 13 tools covering the whole flow — environment check, start, poll, wait,
    cancel, measured motion, prompt-ready description, outputs, trajectory samples,
    re-render, list and delete — plus resources describing the coordinate
    conventions, the environment and the job list
  - a **JSON CLI** (`scripts/cpl`) that prints exactly one JSON document to stdout,
    keeps progress on stderr, and maps error classes to distinct exit codes
  - a `local` / `delegate` / `auto` execution model: jobs run in-process, or are
    handed to a backend that is already running, and land in the same workspace
    either way, so any agent-created job opens in the UI at `?job=<id>`
  - [docs/agents.md](docs/agents.md) with client configuration, the tool reference
    and how to read the results without over-claiming
- Long-running work is split into start and bounded wait, because a solve outlasts
  the tool-call timeout most MCP clients allow.

### Fixed

- The delegated runner assumed the backend on the port shares this checkout's
  workspace. A second clone serving that port silently accepted uploads and wrote
  them elsewhere, surfacing much later as an unrecognisable missing path. The
  mismatch is now detected at the first write and reported with both paths.

## [0.1.0] - 2026-09-14

First public release. The pipeline works end to end, from reference video to camera
trajectory to rendered motion-proxy MP4, and is validated against synthetic Blender
ground truth.

### Added

- **Video ingest**
  - ffprobe PTS timing, so output duration and frame count match the source exactly
  - streaming ffmpeg decode at analysis resolution
- **Shot detection**
  - motion-compensated residual cue with neighbour dominance
  - second path for same-location 3D cuts (tracking collapse plus histogram jump)
- **Tracking**
  - Lucas-Kanade optical flow with a forward-backward check
  - dynamic-object rejection using rigid-scene residuals, coherence, stuck tracks and
    a frame-rate-independent drift accumulator
  - wide-baseline parallax measurement to decide whether translation is observable
- **Solver ladder** (a solver failure degrades to the next rung and never crashes the
  job)
  - COLMAP (pycolmap) structure-from-motion, with a focal observability probe, a
    divergence guard and misregistered-image rejection
  - OpenCV rotation backend, which gets rotation from long-baseline SIFT keyframe
    homographies (pan, tilt, roll, zoom)
  - Perceptual Match, a screen-space fallback
  - 2D motion proxy as the last resort, with LOW confidence
- **Fusion**
  - SQUAD rotation interpolation through anchors
  - per-frame residual at a gain chosen by held-out anchors
  - Hermite translation timed by source timestamps
- **Lens**
  - zoom factor from K⁻¹HK
  - focal length from a chained-homography structure cost
  - labelled FOV prior and UI override
- **World frame**: gravity (Z-up) estimation, rank-aware Sim(3) alignment, and move
  classification (dolly, truck, pedestal, pan, tilt, roll, orbit, crane, and so on)
- **Confidence**: scored from evidence, with reasons
- **Exports**
  - `trajectory.json` and `trajectory.csv` (per frame, per shot)
  - `analysis.json`
  - Nuke `.chan`
- **Blender**
  - headless motion-proxy render with ffprobe verification of the output MP4
  - per-shot renders joined on the original timeline
  - synthetic ground-truth scene generator (19 camera moves)
- **Web UI** (React, TypeScript, Vite, Three.js)
  - upload
  - 3D trajectory viewer
  - kinematics and motion curves
  - per-shot results
  - solver audit log
  - shareable `?job=` links
- **Tooling**
  - `bootstrap_macos.sh`, `dev.sh`, `doctor.py`
  - test-clip and synthetic-render generators
  - dense benchmark suite
  - 302 unit tests

### Validation

- 18 of 19 synthetic scenes pass.
- Rotation error is 0.001°–0.098° on all passing scenes except handheld (0.457°).
- Trajectory shape error is ≤ 0.42% on the passing translation scenes, and timing is
  exact on every scene.
- On real locked-off footage, phantom rotation is 0.14° and 0.36° over 15 s, with no
  phantom zoom.

### Known limitations

- Dolly zoom fails: COLMAP cannot initialise with a zooming lens.
- Handheld positional shake is only partly reproduced (27% speed error).
- Gravity can be about 8° off on a steadily pitched shot with no yaw.
- Same-location cuts with almost no histogram change can be missed.
- VGGT backend, trajectory preview video, diagnostic overlays, solver comparison and
  variable-frame-rate resampling are not built yet.

[0.1.0]: https://github.com/Kayariyan28/CameraPath-Lab/releases/tag/v0.1.0
