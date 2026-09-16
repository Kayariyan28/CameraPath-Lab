# CameraPath Lab

[![CI](https://github.com/Kayariyan28/CameraPath-Lab/actions/workflows/ci.yml/badge.svg)](https://github.com/Kayariyan28/CameraPath-Lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)

**Recover the camera motion of a reference video and re-emit it as a clean 3D
motion reference.**

CameraPath Lab is a local macOS (Apple Silicon) app. You give it a video clip. It
reconstructs how the camera moved: trajectory, rotation, timing, speed and lens
(FOV/zoom). It also finds every hard cut. The output is a trajectory you can inspect
in 3D and an MP4 rendered by a Blender camera that carries exactly that motion
through neutral geometry. You can use that MP4 as a motion reference for
generative-video models such as Seedance.

The motion is kept and the content is dropped. No source pixels reach the exported
proxy.

> **Status: v0.1.0, early release.** The pipeline works end to end and is validated
> against synthetic ground truth. Some camera moves still fail or are weak. See
> [Known limitations](#known-limitations).

---

## Demo

A real session, recorded end to end: drop in a 14-second clip, analyse it, and generate
the motion reference. The long solve is time-lapsed and labelled; everything else runs
at normal speed.

![CameraPath Lab, from upload to rendered motion reference](docs/images/demo.gif)

*[Full-resolution recording (MP4, 43 s)](docs/media/camerapath-demo.mp4)*

The clip is three Blender shots joined by hard cuts — an FPV curve, an orbit and a
crane up. The app found both cuts at the exact frame and solved each shot on its own.

**Recovered camera trajectory** — shot 2, the orbit. Keyframe frustums along the path,
the live pose readout, per-frame kinematics, and the rendered proxy beside it:

![Recovered 3D camera trajectory for the orbit shot](docs/images/app-trajectory.png)

**Analysis, before any 3D solve** — shots and cuts, measured image motion, and the
per-shot routing decision:

![Shot detection and measured image motion](docs/images/app-analysis.png)

**Reference against proxy**, at the same timestamps:

![The reference video and the motion proxy at matching timestamps](docs/images/reference-vs-proxy.png)

The proxy reproduces the motion and the timing. It reproduces none of the content —
that is the point, since it is a motion reference, not a copy of the source.

## Features

- **Shot and cut detection.** Cuts are found with several cues: motion-compensated
  residuals, tracking collapse and histogram jumps. Each shot is solved on its own and
  the results are joined on the original timeline.
- **Camera solving through a fallback ladder.** Each method is tried in turn and the
  app says which one it used and why:
  - **COLMAP** structure-from-motion when parallax makes translation observable
  - long-baseline **keyframe homographies** for pans, tilts, rolls and zooms
  - screen-space **Perceptual Match** when feature matching fails
  - a labelled **2D motion proxy** as the last resort
- **Moving content is rejected.** People, cars and other moving objects are filtered
  out so they don't pull the camera solve.
- **Lens recovery.** Zoom is measured from K⁻¹HK and focal length is probed for
  observability. When the footage doesn't constrain focal length, the prior is
  labelled as a prior and never shown as a measurement.
- **Honest scale and confidence.** Translation is in *normalized* units unless you
  calibrate it. Confidence comes from evidence and can be LOW, with reasons given.
- **Exact timing.** Timestamps come from ffprobe PTS, and the output duration and frame
  count match the source.
- **Gravity-aligned world.** The world is oriented Z-up, and moves are classified as
  dolly, truck, pedestal, pan, tilt, roll, orbit, crane, and so on.
- **Web UI.** React and Three.js, with:
  - a 3D trajectory viewer
  - kinematics and motion curves
  - per-shot results and a solver audit log
- **Exports**
  - `trajectory.json` and `trajectory.csv` (per frame)
  - `analysis.json`
  - Nuke `.chan`
  - rendered motion-proxy MP4

## How it works

```
video ──► ffprobe timing ──► shot detection ──► per-shot optical flow + dynamic rejection
                                                        │
                        wide-baseline parallax measurement
                                                        │
          ┌─────────────────────────────────────────────┼──────────────────────────┐
   translation observable                        no parallax                 matching fails
          │                                             │                          │
   COLMAP SfM (focal probed,              rotation from keyframe            Perceptual Match
   misregistrations rejected)             homographies, position fixed      (screen-space)
          └─────────────────────────────────────────────┼──────────────────────────┘
                                                        ▼
            fusion: SQUAD through anchors + per-frame residual (gain chosen by held-out anchors)
                                                        ▼
            gravity alignment ─► classification ─► exports ─► Blender headless render ─► MP4
```

Two ideas carry most of the accuracy:

1. **Absolute quantities come from long baselines, and detail comes from per-frame
   motion.** Per-frame estimates get the shape right but drift when accumulated. On a
   real 900-frame shot, accumulating them produced a 24% phantom zoom. Keyframe
   geometry fixes the totals, and optical flow fills in between.
2. **Held-out anchors decide how much per-frame detail to trust.** The app doesn't use
   a fixed rule for this.

The full design, hard invariants and coordinate conventions are in
[docs/architecture.md](docs/architecture.md).

## Accuracy

The synthetic benchmark is rendered in Blender with known ground truth: 19 scenes, 60
frames at 30 fps, and every frame scored. **18 of 19 scenes pass.**

| Move type | Solver | Rotation error |
|-----------|--------|---------------:|
| dolly, truck, pedestal, crane, orbit, FPV curve, accelerate/decelerate | COLMAP | 0.006° – 0.098° |
| pan, tilt, roll, static, zoom | keyframe rotation | 0.001° – 0.013° |
| handheld | COLMAP | 0.457° (27% speed error, weak) |
| hard cut (orbit → pan) | per shot | 0.449° / 0.007°, cut found at the exact frame |
| **dolly zoom** | Perceptual fallback | **fails** |

Trajectory shape error is ≤ 0.42% on the passing translation scenes, and timing is
exact on every scene. The real test clips are locked-off footage with people walking
through, so the true camera motion is zero. On them the app measured 0.14° and 0.36° of
phantom rotation over 15 s, no phantom zoom and no false cuts. The full table is in
[docs/architecture.md §13](docs/architecture.md#13-validation-results).

## Requirements

- macOS on Apple Silicon (developed on an M4)
- Python 3.11
- Node.js 18+
- FFmpeg (`brew install ffmpeg`)
- [Blender](https://www.blender.org/download/) 5.x (tested with 5.1). It is optional:
  without it, analysis and exports still work, but you can't render the MP4 proxy.

## Quick start

```bash
git clone https://github.com/Kayariyan28/CameraPath-Lab.git
cd CameraPath-Lab
./scripts/bootstrap_macos.sh     # create .venv, install Python + Node deps, check tools
./scripts/dev.sh                 # start backend (:8848) and frontend (:5173)
```

Open http://localhost:5173 and follow these steps:

1. Drop in a video.
2. Click **Analyse video** to detect shots and measure motion.
3. Click **Generate motion reference**. This solves the camera, writes the trajectory
   exports and renders the MP4 proxy.

Other useful commands:

```bash
.venv/bin/python scripts/doctor.py          # what works on this machine, and what doesn't
./scripts/bootstrap_macos.sh --check        # report only, change nothing
```

## What it will and will not claim

- Monocular video doesn't give absolute scale, so translation is **normalized** unless
  you calibrate a real-world reference. It is never labelled as metres by default.
- A pure pan, a pure zoom, distant scenery and near-zero parallax are *degenerate* for
  structure-from-motion. In those cases the app says translation is not observable and
  holds the position fixed. It doesn't invent a camera path.
- The recovered pose is the **optical camera** pose. For drone footage with a gimbal,
  that is not the aircraft body pose.
- A solver failure degrades to the next rung of the ladder and never crashes the job.

## Known limitations

- **Dolly zoom fails.** A zooming lens leaves COLMAP's image pairs uncalibrated, so it
  cannot initialise. The app reports LOW confidence and doesn't invent a path.
- **Handheld positional shake** is only partly reproduced, because it is finer than
  the anchor spacing. Rotational shake is kept.
- **Gravity** is estimated from camera orientations, so a steadily pitched shot with no
  yaw can be about 8° off.
- **Same-location cuts** with almost no histogram change can be missed.
- **FOV** falls back to a labelled 60° prior when the footage doesn't constrain focal
  length. You can set **FOV override** in the UI.
- **Not built yet:**
  - VGGT learned-geometry backend
  - third-person trajectory preview video
  - diagnostic overlays on the source video
  - solver comparison view
  - variable-frame-rate resampling

## Project layout

```
backend/     FastAPI service: video I/O, shot detection, tracking, solvers, fusion, exports
frontend/    React + TypeScript + Vite + Three.js UI
blender/     scripts run inside Blender's Python (proxy render, synthetic scenes)
scripts/     bootstrap, dev runner, doctor, test-clip and benchmark generators
tests/       unit tests (test clips are generated on first run)
benchmarks/  evaluation scripts; rendered data is generated locally, not committed
docs/        architecture, invariants, validation results
```

## Testing

```bash
.venv/bin/python -m pytest                  # unit suite; generates test clips on first run
./scripts/render_synthetic.sh               # render Blender ground-truth scenes
./benchmarks/run_dense_suite.sh             # score the solver against them
```

## License

[MIT](LICENSE) © 2026 Karan Chandra Dey
