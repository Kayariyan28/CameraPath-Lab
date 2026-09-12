# CameraPath Lab

Local macOS / Apple Silicon tool that recovers the **camera motion** of a
reference video and re-emits it as an MP4 of neutral 3D geometry filmed by a
Blender camera carrying that motion — a motion reference for generative-video
workflows such as Seedance.

It reconstructs trajectory, rotation, timing, speed and lens changes. It does
**not** reproduce the appearance, content or environment of the source. No source
pixels reach the exported proxy.

## Quick start

```bash
./scripts/bootstrap_macos.sh     # check/install local dependencies
./scripts/dev.sh                 # start backend + frontend
```

Then open http://localhost:5173.

## What it will and will not claim

- Monocular video does not determine absolute scale, so translation is reported
  in **normalized** units unless you calibrate a real-world reference. It is
  never labelled metres by default.
- Pure pan, pure zoom, distant scenery and near-zero parallax are *degenerate*
  for structure-from-motion. In those cases the app says translation is not
  observable and switches to screen-space Perceptual Match rather than inventing
  a camera path.
- The recovered pose is the **optical camera** pose. For drone footage with an
  articulated gimbal that is not the aircraft body pose.
- Confidence is computed from evidence and is allowed to be LOW, with reasons.

See [docs/architecture.md](docs/architecture.md) for the design, the hard
invariants, and the phase plan.

## Layout

```
backend/    FastAPI service, pipeline, solvers   (Python 3.11)
frontend/   React + TypeScript + Vite + Three.js
blender/    scripts run inside Blender's own Python
scripts/    bootstrap, dev runner, doctor
tests/      unit / integration / synthetic ground truth
benchmarks/ generated test clips and results (not committed)
workspace/  per-job scratch (not committed)
```
