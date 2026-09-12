# CameraPath Lab — Architecture

**Goal.** Given a reference video, recover the *optical camera motion* — trajectory,
rotation, timing, speed, and lens/FOV changes — and re-emit it as an MP4 of neutral
3D geometry filmed by a Blender camera carrying that motion. The camera move is the
product. The source imagery is never reproduced.

**Non-goal.** Reconstructing the appearance, content, geometry or environment of the
source video. No source pixels reach the exported proxy.

---

## 0. Hard engineering invariants

These are load-bearing. Violating any of them makes the output a lie, not a
reconstruction. Each maps to a test in `tests/`.

| # | Invariant | Enforced by |
|---|-----------|-------------|
| I1 | Source timing is preserved. Output duration differs from source by ≤ 1 output frame. | `trajectory/kinematics.py`, `tests/unit/test_timing.py` |
| I2 | Timestamps come from the container (PTS), not `index / assumed_fps`, whenever the container provides them. | `video/ffprobe.py`, `video/decoder.py` |
| I3 | Rotations are never interpolated as Euler angles. Quaternions + SLERP/SQUAD only. | `geometry/rotations.py`, `trajectory/interpolation.py` |
| I4 | A single trajectory never spans a hard cut. Each shot is an independent coordinate system. | `video/shots.py`, `workers/pipeline.py` |
| I5 | Monocular translation is labelled `normalized` unless the user calibrates a scale. Never labelled metres by default. | `trajectory/normalize.py`, `scale_mode` in every export |
| I6 | Optical flow is *image-space motion*, never reported as physical translation. | `tracking/global_motion.py` type separation |
| I7 | Confidence is computed from evidence and is allowed to be LOW. No fabricated certainty. | `validation/confidence.py` |
| I8 | Every world/camera convention change is an explicit, named, unit-tested function. | `geometry/conventions.py`, `tests/unit/test_conventions.py` |
| I9 | Solver decisions (which backend ran, why, why it was rejected) are logged and returned to the UI. | `core/logging.py`, `SolverDecision` records |
| I10 | Default motion fidelity is EXACT. Handheld jitter present in the source survives to the output. | `trajectory/fidelity.py` |
| I11 | The recovered pose is the *optical camera* pose, not a drone body pose. Labelled as such. | schema field naming, UI copy |
| I12 | A solver failure degrades the result; it never crashes the app. | `solvers/base.py` fallback ladder |

---

## 1. Runtime topology

```
┌──────────────────────────────┐
│  Frontend (Vite + React TS)  │  localhost:5173
│  Three.js trajectory viewer  │
└──────────────┬───────────────┘
               │  REST + SSE  (/api proxied by Vite)
┌──────────────▼───────────────┐
│  FastAPI (uvicorn)           │  localhost:8848
│  ├ api/        thin routes   │
│  ├ workers/    pipeline      │  in-process ThreadPool; one job = one worker
│  └ core/       env + paths   │
└──────────────┬───────────────┘
               │  subprocess
    ┌──────────┴──────────┬─────────────────┐
    ▼                     ▼                 ▼
ffprobe/ffmpeg      Blender 5.1.1       pycolmap (in-proc)
(decode, encode)    (--background)      (SfM)
```

Everything is local. No network egress at runtime. Blender is invoked headless as a
subprocess so a Blender crash cannot take down the API.

---

## 2. Pipeline stages

The pipeline is a sequence of **cached, independently re-runnable stages**. Each stage
writes its artefact into the job directory and records a fingerprint of its inputs.
Changing a downstream-only setting (e.g. proxy scene style) re-runs only the tail.

```
 1 PROBE          ffprobe → VideoInfo, FrameMetadata[]          cache: video hash
 2 DECODE         ffmpeg → analysis-resolution gray frames      cache: (hash, max_res)
 3 SHOTS          cut detection → Shot[]                        cache: (hash, params)
 ── per shot, independent coordinate system ──────────────────────────────────
 4 MOTION         per-frame-transition MotionFrame[]            cache: (shot, params)
 5 REJECT         background/dynamic track confidence           cache: (shot, params)
 6 INTRINSICS     focal prior + estimate → LensFrame[]          cache: (shot, params)
 7 GEOMETRY       keyframe SfM (COLMAP / VGGT) → anchor poses   cache: (shot, solver)
 8 FUSE           anchors + dense motion → per-frame CameraPose cache: (7, fidelity)
 9 VALIDATE       confidence + metrics                          cache: (8)
── per job ───────────────────────────────────────────────────────────────────
10 SCENE          Blender scene build + camera keyframes        cache: (8, style)
11 RENDER         motion_proxy.mp4 / trajectory_preview.mp4     cache: (10, output)
12 EXPORT         trajectory.json/.csv, analysis.json, .blend
```

Stage 4 runs on **every frame transition**. Stage 7 runs only on **selected keyframes**.
That asymmetry is the core design decision: global 3D structure is expensive and sparse,
fine temporal texture is cheap and dense, and the product needs both (§5).

---

## 3. Module map

```
backend/app/
  core/environment.py    chip, unified memory, MPS, Blender, FFmpeg, pycolmap probe;
                         derives adaptive window/batch sizes from *measured* free memory
  core/paths.py          per-job workspace, atomic artefact writes, safe GC of old jobs
  core/logging.py        structured stage + SolverDecision log, streamed to UI

  video/ffprobe.py       authoritative container facts; PTS extraction; VFR detection
  video/decoder.py       ffmpeg → ndarray frames at analysis resolution (no full dump)
  video/shots.py         multi-cue cut detection (§4)

  tracking/features.py       Shi-Tomasi / SIFT extraction
  tracking/flow.py           pyramidal Lucas-Kanade, forward-backward validated
  tracking/global_motion.py  dominant 2D model per transition → MotionFrame
  tracking/dynamic_rejection.py  residual-flow clustering → background confidence

  geometry/conventions.py  ALL coordinate-frame changes (COLMAP↔CV↔Blender). Tested.
  geometry/rotations.py    quaternion algebra, SLERP/SQUAD, log/exp maps
  geometry/intrinsics.py   focal prior, FOV, lens curve
  geometry/alignment.py    Sim(3) / Umeyama alignment for window merge + benchmarks

  solvers/base.py          GeometryBackend protocol + fallback ladder
  solvers/opencv_solver.py     essential-matrix / homography incremental pose
  solvers/colmap_solver.py     pycolmap SfM (primary geometric backend)
  solvers/vggt_solver.py       optional learned geometry, MPS with CPU fallback
  solvers/perceptual_solver.py screen-space motion matching (§6)

  trajectory/interpolation.py  Hermite (translation) + SQUAD (rotation)
  trajectory/fusion.py         anchors + dense motion → per-frame poses
  trajectory/kinematics.py     velocity, accel, angular rates, curvature, jerk
  trajectory/normalize.py      normalized ↔ metric scale modes
  trajectory/classify.py       post-hoc motion labelling (never drives the solve)
  trajectory/fidelity.py       EXACT / CLEAN / SMOOTH

  validation/confidence.py     HIGH/MEDIUM/LOW + human-readable reasons
  validation/metrics.py        reprojection error, inlier %, solver agreement

  blender/runner.py        headless Blender subprocess driver

blender/                   scripts executed *inside* Blender's Python (not importable
                           by the backend — Blender ships its own interpreter)
  render_motion_proxy.py   neutral motion cage + animated camera → MP4
  render_trajectory_preview.py  third-person path visualisation
  gen_synthetic_scene.py   ground-truth test renders (§8)
```

---

## 4. Shot detection

A single physical camera trajectory must never span an edit (I4). But fast camera
movement looks a lot like a cut to any single cue, so detection fuses four:

| Cue | Signal | Fires on a cut | Fires on fast motion |
|-----|--------|----------------|----------------------|
| HSV histogram correlation drop | global colour statistics | yes | sometimes |
| Frame structural change | downsampled SAD / SSIM | yes | sometimes |
| Feature-match collapse | LK track survival ratio | yes, hard | **no** — tracks survive motion |
| Flow coherence collapse | inlier ratio of dominant model | yes | **no** — motion stays coherent |

Fast camera movement produces *large but coherent* flow with surviving tracks. A cut
produces *incoherent* flow with total track death. The match-collapse and
flow-coherence cues are therefore weighted highest, and a cut requires agreement
across cues. This is the guard against the classic false positive.

---

## 5. Fusion: sparse anchors + dense temporal texture

The central problem: SfM gives globally consistent poses but only on keyframes, and
naive interpolation between keyframes destroys exactly what this product exists to
preserve — acceleration, jitter, micro-movement, direction changes.

Approach:

1. **Anchors.** Geometric poses on registered keyframes, each with a confidence.
2. **Dense motion.** `MotionFrame` for every transition: image-space translation,
   rotation, scale, radial flow, divergence, curl, inlier ratio.
3. **Rotation.** Anchor orientations drive the global path; per-frame image-space
   rotation and flow-derived yaw/pitch increments supply the inter-anchor shape.
   Composed in quaternion space, then SQUAD through the anchors so anchors are hit
   exactly and the inter-anchor curve carries the measured high-frequency content.
4. **Translation.** Anchors are interpolated with a Hermite spline whose tangents come
   from the *measured* flow magnitude profile, not from uniform parameterisation. A
   2-second deceleration in the source stays a 2-second deceleration.
5. **Residual.** The dense signal's high-pass component is added back at a gain set by
   Motion Fidelity (EXACT = 1.0). This is what keeps handheld handheld (I10).

> Never a plain `lerp` between anchors, and never Euler interpolation (I3).

---

## 6. Perceptual Motion Match

Geometric SfM is *degenerate*, not merely inaccurate, for pure rotation, pure zoom,
distant scenery, and near-zero parallax. In those cases translation is genuinely
unobservable and a physical solver will happily invent a large fake baseline.

Perceptual mode instead treats the problem as inverse rendering of *motion only*:

```
minimise  Σ_t ‖ signature(render(camera_θ, canonical_scene), t) − signature(source, t) ‖²
          over θ = (yaw, pitch, roll, tx, ty, tz, focal) per frame
```

The signature is the same `MotionFrame` feature vector used everywhere else, so source
and candidate are compared in identical units. The result is *not* a claim about the
physical camera path — it is a camera whose screen-space motion matches the reference.
That is exactly what a downstream generative-video model consumes.

AUTO selects this when geometric confidence is LOW, and says so in the UI with the
reason (I7).

---

## 7. Scale

Monocular video does not determine absolute scale. Two explicit modes:

- **`normalized`** (default): first camera at the origin, trajectory scaled so total
  path length lands in a useful Blender range (~10–25 units). Relative translation
  ratios, all timing, and rotations are preserved exactly.
- **`metric`**: requires an explicit user calibration (known height, known distance
  between two points, or known travelled distance). Only then is the unit `m`.

`scale_mode` is a required field on every export. Normalized translation is never
labelled metres (I5).

---

## 8. Synthetic ground truth

Correctness claims about a camera solver are worthless without ground truth, so
Blender generates its own: `blender/gen_synthetic_scene.py` renders a textured static
scene with a *known* camera animation and writes the true poses alongside the MP4.

16 cases: dolly fwd/back, pan, tilt, roll, orbit, crane, diagonal fly-through, FPV
curve, accelerating, decelerating, handheld, zoom-only, dolly-zoom, moving foreground
object, hard cut.

Estimated vs. true trajectories are aligned with Sim(3) (Umeyama) before measuring,
because monocular scale is ambiguous. Reported: rotation MAE, normalized
trajectory-shape error, timing error, FOV error, registered-frame %.

Initial acceptance targets for textured static scenes: rotation MAE < 2°, shape error
< 8% post-alignment, duration delta ≤ 1 frame. Degenerate cases (pure pan, zoom-only)
assert the *opposite*: that the app reports low translation observability and switches
to Perceptual Match rather than inventing a baseline (§25 of the spec).

Failures are reported, never suppressed.

---

## 9. Coordinate systems

Three conventions are in play. Confusing them is the single most likely source of a
silently wrong result, so each conversion is one named function with a unit test (I8).

| Frame | Camera looks along | Up | Notes |
|-------|--------------------|----|-------|
| **OpenCV / COLMAP camera** | `+Z` | `−Y` | right-handed; COLMAP stores world→cam |
| **CameraPath world** | — | `+Z` | right-handed, Z-up; our canonical interchange |
| **Blender camera** | `−Z` | `+Y` | right-handed, Z-up world |

COLMAP gives `(q_cw, t_cw)` mapping world→camera. Camera centre in world is
`C = −R_cwᵀ · t_cw`. Getting this backwards yields a trajectory that is mirrored and
inverted — plausible-looking and completely wrong. Hence `tests/unit/test_conventions.py`
round-trips every pair.

See `docs/coordinate_systems.md` for the full derivation.

---

## 10. Adaptive resource policy

Nothing assumes a fixed RAM figure. `core/environment.py` measures total and *currently
free* unified memory and derives:

- analysis resolution cap (long edge)
- decode chunk size
- VGGT temporal window length and stride
- COLMAP feature budget
- Blender render tile/thread hints

This machine (M4 Max, 48 GB) lands on generous settings; an 8 GB M1 Air lands on
conservative ones from the same code path.

---

## 11. Phase plan

| Phase | Content | Status |
|-------|---------|--------|
| 1 | React shell, FastAPI, upload, ffprobe, decode, shot detect, optical flow, motion curves | **this milestone** |
| 2 | pycolmap SfM, pose extraction, trajectory JSON, Three.js viewer, synthetic validation | next |
| 3 | Blender headless motion cage, animated camera, MP4 render — usable end-to-end | next |
| 4 | Perceptual Match, zoom/FOV estimation, confidence system, dynamic rejection | later |
| 5 | Optional VGGT, MPS, sliding windows, pose fusion, BA refinement | later |
| 6 | Performance, UX polish, full test suite | later |

Phase 5 is deliberately last: a learned backend is impossible to evaluate without the
deterministic baseline and the synthetic ground-truth harness already in place.
