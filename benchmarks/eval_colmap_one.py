"""Run motion analysis + COLMAP on one synthetic clip and score against truth.

    python benchmarks/eval_colmap_one.py <scene_name>
"""
import json, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np
from app.geometry.alignment import align_and_measure
from app.geometry.conventions import blender_quat_to_camerapath
from app.geometry.intrinsics import initial_intrinsics
from app.solvers.base import SolveContext
from app.solvers.colmap_solver import PyColmapBackend
from app.tracking.shot_motion import analyze_shot_motion
from app.video.ffprobe import probe_frames, probe_video
from app.video.shots import detect_shots

scene = sys.argv[1] if len(sys.argv) > 1 else "dolly_forward"
clip = REPO / "benchmarks/synthetic" / f"{scene}.mp4"
truth = json.loads((REPO / "benchmarks/synthetic" / f"{scene}.truth.json").read_text())

info = probe_video(clip); frames, info = probe_frames(info)
shots, _ = detect_shots(info, frames)
motion = analyze_shot_motion(info, shots[0], frames, long_edge=1080, max_features=2400)
intr, prov = initial_intrinsics(info, *motion.analysis_size)
print(f"[{scene}] shots={len(shots)} flow={motion.signature.mean_flow_magnitude:.2f}px "
      f"parallax={motion.parallax.score:.2f} texture={motion.texture:.2f}")

ctx = SolveContext(
    info=info, shot=shots[0], frames_meta=frames, motion_frames=motion.motion_frames,
    intrinsics=intr, analysis_size=motion.analysis_size,
    parallax_score=motion.parallax.score, texture_score=motion.texture,
    work_dir=Path(tempfile.mkdtemp(prefix="cpl_colmap_")),
    geometry_long_edge=2048, max_features=8000,
)
backend = PyColmapBackend()
ok, why = backend.suitable_for(ctx)
print(f"suitable={ok} {why}")
if not ok:
    # Declining is the correct outcome for a degenerate shot (spec section 25).
    print(json.dumps({"scene": scene, "refused": True, "reason": why}))
    sys.exit(0)
t0 = time.time(); res = backend.estimate(ctx)
print(f"COLMAP {time.time()-t0:.1f}s succeeded={res.succeeded} :: {res.message}")
print(f"  registered {res.registered_frames}/{res.total_frames} conf={res.confidence:.2f} "
      f"reproj={res.mean_reprojection_error} focal={res.focal_pixels}")
true_fov = float(truth["frames"][0]["fov_horizontal"])
if true_fov < 3.2:  # stored in radians by Blender
    true_fov = float(np.degrees(true_fov))
fov_err = None
if res.focal_pixels:
    # COLMAP focal is in the pixels of the images it was given: the source width,
    # capped at geometry_long_edge — NOT the flow-analysis width.
    w = res.focal_image_width
    est_fov = float(np.degrees(2 * np.arctan(w / (2 * res.focal_pixels))))
    fov_err = est_fov - true_fov
    tag = "MEASURED" if res.focal_observable else "PRIOR (unobservable)"
    sens = "n/a" if res.focal_sensitivity is None else f"{res.focal_sensitivity:.2%}"
    print(f"  FOV {est_fov:.2f} deg [{tag}, probe rise {sens}] vs true {true_fov:.2f} deg "
          f"(error {fov_err:+.2f} deg)")

if not (res.succeeded and res.pose_count >= 3):
    sys.exit(1)

tmap = {t["frame"]: t for t in truth["frames"]}
rp, rq, ep, eq = [], [], [], []
for i, fi in enumerate(res.frame_indices):
    t = tmap.get(fi)
    if t is None:
        continue
    rp.append(t["location"]); rq.append(blender_quat_to_camerapath(np.array(t["quaternion_wxyz"])))
    ep.append(res.positions[i]); eq.append(res.quaternions[i])
tf, err = align_and_measure(np.array(ep), eq, np.array(rp), rq, allow_scale=True)
from app.geometry.rotations import quat_angular_distance
per_pose = [float(np.degrees(quat_angular_distance(tf.apply_quaternion(e), r))) for e, r in zip(eq, rq)]
rot_median = float(np.median(per_pose))
passed = err.rotation_mae_degrees < 2.0 and err.normalized_shape_error < 0.08
print(f"  vs truth ({err.sample_count} poses, align={err.alignment_method}): rot MAE "
      f"{err.rotation_mae_degrees:.3f} deg, median {rot_median:.3f}, "
      f"max {err.rotation_max_degrees:.3f}, shape err {err.normalized_shape_error*100:.2f}%, "
      f"path ratio {err.path_length_ratio:.3f}  => {'PASS' if passed else 'FAIL'}")
print(json.dumps({"scene": scene, "work_dir": str(ctx.work_dir), "fov_err": fov_err, "focal_observable": res.focal_observable,
                  "rot_median": rot_median, "rot_mae": err.rotation_mae_degrees,
                  "shape_err": err.normalized_shape_error, "registered": res.registered_frames,
                  "passed": passed}))
