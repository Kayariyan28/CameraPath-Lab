"""End-to-end dense evaluation against Blender ground truth.

    python benchmarks/eval_dense.py <scene> [--solver colmap|perceptual]

video -> shot/motion analysis -> geometry solve -> lens curve -> fusion ->
kinematics, compared with the true camera on EVERY source frame (not only the
solver's keyframes). This is the check that the dense trajectory — the thing that
actually drives the proxy render — is right, including its timing and speed.
"""
from __future__ import annotations

import argparse, json, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

import numpy as np

from app.geometry.alignment import align_and_measure
from app.geometry.conventions import blender_quat_to_camerapath
from app.geometry.intrinsics import build_lens_curve, initial_intrinsics
from app.geometry.rotations import quat_angular_distance
from app.models.schemas.trajectory import MotionFidelity, ScaleMode
from app.solvers.base import SolveContext
from app.tracking.shot_motion import analyze_shot_motion
from app.trajectory.fusion import fuse_lens_curve, fuse_trajectory
from app.trajectory.kinematics import compute_kinematics
from app.video.ffprobe import probe_frames, probe_video
from app.video.shots import detect_shots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--solver", default="auto", choices=["auto", "colmap", "perceptual"])
    args = ap.parse_args()

    clip = REPO / "benchmarks/synthetic" / f"{args.scene}.mp4"
    truth = json.loads((REPO / "benchmarks/synthetic" / f"{args.scene}.truth.json").read_text())
    t0 = time.time()
    info = probe_video(clip); frames, info = probe_frames(info)
    shots, _ = detect_shots(info, frames)
    shot = shots[0]
    motion = analyze_shot_motion(info, shot, frames, long_edge=1080, max_features=2400)
    intr, _ = initial_intrinsics(info, *motion.analysis_size)
    lens, lens_note = build_lens_curve(motion.motion_frames, intr, parallax_score=motion.parallax.score)
    ctx = SolveContext(
        info=info, shot=shot, frames_meta=frames, motion_frames=motion.motion_frames,
        intrinsics=intr, analysis_size=motion.analysis_size,
        parallax_score=motion.parallax.score, texture_score=motion.texture,
        work_dir=Path(tempfile.mkdtemp(prefix="cpl_dense_")), geometry_long_edge=2048, max_features=8000,
    )
    from app.solvers.colmap_solver import PyColmapBackend
    from app.solvers.perceptual_solver import PerceptualMatchBackend
    order = {"colmap": [PyColmapBackend], "perceptual": [PerceptualMatchBackend],
             "auto": [PyColmapBackend, PerceptualMatchBackend]}[args.solver]
    geo, why = None, ""
    for Backend in order:
        backend = Backend()
        ok, why = backend.suitable_for(ctx)
        if not ok:
            print(f"  {Backend.__name__} declined: {why[:120]}")
            continue
        geo = backend.estimate(ctx)
        if geo.succeeded:
            args.solver = Backend.__name__
            break
        print(f"  {Backend.__name__} failed: {geo.message[:120]}")
    if geo is None:
        print(json.dumps({"scene": args.scene, "solver": args.solver, "refused": why}))
        return 0
    print(f"[{args.scene}/{args.solver}] solve ok={geo.succeeded} anchors={geo.pose_count} "
          f"translation_observable={geo.translation_observable} :: {geo.message[:160]}")
    if not geo.succeeded:
        return 1

    fused_lens, fused_note = fuse_lens_curve(geo, lens, frames, shot, intr)
    poses = fuse_trajectory(
        geo, motion.motion_frames, frames, shot, fused_lens, intr, motion.analysis_size,
        fidelity=MotionFidelity.EXACT, translation_observable=geo.translation_observable,
    )
    kin = compute_kinematics(poses, ScaleMode.NORMALIZED)

    tmap = {t["frame"]: t for t in truth["frames"]}
    common = [p for p in poses if p.frame_index in tmap]
    ep = np.array([p.position for p in common]); eq = [np.array(p.quaternion) for p in common]
    rp = np.array([tmap[p.frame_index]["location"] for p in common])
    rq = [blender_quat_to_camerapath(np.array(tmap[p.frame_index]["quaternion_wxyz"])) for p in common]
    tf, err = align_and_measure(ep, eq, rp, rq, allow_scale=True)
    per = np.degrees([quat_angular_distance(tf.apply_quaternion(e), r) for e, r in zip(eq, rq)])

    # Rotation error in the SOLVER's gauge: align on its anchors only, so position
    # interpolation between anchors cannot move the alignment and leak into the
    # rotation score.
    from app.geometry.alignment import align_poses
    arow = [i for i, p in enumerate(common) if p.is_anchor]
    atf, _ = align_poses(ep[arow], [eq[i] for i in arow], rp[arow], [rq[i] for i in arow])
    rot_anchor_gauge = float(np.mean(np.degrees([quat_angular_distance(atf.apply_quaternion(e), r) for e, r in zip(eq, rq)])))

    # Timing (I1/I2): fused timestamps must be the source's.
    t_err = max(abs(p.timestamp - tmap[p.frame_index]["time"]) for p in common)

    # Speed profile: normalized per-frame speed vs truth, correlation + shape.
    true_speed = np.linalg.norm(np.diff(rp, axis=0), axis=1)
    est_speed = np.linalg.norm(np.diff(tf.apply(ep), axis=0), axis=1)
    speed_corr = float(np.corrcoef(true_speed, est_speed)[0, 1]) if true_speed.std() > 1e-9 and est_speed.std() > 1e-9 else float("nan")
    speed_rel = float(np.abs(est_speed - true_speed).sum() / max(true_speed.sum(), 1e-9))

    # Relative rotation error (RPE): frame-to-frame rotation vs truth. Alignment
    # free, so it measures the perceived rotational motion directly.
    rpe = [np.degrees(quat_angular_distance(
        __import__("app.geometry.rotations", fromlist=["quat_relative"]).quat_relative(eq[i - 1], eq[i]),
        __import__("app.geometry.rotations", fromlist=["quat_relative"]).quat_relative(rq[i - 1], rq[i])))
        for i in range(1, len(eq))]
    rpe_mean = float(np.mean(rpe)) if rpe else 0.0
    true_rate = float(np.mean([np.degrees(quat_angular_distance(rq[i - 1], rq[i])) for i in range(1, len(rq))]))

    fov_err = float(np.mean([abs(p.fov_horizontal - float(tmap[p.frame_index]["fov_horizontal"])) for p in common]))
    # Zoom ratio: what a pure zoom actually reveals (its absolute focal is not
    # observable). Ratio of tan(fov/2) first-to-last, estimate vs truth.
    tan_half = lambda f: np.tan(np.radians(f) / 2)  # noqa: E731
    true_ratio = tan_half(float(tmap[common[0].frame_index]["fov_horizontal"])) / tan_half(float(tmap[common[-1].frame_index]["fov_horizontal"]))
    est_ratio = tan_half(common[0].fov_horizontal) / tan_half(common[-1].fov_horizontal)
    zoom_ratio_err = float(est_ratio / true_ratio - 1.0)
    anchors = sum(p.is_anchor for p in common)
    # Acceptance (spec section 24): rotation is scored in the solver's gauge
    # (aligned on its anchors), shape on every dense position. Aligning rotation
    # on dense positions instead lets position interpolation leak into the
    # rotation score: on handheld it read 2.7 deg for orientations accurate to
    # 0.45 deg. Both numbers are reported.
    passed = rot_anchor_gauge < 2.0 and err.normalized_shape_error < 0.08 and t_err < 1e-6
    print(f"  dense {len(common)}/{len(truth['frames'])} frames ({anchors} anchors): rot MAE {err.rotation_mae_degrees:.3f} "
          f"deg (median {np.median(per):.3f}, max {per.max():.3f}; anchor-gauge {rot_anchor_gauge:.3f}), shape {err.normalized_shape_error*100:.2f}%, "
          f"RPE {rpe_mean:.3f} deg/frame (true rate {true_rate:.2f}), timing err {t_err:.2e}s, speed corr {speed_corr:.3f}, speed err {speed_rel*100:.1f}%, "
          f"FOV err {fov_err:.2f} deg (focal {'measured' if geo.focal_observable else 'prior'}), zoom ratio err {zoom_ratio_err*100:+.1f}%, kinematics {len(kin)} => {'PASS' if passed else 'FAIL'}  [{time.time()-t0:.0f}s]")
    print(f"  lens: {fused_note[:140]}")
    print(json.dumps({"scene": args.scene, "solver": args.solver, "rpe": rpe_mean, "true_rate": true_rate,
                      "translation_observable": geo.translation_observable, "anchors": int(anchors),
                      "rot_mae": err.rotation_mae_degrees, "rot_mae_anchor_gauge": rot_anchor_gauge,
                      "rot_max": float(per.max()), "shape_err": err.normalized_shape_error, "timing_err": t_err,
                      "speed_corr": speed_corr, "speed_err": speed_rel, "fov_err": fov_err,
                      "focal_observable": geo.focal_observable, "zoom_ratio_err": zoom_ratio_err, "passed": passed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
