"""Experiment: how should translation be timed between anchors?

    python benchmarks/position_variants.py <scene> ...

Compares the fused positions (speed-warped Hermite, parameterised by image flow
magnitude) against plain time-parameterised Hermite through the same anchors,
scoring per-frame SPEED and path shape against ground truth.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend")); sys.path.insert(0, str(REPO / "benchmarks"))
import numpy as np
from fusion_variants import prepare
from app.geometry.alignment import align_poses
from app.geometry.conventions import blender_quat_to_camerapath
from app.models.schemas.trajectory import MotionFidelity
from app.trajectory.fusion import fuse_trajectory
from app.trajectory.interpolation import hermite_resample, speed_warped_parameterization
import app.trajectory.fusion as F


def evaluate(scene: str) -> None:
    d = prepare(scene)
    if d["solver"] != "PyColmapBackend":
        print(f"== {scene}: {d['solver']} (no translation) — skipped"); return
    truth = json.loads((REPO / "benchmarks/synthetic" / f"{scene}.truth.json").read_text())
    tm = {t["frame"]: t for t in truth["frames"]}
    geo = d["geo"]
    poses = fuse_trajectory(geo, d["motion"], d["frames"], d["shot"], d["lens"], d["intr"], d["analysis_size"],
                            fidelity=MotionFidelity.EXACT, translation_observable=geo.translation_observable)
    idx = [p.frame_index for p in poses]
    times = np.array([p.timestamp for p in poses])
    fused = np.array([p.position for p in poses])
    arow = [idx.index(f) for f in geo.frame_indices if f in idx]
    anchors = fused[arow]
    timed = hermite_resample(times[arow], anchors, times)
    # extrapolate like fusion does beyond the anchor range (constant velocity) for a fair comparison
    lo, hi = min(arow), max(arow)
    for i in range(hi + 1, len(idx)):
        v = (anchors[-1] - anchors[-2]) / (times[arow[-1]] - times[arow[-2]])
        timed[i] = anchors[-1] + v * (times[i] - times[arow[-1]])

    # --- held-out anchor choice of warp weight --------------------------------
    mbi = {m.frame_index: m for m in d["motion"]}
    speed_profile = F._dense_speed_profile(mbi, idx, times)
    warp = speed_warped_parameterization(times, speed_profile)
    tnorm = (times - times[0]) / max(times[-1] - times[0], 1e-9)
    def blended(w):
        return (1 - w) * tnorm + w * warp
    order = sorted(range(len(arow)), key=lambda k: arow[k])
    rows_s = [arow[k] for k in order]; anc_s = anchors[order]
    scores = {}
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        prm = blended(w); errs = []
        for k in range(1, len(rows_s) - 1):
            keep = [j for j in range(max(0, k - 3), min(len(rows_s), k + 4)) if j != k]
            pred = hermite_resample(prm[[rows_s[j] for j in keep]], anc_s[keep], prm[[rows_s[k]]])[0]
            spacing = np.linalg.norm(anc_s[k + 1] - anc_s[k - 1]) + 1e-12
            errs.append(np.linalg.norm(pred - anc_s[k]) / spacing)
        scores[w] = float(np.mean(errs))
    best = min(scores, key=scores.get)
    if best > 0 and scores[best] > 0.9 * scores[0.0]:
        best = 0.0
    chosen = hermite_resample(blended(best)[arow], anchors, blended(best))
    for i in range(hi + 1, len(idx)):
        v = (anchors[-1] - anchors[-2]) / (times[arow[-1]] - times[arow[-2]])
        chosen[i] = anchors[-1] + v * (times[i] - times[arow[-1]])
    print("   held-out position error by warp weight: " + ", ".join(f"{w}:{e:.4f}" for w, e in scores.items()) + f" -> {best}")

    ref = np.array([tm[f]["location"] for f in idx])
    rq = [blender_quat_to_camerapath(np.array(tm[f]["quaternion_wxyz"])) for f in idx]
    # One gauge for both variants: fit on the anchors.
    tf, _ = align_poses(anchors, [np.array(geo.quaternions[i]) for i in range(len(arow))], ref[arow],
                        [rq[r] for r in arow])
    true_speed = np.linalg.norm(np.diff(ref, axis=0), axis=1) / np.diff(times)
    print(f"== {scene}: anchors {len(arow)} over {len(idx)} frames; true speed {true_speed.min():.2f}..{true_speed.max():.2f} m/s")
    for name, pos in (("flow-warped (current)", fused), ("time-parameterised", timed), ("held-out choice", chosen)):
        a = tf.apply(pos)
        speed = np.linalg.norm(np.diff(a, axis=0), axis=1) / np.diff(times)
        rel = float(np.abs(speed - true_speed).mean() / max(true_speed.mean(), 1e-9))
        shape = float(np.sqrt(np.mean(np.sum((a - ref) ** 2, axis=1))) / max(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum(), 1e-9))
        corr = float(np.corrcoef(speed, true_speed)[0, 1]) if true_speed.std() > 1e-6 * true_speed.mean() else float("nan")
        wobble = float(np.std(speed) / max(np.mean(speed), 1e-9))
        print(f"   {name:24s} speed err {rel*100:5.1f}%  speed CV {wobble*100:5.1f}% (truth {np.std(true_speed)/np.mean(true_speed)*100:5.1f}%)  "
              f"corr {corr:6.3f}  shape {shape*100:.3f}%")


for s in sys.argv[1:]:
    evaluate(s)
