"""Experiment: which inter-anchor rotation fill matches ground truth?

    python benchmarks/fusion_variants.py <scene> [<scene> ...]

Runs analysis + COLMAP once per scene (cached to a pickle), then evaluates rotation
fill variants on every source frame. Not product code — it decides what fusion.py
should do.
"""
from __future__ import annotations

import json, pickle, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
import numpy as np

from app.geometry.alignment import align_poses
from app.geometry.conventions import blender_quat_to_camerapath
from app.geometry.intrinsics import build_lens_curve, initial_intrinsics
from app.geometry.rotations import (quat_angular_distance, quat_exp, quat_log, quat_multiply,
                                    quat_normalize, quat_relative, unroll_quaternions)
from app.models.schemas.trajectory import MotionFidelity
from app.solvers.base import SolveContext
from app.trajectory.fusion import fuse_lens_curve, fuse_trajectory
from app.trajectory.interpolation import resample_rotations

CACHE = REPO / "benchmarks/results/fusion_cache"


def prepare(scene):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{scene}.pkl"
    if path.is_file():
        return pickle.loads(path.read_bytes())
    from app.solvers.colmap_solver import PyColmapBackend
    from app.solvers.perceptual_solver import PerceptualMatchBackend
    from app.tracking.shot_motion import analyze_shot_motion
    from app.video.ffprobe import probe_frames, probe_video
    from app.video.shots import detect_shots
    clip = REPO / "benchmarks/synthetic" / f"{scene}.mp4"
    info = probe_video(clip); frames, info = probe_frames(info)
    shot = detect_shots(info, frames)[0][0]
    motion = analyze_shot_motion(info, shot, frames, long_edge=1080, max_features=2400)
    intr, _ = initial_intrinsics(info, *motion.analysis_size)
    lens, _ = build_lens_curve(motion.motion_frames, intr, parallax_score=motion.parallax.score)
    ctx = SolveContext(info=info, shot=shot, frames_meta=frames, motion_frames=motion.motion_frames,
                       intrinsics=intr, analysis_size=motion.analysis_size, parallax_score=motion.parallax.score,
                       texture_score=motion.texture, work_dir=Path(tempfile.mkdtemp(prefix="cpl_fv_")),
                       geometry_long_edge=2048, max_features=8000)
    backend = PyColmapBackend()
    if not backend.suitable_for(ctx)[0]:
        backend = PerceptualMatchBackend()
    geo = backend.estimate(ctx)
    data = dict(info=info, frames=frames, shot=shot, motion=motion.motion_frames, analysis_size=motion.analysis_size,
                intr=intr, lens=lens, geo=geo, solver=type(backend).__name__)
    path.write_bytes(pickle.dumps(data))
    return data


def hp(series, window):
    if window <= 1:
        return series
    k = np.ones(window) / window
    pad = window // 2
    padded = np.pad(series, ((pad, pad), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, c], k, mode="valid") for c in range(series.shape[1])], 1)
    return series - smooth[: len(series)]


def variants(d, draft_quats):
    frames = [f for f in d["frames"] if d["shot"].start_frame <= f.frame_index <= d["shot"].end_frame]
    times = np.array([f.time_seconds for f in frames]); idx = [f.frame_index for f in frames]
    row = {fi: i for i, fi in enumerate(idx)}
    geo = d["geo"]
    arows = np.array([row[f] for f in geo.frame_indices if f in row])
    aq = np.array([geo.quaternions[i] for i, f in enumerate(geo.frame_indices) if f in row])
    at = times[arows]
    squad = unroll_quaternions(resample_rotations(at, aq, times))
    # dense path exactly as the draft builds it
    import app.trajectory.fusion as F
    mbi = {m.frame_index: m for m in d["motion"]}
    w, h = d["analysis_size"]
    fx = np.full(len(idx), d["intr"].scaled_to(w, h).fx)
    inc, _ = F._dense_rotation_increments(mbi, idx, fx, fx)
    dense = F._integrate_rotation(inc)
    trend = resample_rotations(at, dense[arows], times)
    resid = np.array([quat_log(quat_relative(trend[i], dense[i])) for i in range(len(idx))])

    def with_residual(r):
        # keep anchors exact: remove the residual's value interpolated from anchor rows
        corr = np.stack([np.interp(times, at, r[arows, c]) for c in range(3)], 1)
        r = r - corr
        return unroll_quaternions(np.array([quat_normalize(quat_multiply(squad[i], quat_exp(r[i]))) for i in range(len(idx))]))

    def extrapolate(q):
        q = q.copy()
        lo, hi = arows.min(), arows.max()
        if len(arows) >= 2:
            a1, a0 = arows[np.argsort(arows)][-1], arows[np.argsort(arows)][-2]
            rate = quat_log(quat_relative(q[a0], q[a1])) / max(times[a1] - times[a0], 1e-9)
            for i in range(hi + 1, len(idx)):
                q[i] = quat_normalize(quat_multiply(q[a1], quat_exp(rate * (times[i] - times[a1]))))
            b0, b1 = arows[np.argsort(arows)][0], arows[np.argsort(arows)][1]
            rate = quat_log(quat_relative(q[b0], q[b1])) / max(times[b1] - times[b0], 1e-9)
            for i in range(0, lo):
                q[i] = quat_normalize(quat_multiply(q[b0], quat_exp(rate * (times[i] - times[b0]))))
        return q

    def loo_gain(window=5, gains=(0.0, 0.25, 0.5, 0.75, 1.0), margin=0.9):
        order = np.argsort(arows)
        rows_sorted, quats_sorted = arows[order], aq[order]
        scores = {}
        for g in gains:
            errs = []
            for k in range(1, len(rows_sorted) - 1):
                keep = np.array([j for j in range(len(rows_sorted)) if j != k])
                rk, tk, qk = rows_sorted[keep], times[rows_sorted[keep]], quats_sorted[keep]
                sq = resample_rotations(tk, qk, times)
                if g == 0.0:
                    pred = sq[rows_sorted[k]]
                else:
                    tr = resample_rotations(tk, dense[rk], times)
                    r = np.array([quat_log(quat_relative(tr[i], dense[i])) for i in range(len(idx))])
                    r = hp(r, window)
                    corr = np.stack([np.interp(times, tk, r[rk, c]) for c in range(3)], 1)
                    r = r - corr
                    pred = quat_multiply(sq[rows_sorted[k]], quat_exp(g * r[rows_sorted[k]]))
                errs.append(np.degrees(quat_angular_distance(pred, quats_sorted[k])))
            scores[g] = float(np.mean(errs))
        best = min(scores, key=scores.get)
        if best > 0 and scores[best] > margin * scores[0.0]:
            best = 0.0
        return best, scores

    g_best, g_scores = loo_gain()
    print(f"   LOO held-out anchor error by gain: " + ", ".join(f"{g}:{v:.3f}" for g, v in g_scores.items()) + f"  -> chose {g_best}")

    out = {
        "V3 LOO gain + extrapolate": extrapolate(with_residual(hp(resid, 5) * g_best)),
        "V0 draft (dense gain 1)": np.array(draft_quats),
        "V1 squad anchors only": squad,
        "V1e squad + extrapolate": extrapolate(squad),
        "V2(5) squad + hp5 resid": with_residual(hp(resid, 5)),
        "V2(9) squad + hp9 resid": with_residual(hp(resid, 9)),
        "V2e(5) hp5 + extrapolate": extrapolate(with_residual(hp(resid, 5))),
    }
    return out, idx, times, arows


def score(scene, d):
    truth = json.loads((REPO / "benchmarks/synthetic" / f"{scene}.truth.json").read_text())
    tmap = {t["frame"]: t for t in truth["frames"]}
    poses = fuse_trajectory(d["geo"], d["motion"], d["frames"], d["shot"], d["lens"], d["intr"],
                            d["analysis_size"], fidelity=MotionFidelity.EXACT,
                            translation_observable=d["geo"].translation_observable)
    vs, idx, times, arows = variants(d, [p.quaternion for p in poses])
    rq = [blender_quat_to_camerapath(np.array(tmap[f]["quaternion_wxyz"])) for f in idx]
    # align ONCE using anchors (the gauge is the solver's), then measure every variant in that frame
    tf, _ = align_poses(np.array(d["geo"].positions), list(d["geo"].quaternions),
                        np.array([tmap[f]["location"] for f in d["geo"].frame_indices]),
                        [blender_quat_to_camerapath(np.array(tmap[f]["quaternion_wxyz"])) for f in d["geo"].frame_indices])
    lo, hi = arows.min(), arows.max()
    inside = np.array([lo <= i <= hi for i in range(len(idx))])
    # truth high-frequency angular increments (for jitter preservation)
    def incs(qs):
        return np.array([quat_log(quat_relative(qs[i - 1], qs[i])) for i in range(1, len(qs))])
    t_hf = hp(incs(rq), 5)
    print(f"\n== {scene} [{d['solver']}]  anchors={len(arows)} range {idx[lo]}..{idx[hi]} of {idx[0]}..{idx[-1]}")
    print(f"   {'variant':28s} {'MAE all':>8s} {'inside':>7s} {'outside':>8s} {'max':>7s} {'jitter corr':>12s}")
    for name, q in vs.items():
        aligned = [tf.apply_quaternion(x) for x in q]
        e = np.degrees([quat_angular_distance(a, r) for a, r in zip(aligned, rq)])
        e_hf = hp(incs(aligned), 5)
        num = float((e_hf * t_hf).sum()); den = float(np.sqrt((e_hf ** 2).sum() * (t_hf ** 2).sum()))
        jc = num / den if den > 1e-12 else float("nan")
        print(f"   {name:28s} {e.mean():8.3f} {e[inside].mean():7.3f} "
              f"{(e[~inside].mean() if (~inside).any() else 0):8.3f} {e.max():7.3f} {jc:12.3f}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        score(s, prepare(s))
