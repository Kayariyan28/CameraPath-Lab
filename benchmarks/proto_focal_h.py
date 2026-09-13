"""Prototype: focal from chained homographies via a zoom-invariant structure test."""
import sys, json, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, "backend")
from app.video.ffprobe import probe_video, probe_frames
from app.video.shots import detect_shots
from app.tracking.shot_motion import analyze_shot_motion

def chains(mfs, min_inliers=0.6, max_len=20):
    out, cur, n = [], np.eye(3), 0
    for mf in mfs:
        if mf.homography is None or mf.inlier_ratio < min_inliers:
            if n >= 3: out.append(cur)
            cur, n = np.eye(3), 0; continue
        h = np.array(mf.homography).reshape(3, 3)
        cur = h @ cur; n += 1
        if n >= max_len:
            out.append(cur); cur, n = np.eye(3), 0
    if n >= 3: out.append(cur)
    return out

def structure_cost(hs, f, w, h):
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]); Ki = np.linalg.inv(K)
    c = 0.0
    for H in hs:
        M = Ki @ H @ K
        S = M @ M.T
        S = S / (np.trace(S) / 3)
        c += 2 * (S[0, 1] ** 2 + S[0, 2] ** 2 + S[1, 2] ** 2) + (S[0, 0] - S[1, 1]) ** 2
    return c / max(len(hs), 1)

def estimate(mfs, w, h):
    hs = chains(mfs)
    if not hs: return None, None, 0
    fovs = np.linspace(15, 130, 231)
    fs = (w / 2) / np.tan(np.radians(fovs) / 2)
    costs = np.array([structure_cost(hs, f, w, h) for f in fs])
    i = int(np.argmin(costs)); f = fs[i]
    lo, hi = structure_cost(hs, f * 0.85, w, h), structure_cost(hs, f * 1.15, w, h)
    sharp = min(lo, hi) / (costs[i] + 1e-12)
    return f, sharp, len(hs)

cache = Path("benchmarks/results/proto_motion"); cache.mkdir(parents=True, exist_ok=True)
for scene in sys.argv[1:]:
    p = cache / f"{scene}.pkl"
    if p.is_file():
        mfs, (w, h) = pickle.loads(p.read_bytes())
    else:
        info = probe_video(f"benchmarks/synthetic/{scene}.mp4"); fr, info = probe_frames(info)
        m = analyze_shot_motion(info, detect_shots(info, fr)[0][0], fr, long_edge=1080, max_features=2400)
        mfs, (w, h) = m.motion_frames, m.analysis_size
        p.write_bytes(pickle.dumps((mfs, (w, h))))
    f, sharp, nch = estimate(mfs, w, h)
    truth = json.load(open(f"benchmarks/synthetic/{scene}.truth.json"))["frames"][0]["fov_horizontal"]
    fov = np.degrees(2 * np.arctan((w / 2) / f)) if f else float("nan")
    print(f"{scene:14s} chains {nch:2d}  FOV {fov:6.2f} (truth {truth:.2f}, err {fov-truth:+.2f})  sharpness {sharp:10.2f}")
