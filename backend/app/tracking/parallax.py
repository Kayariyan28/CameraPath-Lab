"""Parallax estimation over wide baselines.

Parallax is *the* deciding evidence for whether camera translation is
observable, which makes it the single most consequential measurement in the
system: it routes a shot between physical reconstruction and screen-space
matching, and getting it wrong is wrong in both directions — fabricating a dolly
that was never there, or refusing to reconstruct one that was.

The original implementation measured it between adjacent frames and was wrong
for a fundamental reason: **parallax requires baseline**. A camera dollying at
0.37 m/frame has almost no baseline between consecutive frames, so consecutive
views really are related by a homography to within noise. Measured on a
synthetic forward dolly with 22 m of genuine translation through a depth-varied
scene, adjacent-frame analysis reported a parallax score of 0.008 and routed the
shot away from 3D reconstruction. Between frames 0 and 30 of the same clip there
is 11 m of baseline and the parallax is unmistakable.

So this module works on *wide-baseline* pairs, drawn from tracks that survive
across a gap, and compares two hypotheses:

  H (homography, 8 DoF)  — fits if the scene is planar, distant, or the camera
                           only rotated and zoomed
  F (fundamental, 7 DoF) — fits any rigid scene under any camera motion,
                           including depth variation

When F explains correspondences that H cannot, the extra structure is depth, and
depth-dependent flow means translation is real. When H explains everything, it
is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.logging import get_logger

log = get_logger("tracking.parallax")

#: Accumulated image motion a baseline should span before parallax is judged, px.
#: Below roughly this, H and F are statistically indistinguishable and the
#: comparison measures noise.
TARGET_BASELINE_FLOW_PX = 22.0

#: Minimum correspondences for a meaningful H-vs-F comparison. F has 7 degrees
#: of freedom; fitting it to a handful of points overfits and always "wins".
MIN_CORRESPONDENCES = 40

#: RANSAC thresholds, px at analysis resolution.
H_THRESHOLD = 2.5
F_THRESHOLD = 1.5

#: Epipolar geometry must fit THIS well before any conclusion about depth is
#: drawn. Necessary because F has more freedom than H and will always fit at
#: least as many points: what matters is whether it fits WELL in absolute terms.
#: When it does not, the correspondences themselves are unreliable and the
#: measurement is inconclusive — which is a different statement from "no
#: parallax" and must not be reported as one.
#: Measured: a genuine 3D dolly reaches F=0.95, while a 145 px/frame whip pan
#: over a flat scene reaches only F=0.64 because its tracking has degraded.
MIN_FUNDAMENTAL_INLIERS = 0.80

#: A homography must leave at least this fraction of the motion unexplained.
#: This is the primary, physically meaningful condition: parallax IS the part of
#: the flow a homography cannot represent. Measured: real parallax leaves 43% of
#: the motion unexplained; pure 2D pans, rolls and zooms leave 0.2-2.0%.
MIN_RESIDUAL_RATIO = 0.03


@dataclass
class ParallaxEvidence:
    """What the wide-baseline comparison found."""

    score: float = 0.0
    """0-1 confidence that depth-dependent flow (hence translation) is present."""

    pair_count: int = 0
    mean_baseline_px: float = 0.0
    homography_inlier_ratio: float = 0.0
    fundamental_inlier_ratio: float = 0.0
    homography_residual_px: float = 0.0
    """Median symmetric transfer error of the best homography."""
    residual_ratio: float = 0.0
    """Homography residual as a fraction of the baseline motion. Scale-free."""
    translation_observable: bool = False

    measurement_reliable: bool = False
    """False when no baseline could be measured, or when epipolar geometry fitted
    the correspondences too poorly to conclude anything. "Inconclusive" and "no
    parallax" are different findings and the UI must not conflate them (I7)."""

    reason: str = ""
    per_pair_scores: list[float] = field(default_factory=list)


class TrackSnapshotRecorder:
    """Records track positions at anchor frames to build wide-baseline pairs.

    Reuses the tracker's persistent tracks rather than re-matching: a track alive
    from frame i to frame j already *is* a wide-baseline correspondence, verified
    frame by frame along the way. That is both cheaper and more reliable than
    matching frames i and j directly.
    """

    def __init__(self, target_baseline_px: float = TARGET_BASELINE_FLOW_PX):
        self.target = target_baseline_px
        self._snapshots: list[tuple[int, dict[int, tuple[float, float]]]] = []
        self._accumulated = 0.0

    def observe(self, frame_index: int, track_ids: np.ndarray,
                points: np.ndarray, flow_magnitude: float) -> None:
        """Offer a frame. A snapshot is taken once enough motion has accrued."""
        if not self._snapshots:
            self._store(frame_index, track_ids, points)
            return
        self._accumulated += max(flow_magnitude, 0.0)
        if self._accumulated >= self.target:
            self._store(frame_index, track_ids, points)
            self._accumulated = 0.0

    def _store(self, frame_index: int, track_ids: np.ndarray, points: np.ndarray) -> None:
        if len(track_ids) == 0:
            return
        self._snapshots.append((
            frame_index,
            {int(t): (float(p[0]), float(p[1]))
             for t, p in zip(track_ids.tolist(), points)},
        ))
        # Bound memory on a long shot; consecutive pairs are what matter.
        if len(self._snapshots) > 64:
            self._snapshots.pop(0)

    def pairs(self) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
        """Correspondences for each consecutive snapshot pair."""
        out: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for (idx_a, map_a), (idx_b, map_b) in zip(self._snapshots, self._snapshots[1:]):
            shared = map_a.keys() & map_b.keys()
            if len(shared) < MIN_CORRESPONDENCES:
                continue
            ordered = sorted(shared)
            src = np.array([map_a[t] for t in ordered], dtype=np.float32)
            dst = np.array([map_b[t] for t in ordered], dtype=np.float32)
            out.append((idx_a, idx_b, src, dst))
        return out

    @property
    def snapshot_count(self) -> int:
        return len(self._snapshots)


def _symmetric_transfer_error(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    def apply(mat: np.ndarray, pts: np.ndarray) -> np.ndarray:
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        out = homo @ mat.T
        w = out[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        return out[:, :2] / w

    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return np.full(len(src), 1e6)
    return (
        np.linalg.norm(apply(h, src) - dst, axis=1)
        + np.linalg.norm(apply(h_inv, dst) - src, axis=1)
    ) * 0.5


def analyze_pair(src: np.ndarray, dst: np.ndarray) -> tuple[float, dict]:
    """Parallax score for one wide-baseline pair, plus diagnostics."""
    diag = {
        "baseline_px": 0.0, "h_inliers": 0.0, "f_inliers": 0.0,
        "h_residual": 0.0, "residual_ratio": 0.0, "count": len(src),
        "inconclusive": False,
    }
    if len(src) < MIN_CORRESPONDENCES:
        return 0.0, diag

    baseline = float(np.median(np.linalg.norm(dst - src, axis=1)))
    diag["baseline_px"] = baseline
    if baseline < 3.0:
        # Too little motion to distinguish anything.
        return 0.0, diag

    h, h_mask = cv2.findHomography(
        src, dst, method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=H_THRESHOLD, maxIters=5000, confidence=0.999,
    )
    if h is None or h_mask is None:
        h_ratio, h_residual = 0.0, baseline
    else:
        h_ratio = float(h_mask.ravel().mean())
        h_residual = float(np.median(_symmetric_transfer_error(h, src, dst)))
    diag["h_inliers"] = h_ratio
    diag["h_residual"] = h_residual

    f, f_mask = cv2.findFundamentalMat(
        src, dst, method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=F_THRESHOLD, confidence=0.999, maxIters=5000,
    )
    f_ratio = float(f_mask.ravel().mean()) if f is not None and f_mask is not None else 0.0
    diag["f_inliers"] = f_ratio

    # Scale-free: how large is the unexplained-by-H residual compared with the
    # motion itself? An absolute pixel threshold fails on slow motion, which is
    # exactly how the original implementation missed a genuine 22 m dolly.
    residual_ratio = h_residual / max(baseline, 1e-6)
    diag["residual_ratio"] = residual_ratio

    # --- gate 1: are the correspondences trustworthy at all? ---------------
    # F always fits at least as many points as H (more degrees of freedom), so
    # the H-vs-F gap is only meaningful when F itself fits well. A low F ratio
    # means the tracking is bad, not that the scene is flat.
    if f_ratio < MIN_FUNDAMENTAL_INLIERS:
        diag["inconclusive"] = True
        return 0.0, diag

    # --- gate 2: does a homography actually fail to explain the motion? -----
    # This is the physical definition of parallax: the component of the flow a
    # homography cannot represent. Judged relative to the motion, because an
    # absolute pixel threshold is meaningless across slow and fast shots.
    if residual_ratio < MIN_RESIDUAL_RATIO:
        return 0.0, diag

    # Beyond the gates, score on how badly the homography fails, corroborated by
    # how much more of the scene epipolar geometry explains.
    residual_term = float(np.clip((residual_ratio - MIN_RESIDUAL_RATIO) / 0.12, 0.0, 1.0))
    advantage_term = float(np.clip((f_ratio - h_ratio) / 0.25, 0.0, 1.0))

    score = float(np.clip(0.70 * residual_term + 0.30 * advantage_term, 0.0, 1.0))
    return score, diag


def summarize(pairs: list[tuple[int, int, np.ndarray, np.ndarray]]) -> ParallaxEvidence:
    """Aggregate wide-baseline pairs into one verdict for the shot."""
    evidence = ParallaxEvidence()
    if not pairs:
        evidence.reason = (
            "no wide-baseline correspondences survived — too little camera motion, "
            "or tracks did not persist long enough to measure depth structure"
        )
        return evidence

    scores: list[float] = []
    baselines: list[float] = []
    h_ratios: list[float] = []
    f_ratios: list[float] = []
    residuals: list[float] = []
    ratios: list[float] = []
    inconclusive = 0

    for _, _, src, dst in pairs:
        score, diag = analyze_pair(src, dst)
        if diag["baseline_px"] < 3.0:
            continue
        if diag["inconclusive"]:
            inconclusive += 1
        scores.append(score)
        baselines.append(diag["baseline_px"])
        h_ratios.append(diag["h_inliers"])
        f_ratios.append(diag["f_inliers"])
        residuals.append(diag["h_residual"])
        ratios.append(diag["residual_ratio"])

    if not scores:
        evidence.reason = (
            "camera motion over every available baseline was too small to measure "
            "depth structure"
        )
        return evidence

    evidence.per_pair_scores = scores
    evidence.pair_count = len(scores)
    # Median rather than mean: one badly-tracked pair should not decide the
    # routing of a whole shot.
    evidence.score = float(np.median(scores))
    evidence.mean_baseline_px = float(np.mean(baselines))
    evidence.homography_inlier_ratio = float(np.mean(h_ratios))
    evidence.fundamental_inlier_ratio = float(np.mean(f_ratios))
    evidence.homography_residual_px = float(np.median(residuals))
    evidence.residual_ratio = float(np.median(ratios))
    # Reliable only if most baselines produced a usable epipolar fit.
    evidence.measurement_reliable = inconclusive <= len(scores) // 2
    evidence.translation_observable = (
        evidence.score >= 0.15 and evidence.measurement_reliable
    )

    if not evidence.measurement_reliable:
        evidence.reason = (
            f"parallax could not be measured reliably: epipolar geometry fitted only "
            f"{evidence.fundamental_inlier_ratio:.0%} of correspondences across "
            f"{evidence.pair_count} baselines, so the tracking is too degraded to tell "
            "depth structure from tracking error. Translation is treated as "
            "unobservable because it was not demonstrated, not because it is absent"
        )
    elif evidence.translation_observable:
        evidence.reason = (
            f"depth-dependent flow measured over {evidence.pair_count} baselines of "
            f"~{evidence.mean_baseline_px:.0f} px: a homography leaves "
            f"{evidence.residual_ratio:.0%} of the motion unexplained and fits "
            f"{evidence.homography_inlier_ratio:.0%} of points against "
            f"{evidence.fundamental_inlier_ratio:.0%} for epipolar geometry — "
            "translation is observable"
        )
    else:
        evidence.reason = (
            f"no measurable parallax: a single homography explains the motion over "
            f"{evidence.pair_count} "
            f"baselines of ~{evidence.mean_baseline_px:.0f} px "
            f"({evidence.homography_inlier_ratio:.0%} of points, residual "
            f"{evidence.residual_ratio:.1%} of the motion) — the scene is planar, "
            "distant, or the camera only rotated and zoomed, so translation is not "
            "observable"
        )

    log.info(
        "parallax: score=%.3f over %d baselines (~%.0f px), H=%.2f F=%.2f "
        "residual=%.2f px (%.1f%% of motion)",
        evidence.score, evidence.pair_count, evidence.mean_baseline_px,
        evidence.homography_inlier_ratio, evidence.fundamental_inlier_ratio,
        evidence.homography_residual_px, evidence.residual_ratio * 100,
    )
    return evidence
