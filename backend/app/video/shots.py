"""Hard-cut detection.

A trajectory must never span an edit (invariant I4), so this runs before any
reconstruction. The hard part is not finding cuts — it is *not* finding them in
fast camera movement, which produces an appearance change just as large as a cut.

The discriminator is coherence, not magnitude:

    fast camera motion  ->  large flow, but COHERENT: tracks survive, and one
                            geometric model explains most of them
    a real cut          ->  large flow, INCOHERENT: tracks die wholesale and no
                            single model fits what remains

So appearance cues (histogram, structural) are necessary but never sufficient; a
cut additionally requires track collapse. Both are judged against a local
baseline, because a clip that is uniformly fast has a high floor and a clip that
is uniformly static has a very low one.

This pass runs at a deliberately small resolution (~256 px long edge) — cut
detection needs gross change, not detail, and staying small keeps it nearly free.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.logging import StageReporter, get_logger
from app.models.schemas.video import (
    CutCandidate,
    FrameMetadata,
    Shot,
    ShotComplexity,
    VideoInfo,
)
from app.video.decoder import FrameDecoder

log = get_logger("video.shots")

#: Long edge for the cut-detection decode pass.
CUT_ANALYSIS_LONG_EDGE = 256

#: Grid for spatially-aware histograms. A global histogram is blind to a cut
#: between two shots that happen to share a palette (very common: same location,
#: different angle), which a 4x4 grid catches.
HIST_GRID = 4
HIST_BINS = 32


@dataclass
class CutThresholds:
    """Tuning knobs, all as ratios against a local baseline rather than absolutes."""

    appearance_ratio: float = 3.2
    """Appearance change must exceed this multiple of the local median."""

    appearance_floor: float = 0.22
    """...and this absolute floor, so noise in a static shot cannot trip it."""

    max_track_survival: float = 0.42
    """A cut requires track survival BELOW this. Fast motion stays well above."""

    max_flow_coherence: float = 0.55
    """A cut requires model-inlier coherence BELOW this."""

    min_shot_frames: int = 8
    """Non-maximum suppression window; also the minimum emitted shot length."""


def _grid_histogram(frame: np.ndarray) -> np.ndarray:
    """Concatenated per-cell intensity histograms, L1-normalised per cell."""
    h, w = frame.shape[:2]
    cells = []
    for gy in range(HIST_GRID):
        for gx in range(HIST_GRID):
            y0, y1 = gy * h // HIST_GRID, (gy + 1) * h // HIST_GRID
            x0, x1 = gx * w // HIST_GRID, (gx + 1) * w // HIST_GRID
            cell = frame[y0:y1, x0:x1]
            hist = cv2.calcHist([cell], [0], None, [HIST_BINS], [0, 256]).ravel()
            total = hist.sum()
            cells.append(hist / total if total > 0 else hist)
    return np.concatenate(cells).astype(np.float32)


def _appearance_scores(prev: np.ndarray, cur: np.ndarray,
                       prev_hist: np.ndarray, cur_hist: np.ndarray) -> tuple[float, float]:
    """(histogram_change, structural_change), both roughly 0..1."""
    # Bhattacharyya distance is better behaved than correlation near-zero bins.
    hist_change = float(cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_BHATTACHARYYA))
    diff = cv2.absdiff(prev, cur)
    structural = float(diff.mean()) / 255.0
    return hist_change, structural


def _track_cues(prev: np.ndarray, cur: np.ndarray, max_corners: int = 220
                ) -> tuple[float, float]:
    """(track_survival_ratio, flow_coherence).

    survival  — fraction of corners that LK could still follow, forward-backward
                validated. Collapses to ~0 across a cut because the content the
                corners describe no longer exists.
    coherence — fraction of surviving tracks explained by a single affine model.
                Stays high under fast motion (one model explains it) and falls
                across a cut (surviving "matches" are coincidences).
    """
    corners = cv2.goodFeaturesToTrack(
        prev, maxCorners=max_corners, qualityLevel=0.01, minDistance=6, blockSize=7
    )
    if corners is None or len(corners) < 12:
        # Untrackable content (sky, water, extreme blur). No evidence either way:
        # report neutral values so appearance cues alone cannot declare a cut.
        return 1.0, 1.0

    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.02))
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, corners, None, **lk)
    if nxt is None or status is None:
        return 0.0, 0.0
    back, status_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **lk)
    if back is None or status_b is None:
        return 0.0, 0.0

    ok = (status.ravel() == 1) & (status_b.ravel() == 1)
    if not ok.any():
        return 0.0, 0.0
    fb_error = np.linalg.norm(corners[ok] - back[ok], axis=-1).ravel()
    # Forward-backward consistency: a genuine track returns to where it started.
    good = fb_error < 2.0
    survival = float(good.sum()) / float(len(corners))

    src = corners[ok][good].reshape(-1, 2)
    dst = nxt[ok][good].reshape(-1, 2)
    if len(src) < 8:
        return survival, 0.0

    _, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        maxIters=900, confidence=0.985,
    )
    coherence = float(inliers.ravel().mean()) if inliers is not None else 0.0
    return survival, coherence


def _local_median(values: np.ndarray, index: int, half_window: int = 12) -> float:
    lo = max(0, index - half_window)
    hi = min(len(values), index + half_window + 1)
    window = np.delete(values[lo:hi], min(index - lo, hi - lo - 1))
    if window.size == 0:
        return float(np.median(values)) if values.size else 0.0
    return float(np.median(window))


def detect_shots(
    info: VideoInfo,
    frames_meta: list[FrameMetadata],
    *,
    thresholds: CutThresholds | None = None,
    reporter: StageReporter | None = None,
) -> tuple[list[Shot], list[CutCandidate]]:
    """Split a video into continuous shots.

    Returns the shots plus every evaluated boundary (accepted or not), so the UI
    can explain a decision — including "this looked like a cut but the tracks
    survived, so it was kept as one shot".
    """
    th = thresholds or CutThresholds()
    n = len(frames_meta)
    if n == 0:
        return [], []
    if n < 3:
        return [_whole_video_shot(info, frames_meta)], []

    decoder = FrameDecoder(info, long_edge=CUT_ANALYSIS_LONG_EDGE, gray=True)

    hist_changes = np.zeros(n, dtype=np.float32)
    structurals = np.zeros(n, dtype=np.float32)
    survivals = np.ones(n, dtype=np.float32)
    coherences = np.ones(n, dtype=np.float32)

    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    seen = 0

    start_time = frames_meta[0].time_seconds if frames_meta else 0.0
    for idx, frame in decoder.iter_frames(
        start_frame=0, end_frame=n - 1, start_time=start_time
    ):
        # Light blur: suppresses codec blocking and sensor noise that would
        # otherwise inflate the structural cue on static shots.
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hist = _grid_histogram(frame)

        if prev_frame is not None and 0 <= idx < n:
            hc, sc = _appearance_scores(prev_frame, frame, prev_hist, hist)
            sv, co = _track_cues(prev_frame, frame)
            hist_changes[idx] = hc
            structurals[idx] = sc
            survivals[idx] = sv
            coherences[idx] = co

        prev_frame, prev_hist = frame, hist
        seen += 1
        if reporter and seen % 25 == 0:
            reporter.progress(min(1.0, seen / max(1, n)), f"cut scan {seen}/{n}")

    # ---------------------------------------------------------- decide cuts
    appearance = np.maximum(hist_changes, structurals * 2.0)
    candidates: list[CutCandidate] = []
    cut_frames: list[int] = []

    for i in range(1, n):
        app = float(appearance[i])
        baseline = _local_median(appearance, i)
        ratio = app / (baseline + 1e-4)

        survival = float(survivals[i])
        coherence = float(coherences[i])

        appearance_says_cut = app >= th.appearance_floor and ratio >= th.appearance_ratio
        tracks_collapsed = survival <= th.max_track_survival
        flow_incoherent = coherence <= th.max_flow_coherence

        # Both families of evidence are required. This is the whole guard against
        # calling a whip pan a cut.
        accepted = appearance_says_cut and tracks_collapsed and flow_incoherent

        if accepted:
            reason = (
                f"appearance {app:.3f} = {ratio:.1f}x local baseline, "
                f"track survival {survival:.0%}, model coherence {coherence:.0%}"
            )
        elif appearance_says_cut and not tracks_collapsed:
            reason = (
                f"rejected: large appearance change ({ratio:.1f}x baseline) but "
                f"{survival:.0%} of tracks survived — fast camera motion, not a cut"
            )
        elif appearance_says_cut and not flow_incoherent:
            reason = (
                f"rejected: appearance changed but a single model still explains "
                f"{coherence:.0%} of tracks — coherent motion, not a cut"
            )
        else:
            reason = "no significant appearance change"

        combined = float(
            min(1.0, ratio / max(th.appearance_ratio, 1e-6))
            * (1.0 - survival)
            * (1.0 - coherence)
        )

        if accepted or appearance_says_cut:
            candidates.append(
                CutCandidate(
                    frame_index=i,
                    time_seconds=frames_meta[i].time_seconds,
                    histogram_score=float(hist_changes[i]),
                    structural_score=float(structurals[i]),
                    match_collapse_score=1.0 - survival,
                    flow_coherence_score=coherence,
                    combined_score=combined,
                    accepted=accepted,
                    reason=reason,
                )
            )
        if accepted:
            cut_frames.append(i)

    # Non-maximum suppression: a real cut often flags its neighbour too
    # (the frame after a cut is also unlike the frame before it).
    suppressed: list[int] = []
    for f in cut_frames:
        if suppressed and f - suppressed[-1] < th.min_shot_frames:
            continue
        suppressed.append(f)

    shots = _build_shots(info, frames_meta, suppressed, th.min_shot_frames)

    if reporter:
        accepted_n = sum(1 for c in candidates if c.accepted)
        rejected_n = len(candidates) - accepted_n
        reporter.info(
            f"shot detection: {len(shots)} shot(s), {accepted_n} cut(s) accepted"
            + (f", {rejected_n} appearance spike(s) rejected as camera motion"
               if rejected_n else "")
        )
    log.info("detected %d shots from %d frames (%d cuts)", len(shots), n, len(suppressed))
    return shots, candidates


def _whole_video_shot(info: VideoInfo, frames_meta: list[FrameMetadata]) -> Shot:
    return Shot(
        id=0,
        start_frame=frames_meta[0].frame_index,
        end_frame=frames_meta[-1].frame_index,
        start_time=frames_meta[0].time_seconds,
        end_time=frames_meta[-1].time_seconds,
        confidence=1.0,
    )


def _build_shots(
    info: VideoInfo,
    frames_meta: list[FrameMetadata],
    cut_frames: list[int],
    min_shot_frames: int,
) -> list[Shot]:
    n = len(frames_meta)
    boundaries = [0, *cut_frames, n]
    shots: list[Shot] = []
    for shot_id, (start, end_exclusive) in enumerate(zip(boundaries, boundaries[1:])):
        end = end_exclusive - 1
        if end < start:
            continue
        if (end - start + 1) < min_shot_frames and shots:
            # Too short to solve independently; absorb into the previous shot
            # rather than emitting a degenerate 3-frame "trajectory".
            prev = shots[-1]
            shots[-1] = Shot(
                id=prev.id,
                start_frame=prev.start_frame,
                end_frame=end,
                start_time=prev.start_time,
                end_time=frames_meta[end].time_seconds,
                confidence=prev.confidence * 0.9,
            )
            continue
        shots.append(
            Shot(
                id=shot_id,
                start_frame=frames_meta[start].frame_index,
                end_frame=frames_meta[end].frame_index,
                start_time=frames_meta[start].time_seconds,
                end_time=frames_meta[end].time_seconds,
                confidence=1.0 if start == 0 else 0.9,
            )
        )
    # Re-index so ids are contiguous after any absorption.
    return [s.model_copy(update={"id": i}) for i, s in enumerate(shots)]


def estimate_complexity(
    frame_count: int,
    duration: float,
    mean_flow: float,
    inlier_ratio: float,
    texture_score: float,
    parallax_score: float,
) -> tuple[ShotComplexity, list[str]]:
    """Cheap pre-solve difficulty estimate, shown in the analysis panel.

    This is an *expectation* of solver difficulty, not a confidence score — the
    real confidence comes from `validation/confidence.py` after solving.
    """
    reasons: list[str] = []
    score = 0.0

    if mean_flow > 26:
        score += 2.0
        reasons.append(f"very fast image motion ({mean_flow:.0f} px/frame)")
    elif mean_flow > 12:
        score += 1.0
        reasons.append(f"fast image motion ({mean_flow:.0f} px/frame)")
    elif mean_flow < 0.7:
        reasons.append("near-static framing")

    if texture_score < 0.25:
        score += 2.0
        reasons.append("low texture — few reliable features to track")
    elif texture_score < 0.5:
        score += 1.0
        reasons.append("moderate texture")

    if inlier_ratio < 0.5:
        score += 1.5
        reasons.append(f"only {inlier_ratio:.0%} of tracks agree with a single model")

    if parallax_score < 0.15:
        score += 1.5
        reasons.append(
            "little parallax — translation may not be geometrically observable"
        )

    if duration > 0 and frame_count / max(duration, 1e-6) > 90:
        reasons.append("high frame rate")
    if frame_count < 16:
        score += 1.0
        reasons.append(f"very short shot ({frame_count} frames)")

    if score >= 5.0:
        level = ShotComplexity.EXTREME
    elif score >= 3.5:
        level = ShotComplexity.HIGH
    elif score >= 2.0:
        level = ShotComplexity.MODERATE
    elif score >= 0.75:
        level = ShotComplexity.LOW
    else:
        level = ShotComplexity.TRIVIAL

    if not reasons:
        reasons.append("well-textured, moderate motion — straightforward to solve")
    return level, reasons
