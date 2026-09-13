"""Hard-cut detection.

A trajectory must never span an edit (invariant I4), so this runs before any
reconstruction. The hard part is not finding cuts — it is *not* finding them in
fast camera movement, which changes a frame's appearance as much as an edit does.

The primary cue is **motion-compensated residual**: align the previous frame to
the current one with the best global similarity transform, then measure what is
left over.

    fast camera motion  ->  one transform explains the whole frame, so almost
                            nothing is left over, however large the motion
    a real cut          ->  no transform explains anything, so the residual
                            stays near the raw frame difference

That cue was chosen over histogram comparison after measuring both. Histogram
difference fails in two common situations, and both are represented in the test
clips:

  * a cut between two shots of the same place (reverse angle, same lighting)
    barely changes the histogram at all — measured appearance ratio 1.11x against
    the local baseline, far below any usable threshold
  * a whip pan changes the histogram as much as a cut does

Measured separation for the compensated residual, at 384 px analysis resolution:
a real cut scores 0.21 absolute residual while every non-cut transition across
nine clips — including a 145 px/frame sinusoidal whip pan, a pure roll and a
pure zoom — stays at or below 0.007.

The second cue is **immediate-neighbour dominance**. A cut is a one-frame
discontinuity between two well-aligned neighbours; fast motion degrades a whole
run of frames. So the residual at a cut must dominate the frames either side of
it, not merely be large.

The comparison is against the immediate neighbours (+/-2 frames) rather than a
wider local median, and that detail decides the outcome. A sinusoidal whip pan
alternates fast and slow phases, so a +/-12 frame median straddles both regimes
and lands low, making every fast phase look like a 22x spike — that formulation
split a continuous 2-second clip into four shots. Against immediate neighbours
the same frames score ~0.9x (their neighbours are equally bad) while a real cut
scores ~30x (its neighbours are clean).

Absolute magnitude alone cannot separate the two cases either: the whip pan
reaches a *higher* absolute residual than the real cut does. Both cues are
required.

Track survival and model coherence are kept as corroboration, and appearance
cues are retained because they genuinely help when two shots differ in content.
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
from app.tracking.robust import robust

log = get_logger("video.shots")

#: Long edge for the cut-detection decode pass. Small enough to be nearly free,
#: large enough that fine texture survives downsampling — at 256 px the fine
#: high-frequency detail aliases away, tracking fails under fast motion, and the
#: compensation cue degrades exactly where it is needed most.
CUT_ANALYSIS_LONG_EDGE = 384

#: Grid for spatially-aware histograms. A single global histogram is blind to a
#: cut between two shots that share a palette, which a 4x4 grid partly catches.
HIST_GRID = 4
HIST_BINS = 32

LK_PARAMS = dict(
    winSize=(25, 25),
    maxLevel=5,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


@dataclass
class CutThresholds:
    """Tuning knobs. Defaults were measured, not guessed — see module docstring."""

    residual_floor: float = 0.030
    """Absolute compensated residual (0-1) below which nothing is a cut.
    Non-cut transitions measured at most 0.007; a real cut measured 0.21."""

    peak_ratio: float = 5.0
    """How many times its immediate neighbours the residual must reach. A real
    cut measured ~30x; the whip pan's worst frames measured ~0.9x."""

    max_track_survival: float = 0.70
    """Corroboration: a cut must also break tracking. Deliberately loose,
    because fast motion breaks tracking too and the discrimination is done by
    the two cues above."""

    max_flow_coherence: float = 0.85
    """Corroboration: no single model should explain what survives."""

    min_shot_frames: int = 8
    """Non-maximum suppression window, and the minimum emitted shot length."""

    neighbour_span: int = 2
    """How many frames either side form the comparison neighbourhood. Narrow on
    purpose: it must not reach into a different motion regime."""


def _grid_histogram(frame: np.ndarray) -> np.ndarray:
    """Concatenated per-cell intensity histograms, L1-normalised per cell."""
    h, w = frame.shape[:2]
    cells = []
    for gy in range(HIST_GRID):
        for gx in range(HIST_GRID):
            y0, y1 = gy * h // HIST_GRID, (gy + 1) * h // HIST_GRID
            x0, x1 = gx * w // HIST_GRID, (gx + 1) * w // HIST_GRID
            hist = cv2.calcHist([frame[y0:y1, x0:x1]], [0], None, [HIST_BINS], [0, 256]).ravel()
            total = hist.sum()
            cells.append(hist / total if total > 0 else hist)
    return np.concatenate(cells).astype(np.float32)


@dataclass
class TransitionCues:
    residual: float = 0.0
    """Absolute motion-compensated residual, 0-1. The primary cue."""
    residual_ratio: float = 0.0
    """Compensated residual / raw difference, 0-1."""
    survival: float = 1.0
    coherence: float = 1.0
    hist_change: float = 0.0
    structural: float = 0.0


def _measure_transition(prev: np.ndarray, cur: np.ndarray,
                        prev_hist: np.ndarray, cur_hist: np.ndarray) -> TransitionCues:
    """All cues for one transition, from a single shared LK pass."""
    cues = TransitionCues()

    cues.hist_change = float(cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_BHATTACHARYYA))
    raw = float(cv2.absdiff(prev, cur).mean())
    cues.structural = raw / 255.0

    if raw < 1.0:
        # Frames are effectively identical: nothing moved and nothing changed.
        cues.residual = 0.0
        cues.residual_ratio = 0.0
        return cues

    # --- one LK pass serves both the compensation model and the track cues ---
    model: np.ndarray | None = None
    corners = cv2.goodFeaturesToTrack(
        prev, maxCorners=400, qualityLevel=0.01, minDistance=5, blockSize=7
    )
    if corners is not None and len(corners) >= 10:
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, corners, None, **LK_PARAMS)
        if nxt is not None and status is not None:
            back, status_b, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, **LK_PARAMS)
            ok = status.ravel() == 1
            if back is not None and status_b is not None:
                ok = ok & (status_b.ravel() == 1)
                fb = np.linalg.norm((corners - back).reshape(-1, 2), axis=1)
                ok = ok & (fb < 2.0)
            cues.survival = float(ok.sum()) / float(len(corners))
            if ok.sum() >= 8:
                src = corners[ok].reshape(-1, 2)
                dst = nxt[ok].reshape(-1, 2)
                model, inliers = robust(cv2.estimateAffinePartial2D, 
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                    maxIters=2000, confidence=0.99,
                )
                cues.coherence = (
                    float(inliers.ravel().mean()) if inliers is not None else 0.0
                )
            else:
                cues.coherence = 0.0
    else:
        # Untrackable content (sky, water, extreme blur). No track evidence
        # either way, so leave survival/coherence neutral and let the
        # compensation cue decide on its own.
        cues.survival = 1.0
        cues.coherence = 1.0

    if model is None:
        # Featureless fallback: phase correlation recovers a global translation
        # from raw image structure, with no features required.
        window = cv2.createHanningWindow((prev.shape[1], prev.shape[0]), cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(
            prev.astype(np.float32), cur.astype(np.float32), window
        )
        model = np.float32([[1, 0, dx], [0, 1, dy]])

    h, w = prev.shape[:2]
    warped = cv2.warpAffine(prev, model, (w, h), flags=cv2.INTER_LINEAR)
    coverage = cv2.warpAffine(
        np.full_like(prev, 255), model, (w, h), flags=cv2.INTER_NEAREST
    )
    valid = coverage > 200
    if valid.sum() < 0.2 * valid.size:
        # The transform moved almost everything off-frame; there is nothing left
        # to compare, so this is treated as complete alignment failure.
        cues.residual = cues.structural
        cues.residual_ratio = 1.0
        return cues

    resid = float(
        np.abs(warped[valid].astype(np.float32) - cur[valid].astype(np.float32)).mean()
    )
    cues.residual = resid / 255.0
    cues.residual_ratio = resid / max(raw, 1e-6)
    return cues


def _neighbour_baseline(values: np.ndarray, index: int, span: int) -> float:
    """Worst residual among the immediate neighbours, excluding `index` itself.

    The *maximum* rather than the mean or median: a cut must be worse than the
    worst of its neighbours. Using an average would let one clean neighbour
    excuse a frame that sits in the middle of a sustained bad run.
    """
    lo = max(0, index - span)
    hi = min(len(values), index + span + 1)
    window = np.concatenate([values[lo:index], values[index + 1:hi]])
    if window.size == 0:
        return 0.0
    return float(window.max())


def detect_shots(
    info: VideoInfo,
    frames_meta: list[FrameMetadata],
    *,
    thresholds: CutThresholds | None = None,
    reporter: StageReporter | None = None,
) -> tuple[list[Shot], list[CutCandidate]]:
    """Split a video into continuous shots.

    Returns the shots plus every evaluated boundary, accepted or not, so the UI
    can explain a decision — including "this looked like a cut but one transform
    still explained the whole frame, so it was kept as one shot".
    """
    th = thresholds or CutThresholds()
    n = len(frames_meta)
    if n == 0:
        return [], []
    if n < 3:
        return [_whole_video_shot(frames_meta)], []

    decoder = FrameDecoder(info, long_edge=CUT_ANALYSIS_LONG_EDGE, gray=True)

    cues: list[TransitionCues] = [TransitionCues() for _ in range(n)]
    prev_frame: np.ndarray | None = None
    prev_hist: np.ndarray | None = None
    seen = 0

    start_time = frames_meta[0].time_seconds
    for idx, frame in decoder.iter_frames(
        start_frame=0, end_frame=n - 1, start_time=start_time
    ):
        # Light blur suppresses codec blocking and sensor noise, which would
        # otherwise inflate both the structural and residual cues.
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        hist = _grid_histogram(frame)

        if prev_frame is not None and 0 <= idx < n:
            cues[idx] = _measure_transition(prev_frame, frame, prev_hist, hist)

        prev_frame, prev_hist = frame, hist
        seen += 1
        if reporter and seen % 25 == 0:
            reporter.progress(min(1.0, seen / max(1, n)), f"cut scan {seen}/{n}")

    residuals = np.array([c.residual for c in cues], dtype=np.float64)

    candidates: list[CutCandidate] = []
    cut_frames: list[int] = []

    for i in range(1, n):
        c = cues[i]
        baseline = _neighbour_baseline(residuals, i, th.neighbour_span)
        ratio = c.residual / (baseline + 1e-5)

        substantial = c.residual >= th.residual_floor
        isolated = ratio >= th.peak_ratio
        broken = c.survival <= th.max_track_survival or c.coherence <= th.max_flow_coherence

        accepted = substantial and isolated and broken

        if accepted:
            reason = (
                f"no transform explains this transition: {c.residual * 100:.1f}% "
                f"residual after motion compensation ({ratio:.0f}x its neighbours), "
                f"track survival {c.survival:.0%}, model coherence "
                f"{c.coherence:.0%}"
            )
        elif substantial and not isolated:
            reason = (
                f"rejected: large residual ({c.residual * 100:.1f}%) but only "
                f"{ratio:.1f}x its neighbours, which are just as poorly aligned — "
                "sustained fast camera motion, not an isolated break"
            )
        elif substantial and not broken:
            reason = (
                f"rejected: residual {c.residual * 100:.1f}% but {c.survival:.0%} of "
                f"tracks survived and one model still explains {c.coherence:.0%} of "
                "them — coherent camera motion"
            )
        elif isolated and not substantial:
            reason = (
                f"rejected: a local spike, but only {c.residual * 100:.2f}% residual "
                "remains after motion compensation — the movement is fully explained"
            )
        else:
            reason = "motion fully explained by a single transform"

        if accepted or substantial or isolated:
            candidates.append(
                CutCandidate(
                    frame_index=i,
                    time_seconds=frames_meta[i].time_seconds,
                    histogram_score=c.hist_change,
                    structural_score=c.structural,
                    match_collapse_score=1.0 - c.survival,
                    flow_coherence_score=c.coherence,
                    combined_score=float(min(1.0, c.residual * min(ratio / th.peak_ratio, 2.0))),
                    accepted=accepted,
                    reason=reason,
                )
            )
        if accepted:
            cut_frames.append(i)

    # Non-maximum suppression: the frame after a cut is also unlike the frame
    # before it, so a real cut can flag its neighbour.
    suppressed: list[int] = []
    for f in cut_frames:
        if suppressed and f - suppressed[-1] < th.min_shot_frames:
            continue
        suppressed.append(f)

    shots = _build_shots(frames_meta, suppressed, th.min_shot_frames)

    if reporter:
        accepted_n = len(suppressed)
        rejected_n = sum(1 for c in candidates if not c.accepted)
        reporter.info(
            f"shot detection: {len(shots)} shot(s), {accepted_n} cut(s) accepted"
            + (f", {rejected_n} candidate(s) rejected as camera motion"
               if rejected_n else "")
        )
    log.info("detected %d shots from %d frames (%d cuts)", len(shots), n, len(suppressed))
    return shots, candidates


def _whole_video_shot(frames_meta: list[FrameMetadata]) -> Shot:
    return Shot(
        id=0,
        start_frame=frames_meta[0].frame_index,
        end_frame=frames_meta[-1].frame_index,
        start_time=frames_meta[0].time_seconds,
        end_time=frames_meta[-1].time_seconds,
        confidence=1.0,
    )


def _build_shots(
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

    An *expectation* of solver difficulty, not a confidence score — real
    confidence comes from `validation/confidence.py` after solving.
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
        reasons.append("little parallax — translation may not be geometrically observable")

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
