"""Feature detection.

Two detectors, two jobs:

  * Shi-Tomasi corners — what Lucas-Kanade wants. Cheap, well-localised, and
    re-detected continuously as tracks die. This drives the dense per-frame
    motion signature.
  * SIFT — scale/rotation invariant descriptors for wide-baseline matching,
    used to bridge frames that LK cannot connect (large rotation, long gaps)
    and to seed geometric verification.

Corners are spread with a spatial grid quota. Without that, `goodFeaturesToTrack`
piles detections onto the highest-contrast region — often a single foreground
subject — and the resulting "global" motion estimate is really the subject's
motion. Forcing coverage across the frame is the cheapest available defence
against that, before any dynamic-rejection logic runs.
"""

from __future__ import annotations

import cv2
import numpy as np

from app.core.logging import get_logger

log = get_logger("tracking.features")


def detect_corners(
    frame: np.ndarray,
    *,
    max_corners: int = 2000,
    quality_level: float = 0.008,
    min_distance: int = 7,
    mask: np.ndarray | None = None,
    grid: tuple[int, int] = (4, 4),
) -> np.ndarray:
    """Shi-Tomasi corners with a per-cell quota for spatial spread.

    Returns an (N, 1, 2) float32 array in OpenCV's point layout, or an empty
    (0, 1, 2) array when nothing trackable is found.
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gh, gw = grid
    h, w = frame.shape[:2]
    per_cell = max(4, max_corners // max(1, gh * gw))
    found: list[np.ndarray] = []

    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * h // gh, (gy + 1) * h // gh
            x0, x1 = gx * w // gw, (gx + 1) * w // gw
            cell = frame[y0:y1, x0:x1]
            cell_mask = mask[y0:y1, x0:x1] if mask is not None else None
            if cell_mask is not None and not cell_mask.any():
                continue
            pts = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=per_cell,
                qualityLevel=quality_level,
                minDistance=min_distance,
                blockSize=7,
                mask=cell_mask,
            )
            if pts is not None and len(pts):
                pts = pts.reshape(-1, 2)
                pts[:, 0] += x0
                pts[:, 1] += y0
                found.append(pts)

    if not found:
        return np.empty((0, 1, 2), dtype=np.float32)

    stacked = np.concatenate(found, axis=0).astype(np.float32)
    if len(stacked) > max_corners:
        # Keep a spatially random subset rather than the strongest, which would
        # undo the grid quota we just paid for.
        idx = np.random.default_rng(0).choice(len(stacked), max_corners, replace=False)
        stacked = stacked[idx]
    return stacked.reshape(-1, 1, 2)


def refine_corners(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Sub-pixel corner refinement. Worth it: LK residuals and the resulting
    rotation estimate are both sensitive to sub-pixel localisation."""
    if points is None or len(points) == 0:
        return points
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float32)
    try:
        cv2.cornerSubPix(
            frame, pts, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
    except cv2.error:
        return points
    return pts


def texture_score(frame: np.ndarray, *, sample_corners: int = 900) -> float:
    """0-1 estimate of how trackable a frame is.

    Combines corner density with gradient energy. Sky, water, fog and blown-out
    exposures all score low, which is what routes a shot toward Perceptual Match
    instead of a doomed SfM attempt.
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(
        frame, maxCorners=sample_corners, qualityLevel=0.01, minDistance=6, blockSize=7
    )
    density = (len(pts) if pts is not None else 0) / float(sample_corners)

    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.sqrt(gx * gx + gy * gy).mean())
    # ~18 mean gradient magnitude is a comfortably textured frame.
    grad_norm = min(1.0, grad / 18.0)

    return float(np.clip(0.6 * density + 0.4 * grad_norm, 0.0, 1.0))


def blur_score(frame: np.ndarray) -> float:
    """0-1 sharpness, 1 = sharp. Variance of Laplacian, normalised.

    Motion blur destroys corner localisation, so this feeds both the complexity
    estimate and the per-transition confidence.
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(frame, cv2.CV_64F).var())
    # Empirically: <40 is visibly soft, >400 is crisp. Log-scaled between.
    if var <= 1.0:
        return 0.0
    return float(np.clip((np.log10(var) - 1.4) / (np.log10(400.0) - 1.4), 0.0, 1.0))


class SiftMatcher:
    """SIFT detect + ratio-test match, for wide-baseline frame pairs.

    Used where LK gives up: long temporal gaps, large in-plane rotation, and the
    strategically-separated pairs that keep a long shot's reconstruction from
    drifting.
    """

    def __init__(self, n_features: int = 3000):
        self.sift = cv2.SIFT_create(nfeatures=n_features)
        # FLANN over 128-D SIFT descriptors; approximate but ~5x faster than
        # brute force at this feature count, and the ratio test absorbs the
        # occasional approximate-neighbour miss.
        self.matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=48)
        )

    def detect(self, frame: np.ndarray) -> tuple[list, np.ndarray | None]:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, desc = self.sift.detectAndCompute(frame, None)
        return list(kp), desc

    def match(
        self, desc_a: np.ndarray | None, desc_b: np.ndarray | None, ratio: float = 0.75
    ) -> list[tuple[int, int]]:
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            return []
        try:
            knn = self.matcher.knnMatch(desc_a.astype(np.float32), desc_b.astype(np.float32), k=2)
        except cv2.error as exc:
            log.warning("SIFT match failed: %s", exc)
            return []
        out: list[tuple[int, int]] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                out.append((m.queryIdx, m.trainIdx))
        return out

    def match_points(
        self, frame_a: np.ndarray, frame_b: np.ndarray, ratio: float = 0.75
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: returns (pts_a, pts_b) as (N,2) float32 arrays."""
        kp_a, desc_a = self.detect(frame_a)
        kp_b, desc_b = self.detect(frame_b)
        pairs = self.match(desc_a, desc_b, ratio)
        if not pairs:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pts_a = np.array([kp_a[i].pt for i, _ in pairs], dtype=np.float32)
        pts_b = np.array([kp_b[j].pt for _, j in pairs], dtype=np.float32)
        return pts_a, pts_b
