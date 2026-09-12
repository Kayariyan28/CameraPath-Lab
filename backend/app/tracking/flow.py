"""Pyramidal Lucas-Kanade tracking with persistent tracks.

Tracks persist across frames rather than being re-matched pair-by-pair, because
persistence is itself evidence: a point that has followed the dominant scene
model for 40 frames is almost certainly static world geometry, and one that
repeatedly disagrees is almost certainly on a moving object. Those counters are
what `dynamic_rejection.py` consumes (spec §6, cue 5).

Every correspondence is forward-backward validated. LK will happily return a
confident, completely wrong match in a repetitive texture or across an occlusion
boundary; re-tracking the result back to the previous frame and requiring it to
land where it started removes most of that, cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.logging import get_logger
from app.tracking.features import detect_corners, refine_corners

log = get_logger("tracking.flow")

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=4,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
    minEigThreshold=1e-4,
)

#: Forward-backward round-trip tolerance, pixels.
FB_TOLERANCE = 1.2


@dataclass
class Track:
    """One tracked point's life story — current state plus agreement history."""

    id: int
    x: float
    y: float
    start_frame: int
    last_frame: int

    age: int = 1
    agreements: int = 0
    """Times this track matched the dominant scene model."""
    disagreements: int = 0
    """Times it did not. Repeated disagreement => probably a moving object."""

    residual_ema: float = 0.0
    """Smoothed reprojection residual against the dominant model, px."""

    background_confidence: float = 0.5
    """0-1. Only high-confidence tracks drive camera estimation."""

    @property
    def total_votes(self) -> int:
        return self.agreements + self.disagreements

    @property
    def agreement_ratio(self) -> float:
        return self.agreements / self.total_votes if self.total_votes else 0.5


@dataclass
class FlowResult:
    """One frame transition's correspondences, in analysis-resolution pixels."""

    frame_index: int
    prev_points: np.ndarray   # (N,2) positions in frame N-1
    curr_points: np.ndarray   # (N,2) positions in frame N
    track_ids: np.ndarray     # (N,) ids, for cross-frame bookkeeping
    fb_error: np.ndarray      # (N,) forward-backward round-trip error, px
    tracks_in: int = 0
    tracks_survived: int = 0

    @property
    def count(self) -> int:
        return len(self.curr_points)

    def flow_vectors(self) -> np.ndarray:
        if self.count == 0:
            return np.empty((0, 2), np.float32)
        return self.curr_points - self.prev_points


class LucasKanadeTracker:
    """Stateful tracker over a frame sequence.

    Call `track(frame_index, frame)` for each frame in order. The first call
    seeds; every later call returns the transition from the previous frame.
    """

    def __init__(
        self,
        *,
        max_features: int = 2000,
        min_features: int = 220,
        redetect_ratio: float = 0.55,
        quality_level: float = 0.008,
        min_distance: int = 7,
    ):
        self.max_features = max_features
        self.min_features = min_features
        self.redetect_ratio = redetect_ratio
        self.quality_level = quality_level
        self.min_distance = min_distance

        self.tracks: dict[int, Track] = {}
        self._next_id = 0
        self._prev_frame: np.ndarray | None = None
        self._prev_index: int | None = None
        self._prev_points: np.ndarray = np.empty((0, 1, 2), np.float32)
        self._prev_ids: np.ndarray = np.empty((0,), np.int64)
        self._seed_count = 0

    # ---------------------------------------------------------------- seeding

    def _spawn(self, frame: np.ndarray, frame_index: int, wanted: int,
               existing: np.ndarray) -> None:
        """Detect new corners away from the points we already follow."""
        if wanted <= 0:
            return
        mask = np.full(frame.shape[:2], 255, dtype=np.uint8)
        for px, py in existing.reshape(-1, 2):
            cv2.circle(mask, (int(px), int(py)), self.min_distance, 0, -1)

        fresh = detect_corners(
            frame,
            max_corners=wanted,
            quality_level=self.quality_level,
            min_distance=self.min_distance,
            mask=mask,
        )
        if len(fresh) == 0:
            return
        fresh = refine_corners(frame, fresh)

        new_ids = []
        for px, py in fresh.reshape(-1, 2):
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(
                id=tid, x=float(px), y=float(py),
                start_frame=frame_index, last_frame=frame_index,
            )
            new_ids.append(tid)

        self._prev_points = (
            np.concatenate([existing.reshape(-1, 1, 2), fresh.reshape(-1, 1, 2)])
            if len(existing) else fresh.reshape(-1, 1, 2)
        ).astype(np.float32)
        self._prev_ids = np.concatenate([self._prev_ids, np.array(new_ids, np.int64)])
        self._seed_count += len(new_ids)

    # ----------------------------------------------------------------- public

    def track(self, frame_index: int, frame: np.ndarray) -> FlowResult | None:
        """Advance by one frame. Returns None on the seeding call."""
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_frame is None:
            self._prev_points = np.empty((0, 1, 2), np.float32)
            self._prev_ids = np.empty((0,), np.int64)
            self._spawn(frame, frame_index, self.max_features, self._prev_points)
            self._prev_frame = frame
            self._prev_index = frame_index
            log.debug("seeded %d tracks at frame %d", len(self._prev_ids), frame_index)
            return None

        n_in = len(self._prev_points)
        if n_in == 0:
            # Nothing survived; reseed and report an empty transition so the
            # caller still gets a MotionFrame (with zero confidence) for this
            # index rather than a hole in the timeline.
            self._spawn(frame, frame_index, self.max_features, np.empty((0, 1, 2), np.float32))
            self._prev_frame = frame
            self._prev_index = frame_index
            return FlowResult(
                frame_index=frame_index,
                prev_points=np.empty((0, 2), np.float32),
                curr_points=np.empty((0, 2), np.float32),
                track_ids=np.empty((0,), np.int64),
                fb_error=np.empty((0,), np.float32),
                tracks_in=0, tracks_survived=0,
            )

        fwd, status_f, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_frame, frame, self._prev_points, None, **LK_PARAMS
        )
        if fwd is None or status_f is None:
            fwd = self._prev_points.copy()
            status_f = np.zeros((n_in, 1), np.uint8)

        back, status_b, _ = cv2.calcOpticalFlowPyrLK(
            frame, self._prev_frame, fwd, None, **LK_PARAMS
        )
        if back is None or status_b is None:
            back = self._prev_points.copy()
            status_b = np.zeros((n_in, 1), np.uint8)

        ok = (status_f.ravel() == 1) & (status_b.ravel() == 1)
        fb = np.linalg.norm(
            (self._prev_points - back).reshape(-1, 2), axis=1
        ).astype(np.float32)

        h, w = frame.shape[:2]
        pts_new = fwd.reshape(-1, 2)
        in_bounds = (
            (pts_new[:, 0] >= 0) & (pts_new[:, 0] < w)
            & (pts_new[:, 1] >= 0) & (pts_new[:, 1] < h)
        )
        keep = ok & in_bounds & (fb < FB_TOLERANCE)

        prev_kept = self._prev_points.reshape(-1, 2)[keep]
        curr_kept = pts_new[keep]
        ids_kept = self._prev_ids[keep]
        fb_kept = fb[keep]

        # Retire the dead, advance the living.
        dead = set(self._prev_ids[~keep].tolist())
        for tid in dead:
            self.tracks.pop(tid, None)
        for tid, (cx, cy) in zip(ids_kept.tolist(), curr_kept):
            tr = self.tracks.get(tid)
            if tr is None:
                continue
            tr.x, tr.y = float(cx), float(cy)
            tr.last_frame = frame_index
            tr.age += 1

        result = FlowResult(
            frame_index=frame_index,
            prev_points=prev_kept.astype(np.float32),
            curr_points=curr_kept.astype(np.float32),
            track_ids=ids_kept,
            fb_error=fb_kept,
            tracks_in=n_in,
            tracks_survived=int(keep.sum()),
        )

        # Top up when the population thins, so long shots do not slowly go blind.
        self._prev_points = curr_kept.reshape(-1, 1, 2).astype(np.float32)
        self._prev_ids = ids_kept
        if len(curr_kept) < max(self.min_features, int(self.max_features * self.redetect_ratio)):
            self._spawn(
                frame, frame_index,
                self.max_features - len(curr_kept),
                self._prev_points,
            )

        self._prev_frame = frame
        self._prev_index = frame_index
        return result

    # -------------------------------------------------------------- accessors

    def persistent_track_count(self, min_age: int = 10) -> int:
        return sum(1 for t in self.tracks.values() if t.age >= min_age)

    def mean_track_age(self) -> float:
        if not self.tracks:
            return 0.0
        return float(np.mean([t.age for t in self.tracks.values()]))

    def total_spawned(self) -> int:
        return self._seed_count


def dense_flow(prev: np.ndarray, curr: np.ndarray, *, scale: float = 0.5) -> np.ndarray:
    """Farneback dense flow at reduced scale.

    Only used for diagnostics overlays and for the flow-field statistics on
    low-texture frames where sparse tracking gives too few samples to estimate
    divergence and curl. Returns an (H, W, 2) float32 field at the *reduced*
    resolution, with vectors in reduced-resolution pixels.
    """
    if prev.ndim == 3:
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    if curr.ndim == 3:
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    if scale != 1.0:
        prev = cv2.resize(prev, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        curr = cv2.resize(curr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        pyr_scale=0.5, levels=4, winsize=17,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
