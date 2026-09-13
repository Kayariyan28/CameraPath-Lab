"""Long-baseline keyframe homographies: the absolute reference for rotation and zoom.

Per-frame motion is precise about SHAPE and poor about ACCUMULATION. Every
frame-to-frame estimate carries a tiny bias, and anything obtained by multiplying
or summing hundreds of them inherits hundreds of copies of it. Measured on a real
59.94 fps locked-off shot: per-frame homographies carried a +0.024%/frame zoom
bias — 11x what a random walk would give — which compounded over 899 frames into
a 24% phantom zoom, while a direct homography between the first and last frames
measured 1.0006. The same effect leaked 4 deg of phantom rotation.

So, following spec section 31 (lightweight flow every frame, SIFT geometry at
keyframes), keyframes are chosen during the motion pass and consecutive keyframes
are related by a SIFT + robust-RANSAC homography. Over a baseline of up to a
second, moving people have moved far enough that RANSAC rejects them outright,
and a static background gives an exact identity. Absolute zoom and rotation are
then products of ~1 factor per second instead of ~60, and the per-frame signal is
used only for the detail between keyframes (lens curve, OpenCVPoseBackend, fusion).

The homography model is exact for a rotating/zooming camera or a distant scene.
Under real translation with depth it is not, which is why the consumers of these
matrices gate on measured parallax.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

import cv2

from app.core.logging import get_logger
from app.models.schemas.motion import MotionFrame
from app.tracking.features import SiftMatcher
from app.tracking.robust import robust

log = get_logger("tracking.keyframes")

#: A new keyframe once the image has moved this fraction of its width since the
#: last one: comfortably inside SIFT's overlap requirement.
KEYFRAME_DISPLACEMENT_FRACTION = 0.12
#: ...or rotated this much in-plane,
KEYFRAME_ROTATION_DEG = 4.0
#: ...or its scale changed by this much (log),
KEYFRAME_LOG_SCALE = 0.06
#: ...or this much time has passed. Bounds accumulation even on a static shot.
KEYFRAME_MAX_SECONDS = 1.0
#: Never closer than this many frames.
KEYFRAME_MIN_GAP = 2

#: A keyframe pair needs this many RANSAC inliers to be trusted.
MIN_HOMOGRAPHY_INLIERS = 40
RANSAC_THRESHOLD_PX = 1.5


class KeyframeHomography(BaseModel):
    """H maps pixel coordinates in `frame_a` to `frame_b`, at analysis resolution."""

    frame_a: int
    frame_b: int
    time_a: float
    time_b: float
    homography: list[float] = Field(..., min_length=9, max_length=9)
    inliers: int
    matches: int

    def matrix(self) -> np.ndarray:
        return np.asarray(self.homography, dtype=np.float64).reshape(3, 3)


class KeyframeTracker:
    """Online keyframe selection and pairwise homographies, fed by the motion pass.

    Call `observe` for every frame after its MotionFrame is known (the first frame
    with `motion=None`), then `finish` with the last frame.
    """

    def __init__(self, width: int, n_features: int = 3000):
        self.width = width
        self.matcher = SiftMatcher(n_features=n_features)
        self.pairs: list[KeyframeHomography] = []
        self.keyframes: list[int] = []
        self._last: tuple[int, float, list, np.ndarray | None] | None = None
        self._acc_disp = 0.0
        self._acc_rot = 0.0
        self._acc_scale = 0.0
        self._pending: tuple[int, float, np.ndarray] | None = None
        self.failed_pairs = 0

    def _add_keyframe(self, frame_index: int, time: float, frame: np.ndarray) -> None:
        keypoints, descriptors = self.matcher.detect(frame)
        if self._last is not None:
            last_index, last_time, last_kp, last_desc = self._last
            pair = self._relate(last_index, last_time, last_kp, last_desc,
                                frame_index, time, keypoints, descriptors)
            if pair is not None:
                self.pairs.append(pair)
            else:
                self.failed_pairs += 1
        self._last = (frame_index, time, keypoints, descriptors)
        self.keyframes.append(frame_index)
        self._acc_disp = self._acc_rot = self._acc_scale = 0.0

    def _relate(self, a, ta, kp_a, desc_a, b, tb, kp_b, desc_b) -> KeyframeHomography | None:
        pairs = self.matcher.match(desc_a, desc_b, ratio=0.72)
        if len(pairs) < MIN_HOMOGRAPHY_INLIERS:
            return None
        src = np.float32([kp_a[i].pt for i, _ in pairs])
        dst = np.float32([kp_b[j].pt for _, j in pairs])
        h, mask = robust(cv2.findHomography, src, dst, cv2.USAC_MAGSAC, RANSAC_THRESHOLD_PX,
                         maxIters=5000, confidence=0.999)
        if h is None or mask is None or not np.isfinite(h).all():
            return None
        inliers = int(mask.ravel().sum())
        if inliers < MIN_HOMOGRAPHY_INLIERS:
            return None
        h = h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h
        return KeyframeHomography(frame_a=a, frame_b=b, time_a=ta, time_b=tb,
                                  homography=[float(v) for v in h.ravel()],
                                  inliers=inliers, matches=len(pairs))

    def observe(self, frame_index: int, time: float, frame: np.ndarray,
                motion: MotionFrame | None) -> None:
        if self._last is None:
            self._add_keyframe(frame_index, time, frame)
            return
        self._pending = (frame_index, time, frame)
        if motion is not None:
            self._acc_disp += float(np.hypot(motion.dx_pixels, motion.dy_pixels))
            self._acc_rot += abs(float(motion.rotation_deg))
            self._acc_scale += abs(float(np.log(max(motion.scale, 1e-3))))
        last_index, last_time = self._last[0], self._last[1]
        if frame_index - last_index < KEYFRAME_MIN_GAP:
            return
        if (self._acc_disp >= KEYFRAME_DISPLACEMENT_FRACTION * self.width
                or self._acc_rot >= KEYFRAME_ROTATION_DEG
                or self._acc_scale >= KEYFRAME_LOG_SCALE
                or time - last_time >= KEYFRAME_MAX_SECONDS):
            self._add_keyframe(frame_index, time, frame)
            self._pending = None

    def finish(self) -> list[KeyframeHomography]:
        """Close the chain at the last observed frame."""
        if self._pending is not None and self._last is not None and self._pending[0] > self._last[0]:
            self._add_keyframe(*self._pending)
            self._pending = None
        log.info("keyframe geometry: %d keyframes, %d homographies, %d pair(s) failed",
                 len(self.keyframes), len(self.pairs), self.failed_pairs)
        return self.pairs


def chained_matrices(pairs: list[KeyframeHomography]) -> list[np.ndarray]:
    return [p.matrix() for p in pairs]
