"""Per-frame zoom accumulates bias; keyframe homographies must pin it (real-footage
regression: a locked-off 59.94 fps shot accumulated a 24% phantom zoom)."""

from __future__ import annotations

import numpy as np

from app.geometry.intrinsics import build_lens_curve, intrinsics_from_fov
from app.models.schemas.motion import MotionFrame
from app.tracking.keyframe_geometry import KeyframeHomography

W, H = 1080, 608


def _zoom_h(s: float) -> list[float]:
    cx, cy = W / 2, H / 2
    m = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)], [0, 0, 1.0]])
    return [float(v) for v in m.ravel()]


def _frames(per_frame_zoom: list[float]) -> list[MotionFrame]:
    return [
        MotionFrame(frame_index=i + 1, timestamp=(i + 1) / 60, dt=1 / 60, homography=_zoom_h(z),
                    inlier_ratio=0.95, confidence=0.9)
        for i, z in enumerate(per_frame_zoom)
    ]


def _keyframes(total_zoom_at: dict[int, float]) -> list[KeyframeHomography]:
    frames = sorted(total_zoom_at)
    out = []
    for a, b in zip(frames, frames[1:]):
        out.append(KeyframeHomography(frame_a=a, frame_b=b, time_a=a / 60, time_b=b / 60,
                                      homography=_zoom_h(total_zoom_at[b] / total_zoom_at[a]),
                                      inliers=1500, matches=1550))
    return out


def test_biased_static_shot_is_pinned_flat():
    bias = [1.00024] * 899  # the measured real-footage bias
    base = intrinsics_from_fov(W, H, 60.0)
    unanchored, _ = build_lens_curve(_frames(bias), base, parallax_score=0.0)
    assert unanchored[-1].fov_horizontal < 52.0  # the phantom zoom, reproduced
    keys = _keyframes({f: 1.0 for f in range(0, 900, 60)} | {899: 1.0})
    anchored, note = build_lens_curve(_frames(bias), base, parallax_score=0.0, keyframe_homographies=keys)
    assert abs(anchored[-1].fov_horizontal - 60.0) < 0.5
    assert "no significant scale change" in note


def test_genuine_zoom_survives_anchoring():
    per_frame = [2.25 ** (1 / 59)] * 59
    truth = {0: 1.0}
    for f in range(5, 60, 5):
        truth[f] = 2.25 ** (f / 59)
    truth[59] = 2.25
    base = intrinsics_from_fov(W, H, 60.0)
    curve, _ = build_lens_curve(_frames(per_frame), base, parallax_score=0.0, keyframe_homographies=_keyframes(truth))
    ratio = np.tan(np.radians(curve[0].fov_horizontal) / 2) / np.tan(np.radians(curve[-1].fov_horizontal) / 2)
    assert abs(ratio - 2.25) / 2.25 < 0.01
