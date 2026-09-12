"""Separating world geometry from moving objects, without a segmentation model.

The naive assumption — that the dominant foreground subject is part of the static
world — is wrong exactly when it matters most: a car chase, a walking subject
filling the frame, a drone following a boat. So the camera solver must prefer
*background* motion, and needs a way to find it from geometry alone.

Five cues, per spec §6, in increasing order of how much history they need:

  1. feature tracks            — positions over time
  2. dominant geometric model  — what the majority of the scene is doing
  3. RANSAC                    — robust fit that tolerates a minority of outliers
  4. residual-flow clustering  — scattered residuals are noise; a *spatially
                                 coherent* patch of consistent residual is an
                                 object moving rigidly against the scene
  5. track persistence         — one disagreement is noise; repeated
                                 disagreement over many frames is an object

Cue 4 is what distinguishes this from plain RANSAC. RANSAC alone fails when the
moving object is large enough to become the majority; coherence tells us the
majority is a compact region moving consistently, which the true background never
is.

Deliberately structured behind `DynamicRejectionBackend` so a semantic
segmentation model (people/vehicles/animals) can be added as a second
implementation without touching the solvers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from app.core.logging import get_logger
from app.tracking.flow import FlowResult, Track

log = get_logger("tracking.dynamic")

#: Residual above this (px, analysis resolution) counts as disagreement.
RESIDUAL_THRESHOLD = 2.6

#: Spatial grid for residual coherence testing.
COHERENCE_GRID = 8

#: A cell needs at least this many tracks before its median means anything.
MIN_TRACKS_PER_CELL = 3


@runtime_checkable
class DynamicRejectionBackend(Protocol):
    """Swap-in point for a future semantic implementation."""

    def update(
        self,
        flow_result: FlowResult,
        tracks: dict[int, Track],
        image_size: tuple[int, int],
        *,
        strength: float = 0.5,
    ) -> np.ndarray:
        """Return per-correspondence background confidence in [0,1], aligned
        with `flow_result.track_ids`."""
        ...

    def reset(self) -> None: ...


class GeometricDynamicRejector:
    """Default backend. No neural network, no weights to download."""

    def __init__(self) -> None:
        self._dynamic_cells_history: list[np.ndarray] = []

    def reset(self) -> None:
        self._dynamic_cells_history.clear()

    # ------------------------------------------------------------------ cues

    @staticmethod
    def _provisional_model(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
        """Cue 2+3: dominant transform via RANSAC, fitted to everything.

        A similarity model is used on purpose. A homography has enough freedom to
        partially absorb a large moving object's motion, which would hide the
        very residual we are looking for.
        """
        if len(src) < 6:
            return None
        model, _ = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD_FIT,
            maxIters=2500, confidence=0.995, refineIters=20,
        )
        return model

    @staticmethod
    def _residuals(model: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        pred = (np.asarray(model)[:2, :2] @ src.T).T + np.asarray(model)[:2, 2]
        return np.linalg.norm(pred - dst, axis=1)

    def _coherent_dynamic_mask(
        self,
        points: np.ndarray,
        residual_vectors: np.ndarray,
        residuals: np.ndarray,
        image_size: tuple[int, int],
    ) -> np.ndarray:
        """Cue 4: mark points inside spatially coherent high-residual regions.

        Per grid cell we ask two questions: is the residual large here, and is it
        *consistent in direction* here? Sensor noise and bad tracks produce large
        residuals with random directions; a rigid object produces large residuals
        that all point the same way.
        """
        w, h = image_size
        mask = np.zeros(len(points), dtype=bool)
        if len(points) < MIN_TRACKS_PER_CELL:
            return mask

        cell_x = np.clip((points[:, 0] / max(w, 1) * COHERENCE_GRID).astype(int),
                         0, COHERENCE_GRID - 1)
        cell_y = np.clip((points[:, 1] / max(h, 1) * COHERENCE_GRID).astype(int),
                         0, COHERENCE_GRID - 1)
        cell_id = cell_y * COHERENCE_GRID + cell_x

        dynamic_cells = np.zeros(COHERENCE_GRID * COHERENCE_GRID, dtype=bool)
        for cid in np.unique(cell_id):
            sel = cell_id == cid
            if sel.sum() < MIN_TRACKS_PER_CELL:
                continue
            cell_res = residuals[sel]
            if float(np.median(cell_res)) < RESIDUAL_THRESHOLD:
                continue
            vecs = residual_vectors[sel]
            norms = np.linalg.norm(vecs, axis=1)
            good = norms > 1e-6
            if good.sum() < MIN_TRACKS_PER_CELL:
                continue
            units = vecs[good] / norms[good, None]
            mean_dir = units.mean(axis=0)
            # |mean of unit vectors| is 1 for perfect agreement, ~0 for random.
            coherence = float(np.linalg.norm(mean_dir))
            if coherence >= 0.72:
                dynamic_cells[cid] = True

        if dynamic_cells.any():
            mask = dynamic_cells[cell_id]
        self._dynamic_cells_history.append(dynamic_cells)
        if len(self._dynamic_cells_history) > 12:
            self._dynamic_cells_history.pop(0)
        return mask

    # ---------------------------------------------------------------- update

    def update(
        self,
        flow_result: FlowResult,
        tracks: dict[int, Track],
        image_size: tuple[int, int],
        *,
        strength: float = 0.5,
    ) -> np.ndarray:
        n = flow_result.count
        if n == 0:
            return np.empty((0,), np.float32)

        src, dst = flow_result.prev_points, flow_result.curr_points
        model = self._provisional_model(src, dst)
        if model is None:
            return np.full(n, 0.5, np.float32)

        residuals = self._residuals(model, src, dst)
        pred = (np.asarray(model)[:2, :2] @ src.T).T + np.asarray(model)[:2, 2]
        residual_vectors = dst - pred

        disagrees = residuals > RESIDUAL_THRESHOLD
        coherent = self._coherent_dynamic_mask(src, residual_vectors, residuals, image_size)

        # ---- cue 5: fold this frame's verdict into each track's history ----
        weights = np.full(n, 0.5, np.float32)
        for i, tid in enumerate(flow_result.track_ids.tolist()):
            tr = tracks.get(tid)
            if tr is None:
                continue

            if disagrees[i]:
                # Coherent disagreement is much stronger evidence than isolated
                # disagreement, so it counts double.
                tr.disagreements += 2 if coherent[i] else 1
            else:
                tr.agreements += 1

            alpha = 0.35
            tr.residual_ema = (1 - alpha) * tr.residual_ema + alpha * float(residuals[i])

            # Laplace-smoothed agreement ratio: a brand-new track with one lucky
            # agreement should not immediately be trusted as much as a track with
            # forty.
            votes = tr.agreements + tr.disagreements
            smoothed = (tr.agreements + 1.0) / (votes + 2.0)

            persistence = min(1.0, tr.age / 15.0)
            residual_penalty = float(np.clip(tr.residual_ema / (RESIDUAL_THRESHOLD * 2.0), 0.0, 1.0))

            confidence = smoothed * (0.75 + 0.25 * persistence) * (1.0 - 0.55 * residual_penalty)

            if coherent[i]:
                confidence *= 1.0 - 0.6 * strength

            tr.background_confidence = float(np.clip(confidence, 0.0, 1.0))
            weights[i] = tr.background_confidence

        # `strength` sharpens the separation: at 0 everything is trusted equally,
        # at 1 anything below the midpoint is pushed toward rejection.
        if strength > 0:
            weights = np.clip(
                0.5 + (weights - 0.5) * (1.0 + 2.0 * strength), 0.0, 1.0
            ).astype(np.float32)
        return weights

    # --------------------------------------------------------------- reporting

    def dynamic_region_fraction(self) -> float:
        """Share of the frame recently judged to contain moving content.

        Surfaced to the user because a large value is the honest explanation for
        a low-confidence solve: "most of the frame is a moving subject".
        """
        if not self._dynamic_cells_history:
            return 0.0
        stacked = np.stack(self._dynamic_cells_history)
        return float(stacked.mean())


# Fit threshold kept separate from the judging threshold: RANSAC should be
# slightly stricter when *finding* the model than when deciding who agrees with
# it, so that a marginal track is not used to define the model it is judged by.
RANSAC_THRESHOLD_FIT = 2.0


def background_mask_from_tracks(
    tracks: dict[int, Track],
    image_size: tuple[int, int],
    *,
    min_confidence: float = 0.5,
    dilate_px: int = 28,
) -> np.ndarray:
    """Build an image mask covering probable *dynamic* content.

    Passed to feature detection so new corners are not spawned onto a moving
    subject in the first place, and to COLMAP keyframe extraction as a hint.
    Returned mask is 255 where features are welcome, 0 where they are not.
    """
    w, h = image_size
    mask = np.full((h, w), 255, np.uint8)
    suspicious = [
        t for t in tracks.values()
        if t.background_confidence < min_confidence and t.total_votes >= 3
    ]
    if not suspicious:
        return mask
    for t in suspicious:
        cv2.circle(mask, (int(t.x), int(t.y)), dilate_px, 0, -1)
    return mask
