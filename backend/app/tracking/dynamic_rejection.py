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
from app.tracking.robust import robust

log = get_logger("tracking.dynamic")

#: Baseline residual above which a track counts as disagreeing, in px at
#: analysis resolution. Used as a floor — see `residual_threshold_for()`.
RESIDUAL_THRESHOLD = 2.6

#: Residual tolerance as a fraction of the frame's own motion magnitude.
#: Tracking error grows with displacement (larger LK search, more interpolation,
#: more motion blur), so a fixed threshold makes every fast pan look like it is
#: full of moving objects. 8% of median flow matches the observed spread on
#: synthetic clips that contain no moving content at all.
RESIDUAL_MOTION_FRACTION = 0.08


def residual_threshold_for(median_flow: float) -> float:
    """Disagreement threshold for a transition with this much image motion.

    A 2.6 px residual means very different things at 2 px/frame (gross
    disagreement) and at 48 px/frame (ordinary tracking noise). Without this
    scaling, fast camera movement produces a confident but false "a moving
    subject covers 25% of the frame" warning.
    """
    return max(RESIDUAL_THRESHOLD, RESIDUAL_MOTION_FRACTION * max(median_flow, 0.0))

#: Spatial grid for residual coherence testing.
COHERENCE_GRID = 8

#: A cell needs at least this many tracks before its median means anything.
#: Six rather than three: declaring "a moving object is here" from three points
#: is not a measurement, and at three the estimate is dominated by whichever
#: tracks happened to fail.
MIN_TRACKS_PER_CELL = 6

#: How anti-parallel to the global motion a cell's residual must be, and how
#: close in magnitude, before it is treated as stuck tracks rather than a moving
#: object. See `_is_stuck_track_cluster`.
STUCK_ALIGNMENT = 0.85
STUCK_MAGNITUDE_LO = 0.55
STUCK_MAGNITUDE_HI = 1.6


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
        parallax_present: bool = False,
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """Return per-correspondence background confidence in [0,1], aligned
        with `flow_result.track_ids`."""
        ...

    def reset(self) -> None: ...


#: Time constant of each track's residual drift accumulator, seconds. Defined in
#: seconds so the cue does not depend on frame rate.
DRIFT_TIME_CONSTANT = 0.5

#: Accumulated residual displacement, px at analysis resolution, beyond which a
#: track counts as moving against the scene.
#:
#: Why a per-frame threshold alone is not enough: it is frame-rate dependent. On a
#: real 59.94 fps store-aisle clip, people walking through a locked-off shot moved
#: 1-3 px per frame, never crossed the per-frame threshold, and not one track was
#: rejected in 899 frames. Their approach toward the lens read as a steady
#: expansion, which accumulated into a 24% phantom zoom and 4 deg of phantom
#: rotation. Integrated over time the same walker drifts tens of pixels, while
#: static background noise (~0.1-0.3 px per frame) has a steady-state drift of
#: roughly 0.4-1.2 px at 60 fps. 4 px is well clear of that.
DRIFT_THRESHOLD_PX = 4.0


class GeometricDynamicRejector:
    """Default backend. No neural network, no weights to download."""

    def __init__(self) -> None:
        self._dynamic_cells_history: list[np.ndarray] = []
        self._model_used: str = "similarity"

    def reset(self) -> None:
        self._dynamic_cells_history.clear()
        self._model_used = "similarity"

    @property
    def model_used(self) -> str:
        """Which static-scene model(s) the last verdict was judged against."""
        return self._model_used

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
        model, _ = robust(cv2.estimateAffinePartial2D, 
            src, dst, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD_FIT,
            maxIters=2500, confidence=0.995, refineIters=20,
        )
        return model

    @staticmethod
    def _residuals(model: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        pred = (np.asarray(model)[:2, :2] @ src.T).T + np.asarray(model)[:2, 2]
        return np.linalg.norm(pred - dst, axis=1)

    @staticmethod
    def _rigid_scene_residuals(
        src: np.ndarray, dst: np.ndarray, similarity_residuals: np.ndarray,
        *, use_epipolar: bool, similarity_vectors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, str, np.ndarray | None]:
        """Residual against the best available STATIC-SCENE model.

        The similarity residual alone is not a moving-object detector once the
        camera translates through a scene with depth. Near objects then move
        faster than any single 2D transform predicts, so parallax itself shows up
        as a large, spatially coherent residual — measured at 48% of the frame
        on a synthetic dolly whose scene contains no moving object at all.

        The fix is to ask the right question: is this point consistent with SOME
        rigid interpretation of the scene?

          * a homography explains a static scene under pure rotation, pure zoom,
            or a planar/distant scene
          * the epipolar constraint explains a static scene under ANY camera
            motion, at any depth distribution

        A static world point satisfies at least one of these. An independently
        moving object satisfies neither. So the per-point residual is the MINIMUM
        of the two, and each model covers the case where the other degenerates —
        F is ill-conditioned under pure rotation, H cannot represent parallax.

        The epipolar model is admitted ONLY when parallax has actually been
        measured (`use_epipolar`), and that gate is not a nicety. The epipolar
        constraint is a *weaker* test than a 2D transform: it has 7 degrees of
        freedom and constrains each point to a line rather than a point. Where
        the scene is planar or the camera only rotated, it buys nothing and costs
        real detection power — with two distinct translation directions in the
        frame (a background pan plus a subject crossing it) an F can place
        epipolar lines through BOTH, and the subject becomes invisible. Measured:
        admitting F unconditionally let a synthetic moving subject through at
        weight 0.67 instead of being rejected below 0.4.

        So: no parallax means the stricter 2D test is both correct and safer.
        Parallax means the 2D test is wrong and must be relaxed.

        Known limitation, inherent to any geometry-only method: even with the
        gate, an object moving ALONG its epipolar line satisfies the epipolar
        constraint and is invisible here. Separating that case needs semantic
        segmentation, which is why this class sits behind
        DynamicRejectionBackend.
        """
        best = similarity_residuals
        used = "similarity"
        # Residual VECTOR of whichever rigid model explains each point best, for
        # the drift accumulator. None means "no usable evidence this frame".
        vectors = None if similarity_vectors is None else similarity_vectors.copy()

        if len(src) < 20:
            return best, used, vectors

        motion = float(np.median(np.linalg.norm(dst - src, axis=1)))

        # Homography: right model for rotation-only, zoom, and planar scenes.
        h, _ = robust(cv2.findHomography, 
            src, dst, method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD_FIT, maxIters=3000, confidence=0.995,
        )
        if h is not None and np.isfinite(h).all():
            homo = np.hstack([src, np.ones((len(src), 1))])
            proj = homo @ np.asarray(h).T
            w = proj[:, 2:3]
            w = np.where(np.abs(w) < 1e-12, 1e-12, w)
            h_vec = proj[:, :2] / w - dst
            h_res = np.linalg.norm(h_vec, axis=1)
            if np.isfinite(h_res).all():
                if vectors is not None:
                    take = h_res < best
                    vectors[take] = -h_vec[take]
                best = np.minimum(best, h_res)
                used = "similarity+homography"

        # Epipolar: right model whenever the camera translates through depth.
        # Needs measured parallax (see the docstring) and enough motion to be
        # numerically conditioned at all.
        if use_epipolar and len(src) >= 50 and motion > 1.5:
            f, _ = robust(cv2.findFundamentalMat, 
                src, dst, method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=1.5, confidence=0.995, maxIters=4000,
            )
            if f is not None and f.shape == (3, 3) and np.isfinite(f).all():
                # Sampson distance: first-order geometric distance to the
                # epipolar variety, in pixels.
                x1 = np.hstack([src, np.ones((len(src), 1))])
                x2 = np.hstack([dst, np.ones((len(dst), 1))])
                fx1 = x1 @ np.asarray(f).T
                ftx2 = x2 @ np.asarray(f)
                num = np.sum(x2 * fx1, axis=1) ** 2
                den = fx1[:, 0] ** 2 + fx1[:, 1] ** 2 + ftx2[:, 0] ** 2 + ftx2[:, 1] ** 2
                den = np.where(den < 1e-12, 1e-12, den)
                sampson = np.sqrt(num / den)
                if np.isfinite(sampson).all():
                    if vectors is not None:
                        # Consistent with a rigid scene through depth: no drift.
                        vectors[sampson < best] = 0.0
                    best = np.minimum(best, sampson)
                    used = used + "+epipolar"
        elif use_epipolar:
            # Parallax is present but the epipolar test could not run on this
            # frame (too little motion to condition it). A 2D residual here is
            # parallax as much as motion, so it is not evidence either way.
            vectors = None

        return best, used, vectors

    @staticmethod
    def _is_stuck_track_cluster(
        mean_residual: np.ndarray, global_motion: np.ndarray
    ) -> bool:
        """Is this cell's residual explained by tracks that failed to move?

        This check is what makes the coherence cue usable. A Lucas-Kanade track
        that sticks — locks onto a locally ambiguous patch and reports no motion
        — has a residual vector equal to the *negative of the true motion*. So
        every stuck track in a frame shares one residual direction, and a cluster
        of them is perfectly "coherent", indistinguishable by direction alone
        from a rigid object moving against the camera.

        Forward-backward validation does not catch these: a stuck track
        round-trips to exactly where it started, so its FB error is zero.

        The distinguishing feature is the *relationship to the camera motion*. A
        real moving object's velocity has no particular relationship to the
        camera's. A stuck track's residual is anti-parallel to the global motion
        and of the same magnitude, because it is that motion, negated.

        Measured impact: without this, a clean constant-velocity pan over a
        static scene reported 14% of the frame as moving content, and a fast pan
        reported 27%, while the dominant model was simultaneously fitting 99% of
        tracks.
        """
        gm = float(np.linalg.norm(global_motion))
        rm = float(np.linalg.norm(mean_residual))
        if gm < 1e-6 or rm < 1e-6:
            return False
        alignment = float(np.dot(mean_residual / rm, -global_motion / gm))
        ratio = rm / gm
        return (
            alignment >= STUCK_ALIGNMENT
            and STUCK_MAGNITUDE_LO <= ratio <= STUCK_MAGNITUDE_HI
        )

    def _coherent_dynamic_mask(
        self,
        points: np.ndarray,
        residual_vectors: np.ndarray,
        residuals: np.ndarray,
        image_size: tuple[int, int],
        threshold: float,
        global_motion: np.ndarray,
    ) -> np.ndarray:
        """Cue 4: mark points inside spatially coherent high-residual regions.

        Per grid cell: is the residual large here, is it *consistent in
        direction* here, and is that direction something other than the camera
        motion negated? Sensor noise gives large residuals in random directions;
        stuck tracks give the camera motion negated; only a genuine moving object
        gives a coherent direction unrelated to the camera's.
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
            if float(np.median(cell_res)) < threshold:
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
            if coherence < 0.72:
                continue
            if self._is_stuck_track_cluster(vecs[good].mean(axis=0), global_motion):
                continue  # failed tracks, not an object
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
        parallax_present: bool = False,
        dt: float = 1.0 / 30.0,
    ) -> np.ndarray:
        """`parallax_present` must come from a real measurement (see
        tracking/parallax.py), not a guess — it selects which static-scene model
        this frame is judged against. `dt` is the measured time since the previous
        frame, which makes the drift cue frame-rate independent."""
        n = flow_result.count
        if n == 0:
            return np.empty((0,), np.float32)

        src, dst = flow_result.prev_points, flow_result.curr_points
        model = self._provisional_model(src, dst)
        if model is None:
            return np.full(n, 0.5, np.float32)

        similarity_residuals = self._residuals(model, src, dst)
        pred = (np.asarray(model)[:2, :2] @ src.T).T + np.asarray(model)[:2, 2]
        residual_vectors = dst - pred

        # Judge against the best rigid-scene interpretation, not against a single
        # 2D transform — otherwise parallax reads as moving content.
        residuals, self._model_used, drift_vectors = self._rigid_scene_residuals(
            src, dst, similarity_residuals, use_epipolar=parallax_present,
            similarity_vectors=residual_vectors,
        )
        decay = float(np.exp(-max(dt, 1e-6) / DRIFT_TIME_CONSTANT))

        # Scale the tolerance to this transition's own motion, not a constant.
        median_flow = float(np.median(np.linalg.norm(dst - src, axis=1))) if len(src) else 0.0
        threshold = residual_threshold_for(median_flow)

        # Dominant image motion for this transition, used to recognise stuck
        # tracks. Median over all correspondences, which RANSAC has already
        # shown to be dominated by the true global motion.
        global_motion = np.median(dst - src, axis=0) if len(src) else np.zeros(2)

        disagrees = residuals > threshold
        coherent = self._coherent_dynamic_mask(
            src, residual_vectors, residuals, image_size, threshold, global_motion
        )

        # ---- cue 5: fold this frame's verdict into each track's history ----
        weights = np.full(n, 0.5, np.float32)
        for i, tid in enumerate(flow_result.track_ids.tolist()):
            tr = tracks.get(tid)
            if tr is None:
                continue

            drifting = False
            if drift_vectors is not None:
                tr.drift_x = decay * tr.drift_x + float(drift_vectors[i, 0])
                tr.drift_y = decay * tr.drift_y + float(drift_vectors[i, 1])
                drifting = float(np.hypot(tr.drift_x, tr.drift_y)) > DRIFT_THRESHOLD_PX

            if disagrees[i] or drifting:
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
            residual_penalty = float(np.clip(tr.residual_ema / (threshold * 2.0), 0.0, 1.0))

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
