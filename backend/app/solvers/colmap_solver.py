"""PyCOLMAP structure-from-motion — the primary geometric backend (spec §8).

Pipeline: select keyframes, materialise them, extract SIFT, match sequentially
with loop closure, run incremental mapping with bundle adjustment, read back the
camera poses.

Two details that decide whether the output is correct rather than merely
plausible:

  * COLMAP stores **world-to-camera**. The camera centre is `-R_wc^T . t_wc`.
    This module reads the rotation matrix directly and then cross-checks its own
    centre against COLMAP's `projection_center()`, because a silent convention
    error here produces a mirrored trajectory with no other symptom.
  * `Rotation3d.quat` is **xyzw** (Eigen, scalar last) while everything in this
    codebase is **wxyz**. Rather than convert and hope, the matrix is used as
    the source of truth and the quaternion is derived from it.

Dense reconstruction is never run: this product needs cameras, not geometry.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from app.core.logging import get_logger
from app.geometry.conventions import cv_rotation_to_camerapath
from app.geometry.rotations import matrix_to_quat
from app.models.schemas.trajectory import SolverSource
from app.solvers.anchor_validation import validate_anchor_rotations
from app.solvers.base import GeometryResult, SolveContext, failed_result, select_keyframes
from app.video.decoder import materialize_frames

log = get_logger("solvers.colmap")

#: Below this many keyframes an incremental reconstruction has nothing to work
#: with — it needs an initial pair plus enough views to grow from.
MIN_KEYFRAMES = 6

#: Median triangulation angle an initial pair must reach. COLMAP's default of
#: 16 deg suits wide-baseline photo collections; consecutive video keyframes on a
#: dolly rarely reach it. 4 deg still rejects a near-duplicate pair, and poorly
#: triangulated individual points are removed by filter_min_tri_angle regardless.
INIT_MIN_TRI_ANGLE_DEG = 4.0

#: Reprojection error above which a reconstruction is treated as unreliable, px.
MAX_ACCEPTABLE_REPROJECTION_ERROR = 4.0

#: Focal is probed by pinning it this far either side of the solution and letting
#: poses and points re-optimise.
FOCAL_PROBE_STEP = 0.15

#: Minimum relative rise in reprojection error, on the FLATTER side of the probe,
#: for focal length to count as measured. Calibrated on synthetic ground truth:
#:   forward dolly  rise 0.00%   solved FOV off by  11.1 deg
#:   lateral truck  rise 0.02%   solved FOV off by  51.8 deg  (13.7 vs 65.5)
#:   orbit          rise 45%     solved FOV off by   0.13 deg
#: The gap spans three orders of magnitude; 3% sits well inside it. Bundle
#: adjustment's own "refined" focal is not evidence — on the unobservable shots it
#: simply drifted from whatever prior it was given.
FOCAL_OBSERVABLE_MIN_RISE = 0.03

#: A registered image is treated as mis-registered when it shows BOTH far fewer
#: triangulated observations than its peers AND a much higher reprojection error.
#: Measured on synthetic ground truth: two keyframes COLMAP placed 48-55 deg wrong
#: (camera grazing a wall) had 0.16-0.23x the median observation count and
#: 2.1-2.4x the median error. Correct keyframes reached at most 1.14x error; the
#: first frame of a dolly had 0.26x observations (everything is distant) at a
#: normal 0.99x error, which is why both conditions are required. PROVISIONAL:
#: calibrated on three scenes; revisit as the synthetic suite grows.
MISREGISTERED_MAX_POINTS_RATIO = 0.5
MISREGISTERED_MIN_ERROR_RATIO = 1.7

#: Confidence attached to a focal that is only the prior. Low but not zero: the
#: prior is a stated assumption (see geometry/intrinsics.py), not noise.
PRIOR_FOCAL_CONFIDENCE = 0.1


class PyColmapBackend:
    """Classical SfM via pycolmap."""

    @property
    def source(self) -> SolverSource:
        return SolverSource.COLMAP

    def available(self) -> tuple[bool, str]:
        try:
            import pycolmap  # noqa: PLC0415, F401
        except Exception as exc:  # noqa: BLE001
            return False, f"pycolmap not importable ({type(exc).__name__})"
        return True, ""

    def suitable_for(self, context: SolveContext) -> tuple[bool, str]:
        """Refuse shots where SfM is degenerate rather than merely difficult.

        This is the important judgement in the whole module. Structure-from-motion
        does not fail loudly on a pure rotation — it returns a confident,
        arbitrary baseline. Declining to run is the honest outcome (spec §25).
        """
        if context.shot.frame_count < MIN_KEYFRAMES:
            return False, f"only {context.shot.frame_count} frames in this shot"
        if context.texture_score < 0.15:
            return False, (
                f"texture score {context.texture_score:.2f} — too few stable features "
                "for feature matching"
            )
        if context.parallax_score < 0.10:
            return False, (
                f"parallax {context.parallax_score:.2f} — a single homography explains "
                "the motion, so translation is not observable and SfM would return an "
                "arbitrary baseline rather than a measurement"
            )
        return True, ""

    # ------------------------------------------------------------------ solve

    def estimate(self, context: SolveContext) -> GeometryResult:
        import pycolmap

        total = context.shot.frame_count
        work = context.work_dir
        if work is None:
            return failed_result(self.source, "no work directory provided", total)

        keyframes = select_keyframes(
            context.motion_frames, context.shot, density=context.keyframe_density
        )
        if len(keyframes) < MIN_KEYFRAMES:
            return failed_result(
                self.source,
                f"only {len(keyframes)} keyframes selected, need {MIN_KEYFRAMES}",
                total,
            )

        context.report(
            f"COLMAP: {len(keyframes)} keyframes from {total} frames "
            f"at {context.geometry_long_edge} px"
        )

        image_dir = work / "images"
        db_path = work / "database.db"
        sparse_dir = work / "sparse"
        # A stale database from an earlier attempt would be silently reused and
        # produce results for the wrong frames.
        for path in (db_path,):
            path.unlink(missing_ok=True)
        for directory in (image_dir, sparse_dir):
            shutil.rmtree(directory, ignore_errors=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir.mkdir(parents=True, exist_ok=True)

        by_index = {fm.frame_index: fm for fm in context.frames_meta}
        times = [by_index[i].time_seconds for i in keyframes if i in by_index]
        indices = [i for i in keyframes if i in by_index]
        if len(indices) < MIN_KEYFRAMES:
            return failed_result(self.source, "keyframe timestamps unavailable", total)

        written = materialize_frames(
            context.info, indices, times, image_dir,
            long_edge=context.geometry_long_edge, prefix="k",
        )
        if len(written) < MIN_KEYFRAMES:
            return failed_result(
                self.source, f"only {len(written)} keyframes could be extracted", total
            )
        image_names = [p.name for p in written]
        name_to_index = {p.name: idx for p, idx in zip(written, indices)}
        context.progress(0.2, f"extracted {len(written)} keyframes")

        # --- intrinsics prior ------------------------------------------------
        # A SIMPLE_RADIAL model with a focal prior converges far more reliably
        # than letting COLMAP guess, and the prior comes from
        # geometry/intrinsics.py, which states its own provenance.
        first_w, first_h = _image_size(written[0])
        scaled = context.intrinsics.scaled_to(first_w, first_h)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        reader.camera_params = ",".join(
            str(v) for v in (scaled.fx, scaled.cx, scaled.cy, 0.0)
        )

        extraction = pycolmap.FeatureExtractionOptions()
        extraction.max_image_size = context.geometry_long_edge
        extraction.sift.max_num_features = int(context.max_features)
        # Affine shape estimation and domain-size pooling both improve matching
        # on wide-baseline pairs but cost several times the runtime; left off,
        # and the sequential+loop pairing below covers the same need.
        extraction.sift.estimate_affine_shape = False
        extraction.sift.domain_size_pooling = False

        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=image_dir,
                image_names=image_names,
                camera_mode=pycolmap.CameraMode.SINGLE,
                reader_options=reader,
                extraction_options=extraction,
                device=pycolmap.Device.cpu,
            )
        except Exception as exc:  # noqa: BLE001
            return failed_result(self.source, f"feature extraction failed: {exc}", total)
        context.progress(0.4, "features extracted")

        # --- matching --------------------------------------------------------
        # Sequential with quadratic overlap and loop detection: neighbouring
        # frames give short baselines, and the strategically separated pairs are
        # what stop a long shot's reconstruction from drifting (spec §8).
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = min(12, max(3, len(image_names) // 4))
        pairing.quadratic_overlap = True
        pairing.loop_detection = False  # needs a vocab tree we do not ship

        try:
            pycolmap.match_sequential(
                database_path=db_path,
                pairing_options=pairing,
                device=pycolmap.Device.cpu,
            )
        except Exception as exc:  # noqa: BLE001
            return failed_result(self.source, f"feature matching failed: {exc}", total)
        context.progress(0.6, "features matched")

        # --- incremental mapping --------------------------------------------
        options = _mapping_options(refine_focal=True)

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=image_dir,
                output_path=sparse_dir,
                options=options,
            )
        except Exception as exc:  # noqa: BLE001
            return failed_result(self.source, f"incremental mapping failed: {exc}", total)

        if not reconstructions:
            return failed_result(
                self.source,
                "COLMAP produced no reconstruction — the keyframes could not be "
                "registered into a consistent model",
                total,
            )
        context.progress(0.85, f"{len(reconstructions)} model(s) reconstructed")

        # Multiple models mean the shot fragmented. The largest is the best
        # available answer, but the fragmentation itself lowers confidence.
        best_id = max(reconstructions, key=lambda k: reconstructions[k].num_reg_images())
        reconstruction = reconstructions[best_id]
        fragmented = len(reconstructions) > 1

        # Must run before poses are read: if focal turns out to be unmeasurable
        # the poses are re-solved under the prior, and the poses handed on have
        # to be the ones consistent with the focal that is reported.
        context.progress(0.9, "testing whether focal length is observable")
        reconstruction, focal = self._resolve_focal(
            reconstruction, context, db_path=db_path, image_dir=image_dir, work=work
        )

        result = self._read_poses(
            reconstruction, name_to_index, context, fragmented=fragmented,
            model_count=len(reconstructions), focal=focal,
        )
        if not result.succeeded:
            return result

        # A reconstruction can hold a confidently wrong registration with
        # sub-pixel error; the dense flow is independent evidence against it.
        # The bound needs focal in ANALYSIS pixels, and the smaller candidate is
        # used so that uncertainty widens the bound instead of causing rejections.
        analysis_w, analysis_h = context.analysis_size
        candidates = [float(context.intrinsics.scaled_to(analysis_w, analysis_h).fx)]
        if result.focal_pixels and result.focal_image_width:
            candidates.append(result.focal_pixels * analysis_w / result.focal_image_width)
        validated, rejected = validate_anchor_rotations(
            result, context.motion_frames, min(candidates)
        )
        if rejected:
            context.report(
                f"COLMAP: rejected {len(rejected)} anchor(s) inconsistent with the image motion"
            )
            if not validated.succeeded:
                return failed_result(
                    self.source,
                    f"too few anchors survived flow-consistency checks: {validated.message}",
                    total,
                )
        return validated

    def _resolve_focal(
        self, reconstruction, context: SolveContext, *,
        db_path: Path, image_dir: Path, work: Path,
    ):
        """Decide whether focal was measured; if not, re-solve under the prior.

        Returns (reconstruction, focal_info). The reconstruction may be a new one.

        The re-solve is a full incremental mapping with focal fixed, not a bundle
        adjustment of the existing model at a new focal. That shortcut diverged:
        a lateral truck had solved to 5345 px against a 1108 px prior, and moving
        focal 4.8x with points frozen at their old depths put points behind
        cameras — reported mean reprojection error 7.5e150 px. Mapping from
        scratch under the prior converges cleanly, because along an unobservable
        focal direction the fit is flat by definition.
        """
        import pycolmap

        camera = next(iter(reconstruction.cameras.values()))
        width, height = int(camera.width), int(camera.height)
        solved = float(_value(camera, "focal_length_x"))
        prior = float(context.intrinsics.scaled_to(width, height).fx)
        fov = lambda f: float(np.degrees(2 * np.arctan(width / (2 * f))))  # noqa: E731

        sensitivity = probe_focal_sensitivity(reconstruction)
        if sensitivity is not None and sensitivity >= FOCAL_OBSERVABLE_MIN_RISE:
            confidence = float(min(0.9, 0.5 + 0.4 * min(1.0, sensitivity / 0.30)))
            return reconstruction, {
                "pixels": solved, "width": width, "observable": True,
                "sensitivity": sensitivity, "confidence": confidence,
                "note": f"focal measured (fit worsens {sensitivity:.0%} at +-15%)",
            }

        evidence = (
            f"fit changes only {sensitivity:.2%} at +-15%" if sensitivity is not None
            else "the observability probe could not run"
        )
        context.progress(0.92, "focal unobservable — re-solving under the prior")
        prior_dir = work / "sparse_prior"
        shutil.rmtree(prior_dir, ignore_errors=True)
        prior_dir.mkdir(parents=True, exist_ok=True)
        resolved = None
        try:
            # The database camera already carries the prior (set at extraction).
            candidates = pycolmap.incremental_mapping(
                database_path=db_path, image_path=image_dir, output_path=prior_dir,
                options=_mapping_options(refine_focal=False),
            )
            if candidates:
                best = max(candidates.values(), key=lambda r: r.num_reg_images())
                cost = float(best.compute_mean_reprojection_error())
                if (np.isfinite(cost) and cost <= MAX_ACCEPTABLE_REPROJECTION_ERROR
                        and best.num_reg_images() >= max(2, reconstruction.num_reg_images() // 2)):
                    resolved = best
        except Exception as exc:  # noqa: BLE001
            log.warning("re-solve under prior focal failed: %s", exc)

        if resolved is not None:
            log.info("focal unobservable (%s): using prior %.1f deg, not solved %.1f deg",
                     evidence, fov(prior), fov(solved))
            return resolved, {
                "pixels": prior, "width": width, "observable": False,
                "sensitivity": sensitivity, "confidence": PRIOR_FOCAL_CONFIDENCE,
                "note": (
                    f"focal length is not observable in this shot ({evidence}); poses "
                    f"re-solved under the {fov(prior):.0f} deg prior instead of the "
                    f"unconstrained {fov(solved):.0f} deg — set the FOV override if the "
                    "lens is known"
                ),
            }

        # Could not re-solve. The original poses are only consistent with the
        # solved focal, so report THAT value — flagged unmeasured, confidence
        # halved — rather than pairing those poses with a focal they were not
        # solved under.
        return reconstruction, {
            "pixels": solved, "width": width, "observable": False,
            "sensitivity": sensitivity, "confidence": PRIOR_FOCAL_CONFIDENCE * 0.5,
            "note": (
                f"focal length is not observable in this shot ({evidence}) and a re-solve "
                f"under the prior failed; the reported {fov(solved):.0f} deg is an "
                "unconstrained value, not a measurement"
            ),
        }

    # ------------------------------------------------------------------ poses

    def _read_poses(
        self,
        reconstruction,
        name_to_index: dict[str, int],
        context: SolveContext,
        *,
        fragmented: bool,
        model_count: int,
        focal: dict,
    ) -> GeometryResult:
        total = context.shot.frame_count

        entries: list[tuple[int, np.ndarray, np.ndarray]] = []
        convention_errors: list[float] = []
        misregistered = self._misregistered_images(reconstruction)

        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            if not _value(image, "has_pose"):
                continue
            frame_index = name_to_index.get(image.name)
            if frame_index is None or image.name in misregistered:
                continue

            # world -> camera, in OpenCV axis convention.
            cam_from_world = _value(image, "cam_from_world")
            r_wc = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
            t_wc = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3)

            # Camera centre. Derived here rather than taken on faith, then
            # cross-checked against COLMAP's own answer.
            centre = -r_wc.T @ t_wc
            try:
                colmap_centre = np.asarray(image.projection_center(), dtype=np.float64).reshape(3)
                convention_errors.append(float(np.linalg.norm(centre - colmap_centre)))
            except Exception:  # noqa: BLE001
                pass

            r_cw_cv = r_wc.T
            r_cw_cpl = cv_rotation_to_camerapath(r_cw_cv)
            entries.append((frame_index, centre, matrix_to_quat(r_cw_cpl)))

        if len(entries) < 2:
            return failed_result(
                self.source,
                f"only {len(entries)} camera(s) registered — not enough for a trajectory",
                total,
            )

        if convention_errors:
            worst = max(convention_errors)
            scale = float(np.linalg.norm(np.array([e[1] for e in entries]).std(axis=0))) + 1e-9
            if worst > max(1e-6, scale * 1e-4):
                # Our centre disagrees with COLMAP's. Rather than ship a possibly
                # mirrored trajectory, refuse.
                return failed_result(
                    self.source,
                    f"camera-centre convention check failed (max disagreement "
                    f"{worst:.3e}) — refusing to emit a possibly mirrored trajectory",
                    total,
                )

        if misregistered:
            log.warning("discarding %d likely mis-registered image(s): %s",
                        len(misregistered), ", ".join(sorted(misregistered)))

        entries.sort(key=lambda e: e[0])
        frame_indices = [e[0] for e in entries]
        positions = np.array([e[1] for e in entries])
        quaternions = [e[2] for e in entries]

        # --- quality evidence ------------------------------------------------
        try:
            mean_reproj = float(reconstruction.compute_mean_reprojection_error())
        except Exception:  # noqa: BLE001
            mean_reproj = None
        if mean_reproj is not None and (
            not np.isfinite(mean_reproj) or mean_reproj > MAX_ACCEPTABLE_REPROJECTION_ERROR * 4
        ):
            # A diverged optimisation can still leave plausible-looking cameras
            # behind. Reprojection error this far out means the model is not a
            # solution, whatever its poses look like.
            return failed_result(
                self.source,
                f"reconstruction did not converge (mean reprojection error "
                f"{mean_reproj:.3g} px) — poses discarded",
                total,
            )
        try:
            mean_track = float(reconstruction.compute_mean_track_length())
        except Exception:  # noqa: BLE001
            mean_track = 0.0
        point_count = int(reconstruction.num_points3D())

        # Median over 3D points of each point's mean reprojection error. This is
        # a median of per-track errors rather than per observation; it is labelled
        # median because it is one, which the previous value — the mean copied
        # into the median field — was not.
        try:
            point_errors = [float(pt.error) for pt in reconstruction.points3D.values()]
            median_reproj = float(np.median(point_errors)) if point_errors else None
        except Exception:  # noqa: BLE001
            median_reproj = None

        registered = len(entries)
        keyframe_total = max(len(name_to_index), 1)
        registered_ratio = registered / keyframe_total

        path_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        spread = float(np.linalg.norm(positions.std(axis=0)))
        # A reconstruction whose cameras barely move relative to the scene has
        # produced a rotation estimate, not a translation one. Report that
        # honestly rather than presenting numerical noise as a dolly.
        translation_observable = (
            context.parallax_score >= 0.12 and spread > 1e-6 and path_length > spread * 0.05
        )

        confidence = self._score(
            registered_ratio=registered_ratio,
            mean_reproj=mean_reproj,
            mean_track=mean_track,
            point_count=point_count,
            parallax=context.parallax_score,
            fragmented=fragmented,
        )

        notes = [
            f"{registered}/{keyframe_total} keyframes registered",
            f"{point_count} 3D points",
        ]
        if mean_reproj is not None:
            notes.append(f"mean reprojection error {mean_reproj:.2f} px")
        if mean_track:
            notes.append(f"mean track length {mean_track:.1f}")
        notes.append(focal["note"])
        if misregistered:
            notes.append(
                f"discarded {len(misregistered)} keyframe(s) with few observations and high "
                "reprojection error relative to their peers (likely mis-registered)"
            )
        if fragmented:
            notes.append(
                f"reconstruction fragmented into {model_count} models — using the largest"
            )
        if not translation_observable:
            notes.append(
                "translation is not reliably observable in this shot; the rotation "
                "estimate is usable but the baseline is not"
            )

        result = GeometryResult(
            source=self.source,
            frame_indices=frame_indices,
            positions=positions,
            quaternions=quaternions,
            per_pose_confidence=[confidence] * registered,
            focal_pixels=focal["pixels"],
            focal_confidence=focal["confidence"],
            focal_observable=focal["observable"],
            focal_sensitivity=focal["sensitivity"],
            focal_image_width=focal["width"],
            registered_frames=registered,
            total_frames=keyframe_total,
            mean_reprojection_error=mean_reproj,
            median_reprojection_error=median_reproj,
            track_count=point_count,
            mean_track_length=mean_track,
            translation_observable=translation_observable,
            confidence=confidence,
            message="; ".join(notes),
            succeeded=True,
        )
        log.info("COLMAP result: %s (confidence %.2f)", result.message, confidence)
        return result

    @staticmethod
    def _misregistered_images(reconstruction) -> set[str]:
        """Names of registered images whose registration evidence is far weaker
        than their peers' on BOTH counts. See MISREGISTERED_* for calibration."""
        stats: list[tuple[str, int, float]] = []
        for image_id in reconstruction.reg_image_ids():
            image = reconstruction.image(image_id)
            errors = [
                float(reconstruction.points3D[p.point3D_id].error)
                for p in image.points2D if p.has_point3D()
            ]
            if errors:
                stats.append((image.name, len(errors), float(np.mean(errors))))
        if len(stats) < 5:
            return set()  # too few peers for a median to mean anything
        median_points = float(np.median([n for _, n, _ in stats]))
        median_error = float(np.median([e for _, _, e in stats]))
        if median_points <= 0 or median_error <= 0:
            return set()
        return {
            name for name, n, e in stats
            if n / median_points < MISREGISTERED_MAX_POINTS_RATIO
            and e / median_error > MISREGISTERED_MIN_ERROR_RATIO
        }

    @staticmethod
    def _score(
        *,
        registered_ratio: float,
        mean_reproj: float | None,
        mean_track: float,
        point_count: int,
        parallax: float,
        fragmented: bool,
    ) -> float:
        """Confidence from evidence (spec §22). Deliberately not generous."""
        registration = float(np.clip(registered_ratio, 0.0, 1.0))

        if mean_reproj is None:
            reprojection = 0.5
        else:
            # 1.0 px is excellent, MAX_ACCEPTABLE is worthless.
            reprojection = float(np.clip(
                1.0 - (mean_reproj - 1.0) / (MAX_ACCEPTABLE_REPROJECTION_ERROR - 1.0),
                0.0, 1.0,
            ))

        # A mean track length of 4+ views means points are genuinely
        # multiply-observed rather than stitched from pairs.
        track_quality = float(np.clip((mean_track - 2.0) / 4.0, 0.0, 1.0))
        density = float(np.clip(point_count / 2500.0, 0.0, 1.0))
        parallax_term = float(np.clip(parallax / 0.4, 0.0, 1.0))

        score = (
            0.30 * registration
            + 0.25 * reprojection
            + 0.18 * track_quality
            + 0.12 * density
            + 0.15 * parallax_term
        )
        if fragmented:
            score *= 0.7
        return float(np.clip(score, 0.0, 1.0))


def _mapping_options(*, refine_focal: bool):
    """Incremental-mapping configuration.

    With `refine_focal` False, focal AND radial distortion are held fixed. The two
    trade against each other, and on a shot where focal is unobservable letting k1
    float made the drift worse (measured: FOV error -11.1 deg with k1 free versus
    -4.7 deg with it fixed, from the same prior).
    """
    import pycolmap

    options = pycolmap.IncrementalPipelineOptions()
    options.ba_refine_focal_length = refine_focal
    options.ba_refine_principal_point = False  # unreliable without a target
    options.ba_refine_extra_params = refine_focal
    options.extract_colors = False  # cameras only; colours cost time for nothing
    options.max_num_models = 3

    # Initial-pair gates. COLMAP's defaults are tuned for unordered photo
    # collections and reject exactly the camera moves this product exists to
    # recover. Measured on a synthetic dolly with thousands of verified
    # inliers per pair, every pair was refused and mapping failed with "No
    # good initial image pair found", for two reasons:
    #
    #  * init_max_forward_motion (default 0.95) rejects pairs whose
    #    translation lies along the optical axis. A dolly, drone fly-through
    #    or FPV push has a forward-motion fraction of ~1.0 by definition —
    #    the synthetic dolly measured 1.0000. Forward motion is not
    #    degenerate: the epipole sits at the image centre, and only points
    #    NEAR it triangulate poorly. Those are culled per point by
    #    filter_min_tri_angle, which stays at its default.
    #  * init_min_tri_angle (default 16 deg) demands a median triangulation
    #    angle that consecutive video keyframes rarely reach; it exists to
    #    stop a wide-baseline photo set initialising on a near-duplicate.
    #
    # Relaxing these is safe here specifically because degeneracy is already
    # guarded upstream: a shot only reaches COLMAP when wide-baseline parallax
    # has been measured (tracking/parallax.py), so a pure rotation or zoom —
    # the case these gates really protect against — never gets this far.
    options.mapper.init_max_forward_motion = 1.0
    options.mapper.init_min_tri_angle = INIT_MIN_TRI_ANGLE_DEG
    try:
        options.max_runtime_seconds = 900
    except AttributeError:
        pass
    return options


def _set_focal(camera, focal: float) -> None:
    """Set focal on single- or dual-focal camera models."""
    try:
        camera.focal_length = focal
    except Exception:  # noqa: BLE001 - models with separate fx/fy reject the joint setter
        camera.focal_length_x = focal
        camera.focal_length_y = focal


def _bundle_adjust(reconstruction, *, refine_focal: bool, refine_extra: bool) -> None:
    import pycolmap

    options = pycolmap.BundleAdjustmentOptions()
    options.refine_focal_length = refine_focal
    options.refine_extra_params = refine_extra
    options.refine_principal_point = False
    try:
        options.print_summary = False
    except AttributeError:
        pass
    pycolmap.bundle_adjustment(reconstruction, options)


def _pinned_focal_cost(reconstruction, factor: float) -> float:
    """Mean reprojection error with focal pinned at `factor` x its current value
    and everything else re-optimised. Works on a copy."""
    import pycolmap

    trial = pycolmap.Reconstruction(reconstruction)
    for camera in trial.cameras.values():
        _set_focal(camera, float(_value(camera, "focal_length_x")) * factor)
    _bundle_adjust(trial, refine_focal=False, refine_extra=False)
    return float(trial.compute_mean_reprojection_error())


def probe_focal_sensitivity(reconstruction) -> float | None:
    """Profile-likelihood test: how much worse does the fit get when focal is
    forced away from the solution?

    Returns the relative rise on the flatter side, or None if the probe could not
    run. The flatter side decides because a focal bounded in only one direction
    is still not measured. A pinned solve that diverges (non-finite cost) counts
    as a large rise on that side — bundle adjustment converges easily along a flat
    valley, so divergence is evidence that focal was constrained.
    """
    try:
        base = _pinned_focal_cost(reconstruction, 1.0)
    except Exception as exc:  # noqa: BLE001
        log.warning("focal probe baseline failed: %s", exc)
        return None
    if not np.isfinite(base) or base <= 0:
        return None
    rises: list[float] = []
    for factor in (1.0 - FOCAL_PROBE_STEP, 1.0 + FOCAL_PROBE_STEP):
        try:
            cost = _pinned_focal_cost(reconstruction, factor)
        except Exception as exc:  # noqa: BLE001
            log.warning("focal probe at x%.2f failed: %s", factor, exc)
            continue
        rises.append((cost - base) / base if np.isfinite(cost) else float("inf"))
    return float(min(rises)) if rises else None


def _value(obj, name: str):
    """Read a pycolmap accessor that may be a property or a method.

    pycolmap has flipped several of these between releases, in both directions:
    `Image.cam_from_world` became a method in 4.x while `Camera.focal_length_x`
    became a plain float. Calling a float raises TypeError, and the broad
    handlers around quality evidence then silently reported "no focal length"
    for a reconstruction that had one. Resolving the shape at runtime keeps both
    3.x and 4.x correct without a version switch.
    """
    attr = getattr(obj, name)
    return attr() if callable(attr) else attr


def _image_size(path: Path) -> tuple[int, int]:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 1600, 900
    return int(img.shape[1]), int(img.shape[0])
