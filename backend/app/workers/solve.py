"""Stages 7-12: geometry, fusion, validation, export and render.

Runs on a job that has already been analysed. Every shot is solved independently
in its own coordinate system (I4), then all shots are exported and rendered.

Render is deliberately separable (`SolvePipeline.render`): it reads the exported
trajectory JSON, so changing only the proxy style or output resolution re-renders
without recomputing any geometry (spec §31).

Nothing here may take the application down (I12). A shot whose solve fails
degrades down the ladder — COLMAP, then Perceptual Match, then a 2D motion proxy —
and a render failure leaves the trajectory exports intact.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.blender.runner import BlenderRunner, probe_mp4
from app.core.environment import Environment, detect_environment
from app.core.logging import StageReporter, get_logger
from app.core.paths import Workspace
from app.geometry.conventions import orient_trajectory_z_up
from app.geometry.intrinsics import build_lens_curve, initial_intrinsics
from app.models.schemas.jobs import Job, JobState, SolveSettings, Stage, StageProgress
from app.models.schemas.motion import MotionFrame, MotionSignature
from app.models.schemas.trajectory import (
    ConfidenceLevel,
    ConfidenceReport,
    PipelineMode,
    ScaleMode,
    ShotTrajectory,
    SolverSource,
)
from app.solvers.base import SolveContext, solve_with_fallback
from app.solvers.colmap_solver import PyColmapBackend
from app.solvers.opencv_solver import OpenCVPoseBackend
from app.solvers.perceptual_solver import PerceptualMatchBackend
from app.tracking.keyframe_geometry import KeyframeHomography
from app.trajectory.classify import classify_motion
from app.trajectory.exporters import TRAJECTORY_STEM, export_all, shot_stem
from app.trajectory.fusion import fuse_lens_curve, fuse_trajectory
from app.trajectory.kinematics import compute_kinematics, path_length, total_rotation_degrees
from app.trajectory.normalize import apply_metric_calibration, normalize_trajectory
from app.video.ffprobe import probe_frames, probe_video
from app.workers.jobstore import JobStore
from app.workers.pipeline import Cancelled, load_cached_motion

log = get_logger("workers.solve")

#: FAST mode trades geometric resolution and feature budget for speed.
FAST_GEOMETRY_SCALE = 0.6


@dataclass
class ShotInputs:
    motion_frames: list[MotionFrame]
    signature: MotionSignature
    analysis_size: tuple[int, int]
    texture: float
    blur: float
    keyframe_homographies: list = None  # list[KeyframeHomography]


class SolvePipeline:
    def __init__(self, store: JobStore, workspace: Workspace,
                 environment: Environment | None = None,
                 blender: BlenderRunner | None = None):
        self.store = store
        self.workspace = workspace
        self.environment = environment or detect_environment(workspace.root)
        self.blender = blender or BlenderRunner()

    # --------------------------------------------------------------- helpers

    def _check_cancelled(self, job_id: str) -> None:
        if self.store.is_cancelled(job_id):
            raise Cancelled(job_id)

    def _set_stage(self, job_id: str, reporter: StageReporter, stage: Stage, message: str = "") -> None:
        reporter.stage(stage.value, message)

        def mutate(job: Job) -> None:
            now = datetime.now(timezone.utc)
            if job.current_stage is not None:
                job.current_stage.completed_at = now
                job.stage_history.append(job.current_stage)
            job.current_stage = StageProgress(stage=stage, message=message, started_at=now)
        self.store.update(job_id, mutate)

    def _shot_inputs(self, job_id: str, shot_id: int) -> ShotInputs:
        raw = load_cached_motion(self.workspace, job_id, shot_id)
        if raw is None:
            raise RuntimeError(f"no cached motion for shot {shot_id}; re-run analysis")
        return ShotInputs(
            motion_frames=[MotionFrame.model_validate(m) for m in raw["motion_frames"]],
            signature=MotionSignature.model_validate(raw["signature"]),
            analysis_size=(int(raw["analysis_size"][0]), int(raw["analysis_size"][1])),
            texture=float(raw.get("texture", 0.0)),
            blur=float(raw.get("blur", 1.0)),
            keyframe_homographies=[
                KeyframeHomography.model_validate(k) for k in raw.get("keyframe_homographies", [])
            ],
        )

    def _backends(self, settings: SolveSettings, reporter: StageReporter) -> list:
        colmap_ok = settings.enable_colmap and PyColmapBackend().available()[0]
        if settings.mode is PipelineMode.PERCEPTUAL_MATCH:
            return [PerceptualMatchBackend()]
        if not colmap_ok:
            reporter.warning(
                "COLMAP disabled or unavailable; translation cannot be reconstructed, so "
                "rotation comes from keyframe geometry or Perceptual Match"
            )
            return [OpenCVPoseBackend(), PerceptualMatchBackend()]
        # PHYSICAL_3D still keeps Perceptual Match as the last rung: the product
        # always returns the best honest result (spec §28), and the confidence
        # report states that the physical solve did not succeed. The OpenCV rung
        # measures rotation from keyframe homographies on shots COLMAP declines
        # for lack of parallax.
        return [PyColmapBackend(), OpenCVPoseBackend(), PerceptualMatchBackend()]

    # ------------------------------------------------------------------ solve

    def run(self, job_id: str, settings: SolveSettings | None = None, *, render: bool = True) -> Job:
        """Solve every shot, export, and (optionally) render. Returns the job."""
        paths = self.workspace.job(job_id)
        reporter = StageReporter(job_id, paths.log_file)
        self.store.clear_cancel(job_id)
        if settings is not None:
            self.store.update(job_id, lambda j: setattr(j, "settings", settings))
        job = self.store.get(job_id)
        settings = job.settings

        if job.analysis is None or job.video is None:
            self.store.set_state(job_id, JobState.FAILED, "analyse the video before solving")
            return self.store.get(job_id)

        self.store.set_state(job_id, JobState.SOLVING)
        started = time.time()
        try:
            source = paths.source_video()
            info = probe_video(source)
            frames_meta, info = probe_frames(info)
            policy = self.environment.resource_policy(
                high_accuracy=settings.mode is PipelineMode.HIGH_ACCURACY
            )
            geometry_edge = policy.geometry_long_edge
            max_features = policy.colmap_max_num_features
            density = settings.keyframe_density
            if settings.mode is PipelineMode.FAST:
                geometry_edge = int(geometry_edge * FAST_GEOMETRY_SCALE)
                max_features = int(max_features * FAST_GEOMETRY_SCALE)
            elif settings.mode is PipelineMode.HIGH_ACCURACY:
                density *= 1.5

            trajectories: list[ShotTrajectory] = []
            for analysis in job.analysis.shot_analyses:
                shot = analysis.shot
                self._check_cancelled(job_id)
                reporter.set_shot(shot.id)
                trajectories.append(self._solve_shot(
                    job_id, info, frames_meta, shot, settings, reporter,
                    geometry_edge=geometry_edge, max_features=max_features, density=density,
                ))
            reporter.set_shot(None)

            def store_trajectories(j: Job) -> None:
                j.trajectories = trajectories
                j.state = JobState.SOLVED
                j.error = None
            self.store.update(job_id, store_trajectories)

            self._set_stage(job_id, reporter, Stage.VALIDATING_TRAJECTORY, "writing exports")
            job = self.store.get(job_id)
            outputs = export_all(job, trajectories, paths)
            self.store.update(job_id, lambda j: setattr(j, "outputs", outputs))
            reporter.info(f"exports written in {time.time() - started:.1f}s")

            if render:
                return self.render(job_id, reporter=reporter)

            reporter.stage(Stage.COMPLETE.value, "solve complete (render skipped)")
            return self.store.get(job_id)

        except Cancelled:
            reporter.warning("cancelled by user")
            self.store.set_state(job_id, JobState.CANCELLED, "cancelled")
            return self.store.get(job_id)
        except Exception as exc:  # noqa: BLE001 - I12
            log.exception("solve failed for %s", job_id)
            reporter.error(f"solve failed: {type(exc).__name__}: {exc}")
            detail = traceback.format_exc(limit=12)

            def fail(j: Job) -> None:
                j.state = JobState.FAILED
                j.error = f"{type(exc).__name__}: {exc}"
                j.error_detail = detail
            self.store.update(job_id, fail)
            return self.store.get(job_id)

    def _solve_shot(self, job_id, info, frames_meta, shot, settings: SolveSettings,
                    reporter: StageReporter, *, geometry_edge: int, max_features: int,
                    density: float) -> ShotTrajectory:
        inputs = self._shot_inputs(job_id, shot.id)
        width, height = inputs.analysis_size
        parallax = inputs.signature.parallax_score

        self._set_stage(job_id, reporter, Stage.RECOVERING_LENS_MOTION, f"shot {shot.id + 1}")
        intrinsics, provenance = initial_intrinsics(
            info, width, height, fov_override_degrees=settings.fov_override_degrees
        )
        lens, lens_note = build_lens_curve(
            inputs.motion_frames, intrinsics, parallax_score=parallax,
            lens_mode=settings.lens_mode.value,
            keyframe_homographies=inputs.keyframe_homographies,
        )
        reporter.info(f"shot {shot.id}: intrinsics — {provenance}; lens — {lens_note}")

        self._set_stage(job_id, reporter, Stage.ESTIMATING_CAMERA_GEOMETRY, f"shot {shot.id + 1}")
        work_dir = self.workspace.job(job_id).geometry_dir / f"shot_{shot.id:03d}"
        work_dir.mkdir(parents=True, exist_ok=True)
        context = SolveContext(
            info=info, shot=shot, frames_meta=frames_meta,
            motion_frames=inputs.motion_frames, intrinsics=intrinsics,
            analysis_size=inputs.analysis_size, parallax_score=parallax,
            texture_score=inputs.texture, work_dir=work_dir,
            geometry_long_edge=geometry_edge, max_features=max_features,
            keyframe_density=density, reporter=reporter,
            keyframe_homographies=inputs.keyframe_homographies, lens=lens,
        )
        geometry, decisions = solve_with_fallback(
            self._backends(settings, reporter), context,
            confidence_threshold=settings.confidence_threshold,
        )
        if settings.mode is PipelineMode.PHYSICAL_3D and geometry.source is not SolverSource.COLMAP:
            reporter.warning(
                f"shot {shot.id}: physical 3D reconstruction was requested but could not be "
                f"produced; returning {geometry.source.value} instead"
            )
        # Both COLMAP and the keyframe-rotation rung MEASURE the camera physically
        # (the latter rotation only); Perceptual Match optimises screen-space
        # appearance instead. `translation_observable` says which physical kind.
        mode_used = (
            PipelineMode.PHYSICAL_3D
            if geometry.source in (SolverSource.COLMAP, SolverSource.OPENCV)
            else PipelineMode.PERCEPTUAL_MATCH
        )

        self._set_stage(job_id, reporter, Stage.OPTIMIZING_CAMERA_POSES, f"shot {shot.id + 1}")
        fused_lens, fused_lens_note = fuse_lens_curve(geometry, lens, frames_meta, shot, intrinsics)
        poses = fuse_trajectory(
            geometry, inputs.motion_frames, frames_meta, shot, fused_lens, intrinsics,
            inputs.analysis_size, fidelity=settings.motion_fidelity,
            translation_observable=geometry.translation_observable,
        )

        # ---- world orientation --------------------------------------------------
        # A geometric solve's world has no gravity: COLMAP's is its first camera's
        # OpenCV frame, so exports claimed +Z up while the cameras' up pointed
        # along -Y and the proxy filmed the cage sideways. Perceptual Match
        # already defines its world from a level first camera.
        if geometry.source in (SolverSource.COLMAP, SolverSource.OPENCV, SolverSource.VGGT) and poses:
            poses = self._orient_z_up(poses, reporter, shot.id)

        # ---- scale (I5) --------------------------------------------------------
        poses, normalized_scale = normalize_trajectory(poses)
        scale_mode, units, metric_factor = ScaleMode.NORMALIZED, "normalized", None
        if settings.scale_mode is ScaleMode.METRIC:
            if settings.scale_calibration is None:
                reporter.warning(
                    "metric scale requested without a calibration; monocular video has no "
                    "absolute scale, so the trajectory stays normalized"
                )
            else:
                poses, metric_factor, statement = apply_metric_calibration(
                    poses, settings.scale_calibration, normalized_scale
                )
                scale_mode, units = ScaleMode.METRIC, "m"
                reporter.info(f"shot {shot.id}: metric scale — {statement}")

        kinematics = compute_kinematics(poses, scale_mode)
        # Post-hoc description only; nothing below feeds back into the solve.
        moves, move_summary = classify_motion(
            poses, kinematics, fused_lens,
            jitter_score=inputs.signature.jitter_score,
            translation_observable=geometry.translation_observable,
            units=units,
        )

        self._set_stage(job_id, reporter, Stage.VALIDATING_TRAJECTORY, f"shot {shot.id + 1}")
        confidence = self._confidence(
            geometry, inputs, poses, fused_lens, decisions, mode_used, reporter
        )
        fps = float(info.fps_average or info.fps_nominal or 30.0)
        trajectory = ShotTrajectory(
            shot_id=shot.id,
            frame_count=len(poses),
            duration=float(shot.duration),
            fps=fps,
            scale_mode=scale_mode,
            scale_units=units,
            metric_scale_factor=metric_factor,
            poses=poses,
            kinematics=kinematics,
            lens=fused_lens,
            confidence=confidence,
            classified_moves=moves,
            summary=f"{move_summary} {self._summary(poses, geometry, mode_used, fused_lens_note, units)}",
            solver_decisions=decisions,
            pipeline_mode_used=mode_used,
            motion_fidelity=settings.motion_fidelity,
            total_path_length=path_length(poses),
            total_rotation_deg=total_rotation_degrees(poses),
        )
        reporter.info(
            f"shot {shot.id}: {len(poses)} poses via {geometry.source.value}, "
            f"confidence {confidence.level.value} ({confidence.score:.2f})"
        )
        return trajectory

    @staticmethod
    def _orient_z_up(poses, reporter: StageReporter, shot_id: int):
        positions = np.array([p.position for p in poses], dtype=np.float64)
        quats = [np.array(p.quaternion, dtype=np.float64) for p in poses]
        positions, quats, rotation = orient_trajectory_z_up(positions, quats)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        reporter.info(
            f"shot {shot_id}: world re-oriented to Z-up by {tilt:.1f} deg "
            "(estimated from camera orientations; relative motion unchanged)"
        )
        return [
            p.model_copy(update={"position": [float(v) for v in pos], "quaternion": [float(v) for v in q]})
            for p, pos, q in zip(poses, positions, quats)
        ]

    def _confidence(self, geometry, inputs: ShotInputs, poses, lens, decisions, mode_used,
                    reporter: StageReporter) -> ConfidenceReport:
        try:
            from app.validation.confidence import assess_confidence
            return assess_confidence(
                geometry, inputs.signature, inputs.motion_frames, poses, lens, decisions,
                inputs.signature.parallax_score, inputs.texture, inputs.blur, mode_used,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a solved shot over its report
            log.exception("confidence assessment failed")
            reporter.warning(f"confidence assessment failed ({exc}); reporting solver confidence only")
            score = float(np.clip(geometry.confidence, 0.0, 1.0))
            # Capped at MEDIUM: without the full assessment nothing justifies HIGH (I7).
            level = ConfidenceLevel.MEDIUM if score >= 0.45 else ConfidenceLevel.LOW
            return ConfidenceReport(
                level=level, score=score,
                headline=f"{level.value.upper()} CONFIDENCE (solver estimate only)",
                reasons=[geometry.message or "no solver message",
                         "the full confidence assessment could not run"],
                translation_observable=geometry.translation_observable,
            )

    @staticmethod
    def _summary(poses, geometry, mode_used: PipelineMode, lens_note: str, units: str) -> str:
        """Plain statement of what was measured. Numbers only — the post-hoc motion
        labels come from trajectory/classify.py."""
        if not poses:
            return "No poses were produced."
        rotation = total_rotation_degrees(poses)
        length = path_length(poses)
        fovs = [p.fov_horizontal for p in poses]
        parts = [
            f"{len(poses)} frames over {poses[-1].timestamp - poses[0].timestamp:.2f} s",
            f"{rotation:.1f} deg of total rotation",
        ]
        if geometry.translation_observable:
            parts.append(f"path length {length:.2f} {units}")
        else:
            parts.append("translation not observable — rotation and lens only")
        if max(fovs) - min(fovs) > 0.5:
            parts.append(f"FOV {fovs[0]:.1f} to {fovs[-1]:.1f} deg")
        if geometry.source is SolverSource.OPENCV:
            how = "rotation measured from long-baseline keyframe geometry, position held fixed"
        elif mode_used is PipelineMode.PHYSICAL_3D:
            how = "physical 3D reconstruction"
        else:
            how = "screen-space Perceptual Match (not a physical camera path)"
        return f"{'; '.join(parts)}. Solved by {how}. Optical camera pose, not a drone body pose."

    # ----------------------------------------------------------------- render

    def render(self, job_id: str, *, reporter: StageReporter | None = None) -> Job:
        """Render the motion proxy from the exported trajectory JSON."""
        paths = self.workspace.job(job_id)
        reporter = reporter or StageReporter(job_id, paths.log_file)
        job = self.store.get(job_id)
        settings = job.settings
        if not job.trajectories:
            self.store.set_state(job_id, JobState.FAILED, "nothing to render; solve first")
            return self.store.get(job_id)
        if not self.blender.available:
            reporter.warning("Blender not found; trajectory exports are complete but no MP4 was rendered")
            self.store.set_state(job_id, JobState.COMPLETE)
            return self.store.get(job_id)

        self.store.set_state(job_id, JobState.RENDERING)
        width, height = settings.output_width, settings.output_height
        if settings.match_source_aspect and job.video:
            aspect = job.video.width / max(job.video.height, 1)
            if aspect >= 1:
                width, height = int(round(height * aspect / 2) * 2), height
            else:
                width, height = width, int(round(width / aspect / 2) * 2)

        outputs_dir = paths.outputs_dir
        multi = len(job.trajectories) > 1
        shot_mp4s: list[Path] = []
        try:
            for index, traj in enumerate(job.trajectories):
                self._check_cancelled(job_id)
                stem = shot_stem(traj.shot_id) if multi else TRAJECTORY_STEM
                trajectory_json = outputs_dir / f"{stem}.json"
                out = outputs_dir / (f"motion_proxy_shot_{traj.shot_id:03d}.mp4" if multi else "motion_proxy.mp4")
                blend = paths.outputs_dir / ("camera_scene.blend" if index == 0 else f"camera_scene_shot_{traj.shot_id:03d}.blend")
                self._set_stage(job_id, reporter, Stage.RENDERING_MP4,
                                f"shot {index + 1}/{len(job.trajectories)} at {width}x{height}")
                result = self.blender.render_motion_proxy(
                    trajectory_json, out, style=settings.proxy_style.value,
                    width=width, height=height, fps=settings.output_fps,
                    blend_path=blend,
                    progress=lambda f, m: reporter.progress(f, m),
                )
                if not result.success:
                    raise RuntimeError(f"render failed for shot {traj.shot_id}: {result.error}")
                reporter.info(
                    f"shot {traj.shot_id}: rendered {result.frame_count} frames at "
                    f"{result.fps:.3f} fps ({result.duration_seconds:.3f}s) — timing verified"
                )
                shot_mp4s.append(out)

            final = outputs_dir / "motion_proxy.mp4"
            if multi:
                self._concat(shot_mp4s, final)

            def done(j: Job) -> None:
                j.outputs.motion_proxy_mp4 = str(final)
                first_blend = outputs_dir / "camera_scene.blend"
                j.outputs.camera_scene_blend = str(first_blend) if first_blend.is_file() else None
                j.state = JobState.COMPLETE
                j.error = None
            self.store.update(job_id, done)
            reporter.done("motion proxy rendered")
            return self.store.get(job_id)
        except Cancelled:
            self.store.set_state(job_id, JobState.CANCELLED, "cancelled")
            return self.store.get(job_id)
        except Exception as exc:  # noqa: BLE001 - I12: exports stay usable
            log.exception("render failed for %s", job_id)
            reporter.error(f"render failed: {exc}")

            def fail(j: Job) -> None:
                j.state = JobState.FAILED
                j.error = f"render failed (trajectory exports are still available): {exc}"
            self.store.update(job_id, fail)
            return self.store.get(job_id)

    @staticmethod
    def _concat(parts: list[Path], output: Path) -> None:
        """Join per-shot proxies, cuts preserved, and verify no frame was lost."""
        expected = sum(probe_mp4(p)[0] for p in parts)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            for p in parts:
                fh.write(f"file '{p.resolve().as_posix()}'\n")
            listing = fh.name
        cmd = [shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
               "-i", listing, "-c", "copy", "-movflags", "+faststart", str(output)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        Path(listing).unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"concatenation failed: {proc.stderr.strip()[:300]}")
        got = probe_mp4(output)[0]
        if got != expected:
            raise RuntimeError(f"concatenated proxy has {got} frames, expected {expected}")
