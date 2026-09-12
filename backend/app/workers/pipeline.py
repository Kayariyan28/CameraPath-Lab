"""Pipeline orchestration.

Runs the analysis stages for a job, in a worker thread, reporting progress to the
event bus. The pipeline never raises into the caller: a stage failure is caught,
recorded on the job, and the best honest partial result is kept (invariant I12).

Phase 1 implements stages 1-6 (probe, decode, shots, motion, rejection, routing).
Stages 7-12 (geometry, fusion, validation, Blender, render, export) attach to the
same structure and are wired in later phases; `route_shot` already decides which
solver each shot *should* get so the routing logic is exercised and visible now.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass

from app.core.environment import Environment, detect_environment
from app.core.logging import StageReporter, get_logger
from app.core.paths import Workspace
from app.models.schemas.jobs import (
    AnalysisResult,
    Job,
    JobState,
    ShotAnalysis,
    SolveSettings,
    Stage,
    StageProgress,
)
from app.models.schemas.motion import MotionSignature
from app.models.schemas.trajectory import PipelineMode
from app.models.schemas.video import Shot, VideoInfo
from app.tracking.shot_motion import ShotMotionResult, analyze_shot_motion
from app.video.ffprobe import probe_frames, probe_video
from app.video.shots import detect_shots, estimate_complexity
from app.workers.jobstore import JobStore

log = get_logger("workers.pipeline")


class Cancelled(RuntimeError):
    pass


@dataclass
class ShotRouting:
    mode: PipelineMode
    reason: str


def route_shot(
    signature: MotionSignature,
    requested: PipelineMode,
    *,
    texture: float,
    blur: float,
) -> ShotRouting:
    """Decide which pipeline a shot should use.

    AUTO is the interesting case. The question is not "is this shot hard?" but
    "is physical translation *observable*?" — a distinction that matters because
    structure-from-motion does not fail loudly on a pure pan. It happily returns
    a confident, large, entirely fabricated baseline. So AUTO routes away from
    the physical solver whenever the evidence for parallax is absent, and says
    why (spec §12, §22, §25).
    """
    if requested is not PipelineMode.AUTO:
        return ShotRouting(requested, f"{requested.value} requested explicitly")

    if signature.frame_count < 8:
        return ShotRouting(
            PipelineMode.PERCEPTUAL_MATCH,
            f"only {signature.frame_count} frames — too short for a stable geometric solve",
        )

    if signature.mean_flow_magnitude < 0.4:
        return ShotRouting(
            PipelineMode.PERCEPTUAL_MATCH,
            "the camera is effectively static; there is no motion to triangulate",
        )

    if texture < 0.2:
        return ShotRouting(
            PipelineMode.PERCEPTUAL_MATCH,
            f"texture score {texture:.2f} — too few stable features for reconstruction",
        )

    if signature.parallax_score < 0.15:
        pct = signature.homography_dominance
        return ShotRouting(
            PipelineMode.PERCEPTUAL_MATCH,
            (
                f"a single homography explains {pct:.0%} of the motion and parallax "
                f"is {signature.parallax_score:.2f}. The scene is planar, distant, or "
                "the motion is pure rotation/zoom, so physical translation is not "
                "observable — screen-space matching is used instead of inventing a "
                "camera path"
            ),
        )

    if blur < 0.25:
        return ShotRouting(
            PipelineMode.PERCEPTUAL_MATCH,
            f"heavy motion blur (sharpness {blur:.2f}) degrades feature geometry",
        )

    if signature.parallax_score >= 0.4 and signature.mean_inlier_ratio >= 0.6:
        return ShotRouting(
            PipelineMode.PHYSICAL_3D,
            (
                f"strong parallax ({signature.parallax_score:.2f}) and "
                f"{signature.mean_inlier_ratio:.0%} model agreement — translation is "
                "geometrically observable"
            ),
        )

    return ShotRouting(
        PipelineMode.PHYSICAL_3D,
        (
            f"moderate parallax ({signature.parallax_score:.2f}) — attempting a "
            "geometric solve, with perceptual fallback if confidence is low"
        ),
    )


class AnalysisPipeline:
    """Stages 1-6: everything available before a geometric solve."""

    def __init__(self, store: JobStore, workspace: Workspace,
                 environment: Environment | None = None):
        self.store = store
        self.workspace = workspace
        self.environment = environment or detect_environment(workspace.root)

    # ----------------------------------------------------------------- helper

    def _check_cancelled(self, job_id: str) -> None:
        if self.store.is_cancelled(job_id):
            raise Cancelled(job_id)

    def _set_stage(self, job_id: str, reporter: StageReporter, stage: Stage,
                   message: str = "") -> None:
        reporter.stage(stage.value, message)

        def mutate(job: Job) -> None:
            from datetime import datetime, timezone
            if job.current_stage is not None:
                job.current_stage.completed_at = datetime.now(timezone.utc)
                job.stage_history.append(job.current_stage)
            job.current_stage = StageProgress(
                stage=stage, progress=0.0, message=message,
                started_at=datetime.now(timezone.utc),
            )
        self.store.update(job_id, mutate)

    # ------------------------------------------------------------------ stages

    def run(self, job_id: str, settings: SolveSettings | None = None) -> Job:
        """Full analysis. Returns the job whatever happens."""
        job = self.store.get(job_id)
        paths = self.workspace.job(job_id)
        reporter = StageReporter(job_id, paths.log_file)
        self.store.clear_cancel(job_id)

        if settings is not None:
            self.store.update(job_id, lambda j: setattr(j, "settings", settings))
            job = self.store.get(job_id)
        settings = job.settings

        source = paths.source_video()
        if source is None:
            self.store.set_state(job_id, JobState.FAILED, "no video uploaded")
            reporter.error("no video uploaded")
            return self.store.get(job_id)

        started = time.time()
        self.store.set_state(job_id, JobState.ANALYZING)

        try:
            policy = self.environment.resource_policy(
                high_accuracy=settings.mode is PipelineMode.HIGH_ACCURACY
            )
            long_edge = settings.max_analysis_resolution or policy.analysis_long_edge
            if settings.mode is PipelineMode.FAST:
                long_edge = min(long_edge, 720)

            # ---- stage 1-2: probe ------------------------------------------
            self._set_stage(job_id, reporter, Stage.PREPARING_FRAMES, f"probing {source.name}")
            info = probe_video(source)
            frames_meta, info = probe_frames(info)
            self._check_cancelled(job_id)

            reporter.info(
                f"{info.width}x{info.height} {info.codec_name}, "
                f"{len(frames_meta)} frames, {info.duration_seconds:.3f}s, "
                f"{info.fps_average:.3f} fps ({info.frame_rate_mode.value}), "
                f"timing from {info.timing_source.value}"
            )
            self.store.update(job_id, lambda j: setattr(j, "video", info))

            warnings: list[str] = []
            if info.frame_rate_mode.value == "variable":
                warnings.append(
                    f"Variable frame rate detected (jitter {info.fps_jitter:.1%}). "
                    "Real per-frame timestamps are being used, so timing is preserved."
                )
            if not frames_meta:
                raise RuntimeError("no frames could be read from this file")
            if len(frames_meta) < 4:
                warnings.append(
                    f"Only {len(frames_meta)} frames — too short to recover a trajectory."
                )

            # ---- stage 3: shots --------------------------------------------
            self._set_stage(job_id, reporter, Stage.PREPARING_FRAMES, "detecting shot cuts")
            shots, cut_candidates = detect_shots(info, frames_meta, reporter=reporter)
            self._check_cancelled(job_id)
            if len(shots) > 1:
                reporter.info(
                    f"{len(shots)} shots detected — each gets an independent "
                    "coordinate system; no trajectory will span a cut"
                )

            # ---- stages 4-6: per-shot motion -------------------------------
            analyses: list[ShotAnalysis] = []
            motion_results: dict[int, ShotMotionResult] = {}

            for shot_index, shot in enumerate(shots):
                self._check_cancelled(job_id)
                reporter.set_shot(shot.id, index=shot_index, count=len(shots))
                self._set_stage(
                    job_id, reporter, Stage.TRACKING_FEATURES,
                    f"shot {shot.id + 1}/{len(shots)}: {shot.frame_count} frames",
                )
                motion = analyze_shot_motion(
                    info, shot, frames_meta,
                    long_edge=long_edge,
                    max_features=policy.max_tracked_features,
                    dynamic_strength=settings.dynamic_rejection_strength,
                    reporter=reporter,
                )
                motion_results[shot.id] = motion
                warnings.extend(f"Shot {shot.id + 1}: {w}" for w in motion.warnings)

                complexity, reasons = estimate_complexity(
                    frame_count=shot.frame_count,
                    duration=shot.duration,
                    mean_flow=motion.signature.mean_flow_magnitude,
                    inlier_ratio=motion.signature.mean_inlier_ratio,
                    texture_score=motion.texture,
                    parallax_score=motion.signature.parallax_score,
                )
                routing = route_shot(
                    motion.signature, settings.mode,
                    texture=motion.texture, blur=motion.blur,
                )
                reporter.info(
                    f"shot {shot.id}: {complexity.value} complexity, "
                    f"routed to {routing.mode.value} — {routing.reason}"
                )
                analyses.append(
                    ShotAnalysis(
                        shot=shot,
                        signature=motion.signature,
                        complexity=complexity,
                        complexity_reasons=reasons,
                        recommended_mode=routing.mode,
                        recommendation_reason=routing.reason,
                    )
                )

            reporter.set_shot(None)

            from app.video.decoder import fit_long_edge
            aw, ah = fit_long_edge(info.width, info.height, long_edge)
            result = AnalysisResult(
                video=info,
                shots=shots,
                cut_candidates=cut_candidates,
                shot_analyses=analyses,
                analysis_resolution=[aw, ah],
                warnings=warnings,
            )
            self._persist_motion(job_id, motion_results)

            def mutate(job: Job) -> None:
                job.analysis = result
                job.state = JobState.ANALYZED
                job.error = None
            self.store.update(job_id, mutate)

            elapsed = time.time() - started
            reporter.info(f"analysis complete in {elapsed:.1f}s")
            reporter.stage(Stage.COMPLETE.value, "analysis complete")
            return self.store.get(job_id)

        except Cancelled:
            reporter.warning("cancelled by user")
            self.store.set_state(job_id, JobState.CANCELLED, "cancelled")
            return self.store.get(job_id)
        except Exception as exc:  # noqa: BLE001 - I12: never take down the app
            detail = traceback.format_exc(limit=12)
            log.exception("analysis failed for %s", job_id)
            reporter.error(f"analysis failed: {type(exc).__name__}: {exc}")

            def mutate(job: Job) -> None:
                job.state = JobState.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                job.error_detail = detail
            self.store.update(job_id, mutate)
            return self.store.get(job_id)

    # ------------------------------------------------------------- artefacts

    def _persist_motion(self, job_id: str, results: dict[int, ShotMotionResult]) -> None:
        """Cache per-shot motion frames so later stages (and a proxy-style change)
        do not have to recompute them — spec §31."""
        from app.core.paths import atomic_write_json
        paths = self.workspace.job(job_id)
        for shot_id, res in results.items():
            try:
                atomic_write_json(
                    paths.shot_dir(shot_id) / "motion_frames.json",
                    {
                        "shot_id": shot_id,
                        "analysis_size": list(res.analysis_size),
                        "texture": res.texture,
                        "blur": res.blur,
                        "persistent_track_count": res.persistent_track_count,
                        "mean_track_age": res.mean_track_age,
                        "dynamic_region_fraction": res.dynamic_region_fraction,
                        "signature": res.signature.model_dump(),
                        "motion_frames": [m.model_dump() for m in res.motion_frames],
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("could not cache motion for shot %d: %s", shot_id, exc)


def load_cached_motion(workspace: Workspace, job_id: str, shot_id: int) -> dict | None:
    from app.core.paths import read_json
    path = workspace.job(job_id).shot_dir(shot_id) / "motion_frames.json"
    raw = read_json(path)
    return raw if isinstance(raw, dict) else None
