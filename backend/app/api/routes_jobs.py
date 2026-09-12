"""Job endpoints: create, upload, analyze, inspect, stream progress."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.api.deps import Services, get_services
from app.core.logging import EVENT_BUS, get_logger
from app.core.paths import InvalidJobId
from app.models.schemas.jobs import Job, JobState, JobSummary, SolveSettings
from app.workers.jobstore import JobNotFound

log = get_logger("api.jobs")

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: Upload chunk size. 4 MiB keeps memory flat on an 8 GiB source without
#: making the write syscall-bound.
UPLOAD_CHUNK = 4 * 1024 * 1024


def _get_job(services: Services, job_id: str) -> Job:
    try:
        return services.store.get(job_id)
    except InvalidJobId as exc:
        raise HTTPException(status_code=400, detail=f"malformed job id") from exc
    except (JobNotFound, KeyError) as exc:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found") from exc


# --------------------------------------------------------------------- create


@router.post("", status_code=201)
def create_job(services: Services = Depends(get_services)) -> Job:
    # Opportunistic cleanup: the workspace is bounded by age and size, and doing
    # it here means a long-running backend never needs a cron.
    try:
        services.store.gc()
    except Exception as exc:  # noqa: BLE001
        log.warning("gc failed: %s", exc)
    return services.store.create()


@router.get("")
def list_jobs(limit: int = 50, services: Services = Depends(get_services)) -> list[JobSummary]:
    return services.store.list_summaries(limit=limit)


# --------------------------------------------------------------------- upload


@router.post("/{job_id}/video")
async def upload_video(
    job_id: str,
    file: UploadFile = File(...),
    services: Services = Depends(get_services),
) -> Job:
    job = _get_job(services, job_id)
    settings = services.settings

    filename = Path(file.filename or "upload.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=(
                f"unsupported file type '{suffix}'. Supported: "
                f"{', '.join(settings.allowed_extensions)}"
            ),
        )

    paths = services.workspace.job(job_id).ensure()
    # Clear any previous upload so `source_video()` stays unambiguous.
    for existing in paths.source_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    target = paths.source_dir / filename
    written = 0
    try:
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"file exceeds {settings.max_upload_bytes / 1024**3:.1f} GiB limit",
                    )
                out.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"upload failed: {exc}") from exc
    finally:
        await file.close()

    if written == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="uploaded file is empty")

    # Probe immediately so the UI can show real facts before analysis runs, and
    # so an unreadable file fails here rather than deep in the pipeline.
    from app.video.ffprobe import FFprobeError, probe_frames, probe_video

    try:
        info = probe_video(target)
        # Also read per-frame timestamps now. This is what upgrades the reported
        # timing to CONTAINER_PTS and makes frame_count exact, so the info panel
        # shows measured facts rather than the stream's declared rate. It also
        # surfaces VFR before the user commits to a solve. Packet-mode probing
        # keeps this fast even on a long source.
        _frames, info = probe_frames(info)
    except FFprobeError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"could not read this file as video: {exc}",
        ) from exc

    def mutate(j: Job) -> None:
        j.video = info
        j.state = JobState.UPLOADED
        j.error = None
        j.analysis = None
        j.trajectories = []

    log.info("uploaded %s (%.1f MB) to job %s", filename, written / 1024**2, job_id)
    return services.store.update(job_id, mutate)


# -------------------------------------------------------------------- analyze


@router.post("/{job_id}/analyze")
def analyze(
    job_id: str,
    settings: SolveSettings | None = None,
    services: Services = Depends(get_services),
) -> dict:
    job = _get_job(services, job_id)
    if job.video is None:
        raise HTTPException(status_code=409, detail="upload a video first")
    if not services.mark_running(job_id):
        raise HTTPException(status_code=409, detail="this job is already running")

    pipeline = services.analysis_pipeline()

    def work() -> None:
        try:
            pipeline.run(job_id, settings)
        finally:
            services.mark_done(job_id)

    services.executor.submit(work)
    return {"job_id": job_id, "started": True, "stream": f"/api/jobs/{job_id}/events"}


@router.post("/{job_id}/cancel")
def cancel(job_id: str, services: Services = Depends(get_services)) -> dict:
    _get_job(services, job_id)
    services.store.request_cancel(job_id)
    return {"job_id": job_id, "cancelling": True}


# ------------------------------------------------------------------ inspection


@router.get("/{job_id}")
def get_job(job_id: str, services: Services = Depends(get_services)) -> Job:
    return _get_job(services, job_id)


@router.get("/{job_id}/analysis")
def get_analysis(job_id: str, services: Services = Depends(get_services)) -> dict:
    job = _get_job(services, job_id)
    if job.analysis is None:
        raise HTTPException(status_code=404, detail="not analysed yet")
    return job.analysis.model_dump()


@router.get("/{job_id}/motion/{shot_id}")
def get_motion(
    job_id: str, shot_id: int, services: Services = Depends(get_services)
) -> dict:
    """Dense per-frame motion curves for one shot. This is what the timeline
    charts plot, and it is the honest image-space signal — not camera poses."""
    _get_job(services, job_id)
    from app.workers.pipeline import load_cached_motion

    data = load_cached_motion(services.workspace, job_id, shot_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"no motion data for shot {shot_id}")
    return data


@router.get("/{job_id}/frame/{frame_index}")
def get_frame(
    job_id: str,
    frame_index: int,
    width: int = 640,
    services: Services = Depends(get_services),
):
    """Single source frame as JPEG, for the reference player and diagnostics."""
    job = _get_job(services, job_id)
    if job.video is None:
        raise HTTPException(status_code=409, detail="no video")
    paths = services.workspace.job(job_id)
    cache = paths.cache_dir / "preview"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"p{frame_index:06d}_{width}.jpg"

    if not out.is_file():
        import subprocess

        time_s = frame_index / max(job.video.fps_average or 30.0, 1e-6)
        cmd = [
            shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-nostdin",
            "-accurate_seek", "-ss", f"{time_s:.6f}",
            "-i", job.video.path, "-frames:v", "1",
            "-vf", f"scale={width}:-2:flags=area", "-q:v", "4", str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0 or not out.is_file():
            raise HTTPException(status_code=404, detail="could not extract that frame")
    return FileResponse(out, media_type="image/jpeg")


@router.get("/{job_id}/source")
def get_source_video(job_id: str, services: Services = Depends(get_services)):
    """Serve the uploaded video back for the reference player."""
    _get_job(services, job_id)
    source = services.workspace.job(job_id).source_video()
    if source is None or not source.is_file():
        raise HTTPException(status_code=404, detail="no source video")
    return FileResponse(source, filename=source.name)


@router.get("/{job_id}/log")
def get_log(job_id: str, tail: int = 500, services: Services = Depends(get_services)) -> dict:
    _get_job(services, job_id)
    path = services.workspace.job(job_id).log_file
    if not path.is_file():
        return {"lines": []}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-tail:]}


@router.delete("/{job_id}")
def delete_job(job_id: str, services: Services = Depends(get_services)) -> dict:
    _get_job(services, job_id)
    ok = services.store.delete(job_id)
    return {"deleted": ok}


# ----------------------------------------------------------------------- SSE


@router.get("/{job_id}/events")
async def stream_events(
    job_id: str, request: Request, services: Services = Depends(get_services)
) -> StreamingResponse:
    """Server-sent events for live progress.

    Replays the backlog first so a client that connects mid-run (or reconnects)
    sees everything that already happened rather than starting blind.
    """
    _get_job(services, job_id)
    queue = EVENT_BUS.subscribe(job_id)

    async def generator():
        try:
            for event in EVENT_BUS.replay(job_id):
                yield f"data: {json.dumps(event.to_dict())}\n\n"
            # Tell the client the backlog is done, so it can stop showing a
            # "connecting" state without waiting for the next real event.
            yield f"data: {json.dumps({'kind': 'synced', 'job_id': job_id})}\n\n"

            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Comment frame: keeps proxies and the browser from closing
                    # an idle connection.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event.to_dict())}\n\n"
                if event.kind in ("done",) or (
                    event.kind == "state"
                    and event.data.get("state") in ("failed", "cancelled", "complete")
                ):
                    # Do not close: the user may run a further stage on this job.
                    pass
        except asyncio.CancelledError:
            raise
        finally:
            EVENT_BUS.unsubscribe(job_id, queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
