"""FastAPI application entrypoint.

Run with:
    .venv/bin/python -m uvicorn app.main:app --reload --port 8848
(or via scripts/dev.sh, which starts the frontend too)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.api import routes_jobs, routes_system
from app.api.deps import build_services, get_services, shutdown_services
from app.config import get_settings
from app.core.logging import configure_root_logging, get_logger

log = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_root_logging(getattr(logging, settings.log_level.upper(), logging.INFO))
    services = build_services()
    env = services.environment

    log.info("=" * 72)
    log.info("CameraPath Lab backend starting")
    log.info("  host          : %s (%s)", env.chip, env.platform)
    log.info(
        "  memory        : %.0f GB total, %.1f GB available",
        env.total_memory_gb, env.available_memory_gb,
    )
    log.info(
        "  cores         : %d (%d performance, %d efficiency)",
        env.cpu_cores_total, env.cpu_cores_performance, env.cpu_cores_efficiency,
    )
    log.info("  ffmpeg        : %s", "yes" if env.ffmpeg_ok else "MISSING")
    log.info(
        "  blender       : %s",
        env.tools["blender"].version if env.blender_ok else "MISSING (render disabled)",
    )
    log.info(
        "  pycolmap      : %s",
        env.packages["pycolmap"].version if env.colmap_ok else "MISSING (OpenCV fallback)",
    )
    log.info("  torch / MPS   : %s", env.mps_detail)
    policy = env.resource_policy()
    log.info("  resource tier : %s", policy.reason)
    log.info(
        "  analysis res  : %d px long edge (geometry %d px)",
        policy.analysis_long_edge, policy.geometry_long_edge,
    )
    for warning in env.warnings:
        log.warning("  ! %s", warning)
    log.info("  workspace     : %s", services.workspace.root)
    log.info("  listening     : http://%s:%d", settings.host, settings.port)
    log.info("=" * 72)

    try:
        yield
    finally:
        shutdown_services()
        log.info("backend stopped")


app = FastAPI(
    title="CameraPath Lab",
    description=(
        "Recovers camera motion from a reference video and re-emits it as a "
        "scene-neutral motion proxy. Local-only; no cloud APIs."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_system.router)
app.include_router(routes_jobs.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Invariant I12 at the transport layer: a bug in one request returns a
    readable error instead of killing the worker."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"{type(exc).__name__}: {exc}",
            "path": request.url.path,
        },
    )


@app.get("/api")
def api_root() -> dict:
    return {
        "name": "CameraPath Lab",
        "version": "0.1.0",
        "phase": "1 — ingestion, shot detection, dense motion analysis",
        "endpoints": [
            "GET  /api/system/environment",
            "GET  /api/system/health",
            "GET  /api/system/capabilities",
            "POST /api/jobs",
            "GET  /api/jobs",
            "POST /api/jobs/{id}/video",
            "POST /api/jobs/{id}/analyze",
            "POST /api/jobs/{id}/cancel",
            "GET  /api/jobs/{id}",
            "GET  /api/jobs/{id}/analysis",
            "GET  /api/jobs/{id}/motion/{shot_id}",
            "GET  /api/jobs/{id}/frame/{frame_index}",
            "GET  /api/jobs/{id}/source",
            "GET  /api/jobs/{id}/log",
            "GET  /api/jobs/{id}/events  (SSE)",
        ],
    }
