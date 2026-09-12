"""System/environment endpoints — what the app found on this machine."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import Services, get_services
from app.core.environment import detect_environment

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/environment")
def environment(services: Services = Depends(get_services)) -> dict:
    """Full host probe. Free memory and disk are re-measured on each call so the
    UI shows live headroom, not a startup snapshot."""
    env = detect_environment(services.workspace.root)
    services.environment = env
    return env.to_dict()


@router.get("/health")
def health(services: Services = Depends(get_services)) -> dict:
    env = services.environment
    # "ready" means the deterministic pipeline can run. Blender and pycolmap are
    # reported separately because their absence degrades rather than blocks.
    return {
        "status": "ok" if (env.ffmpeg_ok and env.ffprobe_ok) else "degraded",
        "can_ingest": env.ffmpeg_ok and env.ffprobe_ok,
        "can_reconstruct_3d": env.colmap_ok,
        "can_render": env.blender_ok,
        "can_use_vggt": env.torch_ok and env.mps_available,
        "warnings": env.warnings,
    }


@router.get("/capabilities")
def capabilities(services: Services = Depends(get_services)) -> dict:
    """Which pipeline modes are actually available, and why not if not."""
    env = services.environment
    policy = env.resource_policy()
    return {
        "modes": {
            "auto": {"available": True, "reason": ""},
            "physical_3d": {
                "available": True,
                "reason": "" if env.colmap_ok else
                          "pycolmap unavailable — will use the OpenCV solver, which is "
                          "less accurate on long shots",
            },
            "perceptual_match": {"available": True, "reason": ""},
            "fast": {"available": True, "reason": ""},
            "high_accuracy": {"available": True, "reason": ""},
        },
        "vggt": {
            "available": env.torch_ok,
            "mps": env.mps_available,
            "reason": env.mps_detail if not env.mps_available else "",
        },
        "render": {
            "available": env.blender_ok,
            "blender": env.tools.get("blender").version if env.blender_ok else None,
            "reason": "" if env.blender_ok else "Blender not found",
        },
        "resource_policy": vars(policy),
    }
