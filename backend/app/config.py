"""Application settings.

Local-first defaults: the service binds to loopback only, and every path lives
inside the repo unless overridden. Environment variables are prefixed CPL_.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CPL_", extra="ignore")

    # --- server ---
    host: str = Field("127.0.0.1", description="Loopback by default; this is a local tool.")
    port: int = 8848
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:4173", "http://127.0.0.1:4173",
        ]
    )

    # --- paths ---
    workspace_dir: Path = REPO_ROOT / "workspace"
    blender_scripts_dir: Path = REPO_ROOT / "blender"
    benchmarks_dir: Path = REPO_ROOT / "benchmarks"

    # --- uploads ---
    max_upload_bytes: int = Field(8 * 1024**3, description="8 GiB ceiling on a source video.")
    allowed_extensions: tuple[str, ...] = (
        ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".mts", ".m2ts",
    )

    # --- workspace hygiene ---
    job_retention_hours: float = 72.0
    job_workspace_budget_gb: float = 20.0
    keep_min_jobs: int = 3

    # --- execution ---
    max_concurrent_jobs: int = 1
    """One at a time by default: the solve is already parallel internally, and
    two concurrent COLMAP runs on one machine mostly fight over memory."""

    blender_path: str | None = None
    """Override for Blender's location. None = auto-detect."""

    log_level: str = "INFO"

    @property
    def jobs_dir(self) -> Path:
        return self.workspace_dir / "jobs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
