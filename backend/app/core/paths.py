"""Per-job workspace management.

Layout for one job:

    workspace/jobs/<job_id>/
        source/         the uploaded video, original bytes, untouched
        frames/         analysis-resolution frames (only when materialised)
        geometry/       COLMAP database + sparse model
        cache/          stage artefacts + fingerprints
        outputs/        motion_proxy.mp4, trajectory.json, ...
        job.json        serialised Job record
        job.log         plain-text stage log

Design rules:
  * Artefacts are written atomically (temp file + os.replace) so a crash mid-write
    never leaves a half-parsed JSON that poisons the cache.
  * `job_id` is validated before it touches the filesystem — it arrives from the
    URL path, so it is untrusted input and must not be able to escape the root.
  * GC only ever deletes directories it can prove are job directories.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

JOB_ID_RE = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")

SUBDIRS = ("source", "frames", "geometry", "cache", "outputs")


class InvalidJobId(ValueError):
    pass


def new_job_id() -> str:
    return str(uuid.uuid4())


def validate_job_id(job_id: str) -> str:
    """Reject anything that is not a canonical UUID4 string.

    This is the only guard between a URL path segment and `Path.joinpath`, so it
    is strict by design: a whitelist of hex-and-dashes, not a blacklist of '..'.
    """
    if not isinstance(job_id, str) or not JOB_ID_RE.match(job_id):
        raise InvalidJobId(f"malformed job id: {job_id!r}")
    return job_id


@dataclass(frozen=True)
class JobPaths:
    root: Path

    @property
    def source_dir(self) -> Path:
        return self.root / "source"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def geometry_dir(self) -> Path:
        return self.root / "geometry"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def job_file(self) -> Path:
        return self.root / "job.json"

    @property
    def log_file(self) -> Path:
        return self.root / "job.log"

    def shot_dir(self, shot_id: int) -> Path:
        d = self.cache_dir / f"shot_{shot_id:03d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure(self) -> JobPaths:
        for sub in SUBDIRS:
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        return self

    def source_video(self) -> Path | None:
        if not self.source_dir.is_dir():
            return None
        files = sorted(p for p in self.source_dir.iterdir() if p.is_file() and not p.name.startswith("."))
        return files[0] if files else None

    def size_bytes(self) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(self.root):
            for fn in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, fn))
                except OSError:
                    pass
        return total


class Workspace:
    """Owns `workspace/jobs` and hands out validated JobPaths."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.jobs_root = self.root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job(self, job_id: str) -> JobPaths:
        validate_job_id(job_id)
        path = (self.jobs_root / job_id).resolve()
        # Belt and braces: even with a validated id, assert containment.
        if not str(path).startswith(str(self.jobs_root) + os.sep):
            raise InvalidJobId(f"job path escapes workspace: {job_id!r}")
        return JobPaths(path)

    def create(self, job_id: str | None = None) -> tuple[str, JobPaths]:
        job_id = job_id or new_job_id()
        paths = self.job(job_id).ensure()
        return job_id, paths

    def exists(self, job_id: str) -> bool:
        try:
            return self.job(job_id).root.is_dir()
        except InvalidJobId:
            return False

    def list_job_ids(self) -> list[str]:
        if not self.jobs_root.is_dir():
            return []
        out = []
        for p in self.jobs_root.iterdir():
            if p.is_dir() and JOB_ID_RE.match(p.name):
                out.append(p.name)
        out.sort(key=lambda j: (self.jobs_root / j).stat().st_mtime, reverse=True)
        return out

    def delete(self, job_id: str) -> bool:
        paths = self.job(job_id)
        if not paths.root.is_dir():
            return False
        shutil.rmtree(paths.root, ignore_errors=True)
        return True

    def gc(self, *, max_age_hours: float = 72.0, max_total_gb: float = 20.0,
           keep_min: int = 3) -> list[str]:
        """Remove old job directories.

        Conservative on purpose: only deletes paths whose basename is a valid
        job UUID and which sit directly under `jobs_root`, always keeps the
        `keep_min` most recent, and never touches a job that is still being
        written to (mtime within the last 5 minutes).
        """
        removed: list[str] = []
        now = time.time()
        ids = self.list_job_ids()  # newest first

        for job_id in ids[keep_min:]:
            root = self.jobs_root / job_id
            try:
                age_h = (now - root.stat().st_mtime) / 3600.0
            except OSError:
                continue
            if age_h < (5 / 60):
                continue  # actively in use
            if age_h > max_age_hours:
                shutil.rmtree(root, ignore_errors=True)
                removed.append(job_id)

        # Then trim by total size, oldest first, until under budget.
        budget = max_total_gb * (1024**3)
        remaining = [j for j in self.list_job_ids()]
        sizes = {j: self.job(j).size_bytes() for j in remaining}
        total = sum(sizes.values())
        for job_id in reversed(remaining):  # oldest first
            if total <= budget or len(remaining) <= keep_min:
                break
            root = self.jobs_root / job_id
            try:
                if (now - root.stat().st_mtime) / 3600.0 < (5 / 60):
                    continue
            except OSError:
                continue
            shutil.rmtree(root, ignore_errors=True)
            total -= sizes.get(job_id, 0)
            remaining.remove(job_id)
            removed.append(job_id)
        return removed


# ------------------------------------------------------------- atomic writes


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: object, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=str))


def read_json(path: Path) -> object | None:
    """Tolerant read — a corrupt cache entry is a cache miss, not a crash."""
    try:
        with open(path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
