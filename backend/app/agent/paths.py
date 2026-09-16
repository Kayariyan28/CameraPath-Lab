"""Input-path resolution and validation for agent-supplied video paths.

The web API never sees a path: a browser uploads bytes into the job directory.
An agent hands us a string instead, which is a different trust problem — the
string arrives from a model that may itself have read it out of a web page or a
file. So every path is resolved to an absolute real path, checked against an
explicit policy, and reported back in full, so the caller can see exactly which
file was opened rather than which file it thought it named.

Nothing here reads a byte of the file. It answers one question: may this path be
attached to a job at all?
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from app.agent.models import AgentError
from app.config import Settings

#: `:`-separated absolute directories. Unset means "no root restriction", which
#: matches the web API's posture (loopback-only, local-first, the user's own
#: machine). Setting it is for someone running the MCP server against a less
#: trusted client.
ENV_ALLOWED_ROOTS = "CPL_AGENT_ALLOWED_INPUT_ROOTS"

#: Base for relative `video_path` arguments. An MCP client's working directory
#: is usually not the user's, so a relative path is ambiguous unless the base is
#: stated; this makes it configurable and the resolved result is always echoed.
ENV_INPUT_CWD = "CPL_AGENT_INPUT_CWD"


def allowed_input_roots() -> list[Path]:
    """Configured allowlist, resolved. Empty list = unrestricted."""
    raw = os.environ.get(ENV_ALLOWED_ROOTS, "").strip()
    if not raw:
        return []
    roots: list[Path] = []
    for part in raw.split(os.pathsep):
        part = part.strip()
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except OSError:
            continue
    return roots


def input_base_dir() -> Path:
    raw = os.environ.get(ENV_INPUT_CWD, "").strip()
    if raw:
        try:
            return Path(raw).expanduser().resolve()
        except OSError:
            pass
    return Path.cwd()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_input_video(
    raw: str,
    settings: Settings,
    *,
    allow_job_id: str | None = None,
) -> Path:
    """Resolve and validate a caller-supplied video path.

    `allow_job_id` names the one job whose own `source/` file may be re-used as
    an input (a re-run of an existing job). Everything else inside the workspace
    is refused: feeding one job's scratch into another is never intentional, and
    allowing it would turn a job id into a file-read primitive over the whole
    workspace.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise AgentError(
            "invalid_input_path",
            "video_path must be a non-empty string.",
            hint="Pass an absolute path to a video file on this machine.",
        )
    if "\x00" in raw:
        raise AgentError("invalid_input_path", "video_path contains a NUL byte.")

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = input_base_dir() / candidate

    try:
        # strict=True so a missing target — including a dangling symlink — fails
        # here with a readable message rather than deep inside ffprobe.
        path = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AgentError(
            "invalid_input_path",
            f"no such file: {raw}",
            detail=f"resolved to {candidate}: {type(exc).__name__}: {exc}",
            hint=(
                "Relative paths resolve against "
                f"{input_base_dir()} (override with {ENV_INPUT_CWD})."
            ),
        ) from exc

    if not path.is_file():
        raise AgentError(
            "invalid_input_path",
            f"not a regular file: {path}",
            detail="Directories, FIFOs, sockets and device files are not accepted.",
        )

    suffix = path.suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise AgentError(
            "unsupported_media_type",
            f"unsupported file type '{suffix or path.name}'.",
            detail=f"Supported extensions: {', '.join(settings.allowed_extensions)}",
        )

    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AgentError(
            "invalid_input_path", f"could not stat {path}: {exc}"
        ) from exc
    if size == 0:
        raise AgentError("invalid_input_path", f"file is empty: {path}")
    if size > settings.max_upload_bytes:
        raise AgentError(
            "file_too_large",
            f"file is {size / 1024**3:.2f} GiB; the limit is "
            f"{settings.max_upload_bytes / 1024**3:.1f} GiB.",
            hint="Raise CPL_MAX_UPLOAD_BYTES, or trim the clip first.",
        )

    jobs_root = Path(settings.workspace_dir).expanduser().resolve() / "jobs"
    if _is_within(path, jobs_root):
        permitted = None
        if allow_job_id:
            permitted = jobs_root / allow_job_id / "source"
        if permitted is None or not _is_within(path, permitted.resolve()
                                               if permitted.exists() else permitted):
            raise AgentError(
                "invalid_input_path",
                "that path is inside the CameraPath Lab workspace.",
                detail=str(path),
                hint=(
                    "Point at the original video. Job directories hold derived "
                    "data, and re-feeding one job's scratch into another produces "
                    "a result nobody can trace."
                ),
            )

    roots = allowed_input_roots()
    if roots and not any(_is_within(path, root) for root in roots):
        raise AgentError(
            "invalid_input_path",
            "that path is outside the configured input roots.",
            detail=str(path),
            hint=f"{ENV_ALLOWED_ROOTS} = {os.pathsep.join(str(r) for r in roots)}",
        )

    if not os.access(path, os.R_OK):
        raise AgentError(
            "invalid_input_path",
            f"no read permission: {path}",
        )

    return path


def attach_source(src: Path, source_dir: Path, *, mode: str = "link") -> tuple[Path, str]:
    """Put the source video in the job directory. Returns (path, how).

    A hardlink by default. A copy of an 8 GiB source costs minutes and doubles
    the disk for a file that is only ever opened read-only; a hardlink makes the
    job directory self-contained and survives the workspace GC deleting the job,
    because the original inode keeps its other link. When the link cannot be made
    — different filesystem, a filesystem without links, permissions — we copy and
    say so, rather than failing on a detail the caller cannot see.
    """
    source_dir.mkdir(parents=True, exist_ok=True)
    for existing in source_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    target = source_dir / src.name
    if mode == "link":
        try:
            os.link(src, target)
            return target, "link"
        except OSError:
            target.unlink(missing_ok=True)
    try:
        shutil.copy2(src, target)
    except OSError as exc:
        raise AgentError(
            "invalid_input_path",
            f"could not attach {src.name} to the job: {exc}",
            hint="Check free disk space and permissions on the workspace.",
        ) from exc
    return target, "copy"
