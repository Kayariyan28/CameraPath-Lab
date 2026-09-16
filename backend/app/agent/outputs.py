"""Output discovery, absolute paths, and MP4 verification.

Two jobs. The first is naming every file a job produced with an absolute path,
because an agent's next move is usually to hand that path to something else and
a relative path is useless the moment the process's working directory differs.

The second is verification. `motion_proxy_mp4` reports what `ffprobe` reads back
off the file, not what Blender was asked to render — the product's core promise
is that the proxy's timing matches the source (I1), and the only evidence for
that is the encoded file itself.

Containment is enforced the same way `routes_jobs.py` enforces it for the HTTP
API: a plain-name whitelist, then a `resolve()` parent check. Nothing outside
`workspace/jobs/<job_id>/outputs/` is ever read or named.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.agent.models import (
    AgentError,
    ExportText,
    FileFacts,
    OutputFile,
    OutputsReport,
    ShotOutputs,
    VerifiedMp4,
)
from app.core.paths import JobPaths
from app.models.schemas.jobs import Job
from app.trajectory.exporters import (
    TRAJECTORY_INDEX_NAME,
    TRAJECTORY_STEM,
    chan_meta_path,
    shot_stem,
)

#: Output file names this layer will name or read. Kept identical to
#: `routes_jobs._SAFE_NAME` — plain names only, no separators — so the agent
#: surface and the HTTP API agree on exactly which files are addressable. A test
#: asserts the two patterns are the same string.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,120}$")

#: Text exports an agent may pull inline. Binary outputs are handed over as
#: paths instead: a 40 MB MP4 in a tool result is a context-window accident.
TEXT_SUFFIXES = {".json": "application/json", ".csv": "text/csv", ".chan": "text/plain"}

#: Cap on an inline export read, per the resource contract.
MAX_EXPORT_BYTES = 4 * 1024 * 1024

_KINDS = {
    ".mp4": "video",
    ".blend": "blender_scene",
    ".json": "data",
    ".csv": "data",
    ".chan": "camera_track",
    ".log": "log",
}


def _facts(path: Path | None) -> FileFacts:
    if path is None:
        return FileFacts()
    try:
        stat = path.stat()
    except OSError:
        return FileFacts(path=str(path), exists=False)
    return FileFacts(path=str(path), size_bytes=stat.st_size, exists=path.is_file())


def _kind(name: str) -> str:
    return _KINDS.get(Path(name).suffix.lower(), "other")


def verify_mp4(path: Path, job: Job, *, verify: bool = True) -> VerifiedMp4:
    """Facts about the rendered proxy, read back off the file.

    A missing ffprobe is reported as a stated reason rather than an error: the
    file may well be fine, and claiming a verification we did not perform is
    worse than saying we could not perform it.
    """
    base = _facts(path)
    out = VerifiedMp4(path=base.path, size_bytes=base.size_bytes, exists=base.exists)
    if not out.exists:
        out.verify_reason = "no motion proxy has been rendered for this job"
        return out
    if not verify:
        out.verify_reason = "verify_mp4=false — the numbers below were not read from the file"
        return out

    from app.blender.runner import probe_mp4

    try:
        frames, fps, duration, width, height = probe_mp4(path)
    except Exception as exc:  # noqa: BLE001 - verification must not fail the call
        out.verify_reason = f"ffprobe could not read the proxy: {type(exc).__name__}: {exc}"
        return out

    out.frame_count = frames
    out.fps = fps
    out.duration_seconds = duration
    out.width = width
    out.height = height

    if job.video is None:
        out.verify_reason = "no source video facts on the job to compare against"
        return out

    # Frame count is the strict test; duration is allowed the slack of one frame
    # plus container rounding, because a muxer writes duration as a rational in
    # its own time base.
    slack = max(2.0 / fps, 0.05) if fps else 0.05
    out.timing_matches_source = bool(
        frames == job.video.frame_count
        and abs(duration - job.video.duration_seconds) <= slack
    )
    return out


def build_outputs_report(
    job: Job,
    paths: JobPaths,
    *,
    ui_url: str,
    verify: bool = True,
    render_skipped_reason: str | None = None,
) -> OutputsReport:
    outputs_dir = paths.outputs_dir
    multi = len(job.trajectories) > 1

    proxy = outputs_dir / "motion_proxy.mp4"
    report = OutputsReport(
        job_id=job.id,
        job_dir=str(paths.root),
        outputs_dir=str(outputs_dir),
        ui_url=ui_url,
        motion_proxy_mp4=verify_mp4(proxy, job, verify=verify),
        trajectory_json=_facts(outputs_dir / f"{TRAJECTORY_STEM}.json"),
        trajectory_csv=_facts(outputs_dir / f"{TRAJECTORY_STEM}.csv"),
        trajectory_index_json=_facts(outputs_dir / TRAJECTORY_INDEX_NAME),
        analysis_json=_facts(outputs_dir / "analysis.json"),
        camera_scene_blend=_facts(outputs_dir / "camera_scene.blend"),
        trajectory_preview_mp4=_facts(outputs_dir / "trajectory_preview.mp4"),
        render_skipped_reason=render_skipped_reason,
    )

    named: set[str] = set()
    for trajectory in job.trajectories:
        stem = shot_stem(trajectory.shot_id) if multi else TRAJECTORY_STEM
        chan = outputs_dir / f"{stem}.chan"
        shot_mp4 = (
            outputs_dir / f"motion_proxy_shot_{trajectory.shot_id:03d}.mp4" if multi else proxy
        )
        entry = ShotOutputs(
            shot_id=trajectory.shot_id,
            trajectory_json=_facts(outputs_dir / f"{stem}.json"),
            trajectory_csv=_facts(outputs_dir / f"{stem}.csv"),
            chan=_facts(chan),
            chan_meta_json=_facts(chan_meta_path(chan)),
            motion_proxy_mp4=_facts(shot_mp4),
        )
        report.per_shot.append(entry)
        named.update(
            Path(f.path).name
            for f in (
                entry.trajectory_json,
                entry.trajectory_csv,
                entry.chan,
                entry.chan_meta_json,
                entry.motion_proxy_mp4,
            )
            if f.path
        )

    named.update(
        Path(f.path).name
        for f in (
            report.trajectory_json,
            report.trajectory_csv,
            report.trajectory_index_json,
            report.analysis_json,
            report.camera_scene_blend,
            report.trajectory_preview_mp4,
        )
        if f.path
    )
    named.add(proxy.name)

    if outputs_dir.is_dir():
        for candidate in sorted(outputs_dir.iterdir()):
            if not candidate.is_file() or candidate.name.startswith("."):
                continue
            if not SAFE_NAME.match(candidate.name) or candidate.name in named:
                continue
            report.files.append(
                OutputFile(
                    name=candidate.name,
                    path=str(candidate),
                    size_bytes=candidate.stat().st_size,
                    kind=_kind(candidate.name),
                )
            )

    if not report.motion_proxy_mp4.exists and render_skipped_reason is None:
        report.notes.append(
            "No motion proxy MP4 is present. The trajectory exports above are complete "
            "and usable on their own; render one with render_motion_proxy."
        )
    if multi:
        report.notes.append(
            f"{len(job.trajectories)} shots: motion_proxy.mp4 is the concatenation, and "
            "each shot also has its own proxy and its own coordinate system."
        )
    return report


def resolve_output_file(paths: JobPaths, filename: str) -> Path:
    """Absolute path to one output file, or an AgentError.

    The whitelist runs before the filesystem is touched, and the containment
    check runs after `resolve()` so a symlink planted in the outputs directory
    cannot point out of it.
    """
    if not isinstance(filename, str) or not SAFE_NAME.match(filename):
        raise AgentError(
            "invalid_input_path",
            f"invalid output file name: {filename!r}",
            detail="Output names are plain file names — no directories, no separators.",
        )
    outputs_dir = paths.outputs_dir.resolve()
    path = (outputs_dir / filename).resolve()
    if path.parent != outputs_dir or not path.is_file():
        raise AgentError(
            "job_not_ready",
            f"no such output: {filename}",
            hint="Call list_outputs to see what this job actually produced.",
        )
    return path


def read_export(
    job_id: str, paths: JobPaths, filename: str, *, max_bytes: int = MAX_EXPORT_BYTES
) -> ExportText:
    """Read one text export inline. Binary outputs are refused by design."""
    path = resolve_output_file(paths, filename)
    suffix = path.suffix.lower()
    if path.name.endswith(".chan.meta.json"):
        suffix = ".json"
    if suffix not in TEXT_SUFFIXES:
        raise AgentError(
            "unsupported_media_type",
            f"{filename} is a binary output.",
            hint="Use the absolute path from list_outputs instead of reading it inline.",
        )
    size = path.stat().st_size
    cap = min(max_bytes, MAX_EXPORT_BYTES)
    data = path.read_bytes()[:cap]
    return ExportText(
        job_id=job_id,
        filename=filename,
        path=str(path),
        mime_type=TEXT_SUFFIXES[suffix],
        size_bytes=size,
        truncated=size > cap,
        text=data.decode("utf-8", errors="replace"),
    )
