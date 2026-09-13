"""Headless Blender driver.

Blender runs as a subprocess, never in-process: its interpreter cannot import the
backend, and a crash inside it must degrade one job rather than take down the API
(invariant I12). Every failure mode — Blender missing, a Python error inside the
scene script, a timeout, an MP4 that came out wrong — becomes a `RenderResult`
with `success=False` and a readable reason.

The MP4 is re-verified with ffprobe after every render. The render script prints
its own frame count and fps, but that is Blender reporting what it was asked to
do, not what the encoder wrote. A proxy whose duration drifted from the source
would silently break the product's core promise (I1), so the numbers are checked
against the file itself and a mismatch is a failure, not a warning.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from app.core.environment import BLENDER_CANDIDATES
from app.core.logging import get_logger

log = get_logger("blender.runner")

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "blender"

#: Frame-rate agreement required between request and file. Container time bases
#: round rational rates (30000/1001 is stored as 29.97002997...), so exact float
#: equality is wrong, but anything beyond this is a genuinely different rate.
FPS_TOLERANCE = 1e-3

# Blender 5.x logs "Video append frame N"; 3.x/4.x logged "Append frame N".
_APPEND_RE = re.compile(r"[Aa]ppend frame (\d+)")
_FRA_RE = re.compile(r"Fra:(\d+)")
_RESULT_RE = re.compile(r"^CPL_RESULT (.*)$", re.MULTILINE)

ProgressCallback = Callable[[float, str], None]


@dataclass
class RenderResult:
    success: bool
    output_path: Path | None = None
    frame_count: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    blend_path: Path | None = None
    elapsed_seconds: float = 0.0
    error: str | None = None
    stdout_tail: list[str] = field(default_factory=list)
    reported: dict[str, str] = field(default_factory=dict)
    """The script's own CPL_RESULT fields, before verification."""


def find_blender(explicit: str | None = None) -> str | None:
    """Explicit path, then PATH, then the standard macOS/Homebrew locations."""
    candidates = [explicit] if explicit else []
    on_path = shutil.which("blender")
    if on_path:
        candidates.append(on_path)
    candidates.extend(BLENDER_CANDIDATES)
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def probe_mp4(path: Path) -> tuple[int, float, float, int, int]:
    """(frame_count, fps, duration, width, height) as the FILE reports them.

    Frames are counted by reading packets rather than trusting the container's
    nb_frames, which some muxers write as an estimate.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found; cannot verify the render")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,r_frame_rate,avg_frame_rate,width,height:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:300]}")
    data = json.loads(proc.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    rate = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "0/1"
    fps = float(Fraction(rate)) if rate not in ("0/0", "") else 0.0
    frames = int(stream.get("nb_read_packets") or 0)
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    return frames, fps, duration, int(stream.get("width") or 0), int(stream.get("height") or 0)


class BlenderRunner:
    def __init__(self, blender_path: str | None = None, scripts_dir: Path | None = None):
        self.blender_path = find_blender(blender_path)
        self.scripts_dir = Path(scripts_dir) if scripts_dir else SCRIPTS_DIR

    @property
    def available(self) -> bool:
        return self.blender_path is not None

    # ---------------------------------------------------------------- core

    def _run(
        self,
        script: str,
        script_args: list[str],
        *,
        expected_frames: int | None,
        progress: ProgressCallback | None,
        timeout: float,
    ) -> tuple[int | None, list[str], dict[str, str], str | None]:
        """Run a Blender script. Returns (returncode, tail, CPL_RESULT, error)."""
        if not self.blender_path:
            return None, [], {}, (
                "Blender was not found. Install it from blender.org or with "
                "`brew install --cask blender`."
            )
        script_path = self.scripts_dir / script
        if not script_path.is_file():
            return None, [], {}, f"render script missing: {script_path}"

        cmd = [
            self.blender_path, "--background", "--factory-startup",
            "--python-exit-code", "3",
            "--python", str(script_path), "--", *script_args,
        ]
        log.info("blender: %s", " ".join(cmd))
        tail: deque[str] = deque(maxlen=60)
        reported: dict[str, str] = {}
        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except OSError as exc:
            return None, [], {}, f"could not start Blender: {exc}"

        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                tail.append(line)
                match = _RESULT_RE.match(line)
                if match:
                    for token in match.group(1).split(" "):
                        key, _, value = token.partition("=")
                        if key:
                            reported[key] = value
                if progress and expected_frames:
                    hit = _APPEND_RE.search(line) or _FRA_RE.search(line)
                    if hit:
                        done = int(hit.group(1))
                        progress(min(1.0, done / expected_frames), f"frame {done}/{expected_frames}")
                if time.monotonic() - started > timeout:
                    proc.kill()
                    proc.wait(timeout=10)
                    return proc.returncode, list(tail), reported, (
                        f"Blender exceeded the {timeout:.0f}s render timeout"
                    )
            proc.wait(timeout=max(1.0, timeout - (time.monotonic() - started)))
        except subprocess.TimeoutExpired:
            proc.kill()
            return None, list(tail), reported, f"Blender exceeded the {timeout:.0f}s render timeout"
        finally:
            if proc.poll() is None:
                proc.kill()

        if proc.returncode != 0:
            errors = [l for l in tail if "Error" in l or "Traceback" in l or l.startswith("  File")]
            detail = " | ".join((errors or list(tail))[-6:])
            return proc.returncode, list(tail), reported, (
                f"Blender exited with code {proc.returncode}: {detail[:600]}"
            )
        return proc.returncode, list(tail), reported, None

    def _verify(
        self,
        output: Path,
        *,
        expected_frames: int,
        expected_fps: float | None,
        result: RenderResult,
    ) -> RenderResult:
        if not output.is_file() or output.stat().st_size == 0:
            result.success = False
            result.error = f"Blender reported success but produced no file at {output}"
            return result
        try:
            frames, fps, duration, width, height = probe_mp4(output)
        except Exception as exc:  # noqa: BLE001
            result.success = False
            result.error = f"could not verify the rendered MP4: {exc}"
            return result
        result.frame_count, result.fps = frames, fps
        result.duration_seconds, result.width, result.height = duration, width, height

        problems = []
        if frames != expected_frames:
            problems.append(f"file has {frames} frames, expected exactly {expected_frames}")
        if expected_fps and abs(fps - expected_fps) > FPS_TOLERANCE * max(1.0, expected_fps):
            problems.append(f"file plays at {fps:.6f} fps, expected {expected_fps:.6f}")
        if problems:
            # Loud on purpose: a duration-shifted proxy breaks invariant I1.
            result.success = False
            result.error = "render timing verification FAILED: " + "; ".join(problems)
            log.error(result.error)
        else:
            result.success = True
        return result

    # --------------------------------------------------------------- public

    def render_motion_proxy(
        self,
        trajectory_path: Path,
        output_path: Path,
        *,
        style: str = "motion_cage",
        width: int = 1920,
        height: int = 1080,
        fps: float | None = None,
        samples: int = 16,
        blend_path: Path | None = None,
        progress: ProgressCallback | None = None,
        timeout: float = 3600.0,
    ) -> RenderResult:
        """Render the Seedance motion proxy. Never raises."""
        started = time.monotonic()
        trajectory_path, output_path = Path(trajectory_path), Path(output_path)
        try:
            doc = json.loads(trajectory_path.read_text())
            frames = doc.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError("trajectory has no frames")
            doc_fps = float(doc.get("fps")) if doc.get("fps") else None
        except Exception as exc:  # noqa: BLE001
            return RenderResult(False, error=f"unreadable trajectory {trajectory_path.name}: {exc}")

        expected_frames = len(frames)
        expected_fps = fps or doc_fps
        args = ["--trajectory", str(trajectory_path), "--out", str(output_path),
                "--style", style, "--width", str(width), "--height", str(height),
                "--samples", str(samples)]
        if fps:
            args += ["--fps", repr(float(fps))]
        if blend_path:
            args += ["--blend", str(blend_path)]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        rc, tail, reported, error = self._run(
            "render_motion_proxy.py", args, expected_frames=expected_frames,
            progress=progress, timeout=timeout,
        )
        result = RenderResult(
            False, output_path=output_path, stdout_tail=tail[-25:], reported=reported,
            blend_path=Path(blend_path) if blend_path and Path(blend_path).is_file() else None,
        )
        if error:
            result.error = error
        else:
            result = self._verify(output_path, expected_frames=expected_frames,
                                  expected_fps=expected_fps, result=result)
        result.elapsed_seconds = time.monotonic() - started
        return result

    def render_trajectory_preview(
        self,
        trajectory_path: Path,
        output_path: Path,
        *,
        width: int = 1280,
        height: int = 720,
        samples: int = 8,
        progress: ProgressCallback | None = None,
        timeout: float = 3600.0,
    ) -> RenderResult:
        """Third-person debug render. Optional: absence of the script is a
        failed result, not an error."""
        started = time.monotonic()
        try:
            doc = json.loads(Path(trajectory_path).read_text())
            expected_frames = len(doc.get("frames") or [])
            expected_fps = float(doc["fps"]) if doc.get("fps") else None
        except Exception as exc:  # noqa: BLE001
            return RenderResult(False, error=f"unreadable trajectory: {exc}")
        rc, tail, reported, error = self._run(
            "render_trajectory_preview.py",
            ["--trajectory", str(trajectory_path), "--out", str(output_path),
             "--width", str(width), "--height", str(height), "--samples", str(samples)],
            expected_frames=expected_frames, progress=progress, timeout=timeout,
        )
        result = RenderResult(False, output_path=Path(output_path), stdout_tail=tail[-25:], reported=reported)
        if error:
            result.error = error
        else:
            result = self._verify(Path(output_path), expected_frames=expected_frames,
                                  expected_fps=expected_fps, result=result)
        result.elapsed_seconds = time.monotonic() - started
        return result
