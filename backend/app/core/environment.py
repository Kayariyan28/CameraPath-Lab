"""Machine + toolchain probe, and the adaptive resource policy derived from it.

Nothing in CameraPath Lab hard-codes a RAM figure or a window size. This module
measures the host once at startup (and re-measures free memory on demand), then
every expensive stage asks it how big a bite it may take. The same code path
produces generous settings on an M4 Max/48 GB and conservative ones on an
8 GB M1 Air.
"""

from __future__ import annotations

import functools
import json
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Blender does not put itself on PATH on macOS; these are the standard spots.
BLENDER_CANDIDATES = (
    "/Applications/Blender.app/Contents/MacOS/Blender",
    str(Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"),
    "/opt/homebrew/bin/blender",
    "/usr/local/bin/blender",
)

BYTES_PER_GB = 1024**3


@dataclass
class ToolInfo:
    name: str
    available: bool
    path: str | None = None
    version: str | None = None
    detail: str = ""


@dataclass
class PythonPackageInfo:
    name: str
    available: bool
    version: str | None = None
    detail: str = ""


@dataclass
class ResourcePolicy:
    """Adaptive limits. All derived, none hard-coded."""

    analysis_long_edge: int
    """Long-edge cap for frames fed to optical flow / feature tracking."""

    geometry_long_edge: int
    """Long-edge cap for frames fed to SfM (higher — features need detail)."""

    decode_chunk_frames: int
    """How many frames to hold in memory per decode batch."""

    max_tracked_features: int
    """Shi-Tomasi corner budget per frame."""

    colmap_max_image_size: int
    colmap_max_num_features: int

    vggt_window_frames: int
    vggt_window_stride: int

    worker_threads: int
    blender_threads: int

    reason: str = ""


@dataclass
class Environment:
    platform: str
    machine: str
    is_apple_silicon: bool
    chip: str
    macos_version: str

    total_memory_gb: float
    available_memory_gb: float
    cpu_cores_total: int
    cpu_cores_performance: int
    cpu_cores_efficiency: int

    disk_free_gb: float

    mps_available: bool
    mps_detail: str

    tools: dict[str, ToolInfo] = field(default_factory=dict)
    packages: dict[str, PythonPackageInfo] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)

    @property
    def ffmpeg_ok(self) -> bool:
        return self.tools.get("ffmpeg", ToolInfo("ffmpeg", False)).available

    @property
    def ffprobe_ok(self) -> bool:
        return self.tools.get("ffprobe", ToolInfo("ffprobe", False)).available

    @property
    def blender_ok(self) -> bool:
        return self.tools.get("blender", ToolInfo("blender", False)).available

    @property
    def blender_path(self) -> str | None:
        t = self.tools.get("blender")
        return t.path if t and t.available else None

    @property
    def colmap_ok(self) -> bool:
        return self.packages.get("pycolmap", PythonPackageInfo("pycolmap", False)).available

    @property
    def torch_ok(self) -> bool:
        return self.packages.get("torch", PythonPackageInfo("torch", False)).available

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "is_apple_silicon": self.is_apple_silicon,
            "chip": self.chip,
            "macos_version": self.macos_version,
            "total_memory_gb": round(self.total_memory_gb, 1),
            "available_memory_gb": round(self.available_memory_gb, 1),
            "cpu_cores_total": self.cpu_cores_total,
            "cpu_cores_performance": self.cpu_cores_performance,
            "cpu_cores_efficiency": self.cpu_cores_efficiency,
            "disk_free_gb": round(self.disk_free_gb, 1),
            "mps_available": self.mps_available,
            "mps_detail": self.mps_detail,
            "tools": {k: vars(v) for k, v in self.tools.items()},
            "packages": {k: vars(v) for k, v in self.packages.items()},
            "warnings": self.warnings,
            "resource_policy": vars(self.resource_policy()),
        }

    # ------------------------------------------------------------------ policy

    def resource_policy(self, *, high_accuracy: bool = False) -> ResourcePolicy:
        """Derive concrete limits from measured memory and core count.

        We budget against *available* memory rather than total, because the
        machine may already be under pressure, but we never drop below a floor
        that would make results meaningless.
        """
        avail = max(self.available_memory_gb, 1.0)
        # Budget against available memory (leaving headroom for ffmpeg + Blender),
        # but floored against a share of total. macOS counts compressed and wired
        # pages as used, so `available` badly understates what a 48 GB machine can
        # actually commit; without the floor an M4 Max gets driven like an 8 GB Air.
        budget = max(avail * 0.6, self.total_memory_gb * 0.25, 1.0)

        if budget >= 24:
            tier, analysis, geometry, chunk, feats = "very_large", 1440, 2560, 256, 3000
        elif budget >= 12:
            tier, analysis, geometry, chunk, feats = "large", 1080, 2048, 192, 2400
        elif budget >= 6:
            tier, analysis, geometry, chunk, feats = "medium", 960, 1600, 128, 1800
        elif budget >= 3:
            tier, analysis, geometry, chunk, feats = "small", 720, 1280, 64, 1200
        else:
            tier, analysis, geometry, chunk, feats = "minimal", 540, 960, 32, 800

        if high_accuracy:
            analysis = int(analysis * 1.25)
            geometry = int(geometry * 1.25)
            feats = int(feats * 1.5)

        # VGGT windows scale with memory; attention cost is ~quadratic in frames.
        if budget >= 24:
            vggt_win, vggt_stride = 32, 24
        elif budget >= 12:
            vggt_win, vggt_stride = 24, 18
        elif budget >= 6:
            vggt_win, vggt_stride = 16, 12
        else:
            vggt_win, vggt_stride = 8, 6

        perf = self.cpu_cores_performance or max(1, self.cpu_cores_total // 2)
        workers = max(1, min(perf, 10))

        return ResourcePolicy(
            analysis_long_edge=analysis,
            geometry_long_edge=geometry,
            decode_chunk_frames=chunk,
            max_tracked_features=feats,
            colmap_max_image_size=geometry,
            colmap_max_num_features=feats * 4,
            vggt_window_frames=vggt_win,
            vggt_window_stride=vggt_stride,
            worker_threads=workers,
            blender_threads=max(1, self.cpu_cores_total - 2),
            reason=(
                f"tier={tier} from {avail:.1f} GB available "
                f"({self.total_memory_gb:.0f} GB total), {perf} performance cores"
                + (", high-accuracy uplift applied" if high_accuracy else "")
            ),
        )


# --------------------------------------------------------------------- probes


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 1, str(exc)


def _sysctl(key: str) -> str | None:
    rc, out = _run(["sysctl", "-n", key], timeout=3)
    return out.strip() if rc == 0 and out.strip() else None


def _available_memory_gb(total_gb: float) -> float:
    """Free + inactive + speculative pages, via vm_stat. Inactive pages are
    reclaimable, so counting only 'free' would wildly understate what we can use
    on a machine with a warm page cache."""
    rc, out = _run(["vm_stat"], timeout=3)
    if rc != 0:
        return total_gb * 0.5
    page_size = 16384 if platform.machine() == "arm64" else 4096
    counts: dict[str, int] = {}
    for line in out.splitlines():
        if "page size of" in line:
            try:
                page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
            except (ValueError, IndexError):
                pass
            continue
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        v = v.strip().rstrip(".")
        if v.isdigit():
            counts[k.strip().lower()] = int(v)
    reclaimable = (
        counts.get("pages free", 0)
        + counts.get("pages inactive", 0)
        + counts.get("pages speculative", 0)
        + counts.get("pages purgeable", 0)
    )
    if not reclaimable:
        return total_gb * 0.5
    return (reclaimable * page_size) / BYTES_PER_GB


def _probe_tool(name: str, version_args: list[str], explicit: str | None = None) -> ToolInfo:
    path = explicit or shutil.which(name)
    if not path:
        return ToolInfo(name, False, detail=f"{name} not found on PATH")
    rc, out = _run([path, *version_args], timeout=20)
    version = out.strip().splitlines()[0] if out.strip() else None
    if rc != 0 and not version:
        return ToolInfo(name, False, path=path, detail=f"{name} found but failed to run")
    return ToolInfo(name, True, path=path, version=version)


def _probe_blender() -> ToolInfo:
    path = shutil.which("blender")
    if not path:
        for cand in BLENDER_CANDIDATES:
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                path = cand
                break
    if not path:
        return ToolInfo(
            "blender",
            False,
            detail=(
                "Blender not found. Install from blender.org or "
                "`brew install --cask blender`, then re-run scripts/doctor.py."
            ),
        )
    rc, out = _run([path, "--version"], timeout=45)
    version = None
    for line in out.splitlines():
        if line.strip().lower().startswith("blender"):
            version = line.strip()
            break
    if rc != 0 and not version:
        return ToolInfo("blender", False, path=path, detail="Blender found but --version failed")
    return ToolInfo("blender", True, path=path, version=version or "unknown")


def _probe_package(name: str, import_name: str | None = None) -> PythonPackageInfo:
    import importlib

    mod_name = import_name or name
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001 - any import failure is just "unavailable"
        return PythonPackageInfo(name, False, detail=f"{type(exc).__name__}: {exc}")
    return PythonPackageInfo(name, True, version=str(getattr(mod, "__version__", "unknown")))


def _probe_mps() -> tuple[bool, str]:
    """MPS availability. Deliberately tolerant: torch is optional and a broken
    torch install must not affect the deterministic pipeline."""
    try:
        import torch  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False, "torch not installed (optional — only the VGGT backend needs it)"
    try:
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                return False, "this torch build has no MPS support"
            return False, "MPS built but not available on this machine"
        # Prove it actually executes; is_available() can be optimistic.
        t = torch.ones(8, 8, device="mps")
        _ = (t @ t).sum().item()
        return True, f"MPS functional (torch {torch.__version__})"
    except Exception as exc:  # noqa: BLE001
        return False, f"MPS present but a probe op failed: {type(exc).__name__}: {exc}"


@functools.lru_cache(maxsize=1)
def _static_probe() -> dict:
    """The expensive parts — subprocess version checks — cached for process life."""
    tools = {
        "ffmpeg": _probe_tool("ffmpeg", ["-version"]),
        "ffprobe": _probe_tool("ffprobe", ["-version"]),
        "blender": _probe_blender(),
        "git": _probe_tool("git", ["--version"]),
    }
    packages = {
        "numpy": _probe_package("numpy"),
        "scipy": _probe_package("scipy"),
        "opencv": _probe_package("opencv", "cv2"),
        "pycolmap": _probe_package("pycolmap"),
        "torch": _probe_package("torch"),
        "fastapi": _probe_package("fastapi"),
    }
    mps_available, mps_detail = _probe_mps()
    return {
        "tools": tools,
        "packages": packages,
        "mps_available": mps_available,
        "mps_detail": mps_detail,
    }


def detect_environment(workspace: Path | None = None) -> Environment:
    """Full host probe. Free memory and disk are re-measured on every call;
    tool/package probes are cached."""
    static = _static_probe()

    machine = platform.machine()
    chip = _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model") or "unknown"
    total_bytes = _sysctl("hw.memsize")
    total_gb = int(total_bytes) / BYTES_PER_GB if total_bytes else 8.0

    try:
        cores_total = int(_sysctl("hw.logicalcpu") or os.cpu_count() or 4)
    except ValueError:
        cores_total = os.cpu_count() or 4
    try:
        perf = int(_sysctl("hw.perflevel0.logicalcpu") or 0)
        eff = int(_sysctl("hw.perflevel1.logicalcpu") or 0)
    except ValueError:
        perf, eff = 0, 0

    target = workspace or Path.cwd()
    try:
        usage = shutil.disk_usage(target)
        disk_free_gb = usage.free / BYTES_PER_GB
    except OSError:
        disk_free_gb = 0.0

    env = Environment(
        platform=f"{platform.system()} {platform.release()}",
        machine=machine,
        is_apple_silicon=(platform.system() == "Darwin" and machine == "arm64"),
        chip=chip,
        macos_version=platform.mac_ver()[0] or "unknown",
        total_memory_gb=total_gb,
        available_memory_gb=_available_memory_gb(total_gb),
        cpu_cores_total=cores_total,
        cpu_cores_performance=perf,
        cpu_cores_efficiency=eff,
        disk_free_gb=disk_free_gb,
        mps_available=static["mps_available"],
        mps_detail=static["mps_detail"],
        tools=dict(static["tools"]),
        packages=dict(static["packages"]),
    )

    # Warnings the UI shows. Only genuine blockers are phrased as blockers.
    if not env.ffprobe_ok or not env.ffmpeg_ok:
        env.warnings.append(
            "FFmpeg/FFprobe missing — video ingestion cannot run. "
            "Install with `brew install ffmpeg`."
        )
    if not env.blender_ok:
        env.warnings.append(
            "Blender not found — analysis and trajectory export still work, but "
            "MP4 proxy rendering is unavailable."
        )
    if not env.colmap_ok:
        env.warnings.append(
            "pycolmap unavailable — falling back to the OpenCV geometric solver, "
            "which is less accurate on long shots."
        )
    if not env.torch_ok:
        env.warnings.append(
            "PyTorch not installed — the optional VGGT backend is disabled. "
            "The deterministic pipeline is unaffected."
        )
    if env.disk_free_gb < 10:
        env.warnings.append(
            f"Only {env.disk_free_gb:.1f} GB free. Renders and job scratch need "
            "headroom; consider freeing space."
        )
    if not env.is_apple_silicon:
        env.warnings.append(
            "Not running on Apple Silicon — the app works but is untuned for this host."
        )
    return env


if __name__ == "__main__":  # `python -m app.core.environment`
    print(json.dumps(detect_environment().to_dict(), indent=2))
