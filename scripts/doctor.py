#!/usr/bin/env python3
"""Report what CameraPath Lab can and cannot do on this machine.

Run directly (`.venv/bin/python scripts/doctor.py`) or via the bootstrap script.
Exits non-zero only when something *required* is missing — a missing Blender or
PyTorch degrades the app rather than blocking it, and is reported as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BOLD, RED, GRN, YEL, DIM, RST = "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"


def main() -> int:
    try:
        from app.core.environment import detect_environment
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED}✗{RST} backend package not importable: {exc}")
        print(f"    Run ./scripts/bootstrap_macos.sh")
        return 1

    env = detect_environment(ROOT / "workspace")
    print(f"\n{BOLD}CameraPath Lab — environment{RST}")
    print(f"  host    : {env.chip} · {env.platform}")
    print(f"  memory  : {env.total_memory_gb:.0f} GB total, "
          f"{env.available_memory_gb:.1f} GB available")
    print(f"  cores   : {env.cpu_cores_total} "
          f"({env.cpu_cores_performance}P / {env.cpu_cores_efficiency}E)")
    print(f"  disk    : {env.disk_free_gb:.1f} GB free")

    print(f"\n{BOLD}Tools{RST}")
    required = {"ffmpeg", "ffprobe"}
    failures = 0
    for name, tool in env.tools.items():
        if tool.available:
            print(f"  {GRN}✓{RST} {name:9s} {(tool.version or '')[:56]}")
        elif name in required:
            print(f"  {RED}✗{RST} {name:9s} {tool.detail}")
            failures += 1
        else:
            print(f"  {YEL}!{RST} {name:9s} {tool.detail}")

    print(f"\n{BOLD}Python packages{RST}")
    required_pkgs = {"numpy", "scipy", "opencv", "fastapi"}
    for name, pkg in env.packages.items():
        if pkg.available:
            print(f"  {GRN}✓{RST} {name:9s} {pkg.version}")
        elif name in required_pkgs:
            print(f"  {RED}✗{RST} {name:9s} {pkg.detail[:60]}")
            failures += 1
        else:
            print(f"  {YEL}!{RST} {name:9s} {pkg.detail[:60]}")

    print(f"\n{BOLD}Acceleration{RST}")
    mark = f"{GRN}✓{RST}" if env.mps_available else f"{YEL}!{RST}"
    print(f"  {mark} MPS     {env.mps_detail}")

    policy = env.resource_policy()
    print(f"\n{BOLD}Derived resource policy{RST}")
    print(f"  {DIM}{policy.reason}{RST}")
    print(f"  analysis resolution : {policy.analysis_long_edge} px long edge")
    print(f"  geometry resolution : {policy.geometry_long_edge} px long edge")
    print(f"  decode chunk        : {policy.decode_chunk_frames} frames")
    print(f"  tracked features    : {policy.max_tracked_features} per frame")
    print(f"  VGGT window         : {policy.vggt_window_frames} frames "
          f"(stride {policy.vggt_window_stride})")
    print(f"  worker threads      : {policy.worker_threads}")

    print(f"\n{BOLD}Capabilities{RST}")
    caps = [
        ("ingest video", env.ffmpeg_ok and env.ffprobe_ok, "requires ffmpeg"),
        ("shot detection + motion analysis", env.packages["opencv"].available, "requires opencv"),
        ("geometric 3D reconstruction", env.colmap_ok, "pycolmap missing — OpenCV fallback used"),
        ("Blender MP4 proxy render", env.blender_ok, "Blender missing"),
        ("VGGT learned geometry", env.torch_ok, "PyTorch missing (optional)"),
    ]
    for label, available, why in caps:
        print(f"  {GRN + '✓' + RST if available else YEL + '!' + RST} {label}"
              + ("" if available else f"  {DIM}— {why}{RST}"))

    if env.warnings:
        print(f"\n{BOLD}Notes{RST}")
        for w in env.warnings:
            print(f"  {YEL}!{RST} {w}")

    if failures:
        print(f"\n{RED}{failures} required component(s) missing.{RST} "
              "Run ./scripts/bootstrap_macos.sh\n")
        return 1
    print(f"\n{GRN}Ready.{RST}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
