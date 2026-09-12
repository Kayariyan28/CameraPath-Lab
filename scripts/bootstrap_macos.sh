#!/usr/bin/env bash
# CameraPath Lab — local dependency bootstrap for macOS / Apple Silicon.
#
#   ./scripts/bootstrap_macos.sh              install everything needed
#   ./scripts/bootstrap_macos.sh --with-torch also install PyTorch (~2.5 GB,
#                                             only needed for the optional VGGT
#                                             backend)
#   ./scripts/bootstrap_macos.sh --check      report only, change nothing
#
# Nothing here requires CUDA and nothing calls out to a cloud service.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

WITH_TORCH=0
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --with-torch) WITH_TORCH=1 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; DIM=$'\033[2m'; RST=$'\033[0m'
ok()   { printf '  %s✓%s %s\n' "$GRN" "$RST" "$1"; }
warn() { printf '  %s!%s %s\n' "$YEL" "$RST" "$1"; }
bad()  { printf '  %s✗%s %s\n' "$RED" "$RST" "$1"; }
head_() { printf '\n%s%s%s\n' "$BOLD" "$1" "$RST"; }

MISSING=()
PY_MIN_MINOR=11

head_ "Host"
if [ "$(uname -s)" != "Darwin" ]; then
  bad "not macOS — this script targets macOS. The app may still run."
else
  ok "macOS $(sw_vers -productVersion)"
fi
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  ok "Apple Silicon: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo arm64)"
else
  warn "architecture $ARCH — tuned for Apple Silicon, will still run"
fi
MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 8589934592) / 1073741824 ))
ok "unified memory: ${MEM_GB} GB  ${DIM}(analysis sizes are derived from this at runtime)${RST}"
DISK_GB=$(df -g / 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${DISK_GB:-}" ] && [ "$DISK_GB" -lt 10 ]; then
  warn "only ${DISK_GB} GB free — renders and job scratch need headroom"
else
  ok "disk free: ${DISK_GB:-?} GB"
fi

head_ "Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "brew at $(command -v brew)"
  HAVE_BREW=1
else
  HAVE_BREW=0
  warn "Homebrew not found. Install it from https://brew.sh, or install the"
  echo "    dependencies below manually."
fi

install_brew() {  # $1 = formula, $2 = human name
  if [ "$CHECK_ONLY" = "1" ]; then MISSING+=("brew install $1"); return; fi
  if [ "$HAVE_BREW" = "1" ]; then
    echo "    installing $2 via Homebrew..."
    brew install "$1" || warn "brew install $1 failed — install $2 manually"
  else
    MISSING+=("brew install $1")
  fi
}

head_ "FFmpeg"
if command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  ok "$(ffmpeg -version 2>/dev/null | head -1 | cut -c1-60)"
else
  bad "ffmpeg/ffprobe missing — required for all video ingestion"
  install_brew ffmpeg FFmpeg
fi

head_ "Blender"
BLENDER=""
for cand in \
  "$(command -v blender 2>/dev/null || true)" \
  "/Applications/Blender.app/Contents/MacOS/Blender" \
  "$HOME/Applications/Blender.app/Contents/MacOS/Blender"; do
  if [ -n "$cand" ] && [ -x "$cand" ]; then BLENDER="$cand"; break; fi
done
if [ -n "$BLENDER" ]; then
  ok "$("$BLENDER" --version 2>/dev/null | head -1) ${DIM}at $BLENDER${RST}"
else
  bad "Blender not found — analysis works, but MP4 proxy rendering will not"
  if [ "$CHECK_ONLY" = "1" ] || [ "$HAVE_BREW" = "0" ]; then
    MISSING+=("brew install --cask blender  # or download from blender.org")
  else
    echo "    installing Blender via Homebrew cask..."
    brew install --cask blender || warn "install Blender manually from blender.org"
  fi
fi

head_ "Node"
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node -v | sed 's/^v\([0-9]*\).*/\1/')"
  if [ "$NODE_MAJOR" -ge 18 ]; then
    ok "node $(node -v), npm $(npm -v 2>/dev/null)"
  else
    warn "node $(node -v) is older than v18 — the frontend needs v18+"
    install_brew node Node.js
  fi
else
  bad "Node.js missing — required for the frontend"
  install_brew node Node.js
fi

head_ "Python"
PYBIN=""
for cand in python3.13 python3.12 python3.11 python3; do
  p="$(command -v $cand 2>/dev/null || true)"
  [ -z "$p" ] && continue
  minor="$("$p" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
  major="$("$p" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)"
  if [ "$major" = "3" ] && [ "$minor" -ge "$PY_MIN_MINOR" ]; then
    # Prefer 3.11/3.12: pycolmap and torch publish arm64 wheels there first, and
    # a very new interpreter often has no wheel yet, forcing a source build.
    if [ "$minor" -le 12 ]; then PYBIN="$p"; break; fi
    [ -z "$PYBIN" ] && PYBIN="$p"
  fi
done
if [ -z "$PYBIN" ]; then
  bad "no Python 3.$PY_MIN_MINOR+ found"
  install_brew python@3.11 "Python 3.11"
  PYBIN="$(command -v python3.11 2>/dev/null || true)"
else
  ok "$("$PYBIN" -V) at $PYBIN"
  PYMINOR="$("$PYBIN" -c 'import sys; print(sys.version_info.minor)')"
  if [ "$PYMINOR" -ge 13 ]; then
    warn "Python 3.$PYMINOR is very new; pycolmap/torch wheels may not exist yet."
    echo "    If the install below fails, run: brew install python@3.11"
  fi
fi

head_ "Python environment"
if [ "$CHECK_ONLY" = "1" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then
    ok ".venv exists ($("$ROOT/.venv/bin/python" -V))"
  else
    warn ".venv not created yet"
  fi
elif [ -n "$PYBIN" ]; then
  if [ ! -x "$ROOT/.venv/bin/python" ]; then
    echo "    creating .venv with $PYBIN..."
    "$PYBIN" -m venv "$ROOT/.venv" || { bad "venv creation failed"; exit 1; }
  fi
  VPY="$ROOT/.venv/bin/python"
  ok "venv at .venv ($("$VPY" -V))"
  echo "    upgrading pip..."
  "$VPY" -m pip install -q --upgrade pip wheel
  echo "    installing backend requirements (this can take a few minutes)..."
  if "$VPY" -m pip install -r "$ROOT/backend/requirements-dev.txt"; then
    ok "backend dependencies installed"
  else
    bad "dependency install failed — see the output above"
  fi
  if [ "$WITH_TORCH" = "1" ]; then
    echo "    installing PyTorch (~2.5 GB)..."
    "$VPY" -m pip install -r "$ROOT/backend/requirements-optional.txt" \
      && ok "PyTorch installed (VGGT backend available)" \
      || warn "PyTorch install failed — VGGT stays disabled; nothing else is affected"
  else
    echo "    ${DIM}skipping PyTorch. It is only needed for the optional VGGT"
    echo "    backend; re-run with --with-torch to add it.${RST}"
  fi
fi

head_ "Frontend dependencies"
if [ "$CHECK_ONLY" = "1" ]; then
  [ -d "$ROOT/frontend/node_modules" ] && ok "node_modules present" || warn "not installed yet"
elif command -v npm >/dev/null 2>&1; then
  echo "    running npm install..."
  (cd "$ROOT/frontend" && npm install --no-fund --no-audit) \
    && ok "frontend dependencies installed" \
    || bad "npm install failed"
else
  warn "npm unavailable — skipping frontend install"
fi

head_ "Verification"
if [ -x "$ROOT/.venv/bin/python" ]; then
  "$ROOT/.venv/bin/python" "$ROOT/scripts/doctor.py" || true
fi

if [ ${#MISSING[@]} -gt 0 ]; then
  head_ "Action required"
  echo "  Run these, then re-run this script:"
  for m in "${MISSING[@]}"; do echo "    $m"; done
  exit 1
fi

head_ "Ready"
echo "  Start the app with:"
echo "    ./scripts/dev.sh"
echo "  Then open http://localhost:5173"
