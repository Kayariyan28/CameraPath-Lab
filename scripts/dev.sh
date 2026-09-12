#!/usr/bin/env bash
# Start the CameraPath Lab backend and frontend together.
#   ./scripts/dev.sh            both
#   ./scripts/dev.sh backend    backend only
#   ./scripts/dev.sh frontend   frontend only
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV="$ROOT/.venv"
PORT="${CPL_PORT:-8848}"
WHAT="${1:-both}"

# The project path can contain spaces, so always invoke the interpreter
# directly rather than relying on venv console-script shebangs.
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "No virtualenv found. Run ./scripts/bootstrap_macos.sh first." >&2
  exit 1
fi

free_port() {
  local pids
  pids="$(lsof -ti:"$1" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "port $1 in use by PID(s) $pids — stopping them"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -ti:"$1" 2>/dev/null || true)"
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

PIDS=()
cleanup() {
  for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ "$WHAT" = "both" ] || [ "$WHAT" = "backend" ]; then
  free_port "$PORT"
  echo "starting backend on http://127.0.0.1:$PORT"
  "$PY" -m uvicorn app.main:app \
    --host 127.0.0.1 --port "$PORT" \
    --app-dir "$ROOT/backend" --reload \
    --reload-dir "$ROOT/backend" &
  PIDS+=($!)
fi

if [ "$WHAT" = "both" ] || [ "$WHAT" = "frontend" ]; then
  if [ ! -d "$ROOT/frontend/node_modules" ]; then
    echo "installing frontend dependencies..."
    (cd "$ROOT/frontend" && npm install --no-fund --no-audit)
  fi
  free_port 5173
  echo "starting frontend on http://localhost:5173"
  (cd "$ROOT/frontend" && npm run dev) &
  PIDS+=($!)
fi

echo
echo "CameraPath Lab is starting. Open http://localhost:5173"
echo "Press Ctrl-C to stop."
wait
