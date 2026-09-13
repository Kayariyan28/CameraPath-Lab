#!/usr/bin/env bash
# Render the synthetic ground-truth benchmark set (spec section 24).
#   ./scripts/render_synthetic.sh [out_dir] [frames] [scene ...]
set -uo pipefail
cd "$(dirname "$0")/.."
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
OUT="${1:-benchmarks/synthetic}"; shift || true
FRAMES="${1:-60}"; shift || true
SCENES=("$@")
if [ ${#SCENES[@]} -eq 0 ]; then
  SCENES=(dolly_forward dolly_backward truck_right pedestal_up pan tilt roll orbit
          crane_up rise_and_tilt diagonal_flythrough fpv_curve accelerating
          decelerating handheld zoom_only dolly_zoom static moving_object)
fi
[ -x "$BLENDER" ] || { echo "Blender not found at $BLENDER" >&2; exit 1; }
mkdir -p "$OUT"
for s in "${SCENES[@]}"; do
  if [ -f "$OUT/$s.mp4" ] && [ -f "$OUT/$s.truth.json" ]; then
    echo "skip  $s (already rendered)"; continue
  fi
  echo "render $s ..."
  "$BLENDER" --background --python blender/gen_synthetic_scene.py -- \
    --scene "$s" --out "$OUT" --frames "$FRAMES" --fps 30 --samples 8 \
    >/tmp/cpl_render_$s.log 2>&1 \
    && echo "   ok   $s" \
    || { echo "   FAIL $s — see /tmp/cpl_render_$s.log"; tail -5 /tmp/cpl_render_$s.log; }
done
echo; echo "rendered set:"
ls -1 "$OUT"/*.mp4 2>/dev/null | while read -r f; do
  printf '  %-28s %s frames\n' "$(basename "$f")" \
    "$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$f")"
done
