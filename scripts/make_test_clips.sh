#!/usr/bin/env bash
# Quick ffmpeg-only test clips for plumbing tests (probe / decode / shots / flow).
# These are 2D image transforms, NOT physically-correct 3D camera moves — they
# have zero parallax by construction. Real ground truth comes from
# blender/gen_synthetic_scene.py (spec §24); these exist so the ingestion and
# motion-signature code can be exercised without launching Blender.
set -euo pipefail
OUT="${1:-benchmarks/clips}"
mkdir -p "$OUT"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# A texture-rich still: mandelbrot has strong corners at many scales, which is
# what Shi-Tomasi and SIFT both want.
ffmpeg -y -v error -f lavfi -i "mandelbrot=size=3840x2160:maxiter=180" -frames:v 1 "$TMP/tex.png"
ffmpeg -y -v error -f lavfi -i "mandelbrot=size=3840x2160:maxiter=400:start_x=-0.6:start_y=0.3:start_scale=1.2" -frames:v 1 "$TMP/tex2.png"

enc() { ffmpeg -y -v error -i "$1" -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart "$2"; }

# 1. Constant-velocity horizontal pan (pure translation in image space).
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 3 \
  -vf "crop=1280:720:x='(in_w-1280)*t/3':y='(in_h-720)/2',fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/pan_constant.mp4"

# 2. Accelerating pan — ease-in on t^2. Tests that acceleration survives.
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 3 \
  -vf "crop=1280:720:x='(in_w-1280)*pow(t/3,2)':y='(in_h-720)/2',fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/pan_accelerating.mp4"

# 3. Zoom only — pure scale change about the centre, no translation.
# zoompan has no `t`; it exposes the output frame counter `on`. 90 frames = 3 s.
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -frames:v 90 \
  -vf "zoompan=z='1+0.6*on/90':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/zoom_only.mp4"

# 4. In-plane rotation (roll-like).
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 3 \
  -vf "rotate='0.35*t':c=black:ow=1280:oh=720,fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/roll.mp4"

# 5. Hard cut in the middle — two visually unrelated shots, one file.
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 1.5 \
  -vf "crop=1280:720:x='(in_w-1280)*t/1.5':y=200,fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/a.mp4"
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex2.png" -t 1.5 \
  -vf "crop=1280:720:x=400:y='(in_h-720)*t/1.5',fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/b.mp4"
printf "file '%s'\nfile '%s'\n" "$TMP/a.mp4" "$TMP/b.mp4" > "$TMP/list.txt"
ffmpeg -y -v error -f concat -safe 0 -i "$TMP/list.txt" -c:v libx264 -preset veryfast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart "$OUT/hard_cut.mp4"

# 6. Fast motion WITHOUT a cut — the classic false-positive case for cut
#    detection. Must be detected as one continuous shot (spec §4).
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 2 \
  -vf "crop=1280:720:x='(in_w-1280)*(0.5+0.5*sin(6*t))':y='(in_h-720)*(0.5+0.5*cos(5*t))',fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/fast_motion_no_cut.mp4"

# 7. Handheld-like jitter riding on a slow drift. Tests EXACT fidelity.
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 3 \
  -vf "crop=1280:720:x='(in_w-1280)*(0.3+0.1*t/3)+14*sin(19*t)+7*sin(31*t)':y='(in_h-720)*0.5+11*cos(23*t)+5*sin(37*t)',fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/handheld_jitter.mp4"

# 8. Static camera — must classify as static, not as drifting noise.
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 2 \
  -vf "crop=1280:720:x=1200:y=700,fps=30,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/static.mp4"

# 9. Portrait orientation + odd frame rate, to exercise ingestion edge cases.
ffmpeg -y -v error -loop 1 -framerate 24000/1001 -i "$TMP/tex.png" -t 2 \
  -vf "crop=720:1280:x='(in_w-720)*t/2':y='(in_h-1280)/2',format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 -movflags +faststart "$OUT/portrait_23976.mp4"

echo "wrote:"
for f in "$OUT"/*.mp4; do
  printf '  %-28s %s\n' "$(basename "$f")" "$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$f")"
done
