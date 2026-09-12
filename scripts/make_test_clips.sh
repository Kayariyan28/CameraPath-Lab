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

# Reference texture. This must be feature-rich but NOT self-similar.
#
# The obvious choice, `mandelbrot`, is actively harmful here: a fractal is
# scale-invariant, so many locations look locally identical, which is the
# adversarial worst case for Lucas-Kanade and for SIFT ratio-test matching. LK
# locks onto the wrong-but-identical structure and forward-backward validation
# cannot catch it, because the round trip returns to the start via an equally
# valid-looking path. Measured on an otherwise identical pan: the fractal
# texture produced 29.6% of the frame falsely flagged as moving content, while a
# non-self-similar texture under the same camera motion produced 0.0% and 100%
# model agreement.
#
# So: low-frequency plasma for large-scale structure SIFT can anchor on, plus
# fine random noise for the high-frequency corners LK wants, with no structure
# repeated anywhere in the image.
mk_texture() {  # $1=out  $2=seed offset  $3..: phase tweaks
  local out="$1" o="$2"
  ffmpeg -y -v error -f lavfi -i "nullsrc=s=3840x2160,format=gray,geq=\
'128\
+46*sin((X+$o)/53)\
+34*sin((Y+$o)/31)\
+40*sin((X+Y+$o)/79)\
+28*sin((X-Y+$o)/23)\
+22*sin(hypot(X-1900,Y-1000)/41)'" -frames:v 1 "$TMP/plasma_$o.png"
  ffmpeg -y -v error -f lavfi \
    -i "nullsrc=s=3840x2160,format=gray,geq=random($((o+1)))*255" -frames:v 1 "$TMP/noise_$o.png"
  ffmpeg -y -v error -i "$TMP/plasma_$o.png" -i "$TMP/noise_$o.png" \
    -filter_complex "[0][1]blend=all_mode=overlay:all_opacity=0.42,format=gray" \
    -frames:v 1 "$out"
}
mk_texture "$TMP/tex.png" 0
mk_texture "$TMP/tex2.png" 811

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

# 10. Three shots, two cuts. Verifies multi-cut handling and that shot ids and
#     boundaries stay contiguous.
mk_texture "$TMP/tex3.png" 1607
for i in 1 2 3; do :; done
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 1.0 \
  -vf "crop=1280:720:x='(in_w-1280)*t':y=300,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/s1.mp4"
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex2.png" -t 1.0 \
  -vf "crop=1280:720:x=900:y='(in_h-720)*t',format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/s2.mp4"
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex3.png" -t 1.0 \
  -vf "rotate='0.4*t':c=black:ow=1280:oh=720,format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/s3.mp4"
printf "file '%s'\nfile '%s'\nfile '%s'\n" "$TMP/s1.mp4" "$TMP/s2.mp4" "$TMP/s3.mp4" > "$TMP/list3.txt"
ffmpeg -y -v error -f concat -safe 0 -i "$TMP/list3.txt" -c:v libx264 -preset veryfast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart "$OUT/three_shots.mp4"

# 11. A whip pan immediately followed by a hard cut. The adversarial case: the
#     detector must reject the whip and still catch the cut right after it.
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex.png" -t 1.2 \
  -vf "crop=1280:720:x='(in_w-1280)*(0.5+0.5*sin(7*t))':y='(in_h-720)*0.5',format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/w1.mp4"
ffmpeg -y -v error -loop 1 -framerate 30 -i "$TMP/tex2.png" -t 1.0 \
  -vf "crop=1280:720:x=200:y='(in_h-720)*0.4+40*t',format=yuv420p" \
  -c:v libx264 -preset veryfast -crf 18 "$TMP/w2.mp4"
printf "file '%s'\nfile '%s'\n" "$TMP/w1.mp4" "$TMP/w2.mp4" > "$TMP/listw.txt"
ffmpeg -y -v error -f concat -safe 0 -i "$TMP/listw.txt" -c:v libx264 -preset veryfast -crf 18 \
  -pix_fmt yuv420p -movflags +faststart "$OUT/cut_after_whip.mp4"

echo "wrote:"
for f in "$OUT"/*.mp4; do
  printf '  %-28s %s\n' "$(basename "$f")" "$(ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate -of csv=p=0 "$f")"
done
