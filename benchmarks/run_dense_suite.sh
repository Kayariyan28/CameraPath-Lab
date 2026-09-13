#!/usr/bin/env bash
# Run the dense end-to-end benchmark over every rendered synthetic scene.
cd "$(dirname "$0")/.."
OUT=benchmarks/results/dense_suite.jsonl
mkdir -p benchmarks/results; : > "$OUT"
for f in benchmarks/synthetic/*.truth.json; do
  scene=$(basename "$f" .truth.json)
  line=$(.venv/bin/python benchmarks/eval_dense.py "$scene" 2>&1 | tee /dev/stderr | grep '^{' | tail -1)
  [ -n "$line" ] && echo "$line" >> "$OUT" || echo "{\"scene\": \"$scene\", \"crashed\": true}" >> "$OUT"
done
echo "wrote $OUT"
