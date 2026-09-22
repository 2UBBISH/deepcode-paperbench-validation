#!/usr/bin/env bash
# Step 1: evaluate all 75 models on ImageNet + the 5 OOD datasets.
#
# Produces results/metrics.csv plus cached logits/targets that every later step
# consumes.  On a single GPU this takes a few hours (fp16 recommended).
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-./data/datasets}
OUT_DIR=${OUT_DIR:-results}
DEVICE=${DEVICE:-cuda}

python -m lca_on_the_line.evaluate \
  --data-root "$DATA_ROOT" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "${BATCH_SIZE:-64}"
