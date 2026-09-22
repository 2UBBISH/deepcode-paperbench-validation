#!/usr/bin/env bash
# App. C.4: clean and adversarial embedding loss E[L_clean], E[L_adv] on
# ImageNet val, eps = 4/255 (Table 14).
set -euo pipefail

OUT="${OUT:-results/embedding_loss}"
mkdir -p "$OUT"

python -m robust_clip.eval.embedding_loss \
  --num-samples 500 \
  --eps 4/255 \
  --n-iter 100 \
  --checkpoint "${FARE2_CKPT:-}" \
  --output "$OUT/FARE2.json"
