#!/usr/bin/env bash
# Zero-shot classification (Table 4): clean accuracy plus APGD-CE / targeted
# APGD-DLR robustness at eps = 2/255 and 4/255 for every CLIP variant.
#
# Usage:  CHECKPOINT=runs/robust_clip/FARE2-CLIP_ViT-L-14.pt bash scripts/eval_zero_shot.sh
set -euo pipefail

OUT="${OUT:-results/zero_shot}"
mkdir -p "$OUT"

run_one () {
  local name="$1"; shift
  echo "=== zero-shot evaluation: $name ==="
  python -m robust_clip.eval.zero_shot \
    --eps-list 2/255 4/255 \
    --n-samples 1000 \
    --n-iter 100 \
    --output "$OUT/$name.json" \
    "$@"
}

run_one CLIP
run_one TeCoA2 --checkpoint "${TECOA2_CKPT:-runs/robust_clip/TeCoA2-CLIP_ViT-L-14.pt}"
run_one FARE2  --checkpoint "${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}"
run_one TeCoA4 --checkpoint "${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}"
run_one FARE4  --checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
