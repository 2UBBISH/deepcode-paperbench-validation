#!/usr/bin/env bash
# Stealthy targeted attacks against LLaVA (Table 3, Sec. 4.2).
# 10 000 iterations of APGD, 6 target captions, 25 images each.
set -euo pipefail

OUT="${OUT:-results/targeted}"
mkdir -p "$OUT"

run_one () {
  local name="$1"; shift
  echo "=== targeted attacks: $name ==="
  python -m robust_clip.eval.targeted_attack \
    --eps-list 2/255 4/255 \
    --iterations 10000 \
    --num-images 25 \
    --output "$OUT/$name.json" \
    "$@"
}

run_one CLIP
run_one TeCoA2 --clip-checkpoint "${TECOA2_CKPT:-runs/robust_clip/TeCoA2-CLIP_ViT-L-14.pt}"
run_one FARE2  --clip-checkpoint "${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}"
run_one TeCoA4 --clip-checkpoint "${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}"
run_one FARE4  --clip-checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
