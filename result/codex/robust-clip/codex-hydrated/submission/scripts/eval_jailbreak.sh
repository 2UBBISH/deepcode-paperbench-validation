#!/usr/bin/env bash
# Jailbreaking attacks against LLaVA-1.5 (Table 7, Sec. 4.4).
# 5000 PGD iterations, alpha = 1/255, no momentum, one image, radii 0..64/255.
set -euo pipefail

OUT="${OUT:-results/jailbreak}"
mkdir -p "$OUT"

run_one () {
  local name="$1"; shift
  echo "=== jailbreak: $name ==="
  python -m robust_clip.eval.jailbreak \
    --eps-list 0 16/255 32/255 64/255 \
    --iterations 5000 \
    --alpha 1/255 \
    --output "$OUT/$name.json" \
    "$@"
}

run_one CLIP
run_one TeCoA4 --clip-checkpoint "${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}"
run_one FARE4  --clip-checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
