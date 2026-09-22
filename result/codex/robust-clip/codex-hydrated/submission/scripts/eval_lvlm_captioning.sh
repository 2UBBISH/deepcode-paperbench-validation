#!/usr/bin/env bash
# LLaVA-1.5 7B captioning (COCO / Flickr30k) with clean and adversarial CIDEr.
#
# Usage:  bash scripts/eval_lvlm_captioning.sh
set -euo pipefail

OUT="${OUT:-results/captioning}"
mkdir -p "$OUT"

run_one () {
  local name="$1"; local dataset="$2"; shift 2
  echo "=== captioning: $name / $dataset ==="
  python -m robust_clip.eval.captioning \
    --dataset "$dataset" \
    --num-attack-samples 500 \
    --attack-iterations 100 \
    --output "$OUT/${name}_${dataset}.json" \
    "$@"
}

for DATASET in coco flickr30k; do
  run_one CLIP  "$DATASET"
  run_one FARE2 "$DATASET" --clip-checkpoint "${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}"
  run_one FARE4 "$DATASET" --clip-checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
done
