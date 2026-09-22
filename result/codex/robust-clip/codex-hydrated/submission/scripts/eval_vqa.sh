#!/usr/bin/env bash
# LLaVA-1.5 7B visual question answering (VQAv2 / TextVQA), Table 1.
set -euo pipefail

OUT="${OUT:-results/vqa}"
mkdir -p "$OUT"

run_one () {
  local name="$1"; local dataset="$2"; shift 2
  echo "=== vqa: $name / $dataset ==="
  python -m robust_clip.eval.vqa \
    --dataset "$dataset" \
    --num-attack-samples 500 \
    --attack-iterations 100 \
    --output "$OUT/${name}_${dataset}.json" \
    "$@"
}

for DATASET in vqav2 textvqa; do
  run_one CLIP  "$DATASET"
  run_one FARE2 "$DATASET" --clip-checkpoint "${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}"
  run_one FARE4 "$DATASET" --clip-checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
done
