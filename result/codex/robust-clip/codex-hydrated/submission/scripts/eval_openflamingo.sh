#!/usr/bin/env bash
# OpenFlamingo-9B captioning / VQA (rows 7-11 of Table 1).
#
# OpenFlamingo is evaluated zero-shot (context text, no context images) and
# consumes the *full* token sequence of the CLIP vision transformer.
#
# Requires the `open_flamingo` package (see requirements.txt).
set -euo pipefail

OUT="${OUT:-results/openflamingo}"
mkdir -p "$OUT"

run_caption () {
  local name="$1"; local dataset="$2"; shift 2
  echo "=== OpenFlamingo captioning: $name / $dataset ==="
  python -m robust_clip.eval.captioning \
    --backend openflamingo --dataset "$dataset" \
    --num-attack-samples 500 --attack-iterations 100 \
    --output "$OUT/${name}_${dataset}.json" "$@"
}

run_vqa () {
  local name="$1"; local dataset="$2"; shift 2
  echo "=== OpenFlamingo VQA: $name / $dataset ==="
  python -m robust_clip.eval.vqa \
    --backend openflamingo --dataset "$dataset" \
    --num-attack-samples 500 --attack-iterations 100 \
    --output "$OUT/${name}_${dataset}.json" "$@"
}

for DATASET in coco flickr30k; do
  run_caption CLIP  "$DATASET"
  run_caption TeCoA2 "$DATASET" --clip-checkpoint "${TECOA2_CKPT:-runs/robust_clip/TeCoA2-CLIP_ViT-L-14.pt}"
  run_caption FARE2  "$DATASET" --clip-checkpoint "${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}"
  run_caption TeCoA4 "$DATASET" --clip-checkpoint "${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}"
  run_caption FARE4  "$DATASET" --clip-checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
done

for DATASET in vqav2 textvqa; do
  run_vqa CLIP  "$DATASET"
  run_vqa TeCoA2 "$DATASET" --clip-checkpoint "${TECOA2_CKPT:-runs/robust_clip/TeCoA2-CLIP_ViT-L-14.pt}"
  run_vqa FARE2  "$DATASET" --clip-checkpoint "${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}"
  run_vqa TeCoA4 "$DATASET" --clip-checkpoint "${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}"
  run_vqa FARE4  "$DATASET" --clip-checkpoint "${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"
done
