#!/usr/bin/env bash
# Adversarial fine-tuning of the CLIP vision encoder with FARE and TeCoA.
#
# Trains the four robust models of the paper (2 epochs on ImageNet, 10 PGD
# steps, step size 1/255) and writes one checkpoint per model to $OUT.
#
# Usage:  IMAGENET_ROOT=/path/to/imagenet bash scripts/train_fare_tecoa.sh
set -euo pipefail

OUT="${OUT:-runs/robust_clip}"
IMAGENET_ROOT="${IMAGENET_ROOT:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NGPU="${NGPU:-1}"

DATASET_ARGS=()
if [[ -n "$IMAGENET_ROOT" ]]; then
  DATASET_ARGS=(--imagenet-root "$IMAGENET_ROOT")
fi

for CONFIG in fare_vitl14_eps2 fare_vitl14_eps4 tecoa_vitl14_eps2 tecoa_vitl14_eps4; do
  echo "=== training $CONFIG ==="
  if [[ "$NGPU" -gt 1 ]]; then
    torchrun --standalone --nproc_per_node="$NGPU" -m robust_clip.training.train \
      --config "configs/${CONFIG}.json" \
      --output-dir "$OUT" \
      --batch-size 128 \
      --grad-accum-steps 1 \
      --precision bf16 \
      --num-workers 8 \
      --gradient-checkpointing \
      "${DATASET_ARGS[@]}"
  else
    python -m robust_clip.training.train \
      --config "configs/${CONFIG}.json" \
      --output-dir "$OUT" \
      --batch-size 32 \
      --grad-accum-steps 4 \
      --precision bf16 \
      --num-workers 8 \
      --gradient-checkpointing \
      "${DATASET_ARGS[@]}"
  fi
done

echo "checkpoints written to $OUT"
