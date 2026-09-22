#!/usr/bin/env bash
# POPE hallucination benchmark (Table 5) and ScienceQA-I (Table 6).
set -euo pipefail

OUT="${OUT:-results/other_tasks}"
mkdir -p "$OUT"

run_pope () {
  local name="$1"; shift
  python -m robust_clip.eval.pope --output "$OUT/pope_$name.json" "$@"
}

run_sqa () {
  local name="$1"; shift
  python -m robust_clip.eval.sqa --output "$OUT/sqa_$name.json" "$@"
}

for MODEL in "CLIP:" "TeCoA2:--clip-checkpoint ${TECOA2_CKPT:-runs/robust_clip/TeCoA2-CLIP_ViT-L-14.pt}" \
             "FARE2:--clip-checkpoint ${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}" \
             "TeCoA4:--clip-checkpoint ${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}" \
             "FARE4:--clip-checkpoint ${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"; do
  NAME="${MODEL%%:*}"
  ARGS="${MODEL#*:}"
  echo "=== POPE / SQA-I: $NAME ==="
  run_pope "$NAME" $ARGS
  run_sqa "$NAME" $ARGS
done
