#!/usr/bin/env bash
# Transfer attacks (Table 2): craft adversarial COCO images against the
# surrogate (LLaVA-CLIP / OF-CLIP) and evaluate them on the same architecture
# with CLIP / TeCoA / FARE as vision encoder.
set -euo pipefail

OUT="${OUT:-results/transfer}"
mkdir -p "$OUT"

backend="${BACKEND:-llava}"
samples="${NUM_ATTACK_SAMPLES:-500}"

echo "=== crafting transfer images against the surrogate ($backend, CLIP encoder) ==="
python -m robust_clip.eval.transfer_attack \
  --backend "$backend" \
  --generate-only \
  --num-attack-samples "$samples" \
  --save-images "$OUT/adversarial_${backend}_clip" \
  --output "$OUT/surrogate_${backend}.json"

for MODEL in \
  "CLIP:" \
  "TeCoA2:--clip-checkpoint ${TECOA2_CKPT:-runs/robust_clip/TeCoA2-CLIP_ViT-L-14.pt}" \
  "FARE2:--clip-checkpoint ${FARE2_CKPT:-runs/robust_clip/FARE2-CLIP_ViT-L-14.pt}" \
  "TeCoA4:--clip-checkpoint ${TECOA4_CKPT:-runs/robust_clip/TeCoA4-CLIP_ViT-L-14.pt}" \
  "FARE4:--clip-checkpoint ${FARE4_CKPT:-runs/robust_clip/FARE4-CLIP_ViT-L-14.pt}"; do
  NAME="${MODEL%%:*}"
  ARGS="${MODEL#*:}"
  echo "=== transfer to $NAME ==="
  python -m robust_clip.eval.transfer_attack \
    --backend "$backend" \
    --reuse-images "$OUT/adversarial_${backend}_clip" \
    --output "$OUT/${backend}_${NAME}.json" \
    $ARGS
done
