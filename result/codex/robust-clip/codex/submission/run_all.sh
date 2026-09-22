#!/usr/bin/env bash
# End-to-end reproduction pipeline (run on a GPU machine).
#
#   bash run_all.sh                    # everything
#   bash run_all.sh train              # only the adversarial fine-tuning
#   bash run_all.sh eval               # only the evaluations
#
# Each step writes its results to results/ and can be adapted with the CLI flags
# documented in the README.

set -euo pipefail

STAGE="${1:-all}"
ARCH="${ARCH:-ViT-L-14}"
PRETRAINED="${PRETRAINED:-openai}"          # use laion2b_s32b_b82k for OpenFlamingo
CKPT_DIR="${CKPT_DIR:-checkpoints}"
RESULTS="${RESULTS:-results}"

mkdir -p "${CKPT_DIR}" "${RESULTS}" outputs

train() {
  for method in fare tecoa; do
    for eps in 2/255 4/255; do
      echo "== training ${method} at eps=${eps}"
      python scripts/train_clip.py \
        --method "${method}" --arch "${ARCH}" --pretrained "${PRETRAINED}" \
        --eps "${eps}" --epochs 2 --batch-size 128 --micro-batch-size 16 \
        --lr 1e-5 --wd 1e-4 --pgd-steps 10 --pgd-alpha 1/255 \
        --output-dir "${CKPT_DIR}" \
        --run-name "${method^^}${eps%%/*}-${ARCH}-${PRETRAINED}"
    done
  done
}

evaluate() {
  for eps in 2/255 4/255; do
    tag="FARE${eps%%/*}-${ARCH}-${PRETRAINED}"
    ckpt="${CKPT_DIR}/${tag}.pt"

    # Table 1: LVLM clean + robust performance (one task per invocation)
    for task in coco flickr30k vqav2 textvqa; do
      python scripts/eval_lvlm.py --backend llava --task "${task}" \
        --clip-arch "${ARCH}" --clip-pretrained "${PRETRAINED}" --checkpoint "${ckpt}" \
        --radii "${eps}" --n-adv 500 \
        --output "${RESULTS}/llava_${task}_${tag}.json"
    done

    # Table 4: zero-shot classification + AutoAttack
    python scripts/eval_zeroshot.py --arch "${ARCH}" --pretrained "${PRETRAINED}" \
      --checkpoint "${ckpt}" --datasets all --radii 2/255 4/255 \
      --output "${RESULTS}/zeroshot_${tag}.json"

    # Tables 5 and 6
    python scripts/eval_pope_sqa.py --clip-arch "${ARCH}" --clip-pretrained "${PRETRAINED}" \
      --checkpoint "${ckpt}" --benchmark both --output "${RESULTS}/pope_sqa_${tag}.json"

    # Tables 2, 3 and 7
    python scripts/eval_transfer.py --target-checkpoint "${ckpt}" --eps "${eps}" \
      --output "${RESULTS}/transfer_${tag}.json"
    python scripts/eval_targeted.py --clip-arch "${ARCH}" --clip-pretrained "${PRETRAINED}" \
      --checkpoint "${ckpt}" --radii "${eps}" --iterations 10000
    python scripts/eval_jailbreak.py --mode both --eps 64/255 \
      --clip-arch "${ARCH}" --clip-pretrained "${PRETRAINED}" --checkpoint "${ckpt}"
  done

  # the original CLIP model is the reference row of every table
  python scripts/eval_zeroshot.py --arch "${ARCH}" --pretrained "${PRETRAINED}" \
    --datasets all --output "${RESULTS}/zeroshot_clip.json"
  python scripts/eval_lvlm.py --backend llava --task coco \
    --clip-arch "${ARCH}" --clip-pretrained "${PRETRAINED}" \
    --output "${RESULTS}/llava_coco_clip.json"
}

case "${STAGE}" in
  train) train ;;
  eval) evaluate ;;
  all) train; evaluate ;;
  *) echo "usage: $0 [all|train|eval]"; exit 1 ;;
esac

echo "done -- see ${RESULTS}/"
