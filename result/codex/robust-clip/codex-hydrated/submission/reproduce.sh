#!/usr/bin/env bash
# End-to-end reproduction of "Robust CLIP" (ICML 2024).
#
#   bash reproduce.sh train      # fine-tune FARE^2/4 and TeCoA^2/4 on ImageNet
#   bash reproduce.sh zero_shot  # Table 4: clean + adversarial zero-shot accuracy
#   bash reproduce.sh caption    # Table 1: COCO / Flickr30k captioning with LLaVA
#   bash reproduce.sh vqa        # Table 1: VQAv2 / TextVQA with LLaVA
#   bash reproduce.sh targeted   # Table 3: stealthy targeted attacks
#   bash reproduce.sh other      # Tables 5-7: POPE, SQA-I, jailbreaking
#   bash reproduce.sh transfer   # Table 2: transfer attacks between LVLMs
#   bash reproduce.sh figures    # Figure 1 (radar plot) from the result JSONs
#   bash reproduce.sh download   # fetch the datasets
#   bash reproduce.sh all
#
# Environment variables:
#   IMAGENET_ROOT  local ImageNet (ImageFolder style: <root>/<wnid>/<img>)
#   OUT            output directory for checkpoints / results
set -euo pipefail

STAGE="${1:-all}"
export OUT="${OUT:-runs/robust_clip}"

run_train()    { bash scripts/train_fare_tecoa.sh; }
run_zeroshot() { bash scripts/eval_zero_shot.sh; }
run_caption()  { bash scripts/eval_lvlm_captioning.sh; }
run_vqa()      { bash scripts/eval_vqa.sh; }
run_targeted() { bash scripts/eval_targeted.sh; }
run_other()    { bash scripts/eval_pope_sqa.sh; bash scripts/eval_jailbreak.sh; }
run_transfer() { bash scripts/eval_transfer.sh; }
run_download() { bash scripts/download_data.sh "${DATA_PARTS:-coco flickr vqa pope sqa jailbreak}"; }

run_figures() {
  mkdir -p results
  python -m robust_clip.eval.figures \
    --results \
      "CLIP=results/zero_shot/CLIP.json,results/captioning/CLIP_coco.json,results/vqa/CLIP_vqav2.json" \
      "TeCoA2=results/zero_shot/TeCoA2.json,results/captioning/TeCoA2_coco.json,results/vqa/TeCoA2_vqav2.json" \
      "FARE2=results/zero_shot/FARE2.json,results/captioning/FARE2_coco.json,results/vqa/FARE2_vqav2.json" \
    --out results/figure1.png
}

case "$STAGE" in
  train)     run_train ;;
  zero_shot) run_zeroshot ;;
  caption)   run_caption ;;
  vqa)       run_vqa ;;
  targeted)  run_targeted ;;
  other)     run_other ;;
  transfer)  run_transfer ;;
  figures)   run_figures ;;
  download)  run_download ;;
  all)       run_train; run_zeroshot; run_caption; run_vqa; run_targeted; run_other; run_transfer; run_figures ;;
  *) echo "unknown stage '$STAGE'"; exit 1 ;;
esac
