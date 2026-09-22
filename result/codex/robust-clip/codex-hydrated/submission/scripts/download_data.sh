#!/usr/bin/env bash
# Fetch the datasets and the model weights that the reproduction needs.
#
# Everything is optional: every evaluation script can also read a local
# ``--data-root`` in the layout of the original datasets (COCO, Flickr30k,
# VQA v2, TextVQA, POPE, ScienceQA, CLIP_benchmark).
#
# Usage:  bash scripts/download_data.sh [imagenet] [coco] [flickr] [vqa] [pope] [sqa] [jailbreak]
#         bash scripts/download_data.sh all
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
mkdir -p "$DATA_ROOT"

hf_download () {  # repo_id, local_dir
  python - "$1" "$2" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo_id, local_dir=local_dir)
print(f"downloaded {repo_id} -> {local_dir}")
PY
}

imagenet() {
  echo "=== ImageNet (HuggingFace: imagenet-1k, ~150 GB) ==="
  python - <<'PY'
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)   # addendum
print(dataset)
PY
}

coco() {
  echo "=== COCO 2014 captions + val2014 images ==="
  mkdir -p "$DATA_ROOT/coco"
  for url in \
    http://images.cocodataset.org/annotations/annotations_trainval2014.zip \
    http://images.cocodataset.org/zips/val2014.zip ; do
    (cd "$DATA_ROOT/coco" && curl -LO "$url")
  done
  (cd "$DATA_ROOT/coco" && unzip -o -q annotations_trainval2014.zip && unzip -o -q val2014.zip)
  echo "COCO ready in $DATA_ROOT/coco"
}

flickr() {
  echo "=== Flickr30k (Karpathy split, 5 captions per image) ==="
  hf_download "nlphuji/flickr30k" "$DATA_ROOT/flickr30k"
}

vqa() {
  echo "=== VQA v2 (val) ==="
  mkdir -p "$DATA_ROOT/vqav2"
  for url in \
    https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip \
    https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip ; do
    (cd "$DATA_ROOT/vqav2" && curl -LO "$url")
  done
  (cd "$DATA_ROOT/vqav2" && unzip -o -q 'v2_*.zip')
  echo "=== TextVQA 0.5.1 ==="
  mkdir -p "$DATA_ROOT/textvqa"
  (cd "$DATA_ROOT/textvqa" && curl -LO https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json)
}

pope() {
  echo "=== POPE annotations ==="
  mkdir -p "$DATA_ROOT/pope"
  for split in random popular adversarial; do
    curl -L "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_${split}.json" \
      -o "$DATA_ROOT/pope/coco_pope_${split}.json"
  done
}

sqa() {
  echo "=== ScienceQA (test split, image questions) ==="
  hf_download "derek-thomas/ScienceQA" "$DATA_ROOT/scienceqa"
}

jailbreak() {
  echo "=== Harmful corpora and the clean image of Qi et al. (2023) ==="
  mkdir -p "$DATA_ROOT/harmful_corpus"
  base="https://raw.githubusercontent.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/main"
  curl -L "$base/harmful_corpus/derogatory_corpus.csv" -o "$DATA_ROOT/harmful_corpus/derogatory_corpus.csv"
  curl -L "$base/harmful_corpus/manual_harmful_instructions.csv" -o "$DATA_ROOT/harmful_corpus/manual_harmful_instructions.csv"
  curl -L "$base/adversarial_images/clean.jpeg" -o "$DATA_ROOT/harmful_corpus/clean.jpeg"
}

targets=("$@")
if [[ ${#targets[@]} -eq 0 || "${targets[0]}" == "all" ]]; then
  targets=(imagenet coco flickr vqa pope sqa jailbreak)
fi
for target in "${targets[@]}"; do
  "$target"
done
echo "all requested data is under $DATA_ROOT"
