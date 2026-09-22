#!/usr/bin/env bash
# Full reproduction pipeline.  Everything is written under ./results and ./tables.
set -euo pipefail

bash scripts/download_data.py --data-root "${DATA_ROOT:-./data/datasets}" --dataset all
bash scripts/run_benchmark.sh
bash scripts/run_reproduce_tables.sh
bash scripts/run_latent_hierarchy.sh

for backbone in resnet18 resnet50 vit_b_32 vit_l_32 convnext_tiny swin_b; do
  bash scripts/run_soft_labels.sh "$backbone" WordNet
done

bash scripts/run_prompt_engineering.sh
