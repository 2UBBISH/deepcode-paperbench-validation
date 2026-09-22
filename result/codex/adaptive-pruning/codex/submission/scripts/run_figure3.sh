#!/usr/bin/env bash
# Pruning-sparsity analysis (Figure 3 / Section 5.5).
set -euo pipefail
cd "$(dirname "$0")/.."

for model in roberta-base t5-base; do
  python scripts/run.py sweep --model "$model" --task sst2 --sparsities 0.2 0.4 0.6 0.8 \
      --output-dir "runs/sweep_${model}"
  python scripts/run.py baseline --baseline lora_prune --model "$model" --task sst2 --sparsity 0.6 \
      --output-dir "runs/sweep_${model}_lora_prune"
done
