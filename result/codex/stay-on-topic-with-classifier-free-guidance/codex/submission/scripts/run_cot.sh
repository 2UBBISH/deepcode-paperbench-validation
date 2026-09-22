#!/usr/bin/env bash
# Section 3.2: Chain-of-Thought with CFG on GSM8K and AQuA (Figures 2, 17),
# plus the self-consistency stacking experiment (contribution 3).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 experiments/run_cot.py \
  --models wizardlm-30b guanaco-65b \
  --datasets gsm8k aqua \
  --gammas 1.0 1.1 1.25 1.5 1.75 2.0 \
  --output results/cot.json

python3 experiments/run_cot.py \
  --models wizardlm-30b \
  --datasets gsm8k \
  --gammas 1.0 1.25 1.5 \
  --self-consistency 5 --sample \
  --output results/cot_self_consistency.json
