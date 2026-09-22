#!/usr/bin/env bash
# Section 5: entropy, instruction-tuning comparison and the vocabulary
# visualisation, all on Falcon-7b + a sample of P3.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 experiments/run_section5.py \
  --model tiiuae/falcon-7b \
  --instruct-model tiiuae/falcon-7b-instruct \
  --gamma 1.5 \
  --n-per-dataset 50 \
  --output-dir results/section5

python3 experiments/run_visualize_logits.py \
  --model gpt2-large \
  --prompt "The dragon flew over Paris, France" \
  --gamma 1.5 \
  --output results/vocab_ranking.txt \
  --json-output results/vocab_ranking.json
