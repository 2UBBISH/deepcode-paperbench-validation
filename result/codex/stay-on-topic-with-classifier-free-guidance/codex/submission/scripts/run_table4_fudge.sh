#!/usr/bin/env bash
# Section 6 / Table 4: CFG vs FUDGE for sentiment and toxicity control.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 experiments/run_fudge_comparison.py \
  --gpt2 gpt2 \
  --tasks sentiment toxicity \
  --gammas 1 2 3 4 5 6 \
  --lambdas 0.5 1 2 5 \
  --n-samples 64 \
  --output results/table4.json
