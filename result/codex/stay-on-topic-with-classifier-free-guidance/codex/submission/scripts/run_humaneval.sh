#!/usr/bin/env bash
# Section 3.3.1: CodeGen-mono on HumanEval with CFG (Tables 2, 7-9; Figure 3).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 experiments/run_humaneval.py \
  --models codegen-350m-mono codegen-2b-mono codegen-6b-mono \
  --gammas 1.0 1.1 1.25 1.5 1.75 2.0 \
  --temperatures 0.2 0.6 0.8 \
  --k 100 \
  --pass-k 1 10 100 \
  --max-new-tokens 512 \
  --output results/humaneval.json
