#!/usr/bin/env bash
# Section 4 / Appendix C.2: accuracy vs inference FLOPs and the ANCOVA table.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 experiments/run_flops_ancova.py \
  --input results/zeroshot.json \
  --output-dir results/flops \
  --make-plots
