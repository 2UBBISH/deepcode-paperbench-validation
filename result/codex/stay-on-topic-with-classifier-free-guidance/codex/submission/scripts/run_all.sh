#!/usr/bin/env bash
# Run every experiment end to end.  This is the script the paper's results
# come from; it needs a GPU box and several GPU-days, so it is *not* run by
# the offline test suite (see scripts/run_tests.sh for what runs here).
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/run_zeroshot.sh
./scripts/run_flops_ancova.sh
./scripts/run_humaneval.sh
./scripts/run_cot.sh
./scripts/run_section5.sh
./scripts/run_table4_fudge.sh

python3 experiments/plot_figures.py \
  --zeroshot results/zeroshot.json \
  --cot results/cot.json \
  --section5 results/section5/per_example.json \
  --output-dir results/figures

python3 experiments/report_tables.py \
  --zeroshot results/zeroshot.json \
  --humaneval results/humaneval.json \
  --ancova results/flops/ancova_table.json \
  --output results/tables.md
