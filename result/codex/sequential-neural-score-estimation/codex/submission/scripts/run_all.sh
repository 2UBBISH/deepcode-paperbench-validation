#!/usr/bin/env bash
# Reproduce every numerical experiment reported in the main body of the paper.
#
# This runs for a long time (see the README): the benchmark grid is
# 8 tasks x 3 simulation budgets x 4 (TS)NPSE configurations plus baselines.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HERE}"
mkdir -p results

# --- Section 5.2: NPSE and TSNPSE on the eight sbibm benchmarks -------------
python experiments/run_benchmark.py \
  --tasks all --budgets 1000 10000 100000 \
  --methods npse_ve npse_vp tsnpse_ve tsnpse_vp \
  --output results/benchmark.jsonl

# --- Section 5.2: baselines (NPE, SNPE-C via sbibm; TSNPE via tsnpe_neurips) -
python experiments/run_baselines.py \
  --tasks all --budgets 1000 10000 100000 \
  --methods npe snpe_c tsnpe \
  --output results/baselines.jsonl

# --- Appendix C.5: SNPSE-A / SNPSE-B / SNPSE-C ablation --------------------
python experiments/run_benchmark.py \
  --tasks slcp gaussian_linear_uniform --budgets 1000 10000 \
  --methods snpse_a snpse_b snpse_c \
  --output results/snpse_variants.jsonl

# --- Section 5.3: pyloric network (requires the pyloric + NEURON setup) -----
python experiments/run_pyloric.py --output-dir results/pyloric

# --- Figures 2 and 3 --------------------------------------------------------
python experiments/collect_results.py \
  results/benchmark.jsonl results/baselines.jsonl --csv results/c2st_table.csv
