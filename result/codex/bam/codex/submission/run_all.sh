#!/usr/bin/env bash
# Reproduce everything that is feasible on a CPU machine:
#   1. unit tests
#   2. quick versions of the three empirical sections
#   3. the numerical verification of Theorem 3.1
#   4. the posteriordb data preparation (no-op if already prepared)
#
# The full experiments (Figures 5.1-5.4) and the VAE pre-training are not run
# here; see README.md section 4 for the commands.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"   # override with PYTHON=python if needed
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

echo "== unit tests =="
${PYTHON} -m unittest discover -s tests -v

echo "== posteriordb data =="
${PYTHON} experiments/prepare_posteriordb.py

echo "== Section 5.1 (Gaussian, quick) =="
${PYTHON} experiments/run_gaussian_targets.py --quick --outdir results/smoke

echo "== Section 5.1 (sinh-arcsinh, quick) =="
${PYTHON} experiments/run_shash_targets.py --quick --outdir results/smoke

echo "== Section 5.2 (posteriordb, quick) =="
${PYTHON} experiments/run_posteriordb.py --quick --outdir results/smoke

echo "== Theorem 3.1 =="
${PYTHON} experiments/verify_theorem31.py --dim 8 --n-iters 60 --batch-check 6 --outdir results/smoke

echo "done; see results/"
