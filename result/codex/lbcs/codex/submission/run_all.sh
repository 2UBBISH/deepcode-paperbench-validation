#!/usr/bin/env bash
# Reproduce every in-scope artefact of the paper.
#
#   bash run_all.sh              # the paper's full configuration
#   DRY=1 bash run_all.sh        # tiny smoke run of every script
#   DEVICE=cuda bash run_all.sh
#
# Each script writes CSV + JSON results into $RESULTS_DIR (default ./results).
set -euo pipefail

DEVICE="${DEVICE:-}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_ROOT="${DATA_ROOT:-}"
DRY="${DRY:-}"

PYTHON="${PYTHON:-python3}"

common_args=(--results-dir "${RESULTS_DIR}")
[[ -n "${DEVICE}" ]] && common_args+=(--device "${DEVICE}")
[[ -n "${DATA_ROOT}" ]] && common_args+=(--data-root "${DATA_ROOT}")
[[ -n "${DRY}" ]] && common_args+=(--dry-run)

echo "== unit tests =="
${PYTHON} -m unittest discover -s tests -v

echo "== analyses =="
${PYTHON} -m experiments.exp_theorem2_convergence "${common_args[@]}"
${PYTHON} -m experiments.exp_gradient_analysis "${common_args[@]}"
${PYTHON} -m experiments.exp_quick_validation "${common_args[@]}"

echo "== Figure 1 (trivial solutions) =="
${PYTHON} -m experiments.exp_fig1_trivial "${common_args[@]}"

echo "== Table 1 (MNIST-S) =="
${PYTHON} -m experiments.exp_table1_mnist_s "${common_args[@]}"

echo "== Tables 2 and 3 (F-MNIST, SVHN, CIFAR-10) =="
${PYTHON} -m experiments.exp_table2_table3 "${common_args[@]}"

echo "== Figure 2 (imperfect supervision) =="
${PYTHON} -m experiments.exp_fig2_robustness "${common_args[@]}"

echo "== Table 5 (mask initialisation) =="
${PYTHON} -m experiments.exp_table5_moderate_init "${common_args[@]}"

echo "== Table 6 (cross architecture) =="
${PYTHON} -m experiments.exp_table6_cross_arch "${common_args[@]}"

echo "all artefacts written to ${RESULTS_DIR}"
