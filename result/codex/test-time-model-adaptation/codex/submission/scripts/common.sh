#!/usr/bin/env bash
# Shared configuration for the reproduction scripts.
#
#   export DATA_ROOT=/path/to/data          # ImageNet-C/R/V2/Sketch + ImageNet-1K
#   export STATS=/path/to/vit_b16_in1k.pt   # source in-distribution statistics
#   export OUT_ROOT=results
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results}"
STATS="${STATS:-${REPO_ROOT}/stats/vit_b16_in1k.pt}"
CKPT="${CKPT:-vit_base_patch16_224.augreg_in21k_ft_in1k}"

mkdir -p "${OUT_ROOT}" "$(dirname "${STATS}")"

run() {
  echo "+ python -m foa.cli $*"
  python -m foa.cli "$@"
}

# 1) source in-distribution statistics (32 unlabelled ImageNet-1K val images)
stats() {
  run source-stats --data-root "${DATA_ROOT}/imagenet" --checkpoint "${CKPT}" \
    --num-samples 32 --out "${STATS}"
}
