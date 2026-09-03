#!/usr/bin/env bash
#
# Phase 2: FRE-conditioned offline IQL training.
#
# Usage:
#   bash scripts/run_train.sh [CONFIG] [-- extra args...]
#
# Examples:
#   bash scripts/run_train.sh configs/antmaze.yaml
#   FRE_DEVICE=cuda:0 bash scripts/run_train.sh configs/exorl.yaml -- --model-path checkpoints/fre_encoder.pt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/antmaze.yaml}"
shift $(( $# > 0 ? 1 : 0 ))

# Allow optional explicit device via environment variable.
DEVICE="${FRE_DEVICE:-auto}"

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

echo "=== FRE Phase 2: train_agent ==="
echo "Config: ${CONFIG}"
echo "Device: ${DEVICE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "Extra args: ${EXTRA_ARGS[*]}"
fi

python -m fre.pipeline.train_agent \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
