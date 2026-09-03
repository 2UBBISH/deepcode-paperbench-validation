#!/usr/bin/env bash
#
# Run baseline evaluation for the FRE reproduction (GC-IQL, GC-BC, OPAL, FB, SF).
#
# Usage:
#   bash scripts/run_baselines.sh [CONFIG] [-- extra args...]
#
# Arguments:
#   CONFIG      Path to a YAML config, default: configs/antmaze.yaml
#   --          Everything after this separator is forwarded verbatim to the
#               baseline evaluation Python module.
#
# Environment:
#   FRE_DEVICE  Torch device, default: auto (resolves CUDA/CPU at runtime).

set -euo pipefail

# Resolve repository root relative to this script so the command works from any
# working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-configs/antmaze.yaml}"
ORIGINAL_CONFIG="${CONFIG}"
DEVICE="${FRE_DEVICE:-auto}"

shift || true

EXTRA_ARGS=()
if [ "$#" -gt 0 ]; then
  if [ "${1:-}" = "--" ]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

cd "${REPO_ROOT}"

echo "[run_baselines] repo root : ${REPO_ROOT}"
echo "[run_baselines] config    : ${ORIGINAL_CONFIG}"
echo "[run_baselines] device    : ${DEVICE}"
if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  echo "[run_baselines] extra args: ${EXTRA_ARGS[*]}"
fi

# shellcheck disable=SC2068
python -m fre.pipeline.evaluate_baselines \
  --config "${ORIGINAL_CONFIG}" \
  --device "${DEVICE}" \
  ${EXTRA_ARGS[@]}
