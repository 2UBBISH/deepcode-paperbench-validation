#!/usr/bin/env bash
#
# Phase 3 entry point: zero-shot downstream evaluation for FRE.
#
# Usage:
#   bash scripts/run_eval.sh [CONFIG] [-- extra args...]
#
# Positional arguments:
#   CONFIG    Path to a YAML configuration file.
#             Default: configs/antmaze.yaml
#
# Environment variables:
#   FRE_DEVICE   Torch device to use. Default: auto
#
# Any arguments after a literal `--` are forwarded to the evaluation module.
set -euo pipefail

# Resolve repository root relative to this script, so the command can be
# executed from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${1:-configs/antmaze.yaml}"
DEVICE="${FRE_DEVICE:-auto}"

# Keep the original config path before shifting the positional argument.
ORIGINAL_CONFIG="${CONFIG}"
shift 1 || true

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift 1
  EXTRA_ARGS=("$@")
fi

cd "${REPO_ROOT}"

echo "Running FRE zero-shot evaluation"
echo "  Config: ${ORIGINAL_CONFIG}"
echo "  Device: ${DEVICE}"
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  echo "  Extra args: ${EXTRA_ARGS[*]}"
fi

python -m fre.pipeline.evaluate \
  --config "${ORIGINAL_CONFIG}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"
