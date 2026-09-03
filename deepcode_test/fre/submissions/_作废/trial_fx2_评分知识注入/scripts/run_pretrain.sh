#!/usr/bin/env bash
# Pretrain the FRE reward encoder (Phase 1).
#
# Usage:
#   bash scripts/run_pretrain.sh [CONFIG] [-- extra args...]
#
# CONFIG defaults to configs/antmaze.yaml. Extra arguments are forwarded to
# `python -m fre.pipeline.pretrain_encoder`.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${1:-configs/antmaze.yaml}"
shift || true

# Optional GPU override.
DEVICE="${FRE_DEVICE:-auto}"

echo "=============================================="
echo "FRE Phase 1: pretrain reward encoder"
echo "  config : ${CONFIG}"
echo "  device : ${DEVICE}"
echo "=============================================="

python -m fre.pipeline.pretrain_encoder \
  --config "${CONFIG}" \
  --device "${DEVICE}" \
  "$@"
