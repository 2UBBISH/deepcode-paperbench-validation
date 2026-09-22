#!/usr/bin/env bash
# Table 9 - design choices (learnable parameters x optimiser x loss).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 9 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --batch-size 64 \
  --out "${OUT_ROOT}/table9_design_choices.json"
