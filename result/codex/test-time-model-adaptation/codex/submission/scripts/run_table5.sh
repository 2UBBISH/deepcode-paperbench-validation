#!/usr/bin/env bash
# Table 5 - ablation of the FOA components (ImageNet-C, severity 5).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 5 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --batch-size 64 \
  --out "${OUT_ROOT}/table5_ablation.json"
