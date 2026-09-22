#!/usr/bin/env bash
# Table 11 - non-i.i.d. scenarios (online label shift / mixed domain shifts).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 11 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --methods "${METHODS:-TENT,SAR,FOA}" \
  --batch-size 64 \
  --out "${OUT_ROOT}/table11_non_iid.json"
