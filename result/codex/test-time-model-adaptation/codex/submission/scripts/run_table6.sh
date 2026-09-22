#!/usr/bin/env bash
# Table 6 - FOA-I (interval update, batch size 1) on ImageNet-C gaussian noise, level 5.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 6 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --intervals "${INTERVALS:-4,8,16,32,64}" \
  --out "${OUT_ROOT}/table6_foa_i.json"
