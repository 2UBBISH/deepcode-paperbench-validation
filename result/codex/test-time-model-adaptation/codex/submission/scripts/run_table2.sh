#!/usr/bin/env bash
# Table 2 - ImageNet-C (severity 5), full-precision ViT-Base.
#
#   bash scripts/run_table2.sh            # every method of the table
#   METHODS=FOA bash scripts/run_table2.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 2 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --methods "${METHODS:-NoAdapt,LAME,T3A,TENT,CoTTA,SAR,FOA}" \
  --batch-size 64 \
  --out "${OUT_ROOT}/table2_imagenet_c.json"
