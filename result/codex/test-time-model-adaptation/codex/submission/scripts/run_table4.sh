#!/usr/bin/env bash
# Table 4 - quantised ViT-Base (8-bit / 6-bit, PTQ4ViT) on ImageNet-C severity 5.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 4 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --methods "${METHODS:-NoAdapt,T3A,FOA}" \
  --bits "${BITS:-8,6}" \
  --batch-size 64 \
  --out "${OUT_ROOT}/table4_quantized.json"
