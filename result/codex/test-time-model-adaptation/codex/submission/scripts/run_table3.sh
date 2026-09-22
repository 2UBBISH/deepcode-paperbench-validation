#!/usr/bin/env bash
# Table 3 - ImageNet-R / ImageNet-V2 / ImageNet-Sketch with ViT-Base.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 3 \
  --data-root "${DATA_ROOT}" \
  --stats "${STATS}" \
  --methods "${METHODS:-NoAdapt,LAME,T3A,TENT,CoTTA,SAR,FOA}" \
  --batch-size 64 \
  --out "${OUT_ROOT}/table3_ood_benchmarks.json"
