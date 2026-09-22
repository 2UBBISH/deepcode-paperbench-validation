#!/usr/bin/env bash
# Table 10 - FOA on ResNet-50 and VisionMamba (ImageNet-C gaussian noise, level 5).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "${STATS}" ] || stats

run table --table 10 \
  --data-root "${DATA_ROOT}/imagenet-c" \
  --stats "${STATS}" \
  --backbones "${BACKBONES:-resnet50,visionmamba}" \
  --methods "${METHODS:-NoAdapt,BNAdapt,TENT,SAR,FOA,FOA-dagger}" \
  --out "${OUT_ROOT}/table10_resnet_vim.json"
