#!/usr/bin/env bash
# Fast end-to-end check on a tiny subset: verifies that the pipeline runs and
# that shapes/parameter counts are as expected. Requires the CIFAR-10 download
# (or CIFAR100) but only trains for one epoch.
set -euo pipefail

python - <<'PY'
import torch
from smm.mask_generator import MaskNet, DEFAULT_5_LAYER_CHANNELS, DEFAULT_6_LAYER_CHANNELS
from smm.theory import verify_shared_mask_inclusion, verify_watermark_inclusion, verify_sample_specific_inclusion

m5 = MaskNet((224, 224))
m6 = MaskNet((384, 384), hidden_channels=DEFAULT_6_LAYER_CHANNELS)
print("5-layer mask generator parameters:", sum(p.numel() for p in m5.parameters()), "(Table 4: 26,499)")
print("6-layer mask generator parameters:", sum(p.numel() for p in m6.parameters()), "(Table 4: 102,339)")
assert verify_shared_mask_inclusion()["included"] == 1.0
assert verify_watermark_inclusion()["included"] == 1.0
assert verify_sample_specific_inclusion()["included"] == 1.0
print("theory checks OK")
PY

python -m smm.main --config configs/resnet18.yaml --dataset cifar10 --method smm \
  --epochs 1 --eval-every 1 --num-workers 2 --device cpu --output-dir /tmp/smm_smoke

python -m smm.aggregate --runs /tmp/smm_smoke
