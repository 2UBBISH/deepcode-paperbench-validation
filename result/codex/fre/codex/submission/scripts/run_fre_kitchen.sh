#!/usr/bin/env bash
# Train FRE on the D4RL Kitchen dataset (1M encoder + 1M policy steps).

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"

for seed in $SEEDS; do
  python train_fre.py \
    --domain kitchen \
    --prior FRE-all \
    --seed "$seed" \
    --device "$DEVICE" \
    --run-name "kitchen-FRE-all-s${seed}"
done
