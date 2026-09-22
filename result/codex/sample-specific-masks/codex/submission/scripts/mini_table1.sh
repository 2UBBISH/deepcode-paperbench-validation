#!/usr/bin/env bash
# Fast sanity check of the paper's main trend on a small subset of CIFAR-10.
#
# The real Table 1 needs 200 epochs on the full training set of every dataset
# (a few hundred A100-hours overall); this script instead trains every method
# on a small subset for a few epochs so that the *relative* ordering
# (SMM > shared masks) can be checked in minutes, on CPU or a single GPU.
#
#   bash scripts/mini_table1.sh                # 2000 train / 1000 test images, 8 epochs
#   TRAIN=8000 EPOCHS=20 IMAGE_SIZE=224 bash scripts/mini_table1.sh
#
# Note: with so few samples SMM has much less data than the shared-mask
# baselines to estimate its sample-specific masks, so the gap is smaller than
# in the paper; the script is a pipeline sanity check, not a reproduction of
# the reported numbers.
set -euo pipefail

TRAIN="${TRAIN:-2000}"
TEST="${TEST:-1000}"
EPOCHS="${EPOCHS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-64}"
DATASET="${DATASET:-cifar10}"
OUT="${OUT:-runs_mini}"
BACKBONE="${BACKBONE:-resnet18}"
CONFIG="${CONFIG:-configs/resnet18.yaml}"

for method in pad narrow medium full smm; do
  python -m smm.main --config "$CONFIG" --dataset "$DATASET" --method "$method" \
    --image-size "$IMAGE_SIZE" --epochs "$EPOCHS" --train-subset "$TRAIN" \
    --test-subset "$TEST" --eval-every 1 --seed 0 --output-dir "$OUT" \
    --run-name "${method}_mini"
done

python -m smm.aggregate --runs "$OUT" --table 1
