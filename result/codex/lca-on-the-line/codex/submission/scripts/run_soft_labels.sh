#!/usr/bin/env bash
# Step 4 (Section 4.3.2 / Tables 5, 6): linear probing with LCA soft labels.
#
#   ./scripts/run_soft_labels.sh resnet18 WordNet
#   ./scripts/run_soft_labels.sh resnet18 --source-model resnet18   # latent
set -euo pipefail

BACKBONE=${1:-resnet18}
HIERARCHY=${2:-WordNet}
DATA_ROOT=${DATA_ROOT:-./data/datasets}
DEVICE=${DEVICE:-cuda}

python -m lca_on_the_line.experiments_soft_labels \
  --backbone "$BACKBONE" \
  --hierarchy "$HIERARCHY" \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE" \
  --feature-cache "${FEATURE_CACHE:-results/class_features}"
