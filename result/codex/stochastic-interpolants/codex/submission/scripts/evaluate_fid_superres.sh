#!/usr/bin/env bash
# FID-50k for the super-resolution model (Table 3)
set -euo pipefail
CONFIG=${1:-configs/superres_64_256_coupled.yaml}
CKPT=${2:-runs/superres_64_256_coupled/checkpoint_last.pt}
python -m si_couplings.fid --config "$CONFIG" --checkpoint "$CKPT" --n 50000 \
    --out results/fid_superres.json
