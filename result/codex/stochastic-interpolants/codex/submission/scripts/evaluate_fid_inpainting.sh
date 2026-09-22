#!/usr/bin/env bash
# FID-50k for the in-painting model (Table 2)
set -euo pipefail
CONFIG=${1:-configs/inpainting_256_coupled.yaml}
CKPT=${2:-runs/inpainting_256_coupled/checkpoint_last.pt}
python -m si_couplings.fid --config "$CONFIG" --checkpoint "$CKPT" --n 50000 \
    --out results/fid_inpainting.json
