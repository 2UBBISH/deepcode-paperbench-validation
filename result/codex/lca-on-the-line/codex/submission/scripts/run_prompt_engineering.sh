#!/usr/bin/env bash
# Step 5 (Section 4.3.3 / Table 14): taxonomy-alignment prompt engineering.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-./data/datasets}
MODEL=${MODEL:-CLIP_ViT-B_32}
DEVICE=${DEVICE:-cuda}

python -m lca_on_the_line.experiments_prompt \
  --model "$MODEL" \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE"
