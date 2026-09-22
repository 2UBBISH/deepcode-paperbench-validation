#!/usr/bin/env bash
# Step 2: turn the cached results into Tables 1-3 and Figures 1/5/9.
set -euo pipefail

python scripts/reproduce_tables.py \
  --results-dir "${OUT_DIR:-results}" \
  --out-dir "${TABLE_DIR:-tables}"
