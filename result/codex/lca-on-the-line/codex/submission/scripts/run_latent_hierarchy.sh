#!/usr/bin/env bash
# Step 3 (Section 4.3.1 / Table 4): build the 75 latent hierarchies with
# K-means and correlate their ID LCA with OOD Top-1.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-./data/datasets}
OUT_DIR=${OUT_DIR:-results}
DEVICE=${DEVICE:-cuda}

python -m lca_on_the_line.experiments_latent \
  --data-root "$DATA_ROOT" \
  --results-dir "$OUT_DIR" \
  --feature-cache "${FEATURE_CACHE:-results/class_features}" \
  --per-class "${PER_CLASS:-20}" \
  --device "$DEVICE"
