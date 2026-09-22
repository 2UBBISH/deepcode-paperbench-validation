#!/usr/bin/env bash
# Section 6: the full hyper-parameter grid (Table 1, Figures 2 and 8).
# 3 PDEs x 4 widths x 5 seeds x (5 Adam learning rates x (Adam + 3 Adam+L-BFGS)
# + 1 L-BFGS) = 1260 runs of 41000 iterations each.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pinn.cli grid --outdir runs "$@"
python3 -m pinn.cli analyze --outdir runs
