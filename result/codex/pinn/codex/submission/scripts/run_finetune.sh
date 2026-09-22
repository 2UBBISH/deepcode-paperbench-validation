#!/usr/bin/env bash
# Section 7 (Figures 1, 4, 5 and Tables 2, 3): NNCG and GD after Adam+L-BFGS.
set -euo pipefail
cd "$(dirname "$0")/.."
for pde in convection reaction wave; do
  python3 -m pinn.cli finetune --pde "$pde" --select-best \
      --nncg-iters 2000 --gd-iters 2000 --outdir runs "$@"
done
