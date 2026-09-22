#!/usr/bin/env bash
# Full reproduction: grid search -> analysis -> spectra -> fine-tuning.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pinn.cli grid --outdir runs
python3 -m pinn.cli analyze --outdir runs
for pde in convection reaction wave; do
  python3 -m pinn.cli spectra --pde "$pde" --select-best --component --outdir runs
  python3 -m pinn.cli finetune --pde "$pde" --select-best --outdir runs
done
