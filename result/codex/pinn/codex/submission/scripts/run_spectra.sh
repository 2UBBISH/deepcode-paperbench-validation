#!/usr/bin/env bash
# Section 5 (Figures 3 and 7): spectral densities of the Hessian and of the
# L-BFGS-preconditioned Hessian, for the configuration with the smallest L2RE.
set -euo pipefail
cd "$(dirname "$0")/.."
for pde in convection reaction wave; do
  python3 -m pinn.cli spectra --pde "$pde" --select-best --component \
      --n-runs 1 --n-lanczos 100 --outdir runs "$@"
done
