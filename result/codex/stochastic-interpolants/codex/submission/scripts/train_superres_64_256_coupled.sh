#!/usr/bin/env bash
# Super-resolution 64x64 -> 256x256 with the data-dependent coupling (Table 3, "Ours")
set -euo pipefail
python -m si_couplings.train --config configs/superres_64_256_coupled.yaml "$@"
