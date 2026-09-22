#!/usr/bin/env bash
# Super-resolution 256x256 -> 512x512 with the data-dependent coupling (Figure 6)
set -euo pipefail
python -m si_couplings.train --config configs/superres_256_512_coupled.yaml "$@"
