#!/usr/bin/env bash
# In-painting, ImageNet-256x256 with the data-dependent coupling (Table 2, "Ours")
set -euo pipefail
python -m si_couplings.train --config configs/inpainting_256_coupled.yaml "$@"
