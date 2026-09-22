#!/usr/bin/env bash
# In-painting, ImageNet-512x512 with the data-dependent coupling (Figure 3)
set -euo pipefail
python -m si_couplings.train --config configs/inpainting_512_coupled.yaml "$@"
