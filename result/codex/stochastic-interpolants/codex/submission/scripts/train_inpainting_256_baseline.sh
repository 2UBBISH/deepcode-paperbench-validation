#!/usr/bin/env bash
# In-painting, ImageNet-256x256 with the independent coupling (Table 2, baseline)
set -euo pipefail
python -m si_couplings.train --config configs/inpainting_256_baseline.yaml "$@"
