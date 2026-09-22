#!/usr/bin/env bash
# End-to-end check on synthetic images: train a few steps, sample, run tests.
set -euo pipefail
python -m pytest tests -q
python -m si_couplings.train --config configs/smoke_synthetic.yaml --max-steps 20 --out-dir /tmp/si_smoke
python -m si_couplings.sample --config configs/smoke_synthetic.yaml \
    --checkpoint /tmp/si_smoke/checkpoint_last.pt --task inpainting --steps 20 --method euler \
    --n 4 --out /tmp/si_smoke/samples.pt
