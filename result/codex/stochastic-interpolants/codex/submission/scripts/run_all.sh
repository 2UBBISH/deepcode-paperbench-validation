#!/usr/bin/env bash
# Full reproduction pipeline.
#
# Steps 1-2 run on CPU in a few minutes and produce everything reported in
# Section 1 of README.md.  Steps 3-5 are the ImageNet runs of the paper
# (batch size 32, 200,000 steps each) and need GPUs; they are written so that
# they can be launched as-is on a machine with accelerators.
set -euo pipefail

echo "== 1. unit tests =="
python -m pytest tests -q

echo "== 2. CPU experiments (Figure 2, Section 3.3, Proposition 3.1) =="
python experiments/gmm_coupling.py --steps 4000 --out results
python experiments/transport_cost.py --out results
python experiments/inpainting_demo.py --steps 200 --out results
python experiments/inpainting_demo.py --task super_resolution --steps 150 --out results

echo "== 3. ImageNet in-painting (256 and 512) =="
bash scripts/train_inpainting_256_coupled.sh
bash scripts/train_inpainting_256_baseline.sh
bash scripts/train_inpainting_512_coupled.sh

echo "== 4. ImageNet super-resolution =="
bash scripts/train_superres_64_256_coupled.sh
bash scripts/train_superres_256_512_coupled.sh

echo "== 5. FID-50k (Tables 2 and 3) =="
bash scripts/evaluate_fid_inpainting.sh
bash scripts/evaluate_fid_inpainting.sh configs/inpainting_256_baseline.yaml \
    runs/inpainting_256_baseline/checkpoint_last.pt
bash scripts/evaluate_fid_superres.sh
