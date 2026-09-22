#!/usr/bin/env bash
# Appendix C, Table 7: tuning the initial learning rate (alpha) and the decay
# (gamma) of SMM with ViT-B/32 on CIFAR-10.  The paper's grid is
#   alpha in {0.1, 0.01, 0.001, 0.0001} x gamma in {1, 0.1}
# and the best setting (alpha = 0.001, gamma = 1) is the unified setting used for
# every dataset in Table 2 (it is the default of configs/vitb32.yaml).
#
#   bash scripts/lr_search_vit.sh            # full grid, 200 epochs (expensive)
#   EPOCHS=5 bash scripts/lr_search_vit.sh   # quick grid for pipeline checks
#
# Table 8 (UCF101) shows that a dataset-specific setting can be better than the
# unified one; reproduce those two rows with:
#   python -m smm.main --config configs/vitb32.yaml --dataset ucf101 --method smm \
#       --lr-delta 0.01 --gamma-delta 0.1 --lr-mask 0.01 --gamma-mask 0.1
set -euo pipefail

EPOCHS="${EPOCHS:-200}"
DATASET="${DATASET:-cifar10}"
OUT="${OUT:-runs_lr_search}"

for alpha in 0.1 0.01 0.001 0.0001; do
  for gamma in 1 0.1; do
    python -m smm.main --config configs/vitb32.yaml --dataset "$DATASET" --method smm \
      --epochs "$EPOCHS" --lr-delta "$alpha" --gamma-delta "$gamma" \
      --lr-mask "$alpha" --gamma-mask "$gamma" --seed 0 \
      --output-dir "$OUT" --run-name "smm_alpha${alpha}_gamma${gamma}"
  done
done

python -m smm.aggregate --runs "$OUT" --table 1
