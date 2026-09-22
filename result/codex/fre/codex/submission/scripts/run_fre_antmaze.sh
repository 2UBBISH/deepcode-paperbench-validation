#!/usr/bin/env bash
# Train every FRE variant used in the AntMaze experiments (Tables 1 and 4).
#
#   * FRE-all / FRE-goals / FRE-lin / FRE-mlp / FRE-lin-mlp /
#     FRE-goal-mlp / FRE-goal-lin  -> Section 5.3 scaling study (Table 4, Fig. 5)
#   * FRE-hint                    -> Section 5.4 domain-knowledge study (Fig. 6)
#
# Each agent is trained with 5 seeds, as reported in the paper.

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"
PRIORS="FRE-all FRE-goals FRE-lin FRE-mlp FRE-lin-mlp FRE-goal-mlp FRE-goal-lin"

for prior in $PRIORS; do
  for seed in $SEEDS; do
    python train_fre.py \
      --domain antmaze \
      --prior "$prior" \
      --seed "$seed" \
      --device "$DEVICE" \
      --run-name "antmaze-${prior}-s${seed}"
  done
done

# Section 5.4: augment the prior with unit-direction movement rewards.
for seed in $SEEDS; do
  python train_fre.py \
    --domain antmaze \
    --prior FRE-hint \
    --use-hint-priors \
    --hint-ratio 0.5 \
    --seed "$seed" \
    --device "$DEVICE" \
    --run-name "antmaze-FRE-hint-s${seed}"
done
