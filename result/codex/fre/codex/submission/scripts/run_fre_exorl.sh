#!/usr/bin/env bash
# Train FRE on the ExORL RND datasets (walker and cheetah).
#
# ExORL / Kitchen use 1M encoder steps and 1M policy steps (Appendix A); the
# trainer applies that schedule automatically for these domains.

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"

for domain in walker cheetah; do
  for seed in $SEEDS; do
    python train_fre.py \
      --domain "$domain" \
      --prior FRE-all \
      --seed "$seed" \
      --device "$DEVICE" \
      --run-name "${domain}-FRE-all-s${seed}"
  done
  # Section 5.4: augment the prior with specific-velocity rewards.
  for seed in $SEEDS; do
    python train_fre.py \
      --domain "$domain" \
      --prior FRE-hint \
      --use-hint-priors \
      --seed "$seed" \
      --device "$DEVICE" \
      --run-name "${domain}-FRE-hint-s${seed}"
  done
done
