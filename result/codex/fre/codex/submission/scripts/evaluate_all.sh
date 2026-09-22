#!/usr/bin/env bash
# Evaluate trained agents and aggregate the Table 1 / Table 4 numbers.
#
# Runs `evaluate_fre.py` for each (domain, prior, seed) checkpoint produced by
# the training scripts, then averages across seeds with the across-seed standard
# deviation reported in the paper.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cpu}"
RUNS_DIR="${RUNS_DIR:-runs}"
OUT_DIR="${OUT_DIR:-eval}"
SEEDS="${SEEDS:-0 1 2 3 4}"

mkdir -p "$OUT_DIR"

evaluate_family() {
  local domain="$1"; shift
  local prior="$1"; shift
  for seed in $SEEDS; do
    local ckpt="$RUNS_DIR/${domain}-${prior}-s${seed}/policy.pt"
    if [ ! -f "$ckpt" ]; then
      echo "skip (missing): $ckpt"
      continue
    fi
    python evaluate_fre.py \
      --domain "$domain" \
      --checkpoint "$ckpt" \
      --device "$DEVICE" \
      --seed "$seed" \
      --output "$OUT_DIR/${domain}-${prior}-s${seed}.json"
  done
}

for prior in FRE-all FRE-goals FRE-lin FRE-mlp FRE-lin-mlp FRE-goal-mlp FRE-goal-lin FRE-hint; do
  evaluate_family antmaze "$prior"
done
for domain in walker cheetah; do
  evaluate_family "$domain" FRE-all
  evaluate_family "$domain" FRE-hint
done
evaluate_family kitchen FRE-all

python scripts/aggregate_results.py --input-dir "$OUT_DIR" --output "$OUT_DIR/table1.json"
