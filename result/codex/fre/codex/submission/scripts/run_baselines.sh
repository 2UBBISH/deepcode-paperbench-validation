#!/usr/bin/env bash
# Train the GC-IQL, GC-BC and OPAL baselines used in Table 1.

set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-1000000}"

for domain in antmaze walker cheetah kitchen; do
  for seed in $SEEDS; do
    python train_baseline.py --algo gc_iql --domain "$domain" --seed "$seed" --steps "$STEPS" --device "$DEVICE"
    python train_baseline.py --algo gc_bc  --domain "$domain" --seed "$seed" --steps "$STEPS" --device "$DEVICE"
    python train_baseline.py --algo opal   --domain "$domain" --seed "$seed" --steps "$STEPS" --device "$DEVICE"
  done
done

# FB and SF are trained/evaluated with facebookresearch/controllable_agent.
bash scripts/run_fb_sf.sh "${WORKDIR:-third_party}"
