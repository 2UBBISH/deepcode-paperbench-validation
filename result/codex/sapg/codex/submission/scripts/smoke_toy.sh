#!/usr/bin/env bash
# End-to-end smoke run of SAPG, PPO and the Sec. 6.3 ablations on the CPU toy
# suite, followed by the plots and the diversity analysis.
#
#   bash scripts/smoke_toy.sh [iterations] [num_envs]
set -euo pipefail

ITERATIONS="${1:-150}"
NUM_ENVS="${2:-768}"
ROOT="${SAPG_SMOKE_ROOT:-runs_toy}"
DEVICE="${SAPG_DEVICE:-cpu}"

run() {  # label config
  local label="$1"; shift
  echo "=== ${label} ==="
  python scripts/train.py --config "$1" --logdir "${ROOT}/${label}_multimodal_collect_seed0" \
      --seed 0 --num-iterations "${ITERATIONS}" --num-envs "${NUM_ENVS}" --device "${DEVICE}" \
      --set algo.record_states_interval=25 --set logging.use_tensorboard=false
}

run sapg               sapg/configs/toy_sapg.yaml
run ppo                sapg/configs/toy_ppo.yaml
run sapg_symmetric     sapg/configs/toy_ablations/sapg_symmetric.yaml
run sapg_no_offpolicy  sapg/configs/toy_ablations/sapg_no_offpolicy.yaml
run sapg_high_offpolicy sapg/configs/toy_ablations/sapg_high_offpolicy.yaml
run sapg_entropy0.003  sapg/configs/toy_ablations/sapg_entropy_0003.yaml
run sapg_entropy0.005  sapg/configs/toy_ablations/sapg_entropy_0005.yaml

python scripts/plot_results.py --root "${ROOT}" --out figures_toy --results results_toy \
    --tasks multimodal_collect \
    --methods sapg ppo \
    --ablations sapg sapg_no_offpolicy sapg_symmetric sapg_high_offpolicy sapg_entropy0.003 sapg_entropy0.005

python scripts/run_diversity_analysis.py \
    --run "sapg=${ROOT}/sapg_multimodal_collect_seed0" \
    --run "ppo=${ROOT}/ppo_multimodal_collect_seed0" \
    --random-config sapg/configs/toy_sapg.yaml \
    --num-random-steps 500 \
    --components 1 2 4 6 8 10 12 14 16 \
    --hidden-sizes 2 4 8 16 32 64 \
    --transitions 20000 \
    --out figures_toy/fig7_diversity.png \
    --out-mlp figures_toy/fig8_diversity.png

echo "smoke run finished; see ${ROOT}, figures_toy/ and results_toy/"
