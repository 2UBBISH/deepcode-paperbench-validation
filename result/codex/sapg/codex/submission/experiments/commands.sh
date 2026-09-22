#!/usr/bin/env bash
set -euo pipefail

python scripts/train.py --config sapg/configs/sapg_allegrokuka_regrasping.yaml --logdir runs/sapg_regrasping_seed0 --seed 0
python scripts/train.py --config sapg/configs/sapg_allegrokuka_regrasping.yaml --logdir runs/sapg_regrasping_seed1 --seed 1
python scripts/train.py --config sapg/configs/sapg_allegrokuka_regrasping.yaml --logdir runs/sapg_regrasping_seed2 --seed 2
python scripts/train.py --config sapg/configs/sapg_allegrokuka_regrasping.yaml --logdir runs/sapg_regrasping_seed3 --seed 3
python scripts/train.py --config sapg/configs/sapg_allegrokuka_regrasping.yaml --logdir runs/sapg_regrasping_seed4 --seed 4
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_regrasping_seed0 --seed 0
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_regrasping_seed1 --seed 1
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_regrasping_seed2 --seed 2
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_regrasping_seed3 --seed 3
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_regrasping_seed4 --seed 4
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_regrasping_seed0 --seed 0
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_regrasping_seed1 --seed 1
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_regrasping_seed2 --seed 2
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_regrasping_seed3 --seed 3
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_regrasping_seed4 --seed 4
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_regrasping_seed0 --seed 0
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_regrasping_seed1 --seed 1
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_regrasping_seed2 --seed 2
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_regrasping_seed3 --seed 3
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_regrasping_seed4 --seed 4
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_regrasping_seed0 --seed 0 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_regrasping_seed1 --seed 1 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_regrasping_seed2 --seed 2 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_regrasping_seed3 --seed 3 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_regrasping_seed4 --seed 4 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_regrasping_seed0 --seed 0 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_regrasping_seed1 --seed 1 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_regrasping_seed2 --seed 2 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_regrasping_seed3 --seed 3 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_regrasping_seed4 --seed 4 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_regrasping_seed0 --seed 0 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_regrasping_seed1 --seed 1 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_regrasping_seed2 --seed 2 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_regrasping_seed3 --seed 3 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_regrasping_seed4 --seed 4 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_regrasping_seed0 --seed 0 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_regrasping_seed1 --seed 1 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_regrasping_seed2 --seed 2 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_regrasping_seed3 --seed 3 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_regrasping_seed4 --seed 4 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_regrasping_seed0 --seed 0 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_regrasping_seed1 --seed 1 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_regrasping_seed2 --seed 2 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_regrasping_seed3 --seed 3 --set env.name=regrasping
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_regrasping_seed4 --seed 4 --set env.name=regrasping
python scripts/train.py --config sapg/configs/sapg_allegrokuka_throw.yaml --logdir runs/sapg_throw_seed0 --seed 0
python scripts/train.py --config sapg/configs/sapg_allegrokuka_throw.yaml --logdir runs/sapg_throw_seed1 --seed 1
python scripts/train.py --config sapg/configs/sapg_allegrokuka_throw.yaml --logdir runs/sapg_throw_seed2 --seed 2
python scripts/train.py --config sapg/configs/sapg_allegrokuka_throw.yaml --logdir runs/sapg_throw_seed3 --seed 3
python scripts/train.py --config sapg/configs/sapg_allegrokuka_throw.yaml --logdir runs/sapg_throw_seed4 --seed 4
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_throw_seed0 --seed 0
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_throw_seed1 --seed 1
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_throw_seed2 --seed 2
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_throw_seed3 --seed 3
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_throw_seed4 --seed 4
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_throw_seed0 --seed 0
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_throw_seed1 --seed 1
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_throw_seed2 --seed 2
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_throw_seed3 --seed 3
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_throw_seed4 --seed 4
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_throw_seed0 --seed 0
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_throw_seed1 --seed 1
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_throw_seed2 --seed 2
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_throw_seed3 --seed 3
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_throw_seed4 --seed 4
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_throw_seed0 --seed 0 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_throw_seed1 --seed 1 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_throw_seed2 --seed 2 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_throw_seed3 --seed 3 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_throw_seed4 --seed 4 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_throw_seed0 --seed 0 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_throw_seed1 --seed 1 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_throw_seed2 --seed 2 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_throw_seed3 --seed 3 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_throw_seed4 --seed 4 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_throw_seed0 --seed 0 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_throw_seed1 --seed 1 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_throw_seed2 --seed 2 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_throw_seed3 --seed 3 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_throw_seed4 --seed 4 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_throw_seed0 --seed 0 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_throw_seed1 --seed 1 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_throw_seed2 --seed 2 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_throw_seed3 --seed 3 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_throw_seed4 --seed 4 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_throw_seed0 --seed 0 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_throw_seed1 --seed 1 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_throw_seed2 --seed 2 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_throw_seed3 --seed 3 --set env.name=throw
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_throw_seed4 --seed 4 --set env.name=throw
python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml --logdir runs/sapg_reorientation_seed0 --seed 0
python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml --logdir runs/sapg_reorientation_seed1 --seed 1
python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml --logdir runs/sapg_reorientation_seed2 --seed 2
python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml --logdir runs/sapg_reorientation_seed3 --seed 3
python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml --logdir runs/sapg_reorientation_seed4 --seed 4
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_reorientation_seed0 --seed 0
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_reorientation_seed1 --seed 1
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_reorientation_seed2 --seed 2
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_reorientation_seed3 --seed 3
python scripts/train.py --config sapg/configs/ppo_allegrokuka.yaml --logdir runs/ppo_reorientation_seed4 --seed 4
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_reorientation_seed0 --seed 0
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_reorientation_seed1 --seed 1
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_reorientation_seed2 --seed 2
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_reorientation_seed3 --seed 3
python scripts/train.py --config sapg/configs/pbt_allegrokuka.yaml --logdir runs/pbt_reorientation_seed4 --seed 4
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_reorientation_seed0 --seed 0
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_reorientation_seed1 --seed 1
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_reorientation_seed2 --seed 2
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_reorientation_seed3 --seed 3
python scripts/train.py --config sapg/configs/pql_allegrokuka.yaml --logdir runs/pql_reorientation_seed4 --seed 4
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_reorientation_seed0 --seed 0 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_reorientation_seed1 --seed 1 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_reorientation_seed2 --seed 2 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_reorientation_seed3 --seed 3 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_reorientation_seed4 --seed 4 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_reorientation_seed0 --seed 0 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_reorientation_seed1 --seed 1 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_reorientation_seed2 --seed 2 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_reorientation_seed3 --seed 3 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_symmetric.yaml --logdir runs/sapg_symmetric_reorientation_seed4 --seed 4 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_reorientation_seed0 --seed 0 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_reorientation_seed1 --seed 1 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_reorientation_seed2 --seed 2 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_reorientation_seed3 --seed 3 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_reorientation_seed4 --seed 4 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_reorientation_seed0 --seed 0 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_reorientation_seed1 --seed 1 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_reorientation_seed2 --seed 2 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_reorientation_seed3 --seed 3 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_reorientation_seed4 --seed 4 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_reorientation_seed0 --seed 0 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_reorientation_seed1 --seed 1 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_reorientation_seed2 --seed 2 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_reorientation_seed3 --seed 3 --set env.name=reorientation
python scripts/train.py --config sapg/configs/ablations/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_reorientation_seed4 --seed 4 --set env.name=reorientation
python scripts/train.py --config sapg/configs/sapg_shadow_hand.yaml --logdir runs/sapg_shadow_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/sapg_shadow_hand.yaml --logdir runs/sapg_shadow_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/sapg_shadow_hand.yaml --logdir runs/sapg_shadow_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/sapg_shadow_hand.yaml --logdir runs/sapg_shadow_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/sapg_shadow_hand.yaml --logdir runs/sapg_shadow_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_shadow_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_shadow_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_shadow_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_shadow_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_shadow_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_shadow_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_shadow_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_shadow_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_shadow_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_shadow_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_shadow_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_shadow_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_shadow_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_shadow_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_shadow_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_shadow_hand_seed0 --seed 0 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_shadow_hand_seed1 --seed 1 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_shadow_hand_seed2 --seed 2 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_shadow_hand_seed3 --seed 3 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_shadow_hand_seed4 --seed 4 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_shadow_hand_seed0 --seed 0 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_shadow_hand_seed1 --seed 1 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_shadow_hand_seed2 --seed 2 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_shadow_hand_seed3 --seed 3 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_shadow_hand_seed4 --seed 4 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_shadow_hand_seed0 --seed 0 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_shadow_hand_seed1 --seed 1 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_shadow_hand_seed2 --seed 2 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_shadow_hand_seed3 --seed 3 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_shadow_hand_seed4 --seed 4 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_shadow_hand_seed0 --seed 0 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_shadow_hand_seed1 --seed 1 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_shadow_hand_seed2 --seed 2 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_shadow_hand_seed3 --seed 3 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_shadow_hand_seed4 --seed 4 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_shadow_hand_seed0 --seed 0 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_shadow_hand_seed1 --seed 1 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_shadow_hand_seed2 --seed 2 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_shadow_hand_seed3 --seed 3 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_shadow_hand_seed4 --seed 4 --set env.name=shadow_hand
python scripts/train.py --config sapg/configs/sapg_allegro_hand.yaml --logdir runs/sapg_allegro_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/sapg_allegro_hand.yaml --logdir runs/sapg_allegro_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/sapg_allegro_hand.yaml --logdir runs/sapg_allegro_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/sapg_allegro_hand.yaml --logdir runs/sapg_allegro_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/sapg_allegro_hand.yaml --logdir runs/sapg_allegro_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_allegro_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_allegro_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_allegro_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_allegro_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/ppo_inhand.yaml --logdir runs/ppo_allegro_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_allegro_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_allegro_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_allegro_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_allegro_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/pbt_inhand.yaml --logdir runs/pbt_allegro_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_allegro_hand_seed0 --seed 0
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_allegro_hand_seed1 --seed 1
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_allegro_hand_seed2 --seed 2
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_allegro_hand_seed3 --seed 3
python scripts/train.py --config sapg/configs/pql_inhand.yaml --logdir runs/pql_allegro_hand_seed4 --seed 4
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_allegro_hand_seed0 --seed 0 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_allegro_hand_seed1 --seed 1 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_allegro_hand_seed2 --seed 2 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_allegro_hand_seed3 --seed 3 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_no_offpolicy.yaml --logdir runs/sapg_no_offpolicy_allegro_hand_seed4 --seed 4 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_allegro_hand_seed0 --seed 0 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_allegro_hand_seed1 --seed 1 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_allegro_hand_seed2 --seed 2 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_allegro_hand_seed3 --seed 3 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_symmetric.yaml --logdir runs/sapg_symmetric_allegro_hand_seed4 --seed 4 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_allegro_hand_seed0 --seed 0 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_allegro_hand_seed1 --seed 1 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_allegro_hand_seed2 --seed 2 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_allegro_hand_seed3 --seed 3 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_high_offpolicy_ratio.yaml --logdir runs/sapg_high_offpolicy_allegro_hand_seed4 --seed 4 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_allegro_hand_seed0 --seed 0 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_allegro_hand_seed1 --seed 1 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_allegro_hand_seed2 --seed 2 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_allegro_hand_seed3 --seed 3 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0003.yaml --logdir runs/sapg_entropy0.003_allegro_hand_seed4 --seed 4 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_allegro_hand_seed0 --seed 0 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_allegro_hand_seed1 --seed 1 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_allegro_hand_seed2 --seed 2 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_allegro_hand_seed3 --seed 3 --set env.name=allegro_hand
python scripts/train.py --config sapg/configs/ablations_inhand/sapg_entropy_0005.yaml --logdir runs/sapg_entropy0.005_allegro_hand_seed4 --seed 4 --set env.name=allegro_hand
