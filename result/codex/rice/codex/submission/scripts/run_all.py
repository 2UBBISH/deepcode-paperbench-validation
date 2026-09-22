#!/usr/bin/env python3
"""Run the full reproduction pipeline (intended for the GPU machine).

    python scripts/run_all.py --envs Hopper Walker2d Reacher HalfCheetah \
        --sparse-envs SparseHopper SparseHalfCheetah \
        --seeds 0 1 2

The pipeline of one application is

    1. pre-train the target agent (PPO)                       -> checkpoints/agents
    2. train the mask networks (ours and StateMask)           -> checkpoints/masks
    3. Experiment I  : fidelity + mask training cost           -> results/*_fidelity.json
    4. Experiment II/III: Table 1 / Figure 2                   -> results/*_refining.json
    5. Experiment V  : p / lambda / alpha sensitivity          -> results/*_p_lambda.json

Experiment IV (SAC + GAIL) is run separately with ``scripts/run_experiment4_sac.py``
because it depends on the SAC implementation of Stable-Baselines3.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import ENV_SPECS  # noqa: E402
from rice.experiments.common import ckpt_path, save_json  # noqa: E402
from rice.experiments.explanation import run_experiment_i, train_explanation  # noqa: E402
from rice.experiments.hyperparams import run_p_lambda_grid  # noqa: E402
from rice.experiments.refining import run_experiment_ii_and_iii  # noqa: E402
from rice.training import PretrainConfig, train_ppo_policy  # noqa: E402


def scale_budgets(factor: float, floor_samples: int = 1000, floor_steps: int = 1500):
    """Scale every step budget of the registry (useful for dry runs)."""
    for spec in ENV_SPECS.values():
        spec.mask_samples = max(floor_samples, int(spec.mask_samples * factor))
        spec.mask_iterations = max(
            4, min(spec.mask_iterations, max(4, spec.mask_samples // max(1, spec.horizon)))
        )
        spec.refine_steps = max(floor_steps, int(spec.refine_steps * factor))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=["Hopper"])
    parser.add_argument("--sparse-envs", nargs="+", default=[])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--pretrain-steps", type=int, default=None)
    parser.add_argument("--refine-steps", type=int, default=None)
    parser.add_argument("--skip-hyperparams", action="store_true")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every step budget of the registry (e.g. 0.01 for a dry run)",
    )
    parser.add_argument("--fidelity-trajectories", type=int, default=500)
    parser.add_argument("--fidelity-seeds", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.scale != 1.0:
        scale_budgets(args.scale)
        print("scaled all budgets by {}".format(args.scale), flush=True)

    for env_name in list(args.envs) + list(args.sparse_envs):
        spec = ENV_SPECS[env_name]
        agent_path = ckpt_path("{}_seed0.pt".format(env_name), kind="agents")
        for seed in args.seeds:
            path = ckpt_path("{}_seed{}.pt".format(env_name, seed), kind="agents")
            if os.path.exists(path):
                continue
            print("[1/5] pre-training {} (seed {})".format(env_name, seed), flush=True)
            out = train_ppo_policy(
                env_name,
                PretrainConfig(
                    total_steps=args.pretrain_steps or spec.refine_steps,
                    seed=seed,
                    normalize_obs=env_name in ("Walker2d", "HalfCheetah"),
                ),
                device=args.device,
            )
            from rice.experiments.common import ensure_dir

            ensure_dir(os.path.dirname(path))
            out["policy"].save(path)
            save_json(
                {"env": env_name, "seed": seed, "eval_history": out["eval_history"]},
                ckpt_path("{}_seed{}.json".format(env_name, seed), kind="results"),
            )
            if seed == 0:
                agent_path = path

        print("[2/5] mask networks for {}".format(env_name), flush=True)
        for method in ("ours", "statemask"):
            mask_path = ckpt_path(
                "{}_{}_seed0.pt".format(env_name, method), kind="masks"
            )
            if not os.path.exists(mask_path):
                train_explanation(
                    env_name, agent_path, method=method, seed=0, device=args.device
                )

        print("[3/5] Experiment I (fidelity / efficiency) on {}".format(env_name), flush=True)
        run_experiment_i(
            env_name,
            agent_path,
            explanation_paths={
                "ours": ckpt_path("{}_ours_seed0.pt".format(env_name), kind="masks"),
                "statemask": ckpt_path(
                    "{}_statemask_seed0.pt".format(env_name), kind="masks"
                ),
            },
            n_trajectories=args.fidelity_trajectories,
            n_seeds=args.fidelity_seeds,
            device=args.device,
        )

        print("[4/5] Experiments II + III on {}".format(env_name), flush=True)
        run_experiment_ii_and_iii(
            env_name,
            agent_path,
            {
                "ours": ckpt_path("{}_ours_seed0.pt".format(env_name), kind="masks"),
                "statemask": ckpt_path(
                    "{}_statemask_seed0.pt".format(env_name), kind="masks"
                ),
            },
            total_steps=args.refine_steps,
            seeds=args.seeds,
            device=args.device,
        )

        if env_name in args.envs and not args.skip_hyperparams:
            print("[5/5] Experiment V (p / lambda) on {}".format(env_name), flush=True)
            run_p_lambda_grid(
                env_name,
                agent_path,
                ckpt_path("{}_ours_seed0.pt".format(env_name), kind="masks"),
                total_steps=args.refine_steps,
                seeds=args.seeds[:1] if len(args.seeds) > 1 else args.seeds,
                device=args.device,
            )

    print("done. Run `python scripts/plot_results.py` to draw the figures.")


if __name__ == "__main__":
    main()
