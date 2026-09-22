#!/usr/bin/env python3
"""End-to-end pipeline on a drastically reduced budget (for CI / sanity checks).

    python scripts/smoke_pipeline.py --env SelfishMining
    python scripts/smoke_pipeline.py --env Hopper --factor 0.01

The script scales down every step budget of the registry, then runs
pre-training → mask training (ours + StateMask) → Experiment I →
Experiments II/III → a single point of Experiment V → plotting, so that the
whole reproduction path is exercised in a couple of minutes on CPU.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import ENV_SPECS  # noqa: E402
from rice.experiments.common import ckpt_path, ensure_dir  # noqa: E402
from rice.experiments.explanation import run_experiment_i  # noqa: E402
from rice.experiments.hyperparams import run_p_lambda_grid  # noqa: E402
from rice.experiments.refining import run_experiment_ii_and_iii  # noqa: E402
from rice.training import PretrainConfig, train_ppo_policy  # noqa: E402


def scale_registry(factor: float, floor_samples: int = 1000, floor_steps: int = 1500):
    for spec in ENV_SPECS.values():
        spec.mask_samples = max(floor_samples, int(spec.mask_samples * factor))
        spec.mask_iterations = max(
            4, min(spec.mask_iterations, max(4, spec.mask_samples // max(1, spec.horizon)))
        )
        spec.refine_steps = max(floor_steps, int(spec.refine_steps * factor))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="SelfishMining")
    parser.add_argument("--factor", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    scale_registry(args.factor)
    env_name = args.env
    spec = ENV_SPECS[env_name]
    print(
        "reduced budget: mask {} samples / {} iterations, refining {} steps".format(
            spec.mask_samples, spec.mask_iterations, spec.refine_steps
        )
    )
    t0 = time.time()

    agent_path = ckpt_path("{}_seed0.pt".format(env_name), kind="agents")
    if not os.path.exists(agent_path):
        out = train_ppo_policy(
            env_name,
            PretrainConfig(
                total_steps=max(2000, spec.refine_steps),
                seed=0,
                n_steps=1000,
                eval_interval=max(1000, spec.refine_steps // 2),
                eval_episodes=2,
            ),
            device=args.device,
            verbose=True,
        )
        ensure_dir(os.path.dirname(agent_path))
        out["policy"].save(agent_path)

    from rice.experiments.explanation import train_explanation

    for method in ("ours", "statemask"):
        path = ckpt_path("{}_{}_seed0.pt".format(env_name, method), kind="masks")
        if not os.path.exists(path):
            train_explanation(env_name, agent_path, method=method, seed=0, verbose=True)

    print("=== Experiment I ===")
    run_experiment_i(
        env_name,
        agent_path,
        explanation_paths={
            "ours": ckpt_path("{}_ours_seed0.pt".format(env_name), kind="masks"),
            "statemask": ckpt_path(
                "{}_statemask_seed0.pt".format(env_name), kind="masks"
            ),
        },
        n_trajectories=5,
        n_seeds=1,
        device=args.device,
    )

    print("=== Experiments II + III ===")
    run_experiment_ii_and_iii(
        env_name,
        agent_path,
        {
            "ours": ckpt_path("{}_ours_seed0.pt".format(env_name), kind="masks"),
            "statemask": ckpt_path(
                "{}_statemask_seed0.pt".format(env_name), kind="masks"
            ),
        },
        seeds=[0],
        device=args.device,
    )

    print("=== Experiment V (single point) ===")
    run_p_lambda_grid(
        env_name,
        agent_path,
        ckpt_path("{}_ours_seed0.pt".format(env_name), kind="masks"),
        p_values=(0.0, 0.5),
        lambda_values=(0.0, 0.01),
        seeds=[0],
        device=args.device,
    )

    print("=== plotting ===")
    os.system(
        "{} {} --results results --out figures".format(
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_results.py"),
        )
    )
    print("smoke pipeline finished in {:.1f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
