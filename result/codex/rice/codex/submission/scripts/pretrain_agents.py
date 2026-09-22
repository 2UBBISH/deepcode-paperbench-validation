#!/usr/bin/env python3
"""Pre-train the target agents of the paper (the bottlenecked DRL agents).

Usage
-----
    python scripts/pretrain_agents.py --envs Hopper Walker2d Reacher HalfCheetah \
        --steps 300000 --seeds 0 1 2

The checkpoints are written to ``checkpoints/agents/<env>_seed<k>.pt`` together
with the reward of the pre-trained agent ("No Refine" column of Table 1).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rice.envs.registry import ENV_SPECS, make_env  # noqa: E402
from rice.experiments.common import (  # noqa: E402
    ckpt_path,
    close_env,
    ensure_dir,
    evaluate_agent,
    save_json,
)
from rice.training import PretrainConfig, train_ppo_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", default=["Hopper"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-root", default=".")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    summary = {}
    for env_name in args.envs:
        spec = ENV_SPECS[env_name]
        steps = args.steps or spec.refine_steps
        for seed in args.seeds:
            print("[pretrain] {} seed {} ({} steps)".format(env_name, seed, steps), flush=True)
            if args.algo == "ppo":
                out = train_ppo_policy(
                    env_name,
                    PretrainConfig(
                        total_steps=steps,
                        seed=seed,
                        normalize_obs=env_name in ("Walker2d", "HalfCheetah"),
                    ),
                    device=args.device,
                    verbose=not args.quiet,
                )
                policy = out["policy"]
            else:
                from rice.training import train_sb3_agent

                model = train_sb3_agent(env_name, algo="sac", total_steps=steps, seed=seed)
                path = ckpt_path(
                    "{}_{}_seed{}.zip".format(env_name, "sac", seed),
                    kind="agents",
                    root=args.out_root,
                )
                ensure_dir(os.path.dirname(path))
                model.save(path)
                policy = None
            if policy is not None:
                path = ckpt_path(
                    "{}_seed{}.pt".format(env_name, seed), kind="agents", root=args.out_root
                )
                ensure_dir(os.path.dirname(path))
                policy.save(path)
            env = make_env(env_name, seed=1000 + seed)
            if policy is not None:
                metrics = evaluate_agent(env_name, policy, n_episodes=10, seed=1000 + seed)
            else:
                from rice.policies import SB3Policy
                from rice.training import evaluate_policy

                metrics = evaluate_policy(env, SB3Policy(model), n_episodes=10)
            close_env(env)
            summary.setdefault(env_name, {})["seed{}".format(seed)] = {
                "path": path,
                "no_refine": metrics,
            }
            print(
                "          no-refine reward: {:.2f} +/- {:.2f}".format(
                    metrics["mean_return"], metrics["std_return"]
                ),
                flush=True,
            )

    out_path = ckpt_path("pretrained_agents.json", kind="results", root=args.out_root)
    save_json(summary, out_path)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
