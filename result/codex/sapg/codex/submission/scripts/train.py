#!/usr/bin/env python3
"""Train SAPG or one of its baselines.

Examples
--------
Smoke run on the CPU toy suite::

    python scripts/train.py --config sapg/configs/toy_sapg.yaml \
        --logdir runs/toy_sapg --seed 0

Full-scale run of the paper (needs IsaacGym + a GPU)::

    python scripts/train.py --config sapg/configs/sapg_allegrokuka_reorientation.yaml \
        --logdir runs/sapg_reorientation --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms import make_trainer  # noqa: E402
from sapg.envs import make_env  # noqa: E402
from sapg.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAPG / PPO / DexPBT / PQL")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument("--logdir", default=None, help="output directory")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu or cuda")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--max-env-steps", type=float, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--num-policies", type=int, default=None)
    parser.add_argument("--algo", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="extra config override, e.g. --set algo.entropy_coef=0.005",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)

    if args.seed is not None:
        cfg.set_path("seed", args.seed)
    if args.num_iterations is not None:
        cfg.set_path("algo.num_iterations", args.num_iterations)
    if args.max_env_steps is not None:
        cfg.set_path("algo.max_env_steps", args.max_env_steps)
    if args.num_envs is not None:
        cfg.set_path("env.num_envs", args.num_envs)
    if args.num_policies is not None:
        cfg.set_path("algo.num_policies", args.num_policies)
    if args.algo is not None:
        cfg.set_path("algo.name", args.algo)
    if args.quiet:
        cfg.set_path("verbose", False)

    device = args.device or cfg.get("device", "cpu")
    logdir = args.logdir or os.path.join(
        cfg.get_path("logging.logdir", "runs"),
        f"{cfg.get_path('algo.name')}_{cfg.get_path('env.name')}_seed{cfg.get_path('seed', 0)}",
    )

    env = make_env(cfg, device=device)
    trainer = make_trainer(cfg, env, device=device, logdir=logdir)
    if args.resume:
        trainer.load(args.resume)
        print(f"[train] resumed from {args.resume} (iteration {trainer.iteration})", flush=True)
    print(f"[train] algo={cfg.get_path('algo.name')} env={cfg.get_path('env.name')} "
          f"num_envs={env.num_envs} policies={trainer.num_policies} logdir={logdir}", flush=True)
    trainer.train()
    print(f"[train] done after {trainer.env_steps:,} environment steps", flush=True)


if __name__ == "__main__":
    main()
