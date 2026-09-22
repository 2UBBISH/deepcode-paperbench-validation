#!/usr/bin/env python3
"""Evaluate a trained checkpoint.

Example::

    python scripts/eval.py --config sapg/configs/toy_sapg.yaml \
        --checkpoint runs/sapg_multimodal_collect_seed0/checkpoints/final.pt \
        --episodes 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.algorithms import make_trainer  # noqa: E402
from sapg.envs import make_env  # noqa: E402
from sapg.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SAPG/PPO/DexPBT/PQL checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2048)
    parser.add_argument("--policy-id", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    env = make_env(cfg, device=args.device)
    trainer = make_trainer(cfg, env, device=args.device, logdir=os.path.dirname(args.checkpoint))
    trainer.load(args.checkpoint)
    stats = trainer.evaluate_policy(
        args.policy_id, num_episodes=args.episodes, max_steps=args.max_steps
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
