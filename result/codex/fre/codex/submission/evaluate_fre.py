#!/usr/bin/env python
"""Zero-shot evaluation of a pre-trained FRE agent (Tables 1 and 4).

    python evaluate_fre.py --domain antmaze --checkpoint runs/antmaze-FRE-all-s0/policy.pt
    python evaluate_fre.py --domain antmaze --checkpoint ... --suites ant-goal-reaching,ant-directional
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fre.configs import EvalConfig
from fre.evaluate import FREPolicy, aggregate_seeds, evaluate_suite, save_results
from fre.experiment import build_dataset, build_suites, default_config, make_env_factory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot evaluation of FRE")
    p.add_argument("--domain", default="antmaze", choices=["antmaze", "walker", "cheetah", "kitchen"])
    p.add_argument("--checkpoint", required=True, help="path to policy.pt (or encoder.pt)")
    p.add_argument("--suites", default=None, help="comma-separated suite names (default: all)")
    p.add_argument("--num-episodes", type=int, default=20)
    p.add_argument("--num-encode-pairs", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="eval/results.json")
    p.add_argument("--dry-run", action="store_true", help="encode tasks only, no environment rollouts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_config = EvalConfig(num_episodes=args.num_episodes, num_encode_pairs=args.num_encode_pairs)
    config = default_config(args.domain)
    dataset = build_dataset(config)
    suites = build_suites(config, dataset)
    if args.suites:
        wanted = set(args.suites.split(","))
        suites = [s for s in suites if s.name in wanted]

    policy = FREPolicy.load(args.checkpoint, dataset, device=args.device)

    if args.dry_run:
        results = {}
        for suite in suites:
            results[suite.name] = [
                {"task": t.name, "latent_norm": float(policy.encode_task(t, dataset, eval_config.num_encode_pairs).__abs__().sum())}
                for t in suite.tasks
            ]
        save_results(results, args.output)
        print(json.dumps(results, indent=2, default=float))
        return

    env_factory = make_env_factory(config)
    results = {}
    for suite in suites:
        results[suite.name] = evaluate_suite(
            env_factory, suite, policy, dataset, eval_config, seed=args.seed
        )
        print(f"{suite.name}: {results[suite.name]['mean']:.2f}")
    save_results(results, args.output)


if __name__ == "__main__":
    main()
