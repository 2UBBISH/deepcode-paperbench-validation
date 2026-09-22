#!/usr/bin/env python
"""Train a FRE agent on an offline dataset (Algorithm 1).

Examples
--------
    python train_fre.py --domain antmaze --prior FRE-all --seed 0
    python train_fre.py --domain walker  --prior FRE-hint --use-hint-priors
    python train_fre.py --domain antmaze --prior FRE-goals --encoder-steps 1000 --policy-steps 1000

All hyperparameters default to Appendix A; ExORL / Kitchen runs automatically
use 1M encoder steps and 1M policy steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fre.configs import TrainConfig
from fre.experiment import build_dataset, build_domain_prior, default_config
from fre.training import FRETrainer, make_preprocess_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unsupervised pre-training of FRE")
    p.add_argument("--domain", default="antmaze", choices=["antmaze", "walker", "cheetah", "kitchen"])
    p.add_argument("--config", default=None,
                   help="JSON file of TrainConfig overrides (see configs/); CLI flags take precedence")
    p.add_argument("--prior", dest="prior_name", default="FRE-all",
                   help="FRE-all | FRE-goals | FRE-lin | FRE-mlp | FRE-lin-mlp | FRE-goal-mlp | FRE-goal-lin | FRE-hint")
    p.add_argument("--use-hint-priors", action="store_true")
    p.add_argument("--hint-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder-steps", type=int, default=150_000)
    p.add_argument("--policy-steps", type=int, default=850_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--beta", type=float, default=0.01)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--num-encode-pairs", type=int, default=32)
    p.add_argument("--num-decode-pairs", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--checkpoint-interval", type=int, default=50_000)
    p.add_argument("--exorl-data-dir", default=None)
    p.add_argument("--exorl-algo", default="rnd")
    p.add_argument("--no-discretize-xy", action="store_true")
    args = p.parse_args()
    # Record which arguments the user actually supplied, so that a --config
    # file is only overridden by explicit CLI flags.
    parser_defaults = {action.dest: action.default for action in p._actions}
    args._explicit = {
        dest: value
        for dest, value in vars(args).items()
        if dest != "_explicit" and value != parser_defaults.get(dest)
    }
    return args


def main() -> None:
    args = parse_args()
    explicit = dict(getattr(args, "_explicit", {}))
    explicit.pop("config", None)
    if explicit.pop("no_discretize_xy", False):
        explicit["discretize_xy"] = False
    overrides = {}
    if args.config:
        with open(args.config) as f:
            file_overrides = json.load(f)
        overrides.update(file_overrides)
    # Explicit CLI flags take precedence over the config file.
    overrides.update(explicit)
    domain = overrides.pop("domain", args.domain)
    config = default_config(domain, **overrides)
    print(f"[train_fre] config: {json.dumps(config.to_dict(), indent=2, default=str)}")
    dataset = build_dataset(config)
    prior = build_domain_prior(config, dataset)
    preprocess_fn = make_preprocess_fn(config, dataset.obs_dim)

    trainer = FRETrainer(config, dataset, prior, preprocess_fn)
    run_dir = os.path.join(config.output_dir, config.run_name or f"{config.domain}-{config.prior_name}-s{config.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    def log(entry):
        print(json.dumps(entry, default=float), flush=True)

    trainer.fit(log_fn=log)
    trainer.save_history()
    print(f"[train_fre] finished; checkpoints in {run_dir}")


if __name__ == "__main__":
    main()
