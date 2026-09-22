#!/usr/bin/env python3
"""NetHack: APPO fine-tuning with knowledge retention (Section 3-5, Table 4/5).

Example::

    # compute the Fisher matrix (10 000 NLD-AA batches) for EWC
    python scripts/train_nethack.py --compute-fisher

    # fine-tune with kickstarting (the best-performing method)
    python scripts/train_nethack.py --method ks --steps 500000000

    # fine-tune with behavioral cloning (expert trajectories from NLD-AA)
    python scripts/train_nethack.py --method bc --steps 500000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fpc.nethack.appo import APPO
from fpc.nethack.config import NetHackConfig
from fpc.nethack.dataset import NLDLoader, build_bc_buffer, to_observation
from fpc.nethack.env import make_env_factory
from fpc.nethack.eval import full_evaluation, level_visitation
from fpc.nethack.model import NetHackModel
from fpc.retention.base import RetentionConfig
from fpc.retention.fisher import DiagonalFisher


def build_config(args) -> NetHackConfig:
    config = NetHackConfig()
    config.total_steps = args.steps
    config.dataset_path = args.dataset
    config.device = args.device
    config.seed = args.seed
    config.log_dir = args.log_dir
    config.pretrained_checkpoint = args.pretrained
    if args.method == "ks":
        # Appendix B.1: loss scaled by 0.5 and decayed with 0.99998 per train step.
        config.retention = RetentionConfig(method="ks", coefficient=0.5, decay=0.99998)
    elif args.method == "bc":
        # Appendix B.1: auxiliary loss scaled by 2.0, no decay.
        config.retention = RetentionConfig(method="bc", coefficient=2.0, decay=1.0)
    elif args.method == "ewc":
        # Appendix B.1: regularization coefficient 2e6.
        config.retention = RetentionConfig(method="ewc", coefficient=2e6, decay=1.0)
    else:
        config.retention = RetentionConfig(method="none")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["none", "ks", "bc", "ewc"], default="ks")
    parser.add_argument("--steps", type=int, default=500_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset", default=None, help="path to the unpacked NLD-AA directory")
    parser.add_argument("--pretrained", default=None, help="checkpoint of the Tuyls et al. (2023) model")
    parser.add_argument("--log-dir", default="results/nethack")
    parser.add_argument("--compute-fisher", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    config = build_config(args)
    loader = NLDLoader(config.dataset_path, config, num_games=config.num_games, seed=args.seed)

    teacher = None
    if args.pretrained:
        teacher = NetHackModel(config).to(args.device)
        teacher.load_state_dict(torch.load(args.pretrained, map_location=args.device))

    if args.compute_fisher:
        model = teacher or NetHackModel(config).to(args.device)

        def log_prob_fn(observations, **__):
            logits, _, _ = model(observations)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(dist.sample())

        batches = []
        iterator = loader.iterations(batch_size=config.fisher_batch_size)
        for _ in range(config.fisher_batches):  # 10 000 batches (addendum)
            batch = next(iterator)
            batches.append({"observations": to_observation(batch, config, args.device)})
        fisher = DiagonalFisher.estimate(model, log_prob_fn, batches, num_batches=config.fisher_batches)
        torch.save(fisher, os.path.join(args.log_dir, f"fisher-{args.seed}.pt"))
        print(f"saved Fisher matrix with {len(fisher)} tensors")
        return

    if args.eval:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required for evaluation")
        model = NetHackModel(config).to(args.device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        metrics = full_evaluation(model, config, episodes=1000, seed=args.seed)
        levels, turns = level_visitation(model, config, episodes=100, seed=args.seed)
        print(metrics)
        return

    bc_buffer = None
    if args.method == "bc":
        bc_buffer = build_bc_buffer(loader, config, max_samples=config.bc_buffer_size)

    fisher = None
    fisher_path = os.path.join(args.log_dir, f"fisher-{args.seed}.pt")
    if args.method == "ewc" and os.path.exists(fisher_path):
        fisher = torch.load(fisher_path, map_location=args.device)

    appo = APPO(config, make_env_factory(config), device=args.device, teacher=teacher,
                bc_buffer=bc_buffer, fisher=fisher)
    appo.pretrain_baseline_head()
    history = appo.train(total_steps=config.total_steps)
    torch.save(appo.model.state_dict(), os.path.join(args.log_dir, f"{args.method}-{args.seed}.pt"))
    with open(os.path.join(args.log_dir, f"{args.method}-{args.seed}.json"), "w") as handle:
        json.dump(history, handle)


if __name__ == "__main__":
    main()
