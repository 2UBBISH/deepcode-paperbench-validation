#!/usr/bin/env python3
"""RoboticSequence (Meta-World): pre-training, fine-tuning and analysis.

Example::

    # pre-train pi_* on the last two stages (peg-unplug-side, push-wall)
    python scripts/train_metaworld.py --phase pretrain --seed 0

    # fine-tune on the full sequence with behavioral cloning
    python scripts/train_metaworld.py --phase finetune --method bc --seed 0

    # run the whole grid over 20 seeds for Figure 3c / Figure 7
    python scripts/train_metaworld.py --phase grid --seeds 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from fpc.analysis.cka import cka_over_training, layer_activations
from fpc.metaworld.config import MetaworldConfig
from fpc.metaworld.eval import evaluate_stages, success_rate_curve
from fpc.metaworld.train import (
    collect_bc_dataset,
    collect_episodic_memory,
    fine_tune,
    pretrain_policy,
)
from fpc.retention.base import RetentionConfig


def build_config(args) -> MetaworldConfig:
    config = MetaworldConfig()
    config.num_train_steps = args.steps
    config.seed = args.seed
    config.device = args.device
    config.log_dir = args.log_dir
    config.observation_translation = args.translate
    config.reset_last_layer = args.reset_last_layer
    config.retention = RetentionConfig(
        method=args.method,
        coefficient={"ewc": 100.0, "bc": 1.0, "em": 0.0, "none": 0.0}[args.method],
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pretrain", "finetune", "grid", "analyze"], default="finetune")
    parser.add_argument("--method", choices=["none", "bc", "ewc", "em"], default="bc")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-dir", default="results/metaworld")
    parser.add_argument("--translate", type=float, default=0.0, help="observation shift c (Appendix F)")
    parser.add_argument("--reset-last-layer", action="store_true", help="Figure 21")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    config = build_config(args)
    stages = config.pretrained_stages

    if args.phase == "pretrain":
        agent = pretrain_policy(config, stages, num_steps=args.steps, seed=args.seed)
        agent.save(os.path.join(args.log_dir, f"pretrained-{args.seed}.pt"))
        return

    # Pre-train pi_* and collect the retention buffers.
    teacher = pretrain_policy(config, stages, num_steps=max(args.steps // 3, 1000), seed=args.seed)
    bc_dataset = collect_bc_dataset(teacher, config, stages, config.memory_size, seed=args.seed)
    em_dataset = collect_episodic_memory(teacher, config, stages, config.memory_size, seed=args.seed)

    if args.phase == "grid":
        curves = {method: {stage: [] for stage in stages} for method in ("none", "bc", "ewc", "em")}
        for method in ("none", "bc", "ewc", "em"):
            for seed in range(args.seeds):
                method_config = build_config(args)
                method_config.retention = RetentionConfig(
                    method=method, coefficient={"ewc": 100.0, "bc": 1.0, "em": 0.0, "none": 0.0}[method]
                )
                history = fine_tune(method_config, teacher, seed=seed, bc_dataset=bc_dataset, em_dataset=em_dataset)
                for stage, curve in history["stage_success"].items():
                    curves[method][stage].append(curve)
        with open(os.path.join(args.log_dir, "grid.json"), "w") as handle:
            json.dump(curves, handle)
        return

    history = fine_tune(config, teacher, seed=args.seed, bc_dataset=bc_dataset, em_dataset=em_dataset)
    agent = history.pop("agent")

    if args.phase == "analyze":
        from fpc.metaworld.analysis import (
            collect_cka_curve,
            collect_expert_loglikelihoods,
            collect_expert_trajectories,
            collect_reference_activations,
        )

        states, actions = collect_expert_trajectories(teacher, config, stages, seed=args.seed)
        checkpoints = list(range(0, args.steps + 1, 50_000))  # every 50K steps (addendum)
        likelihoods = collect_expert_loglikelihoods(agent, states, actions, checkpoints, device=args.device)
        np.savez(os.path.join(args.log_dir, "expert_loglikelihoods.npz"), **likelihoods)
        reference = collect_reference_activations(teacher.actor, states, device=args.device)
        cka = collect_cka_curve(agent, reference, states, checkpoints, device=args.device)
        with open(os.path.join(args.log_dir, "cka.json"), "w") as handle:
            json.dump(cka, handle)
        print("analysis artefacts written")
        return

    success = success_rate_curve(agent, config, seed=args.seed)
    per_stage = evaluate_stages(agent, config, episodes=config.num_eval_episodes, seed=args.seed)
    print(f"overall success rate: {success:.3f}")
    print("per-stage success:", per_stage)
    agent.save(os.path.join(args.log_dir, f"{args.method}-{args.seed}.pt"))
    with open(os.path.join(args.log_dir, f"{args.method}-{args.seed}.json"), "w") as handle:
        json.dump({"success": success, "per_stage": per_stage, "history": history}, handle, default=str)


if __name__ == "__main__":
    main()
