#!/usr/bin/env python
"""Train the goal-conditioned / skill-discovery baselines of Table 1.

    python train_baseline.py --algo gc_iql --domain antmaze --steps 1_000_000
    python train_baseline.py --algo gc_bc  --domain antmaze --steps 500_000
    python train_baseline.py --algo opal   --domain walker  --steps 2_000_000

GC-IQL and GC-BC share the FRE codebase and network structure (Section 5.2);
OPAL trains the skill VAE and the skill policy that are later evaluated in the
privileged setting described in the addendum.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines.gc_bc import GCBCAgent, GCBCConfig, GeometricGoalSampler
from baselines.gc_iql import GCIQLAgent, GCIQLConfig, GoalRelabeler
from baselines.opal import OPALAgent, OPALConfig, sample_trajectory_chunks
from fre.experiment import build_dataset, default_config
from fre.training import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train zero-shot RL baselines")
    p.add_argument("--algo", required=True, choices=["gc_iql", "gc_bc", "opal"])
    p.add_argument("--domain", default="antmaze", choices=["antmaze", "walker", "cheetah", "kitchen"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="runs")
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--context-length", type=int, default=32, help="OPAL trajectory chunk length")
    p.add_argument("--max-steps", type=int, default=None, help="cap the number of update steps (smoke tests)")
    return p.parse_args()


def train_gc_iql(args, dataset, device):
    config = GCIQLConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim, learning_rate=args.learning_rate)
    agent = GCIQLAgent(config, device=device)
    relabeler = GoalRelabeler(dataset, config, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        batch = dataset.sample_transitions(args.batch_size, rng=rng)
        idx = rng.integers(0, len(dataset), size=args.batch_size)
        next_idx = np.minimum(idx + 1, len(dataset) - 1)
        goals, rewards, masks = relabeler.sample(idx, next_idx)
        stats = agent.update(batch, goals, rewards, masks)
        if (step + 1) % args.log_interval == 0:
            print(json.dumps({"step": step + 1, **stats}, default=float), flush=True)
    return agent


def train_gc_bc(args, dataset, device):
    config = GCBCConfig(obs_dim=dataset.obs_dim, action_dim=dataset.action_dim, learning_rate=args.learning_rate)
    agent = GCBCAgent(config, device=device)
    sampler = GeometricGoalSampler(dataset, geometric_p=config.geometric_p, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        idx = rng.integers(0, len(dataset), size=args.batch_size)
        goal = sampler.sample(idx)
        obs = torch.as_tensor(dataset.observations[idx], device=device)
        action = torch.as_tensor(dataset.actions[idx], device=device)
        stats = agent.update(obs, torch.as_tensor(goal, device=device), action)
        if (step + 1) % args.log_interval == 0:
            print(json.dumps({"step": step + 1, **stats}, default=float), flush=True)
    return agent


def train_opal(args, dataset, device):
    config = OPALConfig(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        context_length=args.context_length,
        learning_rate=args.learning_rate,
    )
    agent = OPALAgent(config, device=device)
    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        chunk = sample_trajectory_chunks(dataset, args.batch_size, args.context_length, rng)
        states = chunk["states"].to(device)
        actions = chunk["actions"].to(device)
        next_states = chunk["next_states"].to(device)
        stats = agent.update_vae(states, actions, next_states)
        # Train the skill policy by behavioural cloning on the inferred skills.
        with torch.no_grad():
            skills = agent.infer_skills(states, actions)
        flat_states = states.reshape(-1, states.shape[-1])
        flat_actions = actions.reshape(-1, actions.shape[-1])
        flat_skills = skills.unsqueeze(1).expand(-1, states.shape[1], -1).reshape(-1, skills.shape[-1])
        stats.update(agent.update_policy(flat_states, flat_actions, flat_skills))
        if (step + 1) % args.log_interval == 0:
            print(json.dumps({"step": step + 1, **stats}, default=float), flush=True)
    return agent


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    config = default_config(args.domain)
    dataset = build_dataset(config)
    if args.max_steps is not None:
        args.steps = min(args.steps, args.max_steps)
        args.log_interval = min(args.log_interval, max(1, args.steps))

    trainer = {"gc_iql": train_gc_iql, "gc_bc": train_gc_bc, "opal": train_opal}[args.algo]
    agent = trainer(args, dataset, device)

    run_dir = os.path.join(args.output_dir, f"{args.algo}-{args.domain}-s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(run_dir, "agent.pt"))
    print(f"[train_baseline] saved to {run_dir}")


if __name__ == "__main__":
    main()
