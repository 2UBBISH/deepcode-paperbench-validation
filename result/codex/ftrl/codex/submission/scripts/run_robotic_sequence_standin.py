#!/usr/bin/env python3
"""Run the RoboticSequence experiment on a CPU-runnable stand-in environment.

Meta-World (the environment used in the paper) requires Python >= 3.10 and
MuJoCo, neither of which is available in the environment where this reproduction
was prepared.  This script therefore runs the *same* SAC + knowledge-retention
code on :class:`fpc.metaworld.point_chain.PointReachChain`, a multi-stage
continuous-control task with exactly the structure of RoboticSequence:

* ``pi_*`` is pre-trained on the last two stages (FAR),
* fine-tuning has to solve the whole chain starting from the first stage,
  creating a state coverage gap for the stages that were not in pre-training,
* the per-stage success rates during fine-tuning are the quantity plotted in
  Figure 7 of the paper.

The script writes ``results/metaworld_standin/<method>.json`` and a figure that
can be compared with Figures 3c and 7.

    python scripts/run_robotic_sequence_standin.py --steps 30000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

# Small networks on CPU are faster single-threaded; override with FPC_NUM_THREADS.
torch.set_num_threads(int(os.environ.get("FPC_NUM_THREADS", "1")))

from fpc.metaworld.config import MetaworldConfig
from fpc.metaworld.point_chain import PointReachChain, evaluate_point_chain
from fpc.metaworld.sac import ReplayBuffer, SAC, Transition
from fpc.retention.base import RetentionConfig
from fpc.retention.episodic_memory import ProtectedReplayBuffer


def collect_teacher_data(agent, pretrained_stages, num_samples, seed, num_stages=4):
    """Collect BC (states) and EM (transitions) buffers from ``pi_*``."""

    env = PointReachChain(num_stages=num_stages, seed=seed, stage_indices=list(pretrained_stages))
    states, stage_ids = [], []
    em = {key: [] for key in ("obs", "next_obs", "actions", "rewards", "dones", "stages", "next_stages")}
    obs, info = env.reset(seed=seed)
    stage = int(info["stage"])
    attempts = 0
    while len(states) < num_samples and attempts < num_samples * 200:
        attempts += 1
        action = agent.select_action(obs, stage)
        result = env.step(action)
        # The chain only contains the pre-trained (FAR) stages, so every
        # transition belongs to the pre-training distribution.
        states.append(obs)
        stage_ids.append(stage)
        em["obs"].append(obs)
        em["next_obs"].append(result.observation)
        em["actions"].append(action)
        em["rewards"].append(result.reward)
        em["dones"].append(float(result.terminated or result.truncated))
        em["stages"].append(stage)
        em["next_stages"].append(result.stage)
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, info = env.reset(seed=seed + len(states))
            stage = int(info["stage"])
    env.close()
    bc = {"obs": np.asarray(states, dtype=np.float32), "stages": np.asarray(stage_ids, dtype=np.int64)}
    em = {k: np.asarray(v,
                        dtype=np.float32 if k not in ("stages", "next_stages") else np.int64)
          for k, v in em.items()}
    return bc, em


def run_method(method, args, teacher, bc_dataset, em_dataset, seed):
    """Fine-tune one retention method and return the per-stage success curves."""

    config = MetaworldConfig()
    config.hidden_dim = args.hidden_dim
    config.batch_size = args.batch_size
    config.num_stages = 4
    config.retention = RetentionConfig(
        method=method, coefficient={"ewc": 100.0, "bc": 1.0, "em": 0.0, "none": 0.0}[method]
    )
    num_stages = 4
    env = PointReachChain(num_stages=num_stages, seed=seed)
    obs_dim, action_dim = env.observation_dim, env.num_actions

    agent = SAC(config, obs_dim, action_dim, num_stages, device="cpu", teacher=teacher, bc_dataset=bc_dataset)
    agent.actor.load_state_dict(teacher.actor.state_dict())

    if method == "ewc":
        from fpc.retention.fisher import DiagonalFisher

        def log_prob(observations, stage, **_):
            _, _, mu, std = agent.actor(observations, stage, with_logprob=False)
            dist = torch.distributions.Normal(mu, std)
            return dist.log_prob(dist.sample()).sum(-1)

        batches = []
        for _ in range(20):
            idx = np.random.randint(0, len(bc_dataset["obs"]), 32)
            batches.append(
                {
                    "observations": torch.as_tensor(bc_dataset["obs"][idx], dtype=torch.float32),
                    "stage": torch.as_tensor(bc_dataset["stages"][idx], dtype=torch.long),
                }
            )
        agent.retention.set_fisher(DiagonalFisher.estimate(agent.actor, log_prob, batches, num_batches=20))

    if method == "em":
        protected = ProtectedReplayBuffer(args.buffer_size, protected_size=len(em_dataset["obs"]))
        replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim, protected=protected)
        replay.load_protected(em_dataset)
    else:
        replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim)

    history = {"steps": [], "stage_success": {f"stage_{i}": [] for i in range(num_stages)}, "overall": []}
    obs, info = env.reset(seed=seed)
    stage = int(info["stage"])
    for step in range(args.steps):
        action = agent.select_action(obs, stage)
        result = env.step(action)
        replay.add(
            Transition(obs, action, result.reward, result.observation,
                       float(result.terminated or result.truncated), stage, result.stage)
        )
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, info = env.reset(seed=seed + step)
            stage = int(info["stage"])
        if replay.size >= config.batch_size:
            agent.update(replay)
        if step % args.eval_every == 0:
            metrics = evaluate_point_chain(agent, num_stages=num_stages, episodes=5,
                                           time_limit=env.time_limit, seed=seed)
            history["steps"].append(step)
            for i in range(num_stages):
                history["stage_success"][f"stage_{i}"].append(metrics[f"stage_{i}"])
            history["overall"].append(metrics["overall"])
            print(f"  [{method}] step {step:6d}  overall {metrics['overall']:.2f}  "
                  + " ".join(f"s{i}={metrics[f'stage_{i}']:.2f}" for i in range(num_stages)))
    env.close()
    return history


def pretrain_teacher(args, seed):
    """Train ``pi_*`` with SAC on the last two stages (the FAR states)."""

    config = MetaworldConfig()
    config.hidden_dim = args.hidden_dim
    config.batch_size = args.batch_size
    num_stages = 4
    # Pre-training restricted to the FAR stages, as in the paper.
    env = PointReachChain(num_stages=num_stages, seed=seed, stage_indices=[num_stages - 2, num_stages - 1])
    obs_dim, action_dim = env.observation_dim, env.num_actions
    teacher = SAC(config, obs_dim, action_dim, num_stages, device="cpu")
    replay = ReplayBuffer(args.buffer_size, obs_dim, action_dim)
    obs, info = env.reset(seed=seed)
    stage = int(info["stage"])
    for step in range(args.pretrain_steps):
        action = teacher.select_action(obs, stage)
        result = env.step(action)
        replay.add(
            Transition(obs, action, result.reward, result.observation,
                       float(result.terminated or result.truncated), stage, result.stage)
        )
        obs, stage = result.observation, result.stage
        if result.terminated or result.truncated:
            obs, info = env.reset(seed=seed + step)
            stage = int(info["stage"])
        if replay.size >= config.batch_size:
            teacher.update(replay)
    env.close()
    return teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["none", "bc", "ewc", "em"])
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--pretrain-steps", type=int, default=20_000)
    parser.add_argument("--eval-every", type=int, default=3_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/metaworld_standin")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    num_stages = 4
    print("== pre-training pi_* on the last two stages ==")
    teacher = pretrain_teacher(args, args.seed)
    # Evaluate pi_* on its own (FAR) chain -- this is the starting point of
    # fine-tuning for stages 2 and 3.
    far_env_metrics = evaluate_point_chain(
        teacher, num_stages=num_stages, episodes=10, seed=args.seed,
    )
    print(f"  pi_* on the FAR chain: {far_env_metrics}")
    bc_dataset, em_dataset = collect_teacher_data(teacher, {num_stages - 2, num_stages - 1}, 2000, args.seed)

    results = {}
    for method in args.methods:
        print(f"== method: {method} ==")
        results[method] = run_method(method, args, teacher, bc_dataset, em_dataset, args.seed)
        with open(os.path.join(args.out, f"{method}.json"), "w") as handle:
            json.dump(results[method], handle)

    try:
        plot(results, args.out)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plotting failed: {exc}")


def plot(results, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, history in results.items():
        axes[0].plot(history["steps"], history["overall"], marker="o", label=method)
        axes[1].plot(history["steps"], history["stage_success"]["stage_3"], marker="s", label=method)
    axes[0].set_title("Overall chain success rate (cf. Figure 3c)")
    axes[0].set_xlabel("fine-tuning steps")
    axes[0].set_ylabel("success rate")
    axes[0].legend()
    axes[1].set_title("FAR stage (stage 3) success rate (cf. Figure 7)")
    axes[1].set_xlabel("fine-tuning steps")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure3c_standin.png"), dpi=150)
    print(f"figure written to {os.path.join(out_dir, 'figure3c_standin.png')}")


if __name__ == "__main__":
    main()
