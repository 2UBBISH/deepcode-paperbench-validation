"""Zero-shot evaluation for Functional Reward Encodings (FRE).

This script loads a trained FRE agent (frozen FRE VAE + IQL networks), samples
exactly ``num_examples`` state-reward pairs for each downstream task, encodes
the task into a latent vector ``z``, and evaluates the resulting policy in the
corresponding environment.

Example:
    python scripts/eval_zero_shot.py \
        --domain antmaze \
        --checkpoint outputs/antmaze/fre_agent.pt \
        --seed 0 \
        --num_episodes 20 \
        --num_examples 32
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from fre.agent import FREAgent
from fre.dataset import build_state_pool, load_offline_dataset

from envs.antmaze_wrapper import (
    ANTMAZE_TASKS,
    evaluate_antmaze_policy,
    sample_task_reward_states as sample_antmaze_states,
)
from envs.exorl_wrapper import (
    EXORL_TASKS,
    evaluate_exorl_policy,
    sample_task_reward_states as sample_exorl_states,
)
from envs.kitchen_wrapper import (
    KITCHEN_TASKS,
    evaluate_kitchen_policy,
    sample_task_reward_states as sample_kitchen_states,
)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_tasks(domain: str, task_override: Optional[List[str]] = None) -> List[str]:
    """Return the downstream task names for a domain."""
    if task_override:
        return list(task_override)

    if domain == "antmaze":
        return list(ANTMAZE_TASKS)
    if domain == "kitchen":
        return list(KITCHEN_TASKS)
    if domain == "walker":
        return [t for t in EXORL_TASKS if t.startswith("walker")]
    if domain == "cheetah":
        return [t for t in EXORL_TASKS if t.startswith("cheetah")]
    raise ValueError(f"Unknown domain: {domain}")


def sample_task_pairs(
    domain: str,
    task_name: str,
    state_pool: np.ndarray,
    num_examples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample exactly ``num_examples`` state-reward pairs for a task."""
    if domain == "antmaze":
        return sample_antmaze_states(
            task_name, state_pool, num_examples=num_examples, seed=seed
        )
    if domain == "kitchen":
        return sample_kitchen_states(
            task_name, state_pool, num_examples=num_examples, seed=seed
        )
    if domain in ("walker", "cheetah"):
        return sample_exorl_states(
            task_name, state_pool, num_examples=num_examples, seed=seed
        )
    raise ValueError(f"Unknown domain: {domain}")


def evaluate_task(
    domain: str,
    task_name: str,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int,
    seed: int,
    env_name: Optional[str] = None,
) -> Dict[str, float]:
    """Dispatch to the domain-specific policy evaluator."""
    if domain == "antmaze":
        return evaluate_antmaze_policy(
            policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            env_name=env_name or "antmaze-large-diverse-v2",
        )
    if domain == "kitchen":
        return evaluate_kitchen_policy(
            policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            env_name=env_name or "kitchen-complete-v0",
        )
    if domain in ("walker", "cheetah"):
        return evaluate_exorl_policy(
            policy_fn,
            task_name=task_name,
            domain=domain,
            num_episodes=num_episodes,
            seed=seed,
        )
    raise ValueError(f"Unknown domain: {domain}")


def extract_score(result: Dict[str, float]) -> float:
    """Extract a normalized score from an evaluator result dict."""
    for key in ("normalized_score", "score", "mean_score", "return", "mean_return"):
        if key in result:
            return float(result[key])
    # Last-resort: average all finite scalar values.
    values = [float(v) for v in result.values() if np.isfinite(float(v))]
    if not values:
        return 0.0
    return float(np.mean(values))


def load_agent(
    checkpoint: str,
    state_dim: int,
    action_dim: int,
    state_pool: np.ndarray,
    dataset: Any,
    device: torch.device,
    latent_dim: int,
) -> FREAgent:
    """Construct a FREAgent and load a checkpoint with flexible key handling."""
    agent = FREAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        state_pool=state_pool,
        dataset=dataset,
        device=device,
        freeze_vae=True,
    )
    agent.to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        # Support checkpoints wrapped in 'state_dict', 'agent', or 'model'.
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "agent" in ckpt:
            ckpt = ckpt["agent"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]

    try:
        agent.load_state_dict(ckpt, strict=False)
    except TypeError:
        # Some load_state_dict implementations require strict as positional.
        agent.load_state_dict(ckpt)

    agent.freeze_vae()
    for module in (getattr(agent, "vae", None), getattr(agent, "iql", None)):
        if module is not None:
            module.eval()
    return agent


def build_policy_fn(
    agent: FREAgent, z: torch.Tensor
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a policy closure that conditions on latent code ``z``."""
    z_np = z.detach().cpu().numpy().reshape(-1)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return agent.get_action(obs, z=z_np, deterministic=True)

    return policy_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FRE agent on zero-shot downstream tasks."
    )
    parser.add_argument("--domain", required=True, choices=["antmaze", "kitchen", "walker", "cheetah"])
    parser.add_argument("--checkpoint", required=True, help="Path to FREAgent checkpoint (.pt)")
    parser.add_argument("--dataset_name", default=None, help="Optional D4RL dataset name override")
    parser.add_argument("--dataset_path", default=None, help="Optional ExORL HDF5 dataset path")
    parser.add_argument("--env_name", default=None, help="Optional gym environment name override")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=32)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--state_pool_size", type=int, default=None)
    parser.add_argument("--tasks", nargs="+", default=None, help="Task names to evaluate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Optional JSON path for results")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample z from the posterior instead of using the posterior mean.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Load the offline dataset and construct the state pool used for both the
    # reward prior and the 32-example task encoding.
    if args.domain == "antmaze":
        dataset_name = args.dataset_name or "antmaze-large-diverse-v2"
        dataset = load_offline_dataset(domain="antmaze", dataset_name=dataset_name)
    elif args.domain == "kitchen":
        dataset_name = args.dataset_name or "kitchen-complete-v0"
        dataset = load_offline_dataset(domain="kitchen", dataset_name=dataset_name)
    else:
        dataset = load_offline_dataset(domain=args.domain, dataset_path=args.dataset_path)

    state_pool = build_state_pool(dataset, max_pool_size=args.state_pool_size)
    state_dim = int(dataset.states.shape[1])
    action_dim = int(dataset.actions.shape[1])

    agent = load_agent(
        checkpoint=args.checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        state_pool=state_pool,
        dataset=dataset,
        device=device,
        latent_dim=args.latent_dim,
    )

    tasks = resolve_tasks(args.domain, args.tasks)
    results: Dict[str, Dict[str, float]] = {}
    summary: Dict[str, float] = {}

    for task_name in tasks:
        states, rewards = sample_task_pairs(
            domain=args.domain,
            task_name=task_name,
            state_pool=state_pool,
            num_examples=args.num_examples,
            seed=args.seed,
        )

        states_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        with torch.no_grad():
            mu, logvar, z_sampled = agent.vae.encode(states_t, rewards_t)
            z = z_sampled if args.stochastic else mu

        policy_fn = build_policy_fn(agent, z)
        eval_result = evaluate_task(
            domain=args.domain,
            task_name=task_name,
            policy_fn=policy_fn,
            num_episodes=args.num_episodes,
            seed=args.seed,
            env_name=args.env_name,
        )

        score = extract_score(eval_result)
        results[task_name] = {k: float(v) for k, v in eval_result.items()}
        summary[task_name] = score
        print(f"[{args.domain}] {task_name}: {score:.2f}")

    if tasks:
        mean_score = float(np.mean(list(summary.values())))
        std_score = float(np.std(list(summary.values())))
        print(f"[{args.domain}] mean={mean_score:.2f} std={std_score:.2f}")
        summary["mean"] = mean_score
        summary["std"] = std_score

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"domain": args.domain, "tasks": results, "summary": summary}, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
