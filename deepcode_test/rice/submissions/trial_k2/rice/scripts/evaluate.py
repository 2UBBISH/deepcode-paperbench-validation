#!/usr/bin/env python3
"""
Evaluation script for RICE.

Loads a trained target or refined policy and runs a fixed number of evaluation
episodes (default 500, matching the paper).  Reports mean episode return,
standard deviation / standard error, task-specific success metrics, and
optionally explanation fidelity scores when a mask network is provided.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rice.agents import PPOConfig, PPOTrainer, load_target_policy
from rice.agents.target_policy import BaseTargetPolicy, TorchTargetPolicy
from rice.envs import make_mujoco_env, make_sparse_mujoco_env
from rice.envs.cage_env import make_cage_env
from rice.envs.malware_env import make_malware_env
from rice.envs.metadrive_env import make_metadrive_env
from rice.envs.selfish_mining_env import make_selfish_mining_env
from rice.masknet import MaskNetwork, build_mask_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a RICE policy.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the policy checkpoint (.zip for SB3, .pt/.pth for torch).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco",
        choices=[
            "mujoco",
            "sparse_mujoco",
            "selfish_mining",
            "cage",
            "metadrive",
            "malware",
            "mountaincar",
        ],
        help="Task family.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gym/Gymnasium environment id.  Inferred from metadata if omitted.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=500,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for torch policies ('auto' selects cuda if available).",
    )
    parser.add_argument(
        "--normalize-obs",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Normalize observations (default True for Walker2d/HalfCheetah).",
    )
    parser.add_argument(
        "--deterministic",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Use deterministic policy actions.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to a trained mask checkpoint for fidelity evaluation.",
    )
    parser.add_argument(
        "--fidelity-episodes",
        type=int,
        default=50,
        help="Number of episodes used to estimate explanation fidelity.",
    )
    parser.add_argument(
        "--fidelity-samples",
        type=int,
        default=200,
        help="Max number of states to evaluate for fidelity.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (if supported).",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.txt path next to the checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path to write evaluation results.",
    )
    parser.add_argument(
        "--use-sb3",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        help="Force SB3 backend for .zip checkpoints.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_detect_metadata(checkpoint_path: str, metadata_path: Optional[str]) -> Optional[Path]:
    if metadata_path is not None:
        return Path(metadata_path)
    candidate = Path(checkpoint_path).parent / "metadata.txt"
    return candidate if candidate.exists() else None


def _read_metadata(metadata_path: Optional[Path]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if metadata_path is None or not metadata_path.exists():
        return meta
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def _should_normalize_obs(env_id: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return env_id.lower().startswith(("walker2d", "halfcheetah"))


def make_env(args: argparse.Namespace, seed: int) -> Any:
    """Create the evaluation environment for the selected task."""
    env_id = args.env_id
    normalize = _should_normalize_obs(env_id or "", args.normalize_obs)

    if args.task == "mujoco":
        if env_id is None:
            raise ValueError("--env-id is required for mujoco task")
        env = make_mujoco_env(env_id, normalize_obs=normalize, seed=seed)
    elif args.task == "sparse_mujoco":
        if env_id is None:
            raise ValueError("--env-id is required for sparse_mujoco task")
        env = make_sparse_mujoco_env(env_id, normalize_obs=normalize, seed=seed)
    elif args.task == "selfish_mining":
        env = make_selfish_mining_env(seed=seed)
    elif args.task == "cage":
        env = make_cage_env(seed=seed)
    elif args.task == "metadrive":
        env = make_metadrive_env(seed=seed)
    elif args.task == "malware":
        env = make_malware_env(seed=seed)
    elif args.task == "mountaincar":
        import gymnasium as gym

        env = gym.make("MountainCarContinuous-v0")
        env.reset(seed=seed)
        env.action_space.seed(seed)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    return env


def load_policy(args: argparse.Namespace, env: Any) -> BaseTargetPolicy:
    """Load a target/refined policy from checkpoint."""
    backend = "sb3" if args.use_sb3 else None
    policy = load_target_policy(
        args.checkpoint,
        backend=backend,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=args.device,
    )
    return policy


def evaluate_policy(
    policy: BaseTargetPolicy,
    env: Any,
    n_episodes: int = 500,
    deterministic: bool = True,
    render: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ``n_episodes`` and aggregate returns and task-specific metrics."""
    returns: List[float] = []
    lengths: List[int] = []
    successes: List[float] = []
    revenues: List[float] = []
    collisions: List[float] = []
    evasions: List[float] = []

    for ep in range(n_episodes):
        obs, info = env.reset(), {}
        if isinstance(obs, tuple):
            obs, info = obs
        done = False
        ep_return = 0.0
        steps = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=deterministic)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            ep_return += float(reward)
            steps += 1
            if render:
                env.render()
            if max_steps is not None and steps >= max_steps:
                done = True

        returns.append(ep_return)
        lengths.append(steps)

        # Task-specific metrics extracted from info when available.
        if "episode_revenue" in info:
            revenues.append(float(info["episode_revenue"]))
        if "success" in info:
            successes.append(float(info["success"]))
        if "collision" in info:
            collisions.append(float(info["collision"]))
        if "evasion" in info:
            evasions.append(float(info["evasion"]))

    result: Dict[str, Any] = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "se_return": float(np.std(returns) / max(1, np.sqrt(len(returns)))),
        "median_return": float(np.median(returns)),
        "mean_length": float(np.mean(lengths)),
        "n_episodes": n_episodes,
    }
    if revenues:
        result["mean_revenue"] = float(np.mean(revenues))
    if successes:
        result["success_rate"] = float(np.mean(successes))
    if collisions:
        result["collision_rate"] = float(np.mean(collisions))
    if evasions:
        result["evasion_rate"] = float(np.mean(evasions))
    return result


def _collect_target_trajectories(
    policy: BaseTargetPolicy,
    env: Any,
    n_episodes: int,
    max_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect target-policy trajectories as list of dicts with observations."""
    trajectories: List[Dict[str, Any]] = []
    for _ in range(n_episodes):
        obs, info = env.reset(), {}
        if isinstance(obs, tuple):
            obs, info = obs
        traj_obs: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
        traj_rewards: List[float] = []
        done = False
        steps = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            traj_obs.append(np.asarray(obs, dtype=np.float32))
            traj_rewards.append(float(reward))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                done = True
        trajectories.append({"observations": np.stack(traj_obs), "rewards": np.array(traj_rewards)})
    return trajectories


def _estimate_return_from_state(
    policy: BaseTargetPolicy,
    env: Any,
    state_obs: np.ndarray,
    first_action: Optional[np.ndarray] = None,
    n_rollouts: int = 1,
    max_steps: int = 1000,
) -> float:
    """
    Estimate the expected return starting from ``state_obs``.

    If ``first_action`` is provided it is executed at the first step; afterwards
    the target policy is followed.  Because many environments do not support
    true state restoration, this function resets the environment and returns
    the rollout return as a Monte-Carlo proxy.
    """
    returns = []
    for _ in range(n_rollouts):
        obs, info = env.reset(), {}
        if isinstance(obs, tuple):
            obs, info = obs
        # Best-effort state restoration is not generally available, so we treat
        # the provided observation as the starting point for the value estimate.
        obs = np.asarray(state_obs, dtype=obs.dtype)
        done = False
        ep_return = 0.0
        steps = 0
        use_first = first_action is not None
        while not done and steps < max_steps:
            if use_first:
                action = np.asarray(first_action)
                use_first = False
            else:
                action, _ = policy.predict(obs, deterministic=True)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            ep_return += float(reward)
            steps += 1
        returns.append(ep_return)
    return float(np.mean(returns))


def compute_fidelity(
    policy: BaseTargetPolicy,
    mask_network: MaskNetwork,
    env: Any,
    n_episodes: int = 50,
    n_samples: int = 200,
    n_action_samples: int = 8,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute explanation fidelity for a mask network.

    Fidelity is measured as the correlation between the mask score xi(s) and the
    estimated performance drop when the action at state s is randomized.  High
    positive correlation means the mask correctly identifies critical states.
    """
    trajectories = _collect_target_trajectories(policy, env, n_episodes, max_steps=max_steps)

    # Flatten candidate states.
    candidates: List[Tuple[np.ndarray, np.ndarray]] = []
    for traj in trajectories:
        obs = traj["observations"]
        # Use states before the terminal observation.
        for t in range(len(obs) - 1):
            candidates.append((obs[t], obs[t]))

    if len(candidates) == 0:
        return {"fidelity_pearson": 0.0, "fidelity_spearman": 0.0, "n_samples": 0}

    if len(candidates) > n_samples:
        indices = np.random.choice(len(candidates), size=n_samples, replace=False)
        candidates = [candidates[i] for i in indices]

    xi_scores: List[float] = []
    q_diffs: List[float] = []

    for state_obs, _ in candidates:
        xi = float(mask_network.predict(state_obs))
        xi_scores.append(xi)

        # Estimate Q^pi(s, a) where a is the target policy action.
        target_action, _ = policy.predict(state_obs, deterministic=True)
        q_sa = _estimate_return_from_state(
            policy, env, state_obs, first_action=target_action, n_rollouts=1, max_steps=max_steps or 1000
        )

        # Estimate E_{a' ~ random}[Q^pi(s, a')].
        random_returns = []
        for _ in range(n_action_samples):
            random_action = env.action_space.sample()
            q_rand = _estimate_return_from_state(
                policy, env, state_obs, first_action=random_action, n_rollouts=1, max_steps=max_steps or 1000
            )
            random_returns.append(q_rand)
        q_mean_random = float(np.mean(random_returns))

        # Q_diff approximates how much better the target action is than random.
        q_diffs.append(q_sa - q_mean_random)

    xi_arr = np.array(xi_scores)
    q_arr = np.array(q_diffs)

    # Pearson correlation.
    if np.std(xi_arr) < 1e-8 or np.std(q_arr) < 1e-8:
        pearson = 0.0
        spearman = 0.0
    else:
        pearson = float(np.corrcoef(xi_arr, q_arr)[0, 1])
        from scipy.stats import spearmanr

        spearman = float(spearmanr(xi_arr, q_arr)[0])

    return {
        "fidelity_pearson": pearson,
        "fidelity_spearman": spearman,
        "n_samples": len(xi_scores),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata_path = _auto_detect_metadata(args.checkpoint, args.metadata)
    metadata = _read_metadata(metadata_path)
    if args.env_id is None and "env_id" in metadata:
        args.env_id = metadata["env_id"]

    env = make_env(args, args.seed)
    policy = load_policy(args, env)

    print(f"Evaluating {args.checkpoint} on {args.task} ({args.env_id or 'n/a'})")
    print(f"Episodes: {args.n_episodes}, deterministic: {args.deterministic}, seed: {args.seed}")

    start = time.time()
    results = evaluate_policy(
        policy,
        env,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render,
    )
    eval_time = time.time() - start
    results["eval_time_sec"] = eval_time

    print("\n=== Evaluation Results ===")
    print(f"Mean return: {results['mean_return']:.2f} +/- {results['se_return']:.2f}")
    print(f"Std return:  {results['std_return']:.2f}")
    print(f"Median return: {results['median_return']:.2f}")
    print(f"Mean length: {results['mean_length']:.1f}")
    for key in ["mean_revenue", "success_rate", "collision_rate", "evasion_rate"]:
        if key in results:
            print(f"{key}: {results[key]:.4f}")
    print(f"Eval time: {eval_time:.1f}s")

    if args.mask is not None:
        print("\n=== Fidelity Evaluation ===")
        mask_network = build_mask_network(env.observation_space)
        state = torch.load(args.mask, map_location="cpu")
        if "mask_net" in state:
            mask_network.load_state_dict(state["mask_net"])
        else:
            mask_network.load_state_dict(state)
        mask_network.eval()
        fidelity = compute_fidelity(
            policy,
            mask_network,
            env,
            n_episodes=args.fidelity_episodes,
            n_samples=args.fidelity_samples,
        )
        results.update(fidelity)
        print(f"Fidelity Pearson:  {fidelity['fidelity_pearson']:.4f}")
        print(f"Fidelity Spearman: {fidelity['fidelity_spearman']:.4f}")
        print(f"Fidelity samples:  {fidelity['n_samples']}")

    if args.output is not None:
        import json

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")

    env.close()


if __name__ == "__main__":
    main()
