#!/usr/bin/env python3
"""
Refinement stage entry point for RICE.

Loads a frozen target policy and a critical-state buffer (or a trained mask
checkpoint), then trains a refined policy with:
  1. Mixed initial-state resets (critical-state restarts with probability p).
  2. Random Network Distillation (RND) exploration bonus.
  3. PPO optimization.

Example:
    python scripts/refine.py \
        --target-policy outputs/hopper_target/policy.zip \
        --critical-buffer outputs/hopper_mask/critical_buffer.npz \
        --env-id Hopper-v3 --task mujoco \
        --p 0.5 --lambda-rnd 0.01 --total-timesteps 1000000 \
        --seed 0 --output-dir outputs/hopper_refine
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rice.agents import PPOConfig, load_target_policy
from rice.agents.target_policy import BaseTargetPolicy
from rice.envs import make_mujoco_env, make_sparse_mujoco_env
from rice.envs.cage_env import make_cage_env
from rice.envs.malware_env import make_malware_env
from rice.envs.metadrive_env import make_metadrive_env
from rice.envs.selfish_mining_env import make_selfish_mining_env
from rice.masknet import MaskNetwork, build_mask_network
from rice.refine import (
    CriticalStateBuffer,
    RefineTrainer,
    build_critical_buffer_from_trajectories,
    refine_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RICE refinement stage: train a refined policy with mixed resets and RND."
    )

    # Inputs
    parser.add_argument(
        "--target-policy",
        type=str,
        required=True,
        help="Path to the frozen target-policy checkpoint (.zip for SB3, .pt/.pth for torch).",
    )
    parser.add_argument(
        "--critical-buffer",
        type=str,
        default=None,
        help="Path to a saved critical-state buffer (.npz).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to a trained mask-network checkpoint (.pt). If provided and no "
             "critical-buffer is given, the mask is used to build a buffer by rolling out "
             "the target policy.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.txt describing the target policy / task.",
    )

    # Task / environment
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco",
        choices=["mujoco", "sparse_mujoco", "selfish_mining", "cage", "metadrive", "malware"],
        help="Task family.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gym/Gymnasium environment id. Inferred from metadata if omitted.",
    )
    parser.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize observations (default: True for Walker2d/HalfCheetah).",
    )
    parser.add_argument(
        "--use-sb3",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use Stable-Baselines3 backend for the target policy (auto-detected by default).",
    )

    # RICE refinement hyperparameters
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Probability of resetting from a critical state (mixed reset).",
    )
    parser.add_argument(
        "--lambda-rnd",
        type=float,
        default=0.01,
        help="Scale factor for the RND exploration bonus.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Mask intrinsic-reward coefficient (used only when building buffer from mask).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.25,
        help="Top-p percentile for critical-state selection.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Hard threshold alternative for critical-state selection.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="top_p",
        choices=["top_p", "threshold"],
        help="Critical-state selection mode.",
    )
    parser.add_argument(
        "--n-critical-trajectories",
        type=int,
        default=100,
        help="Number of target-policy trajectories used to build a buffer from a mask.",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps for refinement.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Steps per rollout.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per rollout.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value-function loss coefficient.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping.",
    )
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize advantages.",
    )

    # Misc
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
        help="Device for torch models.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/refine",
        help="Directory to save refined policy and metadata.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save intermediate checkpoints every N rollouts.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log training progress every N rollouts.",
    )
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the refined policy from the target policy.",
    )

    args = parser.parse_args()

    # Resolve backend preference from checkpoint extension if not specified.
    if args.use_sb3 is None:
        args.use_sb3 = str(args.target_policy).endswith(".zip")

    return args


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_detect_metadata(checkpoint_path: str, metadata_path: Optional[str]) -> Optional[Path]:
    """Locate a metadata.txt file next to the checkpoint if one exists."""
    if metadata_path is not None:
        path = Path(metadata_path)
        return path if path.exists() else None
    candidate = Path(checkpoint_path).parent / "metadata.txt"
    return candidate if candidate.exists() else None


def _read_metadata(metadata_path: Optional[Path]) -> Dict[str, str]:
    """Parse a simple key: value metadata file."""
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
    """Resolve observation normalization."""
    if explicit is not None:
        return explicit
    return env_id.startswith(("Walker2d", "HalfCheetah"))


def make_env(args: argparse.Namespace, seed: int) -> Any:
    """Create the task-specific environment."""
    env_id = args.env_id
    normalize = _should_normalize_obs(env_id or "", args.normalize_obs)

    if args.task == "mujoco":
        if env_id is None:
            raise ValueError("--env-id is required for mujoco task.")
        return make_mujoco_env(env_id, normalize_obs=normalize, seed=seed)

    if args.task == "sparse_mujoco":
        if env_id is None:
            raise ValueError("--env-id is required for sparse_mujoco task.")
        return make_sparse_mujoco_env(env_id, normalize_obs=normalize, seed=seed)

    if args.task == "selfish_mining":
        return make_selfish_mining_env(seed=seed)

    if args.task == "cage":
        return make_cage_env(seed=seed)

    if args.task == "metadrive":
        return make_metadrive_env(seed=seed)

    if args.task == "malware":
        return make_malware_env(seed=seed)

    raise ValueError(f"Unknown task: {args.task}")


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    """Build a PPOConfig from CLI arguments."""
    return PPOConfig(
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        normalize_advantage=args.normalize_advantage,
        device=args.device,
        seed=args.seed,
    )


def load_critical_buffer(args: argparse.Namespace, target_policy: BaseTargetPolicy, env: Any) -> CriticalStateBuffer:
    """Load or build the critical-state buffer."""
    if args.critical_buffer is not None:
        path = Path(args.critical_buffer)
        if not path.exists():
            raise FileNotFoundError(f"Critical buffer not found: {path}")
        buffer = CriticalStateBuffer(
            capacity=None,
            selection_mode=args.selection_mode,
            top_p=args.top_p,
            threshold=args.threshold,
        )
        buffer.load(path)
        print(f"Loaded critical-state buffer from {path} ({len(buffer)} states).")
        return buffer

    if args.mask is not None:
        mask_path = Path(args.mask)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask checkpoint not found: {mask_path}")
        mask_net = build_mask_network(env.observation_space)
        state = torch.load(mask_path, map_location="cpu")
        mask_net.load_state_dict(state)
        mask_net.eval()
        print(f"Loaded mask network from {mask_path}; building critical-state buffer...")

        # Roll out the target policy and score states with the mask.
        trajectories = []
        for _ in range(args.n_critical_trajectories):
            obs, info = env.reset(seed=args.seed)
            if isinstance(obs, tuple):
                obs = obs[0]
            traj = {"observations": [], "xi": [], "actions": [], "rewards": []}
            done = False
            while not done:
                action, _ = target_policy.predict(obs, deterministic=True)
                xi = mask_net.predict(obs)
                traj["observations"].append(np.asarray(obs, dtype=np.float32))
                traj["xi"].append(float(xi))
                traj["actions"].append(action)
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = result
                traj["rewards"].append(float(reward))
            trajectories.append(traj)

        buffer = build_critical_buffer_from_trajectories(
            trajectories,
            capacity=None,
            selection_mode=args.selection_mode,
            top_p=args.top_p,
            threshold=args.threshold,
        )
        print(f"Built critical-state buffer from mask ({len(buffer)} states).")
        return buffer

    raise ValueError(
        "Either --critical-buffer or --mask must be provided to obtain critical states."
    )


def save_checkpoint(
    trainer: RefineTrainer,
    args: argparse.Namespace,
    output_dir: Path,
    elapsed: float,
) -> Dict[str, str]:
    """Persist refined policy, RND bonus, and run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_path = output_dir / "policy.pt"
    rnd_path = output_dir / "rnd.pt"
    meta_path = output_dir / "metadata.txt"

    trainer.save(str(policy_path), rnd_path=str(rnd_path))

    meta = {
        "task": args.task,
        "env_id": args.env_id or "",
        "seed": str(args.seed),
        "total_timesteps": str(args.total_timesteps),
        "p": str(args.p),
        "lambda_rnd": str(args.lambda_rnd),
        "learning_rate": str(args.learning_rate),
        "n_steps": str(args.n_steps),
        "gamma": str(args.gamma),
        "gae_lambda": str(args.gae_lambda),
        "target_policy": str(args.target_policy),
        "critical_buffer": str(args.critical_buffer or ""),
        "mask": str(args.mask or ""),
        "policy_path": str(policy_path),
        "elapsed_sec": f"{elapsed:.2f}",
    }
    with open(meta_path, "w") as f:
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved refined policy to {policy_path}")
    print(f"Saved RND checkpoint to {rnd_path}")
    print(f"Saved metadata to {meta_path}")
    return meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    metadata_path = _auto_detect_metadata(args.target_policy, args.metadata)
    metadata = _read_metadata(metadata_path)

    # Infer env-id from metadata if not provided.
    if args.env_id is None and "env_id" in metadata:
        args.env_id = metadata["env_id"]
    if args.env_id is None:
        raise ValueError(
            "--env-id is required when it cannot be inferred from target-policy metadata."
        )

    print(f"Task: {args.task}, env_id: {args.env_id}, seed: {args.seed}")
    print(f"Mixed-reset probability p={args.p}, lambda_rnd={args.lambda_rnd}")

    env = make_env(args, seed=args.seed)

    # Load frozen target policy.
    target_policy = load_target_policy(
        args.target_policy,
        backend="sb3" if args.use_sb3 else "torch",
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=args.device,
    )
    print(f"Loaded target policy from {args.target_policy}")

    # Load or build critical-state buffer.
    critical_buffer = load_critical_buffer(args, target_policy, env)

    ppo_config = build_ppo_config(args)

    start = time.time()
    trainer = refine_policy(
        env=env,
        target_policy=target_policy,
        critical_buffer=critical_buffer,
        total_timesteps=args.total_timesteps,
        p=args.p,
        lambda_rnd=args.lambda_rnd,
        ppo_config=ppo_config,
        device=args.device,
        save_path=str(Path(args.output_dir) / "checkpoints"),
    )
    elapsed = time.time() - start

    save_checkpoint(trainer, args, Path(args.output_dir), elapsed)
    print(f"Refinement finished in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
