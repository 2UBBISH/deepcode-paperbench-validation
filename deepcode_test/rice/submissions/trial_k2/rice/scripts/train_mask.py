#!/usr/bin/env python3
"""
Train the RICE MaskNet explanation module.

This script loads a frozen target policy produced by ``train_target.py``,
trains a binary mask network that decides when to randomize the target
policy's action, and extracts a critical-state buffer for the refinement
stage.

Example
-------
.. code-block:: bash

    python scripts/train_mask.py \
        --task mujoco \
        --env-id Hopper-v3 \
        --checkpoint outputs/hopper_target/policy.zip \
        --output-dir outputs/hopper_mask \
        --alpha 1e-4 \
        --total-timesteps 500000 \
        --n-critical-trajectories 100 \
        --top-p 0.25 \
        --seed 0
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

# Allow running from repository root without installation.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rice.agents import PPOConfig, load_target_policy
from rice.agents.target_policy import BaseTargetPolicy
from rice.envs import make_mujoco_env, make_sparse_mujoco_env
from rice.envs.cage_env import make_cage_env
from rice.envs.malware_env import make_malware_env
from rice.envs.metadrive_env import make_metadrive_env
from rice.envs.selfish_mining_env import make_selfish_mining_env
from rice.masknet import MaskTrainer, train_mask_network
from rice.refine import CriticalStateBuffer, build_critical_buffer_from_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RICE MaskNet and extract critical states."
    )

    # Task / environment -----------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "mujoco",
            "sparse_mujoco",
            "selfish_mining",
            "cage",
            "metadrive",
            "malware",
        ],
        help="Task family to run.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gym/Gymnasium environment id (required for MuJoCo tasks).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the frozen target-policy checkpoint from train_target.py.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.txt from train_target.py (auto-detected if omitted).",
    )

    # MaskNet hyper-parameters -----------------------------------------------
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="Intrinsic blinding coefficient alpha (default: 1e-4).",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total environment steps for mask training.",
    )
    parser.add_argument(
        "--n-critical-trajectories",
        type=int,
        default=100,
        help="Number of target-policy trajectories used to build the critical buffer.",
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
        default=None,
        help="Optional hard threshold on xi(s) for critical-state selection.",
    )

    # PPO hyper-parameters ---------------------------------------------------
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--normalize-advantage",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
    )

    # Misc -------------------------------------------------------------------
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for torch models ('auto' selects cuda if available).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mask",
        help="Directory where the mask checkpoint and buffer will be saved.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="If set, save intermediate mask checkpoints every N updates.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Log training progress every N updates.",
    )
    parser.add_argument(
        "--use-sb3",
        action="store_true",
        default=None,
        help="Force loading the target policy as a Stable-Baselines3 model.",
    )
    parser.add_argument(
        "--no-sb3",
        action="store_true",
        default=False,
        help="Force loading the target policy as a custom torch model.",
    )

    args = parser.parse_args()

    # Resolve backend preference.
    if args.use_sb3 and args.no_sb3:
        raise ValueError("Cannot specify both --use-sb3 and --no-sb3.")
    if args.no_sb3:
        args.use_sb3 = False
    elif args.use_sb3 is None:
        # Auto-detect from checkpoint extension unless overridden.
        args.use_sb3 = str(args.checkpoint).endswith(".zip")

    return args


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_detect_metadata(checkpoint_path: str, metadata_path: Optional[str]) -> Optional[Path]:
    """Return a metadata.txt path next to the checkpoint if it exists."""
    if metadata_path is not None:
        return Path(metadata_path)
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


def make_env(args: argparse.Namespace, seed: int) -> Any:
    """Create the task environment used to train the mask network."""
    task = args.task

    if task == "mujoco":
        if args.env_id is None:
            raise ValueError("--env-id is required for MuJoCo tasks.")
        normalize = "Walker2d" in args.env_id or "HalfCheetah" in args.env_id
        return make_mujoco_env(args.env_id, normalize_obs=normalize, seed=seed)

    if task == "sparse_mujoco":
        if args.env_id is None:
            raise ValueError("--env-id is required for sparse MuJoCo tasks.")
        normalize = "Walker2d" in args.env_id or "HalfCheetah" in args.env_id
        return make_sparse_mujoco_env(args.env_id, normalize_obs=normalize, seed=seed)

    if task == "selfish_mining":
        return make_selfish_mining_env(seed=seed)

    if task == "cage":
        return make_cage_env(seed=seed)

    if task == "metadrive":
        return make_metadrive_env(seed=seed)

    if task == "malware":
        return make_malware_env(seed=seed)

    raise ValueError(f"Unknown task: {task}")


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    """Build a PPOConfig from CLI arguments."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        device=device,
        seed=args.seed,
    )


def save_checkpoint(
    trainer: MaskTrainer,
    critical_buffer: CriticalStateBuffer,
    args: argparse.Namespace,
    output_dir: Path,
    elapsed: float,
) -> Dict[str, str]:
    """Persist the trained mask, critical-state buffer, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / "mask.pt"
    buffer_path = output_dir / "critical_buffer.npz"
    meta_path = output_dir / "metadata.txt"

    trainer.save(str(mask_path))
    critical_buffer.save(str(buffer_path))

    meta = {
        "task": args.task,
        "env_id": str(args.env_id or ""),
        "checkpoint": str(args.checkpoint),
        "alpha": str(args.alpha),
        "total_timesteps": str(args.total_timesteps),
        "n_critical_trajectories": str(args.n_critical_trajectories),
        "top_p": str(args.top_p),
        "threshold": str(args.threshold or ""),
        "elapsed_seconds": f"{elapsed:.2f}",
        "mask_path": str(mask_path),
        "buffer_path": str(buffer_path),
        "seed": str(args.seed),
    }

    with open(meta_path, "w") as f:
        for key, value in meta.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved mask checkpoint to {mask_path}")
    print(f"Saved critical-state buffer to {buffer_path}")
    print(f"Saved metadata to {meta_path}")
    return meta


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve device.
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load metadata from target training run.
    metadata_path = _auto_detect_metadata(args.checkpoint, args.metadata)
    metadata = _read_metadata(metadata_path)

    # Infer env-id from metadata if not provided.
    if args.env_id is None and metadata.get("env_id"):
        args.env_id = metadata["env_id"]

    # Create environment.
    env = make_env(args, seed=args.seed)

    # Load frozen target policy.
    backend = "sb3" if args.use_sb3 else "torch"
    target_policy: BaseTargetPolicy = load_target_policy(
        args.checkpoint,
        backend=backend,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    print(f"Loaded target policy from {args.checkpoint} (backend={backend})")

    # Build PPO config and train the mask network.
    ppo_config = build_ppo_config(args)
    print(f"Training MaskNet on {args.task} with alpha={args.alpha} for {args.total_timesteps} steps...")

    start = time.time()
    trainer = train_mask_network(
        env=env,
        target_policy=target_policy,
        total_timesteps=args.total_timesteps,
        alpha=args.alpha,
        ppo_config=ppo_config,
        device=device,
        save_path=None,  # We save manually after training.
    )
    train_time = time.time() - start
    print(f"MaskNet training completed in {train_time:.2f}s")

    # Extract critical states from target-policy trajectories.
    print(
        f"Collecting {args.n_critical_trajectories} target-policy trajectories "
        f"for critical-state extraction (top_p={args.top_p})..."
    )
    critical_states = trainer.collect_critical_states(
        n_trajectories=args.n_critical_trajectories,
        top_p=args.top_p,
        threshold=args.threshold,
    )
    critical_buffer = build_critical_buffer_from_trajectories(
        critical_states,
        selection_mode="top_p" if args.threshold is None else "threshold",
        top_p=args.top_p,
        threshold=args.threshold or 0.5,
    )
    print(f"Critical-state buffer size: {len(critical_buffer)}")
    if len(critical_buffer) > 0:
        summary = critical_buffer.summary()
        print(
            f"xi stats: mean={summary['mean_xi']:.4f}, "
            f"min={summary['min_xi']:.4f}, max={summary['max_xi']:.4f}"
        )

    # Save everything.
    output_dir = Path(args.output_dir)
    meta = save_checkpoint(
        trainer=trainer,
        critical_buffer=critical_buffer,
        args=args,
        output_dir=output_dir,
        elapsed=train_time,
    )

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
