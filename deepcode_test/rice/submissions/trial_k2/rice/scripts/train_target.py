#!/usr/bin/env python3
"""Train a frozen target policy π for a RICE task.

This script is the first stage of the RICE pipeline. It trains a base RL agent
on the requested task/environment and saves the policy checkpoint so that the
mask network and refinement stages can use it.

Supported tasks
---------------
- Dense MuJoCo: Hopper-v3, Walker2d-v3, Reacher-v2, HalfCheetah-v3
- Sparse MuJoCo: SparseHopper-v3, SparseWalker2d-v3, SparseHalfCheetah-v3
- Selfish mining (custom discrete MDP)
- CAGE Challenge 2 / CybORG blue agent
- MetaDrive autonomous driving (Macro-v1)
- Malware mutation (MalConv gym)

Examples
--------
    # Dense MuJoCo
    python scripts/train_target.py --task mujoco --env-id Hopper-v3 --seed 0

    # Sparse MuJoCo
    python scripts/train_target.py --task sparse_mujoco --env-id SparseHopper-v3 --seed 0

    # Selfish mining
    python scripts/train_target.py --task selfish_mining --seed 0

    # CAGE Challenge 2
    python scripts/train_target.py --task cage --seed 0

    # MetaDrive
    python scripts/train_target.py --task metadrive --seed 0

    # Malware mutation
    python scripts/train_target.py --task malware --seed 0
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Make the repository importable when running from the scripts directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rice.agents import PPOConfig, PPOTrainer, TorchTargetPolicy
from rice.agents.target_policy import MLPActorCritic, SB3TargetPolicy, load_target_policy
from rice.envs import make_mujoco_env, make_sparse_mujoco_env
from rice.envs.cage_env import make_cage_env
from rice.envs.malware_env import make_malware_env
from rice.envs.metadrive_env import make_metadrive_env
from rice.envs.selfish_mining_env import make_selfish_mining_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RICE target policy.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["mujoco", "sparse_mujoco", "selfish_mining",
                                 "cage", "metadrive", "malware"],
                        help="Task family to train on.")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Gym/Gymnasium environment id (MuJoCo tasks).")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total environment steps for training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device (cpu/cuda/auto).")
    parser.add_argument("--output-dir", type=str, default="checkpoints/target",
                        help="Directory where the policy checkpoint is saved.")
    parser.add_argument("--use-sb3", action="store_true", default=None,
                        help="Use Stable-Baselines3 PPO when available.")
    parser.add_argument("--no-sb3", action="store_true",
                        help="Force the custom PPO trainer (no SB3).")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="Number of steps per PPO update.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Minibatch size for PPO updates.")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="Number of optimization epochs per update.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value-function loss coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Gradient clipping norm.")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=None,
                        help="Hidden layer sizes for the custom MLP policy.")
    parser.add_argument("--normalize-obs", action="store_true", default=None,
                        help="Normalize observations (MuJoCo only).")
    parser.add_argument("--no-normalize-obs", action="store_true",
                        help="Disable observation normalization.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--save-format", type=str, default="auto",
                        choices=["auto", "sb3", "torch"],
                        help="Checkpoint format to use.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_env(args: argparse.Namespace, seed: int):
    """Create the raw task environment based on CLI arguments."""
    task = args.task
    if task == "mujoco":
        env_id = args.env_id or "Hopper-v3"
        normalize = args.normalize_obs
        if normalize is None:
            normalize = env_id.startswith(("Walker2d", "HalfCheetah"))
        if args.no_normalize_obs:
            normalize = False
        env = make_mujoco_env(env_id, normalize_obs=normalize, seed=seed)
    elif task == "sparse_mujoco":
        env_id = args.env_id or "SparseHopper-v3"
        normalize = args.normalize_obs
        if normalize is None:
            normalize = env_id.startswith(("SparseWalker2d", "SparseHalfCheetah"))
        if args.no_normalize_obs:
            normalize = False
        env = make_sparse_mujoco_env(env_id, normalize_obs=normalize, seed=seed)
    elif task == "selfish_mining":
        env = make_selfish_mining_env(seed=seed)
    elif task == "cage":
        env = make_cage_env(seed=seed)
    elif task == "metadrive":
        env = make_metadrive_env(seed=seed)
    elif task == "malware":
        env = make_malware_env(seed=seed)
    else:
        raise ValueError(f"Unknown task: {task}")
    return env


def infer_obs_action_dims(env) -> Dict[str, Any]:
    """Infer observation/action dimensions and discreteness from a Gym env."""
    obs_space = env.observation_space
    act_space = env.action_space
    info: Dict[str, Any] = {}

    if hasattr(obs_space, "shape"):
        info["obs_shape"] = obs_space.shape
        info["obs_dim"] = int(np.prod(obs_space.shape))
    else:
        raise ValueError("Unsupported observation space type.")

    discrete = hasattr(act_space, "n")
    info["discrete"] = discrete
    if discrete:
        info["action_dim"] = int(act_space.n)
    elif hasattr(act_space, "shape"):
        info["action_shape"] = act_space.shape
        info["action_dim"] = int(np.prod(act_space.shape))
    else:
        raise ValueError("Unsupported action space type.")

    info["obs_space"] = obs_space
    info["action_space"] = act_space
    return info


def build_custom_policy(env, args: argparse.Namespace, device: str) -> TorchTargetPolicy:
    """Build a TorchTargetPolicy using the custom MLPActorCritic backbone."""
    info = infer_obs_action_dims(env)
    hidden_sizes = tuple(args.hidden_sizes) if args.hidden_sizes else (64, 64)
    # Selfish mining paper spec: 4-layer MLP [128,128,128,128]
    if args.task == "selfish_mining" and args.hidden_sizes is None:
        hidden_sizes = (128, 128, 128, 128)

    model = MLPActorCritic(
        obs_dim=info["obs_dim"],
        action_dim=info["action_dim"],
        hidden_sizes=hidden_sizes,
        discrete=info["discrete"],
        share_backbone=False,
    )
    policy = TorchTargetPolicy(
        model=model,
        observation_space=info["obs_space"],
        action_space=info["action_space"],
        device=device,
    )
    return policy


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    """Build a PPOConfig from CLI arguments."""
    return PPOConfig(
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        normalize_advantage=True,
        device=args.device,
        seed=args.seed,
    )


def train_with_sb3(env, args: argparse.Namespace) -> Any:
    """Train a target policy using Stable-Baselines3 PPO."""
    try:
        from stable_baselines3 import PPO
    except Exception as exc:
        raise ImportError(
            "Stable-Baselines3 is required for --use-sb3. Install it or use --no-sb3."
        ) from exc

    policy_kwargs = {}
    if args.hidden_sizes is not None:
        policy_kwargs["net_arch"] = [dict(pi=list(args.hidden_sizes),
                                          vf=list(args.hidden_sizes))]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1 if args.verbose else 0,
        seed=args.seed,
        device=args.device if args.device != "auto" else "auto",
    )
    model.learn(total_timesteps=args.total_timesteps)
    return model


def train_with_custom_ppo(env, args: argparse.Namespace) -> TorchTargetPolicy:
    """Train a target policy using the custom RICE PPO trainer."""
    policy = build_custom_policy(env, args, args.device)
    config = build_ppo_config(args)
    trainer = PPOTrainer(policy=policy, env=env, config=config)
    trainer.learn(
        total_timesteps=args.total_timesteps,
        log_interval=1 if args.verbose else 10,
    )
    return policy


def save_checkpoint(policy, args: argparse.Namespace, output_dir: Path) -> Dict[str, str]:
    """Persist the trained policy and return metadata paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    # Determine save format.
    save_format = args.save_format
    if save_format == "auto":
        if isinstance(policy, SB3TargetPolicy):
            save_format = "sb3"
        else:
            save_format = "torch"

    if save_format == "sb3":
        sb3_path = output_dir / "policy.zip"
        policy.model.save(str(sb3_path))
        paths["policy"] = str(sb3_path)
    else:
        torch_path = output_dir / "policy.pt"
        policy.save(str(torch_path))
        paths["policy"] = str(torch_path)

    # Save a small metadata file for downstream scripts.
    meta = {
        "task": args.task,
        "env_id": args.env_id,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "policy_path": paths["policy"],
        "save_format": save_format,
    }
    meta_path = output_dir / "metadata.txt"
    with open(meta_path, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")
    paths["metadata"] = str(meta_path)
    return paths


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve output directory.
    task_name = args.task
    if args.env_id:
        task_name = f"{task_name}_{args.env_id}"
    output_dir = Path(args.output_dir) / task_name / f"seed_{args.seed}"

    # Decide whether to use SB3.
    use_sb3 = args.use_sb3
    if use_sb3 is None and not args.no_sb3:
        # Default: use SB3 for MuJoCo/CAGE/selfish mining if available.
        try:
            import stable_baselines3  # noqa: F401
            use_sb3 = args.task in ("mujoco", "sparse_mujoco", "cage", "selfish_mining")
        except Exception:
            use_sb3 = False
    if args.no_sb3:
        use_sb3 = False

    print(f"Task: {args.task}, env_id={args.env_id}, seed={args.seed}")
    print(f"Output directory: {output_dir}")
    print(f"Using SB3: {use_sb3}")

    env = make_env(args, seed=args.seed)

    start_time = time.time()
    if use_sb3:
        sb3_model = train_with_sb3(env, args)
        policy = SB3TargetPolicy(sb3_model, device=args.device)
    else:
        policy = train_with_custom_ppo(env, args)
    elapsed = time.time() - start_time

    paths = save_checkpoint(policy, args, output_dir)
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Saved policy to {paths['policy']}")
    print(f"Saved metadata to {paths['metadata']}")

    env.close()


if __name__ == "__main__":
    main()
