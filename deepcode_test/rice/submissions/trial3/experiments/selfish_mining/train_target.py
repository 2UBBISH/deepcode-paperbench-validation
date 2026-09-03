#!/usr/bin/env python3
"""
Train a target PPO agent on the Selfish Mining environment.

This script trains an initial policy on the selfish mining domain using
Stable-Baselines3 PPO (discrete action space). The trained agent serves
as the starting point for mask network training and RICE refinement.

Usage:
    python experiments/selfish_mining/train_target.py \
        --total-steps 1000000 --seed 42 --device cuda

The trained model, normalization stats, and metadata are saved to
`./trained_agents/selfish_mining/` by default.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import gym
import numpy as np
import torch
import yaml

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Stable-Baselines3 imports (optional but recommended)
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable_baselines3 not installed. Will use custom PPO fallback.")

from rice.utils import set_seed, evaluate_policy

# Import selfish mining environment
from experiments.selfish_mining.env import (
    make_env as make_sm_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train target PPO agent on Selfish Mining environment"
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="SelfishMining-v0",
        help="Environment name (default: SelfishMining-v0)",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_agents/selfish_mining",
        help="Directory to save trained model and metadata",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for final evaluation (default: 10)",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50000,
        help="Evaluation frequency in timesteps (default: 50000)",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Attacker hash rate fraction (default: 0.35)",
    )
    parser.add_argument(
        "--gamma-sm",
        type=float,
        default=0.5,
        help="Honest adoption ratio gamma (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level: 0 (silent), 1 (info), 2 (debug)",
    )
    return parser.parse_args()


def load_config(
    env_name: str, config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and merge configuration from default and environment-specific YAML files.

    Args:
        env_name: Environment name (e.g., 'SelfishMining-v0')
        config_path: Optional path to a custom YAML config to merge on top

    Returns:
        Merged configuration dictionary
    """
    config = {}

    # Load default mask config (contains PPO hyperparameters)
    default_mask_path = (
        _project_root / "configs" / "default_mask.yaml"
    )
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            default_cfg = yaml.safe_load(f)
            if default_cfg:
                config.update(default_cfg)

    # Load default refine config (contains additional PPO settings)
    default_refine_path = (
        _project_root / "configs" / "default_refine.yaml"
    )
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            default_cfg = yaml.safe_load(f)
            if default_cfg:
                # Merge refine-specific keys
                for key in ["ppo", "refine_ppo", "policy", "training"]:
                    if key in default_cfg:
                        if key not in config:
                            config[key] = {}
                        if isinstance(default_cfg[key], dict):
                            config[key].update(default_cfg[key])

    # Load environment-specific config
    env_config_path = (
        _project_root / "configs" / "env_specific" / "selfish_mining.yaml"
    )
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_cfg = yaml.safe_load(f)
            if env_cfg:
                # Deep merge: env-specific overrides defaults
                for key, value in env_cfg.items():
                    if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                        config[key].update(value)
                    else:
                        config[key] = value

    # Load custom config if provided
    if config_path is not None:
        custom_path = Path(config_path)
        if custom_path.exists():
            with open(custom_path, "r") as f:
                custom_cfg = yaml.safe_load(f)
                if custom_cfg:
                    for key, value in custom_cfg.items():
                        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                            config[key].update(value)
                        else:
                            config[key] = value

    return config


def make_env(
    env_name: str = "SelfishMining-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    alpha: float = 0.35,
    gamma_sm: float = 0.5,
) -> gym.Env:
    """
    Create a single selfish mining environment wrapped with Monitor.

    Args:
        env_name: Environment name
        seed: Random seed
        max_episode_steps: Maximum steps per episode (None = env default)
        alpha: Attacker hash rate fraction
        gamma_sm: Honest adoption ratio

    Returns:
        Wrapped gym environment
    """
    env = make_sm_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        alpha=alpha,
        gamma=gamma_sm,
    )
    env = Monitor(env)
    return env


def make_vec_env(
    env_name: str = "SelfishMining-v0",
    seed: int = 42,
    n_envs: int = 4,
    max_episode_steps: Optional[int] = None,
    alpha: float = 0.35,
    gamma_sm: float = 0.5,
) -> DummyVecEnv:
    """
    Create a vectorized selfish mining environment.

    Args:
        env_name: Environment name
        seed: Base random seed
        n_envs: Number of parallel environments
        max_episode_steps: Maximum steps per episode
        alpha: Attacker hash rate fraction
        gamma_sm: Honest adoption ratio

    Returns:
        DummyVecEnv with Monitor-wrapped environments
    """

    def _make_env(rank: int):
        def _init():
            env_seed = seed + rank
            env = make_sm_env(
                env_name=env_name,
                seed=env_seed,
                max_episode_steps=max_episode_steps,
                alpha=alpha,
                gamma=gamma_sm,
            )
            env = Monitor(env)
            return env

        return _init

    envs = [_make_env(i) for i in range(n_envs)]
    return DummyVecEnv(envs)


def train_target_agent(
    env_name: str = "SelfishMining-v0",
    total_steps: int = 1_000_000,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "./trained_agents/selfish_mining",
    config: Optional[Dict[str, Any]] = None,
    eval_episodes: int = 10,
    eval_freq: int = 50000,
    n_envs: int = 4,
    alpha: float = 0.35,
    gamma_sm: float = 0.5,
    verbose: int = 1,
) -> Tuple[Any, float]:
    """
    Train a PPO agent on the selfish mining environment.

    Args:
        env_name: Environment name
        total_steps: Total training timesteps
        seed: Random seed
        device: Device for training ('cuda' or 'cpu')
        output_dir: Directory to save trained model
        config: Configuration dictionary (loaded from YAML if None)
        eval_episodes: Number of evaluation episodes
        eval_freq: Evaluation frequency in timesteps
        n_envs: Number of parallel environments
        alpha: Attacker hash rate fraction
        gamma_sm: Honest adoption ratio
        verbose: Verbosity level

    Returns:
        Tuple of (trained PPO model, final mean evaluation reward)
    """
    if not HAS_SB3:
        raise ImportError(
            "stable_baselines3 is required for training. "
            "Install with: pip install stable-baselines3"
        )

    # Load config if not provided
    if config is None:
        config = load_config(env_name)

    # Set random seed
    set_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract hyperparameters from config
    ppo_cfg = config.get("ppo", {})
    policy_cfg = config.get("policy", {})
    training_cfg = config.get("training", {})

    learning_rate = ppo_cfg.get("learning_rate", 3e-4)
    gamma = ppo_cfg.get("gamma", 0.99)
    gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
    clip_range = ppo_cfg.get("clip_epsilon", 0.2)
    ent_coef = ppo_cfg.get("entropy_coef", 0.01)
    vf_coef = ppo_cfg.get("value_loss_coef", 0.5)
    max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
    n_steps = training_cfg.get("steps_per_iteration", 2048)
    batch_size = ppo_cfg.get("batch_size", 64)
    n_epochs = ppo_cfg.get("ppo_epochs", 10)

    hidden_sizes = policy_cfg.get("hidden_sizes", [128, 128])
    activation = policy_cfg.get("activation", "tanh")

    # Build policy kwargs for SB3
    if activation == "tanh":
        activation_fn = torch.nn.Tanh
    elif activation == "relu":
        activation_fn = torch.nn.ReLU
    else:
        activation_fn = torch.nn.Tanh

    policy_kwargs = dict(
        net_arch=dict(
            pi=hidden_sizes,
            vf=hidden_sizes,
        ),
        activation_fn=activation_fn,
    )

    # Create vectorized environment
    max_episode_steps = config.get("max_episode_steps", 100)
    env = make_vec_env(
        env_name=env_name,
        seed=seed,
        n_envs=n_envs,
        max_episode_steps=max_episode_steps,
        alpha=alpha,
        gamma_sm=gamma_sm,
    )

    # Wrap with VecNormalize for observation/reward normalization
    # Note: for discrete action spaces, we still normalize observations
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )

    # Create evaluation environment
    eval_env = make_vec_env(
        env_name=env_name,
        seed=seed + 1000,
        n_envs=1,
        max_episode_steps=max_episode_steps,
        alpha=alpha,
        gamma_sm=gamma_sm,
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        training=False,
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose,
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=seed,
        device=device,
        tensorboard_log=output_dir,
    )

    # Train
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Training PPO agent on {env_name}")
        print(f"Total steps: {total_steps}")
        print(f"Environment: {n_envs} parallel envs")
        print(f"Policy: MLP {hidden_sizes}, {activation}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

    start_time = time.time()
    model.learn(total_timesteps=total_steps, callback=eval_callback)
    training_time = time.time() - start_time

    # Save final model
    model_path = os.path.join(output_dir, f"{env_name}_ppo_final.zip")
    model.save(model_path)

    # Save VecNormalize statistics
    vecnorm_path = os.path.join(output_dir, f"{env_name}_vecnormalize.pkl")
    env.save(vecnorm_path)

    # Final evaluation
    if verbose >= 1:
        print(f"\nRunning final evaluation ({eval_episodes} episodes)...")

    # Create a single env for evaluation using rice.utils.evaluate_policy
    eval_env_single = make_env(
        env_name=env_name,
        seed=seed + 2000,
        max_episode_steps=max_episode_steps,
        alpha=alpha,
        gamma_sm=gamma_sm,
    )

    # Wrap model into a policy function for evaluation
    def policy_fn(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return action

    eval_results = evaluate_policy(
        eval_env_single,
        policy_fn,
        num_episodes=eval_episodes,
        max_steps=max_episode_steps or 100,
        deterministic=True,
        verbose=verbose >= 1,
    )
    final_mean_reward = eval_results["mean_reward"]

    # Save metadata
    metadata = {
        "env_name": env_name,
        "total_steps": total_steps,
        "training_time_seconds": training_time,
        "final_mean_reward": float(final_mean_reward),
        "final_std_reward": float(eval_results["std_reward"]),
        "seed": seed,
        "device": device,
        "n_envs": n_envs,
        "alpha": alpha,
        "gamma_sm": gamma_sm,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "hidden_sizes": hidden_sizes,
            "activation": activation,
        },
    }
    metadata_path = os.path.join(output_dir, f"{env_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Time: {training_time:.1f}s ({training_time/60:.1f} min)")
        print(f"  Final mean reward: {final_mean_reward:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"  Model saved to: {model_path}")
        print(f"  VecNormalize saved to: {vecnorm_path}")
        print(f"  Metadata saved to: {metadata_path}")
        print(f"{'='*60}\n")

    eval_env_single.close()
    eval_env.close()
    env.close()

    return model, final_mean_reward


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.env_name, args.config)

    # Train the agent
    model, final_reward = train_target_agent(
        env_name=args.env_name,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        config=config,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        n_envs=args.n_envs,
        alpha=args.alpha,
        gamma_sm=args.gamma_sm,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())