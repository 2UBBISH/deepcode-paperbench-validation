#!/usr/bin/env python3
"""
Train a target PPO agent on the MetaDrive autonomous driving environment.

This script trains a PPO agent using Stable-Baselines3 on MetaDrive's "Macro-v1"
scenario. The trained agent serves as the initial policy for subsequent mask
network training and RICE refinement.

Usage:
    python train_target.py --env "MetaDrive-Macro-v1" --total_steps 2000000
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rice.utils import set_seed, evaluate_policy

# Import autonomous driving environment
from experiments.autonomous_driving.env import (
    make_env as make_ad_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Install with: pip install stable-baselines3")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train target PPO agent on MetaDrive autonomous driving environment"
    )
    parser.add_argument(
        "--env", type=str, default="MetaDrive-Macro-v1",
        help="Environment name (default: MetaDrive-Macro-v1)"
    )
    parser.add_argument(
        "--total_steps", type=int, default=2_000_000,
        help="Total training timesteps (default: 2,000,000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./trained_agents/autonomous_driving",
        help="Directory to save trained models (default: ./trained_agents/autonomous_driving)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML config file (default: auto-detect from configs/)"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10,
        help="Number of episodes for final evaluation (default: 10)"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50000,
        help="Evaluation frequency during training (default: 50000)"
    )
    parser.add_argument(
        "--n_envs", type=int, default=1,
        help="Number of parallel environments (default: 1; MetaDrive may not support >1)"
    )
    parser.add_argument(
        "--use_sparse_reward", action="store_true",
        help="Use sparse reward variant of MetaDrive"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level (0: silent, 1: info, 2: debug)"
    )
    return parser.parse_args()


def load_config(env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and merge default and environment-specific YAML configurations.

    Args:
        env_name: Name of the environment (e.g., "MetaDrive-Macro-v1")
        config_path: Optional path to a custom config file

    Returns:
        Merged configuration dictionary
    """
    config = {}

    # Load default mask config
    default_mask_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default_mask.yaml"
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            config.update(yaml.safe_load(f) or {})

    # Load default refine config
    default_refine_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            refine_cfg = yaml.safe_load(f) or {}
            # Merge refine config under appropriate keys
            for key in refine_cfg:
                if key not in config:
                    config[key] = refine_cfg[key]

    # Load environment-specific config
    env_config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "env_specific" / "autonomous_driving.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_specific = yaml.safe_load(f) or {}
            # Deep merge: env-specific overrides defaults
            for key, value in env_specific.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value

    # Load custom config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom = yaml.safe_load(f) or {}
            for key, value in custom.items():
                if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                    config[key].update(value)
                else:
                    config[key] = value

    return config


def make_env(
    env_name: str = "MetaDrive-Macro-v1",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_sparse_reward: bool = False,
) -> gym.Env:
    """
    Create a MetaDrive environment for training.

    Args:
        env_name: Environment name
        seed: Random seed
        max_episode_steps: Maximum steps per episode
        use_sparse_reward: Whether to use sparse rewards

    Returns:
        Gym environment
    """
    env = make_ad_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_sparse_reward=use_sparse_reward,
    )
    env = Monitor(env)
    return env


def make_vec_env(
    env_name: str = "MetaDrive-Macro-v1",
    seed: int = 42,
    n_envs: int = 1,
    max_episode_steps: Optional[int] = None,
    use_sparse_reward: bool = False,
) -> DummyVecEnv:
    """
    Create a vectorized MetaDrive environment.

    Args:
        env_name: Environment name
        seed: Random seed
        n_envs: Number of parallel environments
        max_episode_steps: Maximum steps per episode
        use_sparse_reward: Whether to use sparse rewards

    Returns:
        Vectorized environment
    """
    def _make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            env_seed = seed + rank
            env = make_env(
                env_name=env_name,
                seed=env_seed,
                max_episode_steps=max_episode_steps,
                use_sparse_reward=use_sparse_reward,
            )
            return env
        return _init

    env = DummyVecEnv([_make_env(i) for i in range(n_envs)])
    return env


def train_target_agent(
    env_name: str = "MetaDrive-Macro-v1",
    total_steps: int = 2_000_000,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "./trained_agents/autonomous_driving",
    config: Optional[Dict[str, Any]] = None,
    eval_episodes: int = 10,
    eval_freq: int = 50000,
    n_envs: int = 1,
    use_sparse_reward: bool = False,
    verbose: int = 1,
) -> Tuple[Any, float]:
    """
    Train a target PPO agent on the MetaDrive autonomous driving environment.

    Args:
        env_name: Environment name
        total_steps: Total training timesteps
        seed: Random seed
        device: Device to use
        output_dir: Directory to save models
        config: Configuration dictionary (loaded from YAML if None)
        eval_episodes: Number of evaluation episodes
        eval_freq: Evaluation frequency during training
        n_envs: Number of parallel environments
        use_sparse_reward: Whether to use sparse rewards
        verbose: Verbosity level

    Returns:
        Tuple of (trained PPO model, final mean reward)
    """
    if not HAS_SB3:
        raise ImportError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        )

    # Set seeds
    set_seed(seed)

    # Load config if not provided
    if config is None:
        config = load_config(env_name)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract hyperparameters from config
    ppo_config = config.get("ppo", {})
    training_config = config.get("training", {})
    policy_config = config.get("policy", {})

    learning_rate = ppo_config.get("learning_rate", 3e-4)
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_range = ppo_config.get("clip_epsilon", 0.2)
    ent_coef = ppo_config.get("entropy_coef", 0.0)
    vf_coef = ppo_config.get("value_loss_coef", 0.5)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    batch_size = ppo_config.get("batch_size", 64)
    n_epochs = ppo_config.get("ppo_epochs", 10)
    n_steps = training_config.get("steps_per_iteration", 2048)

    # Policy architecture
    policy_hidden_sizes = policy_config.get("hidden_sizes", [256, 256])
    activation = policy_config.get("activation", "tanh")

    # Create environment
    max_episode_steps = config.get("max_episode_steps", None)
    if max_episode_steps is None:
        # MetaDrive default is typically 1000-2000
        max_episode_steps = 2000

    if n_envs > 1:
        env = make_vec_env(
            env_name=env_name,
            seed=seed,
            n_envs=n_envs,
            max_episode_steps=max_episode_steps,
            use_sparse_reward=use_sparse_reward,
        )
    else:
        env = make_env(
            env_name=env_name,
            seed=seed,
            max_episode_steps=max_episode_steps,
            use_sparse_reward=use_sparse_reward,
        )
        env = DummyVecEnv([lambda: env])

    # Wrap with VecNormalize for observation/reward normalization
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )

    # Create evaluation environment
    eval_env = make_env(
        env_name=env_name,
        seed=seed + 1000,
        max_episode_steps=max_episode_steps,
        use_sparse_reward=use_sparse_reward,
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Don't normalize rewards for evaluation
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
    )

    # Build policy kwargs for custom architecture
    policy_kwargs = {
        "net_arch": {
            "pi": list(policy_hidden_sizes),
            "vf": list(policy_hidden_sizes),
        },
        "activation_fn": {
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
        }.get(activation, torch.nn.Tanh),
    }

    # Create PPO model
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"Training target PPO agent on {env_name}")
        print(f"Total steps: {total_steps:,}")
        print(f"Learning rate: {learning_rate}")
        print(f"Policy hidden sizes: {policy_hidden_sizes}")
        print(f"Device: {device}")
        print(f"{'='*60}\n")

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
        seed=seed,
        device=device,
        verbose=verbose,
        tensorboard_log=os.path.join(output_dir, "tensorboard"),
    )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval_logs"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    # Train the agent
    start_time = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
        progress_bar=(verbose >= 1),
    )
    training_time = time.time() - start_time

    # Save the final model
    model_path = os.path.join(output_dir, f"{env_name}_ppo_final.zip")
    model.save(model_path)
    if verbose >= 1:
        print(f"\nModel saved to: {model_path}")

    # Save VecNormalize statistics
    vecnorm_path = os.path.join(output_dir, f"{env_name}_vecnormalize.pkl")
    env.save(vecnorm_path)
    if verbose >= 1:
        print(f"VecNormalize stats saved to: {vecnorm_path}")

    # Final evaluation
    if verbose >= 1:
        print("\nRunning final evaluation...")

    # For final evaluation, use the eval env
    eval_env_norm = VecNormalize.load(vecnorm_path, eval_env)
    eval_env_norm.training = False
    eval_env_norm.norm_reward = False

    all_rewards = []
    for ep in range(eval_episodes):
        obs = eval_env_norm.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env_norm.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            ep_length += 1
        all_rewards.append(ep_reward)

    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))

    if verbose >= 1:
        print(f"Final evaluation over {eval_episodes} episodes:")
        print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f} min)")

    # Save training metadata
    metadata = {
        "env_name": env_name,
        "total_steps": total_steps,
        "seed": seed,
        "device": device,
        "training_time": training_time,
        "final_mean_reward": mean_reward,
        "final_std_reward": std_reward,
        "all_rewards": all_rewards,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "n_steps": n_steps,
            "policy_hidden_sizes": list(policy_hidden_sizes),
            "activation": activation,
        },
        "config": config,
    }

    metadata_path = os.path.join(output_dir, f"{env_name}_training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    if verbose >= 1:
        print(f"Training metadata saved to: {metadata_path}")

    return model, mean_reward


def main():
    """Main entry point."""
    args = parse_args()

    if not HAS_SB3:
        print("ERROR: stable-baselines3 is required. Install with: pip install stable-baselines3")
        sys.exit(1)

    # Load config
    config = load_config(args.env, args.config)

    # Train the agent
    model, mean_reward = train_target_agent(
        env_name=args.env,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        config=config,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        n_envs=args.n_envs,
        use_sparse_reward=args.use_sparse_reward,
        verbose=args.verbose,
    )

    print(f"\nTraining complete! Final mean reward: {mean_reward:.2f}")


if __name__ == "__main__":
    main()