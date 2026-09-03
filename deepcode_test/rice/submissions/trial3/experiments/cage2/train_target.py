#!/usr/bin/env python3
"""
Train a target PPO agent on the CAGE Challenge 2 environment.

This script trains a PPO agent using Stable-Baselines3 on the CAGE2
cybersecurity domain. The trained agent serves as the initial policy for
subsequent mask network training and RICE refinement.

Usage:
    python experiments/cage2/train_target.py --env CAGE2-v0 --total_steps 2000000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rice.utils import set_seed, evaluate_policy

# Optional Stable-Baselines3 import
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable_baselines3 not installed. Install with: pip install stable-baselines3")

# Import CAGE2 environment
from experiments.cage2.env import make_env as make_cage2_env
from experiments.cage2.env import get_state_dim, get_action_dim, is_discrete_action


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train target PPO agent on CAGE Challenge 2"
    )
    parser.add_argument(
        "--env", type=str, default="CAGE2-v0",
        help="Environment name (default: CAGE2-v0)"
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
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./trained_agents/cage2",
        help="Output directory for trained model (default: ./trained_agents/cage2)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML config file"
    )
    parser.add_argument(
        "--eval_episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=50000,
        help="Evaluation frequency in steps (default: 50000)"
    )
    parser.add_argument(
        "--n_envs", type=int, default=1,
        help="Number of parallel environments (default: 1)"
    )
    parser.add_argument(
        "--use_real_env", action="store_true",
        help="Use real CybORG environment instead of simulated fallback"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level (0=none, 1=info, 2=debug)"
    )
    return parser.parse_args()


def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively update a dictionary with another."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    env_name: str,
    config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and merge default and environment-specific YAML configurations.

    Args:
        env_name: Environment name (e.g., 'CAGE2-v0')
        config_path: Optional path to custom config file

    Returns:
        Merged configuration dictionary
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # Load default mask config
    default_mask_path = project_root / "configs" / "default_mask.yaml"
    config = {}
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load default refine config
    default_refine_path = project_root / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            refine_config = yaml.safe_load(f) or {}
            config = _deep_update(config, refine_config)

    # Load environment-specific config
    env_config_path = project_root / "configs" / "env_specific" / "cage2.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
            config = _deep_update(config, env_config)

    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f) or {}
            config = _deep_update(config, custom_config)

    return config


def make_env(
    env_name: str = "CAGE2-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False
) -> gym.Env:
    """
    Create a single CAGE2 environment wrapped with Monitor.

    Args:
        env_name: Environment name
        seed: Random seed
        max_episode_steps: Maximum episode steps
        use_real_env: Whether to use real CybORG environment

    Returns:
        Wrapped gym environment
    """
    env = make_cage2_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env
    )
    env = Monitor(env)
    return env


def make_vec_env(
    env_name: str = "CAGE2-v0",
    seed: int = 42,
    n_envs: int = 1,
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False
) -> DummyVecEnv:
    """
    Create a vectorized CAGE2 environment.

    Args:
        env_name: Environment name
        seed: Random seed
        n_envs: Number of parallel environments
        max_episode_steps: Maximum episode steps
        use_real_env: Whether to use real CybORG environment

    Returns:
        DummyVecEnv wrapping multiple environments
    """
    def _make_env(rank: int) -> gym.Env:
        env_seed = seed + rank
        return make_env(
            env_name=env_name,
            seed=env_seed,
            max_episode_steps=max_episode_steps,
            use_real_env=use_real_env
        )

    env = DummyVecEnv([lambda: _make_env(i) for i in range(n_envs)])
    return env


def train_target_agent(
    env_name: str = "CAGE2-v0",
    total_steps: int = 2_000_000,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "./trained_agents/cage2",
    config: Optional[Dict[str, Any]] = None,
    eval_episodes: int = 10,
    eval_freq: int = 50000,
    n_envs: int = 1,
    use_real_env: bool = False,
    verbose: int = 1
) -> Tuple[Any, float]:
    """
    Train a PPO agent on the CAGE2 environment.

    Args:
        env_name: Environment name
        total_steps: Total training timesteps
        seed: Random seed
        device: Device for training ('cuda' or 'cpu')
        output_dir: Directory to save trained model
        config: Configuration dictionary (loaded from YAML if None)
        eval_episodes: Number of evaluation episodes
        eval_freq: Evaluation frequency in steps
        n_envs: Number of parallel environments
        use_real_env: Whether to use real CybORG environment
        verbose: Verbosity level

    Returns:
        Tuple of (trained PPO model, final mean reward)
    """
    if not HAS_SB3:
        raise ImportError(
            "stable_baselines3 is required for training. "
            "Install with: pip install stable-baselines3"
        )

    # Load config if not provided
    if config is None:
        config = load_config(env_name)

    # Set seed
    set_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get environment info
    temp_env = make_cage2_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=config.get("max_episode_steps", 100),
        use_real_env=use_real_env
    )
    state_dim = get_state_dim(temp_env)
    action_dim = get_action_dim(temp_env)
    discrete = is_discrete_action(temp_env)
    max_episode_steps = config.get("max_episode_steps", 100)
    temp_env.close()

    if verbose:
        print(f"Environment: {env_name}")
        print(f"State dim: {state_dim}, Action dim: {action_dim}, Discrete: {discrete}")
        print(f"Max episode steps: {max_episode_steps}")
        print(f"Total training steps: {total_steps}")
        print(f"Device: {device}")

    # Extract PPO hyperparameters from config
    ppo_config = config.get("ppo", {})
    learning_rate = ppo_config.get("learning_rate", 3e-4)
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_range = ppo_config.get("clip_epsilon", 0.2)
    ent_coef = ppo_config.get("entropy_coef", 0.01)
    vf_coef = ppo_config.get("value_loss_coef", 0.5)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    batch_size = ppo_config.get("batch_size", 64)
    n_epochs = ppo_config.get("ppo_epochs", 10)

    # Policy architecture
    policy_config = config.get("policy", {})
    policy_hidden_sizes = policy_config.get("hidden_sizes", [128, 128])
    activation = policy_config.get("activation", "tanh")

    # Build policy kwargs
    policy_kwargs = {
        "net_arch": dict(
            pi=policy_hidden_sizes,
            vf=policy_hidden_sizes
        ),
        "activation_fn": torch.nn.Tanh if activation == "tanh" else torch.nn.ReLU,
    }

    # Create vectorized environment
    vec_env = make_vec_env(
        env_name=env_name,
        seed=seed,
        n_envs=n_envs,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env
    )

    # Wrap with VecNormalize
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma
    )

    # Create evaluation environment
    eval_env = make_vec_env(
        env_name=env_name,
        seed=seed + 1000,
        n_envs=1,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env
    )
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        training=False
    )

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=verbose
    )

    # Create PPO model
    if discrete:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=config.get("training", {}).get("steps_per_iteration", 2048),
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
            verbose=verbose
        )
    else:
        # CAGE2 actions are discrete, but handle continuous case
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=config.get("training", {}).get("steps_per_iteration", 2048),
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
            verbose=verbose
        )

    # Train
    start_time = time.time()
    if verbose:
        print(f"\nStarting training for {total_steps} steps...")

    model.learn(
        total_timesteps=total_steps,
        callback=eval_callback,
        progress_bar=verbose > 0
    )

    training_time = time.time() - start_time

    # Save model
    model_path = os.path.join(output_dir, f"{env_name}_ppo_final.zip")
    model.save(model_path)
    if verbose:
        print(f"Model saved to {model_path}")

    # Save VecNormalize statistics
    vec_norm_path = os.path.join(output_dir, f"{env_name}_vecnormalize.pkl")
    vec_env.save(vec_norm_path)
    if verbose:
        print(f"VecNormalize stats saved to {vec_norm_path}")

    # Save metadata
    metadata = {
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discrete_action": discrete,
        "max_episode_steps": max_episode_steps,
        "total_steps": total_steps,
        "training_time": training_time,
        "seed": seed,
        "device": device,
        "ppo_config": {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        },
        "policy_hidden_sizes": policy_hidden_sizes,
        "activation": activation,
    }
    metadata_path = os.path.join(output_dir, f"{env_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"Metadata saved to {metadata_path}")

    # Final evaluation
    if verbose:
        print("\nRunning final evaluation...")

    # Create a single env for evaluation
    final_eval_env = make_env(
        env_name=env_name,
        seed=seed + 2000,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env
    )

    # Wrap with VecNormalize for evaluation
    from stable_baselines3.common.vec_env import DummyVecEnv as _DummyVecEnv
    final_vec_env = _DummyVecEnv([lambda: final_eval_env])
    final_vec_env = VecNormalize(
        final_vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=gamma,
        training=False
    )
    # Copy running stats from training env
    final_vec_env.obs_rms = vec_env.obs_rms

    # Evaluate using model
    all_rewards = []
    for ep in range(eval_episodes):
        obs = final_vec_env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = final_vec_env.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            ep_steps += 1
        all_rewards.append(ep_reward)

    mean_reward = float(np.mean(all_rewards))
    std_reward = float(np.std(all_rewards))

    if verbose:
        print(f"Final evaluation: mean_reward={mean_reward:.4f}, std_reward={std_reward:.4f}")

    # Save evaluation results
    eval_results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "all_rewards": [float(r) for r in all_rewards],
        "num_episodes": eval_episodes,
    }
    eval_path = os.path.join(output_dir, f"{env_name}_eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    # Clean up
    vec_env.close()
    eval_env.close()
    final_vec_env.close()

    return model, mean_reward


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not HAS_SB3:
        print("ERROR: stable_baselines3 is required. Install with: pip install stable-baselines3")
        return 1

    # Load config
    config = load_config(args.env, args.config)

    # Override with CLI arguments
    if args.total_steps:
        if "training" not in config:
            config["training"] = {}
        config["training"]["total_steps"] = args.total_steps

    # Train
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
        use_real_env=args.use_real_env,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"\nTraining complete! Final mean reward: {mean_reward:.4f}")
        print(f"Model saved to: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())