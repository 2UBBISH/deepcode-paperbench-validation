#!/usr/bin/env python3
"""
Train a target PPO agent on MuJoCo environments.

This script trains an initial PPO policy on standard MuJoCo tasks (Hopper, Walker2d,
Reacher, HalfCheetah) using Stable-Baselines3. The trained agent serves as the target
policy for subsequent mask network training and RICE refinement.

Usage:
    python experiments/mujoco/train_target.py --env Hopper-v4 --total_steps 1000000
    python experiments/mujoco/train_target.py --env Walker2d-v4 --total_steps 1000000
    python experiments/mujoco/train_target.py --env Reacher-v4 --total_steps 500000
    python experiments/mujoco/train_target.py --env HalfCheetah-v4 --total_steps 1000000
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch
import gym

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rice.utils import set_seed, evaluate_policy


def parse_args():
    parser = argparse.ArgumentParser(description="Train target PPO agent on MuJoCo")
    parser.add_argument("--env", type=str, default="Hopper-v4",
                        choices=["Hopper-v4", "Walker2d-v4", "Reacher-v4", "HalfCheetah-v4"],
                        help="MuJoCo environment name")
    parser.add_argument("--total_steps", type=int, default=1_000_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for training")
    parser.add_argument("--output_dir", type=str, default="./trained_agents",
                        help="Directory to save trained agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to environment-specific YAML config")
    parser.add_argument("--eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=progress, 2=debug)")
    return parser.parse_args()


def load_config(env_name: str, config_path: str = None) -> dict:
    """Load configuration, merging defaults with env-specific overrides."""
    config = {}

    # Load default configs
    default_mask_path = os.path.join(
        os.path.dirname(__file__), "../../configs/default_mask.yaml"
    )
    if os.path.exists(default_mask_path):
        with open(default_mask_path, "r") as f:
            config.update(yaml.safe_load(f) or {})

    # Load env-specific config
    if config_path is None:
        env_short = env_name.lower().replace("-v4", "").replace("-v3", "").replace("-v2", "")
        env_config_path = os.path.join(
            os.path.dirname(__file__), f"../../configs/env_specific/{env_short}.yaml"
        )
    else:
        env_config_path = config_path

    if os.path.exists(env_config_path):
        with open(env_config_path, "r") as f:
            env_specific = yaml.safe_load(f) or {}
            config.update(env_specific)

    return config


def make_env(env_name: str, seed: int = 42) -> gym.Env:
    """Create a MuJoCo environment with appropriate wrappers."""
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train_target_agent(
    env_name: str,
    total_steps: int = 1_000_000,
    seed: int = 42,
    device: str = "cuda",
    output_dir: str = "./trained_agents",
    config: dict = None,
    eval_episodes: int = 10,
    verbose: int = 1,
):
    """
    Train a PPO agent on the specified MuJoCo environment using Stable-Baselines3.

    Args:
        env_name: Gym environment name (e.g., "Hopper-v4")
        total_steps: Total training timesteps
        seed: Random seed
        device: Device for training ("cuda" or "cpu")
        output_dir: Directory to save the trained model
        config: Configuration dictionary (optional)
        eval_episodes: Number of episodes for evaluation
        verbose: Verbosity level

    Returns:
        Tuple of (trained_model, final_mean_reward)
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        raise ImportError(
            "Stable-Baselines3 is required. Install with: pip install stable-baselines3"
        )

    set_seed(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract PPO hyperparameters from config or use defaults
    if config is None:
        config = {}

    ppo_config = config.get("ppo", {})
    learning_rate = ppo_config.get("learning_rate", 3e-4)
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_range = ppo_config.get("clip_epsilon", 0.2)
    ent_coef = ppo_config.get("entropy_coef", 0.0)
    vf_coef = ppo_config.get("value_loss_coef", 0.5)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    batch_size = ppo_config.get("batch_size", 64)
    n_steps = config.get("training", {}).get("steps_per_iteration", 2048)
    n_epochs = ppo_config.get("ppo_epochs", 10)

    # Policy architecture
    policy_config = config.get("policy", {})
    policy_hidden_sizes = policy_config.get("hidden_sizes", [64, 64])
    activation = policy_config.get("activation", "tanh")

    # Build policy kwargs
    if activation == "tanh":
        activation_fn = torch.nn.Tanh
    elif activation == "relu":
        activation_fn = torch.nn.ReLU
    else:
        activation_fn = torch.nn.Tanh

    policy_kwargs = dict(
        net_arch=dict(
            pi=policy_hidden_sizes,
            vf=policy_hidden_sizes,
        ),
        activation_fn=activation_fn,
    )

    # Create vectorized environment
    def _make_env():
        env = gym.make(env_name)
        env = Monitor(env)
        return env

    env = DummyVecEnv([_make_env])
    env.seed(seed)

    # Optionally wrap with VecNormalize
    use_vec_normalize = config.get("training", {}).get("use_vec_normalize", True)
    if use_vec_normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
    )

    # Setup evaluation callback
    eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_name))])
    if use_vec_normalize:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(n_steps, 10000),
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )

    # Train
    print(f"Training PPO on {env_name} for {total_steps} steps...")
    model.learn(total_timesteps=total_steps, callback=eval_callback)

    # Save final model
    model_path = os.path.join(output_dir, f"{env_name}_ppo_final")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Also save the VecNormalize statistics if used
    if use_vec_normalize:
        norm_path = os.path.join(output_dir, f"{env_name}_vecnormalize.pkl")
        env.save(norm_path)
        print(f"VecNormalize stats saved to {norm_path}")

    # Final evaluation
    print("Running final evaluation...")
    final_env = gym.make(env_name)
    final_env.reset(seed=seed + 1000)

    # Extract the policy function for evaluation
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    eval_results = evaluate_policy(
        final_env, policy_fn, num_episodes=eval_episodes, max_steps=1000, deterministic=True
    )
    print(f"Final evaluation: mean_reward={eval_results['mean_reward']:.2f} "
          f"± {eval_results['std_reward']:.2f}")

    final_env.close()

    return model, eval_results["mean_reward"]


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.env, args.config)

    # Override with command-line arguments
    if args.total_steps:
        if "training" not in config:
            config["training"] = {}
        config["training"]["total_steps"] = args.total_steps

    # Train
    model, final_reward = train_target_agent(
        env_name=args.env,
        total_steps=args.total_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        config=config,
        eval_episodes=args.eval_episodes,
        verbose=args.verbose,
    )

    print(f"\nTraining complete! Final mean reward: {final_reward:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())