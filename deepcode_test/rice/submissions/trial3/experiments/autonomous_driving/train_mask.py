#!/usr/bin/env python3
"""
Train Mask Network for Autonomous Driving (MetaDrive) Environment.

This script loads a pre-trained target PPO agent for MetaDrive, wraps it into
a policy function, creates a MaskNetwork and MaskNetworkTrainer, runs the
training loop (PPO with intrinsic reward α·I(a^e=1)), computes fidelity, and
saves all results for subsequent RICE refinement.

Usage:
    python experiments/autonomous_driving/train_mask.py \
        --env MetaDrive-Macro-v1 \
        --model-dir ./trained_agents/autonomous_driving \
        --output-dir ./mask_agents/autonomous_driving \
        --total-steps 300000 \
        --alpha 0.0001 \
        --seed 42 \
        --device cuda
"""

import argparse
import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
import gym

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rice.mask_net import (
    MaskNetwork,
    MaskNetworkTrainer,
    PerturbedPolicy,
    compute_fidelity,
    compute_fidelity_from_env,
    train_mask_network,
)
from rice.utils import evaluate_policy, set_seed, to_numpy
from rice.env_wrappers import make_state_saveable, StateSaveWrapper

# Import autonomous driving environment
from experiments.autonomous_driving.env import (
    make_env as make_ad_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
)

# Check for Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not installed. Some functionality may be limited.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Mask Network for Autonomous Driving (MetaDrive)"
    )
    parser.add_argument(
        "--env", type=str, default="MetaDrive-Macro-v1",
        help="Environment name (default: MetaDrive-Macro-v1)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents/autonomous_driving",
        help="Directory containing pre-trained target agent"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./mask_agents/autonomous_driving",
        help="Directory to save trained mask network and results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML configuration file"
    )
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Total training steps (overrides config)"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Intrinsic reward coefficient (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu, overrides config)"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress verbose output"
    )
    return parser.parse_args()


def load_config(
    env_name: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and merge default mask config with environment-specific overrides.

    Args:
        env_name: Environment name (e.g., "MetaDrive-Macro-v1")
        config_path: Optional path to custom YAML config

    Returns:
        Merged configuration dictionary
    """
    # Load default mask config
    default_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default_mask.yaml"
    config = {}
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load environment-specific config
    env_config_name = "autonomous_driving.yaml"
    env_config_path = (
        Path(__file__).resolve().parent.parent.parent
        / "configs" / "env_specific" / env_config_name
    )
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
        # Deep merge: env-specific overrides default
        config = _deep_merge(config, env_config)

    # Load custom config if provided
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, custom_config)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """
    Load a pre-trained target PPO agent from disk.

    Supports Stable-Baselines3 .zip models and raw PyTorch .pt checkpoints.

    Args:
        env_name: Environment name
        model_dir: Directory containing the trained model
        device: Device to load model on

    Returns:
        Tuple of (model, vec_normalize) where vec_normalize may be None
    """
    model_path = Path(model_dir)
    vec_normalize = None

    # Try loading SB3 PPO model
    sb3_path = model_path / f"{env_name}_ppo_final.zip"
    if sb3_path.exists() and HAS_SB3:
        model = PPO.load(str(sb3_path), device=device)
        # Try loading VecNormalize stats
        vn_path = model_path / f"{env_name}_vecnormalize.pkl"
        if vn_path.exists():
            with open(vn_path, "rb") as f:
                vec_normalize = pickle.load(f)
        return model, vec_normalize

    # Try loading raw PyTorch checkpoint
    pt_path = model_path / f"{env_name}_policy.pt"
    if pt_path.exists():
        model = torch.load(pt_path, map_location=device)
        return model, vec_normalize

    # Try generic names
    for name in ["ppo_final.zip", "target_policy.zip", "model.zip"]:
        candidate = model_path / name
        if candidate.exists() and HAS_SB3:
            model = PPO.load(str(candidate), device=device)
            return model, vec_normalize

    for name in ["policy.pt", "target_policy.pt", "model.pt"]:
        candidate = model_path / name
        if candidate.exists():
            model = torch.load(candidate, map_location=device)
            return model, vec_normalize

    raise FileNotFoundError(
        f"No trained model found in {model_dir}. "
        f"Expected {env_name}_ppo_final.zip or {env_name}_policy.pt"
    )


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], Tuple[np.ndarray, float, float, float]]:
    """
    Create a policy function from a loaded model.

    The returned function takes a state (np.ndarray) and returns
    (action, log_prob, value, entropy), compatible with MaskNetworkTrainer.

    Args:
        model: Loaded model (SB3 PPO or PyTorch nn.Module)
        vec_normalize: Optional VecNormalize for observation normalization
        device: Device for tensor operations

    Returns:
        Policy function: state -> (action, log_prob, value, entropy)
    """
    if HAS_SB3 and isinstance(model, PPO):
        # SB3 PPO model
        policy_net = model.policy

        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            # Normalize observation if needed
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)

            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # Extract features
                features = policy_net.mlp_extractor.forward_actor(
                    policy_net.features_extractor(state_tensor)
                )
                if isinstance(features, tuple):
                    features = features[0]

                # Get action distribution parameters
                mean_actions = policy_net.action_net(features)
                log_std = policy_net.log_std.expand_as(mean_actions)
                std = torch.exp(log_std)

                # Sample action
                dist = torch.distributions.Normal(mean_actions, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                # Get value
                value_features = policy_net.mlp_extractor.forward_critic(
                    policy_net.features_extractor(state_tensor)
                )
                if isinstance(value_features, tuple):
                    value_features = value_features[0]
                value = policy_net.value_net(value_features)

                # Entropy
                entropy = dist.entropy().sum(dim=-1)

            return (
                action.cpu().numpy().flatten(),
                log_prob.item(),
                value.item(),
                entropy.item(),
            )

        return policy_fn

    elif isinstance(model, torch.nn.Module):
        # Raw PyTorch model
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                output = model(state_tensor)
                if isinstance(output, tuple):
                    action, log_prob, value, entropy = output
                elif isinstance(output, dict):
                    action = output.get("action", output.get("mean", np.zeros(2)))
                    log_prob = output.get("log_prob", 0.0)
                    value = output.get("value", 0.0)
                    entropy = output.get("entropy", 0.0)
                else:
                    action = output
                    log_prob = 0.0
                    value = 0.0
                    entropy = 0.0

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().flatten()
            if isinstance(log_prob, torch.Tensor):
                log_prob = log_prob.item()
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(entropy, torch.Tensor):
                entropy = entropy.item()

            return action, float(log_prob), float(value), float(entropy)

        return policy_fn

    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def make_env(
    env_name: str = "MetaDrive-Macro-v1",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """
    Create a MetaDrive environment wrapped with state save/restore.

    Args:
        env_name: Environment name
        seed: Random seed
        max_episode_steps: Maximum episode steps

    Returns:
        Wrapped gym environment
    """
    env = make_ad_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_sparse_reward=False,
    )
    # Ensure state save/restore capability
    env = make_state_saveable(env)
    return env


def train_mask(
    env_name: str = "MetaDrive-Macro-v1",
    model_dir: str = "./trained_agents/autonomous_driving",
    output_dir: str = "./mask_agents/autonomous_driving",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a mask network for the autonomous driving environment.

    Args:
        env_name: Environment name
        model_dir: Directory with pre-trained target agent
        output_dir: Directory to save results
        config_path: Optional custom config path
        total_steps: Override total training steps
        alpha: Override intrinsic reward coefficient
        seed: Random seed
        device: Device (cuda/cpu)
        verbose: Print progress

    Returns:
        Dictionary with mask_network, trainer, history, fidelity, training_time
    """
    # Load configuration
    config = load_config(env_name, config_path)

    # Determine device
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Override config with CLI args
    if total_steps is not None:
        config.setdefault("training", {})["total_steps"] = total_steps
    if alpha is not None:
        config["alpha"] = alpha

    # Extract config values
    mask_config = config.get("mask_network", {})
    hidden_sizes = mask_config.get("hidden_sizes", [128, 128])
    activation = mask_config.get("activation", "tanh")
    alpha_val = config.get("alpha", 0.0001)
    ppo_config = config.get("ppo", {})
    training_config = config.get("training", {})
    total_steps_val = training_config.get("total_steps", 300000)
    steps_per_iteration = training_config.get("steps_per_iteration", 2048)
    eval_interval = training_config.get("eval_interval", 10)
    eval_episodes = training_config.get("eval_episodes", 10)
    save_interval = training_config.get("save_interval", 50)
    seed_val = training_config.get("seed", seed)

    # Set seeds
    set_seed(seed_val)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"=== Training Mask Network for {env_name} ===")
        print(f"Device: {device}")
        print(f"Alpha: {alpha_val}")
        print(f"Total steps: {total_steps_val}")
        print(f"Model dir: {model_dir}")
        print(f"Output dir: {output_dir}")

    # Load target policy
    if verbose:
        print("Loading target policy...")
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

    # Create environment
    if verbose:
        print("Creating environment...")
    env = make_env(env_name, seed=seed_val)

    # Get state and action dimensions
    state_dim = get_state_dim(env)
    action_dim = get_action_dim(env)
    discrete_action = is_discrete_action(env)

    if verbose:
        print(f"State dim: {state_dim}, Action dim: {action_dim}, Discrete: {discrete_action}")

    # Get action space bounds for continuous actions
    action_space_low = None
    action_space_high = None
    num_discrete_actions = None
    if not discrete_action:
        action_space_low = env.action_space.low
        action_space_high = env.action_space.high
    else:
        num_discrete_actions = env.action_space.n

    # Create mask network
    mask_network = MaskNetwork(
        state_dim=state_dim,
        hidden_sizes=tuple(hidden_sizes),
        activation=activation,
    ).to(device)

    # Create trainer
    trainer = MaskNetworkTrainer(
        mask_network=mask_network,
        target_policy=target_policy_fn,
        env=env,
        alpha=alpha_val,
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_epsilon=ppo_config.get("clip_epsilon", 0.2),
        value_loss_coef=ppo_config.get("value_loss_coef", 0.5),
        entropy_coef=ppo_config.get("entropy_coef", 0.01),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        ppo_epochs=ppo_config.get("ppo_epochs", 10),
        batch_size=ppo_config.get("batch_size", 64),
        device=device,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
        discrete_action=discrete_action,
        num_discrete_actions=num_discrete_actions,
    )

    # Train mask network
    if verbose:
        print("Training mask network...")
    start_time = time.time()

    history = trainer.train(
        total_steps=total_steps_val,
        steps_per_iteration=steps_per_iteration,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        save_interval=save_interval,
        save_path=output_dir,
        verbose=verbose,
    )

    training_time = time.time() - start_time

    # Compute fidelity
    if verbose:
        print("Computing fidelity...")
    fidelity = compute_fidelity_from_env(
        mask_network=mask_network,
        env=env,
        target_policy=target_policy_fn,
        num_episodes=config.get("fidelity", {}).get("num_episodes", 10),
        q_function=None,
        device=device,
    )

    if verbose:
        print(f"Fidelity (Pearson r): {fidelity:.4f}")
        print(f"Training time: {training_time:.1f}s")

    # Save results
    results = {
        "env_name": env_name,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discrete_action": discrete_action,
        "alpha": alpha_val,
        "total_steps": total_steps_val,
        "fidelity": fidelity,
        "training_time": training_time,
        "config": config,
    }

    # Save mask network
    mask_path = os.path.join(output_dir, f"{env_name}_mask_network.pt")
    trainer.save(mask_path)
    if verbose:
        print(f"Mask network saved to {mask_path}")

    # Save history
    history_path = os.path.join(output_dir, f"{env_name}_mask_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save results metadata
    results_path = os.path.join(output_dir, f"{env_name}_mask_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"Results saved to {output_dir}")
        print("=== Mask Network Training Complete ===")

    env.close()

    return {
        "mask_network": mask_network,
        "trainer": trainer,
        "history": history,
        "fidelity": fidelity,
        "training_time": training_time,
    }


def main():
    """Main entry point."""
    args = parse_args()

    if args.quiet:
        args.verbose = False

    train_mask(
        env_name=args.env,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        total_steps=args.total_steps,
        alpha=args.alpha,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()