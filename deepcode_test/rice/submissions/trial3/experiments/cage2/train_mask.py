#!/usr/bin/env python3
"""
Train Mask Network for CAGE Challenge 2 (Cybersecurity Domain).

This script loads a pre-trained target PPO agent for the CAGE2 environment,
constructs a perturbed policy, trains a MaskNetwork via PPO with intrinsic
reward α·I(a^e=1), computes fidelity, and saves the trained mask network
for subsequent RICE refinement.

Usage:
    python experiments/cage2/train_mask.py --env CAGE2-v0 --model-dir ./trained_agents/cage2 --output-dir ./mask_agents/cage2
"""

import argparse
import json
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
import gym

# Optional Stable-Baselines3 import
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# RICE core imports
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

# CAGE2 environment imports
from experiments.cage2.env import (
    make_env as make_cage2_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dictionary with override values."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    env_name: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and merge configuration from default YAML files and environment-specific overrides.

    Args:
        env_name: Name of the environment (e.g., "CAGE2-v0").
        config_path: Optional path to a custom YAML config file.

    Returns:
        Merged configuration dictionary.
    """
    # Load default mask config
    default_mask_path = Path(__file__).parent.parent.parent / "configs" / "default_mask.yaml"
    config = {}
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load default refine config (for shared parameters)
    default_refine_path = Path(__file__).parent.parent.parent / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            refine_config = yaml.safe_load(f) or {}
        config = _deep_update(config, refine_config)

    # Load environment-specific config
    env_config_path = Path(__file__).parent.parent.parent / "configs" / "env_specific" / "cage2.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
        config = _deep_update(config, env_config)

    # Load custom config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f) or {}
        config = _deep_update(config, custom_config)

    return config


def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """
    Load a pre-trained target PPO agent.

    Supports Stable-Baselines3 .zip models and raw PyTorch .pt checkpoints.

    Args:
        env_name: Environment name.
        model_dir: Directory containing the trained model.
        device: Device to load the model on.

    Returns:
        Tuple of (model, vec_normalize) where vec_normalize may be None.
    """
    model_path = Path(model_dir)
    vec_normalize = None

    # Try Stable-Baselines3 format first
    sb3_path = model_path / f"{env_name}_ppo_final.zip"
    if sb3_path.exists() and HAS_SB3:
        model = PPO.load(str(sb3_path), device=device)
        # Try to load VecNormalize stats
        vecnorm_path = model_path / f"{env_name}_vecnormalize.pkl"
        if vecnorm_path.exists():
            with open(vecnorm_path, "rb") as f:
                vec_normalize = pickle.load(f)
        return model, vec_normalize

    # Try raw PyTorch checkpoint
    pt_path = model_path / f"{env_name}_policy.pt"
    if pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        # Return checkpoint as model-like object
        return checkpoint, None

    # Try any .pt file in the directory
    pt_files = list(model_path.glob("*.pt")) + list(model_path.glob("*.pth"))
    if pt_files:
        checkpoint = torch.load(str(pt_files[0]), map_location=device)
        return checkpoint, None

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

    The returned function takes a state and returns (action, log_prob, value, entropy).

    Args:
        model: Loaded model (SB3 PPO or raw PyTorch checkpoint).
        vec_normalize: Optional VecNormalize for observation normalization.
        device: Device for tensor operations.

    Returns:
        Policy function: state -> (action, log_prob, value, entropy)
    """
    if HAS_SB3 and isinstance(model, PPO):
        # Stable-Baselines3 PPO model
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            # Normalize observation if needed
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Extract features
                features = model.policy.features_extractor(state_tensor)
                # Get action distribution
                if hasattr(model.policy, 'action_dist'):
                    dist = model.policy.action_dist
                else:
                    dist = model.policy._get_action_dist_from_latent(features)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                value = model.policy.value_net(features)
            return (
                action.cpu().numpy().flatten(),
                log_prob.cpu().numpy().item(),
                value.cpu().numpy().item(),
                entropy.cpu().numpy().item(),
            )
        return policy_fn

    # Raw PyTorch model (assumes it has get_action method or similar)
    if hasattr(model, 'get_action'):
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value, entropy = model.get_action(state_tensor)
            return (
                to_numpy(action).flatten(),
                to_numpy(log_prob).item(),
                to_numpy(value).item(),
                to_numpy(entropy).item(),
            )
        return policy_fn

    # Fallback: treat model as a callable
    if callable(model):
        return model

    raise ValueError(
        "Cannot create policy function from model. "
        "Model must be an SB3 PPO instance, have a get_action method, or be callable."
    )


def make_env(
    env_name: str = "CAGE2-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False,
) -> gym.Env:
    """
    Create a CAGE2 environment with state save/restore capability.

    Args:
        env_name: Environment name.
        seed: Random seed.
        max_episode_steps: Maximum steps per episode.
        use_real_env: Whether to use the real CybORG environment.

    Returns:
        Wrapped gym environment.
    """
    env = make_cage2_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env,
    )
    env = make_state_saveable(env)
    return env


def train_mask(
    env_name: str = "CAGE2-v0",
    model_dir: str = "./trained_agents/cage2",
    output_dir: str = "./mask_agents/cage2",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
    use_real_env: bool = False,
) -> Dict[str, Any]:
    """
    Train a mask network for the CAGE2 environment.

    Args:
        env_name: Environment name.
        model_dir: Directory containing the pre-trained target agent.
        output_dir: Directory to save the trained mask network and results.
        config_path: Optional path to custom YAML config.
        total_steps: Override total training steps from config.
        alpha: Override intrinsic reward coefficient from config.
        seed: Random seed.
        device: Device for training ("cuda" or "cpu").
        verbose: Whether to print progress.
        use_real_env: Whether to use the real CybORG environment.

    Returns:
        Dictionary with keys: mask_network, trainer, history, fidelity, training_time.
    """
    # Load configuration
    config = load_config(env_name, config_path)

    # Determine device
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Override config with CLI arguments
    if total_steps is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["total_steps"] = total_steps
    if alpha is not None:
        config["alpha"] = alpha

    # Extract hyperparameters
    mask_config = config.get("mask_network", {})
    hidden_sizes = mask_config.get("hidden_sizes", [128, 128])
    activation = mask_config.get("activation", "tanh")
    alpha_val = config.get("alpha", 0.0001)

    ppo_config = config.get("ppo", {})
    learning_rate = ppo_config.get("learning_rate", 3e-4)
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_epsilon = ppo_config.get("clip_epsilon", 0.2)
    value_loss_coef = ppo_config.get("value_loss_coef", 0.5)
    entropy_coef = ppo_config.get("entropy_coef", 0.01)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    ppo_epochs = ppo_config.get("ppo_epochs", 10)
    batch_size = ppo_config.get("batch_size", 64)

    training_config = config.get("training", {})
    total_steps_val = training_config.get("total_steps", 300000)
    steps_per_iteration = training_config.get("steps_per_iteration", 2048)
    eval_interval = training_config.get("eval_interval", 10)
    eval_episodes = training_config.get("eval_episodes", 10)
    save_interval = training_config.get("save_interval", 50)
    seed_val = training_config.get("seed", seed)

    fidelity_config = config.get("fidelity", {})
    fidelity_episodes = fidelity_config.get("num_episodes", 10)

    max_episode_steps = config.get("max_episode_steps", 100)

    # Set seeds
    set_seed(seed_val)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"Training mask network for {env_name}")
        print(f"  Device: {device}")
        print(f"  Alpha: {alpha_val}")
        print(f"  Total steps: {total_steps_val}")
        print(f"  Model dir: {model_dir}")
        print(f"  Output dir: {output_dir}")

    # Load target policy
    if verbose:
        print("Loading target policy...")
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

    # Create environment
    if verbose:
        print("Creating environment...")
    env = make_env(env_name, seed_val, max_episode_steps, use_real_env)

    # Get environment dimensions
    state_dim = get_state_dim(env)
    action_dim = get_action_dim(env)
    discrete = is_discrete_action(env)

    if verbose:
        print(f"  State dim: {state_dim}, Action dim: {action_dim}, Discrete: {discrete}")

    # Get action space bounds for continuous actions
    action_space_low = None
    action_space_high = None
    num_discrete_actions = None
    if not discrete:
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
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        device=device,
        action_space_low=action_space_low,
        action_space_high=action_space_high,
        discrete_action=discrete,
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
        num_episodes=fidelity_episodes,
        device=device,
    )

    if verbose:
        print(f"  Fidelity: {fidelity:.4f}")
        print(f"  Training time: {training_time:.1f}s")

    # Save mask network
    mask_save_path = os.path.join(output_dir, f"{env_name}_mask_network.pt")
    mask_network.save(mask_save_path)
    if verbose:
        print(f"Mask network saved to {mask_save_path}")

    # Save trainer state
    trainer_save_path = os.path.join(output_dir, f"{env_name}_mask_trainer.pt")
    trainer.save(trainer_save_path)

    # Save training history
    history_path = os.path.join(output_dir, f"{env_name}_mask_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=float)

    # Save fidelity
    fidelity_path = os.path.join(output_dir, f"{env_name}_mask_fidelity.json")
    with open(fidelity_path, "w") as f:
        json.dump({"fidelity": fidelity, "training_time": training_time}, f, indent=2)

    # Save configuration used
    config_save_path = os.path.join(output_dir, f"{env_name}_mask_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    env.close()

    return {
        "mask_network": mask_network,
        "trainer": trainer,
        "history": history,
        "fidelity": fidelity,
        "training_time": training_time,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Mask Network for CAGE Challenge 2"
    )
    parser.add_argument(
        "--env", type=str, default="CAGE2-v0",
        help="Environment name (default: CAGE2-v0)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents/cage2",
        help="Directory containing the pre-trained target agent"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./mask_agents/cage2",
        help="Directory to save the trained mask network"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML configuration file"
    )
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Override total training steps from config"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override intrinsic reward coefficient"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for training (cuda or cpu, default: auto-detect)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--use-real-env", action="store_true",
        help="Use the real CybORG environment instead of simulated"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        results = train_mask(
            env_name=args.env,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            total_steps=args.total_steps,
            alpha=args.alpha,
            seed=args.seed,
            device=args.device,
            verbose=not args.quiet,
            use_real_env=args.use_real_env,
        )
        print(f"\nMask network training completed successfully!")
        print(f"  Fidelity: {results['fidelity']:.4f}")
        print(f"  Training time: {results['training_time']:.1f}s")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())