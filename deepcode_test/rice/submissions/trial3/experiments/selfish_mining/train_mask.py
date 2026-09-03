#!/usr/bin/env python3
"""
Train Mask Network for Selfish Mining Environment
==================================================
Implements mask network training for the selfish mining domain.
Loads a pre-trained target PPO agent, wraps it into a policy function,
creates a MaskNetwork and MaskNetworkTrainer, runs the PPO-based training
loop with intrinsic reward α·I(a^e=1), computes fidelity, and saves
the trained mask network for subsequent RICE refinement.

Usage:
    python experiments/selfish_mining/train_mask.py \
        --env SelfishMining-v0 \
        --model-dir ./trained_agents/selfish_mining \
        --output-dir ./mask_agents/selfish_mining \
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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

# Import selfish mining environment utilities
from experiments.selfish_mining.env import (
    make_env as make_sm_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
)

# Optional Stable-Baselines3 import
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    env_name: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and merge default mask config with environment-specific overrides.

    Args:
        env_name: Environment name (e.g., "SelfishMining-v0")
        config_path: Optional path to custom config YAML

    Returns:
        Merged configuration dictionary
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # Load default mask config
    default_path = project_root / "configs" / "default_mask.yaml"
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load default refine config (for shared params)
    default_refine_path = project_root / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            refine_config = yaml.safe_load(f)
        # Merge relevant sections
        for key in ["ppo", "rnd", "policy", "training"]:
            if key in refine_config:
                if key not in config:
                    config[key] = {}
                config[key] = deep_merge(config.get(key, {}), refine_config[key])

    # Load environment-specific config
    env_config_path = project_root / "configs" / "env_specific" / "selfish_mining.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
        config = deep_merge(config, env_config)

    # Load custom config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f)
        config = deep_merge(config, custom_config)

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
        env_name: Environment name
        model_dir: Directory containing the trained model
        device: Device to load model on

    Returns:
        Tuple of (model, vec_normalize) where vec_normalize may be None
    """
    model_path = Path(model_dir)

    # Try Stable-Baselines3 format
    sb3_path = model_path / f"{env_name}_ppo_final.zip"
    if sb3_path.exists() and HAS_SB3:
        model = PPO.load(str(sb3_path), device=device)
        # Try to load VecNormalize stats
        vecnorm_path = model_path / f"{env_name}_vecnormalize.pkl"
        vec_normalize = None
        if vecnorm_path.exists():
            with open(vecnorm_path, "rb") as f:
                vec_normalize = pickle.load(f)
        return model, vec_normalize

    # Try raw PyTorch checkpoint
    pt_path = model_path / f"{env_name}_policy.pt"
    if pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        return checkpoint, None

    # Try generic model file
    for pattern in [f"{env_name}_model.pt", "policy.pt", "model.pt"]:
        candidate = model_path / pattern
        if candidate.exists():
            checkpoint = torch.load(candidate, map_location=device)
            return checkpoint, None

    raise FileNotFoundError(
        f"No trained model found in {model_dir}. "
        f"Expected {env_name}_ppo_final.zip (SB3) or {env_name}_policy.pt (PyTorch)."
    )


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], Tuple[np.ndarray, float, float, float]]:
    """
    Create a policy function from a loaded model.

    The returned function maps state -> (action, log_prob, value, entropy)
    compatible with MaskNetworkTrainer.

    Args:
        model: Loaded model (SB3 PPO or PyTorch checkpoint)
        vec_normalize: Optional VecNormalize for observation normalization
        device: Device for computation

    Returns:
        Policy function: state -> (action, log_prob, value, entropy)
    """
    if HAS_SB3 and isinstance(model, PPO):
        # Stable-Baselines3 PPO model
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            # Normalize observation if needed
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)

            obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # Extract features
                features = model.policy.features_extractor(obs_tensor)
                # Get action distribution
                latent_pi = model.policy.mlp_extractor.policy_net(features)
                action_logits = model.policy.action_net(latent_pi)
                # Get value
                latent_vf = model.policy.mlp_extractor.value_net(features)
                value = model.policy.value_net(latent_vf)

                # For discrete actions
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

            return (
                action.cpu().numpy().flatten(),
                log_prob.cpu().numpy().flatten()[0],
                value.cpu().numpy().flatten()[0],
                entropy.cpu().numpy().flatten()[0],
            )

        return policy_fn

    # Raw PyTorch model (assume it's a state dict or nn.Module)
    if isinstance(model, dict):
        # Reconstruct from state dict
        state_dim = 52  # default for selfish mining
        action_dim = 2
        hidden_sizes = [128, 128]

        class PolicyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev_dim = state_dim
                for h in hidden_sizes:
                    layers.append(torch.nn.Linear(prev_dim, h))
                    layers.append(torch.nn.Tanh())
                    prev_dim = h
                self.features = torch.nn.Sequential(*layers)
                self.action_head = torch.nn.Linear(prev_dim, action_dim)
                self.value_head = torch.nn.Linear(prev_dim, 1)

            def forward(self, x):
                feat = self.features(x)
                logits = self.action_head(feat)
                value = self.value_head(feat)
                return logits, value

        policy_net = PolicyNet()
        policy_net.load_state_dict(model)
        policy_net.to(device)
        policy_net.eval()

        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            obs_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = policy_net(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
            return (
                action.cpu().numpy().flatten(),
                log_prob.cpu().numpy().flatten()[0],
                value.cpu().numpy().flatten()[0],
                entropy.cpu().numpy().flatten()[0],
            )

        return policy_fn

    # Assume model is a callable
    if callable(model):
        return model

    raise ValueError(f"Unsupported model type: {type(model)}")


def make_env(
    env_name: str = "SelfishMining-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """
    Create a selfish mining environment with state save/restore capability.

    Args:
        env_name: Environment name
        seed: Random seed
        max_episode_steps: Maximum steps per episode

    Returns:
        Wrapped gym environment
    """
    env = make_sm_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    # Ensure state save/restore is available
    env = make_state_saveable(env)
    return env


def train_mask(
    env_name: str = "SelfishMining-v0",
    model_dir: str = "./trained_agents/selfish_mining",
    output_dir: str = "./mask_agents/selfish_mining",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a mask network for the selfish mining environment.

    Args:
        env_name: Environment name
        model_dir: Directory containing the pre-trained target agent
        output_dir: Directory to save the trained mask network
        config_path: Optional path to custom config YAML
        total_steps: Override total training steps
        alpha: Override intrinsic reward coefficient
        seed: Random seed
        device: Device to use ("cuda" or "cpu")
        verbose: Whether to print progress

    Returns:
        Dictionary with keys: mask_network, trainer, history, fidelity, training_time
    """
    # Load configuration
    config = load_config(env_name, config_path)

    # Determine device
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Override with CLI arguments
    if total_steps is not None:
        config["training"]["total_steps"] = total_steps
    if alpha is not None:
        config["alpha"] = alpha

    # Extract config values
    alpha_val = config.get("alpha", 0.0001)
    total_steps_val = config.get("training", {}).get("total_steps", 300000)
    steps_per_iteration = config.get("training", {}).get("steps_per_iteration", 2048)
    eval_interval = config.get("training", {}).get("eval_interval", 10)
    eval_episodes = config.get("training", {}).get("eval_episodes", 10)
    save_interval = config.get("training", {}).get("save_interval", 50)
    seed_val = config.get("training", {}).get("seed", seed)

    # PPO hyperparameters
    ppo_config = config.get("ppo", {})
    lr = ppo_config.get("learning_rate", 3e-4)
    gamma = ppo_config.get("gamma", 0.99)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    clip_epsilon = ppo_config.get("clip_epsilon", 0.2)
    value_loss_coef = ppo_config.get("value_loss_coef", 0.5)
    entropy_coef = ppo_config.get("entropy_coef", 0.01)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    ppo_epochs = ppo_config.get("ppo_epochs", 10)
    batch_size = ppo_config.get("batch_size", 64)

    # Mask network architecture
    mask_config = config.get("mask_network", {})
    hidden_sizes = mask_config.get("hidden_sizes", [128, 128])
    activation = mask_config.get("activation", "tanh")

    # Set seed
    set_seed(seed_val)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"=== Training Mask Network for {env_name} ===")
        print(f"Device: {device}")
        print(f"Alpha: {alpha_val}")
        print(f"Total steps: {total_steps_val}")
        print(f"Output directory: {output_dir}")

    # Load target policy
    if verbose:
        print("Loading target policy...")
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

    # Create environment
    env = make_env(env_name, seed_val)

    # Get environment info
    state_dim = get_state_dim(env)
    action_dim = get_action_dim(env)
    discrete = is_discrete_action(env)

    if verbose:
        print(f"State dim: {state_dim}, Action dim: {action_dim}, Discrete: {discrete}")

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
        learning_rate=lr,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
        device=device,
        discrete_action=discrete,
        num_discrete_actions=action_dim if discrete else None,
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
        save_path=str(output_path / f"{env_name}_mask_checkpoint"),
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
        device=device,
    )

    if verbose:
        print(f"Fidelity (Pearson correlation): {fidelity:.4f}")
        print(f"Training time: {training_time:.1f}s")

    # Save results
    # Save mask network
    mask_save_path = output_path / f"{env_name}_mask_network.pt"
    mask_network.save(str(mask_save_path))

    # Save trainer state
    trainer_save_path = output_path / f"{env_name}_mask_trainer.pt"
    trainer.save(str(trainer_save_path))

    # Save training history
    history_path = output_path / f"{env_name}_mask_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)

    # Save fidelity and metadata
    results = {
        "env_name": env_name,
        "alpha": alpha_val,
        "total_steps": total_steps_val,
        "fidelity": fidelity,
        "training_time": training_time,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discrete_action": discrete,
        "hidden_sizes": list(hidden_sizes),
        "activation": activation,
        "device": device,
        "seed": seed_val,
        "final_loss": history[-1] if history else None,
    }
    results_path = output_path / f"{env_name}_mask_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"Results saved to {output_dir}")

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
        description="Train mask network for selfish mining environment"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="SelfishMining-v0",
        help="Environment name",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./trained_agents/selfish_mining",
        help="Directory containing pre-trained target agent",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./mask_agents/selfish_mining",
        help="Directory to save trained mask network",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config YAML",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total training steps",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override intrinsic reward coefficient",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress output",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    verbose = not args.quiet

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
            verbose=verbose,
        )
        if verbose:
            print(f"\nTraining complete. Fidelity: {results['fidelity']:.4f}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())