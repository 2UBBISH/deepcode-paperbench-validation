#!/usr/bin/env python3
"""
Train Mask Network for MuJoCo Environments
===========================================
Trains a mask network (binary policy) that learns to identify critical states
for a pre-trained target PPO agent on MuJoCo tasks (Hopper, Walker2d, Reacher, HalfCheetah).

The mask network outputs an importance score ξ(s) for each state, indicating whether
the target agent's action should be preserved (critical) or randomized (non-critical).

Usage:
    python experiments/mujoco/train_mask.py --env Hopper-v4
    python experiments/mujoco/train_mask.py --env Walker2d-v4 --alpha 0.0001 --total_steps 500000
    python experiments/mujoco/train_mask.py --env HalfCheetah-v4 --config configs/env_specific/halfcheetah.yaml

Output:
    - trained_mask_net.pt: Saved mask network weights
    - mask_trainer_state.pt: Full trainer state (optimizer, etc.)
    - fidelity_results.json: Fidelity evaluation metrics
    - training_history.json: Training metrics per iteration
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rice.mask_net import (
    MaskNetwork,
    MaskNetworkTrainer,
    PerturbedPolicy,
    compute_fidelity,
    compute_fidelity_from_env,
    train_mask_network,
)
from rice.utils import evaluate_policy, set_seed, to_numpy


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_config(env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and merge default mask configuration with environment-specific overrides.

    Args:
        env_name: Gym environment name (e.g., "Hopper-v4")
        config_path: Optional path to environment-specific YAML config.
                     If None, auto-detects from configs/env_specific/<env_base>.yaml

    Returns:
        Merged configuration dictionary.
    """
    # Load default mask config
    default_path = PROJECT_ROOT / "configs" / "default_mask.yaml"
    if not default_path.exists():
        print(f"[WARNING] Default mask config not found at {default_path}, using built-in defaults.")
        config = _get_builtin_defaults()
    else:
        with open(default_path, "r") as f:
            config = yaml.safe_load(f)

    # Load environment-specific config
    if config_path is None:
        # Auto-detect: strip version suffix (e.g., "Hopper-v4" -> "hopper")
        env_base = env_name.lower().split("-")[0]
        env_config_path = PROJECT_ROOT / "configs" / "env_specific" / f"{env_base}.yaml"
    else:
        env_config_path = Path(config_path)

    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
        # Deep merge: env-specific overrides default
        config = _deep_merge(config, env_config)
        print(f"[INFO] Loaded env-specific config from {env_config_path}")
    else:
        print(f"[WARNING] Env-specific config not found at {env_config_path}, using defaults only.")

    # Add env_name to config for convenience
    config["env_name"] = env_name

    return config


def _get_builtin_defaults() -> Dict[str, Any]:
    """Return built-in default configuration matching default_mask.yaml."""
    return {
        "mask_network": {
            "hidden_sizes": [128, 128],
            "activation": "tanh",
        },
        "alpha": 0.0001,
        "ppo": {
            "learning_rate": 3.0e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 10,
            "batch_size": 64,
        },
        "training": {
            "total_steps": 300000,
            "steps_per_iteration": 2048,
            "eval_interval": 10,
            "eval_episodes": 10,
            "save_interval": 50,
            "seed": 42,
        },
        "fidelity": {
            "num_episodes": 10,
            "q_function": None,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ==============================================================================
# Target Policy Loading
# ==============================================================================

def load_target_policy(
    env_name: str,
    model_dir: str = "./trained_agents",
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """
    Load a pre-trained Stable-Baselines3 PPO agent and its VecNormalize stats.

    Args:
        env_name: Environment name (e.g., "Hopper-v4")
        model_dir: Directory containing saved models
        device: Device to load model on

    Returns:
        Tuple of (ppo_model, vec_normalize) where vec_normalize may be None.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError(
            "Stable-Baselines3 is required. Install with: pip install stable-baselines3"
        )

    model_path = Path(model_dir) / f"{env_name}_ppo_final.zip"
    norm_path = Path(model_dir) / f"{env_name}_vecnormalize.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Pre-trained model not found at {model_path}. "
            f"Run 'python experiments/mujoco/train_target.py --env {env_name}' first."
        )

    print(f"[INFO] Loading target policy from {model_path}")
    model = PPO.load(str(model_path), device=device)

    vec_normalize = None
    if norm_path.exists():
        import pickle
        with open(norm_path, "rb") as f:
            vec_normalize = pickle.load(f)
        print(f"[INFO] Loaded VecNormalize stats from {norm_path}")

    return model, vec_normalize


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], Tuple[np.ndarray, float, float, float]]:
    """
    Create a target policy function compatible with MaskNetworkTrainer.

    The returned function takes a state (np.ndarray) and returns:
        (action, log_prob, value, entropy)

    Args:
        model: Stable-Baselines3 PPO model
        vec_normalize: Optional VecNormalize for observation normalization
        device: Device for tensor operations

    Returns:
        Policy function: state -> (action, log_prob, value, entropy)
    """
    def target_policy(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        # Normalize observation if VecNormalize is available
        if vec_normalize is not None:
            state = vec_normalize.normalize_obs(state)

        # Convert to tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # Get policy distribution and value
            obs = model.policy.obs_to_tensor(state)[0]  # Handle dict obs if needed
            if isinstance(obs, tuple):
                obs = obs[0]

            # Use the policy's extract_features and action_net
            features = model.policy.mlp_extractor.forward_actor(
                model.policy.features_extractor(state_tensor)
            )
            mean_actions = model.policy.action_net(features)

            # Get log_std
            log_std = model.policy.log_std
            if log_std.dim() == 1:
                log_std = log_std.unsqueeze(0).expand_as(mean_actions)

            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean_actions, std)

            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # Get value
            value = model.policy.value_net(
                model.policy.mlp_extractor.forward_critic(
                    model.policy.features_extractor(state_tensor)
                )
            )

            # Entropy
            entropy = dist.entropy().sum(dim=-1)

        return (
            action.cpu().numpy().flatten(),
            log_prob.item(),
            value.item(),
            entropy.item(),
        )

    return target_policy


# ==============================================================================
# Environment Creation
# ==============================================================================

def make_env(env_name: str, seed: int = 42) -> Any:
    """
    Create a MuJoCo environment.

    Args:
        env_name: Gym environment name
        seed: Random seed

    Returns:
        Gym environment instance.
    """
    import gym

    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)

    return env


# ==============================================================================
# Main Training Function
# ==============================================================================

def train_mask(
    env_name: str = "Hopper-v4",
    model_dir: str = "./trained_agents",
    output_dir: str = "./trained_masks",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    alpha: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train a mask network for a given MuJoCo environment.

    Args:
        env_name: Gym environment name
        model_dir: Directory containing pre-trained target agent
        output_dir: Directory to save trained mask network
        config_path: Path to environment-specific YAML config
        total_steps: Override total training steps
        alpha: Override intrinsic reward coefficient
        seed: Random seed
        device: Device to use ("cuda" or "cpu")
        verbose: Whether to print progress

    Returns:
        Dictionary with training results (fidelity, history, timing).
    """
    # Load configuration
    config = load_config(env_name, config_path)

    # Apply overrides
    if total_steps is not None:
        config["training"]["total_steps"] = total_steps
    if alpha is not None:
        config["alpha"] = alpha
    if device is not None:
        config["device"] = device
    if seed is not None:
        config["training"]["seed"] = seed

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["training"]["seed"])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load target policy
    print(f"\n{'='*60}")
    print(f"Training Mask Network for {env_name}")
    print(f"{'='*60}\n")

    target_model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(target_model, vec_normalize, device)

    # Create environment
    env = make_env(env_name, config["training"]["seed"])

    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'n'):
        # Discrete action space
        discrete_action = True
        num_discrete_actions = env.action_space.n
        action_dim = 1
        action_low = np.array([0])
        action_high = np.array([num_discrete_actions - 1])
    else:
        # Continuous action space
        discrete_action = False
        num_discrete_actions = None
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high

    print(f"[INFO] State dim: {state_dim}, Action dim: {action_dim}")
    print(f"[INFO] Discrete action: {discrete_action}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Alpha: {config['alpha']}")
    print(f"[INFO] Total steps: {config['training']['total_steps']}")

    # Create mask network
    mask_net = MaskNetwork(
        state_dim=state_dim,
        hidden_sizes=tuple(config["mask_network"]["hidden_sizes"]),
        activation=config["mask_network"]["activation"],
    ).to(device)

    # Create trainer
    trainer = MaskNetworkTrainer(
        mask_network=mask_net,
        target_policy=target_policy_fn,
        env=env,
        alpha=config["alpha"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_epsilon=config["ppo"]["clip_epsilon"],
        value_loss_coef=config["ppo"]["value_loss_coef"],
        entropy_coef=config["ppo"]["entropy_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        learning_rate=config["ppo"]["learning_rate"],
        ppo_epochs=config["ppo"]["ppo_epochs"],
        batch_size=config["ppo"]["batch_size"],
        device=device,
        action_space_low=action_low,
        action_space_high=action_high,
        discrete_action=discrete_action,
        num_discrete_actions=num_discrete_actions,
    )

    # Train mask network
    print(f"\n[INFO] Starting mask network training...")
    start_time = time.time()

    history = trainer.train(
        total_steps=config["training"]["total_steps"],
        steps_per_iteration=config["training"]["steps_per_iteration"],
        eval_interval=config["training"]["eval_interval"],
        eval_episodes=config["training"]["eval_episodes"],
        save_interval=config["training"]["save_interval"],
        save_path=os.path.join(output_dir, f"{env_name}_mask_checkpoint"),
        verbose=verbose,
    )

    training_time = time.time() - start_time
    print(f"[INFO] Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")

    # Save final mask network
    mask_save_path = os.path.join(output_dir, f"{env_name}_mask_net.pt")
    trainer.save(mask_save_path)
    print(f"[INFO] Mask network saved to {mask_save_path}")

    # Save training history
    history_path = os.path.join(output_dir, f"{env_name}_mask_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=float)
    print(f"[INFO] Training history saved to {history_path}")

    # Compute fidelity
    print(f"\n[INFO] Computing fidelity...")
    fidelity_score = compute_fidelity_from_env(
        mask_network=mask_net,
        env=env,
        target_policy=target_policy_fn,
        num_episodes=config["fidelity"]["num_episodes"],
        q_function=config["fidelity"].get("q_function"),
        device=device,
    )

    print(f"[INFO] Fidelity score: {fidelity_score:.4f}")

    # Save fidelity results
    fidelity_results = {
        "env_name": env_name,
        "fidelity_score": float(fidelity_score),
        "training_time_seconds": training_time,
        "alpha": config["alpha"],
        "total_steps": config["training"]["total_steps"],
        "seed": config["training"]["seed"],
        "final_policy_loss": history[-1].get("policy_loss", None) if history else None,
        "final_value_loss": history[-1].get("value_loss", None) if history else None,
    }
    fidelity_path = os.path.join(output_dir, f"{env_name}_fidelity.json")
    with open(fidelity_path, "w") as f:
        json.dump(fidelity_results, f, indent=2)
    print(f"[INFO] Fidelity results saved to {fidelity_path}")

    env.close()

    return {
        "mask_network": mask_net,
        "trainer": trainer,
        "history": history,
        "fidelity": fidelity_score,
        "training_time": training_time,
    }


# ==============================================================================
# Command-Line Interface
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train mask network for MuJoCo environments (RICE paper)"
    )
    parser.add_argument(
        "--env", type=str, default="Hopper-v4",
        choices=["Hopper-v4", "Walker2d-v4", "Reacher-v4", "HalfCheetah-v4",
                 "Hopper-v2", "Walker2d-v2", "Reacher-v2", "HalfCheetah-v2"],
        help="MuJoCo environment name (default: Hopper-v4)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents",
        help="Directory containing pre-trained target agent (default: ./trained_agents)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./trained_masks",
        help="Directory to save trained mask network (default: ./trained_masks)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to environment-specific YAML config (auto-detected if not provided)"
    )
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Override total training steps (default: from config)"
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Override intrinsic reward coefficient (default: from config)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

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
    )

    print(f"\n{'='*60}")
    print(f"Mask Network Training Complete!")
    print(f"  Environment:     {args.env}")
    print(f"  Fidelity Score:  {results['fidelity']:.4f}")
    print(f"  Training Time:   {results['training_time']:.1f}s")
    print(f"  Output Dir:      {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()