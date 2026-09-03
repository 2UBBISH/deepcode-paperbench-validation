#!/usr/bin/env python3
"""
RICE Refinement Script for Autonomous Driving (MetaDrive Macro-v1).

This script orchestrates the full RICE refinement pipeline for the MetaDrive
autonomous driving environment:
  1. Loads a pre-trained target PPO agent and a trained mask network.
  2. Collects critical states using the mask network's importance scores.
  3. Refines the policy via PPO with an RND exploration bonus and a mixed
     initial state distribution (critical states + default resets).
  4. Also provides a baseline PPO fine-tuning mode for comparison.

Usage:
    python experiments/autonomous_driving/refine.py \
        --env_name MetaDrive-Macro-v1 \
        --model_dir ./trained_agents/autonomous_driving \
        --mask_dir ./trained_masks/autonomous_driving \
        --output_dir ./refined_agents/autonomous_driving \
        --total_steps 2000000 \
        --p_mixed 0.25 \
        --lambda_rnd 0.01 \
        --seed 42 \
        --device cuda

Reference: RICE paper Table 3 (p=0.25, λ=0.01 for autonomous driving).
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
import torch.nn as nn
import yaml
import gym

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Core RICE modules
from rice.mask_net import MaskNetwork
from rice.rnd import RNDModule, BonusNormalizer
from rice.refine import RICERefine, refine_policy
from rice.utils import (
    evaluate_policy, set_seed, to_tensor, to_numpy,
    save_state_dict, load_state_dict, orthogonal_init,
    TrajectoryBuffer, collect_trajectories, compute_gae, compute_returns,
)
from rice.env_wrappers import (
    StateSaveWrapper, MuJoCoStateWrapper, make_state_saveable,
    save_env_state, restore_env_state, reset_env_to_state,
)

# Domain-specific environment
from experiments.autonomous_driving.env import (
    make_env as make_ad_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
    MetaDriveStateWrapper,
)

# Optional Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


# ==============================================================================
# Configuration Loading
# ==============================================================================

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
    env_name: str = "MetaDrive-Macro-v1",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and merge default refine config with environment-specific overrides.

    Args:
        env_name: Environment name (used to find env-specific YAML).
        config_path: Optional path to a custom YAML config for further overrides.

    Returns:
        Merged configuration dictionary.
    """
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"

    # Load default refine config
    default_path = config_dir / "default_refine.yaml"
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load environment-specific config
    env_config_path = config_dir / "env_specific" / "autonomous_driving.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
        config = deep_merge(config, env_config)

    # Load custom config if provided
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f)
        config = deep_merge(config, custom_config)

    return config


# ==============================================================================
# Environment Creation
# ==============================================================================

def make_env(
    env_name: str = "MetaDrive-Macro-v1",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_sparse_reward: bool = False,
) -> gym.Env:
    """
    Create a MetaDrive environment wrapped with state save/restore capability.

    Args:
        env_name: Environment identifier.
        seed: Random seed.
        max_episode_steps: Maximum steps per episode (None = default).
        use_sparse_reward: Whether to use sparse reward variant.

    Returns:
        Wrapped gym environment.
    """
    env = make_ad_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_sparse_reward=use_sparse_reward,
    )
    # Ensure state save/restore is available
    if not isinstance(env, (StateSaveWrapper, MetaDriveStateWrapper)):
        env = make_state_saveable(env)
    return env


# ==============================================================================
# Target Policy Loading
# ==============================================================================

def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """
    Load a pre-trained target PPO agent.

    Supports Stable-Baselines3 .zip models and raw PyTorch .pt checkpoints.

    Args:
        env_name: Environment name (used to locate model file).
        model_dir: Directory containing the trained model.
        device: Device to load the model on.

    Returns:
        Tuple of (model, vec_normalize) where vec_normalize may be None.
    """
    model_path = Path(model_dir)

    # Try Stable-Baselines3 format
    sb3_path = model_path / f"{env_name}_ppo_final.zip"
    if sb3_path.exists() and HAS_SB3:
        model = PPO.load(str(sb3_path), device=device)
        # Try to load VecNormalize stats
        vec_norm_path = model_path / f"{env_name}_vecnormalize.pkl"
        vec_normalize = None
        if vec_norm_path.exists():
            with open(vec_norm_path, "rb") as f:
                vec_normalize = pickle.load(f)
        return model, vec_normalize

    # Try raw PyTorch checkpoint
    pt_path = model_path / f"{env_name}_target_policy.pt"
    if pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        return checkpoint, None

    raise FileNotFoundError(
        f"No target policy found in {model_dir}. "
        f"Expected {env_name}_ppo_final.zip or {env_name}_target_policy.pt"
    )


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], Tuple[np.ndarray, float, float, float]]:
    """
    Wrap a loaded model into a policy function compatible with RICERefine.

    The returned function maps (state) -> (action, log_prob, value, entropy).

    Args:
        model: Loaded model (SB3 PPO or PyTorch nn.Module).
        vec_normalize: Optional VecNormalize for observation normalization.
        device: Device for tensor operations.

    Returns:
        Policy function: state -> (action, log_prob, value, entropy).
    """
    if HAS_SB3 and isinstance(model, PPO):
        # Stable-Baselines3 PPO model
        policy = model.policy
        policy.set_training_mode(False)

        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            # Normalize observation if needed
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # SB3 continuous policy: get action distribution parameters
                features = policy.mlp_extractor.forward_actor(
                    policy.extract_features(state_tensor)
                )
                mean_actions = policy.action_net(features)
                log_std = policy.log_std.expand_as(mean_actions)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean_actions, std)
                action = mean_actions.squeeze(0).cpu().numpy()  # deterministic
                log_prob = dist.log_prob(
                    torch.as_tensor(action, device=device)
                ).sum(dim=-1).item()
                # Value
                value_features = policy.mlp_extractor.forward_critic(
                    policy.extract_features(state_tensor)
                )
                value = policy.value_net(value_features).squeeze().item()
                entropy = dist.entropy().sum(dim=-1).item()
            return action, log_prob, value, entropy

        return policy_fn

    elif isinstance(model, nn.Module):
        # Raw PyTorch model with get_action method
        model.eval()
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value, entropy = model.get_action(state_tensor, deterministic=True)
            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.item() if isinstance(log_prob, torch.Tensor) else log_prob,
                value.item() if isinstance(value, torch.Tensor) else value,
                entropy.item() if isinstance(entropy, torch.Tensor) else entropy,
            )
        return policy_fn

    elif callable(model):
        # Already a callable
        return model

    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


# ==============================================================================
# Mask Network Loading
# ==============================================================================

def load_mask_network(
    env_name: str,
    mask_dir: str,
    state_dim: int,
    device: str = "cpu",
) -> MaskNetwork:
    """
    Load a trained mask network from disk.

    Args:
        env_name: Environment name.
        mask_dir: Directory containing the trained mask network.
        state_dim: State dimension for the mask network.
        device: Device to load on.

    Returns:
        Loaded MaskNetwork in eval mode.
    """
    mask_path = Path(mask_dir) / f"{env_name}_mask_network.pt"

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask network not found at {mask_path}")

    checkpoint = torch.load(mask_path, map_location=device)

    # Determine hidden sizes from checkpoint
    hidden_sizes = checkpoint.get("hidden_sizes", (128, 128))
    activation = checkpoint.get("activation", "tanh")

    mask_network = MaskNetwork(
        state_dim=state_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
    )
    mask_network.load_state_dict(checkpoint["model_state_dict"])
    mask_network.to(device)
    mask_network.eval()

    return mask_network


# ==============================================================================
# Sparse Reward Wrapper (for sparse reward experiments)
# ==============================================================================

class SparseRewardWrapper(gym.Wrapper):
    """
    Converts dense MetaDrive rewards to sparse based on success.
    Reward = 1.0 if episode terminated with success (reached destination),
    else 0.0.
    """

    def __init__(self, env: gym.Env, success_reward: float = 1.0):
        super().__init__(env)
        self.success_reward = success_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Check if episode ended with success
        if info.get("arrive_dest", False) or info.get("success", False):
            reward = self.success_reward
        else:
            reward = 0.0
        return obs, reward, terminated, truncated, info


# ==============================================================================
# Main Refinement Pipeline
# ==============================================================================

def run_refine(
    env_name: str = "MetaDrive-Macro-v1",
    model_dir: str = "./trained_agents/autonomous_driving",
    mask_dir: str = "./trained_masks/autonomous_driving",
    output_dir: str = "./refined_agents/autonomous_driving",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    p_mixed: Optional[float] = None,
    lambda_rnd: Optional[float] = None,
    seed: int = 42,
    device: str = "cuda",
    sparse: bool = False,
    sparse_threshold: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full RICE refinement pipeline for autonomous driving.

    Args:
        env_name: Environment name.
        model_dir: Directory with pre-trained target agent.
        mask_dir: Directory with trained mask network.
        output_dir: Directory to save refined policy and results.
        config_path: Optional custom YAML config path.
        total_steps: Total refinement steps (overrides config).
        p_mixed: Probability of resetting to critical state (overrides config).
        lambda_rnd: RND bonus coefficient (overrides config).
        seed: Random seed.
        device: Device for training.
        sparse: Whether to use sparse reward variant.
        sparse_threshold: Threshold for sparse reward (unused for MetaDrive).
        verbose: Whether to print progress.

    Returns:
        Dictionary with refined policy, training history, evaluation rewards,
        and configuration.
    """
    # --- Load configuration ---
    config = load_config(env_name, config_path)

    # Override with CLI arguments
    if total_steps is not None:
        config.setdefault("refine_training", {})["total_steps"] = total_steps
    if p_mixed is not None:
        config.setdefault("mixed_init", {})["p_mixed"] = p_mixed
    if lambda_rnd is not None:
        config.setdefault("rnd", {})["lambda_coef"] = lambda_rnd

    # Extract key parameters
    refine_cfg = config.get("refine_training", {})
    total_steps = refine_cfg.get("total_steps", 2_000_000)
    steps_per_iteration = refine_cfg.get("steps_per_iteration", 2048)
    eval_interval = refine_cfg.get("eval_interval", 10)
    eval_episodes = refine_cfg.get("eval_episodes", 10)
    save_interval = refine_cfg.get("save_interval", 50)

    critical_cfg = config.get("critical_state", {})
    num_critical_episodes = critical_cfg.get("num_episodes", 100)
    top_k_per_episode = critical_cfg.get("top_k_per_episode", 1)

    mixed_cfg = config.get("mixed_init", {})
    p_mixed = mixed_cfg.get("p_mixed", 0.25)

    rnd_cfg = config.get("rnd", {})
    lambda_rnd = rnd_cfg.get("lambda_coef", 0.01)
    rnd_hidden_sizes = tuple(rnd_cfg.get("hidden_sizes", [64, 64]))
    rnd_embedding_dim = rnd_cfg.get("embedding_dim", 64)
    rnd_lr = rnd_cfg.get("learning_rate", 1e-4)
    normalize_obs = rnd_cfg.get("normalize_obs", True)
    normalize_bonus = rnd_cfg.get("normalize_bonus", True)

    ppo_cfg = config.get("refine_ppo", config.get("ppo", {}))
    ppo_lr = ppo_cfg.get("learning_rate", 3e-4)
    gamma = ppo_cfg.get("gamma", 0.99)
    gae_lambda = ppo_cfg.get("gae_lambda", 0.95)
    clip_epsilon = ppo_cfg.get("clip_epsilon", 0.2)
    value_loss_coef = ppo_cfg.get("value_loss_coef", 0.5)
    entropy_coef = ppo_cfg.get("entropy_coef", 0.01)
    max_grad_norm = ppo_cfg.get("max_grad_norm", 0.5)
    ppo_epochs = ppo_cfg.get("ppo_epochs", 10)
    ppo_batch_size = ppo_cfg.get("batch_size", 64)
    normalize_advantages = ppo_cfg.get("normalize_advantages", True)

    policy_cfg = config.get("policy", {})
    policy_hidden_sizes = policy_cfg.get("hidden_sizes", None)
    value_hidden_sizes = policy_cfg.get("value_hidden_sizes", None)
    policy_std = policy_cfg.get("policy_std", 0.0)
    activation = policy_cfg.get("activation", "tanh")

    device_str = config.get("device", device)
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
        if verbose:
            print("CUDA not available, falling back to CPU")

    # --- Set seed ---
    set_seed(seed)

    # --- Create environment ---
    if verbose:
        print(f"Creating environment: {env_name}")
    env = make_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=refine_cfg.get("max_episode_steps", None),
        use_sparse_reward=sparse,
    )

    state_dim = get_state_dim(env)
    action_dim = get_action_dim(env)
    discrete_action = is_discrete_action(env)
    num_discrete_actions = env.action_space.n if discrete_action else None

    if verbose:
        print(f"State dim: {state_dim}, Action dim: {action_dim}, "
              f"Discrete: {discrete_action}")

    # --- Load target policy ---
    if verbose:
        print(f"Loading target policy from {model_dir}")
    model, vec_normalize = load_target_policy(env_name, model_dir, device_str)
    target_policy_fn = make_target_policy_fn(model, vec_normalize, device_str)

    # --- Load mask network ---
    if verbose:
        print(f"Loading mask network from {mask_dir}")
    mask_network = load_mask_network(env_name, mask_dir, state_dim, device_str)

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Run RICE refinement ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting RICE Refinement for {env_name}")
        print(f"  Total steps: {total_steps}")
        print(f"  Steps per iteration: {steps_per_iteration}")
        print(f"  p_mixed: {p_mixed}")
        print(f"  lambda_rnd: {lambda_rnd}")
        print(f"  Critical episodes: {num_critical_episodes}")
        print(f"  Top-k per episode: {top_k_per_episode}")
        print(f"  Device: {device_str}")
        print(f"{'='*60}\n")

    start_time = time.time()

    refined_policy, history = refine_policy(
        env=env,
        target_policy=target_policy_fn,
        mask_network=mask_network,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        num_discrete_actions=num_discrete_actions,
        device=device_str,
        p_mixed=p_mixed,
        lambda_rnd=lambda_rnd,
        total_steps=total_steps,
        steps_per_iteration=steps_per_iteration,
        num_critical_episodes=num_critical_episodes,
        top_k_per_episode=top_k_per_episode,
        eval_interval=eval_interval,
        eval_episodes=eval_episodes,
        verbose=verbose,
        rnd_hidden_sizes=rnd_hidden_sizes,
        rnd_embedding_dim=rnd_embedding_dim,
        rnd_lr=rnd_lr,
        ppo_lr=ppo_lr,
        ppo_epochs=ppo_epochs,
        ppo_batch_size=ppo_batch_size,
        ppo_clip_epsilon=clip_epsilon,
        gamma=gamma,
        gae_lambda=gae_lambda,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        normalize_advantages=normalize_advantages,
        normalize_obs=normalize_obs,
        normalize_bonus=normalize_bonus,
        policy_std=policy_std,
        value_hidden_sizes=value_hidden_sizes,
        policy_hidden_sizes=policy_hidden_sizes,
        activation=activation,
    )

    training_time = time.time() - start_time

    # --- Evaluate refined policy ---
    if verbose:
        print("\nEvaluating refined policy...")

    # Create a fresh evaluation environment
    eval_env = make_env(
        env_name=env_name,
        seed=seed + 1000,
        max_episode_steps=refine_cfg.get("max_episode_steps", None),
        use_sparse_reward=sparse,
    )

    def refined_policy_fn(state: np.ndarray) -> np.ndarray:
        """Deterministic policy for evaluation."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device_str).unsqueeze(0)
        with torch.no_grad():
            if hasattr(refined_policy, 'get_action'):
                action, _, _, _ = refined_policy.get_action(state_tensor, deterministic=True)
            else:
                action = refined_policy(state_tensor)
        return action.squeeze(0).cpu().numpy()

    eval_results = evaluate_policy(
        eval_env,
        refined_policy_fn,
        num_episodes=eval_episodes,
        max_steps=refine_cfg.get("max_episode_steps", 1000),
        deterministic=True,
        verbose=verbose,
    )

    # --- Evaluate target policy for comparison ---
    if verbose:
        print("Evaluating target policy for comparison...")
    target_eval_env = make_env(
        env_name=env_name,
        seed=seed + 2000,
        max_episode_steps=refine_cfg.get("max_episode_steps", None),
        use_sparse_reward=sparse,
    )

    def target_eval_fn(state: np.ndarray) -> np.ndarray:
        action, _, _, _ = target_policy_fn(state)
        return action

    target_eval_results = evaluate_policy(
        target_eval_env,
        target_eval_fn,
        num_episodes=eval_episodes,
        max_steps=refine_cfg.get("max_episode_steps", 1000),
        deterministic=True,
        verbose=False,
    )

    # --- Save results ---
    results = {
        "env_name": env_name,
        "seed": seed,
        "config": config,
        "training_time": training_time,
        "total_steps": total_steps,
        "p_mixed": p_mixed,
        "lambda_rnd": lambda_rnd,
        "history": history,
        "refined_eval": eval_results,
        "target_eval": target_eval_results,
        "improvement": {
            "mean_reward": eval_results["mean_reward"] - target_eval_results["mean_reward"],
            "relative": (
                (eval_results["mean_reward"] - target_eval_results["mean_reward"])
                / max(abs(target_eval_results["mean_reward"]), 1e-6)
            ),
        },
    }

    # Save results JSON
    results_path = os.path.join(output_dir, f"{env_name}_refine_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save refined policy
    policy_path = os.path.join(output_dir, f"{env_name}_refined_policy.pt")
    if hasattr(refined_policy, 'state_dict'):
        torch.save({
            "policy_state_dict": refined_policy.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "discrete_action": discrete_action,
            "num_discrete_actions": num_discrete_actions,
            "config": config,
        }, policy_path)
    else:
        # If refined_policy is a function, save the RICERefine object
        torch.save({
            "policy_type": "function",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "config": config,
        }, policy_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RICE Refinement Complete!")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Target mean reward: {target_eval_results['mean_reward']:.2f}")
        print(f"  Refined mean reward: {eval_results['mean_reward']:.2f}")
        print(f"  Improvement: {results['improvement']['mean_reward']:.2f}")
        print(f"  Results saved to: {output_dir}")
        print(f"{'='*60}")

    return results


# ==============================================================================
# Baseline: PPO Fine-tuning (no RICE)
# ==============================================================================

def run_ppo_finetune(
    env_name: str = "MetaDrive-Macro-v1",
    model_dir: str = "./trained_agents/autonomous_driving",
    output_dir: str = "./finetuned_agents/autonomous_driving",
    total_steps: int = 2_000_000,
    seed: int = 42,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run standard PPO fine-tuning (no RICE) as a baseline.

    Args:
        env_name: Environment name.
        model_dir: Directory with pre-trained target agent.
        output_dir: Directory to save fine-tuned model.
        total_steps: Total fine-tuning steps.
        seed: Random seed.
        device: Device for training.
        verbose: Whether to print progress.

    Returns:
        Dictionary with evaluation results and training time.
    """
    if not HAS_SB3:
        raise ImportError("Stable-Baselines3 is required for PPO fine-tuning baseline.")

    set_seed(seed)

    # Load target model
    model, vec_normalize = load_target_policy(env_name, model_dir, device)

    if not isinstance(model, PPO):
        raise ValueError("PPO fine-tuning requires a Stable-Baselines3 PPO model.")

    # Create environment
    env = make_ad_env(env_name=env_name, seed=seed)

    # Wrap for SB3
    def make_env_fn():
        return make_ad_env(env_name=env_name, seed=seed)

    vec_env = DummyVecEnv([make_env_fn])

    if vec_normalize is not None:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=vec_normalize.norm_obs,
            norm_reward=vec_normalize.norm_reward,
            clip_obs=vec_normalize.clip_obs,
            clip_reward=vec_normalize.clip_reward,
            gamma=vec_normalize.gamma,
            epsilon=vec_normalize.epsilon,
        )
        # Copy stats from original
        vec_env.obs_rms = vec_normalize.obs_rms
        vec_env.ret_rms = vec_normalize.ret_rms

    model.set_env(vec_env)

    # Fine-tune
    start_time = time.time()
    model.learn(total_timesteps=total_steps, progress_bar=verbose)
    training_time = time.time() - start_time

    # Save
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, f"{env_name}_ppo_finetuned.zip"))
    if vec_normalize is not None:
        vec_env.save(os.path.join(output_dir, f"{env_name}_vecnormalize_finetuned.pkl"))

    # Evaluate
    eval_env = make_ad_env(env_name=env_name, seed=seed + 1000)

    def policy_fn(state):
        action, _ = model.predict(state, deterministic=True)
        return action

    eval_results = evaluate_policy(
        eval_env, policy_fn,
        num_episodes=10,
        max_steps=1000,
        deterministic=True,
        verbose=False,
    )

    results = {
        "env_name": env_name,
        "seed": seed,
        "total_steps": total_steps,
        "training_time": training_time,
        "eval_results": eval_results,
    }

    results_path = os.path.join(output_dir, f"{env_name}_ppo_finetune_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"PPO Fine-tuning complete. Mean reward: {eval_results['mean_reward']:.2f}")

    return results


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RICE Refinement for Autonomous Driving (MetaDrive)"
    )
    parser.add_argument(
        "--env_name", type=str, default="MetaDrive-Macro-v1",
        help="Environment name"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./trained_agents/autonomous_driving",
        help="Directory with pre-trained target agent"
    )
    parser.add_argument(
        "--mask_dir", type=str, default="./trained_masks/autonomous_driving",
        help="Directory with trained mask network"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./refined_agents/autonomous_driving",
        help="Directory to save refined policy and results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML config file"
    )
    parser.add_argument(
        "--total_steps", type=int, default=None,
        help="Total refinement steps (overrides config)"
    )
    parser.add_argument(
        "--p_mixed", type=float, default=None,
        help="Probability of resetting to critical state (overrides config)"
    )
    parser.add_argument(
        "--lambda_rnd", type=float, default=None,
        help="RND bonus coefficient (overrides config)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for training (cuda/cpu)"
    )
    parser.add_argument(
        "--sparse", action="store_true",
        help="Use sparse reward variant"
    )
    parser.add_argument(
        "--baseline", type=str, default="rice",
        choices=["rice", "ppo_finetune"],
        help="Which method to run: 'rice' (default) or 'ppo_finetune'"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    verbose = not args.quiet

    if args.baseline == "ppo_finetune":
        run_ppo_finetune(
            env_name=args.env_name,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            total_steps=args.total_steps or 2_000_000,
            seed=args.seed,
            device=args.device,
            verbose=verbose,
        )
    else:
        run_refine(
            env_name=args.env_name,
            model_dir=args.model_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            total_steps=args.total_steps,
            p_mixed=args.p_mixed,
            lambda_rnd=args.lambda_rnd,
            seed=args.seed,
            device=args.device,
            sparse=args.sparse,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()