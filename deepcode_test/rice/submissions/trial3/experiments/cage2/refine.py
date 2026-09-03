#!/usr/bin/env python3
"""
RICE Refinement Script for CAGE Challenge 2 (Cybersecurity) Domain.

This script orchestrates the full RICE refinement pipeline for the CAGE2
environment: loads a pre-trained target PPO agent and a trained mask network,
collects critical states, then refines the policy via PPO with an RND
exploration bonus and a mixed initial state distribution. Also provides a
baseline PPO fine-tuning mode for comparison.

Usage:
    python experiments/cage2/refine.py --env CAGE2-v0 \
        --model-dir ./trained_agents/cage2 \
        --mask-dir ./trained_masks/cage2 \
        --output-dir ./refined_agents/cage2 \
        --total-steps 1000000 --p-mixed 0.5 --lambda-rnd 0.01

    # Baseline PPO fine-tuning:
    python experiments/cage2/refine.py --env CAGE2-v0 \
        --model-dir ./trained_agents/cage2 \
        --output-dir ./ppo_finetune/cage2 \
        --baseline ppo_finetune
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Optional Stable-Baselines3 import
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# Core RICE modules
from rice.mask_net import MaskNetwork
from rice.rnd import RNDModule, BonusNormalizer
from rice.refine import RICERefine, refine_policy
from rice.utils import (
    TrajectoryBuffer,
    collect_trajectories,
    compute_gae,
    compute_returns,
    evaluate_policy,
    load_state_dict,
    orthogonal_init,
    save_state_dict,
    set_seed,
    to_numpy,
    to_tensor,
)
from rice.env_wrappers import (
    MuJoCoStateWrapper,
    StateSaveWrapper,
    make_state_saveable,
    reset_env_to_state,
    restore_env_state,
    save_env_state,
)

# CAGE2 environment
from experiments.cage2.env import (
    Cage2StateWrapper,
    SimulatedCage2Env,
    SparseRewardWrapper,
    get_action_dim,
    get_state_dim,
    is_discrete_action,
    make_env as make_cage2_env,
)


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def _deep_update(base: Dict, override: Dict) -> Dict:
    """Recursively update base dict with override values."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    env_name: str = "CAGE2-v0",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and merge default refine config, default mask config,
    environment-specific YAML, and optional custom config.

    Args:
        env_name: Environment name (used to locate env-specific YAML).
        config_path: Optional path to a custom YAML config for overrides.

    Returns:
        Merged configuration dictionary.
    """
    config = {}

    # Load default refine config
    default_refine_path = Path(__file__).parent.parent.parent / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            config = _deep_update(config, yaml.safe_load(f) or {})

    # Load default mask config
    default_mask_path = Path(__file__).parent.parent.parent / "configs" / "default_mask.yaml"
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            config = _deep_update(config, yaml.safe_load(f) or {})

    # Load environment-specific config
    env_config_path = (
        Path(__file__).parent.parent.parent / "configs" / "env_specific" / "cage2.yaml"
    )
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            config = _deep_update(config, yaml.safe_load(f) or {})

    # Load custom config if provided
    if config_path is not None and Path(config_path).exists():
        with open(config_path, "r") as f:
            config = _deep_update(config, yaml.safe_load(f) or {})

    return config


# ---------------------------------------------------------------------------
# Environment Creation
# ---------------------------------------------------------------------------

def make_env(
    env_name: str = "CAGE2-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False,
    use_sparse_reward: bool = False,
) -> gym.Env:
    """
    Create a CAGE2 environment wrapped with state save/restore capability.

    Args:
        env_name: Environment name.
        seed: Random seed.
        max_episode_steps: Maximum steps per episode.
        use_real_env: Whether to use the real CybORG environment.
        use_sparse_reward: Whether to apply sparse reward wrapper.

    Returns:
        Wrapped gym environment.
    """
    env = make_cage2_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env,
    )

    if use_sparse_reward:
        env = SparseRewardWrapper(env)

    # Ensure state save/restore capability
    env = make_state_saveable(env)
    return env


# ---------------------------------------------------------------------------
# Policy Loading
# ---------------------------------------------------------------------------

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
    model_path = Path(model_dir) / f"{env_name}_ppo_final.zip"
    pt_path = Path(model_dir) / f"{env_name}_target_policy.pt"
    vecnorm_path = Path(model_dir) / f"{env_name}_vecnormalize.pkl"

    vec_normalize = None
    if vecnorm_path.exists():
        try:
            with open(vecnorm_path, "rb") as f:
                vec_normalize = pickle.load(f)
        except Exception:
            pass

    # Try Stable-Baselines3 first
    if model_path.exists() and HAS_SB3:
        model = PPO.load(str(model_path), device=device)
        return model, vec_normalize

    # Try raw PyTorch checkpoint
    if pt_path.exists():
        model = torch.load(pt_path, map_location=device)
        return model, vec_normalize

    raise FileNotFoundError(
        f"No target policy found at {model_path} or {pt_path}"
    )


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], Tuple[np.ndarray, float, float, float]]:
    """
    Wrap a loaded model into a policy function returning
    (action, log_prob, value, entropy).

    Args:
        model: Loaded model (SB3 PPO or raw PyTorch).
        vec_normalize: Optional VecNormalize for observation normalization.
        device: Device for tensor operations.

    Returns:
        Policy function: state -> (action, log_prob, value, entropy)
    """
    if HAS_SB3 and hasattr(model, "predict"):
        # Stable-Baselines3 model
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            action, _ = model.predict(state, deterministic=False)
            # For discrete actions, return action index
            if isinstance(action, np.ndarray) and action.ndim == 0:
                action_idx = int(action)
            elif isinstance(action, np.ndarray):
                action_idx = int(action[0]) if len(action) > 0 else 0
            else:
                action_idx = int(action)

            # Get log_prob and value from the model's policy network
            obs_tensor = to_tensor(state, device).unsqueeze(0)
            with torch.no_grad():
                try:
                    # SB3 PPO internal access
                    features = model.policy.mlp_extractor.shared_net(obs_tensor)
                    if hasattr(model.policy, 'action_net'):
                        action_logits = model.policy.action_net(features)
                        value = model.policy.value_net(features).item()
                        log_prob = F.log_softmax(action_logits, dim=-1)[0, action_idx].item()
                        entropy = -torch.sum(
                            F.softmax(action_logits, dim=-1) * F.log_softmax(action_logits, dim=-1)
                        ).item()
                    else:
                        value = 0.0
                        log_prob = 0.0
                        entropy = 0.0
                except Exception:
                    value = 0.0
                    log_prob = 0.0
                    entropy = 0.0

            return np.array([action_idx]), log_prob, value, entropy

        return policy_fn

    elif hasattr(model, "get_action"):
        # Raw PyTorch model with get_action method
        def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            action, log_prob, value, entropy = model.get_action(state)
            return action, log_prob, value, entropy
        return policy_fn

    else:
        raise ValueError("Unsupported model type for target policy")


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
        state_dim: State dimension.
        device: Device to load on.

    Returns:
        Loaded MaskNetwork.
    """
    mask_path = Path(mask_dir) / f"{env_name}_mask_network.pt"

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask network not found at {mask_path}")

    checkpoint = torch.load(mask_path, map_location=device)

    # Determine hidden sizes from checkpoint or use defaults
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


# ---------------------------------------------------------------------------
# Main Refinement Pipeline
# ---------------------------------------------------------------------------

def run_refine(
    env_name: str = "CAGE2-v0",
    model_dir: str = "./trained_agents/cage2",
    mask_dir: str = "./trained_masks/cage2",
    output_dir: str = "./refined_agents/cage2",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    p_mixed: Optional[float] = None,
    lambda_rnd: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    use_real_env: bool = False,
    use_sparse_reward: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full RICE refinement pipeline on the CAGE2 environment.

    Steps:
    1. Load configuration and override with CLI arguments.
    2. Load pre-trained target policy.
    3. Load trained mask network.
    4. Collect critical states using the mask network.
    5. Refine the policy via PPO with RND exploration bonus and
       mixed initial state distribution.
    6. Evaluate the refined policy.
    7. Save all results.

    Args:
        env_name: Environment name.
        model_dir: Directory with pre-trained target agent.
        mask_dir: Directory with trained mask network.
        output_dir: Directory to save refined agent and results.
        config_path: Optional custom YAML config path.
        total_steps: Override total refinement steps.
        p_mixed: Override mixed initial distribution probability.
        lambda_rnd: Override RND exploration bonus coefficient.
        seed: Random seed.
        device: Device for computation ("cuda" or "cpu").
        use_real_env: Whether to use real CybORG environment.
        use_sparse_reward: Whether to use sparse reward variant.
        verbose: Whether to print progress.

    Returns:
        Dictionary with refined policy, training history, evaluation
        rewards, and configuration.
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
    state_dim = config.get("state_dim", 19)
    action_dim = config.get("action_dim", 12)
    max_episode_steps = config.get("max_episode_steps", 100)
    discrete_action = True  # CAGE2 is discrete

    refine_total_steps = config.get("refine_training", {}).get("total_steps", 1_000_000)
    steps_per_iteration = config.get("refine_training", {}).get("steps_per_iteration", 2048)
    eval_interval = config.get("refine_training", {}).get("eval_interval", 10)
    eval_episodes = config.get("refine_training", {}).get("eval_episodes", 10)
    save_interval = config.get("refine_training", {}).get("save_interval", 50)

    p_mixed_val = config.get("mixed_init", {}).get("p_mixed", 0.5)
    lambda_rnd_val = config.get("rnd", {}).get("lambda_coef", 0.01)
    rnd_hidden_sizes = tuple(config.get("rnd", {}).get("hidden_sizes", [64, 64]))
    rnd_embedding_dim = config.get("rnd", {}).get("embedding_dim", 64)
    rnd_lr = config.get("rnd", {}).get("learning_rate", 1e-4)
    normalize_obs = config.get("rnd", {}).get("normalize_obs", True)
    normalize_bonus = config.get("rnd", {}).get("normalize_bonus", True)

    num_critical_episodes = config.get("critical_state", {}).get("num_episodes", 100)
    top_k_per_episode = config.get("critical_state", {}).get("top_k_per_episode", 1)
    max_critical_states = config.get("critical_state", {}).get("max_critical_states", 100)

    ppo_lr = config.get("refine_ppo", {}).get("learning_rate", 3e-4)
    ppo_gamma = config.get("refine_ppo", {}).get("gamma", 0.99)
    ppo_gae_lambda = config.get("refine_ppo", {}).get("gae_lambda", 0.95)
    ppo_clip_epsilon = config.get("refine_ppo", {}).get("clip_epsilon", 0.2)
    ppo_value_loss_coef = config.get("refine_ppo", {}).get("value_loss_coef", 0.5)
    ppo_entropy_coef = config.get("refine_ppo", {}).get("entropy_coef", 0.01)
    ppo_max_grad_norm = config.get("refine_ppo", {}).get("max_grad_norm", 0.5)
    ppo_epochs = config.get("refine_ppo", {}).get("ppo_epochs", 10)
    ppo_batch_size = config.get("refine_ppo", {}).get("batch_size", 64)
    normalize_advantages = config.get("refine_ppo", {}).get("normalize_advantages", True)

    policy_hidden_sizes = tuple(config.get("policy", {}).get("hidden_sizes", [128, 128]))
    value_hidden_sizes = tuple(config.get("policy", {}).get("value_hidden_sizes", [128, 128]))
    policy_activation = config.get("policy", {}).get("activation", "tanh")

    device_val = device or config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    log_dir = config.get("logging", {}).get("log_dir", "./logs/cage2")

    if verbose:
        print(f"{'='*60}")
        print(f"RICE Refinement: {env_name}")
        print(f"{'='*60}")
        print(f"Configuration:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  p_mixed: {p_mixed_val}, lambda_rnd: {lambda_rnd_val}")
        print(f"  Refine steps: {refine_total_steps}")
        print(f"  Device: {device_val}")
        print(f"  Use real env: {use_real_env}")
        print(f"  Sparse reward: {use_sparse_reward}")

    # --- Set seed ---
    set_seed(seed)

    # --- Create environment ---
    env = make_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env,
        use_sparse_reward=use_sparse_reward,
    )

    # --- Load target policy ---
    if verbose:
        print(f"\nLoading target policy from {model_dir}...")
    target_model, vec_normalize = load_target_policy(env_name, model_dir, device_val)
    target_policy_fn = make_target_policy_fn(target_model, vec_normalize, device_val)

    # Evaluate target policy before refinement
    if verbose:
        print("Evaluating target policy...")
    target_eval = evaluate_policy(
        env, target_policy_fn,
        num_episodes=eval_episodes,
        max_steps=max_episode_steps,
        deterministic=True,
        verbose=False,
    )
    if verbose:
        print(f"  Target mean reward: {target_eval['mean_reward']:.4f} "
              f"± {target_eval['std_reward']:.4f}")

    # --- Load mask network ---
    if verbose:
        print(f"\nLoading mask network from {mask_dir}...")
    mask_network = load_mask_network(env_name, mask_dir, state_dim, device_val)

    # --- Run refinement ---
    if verbose:
        print(f"\nStarting RICE refinement...")
        print(f"  Total steps: {refine_total_steps}")
        print(f"  Steps per iteration: {steps_per_iteration}")

    start_time = time.time()

    refined_policy, history = refine_policy(
        env=env,
        target_policy=target_policy_fn,
        mask_network=mask_network,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        num_discrete_actions=action_dim,
        device=device_val,
        p_mixed=p_mixed_val,
        lambda_rnd=lambda_rnd_val,
        total_steps=refine_total_steps,
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
        ppo_clip_epsilon=ppo_clip_epsilon,
        gamma=ppo_gamma,
        gae_lambda=ppo_gae_lambda,
        value_loss_coef=ppo_value_loss_coef,
        entropy_coef=ppo_entropy_coef,
        max_grad_norm=ppo_max_grad_norm,
        normalize_advantages=normalize_advantages,
        normalize_obs=normalize_obs,
        normalize_bonus=normalize_bonus,
        policy_hidden_sizes=policy_hidden_sizes,
        value_hidden_sizes=value_hidden_sizes,
        policy_activation=policy_activation,
    )

    training_time = time.time() - start_time

    # --- Evaluate refined policy ---
    if verbose:
        print(f"\nEvaluating refined policy...")

    def refined_policy_fn(state: np.ndarray) -> np.ndarray:
        """Deterministic policy wrapper for evaluation."""
        with torch.no_grad():
            state_tensor = to_tensor(state, device_val).unsqueeze(0)
            if discrete_action:
                logits, _ = refined_policy(state_tensor)
                action = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                action_mean, _ = refined_policy(state_tensor)
                action = action_mean.cpu().numpy().flatten()
            return action

    refined_eval = evaluate_policy(
        env, refined_policy_fn,
        num_episodes=eval_episodes,
        max_steps=max_episode_steps,
        deterministic=True,
        verbose=False,
    )

    if verbose:
        print(f"  Refined mean reward: {refined_eval['mean_reward']:.4f} "
              f"± {refined_eval['std_reward']:.4f}")
        improvement = refined_eval['mean_reward'] - target_eval['mean_reward']
        print(f"  Improvement: {improvement:+.4f}")

    # --- Save results ---
    os.makedirs(output_dir, exist_ok=True)

    # Save refined policy
    policy_path = Path(output_dir) / f"{env_name}_refined_policy.pt"
    torch.save({
        "model_state_dict": refined_policy.state_dict(),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "discrete_action": discrete_action,
        "num_discrete_actions": action_dim,
        "hidden_sizes": policy_hidden_sizes,
        "value_hidden_sizes": value_hidden_sizes,
        "activation": policy_activation,
        "config": {
            "p_mixed": p_mixed_val,
            "lambda_rnd": lambda_rnd_val,
            "total_steps": refine_total_steps,
        },
    }, policy_path)

    # Save training history
    history_path = Path(output_dir) / f"{env_name}_refine_history.json"
    # Convert any non-serializable values
    serializable_history = []
    for entry in history:
        serializable_entry = {}
        for k, v in entry.items():
            if isinstance(v, (np.ndarray,)):
                serializable_entry[k] = v.tolist()
            elif isinstance(v, (np.float32, np.float64)):
                serializable_entry[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                serializable_entry[k] = int(v)
            else:
                serializable_entry[k] = v
        serializable_history.append(serializable_entry)

    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)

    # Save evaluation results
    results = {
        "env_name": env_name,
        "seed": seed,
        "target_mean_reward": target_eval["mean_reward"],
        "target_std_reward": target_eval["std_reward"],
        "refined_mean_reward": refined_eval["mean_reward"],
        "refined_std_reward": refined_eval["std_reward"],
        "improvement": refined_eval["mean_reward"] - target_eval["mean_reward"],
        "improvement_percent": (
            (refined_eval["mean_reward"] - target_eval["mean_reward"])
            / (abs(target_eval["mean_reward"]) + 1e-8) * 100
        ),
        "training_time_seconds": training_time,
        "total_steps": refine_total_steps,
        "p_mixed": p_mixed_val,
        "lambda_rnd": lambda_rnd_val,
        "target_all_rewards": target_eval.get("all_rewards", []),
        "refined_all_rewards": refined_eval.get("all_rewards", []),
    }
    results_path = Path(output_dir) / f"{env_name}_refine_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save configuration used
    config_path_out = Path(output_dir) / f"{env_name}_refine_config.yaml"
    with open(config_path_out, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    if verbose:
        print(f"\nResults saved to {output_dir}/")
        print(f"  - {env_name}_refined_policy.pt")
        print(f"  - {env_name}_refine_history.json")
        print(f"  - {env_name}_refine_results.json")
        print(f"  - {env_name}_refine_config.yaml")
        print(f"\nTraining time: {training_time:.1f}s "
              f"({training_time/3600:.2f}h)")

    return {
        "refined_policy": refined_policy,
        "history": history,
        "target_eval": target_eval,
        "refined_eval": refined_eval,
        "results": results,
        "config": config,
        "training_time": training_time,
    }


# ---------------------------------------------------------------------------
# Baseline: PPO Fine-tuning
# ---------------------------------------------------------------------------

def run_ppo_finetune(
    env_name: str = "CAGE2-v0",
    model_dir: str = "./trained_agents/cage2",
    output_dir: str = "./ppo_finetune/cage2",
    total_steps: int = 1_000_000,
    seed: int = 42,
    device: str = "cpu",
    use_real_env: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run standard PPO fine-tuning as a baseline (no RICE components).

    Args:
        env_name: Environment name.
        model_dir: Directory with pre-trained target agent.
        output_dir: Directory to save fine-tuned agent.
        total_steps: Total fine-tuning steps.
        seed: Random seed.
        device: Device for computation.
        use_real_env: Whether to use real CybORG environment.
        verbose: Whether to print progress.

    Returns:
        Dictionary with evaluation results and training time.
    """
    if not HAS_SB3:
        raise ImportError(
            "Stable-Baselines3 is required for PPO fine-tuning baseline. "
            "Install with: pip install stable-baselines3"
        )

    set_seed(seed)

    # Load target policy
    model, vec_normalize = load_target_policy(env_name, model_dir, device)

    # Create vectorized environment
    def _make_env():
        env = make_cage2_env(
            env_name=env_name,
            seed=seed,
            max_episode_steps=None,
            use_real_env=use_real_env,
        )
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([_make_env])
    if vec_normalize is not None:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=vec_normalize.norm_obs,
            norm_reward=vec_normalize.norm_reward,
            gamma=vec_normalize.gamma,
        )

    # Set up model with environment
    model.set_env(vec_env)

    if verbose:
        print(f"Starting PPO fine-tuning for {total_steps} steps...")

    start_time = time.time()

    # Fine-tune
    model.learn(total_timesteps=total_steps, reset_num_timesteps=False)

    training_time = time.time() - start_time

    # Save fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / f"{env_name}_ppo_finetuned.zip"
    model.save(str(model_path))

    # Evaluate
    eval_env = make_env(
        env_name=env_name,
        seed=seed + 1000,
        max_episode_steps=None,
        use_real_env=use_real_env,
    )

    def policy_fn(state: np.ndarray) -> np.ndarray:
        if vec_normalize is not None:
            state = vec_normalize.normalize_obs(state)
        action, _ = model.predict(state, deterministic=True)
        if isinstance(action, np.ndarray) and action.ndim == 0:
            return np.array([int(action)])
        return action

    eval_results = evaluate_policy(
        eval_env, policy_fn,
        num_episodes=10,
        max_steps=100,
        deterministic=True,
        verbose=False,
    )

    if verbose:
        print(f"  Fine-tuned mean reward: {eval_results['mean_reward']:.4f} "
              f"± {eval_results['std_reward']:.4f}")
        print(f"  Training time: {training_time:.1f}s")

    # Save results
    results = {
        "env_name": env_name,
        "seed": seed,
        "total_steps": total_steps,
        "mean_reward": eval_results["mean_reward"],
        "std_reward": eval_results["std_reward"],
        "training_time_seconds": training_time,
        "all_rewards": eval_results.get("all_rewards", []),
    }
    results_path = Path(output_dir) / f"{env_name}_ppo_finetune_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RICE Refinement for CAGE Challenge 2"
    )
    parser.add_argument(
        "--env", type=str, default="CAGE2-v0",
        help="Environment name (default: CAGE2-v0)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents/cage2",
        help="Directory with pre-trained target agent"
    )
    parser.add_argument(
        "--mask-dir", type=str, default="./trained_masks/cage2",
        help="Directory with trained mask network"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./refined_agents/cage2",
        help="Directory to save refined agent and results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional custom YAML config path"
    )
    parser.add_argument(
        "--total-steps", type=int, default=None,
        help="Override total refinement steps"
    )
    parser.add_argument(
        "--p-mixed", type=float, default=None,
        help="Override mixed initial distribution probability"
    )
    parser.add_argument(
        "--lambda-rnd", type=float, default=None,
        help="Override RND exploration bonus coefficient"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for computation (cuda/cpu, default: auto)"
    )
    parser.add_argument(
        "--use-real-env", action="store_true",
        help="Use real CybORG environment instead of simulated"
    )
    parser.add_argument(
        "--sparse", action="store_true",
        help="Use sparse reward variant"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        choices=["ppo_finetune"],
        help="Run a baseline instead of RICE refinement"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress (default: True)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    verbose = not args.quiet

    try:
        if args.baseline == "ppo_finetune":
            results = run_ppo_finetune(
                env_name=args.env,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                total_steps=args.total_steps or 1_000_000,
                seed=args.seed,
                device=args.device or "cpu",
                use_real_env=args.use_real_env,
                verbose=verbose,
            )
        else:
            results = run_refine(
                env_name=args.env,
                model_dir=args.model_dir,
                mask_dir=args.mask_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                total_steps=args.total_steps,
                p_mixed=args.p_mixed,
                lambda_rnd=args.lambda_rnd,
                seed=args.seed,
                device=args.device,
                use_real_env=args.use_real_env,
                use_sparse_reward=args.sparse,
                verbose=verbose,
            )

        if verbose:
            print(f"\n{'='*60}")
            print("Refinement complete!")
            print(f"{'='*60}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())