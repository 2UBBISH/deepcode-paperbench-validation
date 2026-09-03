#!/usr/bin/env python3
"""
RICE Refinement Script for MuJoCo Environments
===============================================
Runs the full RICE refining pipeline on a pre-trained MuJoCo agent:
1. Loads the target PPO agent (from train_target.py)
2. Loads the trained mask network (from train_mask.py)
3. Collects critical states using the mask network
4. Refines the policy via PPO with RND exploration bonus and mixed initial distribution

Supports: Hopper-v4, Walker2d-v4, Reacher-v4, HalfCheetah-v4
Also supports sparse reward variants.

Usage:
    python experiments/mujoco/refine.py --env Hopper-v4 --model_dir ./trained_agents --mask_dir ./trained_masks --output_dir ./refined_agents
    python experiments/mujoco/refine.py --env Walker2d-v4 --p_mixed 0.25 --lambda_rnd 0.01 --total_steps 1000000
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
import torch.nn as nn
import yaml
import gym

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from rice.mask_net import MaskNetwork
from rice.rnd import RNDModule, BonusNormalizer
from rice.refine import RICERefine, refine_policy
from rice.utils import (
    evaluate_policy, set_seed, to_tensor, to_numpy,
    save_state_dict, load_state_dict, orthogonal_init,
    TrajectoryBuffer, collect_trajectories, compute_gae, compute_returns
)
from rice.env_wrappers import (
    StateSaveWrapper, MuJoCoStateWrapper, make_state_saveable,
    save_env_state, restore_env_state, reset_env_to_state
)


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_config(env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and merge default refine config with environment-specific overrides.
    
    Args:
        env_name: Name of the environment (e.g., "Hopper-v4")
        config_path: Optional path to a custom config file
        
    Returns:
        Merged configuration dictionary
    """
    # Load default refine config
    default_config_path = _project_root / "configs" / "default_refine.yaml"
    if not default_config_path.exists():
        print(f"[WARNING] Default refine config not found at {default_config_path}, using built-in defaults")
        config = _get_builtin_defaults()
    else:
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load environment-specific config
    env_name_short = env_name.lower().replace('-v4', '').replace('-v3', '').replace('-v2', '')
    env_config_path = _project_root / "configs" / "env_specific" / f"{env_name_short}.yaml"
    if env_config_path.exists():
        with open(env_config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        # Deep merge: env-specific overrides default
        config = _deep_merge(config, env_config)
    else:
        print(f"[WARNING] No env-specific config found at {env_config_path}, using defaults")
    
    # Load custom config if provided
    if config_path is not None:
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        config = _deep_merge(config, custom_config)
    
    return config


def _get_builtin_defaults() -> Dict[str, Any]:
    """Return sensible built-in defaults for the refining phase."""
    return {
        "critical_state": {
            "num_episodes": 100,
            "top_k_per_episode": 1,
            "max_critical_states": 100,
        },
        "mixed_init": {
            "p_mixed": 0.25,
        },
        "rnd": {
            "lambda_coef": 0.01,
            "hidden_sizes": [64, 64],
            "embedding_dim": 64,
            "activation": "relu",
            "learning_rate": 1e-4,
            "normalize_obs": True,
            "obs_rms_decay": 0.99,
            "normalize_bonus": True,
            "bonus_decay": 0.99,
            "update_epochs": 4,
            "batch_size": 256,
        },
        "ppo": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "ppo_epochs": 10,
            "batch_size": 64,
            "normalize_advantages": True,
            "normalize_obs": False,
        },
        "policy": {
            "hidden_sizes": [64, 64],
            "value_hidden_sizes": [64, 64],
            "policy_std": 0.0,
            "activation": "tanh",
        },
        "training": {
            "total_steps": 1_000_000,
            "steps_per_iteration": 2048,
            "eval_interval": 10,
            "eval_episodes": 10,
            "eval_max_steps": 1000,
            "save_interval": 50,
            "seed": 42,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "logging": {
            "verbose": True,
            "log_dir": "./logs/refine",
            "use_tensorboard": False,
        },
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
# Environment Creation
# ==============================================================================

def make_env(env_name: str, seed: int = 42, max_episode_steps: Optional[int] = None) -> gym.Env:
    """
    Create a MuJoCo environment with optional max episode steps override.
    
    Args:
        env_name: Gym environment name (e.g., "Hopper-v4")
        seed: Random seed
        max_episode_steps: Optional max steps per episode
        
    Returns:
        Gym environment wrapped with StateSaveWrapper
    """
    env = gym.make(env_name)
    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps
    
    # Set seed
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
    # Wrap for state save/restore
    env = make_state_saveable(env)
    
    return env


def make_sparse_env(env_name: str, seed: int = 42, threshold: float = 1.0) -> gym.Env:
    """
    Create a sparse reward variant of a MuJoCo environment.
    Reward is 1.0 if the agent's x-position exceeds threshold, else 0.0.
    
    Args:
        env_name: Base MuJoCo environment name
        seed: Random seed
        threshold: x-position threshold for sparse reward
        
    Returns:
        Wrapped environment with sparse rewards
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    
    # Wrap with sparse reward
    env = SparseRewardWrapper(env, threshold=threshold)
    env = make_state_saveable(env)
    
    return env


class SparseRewardWrapper(gym.Wrapper):
    """Converts dense MuJoCo rewards to sparse: reward=1 if x > threshold, else 0."""
    
    def __init__(self, env: gym.Env, threshold: float = 1.0):
        super().__init__(env)
        self.threshold = threshold
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Use x-position (index 0 for most MuJoCo locomotion tasks)
        x_position = obs[0] if isinstance(obs, np.ndarray) else obs
        sparse_reward = 1.0 if x_position > self.threshold else 0.0
        # Keep original info but override reward
        info['original_reward'] = reward
        return obs, sparse_reward, terminated, truncated, info


# ==============================================================================
# Target Policy Loading
# ==============================================================================

def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu"
) -> Tuple[Any, Optional[Any]]:
    """
    Load a pre-trained Stable-Baselines3 PPO model and optional VecNormalize stats.
    
    Args:
        env_name: Environment name
        model_dir: Directory containing saved models
        device: Device to load model on
        
    Returns:
        Tuple of (ppo_model, vec_normalize) where vec_normalize may be None
    """
    from stable_baselines3 import PPO
    
    model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Target model not found at {model_path}. Run train_target.py first.")
    
    ppo_model = PPO.load(model_path, device=device)
    
    # Load VecNormalize stats if available
    vec_normalize = None
    norm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            vec_normalize = pickle.load(f)
    
    return ppo_model, vec_normalize


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any],
    device: str = "cpu"
) -> Callable[[np.ndarray], Tuple[np.ndarray, float, float, float]]:
    """
    Create a policy function compatible with RICERefine from an SB3 PPO model.
    
    Args:
        model: Stable-Baselines3 PPO model
        vec_normalize: Optional VecNormalize for observation normalization
        device: Device for computation
        
    Returns:
        Function: state -> (action, log_prob, value, entropy)
    """
    def policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        # Normalize observation if needed
        if vec_normalize is not None:
            state = vec_normalize.normalize_obs(state)
        
        # Convert to tensor
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Extract features from policy network
            if hasattr(model.policy, 'mlp_extractor'):
                # SB3 MLP policy
                features = model.policy.mlp_extractor.policy_net(
                    model.policy.features_extractor(state_tensor)
                )
            elif hasattr(model.policy, 'features_extractor'):
                features = model.policy.features_extractor(state_tensor)
            else:
                features = state_tensor
            
            # Get action distribution parameters
            if hasattr(model.policy, 'action_net'):
                action_mean = model.policy.action_net(features)
            else:
                action_mean = features
            
            # Get log_std
            if hasattr(model.policy, 'log_std'):
                log_std = model.policy.log_std
            else:
                log_std = torch.zeros_like(action_mean)
            
            # Sample action
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            # Get value
            if hasattr(model.policy, 'value_net'):
                value = model.policy.value_net(features)
            else:
                value = torch.zeros(1, device=device)
        
        return (
            action.cpu().numpy().flatten(),
            log_prob.item(),
            value.item(),
            entropy.item()
        )
    
    return policy_fn


# ==============================================================================
# Mask Network Loading
# ==============================================================================

def load_mask_network(
    env_name: str,
    mask_dir: str,
    state_dim: int,
    device: str = "cpu"
) -> MaskNetwork:
    """
    Load a trained mask network from disk.
    
    Args:
        env_name: Environment name
        mask_dir: Directory containing saved mask networks
        state_dim: State dimension
        device: Device to load on
        
    Returns:
        Loaded MaskNetwork
    """
    mask_path = os.path.join(mask_dir, f"{env_name}_mask_network.pt")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask network not found at {mask_path}. Run train_mask.py first.")
    
    # Load config to get hidden sizes
    env_name_short = env_name.lower().replace('-v4', '').replace('-v3', '').replace('-v2', '')
    config_path = _project_root / "configs" / "env_specific" / f"{env_name_short}.yaml"
    hidden_sizes = [128, 128]
    if config_path.exists():
        with open(config_path, 'r') as f:
            env_config = yaml.safe_load(f)
        hidden_sizes = env_config.get("mask_network", {}).get("hidden_sizes", [128, 128])
    
    mask_network = MaskNetwork(
        state_dim=state_dim,
        hidden_sizes=tuple(hidden_sizes),
        activation="tanh"
    )
    
    checkpoint = torch.load(mask_path, map_location=device)
    mask_network.load_state_dict(checkpoint['model_state_dict'])
    mask_network.to(device)
    mask_network.eval()
    
    return mask_network


# ==============================================================================
# Main Refinement Function
# ==============================================================================

def run_refine(
    env_name: str,
    model_dir: str,
    mask_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    p_mixed: Optional[float] = None,
    lambda_rnd: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    sparse: bool = False,
    sparse_threshold: float = 1.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full RICE refinement pipeline on a MuJoCo environment.
    
    Args:
        env_name: MuJoCo environment name (e.g., "Hopper-v4")
        model_dir: Directory with pre-trained target PPO agent
        mask_dir: Directory with trained mask network
        output_dir: Directory to save refined agent and results
        config_path: Optional path to custom YAML config
        total_steps: Override total training steps
        p_mixed: Override mixed initial distribution probability
        lambda_rnd: Override RND exploration bonus coefficient
        seed: Random seed
        device: Device to use ("cuda" or "cpu")
        sparse: Whether to use sparse reward variant
        sparse_threshold: Threshold for sparse reward
        verbose: Print progress information
        
    Returns:
        Dictionary with results: refined_policy, history, eval_rewards, critical_states
    """
    # --- Configuration ---
    config = load_config(env_name, config_path)
    
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    if total_steps is not None:
        config["training"]["total_steps"] = total_steps
    if p_mixed is not None:
        config["mixed_init"]["p_mixed"] = p_mixed
    if lambda_rnd is not None:
        config["rnd"]["lambda_coef"] = lambda_rnd
    
    # Set seed
    set_seed(seed)
    
    # --- Environment Setup ---
    if verbose:
        print(f"[RICE Refine] Creating environment: {env_name}")
    
    if sparse:
        env = make_sparse_env(env_name, seed=seed, threshold=sparse_threshold)
    else:
        env = make_env(env_name, seed=seed)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, 'shape'):
        action_dim = env.action_space.shape[0]
        discrete_action = False
        num_discrete_actions = None
    else:
        action_dim = 1
        discrete_action = True
        num_discrete_actions = env.action_space.n
    
    if verbose:
        print(f"  State dim: {state_dim}, Action dim: {action_dim}, Discrete: {discrete_action}")
    
    # --- Load Target Policy ---
    if verbose:
        print(f"[RICE Refine] Loading target policy from {model_dir}")
    
    ppo_model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(ppo_model, vec_normalize, device)
    
    # Evaluate target policy before refinement
    if verbose:
        target_reward = evaluate_policy(
            env, target_policy_fn,
            num_episodes=config["training"].get("eval_episodes", 10),
            max_steps=config["training"].get("eval_max_steps", 1000),
            deterministic=True
        )
        print(f"  Target policy mean reward: {target_reward['mean_reward']:.2f} ± {target_reward['std_reward']:.2f}")
    
    # --- Load Mask Network ---
    if verbose:
        print(f"[RICE Refine] Loading mask network from {mask_dir}")
    
    mask_network = load_mask_network(env_name, mask_dir, state_dim, device)
    
    # --- Run RICE Refinement ---
    if verbose:
        print(f"[RICE Refine] Starting refinement...")
        print(f"  p_mixed = {config['mixed_init']['p_mixed']}")
        print(f"  lambda_rnd = {config['rnd']['lambda_coef']}")
        print(f"  total_steps = {config['training']['total_steps']}")
        print(f"  steps_per_iteration = {config['training']['steps_per_iteration']}")
    
    start_time = time.time()
    
    # Use the convenience function from rice.refine
    refined_policy, history = refine_policy(
        env=env,
        target_policy=target_policy_fn,
        mask_network=mask_network,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        num_discrete_actions=num_discrete_actions,
        device=device,
        p_mixed=config["mixed_init"]["p_mixed"],
        lambda_rnd=config["rnd"]["lambda_coef"],
        total_steps=config["training"]["total_steps"],
        steps_per_iteration=config["training"]["steps_per_iteration"],
        num_critical_episodes=config["critical_state"]["num_episodes"],
        top_k_per_episode=config["critical_state"]["top_k_per_episode"],
        eval_interval=config["training"]["eval_interval"],
        eval_episodes=config["training"]["eval_episodes"],
        verbose=verbose,
        save_path=None,
        # Additional kwargs from config
        rnd_hidden_sizes=tuple(config["rnd"]["hidden_sizes"]),
        rnd_embedding_dim=config["rnd"]["embedding_dim"],
        rnd_lr=config["rnd"]["learning_rate"],
        ppo_lr=config["ppo"]["learning_rate"],
        ppo_epochs=config["ppo"]["ppo_epochs"],
        ppo_batch_size=config["ppo"]["batch_size"],
        ppo_clip_epsilon=config["ppo"]["clip_epsilon"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        value_loss_coef=config["ppo"]["value_loss_coef"],
        entropy_coef=config["ppo"]["entropy_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        normalize_advantages=config["ppo"]["normalize_advantages"],
        normalize_obs=config["rnd"]["normalize_obs"],
        normalize_bonus=config["rnd"]["normalize_bonus"],
        policy_std=config["policy"].get("policy_std", 0.0),
        value_hidden_sizes=tuple(config["policy"].get("value_hidden_sizes", [64, 64])),
    )
    
    training_time = time.time() - start_time
    
    if verbose:
        print(f"[RICE Refine] Refinement completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    
    # --- Evaluate Refined Policy ---
    if verbose:
        print(f"[RICE Refine] Evaluating refined policy...")
    
    # Create a policy function from the refined policy module
    def refined_policy_fn(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if hasattr(refined_policy, 'get_action'):
                action = refined_policy.get_action(state_tensor, deterministic=True)
            elif hasattr(refined_policy, 'forward'):
                action_mean = refined_policy(state_tensor)
                action = action_mean.cpu().numpy().flatten()
            else:
                action = refined_policy(state_tensor).cpu().numpy().flatten()
        return action
    
    eval_result = evaluate_policy(
        env, refined_policy_fn,
        num_episodes=config["training"].get("eval_episodes", 10),
        max_steps=config["training"].get("eval_max_steps", 1000),
        deterministic=True
    )
    
    if verbose:
        print(f"  Refined policy mean reward: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
    
    # --- Save Results ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Save refined policy
    policy_path = os.path.join(output_dir, f"{env_name}_refined_policy.pt")
    torch.save({
        'model_state_dict': refined_policy.state_dict() if hasattr(refined_policy, 'state_dict') else None,
        'config': {k: v for k, v in config.items() if not callable(v)},
        'env_name': env_name,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'discrete_action': discrete_action,
    }, policy_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"{env_name}_refine_history.json")
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
    
    with open(history_path, 'w') as f:
        json.dump(serializable_history, f, indent=2)
    
    # Save evaluation results
    eval_path = os.path.join(output_dir, f"{env_name}_refine_eval.json")
    eval_serializable = {
        'mean_reward': float(eval_result['mean_reward']),
        'std_reward': float(eval_result['std_reward']),
        'mean_length': float(eval_result['mean_length']),
        'std_length': float(eval_result['std_length']),
        'all_rewards': [float(r) for r in eval_result['all_rewards']],
        'training_time': training_time,
        'target_reward': float(target_reward['mean_reward']) if 'target_reward' in dir() else None,
    }
    with open(eval_path, 'w') as f:
        json.dump(eval_serializable, f, indent=2)
    
    if verbose:
        print(f"[RICE Refine] Results saved to {output_dir}")
        print(f"  Policy: {policy_path}")
        print(f"  History: {history_path}")
        print(f"  Evaluation: {eval_path}")
    
    # --- Return Results ---
    return {
        'refined_policy': refined_policy,
        'history': history,
        'eval_rewards': eval_result,
        'target_reward': target_reward,
        'training_time': training_time,
        'config': config,
    }


# ==============================================================================
# Baseline Comparison: PPO Fine-tuning (no RICE)
# ==============================================================================

def run_ppo_finetune(
    env_name: str,
    model_dir: str,
    output_dir: str,
    total_steps: int = 1_000_000,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run standard PPO fine-tuning (without RICE) as a baseline.
    Continues training the pre-trained agent with standard PPO.
    
    Args:
        env_name: Environment name
        model_dir: Directory with pre-trained agent
        output_dir: Directory to save results
        total_steps: Total fine-tuning steps
        seed: Random seed
        device: Device
        verbose: Print progress
        
    Returns:
        Results dictionary
    """
    from stable_baselines3 import PPO
    
    set_seed(seed)
    
    # Load pre-trained model
    model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
    ppo_model = PPO.load(model_path, device=device)
    
    # Create environment
    env = make_env(env_name, seed=seed)
    
    # Set up the model's environment
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    ppo_model.set_env(vec_env)
    
    if verbose:
        print(f"[PPO Fine-tune] Starting fine-tuning for {total_steps} steps...")
    
    start_time = time.time()
    ppo_model.learn(total_timesteps=total_steps, progress_bar=verbose)
    training_time = time.time() - start_time
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    finetune_path = os.path.join(output_dir, f"{env_name}_ppo_finetuned.zip")
    ppo_model.save(finetune_path)
    
    # Evaluate
    def policy_fn(state):
        action, _ = ppo_model.predict(state, deterministic=True)
        return action
    
    eval_result = evaluate_policy(env, policy_fn, num_episodes=10, max_steps=1000, deterministic=True)
    
    if verbose:
        print(f"  Fine-tuned mean reward: {eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
    
    return {
        'model': ppo_model,
        'eval_rewards': eval_result,
        'training_time': training_time,
    }


# ==============================================================================
# CLI Interface
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RICE Refinement for MuJoCo Environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard refinement on Hopper
  python experiments/mujoco/refine.py --env Hopper-v4
  
  # Refinement with custom parameters
  python experiments/mujoco/refine.py --env Walker2d-v4 --p_mixed 0.25 --lambda_rnd 0.01 --total_steps 500000
  
  # Sparse reward variant
  python experiments/mujoco/refine.py --env Hopper-v4 --sparse --sparse_threshold 1.5
  
  # PPO fine-tuning baseline (no RICE)
  python experiments/mujoco/refine.py --env Hopper-v4 --baseline ppo_finetune
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--env", type=str, required=True,
        choices=["Hopper-v4", "Walker2d-v4", "Reacher-v4", "HalfCheetah-v4",
                 "Hopper-v3", "Walker2d-v3", "Reacher-v3", "HalfCheetah-v3"],
        help="MuJoCo environment name"
    )
    
    # Paths
    parser.add_argument(
        "--model_dir", type=str, default="./trained_agents",
        help="Directory containing pre-trained target PPO agent"
    )
    parser.add_argument(
        "--mask_dir", type=str, default="./trained_masks",
        help="Directory containing trained mask network"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./refined_agents",
        help="Directory to save refined agent and results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML configuration file"
    )
    
    # Hyperparameter overrides
    parser.add_argument(
        "--total_steps", type=int, default=None,
        help="Override total refinement steps"
    )
    parser.add_argument(
        "--p_mixed", type=float, default=None,
        help="Override mixed initial distribution probability"
    )
    parser.add_argument(
        "--lambda_rnd", type=float, default=None,
        help="Override RND exploration bonus coefficient"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )
    
    # Variants
    parser.add_argument(
        "--sparse", action="store_true",
        help="Use sparse reward variant"
    )
    parser.add_argument(
        "--sparse_threshold", type=float, default=1.0,
        help="Threshold for sparse reward (default: 1.0)"
    )
    
    # Baseline mode
    parser.add_argument(
        "--baseline", type=str, default=None,
        choices=["ppo_finetune"],
        help="Run a baseline instead of RICE refinement"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress information"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.quiet:
        args.verbose = False
    
    if args.baseline == "ppo_finetune":
        # Run PPO fine-tuning baseline
        results = run_ppo_finetune(
            env_name=args.env,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            total_steps=args.total_steps or 1_000_000,
            seed=args.seed,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=args.verbose,
        )
        print(f"\nPPO Fine-tuning Results:")
        print(f"  Mean reward: {results['eval_rewards']['mean_reward']:.2f} ± {results['eval_rewards']['std_reward']:.2f}")
        print(f"  Training time: {results['training_time']:.1f}s")
    else:
        # Run RICE refinement
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
            sparse=args.sparse,
            sparse_threshold=args.sparse_threshold,
            verbose=args.verbose,
        )
        
        print(f"\n{'='*60}")
        print(f"RICE Refinement Results for {args.env}")
        print(f"{'='*60}")
        print(f"  Target policy reward:    {results['target_reward']['mean_reward']:.2f} ± {results['target_reward']['std_reward']:.2f}")
        print(f"  Refined policy reward:   {results['eval_rewards']['mean_reward']:.2f} ± {results['eval_rewards']['std_reward']:.2f}")
        print(f"  Improvement:             {results['eval_rewards']['mean_reward'] - results['target_reward']['mean_reward']:.2f}")
        print(f"  Training time:           {results['training_time']:.1f}s ({results['training_time']/60:.1f} min)")
        print(f"  Results saved to:        {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()