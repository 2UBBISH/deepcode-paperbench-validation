#!/usr/bin/env python3
"""
RICE Refinement Script for Selfish Mining Environment.

This script orchestrates the full RICE refinement pipeline for the selfish mining
domain: loads a pre-trained target PPO agent and a trained mask network, collects
critical states, then refines the policy via PPO with an RND exploration bonus and
a mixed initial state distribution. Also provides a baseline PPO fine-tuning mode.

Usage:
    python experiments/selfish_mining/refine.py --env SelfishMining-v0 \
        --model-dir ./trained_agents/selfish_mining \
        --mask-dir ./trained_masks/selfish_mining \
        --output-dir ./refined_agents/selfish_mining \
        --total-steps 1000000 --p-mixed 0.25 --lambda-rnd 0.001

Reference:
    RICE: Refining via Critical State Explanation (Table 3: p=0.25, λ=0.001, α=0.0001)
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

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
from experiments.selfish_mining.env import (
    SelfishMiningEnv,
    SelfishMiningStateWrapper,
    get_action_dim,
    get_state_dim,
    is_discrete_action,
    make_env as make_sm_env,
)

# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(
    env_name: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load and merge default refine config with environment-specific overrides.

    Args:
        env_name: Environment name (e.g., "SelfishMining-v0").
        config_path: Optional path to a custom YAML config file.

    Returns:
        Merged configuration dictionary.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # Load default refine config
    default_refine_path = project_root / "configs" / "default_refine.yaml"
    config = {}
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load default mask config (for shared PPO params)
    default_mask_path = project_root / "configs" / "default_mask.yaml"
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            mask_config = yaml.safe_load(f) or {}
        # Merge mask config into config (don't overwrite existing keys)
        for key, value in mask_config.items():
            if key not in config:
                config[key] = value

    # Load environment-specific config
    env_config_path = project_root / "configs" / "env_specific" / "selfish_mining.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
        # Deep merge
        config = _deep_merge(config, env_config)

    # Load custom config if provided
    if config_path is not None and os.path.exists(config_path):
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


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(
    env_name: str = "SelfishMining-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    alpha: float = 0.35,
    gamma_sm: float = 0.5,
) -> gym.Env:
    """
    Create a selfish mining environment wrapped with state save/restore capability.

    Args:
        env_name: Environment name.
        seed: Random seed.
        max_episode_steps: Maximum steps per episode (overrides env default).
        alpha: Miner's hash rate fraction.
        gamma_sm: Honest network adoption ratio.

    Returns:
        Wrapped gym environment.
    """
    env = make_sm_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        alpha=alpha,
        gamma=gamma_sm,
    )
    # Ensure state save/restore is available
    env = make_state_saveable(env)
    return env


# ---------------------------------------------------------------------------
# Target policy loading
# ---------------------------------------------------------------------------

def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """
    Load a pre-trained target PPO agent from disk.

    Supports Stable-Baselines3 .zip models and raw PyTorch .pt checkpoints.

    Args:
        env_name: Environment name.
        model_dir: Directory containing the saved model.
        device: Torch device string.

    Returns:
        Tuple of (model, vec_normalize) where vec_normalize may be None.
    """
    model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
    pt_path = os.path.join(model_dir, f"{env_name}_target_policy.pt")
    vecnorm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")

    vec_normalize = None
    if os.path.exists(vecnorm_path):
        try:
            with open(vecnorm_path, "rb") as f:
                vec_normalize = pickle.load(f)
        except Exception:
            pass

    # Try Stable-Baselines3 first
    if os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            model = PPO.load(model_path, device=device)
            return model, vec_normalize
        except Exception as e:
            print(f"  [WARN] Failed to load SB3 model: {e}")

    # Try raw PyTorch checkpoint
    if os.path.exists(pt_path):
        try:
            model = torch.load(pt_path, map_location=device)
            return model, vec_normalize
        except Exception as e:
            print(f"  [WARN] Failed to load PyTorch model: {e}")

    raise FileNotFoundError(
        f"No model found at {model_path} or {pt_path}. "
        f"Run train_target.py first."
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
        vec_normalize: Optional VecNormalize statistics.
        device: Torch device.

    Returns:
        Policy function.
    """
    # Check if it's a Stable-Baselines3 model
    try:
        from stable_baselines3 import PPO
        if isinstance(model, PPO):
            def sb3_policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
                # Apply VecNormalize if available
                if vec_normalize is not None:
                    state = vec_normalize.normalize_obs(state)
                state_tensor = to_tensor(state, device).unsqueeze(0)
                with torch.no_grad():
                    # Extract features
                    features = model.policy.mlp_extractor.forward_actor(
                        model.policy.features_extractor(state_tensor)
                    )
                    # Discrete action
                    action_logits = model.policy.action_net(features)
                    action_dist = torch.distributions.Categorical(logits=action_logits)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)
                    entropy = action_dist.entropy()
                    # Value
                    value = model.policy.value_net(
                        model.policy.mlp_extractor.forward_critic(
                            model.policy.features_extractor(state_tensor)
                        )
                    )
                return (
                    action.cpu().numpy().flatten(),
                    log_prob.cpu().numpy().flatten()[0],
                    value.cpu().numpy().flatten()[0],
                    entropy.cpu().numpy().flatten()[0],
                )
            return sb3_policy_fn
    except ImportError:
        pass

    # Assume raw PyTorch model with get_action method
    if hasattr(model, "get_action"):
        def pt_policy_fn(state: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
            state_tensor = to_tensor(state, device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value, entropy = model.get_action(state_tensor)
            return (
                to_numpy(action).flatten(),
                float(to_numpy(log_prob).flatten()[0]),
                float(to_numpy(value).flatten()[0]),
                float(to_numpy(entropy).flatten()[0]),
            )
        return pt_policy_fn

    raise ValueError("Unsupported model type. Provide SB3 PPO or a model with get_action().")


# ---------------------------------------------------------------------------
# Mask network loading
# ---------------------------------------------------------------------------

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
        mask_dir: Directory containing the saved mask network.
        state_dim: State dimension.
        device: Torch device.

    Returns:
        Loaded MaskNetwork.
    """
    mask_path = os.path.join(mask_dir, f"{env_name}_mask_network.pt")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask network not found at {mask_path}. Run train_mask.py first."
        )

    checkpoint = torch.load(mask_path, map_location=device)
    hidden_sizes = checkpoint.get("hidden_sizes", [128, 128])
    activation = checkpoint.get("activation", "tanh")

    mask_net = MaskNetwork(
        state_dim=state_dim,
        hidden_sizes=tuple(hidden_sizes),
        activation=activation,
    ).to(device)
    mask_net.load_state_dict(checkpoint["model_state_dict"])
    mask_net.eval()

    return mask_net


# ---------------------------------------------------------------------------
# Main refinement routine
# ---------------------------------------------------------------------------

def run_refine(
    env_name: str = "SelfishMining-v0",
    model_dir: str = "./trained_agents/selfish_mining",
    mask_dir: str = "./trained_masks/selfish_mining",
    output_dir: str = "./refined_agents/selfish_mining",
    config_path: Optional[str] = None,
    total_steps: Optional[int] = None,
    p_mixed: Optional[float] = None,
    lambda_rnd: Optional[float] = None,
    seed: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full RICE refinement pipeline for selfish mining.

    Args:
        env_name: Environment name.
        model_dir: Directory with pre-trained target agent.
        mask_dir: Directory with trained mask network.
        output_dir: Directory to save refined agent and results.
        config_path: Optional custom config YAML path.
        total_steps: Override total refinement steps.
        p_mixed: Override mixed initial distribution probability.
        lambda_rnd: Override RND exploration bonus coefficient.
        seed: Random seed.
        device: Torch device ("cuda" or "cpu").
        verbose: Print progress information.

    Returns:
        Dictionary with refined policy, training history, evaluation rewards,
        and configuration.
    """
    # --- Load configuration ---
    config = load_config(env_name, config_path)

    # Determine device
    if device is None:
        device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # Override with CLI arguments
    if total_steps is not None:
        if "refine_training" not in config:
            config["refine_training"] = {}
        config["refine_training"]["total_steps"] = total_steps
    if p_mixed is not None:
        if "mixed_init" not in config:
            config["mixed_init"] = {}
        config["mixed_init"]["p_mixed"] = p_mixed
    if lambda_rnd is not None:
        if "rnd" not in config:
            config["rnd"] = {}
        config["rnd"]["lambda_coef"] = lambda_rnd

    # Extract key parameters
    state_dim = config.get("state_dim", 52)
    action_dim = config.get("action_dim", 2)
    max_episode_steps = config.get("max_episode_steps", 100)
    discrete_action = True  # Selfish mining is discrete

    refine_training = config.get("refine_training", {})
    total_steps_val = refine_training.get("total_steps", 1_000_000)
    steps_per_iteration = refine_training.get("steps_per_iteration", 2048)
    eval_interval = refine_training.get("eval_interval", 10)
    eval_episodes = refine_training.get("eval_episodes", 10)
    save_interval = refine_training.get("save_interval", 50)

    critical_state_cfg = config.get("critical_state", {})
    num_critical_episodes = critical_state_cfg.get("num_episodes", 100)
    top_k_per_episode = critical_state_cfg.get("top_k_per_episode", 1)

    mixed_init_cfg = config.get("mixed_init", {})
    p_mixed_val = mixed_init_cfg.get("p_mixed", 0.25)

    rnd_cfg = config.get("rnd", {})
    lambda_rnd_val = rnd_cfg.get("lambda_coef", 0.001)
    rnd_hidden_sizes = tuple(rnd_cfg.get("hidden_sizes", [64, 64]))
    rnd_embedding_dim = rnd_cfg.get("embedding_dim", 64)
    rnd_lr = rnd_cfg.get("learning_rate", 1e-4)
    normalize_obs = rnd_cfg.get("normalize_obs", True)
    normalize_bonus = rnd_cfg.get("normalize_bonus", True)

    refine_ppo_cfg = config.get("refine_ppo", config.get("ppo", {}))
    ppo_lr = refine_ppo_cfg.get("learning_rate", 3e-4)
    gamma = refine_ppo_cfg.get("gamma", 0.99)
    gae_lambda = refine_ppo_cfg.get("gae_lambda", 0.95)
    clip_epsilon = refine_ppo_cfg.get("clip_epsilon", 0.2)
    value_loss_coef = refine_ppo_cfg.get("value_loss_coef", 0.5)
    entropy_coef = refine_ppo_cfg.get("entropy_coef", 0.01)
    max_grad_norm = refine_ppo_cfg.get("max_grad_norm", 0.5)
    ppo_epochs = refine_ppo_cfg.get("ppo_epochs", 10)
    ppo_batch_size = refine_ppo_cfg.get("batch_size", 64)

    policy_cfg = config.get("policy", {})
    policy_hidden_sizes = policy_cfg.get("hidden_sizes", None)

    if verbose:
        print("=" * 70)
        print(f"RICE Refinement: {env_name}")
        print(f"  Device: {device}")
        print(f"  Total steps: {total_steps_val}")
        print(f"  p_mixed: {p_mixed_val}")
        print(f"  lambda_rnd: {lambda_rnd_val}")
        print(f"  Critical episodes: {num_critical_episodes}")
        print(f"  Top-k per episode: {top_k_per_episode}")
        print("=" * 70)

    # --- Set seed ---
    set_seed(seed)

    # --- Create environment ---
    env = make_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )

    # --- Load target policy ---
    if verbose:
        print("\n[1/4] Loading target policy...")
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

    # --- Load mask network ---
    if verbose:
        print("[2/4] Loading mask network...")
    mask_network = load_mask_network(env_name, mask_dir, state_dim, device)

    # --- Create output directory ---
    os.makedirs(output_dir, exist_ok=True)

    # --- Run refinement ---
    if verbose:
        print("[3/4] Running RICE refinement...")
        print(f"  Collecting critical states from {num_critical_episodes} episodes...")

    start_time = time.time()

    refined_policy, history = refine_policy(
        env=env,
        target_policy=target_policy_fn,
        mask_network=mask_network,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        num_discrete_actions=action_dim,
        device=device,
        p_mixed=p_mixed_val,
        lambda_rnd=lambda_rnd_val,
        total_steps=total_steps_val,
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
        normalize_obs=normalize_obs,
        normalize_bonus=normalize_bonus,
        policy_hidden_sizes=policy_hidden_sizes,
    )

    training_time = time.time() - start_time

    # --- Evaluate refined policy ---
    if verbose:
        print("\n[4/4] Evaluating refined policy...")

    def refined_policy_fn(state: np.ndarray) -> np.ndarray:
        """Deterministic policy wrapper for evaluation."""
        state_tensor = to_tensor(state, device).unsqueeze(0)
        with torch.no_grad():
            if discrete_action:
                logits = refined_policy(state_tensor)
                action = torch.argmax(logits, dim=-1)
            else:
                action_mean = refined_policy(state_tensor)
                action = action_mean
        return to_numpy(action).flatten()

    eval_results = evaluate_policy(
        env=env,
        policy_fn=refined_policy_fn,
        num_episodes=eval_episodes,
        max_steps=max_episode_steps,
        deterministic=True,
        verbose=verbose,
    )

    # --- Evaluate target policy for comparison ---
    def target_eval_fn(state: np.ndarray) -> np.ndarray:
        action, _, _, _ = target_policy_fn(state)
        return action

    target_eval_results = evaluate_policy(
        env=env,
        policy_fn=target_eval_fn,
        num_episodes=eval_episodes,
        max_steps=max_episode_steps,
        deterministic=True,
        verbose=False,
    )

    # --- Save results ---
    results = {
        "env_name": env_name,
        "config": config,
        "training_time": training_time,
        "total_steps": total_steps_val,
        "p_mixed": p_mixed_val,
        "lambda_rnd": lambda_rnd_val,
        "seed": seed,
        "target_mean_reward": target_eval_results["mean_reward"],
        "target_std_reward": target_eval_results["std_reward"],
        "refined_mean_reward": eval_results["mean_reward"],
        "refined_std_reward": eval_results["std_reward"],
        "improvement": eval_results["mean_reward"] - target_eval_results["mean_reward"],
        "improvement_pct": (
            (eval_results["mean_reward"] - target_eval_results["mean_reward"])
            / (abs(target_eval_results["mean_reward"]) + 1e-8)
            * 100
        ),
        "history": history,
    }

    results_path = os.path.join(output_dir, f"{env_name}_refine_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save refined policy
    policy_path = os.path.join(output_dir, f"{env_name}_refined_policy.pt")
    torch.save(
        {
            "policy_state_dict": refined_policy.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "discrete_action": discrete_action,
            "num_discrete_actions": action_dim,
            "config": {
                "p_mixed": p_mixed_val,
                "lambda_rnd": lambda_rnd_val,
                "total_steps": total_steps_val,
            },
        },
        policy_path,
    )

    if verbose:
        print("\n" + "=" * 70)
        print("RICE Refinement Complete!")
        print(f"  Target mean reward:  {target_eval_results['mean_reward']:.4f} ± {target_eval_results['std_reward']:.4f}")
        print(f"  Refined mean reward: {eval_results['mean_reward']:.4f} ± {eval_results['std_reward']:.4f}")
        print(f"  Improvement:         {results['improvement']:.4f} ({results['improvement_pct']:.1f}%)")
        print(f"  Training time:       {training_time:.1f}s")
        print(f"  Results saved to:    {output_dir}")
        print("=" * 70)

    return results


# ---------------------------------------------------------------------------
# Baseline: PPO fine-tuning (no RICE)
# ---------------------------------------------------------------------------

def run_ppo_finetune(
    env_name: str = "SelfishMining-v0",
    model_dir: str = "./trained_agents/selfish_mining",
    output_dir: str = "./finetuned_agents/selfish_mining",
    total_steps: int = 1_000_000,
    seed: int = 42,
    device: str = "cpu",
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
        device: Torch device.
        verbose: Print progress.

    Returns:
        Dictionary with evaluation results and training time.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.vec_env import VecNormalize
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        raise ImportError(
            "Stable-Baselines3 is required for PPO fine-tuning. "
            "Install with: pip install stable-baselines3"
        )

    if verbose:
        print("=" * 70)
        print(f"PPO Fine-tuning Baseline: {env_name}")
        print(f"  Total steps: {total_steps}")
        print("=" * 70)

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-trained model
    model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
    vecnorm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")

    # Create environment
    def _make_env():
        env = make_sm_env(env_name=env_name, seed=seed)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([_make_env])

    # Load VecNormalize if available
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True

    # Load model
    model = PPO.load(model_path, env=vec_env, device=device)

    # Setup evaluation callback
    eval_env = DummyVecEnv([_make_env])
    if os.path.exists(vecnorm_path):
        eval_env = VecNormalize.load(vecnorm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = True

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=output_dir,
        log_path=output_dir,
        eval_freq=max(10000, total_steps // 20),
        n_eval_episodes=10,
        deterministic=True,
    )

    # Fine-tune
    start_time = time.time()
    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=verbose)
    training_time = time.time() - start_time

    # Save
    model.save(os.path.join(output_dir, f"{env_name}_ppo_finetuned.zip"))

    # Evaluate
    final_rewards = []
    for _ in range(10):
        obs = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            ep_reward += reward[0] if isinstance(reward, np.ndarray) else reward
        final_rewards.append(ep_reward)

    results = {
        "env_name": env_name,
        "method": "ppo_finetune",
        "total_steps": total_steps,
        "training_time": training_time,
        "mean_reward": float(np.mean(final_rewards)),
        "std_reward": float(np.std(final_rewards)),
        "all_rewards": [float(r) for r in final_rewards],
    }

    results_path = os.path.join(output_dir, f"{env_name}_ppo_finetune_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nPPO Fine-tuning Complete!")
        print(f"  Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        print(f"  Training time: {training_time:.1f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RICE Refinement for Selfish Mining Environment"
    )
    parser.add_argument(
        "--env", type=str, default="SelfishMining-v0",
        help="Environment name (default: SelfishMining-v0)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents/selfish_mining",
        help="Directory with pre-trained target agent"
    )
    parser.add_argument(
        "--mask-dir", type=str, default="./trained_masks/selfish_mining",
        help="Directory with trained mask network"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./refined_agents/selfish_mining",
        help="Directory to save refined agent and results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML config file"
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
        help="Torch device (default: cuda if available)"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        choices=["ppo_finetune"],
        help="Run a baseline instead of RICE refinement"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet

    if args.baseline == "ppo_finetune":
        run_ppo_finetune(
            env_name=args.env,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            total_steps=args.total_steps or 1_000_000,
            seed=args.seed,
            device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=verbose,
        )
    else:
        run_refine(
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
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())