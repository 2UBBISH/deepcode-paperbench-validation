#!/usr/bin/env python3
"""
Evaluation script for CAGE Challenge 2 experiments.

Evaluates target, RICE-refined, and baseline policies on the CAGE2
cybersecurity environment. Computes performance metrics (mean reward,
success rate, malware removal rate), evaluates mask network fidelity,
and saves structured JSON results.

Supports multiple evaluation modes:
    - all: Full evaluation (fidelity + comparison + sparse)
    - fidelity: Only mask network fidelity
    - sparse: Only sparse reward evaluation
    - compare: Only policy comparison

Usage:
    python experiments/cage2/eval.py --env CAGE2-v0 \
        --model-dir ./trained_agents/cage2 \
        --refine-dir ./refined_agents/cage2 \
        --baseline-dir ./baseline_agents/cage2 \
        --mask-dir ./mask_networks/cage2 \
        --output-dir ./results/cage2 \
        --mode all
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
import yaml

# Optional Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# RICE core modules
from rice.mask_net import MaskNetwork, compute_fidelity_from_env
from rice.refine import RICERefine
from rice.utils import evaluate_policy, set_seed, to_numpy
from rice.env_wrappers import make_state_saveable, StateSaveWrapper

# CAGE2 environment
from experiments.cage2.env import (
    make_env as make_cage2_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
    Cage2StateWrapper,
    SparseRewardWrapper,
)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override values."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    env_name: str,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge default refine/mask configs with environment-specific overrides."""
    config = {}

    # Load default mask config
    default_mask_path = Path(__file__).parent.parent.parent / "configs" / "default_mask.yaml"
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load default refine config
    default_refine_path = Path(__file__).parent.parent.parent / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            refine_config = yaml.safe_load(f) or {}
            _deep_update(config, refine_config)

    # Load environment-specific config
    env_config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "env_specific"
        / "cage2.yaml"
    )
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f) or {}
            _deep_update(config, env_config)

    # Load custom config if provided
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f) or {}
            _deep_update(config, custom_config)

    return config


def make_env(
    env_name: str = "CAGE2-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False,
    use_sparse_reward: bool = False,
) -> gym.Env:
    """Create a CAGE2 environment with state save/restore capability."""
    env = make_cage2_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_real_env=use_real_env,
    )
    if use_sparse_reward:
        env = SparseRewardWrapper(env)
    env = make_state_saveable(env)
    return env


def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """Load a pre-trained target PPO agent.

    Supports Stable-Baselines3 .zip files and raw PyTorch .pt checkpoints.

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

    if model_path.exists() and HAS_SB3:
        model = PPO.load(str(model_path), device=device)
        return model, vec_normalize
    elif pt_path.exists():
        model = torch.load(pt_path, map_location=device)
        return model, vec_normalize
    else:
        raise FileNotFoundError(
            f"No target policy found at {model_path} or {pt_path}"
        )


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a loaded model into a deterministic policy function.

    Args:
        model: SB3 PPO model or raw PyTorch model with get_action method.
        vec_normalize: Optional VecNormalize for observation normalization.
        device: Torch device.

    Returns:
        Function: state -> action (deterministic).
    """
    if HAS_SB3 and hasattr(model, "predict"):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            action, _ = model.predict(state, deterministic=True)
            return action
        return policy_fn
    elif hasattr(model, "get_action"):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                if state_t.ndim == 1:
                    state_t = state_t.unsqueeze(0)
                action, _, _, _ = model.get_action(state_t, deterministic=True)
                return to_numpy(action).squeeze(0)
        return policy_fn
    else:
        raise ValueError("Model must have 'predict' or 'get_action' method")


def load_refined_policy(
    refine_dir: str,
    env_name: str,
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load a RICE-refined policy from a PyTorch checkpoint.

    Reconstructs the policy network from checkpoint metadata.
    """
    refine_path = Path(refine_dir) / f"{env_name}_refined_policy.pt"
    if not refine_path.exists():
        print(f"Warning: Refined policy not found at {refine_path}")
        return None

    checkpoint = torch.load(refine_path, map_location=device)

    # Extract metadata
    hidden_sizes = checkpoint.get("hidden_sizes", [128, 128])
    activation_name = checkpoint.get("activation", "tanh")
    discrete_action = checkpoint.get("discrete_action", True)
    num_discrete_actions = checkpoint.get("num_discrete_actions", action_dim)
    policy_std = checkpoint.get("policy_std", 0.0)

    # Build policy network
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(
        activation_name, nn.Tanh
    )

    layers = []
    prev_dim = state_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation_fn())
        prev_dim = h

    if discrete_action:
        layers.append(nn.Linear(prev_dim, num_discrete_actions))
        policy_net = nn.Sequential(*layers)
    else:
        # Continuous: mean + log_std
        mean_net = nn.Sequential(*layers, nn.Linear(prev_dim, action_dim))
        log_std = nn.Parameter(torch.ones(action_dim) * np.log(policy_std))
        policy_net = {"mean_net": mean_net, "log_std": log_std}

    # Load state dict
    if "policy_state_dict" in checkpoint:
        if isinstance(policy_net, dict):
            policy_net["mean_net"].load_state_dict(
                checkpoint["policy_state_dict"].get("mean_net", {})
            )
            if "log_std" in checkpoint["policy_state_dict"]:
                policy_net["log_std"] = nn.Parameter(
                    checkpoint["policy_state_dict"]["log_std"]
                )
        else:
            policy_net.load_state_dict(checkpoint["policy_state_dict"])

    policy_net = policy_net.to(device)
    policy_net.eval()

    def policy_fn(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            if isinstance(policy_net, dict):
                mean = policy_net["mean_net"](state_t)
                action = mean  # deterministic
            else:
                logits = policy_net(state_t)
                action = torch.argmax(logits, dim=-1)
            return to_numpy(action).squeeze(0)

    return policy_fn


def load_baseline_policy(
    baseline_dir: str,
    env_name: str,
    baseline_name: str,
    device: str = "cpu",
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load a baseline policy from disk.

    Supports SB3 .zip and PyTorch .pt formats.
    """
    # Try SB3 format
    sb3_path = Path(baseline_dir) / baseline_name / f"{env_name}_{baseline_name}.zip"
    if sb3_path.exists() and HAS_SB3:
        model = PPO.load(str(sb3_path), device=device)
        def policy_fn(state: np.ndarray) -> np.ndarray:
            action, _ = model.predict(state, deterministic=True)
            return action
        return policy_fn

    # Try PyTorch format
    pt_path = Path(baseline_dir) / baseline_name / f"{env_name}_{baseline_name}.pt"
    if pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        # Try to reconstruct policy from checkpoint
        if "policy_state_dict" in checkpoint:
            state_dim = checkpoint.get("state_dim", 19)
            action_dim = checkpoint.get("action_dim", 12)
            hidden_sizes = checkpoint.get("hidden_sizes", [128, 128])
            activation_name = checkpoint.get("activation", "tanh")
            discrete_action = checkpoint.get("discrete_action", True)
            num_discrete_actions = checkpoint.get("num_discrete_actions", action_dim)

            activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(
                activation_name, nn.Tanh
            )

            layers = []
            prev_dim = state_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(activation_fn())
                prev_dim = h
            layers.append(nn.Linear(prev_dim, num_discrete_actions))
            policy_net = nn.Sequential(*layers)
            policy_net.load_state_dict(checkpoint["policy_state_dict"])
            policy_net = policy_net.to(device)
            policy_net.eval()

            def policy_fn(state: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                    if state_t.ndim == 1:
                        state_t = state_t.unsqueeze(0)
                    logits = policy_net(state_t)
                    action = torch.argmax(logits, dim=-1)
                    return to_numpy(action).squeeze(0)
            return policy_fn

    print(f"Warning: Baseline '{baseline_name}' not found at {sb3_path} or {pt_path}")
    return None


def evaluate_all(
    env_name: str = "CAGE2-v0",
    model_dir: str = "./trained_agents/cage2",
    refine_dir: str = "./refined_agents/cage2",
    baseline_dir: str = "./baseline_agents/cage2",
    mask_dir: str = "./mask_networks/cage2",
    output_dir: str = "./results/cage2",
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target, refined, and baseline policies.

    Computes comparison metrics and saves results as JSON.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(env_name)
    state_dim = config.get("state_dim", 19)
    action_dim = config.get("action_dim", 12)
    if max_episode_steps is None:
        max_episode_steps = config.get("max_episode_steps", 100)

    results = {
        "env_name": env_name,
        "num_episodes": num_episodes,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": {},
    }

    # --- Target Policy ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Target Policy on {env_name}")
        print(f"{'='*60}")

    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_policy_fn = make_target_policy_fn(model, vec_normalize, device)
        env = make_env(env_name, seed, max_episode_steps, use_real_env)
        target_stats = evaluate_policy(
            env, target_policy_fn, num_episodes=num_episodes,
            max_steps=max_episode_steps, deterministic=True, verbose=verbose
        )
        results["policies"]["target"] = {
            "mean_reward": float(target_stats["mean_reward"]),
            "std_reward": float(target_stats["std_reward"]),
            "mean_length": float(target_stats["mean_length"]),
            "all_rewards": [float(r) for r in target_stats["all_rewards"]],
        }
        env.close()
        if verbose:
            print(f"Target: mean_reward={target_stats['mean_reward']:.4f} "
                  f"± {target_stats['std_reward']:.4f}")
    except Exception as e:
        if verbose:
            print(f"Error evaluating target policy: {e}")
        results["policies"]["target"] = {"error": str(e)}

    # --- RICE Refined Policy ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating RICE Refined Policy on {env_name}")
        print(f"{'='*60}")

    refined_policy_fn = load_refined_policy(refine_dir, env_name, state_dim, action_dim, device)
    if refined_policy_fn is not None:
        try:
            env = make_env(env_name, seed, max_episode_steps, use_real_env)
            refined_stats = evaluate_policy(
                env, refined_policy_fn, num_episodes=num_episodes,
                max_steps=max_episode_steps, deterministic=True, verbose=verbose
            )
            results["policies"]["rice_refined"] = {
                "mean_reward": float(refined_stats["mean_reward"]),
                "std_reward": float(refined_stats["std_reward"]),
                "mean_length": float(refined_stats["mean_length"]),
                "all_rewards": [float(r) for r in refined_stats["all_rewards"]],
            }
            env.close()
            if verbose:
                print(f"RICE Refined: mean_reward={refined_stats['mean_reward']:.4f} "
                      f"± {refined_stats['std_reward']:.4f}")
        except Exception as e:
            if verbose:
                print(f"Error evaluating refined policy: {e}")
            results["policies"]["rice_refined"] = {"error": str(e)}
    else:
        results["policies"]["rice_refined"] = {"error": "Policy not found"}

    # --- Baseline Policies ---
    baseline_names = ["statemask", "jsrl", "sil", "random_explanation", "ppo_finetune"]
    for baseline_name in baseline_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {baseline_name} Baseline on {env_name}")
            print(f"{'='*60}")

        baseline_fn = load_baseline_policy(baseline_dir, env_name, baseline_name, device)
        if baseline_fn is not None:
            try:
                env = make_env(env_name, seed, max_episode_steps, use_real_env)
                baseline_stats = evaluate_policy(
                    env, baseline_fn, num_episodes=num_episodes,
                    max_steps=max_episode_steps, deterministic=True, verbose=verbose
                )
                results["policies"][baseline_name] = {
                    "mean_reward": float(baseline_stats["mean_reward"]),
                    "std_reward": float(baseline_stats["std_reward"]),
                    "mean_length": float(baseline_stats["mean_length"]),
                    "all_rewards": [float(r) for r in baseline_stats["all_rewards"]],
                }
                env.close()
                if verbose:
                    print(f"{baseline_name}: mean_reward={baseline_stats['mean_reward']:.4f} "
                          f"± {baseline_stats['std_reward']:.4f}")
            except Exception as e:
                if verbose:
                    print(f"Error evaluating {baseline_name}: {e}")
                results["policies"][baseline_name] = {"error": str(e)}
        else:
            results["policies"][baseline_name] = {"error": "Policy not found"}

    # --- Compute Comparisons ---
    comparison = compute_comparison(results)
    results["comparison"] = comparison

    # --- Save Results ---
    results_path = Path(output_dir) / f"{env_name}_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to {results_path}")

    # --- Print Summary ---
    print_summary_table(env_name, results, comparison)

    return results


def compute_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pairwise improvements over target policy."""
    comparison = {}
    target_reward = None

    target_data = results.get("policies", {}).get("target", {})
    if "mean_reward" in target_data:
        target_reward = target_data["mean_reward"]

    if target_reward is not None:
        for policy_name, policy_data in results.get("policies", {}).items():
            if policy_name == "target":
                continue
            if "mean_reward" in policy_data:
                mean_r = policy_data["mean_reward"]
                abs_improvement = mean_r - target_reward
                pct_improvement = (
                    (abs_improvement / abs(target_reward)) * 100
                    if target_reward != 0
                    else 0.0
                )
                comparison[policy_name] = {
                    "mean_reward": mean_r,
                    "absolute_improvement": abs_improvement,
                    "percent_improvement": pct_improvement,
                }

    return comparison


def print_summary_table(
    env_name: str,
    results: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    """Print a formatted summary table of evaluation results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY: {env_name}")
    print(f"{'='*80}")
    print(f"{'Policy':<25} {'Mean Reward':>14} {'Std Reward':>12} {'Improvement':>14}")
    print(f"{'-'*65}")

    for policy_name, policy_data in results.get("policies", {}).items():
        if "mean_reward" in policy_data:
            mean_r = policy_data["mean_reward"]
            std_r = policy_data.get("std_reward", 0.0)
            if policy_name in comparison:
                imp = comparison[policy_name]["absolute_improvement"]
                imp_str = f"{imp:+.4f}"
            else:
                imp_str = "---"
            print(f"{policy_name:<25} {mean_r:>14.4f} {std_r:>12.4f} {imp_str:>14}")

    print(f"{'='*80}\n")


def evaluate_fidelity(
    env_name: str = "CAGE2-v0",
    mask_dir: str = "./mask_networks/cage2",
    model_dir: str = "./trained_agents/cage2",
    output_dir: str = "./results/cage2",
    num_episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    use_real_env: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate fidelity (Pearson correlation) of the trained mask network.

    Fidelity measures how well the mask network's importance scores
    correlate with actual Q-value differences.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(env_name)
    state_dim = config.get("state_dim", 19)
    max_episode_steps = config.get("max_episode_steps", 100)

    # Load mask network
    mask_path = Path(mask_dir) / f"{env_name}_mask_network.pt"
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask network not found at {mask_path}")

    mask_network = MaskNetwork(state_dim=state_dim, hidden_sizes=(128, 128))
    checkpoint = torch.load(mask_path, map_location=device)
    if "mask_network_state_dict" in checkpoint:
        mask_network.load_state_dict(checkpoint["mask_network_state_dict"])
    elif "model_state_dict" in checkpoint:
        mask_network.load_state_dict(checkpoint["model_state_dict"])
    else:
        mask_network.load_state_dict(checkpoint)
    mask_network = mask_network.to(device)
    mask_network.eval()

    # Load target policy
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

    # Create environment
    env = make_env(env_name, seed, max_episode_steps, use_real_env)

    # Compute fidelity
    if verbose:
        print(f"\nComputing fidelity for {env_name}...")

    fidelity_result = compute_fidelity_from_env(
        mask_network=mask_network,
        env=env,
        target_policy=target_policy_fn,
        num_episodes=num_episodes,
        device=device,
    )

    env.close()

    # Save results
    fidelity_data = {
        "env_name": env_name,
        "fidelity": float(fidelity_result) if isinstance(fidelity_result, (int, float)) else fidelity_result,
        "num_episodes": num_episodes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    fidelity_path = Path(output_dir) / f"{env_name}_fidelity.json"
    with open(fidelity_path, "w") as f:
        json.dump(fidelity_data, f, indent=2, default=str)

    if verbose:
        print(f"Fidelity: {fidelity_result}")
        print(f"Saved to {fidelity_path}")

    return fidelity_data


def evaluate_sparse(
    env_name: str = "CAGE2-v0",
    model_dir: str = "./trained_agents/cage2",
    refine_dir: str = "./refined_agents/cage2",
    output_dir: str = "./results/cage2",
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    use_real_env: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target and RICE policies on sparse reward variant."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    config = load_config(env_name)
    state_dim = config.get("state_dim", 19)
    action_dim = config.get("action_dim", 12)
    max_episode_steps = config.get("max_episode_steps", 100)

    results = {
        "env_name": f"{env_name}_sparse",
        "num_episodes": num_episodes,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": {},
    }

    # Target on sparse
    if verbose:
        print(f"\nEvaluating Target on sparse {env_name}...")
    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_policy_fn = make_target_policy_fn(model, vec_normalize, device)
        env = make_env(env_name, seed, max_episode_steps, use_real_env, use_sparse_reward=True)
        target_stats = evaluate_policy(
            env, target_policy_fn, num_episodes=num_episodes,
            max_steps=max_episode_steps, deterministic=True, verbose=verbose
        )
        results["policies"]["target_sparse"] = {
            "mean_reward": float(target_stats["mean_reward"]),
            "std_reward": float(target_stats["std_reward"]),
            "mean_length": float(target_stats["mean_length"]),
            "all_rewards": [float(r) for r in target_stats["all_rewards"]],
        }
        env.close()
    except Exception as e:
        results["policies"]["target_sparse"] = {"error": str(e)}

    # RICE on sparse
    refined_policy_fn = load_refined_policy(refine_dir, env_name, state_dim, action_dim, device)
    if refined_policy_fn is not None:
        if verbose:
            print(f"Evaluating RICE Refined on sparse {env_name}...")
        try:
            env = make_env(env_name, seed, max_episode_steps, use_real_env, use_sparse_reward=True)
            refined_stats = evaluate_policy(
                env, refined_policy_fn, num_episodes=num_episodes,
                max_steps=max_episode_steps, deterministic=True, verbose=verbose
            )
            results["policies"]["rice_refined_sparse"] = {
                "mean_reward": float(refined_stats["mean_reward"]),
                "std_reward": float(refined_stats["std_reward"]),
                "mean_length": float(refined_stats["mean_length"]),
                "all_rewards": [float(r) for r in refined_stats["all_rewards"]],
            }
            env.close()
        except Exception as e:
            results["policies"]["rice_refined_sparse"] = {"error": str(e)}

    # Save
    sparse_path = Path(output_dir) / f"{env_name}_sparse_evaluation.json"
    with open(sparse_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"Sparse evaluation saved to {sparse_path}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RICE and baselines on CAGE Challenge 2"
    )
    parser.add_argument(
        "--env", type=str, default="CAGE2-v0",
        help="Environment name"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents/cage2",
        help="Directory containing pre-trained target agent"
    )
    parser.add_argument(
        "--refine-dir", type=str, default="./refined_agents/cage2",
        help="Directory containing RICE-refined policy"
    )
    parser.add_argument(
        "--baseline-dir", type=str, default="./baseline_agents/cage2",
        help="Directory containing baseline policies"
    )
    parser.add_argument(
        "--mask-dir", type=str, default="./mask_networks/cage2",
        help="Directory containing trained mask network"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/cage2",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "fidelity", "sparse", "compare"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--num-episodes", type=int, default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=None,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--use-real-env", action="store_true",
        help="Use real CybORG environment instead of simulated"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose output"
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

    if args.mode == "all":
        # Fidelity
        try:
            evaluate_fidelity(
                env_name=args.env,
                mask_dir=args.mask_dir,
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                num_episodes=min(args.num_episodes, 10),
                seed=args.seed,
                device=args.device,
                use_real_env=args.use_real_env,
                verbose=verbose,
            )
        except Exception as e:
            print(f"Fidelity evaluation failed: {e}")

        # Full comparison
        evaluate_all(
            env_name=args.env,
            model_dir=args.model_dir,
            refine_dir=args.refine_dir,
            baseline_dir=args.baseline_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            use_real_env=args.use_real_env,
            verbose=verbose,
        )

        # Sparse
        try:
            evaluate_sparse(
                env_name=args.env,
                model_dir=args.model_dir,
                refine_dir=args.refine_dir,
                output_dir=args.output_dir,
                num_episodes=args.num_episodes,
                seed=args.seed,
                device=args.device,
                use_real_env=args.use_real_env,
                verbose=verbose,
            )
        except Exception as e:
            print(f"Sparse evaluation failed: {e}")

    elif args.mode == "fidelity":
        evaluate_fidelity(
            env_name=args.env,
            mask_dir=args.mask_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_episodes=min(args.num_episodes, 10),
            seed=args.seed,
            device=args.device,
            use_real_env=args.use_real_env,
            verbose=verbose,
        )

    elif args.mode == "sparse":
        evaluate_sparse(
            env_name=args.env,
            model_dir=args.model_dir,
            refine_dir=args.refine_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            use_real_env=args.use_real_env,
            verbose=verbose,
        )

    elif args.mode == "compare":
        evaluate_all(
            env_name=args.env,
            model_dir=args.model_dir,
            refine_dir=args.refine_dir,
            baseline_dir=args.baseline_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            use_real_env=args.use_real_env,
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())