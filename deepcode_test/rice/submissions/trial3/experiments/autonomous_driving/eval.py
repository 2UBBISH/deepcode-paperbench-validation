#!/usr/bin/env python3
"""
Evaluation script for autonomous driving (MetaDrive Macro-v1) experiments.

Evaluates target, RICE-refined, and baseline policies on the MetaDrive environment,
computes comparison metrics (success rate, mean reward, improvement over target),
evaluates mask network fidelity, and saves results as JSON.

Supports:
- Full evaluation (all policies + fidelity)
- Fidelity-only evaluation
- Sparse reward evaluation
- Comparison-only mode

Matches the paper's evaluation protocol for autonomous driving (Table 5, Figure 14).
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

# Optional Stable-Baselines3 import
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# RICE core imports
from rice.mask_net import MaskNetwork, compute_fidelity_from_env
from rice.refine import RICERefine
from rice.utils import evaluate_policy, set_seed, to_numpy
from rice.env_wrappers import make_state_saveable, StateSaveWrapper

# Domain-specific imports
from experiments.autonomous_driving.env import (
    make_env as make_ad_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
    MetaDriveStateWrapper,
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
    env_name: str = "MetaDrive-Macro-v1",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge default refine config with environment-specific overrides."""
    config = {}

    # Load default refine config
    default_refine_path = Path(__file__).parent.parent.parent / "configs" / "default_refine.yaml"
    if default_refine_path.exists():
        with open(default_refine_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Load default mask config
    default_mask_path = Path(__file__).parent.parent.parent / "configs" / "default_mask.yaml"
    if default_mask_path.exists():
        with open(default_mask_path, "r") as f:
            mask_config = yaml.safe_load(f) or {}
            _deep_update(config, mask_config)

    # Load environment-specific config
    env_config_path = (
        Path(__file__).parent.parent.parent
        / "configs"
        / "env_specific"
        / "autonomous_driving.yaml"
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
    env_name: str = "MetaDrive-Macro-v1",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_sparse_reward: bool = False,
) -> gym.Env:
    """Create a MetaDrive environment with state save/restore capability."""
    return make_ad_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        use_sparse_reward=use_sparse_reward,
    )


def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """Load a pre-trained target PPO agent (SB3 or raw PyTorch)."""
    model = None
    vec_normalize = None

    # Try Stable-Baselines3 format
    if HAS_SB3:
        sb3_path = Path(model_dir) / f"{env_name}_ppo_final.zip"
        if sb3_path.exists():
            model = PPO.load(str(sb3_path), device=device)
            # Try loading VecNormalize stats
            vn_path = Path(model_dir) / f"{env_name}_vecnormalize.pkl"
            if vn_path.exists():
                with open(vn_path, "rb") as f:
                    vec_normalize = pickle.load(f)
            return model, vec_normalize

    # Try raw PyTorch format
    pt_path = Path(model_dir) / f"{env_name}_target_policy.pt"
    if pt_path.exists():
        model = torch.load(pt_path, map_location=device)
        return model, vec_normalize

    # Try generic model path
    for ext in [".zip", ".pt", ".pth"]:
        for fname in Path(model_dir).glob(f"*{ext}"):
            if ext == ".zip" and HAS_SB3:
                model = PPO.load(str(fname), device=device)
                return model, vec_normalize
            elif ext in [".pt", ".pth"]:
                model = torch.load(str(fname), map_location=device)
                return model, vec_normalize

    raise FileNotFoundError(f"No target policy found in {model_dir}")


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a deterministic policy function from a loaded model."""
    if HAS_SB3 and isinstance(model, PPO):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            action, _ = model.predict(state, deterministic=True)
            return action
        return policy_fn

    # Raw PyTorch model
    if hasattr(model, "get_action"):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                if state_t.ndim == 1:
                    state_t = state_t.unsqueeze(0)
                action = model.get_action(state_t, deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim == 2:
                    action = action[0]
                return action
        return policy_fn

    if hasattr(model, "forward"):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                if state_t.ndim == 1:
                    state_t = state_t.unsqueeze(0)
                output = model(state_t)
                if isinstance(output, tuple):
                    output = output[0]
                action = output.cpu().numpy()
                if action.ndim == 2:
                    action = action[0]
                return action
        return policy_fn

    # Fallback: treat model as callable
    def policy_fn(state: np.ndarray) -> np.ndarray:
        return model(state)
    return policy_fn


def load_refined_policy(
    refine_dir: str,
    env_name: str,
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load a RICE-refined policy from a PyTorch checkpoint."""
    refine_path = Path(refine_dir) / f"{env_name}_refined_policy.pt"
    if not refine_path.exists():
        # Try alternative names
        for fname in Path(refine_dir).glob("*refined*.pt"):
            refine_path = fname
            break
        else:
            print(f"Warning: No refined policy found in {refine_dir}")
            return None

    checkpoint = torch.load(str(refine_path), map_location=device)

    # Reconstruct policy network
    policy_state_dict = checkpoint.get("policy_state_dict", checkpoint.get("model_state_dict"))
    if policy_state_dict is None:
        print(f"Warning: No policy state dict in checkpoint")
        return None

    # Infer architecture from state dict
    hidden_sizes = checkpoint.get("hidden_sizes", [256, 256])
    activation_name = checkpoint.get("activation", "tanh")
    discrete = checkpoint.get("discrete_action", False)
    num_discrete = checkpoint.get("num_discrete_actions", None)

    # Build policy network
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(activation_name, nn.Tanh)

    layers = []
    prev_dim = state_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation_fn())
        prev_dim = h

    if discrete and num_discrete:
        layers.append(nn.Linear(prev_dim, num_discrete))
    else:
        layers.append(nn.Linear(prev_dim, action_dim))

    policy_net = nn.Sequential(*layers)

    # Load state dict
    try:
        policy_net.load_state_dict(policy_state_dict)
    except Exception as e:
        print(f"Warning: Could not load state dict directly: {e}")
        # Try loading with flexible matching
        model_dict = policy_net.state_dict()
        matched_dict = {}
        for k, v in policy_state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                matched_dict[k] = v
        policy_net.load_state_dict(matched_dict, strict=False)

    policy_net.to(device)
    policy_net.eval()

    def policy_fn(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            if state_t.ndim == 1:
                state_t = state_t.unsqueeze(0)
            output = policy_net(state_t)
            action = output.cpu().numpy()
            if action.ndim == 2:
                action = action[0]
            return action

    return policy_fn


def load_baseline_policy(
    baseline_dir: str,
    env_name: str,
    baseline_name: str,
    device: str = "cpu",
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load a baseline policy from disk."""
    # Try SB3 format
    if HAS_SB3:
        sb3_path = Path(baseline_dir) / baseline_name / f"{env_name}_ppo_final.zip"
        if sb3_path.exists():
            model = PPO.load(str(sb3_path), device=device)
            def policy_fn(state: np.ndarray) -> np.ndarray:
                action, _ = model.predict(state, deterministic=True)
                return action
            return policy_fn

    # Try PyTorch format
    for ext in [".pt", ".pth"]:
        pt_path = Path(baseline_dir) / baseline_name / f"{env_name}_policy{ext}"
        if pt_path.exists():
            checkpoint = torch.load(str(pt_path), map_location=device)
            policy_state_dict = checkpoint.get("policy_state_dict", checkpoint.get("model_state_dict"))
            if policy_state_dict is not None:
                # Simple MLP reconstruction
                state_dim = checkpoint.get("state_dim", 259)
                action_dim = checkpoint.get("action_dim", 2)
                hidden_sizes = checkpoint.get("hidden_sizes", [256, 256])
                activation_name = checkpoint.get("activation", "tanh")
                activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}.get(activation_name, nn.Tanh)

                layers = []
                prev_dim = state_dim
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev_dim, h))
                    layers.append(activation_fn())
                    prev_dim = h
                layers.append(nn.Linear(prev_dim, action_dim))
                policy_net = nn.Sequential(*layers)

                try:
                    policy_net.load_state_dict(policy_state_dict)
                except Exception:
                    model_dict = policy_net.state_dict()
                    matched_dict = {k: v for k, v in policy_state_dict.items()
                                    if k in model_dict and model_dict[k].shape == v.shape}
                    policy_net.load_state_dict(matched_dict, strict=False)

                policy_net.to(device)
                policy_net.eval()

                def policy_fn(state: np.ndarray) -> np.ndarray:
                    with torch.no_grad():
                        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
                        if state_t.ndim == 1:
                            state_t = state_t.unsqueeze(0)
                        output = policy_net(state_t)
                        action = output.cpu().numpy()
                        if action.ndim == 2:
                            action = action[0]
                        return action
                return policy_fn

    print(f"Warning: No baseline policy found for {baseline_name} in {baseline_dir}")
    return None


def evaluate_all(
    env_name: str = "MetaDrive-Macro-v1",
    model_dir: str = "./trained_agents/autonomous_driving",
    refine_dir: str = "./refined_agents/autonomous_driving",
    baseline_dir: str = "./baseline_agents/autonomous_driving",
    mask_dir: str = "./mask_networks/autonomous_driving",
    output_dir: str = "./eval_results/autonomous_driving",
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target, refined, and baseline policies on MetaDrive."""
    set_seed(seed)

    # Get environment info
    temp_env = make_env(env_name, seed=seed, max_episode_steps=max_episode_steps)
    state_dim = get_state_dim(temp_env)
    action_dim = get_action_dim(temp_env)
    temp_env.close()

    results = {
        "env_name": env_name,
        "num_episodes": num_episodes,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": {},
    }

    # --- Evaluate Target Policy ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating TARGET policy on {env_name}")
        print(f"{'='*60}")

    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

        env = make_env(env_name, seed=seed, max_episode_steps=max_episode_steps)
        target_stats = evaluate_policy(
            env, target_policy_fn,
            num_episodes=num_episodes,
            max_steps=max_episode_steps or 1000,
            deterministic=True,
            verbose=verbose,
        )
        env.close()

        results["policies"]["target"] = {
            "mean_reward": float(target_stats["mean_reward"]),
            "std_reward": float(target_stats["std_reward"]),
            "mean_length": float(target_stats["mean_length"]),
            "std_length": float(target_stats["std_length"]),
            "all_rewards": [float(r) for r in target_stats["all_rewards"]],
        }

        if verbose:
            print(f"Target - Mean Reward: {target_stats['mean_reward']:.2f} ± {target_stats['std_reward']:.2f}")
    except Exception as e:
        print(f"Error evaluating target policy: {e}")
        results["policies"]["target"] = {"error": str(e)}

    # --- Evaluate RICE Refined Policy ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating RICE-REFINED policy on {env_name}")
        print(f"{'='*60}")

    try:
        refined_policy_fn = load_refined_policy(refine_dir, env_name, state_dim, action_dim, device)
        if refined_policy_fn is not None:
            env = make_env(env_name, seed=seed, max_episode_steps=max_episode_steps)
            refined_stats = evaluate_policy(
                env, refined_policy_fn,
                num_episodes=num_episodes,
                max_steps=max_episode_steps or 1000,
                deterministic=True,
                verbose=verbose,
            )
            env.close()

            results["policies"]["rice_refined"] = {
                "mean_reward": float(refined_stats["mean_reward"]),
                "std_reward": float(refined_stats["std_reward"]),
                "mean_length": float(refined_stats["mean_length"]),
                "std_length": float(refined_stats["std_length"]),
                "all_rewards": [float(r) for r in refined_stats["all_rewards"]],
            }

            if verbose:
                print(f"RICE Refined - Mean Reward: {refined_stats['mean_reward']:.2f} ± {refined_stats['std_reward']:.2f}")
        else:
            results["policies"]["rice_refined"] = {"error": "No refined policy found"}
    except Exception as e:
        print(f"Error evaluating refined policy: {e}")
        results["policies"]["rice_refined"] = {"error": str(e)}

    # --- Evaluate Baseline Policies ---
    baseline_names = ["ppo_finetune", "statemask", "jsrl", "sil", "random_explanation"]

    for baseline_name in baseline_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {baseline_name.upper()} baseline on {env_name}")
            print(f"{'='*60}")

        try:
            baseline_policy_fn = load_baseline_policy(baseline_dir, env_name, baseline_name, device)
            if baseline_policy_fn is not None:
                env = make_env(env_name, seed=seed, max_episode_steps=max_episode_steps)
                baseline_stats = evaluate_policy(
                    env, baseline_policy_fn,
                    num_episodes=num_episodes,
                    max_steps=max_episode_steps or 1000,
                    deterministic=True,
                    verbose=verbose,
                )
                env.close()

                results["policies"][baseline_name] = {
                    "mean_reward": float(baseline_stats["mean_reward"]),
                    "std_reward": float(baseline_stats["std_reward"]),
                    "mean_length": float(baseline_stats["mean_length"]),
                    "std_length": float(baseline_stats["std_length"]),
                    "all_rewards": [float(r) for r in baseline_stats["all_rewards"]],
                }

                if verbose:
                    print(f"{baseline_name} - Mean Reward: {baseline_stats['mean_reward']:.2f} ± {baseline_stats['std_reward']:.2f}")
            else:
                results["policies"][baseline_name] = {"error": f"No {baseline_name} policy found"}
        except Exception as e:
            print(f"Error evaluating {baseline_name}: {e}")
            results["policies"][baseline_name] = {"error": str(e)}

    # --- Compute Comparisons ---
    comparison = compute_comparison(results)
    results["comparison"] = comparison

    # --- Save Results ---
    os.makedirs(output_dir, exist_ok=True)
    results_path = Path(output_dir) / f"{env_name}_eval_results.json"
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

    target_data = results.get("policies", {}).get("target", {})
    if "mean_reward" not in target_data:
        return comparison

    target_mean = target_data["mean_reward"]

    for policy_name, policy_data in results.get("policies", {}).items():
        if policy_name == "target" or "mean_reward" not in policy_data:
            continue
        policy_mean = policy_data["mean_reward"]
        improvement = policy_mean - target_mean
        pct_improvement = (improvement / abs(target_mean)) * 100 if target_mean != 0 else 0.0

        comparison[policy_name] = {
            "mean_reward": float(policy_mean),
            "target_mean": float(target_mean),
            "absolute_improvement": float(improvement),
            "percent_improvement": float(pct_improvement),
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
    print(f"{'Policy':<25} {'Mean Reward':>15} {'Std Reward':>15} {'Improvement':>15}")
    print(f"{'-'*70}")

    for policy_name, policy_data in results.get("policies", {}).items():
        if "mean_reward" in policy_data:
            mean_r = policy_data["mean_reward"]
            std_r = policy_data["std_reward"]
            imp = comparison.get(policy_name, {}).get("absolute_improvement", 0.0)
            print(f"{policy_name:<25} {mean_r:>15.2f} {std_r:>15.2f} {imp:>+15.2f}")
        else:
            error_msg = policy_data.get("error", "Unknown error")
            print(f"{policy_name:<25} {'ERROR':>15} {'':>15} {error_msg[:30]}")

    print(f"{'='*80}\n")


def evaluate_fidelity(
    env_name: str = "MetaDrive-Macro-v1",
    mask_dir: str = "./mask_networks/autonomous_driving",
    model_dir: str = "./trained_agents/autonomous_driving",
    output_dir: str = "./eval_results/autonomous_driving",
    num_episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate fidelity (Pearson correlation) of the trained mask network."""
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating MASK FIDELITY on {env_name}")
        print(f"{'='*60}")

    # Get state dimension
    temp_env = make_env(env_name, seed=seed)
    state_dim = get_state_dim(temp_env)
    temp_env.close()

    # Load mask network
    mask_path = Path(mask_dir) / f"{env_name}_mask_network.pt"
    if not mask_path.exists():
        # Try alternative names
        for fname in Path(mask_dir).glob("*mask*.pt"):
            mask_path = fname
            break
        else:
            print(f"Warning: No mask network found in {mask_dir}")
            return {"error": "No mask network found"}

    mask_network = MaskNetwork(state_dim=state_dim, hidden_sizes=(128, 128))
    try:
        checkpoint = torch.load(str(mask_path), map_location=device)
        if "model_state_dict" in checkpoint:
            mask_network.load_state_dict(checkpoint["model_state_dict"])
        else:
            mask_network.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading mask network: {e}")
        return {"error": str(e)}

    mask_network.to(device)
    mask_network.eval()

    # Load target policy for fidelity computation
    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_policy_fn = make_target_policy_fn(model, vec_normalize, device)
    except Exception as e:
        print(f"Error loading target policy for fidelity: {e}")
        return {"error": str(e)}

    # Compute fidelity
    env = make_env(env_name, seed=seed)
    fidelity = compute_fidelity_from_env(
        mask_network=mask_network,
        env=env,
        target_policy=target_policy_fn,
        num_episodes=num_episodes,
        device=device,
    )
    env.close()

    result = {
        "env_name": env_name,
        "fidelity": float(fidelity),
        "num_episodes": num_episodes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if verbose:
        print(f"Fidelity (Pearson r): {fidelity:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fidelity_path = Path(output_dir) / f"{env_name}_fidelity.json"
    with open(fidelity_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def evaluate_sparse(
    env_name: str = "MetaDrive-Macro-v1",
    model_dir: str = "./trained_agents/autonomous_driving",
    refine_dir: str = "./refined_agents/autonomous_driving",
    output_dir: str = "./eval_results/autonomous_driving",
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target and RICE policies on sparse reward variant."""
    set_seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating SPARSE reward on {env_name}")
        print(f"{'='*60}")

    temp_env = make_env(env_name, seed=seed)
    state_dim = get_state_dim(temp_env)
    action_dim = get_action_dim(temp_env)
    temp_env.close()

    results = {
        "env_name": env_name,
        "reward_type": "sparse",
        "num_episodes": num_episodes,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": {},
    }

    # Target policy on sparse
    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_policy_fn = make_target_policy_fn(model, vec_normalize, device)

        env = make_env(env_name, seed=seed, use_sparse_reward=True)
        target_stats = evaluate_policy(
            env, target_policy_fn,
            num_episodes=num_episodes,
            max_steps=1000,
            deterministic=True,
            verbose=verbose,
        )
        env.close()

        results["policies"]["target_sparse"] = {
            "mean_reward": float(target_stats["mean_reward"]),
            "std_reward": float(target_stats["std_reward"]),
            "mean_length": float(target_stats["mean_length"]),
            "all_rewards": [float(r) for r in target_stats["all_rewards"]],
        }

        if verbose:
            print(f"Target (sparse) - Mean Reward: {target_stats['mean_reward']:.2f} ± {target_stats['std_reward']:.2f}")
    except Exception as e:
        print(f"Error evaluating target on sparse: {e}")
        results["policies"]["target_sparse"] = {"error": str(e)}

    # RICE refined on sparse
    try:
        refined_policy_fn = load_refined_policy(refine_dir, env_name, state_dim, action_dim, device)
        if refined_policy_fn is not None:
            env = make_env(env_name, seed=seed, use_sparse_reward=True)
            refined_stats = evaluate_policy(
                env, refined_policy_fn,
                num_episodes=num_episodes,
                max_steps=1000,
                deterministic=True,
                verbose=verbose,
            )
            env.close()

            results["policies"]["rice_refined_sparse"] = {
                "mean_reward": float(refined_stats["mean_reward"]),
                "std_reward": float(refined_stats["std_reward"]),
                "mean_length": float(refined_stats["mean_length"]),
                "all_rewards": [float(r) for r in refined_stats["all_rewards"]],
            }

            if verbose:
                print(f"RICE Refined (sparse) - Mean Reward: {refined_stats['mean_reward']:.2f} ± {refined_stats['std_reward']:.2f}")
        else:
            results["policies"]["rice_refined_sparse"] = {"error": "No refined policy found"}
    except Exception as e:
        print(f"Error evaluating refined on sparse: {e}")
        results["policies"]["rice_refined_sparse"] = {"error": str(e)}

    # Save
    os.makedirs(output_dir, exist_ok=True)
    sparse_path = Path(output_dir) / f"{env_name}_sparse_eval.json"
    with open(sparse_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nSparse results saved to {sparse_path}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RICE and baselines on autonomous driving (MetaDrive)"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "fidelity", "sparse", "compare"],
        help="Evaluation mode: all, fidelity, sparse, compare"
    )
    parser.add_argument(
        "--env_name", type=str, default="MetaDrive-Macro-v1",
        help="Environment name"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./trained_agents/autonomous_driving",
        help="Directory containing pre-trained target agent"
    )
    parser.add_argument(
        "--refine_dir", type=str, default="./refined_agents/autonomous_driving",
        help="Directory containing RICE-refined agent"
    )
    parser.add_argument(
        "--baseline_dir", type=str, default="./baseline_agents/autonomous_driving",
        help="Directory containing baseline agents"
    )
    parser.add_argument(
        "--mask_dir", type=str, default="./mask_networks/autonomous_driving",
        help="Directory containing trained mask network"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results/autonomous_driving",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: cpu or cuda"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=None,
        help="Maximum episode steps"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML config file"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress verbose output"
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    verbose = args.verbose and not args.quiet

    if args.mode == "fidelity":
        evaluate_fidelity(
            env_name=args.env_name,
            mask_dir=args.mask_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            verbose=verbose,
        )
    elif args.mode == "sparse":
        evaluate_sparse(
            env_name=args.env_name,
            model_dir=args.model_dir,
            refine_dir=args.refine_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            verbose=verbose,
        )
    elif args.mode == "compare":
        # Just load existing results and print comparison
        results_path = Path(args.output_dir) / f"{args.env_name}_eval_results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results = json.load(f)
            comparison = compute_comparison(results)
            print_summary_table(args.env_name, results, comparison)
        else:
            print(f"No results found at {results_path}. Run 'all' mode first.")
            return 1
    else:  # "all"
        evaluate_all(
            env_name=args.env_name,
            model_dir=args.model_dir,
            refine_dir=args.refine_dir,
            baseline_dir=args.baseline_dir,
            mask_dir=args.mask_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            max_episode_steps=args.max_episode_steps,
            verbose=verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())