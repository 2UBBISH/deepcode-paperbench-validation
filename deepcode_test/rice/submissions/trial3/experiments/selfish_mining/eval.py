#!/usr/bin/env python3
"""
Selfish Mining Evaluation Script for RICE

Evaluates target agents, RICE-refined agents, and baseline agents on the
selfish mining environment. Computes comparison metrics (mean reward,
improvement over target), evaluates mask network fidelity, and saves
results as JSON for downstream analysis.

Matches the paper's Table 5/6 evaluation protocol for the selfish mining domain.
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

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rice.utils import evaluate_policy, set_seed, to_numpy
from rice.env_wrappers import make_state_saveable, StateSaveWrapper
from rice.mask_net import MaskNetwork, compute_fidelity_from_env
from rice.refine import RICERefine

# Selfish mining environment
from experiments.selfish_mining.env import (
    make_env as make_sm_env,
    get_state_dim,
    get_action_dim,
    is_discrete_action,
    SelfishMiningEnv,
    SelfishMiningStateWrapper,
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

def load_config(
    env_name: str = "SelfishMining-v0",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge default refine config with environment-specific overrides."""
    config_dir = Path(__file__).resolve().parent.parent.parent / "configs"

    # Load defaults
    default_refine_path = config_dir / "default_refine.yaml"
    default_mask_path = config_dir / "default_mask.yaml"

    config = {}
    for path in [default_refine_path, default_mask_path]:
        if path.exists():
            with open(path, "r") as f:
                config.update(yaml.safe_load(f) or {})

    # Load env-specific
    env_config_path = config_dir / "env_specific" / "selfish_mining.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_specific = yaml.safe_load(f) or {}
            _deep_update(config, env_specific)

    # Load custom config
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            custom = yaml.safe_load(f) or {}
            _deep_update(config, custom)

    return config


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override values."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


# ==============================================================================
# Environment Creation
# ==============================================================================

def make_env(
    env_name: str = "SelfishMining-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    alpha: float = 0.35,
    gamma_sm: float = 0.5,
) -> gym.Env:
    """Create a selfish mining environment wrapped with state save/restore."""
    env = make_sm_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        alpha=alpha,
        gamma=gamma_sm,
    )
    return env


# ==============================================================================
# Policy Loading
# ==============================================================================

def load_target_policy(
    env_name: str,
    model_dir: str,
    device: str = "cpu",
) -> Tuple[Any, Optional[Any]]:
    """Load a pre-trained target PPO agent.

    Supports Stable-Baselines3 .zip models and raw PyTorch .pt checkpoints.
    Returns (model, vec_normalize) tuple.
    """
    model_path = Path(model_dir) / f"{env_name}_ppo_final.zip"
    pt_path = Path(model_dir) / f"{env_name}_target_policy.pt"
    vecnorm_path = Path(model_dir) / f"{env_name}_vecnormalize.pkl"

    vec_normalize = None
    if vecnorm_path.exists():
        with open(vecnorm_path, "rb") as f:
            vec_normalize = pickle.load(f)

    if model_path.exists() and HAS_SB3:
        model = PPO.load(str(model_path), device=device)
        if vec_normalize is not None:
            try:
                model.set_env(DummyVecEnv([lambda: make_env(env_name)]))
            except Exception:
                pass
        return model, vec_normalize
    elif pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        # Reconstruct policy network
        state_dim = get_state_dim(None)
        action_dim = get_action_dim(None)
        discrete = is_discrete_action(None)

        class PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = checkpoint.get("hidden_sizes", [128, 128])
                activation = checkpoint.get("activation", "tanh")
                act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
                layers = []
                prev = state_dim
                for h in hidden:
                    layers.append(nn.Linear(prev, h))
                    layers.append(act_fn())
                    prev = h
                self.features = nn.Sequential(*layers)
                if discrete:
                    self.actor = nn.Linear(prev, action_dim)
                else:
                    self.actor_mean = nn.Linear(prev, action_dim)
                    self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
                self.critic = nn.Linear(prev, 1)

            def forward(self, x):
                feats = self.features(x)
                if hasattr(self, "actor"):
                    action_logits = self.actor(feats)
                    value = self.critic(feats)
                    return action_logits, value
                else:
                    action_mean = self.actor_mean(feats)
                    value = self.critic(feats)
                    return action_mean, self.actor_log_std, value

        policy = PolicyNet()
        if "policy_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["policy_state_dict"])
        elif "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"])
        policy.to(device)
        policy.eval()
        return policy, vec_normalize
    else:
        raise FileNotFoundError(
            f"No target policy found at {model_path} or {pt_path}"
        )


def make_target_policy_fn(
    model: Any,
    vec_normalize: Optional[Any] = None,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a deterministic policy function from a loaded model.

    Returns a function: state -> action
    """
    if HAS_SB3 and hasattr(model, "predict"):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            if vec_normalize is not None:
                state = vec_normalize.normalize_obs(state)
            action, _ = model.predict(state, deterministic=True)
            return action
        return policy_fn
    elif isinstance(model, nn.Module):
        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=device)
                if s.ndim == 1:
                    s = s.unsqueeze(0)
                output = model(s)
                if isinstance(output, tuple):
                    if len(output) == 2:
                        # discrete: (logits, value)
                        action = torch.argmax(output[0], dim=-1)
                    else:
                        # continuous: (mean, log_std, value)
                        action = output[0]
                else:
                    action = output
                action = action.cpu().numpy()
                if action.ndim == 2 and action.shape[0] == 1:
                    action = action[0]
                return action
        return policy_fn
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


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
        print(f"Warning: Refined policy not found at {refine_path}")
        return None

    checkpoint = torch.load(refine_path, map_location=device)
    discrete = is_discrete_action(None)

    # Reconstruct policy network
    hidden_sizes = checkpoint.get("hidden_sizes", [128, 128])
    activation = checkpoint.get("activation", "tanh")
    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]

    layers = []
    prev = state_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_fn())
        prev = h
    feature_net = nn.Sequential(*layers)

    if discrete:
        actor = nn.Linear(prev, action_dim)
    else:
        actor_mean = nn.Linear(prev, action_dim)
        actor_log_std = nn.Parameter(torch.zeros(action_dim))

    class RefinedPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = feature_net
            if discrete:
                self.actor = actor
            else:
                self.actor_mean = actor_mean
                self.actor_log_std = actor_log_std
            self.discrete = discrete

        def forward(self, x):
            feats = self.features(x)
            if self.discrete:
                return self.actor(feats)
            else:
                return self.actor_mean(feats)

    policy = RefinedPolicy()
    if "policy_state_dict" in checkpoint:
        policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
    policy.to(device)
    policy.eval()

    def policy_fn(state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
            output = policy(s)
            if discrete:
                action = torch.argmax(output, dim=-1)
            else:
                action = output
            action = action.cpu().numpy()
            if action.ndim == 2 and action.shape[0] == 1:
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
    # Try SB3 format first
    baseline_path = Path(baseline_dir) / baseline_name / f"{env_name}_baseline.zip"
    if baseline_path.exists() and HAS_SB3:
        model = PPO.load(str(baseline_path), device=device)
        def policy_fn(state: np.ndarray) -> np.ndarray:
            action, _ = model.predict(state, deterministic=True)
            return action
        return policy_fn

    # Try PyTorch format
    pt_path = Path(baseline_dir) / baseline_name / f"{env_name}_policy.pt"
    if pt_path.exists():
        checkpoint = torch.load(pt_path, map_location=device)
        state_dim = get_state_dim(None)
        action_dim = get_action_dim(None)
        discrete = is_discrete_action(None)
        hidden_sizes = checkpoint.get("hidden_sizes", [128, 128])

        class BaselinePolicy(nn.Module):
            def __init__(self):
                super().__init__()
                activation = checkpoint.get("activation", "tanh")
                act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}[activation]
                layers = []
                prev = state_dim
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev, h))
                    layers.append(act_fn())
                    prev = h
                self.features = nn.Sequential(*layers)
                if discrete:
                    self.actor = nn.Linear(prev, action_dim)
                else:
                    self.actor_mean = nn.Linear(prev, action_dim)
                self.discrete = discrete

            def forward(self, x):
                feats = self.features(x)
                if self.discrete:
                    return self.actor(feats)
                return self.actor_mean(feats)

        policy = BaselinePolicy()
        if "policy_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["policy_state_dict"], strict=False)
        elif "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        policy.to(device)
        policy.eval()

        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=device)
                if s.ndim == 1:
                    s = s.unsqueeze(0)
                output = policy(s)
                if discrete:
                    action = torch.argmax(output, dim=-1)
                else:
                    action = output
                action = action.cpu().numpy()
                if action.ndim == 2 and action.shape[0] == 1:
                    action = action[0]
                return action
        return policy_fn

    print(f"Warning: Baseline policy not found for {baseline_name} at {baseline_dir}")
    return None


# ==============================================================================
# Main Evaluation Functions
# ==============================================================================

def evaluate_all(
    env_name: str = "SelfishMining-v0",
    model_dir: str = "./trained_agents/selfish_mining",
    refine_dir: str = "./refined_agents/selfish_mining",
    baseline_dir: str = "./baseline_agents/selfish_mining",
    mask_dir: str = "./mask_networks/selfish_mining",
    output_dir: str = "./eval_results/selfish_mining",
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target, refined, and baseline policies on selfish mining.

    Returns a dictionary with all evaluation results.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    state_dim = get_state_dim(None)
    action_dim = get_action_dim(None)

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

    target_model, target_vecnorm = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(target_model, target_vecnorm, device)

    env = make_env(env_name, seed, max_episode_steps)
    target_eval = evaluate_policy(
        env, target_policy_fn, num_episodes=num_episodes,
        max_steps=max_episode_steps or 100, deterministic=True, verbose=verbose
    )
    env.close()

    results["policies"]["target"] = {
        "mean_reward": float(target_eval["mean_reward"]),
        "std_reward": float(target_eval["std_reward"]),
        "mean_length": float(target_eval["mean_length"]),
        "all_rewards": [float(r) for r in target_eval["all_rewards"]],
    }

    if verbose:
        print(f"Target: mean_reward={target_eval['mean_reward']:.4f} "
              f"± {target_eval['std_reward']:.4f}")

    # --- Evaluate RICE-Refined Policy ---
    refined_policy_fn = load_refined_policy(
        refine_dir, env_name, state_dim, action_dim, device
    )

    if refined_policy_fn is not None:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating RICE-REFINED policy on {env_name}")
            print(f"{'='*60}")

        env = make_env(env_name, seed + 1, max_episode_steps)
        refined_eval = evaluate_policy(
            env, refined_policy_fn, num_episodes=num_episodes,
            max_steps=max_episode_steps or 100, deterministic=True, verbose=verbose
        )
        env.close()

        results["policies"]["rice_refined"] = {
            "mean_reward": float(refined_eval["mean_reward"]),
            "std_reward": float(refined_eval["std_reward"]),
            "mean_length": float(refined_eval["mean_length"]),
            "all_rewards": [float(r) for r in refined_eval["all_rewards"]],
        }

        if verbose:
            print(f"RICE Refined: mean_reward={refined_eval['mean_reward']:.4f} "
                  f"± {refined_eval['std_reward']:.4f}")
    else:
        results["policies"]["rice_refined"] = None

    # --- Evaluate Baseline Policies ---
    baseline_names = ["ppo_finetune", "statemask", "jsrl", "sil", "random_explanation"]
    for baseline_name in baseline_names:
        baseline_fn = load_baseline_policy(
            baseline_dir, env_name, baseline_name, device
        )
        if baseline_fn is not None:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating {baseline_name.upper()} baseline on {env_name}")
                print(f"{'='*60}")

            env = make_env(env_name, seed + 2, max_episode_steps)
            baseline_eval = evaluate_policy(
                env, baseline_fn, num_episodes=num_episodes,
                max_steps=max_episode_steps or 100, deterministic=True, verbose=verbose
            )
            env.close()

            results["policies"][baseline_name] = {
                "mean_reward": float(baseline_eval["mean_reward"]),
                "std_reward": float(baseline_eval["std_reward"]),
                "mean_length": float(baseline_eval["mean_length"]),
                "all_rewards": [float(r) for r in baseline_eval["all_rewards"]],
            }

            if verbose:
                print(f"{baseline_name}: mean_reward={baseline_eval['mean_reward']:.4f} "
                      f"± {baseline_eval['std_reward']:.4f}")
        else:
            results["policies"][baseline_name] = None

    # --- Compute Comparisons ---
    comparison = compute_comparison(results)
    results["comparison"] = comparison

    # --- Save Results ---
    results_path = Path(output_dir) / f"{env_name}_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to {results_path}")

    # --- Print Summary ---
    print_summary_table(env_name, results, comparison)

    return results


def compute_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pairwise comparisons (improvement over target)."""
    comparison = {}
    target_reward = None
    if results["policies"].get("target") is not None:
        target_reward = results["policies"]["target"]["mean_reward"]

    for policy_name, policy_data in results["policies"].items():
        if policy_name == "target" or policy_data is None:
            continue
        mean_reward = policy_data["mean_reward"]
        if target_reward is not None:
            improvement = mean_reward - target_reward
            pct_improvement = (improvement / abs(target_reward)) * 100 if target_reward != 0 else 0
            comparison[policy_name] = {
                "mean_reward": mean_reward,
                "improvement_over_target": improvement,
                "pct_improvement": pct_improvement,
            }
        else:
            comparison[policy_name] = {
                "mean_reward": mean_reward,
                "improvement_over_target": None,
                "pct_improvement": None,
            }

    return comparison


def print_summary_table(
    env_name: str,
    results: Dict[str, Any],
    comparison: Dict[str, Any],
) -> None:
    """Print a formatted summary table of mean rewards and improvements."""
    print(f"\n{'='*70}")
    print(f"SUMMARY: {env_name}")
    print(f"{'='*70}")
    print(f"{'Policy':<25} {'Mean Reward':>12} {'Improvement':>12} {'% Change':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    target_data = results["policies"].get("target")
    if target_data:
        print(f"{'Target (Pre-trained)':<25} {target_data['mean_reward']:>12.4f} "
              f"{'---':>12} {'---':>10}")

    for policy_name, comp in comparison.items():
        display_name = policy_name.replace("_", " ").title()
        impr = comp.get("improvement_over_target", "N/A")
        pct = comp.get("pct_improvement", "N/A")
        impr_str = f"{impr:>12.4f}" if isinstance(impr, (int, float)) else f"{impr:>12}"
        pct_str = f"{pct:>9.1f}%" if isinstance(pct, (int, float)) else f"{pct:>10}"
        print(f"{display_name:<25} {comp['mean_reward']:>12.4f} {impr_str} {pct_str}")

    print(f"{'='*70}\n")


# ==============================================================================
# Fidelity Evaluation
# ==============================================================================

def evaluate_fidelity(
    env_name: str = "SelfishMining-v0",
    mask_dir: str = "./mask_networks/selfish_mining",
    model_dir: str = "./trained_agents/selfish_mining",
    output_dir: str = "./eval_results/selfish_mining",
    num_episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate fidelity (Pearson correlation) of the trained mask network."""
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    state_dim = get_state_dim(None)

    # Load mask network
    mask_path = Path(mask_dir) / f"{env_name}_mask_network.pt"
    if not mask_path.exists():
        print(f"Warning: Mask network not found at {mask_path}")
        return {"fidelity": None, "error": "Mask network not found"}

    mask_network = MaskNetwork(state_dim=state_dim, hidden_sizes=(128, 128))
    checkpoint = torch.load(mask_path, map_location=device)
    if "mask_network_state_dict" in checkpoint:
        mask_network.load_state_dict(checkpoint["mask_network_state_dict"])
    elif "model_state_dict" in checkpoint:
        mask_network.load_state_dict(checkpoint["model_state_dict"])
    mask_network.to(device)
    mask_network.eval()

    # Load target policy
    target_model, target_vecnorm = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(target_model, target_vecnorm, device)

    # Create environment
    env = make_env(env_name, seed, max_episode_steps=100)

    # Compute fidelity
    if verbose:
        print(f"\nComputing fidelity for {env_name}...")

    fidelity = compute_fidelity_from_env(
        mask_network=mask_network,
        env=env,
        target_policy=target_policy_fn,
        num_episodes=num_episodes,
        q_function=None,
        device=device,
    )
    env.close()

    result = {
        "env_name": env_name,
        "fidelity": float(fidelity) if fidelity is not None else None,
        "num_episodes": num_episodes,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save
    fidelity_path = Path(output_dir) / f"{env_name}_fidelity.json"
    with open(fidelity_path, "w") as f:
        json.dump(result, f, indent=2)

    if verbose:
        print(f"Fidelity: {fidelity:.4f}" if fidelity is not None else "Fidelity: N/A")
        print(f"Saved to {fidelity_path}")

    return result


# ==============================================================================
# Sparse Reward Evaluation (for selfish mining, sparse = default reward structure)
# ==============================================================================

def evaluate_sparse(
    env_name: str = "SelfishMining-v0",
    model_dir: str = "./trained_agents/selfish_mining",
    refine_dir: str = "./refined_agents/selfish_mining",
    output_dir: str = "./eval_results/selfish_mining",
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target and RICE policies on the default (sparse) reward variant.

    For selfish mining, the default reward is already sparse (only at episode end).
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    state_dim = get_state_dim(None)
    action_dim = get_action_dim(None)

    results = {
        "env_name": f"{env_name}_sparse",
        "num_episodes": num_episodes,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": {},
    }

    # Target policy
    target_model, target_vecnorm = load_target_policy(env_name, model_dir, device)
    target_policy_fn = make_target_policy_fn(target_model, target_vecnorm, device)

    env = make_env(env_name, seed, max_episode_steps=100)
    target_eval = evaluate_policy(
        env, target_policy_fn, num_episodes=num_episodes,
        max_steps=100, deterministic=True, verbose=verbose
    )
    env.close()
    results["policies"]["target"] = {
        "mean_reward": float(target_eval["mean_reward"]),
        "std_reward": float(target_eval["std_reward"]),
        "all_rewards": [float(r) for r in target_eval["all_rewards"]],
    }

    # Refined policy
    refined_policy_fn = load_refined_policy(
        refine_dir, env_name, state_dim, action_dim, device
    )
    if refined_policy_fn is not None:
        env = make_env(env_name, seed + 1, max_episode_steps=100)
        refined_eval = evaluate_policy(
            env, refined_policy_fn, num_episodes=num_episodes,
            max_steps=100, deterministic=True, verbose=verbose
        )
        env.close()
        results["policies"]["rice_refined"] = {
            "mean_reward": float(refined_eval["mean_reward"]),
            "std_reward": float(refined_eval["std_reward"]),
            "all_rewards": [float(r) for r in refined_eval["all_rewards"]],
        }

    # Save
    sparse_path = Path(output_dir) / f"{env_name}_sparse_eval.json"
    with open(sparse_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\nSparse evaluation results saved to {sparse_path}")

    return results


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RICE and baselines on Selfish Mining"
    )
    parser.add_argument(
        "--env_name", type=str, default="SelfishMining-v0",
        help="Environment name"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./trained_agents/selfish_mining",
        help="Directory containing pre-trained target agent"
    )
    parser.add_argument(
        "--refine_dir", type=str, default="./refined_agents/selfish_mining",
        help="Directory containing RICE-refined agent"
    )
    parser.add_argument(
        "--baseline_dir", type=str, default="./baseline_agents/selfish_mining",
        help="Directory containing baseline agents"
    )
    parser.add_argument(
        "--mask_dir", type=str, default="./mask_networks/selfish_mining",
        help="Directory containing trained mask network"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results/selfish_mining",
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
        help="Device to use (cpu or cuda)"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=None,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "fidelity", "sparse", "compare"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to custom YAML config file"
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


def main() -> int:
    """Main entry point."""
    args = parse_args()
    verbose = not args.quiet

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
        results = evaluate_all(
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
    else:  # "all"
        # Run fidelity
        evaluate_fidelity(
            env_name=args.env_name,
            mask_dir=args.mask_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_episodes=min(args.num_episodes, 10),
            seed=args.seed,
            device=args.device,
            verbose=verbose,
        )
        # Run full evaluation
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
        # Run sparse evaluation
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

    return 0


if __name__ == "__main__":
    sys.exit(main())