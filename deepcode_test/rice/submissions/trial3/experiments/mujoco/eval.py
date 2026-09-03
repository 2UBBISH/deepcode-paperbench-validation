#!/usr/bin/env python3
"""
Evaluation script for MuJoCo experiments.

Loads pre-trained target agents, RICE-refined agents, and baseline agents,
evaluates them on the specified MuJoCo environment, and produces comparison
metrics matching the paper's Tables 5, 6 and Figures 5, 10.

Baselines:
  - Target (original PPO, no refinement)
  - PPO Fine-tune (continued PPO training)
  - RICE (our method)
  - StateMask-R (StateMask explanation + RICE-style refinement)
  - JSRL (Jump-Start RL)
  - SIL (Self-Imitation Learning)
  - Random Explanation (random importance scores + RICE-style refinement)

Usage:
    python experiments/mujoco/eval.py --env Hopper-v4 --model_dir ./trained_agents \\
        --refine_dir ./refined_agents --mask_dir ./mask_models \\
        --baseline_dir ./baseline_agents --output_dir ./eval_results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rice.utils import evaluate_policy, set_seed, to_numpy
from rice.env_wrappers import make_state_saveable, StateSaveWrapper


# ==============================================================================
# Configuration Loading
# ==============================================================================

def load_config(env_name: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and merge default config with environment-specific overrides."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Load default refine config
    default_path = base_dir / "configs" / "default_refine.yaml"
    if default_path.exists():
        with open(default_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Load environment-specific config
    env_config_path = base_dir / "configs" / "env_specific" / f"{env_name.lower().replace('-v4','')}.yaml"
    if env_config_path.exists():
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
        # Deep merge
        for key, value in env_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            custom_config = yaml.safe_load(f)
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return config


# ==============================================================================
# Environment Creation
# ==============================================================================

def make_env(env_name: str, seed: int = 42, max_episode_steps: Optional[int] = None) -> Any:
    """Create a MuJoCo environment wrapped with StateSaveWrapper."""
    import gym

    env = gym.make(env_name)
    if max_episode_steps is not None:
        env._max_episode_steps = max_episode_steps
    env = make_state_saveable(env)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


# ==============================================================================
# Policy Loading
# ==============================================================================

def load_target_policy(
    env_name: str, model_dir: str, device: str = "cpu"
) -> Tuple[Any, Optional[Any]]:
    """Load a pre-trained Stable-Baselines3 PPO model and optional VecNormalize stats."""
    from stable_baselines3 import PPO

    model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Target model not found: {model_path}")

    model = PPO.load(model_path, device=device)

    # Load VecNormalize if available
    vec_normalize = None
    vecnorm_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        import pickle
        with open(vecnorm_path, "rb") as f:
            vec_normalize = pickle.load(f)

    return model, vec_normalize


def make_target_policy_fn(
    model: Any, vec_normalize: Optional[Any], device: str = "cpu"
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a policy function from an SB3 PPO model.

    Returns a function: state -> action (deterministic).
    """
    def policy_fn(state: np.ndarray) -> np.ndarray:
        if vec_normalize is not None:
            state = vec_normalize.normalize_obs(state)
        action, _ = model.predict(state, deterministic=True)
        return action

    return policy_fn


def load_refined_policy(
    refine_dir: str, env_name: str, state_dim: int, action_dim: int, device: str = "cpu"
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load a RICE-refined policy from disk.

    Returns a policy function: state -> action.
    """
    import torch.nn as nn

    # Try loading the full RICERefine checkpoint
    refine_path = os.path.join(refine_dir, env_name, "refined_policy.pt")
    if not os.path.exists(refine_path):
        # Try alternative path
        refine_path = os.path.join(refine_dir, f"{env_name}_refined.pt")
    if not os.path.exists(refine_path):
        print(f"Warning: Refined policy not found at {refine_path}")
        return None

    checkpoint = torch.load(refine_path, map_location=device)

    # Reconstruct policy network
    from rice.refine import RICERefine

    # We need to reconstruct the policy from the state dict
    # The checkpoint should contain 'policy_state_dict'
    if "policy_state_dict" in checkpoint:
        policy_state = checkpoint["policy_state_dict"]
        # Determine architecture from state dict
        # Look for the first layer weight shape
        for key in policy_state:
            if "weight" in key and "0" in key:
                hidden_size = policy_state[key].shape[0]
                break
        else:
            hidden_size = 64

        # Build a simple MLP policy
        class PolicyNet(nn.Module):
            def __init__(self, state_dim, action_dim, hidden_sizes):
                super().__init__()
                layers = []
                prev_dim = state_dim
                for h in hidden_sizes:
                    layers.append(nn.Linear(prev_dim, h))
                    layers.append(nn.Tanh())
                    prev_dim = h
                self.features = nn.Sequential(*layers)
                self.mean = nn.Linear(prev_dim, action_dim)
                self.log_std = nn.Parameter(torch.zeros(action_dim))

            def forward(self, x):
                x = self.features(x)
                mean = self.mean(x)
                std = torch.exp(self.log_std.clamp(-20, 2))
                return mean, std

        policy_net = PolicyNet(state_dim, action_dim, [hidden_size, hidden_size])
        policy_net.load_state_dict(policy_state)
        policy_net.to(device)
        policy_net.eval()

        def policy_fn(state: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mean, _ = policy_net(s)
                return mean.cpu().numpy().flatten()

        return policy_fn

    return None


def load_baseline_policy(
    baseline_dir: str, env_name: str, baseline_name: str, device: str = "cpu"
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Load a baseline policy from disk.

    baseline_name: one of 'ppo_finetune', 'statemask', 'jsrl', 'sil', 'random_explanation'
    """
    # Try SB3 format first (for PPO fine-tune, JSRL, SIL)
    model_path = os.path.join(baseline_dir, baseline_name, f"{env_name}_final.zip")
    if os.path.exists(model_path):
        from stable_baselines3 import PPO
        model = PPO.load(model_path, device=device)

        def policy_fn(state: np.ndarray) -> np.ndarray:
            action, _ = model.predict(state, deterministic=True)
            return action
        return policy_fn

    # Try PyTorch state dict format
    pt_path = os.path.join(baseline_dir, baseline_name, f"{env_name}_policy.pt")
    if os.path.exists(pt_path):
        checkpoint = torch.load(pt_path, map_location=device)
        # Handle various formats
        if isinstance(checkpoint, dict) and "policy_state_dict" in checkpoint:
            # Similar reconstruction as above
            pass

    print(f"Warning: Baseline policy not found for {baseline_name} at {baseline_dir}")
    return None


# ==============================================================================
# Main Evaluation
# ==============================================================================

def evaluate_all(
    env_name: str,
    model_dir: str,
    refine_dir: str,
    baseline_dir: str,
    mask_dir: str,
    output_dir: str,
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    max_episode_steps: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate target, refined, and baseline policies on the given environment.

    Returns a dictionary with evaluation results for each method.
    """
    set_seed(seed)

    # Load config for environment dimensions
    config = load_config(env_name)
    state_dim = config.get("state_dim", 11)
    action_dim = config.get("action_dim", 3)
    max_steps = max_episode_steps or config.get("max_episode_steps", 1000)

    results = {}
    os.makedirs(output_dir, exist_ok=True)

    # --- Target Policy ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating Target Policy on {env_name}")
        print(f"{'='*60}")

    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_fn = make_target_policy_fn(model, vec_normalize, device)

        env = make_env(env_name, seed, max_steps)
        target_eval = evaluate_policy(
            env, target_fn, num_episodes=num_episodes,
            max_steps=max_steps, deterministic=True, verbose=verbose
        )
        env.close()

        results["target"] = {
            "mean_reward": float(target_eval["mean_reward"]),
            "std_reward": float(target_eval["std_reward"]),
            "mean_length": float(target_eval["mean_length"]),
            "all_rewards": [float(r) for r in target_eval["all_rewards"]],
        }
        if verbose:
            print(f"Target: {target_eval['mean_reward']:.2f} +/- {target_eval['std_reward']:.2f}")
    except Exception as e:
        print(f"Error evaluating target: {e}")
        results["target"] = {"error": str(e)}

    # --- RICE Refined Policy ---
    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating RICE Refined Policy on {env_name}")
        print(f"{'='*60}")

    try:
        refined_fn = load_refined_policy(refine_dir, env_name, state_dim, action_dim, device)
        if refined_fn is not None:
            env = make_env(env_name, seed, max_steps)
            refined_eval = evaluate_policy(
                env, refined_fn, num_episodes=num_episodes,
                max_steps=max_steps, deterministic=True, verbose=verbose
            )
            env.close()

            results["rice"] = {
                "mean_reward": float(refined_eval["mean_reward"]),
                "std_reward": float(refined_eval["std_reward"]),
                "mean_length": float(refined_eval["mean_length"]),
                "all_rewards": [float(r) for r in refined_eval["all_rewards"]],
            }
            if verbose:
                print(f"RICE: {refined_eval['mean_reward']:.2f} +/- {refined_eval['std_reward']:.2f}")
        else:
            results["rice"] = {"error": "Refined policy not found"}
    except Exception as e:
        print(f"Error evaluating RICE: {e}")
        results["rice"] = {"error": str(e)}

    # --- Baselines ---
    baseline_names = ["ppo_finetune", "statemask", "jsrl", "sil", "random_explanation"]
    for baseline_name in baseline_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating {baseline_name} on {env_name}")
            print(f"{'='*60}")

        try:
            baseline_fn = load_baseline_policy(baseline_dir, env_name, baseline_name, device)
            if baseline_fn is not None:
                env = make_env(env_name, seed, max_steps)
                baseline_eval = evaluate_policy(
                    env, baseline_fn, num_episodes=num_episodes,
                    max_steps=max_steps, deterministic=True, verbose=verbose
                )
                env.close()

                results[baseline_name] = {
                    "mean_reward": float(baseline_eval["mean_reward"]),
                    "std_reward": float(baseline_eval["std_reward"]),
                    "mean_length": float(baseline_eval["mean_length"]),
                    "all_rewards": [float(r) for r in baseline_eval["all_rewards"]],
                }
                if verbose:
                    print(f"{baseline_name}: {baseline_eval['mean_reward']:.2f} +/- {baseline_eval['std_reward']:.2f}")
            else:
                results[baseline_name] = {"error": f"Baseline policy not found for {baseline_name}"}
        except Exception as e:
            print(f"Error evaluating {baseline_name}: {e}")
            results[baseline_name] = {"error": str(e)}

    # --- Compute Comparison Statistics ---
    comparison = compute_comparison(results)

    # --- Save Results ---
    output_path = os.path.join(output_dir, f"{env_name}_eval_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "env_name": env_name,
            "num_episodes": num_episodes,
            "seed": seed,
            "results": results,
            "comparison": comparison,
        }, f, indent=2)
    if verbose:
        print(f"\nResults saved to {output_path}")

    # --- Print Summary Table ---
    print_summary_table(env_name, results, comparison)

    return {"results": results, "comparison": comparison}


def compute_comparison(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute pairwise comparisons between methods."""
    comparison = {}

    # Find target as reference
    target_reward = None
    if "target" in results and "mean_reward" in results["target"]:
        target_reward = results["target"]["mean_reward"]

    for method_name, method_results in results.items():
        if "mean_reward" not in method_results:
            continue
        comp = {
            "mean_reward": method_results["mean_reward"],
            "std_reward": method_results["std_reward"],
        }
        if target_reward is not None and method_name != "target":
            improvement = method_results["mean_reward"] - target_reward
            comp["improvement_over_target"] = improvement
            comp["improvement_pct"] = (
                (improvement / abs(target_reward)) * 100 if target_reward != 0 else float("inf")
            )
        comparison[method_name] = comp

    return comparison


def print_summary_table(
    env_name: str, results: Dict[str, Any], comparison: Dict[str, Any]
) -> None:
    """Print a formatted summary table of results."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {env_name}")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Mean Reward':>12} {'Std':>10} {'Improvement':>12} {'% Change':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    for method_name in ["target", "rice", "ppo_finetune", "statemask", "jsrl", "sil", "random_explanation"]:
        if method_name in comparison:
            c = comparison[method_name]
            impr = c.get("improvement_over_target", 0)
            pct = c.get("improvement_pct", 0)
            print(f"{method_name:<25} {c['mean_reward']:>12.2f} {c['std_reward']:>10.2f} {impr:>12.2f} {pct:>10.1f}%")
        elif method_name in results and "error" in results[method_name]:
            print(f"{method_name:<25} {'ERROR':>12} {'-':>10} {'-':>12} {'-':>10}")
        else:
            print(f"{method_name:<25} {'N/A':>12} {'-':>10} {'-':>12} {'-':>10}")

    print(f"{'='*80}\n")


# ==============================================================================
# Fidelity Evaluation
# ==============================================================================

def evaluate_fidelity(
    env_name: str,
    mask_dir: str,
    model_dir: str,
    output_dir: str,
    num_episodes: int = 10,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate the fidelity of the trained mask network.

    Fidelity = Pearson correlation between importance scores ξ(s) and
    Q-value differences Q(s,a) - E_a'[Q(s,a')].
    """
    from rice.mask_net import MaskNetwork, compute_fidelity_from_env

    set_seed(seed)

    # Load mask network
    config = load_config(env_name)
    state_dim = config.get("state_dim", 11)

    mask_path = os.path.join(mask_dir, env_name, "mask_network.pt")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(mask_dir, f"{env_name}_mask.pt")

    if not os.path.exists(mask_path):
        print(f"Warning: Mask network not found at {mask_path}")
        return {"error": "Mask network not found"}

    mask_net = MaskNetwork(state_dim=state_dim, hidden_sizes=(128, 128))
    checkpoint = torch.load(mask_path, map_location=device)
    if "model_state_dict" in checkpoint:
        mask_net.load_state_dict(checkpoint["model_state_dict"])
    else:
        mask_net.load_state_dict(checkpoint)
    mask_net.to(device)
    mask_net.eval()

    # Load target policy
    model, vec_normalize = load_target_policy(env_name, model_dir, device)
    target_fn = make_target_policy_fn(model, vec_normalize, device)

    # Create environment
    env = make_env(env_name, seed)

    # Compute fidelity
    fidelity = compute_fidelity_from_env(
        mask_net, env, target_fn,
        num_episodes=num_episodes,
        device=device,
    )

    env.close()

    if verbose:
        print(f"Fidelity for {env_name}: {fidelity:.4f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fidelity_path = os.path.join(output_dir, f"{env_name}_fidelity.json")
    with open(fidelity_path, "w") as f:
        json.dump({"env_name": env_name, "fidelity": float(fidelity)}, f, indent=2)

    return {"fidelity": float(fidelity)}


# ==============================================================================
# Sparse Reward Evaluation
# ==============================================================================

def evaluate_sparse(
    env_name: str,
    model_dir: str,
    refine_dir: str,
    output_dir: str,
    threshold: float = 1.0,
    num_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate on sparse reward variant of the environment."""
    import gym
    from rice.env_wrappers import make_state_saveable

    set_seed(seed)

    class SparseRewardWrapper(gym.Wrapper):
        """Convert dense reward to sparse: reward = 1 if x > threshold else 0."""
        def __init__(self, env, threshold=1.0):
            super().__init__(env)
            self.threshold = threshold

        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            # Use x-position as progress metric (works for Hopper, Walker2d, HalfCheetah)
            x_pos = info.get("x_position", obs[0] if len(obs) > 0 else 0)
            sparse_reward = 1.0 if x_pos > self.threshold else 0.0
            # For terminated/truncated (Gym 0.26+)
            if len(info) == 0:
                return obs, sparse_reward, done, info
            return obs, sparse_reward, done, info

    results = {}

    # Target on sparse
    try:
        model, vec_normalize = load_target_policy(env_name, model_dir, device)
        target_fn = make_target_policy_fn(model, vec_normalize, device)

        env = gym.make(env_name)
        env = SparseRewardWrapper(env, threshold)
        env = make_state_saveable(env)
        env.seed(seed)

        target_eval = evaluate_policy(
            env, target_fn, num_episodes=num_episodes,
            max_steps=1000, deterministic=True, verbose=verbose
        )
        env.close()
        results["target_sparse"] = {
            "mean_reward": float(target_eval["mean_reward"]),
            "std_reward": float(target_eval["std_reward"]),
        }
    except Exception as e:
        results["target_sparse"] = {"error": str(e)}

    # RICE on sparse
    try:
        config = load_config(env_name)
        state_dim = config.get("state_dim", 11)
        action_dim = config.get("action_dim", 3)

        refined_fn = load_refined_policy(refine_dir, env_name, state_dim, action_dim, device)
        if refined_fn is not None:
            env = gym.make(env_name)
            env = SparseRewardWrapper(env, threshold)
            env = make_state_saveable(env)
            env.seed(seed)

            refined_eval = evaluate_policy(
                env, refined_fn, num_episodes=num_episodes,
                max_steps=1000, deterministic=True, verbose=verbose
            )
            env.close()
            results["rice_sparse"] = {
                "mean_reward": float(refined_eval["mean_reward"]),
                "std_reward": float(refined_eval["std_reward"]),
            }
    except Exception as e:
        results["rice_sparse"] = {"error": str(e)}

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{env_name}_sparse_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"\nSparse Reward Results for {env_name} (threshold={threshold}):")
        for k, v in results.items():
            if "mean_reward" in v:
                print(f"  {k}: {v['mean_reward']:.4f} +/- {v['std_reward']:.4f}")

    return results


# ==============================================================================
# CLI
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RICE and baselines on MuJoCo environments"
    )
    parser.add_argument(
        "--env", type=str, default="Hopper-v4",
        choices=["Hopper-v4", "Walker2d-v4", "Reacher-v4", "HalfCheetah-v4"],
        help="MuJoCo environment name"
    )
    parser.add_argument(
        "--model_dir", type=str, default="./trained_agents",
        help="Directory containing pre-trained target PPO models"
    )
    parser.add_argument(
        "--refine_dir", type=str, default="./refined_agents",
        help="Directory containing RICE-refined policies"
    )
    parser.add_argument(
        "--mask_dir", type=str, default="./mask_models",
        help="Directory containing trained mask networks"
    )
    parser.add_argument(
        "--baseline_dir", type=str, default="./baseline_agents",
        help="Directory containing baseline policies"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_results",
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
        choices=["cpu", "cuda"],
        help="Device to run on"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=None,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--fidelity_only", action="store_true",
        help="Only evaluate fidelity (skip policy evaluation)"
    )
    parser.add_argument(
        "--sparse", action="store_true",
        help="Evaluate on sparse reward variant"
    )
    parser.add_argument(
        "--sparse_threshold", type=float, default=1.0,
        help="Threshold for sparse reward"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed output"
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

    os.makedirs(args.output_dir, exist_ok=True)

    if args.fidelity_only:
        evaluate_fidelity(
            env_name=args.env,
            mask_dir=args.mask_dir,
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
    elif args.sparse:
        evaluate_sparse(
            env_name=args.env,
            model_dir=args.model_dir,
            refine_dir=args.refine_dir,
            output_dir=args.output_dir,
            threshold=args.sparse_threshold,
            num_episodes=args.num_episodes,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
    else:
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
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()