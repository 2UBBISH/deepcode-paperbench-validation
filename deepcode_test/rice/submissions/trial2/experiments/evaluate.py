#!/usr/bin/env python3
"""
RICE Evaluation Script
=======================
Evaluates and compares RICE-refined agents against baseline methods across
all environments. Supports:

- Experiment I: Fidelity comparison (mask-based vs random explanation)
- Experiment II: Efficiency comparison (training time)
- Experiment III: Refining performance on MuJoCo (dense + sparse)
- Experiment IV: Other applications (selfish mining, CAGE, auto driving, malware)
- Experiment V: Case study & sensitivity analysis

Usage:
    # Evaluate a single agent
    python experiments/evaluate.py --env Hopper-v3 --agent-path models/hopper_refined.zip

    # Compare RICE vs baselines
    python experiments/evaluate.py --env Hopper-v3 --compare-all \
        --agent-path models/hopper_agent.zip \
        --rice-path models/hopper_refined.zip \
        --statemask-path models/hopper_statemask_refined.zip \
        --jsrl-path models/hopper_jsrl_refined.zip \
        --sil-path models/hopper_sil_refined.zip \
        --random-path models/hopper_random_refined.zip

    # Run full experiment suite
    python experiments/evaluate.py --experiment all --env Hopper-v3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rice.utils import (
    load_config,
    set_seed,
    Logger,
    ensure_dir,
    get_device,
    evaluate_policy,
    make_env,
    format_time,
    get_project_root,
    CriticalStateBuffer,
    collect_trajectories,
)
from rice.explanation import compute_fidelity_score, ExplanationExtractor
from rice.mask_network import load_mask_network, compute_importance
from rice.perturbed_env import PerturbedEnvWrapper

# Optional imports for baseline methods
try:
    from baselines.statemask import load_statemask, compute_statemask_importance
    HAS_STATEMASK = True
except ImportError:
    HAS_STATEMASK = False

try:
    from baselines.random_explanation import compute_random_fidelity_score
    HAS_RANDOM = True
except ImportError:
    HAS_RANDOM = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="RICE Evaluation Script - Compare RICE against baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Environment
    parser.add_argument(
        "--env", "--env-id", dest="env_id", type=str, default="Hopper-v3",
        help="Environment ID (e.g., Hopper-v3, Walker2d-v3, etc.)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--env-config", type=str, default=None,
        help="Environment-specific config override"
    )

    # Experiment selection
    parser.add_argument(
        "--experiment", type=str, default="evaluate",
        choices=["evaluate", "fidelity", "efficiency", "refining", "all", "sensitivity"],
        help="Which experiment to run"
    )

    # Agent paths
    parser.add_argument(
        "--agent-path", type=str, default=None,
        help="Path to base agent model (.zip)"
    )
    parser.add_argument(
        "--rice-path", type=str, default=None,
        help="Path to RICE-refined agent model (.zip)"
    )
    parser.add_argument(
        "--statemask-path", type=str, default=None,
        help="Path to StateMask-refined agent model (.zip)"
    )
    parser.add_argument(
        "--jsrl-path", type=str, default=None,
        help="Path to JSRL-refined agent model (.zip)"
    )
    parser.add_argument(
        "--sil-path", type=str, default=None,
        help="Path to SIL-refined agent model (.zip)"
    )
    parser.add_argument(
        "--random-path", type=str, default=None,
        help="Path to random-explanation refined agent model (.zip)"
    )
    parser.add_argument(
        "--finetune-path", type=str, default=None,
        help="Path to PPO fine-tuned agent model (.zip)"
    )

    # Mask network paths (for fidelity)
    parser.add_argument(
        "--mask-path", type=str, default=None,
        help="Path to trained mask network (.zip)"
    )
    parser.add_argument(
        "--statemask-mask-path", type=str, default=None,
        help="Path to StateMask-trained mask network (.zip)"
    )

    # Critical states paths
    parser.add_argument(
        "--critical-states-path", type=str, default=None,
        help="Path to critical states buffer (.pkl)"
    )

    # Output
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--save-results", action="store_true", default=True,
        help="Save results to JSON"
    )
    parser.add_argument(
        "--no-save", action="store_false", dest="save_results",
        help="Do not save results"
    )

    # Evaluation settings
    parser.add_argument(
        "--n-episodes", type=int, default=100,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic policy for evaluation"
    )
    parser.add_argument(
        "--stochastic", action="store_false", dest="deterministic",
        help="Use stochastic policy for evaluation"
    )

    # Comparison settings
    parser.add_argument(
        "--compare-all", action="store_true", default=False,
        help="Compare all available methods"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of seeds for statistical comparison"
    )

    # Fidelity settings
    parser.add_argument(
        "--fidelity-episodes", type=int, default=100,
        help="Number of episodes for fidelity computation"
    )

    # Sensitivity analysis
    parser.add_argument(
        "--sensitivity-param", type=str, default="p",
        choices=["p", "lambda", "alpha"],
        help="Parameter for sensitivity analysis"
    )
    parser.add_argument(
        "--sensitivity-values", type=str, default=None,
        help="Comma-separated values for sensitivity analysis"
    )

    return parser.parse_args()


def load_agent(path: str, device: str = "auto") -> Any:
    """Load a trained PPO agent from a .zip file."""
    from stable_baselines3 import PPO

    if not os.path.exists(path):
        raise FileNotFoundError(f"Agent not found: {path}")

    agent = PPO.load(path, device=device)
    return agent


def evaluate_single_agent(
    env_id: str,
    agent_path: str,
    config: Dict[str, Any],
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    deterministic: bool = True,
    verbose: int = 1,
    **env_kwargs,
) -> Dict[str, Any]:
    """Evaluate a single agent and return metrics."""
    set_seed(seed)

    if verbose:
        print(f"Loading agent from: {agent_path}")

    agent = load_agent(agent_path, device=device)
    env = make_env(env_id, seed=seed, **env_kwargs)

    if verbose:
        print(f"Evaluating on {env_id} for {n_episodes} episodes...")

    start_time = time.time()
    results = evaluate_policy(
        env, agent, n_episodes=n_episodes,
        deterministic=deterministic, render=False,
    )
    elapsed = time.time() - start_time

    env.close()

    return {
        "env_id": env_id,
        "agent_path": agent_path,
        "mean_return": float(results["mean_return"]),
        "std_return": float(results["std_return"]),
        "min_return": float(results.get("min_return", np.nan)),
        "max_return": float(results.get("max_return", np.nan)),
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "evaluation_time": elapsed,
        "seed": seed,
    }


def run_fidelity_experiment(
    env_id: str,
    agent_path: str,
    mask_path: str,
    config: Dict[str, Any],
    output_dir: str,
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 1,
    statemask_mask_path: Optional[str] = None,
    **env_kwargs,
) -> Dict[str, Any]:
    """Run Experiment I: Fidelity comparison.

    Computes fidelity scores for RICE mask network and optionally StateMask,
    comparing against random explanation baseline.
    """
    set_seed(seed)
    results = {}

    # Load agent
    agent = load_agent(agent_path, device=device)

    # --- RICE Fidelity ---
    if verbose:
        print("\n" + "=" * 60)
        print("Computing RICE Fidelity Score")
        print("=" * 60)

    mask_network = load_mask_network(mask_path, device=device)

    # Extract critical states from mask
    extractor = ExplanationExtractor(
        mask_network=mask_network,
        agent_policy=agent,
        config=config,
        device=device,
    )

    env = make_env(env_id, seed=seed, **env_kwargs)
    trajectories = collect_trajectories(
        env, agent, num_trajectories=n_episodes,
        max_steps=max_steps, deterministic=True,
    )
    env.close()

    extractor.extract_from_trajectories(trajectories)
    critical_states = extractor.get_top_critical_states(k=n_episodes)

    rice_fidelity = compute_fidelity_score(
        mask_network=mask_network,
        agent_policy=agent,
        env_id=env_id,
        critical_states=critical_states,
        num_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        device=device,
        verbose=verbose,
        **env_kwargs,
    )
    results["rice_fidelity"] = rice_fidelity

    if verbose:
        print(f"RICE Fidelity: {rice_fidelity.get('fidelity_score', 'N/A')}")

    # --- StateMask Fidelity (if available) ---
    if statemask_mask_path and HAS_STATEMASK:
        if verbose:
            print("\n" + "=" * 60)
            print("Computing StateMask Fidelity Score")
            print("=" * 60)

        statemask_network = load_statemask(statemask_mask_path, device=device)

        statemask_extractor = ExplanationExtractor(
            mask_network=statemask_network,
            agent_policy=agent,
            config=config,
            device=device,
        )

        env = make_env(env_id, seed=seed, **env_kwargs)
        trajectories_sm = collect_trajectories(
            env, agent, num_trajectories=n_episodes,
            max_steps=max_steps, deterministic=True,
        )
        env.close()

        statemask_extractor.extract_from_trajectories(trajectories_sm)
        sm_critical_states = statemask_extractor.get_top_critical_states(k=n_episodes)

        sm_fidelity = compute_fidelity_score(
            mask_network=statemask_network,
            agent_policy=agent,
            env_id=env_id,
            critical_states=sm_critical_states,
            num_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            device=device,
            verbose=verbose,
            **env_kwargs,
        )
        results["statemask_fidelity"] = sm_fidelity

        if verbose:
            print(f"StateMask Fidelity: {sm_fidelity.get('fidelity_score', 'N/A')}")

    # --- Random Explanation Fidelity ---
    if HAS_RANDOM:
        if verbose:
            print("\n" + "=" * 60)
            print("Computing Random Explanation Fidelity Score")
            print("=" * 60)

        # Use random critical states
        env = make_env(env_id, seed=seed, **env_kwargs)
        trajectories_rand = collect_trajectories(
            env, agent, num_trajectories=n_episodes,
            max_steps=max_steps, deterministic=True,
        )
        env.close()

        # Randomly select states
        random_states = []
        for traj in trajectories_rand:
            if len(traj["observations"]) > 0:
                idx = np.random.randint(0, len(traj["observations"]))
                random_states.append({
                    "state": traj["observations"][idx],
                    "action": traj["actions"][idx] if idx < len(traj["actions"]) else None,
                    "importance": np.random.random(),
                    "trajectory_id": traj.get("trajectory_id", 0),
                    "step": idx,
                })

        random_fidelity = compute_random_fidelity_score(
            agent_policy=agent,
            env_id=env_id,
            critical_states=random_states,
            num_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            device=device,
            verbose=verbose,
            **env_kwargs,
        )
        results["random_fidelity"] = random_fidelity

        if verbose:
            print(f"Random Fidelity: {random_fidelity.get('fidelity_score', 'N/A')}")

    # Save results
    if output_dir:
        results_path = os.path.join(output_dir, "fidelity_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\nFidelity results saved to: {results_path}")

    return results


def run_refining_comparison(
    env_id: str,
    config: Dict[str, Any],
    output_dir: str,
    agent_paths: Dict[str, str],
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    deterministic: bool = True,
    verbose: int = 1,
    **env_kwargs,
) -> Dict[str, Any]:
    """Run Experiment III/IV: Compare refined agents against baselines.

    Args:
        agent_paths: Dict mapping method name -> model path
            e.g., {"RICE": "path/to/rice.zip", "StateMask-R": "path/to/sm.zip", ...}
    """
    set_seed(seed)
    results = {}

    if verbose:
        print("\n" + "=" * 60)
        print(f"Refining Performance Comparison: {env_id}")
        print("=" * 60)

    for method_name, agent_path in agent_paths.items():
        if not agent_path or not os.path.exists(agent_path):
            if verbose:
                print(f"  Skipping {method_name}: path not found ({agent_path})")
            continue

        if verbose:
            print(f"\n  Evaluating {method_name}...")

        eval_result = evaluate_single_agent(
            env_id=env_id,
            agent_path=agent_path,
            config=config,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            device=device,
            deterministic=deterministic,
            verbose=0,
            **env_kwargs,
        )
        results[method_name] = eval_result

        if verbose:
            print(f"    Mean Return: {eval_result['mean_return']:.2f} ± {eval_result['std_return']:.2f}")

    # Compute rankings
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1]["mean_return"], reverse=True)
        if verbose:
            print("\n  Ranking:")
            for rank, (name, res) in enumerate(ranked, 1):
                print(f"    {rank}. {name}: {res['mean_return']:.2f} ± {res['std_return']:.2f}")

    # Save results
    if output_dir:
        results_path = os.path.join(output_dir, f"comparison_{env_id.replace('-', '_').lower()}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        if verbose:
            print(f"\n  Comparison results saved to: {results_path}")

    return results


def run_efficiency_comparison(
    env_id: str,
    config: Dict[str, Any],
    output_dir: str,
    rice_training_time: float,
    statemask_training_time: Optional[float] = None,
    verbose: int = 1,
) -> Dict[str, Any]:
    """Run Experiment II: Efficiency comparison.

    Compares wall-clock training time for mask network.
    """
    results = {
        "env_id": env_id,
        "rice_training_time": rice_training_time,
        "statemask_training_time": statemask_training_time,
    }

    if statemask_training_time is not None:
        speedup = (statemask_training_time - rice_training_time) / statemask_training_time * 100
        results["speedup_percent"] = speedup
        if verbose:
            print(f"\nEfficiency Comparison ({env_id}):")
            print(f"  RICE:      {format_time(rice_training_time)}")
            print(f"  StateMask: {format_time(statemask_training_time)}")
            print(f"  Speedup:   {speedup:.1f}%")

    if output_dir:
        results_path = os.path.join(output_dir, "efficiency_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"Efficiency results saved to: {results_path}")

    return results


def run_sensitivity_analysis(
    env_id: str,
    agent_path: str,
    critical_states_path: str,
    config: Dict[str, Any],
    output_dir: str,
    param: str = "p",
    values: Optional[List[float]] = None,
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 1,
    **env_kwargs,
) -> Dict[str, Any]:
    """Run Experiment V: Sensitivity analysis.

    Varies a parameter (p, lambda, or alpha) and evaluates the refined agent.
    """
    from rice.refining import refine_agent

    if values is None:
        if param == "p":
            values = [0.0, 0.25, 0.5, 0.75, 1.0]
        elif param == "lambda":
            values = [0.0, 0.001, 0.01, 0.1]
        elif param == "alpha":
            values = [0.01, 0.001, 0.0001]
        else:
            values = [0.0, 0.25, 0.5, 0.75, 1.0]

    set_seed(seed)
    results = {
        "env_id": env_id,
        "param": param,
        "values": values,
        "results": {},
    }

    if verbose:
        print("\n" + "=" * 60)
        print(f"Sensitivity Analysis: {param} on {env_id}")
        print(f"Values: {values}")
        print("=" * 60)

    for val in values:
        if verbose:
            print(f"\n  Testing {param} = {val}...")

        # Set parameter
        refine_kwargs = {}
        if param == "p":
            refine_kwargs["p"] = val
        elif param == "lambda":
            refine_kwargs["lambda_rnd"] = val
        elif param == "alpha":
            # Alpha is used during mask training, not refining
            # For sensitivity, we'd need to retrain mask; skip for now
            if verbose:
                print(f"    Alpha sensitivity requires mask retraining; skipping")
            continue

        # Run refining
        refined_model, logger, model_path = refine_agent(
            env_id=env_id,
            agent_path=agent_path,
            critical_states_path=critical_states_path,
            config=config,
            output_dir=os.path.join(output_dir, f"sensitivity_{param}_{val}"),
            seed=seed,
            device=device,
            verbose=0,
            **refine_kwargs,
            **env_kwargs,
        )

        # Evaluate
        eval_result = evaluate_single_agent(
            env_id=env_id,
            agent_path=model_path,
            config=config,
            n_episodes=n_episodes,
            max_steps=max_steps,
            seed=seed,
            device=device,
            verbose=0,
            **env_kwargs,
        )

        results["results"][str(val)] = {
            "mean_return": eval_result["mean_return"],
            "std_return": eval_result["std_return"],
        }

        if verbose:
            print(f"    Mean Return: {eval_result['mean_return']:.2f} ± {eval_result['std_return']:.2f}")

    # Save results
    if output_dir:
        results_path = os.path.join(output_dir, f"sensitivity_{param}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        if verbose:
            print(f"\nSensitivity results saved to: {results_path}")

    return results


def generate_summary_table(
    all_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    verbose: int = 1,
) -> str:
    """Generate a summary table from all experiment results.

    Returns formatted table string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("RICE Evaluation Summary")
    lines.append("=" * 80)

    for env_id, env_results in all_results.items():
        lines.append(f"\nEnvironment: {env_id}")
        lines.append("-" * 40)

        if "comparison" in env_results:
            lines.append(f"{'Method':<20} {'Mean Return':>15} {'Std':>10}")
            lines.append("-" * 45)
            for method, res in sorted(
                env_results["comparison"].items(),
                key=lambda x: x[1].get("mean_return", -float("inf")),
                reverse=True,
            ):
                mean = res.get("mean_return", float("nan"))
                std = res.get("std_return", float("nan"))
                lines.append(f"{method:<20} {mean:>15.2f} {std:>10.2f}")

        if "fidelity" in env_results:
            lines.append(f"\nFidelity Scores:")
            for method, res in env_results["fidelity"].items():
                fs = res.get("fidelity_score", "N/A")
                lines.append(f"  {method}: {fs}")

    table = "\n".join(lines)

    if output_dir:
        table_path = os.path.join(output_dir, "summary_table.txt")
        with open(table_path, "w") as f:
            f.write(table)
        if verbose:
            print(f"\nSummary table saved to: {table_path}")

    return table


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.env_id if args.config is None else None)
    if args.config:
        with open(args.config, "r") as f:
            override_config = yaml.safe_load(f)
        # Deep merge
        def deep_merge(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    deep_merge(base[k], v)
                else:
                    base[k] = v
        deep_merge(config, override_config)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            get_project_root(), "results", "evaluation",
            time.strftime("%Y%m%d_%H%M%S"),
        )
    ensure_dir(args.output_dir)

    device = get_device(args.device)
    set_seed(args.seed)

    if args.verbose:
        print("=" * 60)
        print("RICE Evaluation")
        print("=" * 60)
        print(f"Environment: {args.env_id}")
        print(f"Experiment: {args.experiment}")
        print(f"Output Dir: {args.output_dir}")
        print(f"Device: {device}")
        print("=" * 60)

    all_results = {}

    # --- Experiment: evaluate single agent ---
    if args.experiment in ["evaluate", "all"]:
        if args.agent_path:
            result = evaluate_single_agent(
                env_id=args.env_id,
                agent_path=args.agent_path,
                config=config,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                device=device,
                deterministic=args.deterministic,
                verbose=args.verbose,
            )
            all_results[args.env_id] = {"single_eval": result}

            if args.verbose:
                print(f"\nEvaluation Results:")
                print(f"  Mean Return: {result['mean_return']:.2f} ± {result['std_return']:.2f}")
                print(f"  Min: {result['min_return']:.2f}, Max: {result['max_return']:.2f}")
        else:
            print("No agent path provided. Use --agent-path to specify a model.")

    # --- Experiment I: Fidelity ---
    if args.experiment in ["fidelity", "all"]:
        if args.agent_path and args.mask_path:
            fidelity_results = run_fidelity_experiment(
                env_id=args.env_id,
                agent_path=args.agent_path,
                mask_path=args.mask_path,
                config=config,
                output_dir=args.output_dir,
                n_episodes=args.fidelity_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                device=device,
                verbose=args.verbose,
                statemask_mask_path=args.statemask_mask_path,
            )
            if args.env_id not in all_results:
                all_results[args.env_id] = {}
            all_results[args.env_id]["fidelity"] = fidelity_results
        else:
            print("Fidelity experiment requires --agent-path and --mask-path")

    # --- Experiment III/IV: Refining comparison ---
    if args.experiment in ["refining", "all"]:
        agent_paths = {}
        if args.rice_path:
            agent_paths["RICE"] = args.rice_path
        if args.statemask_path:
            agent_paths["StateMask-R"] = args.statemask_path
        if args.jsrl_path:
            agent_paths["JSRL"] = args.jsrl_path
        if args.sil_path:
            agent_paths["SIL"] = args.sil_path
        if args.random_path:
            agent_paths["Random"] = args.random_path
        if args.finetune_path:
            agent_paths["PPO-Finetune"] = args.finetune_path
        if args.agent_path:
            agent_paths["Original"] = args.agent_path

        if args.compare_all:
            # Add all available paths
            pass

        if agent_paths:
            comparison_results = run_refining_comparison(
                env_id=args.env_id,
                config=config,
                output_dir=args.output_dir,
                agent_paths=agent_paths,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                device=device,
                deterministic=args.deterministic,
                verbose=args.verbose,
            )
            if args.env_id not in all_results:
                all_results[args.env_id] = {}
            all_results[args.env_id]["comparison"] = comparison_results
        else:
            print("Refining comparison requires at least one agent path")

    # --- Experiment V: Sensitivity ---
    if args.experiment in ["sensitivity", "all"]:
        if args.agent_path and args.critical_states_path:
            values = None
            if args.sensitivity_values:
                values = [float(v.strip()) for v in args.sensitivity_values.split(",")]

            sensitivity_results = run_sensitivity_analysis(
                env_id=args.env_id,
                agent_path=args.agent_path,
                critical_states_path=args.critical_states_path,
                config=config,
                output_dir=args.output_dir,
                param=args.sensitivity_param,
                values=values,
                n_episodes=args.n_episodes,
                max_steps=args.max_steps,
                seed=args.seed,
                device=device,
                verbose=args.verbose,
            )
            if args.env_id not in all_results:
                all_results[args.env_id] = {}
            all_results[args.env_id]["sensitivity"] = sensitivity_results
        else:
            print("Sensitivity analysis requires --agent-path and --critical-states-path")

    # Generate summary
    if all_results and args.save_results:
        generate_summary_table(all_results, args.output_dir, args.verbose)

        # Save full results
        full_results_path = os.path.join(args.output_dir, "all_results.json")
        with open(full_results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        if args.verbose:
            print(f"\nFull results saved to: {full_results_path}")

    if args.verbose:
        print("\n" + "=" * 60)
        print("Evaluation complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()