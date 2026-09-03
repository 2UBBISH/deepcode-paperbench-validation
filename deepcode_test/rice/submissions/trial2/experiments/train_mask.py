#!/usr/bin/env python3
"""
Train Mask Network for RICE

This script trains a mask network ξ(s) that decides whether to trust the agent's
action (aᵉ=0) or take a random action (aᵉ=1). The mask is trained via PPO on a
perturbed environment to maximize the perturbed policy's performance while
receiving an intrinsic reward α for blinding.

Usage:
    python experiments/train_mask.py --env_id Hopper-v3 --agent_path models/hopper_agent.zip
    python experiments/train_mask.py --env_id Walker2d-v3 --agent_path models/walker_agent.zip --config config/env_specific/walker2d.yaml
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

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
)
from rice.mask_network import (
    train_mask_network,
    load_mask_network,
    compute_importance,
    get_mask_probability,
)
from rice.explanation import (
    ExplanationExtractor,
    extract_critical_states,
    compute_fidelity_score,
)
from rice.perturbed_env import PerturbedEnvWrapper


def load_agent_policy(agent_path: str, device: str = "auto"):
    """
    Load a pre-trained agent policy from a saved model.

    Args:
        agent_path: Path to the saved agent model (.zip file)
        device: Device to load the model on

    Returns:
        Loaded PPO model (or policy)
    """
    from stable_baselines3 import PPO

    if not os.path.exists(agent_path):
        raise FileNotFoundError(f"Agent model not found: {agent_path}")

    print(f"Loading agent from: {agent_path}")
    model = PPO.load(agent_path, device=device)
    print(f"Agent loaded successfully.")
    return model


def train_mask(
    env_id: str,
    agent_path: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    alpha: Optional[float] = None,
    hidden_sizes: Optional[list] = None,
    learning_rate: Optional[float] = None,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_epochs: Optional[int] = None,
    gamma: Optional[float] = None,
    ent_coef: Optional[float] = None,
    device: str = "auto",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
    resume_from: Optional[str] = None,
    extract_explanations: bool = True,
    num_explanation_trajectories: int = 100,
    explanation_max_steps: int = 1000,
    buffer_size: int = 10000,
    compute_fidelity: bool = True,
    fidelity_episodes: int = 100,
    **env_kwargs,
) -> Dict[str, Any]:
    """
    Train a mask network and optionally extract explanations.

    Args:
        env_id: Gymnasium environment ID
        agent_path: Path to pre-trained agent model
        config: Configuration dictionary (loaded from YAML if None)
        output_dir: Directory to save outputs
        seed: Random seed
        total_timesteps: Total timesteps for mask training
        alpha: Intrinsic reward coefficient for blinding
        hidden_sizes: Hidden layer sizes for mask network
        learning_rate: Learning rate for mask PPO
        n_steps: Steps per PPO update
        batch_size: Batch size for PPO
        n_epochs: Number of epochs per PPO update
        gamma: Discount factor
        ent_coef: Entropy coefficient
        device: Device to use
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        resume_from: Path to resume training from
        extract_explanations: Whether to extract critical states after training
        num_explanation_trajectories: Number of trajectories for explanation
        explanation_max_steps: Max steps per trajectory
        buffer_size: Size of critical state buffer
        compute_fidelity: Whether to compute fidelity score
        fidelity_episodes: Number of episodes for fidelity computation
        **env_kwargs: Additional environment keyword arguments

    Returns:
        Dictionary with results (model path, explanation path, metrics, etc.)
    """
    # Load configuration
    if config is None:
        config = load_config(env_id)

    # Set seed
    set_seed(seed)

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(
            get_project_root(), "outputs", "mask", env_id, f"seed_{seed}"
        )
    ensure_dir(output_dir)

    # Save config
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Setup logger
    logger = Logger(log_dir=output_dir)

    # Load agent policy
    print("=" * 60)
    print(f"Training Mask Network for {env_id}")
    print("=" * 60)
    agent_policy = load_agent_policy(agent_path, device=device)

    # Get mask config defaults from config file
    mask_config = config.get("mask", {})

    if total_timesteps is None:
        total_timesteps = mask_config.get("total_timesteps", 300000)
    if alpha is None:
        alpha = mask_config.get("alpha", 0.0001)
    if hidden_sizes is None:
        hidden_sizes = mask_config.get("hidden_sizes", [64, 64])
    if learning_rate is None:
        learning_rate = mask_config.get("learning_rate", 3e-4)
    if n_steps is None:
        n_steps = mask_config.get("n_steps", 2048)
    if batch_size is None:
        batch_size = mask_config.get("batch_size", 64)
    if n_epochs is None:
        n_epochs = mask_config.get("n_epochs", 10)
    if gamma is None:
        gamma = mask_config.get("gamma", 0.99)
    if ent_coef is None:
        ent_coef = mask_config.get("ent_coef", 0.0)

    print(f"\nMask Training Configuration:")
    print(f"  Environment: {env_id}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Alpha (intrinsic reward): {alpha}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  n_steps: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  Gamma: {gamma}")
    print(f"  Entropy coef: {ent_coef}")
    print(f"  Device: {device}")
    print(f"  Output dir: {output_dir}")
    print()

    # Train mask network
    start_time = time.time()
    mask_model, mask_logger, mask_path = train_mask_network(
        env_id=env_id,
        agent_policy=agent_policy,
        config=config,
        output_dir=output_dir,
        seed=seed,
        total_timesteps=total_timesteps,
        alpha=alpha,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        device=device,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose,
        resume_from=resume_from,
        **env_kwargs,
    )
    training_time = time.time() - start_time

    print(f"\nMask training completed in {format_time(training_time)}")
    print(f"Mask model saved to: {mask_path}")

    results = {
        "env_id": env_id,
        "seed": seed,
        "mask_path": mask_path,
        "training_time": training_time,
        "total_timesteps": total_timesteps,
        "alpha": alpha,
    }

    # Extract explanations if requested
    if extract_explanations:
        print("\n" + "=" * 60)
        print("Extracting Critical States (Explanations)")
        print("=" * 60)

        explanation_start = time.time()
        extractor, critical_states = extract_critical_states(
            mask_network=mask_model,
            agent_policy=agent_policy,
            env_id=env_id,
            config=config,
            output_dir=output_dir,
            num_trajectories=num_explanation_trajectories,
            max_steps=explanation_max_steps,
            buffer_size=buffer_size,
            seed=seed,
            device=device,
            use_perturbed_policy=False,
            deterministic_agent=True,
            top_k_per_trajectory=1,
            save_buffer=True,
            verbose=verbose,
            **env_kwargs,
        )
        explanation_time = time.time() - explanation_start

        buffer_path = os.path.join(output_dir, "critical_states.pkl")
        print(f"Critical states saved to: {buffer_path}")
        print(f"Number of critical states: {len(critical_states)}")
        print(f"Explanation extraction time: {format_time(explanation_time)}")

        # Log statistics
        stats = extractor.get_statistics()
        print(f"Importance statistics: mean={stats.get('mean_importance', 'N/A'):.4f}, "
              f"std={stats.get('std_importance', 'N/A'):.4f}, "
              f"max={stats.get('max_importance', 'N/A'):.4f}")

        results["explanation_path"] = buffer_path
        results["num_critical_states"] = len(critical_states)
        results["explanation_time"] = explanation_time
        results["importance_stats"] = stats

    # Compute fidelity score if requested
    if compute_fidelity and extract_explanations:
        print("\n" + "=" * 60)
        print("Computing Fidelity Score")
        print("=" * 60)

        fidelity_start = time.time()
        fidelity_results = compute_fidelity_score(
            mask_network=mask_model,
            agent_policy=agent_policy,
            env_id=env_id,
            critical_states=critical_states,
            num_episodes=fidelity_episodes,
            max_steps=explanation_max_steps,
            seed=seed,
            device=device,
            verbose=verbose,
            **env_kwargs,
        )
        fidelity_time = time.time() - fidelity_start

        print(f"Fidelity score: {fidelity_results.get('fidelity_score', 'N/A'):.4f}")
        print(f"Fidelity computation time: {format_time(fidelity_time)}")

        # Save fidelity results
        fidelity_path = os.path.join(output_dir, "fidelity_results.json")
        with open(fidelity_path, "w") as f:
            json.dump(fidelity_results, f, indent=2, default=str)

        results["fidelity"] = fidelity_results
        results["fidelity_path"] = fidelity_path
        results["fidelity_time"] = fidelity_time

    # Save overall results
    results_path = os.path.join(output_dir, "results.json")
    # Convert non-serializable values
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            serializable_results[k] = v
        elif isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        else:
            serializable_results[k] = str(v)

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print("=" * 60)
    print("Mask training pipeline completed!")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train RICE mask network and extract explanations"
    )

    # Required arguments
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Gymnasium environment ID (e.g., Hopper-v3)",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        required=True,
        help="Path to pre-trained agent model (.zip)",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saved models and results",
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Total timesteps for mask training",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Intrinsic reward coefficient for blinding (default: 0.0001)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="Hidden layer sizes (e.g., 64 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for mask PPO",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="Steps per PPO update",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=None,
        help="Number of epochs per PPO update",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Discount factor",
    )
    parser.add_argument(
        "--ent_coef",
        type=float,
        default=None,
        help="Entropy coefficient",
    )

    # Device and logging
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Evaluation frequency (timesteps)",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0, 1, 2)",
    )

    # Resume
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to resume training from",
    )

    # Explanation extraction
    parser.add_argument(
        "--no_extract_explanations",
        action="store_true",
        help="Skip explanation extraction",
    )
    parser.add_argument(
        "--num_explanation_trajectories",
        type=int,
        default=100,
        help="Number of trajectories for explanation extraction",
    )
    parser.add_argument(
        "--explanation_max_steps",
        type=int,
        default=1000,
        help="Max steps per trajectory for explanation",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10000,
        help="Size of critical state buffer",
    )

    # Fidelity
    parser.add_argument(
        "--no_compute_fidelity",
        action="store_true",
        help="Skip fidelity computation",
    )
    parser.add_argument(
        "--fidelity_episodes",
        type=int,
        default=100,
        help="Number of episodes for fidelity computation",
    )

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Run training
    results = train_mask(
        env_id=args.env_id,
        agent_path=args.agent_path,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        alpha=args.alpha,
        hidden_sizes=args.hidden_sizes,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        device=args.device,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
        resume_from=args.resume_from,
        extract_explanations=not args.no_extract_explanations,
        num_explanation_trajectories=args.num_explanation_trajectories,
        explanation_max_steps=args.explanation_max_steps,
        buffer_size=args.buffer_size,
        compute_fidelity=not args.no_compute_fidelity,
        fidelity_episodes=args.fidelity_episodes,
    )

    return results


if __name__ == "__main__":
    main()