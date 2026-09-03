#!/usr/bin/env python3
"""
RICE Refining Experiment Script

This script runs the RICE refining process: loads a pre-trained agent and
critical state buffer, then continues training with mixed initial state
distribution and RND exploration bonus.

Usage:
    python experiments/refine.py --env_id Hopper-v3 --agent_path models/hopper_agent.zip \
        --critical_states_path results/hopper_critical_states.pkl --output_dir results/refined_hopper
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rice.refining import refine_agent
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RICE refining on a pre-trained agent"
    )
    # Required arguments
    parser.add_argument(
        "--env_id",
        type=str,
        required=True,
        help="Gym environment ID (e.g., Hopper-v3)",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        required=True,
        help="Path to pre-trained agent model (.zip)",
    )
    parser.add_argument(
        "--critical_states_path",
        type=str,
        required=True,
        help="Path to critical states buffer (.pkl or .json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for refined model and logs",
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--env_config",
        type=str,
        default=None,
        help="Environment-specific config override (e.g., hopper)",
    )

    # Refining hyperparameters
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Total timesteps for refining (overrides config)",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Probability of sampling initial state from critical buffer",
    )
    parser.add_argument(
        "--lambda_rnd",
        type=float,
        default=None,
        help="RND exploration bonus coefficient",
    )
    parser.add_argument(
        "--use_rnd",
        action="store_true",
        default=None,
        help="Enable RND exploration bonus",
    )
    parser.add_argument(
        "--no_rnd",
        action="store_false",
        dest="use_rnd",
        help="Disable RND exploration bonus",
    )
    parser.add_argument(
        "--use_mixed_init",
        action="store_true",
        default=None,
        help="Enable mixed initial state distribution",
    )
    parser.add_argument(
        "--no_mixed_init",
        action="store_false",
        dest="use_mixed_init",
        help="Disable mixed initial state distribution",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--ppo_lr", type=float, default=None, help="PPO learning rate"
    )
    parser.add_argument(
        "--n_steps", type=int, default=2048, help="PPO n_steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="PPO batch size"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10, help="PPO n_epochs"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor"
    )
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="GAE lambda"
    )
    parser.add_argument(
        "--clip_range", type=float, default=0.2, help="PPO clip range"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.0, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5, help="Max gradient norm"
    )

    # RND hyperparameters
    parser.add_argument(
        "--rnd_embedding_dim", type=int, default=128, help="RND embedding dimension"
    )
    parser.add_argument(
        "--rnd_hidden_sizes",
        type=int,
        nargs="+",
        default=None,
        help="RND hidden layer sizes",
    )
    parser.add_argument(
        "--rnd_lr", type=float, default=1e-4, help="RND predictor learning rate"
    )
    parser.add_argument(
        "--rnd_update_freq",
        type=int,
        default=1000,
        help="Frequency of RND predictor updates",
    )
    parser.add_argument(
        "--rnd_batch_size", type=int, default=64, help="RND update batch size"
    )
    parser.add_argument(
        "--rnd_n_epochs", type=int, default=1, help="RND update epochs"
    )

    # Evaluation
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Frequency of evaluation during refining",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100000,
        help="Frequency of saving checkpoints",
    )

    # Other
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--verbose", type=int, default=1, help="Verbosity level"
    )
    parser.add_argument(
        "--state_buffer_size",
        type=int,
        default=100000,
        help="Size of state buffer for RND updates",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(
        env_name=args.env_config,
        base_config_path=args.config,
    )

    # Override config with command-line arguments
    if args.total_timesteps is not None:
        if "refining" not in config:
            config["refining"] = {}
        config["refining"]["total_timesteps"] = args.total_timesteps
    if args.p is not None:
        config.setdefault("refining", {})["p"] = args.p
    if args.lambda_rnd is not None:
        config.setdefault("refining", {})["lambda"] = args.lambda_rnd
    if args.use_rnd is not None:
        config.setdefault("refining", {})["use_rnd"] = args.use_rnd
    if args.use_mixed_init is not None:
        config.setdefault("refining", {})["use_mixed_init"] = args.use_mixed_init

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(
            get_project_root(), "results", "refined", args.env_id
        )
    output_dir = ensure_dir(args.output_dir)

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = get_device(args.device)

    # Log start
    print(f"Starting RICE refining for {args.env_id}")
    print(f"Agent: {args.agent_path}")
    print(f"Critical states: {args.critical_states_path}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    start_time = time.time()

    # Run refining
    refined_model, logger, model_path = refine_agent(
        env_id=args.env_id,
        agent_path=args.agent_path,
        critical_states_path=args.critical_states_path,
        config=config,
        output_dir=output_dir,
        seed=args.seed,
        total_timesteps=config.get("refining", {}).get("total_timesteps"),
        p=config.get("refining", {}).get("p"),
        lambda_rnd=config.get("refining", {}).get("lambda"),
        use_rnd=config.get("refining", {}).get("use_rnd", True),
        use_mixed_init=config.get("refining", {}).get("use_mixed_init", True),
        rnd_embedding_dim=args.rnd_embedding_dim,
        rnd_hidden_sizes=args.rnd_hidden_sizes,
        rnd_learning_rate=args.rnd_lr,
        ppo_learning_rate=args.ppo_lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        rnd_update_freq=args.rnd_update_freq,
        rnd_batch_size=args.rnd_batch_size,
        rnd_n_epochs=args.rnd_n_epochs,
        state_buffer_size=args.state_buffer_size,
        device=device,
        verbose=args.verbose,
        save_freq=args.save_freq,
    )

    elapsed = time.time() - start_time

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_env = make_env(args.env_id, seed=args.seed + 1000)
    eval_results = evaluate_policy(
        eval_env,
        refined_model,
        n_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    eval_env.close()

    # Save results summary
    results = {
        "env_id": args.env_id,
        "seed": args.seed,
        "agent_path": args.agent_path,
        "critical_states_path": args.critical_states_path,
        "refined_model_path": model_path,
        "total_timesteps": config.get("refining", {}).get("total_timesteps"),
        "p": config.get("refining", {}).get("p"),
        "lambda_rnd": config.get("refining", {}).get("lambda"),
        "use_rnd": config.get("refining", {}).get("use_rnd", True),
        "use_mixed_init": config.get("refining", {}).get("use_mixed_init", True),
        "final_mean_return": float(eval_results["mean_return"]),
        "final_std_return": float(eval_results["std_return"]),
        "elapsed_time": elapsed,
        "elapsed_time_formatted": format_time(elapsed),
    }

    results_path = os.path.join(output_dir, "refining_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save logger data
    logger_path = os.path.join(output_dir, "refining_logger.pkl")
    logger.save(logger_path)

    print(f"\nRefining completed in {format_time(elapsed)}")
    print(f"Final mean return: {eval_results['mean_return']:.2f} ± {eval_results['std_return']:.2f}")
    print(f"Results saved to {results_path}")
    print(f"Refined model saved to {model_path}")


if __name__ == "__main__":
    main()