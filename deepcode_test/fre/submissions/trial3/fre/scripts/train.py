#!/usr/bin/env python3
"""
Main training script for Functional Reward Encodings (FRE).

This script orchestrates the two-phase training pipeline:
  Phase 1: Unsupervised pre-training of the FRE encoder-decoder VAE
           on random reward functions sampled from the prior distribution.
  Phase 2: Offline RL training of a z-conditioned IQL agent using the
           frozen FRE encoder to produce latent reward representations.

Usage:
    python scripts/train.py --config configs/antmaze.yaml
    python scripts/train.py --domain antmaze --task umaze --seed 0
    python scripts/train.py --config configs/kitchen.yaml --device cuda:0

The script supports both YAML configuration files and command-line overrides.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fre.utils import (
    load_config,
    merge_configs,
    set_seed,
    get_device,
    Logger,
    MetricTracker,
    make_env,
    save_json,
    format_time,
)
from fre.trainer import TwoPhaseTrainer, build_trainer
from fre.data_utils import load_dataset, compute_dataset_statistics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Functional Reward Encodings (FRE) for zero-shot offline RL."
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing configuration files (used if --config not specified).",
    )

    # Domain and task
    parser.add_argument(
        "--domain",
        type=str,
        default="antmaze",
        choices=["antmaze", "kitchen", "walker", "cheetah"],
        help="Domain/environment to train on.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Specific task within the domain (e.g., 'umaze', 'complete', 'proto').",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing offline datasets (for ExORL).",
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.")
    parser.add_argument("--phase1_steps", type=int, default=None, help="Number of FRE pre-training steps.")
    parser.add_argument("--phase2_steps", type=int, default=None, help="Number of RL training steps.")
    parser.add_argument("--rl_batch_size", type=int, default=None, help="Batch size for RL training.")
    parser.add_argument("--fre_batch_size", type=int, default=None, help="Batch size for FRE training.")
    parser.add_argument("--latent_dim", type=int, default=None, help="Latent dimension d_z.")
    parser.add_argument("--beta", type=float, default=None, help="KL divergence weight.")
    parser.add_argument("--expectile", type=float, default=None, help="IQL expectile parameter.")
    parser.add_argument("--temperature", type=float, default=None, help="IQL temperature parameter.")
    parser.add_argument("--discount", type=float, default=None, help="Discount factor.")

    # Logging and checkpoints
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for logs and checkpoints.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (auto-generated if not provided).")
    parser.add_argument("--log_interval", type=int, default=None, help="Steps between logging.")
    parser.add_argument("--eval_interval", type=int, default=None, help="Steps between evaluations.")
    parser.add_argument("--checkpoint_interval", type=int, default=None, help="Steps between checkpoints.")
    parser.add_argument("--use_tensorboard", action="store_true", default=True, help="Use TensorBoard logging.")
    parser.add_argument("--no_tensorboard", action="store_true", default=False, help="Disable TensorBoard logging.")
    parser.add_argument("--use_wandb", action="store_true", default=False, help="Use Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="fre", help="W&B project name.")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/username.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output.")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress verbose output.")

    # Resume training
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--resume_tag", type=str, default="latest", help="Checkpoint tag to resume from.")

    # Evaluation during training
    parser.add_argument("--eval_during_training", action="store_true", default=False,
                        help="Run zero-shot evaluation during training.")
    parser.add_argument("--eval_num_episodes", type=int, default=20, help="Number of evaluation episodes.")

    return parser.parse_args()


def build_config_from_args(args):
    """Build a configuration dictionary from command-line arguments."""
    # Start with default configuration
    config = get_default_config(args.domain, args.task)

    # Load YAML config if specified
    if args.config is not None:
        yaml_config = load_config(args.config)
        config = merge_configs(config, yaml_config)
    else:
        # Try to load domain-specific config from config_dir
        config_path = os.path.join(args.config_dir, f"{args.domain}.yaml")
        if os.path.exists(config_path):
            yaml_config = load_config(config_path)
            config = merge_configs(config, yaml_config)

    # Override with command-line arguments (non-None values)
    cli_overrides = {}
    for key, value in vars(args).items():
        if value is not None and key not in [
            "config", "config_dir", "log_dir", "exp_name", "resume", "resume_tag",
            "use_tensorboard", "no_tensorboard", "use_wandb", "wandb_project",
            "wandb_entity", "verbose", "quiet", "eval_during_training",
        ]:
            cli_overrides[key] = value

    config = merge_configs(config, cli_overrides)

    # Store CLI-specific settings
    config["_cli"] = {
        "log_dir": args.log_dir,
        "exp_name": args.exp_name,
        "use_tensorboard": args.use_tensorboard and not args.no_tensorboard,
        "use_wandb": args.use_wandb,
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "verbose": args.verbose and not args.quiet,
        "eval_during_training": args.eval_during_training,
        "eval_num_episodes": args.eval_num_episodes,
        "resume": args.resume,
        "resume_tag": args.resume_tag,
    }

    return config


def get_default_config(domain, task=None):
    """Return default configuration for a given domain."""
    # Base defaults common to all domains
    config = {
        "domain": domain,
        "task": task,
        "data_dir": None,
        "seed": 0,
        "device": "auto",

        # FRE model hyperparameters
        "latent_dim": 64,
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 1024,
        "d_emb": 64,
        "num_bins": 100,
        "reward_min": -10.0,
        "reward_max": 10.0,
        "decoder_hidden_dims": [256, 256],
        "beta": 0.1,
        "dropout": 0.0,
        "max_num_states": 32,

        # Reward prior hyperparameters
        "singleton_threshold": 0.5,
        "linear_sparsity": 0.5,
        "mlp_hidden_dim": 256,

        # IQL hyperparameters
        "iql_hidden_dims": [256, 256],
        "expectile": 0.7,
        "temperature": 3.0,
        "discount": 0.99,
        "soft_target_update_rate": 0.005,
        "log_std_min": -5.0,
        "log_std_max": 2.0,

        # Training hyperparameters
        "K_encoder": 32,
        "K_decoder": 32,
        "fre_learning_rate": 1e-4,
        "fre_weight_decay": 1e-5,
        "iql_learning_rate": 3e-4,
        "iql_weight_decay": 1e-4,
        "fre_steps": 100000,
        "rl_steps": 1000000,
        "rl_batch_size": 256,
        "fre_batch_size": 1,
        "log_interval": 1000,
        "eval_interval": 10000,
        "checkpoint_interval": 50000,
        "checkpoint_dir": "checkpoints",
        "use_amp": False,
    }

    # Domain-specific overrides
    if domain == "antmaze":
        config.update({
            "singleton_threshold": 1.0,
            "reward_min": -2.0,
            "reward_max": 2.0,
            "discount": 0.995,
            "rl_steps": 1000000,
        })
        if task is None:
            config["task"] = "umaze"
    elif domain == "kitchen":
        config.update({
            "singleton_threshold": 0.5,
            "reward_min": -1.0,
            "reward_max": 1.0,
            "discount": 0.99,
            "rl_steps": 500000,
            "rl_batch_size": 256,
        })
        if task is None:
            config["task"] = "complete"
    elif domain in ("walker", "cheetah"):
        config.update({
            "singleton_threshold": 1.0,
            "reward_min": -5.0,
            "reward_max": 5.0,
            "discount": 0.99,
            "rl_steps": 1000000,
            "rl_batch_size": 1024,
        })
        if task is None:
            config["task"] = "proto"

    return config


def setup_experiment(config):
    """Set up experiment directory, logging, and random seeds."""
    cli = config.get("_cli", {})

    # Create experiment name
    if cli.get("exp_name") is None:
        domain = config.get("domain", "unknown")
        task = config.get("task", "default")
        seed = config.get("seed", 0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        exp_name = f"fre_{domain}_{task}_seed{seed}_{timestamp}"
    else:
        exp_name = cli["exp_name"]

    # Create directories
    log_dir = os.path.join(cli.get("log_dir", "logs"), exp_name)
    checkpoint_dir = os.path.join(log_dir, config.get("checkpoint_dir", "checkpoints"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Update config with paths
    config["checkpoint_dir"] = checkpoint_dir
    config["_exp_name"] = exp_name
    config["_log_dir"] = log_dir

    # Set random seeds
    set_seed(config.get("seed", 0))

    # Initialize logger
    logger = Logger(
        log_dir=log_dir,
        use_tensorboard=cli.get("use_tensorboard", True),
        use_wandb=cli.get("use_wandb", False),
        wandb_project=cli.get("wandb_project", "fre"),
        wandb_entity=cli.get("wandb_entity", None),
        wandb_config=config,
        verbose=cli.get("verbose", True),
    )

    # Save configuration
    config_path = os.path.join(log_dir, "config.json")
    # Remove non-serializable CLI dict for saving
    save_config = {k: v for k, v in config.items() if not k.startswith("_")}
    save_json(save_config, config_path)
    logger.log_metrics({"config_saved": 1}, step=0)

    return logger, config


def main():
    """Main training entry point."""
    args = parse_args()
    config = build_config_from_args(args)

    # Setup experiment
    logger, config = setup_experiment(config)
    cli = config.get("_cli", {})

    print("=" * 80)
    print("Functional Reward Encodings (FRE) Training")
    print("=" * 80)
    print(f"Experiment: {config.get('_exp_name', 'unknown')}")
    print(f"Domain: {config['domain']}, Task: {config.get('task', 'default')}")
    print(f"Seed: {config['seed']}, Device: {config.get('device', 'auto')}")
    print(f"Log directory: {config.get('_log_dir', 'unknown')}")
    print("=" * 80)

    # Resolve device
    device = get_device(config.get("device", "auto"))
    config["device"] = str(device)
    print(f"Using device: {device}")

    # Build trainer
    print("\nBuilding trainer...")
    trainer = build_trainer(
        domain=config["domain"],
        task=config.get("task"),
        data_dir=config.get("data_dir"),
        seed=config["seed"],
        device=str(device),
        latent_dim=config.get("latent_dim", 64),
        d_model=config.get("d_model", 256),
        num_layers=config.get("num_layers", 2),
        num_heads=config.get("num_heads", 4),
        d_ff=config.get("d_ff", 1024),
        d_emb=config.get("d_emb", 64),
        num_bins=config.get("num_bins", 100),
        reward_min=config.get("reward_min", -10.0),
        reward_max=config.get("reward_max", 10.0),
        decoder_hidden_dims=config.get("decoder_hidden_dims", [256, 256]),
        beta=config.get("beta", 0.1),
        dropout=config.get("dropout", 0.0),
        max_num_states=config.get("max_num_states", 32),
        singleton_threshold=config.get("singleton_threshold", 0.5),
        linear_sparsity=config.get("linear_sparsity", 0.5),
        mlp_hidden_dim=config.get("mlp_hidden_dim", 256),
        iql_hidden_dims=config.get("iql_hidden_dims", [256, 256]),
        expectile=config.get("expectile", 0.7),
        temperature=config.get("temperature", 3.0),
        discount=config.get("discount", 0.99),
        soft_target_update_rate=config.get("soft_target_update_rate", 0.005),
        log_std_min=config.get("log_std_min", -5.0),
        log_std_max=config.get("log_std_max", 2.0),
        K_encoder=config.get("K_encoder", 32),
        K_decoder=config.get("K_decoder", 32),
        fre_learning_rate=config.get("fre_learning_rate", 1e-4),
        fre_weight_decay=config.get("fre_weight_decay", 1e-5),
        iql_learning_rate=config.get("iql_learning_rate", 3e-4),
        iql_weight_decay=config.get("iql_weight_decay", 1e-4),
        fre_steps=config.get("fre_steps", 100000),
        rl_steps=config.get("rl_steps", 1000000),
        rl_batch_size=config.get("rl_batch_size", 256),
        fre_batch_size=config.get("fre_batch_size", 1),
        log_interval=config.get("log_interval", 1000),
        eval_interval=config.get("eval_interval", 10000),
        checkpoint_interval=config.get("checkpoint_interval", 50000),
        checkpoint_dir=config.get("checkpoint_dir", "checkpoints"),
        use_amp=config.get("use_amp", False),
    )

    # Print model statistics
    stats = trainer.get_metrics_summary()
    print(f"\nModel statistics:")
    print(f"  State dim: {stats.get('state_dim', '?')}")
    print(f"  Action dim: {stats.get('action_dim', '?')}")
    print(f"  Dataset size: {stats.get('dataset_size', '?')}")
    print(f"  FRE encoder params: {stats.get('fre_encoder_params', '?'):,}")
    print(f"  FRE decoder params: {stats.get('fre_decoder_params', '?'):,}")
    print(f"  IQL total params: {stats.get('iql_total_params', '?'):,}")

    # Resume from checkpoint if specified
    if cli.get("resume") is not None:
        print(f"\nResuming from checkpoint: {cli['resume']}")
        trainer.load_checkpoint(cli.get("resume_tag", "latest"))

    # ============================================================
    # Phase 1: FRE Encoder Pre-training
    # ============================================================
    phase1_steps = config.get("fre_steps", 100000)
    if phase1_steps > 0:
        print("\n" + "=" * 80)
        print("PHASE 1: FRE Encoder Pre-training")
        print("=" * 80)
        print(f"Steps: {phase1_steps}")
        print(f"Batch size: {config.get('fre_batch_size', 1)}")
        print(f"Learning rate: {config.get('fre_learning_rate', 1e-4)}")
        print(f"Beta (KL weight): {config.get('beta', 0.1)}")
        print("-" * 80)

        phase1_start = time.time()
        trainer.run_phase1(
            steps=phase1_steps,
            verbose=cli.get("verbose", True),
        )
        phase1_duration = time.time() - phase1_start
        print(f"\nPhase 1 completed in {format_time(phase1_duration)}")

        # Save checkpoint after Phase 1
        trainer.save_checkpoint("phase1_complete")

    # ============================================================
    # Phase 2: Offline RL Training
    # ============================================================
    phase2_steps = config.get("rl_steps", 1000000)
    if phase2_steps > 0:
        print("\n" + "=" * 80)
        print("PHASE 2: Offline RL Training with Frozen Encoder")
        print("=" * 80)
        print(f"Steps: {phase2_steps}")
        print(f"Batch size: {config.get('rl_batch_size', 256)}")
        print(f"Learning rate: {config.get('iql_learning_rate', 3e-4)}")
        print(f"Expectile: {config.get('expectile', 0.7)}")
        print(f"Temperature: {config.get('temperature', 3.0)}")
        print(f"Discount: {config.get('discount', 0.99)}")
        print("-" * 80)

        phase2_start = time.time()
        trainer.run_phase2(
            steps=phase2_steps,
            verbose=cli.get("verbose", True),
        )
        phase2_duration = time.time() - phase2_start
        print(f"\nPhase 2 completed in {format_time(phase2_duration)}")

        # Save final checkpoint
        trainer.save_checkpoint("final")

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    metrics = trainer.get_metrics_summary()
    print(f"\nFinal metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save metrics summary
    metrics_path = os.path.join(config["_log_dir"], "final_metrics.json")
    save_json(metrics, metrics_path)
    print(f"\nMetrics saved to: {metrics_path}")

    # Close logger
    logger.close()

    print("\nDone!")


if __name__ == "__main__":
    main()