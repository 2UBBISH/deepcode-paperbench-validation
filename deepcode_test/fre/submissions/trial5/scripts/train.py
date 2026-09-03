#!/usr/bin/env python3
"""
Main training script for Functional Reward Encodings (FRE).

This script orchestrates the full training pipeline:
  1. Loads configuration (default + domain-specific overrides).
  2. Loads the offline dataset and creates a replay buffer.
  3. Builds the FRE trainer (encoder, decoder, IQL agent, reward distribution).
  4. Runs Phase 1 (encoder/decoder training) and Phase 2 (IQL agent training),
     optionally with strided interleaving.
  5. Saves checkpoints and logs metrics.

Usage:
    python scripts/train.py --config configs/default.yaml --domain antmaze
    python scripts/train.py --config configs/default.yaml --domain exorl --dataset walker
    python scripts/train.py --config configs/default.yaml --domain kitchen
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (
    OfflineDataset,
    ReplayBuffer,
    load_dataset,
    create_replay_buffer,
)
from models import FREEncoder, FREDecoder, IQLAgent
from rewards import MixtureRewardDistribution
from training import FRETrainer, FREEncoderTrainer, IQLTrainer
from utils import (
    set_seed,
    get_device,
    configure_logging,
    Logger,
    WandbLogger,
    to_tensor,
    to_numpy,
)
from evaluation import (
    FREEvaluator,
    build_tasks_for_domain,
    EvaluationResult,
    compute_normalized_score,
    get_domain_normalization,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train FRE (Functional Reward Encodings) for zero-shot offline RL."
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to base YAML configuration file.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=["antmaze", "exorl", "kitchen", None],
        help="Domain to train on (loads domain-specific config override).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name override (e.g., 'antmaze-large-diverse-v2', 'walker', 'cheetah').",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to ExORL data directory (if applicable).",
    )

    # Training control
    parser.add_argument(
        "--encoder_steps",
        type=int,
        default=None,
        help="Override number of Phase 1 (encoder) training steps.",
    )
    parser.add_argument(
        "--rl_steps",
        type=int,
        default=None,
        help="Override number of Phase 2 (RL) training steps.",
    )
    parser.add_argument(
        "--strided",
        action="store_true",
        default=None,
        help="Enable strided training (alternate encoder and RL updates).",
    )
    parser.add_argument(
        "--no_strided",
        action="store_true",
        default=None,
        help="Disable strided training (sequential Phase 1 -> Phase 2).",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=None,
        help="Override checkpoint save interval (steps).",
    )

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for logs.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=None,
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        default=None,
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team).",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="W&B run name.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Enable verbose logging.",
    )

    # Evaluation during training
    parser.add_argument(
        "--eval_during_training",
        action="store_true",
        default=None,
        help="Run zero-shot evaluation periodically during training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="Evaluation interval in training steps.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes per task.",
    )

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device override.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of torch threads.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        config = {}
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override into base. Override values take precedence.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build the final configuration by merging:
      1. default.yaml
      2. domain-specific config (antmaze.yaml, exorl.yaml, kitchen.yaml)
      3. command-line arguments
    """
    # Load base config
    base_config_path = args.config
    if not os.path.isabs(base_config_path):
        # Resolve relative to project root
        base_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            base_config_path,
        )
    config = load_yaml(base_config_path)

    # Load domain-specific config
    if args.domain:
        domain_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs",
            f"{args.domain}.yaml",
        )
        if os.path.exists(domain_config_path):
            domain_config = load_yaml(domain_config_path)
            config = merge_configs(config, domain_config)
            logging.info(f"Loaded domain config: {domain_config_path}")
        else:
            logging.warning(
                f"Domain config not found: {domain_config_path}. Using base config only."
            )

    # Apply command-line overrides
    cli_overrides = _build_cli_overrides(args)
    config = merge_configs(config, cli_overrides)

    return config


def _build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert CLI arguments to a nested config dict for merging."""
    overrides: Dict[str, Any] = {}

    # General
    if args.seed is not None:
        overrides.setdefault("general", {})["seed"] = args.seed
    if args.device is not None:
        overrides.setdefault("general", {})["device"] = args.device
    if args.log_dir is not None:
        overrides.setdefault("general", {})["log_dir"] = args.log_dir
    if args.checkpoint_dir is not None:
        overrides.setdefault("general", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.use_wandb is not None:
        overrides.setdefault("general", {})["use_wandb"] = True
    if args.no_wandb is not None:
        overrides.setdefault("general", {})["use_wandb"] = False
    if args.wandb_project is not None:
        overrides.setdefault("general", {})["wandb_project"] = args.wandb_project
    if args.wandb_entity is not None:
        overrides.setdefault("general", {})["wandb_entity"] = args.wandb_entity
    if args.wandb_name is not None:
        overrides.setdefault("general", {})["wandb_name"] = args.wandb_name
    if args.verbose is not None:
        overrides.setdefault("general", {})["verbose"] = True

    # Dataset
    if args.dataset is not None:
        overrides.setdefault("dataset", {})["name"] = args.dataset
    if args.data_path is not None:
        overrides.setdefault("dataset", {})["exorl_data_path"] = args.data_path

    # Encoder training
    if args.encoder_steps is not None:
        overrides.setdefault("encoder_training", {})["num_steps"] = args.encoder_steps
    if args.save_interval is not None:
        overrides.setdefault("encoder_training", {})["save_interval"] = args.save_interval

    # RL training
    if args.rl_steps is not None:
        overrides.setdefault("rl_training", {})["num_steps"] = args.rl_steps
    if args.save_interval is not None:
        overrides.setdefault("rl_training", {})["save_interval"] = args.save_interval

    # Strided training
    if args.strided is not None:
        overrides.setdefault("strided_training", {})["enabled"] = True
    if args.no_strided is not None:
        overrides.setdefault("strided_training", {})["enabled"] = False

    # Evaluation
    if args.eval_interval is not None:
        overrides.setdefault("evaluation", {})["eval_interval"] = args.eval_interval
    if args.eval_episodes is not None:
        overrides.setdefault("evaluation", {})["num_episodes"] = args.eval_episodes

    return overrides


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def resolve_dataset_name(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    """Determine the dataset name from args, config, or domain."""
    if args.dataset:
        return args.dataset
    if config.get("dataset", {}).get("name"):
        return config["dataset"]["name"]
    # Infer from domain
    domain = args.domain
    if domain == "antmaze":
        return "antmaze-large-diverse-v2"
    elif domain == "exorl":
        return "walker"  # default; user should specify
    elif domain == "kitchen":
        return "kitchen-complete-v0"
    else:
        raise ValueError(
            "Could not determine dataset name. Specify --dataset or --domain."
        )


def load_replay_buffer(
    config: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
) -> ReplayBuffer:
    """Load the offline dataset and wrap in a replay buffer."""
    dataset_cfg = config.get("dataset", {})
    normalize_states = dataset_cfg.get("normalize_states", True)
    clip_to_eps = dataset_cfg.get("clip_to_eps", True)
    exorl_data_path = dataset_cfg.get("exorl_data_path", None)

    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(
        dataset_name=dataset_name,
        normalize_states=normalize_states,
        data_path=exorl_data_path,
    )

    logging.info(
        f"Dataset loaded: {len(dataset)} transitions, "
        f"state_dim={dataset.state_dim}, action_dim={dataset.action_dim}"
    )

    replay_buffer = create_replay_buffer(
        dataset=dataset,
        device=device,
        normalize_states=normalize_states,
    )

    return replay_buffer


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def run_training(
    config: Dict[str, Any],
    replay_buffer: ReplayBuffer,
    device: torch.device,
    logger: Logger,
    args: argparse.Namespace,
) -> FRETrainer:
    """
    Build the FRE trainer and run the full training pipeline.
    """
    general_cfg = config.get("general", {})
    encoder_cfg = config.get("encoder", {})
    decoder_cfg = config.get("decoder", {})
    agent_cfg = config.get("agent", {})
    reward_cfg = config.get("reward", {})
    enc_train_cfg = config.get("encoder_training", {})
    rl_train_cfg = config.get("rl_training", {})
    strided_cfg = config.get("strided_training", {})
    eval_cfg = config.get("evaluation", {})

    # Infer state/action dims
    state_dim = replay_buffer.dataset.state_dim
    action_dim = replay_buffer.dataset.action_dim

    logging.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Build the trainer
    trainer = FRETrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
        config=config,
        device=device,
    )

    # Resume from checkpoint if specified
    start_step_enc = 0
    start_step_rl = 0
    if args.resume:
        logging.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        # Try to recover step counts
        if hasattr(trainer.encoder_trainer, "global_step"):
            start_step_enc = trainer.encoder_trainer.global_step
        if hasattr(trainer.rl_trainer, "global_step"):
            start_step_rl = trainer.rl_trainer.global_step

    # Determine training regime
    use_strided = strided_cfg.get("enabled", False)

    if use_strided:
        logging.info("Using strided training (alternating encoder and RL updates).")
        encoder_steps_per_stride = strided_cfg.get("encoder_steps_per_stride", 1000)
        rl_steps_per_stride = strided_cfg.get("rl_steps_per_stride", 10000)
        total_strides = strided_cfg.get("total_strides", 100)

        trainer.train_strided(
            encoder_steps_per_stride=encoder_steps_per_stride,
            rl_steps_per_stride=rl_steps_per_stride,
            total_strides=total_strides,
            logger=logger,
            eval_callback=(
                _make_eval_callback(config, trainer, args, logger)
                if args.eval_during_training
                else None
            ),
            eval_interval=args.eval_interval or eval_cfg.get("eval_interval", 50000),
        )
    else:
        # Sequential: Phase 1 then Phase 2
        enc_steps = enc_train_cfg.get("num_steps", 100000)
        rl_steps = rl_train_cfg.get("num_steps", 1000000)

        logging.info(f"Phase 1: Training encoder for {enc_steps} steps.")
        trainer.train_encoder_phase(
            num_steps=enc_steps,
            logger=logger,
            log_interval=enc_train_cfg.get("log_interval", 1000),
            eval_interval=enc_train_cfg.get("eval_interval", 10000),
            save_interval=enc_train_cfg.get("save_interval", 50000),
            checkpoint_dir=general_cfg.get("checkpoint_dir", "checkpoints"),
        )

        logging.info(f"Phase 2: Training IQL agent for {rl_steps} steps.")
        trainer.train_rl_phase(
            num_steps=rl_steps,
            logger=logger,
            log_interval=rl_train_cfg.get("log_interval", 1000),
            eval_interval=rl_train_cfg.get("eval_interval", 10000),
            save_interval=rl_train_cfg.get("save_interval", 50000),
            checkpoint_dir=general_cfg.get("checkpoint_dir", "checkpoints"),
        )

    # Final save
    final_checkpoint_dir = general_cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    final_path = os.path.join(final_checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_path)
    logging.info(f"Final checkpoint saved to: {final_path}")

    return trainer


def _make_eval_callback(config, trainer, args, logger):
    """Create an evaluation callback for periodic zero-shot evaluation."""
    eval_cfg = config.get("evaluation", {})

    def eval_callback(step: int):
        logging.info(f"Running zero-shot evaluation at step {step}...")
        try:
            from evaluation import evaluate_from_trainer

            domain = args.domain or "antmaze"
            env_name = resolve_dataset_name(args, config)

            result, norm_stats = evaluate_from_trainer(
                trainer=trainer,
                domain=domain,
                env_name=env_name,
                replay_buffer=trainer.replay_buffer,
                num_episodes=eval_cfg.get("num_episodes", 20),
                K_enc=eval_cfg.get("K_enc", 32),
                deterministic=eval_cfg.get("deterministic_policy", True),
                seed=config.get("general", {}).get("seed", 0),
            )

            # Log results
            for task_name, (mean_norm, std_norm) in norm_stats.items():
                logger.log_metric(f"eval/{task_name}_norm", mean_norm, step)

            overall = result.get_overall_average()
            logger.log_metric("eval/overall_mean", overall[0], step)
            logger.log_metric("eval/overall_std", overall[1], step)

            logging.info(
                f"Step {step} evaluation - Overall: {overall[0]:.1f} ± {overall[1]:.1f}"
            )
        except Exception as e:
            logging.warning(f"Evaluation at step {step} failed: {e}")

    return eval_callback


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Build configuration
    config = build_config(args)

    # Set random seed
    seed = config.get("general", {}).get("seed", 0)
    set_seed(seed)

    # Determine device
    device_str = config.get("general", {}).get("device", "auto")
    if device_str == "auto":
        device = get_device(use_cuda=torch.cuda.is_available())
    elif device_str == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    # Set torch threads
    num_threads = args.num_threads or config.get("general", {}).get("num_threads", None)
    if num_threads is not None:
        torch.set_num_threads(num_threads)

    # Set up logging
    log_dir = config.get("general", {}).get("log_dir", "logs")
    verbose = config.get("general", {}).get("verbose", True)
    os.makedirs(log_dir, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO
    configure_logging(log_dir=log_dir, level=log_level, name="fre_train")

    # Create logger
    use_wandb = config.get("general", {}).get("use_wandb", False)
    if use_wandb:
        logger = WandbLogger(
            log_dir=log_dir,
            window_size=100,
            verbose=verbose,
            use_wandb=True,
            project=config.get("general", {}).get("wandb_project", "fre"),
            entity=config.get("general", {}).get("wandb_entity", None),
            name=config.get("general", {}).get("wandb_name", None),
            config=config,
        )
    else:
        logger = Logger(
            log_dir=log_dir,
            window_size=100,
            verbose=verbose,
        )

    # Log config
    logger.log_info(f"Configuration: {config}")

    # Resolve dataset name
    dataset_name = resolve_dataset_name(args, config)
    logger.log_info(f"Dataset: {dataset_name}")

    # Load replay buffer
    replay_buffer = load_replay_buffer(config, dataset_name, device)

    # Run training
    start_time = time.time()
    trainer = run_training(config, replay_buffer, device, logger, args)
    elapsed = time.time() - start_time

    logger.log_info(f"Training completed in {elapsed:.1f} seconds ({elapsed/3600:.2f} hours).")

    # Optional final evaluation
    if args.eval_during_training:
        logger.log_info("Running final evaluation...")
        try:
            from evaluation import evaluate_from_trainer

            domain = args.domain or "antmaze"
            env_name = resolve_dataset_name(args, config)

            result, norm_stats = evaluate_from_trainer(
                trainer=trainer,
                domain=domain,
                env_name=env_name,
                replay_buffer=replay_buffer,
                num_episodes=config.get("evaluation", {}).get("num_episodes", 20),
                K_enc=config.get("evaluation", {}).get("K_enc", 32),
                deterministic=config.get("evaluation", {}).get("deterministic_policy", True),
                seed=seed,
            )

            # Print results
            logger.log_info("=" * 60)
            logger.log_info("FINAL EVALUATION RESULTS")
            logger.log_info("=" * 60)
            for task_name, (mean_norm, std_norm) in norm_stats.items():
                logger.log_info(f"  {task_name}: {mean_norm:.1f} ± {std_norm:.1f}")
            overall = result.get_overall_average()
            logger.log_info(f"  OVERALL: {overall[0]:.1f} ± {overall[1]:.1f}")
            logger.log_info("=" * 60)

            # Save results
            results_path = os.path.join(log_dir, "final_results.json")
            import json

            results_dict = {
                "config": {k: str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v
                           for k, v in config.items()},
                "normalized_stats": {k: list(v) for k, v in norm_stats.items()},
                "overall": list(overall),
                "raw_results": result.to_dict(),
            }
            with open(results_path, "w") as f:
                json.dump(results_dict, f, indent=2, default=str)
            logger.log_info(f"Results saved to: {results_path}")

        except Exception as e:
            logger.log_error(f"Final evaluation failed: {e}")

    # Clean up
    logger.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()