#!/usr/bin/env python3
"""
Zero-shot evaluation script for Functional Reward Encodings (FRE).

Loads a trained FRE model (encoder + IQL agent) from a checkpoint and evaluates
it on downstream tasks for a given domain (AntMaze, ExORL, Kitchen). Reports
normalized returns as in the paper (Table 1).

Usage:
    python scripts/evaluate.py --config configs/antmaze.yaml --checkpoint checkpoints/antmaze_final.pt --domain antmaze
    python scripts/evaluate.py --config configs/exorl.yaml --checkpoint checkpoints/exorl_walker.pt --domain exorl_walker
    python scripts/evaluate.py --config configs/kitchen.yaml --checkpoint checkpoints/kitchen.pt --domain kitchen

For multi-seed evaluation:
    python scripts/evaluate.py --config configs/antmaze.yaml --checkpoint_dir checkpoints/ --domain antmaze --multi_seed
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import OfflineDataset, ReplayBuffer, load_dataset, create_replay_buffer
from evaluation import (
    FREEvaluator,
    EvaluationResult,
    EvaluationTask,
    build_antmaze_tasks,
    build_exorl_walker_tasks,
    build_exorl_cheetah_tasks,
    build_kitchen_tasks,
    build_tasks_for_domain,
    run_multi_seed_evaluation,
    evaluate_from_trainer,
    compute_normalized_score,
    get_domain_normalization,
    DOMAIN_NORMALIZATION,
)
from models import FREEncoder, FREDecoder, IQLAgent
from training import FRETrainer
from utils import set_seed, get_device, configure_logging, to_tensor, to_numpy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation for FRE agents"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a single checkpoint file (.pt)"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Directory containing multiple seed checkpoints for multi-seed evaluation"
    )
    parser.add_argument(
        "--domain", type=str, required=True,
        choices=["antmaze", "exorl_walker", "exorl_cheetah", "kitchen"],
        help="Evaluation domain"
    )
    parser.add_argument(
        "--env_name", type=str, default=None,
        help="Gym environment name (overrides config if provided)"
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to ExORL dataset files (overrides config if provided)"
    )
    parser.add_argument(
        "--multi_seed", action="store_true",
        help="Run multi-seed evaluation using checkpoints from --checkpoint_dir"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=20,
        help="Number of evaluation episodes per task (default: 20)"
    )
    parser.add_argument(
        "--K_enc", type=int, default=32,
        help="Number of encoding states for reward function (default: 32)"
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic policy for evaluation (default: True)"
    )
    parser.add_argument(
        "--stochastic", dest="deterministic", action="store_false",
        help="Use stochastic policy for evaluation"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for evaluation (default: 0)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run evaluation on (default: auto)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save evaluation results (JSON format)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed per-episode results"
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Recursively merge override_config into base_config."""
    merged = base_config.copy()
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_model_from_checkpoint(
    checkpoint: Dict,
    state_dim: int,
    action_dim: int,
    config: Dict,
    device: torch.device,
) -> Tuple[FREEncoder, FREDecoder, IQLAgent]:
    """
    Reconstruct encoder, decoder, and IQL agent from a checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary.
        state_dim: State dimension.
        action_dim: Action dimension.
        config: Configuration dictionary.
        device: Torch device.

    Returns:
        Tuple of (encoder, decoder, agent).
    """
    enc_cfg = config.get("encoder", {})
    dec_cfg = config.get("decoder", {})
    agent_cfg = config.get("agent", {})

    # Build encoder
    encoder = FREEncoder(
        state_dim=state_dim,
        embed_dim=enc_cfg.get("embed_dim", 256),
        latent_dim=enc_cfg.get("latent_dim", 64),
        num_layers=enc_cfg.get("num_layers", 3),
        num_heads=enc_cfg.get("num_heads", 4),
        dropout=enc_cfg.get("dropout", 0.1),
        num_bins=enc_cfg.get("num_bins", 64),
        reward_min=enc_cfg.get("reward_min", None),
        reward_max=enc_cfg.get("reward_max", None),
    )

    # Build decoder
    decoder = FREDecoder(
        state_dim=state_dim,
        latent_dim=enc_cfg.get("latent_dim", 64),
        hidden_dims=dec_cfg.get("hidden_dims", [256, 256]),
    )

    # Build IQL agent
    agent = IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=enc_cfg.get("latent_dim", 64),
        hidden_dims=agent_cfg.get("hidden_dims", [256, 256]),
        expectile=agent_cfg.get("expectile", 0.7),
        temperature=agent_cfg.get("temperature", 3.0),
        discount=agent_cfg.get("discount", 0.99),
        target_tau=agent_cfg.get("target_tau", 0.005),
        log_std_min=agent_cfg.get("log_std_min", -5.0),
        log_std_max=agent_cfg.get("log_std_max", 2.0),
    )

    # Load state dicts
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if "decoder_state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])

    encoder.to(device)
    decoder.to(device)
    agent.to(device)

    encoder.eval()
    decoder.eval()
    agent.eval()

    # Freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False
    for param in agent.parameters():
        param.requires_grad = False

    return encoder, decoder, agent


def get_env_name(domain: str, config: Dict, args_env_name: Optional[str] = None) -> str:
    """Determine the environment name for a given domain."""
    if args_env_name is not None:
        return args_env_name

    domain_to_env = {
        "antmaze": "antmaze-large-diverse-v2",
        "exorl_walker": "walker",  # ExORL walker
        "exorl_cheetah": "cheetah",  # ExORL cheetah
        "kitchen": "kitchen-complete-v0",
    }
    return domain_to_env.get(domain, config.get("dataset", {}).get("name", ""))


def get_state_action_dims(
    domain: str, env_name: str, replay_buffer: ReplayBuffer
) -> Tuple[int, int]:
    """Infer state and action dimensions from replay buffer."""
    sample = replay_buffer.sample(1)
    state_dim = sample["states"].shape[-1]
    action_dim = sample["actions"].shape[-1]
    return state_dim, action_dim


def evaluate_single_checkpoint(
    checkpoint_path: str,
    domain: str,
    env_name: str,
    replay_buffer: ReplayBuffer,
    config: Dict,
    device: torch.device,
    num_episodes: int = 20,
    K_enc: int = 32,
    deterministic: bool = True,
    seed: int = 0,
    verbose: bool = False,
) -> Tuple[EvaluationResult, Dict[str, Tuple[float, float]]]:
    """
    Evaluate a single checkpoint on all tasks for a domain.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        domain: Domain name.
        env_name: Gym environment name.
        replay_buffer: Replay buffer for encoding states.
        config: Configuration dictionary.
        device: Torch device.
        num_episodes: Number of episodes per task.
        K_enc: Number of encoding states.
        deterministic: Whether to use deterministic policy.
        seed: Random seed.
        verbose: Print detailed results.

    Returns:
        Tuple of (EvaluationResult, normalized_stats_dict).
    """
    set_seed(seed)

    # Load checkpoint
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Infer dimensions
    state_dim, action_dim = get_state_action_dims(domain, env_name, replay_buffer)

    # Build models
    encoder, decoder, agent = build_model_from_checkpoint(
        checkpoint, state_dim, action_dim, config, device
    )

    # Build evaluator
    evaluator = FREEvaluator(
        encoder=encoder,
        agent=agent,
        replay_buffer=replay_buffer,
        device=device,
        K_enc=K_enc,
        deterministic_policy=deterministic,
    )

    # Build tasks
    rng = np.random.RandomState(seed)
    tasks = build_tasks_for_domain(domain, state_dim=state_dim, rng=rng)

    logging.info(f"Evaluating {len(tasks)} tasks for domain '{domain}'")
    logging.info(f"  Environment: {env_name}")
    logging.info(f"  Episodes per task: {num_episodes}")
    logging.info(f"  Encoding states: {K_enc}")
    logging.info(f"  Policy: {'deterministic' if deterministic else 'stochastic'}")

    # Run evaluation
    min_return, max_return = get_domain_normalization(domain)
    result, normalized_stats = evaluator.evaluate_with_normalization(
        tasks=tasks,
        env_factory=env_name,  # evaluator handles gym.make
        domain=domain,
        num_episodes=num_episodes,
        seed=seed,
        verbose=verbose,
    )

    return result, normalized_stats


def print_results(
    result: EvaluationResult,
    normalized_stats: Dict[str, Tuple[float, float]],
    domain: str,
) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "=" * 70)
    print(f"  FRE Zero-Shot Evaluation Results — {domain}")
    print("=" * 70)

    all_task_stats = result.get_all_task_stats()
    overall_mean, overall_std = result.get_overall_average()

    print(f"\n{'Task':<35} {'Raw Return':>15} {'Normalized':>15}")
    print("-" * 65)

    for task_name in sorted(all_task_stats.keys()):
        raw_mean, raw_std = all_task_stats[task_name]
        norm_mean, norm_std = normalized_stats.get(
            task_name, (float("nan"), float("nan"))
        )
        print(
            f"{task_name:<35} {raw_mean:>8.2f} ± {raw_std:<5.2f} "
            f"{norm_mean:>8.1f} ± {norm_std:<5.1f}"
        )

    print("-" * 65)
    print(f"{'OVERALL AVERAGE':<35} {'':>15} {overall_mean:>8.1f} ± {overall_std:<5.1f}")
    print("=" * 70 + "\n")


def save_results(
    result: EvaluationResult,
    normalized_stats: Dict[str, Tuple[float, float]],
    output_path: str,
    domain: str,
    config: Dict,
) -> None:
    """Save evaluation results to a JSON file."""
    import json

    output = {
        "domain": domain,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_mean": result.get_overall_average()[0],
        "overall_std": result.get_overall_average()[1],
        "tasks": {},
    }

    all_task_stats = result.get_all_task_stats()
    for task_name in sorted(all_task_stats.keys()):
        raw_mean, raw_std = all_task_stats[task_name]
        norm_mean, norm_std = normalized_stats.get(
            task_name, (float("nan"), float("nan"))
        )
        output["tasks"][task_name] = {
            "raw_mean": float(raw_mean),
            "raw_std": float(raw_std),
            "normalized_mean": float(norm_mean),
            "normalized_std": float(norm_std),
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Results saved to {output_path}")


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command-line arguments
    if args.data_path:
        if "dataset" not in config:
            config["dataset"] = {}
        config["dataset"]["exorl_data_path"] = args.data_path

    if args.num_episodes != 20:
        if "evaluation" not in config:
            config["evaluation"] = {}
        config["evaluation"]["num_episodes"] = args.num_episodes

    if args.K_enc != 32:
        if "evaluation" not in config:
            config["evaluation"] = {}
        config["evaluation"]["K_enc"] = args.K_enc

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = configure_logging(level=log_level, name="fre_evaluate")

    # Determine device
    if args.device == "auto":
        device = get_device(use_cuda=torch.cuda.is_available())
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Determine environment name
    env_name = get_env_name(args.domain, config, args.env_name)
    logging.info(f"Domain: {args.domain}, Environment: {env_name}")

    # Load dataset and create replay buffer
    dataset_name = config.get("dataset", {}).get("name", env_name)
    normalize_states = config.get("dataset", {}).get("normalize_states", True)
    data_path = config.get("dataset", {}).get("exorl_data_path", None) or args.data_path

    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(
        dataset_name,
        normalize_states=normalize_states,
        data_path=data_path,
    )

    replay_buffer = create_replay_buffer(
        dataset=dataset,
        device=device,
        normalize_states=normalize_states,
    )
    logging.info(
        f"Dataset loaded: {len(dataset)} transitions, "
        f"state_dim={dataset.states.shape[-1]}, action_dim={dataset.actions.shape[-1]}"
    )

    # Run evaluation
    if args.multi_seed and args.checkpoint_dir:
        logging.info(f"Running multi-seed evaluation from {args.checkpoint_dir}")
        result, normalized_stats = run_multi_seed_evaluation(
            checkpoint_paths=None,  # Will be auto-discovered
            domain=args.domain,
            env_name=env_name,
            replay_buffer=replay_buffer,
            config=config,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            num_episodes=args.num_episodes,
            K_enc=args.K_enc,
            deterministic=args.deterministic,
            verbose=args.verbose,
        )
    elif args.checkpoint:
        logging.info(f"Running single-checkpoint evaluation from {args.checkpoint}")
        result, normalized_stats = evaluate_single_checkpoint(
            checkpoint_path=args.checkpoint,
            domain=args.domain,
            env_name=env_name,
            replay_buffer=replay_buffer,
            config=config,
            device=device,
            num_episodes=args.num_episodes,
            K_enc=args.K_enc,
            deterministic=args.deterministic,
            seed=args.seed,
            verbose=args.verbose,
        )
    else:
        logging.error(
            "Either --checkpoint or --checkpoint_dir (with --multi_seed) must be provided."
        )
        sys.exit(1)

    # Print results
    print_results(result, normalized_stats, args.domain)

    # Save results if requested
    if args.output:
        save_results(result, normalized_stats, args.output, args.domain, config)

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()