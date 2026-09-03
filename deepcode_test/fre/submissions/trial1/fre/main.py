#!/usr/bin/env python3
"""
Functional Reward Encodings (FRE) - Main Entry Point

Orchestrates the full training and evaluation pipeline:
  - Phase 1: Train FRE encoder+decoder (VAE) on random reward functions
  - Phase 2: Train IQL agent with frozen encoder
  - Evaluation: Zero-shot evaluation on downstream tasks

Usage:
    # Train encoder only
    python -m fre.main --mode train_encoder --domain antmaze

    # Train IQL only (requires pretrained encoder)
    python -m fre.main --mode train_iql --domain antmaze --encoder_path checkpoints/encoder_final.pt

    # Full strided training (Phase 1 + Phase 2)
    python -m fre.main --mode train_all --domain antmaze

    # Evaluate a trained agent
    python -m fre.main --mode evaluate --domain antmaze \
        --encoder_path checkpoints/encoder_final.pt \
        --iql_path checkpoints/iql_final.pt

    # Evaluate with multiple seeds
    python -m fre.main --mode evaluate_multi --domain antmaze \
        --encoder_path checkpoints/encoder_final.pt \
        --iql_path checkpoints/iql_final.pt --seeds 0 1 2 3 4
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch

from fre.config import Config, config as default_config
from fre.data.dataset import load_dataset
from fre.models.encoder import FREEncoder
from fre.models.decoder import RewardDecoder
from fre.models.iql import IQLNetworks
from fre.training.train_encoder import train_encoder, load_pretrained_encoder
from fre.training.train_iql import train_iql, load_pretrained_iql
from fre.training.utils import set_seed, get_device, print_model_info, count_parameters
from fre.evaluation.evaluate import run_evaluation, evaluate_multiple_seeds


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Functional Reward Encodings (FRE) for Zero-Shot Offline RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train encoder only
  python -m fre.main --mode train_encoder --domain antmaze

  # Full strided training
  python -m fre.main --mode train_all --domain antmaze

  # Evaluate
  python -m fre.main --mode evaluate --domain antmaze \\
      --encoder_path checkpoints/encoder_final.pt \\
      --iql_path checkpoints/iql_final.pt
        """
    )

    # Main mode
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train_encoder", "train_iql", "train_all", "evaluate", "evaluate_multi"],
        help="Operation mode: train_encoder (Phase 1), train_iql (Phase 2), "
             "train_all (strided Phase 1+2), evaluate (single seed), "
             "evaluate_multi (multiple seeds)"
    )

    # Domain
    parser.add_argument(
        "--domain", type=str, default="antmaze",
        choices=["antmaze", "exorl_walker", "exorl_cheetah", "kitchen"],
        help="Domain/environment to use (default: antmaze)"
    )

    # Paths
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory for offline datasets (default: use D4RL/ExORL defaults)"
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Directory for TensorBoard logs (default: logs/<domain>/<timestamp>)"
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory for model checkpoints (default: checkpoints/<domain>/<timestamp>)"
    )
    parser.add_argument(
        "--encoder_path", type=str, default=None,
        help="Path to pretrained encoder checkpoint (for train_iql, evaluate)"
    )
    parser.add_argument(
        "--iql_path", type=str, default=None,
        help="Path to pretrained IQL checkpoint (for evaluate)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for evaluation results JSON (default: results/<domain>)"
    )

    # Device and seed
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda', 'cpu', or 'cuda:0' (default: auto-detect)"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="List of random seeds for multi-seed evaluation (default: 0 1 2 3 4)"
    )

    # Training hyperparameters (override config defaults)
    parser.add_argument(
        "--encoder_steps", type=int, default=None,
        help="Number of encoder training steps (default: from config)"
    )
    parser.add_argument(
        "--iql_steps", type=int, default=None,
        help="Number of IQL training steps (default: from config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size for training (default: from config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--K", type=int, default=None,
        help="Number of encoder states K (default: from config)"
    )
    parser.add_argument(
        "--K_prime", type=int, default=None,
        help="Number of decoder states K' (default: from config)"
    )
    parser.add_argument(
        "--beta_kl", type=float, default=None,
        help="KL divergence weight for VAE (default: from config)"
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Expectile for IQL value loss (default: from config)"
    )
    parser.add_argument(
        "--beta_iql", type=float, default=None,
        help="Temperature for IQL policy AWR (default: from config)"
    )

    # Evaluation hyperparameters
    parser.add_argument(
        "--K_eval", type=int, default=None,
        help="Number of states for evaluation encoding (default: from config)"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=None,
        help="Number of evaluation episodes per task (default: from config)"
    )
    parser.add_argument(
        "--max_episode_steps", type=int, default=None,
        help="Maximum steps per evaluation episode (default: from config)"
    )

    # Misc
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print verbose output (default: True)"
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False,
        help="Disable CUDA even if available"
    )

    return parser.parse_args()


def build_config_from_args(args):
    """
    Build a Config instance from command-line arguments.
    Override defaults with any non-None argument values.
    """
    cfg = Config()

    # Override with command-line arguments
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    elif args.no_cuda:
        overrides["device"] = "cpu"

    if args.encoder_steps is not None:
        overrides["encoder_steps"] = args.encoder_steps
    if args.iql_steps is not None:
        overrides["iql_steps"] = args.iql_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides["lr"] = args.lr
    if args.K is not None:
        overrides["K"] = args.K
    if args.K_prime is not None:
        overrides["K_prime"] = args.K_prime
    if args.beta_kl is not None:
        overrides["beta_kl"] = args.beta_kl
    if args.tau is not None:
        overrides["tau"] = args.tau
    if args.beta_iql is not None:
        overrides["beta"] = args.beta_iql
    if args.K_eval is not None:
        overrides["K_eval"] = args.K_eval
    if args.num_episodes is not None:
        overrides["num_eval_episodes"] = args.num_episodes
    if args.max_episode_steps is not None:
        overrides["max_episode_steps"] = args.max_episode_steps

    for key, value in overrides.items():
        setattr(cfg, key, value)

    return cfg


def setup_directories(args, cfg):
    """Create log, save, and output directories with timestamps."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.log_dir is None:
        args.log_dir = os.path.join("logs", args.domain, timestamp)
    if args.save_dir is None:
        args.save_dir = os.path.join("checkpoints", args.domain, timestamp)
    if args.output_dir is None:
        args.output_dir = os.path.join("results", args.domain)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    return args.log_dir, args.save_dir, args.output_dir


def run_train_encoder(args, cfg):
    """Run Phase 1: Train FRE encoder+decoder."""
    print("=" * 70)
    print("PHASE 1: Training FRE Encoder + Decoder (VAE)")
    print("=" * 70)
    print(f"Domain: {args.domain}")
    print(f"Device: {cfg.device}")
    print(f"Seed: {args.seed}")
    print(f"Log dir: {args.log_dir}")
    print(f"Save dir: {args.save_dir}")
    print(f"Encoder steps: {cfg.encoder_steps}")
    print(f"K: {cfg.K}, K': {cfg.K_prime}")
    print(f"Beta KL: {cfg.beta_kl}")
    print("=" * 70)

    set_seed(args.seed)

    encoder, decoder, stats = train_encoder(
        domain=args.domain,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        device=cfg.device,
        seed=args.seed,
        K=cfg.K,
        K_prime=cfg.K_prime,
        d_embed=cfg.d_embed,
        d_model=cfg.d_model,
        num_layers=cfg.num_encoder_layers,
        num_heads=cfg.num_heads,
        d_latent=cfg.d_latent,
        num_reward_bins=cfg.num_reward_bins,
        r_max=cfg.r_max,
        beta_kl=cfg.beta_kl,
        hidden_dims=cfg.decoder_hidden_dims,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        total_steps=cfg.encoder_steps,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        eval_interval=cfg.eval_interval,
        epsilon=cfg.epsilon,
        sparsity=cfg.sparsity,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    )

    # Save final encoder checkpoint
    encoder_path = os.path.join(args.save_dir, "encoder_final.pt")
    decoder_path = os.path.join(args.save_dir, "decoder_final.pt")
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "stats": stats,
        "config": cfg.to_dict(),
    }, encoder_path)
    print(f"\nFinal encoder saved to: {encoder_path}")

    return encoder, decoder, encoder_path


def run_train_iql(args, cfg):
    """Run Phase 2: Train IQL agent with frozen encoder."""
    print("=" * 70)
    print("PHASE 2: Training IQL Agent with Frozen Encoder")
    print("=" * 70)
    print(f"Domain: {args.domain}")
    print(f"Device: {cfg.device}")
    print(f"Seed: {args.seed}")
    print(f"Encoder path: {args.encoder_path}")
    print(f"Log dir: {args.log_dir}")
    print(f"Save dir: {args.save_dir}")
    print(f"IQL steps: {cfg.iql_steps}")
    print(f"Tau (expectile): {cfg.tau}")
    print(f"Beta (AWR temperature): {cfg.beta}")
    print("=" * 70)

    if args.encoder_path is None:
        raise ValueError("--encoder_path is required for train_iql mode. "
                         "Run train_encoder first or provide a pretrained encoder path.")

    set_seed(args.seed)

    iql_networks, stats = train_iql(
        domain=args.domain,
        encoder_checkpoint=args.encoder_path,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        device=cfg.device,
        seed=args.seed,
        K=cfg.K,
        batch_size=cfg.batch_size,
        total_steps=cfg.iql_steps,
        tau=cfg.tau,
        beta=cfg.beta,
        gamma=cfg.gamma,
        lr=cfg.lr,
        target_update_rate=cfg.target_update_rate,
        hidden_dims=cfg.iql_hidden_dims,
        d_latent=cfg.d_latent,
        log_interval=cfg.log_interval,
        save_interval=cfg.save_interval,
        eval_interval=cfg.eval_interval,
        epsilon=cfg.epsilon,
        sparsity=cfg.sparsity,
        mlp_hidden_dim=cfg.mlp_hidden_dim,
    )

    # Save final IQL checkpoint
    iql_path = os.path.join(args.save_dir, "iql_final.pt")
    torch.save({
        "iql_state_dict": iql_networks.state_dict(),
        "stats": stats,
        "config": cfg.to_dict(),
    }, iql_path)
    print(f"\nFinal IQL agent saved to: {iql_path}")

    return iql_networks, iql_path


def run_train_all(args, cfg):
    """Run strided training: Phase 1 (encoder) then Phase 2 (IQL)."""
    print("=" * 70)
    print("STRIDED TRAINING: Phase 1 (Encoder) + Phase 2 (IQL)")
    print("=" * 70)

    # Phase 1: Train encoder
    encoder, decoder, encoder_path = run_train_encoder(args, cfg)

    # Phase 2: Train IQL with frozen encoder
    args.encoder_path = encoder_path
    iql_networks, iql_path = run_train_iql(args, cfg)

    print("\n" + "=" * 70)
    print("STRIDED TRAINING COMPLETE!")
    print(f"Encoder: {encoder_path}")
    print(f"IQL: {iql_path}")
    print("=" * 70)

    return encoder, decoder, iql_networks, encoder_path, iql_path


def run_evaluate(args, cfg):
    """Run zero-shot evaluation on all downstream tasks (single seed)."""
    print("=" * 70)
    print("ZERO-SHOT EVALUATION")
    print("=" * 70)
    print(f"Domain: {args.domain}")
    print(f"Device: {cfg.device}")
    print(f"Seed: {args.seed}")
    print(f"Encoder path: {args.encoder_path}")
    print(f"IQL path: {args.iql_path}")
    print(f"K_eval: {cfg.K_eval}")
    print(f"Num episodes: {cfg.num_eval_episodes}")
    print("=" * 70)

    if args.encoder_path is None:
        raise ValueError("--encoder_path is required for evaluation.")
    if args.iql_path is None:
        raise ValueError("--iql_path is required for evaluation.")

    set_seed(args.seed)

    results = run_evaluation(
        encoder_path=args.encoder_path,
        iql_path=args.iql_path,
        domain=args.domain,
        data_dir=args.data_dir,
        device=cfg.device,
        seed=args.seed,
        K=cfg.K_eval,
        num_episodes=cfg.num_eval_episodes,
        max_episode_steps=cfg.max_episode_steps,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    for task_name, task_results in results.items():
        if isinstance(task_results, dict) and "mean_return" in task_results:
            print(f"  {task_name}: {task_results['mean_return']:.2f} "
                  f"± {task_results['std_return']:.2f}")
    print("=" * 70)

    return results


def run_evaluate_multi(args, cfg):
    """Run zero-shot evaluation with multiple seeds."""
    seeds = args.seeds if args.seeds is not None else [0, 1, 2, 3, 4]

    print("=" * 70)
    print("MULTI-SEED ZERO-SHOT EVALUATION")
    print("=" * 70)
    print(f"Domain: {args.domain}")
    print(f"Device: {cfg.device}")
    print(f"Seeds: {seeds}")
    print(f"Encoder path: {args.encoder_path}")
    print(f"IQL path: {args.iql_path}")
    print(f"K_eval: {cfg.K_eval}")
    print(f"Num episodes: {cfg.num_eval_episodes}")
    print("=" * 70)

    if args.encoder_path is None:
        raise ValueError("--encoder_path is required for evaluation.")
    if args.iql_path is None:
        raise ValueError("--iql_path is required for evaluation.")

    results = evaluate_multiple_seeds(
        encoder_path=args.encoder_path,
        iql_path=args.iql_path,
        domain=args.domain,
        data_dir=args.data_dir,
        device=cfg.device,
        seeds=seeds,
        K=cfg.K_eval,
        num_episodes=cfg.num_eval_episodes,
        max_episode_steps=cfg.max_episode_steps,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    for task_name, task_results in results.items():
        if isinstance(task_results, dict) and "mean_return" in task_results:
            print(f"  {task_name}: {task_results['mean_return']:.2f} "
                  f"± {task_results['std_return']:.2f}")
        elif isinstance(task_results, dict) and "mean_across_seeds" in task_results:
            print(f"  {task_name}: {task_results['mean_across_seeds']:.2f} "
                  f"± {task_results['std_across_seeds']:.2f}")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    args = parse_args()

    # Build configuration
    cfg = build_config_from_args(args)

    # Setup directories
    log_dir, save_dir, output_dir = setup_directories(args, cfg)
    args.log_dir = log_dir
    args.save_dir = save_dir
    args.output_dir = output_dir

    # Resolve device
    device = get_device(cfg.device)
    cfg.device = str(device)

    if args.quiet:
        args.verbose = False

    # Print header
    print("\n" + "=" * 70)
    print("Functional Reward Encodings (FRE)")
    print("Zero-Shot Offline Reinforcement Learning")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Domain: {args.domain}")
    print(f"Device: {cfg.device}")
    print(f"Seed: {args.seed}")
    print("=" * 70 + "\n")

    # Execute requested mode
    if args.mode == "train_encoder":
        run_train_encoder(args, cfg)

    elif args.mode == "train_iql":
        run_train_iql(args, cfg)

    elif args.mode == "train_all":
        run_train_all(args, cfg)

    elif args.mode == "evaluate":
        run_evaluate(args, cfg)

    elif args.mode == "evaluate_multi":
        run_evaluate_multi(args, cfg)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("\nDone!")


if __name__ == "__main__":
    main()