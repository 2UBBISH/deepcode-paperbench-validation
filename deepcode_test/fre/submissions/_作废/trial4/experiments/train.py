"""
FRE Training Script
====================
Trains a Functional Reward Encodings (FRE) agent on a specified offline RL domain
using the strided training scheme (Algorithm 1 from the paper).

Usage:
    python experiments/train.py --config experiments/configs/antmaze.yaml
    python experiments/train.py --config experiments/configs/kitchen.yaml
    python experiments/train.py --domain antmaze --data_dir ./data --seed 0

The script:
    1. Loads the offline dataset (D4RL or ExORL)
    2. Creates the reward prior distribution
    3. Builds the FRE agent (encoder + decoder + IQL)
    4. Runs Phase 1: VAE training on random reward functions
    5. Runs Phase 2: IQL training with frozen encoder
    6. Saves checkpoints and logs metrics
"""

import os
import sys
import argparse
import yaml
import json
import time
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fre.utils import (
    load_dataset,
    set_seed,
    get_device,
    compute_reward_statistics,
    ReplayBuffer,
)
from fre.reward_prior import create_reward_prior, RewardPrior
from fre.fre_agent import FREAgent, create_fre_agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train FRE agent on offline RL dataset"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain name (antmaze, kitchen, exorl_walker, exorl_cheetah)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name (e.g., antmaze-large-diverse-v2, kitchen-complete-v0)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory for datasets",
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--vae_steps", type=int, default=100000, help="Phase 1 VAE training steps")
    parser.add_argument("--iql_steps", type=int, default=1000000, help="Phase 2 IQL training steps")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for IQL training")
    parser.add_argument("--vae_batch_size", type=int, default=256, help="Batch size for VAE training")
    parser.add_argument("--d_z", type=int, default=64, help="Latent dimension")
    parser.add_argument("--K_encoder", type=int, default=32, help="Number of encoder context pairs")
    parser.add_argument("--K_decoder", type=int, default=64, help="Number of decoder query states")
    parser.add_argument("--beta", type=float, default=1.0, help="KL divergence weight")
    parser.add_argument("--vae_lr", type=float, default=3e-4, help="VAE learning rate")
    parser.add_argument("--iql_lr", type=float, default=3e-4, help="IQL learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--expectile", type=float, default=0.7, help="IQL expectile")
    parser.add_argument("--alpha", type=float, default=3.0, help="AWR temperature")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")

    # Encoder/Decoder architecture
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden dim")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--num_layers", type=int, default=3, help="Transformer layers")
    parser.add_argument("--num_reward_bins", type=int, default=50, help="Reward discretization bins")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--decoder_hidden_dims", type=int, nargs="+", default=[256, 256],
                        help="Decoder MLP hidden dimensions")

    # Reward prior
    parser.add_argument("--reward_families", type=str, nargs="+",
                        default=["goal", "linear", "mlp"],
                        help="Reward prior families to use")
    parser.add_argument("--linear_sparsity", type=float, default=0.8,
                        help="Sparsity for random linear rewards")
    parser.add_argument("--mlp_hidden_dim", type=int, default=256,
                        help="Hidden dim for random MLP rewards")
    parser.add_argument("--goal_epsilon", type=float, default=0.1,
                        help="Goal-reaching tolerance")
    parser.add_argument("--use_xy_prior", action="store_true",
                        help="Use XY-position-only prior for AntMaze")

    # Logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--eval_interval", type=int, default=10000,
                        help="Steps between VAE evaluations")
    parser.add_argument("--save_interval", type=int, default=50000,
                        help="Steps between checkpoint saves")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="fre",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/username")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag for logging")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, cpu, or specific cuda:N)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU training")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(args: argparse.Namespace, config: dict) -> dict:
    """
    Merge command-line arguments with YAML config.
    CLI arguments take precedence over config file values.
    """
    merged = {}

    # Start with config file values
    if config:
        for key, value in config.items():
            merged[key] = value

    # Override with CLI arguments (only if explicitly set)
    cli_dict = vars(args)
    for key, value in cli_dict.items():
        if value is not None and not (
            # Skip defaults that weren't explicitly set
            key in ["config"] and value is None
        ):
            # For boolean flags, only override if True (since default is False)
            if isinstance(value, bool):
                if value:  # Only override if flag was set
                    merged[key] = value
            elif value is not None:
                merged[key] = value

    return merged


def build_config_from_args(args: argparse.Namespace) -> dict:
    """Build configuration dictionary from parsed arguments."""
    config = {}

    # Core settings
    config["domain"] = args.domain
    config["dataset"] = args.dataset
    config["data_dir"] = args.data_dir
    config["seed"] = args.seed

    # Training
    config["vae_steps"] = args.vae_steps
    config["iql_steps"] = args.iql_steps
    config["batch_size"] = args.batch_size
    config["vae_batch_size"] = args.vae_batch_size
    config["d_z"] = args.d_z
    config["K_encoder"] = args.K_encoder
    config["K_decoder"] = args.K_decoder
    config["beta"] = args.beta
    config["vae_lr"] = args.vae_lr
    config["iql_lr"] = args.iql_lr
    config["gamma"] = args.gamma
    config["expectile"] = args.expectile
    config["alpha"] = args.alpha
    config["tau"] = args.tau

    # Encoder/Decoder
    config["d_model"] = args.d_model
    config["nhead"] = args.nhead
    config["num_layers"] = args.num_layers
    config["num_reward_bins"] = args.num_reward_bins
    config["dropout"] = args.dropout
    config["decoder_hidden_dims"] = args.decoder_hidden_dims

    # Reward prior
    config["reward_families"] = args.reward_families
    config["linear_sparsity"] = args.linear_sparsity
    config["mlp_hidden_dim"] = args.mlp_hidden_dim
    config["goal_epsilon"] = args.goal_epsilon
    config["use_xy_prior"] = args.use_xy_prior

    # Logging
    config["checkpoint_dir"] = args.checkpoint_dir
    config["log_dir"] = args.log_dir
    config["eval_interval"] = args.eval_interval
    config["save_interval"] = args.save_interval
    config["use_wandb"] = args.use_wandb
    config["wandb_project"] = args.wandb_project
    config["wandb_entity"] = args.wandb_entity
    config["tag"] = args.tag

    # Device
    config["device"] = args.device
    config["cpu"] = args.cpu

    return config


def main():
    args = parse_args()

    # Load config file if provided
    config = {}
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")

    # Merge configs (CLI overrides file)
    merged = merge_configs(args, config)

    # Extract key settings
    domain = merged.get("domain", args.domain)
    dataset_name = merged.get("dataset", args.dataset)
    data_dir = merged.get("data_dir", args.data_dir)
    seed = merged.get("seed", args.seed)

    # Validate required settings
    if dataset_name is None and domain is None:
        raise ValueError(
            "Must specify either --dataset or --domain (or set in config file). "
            "Use --config for predefined configurations."
        )

    # Auto-detect dataset from domain if not specified
    if dataset_name is None:
        dataset_name = domain  # For D4RL datasets, domain name is the dataset name

    # Set random seed
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Determine device
    if merged.get("cpu", False):
        device = torch.device("cpu")
    elif merged.get("device") is not None:
        device = torch.device(merged["device"])
    else:
        device = get_device()
    print(f"Using device: {device}")

    # ============================================================
    # Step 1: Load dataset
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*60}")

    replay_buffer = load_dataset(dataset_name, data_dir=data_dir)
    print(f"Dataset loaded: {replay_buffer.num_states} transitions")
    print(f"  State dim: {replay_buffer.state_dim}")
    print(f"  Action dim: {replay_buffer.action_dim}")

    # Compute reward statistics for reward bin range
    reward_stats = compute_reward_statistics(replay_buffer)
    print(f"  Reward range: [{reward_stats['min']:.3f}, {reward_stats['max']:.3f}]")
    print(f"  Reward mean: {reward_stats['mean']:.3f}, std: {reward_stats['std']:.3f}")

    # ============================================================
    # Step 2: Create reward prior
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Creating reward prior")
    print(f"{'='*60}")

    # Get all states from dataset for goal sampling
    dataset_states = replay_buffer.get_all_states()

    reward_families = merged.get("reward_families", ["goal", "linear", "mlp"])
    use_xy_prior = merged.get("use_xy_prior", False)

    reward_prior = create_reward_prior(
        dataset_states=dataset_states,
        state_dim=replay_buffer.state_dim,
        domain=domain if use_xy_prior else None,
        reward_families=reward_families,
        linear_sparsity=merged.get("linear_sparsity", 0.8),
        mlp_hidden_dim=merged.get("mlp_hidden_dim", 256),
        goal_epsilon=merged.get("goal_epsilon", 0.1),
        use_xy_prior=use_xy_prior,
    )
    print(f"  Reward families: {reward_families}")
    print(f"  XY prior: {use_xy_prior}")

    # ============================================================
    # Step 3: Build FRE agent
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Building FRE agent")
    print(f"{'='*60}")

    d_z = merged.get("d_z", 64)
    K_encoder = merged.get("K_encoder", 32)
    K_decoder = merged.get("K_decoder", 64)
    beta = merged.get("beta", 1.0)

    # Encoder kwargs
    encoder_kwargs = {
        "d_model": merged.get("d_model", 256),
        "nhead": merged.get("nhead", 4),
        "num_layers": merged.get("num_layers", 3),
        "d_z": d_z,
        "num_reward_bins": merged.get("num_reward_bins", 50),
        "dropout": merged.get("dropout", 0.1),
        "reward_min": reward_stats["min"],
        "reward_max": reward_stats["max"],
    }

    # Decoder kwargs
    decoder_kwargs = {
        "hidden_dims": merged.get("decoder_hidden_dims", [256, 256]),
        "dropout": merged.get("dropout", 0.0),
    }

    # IQL kwargs
    iql_kwargs = {
        "d_z": d_z,
        "hidden_dims": merged.get("decoder_hidden_dims", [256, 256]),
        "gamma": merged.get("gamma", 0.99),
        "tau": merged.get("tau", 0.005),
        "expectile": merged.get("expectile", 0.7),
        "alpha": merged.get("alpha", 3.0),
        "lr": merged.get("iql_lr", 3e-4),
        "device": device,
        "dropout": merged.get("dropout", 0.0),
    }

    # Build agent config
    agent_config = {
        "state_dim": replay_buffer.state_dim,
        "action_dim": replay_buffer.action_dim,
        "replay_buffer": replay_buffer,
        "reward_prior": reward_prior,
        "encoder_kwargs": encoder_kwargs,
        "decoder_kwargs": decoder_kwargs,
        "iql_kwargs": iql_kwargs,
        "d_z": d_z,
        "K_encoder": K_encoder,
        "K_decoder": K_decoder,
        "beta": beta,
        "vae_lr": merged.get("vae_lr", 3e-4),
        "iql_lr": merged.get("iql_lr", 3e-4),
        "device": device,
        "checkpoint_dir": merged.get("checkpoint_dir", "./checkpoints"),
        "log_dir": merged.get("log_dir", "./logs"),
        "use_wandb": merged.get("use_wandb", False),
    }

    agent = create_fre_agent(**agent_config)
    agent.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in agent.encoder.parameters())
    total_params += sum(p.numel() for p in agent.decoder.parameters())
    total_params += sum(p.numel() for p in agent.iql.policy.parameters())
    total_params += sum(p.numel() for p in agent.iql.qf.parameters())
    total_params += sum(p.numel() for p in agent.iql.vf.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Latent dim (d_z): {d_z}")
    print(f"  K_encoder: {K_encoder}, K_decoder: {K_decoder}")
    print(f"  Beta (KL weight): {beta}")

    # ============================================================
    # Step 4: Train FRE agent
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Starting training")
    print(f"{'='*60}")

    vae_steps = merged.get("vae_steps", 100000)
    iql_steps = merged.get("iql_steps", 1000000)
    eval_interval = merged.get("eval_interval", 10000)
    save_interval = merged.get("save_interval", 50000)
    tag = merged.get("tag", f"{dataset_name}_seed{seed}")

    print(f"  Phase 1 (VAE): {vae_steps} steps")
    print(f"  Phase 2 (IQL): {iql_steps} steps")
    print(f"  Eval interval: {eval_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Tag: {tag}")

    start_time = time.time()

    results = agent.train(
        vae_steps=vae_steps,
        iql_steps=iql_steps,
        batch_size=merged.get("batch_size", 256),
        vae_batch_size=merged.get("vae_batch_size", 256),
        eval_interval=eval_interval,
        save_interval=save_interval,
        tag=tag,
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")

    # ============================================================
    # Step 5: Save final checkpoint
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Saving final checkpoint")
    print(f"{'='*60}")

    final_path = agent.save_checkpoint(tag=f"{tag}_final")
    print(f"  Final checkpoint: {final_path}")

    # Save training results
    results_path = os.path.join(
        merged.get("checkpoint_dir", "./checkpoints"),
        f"{tag}_results.json"
    )
    # Convert non-serializable values
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool, list, dict)):
            serializable_results[key] = value
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = str(value)

    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Seed: {seed}")
    print(f"  VAE steps: {vae_steps}")
    print(f"  IQL steps: {iql_steps}")
    print(f"  Final VAE loss: {results.get('final_vae_loss', 'N/A')}")
    print(f"  Final IQL loss: {results.get('final_iql_loss', 'N/A')}")
    print(f"  Total time: {elapsed/3600:.2f} hours")
    print(f"  Checkpoint: {final_path}")

    agent.close()
    print("Done!")


if __name__ == "__main__":
    main()