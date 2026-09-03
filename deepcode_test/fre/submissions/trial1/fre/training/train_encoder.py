"""
Phase 1 Training: FRE Encoder + Decoder (VAE)

Trains the permutation-invariant transformer VAE encoder and feedforward
reward decoder on random unsupervised reward functions from the prior
distribution p(eta). The encoder learns to map sets of (state, reward)
pairs into a latent vector z that captures the reward function structure.
The decoder learns to reconstruct rewards from (state, z) pairs.

Training follows the beta-VAE ELBO objective:
    L = (1/K') * sum(r_k - decoder(s_k, z))^2 + beta * KL(N(mu, sigma^2) || N(0, I))

After Phase 1, the encoder is frozen and used to condition IQL networks
in Phase 2.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm

from fre.config import config
from fre.data.dataset import OfflineDataset, load_dataset
from fre.reward_functions.mixture import MixtureRewardFunction
from fre.models.encoder import FREEncoder
from fre.models.decoder import RewardDecoder
from fre.training.utils import (
    Logger,
    create_prior_reward_function,
    sample_encoder_batch,
    compute_vae_loss,
    save_checkpoint,
    set_seed,
    get_device,
    count_parameters,
    print_model_info,
)


def train_encoder(
    domain: str = "antmaze",
    data_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    # Overridable hyperparameters
    K: Optional[int] = None,
    K_prime: Optional[int] = None,
    d_embed: Optional[int] = None,
    d_model: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    d_latent: Optional[int] = None,
    num_reward_bins: Optional[int] = None,
    r_max: Optional[float] = None,
    beta_kl: Optional[float] = None,
    hidden_dims: Optional[list] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    total_steps: Optional[int] = None,
    log_interval: Optional[int] = None,
    save_interval: Optional[int] = None,
    eval_interval: Optional[int] = None,
    # Reward function parameters
    epsilon: Optional[float] = None,
    sparsity: Optional[float] = None,
    mlp_hidden_dim: Optional[int] = None,
) -> Tuple[FREEncoder, RewardDecoder, dict]:
    """
    Train the FRE encoder and decoder (Phase 1).

    Args:
        domain: Dataset domain ('antmaze', 'walker', 'cheetah', 'kitchen')
        data_dir: Path to dataset directory
        log_dir: Path to logging directory
        save_dir: Path to save checkpoints
        device: Device string ('cuda' or 'cpu')
        seed: Random seed
        K: Number of encoder states
        K_prime: Number of decoder states
        d_embed: Embedding dimension
        d_model: Transformer hidden dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_latent: Latent dimension
        num_reward_bins: Number of reward discretization bins
        r_max: Reward clipping range
        beta_kl: KL divergence weight
        hidden_dims: Decoder hidden layer dimensions
        lr: Learning rate
        batch_size: Not used directly (encoder uses K, K' per iteration)
        total_steps: Total training iterations
        log_interval: Logging interval (steps)
        save_interval: Checkpoint saving interval (steps)
        eval_interval: Evaluation interval (steps)
        epsilon: Singleton reward epsilon threshold
        sparsity: Linear reward sparsity
        mlp_hidden_dim: MLP reward hidden dimension

    Returns:
        Tuple of (trained encoder, trained decoder, training_stats dict)
    """
    # Resolve hyperparameters from config if not overridden
    K = K if K is not None else config.K
    K_prime = K_prime if K_prime is not None else config.K_prime
    d_embed = d_embed if d_embed is not None else config.d_embed
    d_model = d_model if d_model is not None else config.d_model
    num_layers = num_layers if num_layers is not None else config.num_encoder_layers
    num_heads = num_heads if num_heads is not None else config.num_heads
    d_latent = d_latent if d_latent is not None else config.d_latent
    num_reward_bins = num_reward_bins if num_reward_bins is not None else config.num_reward_bins
    r_max = r_max if r_max is not None else config.R_max
    beta_kl = beta_kl if beta_kl is not None else config.beta_kl
    hidden_dims = hidden_dims if hidden_dims is not None else list(config.decoder_hidden_dims)
    lr = lr if lr is not None else config.encoder_lr
    total_steps = total_steps if total_steps is not None else config.encoder_steps
    log_interval = log_interval if log_interval is not None else config.log_interval
    save_interval = save_interval if save_interval is not None else config.save_interval
    eval_interval = eval_interval if eval_interval is not None else config.eval_interval
    epsilon = epsilon if epsilon is not None else config.epsilon
    sparsity = sparsity if sparsity is not None else config.sparsity
    mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.mlp_hidden_dim

    # Set up device
    if device is None:
        device = get_device(config.device)
    else:
        device = get_device(device)

    # Set random seed
    set_seed(seed)

    # Set up directories
    if log_dir is None:
        log_dir = os.path.join(config.log_dir, f"{domain}_encoder_seed{seed}")
    if save_dir is None:
        save_dir = os.path.join(config.save_dir, f"{domain}_encoder_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize logger
    logger = Logger(log_dir, use_tensorboard=True)

    print(f"\n{'='*60}")
    print(f"Phase 1: Training FRE Encoder + Decoder")
    print(f"{'='*60}")
    print(f"Domain: {domain}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Total steps: {total_steps}")
    print(f"K (encoder states): {K}, K' (decoder states): {K_prime}")
    print(f"d_embed: {d_embed}, d_model: {d_model}, d_latent: {d_latent}")
    print(f"num_layers: {num_layers}, num_heads: {num_heads}")
    print(f"num_reward_bins: {num_reward_bins}, R_max: {r_max}")
    print(f"beta_kl: {beta_kl}, lr: {lr}")
    print(f"Log dir: {log_dir}")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(domain, data_dir=data_dir, device=device, normalize=True)
    state_dim = dataset.state_dim
    print(f"Dataset loaded: {len(dataset)} transitions, state_dim={state_dim}")

    # Create prior reward function mixture
    print("Creating prior reward function mixture...")
    reward_mixture = create_prior_reward_function(
        state_dim=state_dim,
        epsilon=epsilon,
        sparsity=sparsity,
        mlp_hidden_dim=mlp_hidden_dim,
        device=device,
    )

    # Build encoder
    encoder = FREEncoder(
        state_dim=state_dim,
        d_embed=d_embed,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_latent=d_latent,
        num_reward_bins=num_reward_bins,
        r_max=r_max,
        dropout=0.0,
    ).to(device)

    # Build decoder
    decoder = RewardDecoder(
        state_dim=state_dim,
        d_latent=d_latent,
        hidden_dims=hidden_dims,
        activation="relu",
        dropout=0.0,
    ).to(device)

    # Print model info
    print_model_info(encoder, decoder, None)

    # Optimizer (trains both encoder and decoder)
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=lr,
    )

    # Learning rate scheduler (optional: cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=lr * 0.01,
    )

    # Training statistics
    stats = {
        "train_loss": [],
        "recon_loss": [],
        "kl_loss": [],
        "eval_loss": [],
        "eval_recon_loss": [],
        "eval_kl_loss": [],
        "step_times": [],
    }

    # Training loop
    print("\nStarting encoder training...")
    encoder.train()
    decoder.train()

    start_time = time.time()
    best_eval_loss = float("inf")

    pbar = tqdm(range(1, total_steps + 1), desc="Encoder Training")
    for step in pbar:
        step_start = time.time()

        # Sample a new reward function from the prior mixture
        reward_mixture.reset()

        # Sample encoder and decoder states with rewards
        batch = sample_encoder_batch(
            dataset=dataset,
            reward_fn=reward_mixture,
            K=K,
            K_prime=K_prime,
            device=device,
        )

        encoder_states = batch["encoder_states"]      # (1, K, state_dim)
        encoder_rewards = batch["encoder_rewards"]    # (1, K)
        decoder_states = batch["decoder_states"]      # (1, K', state_dim)
        decoder_rewards = batch["decoder_rewards"]    # (1, K')

        # Forward pass: encode to get latent z
        z, mu, logvar = encoder(encoder_states, encoder_rewards, deterministic=False)

        # Forward pass: decode to predict rewards for decoder states
        # z shape: (1, d_latent), decoder_states shape: (1, K', state_dim)
        predicted_rewards = decoder(decoder_states, z)  # (1, K')

        # Compute VAE loss
        total_loss, recon_loss, kl_loss = compute_vae_loss(
            decoder=None,  # Not used; we already have predicted_rewards
            decoder_states=None,  # Not used
            decoder_rewards=decoder_rewards,
            z=z,
            mu=mu,
            logvar=logvar,
            beta_kl=beta_kl,
            predicted_rewards=predicted_rewards,
        )

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=10.0,
        )
        optimizer.step()
        scheduler.step()

        step_time = time.time() - step_start

        # Logging
        stats["train_loss"].append(total_loss.item())
        stats["recon_loss"].append(recon_loss.item())
        stats["kl_loss"].append(kl_loss.item())
        stats["step_times"].append(step_time)

        if step % log_interval == 0 or step == 1:
            logger.log_scalar("train/total_loss", total_loss.item(), step)
            logger.log_scalar("train/recon_loss", recon_loss.item(), step)
            logger.log_scalar("train/kl_loss", kl_loss.item(), step)
            logger.log_scalar("train/lr", scheduler.get_last_lr()[0], step)

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

        # Evaluation on a held-out set of reward functions
        if step % eval_interval == 0 or step == 1:
            eval_total, eval_recon, eval_kl = evaluate_encoder(
                encoder=encoder,
                decoder=decoder,
                dataset=dataset,
                reward_mixture=reward_mixture,
                K=K,
                K_prime=K_prime,
                beta_kl=beta_kl,
                num_eval=20,
                device=device,
            )
            stats["eval_loss"].append(eval_total)
            stats["eval_recon_loss"].append(eval_recon)
            stats["eval_kl_loss"].append(eval_kl)

            logger.log_scalar("eval/total_loss", eval_total, step)
            logger.log_scalar("eval/recon_loss", eval_recon, step)
            logger.log_scalar("eval/kl_loss", eval_kl, step)

            print(f"\nStep {step}: Eval Loss={eval_total:.4f}, "
                  f"Recon={eval_recon:.4f}, KL={eval_kl:.4f}")

            # Save best model
            if eval_total < best_eval_loss:
                best_eval_loss = eval_total
                save_checkpoint(
                    save_dir=save_dir,
                    filename="encoder_best.pt",
                    encoder=encoder,
                    decoder=decoder,
                    iql_networks=None,
                    optimizer_encoder=optimizer,
                    optimizer_iql=None,
                    step=step,
                    extra_info={
                        "eval_loss": eval_total,
                        "eval_recon_loss": eval_recon,
                        "eval_kl_loss": eval_kl,
                        "domain": domain,
                        "seed": seed,
                    },
                )
                print(f"  -> Best model saved (eval_loss={eval_total:.4f})")

        # Periodic checkpoint
        if step % save_interval == 0:
            save_checkpoint(
                save_dir=save_dir,
                filename=f"encoder_step{step}.pt",
                encoder=encoder,
                decoder=decoder,
                iql_networks=None,
                optimizer_encoder=optimizer,
                optimizer_iql=None,
                step=step,
                extra_info={
                    "domain": domain,
                    "seed": seed,
                },
            )

    # End of training
    total_time = time.time() - start_time
    print(f"\nEncoder training completed in {total_time:.1f}s "
          f"({total_time/60:.1f} min)")

    # Save final model
    save_checkpoint(
        save_dir=save_dir,
        filename="encoder_final.pt",
        encoder=encoder,
        decoder=decoder,
        iql_networks=None,
        optimizer_encoder=optimizer,
        optimizer_iql=None,
        step=total_steps,
        extra_info={
            "domain": domain,
            "seed": seed,
            "final_train_loss": stats["train_loss"][-1] if stats["train_loss"] else None,
        },
    )
    print(f"Final model saved to {save_dir}/encoder_final.pt")

    logger.close()

    return encoder, decoder, stats


def evaluate_encoder(
    encoder: FREEncoder,
    decoder: RewardDecoder,
    dataset: OfflineDataset,
    reward_mixture: MixtureRewardFunction,
    K: int,
    K_prime: int,
    beta_kl: float,
    num_eval: int = 20,
    device: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate the encoder+decoder on multiple random reward functions.

    Args:
        encoder: FRE encoder
        decoder: Reward decoder
        dataset: Offline dataset
        reward_mixture: Prior reward function mixture
        K: Number of encoder states
        K_prime: Number of decoder states
        beta_kl: KL divergence weight
        num_eval: Number of evaluation reward functions
        device: Device

    Returns:
        Tuple of (avg_total_loss, avg_recon_loss, avg_kl_loss)
    """
    encoder.eval()
    decoder.eval()

    total_losses = []
    recon_losses = []
    kl_losses = []

    with torch.no_grad():
        for _ in range(num_eval):
            # Sample a new reward function
            reward_mixture.reset()

            # Sample encoder and decoder states
            batch = sample_encoder_batch(
                dataset=dataset,
                reward_fn=reward_mixture,
                K=K,
                K_prime=K_prime,
                device=device,
            )

            encoder_states = batch["encoder_states"]
            encoder_rewards = batch["encoder_rewards"]
            decoder_states = batch["decoder_states"]
            decoder_rewards = batch["decoder_rewards"]

            # Encode (deterministic for evaluation)
            z, mu, logvar = encoder(encoder_states, encoder_rewards, deterministic=True)

            # Decode
            predicted_rewards = decoder(decoder_states, z)

            # Compute loss
            total_loss, recon_loss, kl_loss = compute_vae_loss(
                decoder=None,
                decoder_states=None,
                decoder_rewards=decoder_rewards,
                z=z,
                mu=mu,
                logvar=logvar,
                beta_kl=beta_kl,
                predicted_rewards=predicted_rewards,
            )

            total_losses.append(total_loss.item())
            recon_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())

    encoder.train()
    decoder.train()

    avg_total = np.mean(total_losses)
    avg_recon = np.mean(recon_losses)
    avg_kl = np.mean(kl_losses)

    return avg_total, avg_recon, avg_kl


def load_pretrained_encoder(
    checkpoint_path: str,
    state_dim: int,
    device: Optional[str] = None,
    d_embed: Optional[int] = None,
    d_model: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    d_latent: Optional[int] = None,
    num_reward_bins: Optional[int] = None,
    r_max: Optional[float] = None,
    hidden_dims: Optional[list] = None,
) -> Tuple[FREEncoder, RewardDecoder]:
    """
    Load a pretrained encoder and decoder from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        state_dim: State dimension
        device: Device
        (other args override config defaults for model architecture)

    Returns:
        Tuple of (encoder, decoder)
    """
    if device is None:
        device = get_device(config.device)
    else:
        device = get_device(device)

    # Resolve architecture from config if not provided
    d_embed = d_embed if d_embed is not None else config.d_embed
    d_model = d_model if d_model is not None else config.d_model
    num_layers = num_layers if num_layers is not None else config.num_encoder_layers
    num_heads = num_heads if num_heads is not None else config.num_heads
    d_latent = d_latent if d_latent is not None else config.d_latent
    num_reward_bins = num_reward_bins if num_reward_bins is not None else config.num_reward_bins
    r_max = r_max if r_max is not None else config.R_max
    hidden_dims = hidden_dims if hidden_dims is not None else list(config.decoder_hidden_dims)

    # Build models
    encoder = FREEncoder(
        state_dim=state_dim,
        d_embed=d_embed,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_latent=d_latent,
        num_reward_bins=num_reward_bins,
        r_max=r_max,
    ).to(device)

    decoder = RewardDecoder(
        state_dim=state_dim,
        d_latent=d_latent,
        hidden_dims=hidden_dims,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    print(f"Loaded pretrained encoder from {checkpoint_path}")
    print(f"  Step: {checkpoint.get('step', 'unknown')}")
    if "extra_info" in checkpoint:
        for k, v in checkpoint["extra_info"].items():
            print(f"  {k}: {v}")

    return encoder, decoder


if __name__ == "__main__":
    """
    Quick test: Train encoder on a small number of steps to verify pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train FRE Encoder")
    parser.add_argument("--domain", type=str, default="antmaze",
                        help="Dataset domain")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of training steps (override config)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    args = parser.parse_args()

    encoder, decoder, stats = train_encoder(
        domain=args.domain,
        seed=args.seed,
        device=args.device,
        total_steps=args.steps,
    )

    print("\nTraining complete!")
    print(f"Final train loss: {stats['train_loss'][-1]:.4f}")
    print(f"Final recon loss: {stats['recon_loss'][-1]:.4f}")
    print(f"Final KL loss: {stats['kl_loss'][-1]:.4f}")