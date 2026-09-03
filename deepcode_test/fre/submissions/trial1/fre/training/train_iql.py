"""
Phase 2 Training: Implicit Q-Learning (IQL) with frozen FRE encoder.

Trains Q, V, and policy networks conditioned on latent reward encodings z.
The encoder is frozen; at each iteration a random reward function is sampled
from the prior mixture, encoded into z, and used to condition all networks.

Reference: Kostrikov et al. (2021) "Offline Reinforcement Learning with
Implicit Q-Learning", adapted for z-conditioning as described in the
FRE paper (Frans et al., 2024).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm

from fre.config import config
from fre.data.dataset import OfflineDataset, load_dataset
from fre.reward_functions.mixture import MixtureRewardFunction
from fre.models.encoder import FREEncoder
from fre.models.decoder import RewardDecoder
from fre.models.iql import IQLNetworks
from fre.training.utils import (
    Logger,
    create_prior_reward_function,
    sample_iql_batch,
    compute_iql_value_loss,
    compute_iql_q_loss,
    compute_iql_policy_loss,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
    count_parameters,
    print_model_info,
)


def train_iql(
    domain: str = "antmaze",
    encoder: Optional[FREEncoder] = None,
    encoder_checkpoint: Optional[str] = None,
    data_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    save_dir: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    # IQL hyperparameters
    d_latent: Optional[int] = None,
    hidden_dims: Optional[list] = None,
    tau: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    total_steps: Optional[int] = None,
    target_update_rate: Optional[float] = None,
    max_weight: Optional[float] = None,
    log_std_min: Optional[float] = None,
    log_std_max: Optional[float] = None,
    # Encoder sampling
    K: Optional[int] = None,
    # Reward function parameters
    epsilon: Optional[float] = None,
    sparsity: Optional[float] = None,
    mlp_hidden_dim: Optional[int] = None,
    # Logging
    log_interval: Optional[int] = None,
    save_interval: Optional[int] = None,
    eval_interval: Optional[int] = None,
    # Resume
    resume_checkpoint: Optional[str] = None,
) -> Tuple[IQLNetworks, Dict[str, Any]]:
    """
    Train IQL networks with frozen FRE encoder (Phase 2).

    Args:
        domain: Dataset domain ('antmaze', 'walker', 'cheetah', 'kitchen')
        encoder: Pre-trained FRE encoder (if None, loads from encoder_checkpoint)
        encoder_checkpoint: Path to encoder checkpoint to load
        data_dir: Directory containing datasets
        log_dir: Directory for TensorBoard logs
        save_dir: Directory for saving checkpoints
        device: Device string ('cuda' or 'cpu')
        seed: Random seed
        d_latent: Latent dimension of z
        hidden_dims: Hidden layer dimensions for IQL networks
        tau: Expectile for value loss
        beta: Temperature for advantage-weighted regression
        gamma: Discount factor
        lr: Learning rate
        batch_size: Batch size for IQL training
        total_steps: Total training steps
        target_update_rate: Soft target update rate
        max_weight: Maximum advantage weight clipping
        log_std_min: Minimum log std for policy
        log_std_max: Maximum log std for policy
        K: Number of encoder states for z computation
        epsilon: Epsilon for singleton reward
        sparsity: Sparsity for linear reward
        mlp_hidden_dim: Hidden dim for MLP reward
        log_interval: Steps between logging
        save_interval: Steps between checkpoint saves
        eval_interval: Steps between evaluations
        resume_checkpoint: Path to checkpoint to resume from

    Returns:
        Tuple of (trained IQLNetworks, training statistics dict)
    """
    # Resolve configuration with defaults
    device = get_device(device)
    d_latent = d_latent if d_latent is not None else config.d_latent
    hidden_dims = hidden_dims if hidden_dims is not None else config.iql_hidden_dims
    tau = tau if tau is not None else config.iql_tau
    beta = beta if beta is not None else config.iql_beta
    gamma = gamma if gamma is not None else config.gamma
    lr = lr if lr is not None else config.iql_lr
    batch_size = batch_size if batch_size is not None else config.iql_batch_size
    total_steps = total_steps if total_steps is not None else config.iql_steps
    target_update_rate = target_update_rate if target_update_rate is not None else config.target_update_rate
    max_weight = max_weight if max_weight is not None else config.iql_max_weight
    log_std_min = log_std_min if log_std_min is not None else config.log_std_min
    log_std_max = log_std_max if log_std_max is not None else config.log_std_max
    K = K if K is not None else config.K
    epsilon = epsilon if epsilon is not None else config.epsilon
    sparsity = sparsity if sparsity is not None else config.sparsity
    mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else config.mlp_hidden_dim
    log_interval = log_interval if log_interval is not None else config.log_interval
    save_interval = save_interval if save_interval is not None else config.save_interval
    eval_interval = eval_interval if eval_interval is not None else config.eval_interval

    # Set random seed
    set_seed(seed)

    # Set up directories
    if log_dir is None:
        log_dir = os.path.join(config.log_dir, f"{domain}_iql_seed{seed}")
    if save_dir is None:
        save_dir = os.path.join(config.save_dir, f"{domain}_iql_seed{seed}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize logger
    logger = Logger(log_dir, use_tensorboard=True)

    print(f"\n{'='*60}")
    print(f"Phase 2: IQL Training on {domain}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Log dir: {log_dir}")
    print(f"Save dir: {save_dir}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(domain, data_dir=data_dir, device=device, normalize=True)
    state_dim = dataset.state_dim
    action_dim = dataset.action_dim
    print(f"Dataset size: {len(dataset)} transitions")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Load or receive encoder
    if encoder is None:
        if encoder_checkpoint is None:
            raise ValueError("Must provide either encoder or encoder_checkpoint")
        print(f"\nLoading encoder from {encoder_checkpoint}...")
        from fre.training.train_encoder import load_pretrained_encoder
        encoder, _ = load_pretrained_encoder(
            encoder_checkpoint,
            state_dim=state_dim,
            device=device,
            d_latent=d_latent,
        )
    else:
        encoder = encoder.to(device)

    # Freeze encoder
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen.")

    # Create prior reward function mixture
    print("\nCreating prior reward function mixture...")
    reward_mixture = create_prior_reward_function(
        state_dim=state_dim,
        epsilon=epsilon,
        sparsity=sparsity,
        mlp_hidden_dim=mlp_hidden_dim,
        device=device,
    )

    # Initialize IQL networks
    print("\nInitializing IQL networks...")
    iql_networks = IQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        d_latent=d_latent,
        hidden_dims=hidden_dims,
        activation="relu",
        dropout=0.0,
        log_std_min=log_std_min,
        log_std_max=log_std_max,
    ).to(device)

    # Print model info
    print_model_info(encoder, None, iql_networks)

    # Set up optimizers
    # Separate optimizers for Q, V, and policy as in original IQL
    q_params = list(iql_networks.q1.parameters()) + list(iql_networks.q2.parameters())
    v_params = iql_networks.v.parameters()
    policy_params = iql_networks.policy.parameters()

    optimizer_q = optim.Adam(q_params, lr=lr)
    optimizer_v = optim.Adam(v_params, lr=lr)
    optimizer_policy = optim.Adam(policy_params, lr=lr)

    # Learning rate schedulers (cosine annealing)
    scheduler_q = optim.lr_scheduler.CosineAnnealingLR(optimizer_q, T_max=total_steps)
    scheduler_v = optim.lr_scheduler.CosineAnnealingLR(optimizer_v, T_max=total_steps)
    scheduler_policy = optim.lr_scheduler.CosineAnnealingLR(optimizer_policy, T_max=total_steps)

    # Resume from checkpoint if provided
    start_step = 0
    stats = {
        "value_losses": [],
        "q_losses": [],
        "policy_losses": [],
        "total_losses": [],
        "avg_q_values": [],
        "avg_v_values": [],
    }

    if resume_checkpoint is not None:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint_data = load_checkpoint(
            resume_checkpoint,
            encoder=encoder,
            decoder=None,
            iql_networks=iql_networks,
            optimizer_encoder=None,
            optimizer_iql={
                "optimizer_q": optimizer_q,
                "optimizer_v": optimizer_v,
                "optimizer_policy": optimizer_policy,
            },
            device=device,
        )
        start_step = checkpoint_data.get("step", 0)
        stats = checkpoint_data.get("stats", stats)
        print(f"Resumed from step {start_step}")

    # Training loop
    print(f"\nStarting IQL training from step {start_step} to {total_steps}...")
    start_time = time.time()

    pbar = tqdm(range(start_step, total_steps), desc="IQL Training", initial=start_step, total=total_steps)

    for step in pbar:
        # Sample a new reward function for this batch
        reward_mixture.reset()

        # Sample IQL batch: transitions + encode z from encoder states
        batch = sample_iql_batch(
            dataset=dataset,
            reward_fn=reward_mixture,
            encoder=encoder,
            batch_size=batch_size,
            K=K,
            device=device,
        )

        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        z = batch["z"]  # shape: (batch_size, d_latent) or (1, d_latent) broadcast

        # If z is a single vector, expand to batch
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and states.shape[0] > 1:
            z = z.expand(states.shape[0], -1)

        # --- Value Loss (Expectile Regression) ---
        # Detach Q-targets for V update
        with torch.no_grad():
            q1, q2 = iql_networks.get_q(states, actions, z)
            q_target_min = torch.min(q1, q2)

        v_loss = compute_iql_value_loss(
            v_network=iql_networks.v,
            q_target1=q_target_min.detach(),  # Use min Q as target
            q_target2=q_target_min.detach(),  # Not used in current impl but kept for API
            states=states,
            actions=actions,
            z=z,
            tau=tau,
        )

        optimizer_v.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(iql_networks.v.parameters(), max_norm=10.0)
        optimizer_v.step()

        # --- Q Loss (TD Learning) ---
        with torch.no_grad():
            v_target_next = iql_networks.v(next_states, z)

        # Compute Q loss for both Q networks
        q1_loss = compute_iql_q_loss(
            q_network=iql_networks.q1,
            v_target=v_target_next,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
            z=z,
            gamma=gamma,
        )

        q2_loss = compute_iql_q_loss(
            q_network=iql_networks.q2,
            v_target=v_target_next,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            terminals=terminals,
            z=z,
            gamma=gamma,
        )

        q_loss = q1_loss + q2_loss

        optimizer_q.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_params, max_norm=10.0)
        optimizer_q.step()

        # --- Policy Loss (Advantage-Weighted Regression) ---
        policy_loss = compute_iql_policy_loss(
            policy=iql_networks.policy,
            q_network1=iql_networks.q1,
            q_network2=iql_networks.q2,
            v_network=iql_networks.v,
            states=states,
            actions=actions,
            z=z,
            beta=beta,
            max_weight=max_weight,
        )

        optimizer_policy.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(iql_networks.policy.parameters(), max_norm=10.0)
        optimizer_policy.step()

        # --- Soft Update Target Networks ---
        iql_networks.soft_update_targets(tau=target_update_rate)

        # --- Step Schedulers ---
        scheduler_q.step()
        scheduler_v.step()
        scheduler_policy.step()

        # --- Logging ---
        total_loss = v_loss.item() + q_loss.item() + policy_loss.item()

        stats["value_losses"].append(v_loss.item())
        stats["q_losses"].append(q_loss.item())
        stats["policy_losses"].append(policy_loss.item())
        stats["total_losses"].append(total_loss)

        # Compute average Q and V values for monitoring
        with torch.no_grad():
            avg_q = (q1.mean().item() + q2.mean().item()) / 2.0
            avg_v = iql_networks.v(states, z).mean().item()
        stats["avg_q_values"].append(avg_q)
        stats["avg_v_values"].append(avg_v)

        if (step + 1) % log_interval == 0:
            logger.log_scalar("loss/value", v_loss.item(), step)
            logger.log_scalar("loss/q", q_loss.item(), step)
            logger.log_scalar("loss/policy", policy_loss.item(), step)
            logger.log_scalar("loss/total", total_loss, step)
            logger.log_scalar("stats/avg_q", avg_q, step)
            logger.log_scalar("stats/avg_v", avg_v, step)
            logger.log_scalar("stats/lr_q", scheduler_q.get_last_lr()[0], step)

        # Update progress bar
        if (step + 1) % log_interval == 0:
            pbar.set_postfix({
                "V": f"{v_loss.item():.4f}",
                "Q": f"{q_loss.item():.4f}",
                "Pi": f"{policy_loss.item():.4f}",
                "avgQ": f"{avg_q:.2f}",
            })

        # --- Save Checkpoint ---
        if (step + 1) % save_interval == 0 or (step + 1) == total_steps:
            checkpoint_path = os.path.join(save_dir, f"iql_checkpoint_{step+1}.pt")
            save_checkpoint(
                save_dir=save_dir,
                filename=f"iql_checkpoint_{step+1}.pt",
                encoder=encoder,
                decoder=None,
                iql_networks=iql_networks,
                optimizer_encoder=None,
                optimizer_iql={
                    "optimizer_q": optimizer_q,
                    "optimizer_v": optimizer_v,
                    "optimizer_policy": optimizer_policy,
                },
                step=step + 1,
                extra_info={
                    "stats": stats,
                    "domain": domain,
                    "seed": seed,
                },
            )
            print(f"\nCheckpoint saved at step {step+1}")

        # --- Evaluation (on training reward functions) ---
        if (step + 1) % eval_interval == 0:
            eval_metrics = evaluate_iql_training(
                iql_networks=iql_networks,
                encoder=encoder,
                dataset=dataset,
                reward_mixture=reward_mixture,
                K=K,
                num_eval=10,
                device=device,
            )
            logger.log_scalar("eval/avg_return", eval_metrics["avg_return"], step)
            logger.log_scalar("eval/avg_q_value", eval_metrics["avg_q_value"], step)
            print(f"\nEval at step {step+1}: avg_return={eval_metrics['avg_return']:.4f}, "
                  f"avg_q={eval_metrics['avg_q_value']:.4f}")

    # Training complete
    total_time = time.time() - start_time
    print(f"\nIQL training completed in {total_time:.1f}s ({total_time/3600:.2f}h)")

    # Save final model
    final_path = os.path.join(save_dir, "iql_final.pt")
    save_checkpoint(
        save_dir=save_dir,
        filename="iql_final.pt",
        encoder=encoder,
        decoder=None,
        iql_networks=iql_networks,
        optimizer_encoder=None,
        optimizer_iql={
            "optimizer_q": optimizer_q,
            "optimizer_v": optimizer_v,
            "optimizer_policy": optimizer_policy,
        },
        step=total_steps,
        extra_info={
            "stats": stats,
            "domain": domain,
            "seed": seed,
        },
    )
    print(f"Final model saved to {final_path}")

    logger.close()

    return iql_networks, stats


def evaluate_iql_training(
    iql_networks: IQLNetworks,
    encoder: FREEncoder,
    dataset: OfflineDataset,
    reward_mixture: MixtureRewardFunction,
    K: int = 32,
    num_eval: int = 10,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate IQL networks on random reward functions from the prior mixture.
    This is an offline evaluation using the dataset (no environment interaction).

    Computes the average Q-value for states in the dataset under the encoded
    reward function, which serves as a proxy for policy performance.

    Args:
        iql_networks: Trained IQL networks
        encoder: Frozen FRE encoder
        dataset: Offline dataset
        reward_mixture: Prior reward function mixture
        K: Number of encoder states
        num_eval: Number of evaluation reward functions
        device: Device

    Returns:
        Dict with avg_return (proxy) and avg_q_value
    """
    device = get_device(device)
    iql_networks.eval()
    encoder.eval()

    all_q_values = []
    all_returns = []

    # Get a fixed set of evaluation states from the dataset
    eval_states = dataset.sample_states(1000)

    for _ in range(num_eval):
        # Sample a new reward function
        reward_mixture.reset()

        # Sample encoder states and encode
        encoder_states = dataset.sample_states(K)
        with torch.no_grad():
            encoder_rewards = reward_mixture(encoder_states)
            z = encoder.encode_deterministic(
                encoder_states.unsqueeze(0),
                encoder_rewards.unsqueeze(0),
            )  # shape: (1, d_latent)

        # Expand z for batch evaluation
        z_expanded = z.expand(eval_states.shape[0], -1)

        # Compute Q-values and rewards for eval states
        with torch.no_grad():
            # Get actions from policy
            actions, _, _ = iql_networks.policy(eval_states, z_expanded, deterministic=True)
            q1, q2 = iql_networks.get_q(eval_states, actions, z_expanded)
            q_values = torch.min(q1, q2)
            rewards = reward_mixture(eval_states)

        all_q_values.append(q_values.mean().item())
        all_returns.append(rewards.mean().item())

    iql_networks.train()

    return {
        "avg_q_value": np.mean(all_q_values),
        "std_q_value": np.std(all_q_values),
        "avg_return": np.mean(all_returns),
        "std_return": np.std(all_returns),
    }


def load_pretrained_iql(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    device: Optional[str] = None,
    d_latent: Optional[int] = None,
    hidden_dims: Optional[list] = None,
) -> Tuple[IQLNetworks, FREEncoder]:
    """
    Load a pretrained IQL agent (encoder + IQL networks) from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        state_dim: State dimension
        action_dim: Action dimension
        device: Device
        d_latent: Latent dimension
        hidden_dims: Hidden layer dimensions

    Returns:
        Tuple of (IQLNetworks, FREEncoder)
    """
    device = get_device(device)
    d_latent = d_latent if d_latent is not None else config.d_latent
    hidden_dims = hidden_dims if hidden_dims is not None else config.iql_hidden_dims

    # Initialize encoder
    encoder = FREEncoder(
        state_dim=state_dim,
        d_embed=config.d_embed,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_latent=d_latent,
        num_reward_bins=config.num_reward_bins,
        r_max=config.r_max,
        dropout=config.dropout,
    ).to(device)

    # Initialize IQL networks
    iql_networks = IQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        d_latent=d_latent,
        hidden_dims=hidden_dims,
    ).to(device)

    # Load checkpoint
    checkpoint_data = load_checkpoint(
        checkpoint_path,
        encoder=encoder,
        decoder=None,
        iql_networks=iql_networks,
        optimizer_encoder=None,
        optimizer_iql=None,
        device=device,
    )

    print(f"Loaded IQL checkpoint from {checkpoint_path}")
    print(f"  Step: {checkpoint_data.get('step', 'unknown')}")

    encoder.eval()
    iql_networks.eval()

    return iql_networks, encoder


# ============================================================
# Quick Test
# ============================================================

def test_iql_training():
    """
    Quick test to verify IQL training loop runs without errors.
    Uses a small synthetic dataset.
    """
    print("Testing IQL training...")

    device = torch.device("cpu")
    set_seed(42)

    # Create synthetic dataset
    state_dim = 8
    action_dim = 2
    num_samples = 1000

    observations = torch.randn(num_samples, state_dim)
    actions = torch.randn(num_samples, action_dim)
    next_observations = torch.randn(num_samples, state_dim)
    rewards = torch.randn(num_samples)
    terminals = torch.zeros(num_samples)

    from fre.data.dataset import ReplayBuffer, OfflineDataset

    buffer = ReplayBuffer(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        timeouts=None,
        device=device,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Create a simple dataset wrapper
    class SimpleDataset:
        def __init__(self, buffer):
            self.buffer = buffer
            self.state_dim = state_dim
            self.action_dim = action_dim

        def sample_states(self, n):
            return self.buffer.sample_states(n)

        def sample_transitions(self, n):
            return self.buffer.sample_transitions(n)

        def sample_encoder_decoder_states(self, K, K_prime):
            return self.buffer.sample_encoder_decoder_states(K, K_prime)

        def __len__(self):
            return len(self.buffer)

    dataset = SimpleDataset(buffer)

    # Create encoder
    d_latent = 16
    encoder = FREEncoder(
        state_dim=state_dim,
        d_embed=32,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_latent=d_latent,
        num_reward_bins=32,
        r_max=5.0,
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Create reward mixture
    reward_mixture = create_prior_reward_function(
        state_dim=state_dim,
        epsilon=0.5,
        sparsity=0.8,
        mlp_hidden_dim=64,
        device=device,
    )

    # Create IQL networks
    iql_networks = IQLNetworks(
        state_dim=state_dim,
        action_dim=action_dim,
        d_latent=d_latent,
        hidden_dims=[64, 64],
    ).to(device)

    # Quick training loop (just a few steps)
    K = 8
    batch_size = 32
    gamma = 0.99
    tau_val = 0.7
    beta_val = 3.0
    target_update_rate = 0.005
    max_weight = 100.0

    optimizer_q = optim.Adam(
        list(iql_networks.q1.parameters()) + list(iql_networks.q2.parameters()),
        lr=3e-4,
    )
    optimizer_v = optim.Adam(iql_networks.v.parameters(), lr=3e-4)
    optimizer_policy = optim.Adam(iql_networks.policy.parameters(), lr=3e-4)

    for step in range(20):
        reward_mixture.reset()

        # Sample batch
        batch = sample_iql_batch(
            dataset=dataset,
            reward_fn=reward_mixture,
            encoder=encoder,
            batch_size=batch_size,
            K=K,
            device=device,
        )

        states = batch["states"]
        actions = batch["actions"]
        next_states = batch["next_states"]
        rewards_batch = batch["rewards"]
        terminals_batch = batch["terminals"]
        z = batch["z"]

        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] == 1 and states.shape[0] > 1:
            z = z.expand(states.shape[0], -1)

        # Value loss
        with torch.no_grad():
            q1, q2 = iql_networks.get_q(states, actions, z)
            q_target_min = torch.min(q1, q2)

        v_loss = compute_iql_value_loss(
            v_network=iql_networks.v,
            q_target1=q_target_min.detach(),
            q_target2=q_target_min.detach(),
            states=states,
            actions=actions,
            z=z,
            tau=tau_val,
        )

        optimizer_v.zero_grad()
        v_loss.backward()
        optimizer_v.step()

        # Q loss
        with torch.no_grad():
            v_target_next = iql_networks.v(next_states, z)

        q1_loss = compute_iql_q_loss(
            q_network=iql_networks.q1,
            v_target=v_target_next,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards_batch,
            terminals=terminals_batch,
            z=z,
            gamma=gamma,
        )
        q2_loss = compute_iql_q_loss(
            q_network=iql_networks.q2,
            v_target=v_target_next,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards_batch,
            terminals=terminals_batch,
            z=z,
            gamma=gamma,
        )
        q_loss = q1_loss + q2_loss

        optimizer_q.zero_grad()
        q_loss.backward()
        optimizer_q.step()

        # Policy loss
        policy_loss = compute_iql_policy_loss(
            policy=iql_networks.policy,
            q_network1=iql_networks.q1,
            q_network2=iql_networks.q2,
            v_network=iql_networks.v,
            states=states,
            actions=actions,
            z=z,
            beta=beta_val,
            max_weight=max_weight,
        )

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Soft update
        iql_networks.soft_update_targets(tau=target_update_rate)

        if step % 5 == 0:
            print(f"  Step {step}: V={v_loss.item():.4f}, Q={q_loss.item():.4f}, "
                  f"Pi={policy_loss.item():.4f}")

    print("IQL training test PASSED!")
    return True


if __name__ == "__main__":
    test_iql_training()