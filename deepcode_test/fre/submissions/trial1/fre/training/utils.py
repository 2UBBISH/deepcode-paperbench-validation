"""
Training utilities for FRE: data sampling, reward computation, logging, and helpers.

Provides:
- Sampling of encoder/decoder state sets with reward computation
- Batch sampling for IQL training with z-conditioning
- Logging utilities (TensorBoard, console)
- Training loop helpers (progress tracking, checkpointing)
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, Any
from torch.utils.tensorboard import SummaryWriter

from fre.config import config
from fre.data.dataset import OfflineDataset, ReplayBuffer
from fre.reward_functions.mixture import MixtureRewardFunction
from fre.models.encoder import FREEncoder
from fre.models.decoder import RewardDecoder
from fre.models.iql import IQLNetworks


# ============================================================
# Logging Utilities
# ============================================================

class Logger:
    """Simple logger with TensorBoard and console output."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
        self.metrics_history: Dict[str, list] = {}
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
        if tag not in self.metrics_history:
            self.metrics_history[tag] = []
        self.metrics_history[tag].append((step, value))
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under a main tag."""
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close the logger."""
        if self.writer is not None:
            self.writer.close()
    
    def print_progress(self, step: int, total_steps: int, metrics: Dict[str, float], 
                       prefix: str = "", interval: int = 1000):
        """Print progress to console at regular intervals."""
        if step % interval == 0:
            elapsed = time.time() - getattr(self, '_start_time', time.time())
            self._start_time = time.time()
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            progress = 100.0 * step / total_steps if total_steps > 0 else 0.0
            print(f"[{prefix}] Step {step}/{total_steps} ({progress:.1f}%) | {metric_str} | {elapsed:.1f}s")


# ============================================================
# Reward Function Sampling
# ============================================================

def create_prior_reward_function(
    state_dim: int,
    epsilon: float = 0.5,
    sparsity: float = 0.8,
    mlp_hidden_dim: int = 256,
    device: Optional[str] = None
) -> MixtureRewardFunction:
    """
    Create the prior reward function mixture (uniform over singleton, linear, MLP).
    
    Args:
        state_dim: Dimensionality of the state space.
        epsilon: Threshold for singleton goal-reaching reward.
        sparsity: Fraction of dimensions zeroed out in linear reward.
        mlp_hidden_dim: Hidden dimension for random MLP reward.
        device: Device to place the reward function on.
    
    Returns:
        MixtureRewardFunction instance.
    """
    if device is None:
        device = config.device
    
    mixture = MixtureRewardFunction(
        state_dim=state_dim,
        epsilon=epsilon,
        sparsity=sparsity,
        mlp_hidden_dim=mlp_hidden_dim,
        device=device
    )
    return mixture


def sample_reward_function(mixture: MixtureRewardFunction) -> str:
    """
    Sample a new reward function from the mixture (calls reset).
    
    Returns the type of the sampled reward function.
    """
    mixture.reset()
    return mixture.current_type


# ============================================================
# Encoder Training Data Sampling
# ============================================================

def sample_encoder_batch(
    dataset: OfflineDataset,
    reward_fn: MixtureRewardFunction,
    K: int = 32,
    K_prime: int = 32,
    device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Sample a batch for encoder training: K encoder states + K' decoder states,
    compute rewards using the given reward function.
    
    Args:
        dataset: OfflineDataset instance.
        reward_fn: MixtureRewardFunction (already reset to a specific type).
        K: Number of encoder states.
        K_prime: Number of decoder states.
        device: Device for tensors.
    
    Returns:
        Dict with keys:
            - 'encoder_states': (K, state_dim)
            - 'encoder_rewards': (K,)
            - 'decoder_states': (K_prime, state_dim)
            - 'decoder_rewards': (K_prime,)
    """
    if device is None:
        device = config.device
    
    # Sample disjoint encoder and decoder states
    enc_states, dec_states = dataset.sample_encoder_decoder_states(K, K_prime)
    
    # Compute rewards
    enc_rewards = reward_fn(enc_states)
    dec_rewards = reward_fn(dec_states)
    
    return {
        'encoder_states': enc_states,
        'encoder_rewards': enc_rewards,
        'decoder_states': dec_states,
        'decoder_rewards': dec_rewards,
    }


# ============================================================
# IQL Training Data Sampling
# ============================================================

def sample_iql_batch(
    dataset: OfflineDataset,
    reward_fn: MixtureRewardFunction,
    encoder: FREEncoder,
    batch_size: int = 256,
    K: int = 32,
    device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Sample a batch for IQL training: transitions + latent z from encoder.
    
    Process:
    1. Sample a batch of transitions (s, a, s') from the dataset.
    2. Sample K encoder states, compute rewards using reward_fn.
    3. Encode to get latent z.
    4. Compute rewards for the batch transitions using reward_fn.
    
    Args:
        dataset: OfflineDataset instance.
        reward_fn: MixtureRewardFunction (already reset).
        encoder: FREEncoder (frozen, in eval mode).
        batch_size: Number of transitions to sample.
        K: Number of encoder states for z computation.
        device: Device for tensors.
    
    Returns:
        Dict with keys:
            - 'states': (batch_size, state_dim)
            - 'actions': (batch_size, action_dim)
            - 'next_states': (batch_size, state_dim)
            - 'rewards': (batch_size,)
            - 'next_rewards': (batch_size,)
            - 'terminals': (batch_size,)
            - 'z': (batch_size, d_latent) - same z for all transitions in batch
            - 'mu': (1, d_latent)
            - 'logvar': (1, d_latent)
    """
    if device is None:
        device = config.device
    
    # Sample transitions
    transitions = dataset.sample_transitions(batch_size)
    states = transitions['observations']
    actions = transitions['actions']
    next_states = transitions['next_observations']
    terminals = transitions['terminals']
    
    # Sample encoder states and compute rewards
    enc_states = dataset.sample_states(K)
    enc_rewards = reward_fn(enc_states)
    
    # Encode to get z (use deterministic encoding for training stability)
    with torch.no_grad():
        # Add batch dimension for encoder: (1, K, state_dim), (1, K)
        enc_states_batched = enc_states.unsqueeze(0)  # (1, K, state_dim)
        enc_rewards_batched = enc_rewards.unsqueeze(0)  # (1, K)
        z, mu, logvar = encoder(enc_states_batched, enc_rewards_batched, deterministic=True)
        # z: (1, d_latent), expand to batch
        z_batch = z.expand(batch_size, -1)  # (batch_size, d_latent)
    
    # Compute rewards for transitions
    rewards = reward_fn(states)
    next_rewards = reward_fn(next_states)
    
    return {
        'states': states,
        'actions': actions,
        'next_states': next_states,
        'rewards': rewards,
        'next_rewards': next_rewards,
        'terminals': terminals,
        'z': z_batch,
        'mu': mu,
        'logvar': logvar,
    }


# ============================================================
# VAE Loss Computation
# ============================================================

def compute_vae_loss(
    decoder: RewardDecoder,
    decoder_states: torch.Tensor,
    decoder_rewards: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the VAE loss: reconstruction MSE + beta * KL divergence.
    
    Args:
        decoder: RewardDecoder network.
        decoder_states: (K', state_dim) - decoder states.
        decoder_rewards: (K',) - true rewards for decoder states.
        z: (1, d_latent) - sampled latent vector.
        mu: (1, d_latent) - mean from encoder.
        logvar: (1, d_latent) - log variance from encoder.
        beta_kl: Weight for KL divergence term.
    
    Returns:
        Tuple of (total_loss, recon_loss, kl_loss).
    """
    # Reconstruction loss
    # decoder_states: (K', state_dim), z: (1, d_latent)
    # Need to expand z to match K' states
    z_expanded = z.expand(decoder_states.shape[0], -1)  # (K', d_latent)
    pred_rewards = decoder(decoder_states, z_expanded).squeeze(-1)  # (K',)
    
    recon_loss = torch.mean((pred_rewards - decoder_rewards) ** 2)
    
    # KL divergence: KL(N(mu, sigma^2) || N(0, I))
    # = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    total_loss = recon_loss + beta_kl * kl_loss
    
    return total_loss, recon_loss, kl_loss


# ============================================================
# IQL Loss Functions
# ============================================================

def expectile_loss(diff: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """
    Compute the expectile loss: L2_tau(u) = |tau - 1(u<0)| * u^2.
    
    Args:
        diff: Tensor of differences (Q - V).
        tau: Expectile parameter (0.5 = mean, >0.5 gives upper expectile).
    
    Returns:
        Scalar loss.
    """
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * (diff ** 2)).mean()


def compute_iql_value_loss(
    v_network: nn.Module,
    q_target1: nn.Module,
    q_target2: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    z: torch.Tensor,
    tau: float = 0.7
) -> torch.Tensor:
    """
    Compute the IQL value loss (expectile regression).
    
    L_V = E[ L2_tau( Q_target(s,a,z) - V(s,z) ) ]
    
    Args:
        v_network: Value network V(s, z).
        q_target1, q_target2: Target Q networks.
        states: (B, state_dim).
        actions: (B, action_dim).
        z: (B, d_latent).
        tau: Expectile parameter.
    
    Returns:
        Scalar value loss.
    """
    with torch.no_grad():
        q1 = q_target1(states, actions, z)
        q2 = q_target2(states, actions, z)
        q_target = torch.min(q1, q2)
    
    v_pred = v_network(states, z)
    diff = q_target - v_pred
    return expectile_loss(diff, tau)


def compute_iql_q_loss(
    q_network: nn.Module,
    v_target: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    z: torch.Tensor,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute the IQL Q loss (standard TD loss).
    
    L_Q = E[ ( r + gamma * V(s', z) * (1 - terminal) - Q(s, a, z) )^2 ]
    
    Args:
        q_network: Q network.
        v_target: Target value network.
        states: (B, state_dim).
        actions: (B, action_dim).
        next_states: (B, state_dim).
        rewards: (B,).
        terminals: (B,).
        z: (B, d_latent).
        gamma: Discount factor.
    
    Returns:
        Scalar Q loss.
    """
    with torch.no_grad():
        v_next = v_target(next_states, z)
        target = rewards + gamma * v_next * (1.0 - terminals.float())
    
    q_pred = q_network(states, actions, z)
    return torch.mean((q_pred - target) ** 2)


def compute_iql_policy_loss(
    policy: nn.Module,
    q_network1: nn.Module,
    q_network2: nn.Module,
    v_network: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    z: torch.Tensor,
    beta: float = 3.0,
    max_weight: float = 100.0
) -> torch.Tensor:
    """
    Compute the IQL policy loss (advantage-weighted regression).
    
    L_pi = -E[ exp(beta * A(s,a,z)) * log pi(a|s,z) ]
    where A(s,a,z) = Q(s,a,z) - V(s,z)
    
    Args:
        policy: GaussianPolicy.
        q_network1, q_network2: Q networks.
        v_network: Value network.
        states: (B, state_dim).
        actions: (B, action_dim) - actions from dataset.
        z: (B, d_latent).
        beta: Temperature for advantage weighting.
        max_weight: Maximum weight for clipping.
    
    Returns:
        Scalar policy loss.
    """
    with torch.no_grad():
        q1 = q_network1(states, actions, z)
        q2 = q_network2(states, actions, z)
        q = torch.min(q1, q2)
        v = v_network(states, z)
        advantage = q - v
        weights = torch.exp(beta * advantage)
        weights = torch.clamp(weights, max=max_weight)
    
    # Compute log probability of dataset actions under current policy
    log_probs = policy.log_prob(states, actions, z)
    
    # Weighted negative log-likelihood
    policy_loss = -(weights * log_probs).mean()
    
    return policy_loss


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
    save_dir: str,
    filename: str,
    encoder: Optional[FREEncoder] = None,
    decoder: Optional[RewardDecoder] = None,
    iql_networks: Optional[IQLNetworks] = None,
    optimizer_encoder: Optional[torch.optim.Optimizer] = None,
    optimizer_iql: Optional[Dict[str, torch.optim.Optimizer]] = None,
    step: int = 0,
    extra_info: Optional[Dict[str, Any]] = None
):
    """
    Save a training checkpoint.
    
    Args:
        save_dir: Directory to save checkpoint.
        filename: Checkpoint filename.
        encoder: FREEncoder model.
        decoder: RewardDecoder model.
        iql_networks: IQLNetworks container.
        optimizer_encoder: Optimizer for encoder+decoder.
        optimizer_iql: Dict of optimizers for IQL components.
        step: Current training step.
        extra_info: Any additional info to save.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {'step': step}
    
    if encoder is not None:
        checkpoint['encoder_state_dict'] = encoder.state_dict()
    if decoder is not None:
        checkpoint['decoder_state_dict'] = decoder.state_dict()
    if iql_networks is not None:
        checkpoint['iql_state_dict'] = {
            'q1': iql_networks.q1.state_dict(),
            'q2': iql_networks.q2.state_dict(),
            'q1_target': iql_networks.q1_target.state_dict(),
            'q2_target': iql_networks.q2_target.state_dict(),
            'v': iql_networks.v.state_dict(),
            'policy': iql_networks.policy.state_dict(),
        }
    if optimizer_encoder is not None:
        checkpoint['optimizer_encoder_state_dict'] = optimizer_encoder.state_dict()
    if optimizer_iql is not None:
        checkpoint['optimizer_iql_state_dict'] = {
            k: opt.state_dict() for k, opt in optimizer_iql.items()
        }
    if extra_info is not None:
        checkpoint['extra_info'] = extra_info
    
    save_path = os.path.join(save_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    load_path: str,
    encoder: Optional[FREEncoder] = None,
    decoder: Optional[RewardDecoder] = None,
    iql_networks: Optional[IQLNetworks] = None,
    optimizer_encoder: Optional[torch.optim.Optimizer] = None,
    optimizer_iql: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        load_path: Path to checkpoint file.
        encoder: FREEncoder to load state into.
        decoder: RewardDecoder to load state into.
        iql_networks: IQLNetworks to load state into.
        optimizer_encoder: Optimizer to load state into.
        optimizer_iql: Dict of optimizers to load state into.
        device: Device to map tensors to.
    
    Returns:
        Dict with loaded info (step, extra_info).
    """
    if device is None:
        device = config.device
    
    checkpoint = torch.load(load_path, map_location=device)
    
    if encoder is not None and 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    if decoder is not None and 'decoder_state_dict' in checkpoint:
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if iql_networks is not None and 'iql_state_dict' in checkpoint:
        iql_networks.q1.load_state_dict(checkpoint['iql_state_dict']['q1'])
        iql_networks.q2.load_state_dict(checkpoint['iql_state_dict']['q2'])
        iql_networks.q1_target.load_state_dict(checkpoint['iql_state_dict']['q1_target'])
        iql_networks.q2_target.load_state_dict(checkpoint['iql_state_dict']['q2_target'])
        iql_networks.v.load_state_dict(checkpoint['iql_state_dict']['v'])
        iql_networks.policy.load_state_dict(checkpoint['iql_state_dict']['policy'])
    if optimizer_encoder is not None and 'optimizer_encoder_state_dict' in checkpoint:
        optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
    if optimizer_iql is not None and 'optimizer_iql_state_dict' in checkpoint:
        for k, opt in optimizer_iql.items():
            if k in checkpoint['optimizer_iql_state_dict']:
                opt.load_state_dict(checkpoint['optimizer_iql_state_dict'][k])
    
    return {
        'step': checkpoint.get('step', 0),
        'extra_info': checkpoint.get('extra_info', {}),
    }


# ============================================================
# Utility Functions
# ============================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get torch device from string or config."""
    if device_str is None:
        device_str = config.device
    if device_str == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device_str)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(encoder: FREEncoder, decoder: RewardDecoder, iql: IQLNetworks):
    """Print parameter counts for all models."""
    print(f"Encoder parameters: {count_parameters(encoder):,}")
    print(f"Decoder parameters: {count_parameters(decoder):,}")
    print(f"IQL total parameters: {count_parameters(iql):,}")