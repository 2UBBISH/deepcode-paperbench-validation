"""PPO core utilities for SAPG.

Implements the clipped surrogate objective, Generalized Advantage Estimation
(GAE), adaptive KL-based learning-rate scheduling, and the loss assembly used
by both the follower update and the leader aggregation update.

Reference: SAPG: Split and Aggregate Policy Gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    tau: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-lambda advantages and returns.

    Shapes are expected to be ``(T, N)`` for rewards/values/dones where ``T`` is
    the rollout horizon and ``N`` is the number of parallel environments (or
    transitions). ``next_value`` has shape ``(N,)``.

    Returns
    -------
    advantages : (T, N) tensor
    returns : (T, N) tensor  (advantages + values)
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
        values = values.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_values = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * tau * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Clipped surrogate objective
# ---------------------------------------------------------------------------
def clipped_surrogate_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped surrogate objective.

    L = -mean( min(r * A, clip(r, 1-eps, 1+eps) * A) )

    Returns
    -------
    loss : scalar tensor (to be minimized)
    ratio : (N,) tensor of importance ratios
    clip_fraction : scalar fraction of clipped ratios (diagnostic)
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()
    with torch.no_grad():
        clip_fraction = (torch.abs(ratio - 1.0) > clip_eps).float().mean()
    return loss, ratio, clip_fraction


# ---------------------------------------------------------------------------
# Adaptive KL learning-rate scheduler
# ---------------------------------------------------------------------------
class KLScheduler:
    """Adaptive learning-rate schedule based on measured KL divergence.

    If the measured KL exceeds ``kl_threshold`` the LR is reduced; if it is
    below half the threshold the LR is increased. This mirrors the adaptive
    schedule used in the SAPG/IsaacGym PPO implementations.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        kl_threshold: float = 0.016,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        scale: float = 1.5,
    ) -> None:
        self.optimizer = optimizer
        self.kl_threshold = kl_threshold
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.scale = scale

    def step(self, kl: float) -> float:
        """Update the optimizer LR given the measured KL. Returns the new LR."""
        if kl > self.kl_threshold * 2.0:
            factor = 1.0 / self.scale
        elif kl < self.kl_threshold / 2.0:
            factor = self.scale
        else:
            factor = 1.0

        new_lrs = []
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"] * factor
            lr = max(self.min_lr, min(self.max_lr, lr))
            param_group["lr"] = lr
            new_lrs.append(lr)
        return new_lrs[0] if new_lrs else 0.0


# ---------------------------------------------------------------------------
# Full PPO loss assembly
# ---------------------------------------------------------------------------
@dataclass
class PPOHyperParams:
    """Container for PPO hyperparameters (defaults follow the paper)."""

    gamma: float = 0.99
    tau: float = 0.95
    clip_eps: float = 0.1
    entropy_coeff: float = 0.0
    critic_coeff: float = 4.0
    bounds_loss_coeff: float = 1e-4
    kl_threshold: float = 0.016
    grad_norm_clip: float = 1.0
    lr: float = 1e-4
    mini_epochs: int = 2
    horizon: int = 16
    lstm_seq_len: int = 16
    action_bound: float = 1.0
    extra: Dict = field(default_factory=dict)


def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    actions: torch.Tensor,
    clip_eps: float = 0.1,
    entropy_coeff: float = 0.0,
    critic_coeff: float = 4.0,
    bounds_loss_coeff: float = 1e-4,
    action_bound: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Assemble the full PPO loss.

    total = policy_loss + critic_coeff * value_loss
            - entropy_coeff * entropy + bounds_loss_coeff * bounds_loss

    The ``bounds_loss`` penalizes actions that exceed the action bound (used by
    the IsaacGym-based tasks where actions are clipped to [-1, 1]).
    """
    policy_loss, ratio, clip_fraction = clipped_surrogate_loss(
        new_log_probs, old_log_probs, advantages, clip_eps
    )

    value_loss = 0.5 * (values - returns).pow(2).mean()

    entropy_mean = entropy.mean()

    # Bounds loss: penalize actions outside [-action_bound, action_bound].
    bounds_loss = torch.clamp(actions.abs() - action_bound, min=0.0).pow(2).mean()

    total_loss = (
        policy_loss
        + critic_coeff * value_loss
        - entropy_coeff * entropy_mean
        + bounds_loss_coeff * bounds_loss
    )

    with torch.no_grad():
        approx_kl = (old_log_probs - new_log_probs).mean()
        clip_frac = clip_fraction

    info = {
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy_mean.detach().cpu()),
        "bounds_loss": float(bounds_loss.detach().cpu()),
        "approx_kl": float(approx_kl.cpu()),
        "clip_fraction": float(clip_frac.cpu()),
        "total_loss": float(total_loss.detach().cpu()),
    }
    return total_loss, info


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize advantages (zero mean, unit std) over the batch."""
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of variance in ``y_true`` explained by ``y_pred``."""
    y_var = torch.var(y_true)
    if y_var.item() < 1e-8:
        return float("nan")
    return float((1.0 - torch.var(y_true - y_pred) / y_var).cpu())


def clip_grad_norm_(parameters, max_norm: float = 1.0) -> float:
    """Clip gradients by global norm; returns the pre-clip total norm."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(total_norm)
