"""Utility functions for SAPG / PPO.

Contains:
    * Generalized Advantage Estimation (GAE) with gamma=0.99, tau=0.95.
    * KL-based adaptive learning-rate schedule (threshold 0.016).
    * Gradient clipping helper (max norm 1.0).
    * ELU activation helper.
    * Running value normalizer (returns / value normalization).
    * Small helpers for config access and tensor conversion.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def get_activation(name: str = "elu") -> nn.Module:
    """Return an activation module by name. Defaults to ELU (paper uses ELU)."""
    name = (name or "elu").lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    raise ValueError(f"Unknown activation: {name}")


def elu(x: torch.Tensor) -> torch.Tensor:
    """Functional ELU activation."""
    return torch.nn.functional.elu(x)


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    tau: float = 0.95,
    last_values: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-lambda advantages and returns.

    Args:
        rewards: (T, N) rewards.
        values: (T, N) value estimates V(s_t).
        dones: (T, N) float/bool flags, 1.0 if episode terminated at t.
        gamma: discount factor (0.99).
        tau: GAE lambda (0.95).
        last_values: (N,) bootstrapped value V(s_{T}) for the step after the
            rollout. If None, treated as zeros.

    Returns:
        advantages: (T, N) GAE advantages.
        returns: (T, N) = advantages + values (targets for the value function).
    """
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if dones.dim() == 1:
        dones = dones.unsqueeze(-1)

    T, N = rewards.shape
    device = rewards.device
    dones = dones.float()

    if last_values is None:
        last_values = torch.zeros(N, device=device, dtype=values.dtype)
    else:
        last_values = last_values.to(device=device, dtype=values.dtype).reshape(N)

    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_values
            next_nonterminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * tau * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# KL-based adaptive learning rate
# ---------------------------------------------------------------------------
class AdaptiveKLLR:
    """Adaptive learning-rate schedule based on measured KL divergence.

    If KL > 2 * kl_target, halve the learning rate.
    If KL < kl_target / 2, double the learning rate.
    Otherwise keep it unchanged.

    The paper uses a KL threshold of 0.016.
    """

    def __init__(
        self,
        init_lr: float,
        kl_target: float = 0.016,
        min_lr: float = 1e-6,
        max_lr: float = 1e-2,
        factor: float = 2.0,
    ):
        self.init_lr = float(init_lr)
        self.lr = float(init_lr)
        self.kl_target = float(kl_target)
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.factor = float(factor)

    def update(self, kl: float) -> float:
        """Update the learning rate given the measured KL and return the new LR."""
        kl = float(kl)
        if kl > 2.0 * self.kl_target:
            self.lr = max(self.lr / self.factor, self.min_lr)
        elif kl < 0.5 * self.kl_target:
            self.lr = min(self.lr * self.factor, self.max_lr)
        return self.lr

    def state_dict(self) -> dict:
        return {"lr": self.lr, "init_lr": self.init_lr, "kl_target": self.kl_target}

    def load_state_dict(self, state: dict) -> None:
        self.lr = state.get("lr", self.init_lr)
        self.init_lr = state.get("init_lr", self.init_lr)
        self.kl_target = state.get("kl_target", self.kl_target)


def compute_kl(mu_old: torch.Tensor, logstd_old: torch.Tensor,
               mu_new: torch.Tensor, logstd_new: torch.Tensor) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians (mean over batch).

    KL(N_old || N_new) = sum_d [ log(sigma_new/sigma_old)
                                 + (sigma_old^2 + (mu_old-mu_new)^2) / (2 sigma_new^2)
                                 - 0.5 ]
    """
    var_old = torch.exp(2.0 * logstd_old)
    var_new = torch.exp(2.0 * logstd_new)
    kl = (logstd_new - logstd_old) + (var_old + (mu_old - mu_new) ** 2) / (2.0 * var_new) - 0.5
    return kl.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------
def clip_grad_norm_(parameters, max_norm: float = 1.0) -> float:
    """Clip gradients of an iterable of parameters in place. Returns total norm."""
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p is not None and p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    return float(total_norm)


# ---------------------------------------------------------------------------
# Value / return normalization
# ---------------------------------------------------------------------------
class RunningMeanStd:
    """Tracks a running mean/std (Welford's algorithm) for value normalization."""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = float(epsilon)

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False) if x.dim() > 0 else torch.tensor(0.0)
        batch_count = x.shape[0] if x.dim() > 0 else 1

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var + 1e-8)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.var = state["var"]
        self.count = state["count"]


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------
def _cfg_getter(cfg):
    """Return a get(key, default) callable for dicts or attribute objects."""
    if cfg is None:
        return lambda key, default=None: default
    if isinstance(cfg, dict):
        return lambda key, default=None: cfg.get(key, default)

    def _get(key, default=None):
        return getattr(cfg, key, default)

    return _get


def to_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    """Convert numpy/list/scalar to a torch tensor on the given device."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Explained variance of value predictions (1 - Var(y - pred)/Var(y))."""
    y_pred = y_pred.reshape(-1).float()
    y_true = y_true.reshape(-1).float()
    var_y = torch.var(y_true)
    if var_y == 0:
        return float("nan")
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


def discount_cumsum(x: torch.Tensor, discount: float) -> torch.Tensor:
    """Discounted cumulative sum along dim 0 (used for reward-to-go)."""
    T = x.shape[0]
    out = torch.zeros_like(x)
    running = torch.zeros_like(x[0])
    for t in reversed(range(T)):
        running = x[t] + discount * running
        out[t] = running
    return out


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Apply orthogonal initialization to Linear/LSTM layers in a module."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param, gain=gain)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
