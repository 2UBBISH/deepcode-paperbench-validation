"""Vanilla PPO baseline for comparison with SAPG.

This module implements a standard single-policy PPO (Schulman et al., 2017)
that trains one policy across *all* parallel environments.  It is used as the
baseline in the batch-size sweep (Figure 2) and the hard/easy task comparisons.

The key difference from SAPG is that PPO uses a single policy for all
environments, so at very large batch sizes the data is highly correlated and
the effective number of gradient updates per sample is small, causing
asymptotic performance to saturate.  SAPG instead splits environments into
blocks with separate follower policies and aggregates the resulting
(decorrelated) data for the leader update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .policy import GaussianPolicy, build_policy
from .rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PPOConfig:
    """Hyperparameters for vanilla PPO.

    Defaults follow the SAPG paper's shared hyperparameters (Tables 2-4):
    gamma=0.99, tau=0.95, clip_eps=0.2, KL threshold 0.016, etc.
    """

    num_envs: int = 4096
    horizon: int = 16
    gamma: float = 0.99
    tau: float = 0.95
    clip_eps: float = 0.2
    value_loss_coef: float = 1.0
    bounds_loss_coef: float = 0.001
    entropy_coef: float = 0.0
    max_grad_norm: float = 1.0
    lr: float = 3e-4
    kl_threshold: float = 0.016
    lr_adapt: bool = True
    num_mini_epochs: int = 2
    mini_batch_multiplier: int = 4
    normalize_advantages: bool = True
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flatten_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Flatten (B, T, ...) tensors to (B*T, ...) for network forward passes."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2:
            out[k] = v.reshape(-1, *v.shape[2:])
        else:
            out[k] = v
    return out


def _compute_bounds_loss(policy: GaussianPolicy) -> torch.Tensor:
    """Soft bounds penalty on log_std (keeps exploration in a sane range)."""
    log_std = policy.log_std
    lower = torch.clamp(-4.0 - log_std, min=0.0)
    upper = torch.clamp(log_std - 1.0, min=0.0)
    return (lower + upper).sum()


def _approx_kl(old_logprobs: torch.Tensor, new_logprobs: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 estimator of the approximate KL divergence."""
    log_ratio = new_logprobs - old_logprobs
    return torch.mean(torch.exp(log_ratio) - log_ratio - 1.0)


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------
class PPO:
    """Vanilla PPO trainer operating on a single policy over all environments."""

    def __init__(
        self,
        policy: GaussianPolicy,
        value_net: nn.Module,
        config: Optional[PPOConfig] = None,
    ) -> None:
        self.config = config or PPOConfig()
        self.policy = policy
        self.value_net = value_net
        self.device = torch.device(self.config.device)

        params = list(self.policy.parameters()) + list(self.value_net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.config.lr)
        self.current_lr = self.config.lr

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Sample actions for all environments (single worker id = 0)."""
        worker_ids = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)
        return self.policy.act(obs, worker_ids, hidden, deterministic=deterministic)

    @torch.no_grad()
    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)

    # -- learning rate adaptation -----------------------------------------
    def _adapt_lr(self, approx_kl: float) -> None:
        if not self.config.lr_adapt:
            return
        if approx_kl > 1.5 * self.config.kl_threshold:
            self.current_lr = max(self.current_lr / 1.5, 1e-6)
        elif approx_kl < self.config.kl_threshold / 1.5:
            self.current_lr = min(self.current_lr * 1.5, 1e-2)
        for g in self.optimizer.param_groups:
            g["lr"] = self.current_lr

    # -- update ------------------------------------------------------------
    def update(
        self,
        buffer: RolloutBuffer,
        num_mini_epochs: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run PPO updates on the collected rollout buffer."""
        cfg = self.config
        num_mini_epochs = num_mini_epochs or cfg.num_mini_epochs

        if cfg.normalize_advantages:
            buffer.normalize_advantages()

        mini_batch_size = buffer.mini_batch_size(cfg.mini_batch_multiplier)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "num_updates": 0,
        }

        for _ in range(num_mini_epochs):
            for batch in buffer.get_mini_batches(mini_batch_size, num_mini_epochs=1):
                flat = _flatten_batch(batch)
                obs = flat["obs"]
                actions = flat["actions"]
                old_logprobs = flat["old_logprobs"]
                old_values = flat["old_values"]
                advantages = flat["advantages"]
                returns = flat["returns"]
                worker_ids = flat.get("worker_ids")
                if worker_ids is None:
                    worker_ids = torch.zeros(obs.shape[0], dtype=torch.long, device=obs.device)

                # Evaluate current policy / value.
                new_logprobs, entropy, new_values = self.policy.evaluate_actions(
                    obs, actions, worker_ids
                )
                new_values = new_values.squeeze(-1)

                # Importance ratio (on-policy for PPO).
                ratio = torch.exp(new_logprobs - old_logprobs)

                # Clipped surrogate objective.
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped).
                value_clipped = old_values + torch.clamp(
                    new_values - old_values, -cfg.clip_eps, cfg.clip_eps
                )
                v_loss_unclipped = (new_values - returns) ** 2
                v_loss_clipped = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                bounds_loss = _compute_bounds_loss(self.policy)

                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    + cfg.bounds_loss_coef * bounds_loss
                    - cfg.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = _approx_kl(old_logprobs, new_logprobs).item()
                    clip_frac = (
                        (torch.abs(ratio - 1.0) > cfg.clip_eps).float().mean().item()
                    )

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                stats["num_updates"] += 1

        n = max(stats["num_updates"], 1)
        for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
            stats[k] /= n
        stats["lr"] = self.current_lr

        self._adapt_lr(stats["approx_kl"])
        return stats

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> Dict:
        return {
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "current_lr": self.current_lr,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.value_net.load_state_dict(state["value_net"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.current_lr = state.get("current_lr", self.config.lr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_ppo(
    obs_dim: int,
    action_dim: int,
    num_envs: int = 4096,
    hidden_dims: Tuple[int, ...] = (512, 256, 128),
    recurrent: bool = False,
    lstm_hidden: int = 768,
    config: Optional[PPOConfig] = None,
    device: str = "cpu",
) -> PPO:
    """Construct a vanilla PPO trainer with a single shared policy."""
    from .networks import build_critic

    cfg = config or PPOConfig(num_envs=num_envs, device=device)
    policy = build_policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_workers=1,
        hidden_dims=hidden_dims,
        recurrent=recurrent,
        lstm_hidden=lstm_hidden,
    )
    value_net = build_critic(
        obs_dim=obs_dim,
        num_workers=1,
        hidden_dims=hidden_dims,
        recurrent=recurrent,
        lstm_hidden=lstm_hidden,
    )
    return PPO(policy, value_net, cfg)
