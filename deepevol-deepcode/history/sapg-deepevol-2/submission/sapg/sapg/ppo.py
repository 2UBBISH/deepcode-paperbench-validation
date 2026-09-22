"""Baseline Proximal Policy Optimization (PPO).

This module implements the standard PPO algorithm used as the baseline in the
SAPG paper.  A *single* policy is trained across all parallel environments
(no splitting into blocks / followers).  It is used for the batch-size sweep
experiment (Figure 2) that demonstrates PPO's performance saturation as the
number of parallel environments grows.

Key ingredients (matching the paper's hyper-parameters):
    * Clipped surrogate objective with epsilon (0.1 AllegroKuka/ShadowHand,
      0.2 AllegroHand).
    * GAE with gamma = 0.99, tau = 0.95.
    * Critic coefficient lambda' = 4.0.
    * Entropy coefficient = 0.0.
    * Gradient norm clipping = 1.0.
    * KL-based adaptive learning rate with threshold 0.016.
    * Mini-batch size = num_envs * 4.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .policy import GaussianPolicy, build_policy
from .rollout_buffer import RolloutBuffer
from .utils import (
    AdaptiveKLLR,
    RunningMeanStd,
    clip_grad_norm_,
    compute_kl,
    explained_variance,
)


def _cfg_getter(cfg):
    """Return a ``get(key, default)`` callable for dict or attribute configs."""
    if cfg is None:
        return lambda key, default=None: default
    if isinstance(cfg, dict):
        return lambda key, default=None: cfg.get(key, default)

    def _get(key, default=None):
        return getattr(cfg, key, default)

    return _get


class PPO:
    """Standard PPO trainer operating on a single shared policy.

    Parameters
    ----------
    cfg : dict | object
        Configuration object.  Recognised keys (with defaults):
            gamma=0.99, tau=0.95, clip_epsilon=0.1, critic_coef=4.0,
            entropy_coef=0.0, max_grad_norm=1.0, learning_rate=3e-4,
            num_mini_batches=4, num_learning_epochs=5, kl_target=0.016,
            use_kl_adaptive_lr=True, normalize_advantage=True,
            value_loss_coef (alias of critic_coef), use_lstm=False,
            seq_len=16, device='cpu'.
    policy : GaussianPolicy, optional
        Pre-built policy.  If ``None`` the policy is built lazily on the first
        call to :meth:`init` using ``obs_dim``/``action_dim``.
    obs_dim, action_dim : int
        Observation and action dimensionality (used to build the policy).
    num_envs : int
        Number of parallel environments (used for mini-batch sizing).
    """

    def __init__(
        self,
        cfg=None,
        policy: Optional[GaussianPolicy] = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        num_envs: int = 1,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        get = _cfg_getter(cfg)

        self.gamma = float(get("gamma", 0.99))
        self.tau = float(get("tau", 0.95))
        self.clip_epsilon = float(get("clip_epsilon", get("epsilon", 0.1)))
        self.critic_coef = float(get("critic_coef", get("value_loss_coef", 4.0)))
        self.entropy_coef = float(get("entropy_coef", 0.0))
        self.max_grad_norm = float(get("max_grad_norm", 1.0))
        self.learning_rate = float(get("learning_rate", get("lr", 3e-4)))
        self.num_learning_epochs = int(get("num_learning_epochs", get("mini_epochs", 5)))
        self.num_mini_batches = int(get("num_mini_batches", 4))
        self.kl_target = float(get("kl_target", 0.016))
        self.use_kl_adaptive_lr = bool(get("use_kl_adaptive_lr", True))
        self.normalize_advantage = bool(get("normalize_advantage", True))
        self.use_lstm = bool(get("use_lstm", False))
        self.seq_len = int(get("seq_len", 16))
        self.num_envs = int(num_envs)

        self.device = device if device is not None else torch.device(
            get("device", "cpu")
        )

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy = policy
        if self.policy is not None:
            self.policy.to(self.device)

        # Optimizer / LR scheduler are created once the policy exists.
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[AdaptiveKLLR] = None

        # Value normalisation (running mean/std of returns).
        self.value_normalizer = RunningMeanStd(shape=())
        self.normalize_value = bool(get("normalize_value", False))

        # Diagnostics from the last update.
        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def init(self, obs_dim: Optional[int] = None, action_dim: Optional[int] = None):
        """Lazily build the policy and optimizer if not already provided."""
        if obs_dim is not None:
            self.obs_dim = obs_dim
        if action_dim is not None:
            self.action_dim = action_dim

        if self.policy is None:
            if self.obs_dim is None or self.action_dim is None:
                raise ValueError(
                    "obs_dim and action_dim must be provided to build the policy."
                )
            self.policy = build_policy(
                self.cfg, self.obs_dim, self.action_dim, num_workers=1
            )
            self.policy.to(self.device)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.learning_rate
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = AdaptiveKLLR(
                init_lr=self.learning_rate, kl_target=self.kl_target
            )
        return self

    def _ensure_optimizer(self):
        if self.optimizer is None:
            self.init()

    # ------------------------------------------------------------------ #
    # Rollout collection
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, hidden_state=None, masks=None, deterministic: bool = False):
        """Sample an action for the current observation(s).

        Returns ``(actions, log_probs, values, hidden_state)``.
        """
        self._ensure_optimizer()
        obs_t = obs if torch.is_tensor(obs) else torch.as_tensor(obs, dtype=torch.float32)
        obs_t = obs_t.to(self.device)
        worker_ids = torch.zeros(obs_t.shape[0], dtype=torch.long, device=self.device)

        actions, log_probs, mean, std, new_hidden = self.policy.sample(
            obs_t, worker_ids, hidden_state, masks, deterministic=deterministic
        )
        values = self.policy.actor  # placeholder to keep linters quiet
        values = self._value(obs_t, worker_ids)
        return actions, log_probs, values, new_hidden

    def _value(self, obs, worker_ids):
        """Compute the critic value using the policy's value network."""
        if hasattr(self.policy, "critic") and self.policy.critic is not None:
            return self.policy.critic(obs, worker_ids)
        # Fall back to a zero value if no critic is attached.
        return torch.zeros(obs.shape[0], device=obs.device)

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO updates over the data stored in ``buffer``.

        The buffer must already have had
        :meth:`RolloutBuffer.compute_returns_and_advantages` called on it.
        """
        self._ensure_optimizer()

        # Determine mini-batch count: paper uses num_envs * 4 transitions per
        # mini-batch, i.e. num_mini_batches = total_transitions / (num_envs*4).
        total = len(buffer)
        mini_batch_size = max(1, self.num_envs * 4)
        num_mini_batches = max(1, total // mini_batch_size)
        if self.num_mini_batches and self.num_mini_batches > 0:
            # Allow explicit override from config.
            num_mini_batches = self.num_mini_batches

        seq_len = self.seq_len if (self.use_lstm or buffer.recurrent) else 1

        stats_accum = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        num_updates = 0

        for _epoch in range(self.num_learning_epochs):
            for batch in buffer.get_batches(
                num_mini_batches=num_mini_batches,
                seq_len=seq_len,
                shuffle=True,
            ):
                loss, stats = self._update_minibatch(batch)
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                stats["grad_norm"] = grad_norm
                for k, v in stats.items():
                    stats_accum[k] = stats_accum.get(k, 0.0) + float(v)
                num_updates += 1

        if num_updates > 0:
            for k in stats_accum:
                stats_accum[k] /= num_updates

        # KL-adaptive learning rate.
        if self.use_kl_adaptive_lr and self.lr_scheduler is not None:
            new_lr = self.lr_scheduler.update(stats_accum.get("approx_kl", 0.0))
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr
            stats_accum["learning_rate"] = new_lr

        self.last_stats = stats_accum
        return stats_accum

    def _update_minibatch(self, batch: Dict[str, torch.Tensor]):
        """Compute the PPO loss for a single mini-batch."""
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        old_values = batch["old_values"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        returns = batch["returns"].to(self.device)
        worker_ids = batch.get("worker_ids")
        if worker_ids is None:
            worker_ids = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)
        else:
            worker_ids = worker_ids.to(self.device)

        hidden_states = batch.get("hidden_states")
        cell_states = batch.get("cell_states")
        masks = batch.get("masks")
        hidden_state = None
        if hidden_states is not None:
            hidden_state = (hidden_states.to(self.device), cell_states.to(self.device))

        # Re-evaluate actions under the current policy.
        new_log_probs, entropy, new_values = self.policy.evaluate_actions(
            obs, actions, worker_ids, hidden_state, masks
        )

        # Importance-sampling ratio.
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Clipped surrogate objective.
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        ) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (optionally normalised).
        if self.normalize_value:
            value_targets = self.value_normalizer.normalize(returns)
            value_preds = self.value_normalizer.normalize(new_values)
        else:
            value_targets = returns
            value_preds = new_values
        value_loss = 0.5 * (value_preds - value_targets).pow(2).mean()

        entropy_loss = entropy.mean()

        loss = (
            policy_loss
            + self.critic_coef * value_loss
            - self.entropy_coef * entropy_loss
        )

        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean()
            clip_fraction = (
                (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
            )

        stats = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(entropy_loss.detach()),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
        }
        return loss, stats

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict:
        return {
            "policy": self.policy.state_dict() if self.policy is not None else None,
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }

    def load_state_dict(self, state: Dict):
        if self.policy is not None and state.get("policy") is not None:
            self.policy.load_state_dict(state["policy"])
        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.lr_scheduler is not None and state.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])

    def train(self):
        if self.policy is not None:
            self.policy.train()

    def eval(self):
        if self.policy is not None:
            self.policy.eval()
