"""SAPG: Split and Aggregate Policy Gradients.

This module implements the core contribution of the paper: a new class of
on-policy RL algorithms that scale to tens of thousands of parallel
environments.

Instead of running a single PPO policy across all environments, SAPG:

  (a) SPLIT: divides the ``num_envs`` environments into ``B`` blocks. Each
      block ``b`` runs its own *follower* policy ``pi_{phi_b}``. All followers
      share the same network weights ``B_theta`` but are conditioned on a
      distinct per-worker embedding ``phi_b``. Each block collects a
      horizon-length rollout that is on-policy *for that follower*.

  (b) AGGREGATE: one worker is designated the *leader*. The leader is updated
      using ALL off-policy data collected by every follower, using an
      importance-weighted / clipped surrogate objective (PPO-style). The
      followers are updated with their own on-policy data.

The importance sampling ratio for the leader is

    r = pi_leader(a | s) / pi_follower(a | s)

and the clipped surrogate objective is

    L = E[ min(r * A, clip(r, 1 - eps, 1 + eps) * A) ]

with critic coefficient ``lambda' = 4.0``, entropy coefficient ``0`` and
gradient-norm clipping ``1.0``. GAE uses ``gamma = 0.99`` and ``tau = 0.95``.
A KL-based adaptive learning-rate schedule with threshold ``0.016`` is used.

When ``num_workers == 1`` (a single block) SAPG reduces exactly to standard
PPO.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .aggregation import (
    AggregationMode,
    Aggregator,
    UpdatePlan,
    block_slices,
    build_aggregator,
    split_worker_ids,
)
from .policy import GaussianPolicy, build_policy
from .rollout_buffer import MultiWorkerRolloutBuffer, RolloutBuffer
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


class SAPG:
    """Split-and-Aggregate Policy Gradients trainer.

    Parameters
    ----------
    cfg : dict | object | None
        Configuration. Recognised keys include ``gamma``, ``tau``,
        ``clip_epsilon``, ``critic_coef``, ``entropy_coef``, ``max_grad_norm``,
        ``learning_rate``, ``num_learning_epochs``, ``num_mini_batches``,
        ``kl_target``, ``use_kl_adaptive_lr``, ``num_workers``,
        ``aggregation_mode``, ``leader_id``, ``rotate_leader``, ``use_lstm``,
        ``seq_len``, ``normalize_advantage``, ``use_entropy_exploration``.
    policy : GaussianPolicy, optional
        Pre-built shared policy. If ``None`` it is built from ``cfg`` using
        ``obs_dim`` / ``action_dim`` / ``num_workers``.
    obs_dim, action_dim : int, optional
        Observation and action dimensions (required if ``policy`` is None).
    num_envs : int
        Total number of parallel environments (across all blocks).
    num_workers : int, optional
        Number of blocks / followers (``B``). Defaults to ``cfg.num_workers``
        or 1.
    device : torch.device, optional
        Device for tensors and networks.
    """

    def __init__(
        self,
        cfg=None,
        policy: Optional[GaussianPolicy] = None,
        obs_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        num_envs: int = 1,
        num_workers: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = cfg
        get = _cfg_getter(cfg)

        self.device = device if device is not None else torch.device("cpu")
        self.num_envs = int(num_envs)

        # ---- number of blocks / followers -------------------------------
        if num_workers is None:
            num_workers = get("num_workers", None)
        if num_workers is None:
            num_workers = get("num_blocks", 1)
        self.num_workers = max(1, int(num_workers))
        # Cannot have more blocks than environments.
        self.num_workers = min(self.num_workers, max(1, self.num_envs))

        # ---- PPO hyper-parameters ---------------------------------------
        self.gamma = float(get("gamma", 0.99))
        self.tau = float(get("tau", 0.95))
        self.clip_epsilon = float(get("clip_epsilon", 0.1))
        self.critic_coef = float(get("critic_coef", 4.0))
        self.entropy_coef = float(get("entropy_coef", 0.0))
        self.max_grad_norm = float(get("max_grad_norm", 1.0))
        self.learning_rate = float(get("learning_rate", 3e-4))
        self.num_learning_epochs = int(get("num_learning_epochs", 5))
        self.kl_target = float(get("kl_target", 0.016))
        self.use_kl_adaptive_lr = bool(get("use_kl_adaptive_lr", True))
        self.normalize_advantage = bool(get("normalize_advantage", True))
        self.use_lstm = bool(get("use_lstm", False))
        self.seq_len = int(get("seq_len", 16))
        self.use_entropy_exploration = bool(get("use_entropy_exploration", False))
        self.bounds_loss_coef = float(get("bounds_loss_coef", 1e-4))

        # Mini-batch size = num_envs * 4 (paper). ``num_mini_batches`` overrides.
        default_mini_batches = max(1, (self.num_envs * 4) // max(1, self.num_envs))
        self.num_mini_batches = int(get("num_mini_batches", default_mini_batches))
        self.num_mini_batches = max(1, self.num_mini_batches)

        # ---- aggregation -------------------------------------------------
        self.aggregation_mode = AggregationMode.from_str(
            get("aggregation_mode", "leader")
        )
        self.aggregator: Aggregator = build_aggregator(cfg, self.num_workers)
        # Keep clip epsilon consistent with the aggregator.
        self.aggregator.clip_epsilon = self.clip_epsilon

        # ---- policy / optimizer -----------------------------------------
        if policy is None:
            if obs_dim is None or action_dim is None:
                raise ValueError(
                    "obs_dim and action_dim are required when policy is None"
                )
            policy = build_policy(cfg, obs_dim, action_dim, self.num_workers)
        self.policy: GaussianPolicy = policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate, eps=1e-5
        )
        self.lr_scheduler = AdaptiveKLLR(
            init_lr=self.learning_rate, kl_target=self.kl_target
        )

        self.value_normalizer = RunningMeanStd(shape=())

        # ---- environment -> block assignment ----------------------------
        self.env_worker_ids = split_worker_ids(
            self.num_workers, self.num_envs, device=self.device
        )
        self.block_slices = block_slices(self.num_workers, self.num_envs)

        self.last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def is_recurrent(self) -> bool:
        return bool(self.use_lstm or getattr(self.policy, "is_recurrent", False))

    def _worker_ids_for_envs(self, env_indices: Optional[torch.Tensor] = None):
        """Return the worker id for each environment (or a subset)."""
        if env_indices is None:
            return self.env_worker_ids
        return self.env_worker_ids[env_indices]

    def _value(self, obs: torch.Tensor, worker_ids: torch.Tensor) -> torch.Tensor:
        """Compute critic values (critic lives on the policy module)."""
        critic = getattr(self.policy, "critic", None)
        if critic is None:
            raise AttributeError(
                "GaussianPolicy has no `critic` attribute; SAPG requires a critic."
            )
        return critic(obs, worker_ids).squeeze(-1)

    # ------------------------------------------------------------------ #
    # Rollout collection (SPLIT)
    # ------------------------------------------------------------------ #
    def init_hidden(self, batch_size: int):
        """Initialise recurrent hidden state for ``batch_size`` envs."""
        if self.is_recurrent:
            return self.policy.init_hidden(batch_size, self.device)
        return None

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        worker_ids: Optional[torch.Tensor] = None,
        hidden_state=None,
        masks: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """Sample actions for a batch of environments.

        Returns ``(actions, log_probs, values, hidden_state)``.
        """
        if worker_ids is None:
            worker_ids = self.env_worker_ids[: obs.shape[0]]
        worker_ids = worker_ids.to(self.device)

        actions, log_probs, new_hidden = self.policy.sample(
            obs, worker_ids, hidden_state, masks, deterministic=deterministic
        )
        values = self._value(obs, worker_ids)
        return actions, log_probs, values, new_hidden

    # ------------------------------------------------------------------ #
    # Update (AGGREGATE)
    # ------------------------------------------------------------------ #
    def update(self, buffer: MultiWorkerRolloutBuffer) -> Dict[str, float]:
        """Run one SAPG update over all workers.

        ``buffer`` is a :class:`MultiWorkerRolloutBuffer` holding one
        :class:`RolloutBuffer` per worker.
        """
        # ---- compute GAE per worker -------------------------------------
        last_values = []
        for b in range(self.num_workers):
            wb = buffer[b]
            last_obs = getattr(wb, "last_obs", None)
            if last_obs is not None:
                wids = torch.full(
                    (last_obs.shape[0],),
                    b,
                    dtype=torch.long,
                    device=self.device,
                )
                last_values.append(self._value(last_obs.to(self.device), wids))
            else:
                last_values.append(torch.zeros(wb.num_envs, device=self.device))
        buffer.compute_returns_and_advantages(last_values)

        # ---- build update plans -----------------------------------------
        plans = self.aggregator.build_update_plans()

        stats: Dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "explained_variance": 0.0,
        }
        n_updates = 0
        last_kl = 0.0

        for plan in plans:
            data = self.aggregator.gather_data(plan, buffer)
            if data is None or data["obs"].shape[0] == 0:
                continue
            pstats, last_kl = self._update_worker(plan, data)
            for k, v in pstats.items():
                stats[k] = stats.get(k, 0.0) + v
            n_updates += 1

        if n_updates > 0:
            for k in list(stats.keys()):
                if k != "learning_rate":
                    stats[k] /= n_updates

        # ---- KL-adaptive learning rate ----------------------------------
        if self.use_kl_adaptive_lr:
            new_lr = self.lr_scheduler.update(last_kl)
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr
            stats["learning_rate"] = new_lr

        # ---- advance leader if rotation is enabled ----------------------
        self.aggregator.next_leader()

        self.last_stats = stats
        return stats

    def _update_worker(
        self, plan: UpdatePlan, data: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], float]:
        """Run mini-batch epochs for a single worker's update plan."""
        obs = data["obs"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        old_values = data["old_values"]
        advantages = data["advantages"]
        returns = data["returns"]
        source_worker_ids = data["worker_ids"]
        hidden_states = data.get("hidden_states")
        cell_states = data.get("cell_states")
        masks = data.get("masks")

        n = obs.shape[0]
        if n == 0:
            return {}, 0.0

        # Target worker embedding used to re-evaluate actions.
        target_ids = torch.full(
            (n,), plan.worker_id, dtype=torch.long, device=self.device
        )

        if self.normalize_advantage and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Value normalisation (running mean/std of returns).
        if self.value_normalizer is not None:
            self.value_normalizer.update(returns.detach().cpu().numpy())
            returns_norm = self.value_normalizer.normalize(returns)
        else:
            returns_norm = returns

        recurrent = self.is_recurrent
        seq_len = self.seq_len if recurrent else 1

        # Build index batches.
        indices = torch.randperm(n, device=self.device)
        mini_batch_size = max(1, n // self.num_mini_batches)

        agg = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "explained_variance": 0.0,
        }
        n_mb = 0
        last_kl = 0.0

        for _epoch in range(self.num_learning_epochs):
            for start in range(0, n, mini_batch_size):
                mb_idx = indices[start : start + mini_batch_size]
                if mb_idx.numel() == 0:
                    continue

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns_norm[mb_idx]
                mb_old_values = old_values[mb_idx]
                mb_target_ids = target_ids[mb_idx]
                mb_source_ids = source_worker_ids[mb_idx]

                mb_hidden = None
                mb_cell = None
                mb_masks = None
                if recurrent and hidden_states is not None:
                    mb_hidden = hidden_states[mb_idx]
                    if cell_states is not None:
                        mb_cell = cell_states[mb_idx]
                    if masks is not None:
                        mb_masks = masks[mb_idx]

                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    mb_obs,
                    mb_actions,
                    mb_target_ids,
                    mb_hidden,
                    mb_masks,
                    cell_state=mb_cell,
                )

                # ---- importance sampling ratio --------------------------
                # r = pi_leader(a|s) / pi_follower(a|s)
                #   = exp(log pi_leader - log pi_follower)
                ratio = self.aggregator.importance_ratio(
                    new_log_probs,
                    mb_old_log_probs,
                    source_worker_ids=mb_source_ids,
                    target_worker_id=plan.worker_id,
                )

                # ---- clipped surrogate ----------------------------------
                surr1 = ratio * mb_advantages
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- value loss -----------------------------------------
                value_loss = 0.5 * ((new_values - mb_returns) ** 2).mean()

                # ---- entropy --------------------------------------------
                entropy_loss = -entropy.mean()

                # ---- bounds loss (action mean within bounds) ------------
                bounds_loss = self.policy.bounds_loss(
                    self.policy.actor(mb_obs, mb_target_ids)[0]
                    if not recurrent
                    else self.policy.actor(mb_obs, mb_target_ids, mb_hidden, mb_masks)[0]
                )

                loss = (
                    policy_loss
                    + self.critic_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + self.bounds_loss_coef * bounds_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = compute_kl(
                        mb_old_log_probs,
                        torch.zeros_like(mb_old_log_probs),
                        new_log_probs,
                        torch.zeros_like(new_log_probs),
                    )
                    clip_frac = (
                        (ratio - 1.0).abs() > self.clip_epsilon
                    ).float().mean()

                agg["policy_loss"] += float(policy_loss.detach())
                agg["value_loss"] += float(value_loss.detach())
                agg["entropy"] += float(entropy.mean().detach())
                agg["approx_kl"] += float(approx_kl)
                agg["clip_fraction"] += float(clip_frac)
                agg["grad_norm"] += float(grad_norm)
                last_kl = float(approx_kl)
                n_mb += 1

        if n_mb > 0:
            for k in agg:
                agg[k] /= n_mb

        with torch.no_grad():
            agg["explained_variance"] = explained_variance(
                old_values.detach().cpu().numpy(),
                returns.detach().cpu().numpy(),
            )

        return agg, last_kl

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "lr_scheduler" in state:
            self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        if "value_normalizer" in state:
            self.value_normalizer.load_state_dict(state["value_normalizer"])

    def train(self) -> None:
        self.policy.train()

    def eval(self) -> None:
        self.policy.eval()


__all__ = ["SAPG"]
