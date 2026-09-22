"""Aggregation strategies for SAPG.

This module implements the different ways in which the off-policy data
collected by the followers can be aggregated to update a policy.  The paper
introduces *Split and Aggregate Policy Gradients* (SAPG) whose default mode is
a **leader** policy that is trained on the union of all followers'
transitions.  As an ablation the paper also studies a **symmetric**
aggregation scheme in which there is no designated leader: every worker is
updated using the off-policy data produced by all the other workers.

The classes here are thin wrappers around :class:`sapg.sapg.leader.Leader` and
:class:`sapg.sapg.follower.Follower` so that the training loop in ``main.py``
can switch between aggregation variants purely through configuration.

Public interface
----------------
- :class:`AggregationMode` -- enum of supported aggregation strategies.
- :class:`SymmetricAggregator` -- ablation: every worker updated with all
  other workers' off-policy data (no leader).
- :class:`LeaderAggregator` -- default SAPG: a single leader trained on the
  union of all followers' transitions.
- :func:`build_aggregator` -- factory returning the aggregator for a config.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .buffer import AggregatedBuffer, RolloutBuffer
from .follower import Follower, FollowerStats
from .leader import Leader, LeaderStats
from .networks import GaussianPolicy, ValueNetwork
from .ppo import KLScheduler, PPOHyperParams


class AggregationMode(str, enum.Enum):
    """Supported aggregation strategies."""

    LEADER = "leader"
    SYMMETRIC = "symmetric"

    @classmethod
    def from_str(cls, value: str) -> "AggregationMode":
        value = str(value).lower()
        if value in ("leader", "sapg", "default"):
            return cls.LEADER
        if value in ("symmetric", "sym", "no_leader", "noleader"):
            return cls.SYMMETRIC
        raise ValueError(f"Unknown aggregation mode: {value!r}")


@dataclass
class AggregationStats:
    """Container for the diagnostics of one aggregation step."""

    mode: str = AggregationMode.LEADER.value
    num_transitions: int = 0
    num_workers: int = 0
    # Per-worker follower statistics (on-policy updates).
    follower: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Leader statistics (only populated for the leader mode).
    leader: Optional[Dict[str, float]] = None
    # Symmetric-mode statistics: one entry per worker's aggregated update.
    symmetric: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {
            f"{prefix}aggregation/num_transitions": float(self.num_transitions),
            f"{prefix}aggregation/num_workers": float(self.num_workers),
        }
        for block_id, stats in self.follower.items():
            for key, value in stats.items():
                out[f"{prefix}follower_{block_id}/{key}"] = float(value)
        if self.leader is not None:
            for key, value in self.leader.items():
                out[f"{prefix}{key}"] = float(value)
        for block_id, stats in self.symmetric.items():
            for key, value in stats.items():
                out[f"{prefix}symmetric_{block_id}/{key}"] = float(value)
        return out


class LeaderAggregator:
    """Default SAPG aggregation: a single leader trained on all transitions.

    The followers each perform an on-policy PPO update on their own block of
    environments.  Afterwards the leader consumes the *union* of every
    follower's rollout (off-policy data) and performs a PPO update with
    importance weighting anchored on the behaviour (follower) log-probabilities.
    """

    def __init__(self, leader: Leader):
        self.leader = leader
        self.mode = AggregationMode.LEADER

    def aggregate(
        self,
        buffers: List[RolloutBuffer],
        last_values: Optional[List[torch.Tensor]] = None,
        last_dones: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> LeaderStats:
        """Run the leader update over the union of all follower buffers."""
        return self.leader.update(
            buffers,
            last_values=last_values,
            last_dones=last_dones,
            generator=generator,
        )

    def act(self, obs, lstm_state=None, deterministic: bool = False):
        return self.leader.act(obs, lstm_state=lstm_state, deterministic=deterministic)


class SymmetricAggregator:
    """Ablation: symmetric aggregation without a designated leader.

    Every worker is updated using the off-policy data produced by *all* the
    workers (including its own on-policy data).  Concretely, for each worker
    ``j`` we build an :class:`AggregatedBuffer` over the union of all buffers
    and run a PPO update conditioned on worker ``j``'s ``phi_j``.  This
    removes the asymmetry between the deployed leader and the data-collecting
    followers that SAPG relies on.
    """

    def __init__(
        self,
        followers: List[Follower],
        policy: GaussianPolicy,
        value: ValueNetwork,
        optimizer: torch.optim.Optimizer,
        hparams: Optional[PPOHyperParams] = None,
        kl_scheduler: Optional[KLScheduler] = None,
        device: str = "cpu",
        use_importance_weights: bool = True,
        max_importance_weight: float = 10.0,
    ):
        self.followers = followers
        self.policy = policy
        self.value = value
        self.optimizer = optimizer
        self.hparams = hparams or PPOHyperParams()
        self.kl_scheduler = kl_scheduler
        self.device = device
        self.use_importance_weights = use_importance_weights
        self.max_importance_weight = max_importance_weight
        self.mode = AggregationMode.SYMMETRIC

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mini_batch_size(self, num_envs: int) -> int:
        """Mini-batch size = num_envs * 4 (paper default)."""
        return max(1, int(num_envs) * 4)

    def _num_minibatches(self, num_transitions: int, num_envs: int) -> int:
        mini_batch_size = self._mini_batch_size(num_envs)
        return max(1, num_transitions // mini_batch_size)

    def _update_worker(
        self,
        worker: Follower,
        aggregated: AggregatedBuffer,
        generator: Optional[torch.Generator],
    ) -> Dict[str, float]:
        """Run one off-policy PPO update for ``worker`` on aggregated data."""
        from .ppo import (
            clip_grad_norm_,
            compute_ppo_loss,
            explained_variance,
            normalize_advantages,
        )

        hp = self.hparams
        num_transitions = aggregated.num_transitions
        num_envs = max(1, num_transitions // max(1, hp.horizon))
        num_minibatches = self._num_minibatches(num_transitions, num_envs)

        phi = worker.phi
        block_id = worker.block_id

        totals: Dict[str, float] = {}
        count = 0
        for batch in aggregated.get_minibatches(
            num_minibatches, shuffle=True, generator=generator
        ):
            obs = batch["obs"].to(self.device)
            actions = batch["actions"].to(self.device)
            advantages = batch["advantages"].to(self.device)
            returns = batch["returns"].to(self.device)
            beh_log_probs = batch["log_probs"].to(self.device)
            lstm_states = batch.get("lstm_states")
            if lstm_states is not None:
                lstm_states = lstm_states.to(self.device)

            # Condition on this worker's phi for every transition.
            if phi is not None:
                batch_phi = phi.expand(obs.shape[0], -1).to(self.device)
            else:
                batch_phi = None
            batch_block_ids = torch.full(
                (obs.shape[0],), block_id, dtype=torch.long, device=self.device
            )

            new_log_probs, entropy = self.policy.evaluate_actions(
                obs, actions, batch_phi, batch_block_ids, lstm_states
            )
            values, _ = self.value(obs, batch_phi, lstm_states)

            # Importance weighting for off-policy data: anchor the PPO ratio
            # on the behaviour (follower) log-probs.
            if self.use_importance_weights:
                with torch.no_grad():
                    iw = torch.exp(new_log_probs - beh_log_probs).clamp(
                        max=self.max_importance_weight
                    )
                advantages = advantages * iw

            advantages = normalize_advantages(advantages)

            loss, info = compute_ppo_loss(
                new_log_probs=new_log_probs,
                old_log_probs=beh_log_probs,
                values=values,
                returns=returns,
                advantages=advantages,
                entropy=entropy,
                actions=actions,
                clip_eps=hp.clip_eps,
                entropy_coeff=hp.entropy_coeff,
                critic_coeff=hp.critic_coeff,
                bounds_loss_coeff=hp.bounds_loss_coeff,
                action_bound=hp.action_bound,
            )

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value.parameters()),
                max_norm=hp.grad_norm_clip,
            )
            self.optimizer.step()

            if self.kl_scheduler is not None:
                self.kl_scheduler.step(info["approx_kl"])

            info["grad_norm"] = grad_norm
            info["explained_variance"] = explained_variance(
                values.detach(), returns
            )
            for key, value in info.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1

        if count == 0:
            return {}
        return {key: value / count for key, value in totals.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def aggregate(
        self,
        buffers: List[RolloutBuffer],
        last_values: Optional[List[torch.Tensor]] = None,
        last_dones: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Update every worker symmetrically on the union of all buffers."""
        aggregated = AggregatedBuffer(buffers, device=self.device)
        results: Dict[int, Dict[str, float]] = {}
        for worker in self.followers:
            results[worker.block_id] = self._update_worker(
                worker, aggregated, generator
            )
        return results

    def act(self, obs, lstm_state=None, deterministic: bool = False):
        """Symmetric mode has no single deployed policy; use worker 0."""
        return self.followers[0].act(
            obs, lstm_state=lstm_state, deterministic=deterministic
        )


def build_aggregator(
    mode,
    *,
    leader: Optional[Leader] = None,
    followers: Optional[List[Follower]] = None,
    policy: Optional[GaussianPolicy] = None,
    value: Optional[ValueNetwork] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    hparams: Optional[PPOHyperParams] = None,
    kl_scheduler: Optional[KLScheduler] = None,
    device: str = "cpu",
    use_importance_weights: bool = True,
):
    """Factory building the aggregator described by ``mode``.

    Parameters
    ----------
    mode:
        Either an :class:`AggregationMode` or a string (``"leader"`` /
        ``"symmetric"``).
    """
    if not isinstance(mode, AggregationMode):
        mode = AggregationMode.from_str(mode)

    if mode is AggregationMode.LEADER:
        if leader is None:
            raise ValueError("LeaderAggregator requires a `leader` instance.")
        return LeaderAggregator(leader)

    if mode is AggregationMode.SYMMETRIC:
        if followers is None or policy is None or value is None or optimizer is None:
            raise ValueError(
                "SymmetricAggregator requires `followers`, `policy`, `value` "
                "and `optimizer`."
            )
        return SymmetricAggregator(
            followers=followers,
            policy=policy,
            value=value,
            optimizer=optimizer,
            hparams=hparams,
            kl_scheduler=kl_scheduler,
            device=device,
            use_importance_weights=use_importance_weights,
        )

    raise ValueError(f"Unsupported aggregation mode: {mode!r}")
