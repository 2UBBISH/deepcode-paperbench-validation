"""Aggregation logic for SAPG (Split and Aggregate Policy Gradients).

This module implements the "AGGREGATE" half of SAPG.  After the SPLIT phase
each follower ``pi_{phi_b}`` has collected an on-policy rollout for its own
block of environments.  The aggregation phase decides *which* policy is
updated with *which* data:

* ``LEADER`` (default, the paper's main method): one worker is designated the
  *leader*.  The leader is updated using the off-policy data produced by **all**
  followers (importance weighted / PPO-clipped surrogate).  Every follower is
  updated with its own on-policy data.  This lets the leader latch onto the
  high-reward trajectories discovered by any follower while remaining stable
  thanks to the clipped surrogate.

* ``SYMMETRIC`` (ablation, Figure 6): there is no leader.  Every worker is
  updated with the off-policy data from **all other** workers (symmetrically).

* ``NONE`` (ablation): no aggregation at all -- each worker only ever sees its
  own data (equivalent to running ``B`` independent PPO policies).

The module is deliberately free of any optimizer / network code so that it can
be unit-tested in isolation: it only produces *update plans* (which worker is
updated with which data) and the importance-sampling ratios used by the
clipped surrogate objective.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Aggregation modes
# ---------------------------------------------------------------------------
class AggregationMode(str, enum.Enum):
    """How off-policy data is routed to the workers during AGGREGATE."""

    LEADER = "leader"          # one designated leader sees all followers' data
    SYMMETRIC = "symmetric"    # every worker sees every other worker's data
    NONE = "none"              # no aggregation (independent PPO per block)

    @classmethod
    def from_str(cls, value) -> "AggregationMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.LEADER
        value = str(value).strip().lower()
        aliases = {
            "leader": cls.LEADER,
            "sapg": cls.LEADER,
            "asymmetric": cls.LEADER,
            "symmetric": cls.SYMMETRIC,
            "sym": cls.SYMMETRIC,
            "none": cls.NONE,
            "no_aggregation": cls.NONE,
            "independent": cls.NONE,
        }
        if value not in aliases:
            raise ValueError(
                f"Unknown aggregation mode '{value}'. "
                f"Expected one of {[m.value for m in cls]}."
            )
        return aliases[value]


# ---------------------------------------------------------------------------
# Update plan
# ---------------------------------------------------------------------------
@dataclass
class UpdatePlan:
    """Describes one worker's update for a single SAPG iteration.

    Attributes
    ----------
    worker_id:
        Index of the worker (follower or leader) whose parameters are updated.
    is_leader:
        Whether this worker is the designated leader.
    data_worker_ids:
        The source worker ids whose transitions are used for this update.
        For an on-policy follower update this is ``[worker_id]``; for the leader
        it is ``[0, 1, ..., num_workers - 1]``.
    on_policy:
        ``True`` when ``data_worker_ids == [worker_id]`` (no importance
        correction needed beyond the usual PPO ratio).
    """

    worker_id: int
    is_leader: bool
    data_worker_ids: List[int] = field(default_factory=list)
    on_policy: bool = True

    def __post_init__(self):
        if not self.data_worker_ids:
            self.data_worker_ids = [self.worker_id]
        self.on_policy = self.data_worker_ids == [self.worker_id]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class Aggregator:
    """Builds update plans and importance-sampling ratios for SAPG.

    Parameters
    ----------
    num_workers:
        Number of blocks / followers ``B``.  The leader is one of these
        workers (it is *not* an extra policy), matching the paper where the
        leader is a designated worker.
    mode:
        :class:`AggregationMode` (or its string value).
    leader_id:
        Index of the designated leader.  Defaults to ``0``.  The paper does not
        specify a rotation policy, so a fixed leader is used by default; set
        ``rotate_leader=True`` to cycle the leader each iteration.
    rotate_leader:
        If ``True`` the leader index advances by one every call to
        :meth:`next_leader`.
    clip_epsilon:
        PPO clipping range ``epsilon`` used for the importance ratio.
    """

    def __init__(
        self,
        num_workers: int,
        mode: AggregationMode = AggregationMode.LEADER,
        leader_id: int = 0,
        rotate_leader: bool = False,
        clip_epsilon: float = 0.1,
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        self.num_workers = int(num_workers)
        self.mode = AggregationMode.from_str(mode)
        self.leader_id = int(leader_id) % self.num_workers
        self.rotate_leader = bool(rotate_leader)
        self.clip_epsilon = float(clip_epsilon)

    # -- leader bookkeeping -------------------------------------------------
    @property
    def leader(self) -> int:
        return self.leader_id

    def next_leader(self) -> int:
        """Advance (and return) the leader index if rotation is enabled."""
        if self.rotate_leader:
            self.leader_id = (self.leader_id + 1) % self.num_workers
        return self.leader_id

    # -- update plans -------------------------------------------------------
    def build_update_plans(self) -> List[UpdatePlan]:
        """Return the list of updates to perform this iteration.

        * ``LEADER``: one plan per worker.  The leader's plan aggregates data
          from all workers; followers use their own data only.
        * ``SYMMETRIC``: one plan per worker, each aggregating data from all
          *other* workers (plus its own on-policy data).
        * ``NONE``: one on-policy plan per worker.
        """
        plans: List[UpdatePlan] = []
        all_ids = list(range(self.num_workers))

        if self.mode is AggregationMode.NONE or self.num_workers == 1:
            for w in all_ids:
                plans.append(
                    UpdatePlan(worker_id=w, is_leader=False, data_worker_ids=[w])
                )
            return plans

        if self.mode is AggregationMode.LEADER:
            for w in all_ids:
                is_leader = w == self.leader_id
                data_ids = all_ids if is_leader else [w]
                plans.append(
                    UpdatePlan(
                        worker_id=w,
                        is_leader=is_leader,
                        data_worker_ids=list(data_ids),
                    )
                )
            return plans

        # SYMMETRIC
        for w in all_ids:
            data_ids = [w] + [o for o in all_ids if o != w]
            plans.append(
                UpdatePlan(
                    worker_id=w,
                    is_leader=False,
                    data_worker_ids=data_ids,
                )
            )
        return plans

    # -- importance sampling ------------------------------------------------
    def importance_ratio(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        source_worker_ids: Optional[torch.Tensor] = None,
        target_worker_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute the importance-sampling ratio ``r = pi_target / pi_source``.

        Because all workers share the same network weights ``B_theta`` and only
        differ through the conditioning embedding ``phi_j``, the ratio is simply
        the exponentiated difference of log-probabilities evaluated under the
        target and source worker embeddings::

            r = exp(log pi_target(a|s) - log pi_source(a|s))

        Parameters
        ----------
        new_log_probs:
            ``log pi_target(a|s)`` -- log-probs of the stored actions under the
            *target* (updating) worker's current policy.
        old_log_probs:
            ``log pi_source(a|s)`` -- log-probs recorded when the data was
            collected, i.e. under the *source* follower's policy.
        source_worker_ids, target_worker_id:
            Optional metadata (unused numerically, kept for API clarity and
            debugging / logging).

        Returns
        -------
        torch.Tensor
            The (unclipped) importance ratio, same shape as the inputs.
        """
        del source_worker_ids, target_worker_id  # metadata only
        return torch.exp(new_log_probs - old_log_probs)

    def clipped_surrogate(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: Optional[float] = None,
        source_worker_ids: Optional[torch.Tensor] = None,
        target_worker_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO-style clipped surrogate with importance sampling.

        ``L = E[ min(r * A, clip(r, 1 - eps, 1 + eps) * A) ]``

        where ``r = pi_target / pi_source``.  When the target worker is also the
        source worker this reduces exactly to the standard PPO objective.

        Returns
        -------
        (loss, ratio, clip_fraction)
            ``loss`` is the scalar surrogate loss to be *maximised* (the caller
            typically negates it); ``ratio`` is the unclipped ratio; and
            ``clip_fraction`` is the fraction of samples whose ratio was
            clipped (a useful diagnostic).
        """
        eps = self.clip_epsilon if clip_epsilon is None else float(clip_epsilon)
        ratio = self.importance_ratio(
            new_log_probs,
            old_log_probs,
            source_worker_ids=source_worker_ids,
            target_worker_id=target_worker_id,
        )
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
        loss = torch.min(surr1, surr2).mean()

        with torch.no_grad():
            clipped = (ratio < 1.0 - eps) | (ratio > 1.0 + eps)
            clip_fraction = clipped.float().mean()

        return loss, ratio, clip_fraction

    # -- data routing -------------------------------------------------------
    def gather_data(
        self,
        plan: UpdatePlan,
        worker_buffers: Sequence,
    ) -> Dict[str, torch.Tensor]:
        """Merge the buffers listed in ``plan.data_worker_ids``.

        ``worker_buffers`` is a sequence (indexable by worker id) of objects
        exposing ``all_data()`` returning a dict of tensors.  The returned dict
        concatenates the per-worker tensors along the batch dimension and
        re-tags ``worker_ids`` with the *source* worker index so that the
        importance ratio can be computed correctly.
        """
        chunks: Dict[str, List[torch.Tensor]] = {}
        for src in plan.data_worker_ids:
            buf = worker_buffers[src]
            data = buf.all_data() if hasattr(buf, "all_data") else buf
            for key, value in data.items():
                if not torch.is_tensor(value):
                    continue
                chunks.setdefault(key, []).append(value)

        merged: Dict[str, torch.Tensor] = {}
        for key, values in chunks.items():
            merged[key] = torch.cat(values, dim=0)

        # Ensure the source worker id is available for every transition.
        if "worker_ids" not in merged:
            ids = []
            for src in plan.data_worker_ids:
                buf = worker_buffers[src]
                n = len(buf) if hasattr(buf, "__len__") else None
                if n is None:
                    data = buf.all_data() if hasattr(buf, "all_data") else buf
                    n = next(iter(data.values())).shape[0]
                ids.append(torch.full((n,), src, dtype=torch.long))
            merged["worker_ids"] = torch.cat(ids, dim=0)

        return merged

    # -- repr ---------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Aggregator(num_workers={self.num_workers}, mode={self.mode.value}, "
            f"leader_id={self.leader_id}, rotate_leader={self.rotate_leader}, "
            f"clip_epsilon={self.clip_epsilon})"
        )


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def build_aggregator(cfg, num_workers: int) -> Aggregator:
    """Build an :class:`Aggregator` from a config dict / attribute object.

    Recognised config keys (all optional):
    ``aggregation_mode``, ``leader_id``, ``rotate_leader``, ``clip_epsilon``.
    """
    def _get(key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    return Aggregator(
        num_workers=num_workers,
        mode=AggregationMode.from_str(_get("aggregation_mode", "leader")),
        leader_id=int(_get("leader_id", 0) or 0),
        rotate_leader=bool(_get("rotate_leader", False)),
        clip_epsilon=float(_get("clip_epsilon", 0.1)),
    )


def split_worker_ids(num_workers: int, num_envs: int, device=None) -> torch.Tensor:
    """Assign each of ``num_envs`` environments to one of ``num_workers`` blocks.

    Environments are split as evenly as possible; the returned tensor has shape
    ``(num_envs,)`` and dtype ``torch.long``.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    base = num_envs // num_workers
    remainder = num_envs % num_workers
    ids: List[int] = []
    for w in range(num_workers):
        count = base + (1 if w < remainder else 0)
        ids.extend([w] * count)
    return torch.tensor(ids, dtype=torch.long, device=device)


def block_slices(num_workers: int, num_envs: int) -> List[slice]:
    """Return contiguous ``slice`` objects partitioning ``num_envs`` into blocks."""
    slices: List[slice] = []
    start = 0
    base = num_envs // num_workers
    remainder = num_envs % num_workers
    for w in range(num_workers):
        count = base + (1 if w < remainder else 0)
        slices.append(slice(start, start + count))
        start += count
    return slices


__all__ = [
    "AggregationMode",
    "UpdatePlan",
    "Aggregator",
    "build_aggregator",
    "split_worker_ids",
    "block_slices",
]
