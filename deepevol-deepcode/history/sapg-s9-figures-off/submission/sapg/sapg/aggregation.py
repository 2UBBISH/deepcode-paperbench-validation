"""Aggregation strategies for SAPG (Sec 4.2-4.3).

SAPG splits N parallel environments into M blocks, each rolled out by a
*diverse* policy pi_j (shared actor backbone B_theta / critic backbone C_psi
conditioned on a per-policy latent phi_j).  After collecting per-block buffers
D_1..D_M, the data is *aggregated* into the policy updates.

The paper's primary scheme is **leader-follower**:

    * The leader (policy 1) is updated with its own on-policy data D_1 PLUS an
      off-policy term built from the followers' data D_2..D_M (Eq. 3-4).
    * Each follower (policy j = 2..M) is updated with its own on-policy data
      only (X = empty set in Eq. 4).

Two ablations are also implemented:

    * ``symmetric``: every policy is updated with off-policy data from all the
      other policies (performs worse in the paper).
    * ``high_off_policy_ratio``: the leader uses ALL follower data (no
      subsampling down to |D_1|).
    * ``no_off_policy``: independent PPO per block (no aggregation at all).

The off-policy term for policy i is

    L_off(pi_i; X) = (1/|X|) sum_{j in X} E_{(s,a)~pi_j}[
        min( r_{pi_i}, clip(r_{pi_i}, mu(1-eps), mu(1+eps)) ) * A^{pi_{i,old}} ]

with r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a) and mu = pi_{i,old}(s,a) / pi_j(s,a).

This module is deliberately *data-plumbing only*: it decides which buffers feed
which policy's on-policy / off-policy terms and performs the subsampling.  The
actual loss arithmetic lives in :mod:`sapg.losses`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Aggregation plan description
# ---------------------------------------------------------------------------
@dataclass
class PolicyAggregationPlan:
    """Describes the data feeding one policy's update.

    Attributes
    ----------
    policy_idx:
        0-based index of the policy being updated (0 == leader).
    on_policy_block:
        Index of the block whose data is used as the on-policy term (always the
        policy's own block).
    off_policy_blocks:
        Indices of blocks whose data is aggregated into the off-policy term.
        Empty for followers under leader-follower aggregation.
    subsample_to:
        If not ``None``, the off-policy data is subsampled so that its total
        number of transitions equals this value (used to enforce
        |D_1'| = |D_1| for the leader).
    entropy_coef:
        Effective entropy coefficient for this policy (already scaled by
        ``sigma * (i - 1)`` for followers; 0 for the leader).
    """

    policy_idx: int
    on_policy_block: int
    off_policy_blocks: List[int] = field(default_factory=list)
    subsample_to: Optional[int] = None
    entropy_coef: float = 0.0

    @property
    def has_off_policy(self) -> bool:
        return len(self.off_policy_blocks) > 0


# ---------------------------------------------------------------------------
# Off-policy subsampling
# ---------------------------------------------------------------------------
def subsample_off_policy(
    buffers: Sequence["RolloutBuffer"],  # noqa: F821 (duck-typed)
    target_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Sample ``target_size`` transitions uniformly from the union of buffers.

    Implements step 3 of Algorithm 1: sample |D_1| transitions from
    union_{j=2}^{M} D_j to form D_1'.  Sampling is *with replacement* so that
    the requested size is always achievable even when the follower data is
    smaller than the leader's block (it never is in practice, since all blocks
    have equal size, but this keeps the routine robust).

    Parameters
    ----------
    buffers:
        Sequence of :class:`~sapg.rollout.RolloutBuffer` (or any object exposing
        the flattened tensors ``obs``, ``actions``, ``log_probs``, ``rewards``,
        ``dones``, ``values``, ``advantages``, ``value_targets``).
    target_size:
        Number of transitions to sample (|D_1|).
    generator:
        Optional torch RNG for reproducibility.
    device:
        Device for the sampled tensors.  Defaults to the buffers' device.

    Returns
    -------
    dict
        Flattened tensors of length ``target_size`` with the same keys as the
        buffer fields, plus ``behavior_log_probs`` (the log-probs under the
        *behaviour* policy pi_j that generated each transition) and
        ``source_block`` (which block each transition came from).
    """
    if len(buffers) == 0:
        raise ValueError("subsample_off_policy requires at least one buffer")
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    # Flatten each buffer and record its size.
    flat_buffers: List[Dict[str, torch.Tensor]] = []
    sizes: List[int] = []
    for buf in buffers:
        flat = _flatten_buffer(buf)
        flat_buffers.append(flat)
        sizes.append(flat["obs"].shape[0])

    total = int(sum(sizes))
    if total == 0:
        raise ValueError("cannot subsample from empty buffers")

    if device is None:
        device = flat_buffers[0]["obs"].device

    # Uniformly sample global indices in [0, total).
    idx = torch.randint(
        low=0,
        high=total,
        size=(target_size,),
        generator=generator,
        device=device,
    )

    # Map global indices -> (buffer_id, local_index).
    sizes_t = torch.tensor(sizes, device=device, dtype=torch.long)
    offsets = torch.cumsum(sizes_t, dim=0) - sizes_t  # start offset of each buf
    buffer_id = torch.searchsorted(offsets, idx, right=True) - 1
    buffer_id = buffer_id.clamp_(min=0, max=len(buffers) - 1)
    local_idx = idx - offsets[buffer_id]

    out: Dict[str, torch.Tensor] = {}
    keys = list(flat_buffers[0].keys())
    for key in keys:
        # Gather per-key using the (buffer_id, local_idx) pairs.
        gathered = torch.empty(
            (target_size,) + flat_buffers[0][key].shape[1:],
            dtype=flat_buffers[0][key].dtype,
            device=device,
        )
        for b_id, buf in enumerate(flat_buffers):
            mask = buffer_id == b_id
            if mask.any():
                gathered[mask] = buf[key][local_idx[mask]]
        out[key] = gathered

    # The behaviour log-probs are the log-probs stored in the source buffer
    # (they were produced by the behaviour policy pi_j during rollout).
    out["behavior_log_probs"] = out["log_probs"].clone()
    out["source_block"] = buffer_id
    return out


def _flatten_buffer(buf) -> Dict[str, torch.Tensor]:
    """Flatten a time-major RolloutBuffer into a dict of [T*B, ...] tensors."""
    flat: Dict[str, torch.Tensor] = {}
    for key in ("obs", "actions", "log_probs", "rewards", "dones", "values"):
        tensor = getattr(buf, key, None)
        if tensor is None:
            continue
        flat[key] = tensor.reshape(-1, *tensor.shape[2:])
    # Optional fields produced by compute_returns.
    for key in ("advantages", "value_targets", "valid_mask"):
        tensor = getattr(buf, key, None)
        if tensor is not None:
            flat[key] = tensor.reshape(-1, *tensor.shape[2:])
    return flat


# ---------------------------------------------------------------------------
# Aggregation plan builders
# ---------------------------------------------------------------------------
def build_aggregation_plans(
    num_blocks: int,
    aggregation: str = "leader_follower",
    entropy_coef: float = 0.0,
    subsample_off_policy: bool = True,
    block_size: Optional[int] = None,
) -> List[PolicyAggregationPlan]:
    """Build the per-policy aggregation plan for a given scheme.

    Parameters
    ----------
    num_blocks:
        Number of environment blocks / policies M.
    aggregation:
        One of ``leader_follower``, ``symmetric``, ``high_off_policy_ratio``,
        ``no_off_policy``.
    entropy_coef:
        Base entropy coefficient sigma.  Follower i (1-based) receives
        ``sigma * (i - 1)``; the leader receives 0 (Sec 4.5).
    subsample_off_policy:
        Whether to subsample the leader's off-policy data to |D_1|.
    block_size:
        Number of transitions per block (T * B).  Required when
        ``subsample_off_policy`` is True so that the target size can be set.

    Returns
    -------
    list[PolicyAggregationPlan]
        One plan per policy, ordered by policy index.
    """
    aggregation = aggregation.lower()
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")

    plans: List[PolicyAggregationPlan] = []

    if aggregation == "no_off_policy":
        # Independent PPO per block: no aggregation, no entropy coupling.
        for i in range(num_blocks):
            plans.append(
                PolicyAggregationPlan(
                    policy_idx=i,
                    on_policy_block=i,
                    off_policy_blocks=[],
                    subsample_to=None,
                    entropy_coef=0.0,
                )
            )
        return plans

    if aggregation == "leader_follower":
        for i in range(num_blocks):
            if i == 0:
                # Leader: off-policy from all followers, subsampled to |D_1|.
                off_blocks = list(range(1, num_blocks))
                target = block_size if (subsample_off_policy and block_size) else None
                plans.append(
                    PolicyAggregationPlan(
                        policy_idx=i,
                        on_policy_block=i,
                        off_policy_blocks=off_blocks,
                        subsample_to=target,
                        entropy_coef=0.0,  # leader has NO entropy term
                    )
                )
            else:
                # Follower: on-policy only; entropy scaled by (i) with 0-based i
                # so that follower index (i+1) gets sigma * i.
                plans.append(
                    PolicyAggregationPlan(
                        policy_idx=i,
                        on_policy_block=i,
                        off_policy_blocks=[],
                        subsample_to=None,
                        entropy_coef=entropy_coef * i,
                    )
                )
        return plans

    if aggregation == "symmetric":
        # Every policy aggregates off-policy data from all *other* policies.
        for i in range(num_blocks):
            off_blocks = [j for j in range(num_blocks) if j != i]
            target = block_size if (subsample_off_policy and block_size) else None
            plans.append(
                PolicyAggregationPlan(
                    policy_idx=i,
                    on_policy_block=i,
                    off_policy_blocks=off_blocks,
                    subsample_to=target,
                    entropy_coef=entropy_coef * i,
                )
            )
        return plans

    if aggregation == "high_off_policy_ratio":
        # Leader uses ALL follower data (no subsampling).
        for i in range(num_blocks):
            if i == 0:
                plans.append(
                    PolicyAggregationPlan(
                        policy_idx=i,
                        on_policy_block=i,
                        off_policy_blocks=list(range(1, num_blocks)),
                        subsample_to=None,  # no subsampling
                        entropy_coef=0.0,
                    )
                )
            else:
                plans.append(
                    PolicyAggregationPlan(
                        policy_idx=i,
                        on_policy_block=i,
                        off_policy_blocks=[],
                        subsample_to=None,
                        entropy_coef=entropy_coef * i,
                    )
                )
        return plans

    raise ValueError(f"Unknown aggregation scheme: {aggregation!r}")


# ---------------------------------------------------------------------------
# Convenience: gather the off-policy batch for a plan
# ---------------------------------------------------------------------------
def gather_off_policy_batch(
    plan: PolicyAggregationPlan,
    buffers: Sequence["RolloutBuffer"],  # noqa: F821
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return the off-policy batch for ``plan`` (or ``None`` if not applicable).

    The returned dict contains flattened tensors plus ``behavior_log_probs``
    (log-probs under the behaviour policy pi_j) and ``source_block``.
    """
    if not plan.has_off_policy:
        return None

    selected = [buffers[j] for j in plan.off_policy_blocks]
    if plan.subsample_to is not None:
        return subsample_off_policy(
            selected,
            target_size=plan.subsample_to,
            generator=generator,
            device=device,
        )

    # No subsampling: concatenate all selected buffers.
    flats = [_flatten_buffer(b) for b in selected]
    out: Dict[str, torch.Tensor] = {}
    for key in flats[0].keys():
        out[key] = torch.cat([f[key] for f in flats], dim=0)
    out["behavior_log_probs"] = out["log_probs"].clone()
    # Record source block ids for diagnostics.
    src = []
    for b_id, f in zip(plan.off_policy_blocks, flats):
        src.append(torch.full((f["obs"].shape[0],), b_id, dtype=torch.long))
    out["source_block"] = torch.cat(src, dim=0)
    return out


__all__ = [
    "PolicyAggregationPlan",
    "build_aggregation_plans",
    "subsample_off_policy",
    "gather_off_policy_batch",
]
