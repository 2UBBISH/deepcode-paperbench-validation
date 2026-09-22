"""Aggregation schemes for SAPG (Sec. 4.3, Algorithm 1).

SAPG splits the N parallel environments into M blocks, each trained by its own
policy pi_j (j = 1..M) that shares the actor/critic backbones B_theta / C_psi but
is conditioned on a per-policy latent phi_j.

Data aggregation determines *which* policies receive off-policy data from *which*
other policies:

  * Leader-follower (default, Sec. 4.3):
        - Leader  i = 1 : X = {2, ..., M}
              L(pi_1) = L_on(D_1) + lambda * L_off(D_1'; {2..M})
        - Follower j >= 2 : X = empty
              L(pi_j) = L_on(D_j)  [+ sigma * (j-1) * H(pi_j)]

    The off-policy batch D_1' is subsampled from the union of the follower
    buffers so that |D_1'| == |D_1| (equal on/off-policy volume).

  * Symmetric (ablation, Fig. 6):
        Every policy i receives off-policy data from all other policies:
              X_i = {1, ..., M} \\ {i}
        and each D_i' is subsampled to |D_i| transitions.

  * No-off-policy (ablation, Fig. 6):
        X_i = empty for all i -> pure per-policy PPO.

  * High-off-policy-ratio (ablation, Fig. 6):
        Like leader-follower but D_1' is NOT subsampled, i.e. the leader sees the
        full union of follower data (off-policy volume >> on-policy volume).

This module only *describes* the aggregation; the actual loss assembly happens in
``sapg_trainer.py`` using ``losses.py`` primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .rollout import RolloutBuffer, subsample_off_policy


# ---------------------------------------------------------------------------
# Aggregation specification
# ---------------------------------------------------------------------------
@dataclass
class AggregationSpec:
    """Describes, for each policy i, which other policies contribute off-policy data.

    Attributes:
        scheme: One of ``"leader_follower"``, ``"symmetric"``, ``"none"``.
        num_policies: M, the number of policies / environment blocks.
        subsample: If True, each off-policy batch is subsampled to match the size
            of the receiving policy's on-policy batch (Sec. 4.3). If False, the
            full union of source data is used (high-off-policy-ratio ablation).
        lam: Off-policy loss weight lambda (Eq. 4). Default 1.0.
    """

    scheme: str = "leader_follower"
    num_policies: int = 6
    subsample: bool = True
    lam: float = 1.0

    # populated by build()
    sources: Dict[int, List[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sources = self._build_sources()

    # -- construction -------------------------------------------------------
    def _build_sources(self) -> Dict[int, List[int]]:
        M = self.num_policies
        scheme = self.scheme.lower()
        if scheme in ("leader_follower", "leader-follower", "leaderfollower"):
            # Leader (index 0) aggregates all followers; followers get nothing.
            return {i: (list(range(1, M)) if i == 0 else []) for i in range(M)}
        if scheme in ("symmetric", "sym"):
            return {i: [j for j in range(M) if j != i] for i in range(M)}
        if scheme in ("none", "no_off_policy", "no-off-policy", "on_policy"):
            return {i: [] for i in range(M)}
        raise ValueError(f"Unknown aggregation scheme: {self.scheme!r}")

    # -- queries ------------------------------------------------------------
    def is_leader(self, i: int) -> bool:
        return i == 0

    def has_off_policy(self, i: int) -> bool:
        return len(self.sources.get(i, [])) > 0

    def off_policy_weight(self, i: int) -> float:
        """lambda_i: off-policy weight for policy i (0 if it has no sources)."""
        return self.lam if self.has_off_policy(i) else 0.0

    def describe(self) -> str:
        lines = [f"AggregationSpec(scheme={self.scheme}, M={self.num_policies}, "
                 f"subsample={self.subsample}, lambda={self.lam})"]
        for i in range(self.num_policies):
            role = "leader" if self.is_leader(i) else "follower"
            lines.append(f"  pi_{i + 1} [{role}]: X = {[j + 1 for j in self.sources[i]]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Batch assembly helpers
# ---------------------------------------------------------------------------
def gather_off_policy_batch(
    buffers: Sequence[RolloutBuffer],
    source_indices: Sequence[int],
    target_size: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> Optional[Dict[str, torch.Tensor]]:
    """Build the off-policy batch D_i' for a policy.

    Args:
        buffers: All M rollout buffers (indexed by policy).
        source_indices: Indices j of policies whose data forms the source union.
        target_size: If given, subsample exactly this many transitions (Sec. 4.3
            equal-volume rule). If None, use the full union (high-off-policy
            ablation).
        generator: Optional torch RNG for reproducibility.

    Returns:
        A dict of flattened tensors (``obs``, ``actions``, ``log_probs``,
        ``rewards``, ``dones``, ``values``, ``advantages``, ``value_targets``,
        ``policy_index``) or ``None`` if there are no sources.
    """
    if not source_indices:
        return None

    src_buffers = [buffers[j] for j in source_indices]
    if target_size is None:
        # Full union: concatenate everything.
        parts = [b.flat() for b in src_buffers]
        batch: Dict[str, torch.Tensor] = {}
        for key in parts[0].keys():
            if key == "policy_index":
                continue
            batch[key] = torch.cat([p[key] for p in parts], dim=0)
        # Tag each transition with the policy that generated it (needed for the
        # behaviour ratio pi_j(s,a) in Eq. 3).
        idx = torch.cat(
            [torch.full((p["obs"].shape[0],), j, dtype=torch.long,
                        device=p["obs"].device)
             for p, j in zip(parts, source_indices)],
            dim=0,
        )
        batch["policy_index"] = idx
        return batch

    # Subsampled union: sample `target_size` transitions uniformly at random.
    return subsample_off_policy(src_buffers, target_size, generator=generator)


def build_aggregation_batches(
    buffers: Sequence[RolloutBuffer],
    spec: AggregationSpec,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, Optional[Dict[str, torch.Tensor]]]:
    """Assemble the off-policy batch for every policy according to ``spec``.

    Returns a mapping ``i -> D_i'`` (or ``None`` when policy i has no sources).
    """
    batches: Dict[int, Optional[Dict[str, torch.Tensor]]] = {}
    for i in range(spec.num_policies):
        sources = spec.sources.get(i, [])
        if not sources:
            batches[i] = None
            continue
        target_size = buffers[i].size() if spec.subsample else None
        batches[i] = gather_off_policy_batch(
            buffers, sources, target_size=target_size, generator=generator
        )
    return batches


def split_off_policy_by_source(
    batch: Dict[str, torch.Tensor],
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Split a mixed off-policy batch by the generating policy index.

    The off-policy loss (Eq. 3) requires the behaviour log-probability
    ``log pi_j(s, a)`` for each transition, where ``j`` is the policy that
    collected it. This helper groups transitions by ``policy_index`` so the
    trainer can evaluate each group under its own behaviour policy.
    """
    if batch is None or "policy_index" not in batch:
        return {}
    idx = batch["policy_index"]
    groups: Dict[int, Dict[str, torch.Tensor]] = {}
    for j in torch.unique(idx).tolist():
        mask = idx == j
        groups[int(j)] = {k: v[mask] for k, v in batch.items() if k != "policy_index"}
    return groups


# ---------------------------------------------------------------------------
# Convenience constructors for the paper's variants
# ---------------------------------------------------------------------------
def leader_follower(num_policies: int = 6, lam: float = 1.0,
                    subsample: bool = True) -> AggregationSpec:
    return AggregationSpec("leader_follower", num_policies, subsample, lam)


def symmetric(num_policies: int = 6, lam: float = 1.0,
              subsample: bool = True) -> AggregationSpec:
    return AggregationSpec("symmetric", num_policies, subsample, lam)


def no_off_policy(num_policies: int = 6) -> AggregationSpec:
    return AggregationSpec("none", num_policies, True, 0.0)


def high_off_policy_ratio(num_policies: int = 6, lam: float = 1.0) -> AggregationSpec:
    """Leader-follower without subsampling (off-policy volume >> on-policy)."""
    return AggregationSpec("leader_follower", num_policies, subsample=False, lam=lam)


def make_spec(name: str, num_policies: int = 6, lam: float = 1.0) -> AggregationSpec:
    """Factory by name, used by configs / CLI."""
    name = name.lower()
    if name in ("leader_follower", "leader-follower", "default"):
        return leader_follower(num_policies, lam)
    if name in ("symmetric", "sym"):
        return symmetric(num_policies, lam)
    if name in ("none", "no_off_policy", "on_policy"):
        return no_off_policy(num_policies)
    if name in ("high_off_policy", "high_off_policy_ratio", "no_subsample"):
        return high_off_policy_ratio(num_policies, lam)
    raise ValueError(f"Unknown aggregation name: {name!r}")
