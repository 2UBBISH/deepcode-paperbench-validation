"""Leader-follower aggregation for SAPG (Section 4.3, Algorithm 1 in Section 4.6).

The paper's aggregation rule is:

* One policy is designated the *leader* ``i = 1``.  It is updated with off-policy
  data coming from all other policies ``X = {2, 3, ..., M}`` (Section 4.3).
* Every other policy is a *follower* and uses only its own on-policy data,
  i.e. ``X = {}``.
* In both cases ``lambda = 1`` (the off-policy weight, Eq. 4), but the off-policy
  data of the leader is *subsampled* so that an update uses equal amounts of
  on-policy and off-policy data (:math:`|D'_1| = |D_1|`), because off-policy
  gradients are noisier than on-policy gradients (Section 4.3, Section 6.3).

This module provides the controller used by :mod:`sapg.algorithms.sapg`:

* :class:`LeaderFollowerAggregator` -- the leader/follower controller.  It knows
  which policies feed which policy (``data_sources``), whether an off-policy batch
  has to be built and subsampled, and it can compute the combined objective

      ``L = OFFPolicyLoss(D'_1) + ONPolicyLoss(D_1) + sum_{j>=2} ONPolicyLoss(D_j)``

  of Algorithm 1 (the off-policy term only exists for the leader).
* :func:`make_aggregation` -- factory mirroring the plan's
  ``make_aggregation(name, ...)`` entry point; ``sapg.algorithms.sapg`` falls back
  to it when the ``sapg.aggregation`` package is importable.

The module is deliberately independent of IsaacGym: it only needs a policy object
exposing ``evaluate_actions`` (or ``evaluate``) and buffer objects exposing a dict
of flattened tensors, so that it can be unit-tested with synthetic data.

Indexing convention (as in the paper)
-------------------------------------
*Policy indices are 1-based* (``1 .. M``), the leader being index ``1`` and the
followers ``2 .. M``.  Buffers are stored 0-based, hence the ``-1`` translations
in the helper functions below.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..utils.config import NUM_POLICIES, SAPGConfig

__all__ = [
    "LeaderFollowerAggregator",
    "OffPolicySubsampler",
    "AggregationStats",
    "make_aggregation",
    "subsample_indices",
    "LEADER_FOLLOWER",
    "SYMMETRIC",
    "NO_AGGREGATION",
]

#: aggregation scheme names (also used by the trainers / ablation scripts)
LEADER_FOLLOWER = "leader_follower"
SYMMETRIC = "symmetric"
NO_AGGREGATION = "none"

# Keys required to evaluate a policy on stored transitions.
_DEFAULT_KEYS: Tuple[str, ...] = (
    "obs",
    "actions",
    "logprobs",
    "values",
    "advantages",
    "value_targets",
    "value_targets_off",
    "mu",
    "dones",
    "rewards",
    "source_policy",
    "policy_index",
    "masks",
)

# Keys that are metadata rather than arguments to ``policy.evaluate_actions``.
_NON_EVAL_KEYS = frozenset(
    {"source_policy", "rewards", "dones", "value_targets", "value_targets_off", "advantages", "mu"}
)

_TARGET_OLD_KEYS = ("target_old_logprobs", "source_old_logprobs", "old_logprobs_source", "logprobs_j")
_NEW_LOGPROB_KEYS = ("new_logprobs", "logprobs", "logprob", "log_prob")
_ADV_KEYS = ("advantages", "adv", "advantage")
_MU_KEYS = ("mu", "mu_weights", "importance_mu")
_SOURCE_KEYS = ("source_policy", "policy_source", "data_source", "behaviour_policy")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _first_key(data: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in data and data[key] is not None:
            return key
    return None


def _leader_index_0based(leader_index: int) -> int:
    """Translate the paper's 1-based leader index into a buffer index."""
    return int(leader_index) - 1 if leader_index >= 1 else int(leader_index)


def _buffer_size(buffer: Any) -> int:
    """Number of transitions stored in a per-policy buffer."""
    if buffer is None:
        return 0
    for attr in ("size", "num_samples", "total_samples", "n_samples", "numel"):
        value = getattr(buffer, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    horizon = getattr(buffer, "horizon_length", None)
    num_envs = getattr(buffer, "num_envs", None)
    if horizon is not None and num_envs is not None:
        try:
            return int(horizon) * int(num_envs)
        except (TypeError, ValueError):
            pass
    data = _raw_data(buffer)
    if data:
        for value in data.values():
            if torch.is_tensor(value) and value.numel():
                return int(value.reshape(value.shape[0], -1).shape[0])
    return 0


def _raw_data(buffer: Any) -> Dict[str, torch.Tensor]:
    """Best-effort access to the stored tensors of a buffer."""
    if isinstance(buffer, dict):
        return buffer
    data = getattr(buffer, "data", None)
    if isinstance(data, dict):
        return data
    storage = getattr(buffer, "storage", None)
    if isinstance(storage, dict):
        return storage
    result: Dict[str, torch.Tensor] = {}
    for key in _DEFAULT_KEYS:
        value = getattr(buffer, key, None)
        if torch.is_tensor(value):
            result[key] = value
    return result


def _flatten_buffer(buffer: Any, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """Flatten a rolled-out buffer to ``[num_samples, ...]`` tensors.

    Uses the buffer's own ``flatten``/``sample`` API when available and falls back
    to a manual reshape of the raw ``[horizon, num_envs, ...]`` tensors.
    """
    for name in ("flatten", "flat", "to_dict", "as_dict", "get_data"):
        fn = getattr(buffer, name, None)
        if not callable(fn):
            continue
        try:
            data = fn()
        except TypeError:
            try:
                data = fn(device=device)
            except Exception:  # pragma: no cover - defensive
                continue
        except Exception:  # pragma: no cover - defensive
            continue
        if isinstance(data, dict) and data:
            return {
                k: (v.to(device) if device is not None and torch.is_tensor(v) else v)
                for k, v in data.items()
            }

    data = _raw_data(buffer)
    if not data:
        raise TypeError(
            "Cannot flatten buffer of type %r: no flatten()/data attribute found."
            % type(buffer).__name__
        )
    flat: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if not torch.is_tensor(value):
            flat[key] = value
            continue
        if value.dim() <= 1:
            tensor = value
        else:
            tensor = value.reshape(value.shape[0], -1, *value.shape[2:])
            tensor = tensor.reshape(-1, *value.shape[2:])
        flat[key] = tensor.to(device) if device is not None else tensor
    return flat


def subsample_indices(
    total: int,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Uniformly sample ``num_samples`` indices (without replacement) from ``range(total)``."""
    if num_samples >= total:
        return torch.arange(total, device=device)
    try:
        perm = torch.randperm(total, generator=generator, device=device)
    except TypeError:  # older torch: randperm without device kwarg
        perm = torch.randperm(total, generator=generator).to(device)
    return perm[:num_samples]


def _select(data: Dict[str, torch.Tensor], index: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Index every tensor-like entry of ``data`` along its first dimension."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if torch.is_tensor(value) and value.shape[:1] == index.shape[:1] and value.shape[0] == index.numel():
            out[key] = value[index]
        elif torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] > index.numel():
            # tensors that live on a different (larger) batch axis are kept as-is
            out[key] = value
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# statistics container
# ---------------------------------------------------------------------------
@dataclass
class AggregationStats:
    """Book-keeping for one aggregation step."""

    leader_index: int = 1
    on_policy_samples: int = 0
    off_policy_samples: int = 0
    subsampled: bool = False
    sources: Tuple[int, ...] = ()
    stats: Dict[str, float] = field(default_factory=dict)

    def as_dict(self, prefix: str = "agg/") -> Dict[str, float]:
        out = {
            f"{prefix}leader_index": float(self.leader_index),
            f"{prefix}on_policy_samples": float(self.on_policy_samples),
            f"{prefix}off_policy_samples": float(self.off_policy_samples),
            f"{prefix}subsampled": float(bool(self.subsampled)),
            f"{prefix}num_sources": float(len(self.sources)),
        }
        for key, value in self.stats.items():
            out[f"{prefix}{key}"] = float(value)
        return out


# ---------------------------------------------------------------------------
# subsampler (Section 4.3 / Section 6.3 "high off-policy ratio" ablation)
# ---------------------------------------------------------------------------
class OffPolicySubsampler:
    """Builds the leader's off-policy batch ``D'_1`` and subsamples it.

    Section 4.3: *"we choose lambda = 1, but subsample the off-policy data for the
    leader such that we use equal amounts of on-policy and off-policy data in a
    minibatch update"*, i.e. ``|D'_1| = |D_1|``.

    Section 6.3 ("SAPG (high off-policy ratio)") removes this step and uses the
    entire combined dataset; this is supported with ``subsample=False``.
    """

    def __init__(
        self,
        subsample: bool = True,
        sample_fraction: float = 1.0,
        min_samples: int = 0,
        per_source: bool = False,
    ) -> None:
        self.subsample = bool(subsample)
        self.sample_fraction = float(sample_fraction)
        self.min_samples = int(min_samples)
        self.per_source = bool(per_source)

    def num_samples(self, on_policy_size: int, off_policy_size: int, num_sources: int = 1) -> int:
        """Target size of the off-policy batch."""
        if not self.subsample:
            return int(off_policy_size)
        if self.per_source:
            per = int(math.ceil(on_policy_size / max(1, num_sources)))
            return min(int(off_policy_size), per * max(1, num_sources))
        target = int(round(on_policy_size * self.sample_fraction))
        target = max(target, self.min_samples)
        return min(int(off_policy_size), max(0, target))


# ---------------------------------------------------------------------------
# main controller
# ---------------------------------------------------------------------------
class LeaderFollowerAggregator:
    """Leader/follower data-aggregation controller (Section 4.3, Algorithm 1).

    Parameters
    ----------
    config:
        Optional :class:`~sapg.utils.config.SAPGConfig`; when given, the aggregation
        parameters (``num_policies``, ``leader_index``, ``off_policy_weight``,
        ``subsample_off_policy``, ``clip_epsilon``) are read from it.
    num_policies:
        Number of policies ``M`` into which the ``N`` environments are split.
    leader_index:
        *1-based* index of the leader policy (the paper fixes ``i = 1``).
    lam:
        Off-policy weight ``lambda`` of Eq. 4 (the paper uses ``lambda = 1`` with
        the off-policy batch subsampled to the on-policy batch size).
    subsample:
        Whether to subsample the fused off-policy data down to ``|D_1|``.
    sample_fraction:
        Multiplier applied to ``|D_1|`` when computing the off-policy budget
        (``1.0`` reproduces the paper; larger values implement the "high
        off-policy ratio" ablation of Section 6.3 together with
        ``subsample=False``).
    clip_epsilon:
        PPO clipping range ``eps`` used by both the on- and off-policy surrogates.
    """

    def __init__(
        self,
        config: Optional[SAPGConfig] = None,
        num_policies: Optional[int] = None,
        leader_index: Optional[int] = None,
        lam: Optional[float] = None,
        subsample: Optional[bool] = None,
        sample_fraction: float = 1.0,
        per_source_subsample: bool = False,
        clip_epsilon: Optional[float] = None,
        normalize_advantages: Optional[bool] = None,
        critic_coefficient: Optional[float] = None,
        keys: Optional[Sequence[str]] = None,
        device: Optional[Any] = None,
        off_policy_loss_fn: Optional[Callable[..., Any]] = None,
        on_policy_loss_fn: Optional[Callable[..., Any]] = None,
        normalize_adv_leader_only: bool = True,
    ) -> None:
        if config is not None:
            num_policies = num_policies if num_policies is not None else getattr(config, "num_policies", None)
            leader_index = leader_index if leader_index is not None else getattr(config, "leader_index", None)
            if lam is None:
                lam = getattr(config, "off_policy_weight", None)
            if subsample is None:
                subsample = getattr(config, "subsample_off_policy", None)
            if clip_epsilon is None:
                clip_epsilon = getattr(config, "clip_epsilon", None)
            if normalize_advantages is None:
                normalize_advantages = getattr(config, "normalize_advantages", None)
            if critic_coefficient is None:
                critic_coefficient = getattr(config, "critic_coefficient", None)
        self.config = config
        self.num_policies = int(num_policies if num_policies is not None else NUM_POLICIES)
        self.leader_index = int(leader_index if leader_index is not None else 1)
        self.lam = float(1.0 if lam is None else lam)  # Eq. 4 off-policy weight
        self.clip_epsilon = float(0.1 if clip_epsilon is None else clip_epsilon)
        self.sample_fraction = float(sample_fraction)
        self.normalize_advantages = bool(normalize_advantages) if normalize_advantages is not None else False
        self.critic_coefficient = float(
            4.0 if critic_coefficient is None else critic_coefficient
        )
        self.keys = tuple(keys) if keys is not None else _DEFAULT_KEYS
        self.device = device
        self.name = LEADER_FOLLOWER
        self._off_policy_loss_fn = off_policy_loss_fn
        self._on_policy_loss_fn = on_policy_loss_fn
        self.normalize_adv_leader_only = bool(normalize_adv_leader_only)
        self.subsampler = OffPolicySubsampler(
            subsample=True if subsample is None else bool(subsample),
            sample_fraction=self.sample_fraction,
            per_source=bool(per_source_subsample),
        )
        # 1-based policy indices of the followers
        self.follower_indices: Tuple[int, ...] = tuple(
            i for i in range(1, self.num_policies + 1) if i != self.leader_index
        )

    # -- introspection ------------------------------------------------------
    @property
    def leader(self) -> int:
        """1-based leader index."""
        return self.leader_index

    @property
    def followers(self) -> Tuple[int, ...]:
        return self.follower_indices

    def is_leader(self, i: int) -> bool:
        return int(i) == self.leader_index

    def off_policy_weight(self, i: int) -> float:
        """``lambda`` for policy ``i`` (0 for followers: they are on-policy only)."""
        return self.lam if self.is_leader(i) else 0.0

    def data_sources(self, i: int) -> List[int]:
        """Set ``X`` of policies whose data updates policy ``i`` (Section 4.3).

        The leader (``i = 1``) receives ``X = {2, ..., M}``; followers receive
        ``X = {}`` (on-policy only), matching Algorithm 1 of Section 4.6.
        """
        if self.is_leader(i):
            return [j for j in range(1, self.num_policies + 1) if j != self.leader_index]
        return []

    def subsample_for(self, i: int) -> bool:
        """Whether the off-policy data of policy ``i`` is subsampled (leader only)."""
        return self.is_leader(i) and self.subsampler.subsample

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(M={self.num_policies}, leader={self.leader_index}, "
            f"lambda={self.lam}, subsample={self.subsampler.subsample}, eps={self.clip_epsilon})"
        )

    # -- off-policy batch construction --------------------------------------
    def off_policy_budget(self, on_policy_size: int, off_policy_size: int, num_sources: int = 1) -> int:
        """Number of off-policy samples used per update (``|D'_1|``)."""
        return self.subsampler.num_samples(on_policy_size, off_policy_size, num_sources)

    def build_off_policy_batch(
        self,
        leader_buffer: Any,
        follower_buffers: Sequence[Any],
        generator: Optional[torch.Generator] = None,
        num_samples: Optional[int] = None,
        leader_next_values: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        return_stats: bool = False,
    ):
        """Fuse the follower datasets into the leader's off-policy batch ``D'_1``.

        The fused batch is uniformly subsampled so that ``|D'_1| = |D_1|`` unless
        subsampling is disabled (Section 6.3 "high off-policy ratio").
        """
        device = device if device is not None else self.device
        try:
            from ..buffers.rollout_buffer import build_off_policy_batch as _build
        except Exception:  # pragma: no cover - buffer module should always exist
            _build = None

        num_sources = len([b for b in follower_buffers if b is not None])
        on_policy_size = max(_buffer_size(leader_buffer), 1 if num_samples is None else int(num_samples))
        # total available off-policy samples
        off_policy_size = sum(_buffer_size(b) for b in follower_buffers if b is not None)
        if num_samples is None:
            num_samples = self.off_policy_budget(on_policy_size, off_policy_size, num_sources)
        subsample = self.subsampler.subsample and num_samples < off_policy_size

        if _build is not None:
            kwargs: Dict[str, Any] = dict(
                subsample=subsample,
                num_samples=num_samples if subsample else None,
                generator=generator,
                device=device,
                leader_next_values=leader_next_values,
            )
            try:
                batch = _build(leader_buffer, list(follower_buffers), **kwargs)
            except TypeError:
                batch = _build(leader_buffer, list(follower_buffers), subsample=subsample, device=device)
            stats = AggregationStats(
                leader_index=self.leader_index,
                on_policy_samples=on_policy_size,
                off_policy_samples=int(num_samples),
                subsampled=bool(subsample),
                sources=tuple(self.follower_indices),
            )
            return (batch, stats) if return_stats else batch

        # ---- fallback: fuse manually -------------------------------------
        data = self._concat_followers(follower_buffers, device=device)
        if subsample and num_samples < off_policy_size:
            index = subsample_indices(off_policy_size, int(num_samples), generator, device)
            data = _select(data, index)
        stats = AggregationStats(
            leader_index=self.leader_index,
            on_policy_samples=on_policy_size,
            off_policy_samples=int(data[next(iter(data))].shape[0]) if data else 0,
            subsampled=bool(subsample),
            sources=tuple(self.follower_indices),
        )
        return (data, stats) if return_stats else data

    def _concat_followers(
        self, follower_buffers: Sequence[Any], device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Flatten and concatenate follower buffers, tagging the source policy."""
        chunks: List[Dict[str, torch.Tensor]] = []
        meta: List[int] = []
        for offset, buffer in enumerate(follower_buffers):
            if buffer is None:
                continue
            policy_index = getattr(buffer, "policy_index", None)
            source = (int(policy_index) + 1) if policy_index is not None else self.follower_indices[offset]
            flat = _flatten_buffer(buffer, device=device)
            n = next((int(v.shape[0]) for v in flat.values() if torch.is_tensor(v) and v.dim() >= 1), 0)
            flat["source_policy"] = torch.full((n,), int(source), dtype=torch.long, device=device)
            chunks.append(flat)
            meta.append(int(source))
        if not chunks:
            return {}
        keys = set().union(*[set(c.keys()) for c in chunks])
        merged: Dict[str, torch.Tensor] = {}
        for key in keys:
            values = [c[key] for c in chunks if key in c]
            if all(torch.is_tensor(v) for v in values):
                try:
                    merged[key] = torch.cat(values, dim=0)
                except RuntimeError:  # pragma: no cover - shape mismatch
                    merged[key] = values[0]
            else:
                merged[key] = values[0]
        return merged

    # -- objective assembly -------------------------------------------------
    def aggregate(
        self,
        buffers: Sequence[Any],
        policy: Any,
        off_policy_batch: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        include_off_policy: bool = True,
        max_off_policy_samples: Optional[int] = None,
        return_mapping: bool = False,
    ) -> Any:
        """Compute Algorithm 1's summed objective for a single update.

        ``L = ONPolicyLoss(D_1) + OFFPolicyLoss(D'_1) + sum_{j>=2} ONPolicyLoss(D_j)``

        (the off-policy term only exists for the leader; the paper fixes
        ``lambda = 1``, hence the coefficient is ``self.lam``).  Returns the scalar
        loss, or ``(loss, mapping)`` with the per-policy terms when
        ``return_mapping=True``.
        """
        device = device if device is not None else self.device
        mapping: Dict[str, Any] = {"on_policy": {}, "off_policy": None, "stats": {}}
        total = torch.zeros((), device=device)

        leader_buffer = (
            buffers[_leader_index_0based(self.leader_index)]
            if len(buffers) > _leader_index_0based(self.leader_index)
            else None
        )

        # (a) on-policy objective for every policy (Algorithm 1, lines 3-5)
        for i, buffer in enumerate(buffers, start=1):
            if buffer is None:
                continue
            term = self.on_policy_term(buffer, policy, policy_index=i, device=device)
            mapping["on_policy"][i] = term
            total = total + term["loss"]

        # (b) off-policy objective of the leader (Algorithm 1, line 6)
        if include_off_policy and leader_buffer is not None:
            followers = [b for k, b in enumerate(buffers, start=1) if k != self.leader_index and b is not None]
            if followers:
                if off_policy_batch is None:
                    built = self.build_off_policy_batch(
                        leader_buffer,
                        followers,
                        generator=generator,
                        num_samples=max_off_policy_samples,
                        device=device,
                        return_stats=True,
                    )
                    off_policy_batch, stats = built
                    mapping["stats"].update(stats.as_dict(prefix=""))
                term = self.off_policy_term(off_policy_batch, policy, device=device)
                mapping["off_policy"] = term
                total = total + float(self.lam) * term["loss"]

        if return_mapping:
            return total, mapping
        return total

    def on_policy_term(
        self,
        buffer: Any,
        policy: Any,
        policy_index: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """On-policy PPO term for one policy's own dataset (Eq. 2 + critic loss)."""
        data = _flatten_buffer(buffer, device=device)
        outputs = self._evaluate(policy, data, policy_index)
        advantages = self._get(data, outputs, _ADV_KEYS)
        if advantages is None:
            advantages = data.get("advantages")
        new_logprobs = self._get(data, outputs, _NEW_LOGPROB_KEYS)
        values = self._get(data, outputs, ("values", "value", "new_values"))
        targets = data.get("value_targets", data.get("returns"))

        loss_fn = self._on_policy_loss_fn
        if loss_fn is None:
            loss_fn = _default_on_policy_loss
        term = loss_fn(
            new_logprobs=new_logprobs,
            old_logprobs=data.get("logprobs"),
            values=values,
            value_targets=targets,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
            critic_coefficient=self.critic_coefficient,
            normalize_advantages=self.normalize_advantages,
        )
        if not isinstance(term, dict):
            term = {"loss": term}
        return term

    def off_policy_term(
        self,
        off_policy_batch: Any,
        policy: Any,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Off-policy importance-sampled term for the leader's fused batch (Eq. 3)."""
        data = off_policy_batch
        if not isinstance(data, dict):
            try:
                data = data.to_dict() if hasattr(data, "to_dict") else dict(off_policy_batch)
            except Exception:  # pragma: no cover - defensive
                data = _raw_data(off_policy_batch)
        if device is not None:
            data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

        # The fused batch does not carry the source policy's ``phi`` per sample, so
        # every sample is evaluated as the leader's own objective with the
        # ``mu = pi_{i,old}/pi_j`` correction of Eq. 3.
        outputs = self._evaluate(policy, data, self.leader_index)
        new_logprobs = self._get(data, outputs, _NEW_LOGPROB_KEYS)
        behaviour_logprobs = (
            data.get("behaviour_logprobs")
            if "behaviour_logprobs" in data
            else data.get("logprobs_j", data.get("source_logprobs", data.get("logprobs")))
        )
        target_old = self._get(data, outputs, _TARGET_OLD_KEYS)
        if target_old is None:
            target_old = data.get("mu_logprobs")
        advantages = self._get(data, outputs, _ADV_KEYS)
        if advantages is None:
            advantages = data.get("advantages")

        loss_fn = self._off_policy_loss_fn
        if loss_fn is None:
            loss_fn = _default_off_policy_loss
        result = loss_fn(
            new_logprobs=new_logprobs,
            behaviour_logprobs=behaviour_logprobs,
            target_old_logprobs=target_old,
            advantages=advantages,
            mu=data.get("mu"),
            clip_epsilon=self.clip_epsilon,
            source_policy=data.get("source_policy"),
            reduction="source",
            normalize_advantages=self.normalize_advantages,
        )
        if not isinstance(result, dict):
            result = {"loss": result}
        if "off_policy_loss" in result:
            result.setdefault("loss", result["off_policy_loss"])
        if "value_targets_off" in data and "values" in result:
            pass  # value loss handled by the trainer's critic module
        return result

    # -- evaluation plumbing ------------------------------------------------
    def _evaluate(
        self,
        policy: Any,
        data: Dict[str, torch.Tensor],
        policy_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Call ``policy.evaluate_actions`` with a flexible signature."""
        if policy is None:
            return {}
        obs = data.get("obs")
        actions = data.get("actions")
        args: Dict[str, Any] = {}
        if obs is not None:
            args["obs"] = obs
        if actions is not None:
            args["actions"] = actions
        for key in ("phi", "hidden_state", "masks", "policy_index"):
            if key in data and data[key] is not None:
                args[key] = data[key]
        if policy_index is not None and "policy_index" not in args:
            args["policy_index"] = int(policy_index)

        for name in ("evaluate_actions", "evaluate"):
            fn = getattr(policy, name, None)
            if not callable(fn):
                continue
            for call in (
                lambda: fn(args),
                lambda: fn(**args),
                lambda: fn(obs, actions, **{k: v for k, v in args.items() if k not in ("obs", "actions")}),
            ):
                try:
                    out = call()
                except (TypeError, KeyError, IndexError):
                    continue
                except Exception:  # pragma: no cover - runtime errors propagate
                    raise
                if isinstance(out, dict):
                    return out
        return {}

    @staticmethod
    def _get(
        data: Dict[str, torch.Tensor],
        outputs: Dict[str, Any],
        candidates: Sequence[str],
    ) -> Optional[torch.Tensor]:
        key = _first_key(outputs, candidates)
        if key is not None:
            return outputs[key]
        key = _first_key(data, candidates)
        return data[key] if key is not None else None


# ---------------------------------------------------------------------------
# default loss implementations (imported lazily to avoid circular imports)
# ---------------------------------------------------------------------------
def _default_on_policy_loss(
    new_logprobs: Optional[torch.Tensor],
    old_logprobs: Optional[torch.Tensor],
    values: Optional[torch.Tensor],
    value_targets: Optional[torch.Tensor],
    advantages: Optional[torch.Tensor],
    clip_epsilon: float = 0.1,
    critic_coefficient: float = 4.0,
    normalize_advantages: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Eq. 2 clipped surrogate + ``lambda' *`` value loss (λ′ = 4.0, Appendix B)."""
    try:  # prefer the trainer's implementation when importable
        from ..algorithms.ppo import compute_value_loss
    except Exception:  # pragma: no cover
        compute_value_loss = None

    out: Dict[str, Any] = {}
    policy_loss = torch.zeros((), device=_as_device(advantages, new_logprobs, values))
    if new_logprobs is not None and old_logprobs is not None and advantages is not None:
        adv = advantages
        if normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        out["surrogate"] = torch.min(surr1, surr2).mean()
        with torch.no_grad():
            out["clip_frac"] = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()
            out["kl"] = (old_logprobs - new_logprobs).mean()
    out["policy_loss"] = policy_loss

    value_loss = torch.zeros((), device=policy_loss.device)
    if values is not None and value_targets is not None:
        if compute_value_loss is not None:
            vout = compute_value_loss(values, value_targets, coefficient=critic_coefficient)
            value_loss = vout.get("value_loss_scaled", vout.get("value_loss", value_loss))
            out.update({k: v for k, v in vout.items() if k != "value_loss_scaled"})
        else:
            value_loss = critic_coefficient * ((values - value_targets) ** 2).mean()
    out["value_loss"] = value_loss
    out["loss"] = policy_loss + value_loss
    return out


def _default_off_policy_loss(**kwargs: Any) -> Dict[str, Any]:
    """Eq. 3 via :mod:`sapg.losses.off_policy_loss` (falls back to a local copy)."""
    try:
        from ..losses.off_policy_loss import off_policy_loss as _fn
    except Exception:  # pragma: no cover
        _fn = None
    if _fn is not None:
        return _fn(**kwargs)
    return _local_off_policy_loss(**kwargs)


def _local_off_policy_loss(
    new_logprobs: Optional[torch.Tensor] = None,
    behaviour_logprobs: Optional[torch.Tensor] = None,
    target_old_logprobs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    clip_epsilon: float = 0.1,
    source_policy: Optional[torch.Tensor] = None,
    reduction: str = "source",
    normalize_advantages: bool = False,
    weight: float = 1.0,
    eps: float = 1e-8,
    **kwargs: Any,
) -> Dict[str, Any]:
    device = _as_device(advantages, new_logprobs, target_old_logprobs, behaviour_logprobs)
    zero = torch.zeros((), device=device)
    if new_logprobs is None or behaviour_logprobs is None or advantages is None:
        return {"off_policy_loss": zero, "loss": zero, "surrogate": zero}

    adv = advantages
    if normalize_advantages and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + eps)
    ratio = torch.exp(new_logprobs - behaviour_logprobs)  # r = pi_i(s,a) / pi_j(s,a)
    if mu is None:
        if target_old_logprobs is not None:
            mu = torch.exp(target_old_logprobs - behaviour_logprobs)
        else:
            mu = torch.ones_like(ratio)
    mu = mu.detach()
    lower = mu * (1.0 - clip_epsilon)
    upper = mu * (1.0 + clip_epsilon)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, min=None, max=None) .clamp(lower, upper) * adv)

    if reduction == "source" and source_policy is not None:
        losses = []
        for src in torch.unique(source_policy):
            mask = source_policy == src
            if mask.any():
                losses.append(surrogate[mask].mean())
        off = -(torch.stack(losses).mean() if losses else zero)
    else:
        off = -surrogate.mean()
    with torch.no_grad():
        mu_mean = mu.mean() if mu.numel() else zero
        clip_frac = ((ratio < lower) | (ratio > upper)).float().mean()
    return {
        "off_policy_loss": weight * off,
        "loss": weight * off,
        "surrogate": surrogate.mean(),
        "ratio_mean": ratio.mean().detach(),
        "mu_mean": mu_mean,
        "clip_frac": clip_frac.detach(),
    }


def _as_device(*tensors: Any) -> torch.device:
    for t in tensors:
        if torch.is_tensor(t):
            return t.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_aggregation(
    name: str = LEADER_FOLLOWER,
    config: Optional[SAPGConfig] = None,
    num_policies: Optional[int] = None,
    leader_index: Optional[int] = None,
    lam: Optional[float] = None,
    subsample: Optional[bool] = None,
    sample_fraction: float = 1.0,
    clip_epsilon: Optional[float] = None,
    **kwargs: Any,
) -> Any:
    """Build an aggregation controller by name.

    * ``"leader_follower"`` (default) -- Section 4.3: only the leader receives
      off-policy data, subsampled so that ``|D'_1| = |D_1|``.
    * ``"symmetric"`` -- Section 4.2: every policy consumes the data of all others.
    * ``"none"``/``"no_off_policy"`` -- ablation without any aggregation.

    The returned object always exposes ``data_sources(i)`` so it can be used as a
    drop-in aggregation *scheme* by the trainers.
    """
    key = str(name).lower().replace("-", "_")
    if key in ("leader_follower", "leader", "lf", "sapg"):
        return LeaderFollowerAggregator(
            config=config,
            num_policies=num_policies,
            leader_index=leader_index,
            lam=lam,
            subsample=subsample,
            sample_fraction=sample_fraction,
            clip_epsilon=clip_epsilon,
            **kwargs,
        )
    if key in ("symmetric", "sym", "uniform"):
        try:
            from .symmetric import SymmetricAggregator

            return SymmetricAggregator(
                config=config,
                num_policies=num_policies,
                lam=lam,
                subsample=subsample,
                sample_fraction=sample_fraction,
                clip_epsilon=clip_epsilon,
                **kwargs,
            )
        except Exception:  # pragma: no cover - symmetric module optional here
            controller = LeaderFollowerAggregator(
                config=config,
                num_policies=num_policies,
                leader_index=leader_index,
                lam=lam,
                subsample=subsample,
                sample_fraction=sample_fraction,
                clip_epsilon=clip_epsilon,
                **kwargs,
            )
            controller.name = SYMMETRIC
            controller.data_sources = lambda i: [  # type: ignore[assignment]
                j for j in range(1, controller.num_policies + 1) if j != i
            ]
            controller.is_leader = lambda i: False  # type: ignore[assignment]
            return controller
    if key in ("none", "no_aggregation", "no_off_policy", "on_policy"):
        controller = LeaderFollowerAggregator(
            config=config,
            num_policies=num_policies,
            leader_index=leader_index,
            lam=0.0,
            subsample=subsample,
            sample_fraction=sample_fraction,
            clip_epsilon=clip_epsilon,
            **kwargs,
        )
        controller.name = NO_AGGREGATION
        controller.data_sources = lambda i: []  # type: ignore[assignment]
        controller.is_leader = lambda i: False  # type: ignore[assignment]
        return controller
    raise ValueError(
        f"Unknown aggregation scheme {name!r}; expected one of "
        f"{LEADER_FOLLOWER!r}, {SYMMETRIC!r}, {NO_AGGREGATION!r}."
    )
