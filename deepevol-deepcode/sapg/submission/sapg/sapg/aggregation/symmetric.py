"""Symmetric aggregation for SAPG (Section 4.2, ablation in Section 6.3).

The paper's Section 4.2 reads:

    "A simple choice is to update all i's with the data from all policies. In this
     case, we choose to update each policy i in {1, 2, ..., M} and for each i use
     off-policy data from all other policies X = {1, 2, i-1, i+1, ..., M}. Since
     gradients from off-policy data are typically noisier than gradients from
     on-policy data, we choose lambda = 1, but subsample the off-policy data such
     that we use equal amounts of on-policy and off-policy data."

and Section 6.3 explains why this is used as an ablation:

    "Finally, the symmetric variant of our method performs significantly worse
     across the board. This is possibly because using all the data to update each
     policy leads to them converging in behavior. If all the policies start
     executing the same actions, the benefit of data diversity is lost and SAPG
     reduces to vanilla PPO."

This module therefore mirrors :mod:`sapg.aggregation.leader_follower` but with the
following differences:

* there is no privileged leader; **every** policy ``i`` receives off-policy data
  from ``X = {j : j != i}`` (1-based indexing, ``M`` policies);
* the off-policy budget for policy ``i`` is ``|D_i|`` (equal amounts of on-policy
  and off-policy data) when ``subsample=True``;
* ``subsample=False`` reproduces the "SAPG (high off-policy ratio)" ablation of
  Section 6.3, i.e. the gradient is computed on the entire combined
  off-policy + on-policy dataset.

Equation numbers follow the paper: the off-policy surrogate is Eq. (3)

    L_off(theta) = - sum_{j in X} E_{pi_j}[ min( r_pi_i(s,a) A^pi_i(s,a),
                    clip( r_pi_i(s,a), mu (1 - eps), mu (1 + eps) ) A^pi_i(s,a) ) ]

with ``mu = pi_{i,old}(s,a) / pi_j(s,a)`` and the total objective is Eq. (4)

    L = L_on(theta) + lambda * L_off(theta),     lambda = 1.

The module is deliberately simulation agnostic: it operates on any buffer-like
object (a dict, an object with ``.data``/``.storage``, or plain attributes) and on
any policy exposing ``evaluate_actions``/``evaluate`` plus ``phi(policy_index)``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..utils.config import NUM_POLICIES, SAPGConfig

__all__ = [
    "SymmetricAggregator",
    "SymmetricScheme",
    "make_symmetric_aggregator",
    "make_aggregation_symmetric",
    "SYMMETRIC",
]

SYMMETRIC = "symmetric"

# ---------------------------------------------------------------------------
# Key candidates -- kept intentionally permissive so the aggregator tolerates
# slightly different buffer/trainer naming conventions.
# ---------------------------------------------------------------------------
_OBS_KEYS = ("obs", "observations", "states", "state")
_ACTION_KEYS = ("actions", "action", "mus")
_BEHAVIOUR_LOGPROB_KEYS = ("logprobs", "log_prob", "logprob", "old_logprobs", "behaviour_logprobs", "sample_logprobs")
_TARGET_OLD_LOGPROB_KEYS = ("target_old_logprobs", "target_logprobs", "leader_old_logprobs", "off_target_logprobs")
_ADVANTAGE_KEYS = ("advantages", "adv", "advantage")
_MU_KEYS = ("mu", "importance_weight", "importance_weights")
_SOURCE_POLICY_KEYS = ("source_policy", "policy_index", "source_indices", "policy_ids")
_VERSION_KEYS = ("version", "source_version", "policy_version")
_PHI_KEYS = ("phi", "phis", "latent", "latents")
_HIDDEN_KEYS = ("hidden_states", "hidden_state", "rnn_hidden")
_MASK_KEYS = ("masks", "mask", "dones", "done")
_VALUE_KEYS = ("values", "value", "old_values")
_TARGET_KEYS = ("value_targets", "targets", "returns")
_TARGET_OFF_KEYS = ("value_targets_off", "value_targets_offpolicy", "off_policy_targets")

_LOG_RATIO_CLAMP = 20.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_device(device: Any = None) -> Optional[torch.device]:
    """Best-effort conversion of ``device`` to a ``torch.device``."""
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    try:
        return torch.device(device)
    except (TypeError, RuntimeError):
        return None


def _raw_data(buffer: Any) -> Dict[str, Any]:
    """Return the dict-like storage behind ``buffer``."""
    if buffer is None:
        return {}
    if isinstance(buffer, dict):
        return buffer
    for attr in ("data", "storage", "_data", "buffers", "tensors"):
        inner = getattr(buffer, attr, None)
        if isinstance(inner, dict):
            return inner
    # Fall back to the object's own ``__dict__`` of tensors.
    return {
        k: v
        for k, v in vars(buffer).items()
        if isinstance(v, torch.Tensor)
    }


def _first_key(data: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in data and data[key] is not None:
            return key
    return None


def _buffer_size(buffer: Any) -> int:
    data = _raw_data(buffer)
    for key in _OBS_KEYS + _ACTION_KEYS + _ADVANTAGE_KEYS:
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1, value.shape[-1]).shape[0]) if value.dim() > 1 else int(value.numel())
    for name in ("num_samples", "size", "total_samples", "horizon_length"):
        value = getattr(buffer, name, None)
        if isinstance(value, int):
            return value
    return 0


def _flatten_buffer(buffer: Any, keys: Optional[Sequence[str]] = None) -> Dict[str, torch.Tensor]:
    """Flatten a time-major buffer ``[T, N, ...]`` into ``[T*N, ...]`` tensors."""
    data = _raw_data(buffer)
    flat: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if not isinstance(value, torch.Tensor):
            continue
        if keys is not None and key not in keys:
            continue
        if value.dim() >= 2 and key in set(_OBS_KEYS + _ACTION_KEYS + _PHI_KEYS):
            flat[key] = value.reshape(-1, value.shape[-1])
        elif key in set(_HIDDEN_KEYS):
            # Hidden states need their own handling; keep them as-is.
            flat[key] = value
        else:
            flat[key] = value.reshape(-1)
    return flat


def _select(data: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Index every tensor of ``data`` along its first dimension."""
    out: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == indices.shape[0]:
            out[key] = value.index_select(0, indices)
        else:
            out[key] = value
    return out


def subsample_indices(
    total: int,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    device: Any = None,
) -> torch.Tensor:
    """Uniformly sample ``num_samples`` indices (without replacement) from ``range(total)``."""
    if total <= 0:
        return torch.zeros(0, dtype=torch.long, device=_as_device(device))
    num_samples = int(max(0, min(int(num_samples), int(total))))
    if num_samples == 0:
        return torch.zeros(0, dtype=torch.long, device=_as_device(device))
    if num_samples >= total:
        return torch.arange(total, device=_as_device(device))
    if generator is not None:
        perm = torch.randperm(total, generator=generator)
    else:
        perm = torch.randperm(total)
    return perm[:num_samples].to(_as_device(device) or perm.device)


def _resolve_phi(policy: Any, policy_index: Optional[int]) -> Optional[torch.Tensor]:
    """Fetch ``phi_i`` from a policy object (0-based ``policy_index``)."""
    if policy is None or policy_index is None:
        return None
    for name in ("phi_for", "phi"):
        fn = getattr(policy, name, None)
        if fn is None:
            continue
        try:
            return fn(policy_index)
        except TypeError:
            continue
    phis = getattr(policy, "phis", None)
    if isinstance(phis, torch.Tensor) and phis.dim() == 2:
        return phis[policy_index]
    return None


def _call_evaluate(policy: Any, batch: Dict[str, Any], policy_index: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Call the policy's evaluation entry point, tolerating signature variations."""
    if policy is None:
        return {}
    fn = None
    for name in ("evaluate_actions", "evaluate", "evaluate_batch"):
        fn = getattr(policy, name, None)
        if fn is not None:
            break
    if fn is None:
        return {}

    obs_key = _first_key(batch, _OBS_KEYS)
    action_key = _first_key(batch, _ACTION_KEYS)
    obs = batch.get(obs_key) if obs_key else None
    actions = batch.get(action_key) if action_key else None

    kwargs: Dict[str, Any] = {}
    for key, names in (
        ("obs", _OBS_KEYS),
        ("observations", _OBS_KEYS),
        ("actions", _ACTION_KEYS),
        ("phi", _PHI_KEYS),
        ("hidden_state", _HIDDEN_KEYS),
        ("hidden_states", _HIDDEN_KEYS),
        ("masks", _MASK_KEYS),
        ("policy_index", _SOURCE_POLICY_KEYS),
    ):
        src = _first_key(batch, names)
        if src is not None:
            kwargs[key] = batch[src]
    if policy_index is not None:
        kwargs["policy_index"] = policy_index
        kwargs["policy_idx"] = policy_index

    attempts = (
        ((), dict(obs=obs, actions=actions, **kwargs)),
        ((), kwargs),
        ((obs, actions), kwargs),
        ((obs, actions), {k: v for k, v in kwargs.items() if k in ("phi", "policy_index", "hidden_state", "masks")}),
    )
    last_error: Optional[Exception] = None
    for args, kw in attempts:
        try:
            out = fn(*args, **kw)
        except TypeError as exc:  # signature mismatch -> try the next form
            last_error = exc
            continue
        if out is None:
            return {}
        if isinstance(out, dict):
            return _canonicalise(dict(out))
        return _canonicalise(out)
    if last_error is not None:
        raise last_error
    return {}


def _canonicalise(out: Any) -> Dict[str, torch.Tensor]:
    """Normalise a policy output into a canonical dict."""
    result: Dict[str, torch.Tensor] = {}
    if isinstance(out, dict):
        for key, value in out.items():
            result[key] = value
    elif isinstance(out, (tuple, list)):
        for key, value in zip(("actions", "logprobs", "values", "entropy", "hidden_state"), out):
            result[key] = value
    else:
        result["values"] = out

    aliases = {
        "logprob": "logprobs",
        "log_prob": "logprobs",
        "old_logprobs": "logprobs",
        "value": "values",
        "action": "actions",
        "mus": "actions",
    }
    for src, dst in aliases.items():
        if src in result and dst not in result:
            result[dst] = result[src]
    if "values" in result and isinstance(result["values"], torch.Tensor) and result["values"].dim() > 1:
        result["values"] = result["values"].squeeze(-1)
    return result


def _default_on_policy_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    values: Optional[torch.Tensor],
    value_targets: Optional[torch.Tensor],
    clip_epsilon: float,
    critic_coefficient: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Clipped surrogate (Eq. 2) plus ``lambda'``-scaled value loss (Eq. 9)."""
    surrogate = torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages,
    )
    policy_loss = -surrogate.mean()
    stats: Dict[str, torch.Tensor] = {
        "surrogate": surrogate.mean().detach(),
        "clip_frac": ((ratio - 1.0).abs() > clip_epsilon).float().mean().detach(),
    }
    value_loss = torch.zeros((), device=ratio.device, dtype=ratio.dtype)
    if values is not None and value_targets is not None:
        value_loss = ((values - value_targets) ** 2).mean()
        stats["value_loss"] = value_loss.detach()
    total = policy_loss + critic_coefficient * value_loss
    stats["policy_loss"] = policy_loss.detach()
    return total, stats


def _local_off_policy_surrogate(
    new_logprobs: torch.Tensor,
    behaviour_logprobs: torch.Tensor,
    target_old_logprobs: Optional[torch.Tensor],
    advantages: torch.Tensor,
    clip_epsilon: float,
    source_policy: Optional[torch.Tensor] = None,
    reduction: str = "source",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Local fallback for Eq. (3): mu-corrected clipped importance ratio."""
    log_ratio = (new_logprobs - behaviour_logprobs).clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
    ratio = torch.exp(log_ratio)
    if target_old_logprobs is not None:
        mu_log = (target_old_logprobs - behaviour_logprobs).clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
        mu = torch.exp(mu_log)
    else:
        mu = torch.ones_like(ratio)
    lower = mu * (1.0 - clip_epsilon)
    upper = mu * (1.0 + clip_epsilon)
    surrogate = torch.min(ratio * advantages, torch.clamp(ratio, lower, upper) * advantages)
    if reduction == "source" and source_policy is not None:
        per_source = []
        for src in torch.unique(source_policy):
            mask = source_policy == src
            if mask.any():
                per_source.append(surrogate[mask].mean())
        loss = -torch.stack(per_source).mean() if per_source else -surrogate.mean()
    else:
        loss = -surrogate.mean()
    stats = {
        "surrogate": surrogate.mean().detach(),
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach() if ratio.numel() else torch.zeros((), device=ratio.device),
        "mu_mean": mu.mean().detach(),
        "clip_frac": ((ratio < lower) | (ratio > upper)).float().mean().detach(),
    }
    return loss, stats


# ---------------------------------------------------------------------------
# Scheme (lightweight description of the aggregation pattern)
# ---------------------------------------------------------------------------
class SymmetricScheme:
    """Section 4.2 aggregation pattern: every policy learns from all others."""

    name = SYMMETRIC

    def __init__(
        self,
        num_policies: int = NUM_POLICIES,
        off_policy_weight: float = 1.0,
        subsample: bool = True,
        leader_index: Optional[int] = None,
        **_: Any,
    ) -> None:
        self.num_policies = int(num_policies)
        self.off_policy_weight = float(off_policy_weight)
        self.subsample = bool(subsample)
        # Kept for API compatibility: the symmetric variant has no leader.
        self.leader_index = leader_index

    def data_sources(self, i: int) -> List[int]:
        """``X = {1, 2, i-1, i+1, ..., M}`` (1-based indices, excluding ``i``)."""
        return [j for j in range(1, self.num_policies + 1) if j != i]

    def is_leader(self, i: int) -> bool:
        """Every policy aggregates off-policy data in the symmetric variant."""
        return True

    def __len__(self) -> int:
        return self.num_policies

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"SymmetricScheme(num_policies={self.num_policies}, "
            f"lambda={self.off_policy_weight}, subsample={self.subsample})"
        )


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
class SymmetricAggregator:
    """Builds Eq. (3)/(4) losses for the symmetric ablation of Section 4.2.

    Parameters
    ----------
    config:
        Optional :class:`sapg.utils.config.SAPGConfig`; supplies ``num_policies``,
        ``off_policy_weight`` (lambda), ``subsample_off_policy`` and
        ``clip_epsilon`` when the explicit arguments are omitted.
    num_policies:
        Number of policies ``M`` (paper: 6).
    leader_index:
        Unused except for logging/API compatibility -- the symmetric variant has
        no privileged policy.
    lam:
        Off-policy weight ``lambda`` (paper: 1).
    subsample:
        If ``True`` (paper default) the off-policy data is uniformly subsampled so
        that ``|D'_i| = |D_i|``.  ``False`` gives the "high off-policy ratio"
        ablation of Section 6.3.
    on_policy_loss_fn / off_policy_loss_fn:
        Optional callables overriding the default losses (useful for tests).
    """

    name = SYMMETRIC

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
        normalize_advantages: bool = False,
        critic_coefficient: float = 4.0,
        keys: Optional[Sequence[str]] = None,
        device: Any = None,
        off_policy_loss_fn: Optional[Any] = None,
        on_policy_loss_fn: Optional[Any] = None,
        normalize_adv_leader_only: bool = True,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            num_policies = num_policies or getattr(config, "num_policies", NUM_POLICIES)
            lam = lam if lam is not None else getattr(config, "off_policy_weight", 1.0)
            if subsample is None:
                subsample = getattr(config, "subsample_off_policy", True)
            if clip_epsilon is None:
                clip_epsilon = getattr(config, "clip_epsilon", 0.1)
            if critic_coefficient is None:
                critic_coefficient = getattr(config, "critic_coefficient", 4.0)
            device = device if device is not None else getattr(config, "device", None)
            if not normalize_advantages:
                normalize_advantages = getattr(config, "normalize_advantage", False)

        self.config = config
        self.num_policies = int(num_policies or NUM_POLICIES)
        if self.num_policies < 2:
            raise ValueError("symmetric aggregation requires at least 2 policies")
        self.leader_index = leader_index
        self.lam = 1.0 if lam is None else float(lam)
        self.subsample = True if subsample is None else bool(subsample)
        self.sample_fraction = float(sample_fraction)
        self.per_source_subsample = bool(per_source_subsample)
        self.clip_epsilon = 0.1 if clip_epsilon is None else float(clip_epsilon)
        self.normalize_advantages = bool(normalize_advantages)
        self.critic_coefficient = float(critic_coefficient)
        self.keys = tuple(keys) if keys else None
        self.device = _as_device(device)
        self.off_policy_loss_fn = off_policy_loss_fn
        self.on_policy_loss_fn = on_policy_loss_fn
        self.normalize_adv_leader_only = bool(normalize_adv_leader_only)
        self.scheme = SymmetricScheme(
            num_policies=self.num_policies,
            off_policy_weight=self.lam,
            subsample=self.subsample,
        )
        self._off_policy_loss_module: Optional[Any] = None

    # -- pattern description -------------------------------------------------
    @property
    def followers(self) -> List[int]:
        """All policies play the follower role in the symmetric variant."""
        return list(range(1, self.num_policies + 1))

    @property
    def leader(self) -> Optional[int]:
        """The symmetric variant has no privileged policy."""
        return None

    def data_sources(self, i: int) -> List[int]:
        """Off-policy sources for policy ``i``: all policies ``j != i``."""
        return self.scheme.data_sources(i)

    def is_leader(self, i: int) -> bool:
        """``True`` for every policy: each one aggregates data from all others."""
        return True

    def off_policy_weight(self, i: Optional[int] = None) -> float:
        """``lambda = 1`` for every policy (Section 4.2)."""
        return self.lam

    def subsample_for(self, i: Optional[int] = None) -> bool:
        return self.subsample

    def off_policy_budget(
        self,
        on_policy_size: int,
        off_policy_size: int,
        num_sources: int = 1,
    ) -> int:
        """Number of off-policy samples to use for one target policy.

        Matches Section 4.2: equal amounts of on- and off-policy data
        (``|D'_i| = |D_i|``).  With ``subsample=False`` the entire combined
        dataset is used (Section 6.3, high off-policy ratio).
        """
        if on_policy_size <= 0:
            return 0
        if self.subsample:
            budget = int(math.ceil(on_policy_size * self.sample_fraction))
            return int(min(budget, off_policy_size)) if off_policy_size > 0 else 0
        return int(off_policy_size)

    # -- off-policy batch construction --------------------------------------
    def build_off_policy_batch(
        self,
        target_buffer: Any = None,
        source_buffers: Optional[Sequence[Any]] = None,
        policy: Any = None,
        target_index: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        subsample: Optional[bool] = None,
        num_samples: Optional[int] = None,
        buffer_i: Any = None,
        other_buffers: Optional[Sequence[Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Fuse the data of all policies ``j != i`` into the off-policy batch ``D'_i``.

        ``target_buffer``/``buffer_i`` is ``D_i`` (used only to size the subsample),
        ``source_buffers``/``other_buffers`` is ``[D_j for j != i]``.  When
        ``source_buffers`` is ``None`` and ``target_buffer`` is a full list of
        ``M`` buffers, the sources are selected with :meth:`data_sources`.
        """
        if target_buffer is None:
            target_buffer = buffer_i
        if source_buffers is None:
            source_buffers = other_buffers

        # Allow passing the full list of buffers plus an index.
        if isinstance(target_buffer, (list, tuple)) and all(
            not isinstance(x, torch.Tensor) for x in target_buffer
        ):
            buffers = list(target_buffer)
            if target_index is None:
                raise ValueError("target_index is required when passing all buffers")
            target_buffer = buffers[target_index - 1]
            source_buffers = [buffers[j - 1] for j in self.data_sources(target_index)]
        if source_buffers is None:
            source_buffers = []
        source_buffers = [b for b in source_buffers if b is not None]

        subsample = self.subsample if subsample is None else bool(subsample)
        pieces: List[Dict[str, torch.Tensor]] = []
        for offset, src_buffer in enumerate(source_buffers, start=1):
            flat = _flatten_buffer(src_buffer, self.keys)
            if not flat:
                continue
            size = next((v.shape[0] for v in flat.values() if isinstance(v, torch.Tensor) and v.dim() >= 1), 0)
            # Source-policy label (1-based).  Prefer an existing label in the buffer.
            src_key = _first_key(flat, _SOURCE_POLICY_KEYS)
            if src_key is not None:
                flat["source_policy"] = flat[src_key].reshape(-1).long()
            else:
                # Infer the buffer's own policy index when available.
                idx = getattr(src_buffer, "policy_index", None)
                if idx is None:
                    idx = getattr(src_buffer, "policy", None)
                idx = offset if idx is None else int(idx)
                flat["source_policy"] = torch.full((size,), int(idx), dtype=torch.long)
            pieces.append(flat)

        if not pieces:
            return {}

        device = self.device
        if device is None:
            for piece in pieces:
                for value in piece.values():
                    if isinstance(value, torch.Tensor):
                        device = value.device
                        break
                if device is not None:
                    break

        batch: Dict[str, torch.Tensor] = {}
        for key in pieces[0].keys():
            if all(key in piece for piece in pieces):
                batch[key] = torch.cat([piece[key] for piece in pieces], dim=0)

        total = next((v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor) and v.dim() >= 1), 0)

        # phi_i attached so the target policy can be evaluated on foreign states.
        if target_index is not None and policy is not None:
            phi = _resolve_phi(policy, target_index - 1)
            if phi is not None:
                phi = torch.as_tensor(phi)
                batch["phi"] = phi.reshape(1, -1).expand(total, -1).contiguous()

        # ``target_old_logprobs`` = pi_{i,old}(s,a): evaluate the target policy on
        # the off-policy states with its pre-update parameters (Eq. 3's mu).
        if target_index is not None and policy is not None:
            need_target_old = "target_old_logprobs" not in batch or _first_key(batch, _TARGET_OLD_LOGPROB_KEYS) is None
            if need_target_old:
                try:
                    with torch.no_grad():
                        out = _call_evaluate(policy, batch, policy_index=target_index - 1)
                    logprob_key = _first_key(out, _BEHAVIOUR_LOGPROB_KEYS)
                    if logprob_key is not None:
                        batch["target_old_logprobs"] = out[logprob_key].detach().reshape(-1)
                except Exception:  # pragma: no cover - evaluation is best effort
                    pass

        # Importance correction mu = exp(log pi_{i,old} - log pi_j) when possible.
        target_old = batch.get(_first_key(batch, _TARGET_OLD_LOGPROB_KEYS) or "target_old_logprobs")
        behaviour = batch.get(_first_key(batch, _BEHAVIOUR_LOGPROB_KEYS) or "logprobs")
        if target_old is not None and behaviour is not None:
            batch["mu"] = torch.exp(
                (target_old - behaviour).clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP)
            )

        budget = self.off_policy_budget(
            on_policy_size=_buffer_size(target_buffer) if target_buffer is not None else total,
            off_policy_size=total,
            num_sources=len(pieces),
        )
        if num_samples is not None:
            budget = int(num_samples)

        if subsample and 0 < budget < total:
            indices = subsample_indices(total, budget, generator=generator, device=device)
            if device is not None:
                indices = indices.to(device)
            batch = _select(batch, indices)
        return batch

    # -- losses --------------------------------------------------------------
    @property
    def off_policy_loss_module(self) -> Optional[Any]:
        """Lazily import :class:`sapg.losses.off_policy_loss.OffPolicyLoss`."""
        if self.off_policy_loss_fn is not None:
            return None
        if self._off_policy_loss_module is None:
            try:
                from ..losses.off_policy_loss import OffPolicyLoss

                self._off_policy_loss_module = OffPolicyLoss(
                    clip_epsilon=self.clip_epsilon,
                    lam=self.lam,
                    config=self.config,
                )
            except Exception:  # pragma: no cover - optional dependency
                self._off_policy_loss_module = False
        return self._off_policy_loss_module or None

    def on_policy_term(
        self,
        buffer: Any,
        policy: Any,
        policy_index: Optional[int] = None,
        normalize_advantages: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """On-policy PPO term (Eq. 2 + Eq. 9) for one policy's own dataset ``D_i``."""
        if buffer is None:
            empty = torch.zeros((), device=self.device)
            return empty, {}
        if isinstance(buffer, dict):
            batch = dict(buffer)
        else:
            batch = _flatten_buffer(buffer, self.keys)
        if not batch:
            return torch.zeros((), device=self.device), {}

        idx = (policy_index - 1) if policy_index else None
        out = _call_evaluate(policy, batch, policy_index=idx)

        logprob_key = _first_key(batch, _BEHAVIOUR_LOGPROB_KEYS)
        old_logprobs = batch.get(logprob_key) if logprob_key else None
        new_logprobs = out.get("logprobs")
        if new_logprobs is None or old_logprobs is None:
            return torch.zeros((), device=self.device), {}

        ratio = torch.exp((new_logprobs - old_logprobs).clamp(-_LOG_RATIO_CLAMP, _LOG_RATIO_CLAMP))
        adv_key = _first_key(batch, _ADVANTAGE_KEYS)
        advantages = batch.get(adv_key) if adv_key else torch.zeros_like(ratio)
        if advantages is None:
            advantages = torch.zeros_like(ratio)
        advantages = advantages.reshape(-1)
        if normalize_advantages is None:
            normalize_advantages = self.normalize_advantages
        if normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        values = out.get("values")
        tgt_key = _first_key(batch, _TARGET_KEYS)
        value_targets = batch.get(tgt_key) if tgt_key else None

        if self.on_policy_loss_fn is not None:
            result = self.on_policy_loss_fn(
                ratio=ratio,
                advantages=advantages,
                values=values,
                value_targets=value_targets,
            )
            if isinstance(result, tuple):
                return result
            return result, {}

        return _default_on_policy_loss(
            ratio=ratio,
            advantages=advantages,
            values=values,
            value_targets=value_targets,
            clip_epsilon=self.clip_epsilon,
            critic_coefficient=self.critic_coefficient,
        )

    def off_policy_term(
        self,
        off_policy_batch: Dict[str, torch.Tensor],
        policy: Any,
        policy_index: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Off-policy term ``L_off`` (Eq. 3) for policy ``i`` on ``X = {j != i}``."""
        if not off_policy_batch:
            return torch.zeros((), device=self.device), {}

        idx = (policy_index - 1) if policy_index else None
        out = _call_evaluate(policy, off_policy_batch, policy_index=idx)
        new_logprobs = out.get("logprobs")
        if new_logprobs is None:
            return torch.zeros((), device=self.device), {}

        behaviour_key = _first_key(off_policy_batch, _BEHAVIOUR_LOGPROB_KEYS)
        behaviour_logprobs = off_policy_batch.get(behaviour_key) if behaviour_key else None
        if behaviour_logprobs is None:
            return torch.zeros((), device=self.device), {}

        target_old_key = _first_key(off_policy_batch, _TARGET_OLD_LOGPROB_KEYS)
        target_old_logprobs = off_policy_batch.get(target_old_key) if target_old_key else None
        mu_key = _first_key(off_policy_batch, _MU_KEYS)
        mu = off_policy_batch.get(mu_key) if mu_key else None

        adv_key = _first_key(off_policy_batch, _ADVANTAGE_KEYS)
        advantages = off_policy_batch.get(adv_key)
        if advantages is None:
            return torch.zeros((), device=self.device), {}
        advantages = advantages.reshape(-1)

        src_key = _first_key(off_policy_batch, _SOURCE_POLICY_KEYS)
        source_policy = off_policy_batch.get(src_key) if src_key else None
        if source_policy is not None:
            source_policy = source_policy.reshape(-1).long()

        module = self.off_policy_loss_module
        if module is not None:
            try:
                result = module(
                    new_logprobs=new_logprobs,
                    behaviour_logprobs=behaviour_logprobs,
                    target_old_logprobs=target_old_logprobs,
                    advantages=advantages,
                    mu=mu,
                    source_policy=source_policy,
                )
                if isinstance(result, dict):
                    loss = result.get("off_policy_loss", result.get("policy_loss"))
                    stats = {k: v for k, v in result.items() if isinstance(v, torch.Tensor)}
                    if loss is not None:
                        return loss, stats
            except TypeError:
                pass
            except Exception:  # pragma: no cover - fall through to local loss
                pass

        return _local_off_policy_surrogate(
            new_logprobs=new_logprobs,
            behaviour_logprobs=behaviour_logprobs,
            target_old_logprobs=target_old_logprobs,
            advantages=advantages,
            clip_epsilon=self.clip_epsilon,
            source_policy=source_policy,
            reduction="source",
        )

    def aggregate(
        self,
        buffers: Sequence[Any],
        policy: Any,
        policy_index: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        subsample: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Aggregate terms for one or all policies.

        ``policy_index`` (1-based) restricts the computation to a single policy;
        when omitted the total objective over all policies is returned, matching
        Algorithm 1's single summed-loss backward pass.
        """
        buffers = list(buffers)
        if not buffers:
            return {"loss": torch.zeros((), device=self.device), "terms": {}}

        targets = [policy_index] if policy_index is not None else list(range(1, len(buffers) + 1))
        total = torch.zeros((), device=self.device)
        terms: Dict[int, Dict[str, torch.Tensor]] = {}
        for i in targets:
            on_loss, on_stats = self.on_policy_term(buffers[i - 1], policy, policy_index=i)
            sources = self.data_sources(i)
            off_loss = torch.zeros((), device=self.device)
            off_stats: Dict[str, torch.Tensor] = {}
            if sources:
                off_batch = self.build_off_policy_batch(
                    buffers,
                    target_index=i,
                    policy=policy,
                    generator=generator,
                    subsample=subsample,
                )
                off_loss, off_stats = self.off_policy_term(off_batch, policy, policy_index=i)
            lam = self.off_policy_weight(i)
            combined = on_loss + lam * off_loss
            total = total + combined
            terms[i] = {
                "on_policy_loss": on_loss,
                "off_policy_loss": off_loss,
                "combined": combined,
                **{f"on/{k}": v for k, v in on_stats.items()},
                **{f"off/{k}": v for k, v in off_stats.items()},
            }
        return {
            "loss": total,
            "mean_loss": total / max(len(targets), 1),
            "terms": terms,
            "num_policies": len(targets),
            "lam": self.lam,
            "subsample": self.subsample if subsample is None else bool(subsample),
        }

    def stats(self, aggregation_result: Dict[str, Any]) -> Dict[str, float]:
        """Flatten an :meth:`aggregate` result into scalar logging values."""
        out: Dict[str, float] = {}
        for i, term in (aggregation_result.get("terms") or {}).items():
            for key, value in term.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    out[f"agg/policy_{i}/{key}"] = float(value.detach().cpu())
        loss = aggregation_result.get("mean_loss")
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            out["agg/loss"] = float(loss.detach().cpu())
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"SymmetricAggregator(num_policies={self.num_policies}, lam={self.lam}, "
            f"subsample={self.subsample}, clip_epsilon={self.clip_epsilon})"
        )


def make_symmetric_aggregator(
    config: Optional[SAPGConfig] = None,
    **kwargs: Any,
) -> SymmetricAggregator:
    """Factory mirroring ``leader_follower.make_aggregation`` for the symmetric case."""
    return SymmetricAggregator(config=config, **kwargs)


# Alias used by ``sapg.aggregation.__init__`` / ``make_aggregation`` dispatch.
make_aggregation_symmetric = make_symmetric_aggregator
