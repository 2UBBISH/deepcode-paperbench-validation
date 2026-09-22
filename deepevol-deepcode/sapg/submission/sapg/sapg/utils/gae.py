"""Generalized Advantage Estimation (GAE) utilities for SAPG.

The SAPG paper (Sec. 4.1) updates the critic with *three-step* on-policy returns
(Eq. 5) and *one-step* off-policy returns (Eq. 6), while the policy gradient uses
advantage estimates.  The paper never spells out the advantage estimator or the
meaning of the hyperparameter ``tau = 0.95`` listed in Appendix B.1-B.3; the
reproduction plan resolves this ambiguity by interpreting ``tau`` as the GAE
lambda parameter of Schulman et al. (2016)::

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * tau * (1 - done_t) * A_{t+1}
    V_targ  = A_t + V(s_t)

This module is a thin, dependency-light layer on top of
``sapg.utils.returns`` / ``sapg.losses.critic_loss``: the canonical tensor math
lives there (shared with the critic-target tests), while this file exposes the
GAE-specific entry points used by the trainers (``compute_gae``,
``compute_gae_for_buffer``, ``GeneralizedAdvantageEstimator``) together with a
self-contained fallback in case the canonical implementation is unavailable.

All tensors are time-major ``[horizon, num_envs]`` (values may carry a trailing
singleton ``[horizon, num_envs, 1]``), matching ``RolloutBuffer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is expected but the module stays importable
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


# Canonical implementations (preferred).  Imported lazily / defensively so this
# module can be imported before ``sapg.losses`` is fully wired.
try:  # pragma: no cover - depends on import order
    from .returns import (
        compute_gae as _canonical_gae,
        compute_returns as _canonical_returns,
        normalize_advantages as _canonical_normalize,
    )

    _HAS_RETURNS = True
except Exception:  # pragma: no cover
    _canonical_gae = None  # type: ignore[assignment]
    _canonical_returns = None  # type: ignore[assignment]
    _canonical_normalize = None  # type: ignore[assignment]
    _HAS_RETURNS = False


__all__ = [
    "DEFAULT_GAMMA",
    "DEFAULT_TAU",
    "DEFAULT_GAE_LAMBDA",
    "compute_gae",
    "generalized_advantage_estimate",
    "compute_gae_for_buffer",
    "compute_advantages",
    "compute_returns",
    "normalize_advantages",
    "GeneralizedAdvantageEstimator",
    "GAEStats",
]

DEFAULT_GAMMA: float = 0.99
#: The paper's ``tau`` (Appendix B.1-B.3) is interpreted as the GAE lambda.
DEFAULT_TAU: float = 0.95
DEFAULT_GAE_LAMBDA: float = DEFAULT_TAU


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_REWARD_KEYS = ("rewards", "reward", "rew", "r")
_DONE_KEYS = ("dones", "done", "terminals", "terminal", "masks", "mask")
_VALUE_KEYS = ("values", "value", "v_old", "old_values")
_LAST_VALUE_KEYS = ("last_values", "last_value", "bootstrap_values", "next_values")


def _get(source: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Duck-typed key lookup across dicts, buffers and batch objects."""
    if source is None:
        return default
    for key in keys:
        if isinstance(source, dict):
            if key in source:
                return source[key]
            continue
        getter = getattr(source, "get", None)
        if callable(getter):
            try:
                value = getter(key)
            except Exception:  # pragma: no cover - defensive
                value = None
            if value is not None:
                return value
        if hasattr(source, key):
            return getattr(source, key)
    # nested containers (RolloutBuffer.data / OffPolicyBatch.data)
    for container_name in ("data", "storage", "tensors", "fields"):
        inner = getattr(source, container_name, None)
        if inner is not None and inner is not source:
            found = _get(inner, keys, default=None)
            if found is not None:
                return found
    return default


def _squeeze_last(x: Any) -> Any:
    """Remove a trailing singleton dimension produced by value heads."""
    if HAS_TORCH and torch.is_tensor(x) and x.dim() >= 3 and x.shape[-1] == 1:
        return x.squeeze(-1)
    return x


def _zeros_like(x: Any) -> Any:
    if HAS_TORCH and torch.is_tensor(x):
        return torch.zeros_like(x)
    return 0.0


def _require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError(
            "sapg.utils.gae requires PyTorch, which is not importable in this "
            "environment."
        )


# --------------------------------------------------------------------------- #
# core GAE
# --------------------------------------------------------------------------- #
def _fallback_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    last_values: Any,
    gamma: float,
    tau: float,
):
    """Self-contained GAE (used when ``sapg.utils.returns`` is unavailable)."""
    _require_torch()
    rewards = _squeeze_last(rewards)
    values = _squeeze_last(values)
    if dones is None:
        dones = torch.zeros_like(rewards)
    else:
        dones = _squeeze_last(dones).to(rewards.dtype)

    horizon = rewards.shape[0]
    if last_values is None:
        last_values = torch.zeros(
            rewards.shape[1], dtype=rewards.dtype, device=rewards.device
        )
    last_values = _squeeze_last(last_values)
    if last_values.dim() == 0:
        last_values = last_values.expand(rewards.shape[1])
    if last_values.shape[0] != rewards.shape[1]:
        last_values = values[-1].detach()

    advantages = torch.zeros_like(rewards)
    next_advantage = _zeros_like(rewards[0])
    next_value = last_values.detach()

    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_advantage = delta + gamma * tau * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t].detach()

    value_targets = advantages + values
    return advantages, value_targets


def compute_gae(
    rewards: Any = None,
    values: Any = None,
    dones: Any = None,
    last_values: Any = None,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
    *,
    lam: Optional[float] = None,
    masks: Any = None,
    buffer: Any = None,
    normalize: bool = False,
    device: Any = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Compute GAE advantages and value targets.

    Returns a tuple ``(advantages, value_targets)`` with
    ``value_targets = advantages + values`` so the critic can regress directly
    onto the returned targets (standard PPO value loss).

    ``tau`` is the paper's hyperparameter (0.95) and is interpreted as the GAE
    lambda; the explicit ``lam`` keyword takes precedence when supplied.  Inputs
    may be raw tensors or any buffer/batch object exposing ``rewards``,
    ``dones``, ``values`` and ``last_values``.
    """
    if buffer is not None:
        if rewards is None:
            rewards = _get(buffer, _REWARD_KEYS)
        if dones is None:
            dones = _get(buffer, _DONE_KEYS)
        if values is None:
            values = _get(buffer, _VALUE_KEYS)
        if last_values is None:
            last_values = _get(buffer, _LAST_VALUE_KEYS)

    if lam is None:
        lam = kwargs.pop("gae_lambda", None)
    if lam is not None:
        tau = lam
    tau = kwargs.pop("lambda_", tau)

    if rewards is None or values is None:
        raise ValueError(
            "compute_gae requires `rewards` and `values` (or a `buffer` "
            "exposing them)."
        )

    if _HAS_RETURNS and _canonical_gae is not None:
        try:
            result = _canonical_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                last_values=last_values,
                gamma=gamma,
                tau=tau,
                normalize=normalize,
                device=device,
            )
            if isinstance(result, tuple) and len(result) == 2:
                return result
        except TypeError:
            pass
        except Exception:  # pragma: no cover - fall through to local impl
            pass

    advantages, value_targets = _fallback_gae(
        rewards, values, dones, last_values, gamma, tau
    )
    if normalize:
        advantages = normalize_advantages(advantages)
    return advantages, value_targets


# commonly used aliases
generalized_advantage_estimate = compute_gae
compute_advantages = compute_gae


def compute_gae_for_buffer(
    buffer: Any,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_TAU,
    *,
    normalize: bool = False,
    store: bool = True,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """GAE for a :class:`RolloutBuffer`-like object.

    When ``store`` is true (and the buffer supports it) the computed
    ``advantages`` / ``value_targets`` are written back onto the buffer, then
    the advantages are normalized in-place if requested.
    """
    advantages, value_targets = compute_gae(
        buffer=buffer, gamma=gamma, tau=tau, normalize=False, **kwargs
    )
    if normalize:
        advantages = normalize_advantages(advantages)
    if store:
        _store(buffer, "advantages", advantages)
        _store(buffer, "value_targets", value_targets)
        _store(buffer, "returns", value_targets)
        if normalize:
            _store(buffer, "advantages_normalized", advantages)
    return advantages, value_targets


def _store(buffer: Any, name: str, value: Any) -> None:
    if buffer is None or value is None:
        return
    try:
        setattr(buffer, name, value)
    except Exception:  # pragma: no cover - some buffers are frozen
        pass


def compute_returns(
    rewards: Any,
    dones: Any = None,
    gamma: float = DEFAULT_GAMMA,
    *,
    last_values: Any = None,
    masks: Any = None,
    device: Any = None,
    **kwargs: Any,
) -> Any:
    """Discounted Monte-Carlo returns-to-go (delegates to ``utils.returns``)."""
    if _HAS_RETURNS and _canonical_returns is not None:
        try:
            return _canonical_returns(
                rewards,
                dones=dones,
                gamma=gamma,
                last_values=last_values,
                masks=masks,
                device=device,
            )
        except Exception:  # pragma: no cover - fall through
            pass

    _require_torch()
    rewards = _squeeze_last(rewards)
    if dones is None:
        dones = torch.zeros_like(rewards)
    else:
        dones = _squeeze_last(dones).to(rewards.dtype)
    if last_values is None:
        last_values = torch.zeros(
            rewards.shape[1], dtype=rewards.dtype, device=rewards.device
        )
    last_values = _squeeze_last(last_values)
    if last_values.shape[0] != rewards.shape[1]:
        last_values = torch.zeros(
            rewards.shape[1], dtype=rewards.dtype, device=rewards.device
        )

    out = torch.zeros_like(rewards)
    running = last_values
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        out[t] = running
    return out


def normalize_advantages(advantages: Any, epsilon: float = 1e-8, *, dim: Any = None) -> Any:
    """Zero-mean / unit-std advantage normalization (PPO convention)."""
    if _HAS_RETURNS and _canonical_normalize is not None:
        try:
            return _canonical_normalize(advantages, epsilon=epsilon, dim=dim)
        except TypeError:
            try:
                return _canonical_normalize(advantages, epsilon=epsilon)
            except Exception:  # pragma: no cover
                pass
        except Exception:  # pragma: no cover
            pass

    _require_torch()
    if not torch.is_tensor(advantages):
        return advantages
    if advantages.numel() <= 1:
        return advantages
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    return (advantages - mean) / (std + epsilon)


# --------------------------------------------------------------------------- #
# object-oriented wrapper
# --------------------------------------------------------------------------- #
@dataclass
class GAEStats:
    """Diagnostics for one GAE computation."""

    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    advantage_abs_mean: float = 0.0
    value_target_mean: float = 0.0
    value_target_std: float = 0.0
    num_samples: int = 0

    def as_dict(self, prefix: str = "gae/") -> Dict[str, float]:
        return {
            f"{prefix}advantage_mean": self.advantage_mean,
            f"{prefix}advantage_std": self.advantage_std,
            f"{prefix}advantage_abs_mean": self.advantage_abs_mean,
            f"{prefix}value_target_mean": self.value_target_mean,
            f"{prefix}value_target_std": self.value_target_std,
            f"{prefix}num_samples": float(self.num_samples),
        }


class GeneralizedAdvantageEstimator:
    """Callable GAE helper configured with the paper's ``gamma`` / ``tau``."""

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        tau: float = DEFAULT_TAU,
        normalize_advantages: bool = False,
        config: Any = None,
    ) -> None:
        if config is not None:
            gamma = float(getattr(config, "gamma", gamma))
            tau = float(getattr(config, "tau", tau))
            normalize_advantages = bool(
                getattr(config, "normalize_advantage", normalize_advantages)
            )
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.normalize_advantages = bool(normalize_advantages)
        self.last_stats: Optional[GAEStats] = None
        self._count = 0

    # -- API used by the trainers ------------------------------------------ #
    def __call__(self, buffer: Any = None, **kwargs: Any) -> Tuple[Any, Any]:
        return self.estimate(buffer, **kwargs)

    def estimate(
        self,
        buffer: Any = None,
        *,
        rewards: Any = None,
        values: Any = None,
        dones: Any = None,
        last_values: Any = None,
        gamma: Optional[float] = None,
        tau: Optional[float] = None,
        normalize: Optional[bool] = None,
        store: bool = False,
        device: Any = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        gamma = self.gamma if gamma is None else float(gamma)
        tau = self.tau if tau is None else float(tau)
        normalize = self.normalize_advantages if normalize is None else bool(normalize)

        advantages, value_targets = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_values=last_values,
            gamma=gamma,
            tau=tau,
            buffer=buffer,
            normalize=normalize,
            device=device,
            **kwargs,
        )
        if store and buffer is not None:
            _store(buffer, "advantages", advantages)
            _store(buffer, "value_targets", value_targets)
        self.last_stats = self._stats(advantages, value_targets)
        self._count += 1
        return advantages, value_targets

    # convenience aliases
    def compute(self, *args: Any, **kwargs: Any) -> Tuple[Any, Any]:
        return self.estimate(*args, **kwargs)

    def for_buffer(self, buffer: Any, **kwargs: Any) -> Tuple[Any, Any]:
        return self.estimate(buffer, store=kwargs.pop("store", True), **kwargs)

    # -- diagnostics -------------------------------------------------------- #
    @property
    def stats(self) -> Dict[str, float]:
        if self.last_stats is None:
            return GAEStats().as_dict()
        return self.last_stats.as_dict()

    @property
    def num_calls(self) -> int:
        return self._count

    def state_dict(self) -> Dict[str, Any]:
        return {
            "gamma": self.gamma,
            "tau": self.tau,
            "normalize_advantages": self.normalize_advantages,
            "num_calls": self._count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gamma = float(state.get("gamma", self.gamma))
        self.tau = float(state.get("tau", self.tau))
        self.normalize_advantages = bool(
            state.get("normalize_advantages", self.normalize_advantages)
        )
        self._count = int(state.get("num_calls", self._count))

    def _stats(self, advantages: Any, value_targets: Any) -> GAEStats:
        if not (HAS_TORCH and torch.is_tensor(advantages)):
            return GAEStats()
        adv = advantages.detach().float()
        vt = value_targets.detach().float()
        return GAEStats(
            advantage_mean=float(adv.mean()),
            advantage_std=float(adv.std(unbiased=False)) if adv.numel() > 1 else 0.0,
            advantage_abs_mean=float(adv.abs().mean()),
            value_target_mean=float(vt.mean()),
            value_target_std=float(vt.std(unbiased=False)) if vt.numel() > 1 else 0.0,
            num_samples=int(adv.numel()),
        )

    def __repr__(self) -> str:
        return (
            f"GeneralizedAdvantageEstimator(gamma={self.gamma}, tau={self.tau}, "
            f"normalize={self.normalize_advantages})"
        )


def make_gae(config: Any = None, **kwargs: Any) -> GeneralizedAdvantageEstimator:
    """Factory mirroring the other ``make_*`` helpers in the codebase."""
    return GeneralizedAdvantageEstimator(config=config, **kwargs)
