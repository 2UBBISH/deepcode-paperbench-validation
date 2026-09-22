"""Return / advantage estimation utilities for SAPG (Section 4.1).

The SAPG paper defines two distinct critic targets in Section 4.1:

* the **on-policy** target uses *n*-step returns (the paper uses ``n = 3``)

  .. math::
      V^{\\text{target}}_{on,\\pi_j}(s_t) = \\sum_{k=t}^{t+2} \\gamma^{k-t} r_k
                                            + \\gamma^3 V_{\\pi_j,\\text{old}}(s_{t+3})

* the **off-policy** target, since *n*-step returns are not available across
  policies, approximates a 1-step return

  .. math::
      V^{\\text{target}}_{off,\\pi_j}(s'_t) = r_t + \\gamma V_{\\pi_j,\\text{old}}(s'_{t+1})

This module provides those targets (delegating the canonical implementation to
:mod:`sapg.losses.critic_loss`, with a self-contained fallback so the utilities
remain usable on their own), plus the usual advantage estimation helpers:

* :func:`compute_gae` -- generalized advantage estimation, where the paper's
  ``tau`` (0.95, listed in Appendix B.1-B.3 alongside ``gamma = 0.99``) is
  interpreted as the trace decay parameter ``lambda`` (the paper never defines
  ``tau`` explicitly).
* :func:`compute_returns` -- discounted Monte-Carlo returns-to-go.
* :func:`normalize_advantages` -- zero-mean / unit-std advantage normalisation.

All tensors are expected **time-major** with shape ``[horizon, num_envs]`` (or
``[horizon, num_envs, 1]`` for values), matching
:class:`sapg.buffers.rollout_buffer.RolloutBuffer`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is always present in the training environment
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # canonical targets live with the critic losses
    from ..losses.critic_loss import (  # type: ignore
        compute_n_step_targets as _canonical_n_step_targets,
        compute_one_step_targets as _canonical_one_step_targets,
    )

    _HAS_CRITIC_LOSS = True
except Exception:  # pragma: no cover - defensive fallback
    _canonical_n_step_targets = None  # type: ignore[assignment]
    _canonical_one_step_targets = None  # type: ignore[assignment]
    _HAS_CRITIC_LOSS = False

__all__ = [
    # targets
    "compute_n_step_targets",
    "compute_one_step_targets",
    "compute_on_policy_targets",
    "compute_off_policy_targets",
    "three_step_targets",
    "one_step_targets",
    "compute_value_targets",
    # returns / advantages
    "compute_returns",
    "compute_discounted_returns",
    "compute_gae",
    "compute_advantage",
    "normalize_advantages",
    "compute_advantages_and_returns",
    "check_finite",
    # constants
    "DEFAULT_GAMMA",
    "DEFAULT_LAMBDA",
    "DEFAULT_N_STEP",
]

DEFAULT_GAMMA: float = 0.99
DEFAULT_LAMBDA: float = 0.95  # paper's ``tau`` interpreted as GAE lambda
DEFAULT_N_STEP: int = 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_tensor(value: Any, device: Optional[Any] = None, dtype: Any = None) -> Any:
    """Convert ``value`` to a float tensor on ``device`` (duck typing friendly)."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for return computation")
    if isinstance(value, torch.Tensor):
        out = value
    elif isinstance(value, (list, tuple)):
        out = torch.as_tensor(list(value))
    else:
        out = torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype)
    else:
        out = out.float()
    if device is not None:
        out = out.to(device)
    return out


def _squeeze_last(tensor: Any) -> Any:
    """Drop a trailing singleton dimension (values are often ``[T, B, 1]``)."""

    if torch is not None and isinstance(tensor, torch.Tensor) and tensor.dim() > 2:
        if tensor.shape[-1] == 1:
            return tensor.squeeze(-1)
    return tensor


def _get(source: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Fetch the first available key/attribute from ``source``."""

    if source is None:
        return default
    for key in keys:
        if isinstance(source, dict) and key in source:
            return source[key]
        getter = getattr(source, "get", None)
        if callable(getter):
            try:
                value = getter(key, None)
            except Exception:
                value = None
            if value is not None:
                return value
        if hasattr(source, key):
            return getattr(source, key)
        container = getattr(source, "data", None)
        if isinstance(container, dict) and key in container:
            return container[key]
    return default


def _zeros_like(tensor: Any) -> Any:
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required")
    return torch.zeros_like(tensor)


def _cat_shifted(values: Any, last_values: Any = None) -> Any:
    """Append a bootstrap value to ``values`` -> ``[T+1, num_envs]``."""

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required")
    num_envs = values.shape[1] if values.dim() > 1 else values.shape[0]
    if last_values is None:
        last = _zeros_like(values[-1]).reshape(1, num_envs).to(values.dtype)
    else:
        last = _as_tensor(last_values, values.device)
        last = last.reshape(-1)
        if last.numel() == 1:
            last = last.expand(num_envs)
        last = last[:num_envs].reshape(1, -1).to(values.dtype)
    return torch.cat([values, last], dim=0)


# ---------------------------------------------------------------------------
# critic targets (Section 4.1)
# ---------------------------------------------------------------------------
def compute_n_step_targets(
    rewards: Any = None,
    dones: Any = None,
    values: Any = None,
    last_values: Any = None,
    gamma: float = DEFAULT_GAMMA,
    n_step: int = DEFAULT_N_STEP,
    *,
    masks: Any = None,
    buffer: Any = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """*n*-step on-policy critic target (Section 4.1, ``n = 3``).

    ``V_on(s_t) = sum_{k=t}^{t+n-1} gamma^{k-t} r_k + gamma^n V_old(s_{t+n})``

    The sum is truncated whenever an episode boundary is crossed and the
    bootstrap value falls back to ``last_values`` when the window runs past the
    end of the horizon.
    """

    if _HAS_CRITIC_LOSS:
        try:
            return _canonical_n_step_targets(
                rewards=rewards,
                dones=dones if dones is not None else masks,
                values=values,
                last_values=last_values,
                gamma=gamma,
                n_step=n_step,
                buffer=buffer,
                device=device,
                **kwargs,
            )
        except TypeError:  # pragma: no cover - older signature
            pass
        except Exception:  # pragma: no cover - fall through to local version
            pass

    if buffer is not None:
        rewards = _get(buffer, ("rewards", "reward", "rew"), rewards)
        dones = _get(buffer, ("dones", "done", "terminals", "masks"), dones)
        values = _get(buffer, ("values", "old_values", "v_old"), values)
        last_values = _get(
            buffer, ("last_values", "next_values", "bootstrap_values"), last_values
        )

    rewards = _squeeze_last(_as_tensor(rewards, device))
    dones = _squeeze_last(_as_tensor(dones if dones is not None else masks, device))
    values = _squeeze_last(_as_tensor(values, device))
    horizon = rewards.shape[0]

    values_ext = _cat_shifted(values, last_values)
    targets = torch.zeros_like(rewards)
    for t in range(horizon):
        acc = torch.zeros_like(rewards[t])
        alive = torch.ones_like(rewards[t])
        for k in range(n_step):
            if t + k >= horizon:
                break
            acc = acc + (gamma**k) * rewards[t + k] * alive
            alive = alive * (1.0 - dones[t + k])
        acc = acc + (gamma**n_step) * values_ext[t + n_step] * alive
        targets[t] = acc
    return targets


def compute_one_step_targets(
    rewards: Any = None,
    dones: Any = None,
    next_values: Any = None,
    gamma: float = DEFAULT_GAMMA,
    *,
    values: Any = None,
    masks: Any = None,
    batch: Any = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """1-step off-policy critic target (Section 4.1).

    ``V_off(s'_t) = r_t + gamma * V_old(s'_{t+1})``
    (the bootstrap is masked on terminal transitions).
    """

    if _HAS_CRITIC_LOSS:
        try:
            return _canonical_one_step_targets(
                rewards=rewards,
                dones=dones if dones is not None else masks,
                next_values=next_values if next_values is not None else values,
                gamma=gamma,
                batch=batch,
                device=device,
                **kwargs,
            )
        except TypeError:  # pragma: no cover
            pass
        except Exception:  # pragma: no cover
            pass

    if batch is not None:
        rewards = _get(batch, ("rewards", "reward", "rew"), rewards)
        dones = _get(batch, ("dones", "done", "terminals", "masks"), dones)
        next_values = _get(
            batch,
            ("next_values", "value_targets_next", "bootstrap_values", "values"),
            next_values,
        )

    rewards = _squeeze_last(_as_tensor(rewards, device))
    dones = _squeeze_last(_as_tensor(dones if dones is not None else masks, device))
    next_values = _squeeze_last(_as_tensor(next_values, device))
    if next_values.shape != rewards.shape:
        next_values = next_values.reshape(rewards.shape)
    return rewards + gamma * next_values * (1.0 - dones)


# aliases used by the paper notation / trainers
compute_on_policy_targets = compute_n_step_targets
compute_off_policy_targets = compute_one_step_targets
three_step_targets = compute_n_step_targets
one_step_targets = compute_one_step_targets


def compute_value_targets(
    buffer: Any = None,
    gamma: float = DEFAULT_GAMMA,
    n_step: int = DEFAULT_N_STEP,
    *,
    on_policy: bool = True,
    rewards: Any = None,
    dones: Any = None,
    values: Any = None,
    last_values: Any = None,
    next_values: Any = None,
    off_policy_batch: Any = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Compute on-policy (n-step) or off-policy (1-step) critic targets.

    ``on_policy=True`` selects Eq. (3) of Section 4.1, ``on_policy=False``
    selects the 1-step off-policy target Eq. (4).
    """

    if on_policy:
        return compute_n_step_targets(
            rewards=rewards,
            dones=dones,
            values=values,
            last_values=last_values,
            gamma=gamma,
            n_step=n_step,
            buffer=buffer,
            device=device,
            **kwargs,
        )
    source = off_policy_batch if off_policy_batch is not None else buffer
    return compute_one_step_targets(
        rewards=rewards,
        dones=dones,
        next_values=next_values if next_values is not None else values,
        gamma=gamma,
        batch=source,
        device=device,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# returns and advantages
# ---------------------------------------------------------------------------
def compute_returns(
    rewards: Any,
    dones: Any = None,
    gamma: float = DEFAULT_GAMMA,
    *,
    last_values: Any = None,
    masks: Any = None,
    device: Optional[Any] = None,
) -> Any:
    """Discounted Monte-Carlo returns-to-go (backwards recursion)."""

    rewards = _squeeze_last(_as_tensor(rewards, device))
    if dones is not None or masks is not None:
        dones = _squeeze_last(_as_tensor(dones if dones is not None else masks, device))
    else:
        dones = torch.zeros_like(rewards)
    horizon = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    if last_values is not None:
        running = _as_tensor(last_values, device).reshape(-1)
        if running.numel() != rewards.shape[1]:
            running = torch.zeros(
                rewards.shape[1], device=rewards.device, dtype=rewards.dtype
            )
    else:
        running = torch.zeros(
            rewards.shape[1], device=rewards.device, dtype=rewards.dtype
        )
    for t in reversed(range(horizon)):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running
    return returns


# alias
compute_discounted_returns = compute_returns


def compute_gae(
    rewards: Any = None,
    dones: Any = None,
    values: Any = None,
    last_values: Any = None,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_LAMBDA,
    *,
    lam: Optional[float] = None,
    masks: Any = None,
    buffer: Any = None,
    normalize: bool = False,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple[Any, Any]:
    """Generalized advantage estimation, returns ``(advantages, value_targets)``.

    The paper lists ``tau = 0.95`` in Appendix B.1-B.3 without naming it;
    following the reproduction plan it is interpreted as the GAE trace-decay
    ``lambda``.  ``value_targets = advantages + values`` (the TD(lambda)
    returns used for the critic regression).
    """

    if lam is not None:
        tau = lam

    if buffer is not None:
        rewards = _get(buffer, ("rewards", "reward", "rew"), rewards)
        dones = _get(buffer, ("dones", "done", "terminals", "masks"), dones)
        values = _get(buffer, ("values", "old_values", "v_old"), values)
        last_values = _get(
            buffer, ("last_values", "next_values", "bootstrap_values"), last_values
        )

    rewards = _squeeze_last(_as_tensor(rewards, device))
    if dones is not None or masks is not None:
        dones = _squeeze_last(_as_tensor(dones if dones is not None else masks, device))
    else:
        dones = torch.zeros_like(rewards)
    values = _squeeze_last(_as_tensor(values, device))
    horizon = rewards.shape[0]

    if last_values is None:
        last_values = values[-1]
    next_value = _squeeze_last(_as_tensor(last_values, device)).reshape(-1)
    if next_value.numel() != rewards.shape[1]:
        next_value = values[-1]

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(horizon)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        gae = delta + gamma * tau * not_done * gae
        advantages[t] = gae
        next_value = values[t]
    value_targets = advantages + values
    if normalize:
        advantages = normalize_advantages(advantages)
    return advantages, value_targets


# alias
compute_advantage = compute_gae


def compute_advantages_and_returns(
    buffer: Any = None,
    gamma: float = DEFAULT_GAMMA,
    tau: float = DEFAULT_LAMBDA,
    *,
    normalize: bool = True,
    rewards: Any = None,
    dones: Any = None,
    values: Any = None,
    last_values: Any = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper returning a dict of advantages and TD(lambda) targets."""

    advantages, value_targets = compute_gae(
        rewards=rewards,
        dones=dones,
        values=values,
        last_values=last_values,
        gamma=gamma,
        tau=tau,
        buffer=buffer,
        device=device,
        **kwargs,
    )
    out: Dict[str, Any] = {"advantages": advantages, "value_targets": value_targets}
    if normalize:
        out["advantages_normalized"] = normalize_advantages(advantages)
    return out


def normalize_advantages(
    advantages: Any,
    epsilon: float = 1e-8,
    *,
    dim: Optional[Any] = None,
) -> Any:
    """Zero-mean / unit-std normalisation over all elements (PPO convention)."""

    tensor = _as_tensor(advantages)
    if tensor.numel() <= 1:
        return tensor
    return (tensor - tensor.mean()) / (tensor.std(unbiased=False) + epsilon)


def check_finite(*tensors: Any) -> bool:
    """Return ``True`` if all tensors contain only finite values."""

    for tensor in tensors:
        if tensor is None:
            continue
        if torch is not None and isinstance(tensor, torch.Tensor):
            if not bool(torch.isfinite(tensor).all()):
                return False
    return True
