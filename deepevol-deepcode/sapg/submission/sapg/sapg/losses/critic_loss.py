"""Critic targets and critic losses for SAPG (Section 4.1, Eqs. 5-9).

The paper defines the critic update as follows.

On-policy data :math:`(s, a) \\sim \\pi_i` uses an **n-step** return (here
``n = 3``), Eq. (5)::

    V_on_target(s_t) = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_old(s_{t+3})

Off-policy data behaves differently: "we assume that an off-policy transition
can be used to approximate a 1-step return", Eq. (6)::

    V_off_target(s'_t) = r_t + gamma V_old(s'_{t+1})

The corresponding critic losses (Eqs. 7-8) are::

    L_on^critic(pi_i)        = E_{(s,a)~pi_i}[(V(s) - V_on_target(s))^2]
    L_off^critic(pi_i; X)    = 1/|X| sum_{j in X} E_{(s,a)~pi_j}[(V(s) - V_off_target(s))^2]

and are combined with the off-policy weight ``lambda`` (Eq. 9)::

    L^critic(pi_i) = L_on^critic(pi_i) + lambda * L_off^critic(pi_i)

Finally the critic term enters the total objective scaled by the *critic
coefficient* ``lambda'`` (Appendix B.1-B.3: ``lambda' = 4.0``)::

    total = L_policy + lambda' * L^critic

Note that ``lambda`` (off-policy weight, Section 4.1) and ``lambda'`` (critic
coefficient, Tables 2-4) are *different* coefficients; this module keeps them
separate to avoid double-scaling.

All tensor storage follows the ``rl_games``/IsaacGym convention used by
``sapg.buffers.rollout_buffer``: time-major tensors of shape ``[T, num_envs]``
(or ``[T, num_envs, 1]``).  Episode boundaries are handled through the ``dones``
flag: the n-step sum is truncated when an episode terminates inside the window
and the bootstrap value is masked accordingly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "CriticLoss",
    "compute_critic_loss",
    "critic_loss",
    "compute_on_policy_targets",
    "compute_off_policy_targets",
    "compute_n_step_targets",
    "compute_one_step_targets",
    "compute_value_loss",
    "combined_critic_loss",
    "lambda_prime",
    "CRITIC_COEFFICIENT",
    "N_STEP",
]

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

#: ``lambda'`` - critic coefficient from Appendix B.1-B.3 (Tables 2, 3, 4).
CRITIC_COEFFICIENT: float = 4.0

#: n-step horizon for the on-policy critic target ("here n = 3", Section 4.1).
N_STEP: int = 3

#: Off-policy weight ``lambda`` for the critic loss (Eq. 9); the paper uses 1.
OFF_POLICY_LAMBDA: float = 1.0

# ---------------------------------------------------------------------------
# key aliases (tolerant of dict / OffPolicyBatch / RolloutBuffer objects)
# ---------------------------------------------------------------------------

_REWARD_KEYS = ("rewards", "reward", "rew")
_DONE_KEYS = ("dones", "done", "terminals", "terminal", "masks")
_VALUE_KEYS = ("values", "value", "old_values", "v_old")
_NEXT_VALUE_KEYS = ("next_values", "next_value", "values_next", "bootstrap_values", "last_values")
_TARGET_KEYS = ("value_targets", "value_target", "returns", "targets", "vtarg")
_TARGET_OFF_KEYS = ("value_targets_off", "value_target_off", "off_value_targets", "targets_off")
_ON_POLICY_KEYS = ("on_policy_loss", "value_loss", "values_loss", "on_loss")
_OFF_POLICY_KEYS = ("off_policy_loss", "value_loss_off", "off_loss")


def _get(source: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Fetch the first present key from a dict-like / attribute-like object."""
    if source is None:
        return default
    for key in keys:
        if isinstance(source, dict):
            if key in source:
                return source[key]
        else:
            container = getattr(source, "data", None) or getattr(source, "storage", None)
            if container is not None and isinstance(container, dict) and key in container:
                return container[key]
            if hasattr(source, key):
                value = getattr(source, key)
                if value is not None:
                    return value
            # functional accessors such as ``.get(key)``
            getter = getattr(source, "get", None)
            if callable(getter):
                try:
                    value = getter(key)
                    if value is not None:
                        return value
                except Exception:  # pragma: no cover - defensive
                    pass
    return default


def _as_tensor(value: Any, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value if device is None else value.to(device)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _flatten_time(x: torch.Tensor, time_dim: int = 0) -> Any:
    """Return ``(flat, shape)`` with the time dimension moved/flattened away."""
    shape = x.shape
    if time_dim == 0:
        return x.reshape(shape[0], -1), shape
    moved = x.movedim(time_dim, 0)
    return moved.reshape(moved.shape[0], -1), moved.shape


def _unchunk(flat: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return flat.reshape(shape)


def _shift_time(x: torch.Tensor, offset: int, fill: float = 0.0) -> torch.Tensor:
    """``out[t] = x[t + offset]`` with ``fill`` beyond the stored horizon.

    ``x`` is a time-major ``[T, B]`` tensor.  For negative offsets this is a
    look-back shift (``out[t] = x[t + offset]``, ``fill`` before ``t=0``).
    """
    if offset == 0:
        return x
    T = x.shape[0]
    out = x.new_full(x.shape, fill)
    if offset > 0:
        if offset < T:
            out[: T - offset] = x[offset:]
    else:
        k = -offset
        if k < T:
            out[k:] = x[: T - k]
    return out


# ---------------------------------------------------------------------------
# Eq. (5): 3-step on-policy targets
# ---------------------------------------------------------------------------


def compute_n_step_targets(
    rewards: Any = None,
    dones: Any = None,
    values: Any = None,
    last_values: Any = None,
    gamma: float = 0.99,
    n_step: int = N_STEP,
    buffer: Any = None,
    time_dim: int = 0,
    mask_terminals: bool = True,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> torch.Tensor:
    r"""n-step on-policy critic targets, Eq. (5).

    .. math::
        V^{\text{target}}_{on}(s_t)=\sum_{k=t}^{t+n-1}\gamma^{k-t}r_k
                                 +\gamma^{n}V_{old}(s_{t+n})

    with the sum truncated at episode boundaries (when ``mask_terminals``) and
    the bootstrap value taken from ``values[t + n]`` or, when the window runs
    past the rollout horizon, from ``last_values`` (the value of the final
    observation, i.e. ``V_old(s_{T})``).

    Args:
        rewards: time-major rewards ``[T, ...]`` (or fetched from ``buffer``).
        dones: time-major terminal flags ``[T, ...]`` (or fetched from ``buffer``).
        values: time-major predicted values ``[T, ...]`` (or from ``buffer``).
        last_values: ``V_old`` of the state after the last stored step, shape
            broadcastable to ``[..., 1]``/``[num_envs]``.  When ``None`` the
            final stored value is reused.
        gamma: discount factor (0.99 in Appendix B).
        n_step: n in Eq. (5) - the paper uses ``n = 3``.
        buffer: optional dict-like / ``RolloutBuffer`` to resolve the inputs.
        time_dim: which axis is time (default 0, time-major storage).
        mask_terminals: truncate the multi-step sum at episode boundaries.
        device: optional device for the inputs.

    Returns:
        Tensor with the same shape as ``values``.
    """
    if buffer is not None:
        if rewards is None:
            rewards = _get(buffer, _REWARD_KEYS)
        if dones is None:
            dones = _get(buffer, _DONE_KEYS)
        if values is None:
            values = _get(buffer, _VALUE_KEYS)
        if last_values is None:
            last_values = _get(buffer, ("last_values", "bootstrap_values", "last_value"))
    rewards = _as_tensor(rewards, device)
    dones = _as_tensor(dones, device)
    values = _as_tensor(values, device)
    if values is None:
        raise ValueError("compute_n_step_targets requires `values`")
    if rewards is None:
        rewards = torch.zeros_like(values)
    if dones is None:
        dones = torch.zeros_like(values)

    orig_shape = values.shape
    r_flat, _ = _flatten_time(rewards, time_dim)
    d_flat, _ = _flatten_time(dones, time_dim)
    v_flat, _ = _flatten_time(values, time_dim)
    T, B = v_flat.shape

    # Pad rewards / dones with one extra step so index ``t + k`` is always valid.
    r_ext = torch.cat([r_flat, r_flat.new_zeros(1, B)], dim=0)
    d_ext = torch.cat([d_flat, d_flat.new_zeros(1, B)], dim=0)

    # Values extended with the bootstrap value V_old(s_T).
    if last_values is None:
        boot = v_flat[-1:]
    else:
        boot = _as_tensor(last_values, device)
        boot = boot.reshape(-1)[:B].reshape(1, B).to(v_flat.dtype)
    v_ext = torch.cat([v_flat, boot], dim=0)

    targets = torch.zeros_like(v_flat)
    continuity = torch.ones_like(v_flat)
    discount = 1.0
    n_step = int(n_step)
    for k in range(n_step):
        r_k = r_ext[k : k + T]
        targets = targets + discount * continuity * r_k
        if mask_terminals:
            d_k = d_ext[k : k + T]
            continuity = continuity * (1.0 - d_k)
        discount *= gamma

    # Bootstrap V_old(s_{t+n}) - clamped to the padded final entry.
    idx = torch.arange(T, device=v_flat.device) + n_step
    idx = idx.clamp(max=T)
    v_next = v_ext[idx]
    targets = targets + (gamma ** n_step) * continuity * v_next

    return _unchunk(targets, orig_shape)


#: Paper notation alias for Eq. (5).
compute_on_policy_targets = compute_n_step_targets
three_step_targets = compute_n_step_targets


# ---------------------------------------------------------------------------
# Eq. (6): 1-step off-policy targets
# ---------------------------------------------------------------------------


def compute_one_step_targets(
    rewards: Any = None,
    dones: Any = None,
    next_values: Any = None,
    gamma: float = 0.99,
    batch: Any = None,
    mask_terminals: bool = True,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> torch.Tensor:
    r"""1-step off-policy critic targets, Eq. (6).

    .. math::
        V^{\text{target}}_{off}(s'_t) = r_t + \gamma V_{old}(s'_{t+1})

    Terminal transitions contribute only their reward (``1 - done`` masks the
    bootstrap term).  Works for a flat off-policy batch ``[num_samples]`` as
    well as for time-major tensors.

    Args:
        rewards: rewards of the (off-policy) transitions.
        dones: terminal flags of the transitions.
        next_values: ``V_old(s'_{t+1})`` for each transition; falls back to
            ``values`` gathered one step ahead when ``None``.
        gamma: discount factor.
        batch: optional dict-like / ``OffPolicyBatch`` source for the arguments.
        mask_terminals: mask the bootstrap with ``(1 - done)``.
        device: optional device for the inputs.

    Returns:
        Tensor with the same shape as ``rewards``.
    """
    if batch is not None:
        if rewards is None:
            rewards = _get(batch, _REWARD_KEYS)
        if dones is None:
            dones = _get(batch, _DONE_KEYS)
        if next_values is None:
            next_values = _get(batch, _NEXT_VALUE_KEYS)
    rewards = _as_tensor(rewards, device)
    dones = _as_tensor(dones, device)
    next_values = _as_tensor(next_values, device)
    if rewards is None:
        raise ValueError("compute_one_step_targets requires `rewards`")
    if next_values is None:
        # Fall back to an element-wise / shifted look-up if the batch only
        # carries ``values`` (a valid approximation for the surrogate batch).
        values = None if batch is None else _get(batch, _VALUE_KEYS)
        values = _as_tensor(values, device)
        if values is None:
            next_values = torch.zeros_like(rewards)
        else:
            if values.dim() == rewards.dim() and values.shape[0] > rewards.shape[0]:
                next_values = torch.cat([values[1:], values[-1:]], dim=0)
            else:
                next_values = values
    targets = rewards + gamma * next_values
    if mask_terminals and dones is not None:
        targets = rewards + gamma * next_values * (1.0 - dones)
    return targets


#: Paper notation alias for Eq. (6).
compute_off_policy_targets = compute_one_step_targets
one_step_targets = compute_one_step_targets


# ---------------------------------------------------------------------------
# Eqs. (7)-(8): critic losses
# ---------------------------------------------------------------------------


def compute_value_loss(
    values: Any = None,
    value_targets: Any = None,
    coefficient: float = 1.0,
    huber_delta: Optional[float] = None,
    reduction: str = "mean",
    clip_value_loss: bool = False,
    value_clip: Optional[float] = None,
    old_values: Any = None,
    batch: Any = None,
    targets_key: Sequence[str] = _TARGET_KEYS,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    r"""Squared-error critic loss, Eqs. (7) and (8).

    .. math::
        L^{critic} = \mathbb{E}\left[(V(s) - V^{target}(s))^2\right]

    Args:
        values: predicted values ``V(s)``.
        value_targets: targets from Eq. (5) or Eq. (6).
        coefficient: multiplicative factor; ``1.0`` keeps the loss in the same
            units as the paper (the ``lambda' = 4.0`` critic coefficient is
            applied by :func:`combined_critic_loss` / the trainers).
        huber_delta: if given, use a Huber loss with this delta instead of a
            pure squared error (paper uses the plain squared error).
        reduction: ``"mean"`` (paper) or ``"none"``.
        clip_value_loss: optional PPO-style value clipping (off by default; the
            paper does not clip the value loss).
        value_clip: clipping range for value clipping.
        old_values: values from the behaviour policy, required if clipping.
        batch: optional dict-like source for ``values`` / ``value_targets``.
        targets_key: key aliases used when reading from ``batch``.

    Returns:
        Dict with ``value_loss`` (scaled), ``value_loss_raw``, ``value_error``,
        ``value_error_abs``, ``value_clip_frac``.
    """
    if batch is not None:
        if values is None:
            values = _get(batch, _VALUE_KEYS)
        if value_targets is None:
            value_targets = _get(batch, targets_key)
    values = _as_tensor(values, device)
    value_targets = _as_tensor(value_targets, device)
    if values is None or value_targets is None:
        raise ValueError("compute_value_loss requires `values` and `value_targets`")
    value_targets = value_targets.to(values.dtype)
    if value_targets.numel() and values.numel() != value_targets.numel():
        value_targets = value_targets.reshape(values.shape)

    error = values - value_targets
    clip_frac = values.new_zeros(())
    if clip_value_loss and value_clip is not None and old_values is not None:
        old = _as_tensor(old_values, device).to(values.dtype)
        clipped = old + (values - old).clamp(-value_clip, value_clip)
        error = torch.max((values - value_targets) ** 2, (clipped - value_targets) ** 2).sqrt() * torch.sign(error)
        clip_frac = (values - old).abs().gt(value_clip).float().mean()

    if huber_delta is not None and huber_delta > 0:
        abs_err = error.abs()
        raw = torch.where(
            abs_err <= huber_delta,
            0.5 * error.pow(2),
            huber_delta * (abs_err - 0.5 * huber_delta),
        )
    else:
        raw = error.pow(2)

    if reduction == "none":
        loss = raw
    else:
        loss = raw.mean()
    return {
        "value_loss": loss * coefficient,
        "value_loss_raw": loss if loss.dim() == 0 else loss.mean(),
        "value_error": error.mean(),
        "value_error_abs": error.abs().mean(),
        "value_clip_frac": clip_frac,
    }


# ---------------------------------------------------------------------------
# Eq. (9): on-policy + lambda * off-policy
# ---------------------------------------------------------------------------


def combined_critic_loss(
    on_policy_loss: Any,
    off_policy_loss: Any = None,
    lam: float = OFF_POLICY_LAMBDA,
) -> torch.Tensor:
    r"""Eq. (9): ``L^{critic} = L_on^{critic} + lambda * L_off^{critic}``.

    ``lambda`` is the off-policy weight from Section 4.1 (1.0 in the paper).
    The critic coefficient ``lambda' = 4.0`` (Appendix B) is *not* applied here;
    it scales the whole critic term when forming the total objective.
    """
    on = on_policy_loss if isinstance(on_policy_loss, torch.Tensor) else torch.as_tensor(on_policy_loss)
    if off_policy_loss is None:
        return on
    off = off_policy_loss if isinstance(off_policy_loss, torch.Tensor) else torch.as_tensor(off_policy_loss)
    return on + lam * off


def lambda_prime(coef: Optional[float] = None) -> float:
    """Return the critic coefficient ``lambda'`` (4.0 by default, Tables 2-4)."""
    return float(CRITIC_COEFFICIENT if coef is None else coef)


# ---------------------------------------------------------------------------
# Bundled functional loss
# ---------------------------------------------------------------------------


def compute_critic_loss(
    values: Any = None,
    value_targets: Any = None,
    value_targets_off: Any = None,
    values_off: Any = None,
    coefficient: float = CRITIC_COEFFICIENT,
    lam: float = OFF_POLICY_LAMBDA,
    gamma: float = 0.99,
    n_step: int = N_STEP,
    batch: Any = None,
    off_policy_batch: Any = None,
    huber_delta: Optional[float] = None,
    reduction: str = "mean",
    maximize_objective: bool = False,
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Dict[str, torch.Tensor]:
    r"""Critic loss for one policy: Eqs. (7)-(9) plus the ``lambda'`` scaling.

    ``L^{critic} = L_on^{critic} + lambda * L_off^{critic}`` and the returned
    ``critic_loss`` is ``lambda' * L^{critic}`` (``lambda' = 4.0``, Appendix B),
    ready to be added to the (clipped surrogate) policy loss.

    Args:
        values: ``V_{pi_i}(s)`` predicted values for the batch.
        value_targets: n-step on-policy targets (Eq. 5); computed from
            ``batch``/``rewards``/``dones`` when omitted.
        value_targets_off: 1-step off-policy targets (Eq. 6); when omitted and an
            ``off_policy_batch`` is provided they are computed on the fly.
        values_off: ``V_{pi_i}(s')`` for the off-policy batch (defaults to the
            ``values`` found inside ``off_policy_batch``).
        coefficient: critic coefficient ``lambda'`` (4.0).
        lam: off-policy weight ``lambda`` (1.0).
        gamma, n_step: target computation parameters.
        batch: on-policy dict-like batch (may carry ``value_targets``).
        off_policy_batch: off-policy dict-like batch ``D'_1``.
        huber_delta, reduction: passed to :func:`compute_value_loss`.
        maximize_objective: return the (maximised) objective sign instead.

    Returns:
        Dict with ``critic_loss``, ``critic_loss_raw``, ``on_policy_loss``,
        ``off_policy_loss``, ``value_loss``, ``value_loss_off``,
        ``value_error_abs``, ``value_error_abs_off`` and ``lambda_prime``.
    """
    if batch is not None:
        if values is None:
            values = _get(batch, _VALUE_KEYS)
        if value_targets is None:
            value_targets = _get(batch, _TARGET_KEYS)
    if off_policy_batch is not None:
        if values_off is None:
            values_off = _get(off_policy_batch, _VALUE_KEYS)
        if value_targets_off is None:
            value_targets_off = _get(off_policy_batch, _TARGET_OFF_KEYS)

    # Lazily compute the on-policy targets when they are not supplied.
    if value_targets is None:
        value_targets = compute_n_step_targets(
            rewards=kwargs.get("rewards"),
            dones=kwargs.get("dones"),
            values=values,
            last_values=kwargs.get("last_values"),
            gamma=gamma,
            n_step=n_step,
            buffer=batch,
            device=device,
        )

    on_stats = compute_value_loss(
        values=values,
        value_targets=value_targets,
        coefficient=1.0,
        huber_delta=huber_delta,
        reduction=reduction,
        device=device,
    )

    off_loss = None
    off_stats: Dict[str, Any] = {}
    if value_targets_off is not None or off_policy_batch is not None:
        if value_targets_off is None:
            value_targets_off = compute_one_step_targets(batch=off_policy_batch, gamma=gamma, device=device)
        if values_off is None:
            values_off = values
        off_stats = compute_value_loss(
            values=values_off,
            value_targets=value_targets_off,
            coefficient=1.0,
            huber_delta=huber_delta,
            reduction=reduction,
            device=device,
        )
        off_loss = off_stats["value_loss_raw"]

    raw = combined_critic_loss(on_stats["value_loss_raw"], off_loss, lam=lam)
    scaled = raw * coefficient
    loss = -scaled if maximize_objective else scaled

    result: Dict[str, torch.Tensor] = {
        "critic_loss": loss,
        "critic_loss_raw": raw,
        "on_policy_loss": on_stats["value_loss_raw"],
        "value_loss": on_stats["value_loss"],
        "value_error_abs": on_stats["value_error_abs"],
        "lambda_prime": torch.as_tensor(float(coefficient), device=raw.device),
        "lambda": torch.as_tensor(float(lam), device=raw.device),
    }
    if off_loss is not None:
        result["off_policy_loss"] = off_loss
        result["value_loss_off"] = off_stats["value_loss"]
        result["value_error_abs_off"] = off_stats["value_error_abs"]
    return result


#: Convenience alias mirroring ``ppo_loss``/``off_policy_loss`` naming.
def critic_loss(**kwargs: Any) -> Dict[str, torch.Tensor]:
    """Functional alias of :func:`compute_critic_loss`."""
    return compute_critic_loss(**kwargs)


# ---------------------------------------------------------------------------
# nn.Module wrapper
# ---------------------------------------------------------------------------


class CriticLoss(nn.Module):
    r"""Critic loss module implementing Eqs. (5)-(9).

    Combines the n-step on-policy value loss (Eq. 7) with the ``lambda``-scaled
    1-step off-policy value loss (Eq. 8) and applies the critic coefficient
    ``lambda' = 4.0`` (Appendix B.1-B.3).  The module is deliberately
    buffer-agnostic: it accepts raw tensors or any dict-like batch exposing
    ``values``/``value_targets`` (and the off-policy equivalents).

    Constructor params:
        critic_coefficient: ``lambda'`` (default 4.0 from Tables 2-4).
        lam: off-policy weight ``lambda`` for the critic (default 1.0).
        gamma: discount factor (0.99).
        n_step: n for Eq. (5) (3).
        huber_delta: optional Huber delta (paper uses plain squared error).
        clip_value_loss / value_clip: optional PPO-style value clipping.
        normalize_advantages: accepted for API symmetry (unused by the critic).
        config: optional :class:`sapg.utils.config.SAPGConfig`; reads
            ``critic_coefficient``, ``off_policy_weight``, ``gamma`` and
            ``critic_n_step`` when available.
    """

    def __init__(
        self,
        critic_coefficient: float = CRITIC_COEFFICIENT,
        lam: float = OFF_POLICY_LAMBDA,
        gamma: float = 0.99,
        n_step: int = N_STEP,
        huber_delta: Optional[float] = None,
        reduction: str = "mean",
        clip_value_loss: bool = False,
        value_clip: Optional[float] = None,
        normalize_advantages: bool = False,
        config: Optional[Any] = None,
    ) -> None:
        super().__init__()
        if config is not None:
            critic_coefficient = getattr(config, "critic_coefficient", critic_coefficient) or critic_coefficient
            lam = getattr(config, "off_policy_weight", lam) if hasattr(config, "off_policy_weight") else lam
            gamma = getattr(config, "gamma", gamma)
            n_step = getattr(config, "critic_n_step", None) or n_step
            if value_clip is None:
                value_clip = getattr(config, "value_clip", None)
            reduction = getattr(config, "value_loss_reduction", reduction)
        self.critic_coefficient = float(critic_coefficient)
        self.lam = float(lam)
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.huber_delta = huber_delta
        self.reduction = reduction
        self.clip_value_loss = bool(clip_value_loss)
        self.value_clip = value_clip
        self.normalize_advantages = bool(normalize_advantages)

    # -- targets ----------------------------------------------------------
    def targets(self, **kwargs: Any) -> torch.Tensor:
        """On-policy n-step targets (Eq. 5) using the module's gamma/n_step."""
        kwargs.setdefault("gamma", self.gamma)
        kwargs.setdefault("n_step", self.n_step)
        return compute_n_step_targets(**kwargs)

    def off_policy_targets(self, **kwargs: Any) -> torch.Tensor:
        """Off-policy 1-step targets (Eq. 6)."""
        kwargs.setdefault("gamma", self.gamma)
        return compute_one_step_targets(**kwargs)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        batch: Any = None,
        *,
        values: Any = None,
        value_targets: Any = None,
        value_targets_off: Any = None,
        values_off: Any = None,
        off_policy_batch: Any = None,
        coefficient: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Compute the critic loss; see :func:`compute_critic_loss` for keys."""
        if batch is not None and values is None:
            values = _get(batch, _VALUE_KEYS)
        if batch is not None and value_targets is None:
            value_targets = _get(batch, _TARGET_KEYS)
        stats = compute_critic_loss(
            values=values,
            value_targets=value_targets,
            value_targets_off=value_targets_off,
            values_off=values_off,
            coefficient=self.critic_coefficient if coefficient is None else coefficient,
            lam=self.lam,
            gamma=self.gamma,
            n_step=self.n_step,
            batch=batch,
            off_policy_batch=off_policy_batch,
            huber_delta=self.huber_delta,
            reduction=self.reduction,
            **kwargs,
        )
        return stats

    def loss(self, batch: Any = None, **kwargs: Any) -> torch.Tensor:
        """Return the scalar (minimised) critic loss."""
        return self.forward(batch, **kwargs)["critic_loss"]

    def extra_repr(self) -> str:
        return (
            f"critic_coefficient={self.critic_coefficient}, lam={self.lam}, "
            f"gamma={self.gamma}, n_step={self.n_step}, huber_delta={self.huber_delta}"
        )


def _self_test() -> None:  # pragma: no cover - manual sanity check
    torch.manual_seed(0)
    T, B, gamma, n = 6, 3, 0.99, 3
    rewards = torch.ones(T, B)
    dones = torch.zeros(T, B)
    values = torch.zeros(T, B)
    last = torch.full((B,), 5.0)
    targets = compute_n_step_targets(rewards, dones, values, last, gamma=gamma, n_step=n)
    # sum_{k=0}^{2} 0.99^k = 2.9701 ; + 0.99^3 * 5 = 4.8525
    expected_t0 = sum(gamma ** k for k in range(n)) + gamma ** n * 5.0
    assert abs(targets[0, 0].item() - expected_t0) < 1e-4, (targets[0, 0].item(), expected_t0)
    # terminal at t=1 cuts the window: r_0 + gamma * gamma... (continuity=0 -> only r_0)
    dones2 = dones.clone()
    dones2[1] = 1.0
    targets2 = compute_n_step_targets(rewards, dones2, values, last, gamma=gamma, n_step=n)
    assert abs(targets2[0, 0].item() - 1.0) < 1e-6, targets2[0, 0].item()
    t1 = compute_one_step_targets(rewards, dones, torch.full((T, B), 2.0), gamma=gamma)
    assert abs(t1[0, 0].item() - (1.0 + gamma * 2.0)) < 1e-6
    stats = compute_critic_loss(
        values=torch.zeros(4),
        value_targets=torch.ones(4),
        value_targets_off=torch.ones(4) * 2.0,
        values_off=torch.zeros(4),
        coefficient=4.0,
        lam=1.0,
    )
    assert abs(stats["on_policy_loss"].item() - 1.0) < 1e-6
    assert abs(stats["off_policy_loss"].item() - 4.0) < 1e-6
    assert abs(stats["critic_loss_raw"].item() - 5.0) < 1e-6
    assert abs(stats["critic_loss"].item() - 20.0) < 1e-6
    print("critic_loss self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
