"""Back-to-source activation shifting for FOA (Section 3.2, Eqn. (7)-(9)).

FOA shifts the *final-layer* [CLS] feature ``e_N^0`` -- the exact input of the
classification head -- towards the centre of the source in-distribution feature
distribution.  No backpropagation is involved: the shift is a plain vector
addition performed before the task head.

Reference (verbatim from Section 3.2 of the paper)::

    e_N^0 <- e_N^0 + gamma * d                                   Eqn. (7)

    d_t   = mu_N^S - mu_N(t)                                     Eqn. (8)

    mu_N(t) = alpha * mu_N(X_t) + (1 - alpha) * mu_N(t-1)        Eqn. (9)

where

* ``mu_N^S`` is the mean of the N-th layer [CLS] feature over source
  in-distribution samples (same ``D_S`` as Eqn. (5)),
* ``mu_N(X_t)`` is the mean over the current test batch ``X_t``,
* ``alpha`` is the moving average factor, set to ``0.1`` (Section 3.2/4, B.2),
* ``gamma`` is the step size, set to ``1.0`` (Section 4, B.2).

Per-batch ordering (documented default of this reproduction, see plan item 6
and the Addendum): ``initialize -> shift -> EMA update``.  Concretely, for the
very first batch ``X_1`` the EMA state is initialised from that batch,
``mu_N(0) = mu_N(X_1)``, the shift is computed with ``mu_N(t-1)`` and applied
*before* the head, and only then is the EMA state refreshed with the
*un-shifted* batch statistics.  This guarantees that batch 1 is shifted towards
the source centre as well (``d_1 = mu_N^S - mu_N(X_1)``).

Table 14 (EMA ablation): disabling Eqn. (9) means the shifting direction is
recomputed from the raw current-batch statistics,
``d_t = mu_N^S - mu_N(X_t)`` (see ``use_ema=False``).

Everything here is gradient-free and runs in ``torch.no_grad()``.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Union

import torch

__all__ = [
    "ActivationShifting",
    "BackToSourceShifting",
    "build_activation_shifter",
    "shift_direction",
    "DEFAULT_ALPHA",
    "DEFAULT_GAMMA",
]

DEFAULT_ALPHA = 0.1  # Eqn. (9) moving average factor (Section 3.2 / B.2)
DEFAULT_GAMMA = 1.0  # Eqn. (7) step size (Section 4 / B.2)


# ---------------------------------------------------------------------------
# stateless helpers
# ---------------------------------------------------------------------------
def shift_direction(
    source_mean: torch.Tensor,
    test_mean: torch.Tensor,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """Return ``gamma * d`` with ``d = mu_N^S - mu_N(t)`` (Eqn. (7)-(8)).

    Parameters
    ----------
    source_mean:
        ``mu_N^S``, shape ``[d]`` (or ``[1, d]``).
    test_mean:
        ``mu_N(t)`` (or the raw batch mean ``mu_N(X_t)``), shape ``[d]``.
    gamma:
        Step size of Eqn. (7); the paper uses ``1.0``.
    """
    sm = source_mean.detach()
    tm = test_mean.detach()
    if sm.dim() > 1:
        sm = sm.reshape(-1)
    if tm.dim() > 1:
        tm = tm.reshape(-1)
    if sm.shape != tm.shape:
        raise ValueError(
            f"shift_direction: source mean {tuple(sm.shape)} and test mean "
            f"{tuple(tm.shape)} must match"
        )
    return (float(gamma) * (sm - tm)).to(tm.dtype)


def _as_vector(t: torch.Tensor) -> torch.Tensor:
    """Flatten a ``[d]`` / ``[1, d]`` tensor to ``[d]``."""
    return t.reshape(-1) if t.dim() > 1 else t


# ---------------------------------------------------------------------------
# stateful module
# ---------------------------------------------------------------------------
class ActivationShifting:
    """Online back-to-source activation shifting (Eqn. (7), (8), (9)).

    Parameters
    ----------
    source_mean:
        ``mu_N^S`` -- per-dimension mean of the N-th layer source [CLS]
        features.  Either a tensor of shape ``[d]`` / ``[1, d]`` or a
        ``SourceStats``-like object exposing ``mu_final``.
    alpha:
        Moving average factor of Eqn. (9) (paper: ``0.1``).
    gamma:
        Step size of Eqn. (7) (paper: ``1.0``).
    use_ema:
        If ``True`` (default) the direction follows Eqn. (8) with the EMA
        estimate of Eqn. (9).  If ``False`` the raw current-batch statistics
        are used instead (Table 14 "without Eqn. (9)").
    clamp_distance:
        Optional safety guard clipping the L2 norm of the applied shift
        (``None`` = no clipping, the paper's setting).
    """

    def __init__(
        self,
        source_mean: Union[torch.Tensor, Any],
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        use_ema: bool = True,
        clamp_distance: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        mu_s = self._extract_source_mean(source_mean)
        self.dim = int(mu_s.numel())
        if device is not None:
            mu_s = mu_s.to(device=device)
        self.source_mean = mu_s.detach().to(dtype=dtype).clone()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.use_ema = bool(use_ema)
        self.clamp_distance = (
            None if clamp_distance is None else float(clamp_distance)
        )

        # EMA state
        self.mu: Optional[torch.Tensor] = None  # mu_N(t)
        self.initialized = False
        self.num_batches = 0
        self._last_direction: Optional[torch.Tensor] = None  # gamma * d_t
        self._last_batch_mean: Optional[torch.Tensor] = None  # mu_N(X_t)

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _extract_source_mean(source_mean: Any) -> torch.Tensor:
        if torch.is_tensor(source_mean):
            return _as_vector(source_mean.detach().float().clone())
        for attr in ("mu_final", "mu_N", "source_mean"):
            if hasattr(source_mean, attr):
                val = getattr(source_mean, attr)
                if callable(val):
                    val = val()
                if torch.is_tensor(val):
                    return _as_vector(val.detach().float().clone())
        if hasattr(source_mean, "mu"):
            mu = getattr(source_mean, "mu")
            if isinstance(mu, (list, tuple)) and len(mu) > 0:
                return _as_vector(mu[-1].detach().float().clone())
        raise TypeError(
            "ActivationShifting: could not extract the source mean `mu_N^S` "
            "from the provided object; pass a tensor or a SourceStats instance."
        )

    # -- state handling ------------------------------------------------------
    def reset(self) -> None:
        """Clear the EMA state (``mu_N(0)`` will come from the next batch)."""
        self.mu = None
        self.initialized = False
        self.num_batches = 0
        self._last_direction = None
        self._last_batch_mean = None

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "ActivationShifting":
        self.source_mean = self.source_mean.to(device=device)
        if self.mu is not None:
            self.mu = self.mu.to(device=device)
        if dtype is not None:
            self.source_mean = self.source_mean.to(dtype=dtype)
        return self

    def state_dict(self) -> Dict[str, Any]:
        """Serialize the shifting state (for checkpoint/resume)."""
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "use_ema": self.use_ema,
            "clamp_distance": self.clamp_distance,
            "dim": self.dim,
            "num_batches": self.num_batches,
            "initialized": bool(self.initialized),
            "mu": None if self.mu is None else self.mu.detach().cpu().clone(),
            "source_mean": self.source_mean.detach().cpu().clone(),
            "last_direction": (
                None if self._last_direction is None
                else self._last_direction.detach().cpu().clone()
            ),
            "last_batch_mean": (
                None if self._last_batch_mean is None
                else self._last_batch_mean.detach().cpu().clone()
            ),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = False) -> None:
        self.alpha = state.get("alpha", self.alpha)
        self.gamma = state.get("gamma", self.gamma)
        self.use_ema = state.get("use_ema", self.use_ema)
        self.clamp_distance = state.get("clamp_distance", self.clamp_distance)
        self.num_batches = int(state.get("num_batches", 0))
        self.initialized = bool(state.get("initialized", False))
        mu = state.get("mu", None)
        self.mu = None if mu is None else mu.to(self.source_mean.device)
        sm = state.get("source_mean", None)
        if sm is not None:
            self.source_mean = sm.to(self.source_mean.device)
        ld = state.get("last_direction", None)
        self._last_direction = None if ld is None else ld.to(self.source_mean.device)
        lb = state.get("last_batch_mean", None)
        self._last_batch_mean = None if lb is None else lb.to(self.source_mean.device)

    def clone(self) -> "ActivationShifting":
        """Deep copy (used to evaluate several candidates from one state)."""
        other = ActivationShifting(
            source_mean=self.source_mean,
            alpha=self.alpha,
            gamma=self.gamma,
            use_ema=self.use_ema,
            clamp_distance=self.clamp_distance,
        )
        other.mu = None if self.mu is None else self.mu.clone()
        other.initialized = self.initialized
        other.num_batches = self.num_batches
        other._last_direction = (
            None if self._last_direction is None else self._last_direction.clone()
        )
        other._last_batch_mean = (
            None if self._last_batch_mean is None else self._last_batch_mean.clone()
        )
        return other

    # -- core statistics -----------------------------------------------------
    @staticmethod
    def batch_mean(final_cls: torch.Tensor) -> torch.Tensor:
        """``mu_N(X_t)``: mean of the batch's final-layer [CLS] features.

        Accepts ``[B, d]`` (or ``[B, 1, d]``) and returns ``[d]``.
        """
        if final_cls.dim() == 3:
            final_cls = final_cls.reshape(final_cls.shape[0], -1)
        if final_cls.dim() != 2:
            raise ValueError(
                f"batch_mean expects [B, d] final-layer CLS features, got "
                f"{tuple(final_cls.shape)}"
            )
        return final_cls.mean(dim=0)

    def current_mean(self) -> Optional[torch.Tensor]:
        """``mu_N(t)`` -- the current EMA estimate (``None`` before batch 1)."""
        return self.mu

    @property
    def direction(self) -> Optional[torch.Tensor]:
        """The most recently applied shift ``gamma * d_t`` (or ``None``)."""
        return self._last_direction

    # -- online API ----------------------------------------------------------
    def direction_for(self, test_mean: torch.Tensor) -> torch.Tensor:
        """Compute ``gamma * d_t`` for the given (un-shifted) batch mean.

        Uses ``mu_N(t-1)`` when the EMA is enabled and already initialised,
        otherwise the raw batch mean (this is exactly what makes batch 1 shift
        towards the source centre: ``d_1 = mu_N^S - mu_N(X_1)``).
        """
        tm = test_mean.detach().to(self.source_mean.device, dtype=self.source_mean.dtype)
        if self.use_ema and self.initialized and self.mu is not None:
            reference = self.mu
        else:
            reference = tm
        shift = shift_direction(self.source_mean, reference, gamma=self.gamma)
        if self.clamp_distance is not None:
            norm = shift.norm()
            if float(norm) > self.clamp_distance and float(norm) > 0.0:
                shift = shift * (self.clamp_distance / float(norm))
        return shift

    def apply_shift(self, final_cls: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        """Eqn. (7): ``e_N^0 <- e_N^0 + gamma * d`` (broadcast over the batch)."""
        return final_cls + shift.to(device=final_cls.device, dtype=final_cls.dtype)

    def shift(self, final_cls: torch.Tensor) -> torch.Tensor:
        """Shift the features using the *current* state, without updating it.

        Useful for FOA-I V1 (buffered features) and for multi-candidate
        evaluation, where one direction per batch is applied to every
        candidate / every buffered sample.
        """
        if not self.initialized and not self.use_ema:
            # Stateless mode (Table 14, no Eqn. (9)): direction from this batch.
            mean = self.batch_mean(final_cls)
            shift = self.direction_for(mean)
        elif self._last_direction is not None:
            shift = self._last_direction
        else:
            mean = self.batch_mean(final_cls)
            shift = self.direction_for(mean)
        return self.apply_shift(final_cls, shift)

    def update(self, final_cls: torch.Tensor) -> torch.Tensor:
        """Refresh the EMA state from the *un-shifted* batch statistics.

        Implements Eqn. (9): ``mu_N(t) = alpha * mu_N(X_t) + (1-alpha)*mu_N(t-1)``
        (with ``mu_N(0)`` taken from the first observed batch).
        """
        mean = self.batch_mean(final_cls).detach().to(
            self.source_mean.device, dtype=self.source_mean.dtype
        )
        self._last_batch_mean = mean
        if (not self.initialized) or self.mu is None or not self.use_ema:
            if not self.use_ema:
                # No EMA (Table 14): the "estimate" is simply the batch mean.
                self.mu = mean
            else:
                self.mu = mean if not self.initialized else (
                    self.alpha * mean + (1.0 - self.alpha) * self.mu
                )
        else:
            self.mu = self.alpha * mean + (1.0 - self.alpha) * self.mu
        self.initialized = True
        self.num_batches += 1
        return self.mu

    def shift_and_update(self, final_cls: torch.Tensor) -> torch.Tensor:
        """One online step: ``initialize -> shift -> EMA update``.

        Returns the shifted final-layer [CLS] features that must be fed to the
        classification head (``Head(shifted)``).
        """
        # 1) initialize mu_N(0) from the first batch (so batch 1 is shifted)
        if (not self.initialized) and self.use_ema and self.mu is None:
            self.mu = self.batch_mean(final_cls).detach().to(
                self.source_mean.device, dtype=self.source_mean.dtype
            )
            self.initialized = True
        # 2) compute + apply the shift (d_t = mu_N^S - mu_N(t-1))
        mean = self.batch_mean(final_cls)
        shift = self.direction_for(mean)
        self._last_direction = shift.detach()
        shifted = self.apply_shift(final_cls, shift)
        # 3) refresh the EMA with the UN-shifted batch statistics
        self.update(final_cls)
        return shifted

    # convenient alias used by the main loop
    def __call__(self, final_cls: torch.Tensor) -> torch.Tensor:
        return self.shift_and_update(final_cls)

    def shift_and_classify(self, final_cls: torch.Tensor, head) -> torch.Tensor:
        """Apply the shift before the head, then classify (Eqn. (7) + head)."""
        shifted = self.shift_and_update(final_cls)
        return head(shifted)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, alpha={self.alpha}, gamma={self.gamma}, "
            f"use_ema={self.use_ema}, num_batches={self.num_batches}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


# alias kept for readability in the runner code
BackToSourceShifting = ActivationShifting


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def build_activation_shifter(
    source_stats_or_mean: Any,
    cfg: Any = None,
    *,
    enabled: Optional[bool] = None,
    alpha: Optional[float] = None,
    gamma: Optional[float] = None,
    use_ema: Optional[bool] = None,
    clamp_distance: Optional[float] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[ActivationShifting]:
    """Config-driven factory for :class:`ActivationShifting`.

    Reads ``cfg.shifting.{enabled,gamma,alpha,use_ema,clamp_distance}`` and
    returns ``None`` when shifting is disabled (used by the Table 5 ablation
    "entropy + discrepancy without shifting").
    """
    shifting_cfg = _cfg_get(cfg, "shifting", cfg if not isinstance(cfg, dict) else None)
    if shifting_cfg is None:
        shifting_cfg = cfg

    if enabled is None:
        enabled = _cfg_get(shifting_cfg, "enabled", True)
    if gamma is None:
        gamma = _cfg_get(shifting_cfg, "gamma", DEFAULT_GAMMA)
    if alpha is None:
        alpha = _cfg_get(shifting_cfg, "alpha", DEFAULT_ALPHA)
    if use_ema is None:
        use_ema = _cfg_get(shifting_cfg, "use_ema", True)
    if clamp_distance is None:
        clamp_distance = _cfg_get(shifting_cfg, "clamp_distance", None)

    if not enabled:
        return None

    return ActivationShifting(
        source_mean=source_stats_or_mean,
        alpha=float(alpha),
        gamma=float(gamma),
        use_ema=bool(use_ema),
        clamp_distance=clamp_distance,
        device=device,
        dtype=dtype,
    )
