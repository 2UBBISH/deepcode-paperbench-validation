"""Energy regularizer for BBox-Adapter (Eq. (3) of the paper).

The paper states, in Section 3.2 ("Adapter Update"):

    "Arbitrary energy models based on outputs, such as g_theta, may experience sharp
     gradients, leading to instability during training. To address this, we incorporate
     spectral normalization (Du & Mordatch, 2019) to Eq.(2)."

and the resulting gradient (Eq. (3)) is

    grad_theta l(theta) = grad_theta { -E_{y+ ~ p_data(y|x)}[g_theta(x, y+)]
                                       + alpha * E[g_theta(x, y+)^2]
                                       + E_{y- ~ p_theta(y|x)}[g_theta(x, y-)]
                                       + alpha * E[g_theta(x, y-)^2] }

The benchmark Addendum clarifies the implementation:

    "The paper mentions 'spectral normalization' in Section 3.2, which is implemented as
     l_2 regularization of the energies (alpha E[g_theta(x, y_+)^2] + alpha E[g_theta(x, y_-)^2])
     as shown in Equation 3, rather than using power iteration methods."

Therefore this module implements the regularizer as a simple L2 penalty on the *energies*
themselves (NOT on the parameters, and NOT via power iteration / spectral-norm layers):

    R(theta) = alpha * E[g_theta(x, y)^2]

applied separately to the positive energies and to the negative energies.  Because the
expectation is written separately for positives and negatives in Eq. (3), the two terms are
averaged independently and then added (this is what the code below does when both groups are
available; ``mode="concat"`` pools both groups before averaging and is offered only as a
diagnostic alternative).

The paper does not specify the coefficient ``alpha``; the reproduction plan asks for a sweep
over {1e-3, 1e-2, 1e-1} and then fixing it based on energy-curve stability.  A constant
default of 1e-2 is used here, and can be overridden by configs.

Cross-file relationships
------------------------
* ``bbox_adapter/losses/nce.py`` consumes :class:`EnergyRegularizer` to build the complete
  Eq. (3) objective (ranking NCE term + this regularizer).
* ``bbox_adapter/adapter/energy_model.py`` produces the energy tensors that are passed in.
* ``bbox_adapter/configs/*.yaml`` provide the ``alpha`` coefficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    "DEFAULT_ALPHA",
    "ALPHA_SWEEP",
    "RegularizerConfig",
    "squared_energy_penalty",
    "energy_regularizer",
    "positive_negative_penalty",
    "EnergyRegularizer",
    "alpha_schedule",
    "check_energy_scales",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Default coefficient for the energy L2 penalty.  ``alpha`` is *not* specified in the paper;
#: the reproduction plan asks for a sweep over the values below and then fixing the value
#: based on energy-curve stability.  1e-2 is the middle of the sweep and is a safe default.
DEFAULT_ALPHA: float = 1e-2

#: Candidate coefficients suggested by the reproduction plan (Phase 5, "missing details").
ALPHA_SWEEP: Tuple[float, ...] = (1e-3, 1e-2, 1e-1)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

@dataclass
class RegularizerConfig:
    """Configuration of the energy L2 regularizer (Eq. (3)).

    Parameters
    ----------
    alpha:
        Coefficient multiplying ``E[g_theta(x, y)^2]``.  ``alpha <= 0`` disables the penalty
        (the ablation "no regularization" setting).
    mode:
        ``"split"`` (default, paper-faithful): average the squared energies of the positive
        group and of the negative group separately, then add the two scaled terms, exactly as
        written in Eq. (3).  ``"concat"``: pool positives and negatives and take a single
        average (diagnostic only).
    reduction:
        ``"mean"`` (default) or ``"sum"`` over the samples of each group.  The paper writes an
        expectation, hence ``"mean"`` is the faithful choice.
    max_energy:
        Optional per-energy clamp applied *before* squaring, to keep the systematic/energy
        magnitudes bounded when the sweep pushes ``alpha`` low.  ``None`` disables clamping.
    warmup_steps:
        Linearly ramp ``alpha`` from 0 to its target over this many optimizer steps (0 = off).
        Not specified in the paper; provided for numerical stability of long runs.
    schedule:
        ``"constant"`` (default) or ``"inverse"``: ``alpha / (1 + step / warmup_steps)`` after
        the warmup.  Not specified in the paper; ``"constant"`` matches Eq. (3).
    """

    alpha: float = DEFAULT_ALPHA
    mode: str = "split"
    reduction: str = "mean"
    max_energy: Optional[float] = None
    warmup_steps: int = 0
    schedule: str = "constant"

    def __post_init__(self) -> None:
        if self.mode not in ("split", "concat"):
            raise ValueError(f"unknown regularizer mode: {self.mode!r} (use 'split' or 'concat')")
        if self.reduction not in ("mean", "sum"):
            raise ValueError(f"unknown reduction: {self.reduction!r} (use 'mean' or 'sum')")
        if self.schedule not in ("constant", "inverse"):
            raise ValueError(f"unknown schedule: {self.schedule!r}")

    @property
    def enabled(self) -> bool:
        """Whether the penalty contributes anything."""
        return float(self.alpha) != 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RegularizerConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(d).items() if k in known})


# --------------------------------------------------------------------------------------
# Functional forms
# --------------------------------------------------------------------------------------

def _as_tensor(energies: Any) -> torch.Tensor:
    """Coerce an energy container (tensor / sequence / nested sequence) to a flat tensor."""
    if isinstance(energies, torch.Tensor):
        return energies.reshape(-1)
    if isinstance(energies, (list, tuple)):
        flat: List[Any] = []
        for item in energies:
            if isinstance(item, torch.Tensor):
                flat.extend(item.reshape(-1).tolist())
            elif isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return torch.as_tensor(flat, dtype=torch.float32)
    return torch.as_tensor([energies], dtype=torch.float32)


def _reduce(squared: torch.Tensor, reduction: str) -> torch.Tensor:
    if squared.numel() == 0:
        return squared.sum() * 0.0
    return squared.mean() if reduction == "mean" else squared.sum()


def squared_energy_penalty(
    energies: Any,
    alpha: float = DEFAULT_ALPHA,
    reduction: str = "mean",
    clamp: Optional[float] = None,
) -> torch.Tensor:
    r"""Un-scaled L2 penalty on energies: :math:`\alpha \, \mathbb{E}[g_\theta^2]`.

    This is the atomic building block of Eq. (3); it is applied once to the positive energies
    and once to the negative energies.

    Parameters
    ----------
    energies:
        Scalar energies ``g_theta(x, y)``; any shape, flattened internally.  Must carry the
        autograd graph for the penalty to be differentiable (gradients are preserved).
    alpha:
        Coefficient ``alpha`` of Eq. (3).
    reduction:
        ``"mean"`` implements the expectation; ``"sum"`` sums the squared energies.
    clamp:
        If given, ``g`` is clamped to ``[-clamp, +clamp]`` before squaring.

    Returns
    -------
    torch.Tensor
        A scalar tensor ``alpha * E[g^2]``.
    """
    if alpha is None:
        alpha = 0.0
    g = _as_tensor(energies)
    if not torch.is_floating_point(g):
        g = g.to(torch.get_default_dtype())
    if clamp is not None:
        g = torch.clamp(g, -float(clamp), float(clamp))
    squared = g * g
    return float(alpha) * _reduce(squared, reduction)


def energy_regularizer(
    energies: Any,
    alpha: float = DEFAULT_ALPHA,
    reduction: str = "mean",
    clamp: Optional[float] = None,
) -> torch.Tensor:
    """Alias of :func:`squared_energy_penalty` with the paper's name for the term."""
    return squared_energy_penalty(energies, alpha=alpha, reduction=reduction, clamp=clamp)


def positive_negative_penalty(
    positive_energies: Any,
    negative_energies: Any,
    alpha: float = DEFAULT_ALPHA,
    reduction: str = "mean",
    clamp: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Both regularizer terms of Eq. (3), kept separate for logging.

    Returns
    -------
    (pos_term, neg_term):
        ``alpha * E[g_theta(x, y_+)^2]`` and ``alpha * E[g_theta(x, y_-)^2]``.
    """
    pos_term = squared_energy_penalty(positive_energies, alpha=alpha, reduction=reduction, clamp=clamp)
    neg_term = squared_energy_penalty(negative_energies, alpha=alpha, reduction=reduction, clamp=clamp)
    return pos_term, neg_term


# --------------------------------------------------------------------------------------
# Callable module-style wrapper
# --------------------------------------------------------------------------------------

class EnergyRegularizer:
    r"""Callable energy L2 penalty :math:`\alpha\mathbb{E}[g_\theta(x,y)^2]` (Eq. (3)).

    The regularizer operates on *energies*, not on parameters, and it deliberately does not
    implement power iteration: the Addendum states that the "spectral normalization" mentioned
    in Section 3.2 is realised as L2 regularization of the energies.

    Examples
    --------
    >>> import torch
    >>> reg = EnergyRegularizer(alpha=1e-2)
    >>> pos = torch.tensor([2.0, 3.0])           # g_theta(x, y_+)
    >>> neg = torch.tensor([-1.0, 0.5])          # g_theta(x, y_-)
    >>> total = reg(pos, neg)                    # alpha*(E[g_+^2] + E[g_-^2])
    >>> expected = 1e-2 * (torch.tensor([4.0, 9.0]).mean() + torch.tensor([1.0, 0.25]).mean())
    >>> bool(abs(float(total) - float(expected)) < 1e-9)
    True
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        mode: str = "split",
        reduction: str = "mean",
        max_energy: Optional[float] = None,
        warmup_steps: int = 0,
        schedule: str = "constant",
        config: Optional[RegularizerConfig] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = RegularizerConfig(
                alpha=alpha,
                mode=mode,
                reduction=reduction,
                max_energy=max_energy,
                warmup_steps=warmup_steps,
                schedule=schedule,
            )
        self._step = 0

    # -- properties --------------------------------------------------------------------
    @property
    def alpha(self) -> float:
        return float(self.config.alpha)

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def step(self) -> int:
        return self._step

    def set_alpha(self, alpha: float) -> "EnergyRegularizer":
        """Set the coefficient (used by the config-driven alpha sweep)."""
        self.config.alpha = float(alpha)
        return self

    def current_alpha(self) -> float:
        """``alpha`` at the current step, honouring the (optional) warmup/schedule."""
        return alpha_schedule(
            self.config.alpha,
            step=self._step,
            warmup_steps=self.config.warmup_steps,
            schedule=self.config.schedule,
        )

    def step_end(self) -> None:
        """Advance the internal step counter (call once per optimizer step)."""
        self._step += 1

    def reset(self) -> None:
        self._step = 0

    # -- core --------------------------------------------------------------------------
    def penalty(
        self,
        positive_energies: Any = None,
        negative_energies: Any = None,
        *,
        alpha: Optional[float] = None,
        update_step: bool = False,
    ) -> torch.Tensor:
        r"""Regularization term of Eq. (3) (scalar tensor).

        Parameters
        ----------
        positive_energies:
            ``g_theta(x, y_+)`` values; may be ``None`` if only negatives are available.
        negative_energies:
            ``g_theta(x, y_-)`` values; may be ``None`` if only positives are available.
        alpha:
            Overrides ``current_alpha()`` when provided.
        update_step:
            If True, advances the internal schedule counter after computing the penalty.

        Notes
        -----
        With ``mode="split"`` (paper-faithful) the two groups' squared energies are averaged
        separately and the two terms are added, matching the two ``alpha * E[g^2]`` terms of
        Eq. (3).  With ``mode="concat"`` all energies are pooled before the average.
        """
        if alpha is None:
            alpha = self.current_alpha()
        cfg = self.config

        if cfg.mode == "concat":
            parts = [
                _as_tensor(e)
                for e in (positive_energies, negative_energies)
                if e is not None and _as_tensor(e).numel() > 0
            ]
            if parts:
                pooled = torch.cat(parts, dim=0)
            else:
                pooled = torch.zeros(1, dtype=torch.get_default_dtype())
            out = squared_energy_penalty(pooled, alpha=alpha, reduction=cfg.reduction, clamp=cfg.max_energy)
        else:
            pos_term, neg_term = positive_negative_penalty(
                positive_energies if positive_energies is not None else [],
                negative_energies if negative_energies is not None else [],
                alpha=alpha,
                reduction=cfg.reduction,
                clamp=cfg.max_energy,
            )
            out = pos_term + neg_term

        if update_step:
            self.step_end()
        return out

    def split_terms(
        self,
        positive_energies: Any,
        negative_energies: Any,
        *,
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two Eq. (3) terms separately (for per-term logging / Appendix-K curves)."""
        if alpha is None:
            alpha = self.current_alpha()
        return positive_negative_penalty(
            positive_energies,
            negative_energies,
            alpha=alpha,
            reduction=self.config.reduction,
            clamp=self.config.max_energy,
        )

    def __call__(self, positive_energies: Any = None, negative_energies: Any = None, **kwargs: Any) -> torch.Tensor:
        """Shorthand for :meth:`penalty` with the default (paper) argument order."""
        return self.penalty(positive_energies, negative_energies, **kwargs)

    def combine(
        self,
        nce_loss: torch.Tensor,
        positive_energies: Any = None,
        negative_energies: Any = None,
        *,
        alpha: Optional[float] = None,
        update_step: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Add the regularizer to a ranking-NCE loss and return diagnostics.

        The paper's Eq. (3) objective is
        ``-E[g(x,y+)] + E[g(x,y-)] + alpha*E[g(x,y+)^2] + alpha*E[g(x,y-)^2]``;
        ``losses/nce.py`` owns the first two terms, this method adds the last two.

        Returns
        -------
        (total_loss, stats):
            ``total_loss = nce_loss + reg`` and a dict with ``reg``, ``reg_pos``, ``reg_neg``
            and ``alpha`` for logging.
        """
        if self.config.mode == "split":
            pos_term, neg_term = self.split_terms(positive_energies, negative_energies, alpha=alpha)
        else:
            pos_term = self.penalty(positive_energies, negative_energies, alpha=alpha)
            neg_term = torch.zeros((), dtype=pos_term.dtype, device=pos_term.device)
        reg = pos_term + neg_term
        total = nce_loss + reg
        if update_step:
            self.step_end()
        stats = {
            "reg": float(reg.detach()),
            "reg_pos": float(pos_term.detach()),
            "reg_neg": float(neg_term.detach()),
            "alpha": float(self.current_alpha()),
        }
        return total, stats

    def state_dict(self) -> Dict[str, Any]:
        return {"config": self.config.to_dict(), "step": self._step}

    def load_state_dict(self, state: Mapping[str, Any]) -> "EnergyRegularizer":
        if "config" in state:
            self.config = RegularizerConfig.from_dict(state["config"])
        self._step = int(state.get("step", 0))
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = self.config
        return (
            f"EnergyRegularizer(alpha={c.alpha}, mode={c.mode!r}, reduction={c.reduction!r}, "
            f"max_energy={c.max_energy}, warmup_steps={c.warmup_steps}, schedule={c.schedule!r})"
        )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def alpha_schedule(
    alpha: float,
    step: int = 0,
    warmup_steps: int = 0,
    schedule: str = "constant",
) -> float:
    """Effective ``alpha`` at a given optimizer step.

    ``warmup_steps > 0`` linearly ramps ``alpha`` from 0 to its target; ``schedule="inverse"``
    then decays it as ``alpha / (1 + (step - warmup) / warmup)``.  The paper specifies neither
    mechanism, so the default (``warmup_steps=0``, ``"constant"``) reproduces Eq. (3) verbatim.
    """
    alpha = float(alpha)
    if warmup_steps <= 0:
        return alpha
    if step < warmup_steps:
        return alpha * (float(step) / float(warmup_steps))
    if schedule == "inverse":
        return alpha / (1.0 + (float(step) - float(warmup_steps)) / float(warmup_steps))
    return alpha


def check_energy_scales(
    positive_energies: Any,
    negative_energies: Any = None,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, float]:
    """Diagnostics on energy magnitudes, used to fix ``alpha`` from the sweep.

    Returns mean/std/abs-max of the positive and negative energies plus the resulting
    regularizer magnitude, so the stability of the energy curves can be judged objectively
    (the reproduction plan's criterion for choosing ``alpha``).
    """
    out: Dict[str, float] = {"alpha": float(alpha)}
    groups: Iterable[Tuple[str, Any]] = (
        ("pos", positive_energies),
        ("neg", negative_energies),
    )
    for tag, values in groups:
        if values is None:
            continue
        t = _as_tensor(values).detach().float()
        if t.numel() == 0:
            continue
        out[f"{tag}_mean"] = float(t.mean())
        out[f"{tag}_std"] = float(t.std()) if t.numel() > 1 else 0.0
        out[f"{tag}_absmax"] = float(t.abs().max())
        out[f"{tag}_rms"] = float(torch.sqrt((t * t).mean()))
        out[f"{tag}_reg"] = float(alpha) * float((t * t).mean())
    if "pos_reg" in out and "neg_reg" in out:
        out["reg_total"] = out["pos_reg"] + out["neg_reg"]
    # NaN/Inf guard surfaces exploding energies during the alpha sweep.
    out["finite"] = float(all(math.isfinite(v) for v in out.values()))
    return out
