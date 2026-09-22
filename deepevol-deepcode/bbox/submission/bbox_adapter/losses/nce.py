"""Ranking-based Noise Contrastive Estimation (NCE) loss for BBox-Adapter.

This module implements the loss of Section 3.2 of *Lightweight Adapting for
Black-Box Large Language Models* together with the gradient of Eq. (3) and the
proofs of Appendix A / Appendix B.

Paper statements implemented here
---------------------------------
Eq. (1) -- parameterised posterior over a contrastive set
    ``{x_k}_{k=1}^K`` (one data sample ``x_+`` plus LLM samples):

        p_theta(k | {x_k}_{k=1}^K) = exp(g_theta(x_k)) / sum_k exp(g_theta(x_k))

Eq. (2) -- ranking-based NCE objective (minimise the KL between the data
posterior ``q`` and ``p_theta``, Appendix A):

        max_theta  E_{p_data(x)} [ g_theta(x) - log sum_k exp(g_theta(x_k)) ]

    Equivalently, the *loss* that is minimised is the negative of the above.
    With a single positive index (the sample drawn from ``p_data``) and the
    remaining entries of the set drawn from the black-box LLM, this is exactly
    the softmax cross-entropy of the positive entry:

        l_nce = -g_theta(x, y_+) + logsumexp_k g_theta(x, y_k)

Eq. (3) -- gradient, i.e. the objective whose derivative the paper writes as
(Appendix B derives ``-grad l = E_data[grad g] - E_{p_theta}[grad g]``):

        grad l(theta) = grad { -E_{y+ ~ p_data}[g(x, y_+)] + alpha E[g(x, y_+)^2]
                             + E_{y- ~ p_theta}[g(x, y_-)] + alpha E[g(x, y_-)^2] }

    The two ``alpha`` terms are the "spectral normalization" of Eq. (3); per the
    addendum they are realised as an ``l_2`` penalty on the scalar energies and
    NOT as power iteration.  They are produced by
    :mod:`bbox_adapter.adapter.regularizer` (``alpha * E[g^2]``) and combined
    here so that the whole Eq. (3) objective is available from one call.

Notes on the finite-set gradient
--------------------------------
Eq. (3) writes ``E_{y- ~ p_theta(y|x)}[g(x, y_-)]`` -- an expectation over the
*whole* adapted model distribution ``p_theta(y|x) = p_LLM(y|x) exp(g_theta)``,
which is approximated by sampling negatives from the current adapted inference
(Section 3.4).  In code we work with the finite contrastive set that is
actually available per query; the exact, differentiable form of Eq. (2) on that
finite set weights the negatives with the set posterior ``p_theta(k | set)``.
Both variants are exposed:

* :func:`compute_nce_loss` (and :class:`RankingNCELoss`) -- the differentiable
  softmax form of Eq. (1)/(2), used to obtain ``theta`` updates;
* :func:`nce_gradient_terms` -- the four unweighted terms of Eq. (3) exactly as
  printed in the paper (used for gradient diagnostics / curve logging).

The black-box LLM is never differentiated against: only the adapter's scalar
energies take part in the graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..adapter.regularizer import (
    DEFAULT_ALPHA,
    EnergyRegularizer,
    RegularizerConfig,
    squared_energy_penalty,
)

__all__ = [
    "ENERGY_CLAMP_DEFAULT",
    "NCELossConfig",
    "RankingNCELoss",
    "compute_nce_loss",
    "ranking_nce_loss",
    "nce_gradient_terms",
    "nce_objective",
    "binary_nce_loss",
    "pairwise_ranking_loss",
    "softmax_posterior",
    "log_softmax_posterior",
    "ranking_accuracy",
    "build_contrastive_tensors",
    "score_contrastive_sets",
    "nce_loss_from_model",
]

# Clamp used only when the caller explicitly asks for one (``energy_clamp``).
# Default ``None`` keeps the loss numerically identical to Eq. (2); stability is
# otherwise provided by the ``alpha * E[g^2]`` term and by logsumexp.
ENERGY_CLAMP_DEFAULT: Optional[float] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_energy_tensor(x: Any, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Coerce nested lists / numpy / tensors to a flat float tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach() if not x.requires_grad else x
    if x is None:
        return torch.zeros(0, dtype=dtype)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.zeros(0, dtype=dtype)
        if all(isinstance(e, torch.Tensor) for e in x):
            return torch.cat([e.reshape(-1).to(dtype) for e in x], dim=0)
        return torch.tensor([float(e) for e in x], dtype=dtype)
    return torch.tensor([float(x)], dtype=dtype)


def _log_softmax_set(logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Numerically stable log-softmax over the contrastive-set axis.

    ``logits`` has shape ``[B, K]``; ``mask`` (bool, ``True`` = valid) may be
    given for right-padded negative sets.
    """
    if mask is not None:
        neg_inf = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        logits = torch.where(mask, logits, neg_inf)
    return logits - torch.logsumexp(logits, dim=-1, keepdim=True)


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return values.mean() if values.numel() else values.new_zeros(())
    mask = mask.to(values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


# ---------------------------------------------------------------------------
# core loss computation (Eq. 1 / Eq. 2)
# ---------------------------------------------------------------------------
def compute_nce_loss(
    positive_energies: Any,
    negative_energies: Any = None,
    *,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
    energy_clamp: Optional[float] = ENERGY_CLAMP_DEFAULT,
    per_query_reduction: bool = False,
) -> torch.Tensor:
    """Ranking-based NCE loss (Eq. 2) for a batch of contrastive sets.

    Parameters
    ----------
    positive_energies:
        ``[B]`` tensor (or list of floats) of ``g_theta(x, y_+)``.
    negative_energies:
        ``[B, N]`` padded tensor / list of per-query tensors / ``None``.  Each
        entry is ``g_theta(x, y_-)`` for a negative sample of the *same* query.
    mask:
        Optional bool ``[B, N]`` mask (``True`` = valid negative).
    reduction:
        ``"mean"`` (default, implements ``E_{p_data}``), ``"sum"`` or ``"none"``.
    temperature:
        Optional softmax temperature; ``1.0`` reproduces Eq. (1) verbatim.
    label_smoothing:
        Optional uniform smoothing over the contrastive set (``0.0`` = Eq. 2).
    energy_clamp:
        Optional symmetric clamp on energies before the softmax (stability aid).
    per_query_reduction:
        If ``True`` return the per-query losses (``[B]``) instead of a scalar.

    Returns
    -------
    torch.Tensor
        Scalar (or ``[B]``) loss ``-g_+ + logsumexp_k g_k``.
    """
    temp = float(temperature) if temperature else 1.0

    if isinstance(positive_energies, torch.Tensor):
        pos = positive_energies.reshape(-1)
    else:
        pos = _as_energy_tensor(positive_energies).reshape(-1)
    if pos.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=pos.device if pos.numel() else None)

    if energy_clamp is not None:
        pos = pos.clamp(-float(energy_clamp), float(energy_clamp))

    # Per-query contrastive sets: support both padded tensors and ragged lists.
    if negative_energies is None:
        sets = [pos[i : i + 1] for i in range(pos.numel())]
        mask = None
    elif isinstance(negative_energies, torch.Tensor):
        neg = negative_energies
        if neg.dim() == 1:
            neg = neg.unsqueeze(-1)
        neg = neg.reshape(pos.numel(), -1) if neg.dim() == 2 else neg.reshape(pos.numel(), -1)
        if energy_clamp is not None:
            neg = neg.clamp(-float(energy_clamp), float(energy_clamp))
        sets = [torch.cat([pos[i : i + 1], neg[i]], dim=0) for i in range(pos.numel())]
    else:
        neg_list = list(negative_energies)
        if len(neg_list) != pos.numel():
            raise ValueError(
                f"expected {pos.numel()} negative sets, received {len(neg_list)}"
            )
        sets = []
        for i, nset in enumerate(neg_list):
            n = _as_energy_tensor(nset).reshape(-1)
            if n.numel():
                if energy_clamp is not None:
                    n = n.clamp(-float(energy_clamp), float(energy_clamp))
                sets.append(torch.cat([pos[i : i + 1], n], dim=0))
            else:
                sets.append(pos[i : i + 1])

    losses: List[torch.Tensor] = []
    for s in sets:
        logits = (s / temp).unsqueeze(0)  # [1, K]
        log_probs = _log_softmax_set(logits)  # [1, K]
        lp_pos = log_probs[0, 0]
        if label_smoothing and label_smoothing > 0.0:
            k = logits.shape[1]
            uniform = -math.log(k) if k > 0 else torch.zeros((), device=logits.device)
            lp_pos = (1.0 - label_smoothing) * lp_pos + label_smoothing * uniform
        losses.append(-lp_pos)

    loss_vec = torch.stack(losses) if losses else pos.new_zeros(0)
    if per_query_reduction or reduction == "none":
        return loss_vec
    if mask is not None and isinstance(mask, torch.Tensor) and mask.shape[-1] > 1:
        # Masked-negative variant: recompute with the mask applied, query by query.
        masked: List[torch.Tensor] = []
        for i, s in enumerate(sets):
            m = mask[i]
            if m.dim() == 0:
                m = m.unsqueeze(0)
            m = m.to(torch.bool)
            if m.numel() == s.numel():
                keep = torch.cat(
                    [torch.ones(1, dtype=torch.bool, device=s.device), m[1:]], dim=0
                )
            else:
                keep = torch.cat(
                    [torch.ones(1, dtype=torch.bool, device=s.device), m], dim=0
                )
            logits = (s / temp).unsqueeze(0)
            lp = _log_softmax_set(logits, keep.unsqueeze(0))[0, 0]
            masked.append(-lp)
        loss_vec = torch.stack(masked)
    if reduction == "sum":
        return loss_vec.sum()
    return loss_vec.mean()


def ranking_nce_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Alias of :func:`compute_nce_loss` using the paper's term name."""
    return compute_nce_loss(*args, **kwargs)


def softmax_posterior(energies: Any, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``p_theta(k | {x_k})`` of Eq. (1) (``[B, K]`` -> probabilities)."""
    e = energies if isinstance(energies, torch.Tensor) else _as_energy_tensor(energies)
    if e.dim() == 1:
        e = e.unsqueeze(0)
    return _log_softmax_set(e, mask).exp()


def log_softmax_posterior(energies: Any, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``log p_theta(k | {x_k})`` (log of Eq. (1))."""
    e = energies if isinstance(energies, torch.Tensor) else _as_energy_tensor(energies)
    if e.dim() == 1:
        e = e.unsqueeze(0)
    return _log_softmax_set(e, mask)


def ranking_accuracy(
    positive_energies: Any, negative_energies: Any = None, *, mask: Optional[torch.Tensor] = None
) -> float:
    """Fraction of queries whose positive energy exceeds every negative energy."""
    pos = _as_energy_tensor(positive_energies).reshape(-1)
    if pos.numel() == 0:
        return float("nan")
    if negative_energies is None:
        return float("nan")
    if isinstance(negative_energies, torch.Tensor):
        neg = negative_energies.reshape(pos.numel(), -1)
        if mask is not None:
            neg = neg.masked_fill(~mask.to(torch.bool), float("-inf"))
        worst = neg.max(dim=1).values
        valid = torch.isfinite(worst)
        if not bool(valid.any()):
            return float("nan")
        return float(((pos > worst) & valid).sum().item()) / float(valid.sum().item())
    neg_list = [_as_energy_tensor(n) for n in negative_energies]
    hits = 0
    tot = 0
    for i, n in enumerate(neg_list):
        if n.numel() == 0:
            continue
        tot += 1
        hits += int((pos[i] > n.max()).item())
    return float(hits) / float(tot) if tot else float("nan")


# ---------------------------------------------------------------------------
# Eq. (3): the four gradient terms
# ---------------------------------------------------------------------------
def nce_gradient_terms(
    positive_energies: Any,
    negative_energies: Any = None,
    *,
    alpha: float = 0.0,
    reduction: str = "mean",
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """The four terms of Eq. (3) plus the total objective.

    Returns a dict with (all differentiable w.r.t. ``theta`` via the energies):

    ``neg_data``       ``-E_{y+ ~ p_data}[g_theta(x, y+)]``
    ``reg_pos``        ``alpha * E[g_theta(x, y+)^2]``
    ``neg_model``      ``E_{y- ~ p_theta}[g_theta(x, y-)]`` (unweighted mean over
                       the sampled negatives, as printed in Eq. 3, rescaled by
                       the set size to match the softmax form of Eq. 2)
    ``reg_neg``        ``alpha * E[g_theta(x, y-)^2]``
    ``total``          their sum -- this is the objective of Eq. (3)

    ``neg_model_softmax`` (averaged over queries when reductions are applied) is
    the alternative, exactly-differentiable finite-set estimator
    ``sum_k p_theta(k|set) * g_k`` restricted to the negatives.
    """
    pos = _as_energy_tensor(positive_energies).reshape(-1)
    if pos.numel() == 0:
        z = torch.zeros(())
        return {
            k: z
            for k in (
                "neg_data",
                "reg_pos",
                "neg_model",
                "reg_neg",
                "total",
                "neg_model_softmax",
            )
        }

    if negative_energies is None:
        neg = pos.new_zeros((pos.numel(), 0))
        neg_list: List[torch.Tensor] = [pos.new_zeros(0) for _ in range(pos.numel())]
    elif isinstance(negative_energies, torch.Tensor):
        neg = negative_energies.reshape(pos.numel(), -1)
        neg_list = [neg[i] for i in range(pos.numel())]
    else:
        neg_list = [_as_energy_tensor(n).reshape(-1) for n in negative_energies]
        neg = torch.stack(
            [torch.nn.functional.pad(n, (0, max(0, 1 - n.numel()))) for n in neg_list]
        ) if neg_list else pos.new_zeros((pos.numel(), 0))

    # -E_{p_data}[g(x, y+)]
    neg_data = -_masked_mean(pos)

    # alpha * E[g(x, y+)^2]
    reg_pos = squared_energy_penalty(pos, alpha=alpha, reduction=reduction)

    # E_{p_theta}[g(x, y-)] over the sampled negatives (Eq. 3, unweighted)
    softmax_neg_terms: List[torch.Tensor] = []
    flat_terms: List[torch.Tensor] = []
    for i, n in enumerate(neg_list):
        if n.numel() == 0:
            continue
        flat_terms.append(n.mean())
        set_energies = torch.cat([pos[i : i + 1], n], dim=0) / float(temperature or 1.0)
        p = softmax_posterior(set_energies)[0]
        softmax_neg_terms.append((p[1:] * (n / float(temperature or 1.0))).sum())
    neg_model = torch.stack(flat_terms).mean() if flat_terms else pos.new_zeros(())
    neg_model_softmax = (
        torch.stack(softmax_neg_terms).mean() if softmax_neg_terms else pos.new_zeros(())
    )

    # alpha * E[g(x, y-)^2]
    if isinstance(negative_energies, torch.Tensor) and mask is not None:
        m = mask.to(neg.dtype) if mask.dim() == neg.dim() else mask
        sq = (neg.to(pos.dtype) ** 2) * (m if m.shape == neg.shape else m.unsqueeze(0))
        reg_neg = alpha * (sq.sum() / (m.sum().clamp_min(1.0) if m.shape == neg.shape else neg.numel()))
    elif is_list_negatives(negative_energies):
        reg_neg = squared_energy_penalty(
            torch.cat(neg_list) if neg_list and all(n.numel() for n in neg_list) else pos.new_zeros(0),
            alpha=alpha,
            reduction=reduction,
        )
    else:
        reg_neg = squared_energy_penalty(neg, alpha=alpha, reduction=reduction)

    total = neg_data + reg_pos + neg_model + reg_neg
    return {
        "neg_data": neg_data,
        "reg_pos": reg_pos,
        "neg_model": neg_model,
        "reg_neg": reg_neg,
        "total": total,
        "neg_model_softmax": neg_model_softmax,
    }


def is_list_negatives(negative_energies: Any) -> bool:
    """True when negatives were supplied as a ragged list of tensors."""
    return (
        negative_energies is not None
        and not isinstance(negative_energies, torch.Tensor)
        and isinstance(negative_energies, (list, tuple))
    )


def nce_objective(
    positive_energies: Any,
    negative_energies: Any = None,
    *,
    alpha: float = 0.0,
    reg: Optional[EnergyRegularizer] = None,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
    energy_clamp: Optional[float] = ENERGY_CLAMP_DEFAULT,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Complete Eq. (3) objective: ranking-NCE + ``alpha * E[g^2]`` on both groups.

    When ``reg`` is provided (an :class:`~bbox_adapter.adapter.regularizer.EnergyRegularizer`)
    the ``alpha * E[g^2]`` terms come from that object (so alpha schedules and
    diagnostics stay in one place); otherwise an internal
    :class:`EnergyRegularizer` built from ``alpha`` is used.
    """
    nce = compute_nce_loss(
        positive_energies,
        negative_energies,
        mask=mask,
        reduction=reduction,
        temperature=temperature,
        label_smoothing=label_smoothing,
        energy_clamp=energy_clamp,
    )
    pos = _as_energy_tensor(positive_energies)
    if isinstance(negative_energies, torch.Tensor):
        neg = negative_energies.reshape(-1)
    elif is_list_negatives(negative_energies):
        neg = (
            torch.cat([_as_energy_tensor(n).reshape(-1) for n in negative_energies])
            if len(negative_energies)
            else pos.new_zeros(0)
        )
    else:
        neg = pos.new_zeros(0)

    reg_obj = reg if reg is not None else EnergyRegularizer(alpha=alpha, reduction=reduction)
    if reg is not None:
        reg_term = reg_obj.penalty(pos, neg)
        stats = getattr(reg_obj, "last_stats", {}) or {}
        alpha_eff = float(reg_obj.current_alpha())
    else:
        reg_pos, reg_neg = reg_obj.split_terms(pos, neg)
        reg_term = reg_pos + reg_neg
        alpha_eff = float(alpha)
        stats = {
            "reg_pos": float(reg_pos.detach()) if reg_pos.numel() else 0.0,
            "reg_neg": float(reg_neg.detach()) if reg_neg.numel() else 0.0,
        }

    total = nce + reg_term
    if not return_stats:
        return total
    with torch.no_grad():
        out_stats: Dict[str, float] = {
            "nce": float(nce.detach()),
            "reg": float(reg_term.detach()),
            "alpha": alpha_eff,
            "mean_pos": float(pos.mean().detach()) if pos.numel() else 0.0,
            "mean_neg": float(neg.mean().detach()) if neg.numel() else 0.0,
            "ranking_acc": ranking_accuracy(pos, negative_energies, mask=mask),
            "n_contrastive": float(1 + (neg.numel() / max(1, pos.numel()))),
        }
        out_stats.update({k: float(v) for k, v in stats.items()})
    return total, out_stats


# ---------------------------------------------------------------------------
# configuration / nn.Module wrapper
# ---------------------------------------------------------------------------
@dataclass
class NCELossConfig:
    """Configuration of the ranking-based NCE loss (Eq. 2 + Eq. 3).

    ``alpha`` is the Eq. (3) regularization coefficient (``l_2`` penalty on the
    energies, addendum); ``0.0`` disables it and the regularizer is then the
    responsibility of the caller.  ``temperature`` is a loss-side softmax
    temperature (``1.0`` = exact Eq. (1)); it is unrelated to the black-box
    sampling temperature (which is ``1.0`` for BBox-Adapter runs).
    """

    alpha: float = DEFAULT_ALPHA
    reduction: str = "mean"
    temperature: float = 1.0
    label_smoothing: float = 0.0
    energy_clamp: Optional[float] = ENERGY_CLAMP_DEFAULT
    reg_mode: str = "split"
    reg_warmup_steps: int = 0
    reg_schedule: str = "constant"
    use_regularizer: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError(f"unknown reduction: {self.reduction!r}")
        if self.reg_mode not in ("split", "concat"):
            raise ValueError(f"unknown reg_mode: {self.reg_mode!r}")
        if self.reg_schedule not in ("constant", "inverse"):
            raise ValueError(f"unknown reg_schedule: {self.reg_schedule!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NCELossConfig":
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in dict(d).items() if k in known}
        extra = {k: v for k, v in dict(d).items() if k not in known}
        return cls(extra=extra, **kwargs)


class RankingNCELoss(nn.Module):
    """``nn.Module`` form of Eq. (2) + the Eq. (3) gradient objective.

    The module can either be fed energies directly (``positive_energies`` /
    ``negative_energies``) or -- by passing ``model`` with ``questions`` and the
    candidate answers -- compute the energies itself with
    :class:`~bbox_adapter.adapter.energy_model.EnergyModel`.  In both cases only
    adapter parameters live in the autograd graph.

    The positive entry of each contrastive set is always index 0, so the loss is
    the cross-entropy of the set posterior on the ground-truth sample: the
    gradient raises ``g_theta(x, y+)`` and lowers the negatives, i.e. exactly the
    sign structure of Eq. (3).
    """

    def __init__(
        self,
        config: Optional[NCELossConfig] = None,
        *,
        alpha: Optional[float] = None,
        reg: Optional[EnergyRegularizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config or NCELossConfig()
        if alpha is not None:
            self.config.alpha = float(alpha)
        if kwargs:
            self.config = NCELossConfig.from_dict({**self.config.to_dict(), **kwargs})
        if reg is not None:
            self.regularizer: Optional[EnergyRegularizer] = reg
        elif self.config.use_regularizer:
            self.regularizer = EnergyRegularizer(
                config=RegularizerConfig(
                    alpha=self.config.alpha,
                    mode=self.config.reg_mode,
                    reduction="mean" if self.config.reduction == "none" else self.config.reduction,
                    warmup_steps=self.config.reg_warmup_steps,
                    schedule=self.config.reg_schedule,
                )
            )
        else:
            self.regularizer = None

    # -- energy computation -------------------------------------------------
    def _energies_from_model(
        self,
        model: nn.Module,
        questions: Sequence[str],
        positive_answers: Sequence[str],
        negative_answers: Sequence[Sequence[str]],
        batch_size: int = 64,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return score_contrastive_sets(
            model,
            questions,
            positive_answers,
            negative_answers,
            batch_size=batch_size,
            max_length=max_length,
        )

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        positive_energies: Any = None,
        negative_energies: Any = None,
        mask: Optional[torch.Tensor] = None,
        *,
        model: Optional[nn.Module] = None,
        questions: Optional[Sequence[str]] = None,
        positive_answers: Optional[Sequence[str]] = None,
        negative_answers: Optional[Sequence[Sequence[str]]] = None,
        batch_size: int = 64,
        max_length: Optional[int] = None,
        return_stats: bool = False,
        update_reg_step: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        if positive_energies is None:
            if model is None or questions is None or positive_answers is None:
                raise ValueError(
                    "provide either energies or (model, questions, positive_answers[, negative_answers])"
                )
            negs = negative_answers if negative_answers is not None else [[] for _ in questions]
            positive_energies, negative_energies, mask = self._energies_from_model(
                model, questions, positive_answers, negs, batch_size=batch_size, max_length=max_length
            )

        pos = _as_energy_tensor(positive_energies).reshape(-1)
        neg_t: Optional[Union[torch.Tensor, List[torch.Tensor]]] = negative_energies
        if isinstance(negative_energies, torch.Tensor):
            neg_t = negative_energies.reshape(pos.numel(), -1)
        elif _is_ragged(negative_energies):
            neg_t = [_as_energy_tensor(n).reshape(-1) for n in negative_energies]

        loss = compute_nce_loss(
            pos,
            neg_t,
            mask=mask,
            reduction=self.config.reduction,
            temperature=self.config.temperature,
            label_smoothing=self.config.label_smoothing,
            energy_clamp=self.config.energy_clamp,
        )

        stats: Dict[str, float] = {}
        reg_term: Optional[torch.Tensor] = None
        if self.regularizer is not None:
            flat_neg = None
            if isinstance(neg_t, torch.Tensor):
                flat_neg = neg_t.reshape(-1)
            elif isinstance(neg_t, list):
                flat_neg = (
                    torch.cat(neg_t) if len(neg_t) and all(n.numel() for n in neg_t) else pos.new_zeros(0)
                )
            reg_term, reg_stats = self.regularizer.combine(
                loss, pos, flat_neg, update_step=update_reg_step
            )
            loss = reg_term
            stats.update({k: float(v) for k, v in reg_stats.items()})
        if update_reg_step and self.regularizer is not None:
            self.regularizer.step_end()

        if not return_stats:
            return loss
        with torch.no_grad():
            grad_terms = nce_gradient_terms(
                pos,
                neg_t,
                alpha=float(self.config.alpha),
                reduction=self.config.reduction,
                mask=mask,
                temperature=self.config.temperature,
            )
            stats.update(
                {
                    "loss": float(loss.detach()),
                    "nce": float(
                        compute_nce_loss(
                            pos,
                            neg_t,
                            mask=mask,
                            reduction=self.config.reduction,
                            temperature=self.config.temperature,
                            label_smoothing=self.config.label_smoothing,
                            energy_clamp=self.config.energy_clamp,
                        ).detach()
                    ),
                    "mean_pos": float(pos.mean().detach()) if pos.numel() else 0.0,
                    "mean_neg": float(
                        _as_energy_tensor(neg_t).mean().detach()
                    )
                    if neg_t is not None and _as_energy_tensor(neg_t).numel()
                    else 0.0,
                    "ranking_acc": ranking_accuracy(pos, neg_t, mask=mask),
                    "eq3_neg_data": float(grad_terms["neg_data"].detach()),
                    "eq3_neg_model": float(grad_terms["neg_model"].detach()),
                    "eq3_reg_pos": float(grad_terms["reg_pos"].detach()),
                    "eq3_reg_neg": float(grad_terms["reg_neg"].detach()),
                    "alpha": float(self.config.alpha),
                }
            )
        return loss, stats

    # -- conveniences -------------------------------------------------------
    def alpha(self) -> float:
        return float(self.regularizer.current_alpha()) if self.regularizer else float(self.config.alpha)

    def set_alpha(self, alpha: float) -> None:
        self.config.alpha = float(alpha)
        if self.regularizer is not None:
            self.regularizer.set_alpha(alpha)

    def extra_repr(self) -> str:
        return (
            f"alpha={self.config.alpha}, reduction={self.config.reduction}, "
            f"temperature={self.config.temperature}"
        )


def _is_ragged(negative_energies: Any) -> bool:
    return (
        negative_energies is not None
        and not isinstance(negative_energies, torch.Tensor)
        and isinstance(negative_energies, (list, tuple))
    )


# ---------------------------------------------------------------------------
# alternative (ablation / diagnostic) objectives
# ---------------------------------------------------------------------------
def binary_nce_loss(
    positive_energies: Any,
    negative_energies: Any,
    *,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Conventional (non-ranking) NCE: binary discrimination of real vs noise.

    ``-( log sigma(g_+ - g_-) )`` averaged over positives and negatives.  Kept
    for contrast with the paper's ranking-based variant (Eq. 2), which the paper
    shows to be the appropriate objective (Appendix A/B).
    """
    pos = _as_energy_tensor(positive_energies).reshape(-1)
    neg = _as_energy_tensor(negative_energies).reshape(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return pos.new_zeros(())
    diff = (pos.unsqueeze(1) - neg.unsqueeze(0)) / float(temperature or 1.0)
    loss = -F.logsigmoid(diff)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def pairwise_ranking_loss(
    positive_energies: Any,
    negative_energies: Any,
    *,
    margin: float = 0.0,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Margin ranking surrogate ``max(0, margin - (g_+ - g_-))`` (diagnostic)."""
    pos = _as_energy_tensor(positive_energies).reshape(-1)
    if negative_energies is None:
        return pos.new_zeros(())
    if isinstance(negative_energies, torch.Tensor):
        neg = negative_energies.reshape(pos.numel(), -1)
    else:
        neg = torch.stack(
            [_as_energy_tensor(n).reshape(-1) for n in negative_energies]
        ) if len(negative_energies) else pos.new_zeros((pos.numel(), 0))
    if neg.numel() == 0:
        return pos.new_zeros(())
    diff = (pos.unsqueeze(1) - neg) / float(temperature or 1.0)
    loss = torch.clamp(float(margin) - diff, min=0.0)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


# ---------------------------------------------------------------------------
# building contrastive sets from sampled candidate answers
# ---------------------------------------------------------------------------
def score_contrastive_sets(
    model: nn.Module,
    questions: Sequence[str],
    positive_answers: Sequence[str],
    negative_answers: Sequence[Sequence[str]],
    *,
    batch_size: int = 64,
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one positive + its negatives per query with the adapter.

    Returns ``(positive_energies [B], negative_energies [B, N], mask [B, N])``
    where the tensors keep the autograd graph so that ``.backward()`` updates
    only adapter parameters.  ``N`` is the maximum number of negatives; shorter
    sets are right-padded with ``-inf``-masked zeros.
    """
    b = len(questions)
    if len(positive_answers) != b or len(negative_answers) != b:
        raise ValueError("questions, positive_answers and negative_answers must align")
    n_max = max((len(n) for n in negative_answers), default=0)
    device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")

    pos_list: List[torch.Tensor] = []
    for start in range(0, b, batch_size):
        stop = min(start + batch_size, b)
        q_chunk = list(questions[start:stop])
        a_chunk = list(positive_answers[start:stop])
        energies = model.score_batch(q_chunk, a_chunk, batch_size=batch_size, max_length=max_length)
        pos_list.append(energies.reshape(-1))
    pos = torch.cat(pos_list, dim=0) if pos_list else torch.zeros(0, device=device)

    if n_max == 0:
        return pos, pos.new_zeros((b, 0)), torch.zeros((b, 0), dtype=torch.bool, device=pos.device)

    neg = torch.zeros((b, n_max), dtype=pos.dtype, device=pos.device)
    mask = torch.zeros((b, n_max), dtype=torch.bool, device=pos.device)
    for start in range(0, b, batch_size):
        stop = min(start + batch_size, b)
        flat_q: List[str] = []
        flat_a: List[str] = []
        index: List[Tuple[int, int]] = []
        for i in range(start, stop):
            for j, ans in enumerate(negative_answers[i]):
                flat_q.append(questions[i])
                flat_a.append(ans)
                index.append((i, j))
        if not flat_a:
            continue
        energies = model.score_batch(flat_q, flat_a, batch_size=batch_size, max_length=max_length)
        energies = energies.reshape(-1)
        for (i, j), e in zip(index, energies):
            neg[i, j] = e
            mask[i, j] = True
    return pos, neg, mask


def build_contrastive_tensors(
    positive_energies: Any,
    negative_energies: Any,
    *,
    pad_value: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad ragged per-query energy sets into ``(pos [B], neg [B, N], mask [B, N])``."""
    pos = _as_energy_tensor(positive_energies).reshape(-1)
    if isinstance(negative_energies, torch.Tensor):
        neg = negative_energies
        if neg.dim() == 1:
            neg = neg.unsqueeze(-1)
        mask = torch.ones_like(neg, dtype=torch.bool)
        return pos, neg, mask
    negs = [_as_energy_tensor(n).reshape(-1) for n in (negative_energies or [])]
    b = pos.numel()
    n_max = max((n.numel() for n in negs), default=0)
    neg = torch.full((b, n_max), float(pad_value), dtype=pos.dtype, device=device or pos.device)
    mask = torch.zeros((b, n_max), dtype=torch.bool, device=device or pos.device)
    for i, n in enumerate(negs[:b]):
        if n.numel():
            neg[i, : n.numel()] = n
            mask[i, : n.numel()] = True
    return pos, neg, mask


def nce_loss_from_model(
    model: nn.Module,
    questions: Sequence[str],
    positive_answers: Sequence[str],
    negative_answers: Sequence[Sequence[str]],
    *,
    alpha: float = DEFAULT_ALPHA,
    reg: Optional[EnergyRegularizer] = None,
    batch_size: int = 64,
    max_length: Optional[int] = None,
    return_stats: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """End-to-end convenience: score candidate answers, then apply Eq. (2)/(3)."""
    pos, neg, mask = score_contrastive_sets(
        model,
        questions,
        positive_answers,
        negative_answers,
        batch_size=batch_size,
        max_length=max_length,
    )
    return nce_objective(
        pos,
        neg,
        alpha=alpha,
        reg=reg,
        mask=mask,
        return_stats=return_stats,
    )


def _self_test() -> None:  # pragma: no cover - manual sanity check
    """Small numerical sanity check used during development."""
    pos = torch.tensor([2.0, -1.0], requires_grad=True)
    neg = torch.tensor([[0.0, 0.5], [0.0, -2.0]])
    loss = compute_nce_loss(pos, neg)
    loss.backward()
    # gradient sign structure of Eq. (3): d/dpos < 0, d/dneg > 0
    assert pos.grad[0] < 0 and pos.grad[1] < 0, pos.grad
    assert (neg.grad > 0).all(), neg.grad
    terms = nce_gradient_terms(torch.tensor([2.0]), torch.tensor([[0.0, 0.5]]), alpha=1e-2)
    assert terms["neg_data"] < 0 and terms["reg_pos"] > 0
    assert terms["neg_model"] > 0 and terms["reg_neg"] > 0


if __name__ == "__main__":  # pragma: no cover
    _self_test()
    print("nce.py self-test passed")
