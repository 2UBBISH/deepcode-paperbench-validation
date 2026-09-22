"""Taxonomy-alignment soft loss (Algorithm 1) for *LCA-on-the-Line*.

Paper references
----------------
* Section 4.3.2 ("Using Class Taxonomy as Soft Labels"): the total objective is

      L = lambda * L(CE) + L(soft_lca)

  where ``L(CE)`` is the usual top-1 cross entropy and ``L(soft_lca)`` is an
  auxiliary loss whose target for sample ``i`` is the row
  ``reverse_LCA_matrix[y_i]`` of the (reverse) LCA distance matrix.  The problem
  is treated as multi-label classification.
* Appendix E.2 / Algorithm 1: given an ``n x n`` LCA distance matrix ``M`` with
  ``M[i, k] = D_LCA(i, k)``, it is temperature-scaled and MinMax-normalised,

      M_LCA = MinMax(M ** T),

  then inverted into an alignment indicator

      reverse_LCA_matrix = 1 - M_LCA

  so that the ground-truth index carries the largest value ``1`` (this follows
  from the zero diagonal of a distance matrix).  Algorithm 1 pseudo code::

      function LCA_ALIGNMENT_LOSS(logits, targets, alignment_mode, LCA_matrix,
                                  lambda_weight=0.03)
          reverse_LCA_matrix <- 1 - LCA_matrix
          probs <- softmax(logits, dim=1)
          one_hot_targets <- one_hot(targets)
          standard_loss <- -sum(one_hot_targets * log(probs), dim=1)
          if alignment_mode == 'BCE' then
              criterion <- BCEWithLogitsLoss(reduction='none')
              soft_loss <- mean(criterion(logits, reverse_LCA_matrix[targets]), dim=1)
          else if alignment_mode == 'CE' then
              soft_loss <- -mean(reverse_LCA_matrix[targets] * log(probs), dim=1)
          end if
          total_loss <- lambda_weight * standard_loss + soft_loss
          return mean(total_loss)
      end function

  Main-paper hyper-parameters: ``lambda = 0.03``, ``temperature = 25``, and
  ``CE`` as the soft-loss mode.  A *smaller* lambda scales the standard
  cross-entropy term down; a *large* temperature assigns semantically closer
  classes a larger likelihood and boosts generalisation.

Public interface
----------------
* ``LCAAlignmentLoss``  - ``torch.nn.Module`` implementing Algorithm 1 exactly
  (buffers the reverse LCA matrix, returns ``(loss, parts)`` optionally).
* ``LcaAlignmentLoss``  - alias of the above.
* ``lca_alignment_loss`` - functional form (Algorithm 1 pseudo code directly).
* ``reverse_lca_matrix`` - ``1 - M`` (re-exported / reimplemented).
* ``standard_cross_entropy`` / ``one_hot_encode`` / ``log_softmax`` helpers.
* ``build_alignment_targets`` - gathers ``reverse_LCA_matrix[targets]`` rows.
* ``process_lca_matrix`` - re-exported from ``..hierarchy.lca_matrix`` so a raw
  distance matrix can be turned into ``M_LCA = MinMax(M ** T)`` in one call.
* ``build_soft_loss(cfg)`` - config/factory helper used by the probing script.

Numerical conventions
---------------------
All losses are reduced as ``mean`` over the batch, matching "return
mean(total_loss)" in Algorithm 1.  ``log`` is applied with an epsilon clamp and
implemented via ``log_softmax`` for stability, which is algebraically identical
to ``log(softmax(logits))``.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper defaults (Section 4.3.2 / Appendix E.2)
# ---------------------------------------------------------------------------
DEFAULT_LAMBDA_WEIGHT: float = 0.03
DEFAULT_TEMPERATURE: float = 25.0
DEFAULT_ALIGNMENT_MODE: str = "CE"
DEFAULT_REDUCTION: str = "mean"

ALIGNMENT_MODES: Tuple[str, ...] = ("CE", "BCE")
REDUCTIONS: Tuple[str, ...] = ("mean", "sum", "none")

_ALIGNMENT_MODES_UPPER = {m.upper(): m for m in ALIGNMENT_MODES}

# Filled lazily by ``_ensure_lca_matrix_imports``.
process_lca_matrix = None  # type: ignore[assignment]
_reverse_lca_matrix_fn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Optional / lazy imports
# ---------------------------------------------------------------------------
def _require_torch():
    """Import torch lazily so the module can be inspected without it."""
    try:
        import torch  # noqa: WPS433 (local import on purpose)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "src.alignment.soft_loss requires PyTorch for the loss computation; "
            "install torch to use it."
        ) from exc
    return torch


def _ensure_lca_matrix_imports() -> None:
    """Lazily pull the matrix utilities from :mod:`..hierarchy.lca_matrix`.

    The import is deferred (and tolerant to flat ``src`` layouts) to avoid a
    circular import at package import time.
    """
    global process_lca_matrix, _reverse_lca_matrix_fn
    if process_lca_matrix is not None and _reverse_lca_matrix_fn is not None:
        return

    candidates = (
        "..hierarchy.lca_matrix",
        "hierarchy.lca_matrix",
        "src.hierarchy.lca_matrix",
        "lca_matrix",
    )
    module = None
    if __package__ in (None, ""):
        for root in (os.path.dirname(os.path.dirname(os.path.abspath(__file__))),):
            if root not in sys.path:
                sys.path.insert(0, root)
    for name in candidates:
        try:
            if name.startswith("."):
                from importlib import import_module

                module = import_module(name, __package__)
            else:
                from importlib import import_module

                module = import_module(name)
            break
        except Exception:  # pragma: no cover - fallback path
            module = None
            continue

    if module is not None:
        process_lca_matrix = getattr(module, "process_lca_matrix", None)
        _reverse_lca_matrix_fn = getattr(module, "reverse_lca_matrix", None)

    if process_lca_matrix is None:  # local fallback implementation
        process_lca_matrix = _process_lca_matrix_fallback  # type: ignore[assignment]
    if _reverse_lca_matrix_fn is None:
        _reverse_lca_matrix_fn = _reverse_lca_matrix_fallback  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Reverse LCA matrix (alignment indicator)
# ---------------------------------------------------------------------------
def _reverse_lca_matrix_fallback(lca_matrix: Any) -> Any:
    """``1 - M_LCA`` elementwise (works for nested lists or tensors)."""
    _require_torch()
    import torch

    if torch.is_tensor(lca_matrix):
        return 1.0 - lca_matrix
    return [[1.0 - float(v) for v in row] for row in lca_matrix]


def reverse_lca_matrix(lca_matrix: Any) -> Any:
    """Return the alignment indicator ``reverse_LCA_matrix = 1 - LCA_matrix``.

    The ground-truth entry of every row is ``1`` because the LCA distance
    matrix has a zero diagonal (Section E.2 / Algorithm 1).
    """
    _ensure_lca_matrix_imports()
    try:
        out = _reverse_lca_matrix_fn(lca_matrix)  # type: ignore[misc]
        if out is not None:
            return out
    except Exception:  # pragma: no cover - defensive
        pass
    return _reverse_lca_matrix_fallback(lca_matrix)


def _process_lca_matrix_fallback(
    lca_matrix: Any,
    temperature: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Local ``M_LCA = MinMax(M ** T)`` (used only if the hierarchy package is absent)."""
    _require_torch()
    import torch

    if torch.is_tensor(lca_matrix):
        m = lca_matrix.clamp_min(0.0) ** float(temperature)
        lo, hi = m.min(), m.max()
        if float(hi - lo) < 1e-12:
            return torch.zeros_like(m)
        return (m - lo) / (hi - lo)

    flat = [float(v) for row in lca_matrix for v in row]
    if not flat:
        return []
    m = [max(v, 0.0) ** float(temperature) for v in flat]
    lo, hi = min(m), max(m)
    span = hi - lo
    scaled = [(v - lo) / span if span > 1e-12 else 0.0 for v in m]
    width = len(lca_matrix[0])
    return [scaled[i * width:(i + 1) * width] for i in range(len(lca_matrix))]


def build_reverse_lca_matrix(
    lca_matrix: Any,
    temperature: Optional[float] = None,
    processed: bool = True,
    as_tensor: bool = True,
    device: Any = None,
    dtype: Any = None,
) -> Any:
    """Build the ``reverse_LCA_matrix`` used as a soft-label table.

    Parameters
    ----------
    lca_matrix:
        Either a raw pairwise LCA distance matrix or an already processed
        ``M_LCA = MinMax(M ** T)`` matrix.
    temperature:
        Temperature ``T`` applied as ``M ** T`` before MinMax scaling.  Pass
        ``None`` to skip the processing pipeline (i.e. treat ``lca_matrix`` as
        already scaled).
    processed:
        Set to ``False`` to force re-processing of ``lca_matrix``.
    as_tensor:
        Return a ``torch.Tensor`` (default) instead of nested Python lists.
    """
    torch = _require_torch()
    _ensure_lca_matrix_imports()

    matrix = lca_matrix
    if temperature is not None or not processed:
        temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        matrix = process_lca_matrix(  # type: ignore[misc]
            lca_matrix, temperature=temp, as_tensor=False
        )

    if not torch.is_tensor(matrix):
        matrix = torch.as_tensor([list(r) for r in matrix], dtype=dtype or torch.float32)

    if dtype is not None:
        matrix = matrix.to(dtype=dtype)
    if device is not None:
        matrix = matrix.to(device)

    out = 1.0 - matrix
    if not as_tensor:
        return out.detach().cpu().tolist()
    return out


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def one_hot_encode(targets: Any, num_classes: int, dtype: Any = None) -> Any:
    """One-hot encode ``targets`` with ``num_classes`` output columns."""
    torch = _require_torch()
    targets = targets.long().view(-1)
    if dtype is None:
        dtype = torch.float32
    out = torch.zeros((targets.numel(), int(num_classes)), device=targets.device, dtype=dtype)
    if targets.numel() > 0:
        out.scatter_(1, targets.view(-1, 1), 1.0)
    return out


def log_softmax(logits: Any, dim: int = 1, temperature: float = 1.0) -> Any:
    """Numerically stable ``log softmax`` with optional logit temperature."""
    torch = _require_torch()
    z = logits if float(temperature) == 1.0 else logits / float(temperature)
    return torch.log_softmax(z, dim=dim)


def softmax(logits: Any, dim: int = 1, temperature: float = 1.0) -> Any:
    """Numerically stable ``softmax`` (optionally temperature scaled)."""
    torch = _require_torch()
    z = logits if float(temperature) == 1.0 else logits / float(temperature)
    return torch.softmax(z, dim=dim)


def standard_cross_entropy(
    logits: Any,
    targets: Any,
    weight: Any = None,
    reduction: str = "none",
    label_smoothing: float = 0.0,
) -> Any:
    """Standard top-1 cross entropy, per-sample when ``reduction='none'``.

    This mirrors ``-sum(one_hot_targets * log(probs), dim=1)`` in Algorithm 1
    (implemented through ``log_softmax`` for stability).
    """
    torch = _require_torch()
    import torch.nn.functional as F

    loss = F.cross_entropy(
        logits,
        targets.long().view(-1),
        weight=weight,
        reduction="none",
        label_smoothing=float(label_smoothing),
    )
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def build_alignment_targets(reverse_matrix: Any, targets: Any) -> Any:
    """Gather ``reverse_LCA_matrix[targets]`` -> soft multi-label targets ``(B, K)``."""
    torch = _require_torch()
    if not torch.is_tensor(reverse_matrix):
        reverse_matrix = torch.as_tensor(reverse_matrix, dtype=torch.float32)
    idx = targets.long().view(-1)
    if reverse_matrix.device != idx.device:
        idx = idx.to(reverse_matrix.device)
    return reverse_matrix.index_select(0, idx)


def normalize_mode(mode: str) -> str:
    """Normalise an alignment mode string to ``'CE'`` or ``'BCE'``."""
    key = str(mode).strip().upper()
    if key not in _ALIGNMENT_MODES_UPPER:
        raise ValueError(
            f"Unknown alignment_mode {mode!r}; expected one of {ALIGNMENT_MODES}"
        )
    return _ALIGNMENT_MODES_UPPER[key]


# ---------------------------------------------------------------------------
# Algorithm 1 - functional form
# ---------------------------------------------------------------------------
def lca_alignment_loss(
    logits: Any,
    targets: Any,
    lca_matrix: Any,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    reduction: str = DEFAULT_REDUCTION,
    class_weight: Any = None,
    label_smoothing: float = 0.0,
    temperature: Optional[float] = None,
    processed_matrix: bool = True,
    return_parts: bool = False,
    reverse_matrix: Any = None,
) -> Union[Any, Tuple[Any, Dict[str, Any]]]:
    """Algorithm 1 ``LCA_ALIGNMENT_LOSS`` (Appendix E.2).

    Parameters
    ----------
    logits:
        ``(B, K)`` unnormalised model outputs.
    targets:
        ``(B,)`` ground-truth (canonical ImageNet / hierarchy) class indices.
    lca_matrix:
        ``(K, K)`` ``M_LCA`` matrix.  When ``processed_matrix=False`` the raw
        pairwise distance matrix must be supplied together with
        ``temperature`` so that ``M_LCA = MinMax(M ** T)`` is computed first.
    alignment_mode:
        ``'CE'`` (used in the paper) or ``'BCE'``.
    lambda_weight:
        Weight scaling the standard cross-entropy term (paper: ``0.03``).
    reduction:
        ``'mean'`` (Algorithm 1), ``'sum'`` or ``'none'``.
    temperature:
        Temperature ``T`` from ``M_LCA = MinMax(M ** T)`` (paper: ``25``).
        ``None`` means the supplied matrix is already processed.
    return_parts:
        Also return a dict with ``standard_loss`` / ``soft_loss`` values.

    Returns
    -------
    torch.Tensor or (torch.Tensor, dict)
    """
    torch = _require_torch()
    import torch.nn.functional as F

    mode = normalize_mode(alignment_mode)
    if str(reduction).lower() not in REDUCTIONS:
        raise ValueError(f"Unknown reduction {reduction!r}; expected one of {REDUCTIONS}")

    targets = targets.long().view(-1)
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2-D (B, K); got shape {tuple(logits.shape)}")
    num_classes = int(logits.shape[1])

    # ---- reverse_LCA_matrix <- 1 - LCA_matrix (shape: K x K) --------------
    if reverse_matrix is None:
        if not processed_matrix or temperature is not None:
            reverse_matrix = build_reverse_lca_matrix(
                lca_matrix,
                temperature=temperature,
                processed=processed_matrix,
                as_tensor=True,
                device=logits.device,
                dtype=logits.dtype,
            )
        else:
            reverse_matrix = lca_matrix
    if not torch.is_tensor(reverse_matrix):
        reverse_matrix = torch.as_tensor(reverse_matrix, dtype=logits.dtype)
    reverse_matrix = reverse_matrix.to(device=logits.device, dtype=logits.dtype)
    if reverse_matrix.shape[0] != num_classes:
        # tolerate a matrix larger than the logit head (e.g. partial evaluation)
        reverse_matrix = reverse_matrix[:num_classes, :num_classes]

    # ---- probs <- softmax(logits, dim=1); one hot targets -----------------
    log_probs = log_softmax(logits, dim=1)
    one_hot_targets = one_hot_encode(targets, num_classes, dtype=logits.dtype)

    standard_loss = -(one_hot_targets * log_probs).sum(dim=1)  # (B,)
    if class_weight is not None:
        weight = class_weight if torch.is_tensor(class_weight) else torch.as_tensor(
            class_weight, dtype=logits.dtype
        )
        weight = weight.to(device=logits.device, dtype=logits.dtype).view(-1)
        standard_loss = standard_loss * weight.index_select(0, targets)
    if float(label_smoothing) > 0.0:
        smooth = float(label_smoothing)
        smooth_targets = one_hot_targets * (1.0 - smooth) + smooth / float(num_classes)
        standard_loss = -(smooth_targets * log_probs).sum(dim=1)

    # ---- soft loss --------------------------------------------------------
    soft_targets = build_alignment_targets(reverse_matrix, targets)  # (B, K)
    if mode == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        per_entry = criterion(logits, soft_targets)  # (B, K)
        soft_loss = per_entry.mean(dim=1)  # (B,)
    else:  # CE
        probs = torch.exp(log_probs)
        soft_loss = -(soft_targets * torch.log(probs.clamp_min(1e-12))).mean(dim=1)  # (B,)

    # ---- total_loss <- lambda_weight * standard_loss + soft_loss ----------
    total_loss = float(lambda_weight) * standard_loss + soft_loss

    if reduction == "none":
        out = total_loss
    elif reduction == "sum":
        out = total_loss.sum()
    else:
        out = total_loss.mean()

    if not return_parts:
        return out

    parts: Dict[str, Any] = {
        "standard_loss": standard_loss.detach().mean(),
        "soft_loss": soft_loss.detach().mean(),
        "total_loss": total_loss.detach().mean(),
        "alignment_mode": mode,
        "lambda_weight": float(lambda_weight),
        "temperature": None if temperature is None else float(temperature),
        "reduction": str(reduction).lower(),
        "num_classes": num_classes,
        "batch_size": int(targets.numel()),
    }
    # also provide the plain (unweighted) cross-entropy for monitoring
    parts["ce_loss"] = standard_cross_entropy(logits, targets, reduction="mean").detach()
    return out, parts


# Aliases matching the plan / paper naming
LCA_ALIGNMENT_LOSS = lca_alignment_loss
lca_soft_loss = lca_alignment_loss
taxonomy_alignment_loss = lca_alignment_loss


# ---------------------------------------------------------------------------
# Module-style loss
# ---------------------------------------------------------------------------
class LCAAlignmentLoss:
    """``torch.nn.Module`` implementing Algorithm 1.

    The (processed) LCA distance matrix is stored as a non-persistent buffer and
    inverted once into ``reverse_LCA_matrix = 1 - M_LCA``.

    Examples
    --------
    >>> criterion = LCAAlignmentLoss(lca_matrix, lambda_weight=0.03)
    >>> loss = criterion(logits, targets)                       # total loss
    >>> loss, parts = criterion(logits, targets, return_parts=True)
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        """Build a real ``nn.Module`` subclass lazily (torch optional at import)."""
        torch = _require_torch()
        return super().__new__(cls)

    def __init__(
        self,
        lca_matrix: Any = None,
        lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
        alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
        temperature: Optional[float] = DEFAULT_TEMPERATURE,
        reduction: str = DEFAULT_REDUCTION,
        processed_matrix: bool = False,
        class_weight: Any = None,
        label_smoothing: float = 0.0,
        reverse_matrix: Any = None,
        num_classes: Optional[int] = None,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        torch = _require_torch()
        import torch.nn as nn

        # ``self`` is already an nn.Module because ``__new__`` re-based the class
        # the first time; initialise the Module bookkeeping explicitly.
        if not isinstance(self, nn.Module):
            self.__class__ = type(
                "LCAAlignmentLoss", (nn.Module, LCAAlignmentLoss), {}
            )
        nn.Module.__init__(self)

        self.lambda_weight = float(lambda_weight)
        self.alignment_mode = normalize_mode(alignment_mode)
        self.temperature = None if temperature is None else float(temperature)
        self.reduction = str(reduction).lower()
        self.processed_matrix = bool(processed_matrix)
        self.label_smoothing = float(label_smoothing)

        if reverse_matrix is not None:
            matrix = reverse_matrix if torch.is_tensor(reverse_matrix) else torch.as_tensor(
                reverse_matrix, dtype=dtype or torch.float32
            )
        else:
            if lca_matrix is None:
                raise ValueError("LCAAlignmentLoss requires either lca_matrix or reverse_matrix")
            matrix = build_reverse_lca_matrix(
                lca_matrix,
                temperature=self.temperature,
                processed=self.processed_matrix,
                as_tensor=True,
                device=device,
                dtype=dtype,
            )
        if dtype is not None:
            matrix = matrix.to(dtype=dtype)
        if device is not None:
            matrix = matrix.to(device=device)
        self.register_buffer("reverse_lca_matrix", matrix.to(dtype=torch.float32), persistent=False)

        if class_weight is not None:
            weight = class_weight if torch.is_tensor(class_weight) else torch.as_tensor(
                class_weight, dtype=torch.float32
            )
            self.register_buffer("class_weight", weight.view(-1).to(dtype=torch.float32), persistent=False)
        else:
            self.class_weight = None

        self.num_classes = int(num_classes) if num_classes else int(matrix.shape[0])

    # -- forward ---------------------------------------------------------
    def forward(self, logits: Any, targets: Any, return_parts: bool = False):
        reverse = self.reverse_lca_matrix.to(device=logits.device, dtype=logits.dtype)
        weight = None
        if self.class_weight is not None:
            weight = self.class_weight.to(device=logits.device, dtype=logits.dtype)
        return lca_alignment_loss(
            logits,
            targets,
            lca_matrix=None,
            alignment_mode=self.alignment_mode,
            lambda_weight=self.lambda_weight,
            reduction=self.reduction,
            class_weight=weight,
            label_smoothing=self.label_smoothing,
            temperature=None,
            processed_matrix=True,
            return_parts=return_parts,
            reverse_matrix=reverse,
        )

    # -- convenience -----------------------------------------------------
    def alignment_targets(self, targets: Any) -> Any:
        """Return ``reverse_LCA_matrix[targets]`` (soft multi-label targets)."""
        return build_alignment_targets(self.reverse_lca_matrix, targets)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, alignment_mode={self.alignment_mode!r}, "
            f"lambda_weight={self.lambda_weight}, temperature={self.temperature}, "
            f"reduction={self.reduction!r}"
        )


# ``LcaAlignmentLoss`` alias (plan spelling)
LcaAlignmentLoss = LCAAlignmentLoss


# ---------------------------------------------------------------------------
# Config / factory helper
# ---------------------------------------------------------------------------
@dataclass
class SoftLossConfig:
    """Configuration bundle for the alignment loss (see ``configs/config.yaml``)."""

    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT
    temperature: float = DEFAULT_TEMPERATURE
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE
    reduction: str = DEFAULT_REDUCTION
    processed_matrix: bool = False
    label_smoothing: float = 0.0
    hierarchy: str = "wordnet"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "SoftLossConfig":
        payload = dict(payload or {})
        # tolerate a nested ``soft_loss: {...}`` / ``alignment: {...}`` block
        for key in ("soft_loss", "alignment", "lca_alignment"):
            nested = payload.pop(key, None)
            if isinstance(nested, dict):
                payload.update(nested)
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in payload.items() if k in known}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs, extra=extra)


def build_soft_loss(
    lca_matrix: Any,
    config: Optional[Union[SoftLossConfig, Dict[str, Any]]] = None,
    **overrides: Any,
) -> "LCAAlignmentLoss":
    """Create an :class:`LCAAlignmentLoss` from a config dict / dataclass."""
    if config is None:
        cfg = SoftLossConfig()
    elif isinstance(config, SoftLossConfig):
        cfg = config
    else:
        cfg = SoftLossConfig.from_dict(config)
    kwargs = dict(
        lambda_weight=cfg.lambda_weight,
        temperature=cfg.temperature,
        alignment_mode=cfg.alignment_mode,
        reduction=cfg.reduction,
        processed_matrix=cfg.processed_matrix,
        label_smoothing=cfg.label_smoothing,
    )
    kwargs.update(overrides)
    return LCAAlignmentLoss(lca_matrix, **kwargs)


def wordnet_soft_loss(
    hierarchy: Any,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    **kwargs: Any,
) -> "LCAAlignmentLoss":
    """Convenience factory: build ``M_LCA`` from a ``WordNetHierarchy``."""
    _ensure_lca_matrix_imports()
    matrix = process_lca_matrix(  # type: ignore[misc]
        _raw_matrix_from_hierarchy(hierarchy), temperature=temperature, as_tensor=False
    )
    return LCAAlignmentLoss(
        matrix,
        lambda_weight=lambda_weight,
        alignment_mode=alignment_mode,
        temperature=None,
        processed_matrix=True,
        **kwargs,
    )


def latent_hierarchy_soft_loss(
    latent_hierarchy: Any,
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    temperature: float = DEFAULT_TEMPERATURE,
    alignment_mode: str = DEFAULT_ALIGNMENT_MODE,
    **kwargs: Any,
) -> "LCAAlignmentLoss":
    """Convenience factory: build ``M_LCA`` from a latent (K-means) hierarchy.

    Latent hierarchies store a *similarity* matrix (deeper shared K-means level =
    larger value), so they are inverted with ``max(M) - M`` by
    ``process_lca_matrix(..., latent_hierarchy=True)`` before temperature
    scaling and MinMax normalisation (Appendix E.1/E.2).
    """
    _ensure_lca_matrix_imports()
    matrix = getattr(latent_hierarchy, "latent_lca_matrix", None)
    raw = matrix() if callable(matrix) else _raw_matrix_from_hierarchy(latent_hierarchy)
    processed = process_lca_matrix(  # type: ignore[misc]
        raw, temperature=temperature, latent_hierarchy=True, as_tensor=False
    )
    return LCAAlignmentLoss(
        processed,
        lambda_weight=lambda_weight,
        alignment_mode=alignment_mode,
        temperature=None,
        processed_matrix=True,
        **kwargs,
    )


def _raw_matrix_from_hierarchy(hierarchy: Any) -> Any:
    """Best-effort extraction of the raw pairwise distance matrix of a hierarchy."""
    for attr in ("latent_lca_matrix", "distance_matrix", "pairwise_information_matrix"):
        fn = getattr(hierarchy, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # pragma: no cover - defensive
                continue
    _ensure_lca_matrix_imports()
    from importlib import import_module

    try:
        lca_mod = import_module("..hierarchy.lca", __package__)
    except Exception:  # pragma: no cover
        lca_mod = None
    if lca_mod is not None:
        try:
            return lca_mod.pairwise_lca_matrix(hierarchy)
        except Exception:  # pragma: no cover
            pass
    raise ValueError("Could not extract a raw LCA distance matrix from the given hierarchy")


# ---------------------------------------------------------------------------
# Numpy reference implementation (used by tests without torch)
# ---------------------------------------------------------------------------
def numpy_lca_alignment_loss(
    logits: Sequence[Sequence[float]],
    targets: Sequence[int],
    reverse_matrix: Sequence[Sequence[float]],
    lambda_weight: float = DEFAULT_LAMBDA_WEIGHT,
    alignment_mode: str = "CE",
    reduction: str = "mean",
) -> Union[float, List[float]]:
    """Pure-numpy transcription of Algorithm 1 (used for unit testing)."""
    import numpy as np

    mode = normalize_mode(alignment_mode)
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int64).reshape(-1)
    rev = np.asarray(reverse_matrix, dtype=np.float64)

    z = logits_arr - logits_arr.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs /= np.probs if False else probs.sum(axis=1, keepdims=True)  # keep explicit

    n, k = logits_arr.shape
    one_hot = np.zeros((n, k), dtype=np.float64)
    one_hot[np.arange(n), targets_arr] = 1.0
    standard = -(one_hot * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)

    soft_targets = rev[targets_arr]
    if mode == "CE":
        soft = -(soft_targets * np.log(np.clip(probs, 1e-12, None))).mean(axis=1)
    else:  # BCE-with-logits convention
        sx = 1.0 / (1.0 + np.exp(-logits_arr))
        per_entry = -(soft_targets * np.log(np.clip(sx, 1e-12, None))
                      + (1.0 - soft_targets) * np.log(np.clip(1.0 - sx, 1e-12, None)))
        soft = per_entry.mean(axis=1)

    total = float(lambda_weight) * standard + soft
    if reduction == "none":
        return total.tolist()
    if reduction == "sum":
        return float(total.sum())
    return float(total.mean())


# ---------------------------------------------------------------------------
# Self-test / smoke check
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Sanity check Algorithm 1 on a tiny synthetic problem."""
    logging.basicConfig(level=logging.INFO)
    try:
        torch = _require_torch()
    except ImportError:
        LOG.warning("torch unavailable - running the numpy reference implementation only")
        rev = [[1.0, 0.5, 0.0], [0.5, 1.0, 0.25], [0.0, 0.25, 1.0]]
        loss = numpy_lca_alignment_loss(
            [[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]], [0, 1], rev
        )
        LOG.info("numpy reference loss: %.6f", loss)
        return 0

    # 4 classes, zero-diagonal distance matrix -> reverse has ones on diagonal
    distances = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 0.0, 1.0, 2.0],
            [2.0, 1.0, 0.0, 1.0],
            [3.0, 2.0, 1.0, 0.0],
        ]
    )
    processed = process_lca_matrix(distances.tolist(), temperature=1.0, as_tensor=False)
    rev = reverse_lca_matrix(processed)
    if torch.is_tensor(rev):
        rev_np = rev.numpy()
    else:
        import numpy as np

        rev_np = np.asarray(rev)
    diag = [float(rev_np[i, i]) for i in range(4)]
    assert all(abs(d - 1.0) < 1e-6 for d in diag), f"reverse diagonal must be ones, got {diag}"
    LOG.info("reverse_LCA_matrix diagonal: %s", diag)

    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    criterion = LCAAlignmentLoss(processed, lambda_weight=DEFAULT_LAMBDA_WEIGHT,
                                 temperature=None, processed_matrix=True)
    total, parts = criterion(logits, targets, return_parts=True)
    LOG.info("CE-mode loss parts: standard=%.4f soft=%.4f total=%.4f",
             float(parts["standard_loss"]), float(parts["soft_loss"]), float(parts["total_loss"]))
    assert torch.isfinite(total)

    bce = LCAAlignmentLoss(processed, alignment_mode="BCE", temperature=None,
                           processed_matrix=True)
    bce_loss = bce(logits, targets)
    assert torch.isfinite(bce_loss)
    LOG.info("BCE-mode loss: %.4f", float(bce_loss))

    # reversed (distance) matrix must give consistent results
    manual = numpy_lca_alignment_loss(
        logits.detach().numpy(), targets.detach().numpy(), rev_np, reduction="mean"
    )
    LOG.info("numpy check: %.4f", manual)
    assert abs(manual - float(total)) < 1e-4, (manual, float(total))
    LOG.info("soft_loss self-test passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_self_test())
