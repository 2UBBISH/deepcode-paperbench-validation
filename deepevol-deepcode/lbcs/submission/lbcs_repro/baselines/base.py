"""Shared infrastructure for the coreset-selection baselines of Section 5.2.

The paper compares LBCS against seven baselines that *construct* a coreset of a
predefined size ``k`` without minimizing that size any further (§5.2, Appendix
D.1):

    (i)   Uniform sampling ("Uniform")
    (ii)  EL2N            (Paul et al., 2021)
    (iii) GraNd           (Paul et al., 2021)
    (iv)  Influential     (Yang et al., 2023)
    (v)   Moderate        (Xia et al., 2023b)
    (vi)  CCS             (Zheng et al., 2023)
    (vii) Probabilistic   (Zhou et al., 2022)

All of them are score-based: an importance score is computed for every training
example, and the coreset is the ``k`` examples with the most favourable scores
(highest scores for EL2N/GraNd/CCS, scores closest to the median for Moderate,
etc.).  This module centralises

* the ``BaselineSelector`` interface shared by every baseline,
* score -> mask conversion helpers (top-k / median-closest / bottom-k),
* deterministic seeding so the "10 repeats" protocol of §5.2 is reproducible,
* small torch helpers (per-sample losses, logits, a generic SGD/Adam trainer)
  used by the score definitions.

The module works without PyTorch installed as long as only the score->mask
helpers are used; model-dependent paths raise a clear error in that case.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # PyTorch is a soft dependency (mask algebra must work without it)
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in torch-free environments
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Subset = None  # type: ignore
    _TORCH_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, Sequence[float]]
MaskLike = Union[np.ndarray, "torch.Tensor", Sequence[float]]


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def _as_numpy(x: Any) -> np.ndarray:
    """Best-effort conversion of masks/tensors/sequences to a NumPy array."""
    if x is None:
        raise ValueError("cannot convert None to a NumPy array")
    if isinstance(x, np.ndarray):
        return x
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):  # pragma: no cover
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_float_scores(scores: ArrayLike) -> np.ndarray:
    """Coerce scores to a 1-D ``float64`` NumPy array."""
    arr = _as_numpy(scores).reshape(-1).astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        # Non-finite scores would sort unpredictably; map them to the worst value.
        finite = arr[np.isfinite(arr)]
        worst = float(np.min(finite)) - 1.0 if finite.size else 0.0
        arr = np.where(np.isfinite(arr), arr, worst)
    return arr


def indices_to_mask(indices: Iterable[int], n: int, dtype: Any = np.float32) -> np.ndarray:
    """Build a binary mask ``m in {0,1}^n`` with ``m_i = 1`` for ``i in indices``."""
    mask = np.zeros(int(n), dtype=dtype)
    idx = np.asarray(list(indices), dtype=np.int64).reshape(-1)
    if idx.size:
        idx = idx[(idx >= 0) & (idx < int(n))]
        mask[idx] = 1
    return mask


def topk_indices(scores: ArrayLike, k: int, largest: bool = True) -> np.ndarray:
    """Indices of the ``k`` largest (or smallest) scores, deterministically."""
    s = to_float_scores(scores)
    k = int(min(max(int(k), 0), s.size))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    # ``np.argsort`` is stable-ish; add a tiny index tie-breaker for determinism.
    order = np.lexsort((np.arange(s.size), -s if largest else s))
    return np.sort(order[:k].astype(np.int64))


def bottomk_indices(scores: ArrayLike, k: int) -> np.ndarray:
    """Indices of the ``k`` smallest scores."""
    return topk_indices(scores, k, largest=False)


def median_indices(scores: ArrayLike, k: int, median: Optional[float] = None) -> np.ndarray:
    """Indices of the ``k`` scores closest to the (per-class or global) median.

    This is the selection rule of Moderate coreset (Xia et al., 2023b): "chooses
    the examples with the scores close to the score median".
    """
    s = to_float_scores(scores)
    k = int(min(max(int(k), 0), s.size))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    med = float(np.median(s)) if median is None else float(median)
    dist = np.abs(s - med)
    order = np.lexsort((np.arange(s.size), dist))
    return np.sort(order[:k].astype(np.int64))


def stratified_topk_indices(
    scores: ArrayLike, targets: ArrayLike, k: int, num_classes: Optional[int] = None, largest: bool = True
) -> np.ndarray:
    """Top-k selection applied per class, then truncated to ``k`` examples.

    Useful for the class-imbalanced setting of §5.3, where a global top-k would
    collapse onto the majority classes.
    """
    s = to_float_scores(scores)
    y = _as_numpy(targets).reshape(-1).astype(np.int64)
    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size else 0
    k = int(min(max(int(k), 0), s.size))
    if k == 0:
        return np.zeros(0, dtype=np.int64)
    per_class = max(k // max(num_classes, 1), 1)
    chosen: List[int] = []
    for c in range(int(num_classes)):
        idx_c = np.flatnonzero(y == c)
        if idx_c.size == 0:
            continue
        local = topk_indices(s[idx_c], min(per_class, idx_c.size), largest=largest)
        chosen.extend(idx_c[local].tolist())
    if len(chosen) < k:  # top up from the global ranking
        already = np.zeros(s.size, dtype=bool)
        already[np.asarray(chosen, dtype=np.int64)] = True
        rest = np.flatnonzero(~already)
        need = k - len(chosen)
        chosen.extend(rest[topk_indices(s[rest], need, largest=largest)].tolist())
    chosen_arr = np.asarray(chosen, dtype=np.int64)
    if chosen_arr.size > k:  # keep the globally best ``k`` of the per-class picks
        keep = topk_indices(s[chosen_arr], k, largest=largest)
        chosen_arr = chosen_arr[keep]
    return np.sort(chosen_arr)


# ---------------------------------------------------------------------------
# torch utilities used by score-based baselines
# ---------------------------------------------------------------------------
def forward_logits(model: Any, inputs: Any) -> Any:
    """Robust forward pass returning raw logits (handles tuples / HF outputs)."""
    out = model(inputs)
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits
    return out


def per_sample_losses(
    model: Any,
    inputs: Any,
    targets: Any,
    criterion: Optional[Any] = None,
    device: Optional[Any] = None,
) -> np.ndarray:
    """Unreduced cross-entropy loss ``l(h(x_i;θ), y_i)`` for one batch."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required for per-sample losses")
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
    targets = targets.to(device) if hasattr(targets, "to") else targets
    logits = forward_logits(model, inputs)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(logits, targets)
    return loss.detach().cpu().numpy().reshape(-1)


def per_sample_grad_norms(
    model: Any,
    inputs: Any,
    targets: Any,
    criterion: Optional[Any] = None,
    device: Optional[Any] = None,
    parameter_filter: Optional[Callable[[str, Any], bool]] = None,
) -> np.ndarray:
    """L2 norm of the loss gradient w.r.t. the (filtered) model parameters.

    This is the GraNd score of Paul et al. (2021): "the data points with larger
    loss gradient norms during training".
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required for GraNd scores")
    if device is None:
        device = next(model.parameters()).device
    inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
    targets = targets.to(device) if hasattr(targets, "to") else targets
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="sum")

    logits = forward_logits(model, inputs)
    loss = criterion(logits, targets)
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in params], allow_unused=True)
    total = 0.0
    for (name, _), g in zip(params, grads):
        if g is None:
            continue
        if parameter_filter is not None and not parameter_filter(name, g):
            continue
        total = total + float(g.detach().pow(2).sum().item())
    return np.asarray(math.sqrt(total), dtype=np.float64)


def normalize_vector(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise rows of ``v``."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)


def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy and (if available) PyTorch deterministically."""
    if seed is None:
        return
    np.random.seed(int(seed))
    if _TORCH_AVAILABLE:  # pragma: no cover
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def resolve_seed(seed: Optional[int], repeat: int, base: int = 0) -> int:
    """Seed used by repeat ``repeat`` of an experiment (deterministic protocol)."""
    return int(base if seed is None else seed) + int(repeat)


# ---------------------------------------------------------------------------
# a minimal proxy trainer (reference model required by several baselines)
# ---------------------------------------------------------------------------
def train_reference_model(
    model: Any,
    loader: Any,
    epochs: int = 10,
    lr: float = 0.001,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Optional[Any] = None,
    criterion: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    log_every: int = 0,
    verbose: bool = False,
) -> Any:
    """Train the proxy model used to compute score-based coresets.

    Defaults follow §5.2: an Adam optimizer with learning rate ``0.001`` is used
    for the inner loop on every benchmark.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to train a reference model")
    if device is None:
        device = next(model.parameters()).device
    model.to(device)
    if optimizer.lower() == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = criterion if criterion is not None else nn.CrossEntropyLoss()
    model.train()
    for ep in range(int(epochs)):
        total, seen = 0.0, 0
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
            targets = targets.to(device) if hasattr(targets, "to") else targets
            opt.zero_grad()
            loss = crit(forward_logits(model, inputs), targets)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(targets.shape[0])
            seen += int(targets.shape[0])
        if scheduler is not None:
            scheduler.step()
        if verbose and log_every and (ep + 1) % int(log_every) == 0:
            LOGGER.info("reference training epoch %d/%d loss=%.4f", ep + 1, epochs, total / max(seen, 1))
    model.eval()
    return model


def collect_predictions(model: Any, loader: Any, device: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Collect ``(logits, targets, indices)`` over a loader.

    ``indices`` is returned when the loader yields ``(x, y, index)`` triples
    (``IndexedDataset`` / ``TransformSubset`` from ``lbcs_repro.data``).
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to collect predictions")
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    logits_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    indices_all: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise ValueError("expected loader to yield (inputs, targets[, indices]) batches")
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
            logits = forward_logits(model, inputs)
            logits_all.append(logits.detach().cpu().numpy())
            targets_all.append(_as_numpy(targets).reshape(-1))
            if len(batch) >= 3 and batch[2] is not None:
                indices_all.append(_as_numpy(batch[2]).reshape(-1).astype(np.int64))
    logits_np = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, 0), dtype=np.float64)
    targets_np = np.concatenate(targets_all, axis=0) if targets_all else np.zeros(0, dtype=np.int64)
    indices_np = np.concatenate(indices_all, axis=0) if indices_all else None
    return logits_np, targets_np, indices_np


def gather_scores_by_index(
    scores: np.ndarray, indices: Optional[np.ndarray], n: int
) -> np.ndarray:
    """Scatter scores observed in loader order back to canonical example order."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    full = np.full(int(n), np.nan, dtype=np.float64)
    if indices is None:
        if scores.size != int(n):
            raise ValueError(f"got {scores.size} scores for n={n}; pass index-yielding loaders")
        return scores.copy()
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    full[idx] = scores[: idx.size]
    if np.isnan(full).any():  # unvisited examples: use the observed mean
        mean_val = float(np.nanmean(scores)) if scores.size else 0.0
        full = np.where(np.isnan(full), mean_val, full)
    return full


# ---------------------------------------------------------------------------
# the baseline interface
# ---------------------------------------------------------------------------
class BaselineSelector(ABC):
    """Common interface of every coreset-selection baseline.

    Subclasses implement :meth:`compute_scores` (a per-example importance score)
    or override :meth:`select` directly when the rule is not a score ranking
    (e.g. Uniform sampling).  ``select_mask`` always returns a binary mask
    ``m in {0,1}^n`` compatible with ``lbcs.masks`` / ``lbcs.discretize``.
    """

    name: str = "baseline"
    abbreviation: str = "baseline"
    requires_model: bool = False
    higher_is_better: bool = True

    def __init__(self, seed: Optional[int] = None, device: Optional[Any] = None, **kwargs: Any) -> None:
        self.seed = seed
        self.device = device
        self.config: Dict[str, Any] = dict(kwargs)

    # -- naming -------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(seed={self.seed})"

    # -- scoring ------------------------------------------------------------
    def compute_scores(
        self,
        model: Optional[Any] = None,
        loader: Optional[Any] = None,
        dataset: Optional[Any] = None,
        targets: Optional[ArrayLike] = None,
        n: Optional[int] = None,
        device: Optional[Any] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Per-example scores in canonical example order (length ``n``)."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement score computation")

    # -- selection ----------------------------------------------------------
    def select(
        self,
        n: int,
        k: int,
        *,
        model: Optional[Any] = None,
        loader: Optional[Any] = None,
        dataset: Optional[Any] = None,
        targets: Optional[ArrayLike] = None,
        device: Optional[Any] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the indices of the ``k`` selected examples."""
        scores = self.compute_scores(
            model=model, loader=loader, dataset=dataset, targets=targets, n=n, device=device, **kwargs
        )
        return topk_indices(scores, k, largest=self.higher_is_better)

    def select_mask(
        self,
        n: int,
        k: int,
        *,
        model: Optional[Any] = None,
        loader: Optional[Any] = None,
        dataset: Optional[Any] = None,
        targets: Optional[ArrayLike] = None,
        device: Optional[Any] = None,
        return_indices: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the binary coreset mask (or ``(mask, indices)``)."""
        idx = self.select(
            n=n, k=k, model=model, loader=loader, dataset=dataset, targets=targets, device=device, **kwargs
        )
        mask = indices_to_mask(idx, n)
        return (mask, idx) if return_indices else mask

    # -- aliases used by the experiment drivers -----------------------------
    def __call__(self, n: int, k: int, **kwargs: Any) -> np.ndarray:
        return self.select_mask(n, k, **kwargs)

    def mask(self, n: int, k: int, **kwargs: Any) -> np.ndarray:
        return self.select_mask(n, k, **kwargs)


class ScoreBaseline(BaselineSelector):
    """Baseline that ranks a per-example score and takes the top ``k``."""

    requires_model = True

    def __init__(self, seed: Optional[int] = None, device: Optional[Any] = None, num_classes: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(seed=seed, device=device, num_classes=num_classes, **kwargs)
        self.num_classes = num_classes


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of the score -> mask helpers."""
    n, k = 50, 12
    scores = np.linspace(0.0, 1.0, n)
    idx = topk_indices(scores, k)
    assert idx.size == k, "top-k must return exactly k indices"
    assert np.all(np.diff(idx) > 0), "top-k indices must be sorted and unique"
    assert idx.min() >= 0 and idx.max() < n
    mask = indices_to_mask(idx, n)
    assert int(mask.sum()) == k, "mask must contain exactly k ones"
    assert np.count_nonzero(mask - indices_to_mask(idx, n)) == 0, "mask construction must be deterministic"

    idx_small = bottomk_indices(scores, k)
    assert idx_small.size == k
    assert np.all(scores[idx_small] <= scores[idx].min() + 1e-12)

    med = median_indices(scores, k)
    assert med.size == k
    assert np.all(np.abs(scores[med] - np.median(scores)) <= np.abs(scores[idx] - np.median(scores)).max() + 1e-12)

    y = np.arange(n) % 2
    strat = stratified_topk_indices(scores, y, k, num_classes=2)
    assert strat.size == k, "stratified selection must return exactly k indices"

    x = np.array([[1.0, 0.0], [3.0, 4.0]])
    xn = normalize_vector(x)
    assert np.allclose(np.linalg.norm(xn, axis=1), 1.0), "rows must be unit norm"

    info = {
        "n": n,
        "k": k,
        "topk": idx.tolist(),
        "mask_size": int(mask.sum()),
        "median_size": int(med.size),
        "stratified_size": int(strat.size),
        "torch_available": bool(_TORCH_AVAILABLE),
    }
    if verbose:  # pragma: no cover - diagnostics only
        LOGGER.info("baseline self-test passed: %s", info)
        print("baseline self-test passed:", info)
    return info


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest()
