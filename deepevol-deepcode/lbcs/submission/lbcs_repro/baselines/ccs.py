"""CCS (Coverage-centric Coreset Selection) baseline -- Zheng et al., ICLR 2023.

Appendix D.1 of the paper describes the method as:

    "CCS (ICLR 2023) (Zheng et al., 2023). The method proposes a novel one-shot
     coreset selection method that jointly considers overall data coverage upon a
     distribution as well as the importance of each example."

Implementation
--------------
The selector is a plain ``ScoreBaseline`` so that it can be swapped with
Uniform / EL2N / GraNd / Influential / Moderate inside the experiment drivers
(Table 2, Table 3, Figure 2).  Because CCS is *not* a pure score-ranking method,
the actual selection is performed by :meth:`CCSSelector.select_indices`, which

1. preserves the data distribution by allocating the budget ``k`` across classes
   in proportion to their frequency (largest-remainder rounding), and
2. inside every class, runs a one-shot *coverage + importance* greedy pass: the
   first centre is the most important example, and each following centre
   maximises a convex combination

       ``alpha * coverage_gain + (1 - alpha) * importance``

   where ``coverage_gain`` is the (normalised) distance to the already selected
   centres (a k-center / farthest-point criterion) and ``importance`` is an
   EL2N- or feature-norm style per-example score.

``compute_scores`` still returns a length-``n`` importance vector so that the
generic ``ScoreBaseline`` machinery (top-k masks, diagnostics) keeps working.

The module has a soft PyTorch dependency: with PyTorch available a proxy model
supplies features/importances; without it, callers can pass precomputed
``features``/``importance`` arrays and the selection logic still runs (used by
the offline self-test).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import (
    ScoreBaseline,
    forward_logits,
    indices_to_mask,
    normalize_vector,
    per_sample_losses,
    resolve_seed,
    set_seed,
    train_reference_model,
)

try:  # pragma: no cover - optional dependency
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CCSSelector",
    "CCS",
    "normalize01",
    "allocate_class_budget",
    "preserved_class_budgets",
    "min_distance_to_selected",
    "kcenter_greedy",
    "ccs_select_indices",
    "ccs_indices",
    "ccs_mask",
    "ccs_scores",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def normalize01(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Min-max normalise finite entries of ``values`` into ``[0, 1]``.

    Non-finite entries (used internally as "already selected" markers) map to
    ``0.0``.  A constant vector maps to all zeros.
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    out = np.zeros_like(v)
    finite = np.isfinite(v)
    if not finite.any():
        return out
    lo = float(v[finite].min())
    hi = float(v[finite].max())
    if hi - lo <= eps:
        return out
    out[finite] = (v[finite] - lo) / (hi - lo)
    return out


def allocate_class_budget(
    counts: np.ndarray,
    k: int,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Allocate ``k`` slots across classes proportionally to ``counts``.

    Uses largest-remainder rounding so that ``budgets.sum() == min(k, counts.sum())``.
    """
    counts = np.asarray(counts, dtype=np.float64).reshape(-1)
    if num_classes is not None:
        counts = counts[: int(num_classes)]
    total = float(counts.sum())
    k = int(max(0, k))
    if counts.size == 0 or total <= 0 or k == 0:
        return np.zeros(counts.size, dtype=np.int64)

    k_eff = int(min(k, counts.sum()))
    exact = counts / total * k_eff
    budgets = np.floor(exact).astype(np.int64)
    remainder = k_eff - int(budgets.sum())
    if remainder > 0:
        frac = exact - budgets
        for idx in np.argsort(-frac, kind="stable"):
            if remainder == 0:
                break
            if budgets[idx] < counts[idx]:
                budgets[idx] += 1
                remainder -= 1
        # if some classes are saturated, top up the largest classes
        guard = 0
        while remainder > 0 and guard < counts.size:
            progressed = False
            for idx in np.argsort(-counts, kind="stable"):
                if remainder == 0:
                    break
                if budgets[idx] < counts[idx]:
                    budgets[idx] += 1
                    remainder -= 1
                    progressed = True
            if not progressed:
                break
            guard += 1
    budgets = np.minimum(budgets, counts.astype(np.int64))
    return budgets


def preserved_class_budgets(
    targets: np.ndarray,
    k: int,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Class budgets proportional to the empirical class distribution."""
    targets = np.asarray(targets).astype(np.int64).reshape(-1)
    if num_classes is None:
        num_classes = int(targets.max()) + 1 if targets.size else 0
    if targets.size:
        counts = np.bincount(targets, minlength=int(num_classes))[: int(num_classes)]
    else:
        counts = np.zeros(int(num_classes), dtype=np.int64)
    return allocate_class_budget(counts, k, num_classes=int(num_classes))


def min_distance_to_selected(
    features: np.ndarray,
    selected: Sequence[int],
    metric: str = "euclidean",
) -> np.ndarray:
    """Distance of every row to the closest already-selected centre.

    Selected rows receive ``-inf`` so that they are never picked twice.
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    n = feats.shape[0]
    dists = np.full(n, np.inf, dtype=np.float64)
    if metric == "cosine":
        normed = normalize_vector(feats)
    else:
        normed = feats
    for idx in selected:
        idx = int(idx)
        if not (0 <= idx < n):
            continue
        if metric == "cosine":
            d = 1.0 - normed @ normed[idx]
        else:
            d = np.linalg.norm(normed - normed[idx], axis=1)
        dists = np.minimum(dists, d)
    for idx in selected:
        idx = int(idx)
        if 0 <= idx < n:
            dists[idx] = -np.inf
    return dists


def kcenter_greedy(
    features: np.ndarray,
    k: int,
    initial: Optional[Sequence[int]] = None,
    importance: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    metric: str = "euclidean",
) -> np.ndarray:
    """One-shot greedy coverage + importance selection (farthest-point style).

    Parameters
    ----------
    features:
        ``(n, d)`` feature matrix (rows already L2-normalised when cosine
        matching is desired).
    k:
        Number of centres to pick (clipped to ``n``).
    initial:
        Optional seed centre(s); by default the most important example is used.
    importance:
        Per-example importance scores (higher = more important).  ``None`` means
        pure coverage.
    alpha:
        Weight of coverage relative to importance, ``alpha=1`` -> pure k-center,
        ``alpha=0`` -> pure importance ranking.
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    n = feats.shape[0]
    if n == 0 or k <= 0:
        return np.zeros(0, dtype=np.int64)
    k = int(min(k, n))
    if importance is None:
        imp = np.zeros(n, dtype=np.float64)
    else:
        imp = np.asarray(importance, dtype=np.float64).reshape(-1)
        if imp.size != n:
            raise ValueError(f"importance has length {imp.size}, expected {n}")
    imp_n = normalize01(imp)

    selected: List[int] = []
    if initial:
        for idx in initial:
            idx = int(idx)
            if 0 <= idx < n and idx not in selected:
                selected.append(idx)
            if len(selected) >= k:
                break
    if not selected:
        # start from the most important example (ties -> smallest index)
        selected.append(int(np.argmax(imp_n)))

    dists = min_distance_to_selected(feats, selected, metric=metric)
    while len(selected) < k:
        cov_n = normalize01(dists)
        combined = float(alpha) * cov_n + (1.0 - float(alpha)) * imp_n
        combined[np.asarray(selected, dtype=np.int64)] = -np.inf
        nxt = int(np.argmax(combined))
        if not np.isfinite(combined[nxt]):
            # only fully saturated rows remain
            remaining = [i for i in range(n) if i not in selected]
            if not remaining:
                break
            nxt = int(remaining[0])
        selected.append(nxt)
        if metric == "cosine":
            normed = normalize_vector(feats)
            d = 1.0 - normed @ normed[nxt]
        else:
            d = np.linalg.norm(feats - feats[nxt], axis=1)
        dists = np.minimum(dists, d)
        dists[nxt] = -np.inf
    return np.array(sorted(selected), dtype=np.int64)


# ---------------------------------------------------------------------------
# selection API
# ---------------------------------------------------------------------------
def ccs_select_indices(
    features: Optional[np.ndarray] = None,
    targets: Optional[np.ndarray] = None,
    k: Optional[int] = None,
    n: Optional[int] = None,
    num_classes: Optional[int] = None,
    importance: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    preserve_distribution: bool = True,
    normalize_features: bool = True,
    metric: str = "euclidean",
    seed: Optional[int] = 0,
    return_details: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """One-shot coverage + importance coreset selection.

    Returns the selected example indices (sorted) of size ``k``.
    """
    if features is None:
        raise ValueError("ccs_select_indices requires a `features` matrix.")
    raw = np.asarray(features, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    n_feat = raw.shape[0]
    n = int(n_feat if n is None else n)
    if n != n_feat:
        raise ValueError(f"n={n} does not match features rows={n_feat}")
    k = int(n if k is None else k)
    k = int(min(max(k, 0), n))
    if k == 0:
        empty = np.zeros(0, dtype=np.int64)
        return (empty, {"budgets": np.zeros(0, dtype=np.int64)}) if return_details else empty

    feats = normalize_vector(raw) if normalize_features else raw.copy()

    if importance is None:
        importance = np.linalg.norm(raw, axis=1)
    imp = np.asarray(importance, dtype=np.float64).reshape(-1)
    if imp.size != n:
        imp = np.resize(imp, n) if imp.size else np.zeros(n)

    if targets is None:
        targets = np.zeros(n, dtype=np.int64)
    targets = np.asarray(targets).astype(np.int64).reshape(-1)[:n]
    if num_classes is None:
        num_classes = int(targets.max()) + 1 if targets.size else 1

    if preserve_distribution:
        budgets = preserved_class_budgets(targets, k, num_classes=int(num_classes))
    else:
        budgets = allocate_class_budget(
            np.ones(int(num_classes), dtype=np.float64), k, num_classes=int(num_classes)
        )

    selected: List[int] = []
    per_class: Dict[int, List[int]] = {}
    for cls in range(int(num_classes)):
        cls_idx = np.flatnonzero(targets == cls)
        budget = int(min(budgets[cls], cls_idx.size)) if cls < budgets.size else 0
        if budget <= 0 or cls_idx.size == 0:
            per_class[cls] = []
            continue
        local = kcenter_greedy(
            feats[cls_idx],
            budget,
            importance=imp[cls_idx],
            alpha=alpha,
            metric=metric,
        )
        chosen = cls_idx[local].tolist()
        per_class[cls] = chosen
        selected.extend(chosen)

    # Safety net: recover any shortfall (saturated classes) by global importance.
    if len(selected) < k:
        chosen_set = set(selected)
        for idx in np.argsort(-imp, kind="stable"):
            if len(selected) >= k:
                break
            if int(idx) not in chosen_set:
                selected.append(int(idx))
                chosen_set.add(int(idx))

    selected_arr = np.array(sorted(selected[:k]), dtype=np.int64)
    if return_details:
        return selected_arr, {"budgets": budgets, "per_class": per_class}
    return selected_arr


def ccs_indices(
    features: Optional[np.ndarray] = None,
    k: Optional[int] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper around :func:`ccs_select_indices` (positional-friendly)."""
    return np.asarray(ccs_select_indices(features=features, k=k, **kwargs))


def ccs_mask(
    features: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    k: Optional[int] = None,
    dtype: Any = np.float32,
    **kwargs: Any,
) -> np.ndarray:
    """Binary coreset mask ``m in {0,1}^n`` selecting ``k`` examples."""
    if features is None:
        raise ValueError("ccs_mask requires a `features` matrix.")
    n_feat = np.asarray(features).shape[0]
    n = int(n_feat if n is None else n)
    indices = ccs_select_indices(features=features, k=k, n=n, **kwargs)
    return indices_to_mask(indices, n, dtype=dtype)


# ---------------------------------------------------------------------------
# selector class
# ---------------------------------------------------------------------------
class CCSSelector(ScoreBaseline):
    """Coverage-centric Coreset Selection (Zheng et al., ICLR 2023).

    The selector keeps a fixed coreset size ``k`` (like all baselines, it never
    minimizes the size by optimization), jointly trading off distributional
    coverage against per-example importance.
    """

    name = "CCS"
    abbreviation = "CCS"
    requires_model = True
    higher_is_better = True

    def __init__(
        self,
        model: Any = None,
        importance: str = "l2",
        alpha: float = 0.5,
        preserve_distribution: bool = True,
        normalize_features: bool = True,
        metric: str = "euclidean",
        num_classes: Optional[int] = None,
        feature_fn: Any = None,
        reference_epochs: int = 100,
        train_model: bool = False,
        batch_size: int = 128,
        num_workers: int = 0,
        max_batches: Optional[int] = None,
        lr: float = 0.001,
        optimizer: str = "adam",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        device: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, device=device, num_classes=num_classes, **kwargs)
        self.model = model
        self.importance = importance
        self.alpha = float(alpha)
        self.preserve_distribution = bool(preserve_distribution)
        self.normalize_features = bool(normalize_features)
        self.metric = metric
        self.feature_fn = feature_fn
        self.reference_epochs = int(reference_epochs)
        self.train_model = bool(train_model)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.max_batches = max_batches
        self.lr = float(lr)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self._last_features: Optional[np.ndarray] = None
        self._last_scores: Optional[np.ndarray] = None
        self._last_targets: Optional[np.ndarray] = None

    # -- internal helpers -------------------------------------------------
    def _seed_for(self, seed: Optional[int]) -> int:
        base = self.seed if self.seed is not None else 0
        return int(resolve_seed(base, 0 if seed is None else int(seed), base=0))

    def _resolve_n(self, n: Optional[int], dataset: Any = None, targets: Any = None) -> int:
        if n is not None:
            return int(n)
        if targets is not None:
            return int(len(targets))
        if dataset is not None and hasattr(dataset, "__len__"):
            return int(len(dataset))
        if self._last_features is not None:
            return int(self._last_features.shape[0])
        raise ValueError("CCSSelector could not infer `n`; pass n/dataset/targets.")

    def _resolve_targets(
        self, targets: Optional[np.ndarray], dataset: Any = None, n: Optional[int] = None
    ) -> Optional[np.ndarray]:
        if targets is not None:
            return np.asarray(targets).astype(np.int64).reshape(-1)
        if dataset is not None:
            try:
                from lbcs_repro.data.datasets import get_targets

                return np.asarray(get_targets(dataset)).astype(np.int64).reshape(-1)
            except Exception:  # pragma: no cover
                if hasattr(dataset, "targets"):
                    return np.asarray(dataset.targets).astype(np.int64).reshape(-1)
        return None

    def _extract_features(
        self,
        loader: Any = None,
        dataset: Any = None,
        model: Any = None,
        n: Optional[int] = None,
    ) -> np.ndarray:
        """Collect per-example features (delegates to ``moderate.extract_features``)."""
        feats = None
        if loader is not None or dataset is not None:
            try:
                from .moderate import extract_features

                feats, _tgts, _idx = extract_features(
                    model=model,
                    loader=loader,
                    dataset=dataset if loader is None else None,
                    n=n,
                    device=self.device,
                    max_batches=self.max_batches,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    feature_fn=self.feature_fn,
                    return_index=True,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("CCS feature extraction failed (%s); using proxies", exc)
                feats = None
        if feats is None and model is not None and loader is not None and _TORCH_AVAILABLE:
            # fallback: first linear layer output is unavailable -> flatten logits
            collected: List[np.ndarray] = []
            model.eval()
            with torch.no_grad():
                for batch in loader:
                    inputs = batch[0]
                    inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
                    logits = forward_logits(model, inputs)
                    collected.append(logits.detach().cpu().numpy())
            if collected:
                feats = np.concatenate(collected, axis=0).astype(np.float64)
        if feats is None:
            raise ValueError(
                "CCSSelector could not extract features; pass `features` explicitly."
            )
        return np.asarray(feats, dtype=np.float64)

    def _importance_scores(
        self,
        features: np.ndarray,
        loader: Any = None,
        model: Any = None,
    ) -> np.ndarray:
        """Per-example importance in the sense of CCS."""
        feats = np.asarray(features, dtype=np.float64)
        mode = str(self.importance).lower()
        if mode in ("none", "null"):
            return np.ones(feats.shape[0], dtype=np.float64)
        if mode in ("loss", "el2n") and loader is not None and model is not None:
            if not _TORCH_AVAILABLE:
                raise RuntimeError("loss/EL2N importance requires PyTorch.")
            model.eval()
            collected: List[np.ndarray] = []
            with torch.no_grad():
                for batch in loader:
                    inputs, targets_b = batch[0], batch[1]
                    inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
                    targets_b = (
                        targets_b.to(self.device) if hasattr(targets_b, "to") else targets_b
                    )
                    if mode == "loss":
                        collected.append(
                            per_sample_losses(model, inputs, targets_b, device=self.device)
                        )
                    else:  # EL2N-style error norm
                        logits = forward_logits(model, inputs)
                        probs = torch.softmax(logits, dim=-1)
                        onehot = torch.zeros_like(probs)
                        onehot.scatter_(1, targets_b.view(-1, 1).long(), 1.0)
                        err = (probs - onehot).detach().cpu().numpy()
                        collected.append(
                            np.linalg.norm(err.reshape(err.shape[0], -1), axis=1)
                        )
            if collected:
                return np.concatenate(collected).astype(np.float64)
        # default: feature norm ("l2" importance)
        return np.linalg.norm(feats, axis=1)

    # -- public API -------------------------------------------------------
    def compute_scores(
        self,
        dataset: Any = None,
        targets: Optional[np.ndarray] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        model: Any = None,
        loader: Any = None,
        train_loader: Any = None,
        num_classes: Optional[int] = None,
        features: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Length-``n`` importance scores (higher = more important).

        Side effect: caches the extracted features on the instance so that
        :meth:`select_indices` can reuse them without a second forward pass.
        """
        n = self._resolve_n(n, dataset=dataset, targets=targets)
        model = model if model is not None else self.model
        targets_arr = self._resolve_targets(targets, dataset=dataset, n=n)

        # optionally train the proxy model first (shared across baselines)
        if self.train_model and train_loader is not None and model is not None:
            set_seed(self._seed_for(seed))
            model = train_reference_model(
                model,
                train_loader,
                epochs=self.reference_epochs,
                lr=self.lr,
                optimizer=self.optimizer,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                device=self.device,
            )
            self.model = model

        if features is None:
            features = self._extract_features(
                loader=loader, dataset=dataset, model=model, n=n
            )
        feats = np.asarray(features, dtype=np.float64)
        if feats.ndim == 1:
            feats = feats.reshape(-1, 1)
        if feats.shape[0] != n:
            LOGGER.warning(
                "feature rows (%d) != n (%d); truncating/padding with zeros",
                feats.shape[0],
                n,
            )
            fixed = np.zeros((n, feats.shape[1]), dtype=np.float64)
            m = min(n, feats.shape[0])
            fixed[:m] = feats[:m]
            feats = fixed

        scores = self._importance_scores(feats, loader=loader, model=model)
        self._last_features = feats
        self._last_targets = targets_arr
        if scores.size != n:
            scores = np.resize(scores, n) if scores.size else np.zeros(n)
        self._last_scores = np.asarray(scores, dtype=np.float64)
        return self._last_scores

    def select_indices(
        self,
        n: Optional[int] = None,
        k: Optional[int] = None,
        dataset: Any = None,
        targets: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the indices of ``k`` chosen examples."""
        n = self._resolve_n(n, dataset=dataset, targets=targets)
        k = int(k) if k is not None else int(n // 2)
        targets_arr = self._resolve_targets(targets, dataset=dataset, n=n)
        if targets_arr is None:
            targets_arr = self._last_targets
        if targets_arr is None:
            targets_arr = np.zeros(n, dtype=np.int64)

        feats = features if features is not None else self._last_features
        imp = scores if scores is not None else self._last_scores
        if feats is None or imp is None:
            imp = self.compute_scores(
                dataset=dataset,
                targets=targets_arr,
                n=n,
                seed=seed,
                num_classes=num_classes,
                features=features,
                **kwargs,
            )
            feats = self._last_features
        if feats is None:
            raise ValueError("CCSSelector.select_indices needs features.")

        return ccs_select_indices(
            features=np.asarray(feats, dtype=np.float64),
            targets=targets_arr,
            k=k,
            n=n,
            num_classes=num_classes if num_classes is not None else self.num_classes,
            importance=np.asarray(imp, dtype=np.float64),
            alpha=self.alpha if alpha is None else float(alpha),
            preserve_distribution=self.preserve_distribution,
            normalize_features=self.normalize_features,
            metric=self.metric,
            seed=seed,
        )

    def select_mask(
        self,
        n: Optional[int] = None,
        k: Optional[int] = None,
        dataset: Any = None,
        targets: Optional[np.ndarray] = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        return_indices: bool = False,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> Any:
        n = self._resolve_n(n, dataset=dataset, targets=targets)
        indices = self.select_indices(
            n=n,
            k=k,
            dataset=dataset,
            targets=targets,
            num_classes=num_classes,
            seed=seed,
            scores=scores,
            features=features,
            **kwargs,
        )
        mask = indices_to_mask(indices, n, dtype=dtype)
        if return_indices:
            return mask, indices
        return mask

    def mask(self, n: Optional[int] = None, k: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """Convenience alias for :meth:`select_mask`."""
        return self.select_mask(n=n, k=k, **kwargs)


# Alias used by the experiment drivers / aggregator.
CCS = CCSSelector


def ccs_scores(
    model: Any = None,
    loader: Any = None,
    dataset: Any = None,
    features: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    num_classes: Optional[int] = None,
    importance: str = "l2",
    alpha: float = 0.5,
    preserve_distribution: bool = True,
    normalize_features: bool = True,
    metric: str = "euclidean",
    train_loader: Any = None,
    train: bool = False,
    reference_epochs: int = 100,
    lr: float = 0.001,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    num_workers: int = 0,
    max_batches: Optional[int] = None,
    device: Any = None,
    seed: Optional[int] = 0,
    return_index: bool = True,
    **kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Functional entry point: importance scores (CCS's score component).

    Kept for API parity with ``el2n_scores`` / ``grand_scores``.  Use
    :func:`ccs_indices` / :func:`ccs_mask` for the full coverage-aware selection.
    """
    selector = CCSSelector(
        model=model,
        importance=importance,
        alpha=alpha,
        preserve_distribution=preserve_distribution,
        normalize_features=normalize_features,
        metric=metric,
        num_classes=num_classes,
        reference_epochs=reference_epochs,
        train_model=bool(train),
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=max_batches,
        lr=lr,
        optimizer=optimizer,
        momentum=momentum,
        weight_decay=weight_decay,
        device=device,
        seed=seed,
    )
    if features is None and loader is None and dataset is None and train_loader is not None:
        loader = train_loader
    scores = selector.compute_scores(
        dataset=dataset,
        n=n,
        seed=seed,
        model=model,
        loader=loader,
        train_loader=train_loader,
        num_classes=num_classes,
        features=features,
    )
    if return_index:
        return scores, None
    return scores


# ---------------------------------------------------------------------------
# offline self-test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Deterministic checks for the CCS selection logic (no model required)."""
    rng = np.random.default_rng(0)
    n_per_class = 60
    num_classes = 3
    dim = 8
    centers = rng.normal(size=(num_classes, dim)) * 4.0
    feats = np.vstack(
        [
            centers[c] + rng.normal(scale=1.0, size=(n_per_class, dim))
            for c in range(num_classes)
        ]
    )
    targets = np.repeat(np.arange(num_classes), n_per_class)
    n = feats.shape[0]

    out: Dict[str, Any] = {}
    k = 45

    # 1) budget allocation is exact and proportional
    budgets = preserved_class_budgets(targets, k, num_classes=num_classes)
    out["budgets"] = budgets.tolist()
    assert int(budgets.sum()) == k, budgets
    assert budgets.shape[0] == num_classes

    idx = ccs_select_indices(
        features=feats, targets=targets, k=k, n=n, num_classes=num_classes, seed=0
    )
    out["num_selected"] = int(idx.size)
    assert idx.size == k, idx.size
    assert len(set(idx.tolist())) == k, "duplicate indices selected"

    # 2) class distribution preserved
    sel_counts = np.bincount(targets[idx], minlength=num_classes)
    out["selected_class_counts"] = sel_counts.tolist()
    assert np.all(np.abs(sel_counts - budgets) <= 1), sel_counts

    # 3) determinism
    idx2 = ccs_select_indices(
        features=feats, targets=targets, k=k, n=n, num_classes=num_classes, seed=0
    )
    assert np.array_equal(idx, idx2), "selection is not deterministic"
    out["deterministic"] = True

    # 4) alpha=1 is pure coverage, alpha=0 pure importance ranking
    imp = np.linalg.norm(feats, axis=1)
    idx_cov = ccs_select_indices(
        features=feats,
        targets=targets,
        k=k,
        n=n,
        num_classes=num_classes,
        importance=imp,
        alpha=1.0,
        seed=0,
    )
    assert idx_cov.size == k
    out["alpha1_size"] = int(idx_cov.size)

    idx_imp = ccs_select_indices(
        features=feats,
        targets=targets,
        k=k,
        n=n,
        num_classes=num_classes,
        importance=imp,
        alpha=0.0,
        seed=0,
    )
    for c in range(num_classes):
        cls = np.flatnonzero(targets == c)
        b = int(budgets[c])
        chosen_c = [i for i in idx_imp.tolist() if targets[i] == c]
        top_c = cls[np.argsort(-imp[cls], kind="stable")[:b]].tolist()
        assert set(chosen_c) == set(top_c), (c, chosen_c, top_c)
    out["pure_importance_ok"] = True

    # 5) coverage: k-center picks spread-out points
    def mean_pairwise(sel: np.ndarray) -> float:
        pts = feats[sel]
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        iu = np.triu_indices(pts.shape[0], k=1)
        return float(d[iu].mean()) if iu[0].size else 0.0

    random_idx = rng.choice(n, size=k, replace=False)
    out["mean_pairwise_coverage"] = mean_pairwise(idx_cov)
    out["mean_pairwise_random"] = mean_pairwise(random_idx)
    assert out["mean_pairwise_coverage"] > out["mean_pairwise_random"], out

    # 6) mask construction
    mask = ccs_mask(
        features=feats, targets=targets, k=k, n=n, num_classes=num_classes, seed=0
    )
    out["mask_size"] = int(mask.sum())
    assert mask.shape[0] == n
    assert int(mask.sum()) == k
    assert set(np.unique(mask).tolist()) <= {0.0, 1.0}

    # 7) normalize01 / allocate_class_budget edge cases
    assert np.allclose(normalize01(np.array([5.0, 5.0, 5.0])), 0.0)
    assert np.allclose(normalize01(np.array([0.0, 1.0, 2.0])), [0.0, 0.5, 1.0])
    assert int(allocate_class_budget(np.array([10, 1]), 3).sum()) == 3
    assert np.array_equal(allocate_class_budget(np.array([0, 0]), 4), [0, 0])
    out["edge_cases_ok"] = True

    if verbose:
        LOGGER.info("CCS _selftest passed: %s", out)
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
    print("CCS baseline self-test passed.")
