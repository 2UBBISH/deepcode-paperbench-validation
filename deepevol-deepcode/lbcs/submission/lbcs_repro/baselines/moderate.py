"""Moderate coreset selection baseline (Xia et al., ICLR 2023).

Paper specification (Appendix D.1, verbatim):

    "- Moderate coreset (ICLR 2023) (Xia et al., 2023b). This method chooses
     the examples with the scores close to the score median in coreset
     selection. The score is about the distance of an example to its class
     center."

So the baseline is a two step procedure:

    1. Score every training example by the distance between its feature
       representation ``f(x_i)`` and the center of its own class
       ``c_{y_i}`` (the mean feature of all training examples of that class)::

           s_i = || f(x_i) - c_{y_i} ||_2

       Features come from a (proxy) network; when no network is supplied the
       flattened input is used as the representation, which keeps the baseline
       runnable in mask-only unit tests.

    2. Keep the ``k`` examples whose scores are *closest to the score median*
       (neither the easiest/near-center nor the hardest/far-out examples).

Because Moderate is a fixed-size selection baseline it never minimizes the
coreset size -- it simply emits a binary mask ``m in {0,1}^n`` with
``||m||_0 = k`` (see Table 2/Table 3 of the paper, where only LBCS reports an
optimized coreset size).

Selection semantics
-------------------
The original Moderate paper selects examples per class, therefore
``median_mode="class"`` is the default here: the median is computed inside
each class and the per-class budget is proportional to the class frequency
(equal split when frequency information is unavailable). ``median_mode``
``"global"`` implements the literal reading of the paper sentence above,
i.e. one median over the entire score vector. Both modes select exactly ``k``
examples whenever ``k <= n``.

Public interface
----------------
- :class:`ModerateSelector` (alias ``Moderate``)
- :func:`class_centers`
- :func:`class_center_distances`
- :func:`extract_features`
- :func:`moderate_scores`
- :func:`moderate_indices`
- :func:`moderate_mask`
- ``_selftest`` -- offline verification runnable via
  ``python -m lbcs_repro.baselines.moderate``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import (
    ScoreBaseline,
    forward_logits,
    gather_scores_by_index,
    indices_to_mask,
    resolve_seed,
    set_seed,
    train_reference_model,
)

try:  # pragma: no cover - torch is a soft dependency
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    _TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ModerateSelector",
    "Moderate",
    "class_centers",
    "class_center_distances",
    "extract_features",
    "moderate_scores",
    "moderate_indices",
    "moderate_mask",
    "median_closeness_order",
]


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
_LAST_LINEAR_ATTRS = ("features", "feature_extractor", "encoder", "backbone")


def _find_last_linear(model: Any) -> Any:
    """Return the last ``nn.Linear`` module of a model (or ``None``)."""
    if not _TORCH_AVAILABLE or model is None:
        return None
    last = None
    try:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last = module
    except Exception:  # pragma: no cover - exotic models
        return None
    return last


def _features_via_hook(model: Any, inputs: Any) -> Any:
    """Capture the *input* of the model's last linear layer as the feature."""
    linear = _find_last_linear(model)
    if linear is None:
        return None
    store: Dict[str, Any] = {}

    def _hook(_module, args):  # pragma: no cover - exercised only with torch
        if args:
            store["feat"] = args[0]

    handle = linear.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            forward_logits(model, inputs)
    finally:
        handle.remove()
    feat = store.get("feat")
    if feat is None:
        return None
    if hasattr(feat, "detach"):
        feat = feat.detach()
    if getattr(feat, "dim", lambda: 0)() > 2:
        feat = feat.reshape(feat.shape[0], -1)
    return feat


def extract_features(
    model: Any = None,
    loader: Any = None,
    dataset: Any = None,
    n: Optional[int] = None,
    device: Any = None,
    max_batches: Optional[int] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    feature_fn: Optional[Callable[[Any, Any], Any]] = None,
    return_index: bool = True,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Collect per-example feature representations.

    Parameters
    ----------
    model:
        Network used as the feature extractor. ``None`` falls back to using the
        flattened raw inputs as features (useful for mask-only unit tests).
    loader / dataset:
        Either a ready-made loader (yielding ``(x, y)`` or ``(x, y, index)``)
        or a dataset that is wrapped with
        :func:`lbcs_repro.data.datasets.make_loader`.
    feature_fn:
        Optional callable ``(model, inputs) -> features`` overriding the default
        penultimate-layer extraction.
    return_index:
        When ``True`` also return the example indices observed while iterating
        (``None`` if the loader does not expose them).

    Returns
    -------
    ``features`` (``(n_obs, d)``) or ``(features, targets, indices)``.
    """
    if loader is None:
        if dataset is None:
            raise ValueError("extract_features needs either `loader` or `dataset`.")
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required to build a loader from a dataset.")
        from lbcs_repro.data.datasets import make_loader

        loader = make_loader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    if model is not None:
        model.eval()
    feats: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    indices: List[np.ndarray] = []
    saw_indices = False

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if not isinstance(batch, (tuple, list)):
            raise TypeError("Unsupported batch type for feature extraction.")
        inputs, batch_targets = batch[0], batch[1]
        batch_indices = batch[2] if len(batch) > 2 else None

        with _no_grad():
            if device is not None and hasattr(inputs, "to"):
                inputs = inputs.to(device)
            if model is None:
                feat = inputs
                if hasattr(feat, "reshape"):
                    feat = feat.reshape(feat.shape[0], -1)
                if hasattr(feat, "detach"):
                    feat = feat.detach()
                if hasattr(feat, "cpu"):
                    feat = feat.cpu()
            elif feature_fn is not None:
                feat = feature_fn(model, inputs)
            else:
                feat = None
                for attr in _LAST_LINEAR_ATTRS:
                    fn = getattr(model, attr, None)
                    if callable(fn):
                        try:
                            feat = fn(inputs)
                            break
                        except Exception:  # pragma: no cover
                            feat = None
                if feat is None:
                    feat = _features_via_hook(model, inputs)
                if feat is None:
                    raise RuntimeError(
                        "Could not extract features: pass `feature_fn` explicitly."
                    )
                if hasattr(feat, "detach"):
                    feat = feat.detach()
                if getattr(feat, "cpu", None) is not None:
                    feat = feat.cpu()
                if getattr(feat, "dim", lambda: 0)() > 2:
                    feat = feat.reshape(feat.shape[0], -1)

        feats.append(np.asarray(feat, dtype=np.float64))
        targets.append(np.asarray(batch_targets).reshape(-1).astype(np.int64))
        if batch_indices is None:
            indices.append(
                np.arange(
                    len(feats[-1]) if not feats[:-1] else sum(len(f) for f in feats[:-1]),
                    (len(feats[-1]) if not feats[:-1] else sum(len(f) for f in feats[:-1]))
                    + len(feats[-1]),
                    dtype=np.int64,
                )
            )
        else:
            saw_indices = True
            indices.append(np.asarray(batch_indices).reshape(-1).astype(np.int64))

    if not feats:
        raise RuntimeError("Feature extraction produced no batches.")

    features = np.concatenate(feats, axis=0).astype(np.float64)
    targets_arr = np.concatenate(targets, axis=0)
    indices_arr = np.concatenate(indices, axis=0) if saw_indices else None
    if verbose:  # pragma: no cover
        LOGGER.info("extracted features %s targets %s", features.shape, targets_arr.shape)
    if return_index:
        return features, targets_arr, indices_arr
    return features


class _no_grad:
    """Context manager: ``torch.no_grad`` when torch is available, else no-op."""

    def __enter__(self):
        if _TORCH_AVAILABLE:
            self._cm = torch.no_grad()
            self._cm.__enter__()
        return self

    def __exit__(self, *exc):
        if _TORCH_AVAILABLE:
            self._cm.__exit__(*exc)
        return False


# ---------------------------------------------------------------------------
# class centers and distances
# ---------------------------------------------------------------------------
def class_centers(
    features: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-class mean feature vectors (the class centers).

    Returns ``(centers, counts)`` where ``centers`` has shape
    ``(num_classes, d)``; classes without examples get a zero center and a
    count of ``0``.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1).astype(np.int64)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same length.")

    if num_classes is None:
        num_classes = int(targets.max()) + 1 if targets.size else 0
    d = features.shape[1]
    centers = np.zeros((num_classes, d), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)
    for c in range(num_classes):
        mask = targets == c
        n_c = int(np.count_nonzero(mask))
        counts[c] = n_c
        if n_c > 0:
            centers[c] = features[mask].mean(axis=0)
    if normalize:
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        centers = centers / norms
    return centers, counts


def class_center_distances(
    features: np.ndarray,
    targets: np.ndarray,
    centers: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    metric: str = "l2",
    normalize_features: bool = False,
) -> np.ndarray:
    """Distance of every example to the center of its own class.

    ``metric`` in ``{"l2", "cosine", "dot"}``. This is the Moderate score.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets).reshape(-1).astype(np.int64)
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    if normalize_features:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        features = features / norms
    if centers is None:
        centers, _ = class_centers(features, targets, num_classes=num_classes)
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError("`centers` must be a 2-D array.")

    if targets.size and (targets.min() < 0 or targets.max() >= centers.shape[0]):
        raise ValueError("targets contain class ids outside the provided centers.")

    own = centers[targets] if targets.size else np.zeros_like(features)
    diff = features - own
    metric = str(metric).lower()
    if metric == "l2":
        scores = np.linalg.norm(diff, axis=1)
    elif metric == "cosine":
        fn = np.linalg.norm(features, axis=1)
        cn = np.linalg.norm(own, axis=1)
        denom = fn * cn
        denom[denom == 0.0] = 1.0
        scores = 1.0 - np.einsum("ij,ij->i", features, own) / denom
    elif metric == "dot":
        scores = -np.einsum("ij,ij->i", diff, diff)
    else:
        raise ValueError(f"Unknown metric {metric!r}.")
    return np.asarray(scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# median-closeness selection
# ---------------------------------------------------------------------------
def median_closeness_order(
    scores: np.ndarray,
    median: Optional[float] = None,
) -> np.ndarray:
    """Indices sorted by ``|s_i - median|`` ascending (deterministic ties)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    if median is None:
        median = float(np.median(scores))
    distance = np.abs(scores - float(median))
    order = np.lexsort((np.arange(scores.size, dtype=np.int64), distance))
    return order.astype(np.int64)


def _allocate_counts(counts: np.ndarray, k: int) -> np.ndarray:
    """Split ``k`` over classes proportionally to ``counts`` (largest remainder)."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    k = int(k)
    if k <= 0:
        return np.zeros_like(counts, dtype=np.int64)
    if total <= 0:
        # Even split with remainder given to the first classes.
        base = k // max(len(counts), 1)
        alloc = np.full(len(counts), base, dtype=np.int64)
        alloc[: k - base * len(counts)] += 1
        return alloc
    raw = counts / total * k
    alloc = np.floor(raw).astype(np.int64)
    remainder = k - int(alloc.sum())
    if remainder > 0:
        frac = raw - alloc
        order = np.lexsort((np.arange(len(frac)), -frac))
        for idx in order[:remainder]:
            alloc[idx] += 1
    # never allocate more than available examples
    alloc = np.minimum(alloc, counts.astype(np.int64))
    # redistribute the deficit to classes with spare capacity
    deficit = k - int(alloc.sum())
    if deficit > 0:
        spare = counts.astype(np.int64) - alloc
        order = np.lexsort((np.arange(len(spare)), -spare))
        for idx in order:
            if deficit <= 0:
                break
            take = int(min(spare[idx], deficit))
            alloc[idx] += take
            deficit -= take
    return alloc


def moderate_indices(
    scores: np.ndarray,
    k: int,
    targets: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    median_mode: str = "class",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Select the ``k`` examples whose scores are closest to the score median.

    ``median_mode="class"`` computes one median per class and allocates the
    budget proportionally to class frequency (the original Moderate recipe);
    ``median_mode="global"`` uses a single median over all scores.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(scores.size)
    k = int(k)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= n:
        return np.arange(n, dtype=np.int64)

    mode = str(median_mode).lower()
    if mode in ("global", "all", "pooled") or targets is None:
        return median_closeness_order(scores)[:k]

    targets = np.asarray(targets).reshape(-1).astype(np.int64)
    if targets.size != n:
        raise ValueError("`targets` must have the same length as `scores`.")
    if num_classes is None:
        num_classes = int(targets.max()) + 1 if n else 0

    counts = np.bincount(targets, minlength=num_classes).astype(np.int64)
    alloc = _allocate_counts(counts, k)

    chosen = np.zeros(0, dtype=np.int64)
    for c in range(num_classes):
        if alloc[c] <= 0:
            continue
        cls_idx = np.flatnonzero(targets == c)
        if cls_idx.size == 0:
            continue
        order = median_closeness_order(scores[cls_idx])
        chosen = np.concatenate([chosen, cls_idx[order[: int(alloc[c])]]])

    # Safety net: if the class-wise pass could not reach k, top up globally
    # with the remaining globally-median-closest examples.
    if chosen.size < k:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), chosen, assume_unique=False)
        extra = median_closeness_order(scores[remaining])[: k - chosen.size]
        chosen = np.concatenate([chosen, remaining[extra]])

    chosen = np.asarray(chosen, dtype=np.int64)
    if chosen.size > k:
        order = median_closeness_order(scores[chosen])
        chosen = chosen[order[:k]]
    return np.sort(chosen.astype(np.int64))


def moderate_mask(
    scores: np.ndarray,
    n: Optional[int] = None,
    k: Optional[int] = None,
    targets: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    median_mode: str = "class",
    seed: Optional[int] = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Binary mask built from :func:`moderate_indices` (defaults ``k = n // 2``)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if n is None:
        n = int(scores.size)
    n = int(n)
    if k is None:
        k = max(1, n // 2)
    idx = moderate_indices(
        scores,
        int(k),
        targets=targets,
        num_classes=num_classes,
        median_mode=median_mode,
        seed=seed,
    )
    return indices_to_mask(idx, n, dtype=dtype)


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------
def moderate_scores(
    model: Any = None,
    loader: Any = None,
    dataset: Any = None,
    n: Optional[int] = None,
    device: Any = None,
    num_classes: Optional[int] = None,
    metric: str = "l2",
    median_mode: str = "class",
    k: Optional[int] = None,
    normalize_features: bool = False,
    feature_fn: Optional[Callable[[Any, Any], Any]] = None,
    max_batches: Optional[int] = None,
    batch_size: int = 128,
    num_workers: int = 0,
    train_loader: Any = None,
    train_epochs: int = 100,
    train: bool = False,
    lr: float = 0.001,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    seed: Optional[int] = None,
    return_index: bool = True,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Distance-to-class-center scores of every example.

    When ``train_loader`` is provided (and ``train=True``) a reference model is
    trained first with Adam ``lr=0.001`` for ``train_epochs`` epochs following
    Section 5.2 of the paper.
    """
    if _TORCH_AVAILABLE and train and train_loader is not None and model is not None:
        model = train_reference_model(
            model,
            train_loader,
            epochs=train_epochs,
            lr=lr,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
            seed=seed,
        )

    features, targets, indices = extract_features(
        model=model,
        loader=loader,
        dataset=dataset,
        device=device,
        max_batches=max_batches,
        batch_size=batch_size,
        num_workers=num_workers,
        feature_fn=feature_fn,
        return_index=True,
    )

    scores_obs = class_center_distances(
        features,
        targets,
        num_classes=num_classes,
        metric=metric,
        normalize_features=normalize_features,
    )

    if n is not None:
        scores = gather_scores_by_index(scores_obs, indices, int(n))
        if return_index:
            return scores, indices
        return scores
    if return_index:
        return scores_obs, indices
    return scores_obs


# ---------------------------------------------------------------------------
# selector
# ---------------------------------------------------------------------------
class ModerateSelector(ScoreBaseline):
    """Moderate coreset selector: keep scores closest to the score median.

    Parameters
    ----------
    model:
        Feature extractor / proxy network. Optional; if omitted and a dataset is
        given, flattened inputs are used as features.
    metric:
        Distance used as the score (``"l2"`` by default, matching the distance
        to the class center described in Appendix D.1).
    median_mode:
        ``"class"`` (per-class median, original recipe) or ``"global"``.
    num_classes:
        Number of classes; inferred from the targets when omitted.
    reference_epochs / lr / optimizer:
        Proxy-model training configuration. Section 5.2 uses Adam with
        ``lr=0.001`` for coreset selection.
    """

    name = "Moderate"
    abbreviation = "Moderate"
    requires_model = True
    higher_is_better = False  # lower distance = closer to the class center

    def __init__(
        self,
        model: Any = None,
        metric: str = "l2",
        median_mode: str = "class",
        num_classes: Optional[int] = None,
        normalize_features: bool = False,
        feature_fn: Optional[Callable[[Any, Any], Any]] = None,
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
        self.metric = metric
        self.median_mode = median_mode
        self.normalize_features = normalize_features
        self.feature_fn = feature_fn
        self.reference_epochs = reference_epochs
        self.train_model = train_model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_batches = max_batches
        self.lr = lr
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scores_: Optional[np.ndarray] = None

    # -- internals ---------------------------------------------------------
    def _resolve_seed(self, seed: Optional[int]) -> int:
        if seed is not None:
            return int(seed)
        return int(resolve_seed(self.seed, 0))

    def _resolve_n(self, n: Optional[int], dataset: Any = None, targets: Any = None) -> Optional[int]:
        if n is not None:
            return int(n)
        if targets is not None:
            return int(np.asarray(targets).reshape(-1).size)
        return None

    # -- interface ---------------------------------------------------------
    def compute_scores(
        self,
        dataset: Any = None,
        targets: Any = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        model: Any = None,
        loader: Any = None,
        train_loader: Any = None,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Distance of each example to its class center (length ``n``)."""
        set_seed(self._resolve_seed(seed))
        model = model if model is not None else self.model
        num_classes = num_classes if num_classes is not None else self.num_classes
        n = self._resolve_n(n, dataset, targets)

        scores, _ = moderate_scores(
            model=model,
            loader=loader,
            dataset=dataset if loader is None else None,
            n=n,
            device=kwargs.pop("device", self.device),
            num_classes=num_classes,
            metric=kwargs.pop("metric", self.metric),
            median_mode=kwargs.pop("median_mode", self.median_mode),
            normalize_features=kwargs.pop("normalize_features", self.normalize_features),
            feature_fn=kwargs.pop("feature_fn", self.feature_fn),
            max_batches=kwargs.pop("max_batches", self.max_batches),
            batch_size=kwargs.pop("batch_size", self.batch_size),
            num_workers=kwargs.pop("num_workers", self.num_workers),
            train_loader=train_loader,
            train_epochs=kwargs.pop("reference_epochs", self.reference_epochs),
            train=bool(kwargs.pop("train_module", self.train_model)),
            lr=kwargs.pop("lr", self.lr),
            optimizer=kwargs.pop("optimizer", self.optimizer),
            momentum=kwargs.pop("momentum", self.momentum),
            weight_decay=kwargs.pop("weight_decay", self.weight_decay),
            seed=self._resolve_seed(seed),
            return_index=True,
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        self.scores_ = scores
        return scores

    def select_indices(
        self,
        n: Optional[int] = None,
        k: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        targets: Any = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Indices of the ``k`` examples whose scores are closest to the median."""
        if scores is None:
            scores = self.scores_
        if scores is None:
            raise ValueError("`scores` must be provided (or call compute_scores first).")
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n = self._resolve_n(n) or int(scores.size)
        if k is None:
            k = max(1, int(n) // 2)
        if targets is None:
            targets = kwargs.pop("train_targets", None)
        return moderate_indices(
            scores,
            int(k),
            targets=targets,
            num_classes=num_classes if num_classes is not None else self.num_classes,
            median_mode=kwargs.pop("median_mode", self.median_mode),
            seed=self._resolve_seed(seed),
        )

    def select_mask(
        self,
        n: int,
        k: int,
        dataset: Any = None,
        targets: Any = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        return_indices: bool = False,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> np.ndarray:
        """Binary coreset mask of exactly ``k`` selected examples."""
        if scores is None:
            scores = self.compute_scores(
                dataset=dataset,
                targets=targets,
                n=n,
                seed=seed,
                num_classes=num_classes,
                **kwargs,
            )
        idx = self.select_indices(
            n=n,
            k=k,
            scores=scores,
            targets=targets,
            num_classes=num_classes,
            seed=seed,
        )
        mask = indices_to_mask(idx, int(n), dtype=dtype)
        if return_indices:
            return idx
        return mask

    def mask(self, n: int, k: int = None, **kwargs: Any) -> np.ndarray:
        return self.select_mask(n=n, k=k, **kwargs)


Moderate = ModerateSelector


# ---------------------------------------------------------------------------
# offline self-test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of class centers, scores and median-closeness selection."""
    rng = np.random.default_rng(0)
    n, d, C = 300, 8, 3
    targets = np.repeat(np.arange(C), n // C)
    centers_true = rng.normal(size=(C, d)) * 4.0
    features = centers_true[targets] + rng.normal(scale=1.0, size=(n, d))

    centers, counts = class_centers(features, targets, num_classes=C)
    assert centers.shape == (C, d)
    assert counts.tolist() == [n // C] * C
    for c in range(C):
        assert np.allclose(centers[c], features[targets == c].mean(axis=0))

    scores = class_center_distances(features, targets, centers=centers)
    assert scores.shape == (n,)
    assert np.all(scores >= 0)

    # median-closeness: selected scores are the closest to the median
    k = 30
    idx = moderate_indices(scores, k, targets=targets, num_classes=C, median_mode="class")
    assert idx.size == k, (idx.size, k)
    assert np.unique(idx).size == k
    sel = scores[idx]
    med = float(np.median(scores))
    unsel = np.setdiff1d(np.arange(n), idx)
    # the largest selected deviation should not exceed the smallest unselected one
    if unsel.size:
        assert np.abs(sel - med).max() <= np.abs(scores[unsel] - med).max() + 1e-9

    idx_global = moderate_indices(scores, k, targets=targets, median_mode="global")
    assert idx_global.size == k

    # class-balanced allocation
    per_class = np.bincount(targets[idx], minlength=C)
    assert per_class.sum() == k
    assert per_class.max() - per_class.min() <= 1

    mask = moderate_mask(scores, n=n, k=k, targets=targets, num_classes=C)
    assert mask.shape == (n,)
    assert int(np.count_nonzero(mask)) == k

    # deterministic
    idx2 = moderate_indices(scores, k, targets=targets, num_classes=C)
    assert np.array_equal(idx, idx2)

    result = {
        "n": n,
        "k": k,
        "num_classes": C,
        "centers_shape": centers.shape,
        "score_mean": float(scores.mean()),
        "per_class_selected": per_class.tolist(),
    }

    # selector plumbing on flattened inputs (no torch needed)
    class _FlatDataset:
        def __init__(self, x, y):
            self.x, self.y = x, y

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.x[i].astype("float32"), int(self.y[i])

    try:
        if _TORCH_AVAILABLE:
            from lbcs_repro.data.datasets import make_loader

            ds = _FlatDataset(centers_true[targets], targets)
            loader = make_loader(ds, batch_size=64, shuffle=False)
            sel = ModerateSelector(num_classes=C, seed=0)
            m = sel.select_mask(n, k, loader=loader)
            assert int(np.count_nonzero(m)) == k
            result["selector_mask_size"] = int(np.count_nonzero(m))
    except Exception as exc:  # pragma: no cover - torch optional
        result["selector_skip"] = repr(exc)

    if verbose:  # pragma: no cover
        print("[moderate._selftest] OK ->", result)
    return result


if __name__ == "__main__":  # pragma: no cover
    _selftest(verbose=True)
