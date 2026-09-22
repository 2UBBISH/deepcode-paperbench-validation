"""EL2N coreset selection baseline (Paul et al., 2021) -- Appendix D.1.

Paper text (Appendix D.1):

    "EL2N (NeurIPS 2021) (Paul et al., 2021). The method involves the data
     points with larger norms of the error vector that is the predicted class
     probabilities minus one-hot label encoding."

The baseline therefore computes, for every training example ``(x_i, y_i)`` and a
(partially) trained proxy model ``h(.; theta)``, the error vector

    e_i = softmax(h(x_i; theta)) - onehot(y_i)          in R^C

and scores the example by its L2 norm

    s_i = || e_i ||_2 .

Examples with the *largest* scores are kept (``higher_is_better = True``), and
exactly ``k`` of them form the coreset (a binary mask ``m in {0,1}^n`` with
``||m||_0 = k``), matching the "construct the coreset with a predetermined
coreset size" protocol of Section 5.2.

Reference implementation: https://github.com/mansheej/data_diet
(the paper's footnote 2), where EL2N scores are accumulated over the first few
epochs of training and averaged per example.  This module supports both modes:

* ``n_train_epochs=1`` (default here) -- score once with the reference model;
* ``n_train_epochs>1`` -- average the per-epoch scores, following data_diet.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .base import (
    ScoreBaseline,
    collect_predictions,
    gather_scores_by_index,
    indices_to_mask,
    set_seed,
    topk_indices,
    train_reference_model,
)

try:  # PyTorch is a soft dependency (mask-only unit tests must run without it)
    import torch
    import torch.nn.functional as F  # noqa: F401  (kept for parity/extensions)

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

__all__ = [
    "EL2NSelector",
    "EL2N",
    "el2n_scores",
    "el2n_scores_from_logits",
    "error_vectors",
    "el2n_indices",
    "el2n_mask",
]


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def _is_torch(obj: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(obj, torch.Tensor)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-300, None)


def _one_hot(targets: np.ndarray, num_classes: Optional[int]) -> np.ndarray:
    """One-hot encoding of integer targets; ``C`` is inferred when unknown."""
    targets = np.asarray(targets).astype(np.int64).ravel()
    if num_classes is None or int(num_classes) <= 0:
        num_classes = int(targets.max()) + 1 if targets.size else 1
    num_classes = int(num_classes)
    out = np.zeros((targets.size, num_classes), dtype=np.float64)
    valid = (targets >= 0) & (targets < num_classes)
    out[np.arange(targets.size)[valid], targets[valid]] = 1.0
    return out


def error_vectors(
    logits: Union[np.ndarray, Any],
    targets: Union[np.ndarray, Sequence[int]],
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """``e_i = softmax(logits_i) - onehot(y_i)`` for a batch/set of examples.

    Args:
        logits: array/tensor of shape ``(n, C)``.
        targets: integer labels of shape ``(n,)``.
        num_classes: number of classes ``C`` (inferred when omitted).

    Returns:
        Array of shape ``(n, C)`` holding the per-example error vectors.
    """
    logits_np = logits.detach().cpu().numpy() if _is_torch(logits) else np.asarray(logits)
    logits_np = np.asarray(logits_np, dtype=np.float64)
    if logits_np.ndim == 1:
        logits_np = logits_np.reshape(1, -1)
    if num_classes is None:
        num_classes = int(logits_np.shape[1])
    probs = _softmax(logits_np)
    return probs - _one_hot(np.asarray(targets), num_classes)


def el2n_scores_from_logits(
    logits: Union[np.ndarray, Any],
    targets: Union[np.ndarray, Sequence[int]],
    num_classes: Optional[int] = None,
    reduction: str = "norm",
) -> np.ndarray:
    """EL2N score ``|| softmax(h(x)) - onehot(y) ||_2`` from precomputed logits.

    ``reduction="norm"`` (default) is the paper's rule; ``"mean"``/``"sum"``
    are provided for data_diet-style variants that aggregate the absolute error
    vector instead of taking its L2 norm.
    """
    err = error_vectors(logits, targets, num_classes=num_classes)
    if reduction == "norm":
        return np.sqrt(np.sum(err ** 2, axis=1))
    if reduction == "mean":
        return np.mean(np.abs(err), axis=1)
    if reduction == "sum":
        return np.sum(np.abs(err), axis=1)
    raise ValueError(f"unknown reduction '{reduction}'")


def el2n_scores(
    model: Any,
    loader: Any = None,
    num_classes: Optional[int] = None,
    device: Any = None,
    epochs: int = 1,
    optimizer: Any = "adam",
    train_loader: Any = None,
    train_epochs: int = 100,
    lr: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    reduction: str = "norm",
    return_index: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Compute per-example EL2N scores with a proxy model.

    Args:
        model: proxy network delivering logits for ``loader``.
        loader: loader over the candidate examples (yielding ``(x, y)`` or
            ``(x, y, index)``).
        epochs: number of scoring passes; ``>1`` averages the per-epoch scores
            (data_diet recipe), else the paper's single-pass EL2N is used.
        optimizer / train_loader / train_epochs / lr / momentum / weight_decay:
            when ``train_loader`` is given, the proxy model is first trained on
            the *full* training set with the Section 5.2 inner-loop recipe
            (Adam, lr ``0.001``) for ``train_epochs`` epochs.
        return_index: when the loader yields indices, also return them so the
            caller can scatter scores back to canonical example order.

    Returns:
        ``scores`` (float64 array), optionally with the observed indices.
    """
    if train_loader is not None:
        model = train_reference_model(
            model,
            train_loader,
            epochs=max(1, int(train_epochs)),
            lr=float(lr),
            optimizer=optimizer if isinstance(optimizer, str) else "adam",
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            device=device,
            scheduler=None,
            verbose=False,
        )

    if loader is None:
        raise ValueError("el2n_scores requires either `loader` or `train_loader`")

    num_passes = max(1, int(epochs))
    accumulated: Optional[np.ndarray] = None
    observed_idx: Optional[np.ndarray] = None

    for _ in range(num_passes):
        logits, targets, idx = collect_predictions(model, loader, device=device)
        scores = el2n_scores_from_logits(
            logits, targets, num_classes=num_classes, reduction=reduction
        )
        if accumulated is None:
            accumulated = np.zeros_like(scores, dtype=np.float64)
            observed_idx = idx if idx is not None else np.arange(scores.size)
        accumulated += scores

    scores = accumulated / float(num_passes)

    if return_index:
        return scores, observed_idx
    return scores


def el2n_scores_batch(
    model: Any,
    loader: Any,
    num_classes: Optional[int] = None,
    device: Any = None,
    reduction: str = "norm",
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-pass EL2N scores plus the corresponding labels (loader order)."""
    logits, targets, _ = collect_predictions(model, loader, device=device)
    scores = el2n_scores_from_logits(
        logits, targets, num_classes=num_classes, reduction=reduction
    )
    return np.asarray(scores, dtype=np.float64), np.asarray(targets)


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def el2n_indices(
    n: int,
    k: int,
    scores: Union[np.ndarray, Sequence[float]],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Indices of the ``k`` largest EL2N scores (deterministic tie-breaking)."""
    _ = seed  # ranking is deterministic; kept for API symmetry with other baselines
    return topk_indices(np.asarray(scores, dtype=np.float64).ravel(), int(k), largest=True)


def el2n_mask(
    n: int,
    k: int,
    scores: Union[np.ndarray, Sequence[float]],
    seed: Optional[int] = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Binary coreset mask selecting the ``k`` largest EL2N scores."""
    return indices_to_mask(el2n_indices(n, k, scores, seed=seed), int(n), dtype=dtype)


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------
def _looks_like_dataloader(obj: Any) -> bool:
    if obj is None:
        return False
    if _TORCH_AVAILABLE and isinstance(obj, torch.utils.data.DataLoader):
        return True
    return hasattr(obj, "dataset") and hasattr(obj, "batch_size")


def _looks_like_dataset(obj: Any) -> bool:
    if obj is None:
        return False
    if _TORCH_AVAILABLE and isinstance(obj, torch.utils.data.Dataset):
        return True
    return (
        hasattr(obj, "__getitem__")
        and hasattr(obj, "__len__")
        and not _looks_like_dataloader(obj)
    )


def _make_loader(dataset: Any, batch_size: int = 128, num_workers: int = 0) -> Any:
    """Build a DataLoader through :mod:`lbcs_repro.data` (falls back to torch)."""
    try:
        from lbcs_repro.data.datasets import make_loader as _ml

        return _ml(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    except Exception:  # pragma: no cover - fallback path
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to build a DataLoader for EL2N")
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------
class EL2NSelector(ScoreBaseline):
    """EL2N coreset selection: keep the ``k`` examples with the largest error norms.

    Scoring uses the predicted class probabilities minus the one-hot label
    (Appendix D.1).  The proxy model is either supplied directly or trained on
    the fly with the Section 5.2 inner-loop recipe (Adam, lr ``0.001``).
    """

    name = "EL2N"
    abbreviation = "EL2N"
    requires_model = True
    higher_is_better = True

    def __init__(
        self,
        model: Any = None,
        reference_epochs: int = 100,
        n_train_epochs: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        batch_size: int = 128,
        reduction: str = "norm",
        num_classes: Optional[int] = None,
        device: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: proxy network; if ``None`` the driver supplies one via
                ``compute_scores(..., model=...)``.
            reference_epochs: epochs used to train the proxy model on the full
                training set before scoring (Section 5.2 inner-loop recipe).
            n_train_epochs: number of scoring passes whose scores are averaged
                (data_diet style).  ``1`` = single-pass EL2N.
            lr / optimizer / momentum / weight_decay / batch_size: reference
                training hyper-parameters (Section 5.2 uses Adam, lr 0.001).
            reduction: ``"norm"`` (paper) or ``"mean"``/``"sum"`` variants.
            num_classes: number of classes (inferred when possible).
            device: torch device.
        """
        super().__init__(seed=seed, device=device, num_classes=num_classes, **kwargs)
        self.model = model
        self.reference_epochs = int(reference_epochs)
        self.n_train_epochs = int(n_train_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.optimizer = str(optimizer)
        self.momentum = float(momentum)
        self.batch_size = int(batch_size)
        self.reduction = str(reduction)

    # -- scoring -----------------------------------------------------------
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
        """Return per-example EL2N scores of length ``n`` when known.

        ``loader``/``dataset`` may be a ready DataLoader, a torch ``Dataset``
        (a DataLoader is derived) or a loader yielding ``(x, y, index)`` triples.
        """
        set_seed(self._resolve_seed(seed))

        net = model if model is not None else self.model
        if net is None:
            raise ValueError(
                "EL2NSelector requires a proxy model; pass `model=` to "
                "compute_scores/select_mask or set it at construction."
            )

        resolved_loader = self._resolve_loader(loader=loader, dataset=dataset)
        n_examples = self._resolve_n(n=n, dataset=dataset, targets=targets, loader=resolved_loader)
        classes = num_classes if num_classes is not None else self.num_classes

        scores, idx = el2n_scores(
            net,
            loader=resolved_loader,
            num_classes=classes,
            device=self.device,
            epochs=self.n_train_epochs,
            optimizer=self.optimizer,
            train_loader=train_loader,
            train_epochs=self.reference_epochs,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            reduction=self.reduction,
            return_index=True,
        )
        scores = np.asarray(scores, dtype=np.float64).ravel()

        if n_examples is None:
            n_examples = (
                int(idx.max()) + 1 if idx is not None and np.asarray(idx).size else int(scores.size)
            )
        if n_examples > scores.size:
            scores = gather_scores_by_index(scores, idx, int(n_examples))

        return scores

    # -- helpers -----------------------------------------------------------
    def _resolve_seed(self, seed: Optional[int]) -> int:
        return int(seed if seed is not None else (self.seed if self.seed is not None else 0))

    def _resolve_loader(self, loader: Any, dataset: Any) -> Any:
        if loader is not None:
            if _looks_like_dataloader(loader):
                return loader
            return _make_loader(loader, batch_size=self.batch_size)
        if dataset is None:
            raise ValueError("EL2NSelector needs a `loader` or a `dataset`")
        if _looks_like_dataloader(dataset):
            return dataset
        return _make_loader(dataset, batch_size=self.batch_size)

    def _resolve_n(
        self,
        n: Optional[int],
        dataset: Any = None,
        targets: Any = None,
        loader: Any = None,
    ) -> Optional[int]:
        if n is not None:
            return int(n)
        if targets is not None:
            try:
                return int(np.asarray(targets).ravel().size)
            except Exception:  # pragma: no cover
                pass
        for obj in (dataset, loader):
            if obj is None:
                continue
            try:
                length = len(obj)  # type: ignore[arg-type]
            except Exception:  # pragma: no cover
                continue
            if isinstance(length, int) and length > 0:
                return int(length)
        return None


# Alias ---------------------------------------------------------------------
EL2N = EL2NSelector


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline sanity checks for the EL2N score/selection algebra."""
    results: Dict[str, Any] = {}

    # Handmade logits: row 0 perfectly correct, row 2 confidently wrong.
    logits = np.array(
        [
            [10.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 10.0],
        ]
    )
    targets = np.array([0, 1, 1])
    scores = el2n_scores_from_logits(logits, targets, num_classes=3)
    results["scores"] = [round(float(s), 4) for s in scores]
    results["ordering_ok"] = bool(scores[2] > scores[1] > scores[0] - 1e-9)

    err = error_vectors(logits, targets, num_classes=3)
    results["error_shape_ok"] = err.shape == (3, 3)
    # correct confident prediction -> e ~ (-1, 0, 0), norm ~ 1
    results["correct_row_norm"] = round(float(np.linalg.norm(err[0])), 4)
    # confident wrong prediction -> e ~ (0, -1, 1), norm ~ sqrt(2)
    results["wrong_row_norm"] = round(float(np.linalg.norm(err[2])), 4)
    results["norms_ok"] = bool(abs(np.linalg.norm(err[0]) - 1.0) < 1e-2)

    m = el2n_mask(n=3, k=1, scores=scores)
    results["mask_ok"] = bool(np.count_nonzero(m) == 1 and m[2] == 1.0)
    idx = el2n_indices(n=3, k=2, scores=scores)
    results["indices_ok"] = bool(sorted(np.asarray(idx).tolist()) == [1, 2])

    # Selector end-to-end through a stub model producing fixed logits.
    if _TORCH_AVAILABLE:
        class _Stub(torch.nn.Module):
            def forward(self, x):  # noqa: D102
                return logits.clone()

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.zeros(3, 1), torch.tensor(targets, dtype=torch.long)
            ),
            batch_size=3,
        )
        sel = EL2NSelector(model=_Stub(), num_classes=3, seed=0)
        scores2 = sel.compute_scores(loader=loader, n=3)
        results["selector_scores_ok"] = bool(np.allclose(scores2, scores, atol=1e-6))
        mask = sel.select_mask(n=3, k=1, loader=loader)
        results["selector_mask_ok"] = bool(np.count_nonzero(mask) == 1 and mask[2] == 1.0)

    results["all_ok"] = bool(
        results["ordering_ok"]
        and results["error_shape_ok"]
        and results["norms_ok"]
        and results["mask_ok"]
        and results["indices_ok"]
        and results.get("selector_scores_ok", True)
        and results.get("selector_mask_ok", True)
    )

    if verbose:
        for key, value in results.items():
            print(f"  {key}: {value}")
    return results


if __name__ == "__main__":  # pragma: no cover
    print("EL2N self-test")
    _selftest()
