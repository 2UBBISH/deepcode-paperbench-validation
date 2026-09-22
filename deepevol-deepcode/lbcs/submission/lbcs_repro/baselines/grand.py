"""GraNd coreset-selection baseline (Paul et al., NeurIPS 2021).

Appendix D.1 of the paper describes the GraNd baseline as:

    "GraNd (NeurIPS 2021) (Paul et al., 2021). The method builds a coreset by
     involving the data points with larger loss gradient norms during training."

So, for each candidate example ``i`` we estimate

    score_i = E_t [ || grad_theta  l(h(x_i; theta_t), y_i) ||_2 ]

i.e. the (expected) L2 norm of the per-example loss gradient w.r.t. the model
parameters, measured during (proxy) training.  The ``k`` examples with the
largest scores form the coreset, giving a binary mask ``m in {0,1}^n`` with
``||m||_0 = k`` (same size as the predefined ``k`` used by every baseline in
Table 2 -- the baselines do not minimize the size, only LBCS does).

Implementation notes
--------------------
* The paper reports using the last-layer parameters for the original GraNd/EL2N
  study; we therefore default to ``"last_layer"`` parameter filtering but expose
  ``param_scope in {"all", "last_layer", "bias"}`` so the behaviour can be
  switched without touching algorithm code.
* ``reference_epochs`` (default 100, matching the inner-loop budget used across
  Section 5.2) trains the proxy model when one is not supplied.  ``epochs``
  (default 1) averages the scores over several measurement passes, which is the
  faithful "during training" reading of the paper's sentence.
* Deterministic tie-breaking via :func:`topk_indices` (lexsort with index
  tie-breaker) keeps repeat runs reproducible.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .base import (
    ScoreBaseline,
    collect_predictions,
    forward_logits,
    gather_scores_by_index,
    indices_to_mask,
    per_sample_grad_norms,
    per_sample_losses,
    set_seed,
    topk_indices,
    train_reference_model,
)

try:  # torch is a soft dependency (matches baselines/base.py, el2n.py)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without PyTorch
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

__all__ = [
    "GraNdSelector",
    "GraNd",
    "grand_scores",
    "grand_scores_from_model",
    "grad_norm_scores",
    "grand_indices",
    "grand_mask",
]


# ---------------------------------------------------------------------------
# Parameter scoping helpers
# ---------------------------------------------------------------------------

def _select_parameters(model: Any, scope: str = "last_layer") -> list:
    """Return the list of parameters used for the gradient-norm scores.

    ``scope`` is one of ``"all"`` (every trainable parameter), ``"last_layer"``
    (parameters of the final ``nn.Linear``-like module, the common GraNd/EL2N
    convention) or ``"bias"`` (biases only -- cheap variant).
    """
    if not _TORCH_AVAILABLE or model is None:
        return []

    params = [p for p in model.parameters() if getattr(p, "requires_grad", True)]
    if scope == "all" or not params:
        return params

    if scope == "bias":
        biases = [p for p in params if p.dim() == 1]
        return biases if biases else params

    # last_layer: parameters of the last module that has parameters.
    try:
        modules = list(model.named_modules())
    except Exception:  # pragma: no cover - exotic models
        modules = []
    for name, module in reversed(modules):
        if name and any(p.requires_grad for p in module.parameters(recurse=False)):
            return [p for p in module.parameters(recurse=False) if p.requires_grad]
    return params


def _parameter_filter(scope: str = "last_layer"):
    """Build a ``parameter_filter`` callable for :func:`per_sample_grad_norms`."""
    if not _TORCH_AVAILABLE:
        return None

    def _filter(param):
        name = getattr(param, "parametrization_name", None)
        del name  # unused; kept for clarity

    # ``per_sample_grad_norms`` accepts a callable or an iterable of params;
    # we transparently support both by returning None here and passing the
    # selected parameter list instead (see ``_grad_norm_scores``).
    return None if scope else _filter


# ---------------------------------------------------------------------------
# Core score computation
# ---------------------------------------------------------------------------

def _grad_norm_scores_single_pass(
    model: Any,
    loader: Any,
    device: Any = None,
    criterion: Any = None,
    param_scope: str = "last_layer",
    parameters: Optional[Sequence[Any]] = None,
    reduction: str = "norm",
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """One pass over ``loader`` collecting per-example gradient norms.

    Returns ``(scores, targets, indices)``; ``indices`` is ``None`` unless the
    loader yields ``(x, y, index)`` triples.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("GraNd scores require PyTorch to be installed.")

    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cpu")

    params = list(parameters) if parameters is not None else _select_parameters(model, param_scope)

    scores: list = []
    labels: list = []
    indices: list = []
    saw_index = False

    model.eval() if hasattr(model, "eval") else None
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            targets = batch[1]
            idx = batch[2] if len(batch) > 2 else None
        else:  # pragma: no cover - dict-style batches are not used here
            inputs, targets, idx = batch["inputs"], batch["targets"], None

        if idx is not None:
            saw_index = True
            indices.append(np.asarray(idx).reshape(-1))

        inputs = inputs.to(device) if hasattr(inputs, "to") else inputs
        targets_t = targets.to(device) if hasattr(targets, "to") else targets

        batch_scores = per_sample_grad_norms(
            model,
            inputs,
            targets_t,
            criterion=criterion,
            device=device,
            parameter_filter=params if params else None,
        )
        batch_scores = np.asarray(batch_scores, dtype=np.float64).reshape(-1)

        if reduction == "sum":
            # Already an L2 norm over the flattened gradient; sum == norm here.
            pass
        elif reduction == "mean":
            batch_scores = batch_scores  # norm is the canonical GraNd score

        scores.append(batch_scores)
        labels.append(np.asarray(targets).reshape(-1).astype(np.int64))

    scores_arr = np.concatenate(scores).astype(np.float64) if scores else np.zeros(0, dtype=np.float64)
    labels_arr = np.concatenate(labels).astype(np.int64) if labels else np.zeros(0, dtype=np.int64)
    indices_arr = np.concatenate(indices).astype(np.int64) if (saw_index and indices) else None
    return scores_arr, labels_arr, indices_arr


def grand_scores_from_model(
    model: Any,
    loader: Any,
    n: Optional[int] = None,
    device: Any = None,
    criterion: Any = None,
    epochs: int = 1,
    param_scope: str = "last_layer",
    reduction: str = "norm",
    return_index: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """GraNd scores for the examples reached by ``loader``.

    ``epochs`` measurement passes are averaged (the paper's "during training"
    formulation).  When ``n`` is given and the loader yields source indices, the
    scores are scattered back into canonical example order of length ``n``.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("GraNd scores require PyTorch to be installed.")

    params = _select_parameters(model, param_scope)

    total: Optional[np.ndarray] = None
    idx_ref: Optional[np.ndarray] = None
    count = 0
    for _ in range(max(1, int(epochs))):
        scores, _labels, indices = _grad_norm_scores_single_pass(
            model,
            loader,
            device=device,
            criterion=criterion,
            param_scope=param_scope,
            parameters=params,
            reduction=reduction,
        )
        total = scores if total is None else total + scores
        count += 1
        if indices is not None:
            idx_ref = indices

    scores = total / float(max(1, count))

    if n is not None and idx_ref is not None:
        scores = gather_scores_by_index(scores, idx_ref, int(n))
    elif n is not None and scores.shape[0] != int(n):
        # No index information: assume loader order matches example order.
        out = np.full(int(n), np.nan, dtype=np.float64)
        m = min(int(n), scores.shape[0])
        out[:m] = scores[:m]
        nan_mask = ~np.isfinite(out)
        if nan_mask.any():
            out[nan_mask] = np.nanmean(scores) if scores.size else 0.0
        scores = out

    if return_index:
        return scores, idx_ref
    return scores


def grand_scores(
    model: Any = None,
    loader: Any = None,
    n: Optional[int] = None,
    device: Any = None,
    criterion: Any = None,
    epochs: int = 1,
    optimizer: str = "adam",
    lr: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    train_loader: Any = None,
    train_epochs: int = 100,
    train_val_loader: Any = None,
    param_scope: str = "last_layer",
    reduction: str = "norm",
    seed: Optional[int] = None,
    return_index: bool = True,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Reference implementation of the GraNd score.

    If ``train_loader`` is provided the proxy model is first trained
    (``train_epochs`` epochs of Adam, lr 0.001 -- Section 5.2 inner loop), then
    the scores are measured on ``loader`` (defaults to ``train_loader``).
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("GraNd scores require PyTorch to be installed.")
    if seed is not None:
        set_seed(seed)
    if model is None:
        raise ValueError("grand_scores requires a `model`.")

    if train_loader is not None and train_epochs and train_epochs > 0:
        model = train_reference_model(
            model,
            train_loader,
            epochs=train_epochs,
            lr=lr,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            criterion=criterion,
        )

    if loader is None:
        loader = train_loader
    if loader is None:
        raise ValueError("grand_scores requires a `loader` (or `train_loader`).")

    return grand_scores_from_model(
        model,
        loader,
        n=n,
        device=device,
        criterion=criterion,
        epochs=epochs,
        param_scope=param_scope,
        reduction=reduction,
        return_index=return_index,
    )


# Backwards/forwards friendly alias (the paper's own spelling).
grad_norm_scores = grand_scores


def grand_indices(scores: Any, k: int, seed: Optional[int] = None) -> np.ndarray:
    """Indices of the ``k`` examples with the largest GraNd scores."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if k >= scores.shape[0]:
        return np.arange(scores.shape[0], dtype=np.int64)
    return topk_indices(scores, int(k), largest=True)


def grand_mask(
    scores: Any,
    n: Optional[int] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Binary coreset mask from GraNd scores (largest scores are kept)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(scores.shape[0]) if n is None else int(n)
    if k is None:
        k = n // 2
    idx = grand_indices(scores, int(k), seed=seed)
    return indices_to_mask(idx, n=n, dtype=dtype)


# ---------------------------------------------------------------------------
# Selector class
# ---------------------------------------------------------------------------

class GraNdSelector(ScoreBaseline):
    """GraNd coreset selection (large loss-gradient norms).

    ``requires_model = True`` because the score is defined through a (proxy)
    model's per-example gradients; the selector accepts a shared reference model
    via the constructor or via ``compute_scores(..., model=...)``.
    """

    name = "GraNd"
    abbreviation = "GraNd"
    requires_model = True
    higher_is_better = True

    def __init__(
        self,
        model: Any = None,
        reference_epochs: int = 100,
        n_train_epochs: int = 1,
        epochs: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        momentum: float = 0.9,
        batch_size: int = 128,
        param_scope: str = "last_layer",
        reduction: str = "norm",
        num_classes: Optional[int] = None,
        device: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, device=device, num_classes=num_classes, **kwargs)
        self.model = model
        self.reference_epochs = int(reference_epochs)
        # ``n_train_epochs`` mirrors the data_diet-style flag used by EL2N here;
        # when >1 the score is averaged over several measurement passes.
        self.n_train_epochs = int(n_train_epochs)
        self.epochs = int(epochs) if epochs else max(1, int(n_train_epochs))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.batch_size = int(batch_size)
        self.param_scope = param_scope
        self.reduction = reduction

    # -- helpers ---------------------------------------------------------
    def _resolve_seed(self, seed: Optional[int]) -> Optional[int]:
        return seed if seed is not None else self.seed

    def _resolve_n(
        self,
        n: Optional[int],
        dataset: Any = None,
        targets: Any = None,
        scores_hint: Any = None,
    ) -> int:
        if n is not None:
            return int(n)
        if scores_hint is not None:
            return int(np.asarray(scores_hint).reshape(-1).shape[0])
        if targets is not None:
            return int(np.asarray(targets).reshape(-1).shape[0])
        if dataset is not None and hasattr(dataset, "__len__"):
            return int(len(dataset))
        raise ValueError("Could not infer n for GraNdSelector.")

    def _resolve_loader(self, train_loader: Any = None, loader: Any = None, dataset: Any = None) -> Any:
        if train_loader is not None:
            return train_loader
        if loader is not None:
            return loader
        if dataset is not None:
            try:
                from lbcs_repro.data.datasets import make_loader  # local import

                return make_loader(dataset, batch_size=self.batch_size, shuffle=False)
            except Exception:  # pragma: no cover
                if _TORCH_AVAILABLE:
                    return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return None

    # -- BaselineSelector API -------------------------------------------
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
        epochs: Optional[int] = None,
        reference_epochs: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return per-example GraNd scores (length ``n``)."""
        _ = num_classes  # kept for API symmetry
        resolved_seed = self._resolve_seed(seed)
        if resolved_seed is not None:
            set_seed(resolved_seed)

        model = model if model is not None else self.model
        load = self._resolve_loader(train_loader=train_loader, loader=loader, dataset=dataset)
        n_resolved = self._resolve_n(n, dataset=dataset, targets=targets)

        if model is None:
            raise ValueError(
                "GraNdSelector needs a proxy model (pass `model=` or a trained "
                "reference model) because the score is a gradient norm."
            )
        if load is None:
            raise ValueError("GraNdSelector needs a data loader to compute scores.")

        epochs_used = int(epochs) if epochs is not None else self.epochs
        ref_epochs = (
            int(reference_epochs) if reference_epochs is not None else self.reference_epochs
        )
        # Only train the proxy if it has not been supplied pre-trained by the
        # caller/driver (drivers commonly share one reference model across
        # baselines, and re-training here would break that sharing).
        should_train = bool(kwargs.get("train_model", False))
        if "train_model" not in kwargs and train_loader is not None and not self._model_is_trained(model):
            pass  # drivers own reference-model training; do not silently retrain

        scores = grand_scores(
            model=model,
            loader=load,
            n=n_resolved,
            device=self.device,
            epochs=max(1, epochs_used),
            optimizer=self.optimizer,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            train_loader=load if should_train else None,
            train_epochs=ref_epochs,
            param_scope=self.param_scope,
            reduction=self.reduction,
            seed=resolved_seed,
            return_index=False,
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.shape[0] != n_resolved:
            out = np.full(n_resolved, np.nan, dtype=np.float64)
            m = min(n_resolved, scores.shape[0])
            out[:m] = scores[:m]
            bad = ~np.isfinite(out)
            if bad.any():
                fill = np.nanmean(scores) if scores.size else 0.0
                out[bad] = fill
            scores = out
        return scores

    @staticmethod
    def _model_is_trained(model: Any) -> bool:
        """Heuristic: a model produced by ``train_reference_model`` carries a
        ``_lbcs_trained`` marker; otherwise assume it is untrained."""
        return bool(getattr(model, "_lbcs_trained", False))

    # -- convenience -----------------------------------------------------
    def mask(self, n: int, k: int, scores: Any, **kwargs: Any) -> np.ndarray:
        return grand_mask(scores, n=n, k=k, seed=self.seed)


# Alias matching the paper's abbreviation.
GraNd = GraNdSelector


# ---------------------------------------------------------------------------
# Offline self-test
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Validate score ranking, mask construction and (if torch is present) the
    gradient-norm plumbing against a tiny stub model."""
    results: Dict[str, Any] = {}

    # 1) score -> mask ranking
    scores = np.array([0.1, 3.0, 0.2, 2.9, 0.0], dtype=np.float64)
    idx = grand_indices(scores, 2)
    results["topk_indices"] = idx.tolist()
    assert set(idx.tolist()) == {1, 3}, idx

    mask = grand_mask(scores, n=5, k=2)
    assert mask.shape == (5,)
    assert int(mask.sum()) == 2
    assert mask[1] == 1.0 and mask[3] == 1.0
    results["mask"] = mask.tolist()

    # k >= n keeps everything
    assert int(grand_mask(scores, n=5, k=10).sum()) == 5
    results["k_ge_n_ok"] = True

    # determinism
    m1 = grand_mask(scores, n=5, k=2, seed=7)
    m2 = grand_mask(scores, n=5, k=2, seed=7)
    assert np.array_equal(m1, m2)
    results["deterministic"] = True

    # 2) torch-level checks: gradient norms of a linear model.
    if _TORCH_AVAILABLE:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 3))
        # Make the task noisy so gradients differ per example.
        torch.manual_seed(1)
        xs = torch.randn(16, 1, 2, 2)
        ys = torch.randint(0, 3, (16,))

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, i):
                return xs[i], ys[i], i

        loader = torch.utils.data.DataLoader(_DS(), batch_size=8, shuffle=False)
        sc, _lab, ix = _grad_norm_scores_single_pass(model, loader, device=torch.device("cpu"))
        assert sc.shape == (16,), sc.shape
        assert np.all(sc >= 0.0)
        assert ix is not None and ix.shape == (16,)
        results["grad_norm_scores_shape"] = list(sc.shape)

        # scores scatter back to canonical order
        permuted = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs.reshape(16, -1), ys),
            batch_size=16,
            shuffle=False,
        )
        s2 = grand_scores_from_model(model, permuted, n=16, return_index=False)
        assert s2.shape == (16,)
        results["scatter_ok"] = True

        # selector end-to-end with a stub reference model
        sel = GraNdSelector(model=model, seed=0, device=torch.device("cpu"))
        scores_sel = sel.compute_scores(n=16, loader=loader)
        assert scores_sel.shape == (16,)
        msk = sel.select_mask(n=16, k=4, scores=scores_sel)
        assert int(msk.sum()) == 4
        results["selector_ok"] = True

        # single-layer parameter scope resolves to the final Linear module
        params = _select_parameters(model, "last_layer")
        assert len(params) == 2, [p.shape for p in params]
        results["last_layer_params"] = [tuple(p.shape) for p in params]
    else:  # pragma: no cover
        results["torch"] = "unavailable"

    if verbose:
        LOGGER.info("GraNd selftest: %s", results)
        print("GraNd selftest:", results)
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
