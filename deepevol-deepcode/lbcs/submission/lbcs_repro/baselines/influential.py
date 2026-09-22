"""Influential coreset baseline (Yang et al., ICLR 2023).

Paper reference
---------------
Section 5.2 (Competitors) lists *Influential coreset* (Yang et al., 2023),
abbreviated ``Influential``, among the coreset-selection baselines that build a
coreset of **predefined size** ``k`` (the size is *not* further minimized).

Appendix D.1 (Details of Baselines) describes it as:

    "- Influential coreset (ICLR 2023) (Yang et al., 2023). This algorithm
      utilizes the influence function (Hampel, 1974). The examples that yield
      strictly constrained generalization gaps are included in the coreset."

Implementation
--------------
The classical influence-function estimate of how adding a training example
``z`` changes the *validation / generalization* loss is

    I(z) = -grad_z L(z; theta)^T  H_theta^{-1}  grad_theta L_val(theta),

so the induced change of the validation loss when ``z`` receives weight ``eps``
is ``d L_val / d eps |_{eps=0} = I(z)``.  Examples with a large value of the
generalization-gap **reduction**

    score(z) = -I(z) = grad_z L(z)^T H^{-1} grad_val

are exactly the examples that *reduce* (i.e. strictly constrain) the
generalization gap, hence they are the ones included in the coreset -- matching
the Appendix D.1 description.  The selector therefore ranks examples by
``score(z)`` (``higher_is_better = True``) and keeps the top ``k``.

Practical ingredients
---------------------
* per-example gradients w.r.t. a *filtered* parameter subset (default: last
  layer -- the original influence / GraNd convention; ``"all"`` and ``"bias"``
  are also supported);
* an inverse-Hessian-vector product ``H^{-1} grad_val`` with three
  approximations:
    - ``"identity"`` : ``H^{-1} ~ I`` (first-order / TracIn-like),
    - ``"diag"``     : Gauss-Newton (Fisher) diagonal ``diag(H) = E[g_z g_z^T]``
                       (default; cheap and numerically stable),
    - ``"cg"``       : truncated conjugate gradient on Hessian-vector products
                       computed by double back-propagation;
* optional damping ``H + lambda I``;
* optional *strict* constraint (``require_positive``) so that only examples
  whose gap reduction is strictly positive are admissible; remaining slots are
  filled with the next-best scores so ``||m||_0 = k`` is always satisfied.

The module mirrors :mod:`lbcs_repro.baselines.grand` and
:mod:`lbcs_repro.baselines.el2n` so experiment drivers (Table 2 / Table 3 /
Figure 2) can treat every baseline uniformly: ``select_mask(n, k, ...)`` returns
a binary mask ``m in {0,1}^n`` with ``||m||_0 = k``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # PyTorch is a soft dependency so mask-only unit tests still run.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - only without PyTorch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from .base import (
    ScoreBaseline,
    forward_logits,
    gather_scores_by_index,
    indices_to_mask,
    resolve_seed,
    set_seed,
    topk_indices,
    train_reference_model,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "InfluentialSelector",
    "Influential",
    "influential_scores",
    "influential_indices",
    "influential_mask",
    "per_sample_grad_vectors",
    "validation_grad_vector",
    "hessian_diagonal",
    "hessian_vector_product",
    "conjugate_gradient",
    "inverse_hessian_vector",
    "generalization_gap_scores",
]


# ---------------------------------------------------------------------------
# small torch helpers
# ---------------------------------------------------------------------------
def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "Influential coreset scores require PyTorch; install torch to use "
            "lbcs_repro.baselines.influential."
        )


def _looks_like_dataloader(obj: Any) -> bool:
    if obj is None:
        return False
    if _TORCH_AVAILABLE and isinstance(obj, torch.utils.data.DataLoader):
        return True
    return hasattr(obj, "__iter__") and hasattr(obj, "dataset")


def _looks_like_dataset(obj: Any) -> bool:
    if obj is None:
        return False
    if _TORCH_AVAILABLE and isinstance(obj, torch.utils.data.Dataset):
        return True
    return hasattr(obj, "__getitem__") and hasattr(obj, "__len__")


def _make_loader(
    obj: Any, batch_size: int = 128, shuffle: bool = False, num_workers: int = 0
) -> Any:
    """Accept a DataLoader, a Dataset, or ``None``."""
    if obj is None:
        return None
    if _looks_like_dataloader(obj):
        return obj
    if _looks_like_dataset(obj):
        try:  # prefer the project loader factory (consistent seeding/normalization)
            from lbcs_repro.data.datasets import make_loader

            return make_loader(
                obj, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
        except Exception:
            from torch.utils.data import DataLoader

            return DataLoader(
                obj, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
    raise TypeError(f"Cannot build a DataLoader from object of type {type(obj)!r}")


def _select_parameters(model: Any, scope: str = "last_layer") -> List[Any]:
    """Filter the differentiable parameters whose gradient defines ``grad_z``.

    ``scope`` in ``{"last_layer", "all", "bias"}``.  ``"last_layer"`` follows the
    GraNd / EL2N / influence convention and keeps the per-example gradients small
    by using the final learnable layer only.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("Parameter filtering requires PyTorch.")
    params = [p for p in model.parameters() if getattr(p, "requires_grad", True)]
    if not params:
        raise ValueError("Model exposes no trainable parameters.")
    scope = (scope or "last_layer").lower()
    if scope in {"all", "*"}:
        return params
    if scope == "bias":
        sel = [p for p in model.parameters() if p.ndim == 1 and p.requires_grad]
        return sel or params
    # last learnable layer: the final module child that owns parameters
    last: List[Any] = []
    for module in model.modules():
        own = [p for p in module.parameters(recurse=False) if p.requires_grad]
        if own:
            last = own
    if not last:
        last = params[-2:] if len(params) >= 2 else params[-1:]
    return last


def _unpack(batch: Any) -> Tuple[Any, Any, Optional[Any]]:
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise ValueError(f"Unsupported batch structure: {type(batch)!r}")


def _flat_grads(grads: Sequence[Any], parameters: Sequence[Any]) -> Any:
    parts = []
    for gi, p in zip(grads, parameters):
        if gi is None:
            parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        else:
            parts.append(gi.detach().reshape(-1))
    return torch.cat(parts) if parts else torch.zeros(0)


# ---------------------------------------------------------------------------
# per-example gradient vectors
# ---------------------------------------------------------------------------
def _vmap_per_sample_grads(
    model: Any,
    inputs: Any,
    targets: Any,
    parameters: Sequence[Any],
    criterion: Any,
) -> Optional[Any]:
    """Fast path using ``torch.func`` (vmap over grad) when available."""
    try:
        import torch.func as tf  # noqa: WPS433 (torch >= 2.0)
    except Exception:
        return None

    wanted = {id(p) for p in parameters}
    params = {
        name: p for name, p in model.named_parameters() if p.requires_grad and id(p) in wanted
    }
    if not params:
        return None
    buffers = dict(model.named_buffers())

    def loss_fn(p, buffers_, xb, yb):
        out = tf.functional_call(model, (p, buffers_), (xb,))
        return criterion(out, yb)

    try:
        grads = tf.vmap(tf.grad(loss_fn), in_dims=(None, None, 0, 0))(
            params, buffers, inputs, targets
        )
        flat = torch.cat([grads[k].reshape(inputs.shape[0], -1) for k in grads], dim=1)
    except Exception:
        return None
    return flat


def _loop_per_sample_grads(
    model: Any,
    inputs: Any,
    targets: Any,
    parameters: Sequence[Any],
    criterion: Any,
) -> Any:
    """Portable per-example gradient path (one backward pass per example)."""
    rows: List[Any] = []
    for i in range(inputs.shape[0]):
        logits = forward_logits(model, inputs[i : i + 1])
        loss = criterion(logits, targets[i : i + 1])
        grads = torch.autograd.grad(
            loss, list(parameters), allow_unused=True, retain_graph=False, create_graph=False
        )
        rows.append(_flat_grads(grads, parameters))
    if not rows:
        return torch.zeros((0, 0))
    return torch.stack(rows, dim=0)


def per_sample_grad_vectors(
    model: Any,
    loader: Any = None,
    n: Optional[int] = None,
    device: Any = None,
    criterion: Any = None,
    param_scope: str = "last_layer",
    parameters: Optional[Sequence[Any]] = None,
    max_batches: Optional[int] = None,
    return_index: bool = True,
    method: str = "auto",
    batch_size: int = 128,
    num_workers: int = 0,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Per-example loss gradients ``grad_z L(z; theta)``.

    Returns ``grads`` of shape ``(num_examples, num_params)`` (float64) and, when
    ``return_index`` is true, the canonical example indices visited by the
    (optionally index-aware) loader -- else ``None``.
    """
    _require_torch()
    loader = _make_loader(
        loader, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    if loader is None:
        raise ValueError("per_sample_grad_vectors requires a loader or a dataset.")

    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    if parameters is None:
        parameters = _select_parameters(model, param_scope)
    parameters = list(parameters)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()

    method = (method or "auto").lower()
    rows: List[Any] = []
    indices: List[Any] = []
    count = 0
    t0 = time.time()
    with torch.enable_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            inputs, targets, idx = _unpack(batch)
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            if hasattr(inputs, "float"):
                inputs = inputs.float()
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            flat = None
            if method in {"auto", "vmap"}:
                flat = _vmap_per_sample_grads(model, inputs, targets, parameters, criterion)
                if flat is None and method == "vmap":
                    raise RuntimeError("vmap-based per-sample gradients are unavailable.")
            if flat is None:
                flat = _loop_per_sample_grads(model, inputs, targets, parameters, criterion)
            rows.append(flat.detach().cpu().double())
            if idx is not None:
                indices.append(np.asarray(idx, dtype=np.int64))
            count += int(flat.shape[0])
            if verbose and (b_idx + 1) % 20 == 0:
                LOGGER.info(
                    "influence: %d batches / %d examples (%.1fs)",
                    b_idx + 1, count, time.time() - t0,
                )

    if was_training and hasattr(model, "train"):
        model.train()

    if not rows:
        empty = np.zeros((0, 0), dtype=np.float64)
        return (empty, None) if return_index else empty

    grads = torch.cat(rows, dim=0).numpy().astype(np.float64, copy=False)
    index_arr = np.concatenate(indices) if indices else None
    return (grads, index_arr) if return_index else grads


# ---------------------------------------------------------------------------
# validation gradient and inverse-Hessian-vector products
# ---------------------------------------------------------------------------
def validation_grad_vector(
    model: Any,
    loader: Any,
    device: Any = None,
    criterion: Any = None,
    param_scope: str = "last_layer",
    parameters: Optional[Sequence[Any]] = None,
    max_batches: Optional[int] = None,
) -> np.ndarray:
    """``grad_theta L_val(theta)`` over the validation probe set."""
    _require_torch()
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    if parameters is None:
        parameters = _select_parameters(model, param_scope)
    parameters = list(parameters)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    acc: List[Any] = []
    with torch.enable_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            inputs, targets, _ = _unpack(batch)
            if hasattr(inputs, "float"):
                inputs = inputs.float()
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            logits = forward_logits(model, inputs)
            loss = criterion(logits, targets)
            grads = torch.autograd.grad(
                loss, parameters, allow_unused=True, retain_graph=False, create_graph=False
            )
            acc.append(_flat_grads(grads, parameters).detach().cpu().double())
    if was_training and hasattr(model, "train"):
        model.train()
    if not acc:
        return np.zeros(0, dtype=np.float64)
    # Every batch contributed the gradient of its *mean* loss; with (almost)
    # equal batch sizes the unweighted mean is the gradient of the mean loss.
    return torch.stack(acc, dim=0).numpy().mean(axis=0).astype(np.float64, copy=False)


def hessian_diagonal(grads: np.ndarray, damping: float = 0.0) -> np.ndarray:
    """Gauss-Newton (Fisher) diagonal ``diag(H) = E[g_z g_z^T]`` from per-example grads."""
    g = np.asarray(grads, dtype=np.float64)
    if g.size == 0:
        return np.zeros(0, dtype=np.float64)
    diag = np.mean(g * g, axis=0)
    return diag + float(damping)


def hessian_vector_product(
    model: Any,
    loader: Any,
    v: np.ndarray,
    device: Any = None,
    criterion: Any = None,
    param_scope: str = "last_layer",
    parameters: Optional[Sequence[Any]] = None,
    max_batches: Optional[int] = 1,
) -> np.ndarray:
    """``H v`` for the mean training loss, via double back-propagation."""
    _require_torch()
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    if parameters is None:
        parameters = _select_parameters(model, param_scope)
    parameters = list(parameters)
    v_t = torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(-1))
    v_t = v_t.to(device=parameters[0].device, dtype=parameters[0].dtype)

    was_training = getattr(model, "training", False)
    if hasattr(model, "eval"):
        model.eval()
    hs: List[Any] = []
    with torch.enable_grad():
        for b_idx, batch in enumerate(loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            inputs, targets, _ = _unpack(batch)
            if hasattr(inputs, "float"):
                inputs = inputs.float()
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            logits = forward_logits(model, inputs)
            loss = criterion(logits, targets)
            grads = torch.autograd.grad(
                loss, parameters, create_graph=True, allow_unused=True
            )
            flat = _flat_grads(grads, parameters)
            dot = (flat * v_t).sum()
            hvp = torch.autograd.grad(dot, parameters, allow_unused=True, retain_graph=False)
            hs.append(_flat_grads(hvp, parameters).detach().cpu().double())
    if was_training and hasattr(model, "train"):
        model.train()
    if not hs:
        return np.zeros_like(np.asarray(v, dtype=np.float64).reshape(-1))
    return torch.stack(hs, dim=0).numpy().mean(axis=0).astype(np.float64, copy=False)


def conjugate_gradient(
    hvp_fn: Callable[[np.ndarray], np.ndarray],
    b: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    damping: float = 0.0,
) -> np.ndarray:
    """Truncated conjugate gradient solving ``(A + damping I) x = b``."""
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)
    if rs == 0.0:
        return x
    bnorm = max(float(np.linalg.norm(b)), 1e-12)
    for _ in range(int(max_iter)):
        Ap = np.asarray(hvp_fn(p), dtype=np.float64).reshape(-1) + float(damping) * p
        denom = float(p @ Ap)
        if not np.isfinite(denom) or abs(denom) < 1e-30:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = float(r @ r)
        if np.sqrt(rs_new) <= float(tol) * bnorm:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def inverse_hessian_vector(
    val_grad: np.ndarray,
    grads: Optional[np.ndarray] = None,
    hessian_approx: str = "diag",
    damping: float = 1e-2,
    model: Any = None,
    loader: Any = None,
    device: Any = None,
    criterion: Any = None,
    param_scope: str = "last_layer",
    cg_iters: int = 30,
    cg_tol: float = 1e-6,
    max_batches: int = 1,
    parameters: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    """Approximate ``H^{-1} grad_val`` under the chosen Hessian approximation.

    ``hessian_approx`` in ``{"identity", "diag", "cg"}``:
      * ``identity`` : returns ``val_grad`` unchanged (TracIn-like first order);
      * ``diag``     : ``val_grad / (E[g_z^2] + damping)`` from ``grads`` (or from
                       ``loader`` when ``grads`` is not supplied);
      * ``cg``       : truncated CG with Hessian-vector products from ``model``.
    """
    gval = np.asarray(val_grad, dtype=np.float64).reshape(-1)
    mode = (hessian_approx or "diag").lower()
    if mode in {"identity", "none", "tracin"}:
        return gval
    if mode in {"diag", "diagonal", "fisher", "gn"}:
        g = grads
        if g is None:
            if model is None or loader is None:
                raise ValueError("diag Hessian approximation needs `grads` or (`model`, `loader`).")
            g, _ = per_sample_grad_vectors(
                model, loader, device=device, criterion=criterion,
                param_scope=param_scope, parameters=parameters,
                max_batches=max_batches, return_index=True,
            )
        diag = hessian_diagonal(g, damping=0.0)
        if diag.size == 0:
            return gval
        return gval / (diag + float(damping))
    if mode in {"cg", "conjugate", "conjugate_gradient"}:
        if model is None or loader is None:
            raise ValueError("cg Hessian approximation needs `model` and `loader`.")

        def _hvp(v: np.ndarray) -> np.ndarray:
            return hessian_vector_product(
                model, loader, v, device=device, criterion=criterion,
                param_scope=param_scope, parameters=parameters, max_batches=max_batches,
            )

        return conjugate_gradient(_hvp, gval, max_iter=cg_iters, tol=cg_tol, damping=damping)
    raise ValueError(f"Unknown hessian_approx={hessian_approx!r}")


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------
def generalization_gap_scores(
    grads: np.ndarray,
    h_inv_val_grad: np.ndarray,
    normalize: bool = False,
) -> np.ndarray:
    """``score(z) = grad_z^T H^{-1} grad_val`` (generalization-gap reduction).

    A larger score means adding ``z`` reduces the validation / generalization
    loss more strongly, i.e. constrains the generalization gap more tightly.
    """
    g = np.asarray(grads, dtype=np.float64)
    w = np.asarray(h_inv_val_grad, dtype=np.float64).reshape(-1)
    if g.size == 0:
        return np.zeros(0, dtype=np.float64)
    if w.size != g.shape[1]:
        raise ValueError(
            f"Dimension mismatch: grads has {g.shape[1]} parameters but the "
            f"inverse-Hessian-vector has {w.size}."
        )
    if normalize:
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        gn[gn == 0.0] = 1.0
        g = g / gn
    return g @ w


def influential_scores(
    model: Any = None,
    loader: Any = None,
    val_loader: Any = None,
    n: Optional[int] = None,
    device: Any = None,
    criterion: Any = None,
    param_scope: str = "last_layer",
    parameters: Optional[Sequence[Any]] = None,
    hessian_approx: str = "diag",
    damping: float = 1e-2,
    cg_iters: int = 30,
    normalize: bool = False,
    max_batches: Optional[int] = None,
    diag_batches: Optional[int] = 64,
    return_index: bool = True,
    train_loader: Any = None,
    train_epochs: int = 100,
    train: bool = False,
    lr: float = 0.001,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Per-example generalization-influence scores for the candidate pool.

    If ``train_loader`` is given and ``train`` is true, a reference model is
    first trained on it (section 5.2 uses Adam with ``lr=1e-3``); otherwise the
    supplied ``model`` is used as-is.  When ``val_loader`` is ``None`` the
    training loader itself serves as the validation probe (self-influence
    approximation, as done when no held-out generalization set is available).
    """
    _require_torch()
    if model is None:
        raise ValueError("influential_scores requires a (reference) model.")
    if loader is None:
        raise ValueError("influential_scores requires the candidate `loader`.")

    loader = _make_loader(
        loader, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    if train and train_loader is not None:
        train_loader = _make_loader(
            train_loader, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
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
            log_every=0,
            verbose=verbose,
        )
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="mean")
    if parameters is None:
        parameters = _select_parameters(model, param_scope)

    probe = val_loader if val_loader is not None else loader
    probe = _make_loader(
        probe, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if verbose:
        LOGGER.info("Influential: collecting per-example gradients (scope=%s)", param_scope)
    grads, idx = per_sample_grad_vectors(
        model, loader, n=n, device=device, criterion=criterion,
        param_scope=param_scope, parameters=parameters, max_batches=max_batches,
        return_index=True, batch_size=batch_size, num_workers=num_workers, verbose=verbose,
    )
    gval = validation_grad_vector(
        model, probe, device=device, criterion=criterion,
        param_scope=param_scope, parameters=parameters,
    )
    mode = (hessian_approx or "diag").lower()
    use_diag_grads = mode in {"diag", "diagonal", "fisher", "gn"}
    hid = inverse_hessian_vector(
        gval,
        grads=grads if use_diag_grads else None,
        hessian_approx=hessian_approx,
        damping=damping,
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        param_scope=param_scope,
        cg_iters=cg_iters,
        parameters=parameters,
        max_batches=(diag_batches if use_diag_grads else 1),
    )
    scores = generalization_gap_scores(grads, hid, normalize=normalize)

    if n is not None:
        scores = gather_scores_by_index(scores, idx, n)

    if return_index:
        return scores, idx
    return scores


def influential_indices(
    scores: np.ndarray,
    k: int,
    seed: Optional[int] = None,
    require_positive: bool = False,
) -> np.ndarray:
    """Indices of the ``k`` examples that most constrain the generalization gap.

    With ``require_positive``, only examples whose score strictly reduces the gap
    (``score > 0``) are admissible; if fewer than ``k`` exist, the remaining slots
    are filled with the next-best scores so exactly ``k`` indices are returned
    (the baselines must produce a predetermined coreset size ``k``).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    k = int(k)
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    if k >= scores.size:
        return np.arange(scores.size, dtype=np.int64)

    order = topk_indices(scores, scores.size, largest=True)
    if require_positive:
        pos = order[scores[order] > 0.0]
        if pos.size >= k:
            return pos[:k].astype(np.int64)
        rest_mask = np.ones(order.size, dtype=bool)
        rest_mask[: pos.size] = False
        rest = order[rest_mask]
        return np.concatenate([pos, rest[: k - pos.size]]).astype(np.int64)
    return order[:k].astype(np.int64)


def influential_mask(
    scores: np.ndarray,
    n: Optional[int] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    dtype: Any = np.float32,
    require_positive: bool = False,
) -> np.ndarray:
    """Binary mask selecting the top-``k`` influence scores."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = int(n) if n is not None else int(scores.size)
    k = int(k) if k is not None else n // 2
    idx = influential_indices(scores, k, seed=seed, require_positive=require_positive)
    return indices_to_mask(idx, n, dtype=dtype)


# ---------------------------------------------------------------------------
# selector class
# ---------------------------------------------------------------------------
class InfluentialSelector(ScoreBaseline):
    """Influential coreset (Yang et al., ICLR 2023) -- Appendix D.1 baseline.

    Scores examples by the influence-function estimate of the generalization-gap
    reduction ``grad_z^T H^{-1} grad_val`` and keeps the top ``k``
    (``higher_is_better = True``).
    """

    name = "Influential"
    abbreviation = "Influential"
    requires_model = True
    higher_is_better = True

    def __init__(
        self,
        model: Any = None,
        reference_epochs: int = 100,
        train_model: bool = False,
        param_scope: str = "last_layer",
        hessian_approx: str = "diag",
        damping: float = 1e-2,
        cg_iters: int = 30,
        normalize: bool = False,
        batch_size: int = 128,
        train_epochs: int = 100,
        lr: float = 0.001,
        optimizer: str = "adam",
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        max_batches: Optional[int] = None,
        diag_batches: Optional[int] = 64,
        num_workers: int = 0,
        num_classes: Optional[int] = None,
        require_positive: bool = False,
        device: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, device=device, num_classes=num_classes, **kwargs)
        self.model = model
        self.reference_epochs = int(reference_epochs)
        self.train_model = bool(train_model)
        self.param_scope = param_scope
        self.hessian_approx = hessian_approx
        self.damping = float(damping)
        self.cg_iters = int(cg_iters)
        self.normalize = bool(normalize)
        self.batch_size = int(batch_size)
        self.train_epochs = int(train_epochs)
        self.lr = float(lr)
        self.optimizer = optimizer
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.max_batches = max_batches
        self.diag_batches = diag_batches
        self.num_workers = int(num_workers)
        self.require_positive = bool(require_positive)

    # -- internals ---------------------------------------------------------
    def _resolve_n(
        self, n: Optional[int], dataset: Any, targets: Any, scores: Any = None
    ) -> int:
        if n is not None:
            return int(n)
        if targets is not None:
            return int(np.asarray(targets).reshape(-1).size)
        if scores is not None:
            return int(np.asarray(scores).reshape(-1).size)
        if dataset is not None and hasattr(dataset, "__len__"):
            return int(len(dataset))
        raise ValueError("Cannot infer the number of candidate examples `n`.")

    def _resolve_loader(self, dataset: Any, loader: Any) -> Any:
        if loader is not None:
            return loader
        if dataset is not None:
            return _make_loader(
                dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
        return None

    def _resolve_val_loader(self, val_loader: Any, val_dataset: Any) -> Any:
        if val_loader is not None:
            return val_loader
        if val_dataset is not None:
            return _make_loader(
                val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
        return None

    # -- ScoreBaseline API -------------------------------------------------
    def compute_scores(
        self,
        dataset: Any = None,
        targets: Any = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        model: Any = None,
        loader: Any = None,
        val_loader: Any = None,
        val_dataset: Any = None,
        train_loader: Any = None,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return per-example generalization-influence scores of length ``n``."""
        model = model if model is not None else self.model
        if model is None:
            raise ValueError(
                "InfluentialSelector needs a reference `model` (constructor or compute_scores)."
            )
        loader = self._resolve_loader(dataset, loader)
        if loader is None:
            raise ValueError("InfluentialSelector needs a `dataset` or a `loader`.")
        n = self._resolve_n(n, dataset, targets)
        seed = resolve_seed(seed if seed is not None else self.seed, 0)
        set_seed(seed)

        probe = self._resolve_val_loader(val_loader, val_dataset)
        if train_loader is None:
            train_loader = kwargs.get("train_loader", None)

        scores = influential_scores(
            model=model,
            loader=loader,
            val_loader=probe,
            n=n,
            device=self.device,
            param_scope=self.param_scope,
            hessian_approx=self.hessian_approx,
            damping=self.damping,
            cg_iters=self.cg_iters,
            normalize=self.normalize,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            max_batches=self.max_batches,
            diag_batches=self.diag_batches,
            train_loader=train_loader,
            train=bool(self.train_model and train_loader is not None),
            train_epochs=self.train_epochs,
            lr=self.lr,
            optimizer=self.optimizer,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            seed=seed,
            return_index=False,
            verbose=kwargs.get("verbose", False),
        )
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if scores.size != n:
            LOGGER.warning(
                "Influential scores length %d != n %d; padding/truncating.", scores.size, n
            )
            if scores.size > n:
                scores = scores[:n]
            else:
                scores = np.concatenate(
                    [scores, np.zeros(n - scores.size, dtype=np.float64)]
                )
        return scores

    def select_indices(
        self,
        n: int,
        k: int,
        dataset: Any = None,
        targets: Any = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Any = None,
        loader: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        if scores is None:
            scores = self.compute_scores(
                dataset=dataset, targets=targets, n=n, seed=seed, loader=loader, **kwargs
            )
        return influential_indices(
            scores, k, seed=seed, require_positive=self.require_positive
        )

    def select_mask(
        self,
        n: int,
        k: int,
        dataset: Any = None,
        targets: Any = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Any = None,
        loader: Any = None,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> np.ndarray:
        idx = self.select_indices(
            n, k, dataset=dataset, targets=targets, num_classes=num_classes,
            seed=seed, scores=scores, loader=loader, **kwargs,
        )
        return indices_to_mask(idx, n, dtype=dtype)

    def mask(self, n: int, k: int, **kwargs: Any) -> np.ndarray:  # convenience alias
        return self.select_mask(n, k, **kwargs)


# convenient alias matching the paper's abbreviation
Influential = InfluentialSelector


# ---------------------------------------------------------------------------
# offline self-test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of the score algebra (no dataset/GPU needed)."""
    results: Dict[str, Any] = {}
    rng = np.random.default_rng(0)
    m, p = 40, 7
    grads = rng.normal(size=(m, p))
    gval = rng.normal(size=p)

    # identity approximation: score = grad_z . grad_val
    s_id = generalization_gap_scores(grads, gval, normalize=False)
    results["identity_matches_dot"] = bool(np.allclose(s_id, grads @ gval, atol=1e-10))

    # diag approximation: H^{-1} ~ diag(E[g^2]) + damping
    hid = inverse_hessian_vector(gval, grads=grads, hessian_approx="diag", damping=1e-2)
    diag = np.mean(grads * grads, axis=0)
    results["diag_inverse_ok"] = bool(np.allclose(hid, gval / (diag + 1e-2), atol=1e-10))

    # dimension mismatch detection
    try:
        generalization_gap_scores(grads, np.zeros(p + 1))
        results["mismatch_raises"] = False
    except ValueError:
        results["mismatch_raises"] = True

    # selection: exactly k indices, highest scores kept
    s = np.arange(m, dtype=np.float64)
    idx = influential_indices(s, k=10)
    results["topk_indices"] = bool(
        idx.size == 10 and np.array_equal(np.sort(idx), np.arange(m - 10, m))
    )
    mask = influential_mask(s, n=m, k=10)
    results["mask_size"] = bool(int(np.asarray(mask).sum()) == 10)
    results["mask_dtype_ok"] = bool(np.asarray(mask).dtype == np.float32)

    # require_positive path: only 3 positive scores -> filled up to k
    s2 = np.array([2.0, 1.0, 0.5, -1.0, -2.0, -3.0])
    idx2 = influential_indices(s2, k=5, require_positive=True)
    results["require_positive_fill"] = bool(
        idx2.size == 5 and set(idx2[:3].tolist()) == {0, 1, 2}
    )

    # conjugate gradient on a well-conditioned SPD matrix
    A = np.eye(5) * 2.0 + 0.1 * np.ones((5, 5))
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x = conjugate_gradient(lambda vv: A @ vv, b, max_iter=50, tol=1e-12, damping=0.0)
    results["cg_solves"] = bool(np.allclose(A @ x, b, atol=1e-6))

    # Fisher-diagonal helper consistency
    results["hessian_diagonal_ok"] = bool(
        np.allclose(hessian_diagonal(grads, damping=0.0), np.mean(grads ** 2, axis=0))
    )

    # selector plumbing with a stub torch model + tiny in-memory dataset
    if _TORCH_AVAILABLE:
        try:

            class _Stub(torch.utils.data.Dataset):
                def __init__(self, nb: int = 24, dim: int = 8, classes: int = 3) -> None:
                    self.x = torch.randn(nb, dim)
                    self.y = torch.randint(0, classes, (nb,))

                def __len__(self) -> int:
                    return self.x.shape[0]

                def __getitem__(self, i: int):
                    return self.x[i], self.y[i]

            class _Net(nn.Module):
                def __init__(self, dim: int = 8, classes: int = 3) -> None:
                    super().__init__()
                    self.fc1 = nn.Linear(dim, 6)
                    self.fc2 = nn.Linear(6, classes)

                def forward(self, x: Any) -> Any:
                    return self.fc2(torch.relu(self.fc1(x)))

            ds = _Stub()
            net = _Net()
            sel = InfluentialSelector(model=net, hessian_approx="diag")
            scores = sel.compute_scores(dataset=ds, n=len(ds))
            results["selector_scores_len"] = bool(scores.shape == (len(ds),))
            msk = sel.select_mask(len(ds), 9, dataset=ds)
            results["selector_mask_size"] = bool(int(np.asarray(msk).sum()) == 9)

            g_loop, i_loop = per_sample_grad_vectors(net, ds, return_index=True, method="loop")
            results["loop_grads_shape"] = bool(g_loop.shape[0] == len(ds))
            results["loop_indices"] = bool(i_loop is None or i_loop.shape[0] == len(ds))

            g_auto, _ = per_sample_grad_vectors(net, ds, return_index=True, method="auto")
            results["auto_loop_agree"] = bool(g_auto.shape == g_loop.shape)
        except Exception as exc:  # pragma: no cover
            results["selector_error"] = repr(exc)
            results["selector_ran"] = False
        else:
            results["selector_ran"] = True

    if verbose:
        for key, val in results.items():
            LOGGER.info("influential selftest: %-24s %s", key, val)
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
