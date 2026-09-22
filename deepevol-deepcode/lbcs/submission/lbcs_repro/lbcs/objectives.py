"""Objective functions of Refined Coreset Selection (LBCS).

Paper specification (§2 "Objective formulations", §2 "Preliminaries").

The bilevel framework introduces 0-1 masks ``m in {0,1}^n`` where ``m_i = 1``
indicates that the data point ``(x_i, y_i)`` is selected into the coreset.  With
``h(x; theta)`` the deep network and ``l(.,.)`` the cross-entropy loss,

    (1)  f_1(m) := (1/n) sum_{i=1}^{n} l(h(x_i; theta(m)), y_i)
         s.t.     theta(m) in argmin_theta L(m, theta),

         where   L(m, theta) = (1 / ||m||_0) sum_{i=1}^{n} m_i l(h(x_i; theta), y_i)

    (2)  f_2(m) := ||m||_0

``f_1`` is the primary objective and ``f_2`` the secondary one: ``f_2`` must only
be optimized under the premise that ``f_1`` does not get worse (§2: "we aim to
minimize f_1(m) and f_2(m) in order of priority ... f_2(m) should be optimized
under the premise of f_1(m)").

This module implements

* :func:`inner_coreset_loss`  -- ``L(m, theta)`` on the selected coreset (the
  inner loop objective, Algorithm 1 Step 3),
* :func:`full_data_loss`      -- ``f_1(m)`` for a *given* ``theta(m)``,
* :func:`f1`, :func:`f2`      -- the two objectives over a mask,
* :class:`MaskObjectiveEvaluator` -- the "train the inner loop, then evaluate on
  the full data" primitive of Algorithm 1 (Steps 3-4), with memoisation of every
  evaluated mask; the memo also provides the historical set ``H`` consumed by the
  practical lexicographic relations (§3.2 / Appendix A).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is a soft dependency so that pure-mask unit tests still run
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Subset = None  # type: ignore
    _TORCH_AVAILABLE = False

from .masks import l0_norm, num_selected, selected_indices

__all__ = [
    "MaskEvaluation",
    "MaskObjectiveEvaluator",
    "ObjectiveCache",
    "accuracy",
    "coreset_accuracy",
    "default_criterion",
    "f1",
    "f1_full_data",
    "f2",
    "forward_logits",
    "full_data_loss",
    "inner_coreset_loss",
    "mask_key",
    "per_sample_losses",
    "predict_logits",
    "unpack_batch",
]

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# small torch helpers
# ---------------------------------------------------------------------------
def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the LBCS inner loop / full-data evaluation."
        )


def default_criterion(reduction: str = "mean", device: Any = None):
    """Cross-entropy ``l(.)`` used everywhere in the paper (no class weights)."""
    _require_torch()
    crit = nn.CrossEntropyLoss(reduction=reduction)
    if device is not None:
        crit = crit.to(device)
    return crit


def forward_logits(model: Any, inputs: Any) -> Any:
    """``h(x; theta)``: handle plain logits, tuples and HF/timm-style outputs."""
    out = model(inputs)
    if isinstance(out, tuple):
        out = out[0]
    elif hasattr(out, "logits"):  # HF-style ModelOutput
        out = out.logits
    return out


def unpack_batch(batch: Any) -> Tuple[Any, Any, Optional[Any]]:
    """Return ``(inputs, targets, indices_or_None)`` from a dataloader batch."""
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
        if len(batch) == 1:
            return batch[0], None, None
    if isinstance(batch, dict):  # dict-style batches
        inputs = batch.get("input", batch.get("x", batch.get("image")))
        targets = batch.get("target", batch.get("y", batch.get("label")))
        idx = batch.get("index", batch.get("idx"))
        return inputs, targets, idx
    raise ValueError(f"cannot unpack batch of type {type(batch)}")


def _as_tensor(mask: ArrayLike, device: Any = None, dtype: Any = None):
    _require_torch()
    if isinstance(mask, torch.Tensor):
        t = mask
    else:
        t = torch.as_tensor(np.asarray(mask))
    if dtype is not None:
        t = t.to(dtype)
    elif not t.is_floating_point():
        t = t.float()
    if device is not None:
        t = t.to(device)
    return t


def per_sample_losses(
    model: Any,
    inputs: Any,
    targets: Any,
    criterion: Any = None,
    device: Any = None,
) -> Any:
    """``l(h(x_i; theta), y_i)`` for a batch, without reduction."""
    if criterion is None or getattr(criterion, "reduction", "mean") != "none":
        criterion = default_criterion(reduction="none", device=device)
    if device is not None:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
    logits = forward_logits(model, inputs)
    return criterion(logits, targets)


def predict_logits(model: Any, loader: Any, device: Any = None) -> np.ndarray:
    """Collect the logits of ``model`` over a whole loader (restores eval mode)."""
    _require_torch()
    was_training = model.training
    model.eval()
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            inputs, _targets, _idx = unpack_batch(batch)
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
            chunks.append(forward_logits(model, inputs).detach().cpu().numpy())
    if was_training:
        model.train()
    if not chunks:
        raise ValueError("loader produced no batches")
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# inner coreset loss  L(m, theta)   (Eq. (1), "s.t." part)
# ---------------------------------------------------------------------------
def inner_coreset_loss(
    mask: ArrayLike,
    model: Any,
    loader: Any = None,
    criterion: Any = None,
    device: Any = None,
    dataset: Any = None,
    batch_size: int = 256,
    num_workers: int = 0,
    max_batches: Optional[int] = None,
    return_count: bool = False,
    create_graph: bool = True,
) -> Union[float, Tuple[float, Any]]:
    """``L(m, theta) = (1/||m||_0) sum_{i=1}^{n} m_i l(h(x_i; theta), y_i)``.

    Two equivalent code paths:

    * if the loader yields the batch index as its third element, the loss is
      formed as ``sum_i m_i l_i / sum_i m_i`` -- exactly the definition above,
      and it also supports relaxed masks ``m in [-1, 1]``;
    * otherwise the loader is assumed to already contain only the selected
      examples (e.g. a ``Subset`` built from ``m``) and a plain mean is taken.

    The paper normalises by ``||m||_0``, i.e. losses are averaged over the
    *selected* examples only, never over the full data.
    """
    _require_torch()
    if criterion is None:
        criterion = default_criterion(reduction="none", device=device)
    elif getattr(criterion, "reduction", "mean") != "none":
        criterion = default_criterion(reduction="none", device=device)

    if loader is None and dataset is not None:
        idx = selected_indices(mask)
        loader = DataLoader(
            Subset(dataset, [int(i) for i in idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    if loader is None:
        raise ValueError("inner_coreset_loss needs either a loader or a dataset")

    total_loss = 0.0
    total_weight = 0.0
    for b_i, batch in enumerate(loader):
        if max_batches is not None and b_i >= max_batches:
            break
        inputs, targets, batch_idx = unpack_batch(batch)
        with torch.no_grad():
            losses = per_sample_losses(model, inputs, targets, criterion, device)
        if batch_idx is not None:
            w = _as_tensor(mask, device=losses.device, dtype=losses.dtype)[
                batch_idx.to(losses.device).long()
            ]
            total_loss += float((w * losses).sum().item())
            total_weight += float(w.sum().item())
        else:
            total_loss += float(losses.sum().item())
            total_weight += float(losses.numel())
    value = total_loss / max(total_weight, 1e-12)
    if return_count:
        return value, total_weight
    return value


# ---------------------------------------------------------------------------
# full-data loss  f_1(m)
# ---------------------------------------------------------------------------
def full_data_loss(
    model: Any,
    loader: Any,
    criterion: Any = None,
    device: Any = None,
    max_batches: Optional[int] = None,
    return_count: bool = False,
) -> Union[float, Tuple[float, Any]]:
    """``(1/n) sum_{i=1}^{n} l(h(x_i; theta(m)), y_i)`` over the **full** data."""
    _require_torch()
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for b_i, batch in enumerate(loader):
            if max_batches is not None and b_i >= max_batches:
                break
            inputs, targets, _idx = unpack_batch(batch)
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            logits = forward_logits(model, inputs)
            loss = nn.functional.cross_entropy(logits, targets, reduction="sum")
            total += float(loss.item())
            count += int(targets.numel())
    if was_training:
        model.train()
    value = total / max(count, 1)
    if return_count:
        return value, count
    return value


def accuracy(model: Any, loader: Any, device: Any = None) -> float:
    """Top-1 accuracy in percent (the metric reported in the paper's tables)."""
    _require_torch()
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets, _idx = unpack_batch(batch)
            if device is not None:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            logits = forward_logits(model, inputs)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
    if was_training:
        model.train()
    return 100.0 * correct / max(total, 1)


def coreset_accuracy(model: Any, dataset: Any, mask: ArrayLike, **kw) -> float:
    """Accuracy of ``theta(m)`` restricted to the selected examples."""
    _require_torch()
    batch_size = kw.pop("batch_size", 256)
    idx = selected_indices(mask)
    loader = DataLoader(Subset(dataset, [int(i) for i in idx]), batch_size=batch_size, shuffle=False)
    return accuracy(model, loader, **kw)


# ---------------------------------------------------------------------------
# objective wrappers
# ---------------------------------------------------------------------------
def f2(mask: ArrayLike) -> float:
    """``f_2(m) := ||m||_0`` (Eq. (2)) -- the coreset size, exact and free."""
    return float(l0_norm(mask))


def f1(theta: Any, loader: Any = None, criterion: Any = None, device: Any = None, **kw) -> float:
    """``f_1(m)`` evaluated with the *already trained* ``theta = theta(m)``.

    Either pass a full-data ``loader``, or use :class:`MaskObjectiveEvaluator`,
    which owns the loaders and caches ``F(m)`` per mask.
    """
    return float(full_data_loss(theta, loader, criterion=criterion, device=device, **kw))


f1_full_data = f1


# ---------------------------------------------------------------------------
# caching / mask keys
# ---------------------------------------------------------------------------
def mask_key(mask: ArrayLike, decimals: int = 6) -> Tuple[str, str]:
    """Stable cache key: exact bit-packing for binary masks, rounded otherwise."""
    arr = np.asarray(mask)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    binary = bool(np.all(np.isin(arr, (0, 1))))
    if binary:
        packed = np.packbits(arr.astype(np.uint8))
        return ("bin", hashlib.md5(packed.tobytes()).hexdigest())
    rounded = np.round(arr.astype(np.float64), decimals)
    return ("cont", hashlib.md5(rounded.tobytes()).hexdigest())


@dataclass
class MaskEvaluation:
    """One point of ``F(m) = [f_1(m), f_2(m)]`` together with its provenance."""

    mask: np.ndarray
    f1: float
    f2: float
    theta: Any = None
    inner_loss: Optional[float] = None
    acc: Optional[float] = None
    cached: bool = False
    wall_time: float = 0.0
    iteration: int = -1
    key: Optional[Tuple[str, str]] = None

    def F(self) -> np.ndarray:
        """``F(m) = [f_1(m), f_2(m)]`` -- lexicographic order, f1 first."""
        return np.array([self.f1, self.f2], dtype=np.float64)

    def __iter__(self):
        return iter((self.f1, self.f2))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "f1": self.f1,
            "f2": self.f2,
            "inner_loss": self.inner_loss,
            "acc": self.acc,
            "cached": self.cached,
            "wall_time": self.wall_time,
            "iteration": self.iteration,
        }


class ObjectiveCache:
    """Memoisation of already-evaluated masks (§3.2: avoid wasting inner-loop training)."""

    def __init__(self, keep_theta: bool = False, max_entries: Optional[int] = None):
        self.keep_theta = keep_theta
        self.max_entries = max_entries
        self._entries: Dict[Tuple[str, str], MaskEvaluation] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, mask: ArrayLike) -> bool:
        return mask_key(mask) in self._entries

    def get(self, mask: ArrayLike) -> Optional[MaskEvaluation]:
        key = mask_key(mask)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        entry.cached = True
        return entry

    def put(self, evaluation: MaskEvaluation) -> MaskEvaluation:
        key = evaluation.key or mask_key(evaluation.mask)
        evaluation.key = key
        if not self.keep_theta:
            evaluation.theta = None
        self._entries[key] = evaluation
        if self.max_entries is not None and len(self._entries) > self.max_entries:
            self._entries.pop(next(iter(self._entries)))  # drop oldest insertion
        return evaluation

    def clear(self) -> None:
        self._entries.clear()
        self.hits = self.misses = 0

    def history(self) -> List[MaskEvaluation]:
        return list(self._entries.values())

    def F_history(self) -> np.ndarray:
        """The historical set ``H`` of Appendix A as an array of shape (|H|, 2)."""
        if not self._entries:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([e.F() for e in self._entries.values()], axis=0)

    def best_by_f1(self) -> Optional[MaskEvaluation]:
        return min(self._entries.values(), key=lambda e: e.f1, default=None)

    def best_lexicographic(self) -> Optional[MaskEvaluation]:
        return min(self._entries.values(), key=lambda e: (e.f1, e.f2), default=None)

    def stats(self) -> Dict[str, Any]:
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# the train-then-evaluate primitive of Algorithm 1
# ---------------------------------------------------------------------------
class MaskObjectiveEvaluator:
    """Algorithm 1 Steps 3-4: ``theta(m) <- argmin L(m, theta)`` then ``F(m)``.

    Parameters
    ----------
    inner_train_fn:
        Callable ``mask -> model`` (or ``mask -> (model, inner_loss)``).  It must
        solve the inner problem, i.e. train the proxy network on the selected
        coreset until convergence.  Implemented by
        :mod:`lbcs_repro.lbcs.bilevel` / :mod:`lbcs_repro.lbcs.acceleration`.
    full_loader:
        DataLoader over the **full** dataset, used for ``f_1`` (Eq. (1)).
    eval_loader:
        Optional separate loader (e.g. the test set) used for the reported ``acc``.
    cache:
        Reuse ``F(m)`` for masks already evaluated; the memo is also the
        historical set ``H`` used by the practical lexicographic relations.
    cache_theta:
        Keep the trained parameters (needed to warm-start the acceleration
        tricks of §3.2 or to hand the model to the target-model harness).
    """

    def __init__(
        self,
        inner_train_fn: Callable[..., Any],
        full_loader: Any = None,
        criterion: Any = None,
        device: Any = None,
        eval_loader: Any = None,
        cache: bool = True,
        cache_theta: bool = False,
        max_cache_entries: Optional[int] = None,
        logger: Any = None,
        track_time: bool = True,
    ):
        self.inner_train_fn = inner_train_fn
        self.full_loader = full_loader
        self.eval_loader = eval_loader if eval_loader is not None else full_loader
        self.criterion = criterion
        self.device = device
        self.cache = (
            ObjectiveCache(keep_theta=cache_theta, max_entries=max_cache_entries)
            if cache
            else None
        )
        self.logger = logger
        self.track_time = track_time
        self.iterations = 0
        self.train_time = 0.0
        self.eval_time = 0.0
        self._history: List[MaskEvaluation] = []

    # -- inner loop --------------------------------------------------------
    def theta(self, mask: ArrayLike, **train_kwargs) -> Any:
        """``theta(m)``: train the proxy network on the coreset (Algorithm 1 Step 3)."""
        result = self.inner_train_fn(mask, **train_kwargs)
        if isinstance(result, tuple):  # (model, inner_loss)
            return result[0]
        return result

    # -- objectives --------------------------------------------------------
    @staticmethod
    def f2(mask: ArrayLike) -> float:
        """``f_2(m) = ||m||_0`` -- requires no training (Eq. (2))."""
        return float(l0_norm(mask))

    def f1(self, theta: Any) -> float:
        """``f_1(m)`` for an already trained network (Eq. (1))."""
        if self.full_loader is None:
            raise ValueError("MaskObjectiveEvaluator needs a full_loader to compute f1")
        return float(
            full_data_loss(theta, self.full_loader, criterion=self.criterion, device=self.device)
        )

    # -- combined evaluation ----------------------------------------------
    def evaluate(
        self,
        mask: ArrayLike,
        force: bool = False,
        return_theta: bool = False,
        iteration: int = -1,
        **train_kwargs,
    ) -> MaskEvaluation:
        """Full ``F(m) = [f_1(m), f_2(m)]`` for one mask (memoised)."""
        mask = np.asarray(mask, dtype=np.float64).reshape(-1)
        if self.cache is not None and not force:
            cached = self.cache.get(mask)
            if cached is not None:
                if self.logger is not None:
                    self.logger.debug(
                        "mask cache hit f1=%.4f f2=%.1f", cached.f1, cached.f2
                    )
                return cached

        t0 = time.time()
        theta = self.theta(mask, **train_kwargs)
        t1 = time.time()
        value_f1 = self.f1(theta)
        value_f2 = self.f2(mask)
        acc = None
        if self.eval_loader is not None:
            acc = accuracy(theta, self.eval_loader, device=self.device)
        t2 = time.time()

        evaluation = MaskEvaluation(
            mask=mask,
            f1=value_f1,
            f2=value_f2,
            theta=theta,
            acc=acc,
            cached=False,
            wall_time=(t2 - t0) if self.track_time else 0.0,
            iteration=iteration,
        )
        self.iterations += 1
        if self.track_time:
            self.train_time += t1 - t0
            self.eval_time += t2 - t1
        self._history.append(evaluation)
        if self.cache is not None:
            self.cache.put(evaluation)  # may clear .theta if not cache_theta
        if not return_theta:
            evaluation.theta = None
        if self.logger is not None:
            self.logger.info(
                "iter %d | f1=%.4f | f2=%.0f | acc=%s",
                iteration,
                value_f1,
                value_f2,
                "n/a" if acc is None else f"{acc:.2f}",
            )
        return evaluation

    def evaluate_many(self, masks: Iterable[ArrayLike], **kw) -> List[MaskEvaluation]:
        return [self.evaluate(m, **kw) for m in masks]

    def F(self, mask: ArrayLike, **kw) -> np.ndarray:
        """``F(m) = [f_1(m), f_2(m)]`` only (trains if not cached)."""
        return self.evaluate(mask, **kw).F()

    # -- bookkeeping -------------------------------------------------------
    @property
    def history(self) -> List[MaskEvaluation]:
        return self._history

    def F_history(self) -> np.ndarray:
        """All evaluated ``F`` values, i.e. the historical set ``H`` of Appendix A."""
        if self.cache is not None and len(self.cache):
            return self.cache.F_history()
        if not self._history:
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([e.F() for e in self._history], axis=0)

    def historical_masks(self) -> List[np.ndarray]:
        if self.cache is not None and len(self.cache):
            return [e.mask for e in self.cache.history()]
        return [e.mask for e in self._history]

    def best_incumbent(self) -> Optional[MaskEvaluation]:
        """Lexicographically best evaluated point (f1 first, then f2)."""
        entries = (
            self.cache.history()
            if (self.cache is not None and len(self.cache))
            else self._history
        )
        if not entries:
            return None
        return min(entries, key=lambda e: (e.f1, e.f2))

    def stats(self) -> Dict[str, Any]:
        out = {
            "evaluations": self.iterations,
            "train_time_s": self.train_time,
            "eval_time_s": self.eval_time,
        }
        if self.cache is not None:
            out.update(self.cache.stats())
        return out

    def log_summary(self) -> None:
        if self.logger is None:
            return
        best = self.best_incumbent()
        self.logger.info("evaluator stats: %s", self.stats())
        if best is not None:
            self.logger.info("best incumbent: f1=%.4f f2=%.0f", best.f1, best.f2)


# ---------------------------------------------------------------------------
# self-test (run: python -m lbcs_repro.lbcs.objectives)
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover
    from .masks import init_binary_mask

    mask = init_binary_mask(100, k=30, seed=0)
    assert f2(mask) == 30.0, f2(mask)
    assert num_selected(mask) == 30
    assert mask_key(mask.copy()) == mask_key(mask)
    assert mask_key(np.clip(mask + 1e-9, -1, 1))[0] == "cont"
    cache = ObjectiveCache()
    cache.put(MaskEvaluation(mask=mask, f1=1.0, f2=30.0))
    assert cache.get(mask) is not None and cache.hits == 1
    assert np.allclose(cache.history()[0].F(), [1.0, 30.0])

    if _TORCH_AVAILABLE:
        # tiny end-to-end check of L(m, theta) and f_1(m) on synthetic data
        torch.manual_seed(0)
        x = torch.randn(64, 4)
        y = (x.sum(dim=1) > 0).long()
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        ds = torch.utils.data.TensorDataset(x, y)
        full_loader = DataLoader(ds, batch_size=16)
        m = init_binary_mask(64, k=16, seed=0)
        idx = selected_indices(m)
        sub_loader = DataLoader(
            Subset(ds, [int(i) for i in idx]), batch_size=16, shuffle=False
        )
        inner = inner_coreset_loss(m, model, loader=sub_loader, device=None)
        full = full_data_loss(model, full_loader, device=None)
        assert np.isfinite(inner) and np.isfinite(full)
        print(f"objectives.py self-test OK (inner={inner:.4f}, f1={full:.4f})")
    else:
        print("objectives.py self-test OK (torch not available)")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
