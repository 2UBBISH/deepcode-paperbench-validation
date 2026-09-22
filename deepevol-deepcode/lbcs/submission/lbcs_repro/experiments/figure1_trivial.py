"""Figure 1 of the LBCS paper: experiments illustrating the *trivial solutions* of
Refined Coreset Selection (RCS).

Paper specification (read back from the original text)
-----------------------------------------------------
* §2.1 eq. (3)::

      min_m f1(m)   s.t.   theta(m) in argmin_theta L(m, theta)

  "In (3), the minimization of f1(m) is in the outer loop, while the
  minimization of L(m, theta) lies in the inner loop.  Without optimizations
  about the coreset size, f1(m) can be minimized effectively (see Figure
  1(a)).  As a comparison, the coreset size remains close to the predefined one
  (see Figure 1(b)), which is not our desideratum in RCS."

* §2.1 eq. (4)::

      min_m (1 - lambda) f1(m) + lambda f2(m)
      s.t.  theta(m) in argmin_theta L(m, theta)

  "if f1(m) and f2(m) share the same weights, i.e. lambda = 1/2, optimization
  does not implicitly favor f1(m).  Instead, the minimization of f2(m) is
  salient, where after all iterations f2(m) is too small and f1(m) is still
  large (see Figures 1(c) and 1(d))."

* Appendix C.3 (settings for the experiments in Figure 1): "we employ a subset
  of MNIST.  A convolutional neural network stacked with two blocks of
  convolution, dropout, max-pooling, and ReLU activation is used.  Following
  (Zhou et al., 2022), for the inner loop, the model is trained for 100 epochs
  using SGD with a learning rate of 0.1 and momentum of 0.9.  For the outer
  loop, the probabilities are optimized by Adam with a learning rate of 2.5 and
  a cosine scheduler."

* Addendum ("Useful details for Figure 1"):
  - lambda = 0.5 for eq. (4);
  - an arbitrarily random subset of MNIST is used (MNIST-S is allowed);
  - the CNN is the ``ConvNet`` class of Zhou et al. (2022);
  - T = 1000 outer iterations.

Produced artefacts
------------------
- ``figure1.png``: the four panels (a) f1 vs. outer iterations with eq. (3),
  (b) f2 with eq. (3), (c) f1 with eq. (4), (d) f2 with eq. (4);
- ``figure1_curves.json``: the raw per-iteration curves + measured summary;
- ``figure1_summary.md``/``.txt``: human readable failure-mode diagnostics.

Nothing outside of the paper's scope (ImageNet-1k §5.4, continual learning
appendix E.5, streaming appendix E.6) is touched here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("lbcs_repro.experiments.figure1")

try:  # torch is a soft dependency: the module must import without it.
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Subset = None  # type: ignore
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paper-stated constants (Appendix C.3 + addendum "Useful details for Figure 1")
# ---------------------------------------------------------------------------
FIGURE1_LAMBDA = 0.5
FIGURE1_OUTER_ITERS = 1000
FIGURE1_OUTER_LR = 2.5
FIGURE1_OUTER_OPTIMIZER = "adam"
FIGURE1_OUTER_SCHEDULER = "cosine"
FIGURE1_INNER_EPOCHS = 100
FIGURE1_INNER_LR = 0.1
FIGURE1_INNER_MOMENTUM = 0.9
FIGURE1_INNER_OPTIMIZER = "sgd"
FIGURE1_DATASET = "MNIST-S"

#: The paper states that "k denotes the predefined coreset size before
#: optimization" but does not give its numeric value; 20% of MNIST-S (the
#: "arbitrarily random subset of MNIST") is used as a suggested default.
SUGGESTED_K = 200
SUGGESTED_MNIST_SUBSET = 1000
SUGGESTED_BATCH_SIZE = 128

#: Reduced-cost defaults used for smoke tests / CI; explicitly *not* paper
#: values (see README).
SUGGESTED_SMOKE_N = 128
SUGGESTED_SMOKE_K = 32
SUGGESTED_SMOKE_OUTER_ITERS = 20
SUGGESTED_SMOKE_INNER_EPOCHS = 2

FORMULATIONS = ("eq3", "eq4")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _import_attr(module_names: Sequence[str], attr: str, default: Any = None) -> Any:
    """Import ``attr`` from the first importable module in ``module_names``."""
    names = list(module_names)
    for name in names:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if hasattr(mod, attr):
            return getattr(mod, attr)
    return default


def _normalize_formulation(formulation: Any) -> str:
    """Map ``"Eq. (3)"`` / ``"eq3"`` / ``3`` to the canonical ``"eq3"``."""
    text = str(formulation).strip().lower()
    text = text.replace("equation", "eq").replace("(", "").replace(")", "")
    text = text.replace(" ", "").replace(".", "").replace("_", "").replace("-", "")
    if text in ("eq3", "3", "o1", "f1only", "f1"):
        return "eq3"
    if text in ("eq4", "4", "o2", "weighted", "combined"):
        return "eq4"
    raise ValueError("unknown formulation {!r} (expected 'eq3' or 'eq4')".format(formulation))


def _as_array(value: Any) -> Optional[np.ndarray]:
    """Best-effort conversion of a scalar/sequence/tensor into a 1-D float array."""
    if value is None:
        return None
    if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):  # pragma: no cover
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        arr = value
    else:
        try:
            arr = np.asarray(list(value) if not np.isscalar(value) else [value], dtype=float)
        except Exception:
            try:
                arr = np.asarray(value, dtype=float)
            except Exception:
                return None
    arr = np.asarray(arr, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)] if arr.size and not np.all(np.isfinite(arr)) else arr
    return arr


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch/dataclass objects into JSON-safe values."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if _TORCH_AVAILABLE and isinstance(obj, torch.Tensor):  # pragma: no cover
        return _to_jsonable(obj.detach().cpu().numpy())
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}  # type: ignore
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def _set_seed(seed: Optional[int]) -> None:
    """Seed numpy / torch (and CUDA) deterministically."""
    if seed is None:
        return
    fn = _import_attr(["lbcs_repro.baselines.base", "lbcs.baselines.base"], "set_seed")
    if callable(fn):
        try:
            fn(seed)
            return
        except Exception:
            pass
    np.random.seed(int(seed) % (2 ** 32 - 1))
    if _TORCH_AVAILABLE:  # pragma: no cover
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return str(device)
    if _TORCH_AVAILABLE and torch.cuda.is_available():  # pragma: no cover
        return "cuda"
    return "cpu"


def binarize_mask(mask: Any) -> np.ndarray:
    """Project any mask convention onto ``{0, 1}^n``.

    Handles the three conventions used in this codebase:

    * ``{0, 1}`` binary masks (identity, thresholded at 0.5);
    * relaxed masks in ``[-1, 1]`` (Appendix A rule: ``[-1, 0) -> 0``,
      ``[0, 1] -> 1``);
    * probability masks in ``[0, 1]`` (thresholded at 0.5).
    """
    if _TORCH_AVAILABLE and isinstance(mask, torch.Tensor):  # pragma: no cover
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask, dtype=float).reshape(-1)
    if m.size == 0:
        return m.astype(np.float64)
    is_binary = bool(np.all((np.abs(m) < 1e-9) | (np.abs(m - 1.0) < 1e-9)))
    if is_binary:
        return (m > 0.5).astype(np.float64)
    if float(m.min()) < -1e-9:  # relaxed [-1, 1]
        return (m >= 0.0).astype(np.float64)
    return (m > 0.5).astype(np.float64)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of the selected examples (the coreset) of ``mask``."""
    return np.flatnonzero(binarize_mask(mask) > 0.5)


# ---------------------------------------------------------------------------
# evaluation container (mirrors ``lbcs.objectives.MaskEvaluation``)
# ---------------------------------------------------------------------------
@dataclass
class Figure1Evaluation:
    """One evaluated outer point ``F(m) = [f1(m), f2(m)]``."""

    mask: Any = None
    f1: float = float("nan")
    f2: float = float("nan")
    theta: Any = None
    inner_loss: float = float("nan")
    acc: float = float("nan")
    cached: bool = False
    wall_time: float = 0.0
    iteration: int = -1
    key: str = ""

    def F(self) -> np.ndarray:  # noqa: N802 - follows the objective API
        return np.asarray([self.f1, self.f2], dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f1": self.f1,
            "f2": self.f2,
            "inner_loss": self.inner_loss,
            "acc": self.acc,
            "cached": self.cached,
            "wall_time": self.wall_time,
            "iteration": self.iteration,
            "key": self.key,
        }


def _make_evaluation(**kwargs: Any) -> Any:
    """Build the core ``MaskEvaluation`` when available, else the local mirror."""
    cls = _import_attr(["lbcs_repro.lbcs.objectives", "lbcs.objectives"], "MaskEvaluation")
    if cls is not None:
        fields = set(getattr(cls, "__dataclass_fields__", {}) or {})
        payload = {k: v for k, v in kwargs.items() if k in fields}
        try:
            return cls(**payload)
        except Exception:
            pass
    return Figure1Evaluation(**kwargs)


# ---------------------------------------------------------------------------
# Figure 1 configuration
# ---------------------------------------------------------------------------
@dataclass
class Figure1Config:
    """All knobs of the Figure 1 reproduction.

    Every default that the paper states explicitly is held in the module-level
    ``FIGURE1_*`` constants and mirrored here; anything else is marked
    SUGGESTED.
    """

    dataset: str = FIGURE1_DATASET
    n: Optional[int] = SUGGESTED_MNIST_SUBSET
    k: int = SUGGESTED_K
    lam: float = FIGURE1_LAMBDA
    outer_iters: int = FIGURE1_OUTER_ITERS
    outer_lr: float = FIGURE1_OUTER_LR
    outer_optimizer: str = FIGURE1_OUTER_OPTIMIZER
    outer_scheduler: str = FIGURE1_OUTER_SCHEDULER
    inner_epochs: int = FIGURE1_INNER_EPOCHS
    inner_lr: float = FIGURE1_INNER_LR
    inner_momentum: float = FIGURE1_INNER_MOMENTUM
    inner_optimizer: str = FIGURE1_INNER_OPTIMIZER
    inner_weight_decay: float = 0.0  # SUGGESTED
    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = 256  # SUGGESTED
    num_workers: int = 0
    architecture: str = "ConvNet"
    root: Optional[str] = None
    download: bool = True
    device: Optional[str] = None
    seed: int = 0
    eval_on_test: bool = False  # SUGGESTED: f1 is measured on the selection pool D
    samples_per_iter: int = 1  # SUGGESTED
    use_size_objective: Optional[bool] = None  # None -> per formulation
    constrain_size: bool = True
    warm_start: bool = False  # SUGGESTED acceleration (§3.2), off for fidelity
    cache_evaluations: bool = True
    use_core_evaluator: bool = True
    formulations: Tuple[str, ...] = FORMULATIONS
    log_every: int = 0
    save_figure: bool = True
    save_json: bool = True
    output_dir: str = os.path.join("results", "figure1")
    figure_format: str = "png"
    tags: Tuple[str, ...] = field(default_factory=tuple)

    # -- presets ---------------------------------------------------------
    @classmethod
    def paper(cls, **overrides: Any) -> "Figure1Config":
        """Paper-faithful setting (Appendix C.3 + addendum Figure-1 details)."""
        cfg = cls(
            dataset=FIGURE1_DATASET,
            n=SUGGESTED_MNIST_SUBSET,
            k=SUGGESTED_K,
            lam=FIGURE1_LAMBDA,
            outer_iters=FIGURE1_OUTER_ITERS,
            outer_lr=FIGURE1_OUTER_LR,
            inner_epochs=FIGURE1_INNER_EPOCHS,
            inner_lr=FIGURE1_INNER_LR,
            inner_momentum=FIGURE1_INNER_MOMENTUM,
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "Figure1Config":
        """Tiny, cheap setting used for validation; NOT a paper setting."""
        cfg = cls(
            dataset=FIGURE1_DATASET,
            n=SUGGESTED_SMOKE_N,
            k=SUGGESTED_SMOKE_K,
            lam=FIGURE1_LAMBDA,
            outer_iters=SUGGESTED_SMOKE_OUTER_ITERS,
            outer_lr=FIGURE1_OUTER_LR,
            inner_epochs=SUGGESTED_SMOKE_INNER_EPOCHS,
            inner_lr=FIGURE1_INNER_LR,
            inner_momentum=FIGURE1_INNER_MOMENTUM,
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Figure1Config":
        cfg = cls()
        if not data:
            return cfg
        known = set(cfg.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(data).items() if k in known}
        if "formulations" in payload and payload["formulations"] is not None:
            payload["formulations"] = tuple(payload["formulations"])
        if "tags" in payload and payload["tags"] is not None:
            payload["tags"] = tuple(payload["tags"])
        for key in ("outer_iters", "T", "iterations"):
            if key != "outer_iters" and key in dict(data):
                payload.setdefault("outer_iters", int(dict(data)[key]))
        for key in ("inner_epochs", "epochs"):
            if key != "inner_epochs" and key in dict(data):
                payload.setdefault("inner_epochs", int(dict(data)[key]))
        if "lambda" in dict(data) and "lam" not in payload:
            payload["lam"] = float(dict(data)["lambda"])
        return cfg.with_overrides(**payload)

    def with_overrides(self, **overrides: Any) -> "Figure1Config":
        known = set(self.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        clean: Dict[str, Any] = {}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in known:
                clean[key] = value
            elif key == "lambda":
                clean["lam"] = value
            elif key == "T":
                clean["outer_iters"] = value
            else:
                LOGGER.debug("Figure1Config: ignoring unknown override %r", key)
        if "formulations" in clean and clean["formulations"] is not None:
            clean["formulations"] = tuple(str(_normalize_formulation(f)) for f in clean["formulations"])
        return replace(self, **clean) if clean else self

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]
        out["formulations"] = list(out["formulations"])
        out["tags"] = list(out["tags"])
        return out


# ---------------------------------------------------------------------------
# Objective: theta(m) <- argmin L(m, theta), F(m) = [f1(m), f2(m)]
# ---------------------------------------------------------------------------
class Figure1Objective:
    """Train-then-evaluate primitive for Figure 1 (Algorithm 1 steps 3-4).

    ``theta(m)`` is obtained by training the proxy network on the coreset
    indicated by the mask with the *paper-stated* inner loop of Appendix C.3
    (SGD, lr = 0.1, momentum = 0.9, 100 epochs).  ``f1(m)`` is the average
    cross-entropy of ``theta(m)`` over the full data ``D`` (eq. (1)) and
    ``f2(m) = ||m||_0`` (eq. (2)).
    """

    def __init__(
        self,
        dataset: Any,
        model_factory: Callable[..., Any],
        eval_dataset: Any = None,
        criterion: Any = None,
        device: Optional[str] = None,
        inner_epochs: int = FIGURE1_INNER_EPOCHS,
        inner_lr: float = FIGURE1_INNER_LR,
        inner_momentum: float = FIGURE1_INNER_MOMENTUM,
        inner_optimizer: str = FIGURE1_INNER_OPTIMIZER,
        weight_decay: float = 0.0,
        batch_size: int = SUGGESTED_BATCH_SIZE,
        eval_batch_size: int = 256,
        num_workers: int = 0,
        seed: int = 0,
        n: Optional[int] = None,
        warm_start: bool = False,
        cache: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.dataset = dataset
        self.eval_dataset = eval_dataset if eval_dataset is not None else dataset
        self.model_factory = model_factory
        self.device = _resolve_device(device)
        self.inner_epochs = int(inner_epochs)
        self.inner_lr = float(inner_lr)
        self.inner_momentum = float(inner_momentum)
        self.inner_optimizer = str(inner_optimizer or "sgd").lower()
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.eval_batch_size = int(eval_batch_size)
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.n = int(n) if n is not None else self._infer_n()
        self.warm_start = bool(warm_start)
        self.cache_enabled = bool(cache)
        self.logger = logger or LOGGER
        self.criterion = criterion if criterion is not None else self._default_criterion()
        self._cache: Dict[Tuple[int, ...], Any] = {}
        self._last_model: Any = None
        self.num_evaluations = 0

    # -- helpers ---------------------------------------------------------
    def _default_criterion(self) -> Any:
        if _TORCH_AVAILABLE:
            return nn.CrossEntropyLoss()
        fn = _import_attr(["lbcs_repro.lbcs.objectives", "lbcs.objectives"], "default_criterion")
        return fn() if callable(fn) else None

    def _infer_n(self) -> int:
        try:
            return int(len(self.dataset))
        except Exception:
            return 0

    @staticmethod
    def _unpack(batch: Any) -> Tuple[Any, Any]:
        if isinstance(batch, (list, tuple)):
            return batch[0], batch[1]
        return batch, None  # pragma: no cover - defensive

    def _model_key(self, indices: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(i) for i in np.sort(np.asarray(indices).reshape(-1)))

    def binarize(self, mask: Any) -> np.ndarray:
        return binarize_mask(mask)

    def coreset_indices(self, mask: Any) -> np.ndarray:
        return mask_indices(mask)

    def f2(self, mask: Any) -> float:
        return float(np.count_nonzero(binarize_mask(mask) > 0.5))

    # -- inner loop ------------------------------------------------------
    def theta(
        self,
        mask: Any,
        epochs: Optional[int] = None,
        seed: Optional[int] = None,
        model: Any = None,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
        batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Return ``theta(m)``: the proxy network trained on the coreset."""
        if not _TORCH_AVAILABLE:  # pragma: no cover - numpy-only environment
            raise RuntimeError("Figure1Objective.theta requires PyTorch to be installed")

        indices = self.coreset_indices(mask)
        if indices.size == 0:
            raise ValueError("empty coreset: mask selects no example")

        epochs = int(self.inner_epochs if epochs is None else epochs)
        lr = float(self.inner_lr if lr is None else lr)
        momentum = float(self.inner_momentum if momentum is None else momentum)
        batch_size = int(self.batch_size if batch_size is None else batch_size)
        seed = int(self.seed if seed is None else seed)

        key = self._model_key(indices)
        if model is None and self.warm_start and self._last_model is not None:
            import copy as _copy

            try:
                model = _copy.deepcopy(self._last_model)
            except Exception:  # pragma: no cover
                model = None

        if model is None:
            model = self.model_factory()
        model = model.to(self.device)
        model.train()

        subset = Subset(self.dataset, [int(i) for i in indices])
        generator = None
        if _TORCH_AVAILABLE:
            generator = torch.Generator()
            generator.manual_seed(seed % (2 ** 31 - 1))
        loader = DataLoader(
            subset,
            batch_size=min(batch_size, max(1, indices.size)),
            shuffle=True,
            num_workers=self.num_workers,
            generator=generator,
            drop_last=False,
        )

        params = [p for p in model.parameters() if p.requires_grad]
        opt_name = self.inner_optimizer
        if opt_name in ("sgd", "momentum"):
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=self.weight_decay)
        elif opt_name in ("adam", "adamw"):
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        else:  # pragma: no cover - defensive
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=self.weight_decay)

        for _epoch in range(max(0, epochs)):
            for batch in loader:
                inputs, targets = self._unpack(batch)
                if targets is None:  # pragma: no cover
                    continue
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = self.criterion(logits, targets)
                loss.backward()
                optimizer.step()

        model.eval()
        self._last_model = model
        self._last_key = key  # type: ignore[attr-defined]
        return model

    # -- evaluation ------------------------------------------------------
    def mean_loss(self, model: Any, dataset: Any = None, max_batches: Optional[int] = None) -> float:
        """Average cross-entropy of ``model`` over (a subset of) the data."""
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("Figure1Objective.mean_loss requires PyTorch")
        dataset = self.eval_dataset if dataset is None else dataset
        generator = torch.Generator()
        generator.manual_seed(self.seed % (2 ** 31 - 1))
        loader = DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            generator=generator,
            drop_last=False,
        )
        model.to(self.device)
        model.eval()
        total_loss, total_count = 0.0, 0
        was_training = model.training
        with torch.no_grad():
            for step, batch in enumerate(loader):
                if max_batches is not None and step >= int(max_batches):
                    break
                inputs, targets = self._unpack(batch)
                if targets is None:  # pragma: no cover
                    continue
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = model(inputs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = self.criterion(logits, targets)
                count = int(targets.shape[0])
                total_loss += float(loss.detach().cpu()) * count
                total_count += count
        if was_training:
            model.train()
        return float(total_loss / max(1, total_count))

    def accuracy(self, model: Any, dataset: Any = None) -> float:
        """Top-1 accuracy (percent) of ``model`` (diagnostic only)."""
        if not _TORCH_AVAILABLE:  # pragma: no cover
            return float("nan")
        dataset = self.eval_dataset if dataset is None else dataset
        loader = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)
        model.to(self.device)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack(batch)
                if targets is None:  # pragma: no cover
                    continue
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = model(inputs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                pred = logits.argmax(dim=1)
                correct += int((pred == targets).sum().item())
                total += int(targets.shape[0])
        return 100.0 * correct / max(1, total)

    def f1(self, theta: Any, mask: Any = None) -> float:
        """``f1(m)``: average full-data loss of ``theta(m)`` (eq. (1))."""
        if theta is None:
            return float("nan")
        return self.mean_loss(theta)

    def evaluate(
        self,
        mask: Any,
        force: bool = False,
        return_theta: bool = False,
        iteration: int = -1,
        epochs: Optional[int] = None,
        **train_kwargs: Any,
    ) -> Any:
        """Evaluate ``F(m) = [f1(m), f2(m)]`` (Algorithm 1 step 4)."""
        t0 = time.time()
        indices = self.coreset_indices(mask)
        key = self._model_key(indices)
        if self.cache_enabled and not force and key in self._cache:
            cached = self._cache[key]
            self.num_evaluations += 1
            return cached

        theta = self.theta(mask, epochs=epochs, seed=self.seed + max(0, int(iteration)), **train_kwargs)
        inner_loss = self.mean_loss(theta, dataset=Subset(self.dataset, [int(i) for i in indices]))
        f1_value = self.f1(theta)
        f2_value = self.f2(mask)
        acc = self.accuracy(theta)
        evaluation = _make_evaluation(
            mask=np.asarray(binarize_mask(mask)),
            f1=float(f1_value),
            f2=float(f2_value),
            theta=theta if return_theta else None,
            inner_loss=float(inner_loss),
            acc=float(acc),
            cached=False,
            wall_time=float(time.time() - t0),
            iteration=int(iteration),
            key=str(key),
        )
        if self.cache_enabled:
            self._cache[key] = evaluation
            self.num_evaluations += 1
        return evaluation

    def __call__(self, mask: Any, **kwargs: Any) -> Any:
        return self.evaluate(mask, **kwargs)

    # -- API parity with ``MaskObjectiveEvaluator`` ----------------------
    def F(self, mask: Any, **kwargs: Any) -> np.ndarray:  # noqa: N802
        return np.asarray(self.evaluate(mask, **kwargs).F(), dtype=float)

    def historical_masks(self) -> List[Any]:
        return [getattr(ev, "mask", None) for ev in self._cache.values()]

    def stats(self) -> Dict[str, Any]:
        return {"num_evaluations": self.num_evaluations, "cache_size": len(self._cache), "n": self.n}

    def log_summary(self) -> Dict[str, Any]:  # pragma: no cover - convenience
        info = self.stats()
        self.logger.info("Figure1Objective: %s", info)
        return info


class _ObjectiveWithFallback:
    """Delegate to the core evaluator; switch to the self-contained one on error."""

    def __init__(self, primary: Any = None, fallback: Any = None, logger: Optional[logging.Logger] = None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.logger = logger or LOGGER
        self._use_primary = primary is not None
        self.switched = False

    def evaluate(self, mask: Any, **kwargs: Any) -> Any:
        if self._use_primary:
            try:
                return self.primary.evaluate(mask, **kwargs)
            except Exception as exc:  # pragma: no cover - integration safety net
                self.logger.warning(
                    "core MaskObjectiveEvaluator failed (%s); falling back to Figure1Objective", exc
                )
                self._use_primary = False
                self.switched = True
        if self.fallback is None:
            raise RuntimeError("no usable objective evaluator")
        return self.fallback.evaluate(mask, **kwargs)

    def __call__(self, mask: Any, **kwargs: Any) -> Any:
        return self.evaluate(mask, **kwargs)

    def F(self, mask: Any, **kwargs: Any) -> np.ndarray:  # noqa: N802
        return np.asarray(self.evaluate(mask, **kwargs).F(), dtype=float)

    # -- shared (mask-level) helpers, always served by the fallback -------
    @property
    def n(self) -> int:
        return int(getattr(self.fallback, "n", 0) or getattr(self.primary, "n", 0) or 0)

    def binarize(self, mask: Any) -> np.ndarray:
        return binarize_mask(mask)

    def coreset_indices(self, mask: Any) -> np.ndarray:
        return mask_indices(mask)

    def f2(self, mask: Any) -> float:
        return float(np.count_nonzero(binarize_mask(mask) > 0.5))

    def f1(self, theta: Any, mask: Any = None) -> float:
        if hasattr(self.primary, "f1"):
            try:
                return float(self.primary.f1(theta))  # type: ignore[attr-defined]
            except Exception:
                pass
        if self.fallback is not None:
            return float(self.fallback.f1(theta))
        return float("nan")

    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"fallback_used": bool(self.switched)}
        for name, obj in (("primary", self.primary), ("fallback", self.fallback)):
            if obj is not None and hasattr(obj, "stats"):
                try:
                    out[name] = obj.stats()
                except Exception:
                    pass
        return out


# ---------------------------------------------------------------------------
# data / model preparation
# ---------------------------------------------------------------------------
def _build_model_factory(config: Figure1Config, logger: logging.Logger) -> Callable[..., Any]:
    name = str(config.architecture or "ConvNet")
    lowered = name.strip().lower()
    if lowered in ("convnet", "conv", "cnn", "znet", "zhou"):
        factory = _import_attr(["lbcs_repro.models.convnet", "lbcs.models.convnet"], "convnet_factory")
        if callable(factory):
            return factory()
        builder = _import_attr(["lbcs_repro.models.convnet", "lbcs.models.convnet"], "build_convnet")
        if callable(builder):
            return lambda **kw: builder(**kw)
        builder = _import_attr(["lbcs_repro.models"], "build_model")
        if callable(builder):
            return lambda **kw: builder("ConvNet", **kw)
        raise RuntimeError("ConvNet (models/convnet.py) is not importable")
    model_factory = _import_attr(["lbcs_repro.models", "lbcs.models"], "model_factory")
    if callable(model_factory):
        logger.info("using architecture %r from the model registry", name)
        return model_factory(name)
    raise RuntimeError("unknown architecture {!r} and no model registry available".format(name))


def _build_datasets(
    config: Figure1Config, logger: logging.Logger
) -> Tuple[Any, Any, int]:
    """Return ``(selection_pool, eval_dataset, n)`` for Figure 1."""
    subset_dataset = _import_attr(
        ["lbcs_repro.data.datasets", "lbcs.data.datasets"], "subset_dataset"
    )
    name = str(config.dataset)
    is_mnist_s = name.strip().lower() in ("mnist-s", "mnist_s", "mnists", "mnist–s")
    target_n = int(config.n) if config.n else None

    if is_mnist_s:
        get_mnist_s = _import_attr(["lbcs_repro.data.mnist_s", "lbcs.data.mnist_s"], "get_mnist_s")
        if not callable(get_mnist_s):
            get_mnist_s = _import_attr(["lbcs_repro.data", "lbcs.data"], "get_mnist_s")
        if not callable(get_mnist_s):
            raise RuntimeError("MKIST-S builder (data/mnist_s.py) is not importable")
        train_ds = get_mnist_s(root=config.root, download=config.download, train=True)
        test_ds = get_mnist_s(root=config.root, download=config.download, train=False)
    else:
        get_dataset = _import_attr(["lbcs_repro.data.datasets", "lbcs.data.datasets"], "get_dataset")
        if not callable(get_dataset):
            get_dataset = _import_attr(["lbcs_repro.data", "lbcs.data"], "get_dataset")
        if not callable(get_dataset):
            raise RuntimeError("dataset registry (data/datasets.py) is not importable")
        train_ds = get_dataset(name, train=True, root=config.root, download=config.download)
        test_ds = get_dataset(name, train=False, root=config.root, download=config.download)

    n_available = len(train_ds)
    if target_n is not None and target_n < n_available and callable(subset_dataset):
        train_ds = subset_dataset(train_ds, np.arange(target_n))
        n_available = target_n
    elif target_n is not None and target_n < n_available:
        train_ds = Subset(train_ds, list(range(target_n)))
        n_available = target_n
    logger.info(
        "Figure 1 selection pool: %s with %d examples (eval set: %s)",
        name,
        n_available,
        "full test split" if config.eval_on_test else "the pool itself",
    )
    eval_ds = test_ds if config.eval_on_test else train_ds
    return train_ds, eval_ds, n_available


def build_figure1_objective(
    config: Figure1Config,
    dataset: Any = None,
    eval_dataset: Any = None,
    model_factory: Callable[..., Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Build the ``theta(m) -> F(m)`` evaluator used by the outer loops."""
    logger = logger or LOGGER
    if dataset is None or model_factory is None:
        dataset, eval_ds, n = _build_datasets(config, logger)
        eval_dataset = eval_ds if eval_dataset is None else eval_dataset
        config = config.with_overrides(n=n)
    objective = Figure1Objective(
        dataset=dataset,
        eval_dataset=eval_dataset,
        model_factory=model_factory,
        device=config.device,
        inner_epochs=config.inner_epochs,
        inner_lr=config.inner_lr,
        inner_momentum=config.inner_momentum,
        inner_optimizer=config.inner_optimizer,
        weight_decay=config.inner_weight_decay,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        n=config.n,
        warm_start=config.warm_start,
        cache=config.cache_evaluations,
        logger=logger,
    )
    if not config.use_core_evaluator:
        return objective

    # Optional integration with the shared LBCS machinery (lbcs/objectives.py).
    try:  # pragma: no cover - exercised only when the core modules import cleanly
        InnerTrainConfig = _import_attr(["lbcs_repro.lbcs.bilevel", "lbcs.bilevel"], "InnerTrainConfig")
        InnerTrainer = _import_attr(["lbcs_repro.lbcs.bilevel", "lbcs.bilevel"], "InnerTrainer")
        make_inner_train_fn = _import_attr(["lbcs_repro.lbcs.bilevel", "lbcs.bilevel"], "make_inner_train_fn")
        MaskObjectiveEvaluator = _import_attr(
            ["lbcs_repro.lbcs.objectives", "lbcs.objectives"], "MaskObjectiveEvaluator"
        )
        if all(x is not None for x in (InnerTrainConfig, InnerTrainer, make_inner_train_fn, MaskObjectiveEvaluator)):
            inner_cfg = InnerTrainConfig(
                optimizer=config.inner_optimizer,
                lr=config.inner_lr,
                momentum=config.inner_momentum,
                epochs=config.inner_epochs,
                batch_size=config.batch_size,
                weight_decay=config.inner_weight_decay,
                device=config.device,
            )
            trainer = InnerTrainer(inner_cfg, device=config.device)
            train_fn = make_inner_train_fn(trainer, model_factory, dataset=dataset, n=config.n)
            primary = MaskObjectiveEvaluator(
                train_fn,
                full_loader=None,
                eval_loader=None,
                dataset=dataset,
                eval_dataset=eval_dataset,
                device=config.device,
                cache=config.cache_evaluations,
            ) if "dataset" in (getattr(MaskObjectiveEvaluator.__init__, "__code__", None).co_varnames or ()) else MaskObjectiveEvaluator(
                train_fn, device=config.device, cache=config.cache_evaluations
            )
            logger.info("Figure 1 objective: core MaskObjectiveEvaluator (with fallback)")
            return _ObjectiveWithFallback(primary=primary, fallback=objective, logger=logger)
    except Exception as exc:  # pragma: no cover - integration safety net
        logger.debug("core evaluator unavailable (%s); using Figure1Objective", exc)
    return objective


# ---------------------------------------------------------------------------
# outer loops (eq. (3) and eq. (4))
# ---------------------------------------------------------------------------
def run_formulation(
    objective: Any,
    formulation: str,
    config: Figure1Config,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Run one trivial formulation (``eq3`` or ``eq4``) of §2.1."""
    logger = logger or LOGGER
    form = _normalize_formulation(formulation)
    n = int(config.n or objective.n)
    use_size = (form == "eq4")

    # Preferred path: the shared weighted-bilevel driver (delegates to
    # baselines/probabilistic.py, the Zhou et al. 2022 machinery).
    trivial_curves = _import_attr(
        ["lbcs_repro.baselines.weighted_bilevel", "lbcs.baselines.weighted_bilevel"], "trivial_bilevel_curves"
    )
    if callable(trivial_curves):
        try:
            logger.info(
                "running %s via baselines.weighted_bilevel (T=%d, k=%d, lambda=%.2f)",
                form,
                config.outer_iters,
                config.k,
                config.lam,
            )
            return trivial_curves(
                objective,
                formulation=form,
                n=n,
                k=int(config.k),
                lambda_=float(config.lam),
                outer_iters=int(config.outer_iters),
                outer_lr=float(config.outer_lr),
                train_kwargs=None,
                seed=int(config.seed),
                cosine=str(config.outer_scheduler) == "cosine",
                optimizer=str(config.outer_optimizer),
                samples_per_iter=int(config.samples_per_iter),
                constrain_size=bool(config.constrain_size),
            )
        except Exception as exc:  # pragma: no cover - driver fallback
            logger.warning("weighted_bilevel driver failed for %s (%s); using probabilistic driver", form, exc)

    probabilistic_bilevel = _import_attr(
        ["lbcs_repro.baselines.probabilistic", "lbcs.baselines.probabilistic"], "probabilistic_bilevel"
    )
    if not callable(probabilistic_bilevel):  # pragma: no cover
        raise RuntimeError("neither baselines.weighted_bilevel nor baselines.probabilistic is importable")
    logger.info(
        "running %s via baselines.probabilistic (T=%d, k=%d, lambda=%.2f)",
        form,
        config.outer_iters,
        config.k,
        config.lam,
    )
    return probabilistic_bilevel(
        objective,
        n=n,
        k=int(config.k),
        lambda_=float(config.lam),
        use_size_objective=bool(use_size if config.use_size_objective is None else config.use_size_objective),
        outer_iters=int(config.outer_iters),
        outer_lr=float(config.outer_lr),
        cosine=str(config.outer_scheduler) == "cosine",
        samples_per_iter=int(config.samples_per_iter),
        constrain_size=bool(config.constrain_size) if form == "eq3" else False,
        seed=int(config.seed),
    )


def _series_from(obj: Any, key: str) -> Optional[np.ndarray]:
    """Extract a per-iteration series (``f1`` or ``f2``) from a result object."""
    if obj is None:
        return None
    candidates: List[np.ndarray] = []

    if isinstance(obj, dict):
        for cand_key in (key, key.upper(), key + "_history", key + "_curve", "history"):
            if cand_key in obj and obj[cand_key] is not None:
                arr = _as_array(obj[cand_key])
                if arr is None and cand_key == "history":
                    arr = _history_series(obj[cand_key], key)
                if arr is not None and arr.size:
                    candidates.append(arr)
    else:
        for attr in (key, key + "_history", key + "_curve"):
            arr = _as_array(getattr(obj, attr, None))
            if arr is not None and arr.size:
                candidates.append(arr)
        hist = getattr(obj, "history", None)
        if hist is not None:
            arr = _as_array(hist) if not isinstance(hist, (list, tuple)) else _history_series(hist, key)
            if arr is not None and arr.size:
                candidates.append(arr)
        curve_fn = getattr(obj, "curve", None)
        if callable(curve_fn):
            try:
                arr = _as_array(curve_fn(key))
                if arr is not None and arr.size:
                    candidates.append(arr)
            except Exception:
                pass

    if not candidates:
        return None
    # Prefer the longest series (a genuine per-iteration curve) when available.
    longest = max(candidates, key=lambda a: a.size)
    return longest


def _history_series(history: Any, key: str) -> Optional[np.ndarray]:
    if not isinstance(history, (list, tuple)) or not history:
        return None
    idx = 0 if key.lower().startswith("f1") else 1
    values: List[float] = []
    for entry in history:
        if isinstance(entry, dict):
            found = None
            for cand in (key, key.upper()):
                if cand in entry:
                    found = entry[cand]
                    break
            if found is None:
                return None
            values.append(float(found))
        elif isinstance(entry, (list, tuple, np.ndarray)):
            arr = np.asarray(entry, dtype=float).reshape(-1)
            if arr.size <= idx:
                return None
            values.append(float(arr[idx]))
        else:
            values.append(float(entry))
    return np.asarray(values, dtype=float)


def curves_from_result(result: Any) -> Dict[str, List[float]]:
    """Return ``{"f1": [...], "f2": [...], "iterations": [...]}`` for a run."""
    curves: Dict[str, List[float]] = {}
    for key in ("f1", "f2"):
        arr = _series_from(result, key)
        curves[key] = [] if arr is None else [float(v) for v in np.asarray(arr).reshape(-1)]
    length = max(len(curves["f1"]), len(curves["f2"]), 1)
    if not curves["f1"] and curves["f2"]:
        curves["f1"] = [float("nan")] * len(curves["f2"])
    if not curves["f2"] and curves["f1"]:
        curves["f2"] = [float("nan")] * len(curves["f1"])
    curves["iterations"] = [int(i) for i in range(max(len(curves["f1"]), len(curves["f2"])) or length)]
    return curves


def _final_value(result: Any, curves: Dict[str, List[float]], key: str) -> float:
    for attr in ("final_" + key, "best_" + key):
        value = getattr(result, attr, None)
        if isinstance(value, (int, float, np.floating)) and np.isfinite(float(value)):
            return float(value)
    series = curves.get(key) or []
    finite = [v for v in series if np.isfinite(v)]
    return float(finite[-1]) if finite else float("nan")


# ---------------------------------------------------------------------------
# diagnostics / plotting / reporting
# ---------------------------------------------------------------------------
def summarize_figure1(results: Dict[str, Any], config: Figure1Config) -> Dict[str, Any]:
    """Measure the two failure modes described in §2.1 and Figure 1."""
    summary: Dict[str, Any] = {"k": int(config.k), "formulations": {}}
    eq3 = results.get("curves", {}).get("eq3")
    eq4 = results.get("curves", {}).get("eq4")

    if eq3:
        f1 = np.asarray(eq3.get("f1", []), dtype=float)
        f2 = np.asarray(eq3.get("f2", []), dtype=float)
        summary["formulations"]["eq3"] = {
            "f1_initial": _first_finite(f1),
            "f1_final": _last_finite(f1),
            "f2_initial": _first_finite(f2),
            "f2_final": _last_finite(f2),
            "f2_mean": float(np.nanmean(f2)) if f2.size else float("nan"),
            "f2_final_over_k": _safe_ratio(_last_finite(f2), float(config.k)),
            # Figure 1(a)/(b): f1 is minimized while the size stays ~ k.
            "fixed_size_failure": bool(f2.size and abs(_last_finite(f2) - float(config.k)) <= 0.10 * max(1.0, float(config.k))),
            "f1_improved": bool(f1.size > 1 and np.isfinite(f1[0]) and np.isfinite(f1[-1]) and f1[-1] < f1[0]),
        }
    if eq4:
        f1 = np.asarray(eq4.get("f1", []), dtype=float)
        f2 = np.asarray(eq4.get("f2", []), dtype=float)
        summary["formulations"]["eq4"] = {
            "f1_initial": _first_finite(f1),
            "f1_final": _last_finite(f1),
            "f2_initial": _first_finite(f2),
            "f2_final": _last_finite(f2),
            "f2_min": float(np.nanmin(f2)) if f2.size else float("nan"),
            "f2_final_over_k": _safe_ratio(_last_finite(f2), float(config.k)),
            # Figure 1(c)/(d): f2 becomes "too small" and f1 stays "large".
            "over_minimized_failure": bool(f2.size and _last_finite(f2) < 0.75 * float(config.k)),
            "f2_shrunk": bool(f2.size > 1 and np.isfinite(f2[0]) and np.isfinite(f2[-1]) and f2[-1] < f2[0]),
        }
    if eq3 and eq4:
        summary["comparison"] = {
            "f1_eq3_final": summary["formulations"]["eq3"]["f1_final"],
            "f1_eq4_final": summary["formulations"]["eq4"]["f1_final"],
            "f2_eq3_final": summary["formulations"]["eq3"]["f2_final"],
            "f2_eq4_final": summary["formulations"]["eq4"]["f2_final"],
            "eq4_f1_worse_than_eq3": bool(
                np.isfinite(summary["formulations"]["eq3"]["f1_final"])
                and np.isfinite(summary["formulations"]["eq4"]["f1_final"])
                and summary["formulations"]["eq4"]["f1_final"] > summary["formulations"]["eq3"]["f1_final"]
            ),
            "eq4_size_smaller_than_eq3": bool(
                np.isfinite(summary["formulations"]["eq3"]["f2_final"])
                and np.isfinite(summary["formulations"]["eq4"]["f2_final"])
                and summary["formulations"]["eq4"]["f2_final"] < summary["formulations"]["eq3"]["f2_final"]
            ),
        }

    # cross-check with the shared failure-mode diagnosis helper
    diagnose = _import_attr(
        ["lbcs_repro.baselines.weighted_bilevel", "lbcs.baselines.weighted_bilevel"], "diagnose_failure_mode"
    )
    if callable(diagnose):
        for form_key in ("eq3", "eq4"):
            res = results.get("results", {}).get(form_key)
            if res is None:
                continue
            try:
                summary.setdefault("shared_diagnostics", {})[form_key] = _to_jsonable(
                    diagnose(res, fixed_size_tol=0.10, shrink_threshold=0.75)
                )
            except Exception:  # pragma: no cover - diagnostic only
                pass
    return summary


def _first_finite(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)] if arr.size else arr
    return float(finite[0]) if finite.size else float("nan")


def _last_finite(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)] if arr.size else arr
    return float(finite[-1]) if finite.size else float("nan")


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return float("nan")
    return float(num / den)


def plot_figure1(results: Dict[str, Any], out_path: str, config: Figure1Config, dpi: int = 150) -> Optional[str]:
    """Draw the four panels of Figure 1: (a)-(b) eq. (3), (c)-(d) eq. (4)."""
    try:  # matplotlib is optional
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting optional
        LOGGER.warning("matplotlib unavailable (%s); skipping figure", exc)
        return None

    curves = results.get("curves", {})
    eq3, eq4 = curves.get("eq3"), curves.get("eq4")
    panels = [
        (eq3, "f1", "(a) eq. (3): $f_1(m)$ vs. outer iterations", False),
        (eq3, "f2", "(b) eq. (3): $f_2(m)$ vs. outer iterations", True),
        (eq4, "f1", "(c) eq. (4): $f_1(m)$ vs. outer iterations", False),
        (eq4, "f2", "(d) eq. (4): $f_2(m)$ vs. outer iterations", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    for ax, (curve, key, title, show_k) in zip(axes.reshape(-1), panels):
        if curve and curve.get(key):
            values = np.asarray(curve[key], dtype=float)
            iters = np.arange(values.size)
            ax.plot(iters, values, color="#1f77b4", lw=1.4, label=key)
            if show_k:
                ax.axhline(float(config.k), color="#d62728", ls="--", lw=1.1, label="predefined $k$")
                ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("outer iterations")
        ax.set_ylabel("$f_1(m)$" if key == "f1" else "$f_2(m)$")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Figure 1 reproduction - trivial solutions of RCS (MNIST subset, ConvNet, "
        "$T={}$, $k={}$, $\\lambda={}$)".format(config.outer_iters, config.k, config.lam),
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _format_summary_text(results: Dict[str, Any], config: Figure1Config) -> str:
    lines = [
        "Figure 1 reproduction (trivial solutions of RCS, §2.1)",
        "=" * 62,
        "paper setting: inner loop SGD lr=0.1 momentum=0.9 epochs=100 (Appendix C.3);",
        "               outer loop Adam lr=2.5 + cosine scheduler; T=1000; lambda=0.5",
        "run setting:   dataset={} n={} k={} T={} inner_epochs={} lambda={}".format(
            config.dataset, config.n, config.k, config.outer_iters, config.inner_epochs, config.lam
        ),
        "",
    ]
    summary = results.get("summary", {})
    for form in ("eq3", "eq4"):
        info = summary.get("formulations", {}).get(form)
        if not info:
            continue
        lines.append("[{}]".format(form))
        lines.append(
            "  f1: {:.4f} -> {:.4f} | f2: {:.4f} -> {:.4f} (predefined k={})".format(
                info.get("f1_initial", float("nan")),
                info.get("f1_final", float("nan")),
                info.get("f2_initial", float("nan")),
                info.get("f2_final", float("nan")),
                config.k,
            )
        )
        if form == "eq3":
            lines.append(
                "  fixed-size failure mode (f2 stays close to k): {}".format(
                    info.get("fixed_size_failure")
                )
            )
        else:
            lines.append(
                "  over-minimization failure mode (f2 << k while f1 stays large): {}".format(
                    info.get("over_minimized_failure")
                )
            )
        lines.append("")
    comp = summary.get("comparison")
    if comp:
        lines.append("expected qualitative behaviour (Figure 1(c)/(d) vs (a)/(b)):")
        lines.append("  eq.(4) f1 worse than eq.(3): {}".format(comp.get("eq4_f1_worse_than_eq3")))
        lines.append("  eq.(4) size smaller than eq.(3): {}".format(comp.get("eq4_size_smaller_than_eq3")))
        lines.append("")
    lines.append("NOTE: the paper does not state k for Figure 1; every value that is not")
    lines.append("stated in the paper is exposed in Figure1Config and labelled SUGGESTED.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------
def run_figure1(
    config: Optional[Figure1Config] = None,
    dataset: Any = None,
    eval_dataset: Any = None,
    model_factory: Callable[..., Any] = None,
    logger: Optional[logging.Logger] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Reproduce Figure 1 (eq. (3) vs. eq. (4) failure modes).

    Returns a dictionary with the per-iteration ``curves`` for both
    formulations, the failure-mode ``summary``, and the written artefact paths.
    """
    logger = logger or LOGGER
    if config is None:
        config = Figure1Config.from_dict(overrides)
    elif overrides:
        config = config.with_overrides(**overrides)

    t_start = time.time()
    _set_seed(config.seed)
    if config.device is None:
        config = config.with_overrides(device=_resolve_device(None))

    if dataset is None or model_factory is None:
        dataset, eval_ds, n = _build_datasets(config, logger)
        eval_dataset = eval_ds if eval_dataset is None else eval_dataset
        config = config.with_overrides(n=n)
    if model_factory is None:
        model_factory = _build_model_factory(config, logger)

    objective = build_figure1_objective(
        config, dataset=dataset, eval_dataset=eval_dataset, model_factory=model_factory, logger=logger
    )

    results: Dict[str, Any] = {
        "experiment": "figure1",
        "config": config.to_dict(),
        "curves": {},
        "results": {},
        "formulation_meta": {},
        "wall_time": 0.0,
    }
    for form in config.formulations:
        form = _normalize_formulation(form)
        t0 = time.time()
        result = run_formulation(objective, form, config, logger=logger)
        curves = curves_from_result(result)
        results["curves"][form] = curves
        results["results"][form] = result
        results["formulation_meta"][form] = {
            "final_f1": _final_value(result, curves, "f1"),
            "final_f2": _final_value(result, curves, "f2"),
            "final_size": _final_value(result, curves, "f2"),
            "k": int(config.k),
            "lambda": float(config.lam),
            "outer_iters": int(config.outer_iters),
            "wall_time": float(time.time() - t0),
        }
        logger.info(
            "%s finished: f1 %.4f -> %.4f | f2 %.4f -> %.4f (%.1fs)",
            form,
            curves["f1"][0] if curves["f1"] else float("nan"),
            curves["f1"][-1] if curves["f1"] else float("nan"),
            curves["f2"][0] if curves["f2"] else float("nan"),
            curves["f2"][-1] if curves["f2"] else float("nan"),
            time.time() - t0,
        )

    results["summary"] = summarize_figure1(results, config)
    results["objective_stats"] = _to_jsonable(
        objective.stats() if hasattr(objective, "stats") else {}
    )

    # ---- persist artefacts
    paths: Dict[str, Any] = {}
    out_dir = config.output_dir
    if config.save_json or config.save_figure:
        os.makedirs(out_dir, exist_ok=True)
    if config.save_json:
        json_path = os.path.join(out_dir, "figure1_curves.json")
        payload = {
            "experiment": "figure1",
            "config": _to_jsonable(config.to_dict()),
            "curves": _to_jsonable(results["curves"]),
            "formulation_meta": _to_jsonable(results["formulation_meta"]),
            "summary": _to_jsonable(results["summary"]),
            "objective_stats": results["objective_stats"],
            "wall_time": float(time.time() - t_start),
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        paths["json"] = json_path
        logger.info("wrote %s", json_path)
    if config.save_figure:
        fig_path = os.path.join(out_dir, "figure1." + str(config.figure_format).lstrip("."))
        saved = plot_figure1(results, fig_path, config)
        if saved:
            paths["figure"] = saved
            logger.info("wrote %s", saved)
    text = _format_summary_text(results, config)
    text_path = os.path.join(out_dir, "figure1_summary.txt")
    try:
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        paths["summary"] = text_path
    except Exception as exc:  # pragma: no cover
        logger.debug("could not write summary text (%s)", exc)

    results["paths"] = paths
    results["wall_time"] = float(time.time() - t_start)
    # results dict keeps the live result objects for in-process inspection
    logger.info("\n" + text)
    return results


#: alias so ``experiments.run_experiment("figure1")`` finds an entrypoint
run = run_figure1
figure1_trivial = run_figure1


# ---------------------------------------------------------------------------
# offline self-test (synthetic objective, no torch / no dataset download)
# ---------------------------------------------------------------------------
class _SyntheticObjective:
    """Cheap synthetic objective used to exercise the drivers offline.

    * ``f2(m) = ||m||_0`` exactly;
    * ``f1(m)`` decreases with the coreset size, is minimised at
      ``||m||_0 == k`` and grows again for larger coresets (so that eq. (3)
      keeps the size at ``k`` while eq. (4) shrinks it).
    """

    def __init__(self, n: int = 64, k: int = 16, noise: float = 0.0, seed: int = 0) -> None:
        self.n = int(n)
        self.k = int(k)
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed)
        self.num_evaluations = 0
        self._cache: Dict[Tuple[int, ...], Any] = {}

    def stats(self) -> Dict[str, Any]:
        return {"num_evaluations": self.num_evaluations, "cache_size": len(self._cache), "n": self.n}

    def evaluate(self, mask: Any, **kwargs: Any) -> Any:
        f2 = float(np.count_nonzero(binarize_mask(mask) > 0.5))
        # f1 decreases with size until k, then increases (over-selection penalty)
        gap = f2 - float(self.k)
        f1 = 1.0 - 0.5 * min(f2 / max(1.0, float(self.k)), 1.0) + 0.02 * max(0.0, gap)
        if self.noise:
            f1 += float(self.rng.normal(0.0, self.noise))
        self.num_evaluations += 1
        return Figure1Evaluation(mask=np.asarray(binarize_mask(mask)), f1=float(f1), f2=f2)

    def __call__(self, mask: Any, **kwargs: Any) -> Any:
        return self.evaluate(mask, **kwargs)

    def F(self, mask: Any, **kwargs: Any) -> np.ndarray:  # noqa: N802
        return self.evaluate(mask, **kwargs).F()

    def f2(self, mask: Any) -> float:
        return float(np.count_nonzero(binarize_mask(mask) > 0.5))

    binarize = staticmethod(binarize_mask)
    coreset_indices = staticmethod(mask_indices)


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: mask conventions, curve extraction, both drivers."""
    report: Dict[str, Any] = {"ok": True, "checks": {}}

    # mask conventions
    binary = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    relaxed = np.array([1.0, -1.0, 1.0, 0.0, -0.3])
    probs = np.array([0.9, 0.1, 0.6, 0.4, 0.05])
    report["checks"]["binarize_binary"] = bool(np.array_equal(binarize_mask(binary), binary))
    report["checks"]["binarize_relaxed"] = bool(np.array_equal(binarize_mask(relaxed), binary))
    report["checks"]["binarize_probs"] = bool(np.array_equal(binarize_mask(probs), binary))
    report["checks"]["indices"] = bool(np.array_equal(mask_indices(binary), np.array([0, 2, 3])))

    objective = _SyntheticObjective(n=64, k=16, seed=0)
    report["checks"]["f2_is_l0"] = (
        abs(objective.evaluate(np.array([1.0] * 7 + [0.0] * 57)).f2 - 7.0) < 1e-12
    )

    warnings: List[str] = []
    results: Dict[str, Any] = {"curves": {}, "results": {}}
    cfg = Figure1Config.smoke(n=64, k=16, output_dir=os.path.join("results", "figure1_selftest"))
    for form in ("eq3", "eq4"):
        try:
            res = run_formulation(objective, form, cfg)
            curves = curves_from_result(res)
            results["curves"][form] = curves
            results["results"][form] = res
            ok = bool(curves["f1"]) and bool(curves["f2"])
            report["checks"]["driver_" + form] = ok
            if not ok:
                warnings.append("{} produced empty curves".format(form))
        except Exception as exc:
            report["checks"]["driver_" + form] = False
            warnings.append("{} failed: {}".format(form, exc))

    summary = summarize_figure1(results, cfg)
    report["summary"] = summary
    report["checks"]["eq3_fixed_size"] = bool(
        summary.get("formulations", {}).get("eq3", {}).get("fixed_size_failure", False)
    )
    report["checks"]["eq4_shrank"] = bool(
        summary.get("formulations", {}).get("eq4", {}).get("f2_shrunk", False)
    )
    report["warnings"] = warnings
    report["ok"] = all(
        v for k, v in report["checks"].items() if k not in ("eq3_fixed_size", "eq4_shrank")
    )
    if verbose:  # pragma: no cover
        print(json.dumps(_to_jsonable(report), indent=2))
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 1 (trivial solutions of RCS, §2.1) of the LBCS paper."
    )
    parser.add_argument("--dataset", default=FIGURE1_DATASET)
    parser.add_argument("--n", type=int, default=None, help="size of the MNIST subset (default 1000)")
    parser.add_argument("--k", type=int, default=SUGGESTED_K, help="predefined coreset size (SUGGESTED)")
    parser.add_argument("--lambda", dest="lam", type=float, default=FIGURE1_LAMBDA)
    parser.add_argument("--outer-iters", "--T", dest="outer_iters", type=int, default=None)
    parser.add_argument("--outer-lr", type=float, default=FIGURE1_OUTER_LR)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--inner-lr", type=float, default=FIGURE1_INNER_LR)
    parser.add_argument("--inner-momentum", type=float, default=FIGURE1_INNER_MOMENTUM)
    parser.add_argument("--batch-size", type=int, default=SUGGESTED_BATCH_SIZE)
    parser.add_argument("--architecture", default="ConvNet")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-dir", default=os.path.join("results", "figure1"))
    parser.add_argument("--no-figure", action="store_true", help="skip the matplotlib figure")
    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="measure f1 on the full test split instead of the selection pool",
    )
    parser.add_argument("--formulations", default="eq3,eq4")
    parser.add_argument(
        "--paper",
        action="store_true",
        help="use the paper-faithful setting (T=1000, inner epochs=100)",
    )
    parser.add_argument("--smoke", action="store_true", help="cheap validation run (NOT a paper setting)")
    parser.add_argument("--selftest", action="store_true", help="run the offline self-test and exit")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> Any:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    if args.paper:
        config = Figure1Config.paper()
    elif args.smoke:
        config = Figure1Config.smoke()
    else:
        config = Figure1Config()

    overrides: Dict[str, Any] = {
        "dataset": args.dataset,
        "k": args.k,
        "lam": args.lam,
        "outer_lr": args.outer_lr,
        "inner_lr": args.inner_lr,
        "inner_momentum": args.inner_momentum,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
        "device": args.device,
        "seed": args.seed,
        "root": args.root,
        "output_dir": args.output_dir,
        "save_figure": not args.no_figure,
        "eval_on_test": args.eval_on_test,
        "formulations": tuple(_normalize_formulation(f) for f in str(args.formulations).split(",") if f.strip()),
    }
    if args.n is not None:
        overrides["n"] = args.n
    if args.outer_iters is not None:
        overrides["outer_iters"] = args.outer_iters
    if args.inner_epochs is not None:
        overrides["inner_epochs"] = args.inner_epochs

    config = config.with_overrides(**overrides)
    results = run_figure1(config)
    return results


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
