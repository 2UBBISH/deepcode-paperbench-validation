"""Section 6 / Appendix E.4 experiment driver: ablation on the number of search times ``T``.

The paper (Section 6 "The influence of the number of search times" and Appendix E.4
"Ablation on Search Times") studies how the number of outer search iterations ``T`` of
Algorithm 1 influences:

* the test accuracy obtained after training a target model on the constructed coreset, and
* the optimized coreset size ``f_2(m) = ||m||_0``.

The reported behaviour is:

1. initially, increasing ``T`` raises test accuracy and decreases coreset size,
2. later, test accuracy stabilises while the coreset size keeps shrinking,
3. for very large ``T`` the results change only marginally (empirical convergence),
4. so ``T`` can be chosen from the coreset requirement and the search budget
   ("selecting an appropriate value for T can be tailored to specific requirements for
   coresets and the allocated budget for coreset selection").

The experiment is run on F-MNIST (LeNet proxy / LeNet target, Adam, lr=0.001, epsilon=0.2)
with a sweep over ``T in {100, 200, 300, 500, 800, 1500, 2000}`` and
``k in {1000, 2000, 3000, 4000}``.

Paper reference values quoted in the reproduction plan (F-MNIST, k=1000):
``T=100 -> 77.0 +- 1.8 / 998.0 +- 1.9``, ``T=500 -> 79.7 +- 0.7 / 956.7 +- 3.5``,
``T=2000 -> 79.8 +- 0.6 / 935.8 +- 3.8``.

Everything the paper does not state numerically (batch size, weight decay, delta_init,
delta_lower, ...) is exposed as a clearly labelled ``SUGGESTED_*`` default so it can be
overridden from ``configs/section6.yaml`` without touching algorithm code.

Out of scope (never invoked here): ImageNet-1k (Section 5.4), continual learning
(Appendix E.5) and streaming (Appendix E.6).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional / soft dependencies
# ---------------------------------------------------------------------------

try:  # pragma: no cover - torch is a soft dependency
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

try:  # pragma: no cover - yaml is optional
    import yaml

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False

LOGGER = logging.getLogger("lbcs_repro.experiments.table9")

# ---------------------------------------------------------------------------
# Paper-stated protocol (§5.2 dataset/training, §6 T-sweep, Appendix E.4)
# ---------------------------------------------------------------------------

PAPER_DATASET = "F-MNIST"
PAPER_KS: Tuple[int, ...] = (1000, 2000, 3000, 4000)
PAPER_TS: Tuple[int, ...] = (100, 200, 300, 500, 800, 1500, 2000)
PAPER_T = 500
PAPER_EPSILON = 0.2
PAPER_REPEATS = 10
PAPER_INNER_OPTIMIZER = "adam"
PAPER_INNER_LR = 0.001
PAPER_INNER_EPOCHS = 100
PAPER_TARGET_OPTIMIZER = "adam"
PAPER_TARGET_LR = 0.001
PAPER_TARGET_EPOCHS = 100

LBCS_LABEL = "LBCS (ours)"
SIZE_LABEL = "Coreset size (ours)"

#: (k, T) -> (accuracy mean, accuracy std, coreset size mean, coreset size std)
#: Anchor values taken from the reproduction plan (F-MNIST, k=1000).  Used only for
#: reporting/validation, never as algorithm input.
PAPER_TABLE9_K1000: Dict[int, Tuple[float, float, float, float]] = {
    100: (77.0, 1.8, 998.0, 1.9),
    500: (79.7, 0.7, 956.7, 3.5),
    2000: (79.8, 0.6, 935.8, 3.8),
}

# ---------------------------------------------------------------------------
# Suggested defaults (NOT paper-stated)
# ---------------------------------------------------------------------------

SUGGESTED_BATCH_SIZE = 128
SUGGESTED_EVAL_BATCH_SIZE = 256
SUGGESTED_WEIGHT_DECAY = 5e-4
SUGGESTED_DELTA_INIT = 0.1
SUGGESTED_DELTA_LOWER = 1e-3
SUGGESTED_NUM_WORKERS = 0
DEFAULT_OUTPUT_DIR = os.path.join("results", "table9")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Table9Config:
    """Configuration of the Appendix E.4 ``T``-sweep experiment."""

    dataset: str = PAPER_DATASET
    ks: Tuple[int, ...] = PAPER_KS
    ts: Tuple[int, ...] = PAPER_TS
    epsilon: float = PAPER_EPSILON
    repeats: int = PAPER_REPEATS
    inner_optimizer: str = PAPER_INNER_OPTIMIZER
    inner_lr: float = PAPER_INNER_LR
    inner_epochs: int = PAPER_INNER_EPOCHS
    target_optimizer: str = PAPER_TARGET_OPTIMIZER
    target_lr: float = PAPER_TARGET_LR
    target_epochs: int = PAPER_TARGET_EPOCHS
    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1
    num_workers: int = SUGGESTED_NUM_WORKERS
    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0
    output_dir: str = DEFAULT_OUTPUT_DIR
    data_root: Optional[str] = None
    save_artifacts: bool = True
    plot: bool = True
    verbose: bool = False

    # -- constructors ------------------------------------------------------
    @classmethod
    def paper(cls, **overrides: Any) -> "Table9Config":
        """Paper protocol for the ``T``-sweep (Appendix E.4 / Table 9)."""
        return cls().with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "Table9Config":
        """Tiny configuration for offline smoke tests."""
        base = cls(
            ks=(200,),
            ts=(2, 5),
            repeats=1,
            inner_epochs=1,
            target_epochs=1,
            batch_size=16,
            eval_batch_size=32,
            output_dir=os.path.join("results", "table9_smoke"),
        )
        return base.with_overrides(**overrides)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Table9Config":
        """Build a config from a (possibly nested) mapping such as a YAML block."""
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        flat: Dict[str, Any] = {}
        for key, value in dict(data).items():
            if key == "inner" and isinstance(value, dict):
                mapped = {
                    "inner_optimizer": value.get("optimizer"),
                    "inner_lr": value.get("lr"),
                    "inner_epochs": value.get("epochs"),
                }
                flat.update({k: v for k, v in mapped.items() if v is not None})
            elif key == "target":
                entry: Any = value
                if isinstance(value, dict):
                    entry = value.get(PAPER_DATASET, value)
                if isinstance(entry, dict):
                    mapped = {
                        "target_optimizer": entry.get("optimizer"),
                        "target_lr": entry.get("lr"),
                        "target_epochs": entry.get("epochs"),
                    }
                    flat.update({k: v for k, v in mapped.items() if v is not None})
            elif key == "robustness" and isinstance(value, dict):
                for sub in ("ks", "repeats"):
                    if sub in value:
                        flat[sub] = value[sub]
            elif key in known:
                flat[key] = value
        return cls(**flat)

    # -- helpers -----------------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "Table9Config":
        """Return a copy with ``overrides`` applied (unknown keys ignored)."""
        clean = {
            k: v
            for k, v in overrides.items()
            if v is not None and k in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        cfg = replace(self, **clean) if clean else replace(self)
        cfg.ks = tuple(int(k) for k in cfg.ks)
        cfg.ts = tuple(int(t) for t in cfg.ts)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        data["ks"] = list(self.ks)
        data["ts"] = list(self.ts)
        return data


@dataclass
class Table9Cell:
    """Aggregated mean/std result for one ``(k, T)`` cell."""

    k: int
    T: int
    accuracy_mean: float = float("nan")
    accuracy_std: float = 0.0
    size_mean: float = float("nan")
    size_std: float = 0.0
    f1_mean: float = float("nan")
    f1_std: float = 0.0
    repeats: int = 0
    failures: int = 0
    raw: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def cell(self) -> Tuple[int, int]:
        return (int(self.k), int(self.T))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": int(self.k),
            "T": int(self.T),
            "accuracy_mean": float(self.accuracy_mean),
            "accuracy_std": float(self.accuracy_std),
            "size_mean": float(self.size_mean),
            "size_std": float(self.size_std),
            "f1_mean": float(self.f1_mean),
            "f1_std": float(self.f1_std),
            "repeats": int(self.repeats),
            "failures": int(self.failures),
        }


# ---------------------------------------------------------------------------
# Small shared helpers (mirrors the other drivers)
# ---------------------------------------------------------------------------


def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy and (when available) PyTorch deterministically."""
    if seed is None:
        return
    np.random.seed(int(seed) % (2**32 - 1))
    if _TORCH_AVAILABLE:  # pragma: no branch
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():  # pragma: no cover - GPU only
                torch.cuda.manual_seed_all(int(seed))
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover
            pass


def resolve_seed(seed: Optional[int], repeat: int = 0, base: int = 0) -> int:
    """Deterministic per-repeat seed (stable across runs)."""
    base_seed = base if seed is None else int(seed)
    if repeat == 0:
        return base_seed
    return int((base_seed + repeat * 7919) % (2**31 - 1))


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a binary/relaxed/probability mask to ``{0, 1}^n`` (Appendix A rule).

    Values below ``-1`` clamp to ``-1``, above ``1`` to ``1``; then ``[-1, 0) -> 0`` and
    ``[0, 1] -> 1``.
    """
    arr = mask.detach().cpu().numpy() if _is_tensor(mask) else np.asarray(mask)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of the selected examples in a (possibly relaxed) mask."""
    return np.flatnonzero(binarize_mask(mask) > 0.5).astype(np.int64)


def mask_size(mask: Any) -> int:
    """``f_2(m) = ||m||_0`` computed on the discretized mask."""
    return int(mask_indices(mask).size)


def _is_tensor(obj: Any) -> bool:
    return bool(_TORCH_AVAILABLE and isinstance(obj, torch.Tensor))  # type: ignore[union-attr]


def canonical_dataset(name: Optional[str]) -> str:
    """Normalise dataset aliases to the canonical paper names."""
    if not name:
        return PAPER_DATASET
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    mapping = {
        "f-mnist": "F-MNIST",
        "fashionmnist": "F-MNIST",
        "fashion-mnist": "F-MNIST",
        "fmnist": "F-MNIST",
        "fashion": "F-MNIST",
        "mnist": "MNIST",
        "mnist-s": "MNIST-S",
        "svhn": "SVHN",
        "cifar": "CIFAR-10",
        "cifar10": "CIFAR-10",
        "cifar-10": "CIFAR-10",
    }
    return mapping.get(key, str(name))


def mean_std(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:
    """Mean and (sample) standard deviation; empty input gives ``(nan, 0)``."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def format_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    """Format ``mean +- std`` in the paper's table style."""
    if not np.isfinite(mean):
        return "-"
    return f"{mean:.{decimals}f} +- {std:.{decimals}f}"


# ---------------------------------------------------------------------------
# Model / data plumbing
# ---------------------------------------------------------------------------


def _model_factory_for(dataset: str) -> Callable[..., Any]:
    """Return a fresh-model factory for the given dataset (proxy and target)."""
    try:
        from lbcs_repro.models import model_factory as _mf  # type: ignore

        if dataset == "F-MNIST":
            return _mf("LeNet")
        if dataset == "SVHN":
            return _mf("SVHNCNN")
        if dataset == "CIFAR-10":
            return _mf("CIFARCNN")
        return _mf("ConvNet")
    except Exception:  # pragma: no cover - fallback path
        pass
    if dataset == "F-MNIST":
        from lbcs_repro.models.lenet import lenet_factory

        return lenet_factory()
    if dataset == "SVHN":
        from lbcs_repro.models.svhn_cnn import svhn_cnn_factory

        return svhn_cnn_factory()
    if dataset == "CIFAR-10":
        from lbcs_repro.models.cifar_cnn import cifar_cnn_factory

        return cifar_cnn_factory()
    from lbcs_repro.models.convnet import convnet_factory

    return convnet_factory()


def build_context(dataset: str, config: "Table9Config", device: Optional[str] = None) -> Dict[str, Any]:
    """Build train/test loaders and model factories for one benchmark."""
    from lbcs_repro.data.datasets import get_dataset, get_targets, make_loader, num_classes

    dataset = canonical_dataset(dataset)
    root = config.data_root
    train_ds = get_dataset(dataset, train=True, root=root, download=True)
    test_ds = get_dataset(dataset, train=False, root=root, download=True)
    train_targets = get_targets(train_ds)
    n = int(len(train_targets))
    train_loader = make_loader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        return_index=True,
        seed=config.seed,
    )
    eval_loader = make_loader(
        train_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        return_index=True,
    )
    test_loader = make_loader(
        test_ds,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return {
        "dataset": dataset,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "test_loader": test_loader,
        "targets": train_targets,
        "n": n,
        "num_classes": int(num_classes(dataset)),
        "inner_factory": _model_factory_for(dataset),
        "target_factory": _model_factory_for(dataset),
    }


def coreset_loader(
    context: Dict[str, Any],
    mask: Any,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Any:
    """DataLoader over the coreset examples selected by ``mask``."""
    from lbcs_repro.data.datasets import make_loader, subset_dataset

    indices = mask_indices(mask)
    subset = subset_dataset(context["train_dataset"], indices)
    return make_loader(
        subset,
        batch_size=int(batch_size or 128),
        shuffle=shuffle,
        num_workers=int(num_workers or 0),
        seed=seed,
    )


def evaluate_accuracy(model: Any, loader: Any, device: Optional[str] = None) -> float:
    """Top-1 accuracy (%) of ``model`` on ``loader``."""
    if not _TORCH_AVAILABLE:  # pragma: no cover - numpy-only path
        raise RuntimeError("evaluate_accuracy requires PyTorch")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # type: ignore[union-attr]
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.size(0))
    return 100.0 * correct / max(total, 1)


def train_target_model(
    model: Any,
    train_loader: Any,
    test_loader: Any = None,
    epochs: int = PAPER_TARGET_EPOCHS,
    lr: float = PAPER_TARGET_LR,
    optimizer: str = PAPER_TARGET_OPTIMIZER,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Any, List[float]]:
    """Train the post-selection target model on the constructed coreset.

    For F-MNIST the paper uses ``Adam`` with learning rate ``0.001`` for ``100`` epochs
    (§5.2 "Datasets and implementation").
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("train_target_model requires PyTorch")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")  # type: ignore[union-attr]
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  # type: ignore[union-attr]
    if str(optimizer).lower() == "sgd":
        opt = torch.optim.SGD(  # type: ignore[union-attr]
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    else:
        opt = torch.optim.Adam(  # type: ignore[union-attr]
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

    history: List[float] = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            logits = model(inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            loss = criterion(logits, targets)
            loss.backward()
            opt.step()
        if test_loader is not None:
            acc = evaluate_accuracy(model, test_loader, device)
            history.append(acc)
            if verbose:
                LOGGER.info("epoch %d/%d - test acc %.2f", epoch + 1, epochs, acc)
    return model, history


# ---------------------------------------------------------------------------
# LBCS selection with a given T
# ---------------------------------------------------------------------------


def select_lbcs_mask(
    context: Dict[str, Any],
    k: int,
    T: int,
    config: "Table9Config",
    seed: Optional[int] = None,
    lbcs: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run Algorithm 1 (LBCS) with search budget ``T`` and return mask + objectives."""
    from lbcs_repro.lbcs.bilevel import InnerTrainConfig, LBCS, LBCSConfig

    lbcs_config = LBCSConfig(
        k=int(k),
        epsilon=float(config.epsilon),
        T=int(T),
        delta_init=float(config.delta_init),
        delta_lower=float(config.delta_lower),
        warm_start=bool(config.warm_start),
        group_size=int(config.group_size),
        seed=seed,
    )
    inner_config = InnerTrainConfig(
        optimizer=str(config.inner_optimizer),
        lr=float(config.inner_lr),
        weight_decay=float(config.weight_decay),
        epochs=int(config.inner_epochs),
        batch_size=int(config.batch_size),
    )
    runner = lbcs
    if runner is None:
        runner = LBCS(
            model_factory=context["inner_factory"],
            n=int(context["n"]),
            k=int(k),
            epsilon=float(config.epsilon),
            T=int(T),
            dataset=context["train_dataset"],
            train_loader=context["train_loader"],
            eval_loader=context["eval_loader"],
            inner_config=inner_config,
            device=config.device,
            seed=seed,
            delta_init=float(config.delta_init),
            delta_lower=float(config.delta_lower),
            warm_start=bool(config.warm_start),
            group_size=int(config.group_size),
            log_every=int(config.log_every),
            logger=logger,
            config=lbcs_config,
        )
    result = runner.run() if hasattr(runner, "run") else runner.fit()
    mask = getattr(result, "mask", result)
    return {
        "mask": binarize_mask(mask),
        "coreset_size": int(getattr(result, "f2", mask_size(mask))),
        "f1": float(getattr(result, "f1", float("nan"))),
        "T": int(T),
        "k": int(k),
    }


def train_and_evaluate(
    context: Dict[str, Any],
    mask: Any,
    seed: Optional[int] = None,
    config: Optional["Table9Config"] = None,
    model: Any = None,
    train_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Train a target model on the coreset and evaluate the clean test accuracy."""
    config = config or Table9Config()
    set_seed(seed)
    size = mask_size(mask)
    train_loader = coreset_loader(
        context, mask, batch_size=config.batch_size, num_workers=config.num_workers, seed=seed
    )
    if model is None:
        model = context["target_factory"]()
    if train_fn is not None:
        model, history = train_fn(
            model,
            train_loader,
            context["test_loader"],
            epochs=int(config.target_epochs),
            lr=float(config.target_lr),
            optimizer=str(config.target_optimizer),
            weight_decay=float(config.weight_decay),
            device=config.device,
        )
    else:
        model, history = train_target_model(
            model,
            train_loader,
            context["test_loader"],
            epochs=int(config.target_epochs),
            lr=float(config.target_lr),
            optimizer=str(config.target_optimizer),
            weight_decay=float(config.weight_decay),
            device=config.device,
        )
    accuracy = history[-1] if history else evaluate_accuracy(model, context["test_loader"], config.device)
    return {
        "accuracy": float(accuracy),
        "coreset_size": float(size),
        "accuracy_per_point": float(accuracy) / max(size, 1),
        "history": history,
        "model": model,
    }


# ---------------------------------------------------------------------------
# One cell / full driver
# ---------------------------------------------------------------------------


def run_single_cell(
    k: int,
    T: int,
    repeat: int,
    config: Optional["Table9Config"] = None,
    context: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    select_lbcs_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    train_eval_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    lbcs: Any = None,
) -> Dict[str, Any]:
    """Run LBCS with search budget ``T`` once and evaluate the resulting coreset."""
    config = config or Table9Config()
    seed = resolve_seed(config.seed, repeat) if seed is None else int(seed)
    record: Dict[str, Any] = {"k": int(k), "T": int(T), "repeat": int(repeat), "seed": int(seed)}
    started = time.time()
    try:
        if context is None:
            raise RuntimeError("no dataset context provided")
        selector = select_lbcs_fn or select_lbcs_mask
        selection = selector(context, int(k), int(T), config, seed=seed, lbcs=lbcs, logger=logger)
        mask = selection.get("mask")
        if mask is None:
            raise RuntimeError("LBCS selection returned no mask")
        trainer = train_eval_fn or train_and_evaluate
        train_result = trainer(context, mask, seed=seed, config=config)
        size = float(train_result.get("coreset_size", mask_size(mask)))
        record.update(
            {
                "accuracy": float(train_result.get("accuracy", float("nan"))),
                "coreset_size": size,
                "accuracy_per_point": float(train_result.get("accuracy_per_point", float("nan"))),
                "f1": float(selection.get("f1", float("nan"))),
                "f2": float(selection.get("coreset_size", size)),
                "lbcs_size": size,
                "failed": False,
            }
        )
    except Exception as exc:  # keep long sweeps alive
        LOGGER.warning("cell k=%s T=%s repeat=%s failed: %s", k, T, repeat, exc)
        record.update(
            {
                "failed": True,
                "error": repr(exc),
                "accuracy": float("nan"),
                "coreset_size": float("nan"),
            }
        )
    record["wall_time"] = time.time() - started
    return record


def run_table9(
    config: Optional["Table9Config"] = None,
    logger: Optional[logging.Logger] = None,
    cell_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    context_builder: Optional[Callable[..., Dict[str, Any]]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run the Appendix E.4 ``T``-sweep (Table 9) on F-MNIST."""
    logger = logger or LOGGER
    if config is None:
        config = Table9Config.from_dict(overrides) if overrides else Table9Config.paper()
        config = config.with_overrides(**overrides)
    elif overrides:
        config = config.with_overrides(**overrides)
    context_builder = context_builder or build_context
    cell_runner = cell_runner or run_single_cell

    records: List[Dict[str, Any]] = []
    context: Optional[Dict[str, Any]] = None
    try:
        context = context_builder(config.dataset, config, config.device)
        logger.info(
            "Table 9 sweep: dataset=%s n=%s ks=%s ts=%s repeats=%d",
            config.dataset,
            context.get("n"),
            list(config.ks),
            list(config.ts),
            config.repeats,
        )
    except Exception as exc:  # pragma: no cover - offline/no-data path
        logger.warning("could not build dataset context (%s); running dry sweep", exc)
        context = None

    for k in config.ks:
        for T in config.ts:
            for repeat in range(int(config.repeats)):
                record = cell_runner(
                    int(k), int(T), int(repeat), config=config, context=context, logger=logger
                )
                records.append(record)
            logger.info("finished k=%s T=%s", k, T)

    cells = aggregate_results(records)
    checks = direction_checks(cells, config)
    table_text = format_table9(cells, config)
    artifacts = save_results(cells, records, checks, config) if config.save_artifacts else {}
    if config.plot and config.save_artifacts:
        try:
            plot_table9(cells, os.path.join(config.output_dir, "table9.png"), config)
        except Exception as exc:  # pragma: no cover - plotting is optional
            logger.debug("plot skipped: %s", exc)
    return {
        "table": table_text,
        "cells": cells,
        "records": records,
        "checks": checks,
        "config": config.to_dict(),
        "artifacts": artifacts,
    }


def aggregate_results(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, int], Table9Cell]:
    """Aggregate per-repeat records into ``(k, T) -> Table9Cell`` mean/std entries."""
    grouped: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for rec in records:
        key = (int(rec.get("k", 0)), int(rec.get("T", 0)))
        grouped.setdefault(key, []).append(rec)

    cells: Dict[Tuple[int, int], Table9Cell] = {}
    for (k, T), recs in grouped.items():
        ok = [r for r in recs if not r.get("failed", False)]
        acc_mean, acc_std = mean_std([r.get("accuracy") for r in ok])
        size_mean, size_std = mean_std([r.get("coreset_size") for r in ok])
        f1_mean, f1_std = mean_std([r.get("f1") for r in ok])
        cells[(k, T)] = Table9Cell(
            k=int(k),
            T=int(T),
            accuracy_mean=acc_mean,
            accuracy_std=acc_std,
            size_mean=size_mean,
            size_std=size_std,
            f1_mean=f1_mean,
            f1_std=f1_std,
            repeats=len(recs),
            failures=len(recs) - len(ok),
            raw=recs,
        )
    return cells


def direction_checks(
    cells: Dict[Tuple[int, int], Table9Cell], config: Optional["Table9Config"] = None
) -> Dict[str, Any]:
    """Check the Appendix E.4 qualitative claims about the ``T`` sweep.

    Claims: (i) accuracy initially increases with ``T``; (ii) coreset size decreases with
    ``T``; (iii) very large ``T`` brings only marginal changes (empirical convergence).
    """
    config = config or Table9Config()
    per_k: Dict[int, List[Table9Cell]] = {}
    for (k, _T), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if cell.repeats - cell.failures <= 0:
            continue
        per_k.setdefault(int(k), []).append(cell)

    increasing_acc: List[bool] = []
    decreasing_size: List[bool] = []
    converging: List[bool] = []
    for k, row in per_k.items():
        row = sorted(row, key=lambda c: c.T)
        if len(row) >= 2:
            increasing_acc.append(bool(row[-1].accuracy_mean >= row[0].accuracy_mean - 1.0))
        sizes = [c.size_mean for c in row if np.isfinite(c.size_mean)]
        if len(sizes) >= 2:
            decreasing_size.append(bool(sizes[-1] <= sizes[0] + 1e-6))
        if len(row) >= 3:
            last_gap = abs(row[-1].size_mean - row[-2].size_mean)
            converging.append(bool(last_gap <= 50.0))

    reference: Dict[str, Any] = {}
    for T, (am, asd, sm, ssd) in PAPER_TABLE9_K1000.items():
        ref: Dict[str, Any] = {
            "accuracy": am,
            "accuracy_std": asd,
            "size": sm,
            "size_std": ssd,
        }
        measured = cells.get((1000, T))
        if measured is not None:
            ref["measured_accuracy"] = measured.accuracy_mean
            ref["measured_size"] = measured.size_mean
        reference[str(T)] = ref

    return {
        "accuracy_increases_with_T": bool(all(increasing_acc)) if increasing_acc else None,
        "size_decreases_with_T": bool(all(decreasing_size)) if decreasing_size else None,
        "converges_for_large_T": bool(all(converging)) if converging else None,
        "per_k": {
            str(k): [{"T": c.T, "accuracy": c.accuracy_mean, "size": c.size_mean} for c in row]
            for k, row in per_k.items()
        },
        "paper_reference_k1000": reference,
    }


# ---------------------------------------------------------------------------
# Reporting / artifacts
# ---------------------------------------------------------------------------


def format_table9(
    cells: Dict[Tuple[int, int], Table9Cell], config: Optional["Table9Config"] = None
) -> str:
    """Render the Table 9 layout: per ``k`` a row per ``T`` with accuracy and size."""
    config = config or Table9Config()
    lines: List[str] = []
    lines.append("Table 9: Ablation on the number of search times T (Appendix E.4)")
    lines.append(
        f"Dataset: {config.dataset} | epsilon={config.epsilon} | repeats={config.repeats}"
    )
    lines.append("")
    header = f"{'k':>6} | {'T':>6} | {'Test accuracy (%)':>20} | {'Coreset size':>20}"
    lines.append(header)
    lines.append("-" * len(header))
    for k in sorted({kk for kk, _ in cells}):
        for T in sorted({tt for kk, tt in cells if kk == k}):
            cell = cells.get((int(k), int(T)))
            if cell is None:
                continue
            acc = format_mean_std(cell.accuracy_mean, cell.accuracy_std)
            size = format_mean_std(cell.size_mean, cell.size_std)
            lines.append(f"{int(k):>6} | {int(T):>6} | {acc:>20} | {size:>20}")
    lines.append("")
    lines.append("Paper reference values (F-MNIST, k=1000) from the reproduction plan:")
    for T, (am, asd, sm, ssd) in PAPER_TABLE9_K1000.items():
        lines.append(f"  T={T:>4}: accuracy={am:.1f} +- {asd:.1f}, size={sm:.1f} +- {ssd:.1f}")
    return "\n".join(lines)


def plot_table9(
    cells: Dict[Tuple[int, int], Table9Cell],
    out_path: str,
    config: Optional["Table9Config"] = None,
    dpi: int = 150,
) -> Optional[str]:
    """Plot accuracy and coreset size versus ``T`` for each ``k`` (if matplotlib present)."""
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    ks = sorted({kk for kk, _ in cells})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for k in ks:
        row = sorted([c for (kk, _), c in cells.items() if kk == k], key=lambda c: c.T)
        ts = [c.T for c in row]
        axes[0].errorbar(
            ts,
            [c.accuracy_mean for c in row],
            yerr=[c.accuracy_std for c in row],
            marker="o",
            capsize=3,
            label=f"k={k}",
        )
        axes[1].errorbar(
            ts,
            [c.size_mean for c in row],
            yerr=[c.size_std for c in row],
            marker="s",
            capsize=3,
            label=f"k={k}",
        )
    axes[0].set_xlabel("T (number of search times)")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("T (number of search times)")
    axes[1].set_ylabel("Coreset size $f_2(m)$")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    dataset = config.dataset if config else PAPER_DATASET
    fig.suptitle(f"Appendix E.4 / Table 9: influence of search times T ({dataset})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_results(
    cells: Dict[Tuple[int, int], Table9Cell],
    records: Sequence[Dict[str, Any]],
    checks: Dict[str, Any],
    config: "Table9Config",
) -> Dict[str, str]:
    """Persist Table 9 as JSON/CSV/TXT plus raw per-repeat JSONL and check summary."""
    os.makedirs(config.output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    payload = {
        "config": config.to_dict(),
        "cells": [c.to_dict() for _, c in sorted(cells.items())],
        "checks": checks,
    }
    json_path = os.path.join(config.output_dir, "table9.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    paths["json"] = json_path

    csv_path = os.path.join(config.output_dir, "table9.csv")
    columns = (
        "k",
        "T",
        "accuracy_mean",
        "accuracy_std",
        "size_mean",
        "size_std",
        "f1_mean",
        "f1_std",
        "repeats",
        "failures",
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(columns))
        for _, cell in sorted(cells.items()):
            d = cell.to_dict()
            writer.writerow([d[c] for c in columns])
    paths["csv"] = csv_path

    txt_path = os.path.join(config.output_dir, "table9.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(format_table9(cells, config))
        fh.write("\n\n")
        fh.write("Direction checks:\n")
        fh.write(json.dumps(_json_safe(checks), indent=2))
    paths["txt"] = txt_path

    raw_path = os.path.join(config.output_dir, "table9_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as fh:
        for rec in records:
            clean = {k: v for k, v in rec.items() if k != "model"}
            fh.write(json.dumps(_json_safe(clean), default=_json_default))
            fh.write("\n")
    paths["raw"] = raw_path

    checks_path = os.path.join(config.output_dir, "table9_checks.json")
    with open(checks_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(checks), fh, indent=2, default=_json_default)
    paths["checks"] = checks_path
    return paths


def _json_safe(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays to JSON-friendly Python objects."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Section 6 / Appendix E.4 Table 9 driver: LBCS search-time (T) sweep on F-MNIST"
    )
    parser.add_argument("--dataset", default=PAPER_DATASET)
    parser.add_argument("--ks", type=int, nargs="*", default=list(PAPER_KS))
    parser.add_argument("--ts", type=int, nargs="*", default=list(PAPER_TS))
    parser.add_argument("--epsilon", type=float, default=PAPER_EPSILON)
    parser.add_argument("--repeats", type=int, default=PAPER_REPEATS)
    parser.add_argument("--inner-epochs", type=int, default=PAPER_INNER_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=PAPER_TARGET_EPOCHS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=None, help="optional YAML config path")
    parser.add_argument("--paper", action="store_true", help="use the paper protocol")
    parser.add_argument("--smoke", action="store_true", help="tiny smoke configuration")
    parser.add_argument("--selftest", action="store_true", help="run offline self-test only")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> "Table9Config":
    data: Dict[str, Any] = {}
    if args.config:
        if not _YAML_AVAILABLE:  # pragma: no cover
            raise RuntimeError("pyyaml is required to load --config")
        with open(args.config, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    config = Table9Config.from_dict(data)
    overrides: Dict[str, Any] = {}
    if getattr(args, "dataset", None) is not None:
        overrides["dataset"] = args.dataset
    if getattr(args, "ks", None):
        overrides["ks"] = tuple(args.ks)
    if getattr(args, "ts", None):
        overrides["ts"] = tuple(args.ts)
    overrides.update(
        {
            "epsilon": args.epsilon,
            "repeats": args.repeats,
            "inner_epochs": args.inner_epochs,
            "target_epochs": args.target_epochs,
            "device": args.device,
            "seed": args.seed,
            "output_dir": args.output_dir,
            "data_root": args.data_root,
            "verbose": args.verbose,
        }
    )
    if args.smoke:
        return Table9Config.smoke(**overrides)
    return config.with_overrides(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report["ok"] else 1
    config = _config_from_args(args)
    result = run_table9(config)
    print(result["table"])
    return 0


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline self-test: masks, aggregation, formatting, checks, config, dry run."""
    problems: List[str] = []

    # mask helpers (Appendix A projection)
    m = np.array([0, 1, 1, 0, 1], dtype=float)
    if mask_size(m) != 3:
        problems.append("mask_size mismatch")
    if list(mask_indices(m)) != [1, 2, 4]:
        problems.append("mask_indices mismatch")
    if mask_size(np.array([-0.5, 0.0, 1.5, -2.0])) != 2:
        problems.append("Appendix A projection mismatch")

    # aggregation
    records = [
        {"k": 100, "T": 10, "accuracy": 70.0, "coreset_size": 95.0, "f1": 1.0, "failed": False},
        {"k": 100, "T": 10, "accuracy": 72.0, "coreset_size": 93.0, "f1": 1.2, "failed": False},
        {"k": 100, "T": 20, "accuracy": 74.0, "coreset_size": 90.0, "f1": 1.1, "failed": False},
        {"k": 100, "T": 20, "accuracy": 0.0, "coreset_size": 0.0, "failed": True},
    ]
    cells = aggregate_results(records)
    if abs(cells[(100, 10)].accuracy_mean - 71.0) > 1e-9:
        problems.append("aggregate accuracy mismatch")
    if cells[(100, 20)].failures != 1:
        problems.append("failure counting mismatch")

    text = format_table9(cells, Table9Config.smoke())
    if "Table 9" not in text or "Coreset size" not in text:
        problems.append("format_table9 layout mismatch")

    checks = direction_checks(cells, Table9Config.smoke())
    for key in ("accuracy_increases_with_T", "size_decreases_with_T", "converges_for_large_T"):
        if key not in checks:
            problems.append(f"missing check key: {key}")

    # config round trip
    cfg = Table9Config.paper()
    if tuple(cfg.ts) != PAPER_TS or tuple(cfg.ks) != PAPER_KS:
        problems.append("paper config mismatch")
    if Table9Config.from_dict(cfg.to_dict()).epsilon != cfg.epsilon:
        problems.append("config round-trip mismatch")
    if Table9Config.smoke().repeats != 1:
        problems.append("smoke config mismatch")

    # dry run with synthetic runner (no torch / no data)
    def fake_cell_runner(k, T, repeat, config=None, context=None, logger=None, **kw):
        scale = float(T) / 200.0
        return {
            "k": k,
            "T": T,
            "repeat": repeat,
            "accuracy": 60.0 + 20.0 * min(scale, 1.0),
            "coreset_size": float(k) - 50.0 * min(scale, 1.0),
            "f1": 2.0 - min(scale, 1.0),
            "failed": False,
        }

    result = run_table9(
        config=Table9Config.smoke(plot=False, save_artifacts=False),
        cell_runner=fake_cell_runner,
        context_builder=lambda *a, **k: {"n": 1000},
    )
    if not result["cells"]:
        problems.append("dry run produced no cells")

    ok = not problems
    report = {"ok": ok, "problems": problems, "num_cells": len(cells)}
    if verbose:
        print("Table 9 self-test:", "PASS" if ok else "FAIL")
        for p in problems:
            print("  -", p)
    return report


run = run_table9
table9_search_times = run_table9

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
