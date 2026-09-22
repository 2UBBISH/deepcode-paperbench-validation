"""Section 6 / Table 5: the influence of mask initialization on LBCS.

Paper (Section 6, "The influence of mask initialization")::

    If the search space is large and the search time is limited, a suitable
    mask initialization will be beneficial to the final performance. Prior to
    this, we use random mask initialization for fair comparison. Here we show
    that with mask initialization by other methods, the final performance will
    be enhanced. Experimental results are shown in Table 5.

Table 5 (mean +- std test accuracy (%) on F-MNIST, various predefined coreset
sizes k).  "LBCS+Moderate" means the mask is initialized by "Moderate" and then
is refined by our LBCS::

    k      LBCS             LBCS+Moderate
    1000   79.7 +- 0.7      79.8 +- 0.5
    2000   82.8 +- 0.6      83.6 +- 0.7
    3000   84.0 +- 0.6      84.3 +- 0.4
    4000   84.5 +- 0.4      85.1 +- 0.3

Protocol (Section 5.2 settings, unchanged by Section 6: "The other experimental
settings are not changed"):
  * benchmark                : F-MNIST (Xiao et al., 2017)
  * proxy (inner loop) model : LeNet
  * inner optimizer          : Adam, lr = 0.001
  * target model             : LeNet, Adam lr = 0.001, 100 epochs
  * epsilon = 0.2, T = 500
  * 10 repeats (Section 5.2: "All experiments are repeated ten times")

Two initialization modes are compared for the *same* LBCS refinement loop:
  * ``random``   -- the paper's default random mask initialization with exactly
                    ``||m||_0 = k`` (Algorithm 1 Line 2).
  * ``moderate`` -- the mask is initialized by the Moderate coreset baseline
                    (Xia et al., 2023b; Appendix D.1) and then refined by LBCS.

Scope: ImageNet-1k (Section 5.4), continual learning (Appendix E.5) and
streaming (Appendix E.6) are out of scope and are never invoked here.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-stated constants (never modify these to "make results match")
# ---------------------------------------------------------------------------

PAPER_DATASET: str = "F-MNIST"
PAPER_KS: Tuple[int, ...] = (1000, 2000, 3000, 4000)
PAPER_EPSILON: float = 0.2
PAPER_T: int = 500
PAPER_REPEATS: int = 10

PAPER_INNER_OPTIMIZER: str = "adam"
PAPER_INNER_LR: float = 0.001
PAPER_INNER_EPOCHS: int = 100

PAPER_TARGET_OPTIMIZER: str = "adam"
PAPER_TARGET_LR: float = 0.001
PAPER_TARGET_EPOCHS: int = 100

LBCS_LABEL: str = "LBCS"
MODERATE_INIT_LABEL: str = "LBCS+Moderate"
INIT_MODES: Tuple[str, ...] = ("random", "moderate")

#: Paper-reported Table 5 values: init mode -> k -> (mean, std).
PAPER_TABLE5: Dict[str, Dict[int, Tuple[float, float]]] = {
    LBCS_LABEL: {
        1000: (79.7, 0.7),
        2000: (82.8, 0.6),
        3000: (84.0, 0.6),
        4000: (84.5, 0.4),
    },
    MODERATE_INIT_LABEL: {
        1000: (79.8, 0.5),
        2000: (83.6, 0.7),
        3000: (84.3, 0.4),
        4000: (85.1, 0.3),
    },
}

# ---------------------------------------------------------------------------
# Suggested defaults (NOT paper-stated -- labelled and overridable)
# ---------------------------------------------------------------------------

SUGGESTED_BATCH_SIZE: int = 128
SUGGESTED_EVAL_BATCH_SIZE: int = 256
SUGGESTED_WEIGHT_DECAY: float = 0.0
SUGGESTED_DELTA_INIT: float = 0.1
SUGGESTED_DELTA_LOWER: float = 1e-3
SUGGESTED_NUM_WORKERS: int = 0
#: Number of epochs used to train the proxy/reference model that provides the
#: Moderate scores for mask initialization (SUGGESTED; the paper delegates the
#: detailed recipe of every baseline to its own code repository, Appendix D.1).
SUGGESTED_REFERENCE_EPOCHS: int = 100
DEFAULT_OUTPUT_DIR: str = os.path.join("results", "table5")


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy (and PyTorch when available) deterministically."""
    if seed is None:
        return
    seed = int(seed)
    np.random.seed(seed % (2 ** 32))
    try:  # soft torch dependency
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass


def resolve_seed(seed: Optional[int], repeat: int = 0, base: int = 0) -> int:
    """Deterministic per-repeat seed."""
    if seed is None:
        seed = base
    return int((int(seed) + int(repeat) * 7919) % (2 ** 31 - 1))


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a mask to ``{0, 1}`` using the Appendix A rule.

    Continuous values ``v < -1`` clamp to ``-1``, ``v > 1`` clamp to ``1``;
    then ``[-1, 0) -> 0`` and ``[0, 1] -> 1``.
    """
    arr = np.asarray(mask, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of the selected examples (after projection)."""
    return np.nonzero(binarize_mask(mask) > 0)[0]


def mask_size(mask: Any) -> int:
    """``f_2(m) = ||m||_0`` on the discretized mask."""
    return int(binarize_mask(mask).sum())


def canonical_dataset(name: Optional[str]) -> str:
    """Normalize dataset aliases to the paper's benchmark names."""
    if name is None:
        return PAPER_DATASET
    key = str(name).strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "f-mnist": "F-MNIST",
        "fmnist": "F-MNIST",
        "fashion-mnist": "F-MNIST",
        "fashionmnist": "F-MNIST",
        "mnist-s": "MNIST-S",
        "mnists": "MNIST-S",
        "svhn": "SVHN",
        "cifar-10": "CIFAR-10",
        "cifar10": "CIFAR-10",
        "mnist": "MNIST",
    }
    return aliases.get(key, str(name))


def mean_std(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:
    """Mean and sample standard deviation, ignoring non-finite values."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def format_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    """Format ``mean +- std`` the way the paper's tables do."""
    return f"{mean:.{decimals}f} +- {std:.{decimals}f}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Table5Config:
    """Configuration of the Section 6 / Table 5 mask-initialization study."""

    dataset: str = PAPER_DATASET
    ks: Tuple[int, ...] = PAPER_KS
    init_modes: Tuple[str, ...] = INIT_MODES
    epsilon: float = PAPER_EPSILON
    T: int = PAPER_T
    repeats: int = PAPER_REPEATS

    # Inner loop (coreset selection proxy model).
    inner_optimizer: str = PAPER_INNER_OPTIMIZER
    inner_lr: float = PAPER_INNER_LR
    inner_epochs: int = PAPER_INNER_EPOCHS

    # Post-selection target training.
    target_optimizer: str = PAPER_TARGET_OPTIMIZER
    target_lr: float = PAPER_TARGET_LR
    target_epochs: int = PAPER_TARGET_EPOCHS

    # Shared controls.
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1
    num_workers: int = SUGGESTED_NUM_WORKERS
    reference_epochs: int = SUGGESTED_REFERENCE_EPOCHS

    # Runtime / bookkeeping.
    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0
    output_dir: str = DEFAULT_OUTPUT_DIR
    data_root: Optional[str] = None
    save_artifacts: bool = True
    plot: bool = True
    verbose: bool = False

    # ------------------------------------------------------------------ ctors
    @classmethod
    def paper(cls, **overrides: Any) -> "Table5Config":
        """Paper-faithful defaults for Table 5."""
        return cls().with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "Table5Config":
        """Tiny offline-friendly configuration (2 repeats, k = 1000 only)."""
        base = dict(
            ks=(1000,),
            repeats=2,
            T=5,
            inner_epochs=1,
            target_epochs=1,
            reference_epochs=1,
            save_artifacts=False,
            plot=False,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Table5Config":
        """Build from a (possibly nested) mapping, e.g. a YAML section."""
        if not data:
            return cls()
        flat: Dict[str, Any] = {}
        for key, value in dict(data).items():
            if key in {"table5", "mask_init", "section6"} and isinstance(value, dict):
                flat.update(value)
            else:
                flat[key] = value
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in flat.items() if k in known}
        for key in ("ks", "init_modes"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> "Table5Config":
        """Return a copy with the given (known) fields overridden."""
        data = self.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            if key in {"ks", "init_modes"}:
                value = tuple(value)
            if key in data:
                data[key] = value
        known = set(Table5Config.__dataclass_fields__)  # type: ignore[attr-defined]
        return Table5Config(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "ks": tuple(self.ks),
            "init_modes": tuple(self.init_modes),
            "epsilon": self.epsilon,
            "T": self.T,
            "repeats": self.repeats,
            "inner_optimizer": self.inner_optimizer,
            "inner_lr": self.inner_lr,
            "inner_epochs": self.inner_epochs,
            "target_optimizer": self.target_optimizer,
            "target_lr": self.target_lr,
            "target_epochs": self.target_epochs,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "delta_init": self.delta_init,
            "delta_lower": self.delta_lower,
            "warm_start": self.warm_start,
            "group_size": self.group_size,
            "num_workers": self.num_workers,
            "reference_epochs": self.reference_epochs,
            "device": self.device,
            "seed": self.seed,
            "log_every": self.log_every,
            "output_dir": self.output_dir,
            "data_root": self.data_root,
            "save_artifacts": self.save_artifacts,
            "plot": self.plot,
            "verbose": self.verbose,
        }


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class Table5Cell:
    """Aggregated mean/std results for one ``k`` and one initialization mode."""

    k: int
    init_mode: str = "random"
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    size_mean: float = 0.0
    size_std: float = 0.0
    f1_mean: float = 0.0
    f1_std: float = 0.0
    f2_mean: float = 0.0
    f2_std: float = 0.0
    repeats: int = 0
    failures: int = 0
    raw: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def cell(self) -> Tuple[int, str]:
        return (int(self.k), str(self.init_mode))

    @property
    def label(self) -> str:
        return LBCS_LABEL if self.init_mode == "random" else MODERATE_INIT_LABEL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "init_mode": self.init_mode,
            "label": self.label,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "size_mean": self.size_mean,
            "size_std": self.size_std,
            "f1_mean": self.f1_mean,
            "f1_std": self.f1_std,
            "f2_mean": self.f2_mean,
            "f2_std": self.f2_std,
            "repeats": self.repeats,
            "failures": self.failures,
            "accuracy_str": format_mean_std(self.accuracy_mean, self.accuracy_std),
            "size_str": format_mean_std(self.size_mean, self.size_std),
        }


# ---------------------------------------------------------------------------
# Data / model context
# ---------------------------------------------------------------------------

def build_context(dataset: str = PAPER_DATASET, config: Optional[Table5Config] = None,
                  device: Optional[str] = None) -> Dict[str, Any]:
    """Build loaders, targets and model factories for the experiment."""
    cfg = config or Table5Config()
    dataset = canonical_dataset(dataset)

    from lbcs_repro.data.datasets import (  # noqa: WPS433
        get_dataset,
        get_targets,
        make_loader,
        num_classes,
    )

    root = cfg.data_root
    train_ds = get_dataset(dataset, train=True, root=root)
    test_ds = get_dataset(dataset, train=False, root=root)
    targets = get_targets(train_ds)
    n_classes = num_classes(dataset)

    train_loader = make_loader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    eval_loader = make_loader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    test_loader = make_loader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    inner_factory, target_factory = _model_factories(dataset, n_classes)

    return {
        "dataset": dataset,
        "train_dataset": train_ds,
        "test_dataset": test_ds,
        "train_loader": train_loader,
        "eval_loader": eval_loader,
        "test_loader": test_loader,
        "targets": np.asarray(targets),
        "num_classes": int(n_classes),
        "n": int(len(targets)),
        "inner_factory": inner_factory,
        "target_factory": target_factory,
        "device": device or cfg.device,
        "config": cfg,
    }


def _model_factories(dataset: str, num_classes: int) -> Tuple[Any, Any]:
    """Resolve (proxy/inner, target) model factories for a benchmark."""
    inner = target = None
    try:  # preferred: aggregated registry with the paper's defaults
        from lbcs_repro.models import default_model_for, model_factory  # noqa: WPS433

        try:
            inner_name = default_model_for(dataset, "inner")
            inner = model_factory(inner_name, num_classes=num_classes)
        except Exception:  # pragma: no cover - registry incomplete
            inner = None
        try:
            target_name = default_model_for(dataset, "target")
            target = model_factory(target_name, num_classes=num_classes)
        except Exception:  # pragma: no cover
            target = None
    except Exception:  # pragma: no cover - models package unavailable
        pass

    if inner is None:
        try:
            from lbcs_repro.models.lenet import lenet_factory  # noqa: WPS433

            inner = lenet_factory(num_classes=num_classes)
        except Exception:  # pragma: no cover
            inner = None
    if target is None:
        target = inner
    return inner, target


def coreset_loader(context: Dict[str, Any], mask: Any, batch_size: Optional[int] = None,
                   num_workers: Optional[int] = None, shuffle: bool = True,
                   seed: Optional[int] = None) -> Any:
    """DataLoader over the coreset examples selected by ``mask``."""
    cfg: Table5Config = context.get("config") or Table5Config()
    indices = mask_indices(mask)
    from lbcs_repro.data.datasets import make_loader, subset_dataset  # noqa: WPS433

    subset = subset_dataset(context["train_dataset"], indices, return_index=True)
    return make_loader(
        subset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        seed=seed if seed is not None else cfg.seed,
    )


# ---------------------------------------------------------------------------
# Target model training / evaluation
# ---------------------------------------------------------------------------

def evaluate_accuracy(model: Any, loader: Any, device: Optional[str] = None) -> float:
    """Top-1 accuracy (%) of ``model`` on ``loader``."""
    if model is None or loader is None:
        return float("nan")
    try:
        import torch  # noqa: WPS433

        from lbcs_repro.models.resnet18 import evaluate as _evaluate  # noqa: WPS433

        return float(_evaluate(model, loader, device=device))
    except Exception:
        pass
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            if device is not None:
                inputs = inputs.to(device)
                targets = targets.to(device)
            logits = model(inputs)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            preds = logits.argmax(dim=-1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


def train_target_model(model: Any, train_loader: Any, test_loader: Any = None,
                       epochs: int = PAPER_TARGET_EPOCHS, lr: float = PAPER_TARGET_LR,
                       optimizer: str = PAPER_TARGET_OPTIMIZER,
                       weight_decay: float = SUGGESTED_WEIGHT_DECAY,
                       momentum: float = 0.9, device: Optional[str] = None,
                       verbose: bool = False) -> Tuple[Any, List[float]]:
    """Train the post-selection target model (Section 5.2 protocol).

    F-MNIST: LeNet with Adam, lr = 0.001, 100 epochs.
    """
    if model is None:
        return model, []
    import torch  # noqa: WPS433
    import torch.nn as nn  # noqa: WPS433

    criterion = nn.CrossEntropyLoss()
    opt_name = str(optimizer).lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: List[float] = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            if device is not None:
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
            acc = evaluate_accuracy(model, test_loader, device=device)
            history.append(acc)
            if verbose:
                LOGGER.info("epoch %d/%d  test acc = %.2f", epoch + 1, epochs, acc)
    return model, history


def train_and_evaluate(context: Dict[str, Any], mask: Any, seed: Optional[int] = None,
                       config: Optional[Table5Config] = None, model: Any = None,
                       train_fn: Optional[Any] = None) -> Dict[str, Any]:
    """Train a target model on the coreset selected by ``mask`` and evaluate it."""
    cfg = config or context.get("config") or Table5Config()
    if train_fn is not None:
        return train_fn(context, mask, seed=seed, config=cfg, model=model)

    set_seed(seed)
    factory = context.get("target_factory") or context.get("inner_factory")
    if model is None and factory is not None:
        try:
            model = factory()
        except TypeError:
            model = factory(num_classes=context.get("num_classes"))

    loader = coreset_loader(context, mask, seed=seed)
    size = mask_size(mask)
    device = context.get("device") or cfg.device
    trained, history = train_target_model(
        model,
        loader,
        context.get("test_loader"),
        epochs=cfg.target_epochs,
        lr=cfg.target_lr,
        optimizer=cfg.target_optimizer,
        weight_decay=cfg.weight_decay,
        device=device,
    )
    acc = evaluate_accuracy(trained, context.get("test_loader"), device=device)
    return {
        "accuracy": float(acc),
        "coreset_size": int(size),
        "accuracy_per_point": float(acc) / max(size, 1),
        "history": list(history),
        "model": trained,
    }


# ---------------------------------------------------------------------------
# Mask initialization strategies
# ---------------------------------------------------------------------------

def random_init_mask(n: int, k: int, seed: Optional[int] = None) -> np.ndarray:
    """Random mask with exactly ``||m||_0 = k`` (Algorithm 1 Line 2)."""
    mask = np.zeros(int(n), dtype=np.float32)
    k = int(min(max(k, 1), int(n)))
    rng = np.random.default_rng(0 if seed is None else int(seed))
    idx = rng.choice(int(n), size=k, replace=False)
    mask[idx] = 1.0
    return mask


def moderate_init_mask(context: Dict[str, Any], k: int, seed: Optional[int] = None,
                       config: Optional[Table5Config] = None,
                       reference_model: Any = None) -> Dict[str, Any]:
    """Initialize the mask with the Moderate coreset baseline (Appendix D.1).

    Returns a dict with ``mask``, the strategy used and diagnostics.  Falls back
    to a random ``k``-subset (with a warning) if the Moderate selector is
    unavailable, so long sweeps never crash.
    """
    cfg = config or context.get("config") or Table5Config()
    targets = context.get("targets")
    n = int(context.get("n", 0)) or (len(targets) if targets is not None else int(k))
    num_classes = int(context.get("num_classes", 10))
    fallback = random_init_mask(n, int(k), seed=seed)
    try:
        from lbcs_repro.baselines.moderate import ModerateSelector  # noqa: WPS433

        selector = ModerateSelector(
            model=reference_model,
            num_classes=num_classes,
            num_workers=cfg.num_workers,
            seed=seed,
            device=context.get("device") or cfg.device,
        )
        mask = None
        try:
            mask = selector.select_mask(
                n,
                int(k),
                dataset=context.get("train_dataset"),
                targets=targets,
                num_classes=num_classes,
                seed=seed,
                model=reference_model,
                loader=context.get("train_loader"),
                train_loader=context.get("train_loader"),
            )
        except TypeError:
            mask = None
        if mask is None:
            scores = selector.compute_scores(
                dataset=context.get("train_dataset"),
                targets=targets,
                n=n,
                seed=seed,
                model=reference_model,
                loader=context.get("train_loader"),
                train_loader=context.get("train_loader"),
                num_classes=num_classes,
            )
            from lbcs_repro.baselines.moderate import moderate_mask  # noqa: WPS433

            mask = moderate_mask(
                np.asarray(scores),
                n=n,
                k=int(k),
                targets=targets,
                num_classes=num_classes,
                seed=seed,
            )
        mask = binarize_mask(mask)
        if int(mask.sum()) != int(k):
            LOGGER.warning(
                "Moderate initialization produced ||m||_0 = %d instead of k = %d",
                int(mask.sum()), int(k),
            )
        return {"mask": mask, "strategy": "moderate", "fallback": False}
    except Exception as exc:  # pragma: no cover - depends on optional stack
        LOGGER.warning("Moderate mask initialization unavailable (%s); using random init", exc)
        return {"mask": fallback, "strategy": "random(fallback)", "fallback": True, "error": str(exc)}


def train_reference_proxy(context: Dict[str, Any], config: Table5Config,
                          seed: Optional[int] = None) -> Any:
    """Train the proxy/reference model used to compute Moderate scores."""
    factory = context.get("inner_factory")
    if factory is None:
        return None
    set_seed(seed)
    try:
        model = factory()
    except TypeError:
        model = factory(num_classes=context.get("num_classes"))
    try:
        from lbcs_repro.baselines.base import train_reference_model  # noqa: WPS433

        model = train_reference_model(
            model,
            context.get("train_loader"),
            epochs=int(config.reference_epochs),
            lr=float(config.inner_lr),
            optimizer=str(config.inner_optimizer),
            weight_decay=float(config.weight_decay),
            device=context.get("device") or config.device,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Reference-model training failed (%s); Moderate scores use an untrained model", exc)
    return model


def initial_mask(mode: str, context: Dict[str, Any], k: int, config: Table5Config,
                 seed: Optional[int] = None, reference_model: Any = None) -> Dict[str, Any]:
    """Produce the initial mask for a given initialization mode."""
    mode = str(mode).strip().lower()
    if mode in {"moderate", "moderate-init", "lbcs+moderate"}:
        return moderate_init_mask(context, k, seed=seed, config=config,
                                  reference_model=reference_model)
    n = int(context.get("n", k))
    return {"mask": random_init_mask(n, k, seed=seed), "strategy": "random", "fallback": False}


def install_initial_mask(runner: Any, mask: np.ndarray) -> str:
    """Push ``mask`` into an existing LBCS runner if it exposes a hook.

    Returns a short description of the strategy that succeeded (or ``"none"``).
    The driver itself does not depend on this helper: the primary search path
    (see :func:`select_lbcs_mask`) already accepts an explicit initial mask.
    """
    if runner is None or mask is None:
        return "none"
    # 1) explicit attribute hooks
    for name in ("initial_mask", "init_mask", "mask_init", "_initial_mask"):
        try:
            if hasattr(runner, name):
                setattr(runner, name, mask)
                return "attribute:" + name
        except Exception:  # pragma: no cover
            pass
    # 2) explicit method hooks
    for name in ("set_initial_mask", "set_init_mask"):
        fn = getattr(runner, name, None)
        if callable(fn):
            try:
                fn(mask)
                return "method:" + name
            except Exception:  # pragma: no cover
                pass
    # 3) keyword / positional variants of initialize_masks
    init_fn = getattr(runner, "initialize_masks", None)
    if callable(init_fn):
        for kwargs in ({"initial_mask": mask}, {"mask": mask}, {"init_mask": mask}):
            try:
                init_fn(**kwargs)
                return "initialize_masks:" + next(iter(kwargs))
            except TypeError:
                continue
            except Exception:  # pragma: no cover
                break
        try:
            init_fn(mask)
            return "initialize_masks:positional"
        except TypeError:
            pass
        except Exception:  # pragma: no cover
            pass
        # 4) monkeypatch: force the returned initialization to be our mask
        try:
            original = init_fn

            def _patched(*args: Any, **kwargs: Any) -> Any:  # noqa: WPS430
                try:
                    return original(mask, *args, **kwargs)
                except TypeError:
                    pass
                except Exception:  # pragma: no cover
                    pass
                try:
                    return original(*args, initial_mask=mask, **kwargs)
                except Exception:  # pragma: no cover
                    pass
                return mask

            try:
                runner.initialize_masks = _patched  # type: ignore[assignment]
            except Exception:  # pragma: no cover
                return "none"
            return "monkeypatch:initialize_masks"
        except Exception:  # pragma: no cover
            pass
    return "none"


# ---------------------------------------------------------------------------
# Algorithm 1 (LBCS) with a configurable initial mask
# ---------------------------------------------------------------------------

def select_lbcs_mask(context: Dict[str, Any], k: int, config: Table5Config,
                     seed: Optional[int] = None, init_mode: str = "random",
                     initial: Optional[np.ndarray] = None,
                     lbcs: Any = None, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Run Algorithm 1 (LBCS) on ``context`` for coreset size ``k``.

    ``init_mode`` selects the mask initialization; ``initial`` overrides it with
    an explicit binary mask when provided.  A caller-supplied ``lbcs`` callable
    is used verbatim (dependency injection for offline tests).
    """
    log = logger or LOGGER
    cfg = config or context.get("config") or Table5Config()
    n = int(context.get("n", k))
    started = time.time()

    init_info: Dict[str, Any] = {"strategy": "random", "fallback": False}
    if initial is not None:
        mask0 = binarize_mask(initial)
    else:
        init_info = initial_mask(init_mode, context, k, cfg, seed=seed)
        mask0 = init_info["mask"]

    # --- injected runner (tests / custom harnesses) ------------------------
    if callable(lbcs):
        try:
            out = lbcs(context, mask0, k, cfg, seed)
        except TypeError:
            out = lbcs(mask0)
        return _normalize_lbcs_result(out, mask0, k, init_info, started, cfg, init_mode=init_mode)

    device = context.get("device") or cfg.device
    set_seed(seed)

    # --- primary path: LexiFlow + MaskObjectiveEvaluator -------------------
    result = None
    try:
        from lbcs_repro.lbcs.bilevel import InnerTrainConfig, InnerTrainer, make_inner_train_fn  # noqa: WPS433
        from lbcs_repro.lbcs.lexiflow import LexiFlow  # noqa: WPS433
        from lbcs_repro.lbcs.objectives import MaskObjectiveEvaluator  # noqa: WPS433

        inner_cfg = InnerTrainConfig(
            optimizer=str(cfg.inner_optimizer),
            lr=float(cfg.inner_lr),
            epochs=int(cfg.inner_epochs),
            weight_decay=float(cfg.weight_decay),
            batch_size=int(cfg.batch_size),
            device=device,
        )
        trainer = InnerTrainer(config=inner_cfg, device=device)
        inner_train_fn = make_inner_train_fn(
            trainer,
            context.get("inner_factory"),
            dataset=None,
            loader=context.get("train_loader"),
            warm_start=bool(cfg.warm_start),
            n=n,
        )
        evaluator = MaskObjectiveEvaluator(
            inner_train_fn=inner_train_fn,
            full_loader=context.get("eval_loader"),
            device=device,
            cache=True,
        )
        optimizer = LexiFlow(
            objective=evaluator.evaluate,
            epsilon=float(cfg.epsilon),
            delta_init=float(cfg.delta_init),
            delta_lower=float(cfg.delta_lower),
            max_iters=int(cfg.T),
            dimension=n,
            seed=seed,
        )
        result = optimizer.run(initial_mask=mask0, max_iters=int(cfg.T))
    except Exception as exc:  # pragma: no cover - heavyweight optional path
        log.warning("LexiFlow-based Algorithm 1 path failed (%s); trying the LBCS driver", exc)
        result = None

    # --- fallback: the packaged LBCS class ---------------------------------
    if result is None:
        try:
            from lbcs_repro.lbcs.bilevel import LBCS, InnerTrainConfig, LBCSConfig  # noqa: WPS433

            inner_cfg = InnerTrainConfig(
                optimizer=str(cfg.inner_optimizer),
                lr=float(cfg.inner_lr),
                epochs=int(cfg.inner_epochs),
                weight_decay=float(cfg.weight_decay),
                batch_size=int(cfg.batch_size),
                device=device,
            )
            lbcs_cfg = LBCSConfig(
                k=int(k),
                epsilon=float(cfg.epsilon),
                T=int(cfg.T),
                delta_init=float(cfg.delta_init),
                delta_lower=float(cfg.delta_lower),
                warm_start=bool(cfg.warm_start),
                group_size=int(cfg.group_size),
                seed=seed,
                device=device,
            )
            runner = LBCS(
                model_factory=context.get("inner_factory"),
                n=n,
                k=int(k),
                config=lbcs_cfg,
                dataset=None,
                train_loader=context.get("train_loader"),
                eval_loader=context.get("eval_loader"),
                inner_config=inner_cfg,
                device=device,
                seed=seed,
            )
            strategy = install_initial_mask(runner, mask0)
            init_info["install_strategy"] = strategy
            result = runner.run()
        except Exception as exc:  # pragma: no cover
            log.warning("LBCS driver unavailable (%s); returning the initial mask", exc)
            result = None

    if result is None:
        return {
            "mask": mask0,
            "coreset_size": mask_size(mask0),
            "f1": float("nan"),
            "f2": float(mask_size(mask0)),
            "iterations": 0,
            "restarts": 0,
            "wall_time": time.time() - started,
            "init": init_info,
            "init_mode": init_mode,
            "stopped_reason": "no_search",
            "result": None,
        }

    return _normalize_lbcs_result(result, mask0, k, init_info, started, cfg, init_mode=init_mode)


def _normalize_lbcs_result(result: Any, mask0: np.ndarray, k: int,
                           init_info: Dict[str, Any], started: float,
                           cfg: Table5Config, init_mode: str = "random") -> Dict[str, Any]:
    """Coerce an LBCS/LexiFlow result object into a plain dict."""
    f1: Any = None
    f2: Any = None
    iters: Any = 0
    restarts: Any = 0
    reason: Any = ""
    if isinstance(result, dict):
        best = result.get("mask", result.get("best_mask"))
        f1 = result.get("f1")
        f2 = result.get("f2", result.get("size"))
        iters = result.get("iterations", 0)
        restarts = result.get("restarts", 0)
        reason = result.get("stopped_reason", "")
    else:
        best = getattr(result, "mask", None)
        if best is None:
            best = getattr(result, "best_mask", None)
        f1 = getattr(result, "f1", None)
        f2 = getattr(result, "f2", getattr(result, "size", None))
        iters = getattr(result, "iterations", getattr(result, "num_iterations", 0))
        restarts = getattr(result, "restarts", 0)
        reason = getattr(result, "stopped_reason", "")
        if f1 is None:
            best_F = getattr(result, "best_F", None)
            if best_F is not None:
                try:
                    arr = np.asarray(best_F, dtype=np.float64).reshape(-1)
                    f1, f2 = float(arr[0]), float(arr[1])
                except Exception:  # pragma: no cover
                    pass

    if best is None:
        best = mask0
    mask = binarize_mask(best)
    if int(mask.sum()) == 0:
        mask = binarize_mask(mask0)

    f1_ok = f1 is not None and np.isfinite(float(f1))
    f2_ok = f2 is not None and np.isfinite(float(f2))
    return {
        "mask": mask,
        "coreset_size": mask_size(mask),
        "f1": float(f1) if f1_ok else float("nan"),
        "f2": float(f2) if f2_ok else float(mask_size(mask)),
        "iterations": int(iters or 0),
        "restarts": int(restarts or 0),
        "wall_time": time.time() - started,
        "init": init_info,
        "init_mode": init_mode,
        "stopped_reason": str(reason),
        "result": result,
    }


# ---------------------------------------------------------------------------
# Single run / sweep
# ---------------------------------------------------------------------------

def run_single_cell(k: int, init_mode: str, repeat: int, config: Optional[Table5Config] = None,
                    context: Optional[Dict[str, Any]] = None, seed: Optional[int] = None,
                    logger: Optional[logging.Logger] = None,
                    select_lbcs_fn: Optional[Any] = None,
                    train_eval_fn: Optional[Any] = None,
                    lbcs: Any = None,
                    reference_model: Any = None) -> Dict[str, Any]:
    """Run one ``(k, init_mode, repeat)`` trial and record the outcome."""
    cfg = config or Table5Config()
    log = logger or LOGGER
    ctx = context if context is not None else build_context(cfg.dataset, cfg, device=cfg.device)
    seed = resolve_seed(cfg.seed, repeat) if seed is None else int(seed)

    record: Dict[str, Any] = {
        "k": int(k),
        "init_mode": str(init_mode),
        "repeat": int(repeat),
        "seed": int(seed),
        "failed": False,
    }
    try:
        res = initial_mask(init_mode, ctx, k, cfg, seed=seed, reference_model=reference_model)
        if select_lbcs_fn is not None:
            sel = select_lbcs_fn(ctx, k, cfg, seed, init_mode, reference_model)
        else:
            sel = select_lbcs_mask(
                ctx, k, cfg, seed=seed, init_mode=init_mode,
                initial=res["mask"], lbcs=lbcs, logger=log,
            )
        sel.setdefault("init", res)
        size = int(sel.get("coreset_size", mask_size(sel.get("mask"))))
        record.update({
            "coreset_size": size,
            "f1": float(sel.get("f1", float("nan"))),
            "f2": float(sel.get("f2", float(size))),
            "init_strategy": (sel.get("init") or {}).get("strategy", str(init_mode)),
            "init_fallback": bool((sel.get("init") or {}).get("fallback", False)),
            "iterations": int(sel.get("iterations", 0)),
            "restarts": int(sel.get("restarts", 0)),
            "search_wall_time": float(sel.get("wall_time", 0.0)),
        })

        eval_out = train_and_evaluate(ctx, sel["mask"], seed=seed, config=cfg, train_fn=train_eval_fn)
        record.update({
            "accuracy": float(eval_out.get("accuracy", float("nan"))),
            "target_coreset_size": int(eval_out.get("coreset_size", size)),
            "accuracy_per_point": float(eval_out.get("accuracy_per_point", float("nan"))),
        })
        if cfg.log_every and (repeat % cfg.log_every == 0):
            log.info("k=%d %s repeat=%d acc=%.2f size=%d", k, init_mode, repeat,
                     record["accuracy"], record["coreset_size"])
    except Exception as exc:  # pragma: no cover - robustness for long sweeps
        log.warning("trial failed (k=%d, mode=%s, repeat=%d): %s", k, init_mode, repeat, exc)
        fallback_size = mask_size(initial_mask(init_mode, ctx, k, cfg, seed=seed)["mask"])
        record.update({
            "failed": True,
            "error": str(exc),
            "accuracy": float("nan"),
            "coreset_size": int(fallback_size),
            "f1": float("nan"),
            "f2": float("nan"),
        })
    return record


def run_table5(config: Optional[Table5Config] = None, logger: Optional[logging.Logger] = None,
               cell_runner: Optional[Any] = None, context_builder: Optional[Any] = None,
               reference_model: Any = None, **overrides: Any) -> Dict[str, Any]:
    """Run the Table 5 sweep: random init vs. Moderate init, per ``k``.

    ``cell_runner`` may be injected to run offline tests without torch/data.
    """
    log = logger or LOGGER
    if config is None:
        config = Table5Config()
    if overrides:
        config = config.with_overrides(**overrides)
    log.info("Table 5 sweep: dataset=%s ks=%s modes=%s eps=%.2f T=%d repeats=%d",
             config.dataset, tuple(config.ks), tuple(config.init_modes),
             config.epsilon, config.T, config.repeats)

    context = None
    if cell_runner is None:
        builder = context_builder or build_context
        context = builder(config.dataset, config, device=config.device)
        # One shared reference model supplies Moderate scores for every k.
        if reference_model is None and "moderate" in tuple(m.lower() for m in config.init_modes):
            reference_model = train_reference_proxy(context, config, seed=resolve_seed(config.seed, 0))

    records: List[Dict[str, Any]] = []
    for k in config.ks:
        for mode in config.init_modes:
            for repeat in range(int(config.repeats)):
                if cell_runner is not None:
                    try:
                        rec = cell_runner(k, mode, repeat, config)
                    except TypeError:
                        rec = cell_runner(k, mode, repeat)
                else:
                    rec = run_single_cell(
                        k, mode, repeat, config=config, context=context,
                        logger=log, reference_model=reference_model,
                    )
                rec.setdefault("k", int(k))
                rec.setdefault("init_mode", str(mode))
                rec.setdefault("repeat", int(repeat))
                records.append(rec)

    cells = aggregate_results(records)
    checks = direction_checks(cells, config)
    table = format_table5(cells, config)
    artifacts = save_results(cells, records, checks, config) if config.save_artifacts else {}
    if config.plot and config.save_artifacts:
        plot_path = plot_table5(cells, os.path.join(config.output_dir, "table5.png"), config)
        if plot_path:
            artifacts["plot"] = plot_path

    if config.verbose:
        print(table)

    return {
        "table": table,
        "config": config.to_dict(),
        "cells": cells,
        "records": records,
        "checks": checks,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Aggregation / validation / formatting
# ---------------------------------------------------------------------------

def aggregate_results(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[int, str], Table5Cell]:
    """Group per-repeat records into mean/std cells keyed by ``(k, init_mode)``."""
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for rec in records:
        key = (int(rec.get("k", 0)), str(rec.get("init_mode", "random")))
        grouped.setdefault(key, []).append(dict(rec))

    cells: Dict[Tuple[int, str], Table5Cell] = {}
    for key, recs in grouped.items():
        ok = [r for r in recs if not r.get("failed")]
        acc_mean, acc_std = mean_std([float(r.get("accuracy", float("nan"))) for r in ok])
        size_mean, size_std = mean_std([float(r.get("coreset_size", float("nan"))) for r in ok])
        f1_mean, f1_std = mean_std([float(r.get("f1", float("nan"))) for r in ok])
        f2_mean, f2_std = mean_std([float(r.get("f2", float("nan"))) for r in ok])
        cells[key] = Table5Cell(
            k=key[0],
            init_mode=key[1],
            accuracy_mean=acc_mean,
            accuracy_std=acc_std,
            size_mean=size_mean,
            size_std=size_std,
            f1_mean=f1_mean,
            f1_std=f1_std,
            f2_mean=f2_mean,
            f2_std=f2_std,
            repeats=len(recs),
            failures=len(recs) - len(ok),
            raw=recs,
        )
    return cells


def direction_checks(cells: Dict[Tuple[int, str], Table5Cell],
                     config: Optional[Table5Config] = None) -> Dict[str, Any]:
    """Check the paper's qualitative claims for Table 5.

    Claims (Section 6): (i) initializing the mask with Moderate and refining it
    with LBCS yields a *higher* mean test accuracy than random initialization;
    (ii) both variants keep the achieved coreset size below the predefined ``k``.
    """
    cfg = config or Table5Config()
    per_k: Dict[str, Any] = {}
    wins = 0
    compared = 0
    for k in cfg.ks:
        base = cells.get((int(k), "random"))
        mod = cells.get((int(k), "moderate"))
        entry: Dict[str, Any] = {"k": int(k)}
        if base is not None:
            entry["lbcs"] = [base.accuracy_mean, base.accuracy_std]
            entry["lbcs_size"] = [base.size_mean, base.size_std]
            entry["lbcs_size_below_k"] = bool(base.size_mean <= float(k) + 1e-6)
        if mod is not None:
            entry["lbcs_moderate"] = [mod.accuracy_mean, mod.accuracy_std]
            entry["lbcs_moderate_size"] = [mod.size_mean, mod.size_std]
            entry["lbcs_moderate_size_below_k"] = bool(mod.size_mean <= float(k) + 1e-6)
        if base is not None and mod is not None:
            delta = mod.accuracy_mean - base.accuracy_mean
            entry["delta_accuracy"] = float(delta)
            entry["moderate_better"] = bool(delta > 0.0)
            compared += 1
            wins += int(delta > 0.0)
        entry["paper_reference"] = {
            "random": list(PAPER_TABLE5[LBCS_LABEL].get(int(k), ())),
            "moderate": list(PAPER_TABLE5[MODERATE_INIT_LABEL].get(int(k), ())),
        }
        per_k[str(k)] = entry

    sizes_below = [
        bool(v.get("lbcs_size_below_k", True)) and bool(v.get("lbcs_moderate_size_below_k", True))
        for v in per_k.values()
    ]
    return {
        "per_k": per_k,
        "moderate_init_better_count": wins,
        "compared_count": compared,
        "moderate_init_better": bool(compared > 0 and wins == compared),
        "sizes_below_k": bool(all(sizes_below)) if sizes_below else True,
        "note": (
            "Paper claim (Section 6): mask initialization by Moderate followed by LBCS "
            "refinement improves the final test accuracy; sizes remain below k."
        ),
    }


def format_table5(cells: Dict[Tuple[int, str], Table5Cell],
                  config: Optional[Table5Config] = None) -> str:
    """Render the results in the layout of paper Table 5."""
    cfg = config or Table5Config()
    header = f"{'k':>6} | {LBCS_LABEL:>21} | {MODERATE_INIT_LABEL:>21}"
    sep = "-" * len(header)
    lines = [
        "Table 5: Mean and standard deviation of test accuracy (%) on "
        f"{cfg.dataset} with various predefined coreset sizes.",
        f"'{MODERATE_INIT_LABEL}' means the mask is initialized by 'Moderate' and then "
        "is refined by our LBCS.",
        "",
        header,
        sep,
    ]
    for k in cfg.ks:
        base = cells.get((int(k), "random"))
        mod = cells.get((int(k), "moderate"))
        base_txt = format_mean_std(base.accuracy_mean, base.accuracy_std) if base else "-"
        mod_txt = format_mean_std(mod.accuracy_mean, mod.accuracy_std) if mod else "-"
        lines.append(f"{int(k):>6} | {base_txt:>21} | {mod_txt:>21}")
    lines.append(sep)
    lines.append("")
    lines.append("Achieved coreset sizes (mean +- std):")
    for k in cfg.ks:
        base = cells.get((int(k), "random"))
        mod = cells.get((int(k), "moderate"))
        base_txt = format_mean_std(base.size_mean, base.size_std) if base else "-"
        mod_txt = format_mean_std(mod.size_mean, mod.size_std) if mod else "-"
        lines.append(f"{int(k):>6} | {base_txt:>21} | {mod_txt:>21}")
    lines.append("")
    lines.append("Paper reference:")
    for k in cfg.ks:
        base = PAPER_TABLE5[LBCS_LABEL].get(int(k))
        mod = PAPER_TABLE5[MODERATE_INIT_LABEL].get(int(k))
        lines.append(
            f"{int(k):>6} | {format_mean_std(*base) if base else '-':>21} | "
            f"{format_mean_std(*mod) if mod else '-':>21}"
        )
    return "\n".join(lines)


def plot_table5(cells: Dict[Tuple[int, str], Table5Cell], out_path: str,
                config: Optional[Table5Config] = None, dpi: int = 150) -> Optional[str]:
    """Optional accuracy-vs-k plot for the two initialization modes."""
    try:
        import matplotlib  # noqa: WPS433

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return None
    cfg = config or Table5Config()
    xs = [int(k) for k in cfg.ks]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for mode, label, marker in (("random", LBCS_LABEL, "o"), ("moderate", MODERATE_INIT_LABEL, "s")):
        ys, errs = [], []
        for k in xs:
            cell = cells.get((int(k), mode))
            ys.append(cell.accuracy_mean if cell else np.nan)
            errs.append(cell.accuracy_std if cell else 0.0)
        ax.errorbar(xs, ys, yerr=errs, marker=marker, capsize=3, label=label)
    ax.set_xlabel("predefined coreset size k")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Table 5: influence of mask initialization")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_results(cells: Dict[Tuple[int, str], Table5Cell], records: Sequence[Dict[str, Any]],
                 checks: Dict[str, Any], config: Table5Config) -> Dict[str, str]:
    """Persist the Table 5 artifacts (JSON/CSV/TXT/JSONL/checks)."""
    out_dir = config.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    payload = {
        "config": _jsonable(config.to_dict()),
        "cells": [c.to_dict() for c in cells.values()],
        "checks": _jsonable(checks),
        "paper_reference": _jsonable(PAPER_TABLE5),
    }
    json_path = os.path.join(out_dir, "table5.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    paths["json"] = json_path

    csv_path = os.path.join(out_dir, "table5.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["k", "init_mode", "accuracy_mean", "accuracy_std",
                         "size_mean", "size_std", "f1_mean", "f1_std",
                         "repeats", "failures"])
        for cell in cells.values():
            writer.writerow([cell.k, cell.init_mode, f"{cell.accuracy_mean:.4f}",
                             f"{cell.accuracy_std:.4f}", f"{cell.size_mean:.4f}",
                             f"{cell.size_std:.4f}", f"{cell.f1_mean:.4f}",
                             f"{cell.f1_std:.4f}", cell.repeats, cell.failures])
    paths["csv"] = csv_path

    txt_path = os.path.join(out_dir, "table5.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(format_table5(cells, config))
        handle.write("\n\nChecks:\n")
        handle.write(json.dumps(_jsonable(checks), indent=2))
        handle.write("\n")
    paths["txt"] = txt_path

    raw_path = os.path.join(out_dir, "table5_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(_jsonable({k: v for k, v in rec.items() if k != "result"})) + "\n")
    paths["jsonl"] = raw_path

    checks_path = os.path.join(out_dir, "table5_checks.json")
    with open(checks_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(checks), handle, indent=2)
    paths["checks"] = checks_path
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Table 5: influence of mask initialization on LBCS (Section 6)."
    )
    parser.add_argument("--config", type=str, default=None, help="optional YAML config file")
    parser.add_argument("--paper", action="store_true", help="use the paper protocol (10 repeats, T=500)")
    parser.add_argument("--smoke", action="store_true", help="tiny offline smoke configuration")
    parser.add_argument("--selftest", action="store_true", help="run the offline self-test")
    parser.add_argument("--ks", type=int, nargs="*", default=None, help="predefined coreset sizes")
    parser.add_argument("--modes", type=str, nargs="*", default=None,
                        help="initialization modes (random, moderate)")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> Table5Config:
    if args.smoke:
        cfg = Table5Config.smoke()
    elif args.paper:
        cfg = Table5Config.paper()
    else:
        cfg = Table5Config()
    if args.config:
        try:
            import yaml  # noqa: WPS433

            with open(args.config, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            cfg = Table5Config.from_dict(data).with_overrides(**cfg.to_dict())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not load config %s (%s)", args.config, exc)

    overrides: Dict[str, Any] = {}
    if args.ks:
        overrides["ks"] = tuple(args.ks)
    if args.modes:
        overrides["init_modes"] = tuple(args.modes)
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    if args.T is not None:
        overrides["T"] = args.T
    if args.epsilon is not None:
        overrides["epsilon"] = args.epsilon
    if args.inner_epochs is not None:
        overrides["inner_epochs"] = args.inner_epochs
    if args.target_epochs is not None:
        overrides["target_epochs"] = args.target_epochs
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.no_save:
        overrides["save_artifacts"] = False
        overrides["plot"] = False
    if args.verbose:
        overrides["verbose"] = True
    return cfg.with_overrides(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_argparser().parse_args(argv)
    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1
    config = _config_from_args(args)
    out = run_table5(config)
    print(out["table"])
    if out["artifacts"]:
        print("\nArtifacts:", json.dumps(out["artifacts"], indent=2))
    return 0


# ---------------------------------------------------------------------------
# Offline self-test (no torch, no dataset download)
# ---------------------------------------------------------------------------

class _SyntheticCellRunner:
    """Deterministic synthetic runner used by the offline self-test."""

    def __init__(self, moderate_gain: float = 0.5) -> None:
        self.moderate_gain = float(moderate_gain)

    def __call__(self, k: int, mode: str, repeat: int, config: Table5Config) -> Dict[str, Any]:
        base = 70.0 + 4.0 * np.log10(max(int(k), 10))
        gain = self.moderate_gain if str(mode).startswith("moderate") else 0.0
        acc = base + gain + 0.1 * (repeat % 3)
        size = float(k) - 50.0 - 5.0 * (repeat % 2)
        return {
            "k": int(k),
            "init_mode": str(mode),
            "repeat": int(repeat),
            "seed": int(repeat),
            "failed": False,
            "accuracy": float(acc),
            "coreset_size": int(size),
            "f1": float(1.0 + 0.01 * repeat),
            "f2": float(size),
            "init_strategy": str(mode),
        }


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline validation of masks, aggregation, formatting and checks."""
    checks: Dict[str, Any] = {"ok": True, "steps": []}

    def record(name: str, passed: bool, info: Any = None) -> None:
        checks["steps"].append({"name": name, "passed": bool(passed), "info": info})
        checks["ok"] = checks["ok"] and bool(passed)
        if verbose:
            print(f"[{'ok' if passed else 'FAIL'}] {name}")

    # 1) mask helpers / Appendix A projection
    m = np.array([-1.5, -1.0, -0.5, -1e-12, 0.0, 1e-12, 0.7, 1.5], dtype=np.float64)
    proj = binarize_mask(m)
    expected = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    record("appendix_a_projection", np.array_equal(proj, expected), proj.tolist())
    record("mask_size_equals_l0", mask_size(proj) == int(proj.sum()))

    # 2) random initialization has ||m||_0 = k
    mask = random_init_mask(100, 17, seed=3)
    record("random_init_size", mask_size(mask) == 17, mask_size(mask))
    record("random_init_deterministic", np.array_equal(mask, random_init_mask(100, 17, seed=3)))

    # 3) install_initial_mask on a stub runner
    class _StubRunner:
        def __init__(self) -> None:
            self.initial_mask = None

        def initialize_masks(self) -> np.ndarray:
            return np.zeros(10, dtype=np.float32)

    stub = _StubRunner()
    strategy = install_initial_mask(stub, mask[:10])
    record("install_initial_mask", stub.initial_mask is not None and strategy != "none", strategy)

    # 4) aggregation of synthetic records
    runner = _SyntheticCellRunner(moderate_gain=0.5)
    cfg = Table5Config.smoke(ks=(1000, 2000), init_modes=("random", "moderate"), repeats=3,
                             save_artifacts=False, plot=False)
    records = [
        runner(k, mode, r, cfg)
        for k in cfg.ks for mode in cfg.init_modes for r in range(cfg.repeats)
    ]
    cells = aggregate_results(records)
    record("aggregation_cell_count", len(cells) == 4, len(cells))
    cell = cells[(1000, "random")]
    record("aggregation_std_nonnegative", cell.accuracy_std >= 0.0, cell.accuracy_std)

    # 5) direction checks detect the Moderate advantage
    chk = direction_checks(cells, cfg)
    record("direction_checks_moderate_better", bool(chk.get("moderate_init_better")),
           chk.get("moderate_init_better_count"))
    record("direction_checks_sizes", bool(chk.get("sizes_below_k")))

    # 6) formatting
    text = format_table5(cells, cfg)
    record("format_contains_headers", LBCS_LABEL in text and MODERATE_INIT_LABEL in text)
    record("format_contains_k", "1000" in text and "2000" in text)

    # 7) run_table5 through injection (no torch / data)
    out = run_table5(cfg, cell_runner=runner)
    record("run_table5_injected", len(out["records"]) == 12 and bool(out["checks"]),
           len(out["records"]))

    # 8) config round-trip / paper defaults
    restored = Table5Config.from_dict({"table5": cfg.to_dict()})
    record("config_roundtrip", restored.ks == cfg.ks and restored.T == cfg.T)
    paper_cfg = Table5Config.paper()
    record("config_paper_defaults",
           paper_cfg.T == PAPER_T and paper_cfg.epsilon == PAPER_EPSILON
           and tuple(paper_cfg.ks) == PAPER_KS and paper_cfg.repeats == PAPER_REPEATS)

    # 9) all three datasets of Section 5.2 normalize correctly (context builder contract)
    record("canonical_dataset_aliases",
           canonical_dataset("fmnist") == "F-MNIST" and canonical_dataset("CIFAR10") == "CIFAR-10")

    if verbose:
        print("\nSelf-test:", "PASSED" if checks["ok"] else "FAILED")
    return checks


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
