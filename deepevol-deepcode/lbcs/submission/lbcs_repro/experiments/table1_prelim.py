"""Table 1 driver: preliminary presentation of LBCS's superiority (Section 5.1).

Reproduces the paper's Table 1 (Section 5.1, "Preliminary Presentation of
Algorithm's Superiority"):

    k   Objectives      Initial   eps=0.2        eps=0.3        eps=0.4
    200 f1(m)          3.21      1.92 +- 0.33   2.26 +- 0.35   2.48 +- 0.30
        f2(m)          200       190.7 +- 3.9   185.0 +- 4.6   175.5 +- 7.7
    400 f1(m)          2.16      1.05 +- 0.26   1.29 +- 0.33   1.82 +- 0.41
        f2(m)          400       384.1 +- 4.4   373.0 +- 6.0   366.2 +- 8.1

Paper-stated protocol (Section 5.1, verbatim):

    * benchmark: MNIST-S = 1,000 examples randomly sampled from original MNIST
    * proxy network: a CNN stacked with two blocks of convolution, dropout,
      max-pooling and ReLU (the ConvNet of Zhou et al. 2022 / Borsos et al. 2020)
    * predefined coreset size k in {200, 400}
    * voluntary performance compromise eps in {0.2, 0.3, 0.4}
    * 20 repeats, mean +- std reported
    * two objectives reported: f1(m) (objective (O1), Eq. (1)) and
      f2(m) = ||m||_0 (objective (O2), Eq. (2)); f1 has higher priority

Paper-stated qualitative findings checked here:

    1. both achieved f1(m) and f2(m) are lower than the initialized values,
    2. a larger epsilon leads to a smaller f2(m) over multiple experiments,
    3. on average a larger epsilon leads to a larger f1(m), although in any
       single experiment that is not guaranteed (the compromise only upper
       bounds f1, see Eq. (7)).

Values NOT stated by the paper (explicitly labelled SUGGESTED below and
overridable from ``configs/section5_1.yaml``): the number of outer iterations T
(SUGGESTED_T = 500, the addendum's default when a section states no T), the
Section 5.1 inner-loop optimizer (Adam lr=0.001 per Section 5.2's statement, or
SGD lr=0.1 momentum=0.9 100 epochs as used for Figure 1), batch size, weight
decay, delta_init / delta_lower (Algorithm 2 is not numerically specified).

Out of scope: ImageNet-1k (Section 5.4), continual learning (Appendix E.5) and
streaming (Appendix E.6).
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

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paper-stated constants (Section 5.1)
# --------------------------------------------------------------------------
PAPER_KS: Tuple[int, ...] = (200, 400)
PAPER_EPSILONS: Tuple[float, ...] = (0.2, 0.3, 0.4)
PAPER_REPEATS: int = 20
PAPER_N: int = 1000  # MNIST-S = 1,000 random MNIST samples

# Paper's reported Table 1 values: (k, epsilon) -> (f1_mean, f1_std, f2_mean, f2_std).
PAPER_TABLE1: Dict[Tuple[int, float], Tuple[float, float, float, float]] = {
    (200, 0.2): (1.92, 0.33, 190.7, 3.9),
    (200, 0.3): (2.26, 0.35, 185.0, 4.6),
    (200, 0.4): (2.48, 0.30, 175.5, 7.7),
    (400, 0.2): (1.05, 0.26, 384.1, 4.4),
    (400, 0.3): (1.29, 0.33, 373.0, 6.0),
    (400, 0.4): (1.82, 0.41, 366.2, 8.1),
}
# "Initial" column: random mask with ||m||_0 = k.
PAPER_INITIAL_F1: Dict[int, float] = {200: 3.21, 400: 2.16}

# --------------------------------------------------------------------------
# SUGGESTED defaults (NOT stated by the paper)
# --------------------------------------------------------------------------
SUGGESTED_T: int = 500
SUGGESTED_INNER_EPOCHS: int = 100
SUGGESTED_INNER_OPTIMIZER: str = "adam"
SUGGESTED_INNER_LR: float = 0.001
SUGGESTED_INNER_MOMENTUM: float = 0.9
SUGGESTED_WEIGHT_DECAY: float = 0.0
SUGGESTED_BATCH_SIZE: int = 128
SUGGESTED_EVAL_BATCH_SIZE: int = 256
SUGGESTED_DELTA_INIT: float = 0.1
SUGGESTED_DELTA_LOWER: float = 1e-3
DEFAULT_OUTPUT_DIR: str = os.path.join("results", "table1")

__all__ = [
    "Table1Config",
    "Table1Cell",
    "PAPER_TABLE1",
    "PAPER_INITIAL_F1",
    "binarize_mask",
    "mask_indices",
    "build_table1_objective",
    "initial_reference",
    "run_single_configuration",
    "run_table1",
    "aggregate_results",
    "format_table1",
    "save_results",
    "main",
]

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _has_torch() -> bool:
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve ``None``/``"auto"`` to cuda/mps/cpu."""
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a binary / relaxed [-1,1] / probability mask to {0,1}.

    Mirrors Appendix A: values in [-1, 0) -> 0 and values in [0, 1] -> 1.
    """
    arr = np.asarray(mask, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.float32)
    if np.all((arr == 0.0) | (arr == 1.0)):
        return arr.astype(np.float32)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of the selected examples of a (possibly relaxed) mask."""
    return np.flatnonzero(binarize_mask(mask) > 0.0)


def _extract_f1f2(obj: Any) -> Tuple[Optional[float], Optional[float]]:
    """Pull (f1, f2) out of a MaskEvaluation / tuple / dict / scalar."""
    if obj is None:
        return None, None
    if isinstance(obj, dict):
        f1 = obj.get("f1")
        f2 = obj.get("f2")
        return (None if f1 is None else float(f1), None if f2 is None else float(f2))
    if isinstance(obj, (tuple, list)) and obj:
        f1 = float(obj[0])
        f2 = float(obj[1]) if len(obj) > 1 and obj[1] is not None else None
        return f1, f2
    f1 = getattr(obj, "f1", None)
    f2 = getattr(obj, "f2", None)
    if f1 is None and isinstance(obj, (int, float, np.floating)):
        return float(obj), None
    return (
        None if f1 is None else float(f1),
        None if f2 is None else float(f2),
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class Table1Config:
    """Configuration of the Section 5.1 / Table 1 experiment."""

    # --- paper-stated ---
    dataset: str = "MNIST-S"
    n: int = PAPER_N
    ks: Tuple[int, ...] = PAPER_KS
    epsilons: Tuple[float, ...] = PAPER_EPSILONS
    repeats: int = PAPER_REPEATS
    architecture: str = "ConvNet"

    # --- SUGGESTED (not paper-stated) ---
    T: int = SUGGESTED_T
    inner_epochs: int = SUGGESTED_INNER_EPOCHS
    inner_optimizer: str = SUGGESTED_INNER_OPTIMIZER
    inner_lr: float = SUGGESTED_INNER_LR
    inner_momentum: float = SUGGESTED_INNER_MOMENTUM
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1

    # --- run plumbing ---
    device: Optional[str] = None
    seed: int = 0
    num_workers: int = 0
    log_every: int = 0
    output_dir: str = DEFAULT_OUTPUT_DIR
    save_artifacts: bool = True
    verbose: bool = True

    # ---- presets -------------------------------------------------------
    @classmethod
    def paper(cls, **overrides: Any) -> "Table1Config":
        """Exact Section 5.1 protocol (k in {200,400}, eps in {0.2,0.3,0.4}, 20 repeats)."""
        cfg = cls(
            dataset="MNIST-S",
            n=PAPER_N,
            ks=tuple(PAPER_KS),
            epsilons=tuple(PAPER_EPSILONS),
            repeats=PAPER_REPEATS,
            architecture="ConvNet",
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "Table1Config":
        """Tiny configuration for CPU smoke tests."""
        cfg = cls(
            n=PAPER_N,
            ks=(200,),
            epsilons=(0.2,),
            repeats=1,
            T=5,
            inner_epochs=1,
            output_dir=os.path.join("results", "table1_smoke"),
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Table1Config":
        if not data:
            return cls()
        data = dict(data)
        for key in ("ks", "epsilons"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def with_overrides(self, **overrides: Any) -> "Table1Config":
        clean = {k: v for k, v in overrides.items() if v is not None}
        for key in ("ks", "epsilons"):
            if key in clean and clean[key] is not None:
                clean[key] = tuple(clean[key])
        allowed = set(self.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        return replace(self, **{k: v for k, v in clean.items() if k in allowed})

    def to_dict(self) -> Dict[str, Any]:
        out = {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        out["ks"] = list(out.get("ks", ()))
        out["epsilons"] = list(out.get("epsilons", ()))
        return out


# --------------------------------------------------------------------------
# Data / model / objective construction
# --------------------------------------------------------------------------


def _build_dataset_and_model(
    config: Table1Config, logger: Optional[logging.Logger] = None
) -> Tuple[Any, Any, Any]:
    """Return (dataset, eval_dataset, model_factory) for Section 5.1."""
    log = logger or LOGGER

    # MNIST-S: 1,000 random MNIST samples (Section 5.1). Evaluation is the full
    # MNIST test split because f1(m) is the loss on the full data D (Eq. (1)).
    try:
        from lbcs_repro.data.mnist_s import get_mnist_s

        dataset = get_mnist_s(size=config.n, seed=config.seed, train=True)
        eval_dataset = get_mnist_s(size=config.n, seed=config.seed, train=False)
    except Exception as exc:  # pragma: no cover - fallback to generic registry
        log.warning("MNIST-S unavailable (%s); falling back to generic MNIST loader", exc)
        from lbcs_repro.data.datasets import get_dataset

        dataset = get_dataset("MNIST", train=True)
        eval_dataset = get_dataset("MNIST", train=False)

    # ConvNet of Zhou et al. 2022: two blocks of convolution + dropout + max-pool + ReLU.
    try:
        from lbcs_repro.models.convnet import convnet_factory

        factory = convnet_factory()
    except Exception as exc:  # pragma: no cover
        from lbcs_repro.models import model_factory as _mf

        log.warning("ConvNet module unavailable (%s); using model registry", exc)
        factory = _mf(config.architecture)

    return dataset, eval_dataset, factory


def build_table1_objective(
    config: Table1Config,
    dataset: Any = None,
    eval_dataset: Any = None,
    model_factory: Any = None,
    logger: Optional[logging.Logger] = None,
):
    """Build the ``theta(m) -> F(m) = [f1(m), f2(m)]`` evaluator of Algorithm 1.

    Uses the shared :class:`lbcs_repro.lbcs.objectives.MaskObjectiveEvaluator` with
    the :class:`lbcs_repro.lbcs.bilevel.InnerTrainer` inner loop, so the returned
    object exposes ``evaluate(mask) -> MaskEvaluation`` with ``.f1`` / ``.f2``.
    """
    log = logger or LOGGER
    if dataset is None or eval_dataset is None or model_factory is None:
        dataset, eval_dataset, model_factory = _build_dataset_and_model(config, log)

    from lbcs_repro.data.datasets import make_loader
    from lbcs_repro.lbcs.bilevel import InnerTrainConfig, InnerTrainer, make_inner_train_fn
    from lbcs_repro.lbcs.objectives import MaskObjectiveEvaluator

    inner_config = InnerTrainConfig(
        optimizer=config.inner_optimizer,
        lr=config.inner_lr,
        momentum=config.inner_momentum,
        weight_decay=config.weight_decay,
        epochs=config.inner_epochs,
        batch_size=config.batch_size,
        device=config.device,
    )

    train_loader = make_loader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    eval_loader = make_loader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    trainer = InnerTrainer(config=inner_config, device=config.device)
    inner_fn = make_inner_train_fn(
        trainer,
        model_factory,
        dataset=dataset,
        loader=train_loader,
        n=config.n,
        warm_start=config.warm_start,
    )

    evaluator = MaskObjectiveEvaluator(
        inner_train_fn=inner_fn,
        full_loader=eval_loader,
        device=config.device,
        cache=True,
        logger=log,
    )
    # keep handles so callers (and the LBCS runner) reuse the exact loaders
    evaluator._table1_dataset = dataset
    evaluator._table1_eval_loader = eval_loader
    evaluator._table1_train_loader = train_loader
    return evaluator


def _run_lbcs(
    config: Table1Config,
    k: int,
    epsilon: float,
    seed: int,
    dataset: Any,
    eval_dataset: Any,
    model_factory: Any,
    objective: Any = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """One Algorithm 1 run for a given (k, epsilon, seed)."""
    log = logger or LOGGER
    from lbcs_repro.data.datasets import make_loader
    from lbcs_repro.lbcs.bilevel import InnerTrainConfig, LBCS, LBCSConfig

    eval_loader = make_loader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    inner_config = InnerTrainConfig(
        optimizer=config.inner_optimizer,
        lr=config.inner_lr,
        momentum=config.inner_momentum,
        weight_decay=config.weight_decay,
        epochs=config.inner_epochs,
        batch_size=config.batch_size,
        device=config.device,
    )
    lbcs_config = LBCSConfig(
        k=int(k),
        epsilon=float(epsilon),
        T=int(config.T),
        delta_init=config.delta_init,
        delta_lower=config.delta_lower,
        warm_start=config.warm_start,
        group_size=config.group_size,
        grouped=bool(config.group_size and config.group_size > 1),
        cache_evaluations=True,
        seed=int(seed),
        device=config.device,
    )

    runner = LBCS(
        model_factory=model_factory,
        n=config.n,
        k=int(k),
        epsilon=float(epsilon),
        T=int(config.T),
        dataset=dataset,
        eval_loader=eval_loader,
        inner_config=inner_config,
        device=config.device,
        seed=int(seed),
        warm_start=config.warm_start,
        cache_evaluations=True,
        cache_theta=False,
        delta_init=config.delta_init,
        delta_lower=config.delta_lower,
        log_every=config.log_every,
        logger=log,
        config=lbcs_config,
    )

    t0 = time.time()
    result = runner.run()
    wall = time.time() - t0

    f1, f2 = _extract_f1f2(result)
    if f1 is None:
        # LBCSResult stores f1/f2 directly; fall through to mask if needed.
        mask = getattr(result, "mask", None)
        if mask is not None and objective is not None:
            try:
                f1, f2_alt = _extract_f1f2(objective.evaluate(mask))
                if f2 is None:
                    f2 = f2_alt
            except Exception:  # pragma: no cover
                pass
    if f2 is None:
        mask = getattr(result, "mask", None)
        if mask is not None:
            f2 = float(len(mask_indices(mask)))
    return {
        "f1": None if f1 is None else float(f1),
        "f2": None if f2 is None else float(f2),
        "wall_time": float(wall),
        "iterations": int(getattr(result, "iterations", config.T) or config.T),
        "evaluations": int(getattr(result, "evaluations", 0) or 0),
        "restarts": int(getattr(result, "restarts", 0) or 0),
        "mask_size": None if f2 is None else int(round(float(f2))),
    }


def initial_reference(
    config: Table1Config,
    k: int,
    objective: Any,
    seed: int,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[float]]:
    """Evaluate the *initialized* mask (random k-subset) -> (f1, f2).

    Section 5.1's "Initial" column is f1(m) of a random mask with ||m||_0 = k;
    the corresponding f2(m) is exactly k.
    """
    from lbcs_repro.lbcs.masks import init_binary_mask

    mask = init_binary_mask(config.n, k=int(k), seed=int(seed))
    f1: Optional[float] = None
    try:
        ev = objective.evaluate(mask) if hasattr(objective, "evaluate") else objective(mask)
        f1, _f2 = _extract_f1f2(ev)
    except Exception as exc:  # pragma: no cover - evaluation fallback
        (logger or LOGGER).warning("initial evaluation failed: %s", exc)
    return {"initial_f1": f1, "initial_f2": float(len(mask_indices(mask)))}


# --------------------------------------------------------------------------
# Cells / aggregation
# --------------------------------------------------------------------------


@dataclass
class Table1Cell:
    """Aggregated (mean +- std over repeats) results for one (k, epsilon)."""

    k: int
    epsilon: float
    f1_mean: float
    f1_std: float
    f2_mean: float
    f2_std: float
    initial_f1_mean: float
    initial_f1_std: float
    initial_f2_mean: float
    repeats: int = 0
    failures: int = 0
    raw: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__ if f != "raw"}  # type: ignore[attr-defined]


def _mean_std(values: Sequence[Optional[float]]) -> Tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def aggregate_results(
    records: Iterable[Dict[str, Any]],
    config: Optional[Table1Config] = None,
) -> Dict[str, Table1Cell]:
    """Group per-repeat records into mean +- std cells keyed by (k, epsilon)."""
    buckets: Dict[Tuple[int, float], List[Dict[str, Any]]] = {}
    for rec in records:
        buckets.setdefault((int(rec["k"]), float(rec["epsilon"])), []).append(rec)

    cells: Dict[str, Table1Cell] = {}
    for (k, eps) in sorted(buckets.keys()):
        recs = buckets[(k, eps)]
        f1_mean, f1_std = _mean_std([r.get("f1") for r in recs])
        f2_mean, f2_std = _mean_std([r.get("f2") for r in recs])
        i1_mean, i1_std = _mean_std([r.get("initial_f1") for r in recs])
        i2_mean, _ = _mean_std([r.get("initial_f2") for r in recs])
        cells[f"k={k},eps={eps:g}"] = Table1Cell(
            k=k,
            epsilon=eps,
            f1_mean=f1_mean,
            f1_std=f1_std,
            f2_mean=f2_mean,
            f2_std=f2_std,
            initial_f1_mean=i1_mean,
            initial_f1_std=i1_std,
            initial_f2_mean=i2_mean,
            repeats=len(recs),
            failures=sum(1 for r in recs if r.get("f1") is None or r.get("f2") is None),
            raw=list(recs),
        )
    return cells


def _paper_reference(k: int, epsilon: float, which: str) -> Optional[float]:
    """Paper-reported value for (k, epsilon); None if unavailable."""
    entry = PAPER_TABLE1.get((int(k), float(round(epsilon, 2))))
    if entry is None:
        return None
    return entry[0] if which == "f1" else entry[2]


def _direction_checks(
    cells: Dict[str, Table1Cell], config: Optional[Table1Config] = None
) -> Dict[str, Any]:
    """Validate the three qualitative claims of Section 5.1."""
    checks: Dict[str, Any] = {}
    ks = sorted({c.k for c in cells.values()})

    # (1) both achieved f1 and f2 are lower than the initialized values
    lower: Dict[str, bool] = {}
    for key, cell in cells.items():
        f1_ok = (not np.isfinite(cell.f1_mean)) or (cell.f1_mean < cell.initial_f1_mean)
        f2_ok = (not np.isfinite(cell.f2_mean)) or (cell.f2_mean < cell.initial_f2_mean)
        lower[key] = bool(f1_ok and f2_ok)
    checks["achieved_lower_than_initial"] = lower
    checks["claim1_all_lower"] = bool(all(lower.values())) if lower else False

    # (2) larger epsilon -> smaller f2 (per k, non-increasing)
    def _seq(k: int, attr: str) -> List[float]:
        return [
            getattr(c, attr)
            for c in sorted((x for x in cells.values() if x.k == k), key=lambda c: c.epsilon)
        ]

    mono_f2: Dict[int, bool] = {}
    for k in ks:
        seq = _seq(k, "f2_mean")
        if all(np.isfinite(seq)):
            mono_f2[k] = bool(all(a >= b - 1e-6 for a, b in zip(seq, seq[1:])))
        else:
            mono_f2[k] = True
    checks["f2_nonincreasing_in_epsilon"] = mono_f2
    checks["claim2_f2_decreases_with_epsilon"] = bool(all(mono_f2.values())) if mono_f2 else False

    # (3) on average a larger epsilon leads to a larger f1 (mean-level trend)
    mono_f1: Dict[int, bool] = {}
    for k in ks:
        seq = _seq(k, "f1_mean")
        if all(np.isfinite(seq)):
            mono_f1[k] = bool(all(b >= a - 1e-9 for a, b in zip(seq, seq[1:])))
        else:
            mono_f1[k] = True
    checks["f1_nondecreasing_in_epsilon"] = mono_f1
    checks["claim3_f1_increases_with_epsilon_on_average"] = (
        bool(all(mono_f1.values())) if mono_f1 else False
    )

    # (4) direction comparison with the paper's reported means
    comparisons: Dict[str, Any] = {}
    for key, cell in cells.items():
        comparisons[key] = {
            "initial_f1_measured": cell.initial_f1_mean,
            "initial_f1_paper": PAPER_INITIAL_F1.get(cell.k),
            "f1_measured": cell.f1_mean,
            "f1_std_measured": cell.f1_std,
            "f1_paper": _paper_reference(cell.k, cell.epsilon, "f1"),
            "f2_measured": cell.f2_mean,
            "f2_std_measured": cell.f2_std,
            "f2_paper": _paper_reference(cell.k, cell.epsilon, "f2"),
        }
    checks["vs_paper"] = comparisons
    if config is not None:
        checks["config"] = config.to_dict()
    return checks


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_single_configuration(
    config: Table1Config,
    k: int,
    epsilon: float,
    repeats: Optional[int] = None,
    dataset: Any = None,
    eval_dataset: Any = None,
    model_factory: Any = None,
    objective: Any = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Run ``repeats`` independent LBCS runs for one (k, epsilon) pair."""
    log = logger or LOGGER
    repeats = int(repeats if repeats is not None else config.repeats)

    if dataset is None or eval_dataset is None or model_factory is None:
        dataset, eval_dataset, model_factory = _build_dataset_and_model(config, log)
    if objective is None:
        objective = build_table1_objective(
            config,
            dataset=dataset,
            eval_dataset=eval_dataset,
            model_factory=model_factory,
            logger=log,
        )

    records: List[Dict[str, Any]] = []
    for rep in range(repeats):
        from lbcs_repro.baselines.base import resolve_seed

        seed = int(resolve_seed(config.seed, rep))
        log.info("[Table1] k=%d eps=%.2f repeat %d/%d (seed=%d)", k, epsilon, rep + 1, repeats, seed)
        rec: Dict[str, Any] = {"k": int(k), "epsilon": float(epsilon), "repeat": rep, "seed": seed}
        try:
            rec.update(initial_reference(config, k, objective, seed=seed, logger=log))
        except Exception as exc:  # pragma: no cover
            log.warning("initial reference failed: %s", exc)
            rec.update({"initial_f1": None, "initial_f2": float(k)})
        try:
            rec.update(
                _run_lbcs(
                    config,
                    k=k,
                    epsilon=epsilon,
                    seed=seed,
                    dataset=dataset,
                    eval_dataset=eval_dataset,
                    model_factory=model_factory,
                    objective=objective,
                    logger=log,
                )
            )
        except Exception as exc:  # pragma: no cover - keeps the sweep alive
            log.warning("LBCS run failed (k=%d eps=%.2f rep=%d): %s", k, epsilon, rep, exc)
            rec.update({"f1": None, "f2": None, "error": repr(exc)})
        records.append(rec)
    return records


def run_table1(
    config: Optional[Table1Config] = None,
    dataset: Any = None,
    eval_dataset: Any = None,
    model_factory: Any = None,
    runner: Optional[Callable[[Table1Config, int, float, int], Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Top-level Table 1 driver.

    Parameters
    ----------
    config : Table1Config, optional
        Experiment configuration (defaults to the Section 5.1 protocol).
    dataset, eval_dataset, model_factory : optional
        Injected data/model handles (built automatically when omitted).
    runner : callable, optional
        Injection hook for testing; called as
        ``runner(config, k, epsilon, repeat) -> {"f1","f2","initial_f1","initial_f2"}``.
    overrides : kwargs
        Forwarded to :meth:`Table1Config.with_overrides`.

    Returns
    -------
    dict with keys ``table``, ``config``, ``cells``, ``records``, ``checks``,
    ``artifacts``.
    """
    log = logger or LOGGER
    config = (config or Table1Config.paper()).with_overrides(**overrides)
    log.info(
        "[Table1] n=%d ks=%s epsilons=%s repeats=%d T=%d",
        config.n,
        list(config.ks),
        list(config.epsilons),
        config.repeats,
        config.T,
    )

    records: List[Dict[str, Any]] = []

    if runner is not None:
        for k in config.ks:
            for eps in config.epsilons:
                for rep in range(int(config.repeats)):
                    rec = dict(runner(config, int(k), float(eps), rep))
                    rec.setdefault("k", int(k))
                    rec.setdefault("epsilon", float(eps))
                    rec.setdefault("repeat", rep)
                    records.append(rec)
    else:
        if dataset is None or eval_dataset is None or model_factory is None:
            dataset, eval_dataset, model_factory = _build_dataset_and_model(config, log)
        objective = build_table1_objective(
            config,
            dataset=dataset,
            eval_dataset=eval_dataset,
            model_factory=model_factory,
            logger=log,
        )
        for k in config.ks:
            for eps in config.epsilons:
                records.extend(
                    run_single_configuration(
                        config,
                        k=int(k),
                        epsilon=float(eps),
                        dataset=dataset,
                        eval_dataset=eval_dataset,
                        model_factory=model_factory,
                        objective=objective,
                        logger=log,
                    )
                )

    cells = aggregate_results(records, config)
    checks = _direction_checks(cells, config)

    artifacts: Dict[str, str] = {}
    if config.save_artifacts and config.output_dir:
        try:
            artifacts = save_results(cells, records, checks, config)
        except Exception as exc:  # pragma: no cover
            log.warning("failed to save Table 1 artifacts: %s", exc)

    return {
        "table": "Table 1 (Section 5.1 preliminary superiority)",
        "config": config.to_dict(),
        "cells": {key: cell.to_dict() for key, cell in cells.items()},
        "records": records,
        "checks": checks,
        "artifacts": artifacts,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def format_table1(cells: Dict[str, Table1Cell]) -> str:
    """Render the aggregated cells in the layout of the paper's Table 1."""
    lines = [
        "Table 1: Results (mean +- std.) to illustrate the utility of our method "
        "in optimizing the objectives f1(m) and f2(m).",
        "",
        f"{'k':>5} | {'Objectives':<10} | {'Initial':>8} | "
        f"{'eps=0.2':>14} | {'eps=0.3':>14} | {'eps=0.4':>14}",
        "-" * 90,
    ]
    ks = sorted({cell.k for cell in cells.values()})
    epsilons = sorted({cell.epsilon for cell in cells.values()})
    for k in ks:
        row_cells = {c.epsilon: c for c in cells.values() if c.k == k}
        init_f1 = next((c.initial_f1_mean for c in row_cells.values()), float("nan"))
        init_f2 = next((c.initial_f2_mean for c in row_cells.values()), float(k))
        paper_f1 = PAPER_INITIAL_F1.get(k)

        def cell_str(eps: float, which: str) -> str:
            c = row_cells.get(round(eps, 2))
            if c is None:
                return "-"
            if which == "f1":
                return f"{c.f1_mean:.2f} +- {c.f1_std:.2f}"
            return f"{c.f2_mean:.1f} +- {c.f2_std:.1f}"

        lines.append(
            f"{k:>5} | {'f1(m)':<10} | {init_f1:>8.2f} | "
            + " | ".join(f"{cell_str(e, 'f1'):>14}" for e in epsilons)
        )
        lines.append(
            f"{'':>5} | {'f2(m)':<10} | {init_f2:>8.1f} | "
            + " | ".join(f"{cell_str(e, 'f2'):>14}" for e in epsilons)
        )
        lines.append(
            f"{'':>5} | {'(paper f1)':<10} | {paper_f1 if paper_f1 is not None else float('nan'):>8.2f} | "
            + " | ".join(
                f"{(lambda v: v if v is not None else float('nan'))(_paper_reference(k, e, 'f1')):>14.2f}"
                for e in epsilons
            )
        )
        lines.append(
            f"{'':>5} | {'(paper f2)':<10} | {float(k):>8.1f} | "
            + " | ".join(
                f"{(lambda v: v if v is not None else float('nan'))(_paper_reference(k, e, 'f2')):>14.1f}"
                for e in epsilons
            )
        )
        lines.append("-" * 90)
    return "\n".join(lines)


def save_results(
    cells: Dict[str, Table1Cell],
    records: Sequence[Dict[str, Any]],
    checks: Dict[str, Any],
    config: Table1Config,
) -> Dict[str, str]:
    """Write table1.json / table1.csv / table1.txt / table1_checks.json / raw jsonl."""
    os.makedirs(config.output_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}

    json_path = os.path.join(config.output_dir, "table1.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": config.to_dict(),
                "cells": {k: c.to_dict() for k, c in cells.items()},
                "checks": checks,
                "paper_table1": {f"k={k},eps={e:g}": v for (k, e), v in PAPER_TABLE1.items()},
            },
            fh,
            indent=2,
            default=str,
        )
    artifacts["json"] = json_path

    csv_path = os.path.join(config.output_dir, "table1.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "k",
                "epsilon",
                "initial_f1_mean",
                "initial_f1_std",
                "initial_f2_mean",
                "f1_mean",
                "f1_std",
                "f2_mean",
                "f2_std",
                "paper_f1",
                "paper_f2",
                "repeats",
                "failures",
            ]
        )
        for key in sorted(cells.keys(), key=lambda s: (cells[s].k, cells[s].epsilon)):
            c = cells[key]
            writer.writerow(
                [
                    c.k,
                    c.epsilon,
                    f"{c.initial_f1_mean:.4f}",
                    f"{c.initial_f1_std:.4f}",
                    f"{c.initial_f2_mean:.2f}",
                    f"{c.f1_mean:.4f}",
                    f"{c.f1_std:.4f}",
                    f"{c.f2_mean:.2f}",
                    f"{c.f2_std:.2f}",
                    _paper_reference(c.k, c.epsilon, "f1"),
                    _paper_reference(c.k, c.epsilon, "f2"),
                    c.repeats,
                    c.failures,
                ]
            )
    artifacts["csv"] = csv_path

    txt_path = os.path.join(config.output_dir, "table1.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(format_table1(cells))
        fh.write("\n\nDirection checks (Section 5.1 claims):\n")
        for key in (
            "claim1_all_lower",
            "claim2_f2_decreases_with_epsilon",
            "claim3_f1_increases_with_epsilon_on_average",
        ):
            fh.write(f"  {key}: {checks.get(key)}\n")
    artifacts["txt"] = txt_path

    checks_path = os.path.join(config.output_dir, "table1_checks.json")
    with open(checks_path, "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, default=str)
    artifacts["checks"] = checks_path

    raw_path = os.path.join(config.output_dir, "table1_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    artifacts["raw"] = raw_path

    return artifacts


# --------------------------------------------------------------------------
# Offline self-test (no torch, no dataset needed)
# --------------------------------------------------------------------------


class _SyntheticRunner:
    """Deterministic stand-in for Algorithm 1 used by the offline self-test."""

    def __call__(self, config: Table1Config, k: int, epsilon: float, repeat: int) -> Dict[str, Any]:
        rng = np.random.default_rng(1000 * int(k) + int(round(100 * epsilon)) + repeat)
        # mimic the paper's trends: larger eps -> smaller f2, larger f1 on average
        f2 = k * (1.0 - 0.6 * (epsilon / 0.4)) + rng.normal(0.0, 2.0)
        f1 = 1.0 + 2.0 * (epsilon / 0.4) + rng.normal(0.0, 0.1)
        initial_f1 = PAPER_INITIAL_F1.get(int(k), 3.21)
        return {
            "k": int(k),
            "epsilon": float(epsilon),
            "repeat": int(repeat),
            "f1": float(f1),
            "f2": float(f2),
            "initial_f1": float(initial_f1),
            "initial_f2": float(k),
        }


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: mask projection, aggregation, mean/std, formatting, claims."""
    report: Dict[str, Any] = {}

    # mask helpers (Appendix A projection: [-1,0)->0, [0,1]->1)
    relaxed = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    binary = binarize_mask(relaxed)
    assert binary.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0], binary
    assert mask_indices(binary).tolist() == [2, 3, 4]
    assert binarize_mask(np.array([1.0, 0.0, 1.0])).tolist() == [1.0, 0.0, 1.0]
    report["mask_helpers_ok"] = True

    # synthetic sweep -> aggregation, mean/std, determinism
    cfg = Table1Config.smoke(repeats=4)
    runner = _SyntheticRunner()
    records: List[Dict[str, Any]] = [
        runner(cfg, k, eps, rep)
        for k in cfg.ks
        for eps in cfg.epsilons
        for rep in range(cfg.repeats)
    ]
    cells = aggregate_results(records, cfg)
    assert cells, "no cells produced"
    cell = next(iter(cells.values()))
    assert cell.repeats == cfg.repeats, cell.repeats
    assert cell.f1_std >= 0.0 and cell.f2_std >= 0.0
    report["aggregation_ok"] = True

    f1s = [r["f1"] for r in records if r["k"] == cell.k and abs(r["epsilon"] - cell.epsilon) < 1e-9]
    assert abs(cell.f1_mean - float(np.mean(f1s))) < 1e-9
    assert abs(cell.f1_std - float(np.std(f1s, ddof=0))) < 1e-9
    report["mean_std_ok"] = True

    table = format_table1(cells)
    assert "f1(m)" in table and "f2(m)" in table and "Initial" in table
    report["format_ok"] = True

    # full synthetic sweep: claims 1 and 2 should hold by construction
    cfg2 = Table1Config.paper(repeats=3)
    records2 = [
        runner(cfg2, k, eps, rep) for k in cfg2.ks for eps in cfg2.epsilons for rep in range(3)
    ]
    cells2 = aggregate_results(records2, cfg2)
    checks = _direction_checks(cells2, cfg2)
    assert checks["claim1_all_lower"], checks["achieved_lower_than_initial"]
    assert checks["claim2_f2_decreases_with_epsilon"], checks["f2_nonincreasing_in_epsilon"]
    report["direction_checks_ok"] = True

    # end-to-end driver with the injected runner (no torch / no data)
    res = run_table1(config=cfg2, runner=runner, logger=None)
    assert "cells" in res and "checks" in res
    assert len(res["cells"]) == len(cfg2.ks) * len(cfg2.epsilons)
    report["run_table1_ok"] = True
    report["num_cells"] = len(res["cells"])

    if verbose:
        print(table)
        print()
        print("claim1_all_lower:", checks["claim1_all_lower"])
        print("claim2_f2_decreases_with_epsilon:", checks["claim2_f2_decreases_with_epsilon"])
        print(
            "claim3_f1_increases_with_epsilon_on_average:",
            checks["claim3_f1_increases_with_epsilon_on_average"],
        )
        print("selftest: OK")
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Table 1 (Section 5.1) driver for LBCS")
    parser.add_argument("--paper", action="store_true", help="run the exact Section 5.1 protocol")
    parser.add_argument("--smoke", action="store_true", help="tiny CPU smoke run")
    parser.add_argument("--selftest", action="store_true", help="run the offline self-test")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--T", type=int, default=None, help="outer iterations (SUGGESTED default 500)")
    parser.add_argument("--ks", type=int, nargs="+", default=None)
    parser.add_argument("--epsilons", type=float, nargs="+", default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, help="path to a YAML config (optional)")
    return parser


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict) and isinstance(data.get("table1"), dict):
            return dict(data["table1"])
        if isinstance(data, dict) and isinstance(data.get("section5_1"), dict):
            return dict(data["section5_1"])
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("could not load config %s: %s", path, exc)
        return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_argparser().parse_args(argv)

    if args.selftest:
        _selftest(verbose=True)
        return 0

    base = Table1Config.from_dict(_load_yaml(args.config)) if args.config else Table1Config()
    if args.paper:
        base = Table1Config.paper()
    if args.smoke:
        base = Table1Config.smoke()

    overrides: Dict[str, Any] = {}
    if args.repeats is not None:
        overrides["repeats"] = args.repeats
    if args.T is not None:
        overrides["T"] = args.T
    if args.ks is not None:
        overrides["ks"] = tuple(args.ks)
    if args.epsilons is not None:
        overrides["epsilons"] = tuple(args.epsilons)
    if args.inner_epochs is not None:
        overrides["inner_epochs"] = args.inner_epochs
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    config = base.with_overrides(**overrides)

    if not _has_torch():
        LOGGER.error("PyTorch is required for the Table 1 experiment (use --selftest offline).")
        return 2

    result = run_table1(config=config)
    cells = {
        key: Table1Cell(
            **{k: v for k, v in cell.items() if k in Table1Cell.__dataclass_fields__}  # type: ignore[attr-defined]
        )
        for key, cell in result["cells"].items()
    }
    print(format_table1(cells))
    print()
    for key in (
        "claim1_all_lower",
        "claim2_f2_decreases_with_epsilon",
        "claim3_f1_increases_with_epsilon_on_average",
    ):
        print(f"{key}: {result['checks'].get(key)}")
    if result.get("artifacts"):
        print("artifacts:", result["artifacts"])
    return 0


# Aliases so the experiment registry / CLI can resolve this driver uniformly.
run = run_table1
table1_prelim = run_table1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
