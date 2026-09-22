"""Label-mapping study (Appendix D.1, Table 10) for SMM.

Reproduces the experiment of Appendix D.1 "Applying SMM with Different
:math:`f_{\\mathrm{out}}`":

    "As mentioned before, and as shown in Appendix A.1, input VR is agnostic of
     the output label mapping method.  Thus, our SMM can be applied to different
     output label methods other than Ilm.  Experimental results are presented in
     Table 10.

     Our method improves the performance of all output mapping methods.  In most
     cases, the worse the output mapping method is, the more pronounced the
     improvement of SMM will be."

Three non-parametric output mappings from Sec. 2.3 are compared, each with and
without SMM:

* ``Rlm`` -- random injective mapping, drawn once before training (Sec. 2.3);
* ``Flm`` -- frequent label mapping built from the frequency matrix
  (Appendix A.4);
* ``Ilm`` -- iterative label mapping, refreshed every epoch (Appendix A.4).

The paper reports that SMM improves *all* three mappings, with average gains of
roughly ``Rlm +8.94``, ``Flm +4.69`` and ``Ilm +5.68`` accuracy points (see the
plan / Table 10 reference numbers below).

Everything in this module is orchestration glue: the heavy lifting happens in
``engine/train_smm.py`` (Algorithm 1), ``methods/baselines.py`` (shared-mask
"without SMM" reference), ``label_mapping/*`` (the three mappings) and
``engine/metrics.py`` (aggregation / reporting).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Guarded project imports (module stays importable during partial builds)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from ..data.datasets import (
        DEFAULT_BATCH_SIZES,
        MAIN_DATASETS,
        build_dataloaders,
        num_classes,
    )
except Exception:  # pragma: no cover
    MAIN_DATASETS = (
        "cifar10",
        "cifar100",
        "svhn",
        "gtsrb",
        "flowers102",
        "dtd",
        "ucf101",
        "food101",
        "sun397",
        "eurosat",
        "oxfordpets",
    )
    DEFAULT_BATCH_SIZES = {
        "default": 256,
        "dtd": 64,
        "oxfordpets": 64,
    }
    build_dataloaders = None  # type: ignore[assignment]
    num_classes = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..engine.metrics import (
        aggregate_seeds,
        format_mean_std,
        mean_over_datasets,
    )
except Exception:  # pragma: no cover
    def aggregate_seeds(values, ddof: int = 1):
        import math

        vals = [float(v) for v in values]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        if len(vals) < 2:
            return mean, 0.0
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - ddof)
        return mean, math.sqrt(max(var, 0.0))

    def format_mean_std(mean: float, std: float, decimals: int = 2) -> str:
        return f"{mean:.{decimals}f} +- {std:.{decimals}f}"

    def mean_over_datasets(per_dataset, dataset_order=None):
        keys = list(dataset_order) if dataset_order else [
            k for k in per_dataset if k != "average"
        ]
        vals = []
        for k in keys:
            v = per_dataset.get(k)
            if v is None:
                continue
            if isinstance(v, dict):
                v = v.get("mean")
            if v is not None:
                vals.append(float(v))
        return sum(vals) / len(vals) if vals else 0.0

try:  # pragma: no cover - import guard
    from ..engine.seeds import SEEDS, resolve_seeds, set_seed
except Exception:  # pragma: no cover
    SEEDS = (0, 1, 2)

    def resolve_seeds(seeds=None, n_seeds=None):
        if seeds is None:
            return list(SEEDS)
        return list(seeds)

    def set_seed(seed, **kwargs):
        try:
            import random

            import torch

            random.seed(seed)
            torch.manual_seed(seed)
        except Exception:
            pass
        return seed

try:  # pragma: no cover - import guard
    from ..models.pretrained import build_classifier, input_size_for
except Exception:  # pragma: no cover
    build_classifier = None  # type: ignore[assignment]

    def input_size_for(backbone: str = "resnet18", imgsize: Optional[int] = None) -> int:
        if imgsize:
            return int(imgsize)
        return 384 if "vit" in str(backbone).lower() else 224

try:  # pragma: no cover - import guard
    from ..engine.train_smm import (
        MASK_LAYERS_BY_BACKBONE,
        SMMTrainConfig,
        train_smm,
    )
except Exception:  # pragma: no cover
    MASK_LAYERS_BY_BACKBONE = {
        "resnet18": 5,
        "resnet50": 5,
        "resnet101": 5,
        "vit_b32": 6,
        "vit_b_32": 6,
        "vit_large": 6,
    }
    SMMTrainConfig = None  # type: ignore[assignment]
    train_smm = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..engine.train_smm import build_label_mapping as _engine_build_label_mapping
except Exception:  # pragma: no cover
    _engine_build_label_mapping = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..label_mapping import build_label_mapping as _lm_build_label_mapping
except Exception:  # pragma: no cover
    _lm_build_label_mapping = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..methods.baselines import (
        BaselineTrainConfig,
        build_shared_mask_baseline,
        train_baseline,
    )
except Exception:  # pragma: no cover
    BaselineTrainConfig = None  # type: ignore[assignment]
    build_shared_mask_baseline = None  # type: ignore[assignment]
    train_baseline = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..modules.reprogram import build_smm_reprogram
except Exception:  # pragma: no cover
    build_smm_reprogram = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from ..data.dataset_stats import (
        TABLE10_LABEL_MAPPINGS,
        TABLE10_AVERAGES,
    )
except Exception:  # pragma: no cover
    TABLE10_LABEL_MAPPINGS = None  # type: ignore[assignment]
    TABLE10_AVERAGES = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: Output mappings studied in Appendix D.1 / Table 10 (Sec. 2.3, Appendix A.4).
LABEL_MAPPING_ORDER: Tuple[str, ...] = ("rlm", "flm", "ilm")

#: Alias used by the experiment registry / other modules.
LABEL_MAPPINGS: Tuple[str, ...] = LABEL_MAPPING_ORDER

DEFAULT_LABEL_MAPPING = "ilm"

MAPPING_DISPLAY: Dict[str, str] = {
    "rlm": "Rlm",
    "flm": "Flm",
    "ilm": "Ilm",
}

MAPPING_ALIASES: Dict[str, str] = {
    "rlm": "rlm",
    "random": "rlm",
    "random_label_mapping": "rlm",
    "flm": "flm",
    "frequent": "flm",
    "frequency": "flm",
    "frequent_label_mapping": "flm",
    "ilm": "ilm",
    "iterative": "ilm",
    "iterated": "ilm",
    "iterative_label_mapping": "ilm",
}

BACKBONES: Tuple[str, ...] = ("resnet18", "resnet50", "vit_b32")

#: Datasets used for the Table 10 study (Appendix C, Table 6 order).  The full
#: 11-task set can be requested explicitly; the paper's Table 10 covers the same
#: target tasks as Table 1.
STUDY_DATASETS: Tuple[str, ...] = tuple(MAIN_DATASETS)

#: "Without SMM" reference for each mapping is the shared-pattern baseline
#: (i.e. ``Full``), which is the setting of Table 1 / Table 2 baselines.
DEFAULT_REFERENCE_METHOD = "full"

TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": 200,
    "milestones": (100, 145),
    "alpha_delta": 0.01,
    "gamma_delta": 0.1,
    "optimizer": "sgd",
    "momentum": 0.9,
    "batch_size": 256,
    "patch_size": 8,
    "label_mapping": DEFAULT_LABEL_MAPPING,
}

SMALL_BATCH_DATASETS: Tuple[str, ...] = ("dtd", "oxfordpets")

# --------------------------------------------------------------------------- #
# Table 10 reference numbers
# --------------------------------------------------------------------------- #
#: Average accuracy *improvement* brought by SMM for each output mapping.
#: (Appendix D.1: "Our method improves the performance of all output mapping
#: methods"; larger gains for the weaker mappings.)
TABLE10_IMPROVEMENTS: Dict[str, float] = {
    "rlm": 8.94,
    "flm": 4.69,
    "ilm": 5.68,
}

#: Averages reported in Table 10 (mean over the target tasks) when known.
#: ``None`` entries mean the paper's value was not transcribed; only the
#: improvement deltas are used for comparison in that case.
TABLE10_REFERENCE_AVERAGES: Dict[str, Dict[str, Optional[float]]] = {
    "rlm": {"without_smm": None, "with_smm": None, "improvement": 8.94},
    "flm": {"without_smm": None, "with_smm": None, "improvement": 4.69},
    "ilm": {"without_smm": None, "with_smm": None, "improvement": 5.68},
}

if isinstance(TABLE10_AVERAGES, dict):
    for _k, _v in TABLE10_AVERAGES.items():
        _key = str(_k).lower()
        if _key in TABLE10_REFERENCE_AVERAGES:
            if isinstance(_v, dict):
                TABLE10_REFERENCE_AVERAGES[_key].update(
                    {kk: vv for kk, vv in _v.items() if vv is not None}
                )
            elif _v is not None:
                TABLE10_REFERENCE_AVERAGES[_key]["with_smm"] = float(_v)

#: Per-dataset reference values from Table 10, when available.
TABLE10_REFERENCE: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
if isinstance(TABLE10_LABEL_MAPPINGS, dict):
    for _mapping, _rows in TABLE10_LABEL_MAPPINGS.items():
        _key = str(_mapping).lower()
        if _key not in LABEL_MAPPING_ORDER:
            continue
        if isinstance(_rows, dict):
            TABLE10_REFERENCE[_key] = {
                str(k): dict(v) if isinstance(v, dict) else {"with_smm": v}
                for k, v in _rows.items()
            }

#: Module-level accumulator (mirrors ``ABLATION_RESULTS`` in run_ablations.py).
LABEL_MAPPING_RESULTS: Dict[str, Any] = {"table10": {}}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def canonical_mapping(name: str) -> str:
    """Normalise a label-mapping name/alias to ``"rlm" | "flm" | "ilm"``."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in MAPPING_ALIASES:
        return MAPPING_ALIASES[key]
    squashed = key.replace("_", "")
    for alias, target in MAPPING_ALIASES.items():
        if alias.replace("_", "") == squashed:
            return target
    raise ValueError(
        f"Unknown label mapping {name!r}; expected one of {LABEL_MAPPING_ORDER}"
    )


def canonical_backbone(name: Optional[str]) -> str:
    """Normalise a backbone name to the canonical registry key."""
    key = str(name or "resnet18").strip().lower().replace("-", "_")
    if key in ("vit_b32", "vitb32", "vit_b_32", "vit"):
        return "vit_b32"
    if key in ("resnet_18", "resnet18"):
        return "resnet18"
    if key in ("resnet_50", "resnet50"):
        return "resnet50"
    return key


def num_mask_layers_for(backbone: str) -> int:
    """Mask-generator depth (5 for ResNets, 6 for ViT-B/32) -- Sec. 3.2."""
    return int(MASK_LAYERS_BY_BACKBONE.get(canonical_backbone(backbone), 5))


def batch_size_for(dataset: str, backbone: Optional[str] = None) -> int:
    """Table 9 batch size: 256 everywhere except DTD / OxfordPets (64)."""
    key = str(dataset).lower()
    if key in SMALL_BATCH_DATASETS:
        return 64
    if isinstance(DEFAULT_BATCH_SIZES, dict):
        if key in DEFAULT_BATCH_SIZES:
            return int(DEFAULT_BATCH_SIZES[key])
        if "default" in DEFAULT_BATCH_SIZES:
            return int(DEFAULT_BATCH_SIZES["default"])
    return 256


def resolve_device(device: Optional[Any] = None) -> Any:
    """Resolve ``None`` to CUDA when available."""
    if device is not None:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def num_target_classes(dataset: str) -> Optional[int]:
    """Number of target classes for ``dataset`` (Table 6), if known."""
    if callable(num_classes):
        try:
            return int(num_classes(dataset))
        except Exception:
            return None
    return None


def _filter_kwargs(cls_or_fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys a dataclass / callable does not accept (tolerant wiring)."""
    if cls_or_fn is None:
        return dict(kwargs)
    if dataclasses.is_dataclass(cls_or_fn):
        allowed = {f.name for f in dataclasses.fields(cls_or_fn)}
        return {k: v for k, v in kwargs.items() if k in allowed}
    try:
        import inspect

        sig = inspect.signature(cls_or_fn)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _mapping_builder():
    """Return the available ``build_label_mapping`` implementation."""
    return _engine_build_label_mapping or _lm_build_label_mapping


# --------------------------------------------------------------------------- #
# Config construction
# --------------------------------------------------------------------------- #
def make_mapping_train_config(
    dataset: str,
    backbone: str = "resnet18",
    *,
    label_mapping: str = DEFAULT_LABEL_MAPPING,
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    **overrides: Any,
) -> Any:
    """Build an ``SMMTrainConfig`` for one (dataset, mapping) run.

    Schedule (Table 9, Appendix C): 200 epochs, milestones ``[100, 145]``,
    ``alpha_1 = 0.01`` / ``gamma = 0.1`` for ``delta``; ``alpha_2 = 0.01`` /
    ``gamma = 0.1`` for the 5-layer mask generator (ResNets) and
    ``alpha_2 = 0.001`` / ``gamma = 1`` for the 6-layer one (ViT-B/32).
    """
    backbone = canonical_backbone(backbone)
    mapping = canonical_mapping(label_mapping)
    kwargs: Dict[str, Any] = dict(
        epochs=int(TRAINING_DEFAULTS["epochs"]),
        milestones=tuple(TRAINING_DEFAULTS["milestones"]),
        alpha_delta=float(TRAINING_DEFAULTS["alpha_delta"]),
        gamma_delta=float(TRAINING_DEFAULTS["gamma_delta"]),
        optimizer=str(TRAINING_DEFAULTS["optimizer"]),
        momentum=float(TRAINING_DEFAULTS["momentum"]),
        batch_size=batch_size_for(dataset, backbone),
        test_batch_size=batch_size_for(dataset, backbone),
        backbone=backbone,
        input_size=imgsize,
        patch_size=int(patch_size),
        num_mask_layers=num_mask_layers_for(backbone),
        label_mapping=mapping,
        mapping_refresh_every=1 if mapping == "ilm" else 0,
        seed=int(seed),
        device=device,
    )
    kwargs.update(overrides)
    if SMMTrainConfig is None:
        return kwargs
    filtered = _filter_kwargs(SMMTrainConfig, kwargs)
    return SMMTrainConfig(**filtered)  # type: ignore[misc]


def make_mapping_baseline_config(
    name: str,
    dataset: str,
    backbone: str = "resnet18",
    *,
    label_mapping: str = DEFAULT_LABEL_MAPPING,
    seed: int = 0,
    device: Optional[Any] = None,
    imgsize: Optional[int] = None,
    **overrides: Any,
) -> Any:
    """Build a ``BaselineTrainConfig`` for the "without SMM" reference run.

    The schedule is identical to the SMM run (200 epochs, LR 0.01, gamma 0.1,
    milestones 100/145) for a fair comparison, per Sec. 5 "Baselines".
    """
    backbone = canonical_backbone(backbone)
    mapping = canonical_mapping(label_mapping)
    kwargs: Dict[str, Any] = dict(
        name=str(name),
        backbone=backbone,
        input_size=imgsize if imgsize else input_size_for(backbone),
        epochs=int(TRAINING_DEFAULTS["epochs"]),
        milestones=tuple(TRAINING_DEFAULTS["milestones"]),
        lr=0.01,
        gamma=0.1,
        optimizer=str(TRAINING_DEFAULTS["optimizer"]),
        batch_size=batch_size_for(dataset, backbone),
        label_mapping=mapping,
        seed=int(seed),
        device=device,
    )
    kwargs.update(overrides)
    if BaselineTrainConfig is None:
        return kwargs
    filtered = _filter_kwargs(BaselineTrainConfig, kwargs)
    return BaselineTrainConfig(**filtered)  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Single runs
# --------------------------------------------------------------------------- #
def run_single_mapping(
    dataset: str,
    mapping: str = DEFAULT_LABEL_MAPPING,
    backbone: str = "resnet18",
    *,
    use_smm: bool = True,
    seed: int = 0,
    device: Optional[Any] = None,
    data_root: Optional[str] = None,
    root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    imgsize: Optional[int] = None,
    patch_size: int = 8,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    logger: Optional[Callable[[str], None]] = None,
    classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train one (dataset, mapping, with/without-SMM) combination for one seed.

    Returns a dict with the final test accuracy (%, final epoch, matching
    Algorithm 1's reported metric), the per-epoch history and the run metadata.
    """
    backbone = canonical_backbone(backbone)
    mapping = canonical_mapping(mapping)
    device = resolve_device(device)
    log = logger or (print if verbose else (lambda *_: None))
    imgsize = imgsize if imgsize else input_size_for(backbone)

    if build_dataloaders is None:
        raise RuntimeError("data.datasets.build_dataloaders is unavailable")

    set_seed(int(seed))
    train_loader, test_loader, spec = build_dataloaders(
        dataset,
        backbone=backbone,
        root=root,
        data_root=data_root,
        imgsize=imgsize,
        batch_size=batch_size_for(dataset, backbone),
        num_workers=num_workers,
        download=download,
        train_fraction=train_fraction,
        split_seed=split_seed,
        device=device,
    )
    n_classes = getattr(spec, "num_classes", None) or num_target_classes(dataset)

    if classifier is None:
        if build_classifier is None:
            raise RuntimeError("models.pretrained.build_classifier is unavailable")
        classifier = build_classifier(backbone=backbone, input_size=imgsize, device=device)

    overrides = dict(config_overrides or {})
    method = "smm" if use_smm else DEFAULT_REFERENCE_METHOD

    history = None
    if use_smm:
        if train_smm is None:
            raise RuntimeError("engine.train_smm.train_smm is unavailable")
        config = make_mapping_train_config(
            dataset,
            backbone,
            label_mapping=mapping,
            seed=seed,
            device=device,
            imgsize=imgsize,
            patch_size=patch_size,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            save_dir=save_dir,
            verbose=verbose,
            **overrides,
        )
        f_out = None
        builder = _mapping_builder()
        if builder is not None:
            try:
                f_out = builder(
                    mapping,
                    classifier=classifier,
                    data_loader=train_loader,
                    num_target_classes=n_classes,
                    device=device,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log(f"[label_mappings] mapping builder failed ({exc}); using default")
                f_out = None
        history = train_smm(
            model=None,
            classifier=classifier,
            train_loader=train_loader,
            test_loader=test_loader,
            f_out=f_out,
            config=config,
            dataset=dataset,
            device=device,
        )
    else:
        if train_baseline is None or build_shared_mask_baseline is None:
            raise RuntimeError("methods.baselines is unavailable")
        config = make_mapping_baseline_config(
            DEFAULT_REFERENCE_METHOD,
            dataset,
            backbone,
            label_mapping=mapping,
            seed=seed,
            device=device,
            imgsize=imgsize,
            max_train_batches=max_train_batches,
            max_eval_batches=max_eval_batches,
            save_dir=save_dir,
            verbose=verbose,
            **overrides,
        )
        model = build_shared_mask_baseline(
            DEFAULT_REFERENCE_METHOD, backbone=backbone, input_size=imgsize
        )
        f_out = None
        builder = _mapping_builder()
        if builder is not None:
            try:
                f_out = builder(
                    mapping,
                    classifier=classifier,
                    data_loader=train_loader,
                    num_target_classes=n_classes,
                    device=device,
                    seed=seed,
                )
            except Exception as exc:  # pragma: no cover - defensive
                log(f"[label_mappings] mapping builder failed ({exc}); using default")
                f_out = None
        history = train_baseline(
            model,
            classifier,
            train_loader,
            test_loader=test_loader,
            f_out=f_out,
            num_classes=n_classes,
            config=config,
            dataset=dataset,
            device=device,
        )

    accuracy = _final_accuracy(history)
    log(
        f"[label_mappings] {dataset} {backbone} {MAPPING_DISPLAY[mapping]} "
        f"{'SMM' if use_smm else 'no-SMM'} seed={seed}: {accuracy:.2f}%"
    )
    return {
        "dataset": dataset,
        "backbone": backbone,
        "label_mapping": mapping,
        "use_smm": bool(use_smm),
        "method": method,
        "seed": int(seed),
        "accuracy": float(accuracy),
        "num_classes": n_classes,
        "history": history,
        "config": config,
    }


def _final_accuracy(history: Any) -> float:
    """Extract the reported (final-epoch) test accuracy from a history object."""
    if history is None:
        return float("nan")
    for attr in ("final_test_accuracy", "best_test_accuracy", "accuracy"):
        val = getattr(history, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    accs = getattr(history, "test_accuracies", None)
    if accs:
        return float(accs[-1])
    if isinstance(history, dict):
        for key in ("final_test_accuracy", "accuracy", "test_accuracy"):
            if history.get(key) is not None:
                return float(history[key])
    return float("nan")


def run_mapping_pair(
    dataset: str,
    mapping: str = DEFAULT_LABEL_MAPPING,
    backbone: str = "resnet18",
    *,
    seeds: Optional[Sequence[int]] = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run ``mapping`` on ``dataset`` both without and with SMM, over seeds.

    Mirrors Appendix D.1 / Table 10: for every output mapping we compare the
    shared-pattern result with the SMM result and report the improvement.
    """
    mapping = canonical_mapping(mapping)
    backbone = canonical_backbone(backbone)
    seed_list = resolve_seeds(seeds)

    per_seed_without: List[float] = []
    per_seed_with: List[float] = []
    histories: Dict[str, List[Any]] = {"without_smm": [], "with_smm": []}

    for seed in seed_list:
        res_wo = run_single_mapping(
            dataset, mapping, backbone, use_smm=False, seed=int(seed), device=device, **kwargs
        )
        res_w = run_single_mapping(
            dataset, mapping, backbone, use_smm=True, seed=int(seed), device=device, **kwargs
        )
        per_seed_without.append(res_wo["accuracy"])
        per_seed_with.append(res_w["accuracy"])
        histories["without_smm"].append(res_wo["history"])
        histories["with_smm"].append(res_w["history"])

    mean_wo, std_wo = aggregate_seeds(per_seed_without)
    mean_w, std_w = aggregate_seeds(per_seed_with)

    return {
        "dataset": dataset,
        "backbone": backbone,
        "label_mapping": mapping,
        "seeds": [int(s) for s in seed_list],
        "without_smm": {
            "per_seed_accuracy": per_seed_without,
            "mean": mean_wo,
            "std": std_wo,
            "formatted": format_mean_std(mean_wo, std_wo),
        },
        "with_smm": {
            "per_seed_accuracy": per_seed_with,
            "mean": mean_w,
            "std": std_w,
            "formatted": format_mean_std(mean_w, std_w),
        },
        "improvement": float(mean_w - mean_wo),
        "histories": histories,
    }


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def format_label_mapping_table(
    results: Dict[str, Any],
    *,
    dataset_order: Optional[Sequence[str]] = None,
    backbone: Optional[str] = None,
    decimals: int = 1,
) -> str:
    """Render a Table-10-style text table for one backbone.

    Rows are ``<mapping> (w/o SMM)`` and ``<mapping> (SMM)`` pairs; the last
    column of each pair is the average over the target tasks.
    """
    order = list(dataset_order) if dataset_order else list(STUDY_DATASETS)
    mapping_names = [m for m in LABEL_MAPPING_ORDER if m in results]
    header_cells = [MAPPING_DISPLAY[m] for m in mapping_names]
    header_cells.append("AVERAGE")
    width = max(12, max(len(c) for c in header_cells) + 2)
    lines: List[str] = []
    lines.append(" " * width + "  " + "  ".join(f"{c:>{width}}" for c in header_cells))
    lines.append(" " * width + "  " + "  ".join(f"{'-' * 6:>{width}}" for _ in header_cells))

    for label, key in (("w/o SMM", "without_smm"), ("w/ SMM", "with_smm")):
        cells: List[str] = []
        for mapping in mapping_names:
            mean = _mean_for_backbone(results.get(mapping, {}), backbone, key)
            cells.append(f"{mean:>{width}.{decimals}f}")
        avg = _average_over_datasets(results, mapping_names, backbone, key, order)
        cells.append(f"{avg:>{width}.{decimals}f}")
        lines.append(f"{label:<{width}}" + "  " + "  ".join(cells))

    return "\n".join(lines)


def _mean_for_backbone(mapping_results: Dict[str, Any], backbone: Optional[str], key: str) -> float:
    """Overall mean of ``key`` for one mapping, restricted to a backbone."""
    per_dataset: Dict[str, float] = {}
    for ds, entry in mapping_results.items():
        if not isinstance(entry, dict):
            continue
        if backbone and entry.get("backbone") not in (None, backbone):
            continue
        block = entry.get(key)
        if isinstance(block, dict) and block.get("mean") is not None:
            per_dataset[ds] = float(block["mean"])
    if not per_dataset:
        return float("nan")
    return sum(per_dataset.values()) / len(per_dataset)


def _average_over_datasets(
    results: Dict[str, Any],
    mapping_names: Sequence[str],
    backbone: Optional[str],
    key: str,
    dataset_order: Sequence[str],
) -> float:
    """Average across mappings and datasets (Table 10 AVERAGE column)."""
    per_mapping = [
        _mean_for_backbone(results.get(m, {}), backbone, key) for m in mapping_names
    ]
    per_mapping = [v for v in per_mapping if v == v]  # drop NaNs
    if not per_mapping:
        return float("nan")
    return sum(per_mapping) / len(per_mapping)


def compare_table10(
    results: Dict[str, Any],
    *,
    backbone: Optional[str] = None,
    tolerance: float = 2.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare measured improvements with Table 10's reported gains.

    The claim checked is Appendix D.1's: *SMM improves every output mapping*,
    with the improvement magnitude roughly matching the paper's
    ``Rlm +8.94 / Flm +4.69 / Ilm +5.68``.
    """
    comparison: Dict[str, Any] = {"per_mapping": {}, "all_improved": True, "within_tolerance": {}}
    for mapping in LABEL_MAPPING_ORDER:
        if mapping not in results:
            continue
        mean_wo = _mean_for_backbone(results[mapping], backbone, "without_smm")
        mean_w = _mean_for_backbone(results[mapping], backbone, "with_smm")
        improvement = mean_w - mean_wo if (mean_wo == mean_wo and mean_w == mean_w) else float("nan")
        reference = TABLE10_IMPROVEMENTS.get(mapping)
        improved = bool(improvement == improvement and improvement > 0.0)
        within = (
            bool(improvement == improvement and reference is not None
                 and abs(improvement - reference) <= tolerance)
        )
        comparison["per_mapping"][mapping] = {
            "without_smm": mean_wo,
            "with_smm": mean_w,
            "improvement": improvement,
            "reference_improvement": reference,
            "improved": improved,
            "within_tolerance": within,
        }
        comparison["within_tolerance"][mapping] = within
        comparison["all_improved"] = comparison["all_improved"] and improved
        if verbose:
            print(
                f"[label_mappings] {MAPPING_DISPLAY[mapping]}: "
                f"{mean_wo:.2f} -> {mean_w:.2f} "
                f"(Delta {improvement:+.2f}, paper {reference:+.2f})"
            )
    return comparison


def format_comparison_report(comparison: Dict[str, Any]) -> str:
    """Human-readable summary of :func:`compare_table10`."""
    lines = ["Label-mapping study vs. Appendix D.1 / Table 10:"]
    for mapping, row in comparison.get("per_mapping", {}).items():
        ref = row.get("reference_improvement")
        ref_txt = f"{ref:+.2f}" if ref is not None else "n/a"
        lines.append(
            f"  {MAPPING_DISPLAY.get(mapping, mapping):<6}"
            f"w/o SMM {row['without_smm']:6.2f}  "
            f"w/ SMM {row['with_smm']:6.2f}  "
            f"Delta {row['improvement']:+.2f} (paper {ref_txt})"
        )
    lines.append(f"  SMM improves all mappings: {comparison.get('all_improved')}")
    return "\n".join(lines)


def save_label_mapping_results(
    payload: Dict[str, Any],
    output_dir: Optional[str] = None,
    *,
    filename: str = "label_mappings.json",
) -> str:
    """Persist the study results (without history objects) to JSON."""
    output_dir = output_dir or os.environ.get("SMM_OUTPUT_DIR", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_strip_histories(payload), handle, indent=2, default=str)
    return path


def _strip_histories(obj: Any) -> Any:
    """Recursively drop non-serialisable history/config objects."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in ("history", "histories", "config"):
                continue
            out[key] = _strip_histories(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_strip_histories(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Experiment entry points
# --------------------------------------------------------------------------- #
def run_label_mapping_experiment(
    datasets: Optional[Sequence[str]] = None,
    backbones: Optional[Sequence[str]] = None,
    mappings: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    *,
    backbone: Optional[str] = None,
    output_dir: Optional[str] = None,
    save: bool = True,
    verbose: bool = True,
    data_root: Optional[str] = None,
    root: Optional[str] = None,
    num_workers: int = 4,
    download: bool = True,
    device: Optional[Any] = None,
    patch_size: int = 8,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
    config_overrides: Optional[Dict[str, Any]] = None,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    logger: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Reproduce Appendix D.1 / Table 10 (label-mapping study).

    For each backbone, dataset and mapping in ``{Rlm, Flm, Ilm}`` the shared
    pattern ("without SMM") result is compared against the SMM result over the
    paper's three seeds.
    """
    dataset_list = list(datasets) if datasets else list(STUDY_DATASETS)
    backbone_list = [canonical_backbone(b) for b in (backbones or ([backbone] if backbone else ["resnet18"]))]
    mapping_list = [canonical_mapping(m) for m in (mappings or LABEL_MAPPING_ORDER)]
    seed_list = resolve_seeds(seeds)
    device = resolve_device(device)
    log = logger or (print if verbose else (lambda *_: None))

    results: Dict[str, Any] = {}
    for bb in backbone_list:
        results[bb] = {}
        for mapping in mapping_list:
            results[bb][mapping] = {}
            log(f"[label_mappings] === {bb} / {MAPPING_DISPLAY[mapping]} ===")
            for dataset in dataset_list:
                entry = run_mapping_pair(
                    dataset,
                    mapping,
                    bb,
                    seeds=seed_list,
                    device=device,
                    data_root=data_root,
                    root=root,
                    num_workers=num_workers,
                    download=download,
                    patch_size=patch_size,
                    train_fraction=train_fraction,
                    split_seed=split_seed,
                    config_overrides=config_overrides,
                    max_train_batches=max_train_batches,
                    max_eval_batches=max_eval_batches,
                    verbose=verbose,
                    logger=logger,
                    **kwargs,
                )
                results[bb][mapping][dataset] = entry
                log(
                    f"[label_mappings] {dataset}: {entry['without_smm']['formatted']} -> "
                    f"{entry['with_smm']['formatted']} (Delta {entry['improvement']:+.2f})"
                )

    payload: Dict[str, Any] = {
        "experiment": "label_mappings",
        "table": "table10",
        "datasets": dataset_list,
        "backbones": backbone_list,
        "mappings": mapping_list,
        "seeds": [int(s) for s in seed_list],
        "results": results,
        "comparison": {},
        "tables": {},
    }
    for bb in backbone_list:
        payload["comparison"][bb] = compare_table10(results[bb], backbone=bb, verbose=verbose)
        payload["tables"][bb] = format_label_mapping_table(results[bb], backbone=bb)
        if verbose:
            print(f"\n=== Table 10 ({bb}) ===\n{payload['tables'][bb]}")
            print(format_comparison_report(payload["comparison"][bb]))

    LABEL_MAPPING_RESULTS["table10"] = payload
    if save:
        payload["path"] = save_label_mapping_results(payload, output_dir)
        if verbose:
            print(f"[label_mappings] saved to {payload['path']}")
    return payload


#: Registry-compatible alias (``experiments/__init__.py`` expects
#: ``run_label_mappings_experiment``).
def run_label_mappings_experiment(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Alias of :func:`run_label_mapping_experiment`."""
    return run_label_mapping_experiment(*args, **kwargs)


def run_single_mapping_dataset(
    dataset: str,
    mapping: str = DEFAULT_LABEL_MAPPING,
    backbone: str = "resnet18",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Convenience wrapper: one dataset, one mapping, no/with SMM over seeds."""
    return run_mapping_pair(dataset, mapping, backbone, **kwargs)


def describe_label_mapping_study(
    datasets: Optional[Sequence[str]] = None,
    backbones: Optional[Sequence[str]] = None,
    mappings: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Describe the planned label-mapping study (for ``--describe``/logging)."""
    return {
        "experiment": "label_mappings",
        "source": "Appendix D.1, Table 10",
        "datasets": [str(d) for d in (datasets or STUDY_DATASETS)],
        "backbones": [canonical_backbone(b) for b in (backbones or ["resnet18"])],
        "mappings": [canonical_mapping(m) for m in (mappings or LABEL_MAPPING_ORDER)],
        "seeds": [int(s) for s in resolve_seeds(seeds)],
        "reference_improvements": dict(TABLE10_IMPROVEMENTS),
        "schedule": dict(TRAINING_DEFAULTS),
        "note": (
            "SMM improves every output mapping; weaker mappings gain more "
            "(Appendix D.1)."
        ),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMM label-mapping study (Appendix D.1 / Table 10)."
    )
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--backbones", nargs="*", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--mappings", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the label-mapping study."""
    args = build_arg_parser().parse_args(argv)
    if args.describe:
        info = describe_label_mapping_study(
            args.datasets, args.backbones or ([args.backbone] if args.backbone else None),
            args.mappings, args.seeds,
        )
        print(json.dumps(info, indent=2))
        return 0
    run_label_mapping_experiment(
        datasets=args.datasets,
        backbones=args.backbones,
        backbone=args.backbone,
        mappings=args.mappings,
        seeds=args.seeds,
        output_dir=args.output_dir,
        save=not args.no_save,
        verbose=not args.quiet,
        data_root=args.data_root,
        num_workers=args.num_workers,
        download=not args.no_download,
        device=args.device,
        patch_size=args.patch_size,
        train_fraction=args.train_fraction,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
