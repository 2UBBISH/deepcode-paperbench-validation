"""SMM (Sample-specific Multi-channel Masks) — top-level entry point.

This module is the command-line / programmatic dispatcher for the SMM paper
(ICML 2024) reproduction.  It is *glue code*: it loads and merges the YAML
configs in ``smm_vr/configs`` (``base.yaml`` -> ``model_<backbone>.yaml`` ->
experiment config) and forwards the resolved settings to the experiment
runners in ``smm_vr.experiments``.

Examples
--------
>>> python -m smm_vr.main --list
>>> python -m smm_vr.main --experiment main --backbone resnet18 \
...     --datasets cifar10 cifar100 --seeds 0 1 2
>>> python -m smm_vr.main --experiment ablations --mode table3
>>> python -m smm_vr.main --experiment label_mappings --mappings ilm flm rlm
>>> python -m smm_vr.main --experiment scaling
>>> python -m smm_vr.main --experiment finetuning --methods lora finetune_fc
>>> python -m smm_vr.main --experiment stanfordcars
>>> python -m smm_vr.main --experiment dataset_stats --datasets cifar10 svhn

All heavy imports (torch, torchvision, per-experiment runners) are performed
lazily so that ``--list`` / ``--describe`` / ``--check-config`` work in a bare
environment (e.g. no GPU, no datasets downloaded).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_CONFIGS",
    "EXPERIMENT_CONFIGS",
    "load_config",
    "merge_configs",
    "deep_update",
    "resolve_config",
    "resolve_device_str",
    "discover_experiments",
    "dispatch_experiment",
    "run_dataset_stats",
    "main",
]

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(HERE, "configs")

#: YAML layers merged (in order) for every experiment.
DEFAULT_CONFIGS: Sequence[str] = ("base.yaml", "datasets.yaml")

#: Extra YAML layer per experiment name.
EXPERIMENT_CONFIGS: Dict[str, str] = {
    "main": "train_smm.yaml",
    "train": "train_smm.yaml",
    "smm": "train_smm.yaml",
    "ablations": "ablations.yaml",
    "ablation": "ablations.yaml",
    "label_mappings": "label_mappings.yaml",
    "label_mapping": "label_mappings.yaml",
    "scaling": "train_smm.yaml",
    "finetuning": "train_smm.yaml",
    "stanfordcars": "train_smm.yaml",
}

#: Canonical experiment names handled by this dispatcher.
EXPERIMENT_ALIASES: Dict[str, str] = {
    "main": "main",
    "table1": "main",
    "table2": "main",
    "tables12": "main",
    "train": "main",
    "smm": "main",
    "ablations": "ablations",
    "ablation": "ablations",
    "table3": "ablations",
    "figure4": "ablations",
    "patch_size": "ablations",
    "patch": "ablations",
    "label_mappings": "label_mappings",
    "label_mapping": "label_mappings",
    "labelmapping": "label_mappings",
    "table10": "label_mappings",
    "scaling": "scaling",
    "table11": "scaling",
    "scale": "scaling",
    "finetuning": "finetuning",
    "finetune": "finetuning",
    "lora": "finetuning",
    "table13": "finetuning",
    "table14": "finetuning",
    "stanfordcars": "stanfordcars",
    "cars": "stanfordcars",
    "table12": "stanfordcars",
    "failure": "stanfordcars",
    # local utilities (no training)
    "data": "dataset_stats",
    "dataset_stats": "dataset_stats",
    "stats": "dataset_stats",
    "datasets": "dataset_stats",
    "tsne": "tsne",
    "tsne_features": "tsne",
    "verify": "verify",
    "check": "verify",
}

MAIN_DATASETS_FALLBACK: Sequence[str] = (
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

BACKBONES: Sequence[str] = ("resnet18", "resnet50", "vit_b32")


# ---------------------------------------------------------------------------
# Config loading / merging
# ---------------------------------------------------------------------------

def _import_yaml():
    """Import PyYAML lazily with a helpful error message."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "PyYAML is required to read the SMM configs; install it with "
            "`pip install pyyaml`."
        ) from exc
    return yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a single YAML config file (returns ``{}`` if it does not exist)."""
    if not os.path.isabs(path):
        candidate = os.path.join(CONFIG_DIR, path)
    else:
        candidate = path
    if not os.path.isfile(candidate):
        return {}
    yaml = _import_yaml()
    with open(candidate, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config {candidate} must contain a YAML mapping")
    return payload


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in (override or {}).items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config dicts left-to-right (later files win)."""
    merged: Dict[str, Any] = {}
    for config in configs:
        deep_update(merged, config or {})
    return merged


def resolve_config(
    experiment: str = "main",
    *,
    backbone: Optional[str] = None,
    config_paths: Optional[Sequence[str]] = None,
    config_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load and merge the YAML layers for ``experiment``.

    Layer order: ``base.yaml`` -> ``datasets.yaml`` -> experiment YAML ->
    ``model_<backbone>.yaml`` (when a backbone is given).
    """
    global CONFIG_DIR
    previous_dir = CONFIG_DIR
    if config_dir:
        CONFIG_DIR = config_dir
    try:
        layers: List[Dict[str, Any]] = []
        if config_paths:
            for path in config_paths:
                layers.append(load_config(path))
        else:
            for name in DEFAULT_CONFIGS:
                layers.append(load_config(name))
            extra = EXPERIMENT_CONFIGS.get(canonical_experiment(experiment))
            if extra:
                layers.append(load_config(extra))
        config = merge_configs(*layers)
    finally:
        CONFIG_DIR = previous_dir

    if config_paths:
        # Explicit paths already encode everything the caller wants; still let
        # a backbone-specific layer be appended for convenience.
        pass

    config.setdefault("experiment", canonical_experiment(experiment))
    if backbone:
        model_layer = load_config(_model_config_name(backbone))
        deep_update(config, model_layer)
        _set_backbone(config, backbone)
    else:
        _set_backbone(config, config.get("backbone") or _config_backbone(config))
    return config


def _model_config_name(backbone: str) -> str:
    key = str(backbone).lower().replace("-", "_").replace("/", "_")
    if key in ("resnet18", "resnet_18", "r18"):
        return "model_resnet18.yaml"
    if key in ("resnet50", "resnet_50", "r50"):
        return "model_resnet50.yaml"
    if "vit" in key and "large" not in key:
        return "model_vit_b32.yaml"
    return "model_vit_b32.yaml" if "vit" in key else "model_resnet18.yaml"


def _config_backbone(config: Dict[str, Any]) -> str:
    model = config.get("model")
    if isinstance(model, dict) and model.get("backbone"):
        return str(model["backbone"])
    return str(config.get("backbone") or "resnet18")


def _set_backbone(config: Dict[str, Any], backbone: str) -> None:
    config["backbone"] = backbone
    model = config.setdefault("model", {})
    if isinstance(model, dict):
        model["backbone"] = backbone


# ---------------------------------------------------------------------------
# Experiment name normalisation
# ---------------------------------------------------------------------------

def canonical_experiment(name: str) -> str:
    """Normalise an experiment name/alias to its canonical key."""
    if not name:
        return "main"
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in EXPERIMENT_ALIASES:
        return EXPERIMENT_ALIASES[key]
    squashed = key.replace("_", "")
    for alias, target in EXPERIMENT_ALIASES.items():
        if alias.replace("_", "") == squashed:
            return target
    raise ValueError(
        f"Unknown experiment '{name}'. Known: {sorted(set(EXPERIMENT_ALIASES))}"
    )


def discover_experiments() -> List[str]:
    """Return the experiment names exposed by ``smm_vr.experiments``."""
    try:
        from smm_vr.experiments import list_experiments

        names = list(list_experiments())
    except Exception:
        names = ["main", "ablations", "label_mappings", "scaling",
                 "finetuning", "stanfordcars"]
    return names


def resolve_device_str(device: Optional[str]) -> str:
    """Resolve ``None``/``"auto"`` to ``"cuda"`` or ``"cpu"``."""
    if device in (None, "", "auto"):
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return str(device)


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs not accepted by ``func`` (tolerant to signature drift)."""
    import inspect

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # builtins / C callables
        return dict(kwargs)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _default_output_dir(config: Dict[str, Any]) -> str:
    output = config.get("output") or {}
    root = output.get("root") if isinstance(output, dict) else None
    results = output.get("results_dir") if isinstance(output, dict) else None
    if results:
        return str(results)
    return str(root or os.path.join(os.getcwd(), "outputs"))


def dispatch_experiment(name: str, **kwargs) -> Dict[str, Any]:
    """Resolve and invoke the experiment runner for ``name``."""
    canonical = canonical_experiment(name)

    if canonical == "dataset_stats":
        return run_dataset_stats(**kwargs)
    if canonical == "tsne":
        from smm_vr.analysis import tsne_features as _tsne

        entry = getattr(_tsne, "run_tsne_experiment", None) or getattr(
            _tsne, "main", None
        )
        if entry is None:
            raise ImportError("smm_vr.analysis.tsne_features has no entry point")
        return entry(**kwargs)
    if canonical == "verify":
        return verify_environment(**kwargs)

    from smm_vr import experiments as experiments_pkg

    runner = experiments_pkg.get_experiment(canonical)
    call_kwargs = _filter_kwargs(runner, kwargs)
    return runner(**call_kwargs)


def verify_environment(**kwargs) -> Dict[str, Any]:
    """Lightweight self-check of the mask generators / backbones / installs."""
    report: Dict[str, Any] = {"checks": {}, "ok": True}

    def record(key: str, value: Any, ok: bool = True) -> None:
        report["checks"][key] = value
        report["ok"] = report["ok"] and bool(ok)

    try:
        import torch

        record("torch", torch.__version__)
        record("cuda_available", bool(torch.cuda.is_available()))
    except Exception as exc:  # pragma: no cover - env dependent
        record("torch", f"ERROR: {exc}", ok=False)

    try:
        import torchvision

        record("torchvision", torchvision.__version__)
    except Exception as exc:  # pragma: no cover
        record("torchvision", f"ERROR: {exc}", ok=False)

    try:
        from smm_vr.models.mask_generator import EXPECTED_PARAMETERS, build_mask_generator

        counts = {}
        for backbone, expected in EXPECTED_PARAMETERS.items():
            try:
                generator = build_mask_generator(backbone)
                actual = sum(p.numel() for p in generator.parameters())
                counts[backbone] = {"parameters": actual, "expected": expected}
                if actual != expected:
                    record(f"mask_generator[{backbone}]", actual, ok=False)
            except Exception as exc:  # pragma: no cover
                record(f"mask_generator[{backbone}]", f"ERROR: {exc}", ok=False)
        record("mask_generator_parameters", counts)
    except Exception as exc:  # pragma: no cover
        record("mask_generator", f"ERROR: {exc}", ok=False)

    try:
        from smm_vr.data.dataset_stats import summarise_datasets

        record("datasets", [row.get("name") for row in summarise_datasets()])
    except Exception as exc:  # pragma: no cover
        record("datasets", f"ERROR: {exc}", ok=False)

    if kwargs.get("verbose", True):
        print(json.dumps(report, indent=2, default=str))
    return report


def run_dataset_stats(
    datasets: Optional[Sequence[str]] = None,
    *,
    data_root: Optional[str] = None,
    root: Optional[str] = None,
    backbone: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    download: bool = False,
    verbose: bool = True,
    json_path: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Report the paper's Table 6 metadata, optionally measured on disk."""
    from smm_vr.data.dataset_stats import (
        dump_statistics_json,
        format_statistics_table,
        summarise_datasets,
    )

    names = list(datasets) if datasets else list(MAIN_DATASETS_FALLBACK)
    rows = summarise_datasets(names, reference=True)

    if data_root is None and config:
        data = config.get("data") or {}
        if isinstance(data, dict):
            data_root = data.get("root")
            download = bool(data.get("download", download))
    if root and data_root is None:
        data_root = root

    measured = None
    build_datasets = None
    try:
        from smm_vr.data.datasets import build_datasets as build_datasets  # noqa: F811
    except Exception:
        build_datasets = None

    if build_datasets is not None:
        measured = []
        for name in names:
            try:
                train_ds, test_ds, spec = build_datasets(
                    name,
                    backbone=backbone,
                    root=root,
                    data_root=data_root,
                    download=download,
                )
                measured.append(
                    {
                        "name": name,
                        "train_size": len(train_ds),
                        "test_size": len(test_ds),
                        "num_classes": getattr(spec, "num_classes", None),
                    }
                )
            except Exception as exc:  # pragma: no cover - data not present
                measured.append({"name": name, "error": str(exc)})

    payload = {"reference": rows, "measured": measured}
    if verbose:
        print(format_statistics_table(rows))
        if measured:
            print("\nMeasured on disk:")
            for row in measured:
                print(" ", row)

    if json_path:
        dump_statistics_json(json_path, rows)
        if verbose:
            print(f"\nWrote reference statistics to {json_path}")

    return payload


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smm_vr",
        description=(
            "Sample-specific Multi-channel Mask (SMM) visual reprogramming — "
            "reproduction driver (ICML 2024)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        "--exp",
        default="main",
        help="Experiment to run (main, ablations, label_mappings, scaling, "
             "finetuning, stanfordcars, dataset_stats, tsne, verify).",
    )
    parser.add_argument(
        "--backbone", default=None, help="resnet18 | resnet50 | vit_b32"
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Target datasets (default: the 11 Table 6 tasks).")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Comparison methods for --experiment main.")
    parser.add_argument("--mappings", nargs="+", default=None,
                        help="Label mappings for --experiment label_mappings.")
    parser.add_argument("--variants", nargs="+", default=None,
                        help="Masking variants for --experiment ablations.")
    parser.add_argument("--l-values", nargs="+", type=int, default=None,
                        help="Patch-size exponents l for the patch study.")
    parser.add_argument("--mode", default=None,
                        help="Sub-mode for --experiment ablations "
                             "(all|masking|table3|patch|patch_size|figure4).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: 0 1 2).")
    parser.add_argument("--label-mapping", default=None,
                        help="Output label mapping: rlm | flm | ilm.")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="Patch size (default 8 => l = 3).")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda | cpu | auto")
    parser.add_argument("--data-root", default=None,
                        help="Dataset root (default: $SMM_DATA_ROOT or ./data).")
    parser.add_argument("--download", type=_str2bool, default=None,
                        help="Allow dataset download.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None,
                        help="Deterministic subsampling fraction (debugging).")
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config", nargs="+", default=None,
                        help="Explicit YAML config paths (override the defaults).")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write result JSON files.")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="List the available experiments and exit.")
    parser.add_argument("--describe", action="store_true",
                        help="Describe the selected experiment and exit.")
    parser.add_argument("--check-config", action="store_true",
                        help="Print the merged config for the selection and exit.")
    parser.add_argument("--verify", action="store_true",
                        help="Run the environment self-check and exit.")
    return parser


def _build_runner_kwargs(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """Turn parsed CLI args + merged config into runner kwargs."""
    data = config.get("data") or {}
    training = config.get("training") or {}
    output = config.get("output") or {}
    evaluation = config.get("evaluation") or {}

    kwargs: Dict[str, Any] = {}
    if args.datasets:
        kwargs["datasets"] = list(args.datasets)
    if args.methods:
        kwargs["methods"] = list(args.methods)
    if args.mappings:
        kwargs["mappings"] = list(args.mappings)
    if args.variants:
        kwargs["variants"] = list(args.variants)
    if args.l_values:
        kwargs["l_values"] = list(args.l_values)
    if args.mode:
        kwargs["mode"] = args.mode
    if args.seeds:
        kwargs["seeds"] = list(args.seeds)
    if args.backbone:
        kwargs["backbone"] = args.backbone
    if args.label_mapping:
        kwargs["label_mapping"] = args.label_mapping

    if args.patch_size is not None:
        kwargs["patch_size"] = args.patch_size

    overrides: Dict[str, Any] = {}
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.max_train_batches is not None:
        overrides["max_train_batches"] = args.max_train_batches
    if args.max_eval_batches is not None:
        overrides["max_eval_batches"] = args.max_eval_batches
    if overrides:
        kwargs["config_overrides"] = overrides

    if args.data_root is not None:
        kwargs["data_root"] = args.data_root
    elif isinstance(data, dict) and data.get("root"):
        kwargs["data_root"] = data["root"]

    if args.download is not None:
        kwargs["download"] = bool(args.download)
    elif isinstance(data, dict) and data.get("download") is not None:
        kwargs["download"] = bool(data["download"])

    if args.num_workers is not None:
        kwargs["num_workers"] = args.num_workers
    elif isinstance(data, dict) and data.get("num_workers") is not None:
        kwargs["num_workers"] = int(data["num_workers"])

    if args.train_fraction is not None:
        kwargs["train_fraction"] = args.train_fraction
    if args.split_seed is not None:
        kwargs["split_seed"] = args.split_seed
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    elif output:
        kwargs["output_dir"] = _default_output_dir(config)
    kwargs["save"] = not args.no_save
    kwargs["verbose"] = not args.quiet
    kwargs["device"] = resolve_device_str(args.device or config.get("device"))

    # Keep training/evaluation defaults from YAML visible to runners that
    # accept them (e.g. evaluation topk / t-SNE sample count).
    if isinstance(evaluation, dict) and "tsne_samples" in evaluation:
        kwargs.setdefault("tsne_samples", evaluation["tsne_samples"])
    if isinstance(training, dict) and "label_mapping" in training:
        kwargs.setdefault("label_mapping", training["label_mapping"])
    return kwargs


def describe_experiment(name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a human-readable description of an experiment selection."""
    canonical = canonical_experiment(name)
    description: Dict[str, Any] = {
        "experiment": canonical,
        "runner": EXPERIMENT_CONFIGS.get(canonical, "n/a"),
        "backbone": _config_backbone(config),
        "datasets": (config.get("datasets")
                     if isinstance(config.get("datasets"), list)
                     else list(MAIN_DATASETS_FALLBACK)),
        "seeds": config.get("seeds", [0, 1, 2]),
        "config_keys": sorted(config.keys()),
    }
    try:
        from smm_vr.experiments import describe_experiments

        description["runners"] = describe_experiments()
    except Exception:
        pass
    return description


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list:
        print("Experiments:")
        for name in discover_experiments():
            print(f"  - {name}")
        print("\nCanonical dispatch names:")
        for name in sorted(set(EXPERIMENT_ALIASES.values())):
            print(f"  - {name}")
        return 0

    if args.verify:
        report = verify_environment(verbose=not args.quiet)
        return 0 if report.get("ok", True) else 1

    try:
        canonical = canonical_experiment(args.experiment)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = resolve_config(
        canonical,
        backbone=args.backbone,
        config_paths=args.config,
        config_dir=args.config_dir,
    )

    if args.check_config:
        print(json.dumps(config, indent=2, default=str))
        return 0

    if args.describe:
        try:
            from smm_vr.experiments import describe_experiments

            if canonical in ("main",):
                pass
        except Exception:
            pass
        print(json.dumps(describe_experiment(canonical, config), indent=2, default=str))
        return 0

    if canonical == "dataset_stats":
        payload = run_dataset_stats(
            datasets=args.datasets,
            data_root=args.data_root,
            backbone=args.backbone,
            config=config,
            download=bool(args.download) if args.download is not None else False,
            verbose=not args.quiet,
        )
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, "dataset_stats.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, default=str)
            if not args.quiet:
                print(f"Wrote {path}")
        return 0

    kwargs = _build_runner_kwargs(args, config)
    if not args.quiet:
        print(
            f"[smm_vr] experiment={canonical} backbone={kwargs.get('backbone', _config_backbone(config))} "
            f"device={kwargs.get('device')} seeds={kwargs.get('seeds', config.get('seeds', [0, 1, 2]))}"
        )

    result = dispatch_experiment(canonical, **kwargs)

    if isinstance(result, dict) and kwargs.get("verbose", True):
        for key in ("formatted", "table", "report"):
            value = result.get(key)
            if isinstance(value, str) and value:
                print(value)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
