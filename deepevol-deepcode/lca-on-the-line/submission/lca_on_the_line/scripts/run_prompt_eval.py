#!/usr/bin/env python
"""Reproduce Table 14 of *LCA-on-the-Line*: taxonomy-aware prompt engineering.

Paper: "LCA-on-the-Line: In-Distribution Taxonomic Distance (LCA) Predicts
Out-of-Distribution Generalization" -- Section 4.3.3 / Table 14.

This script performs zero-shot classification with CLIP-ViT32 using four prompt
templates and reports Top-1 accuracy (and test-time CrossEntropy) on ImageNet-1k
plus the five severe-shift OOD datasets:

    1. Baseline         : "<class>"
    2. Stack Parent     : "<class, parent, grandparent>"  (path without 'is-a')
    3. Taxonomy Parent  : "<class, which is a type of parent, which is a type of grandparent>"
    4. Shuffle Parent   : same phrasing as (3) but with randomly sampled (wrong) ancestors

The paper's expectation (Table 14) is:

    Baseline 0.589 -> Taxonomy Parent 0.626 on ImageNet (CLIP-ViT32),
    with Stack/Shuffle Parent worse than Baseline on the shifted datasets and
    test CE decreasing for Taxonomy Parent across all six datasets.

The heavy machinery lives in ``src/alignment/prompt_engineering.py``; this file
acts as a defensively-written orchestration layer (lazy imports, signature
filtering, graceful fallbacks) so that it can be executed offline, from cached
image features, or fully online with CLIP / OpenCLIP checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("run_prompt_eval")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# import bootstrap (mirrors the other scripts in this repo)
# ---------------------------------------------------------------------------
def _bootstrap() -> None:
    """Make ``src`` importable regardless of how the script is launched."""
    for candidate in (REPO_ROOT, os.path.join(REPO_ROOT, "src")):
        if candidate and candidate not in sys.path:
            sys.path.insert(0, candidate)


_bootstrap()


def _import_attr(module_names: Sequence[str], attr: str) -> Optional[Any]:
    """Import ``attr`` from the first importable module in ``module_names``."""
    import importlib

    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception:  # pragma: no cover - optional dependency
            continue
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def _filter_kwargs(func: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs a (possibly older) callable does not accept."""
    import inspect

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in signature.parameters}


def _call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` while tolerating signature drift between modules."""
    try:
        return func(*args, **kwargs)
    except TypeError:
        return func(*args, **_filter_kwargs(func, kwargs))


# ---------------------------------------------------------------------------
# prompt-engineering helpers (resolved lazily)
# ---------------------------------------------------------------------------
def _pe() -> Any:
    module = _import_attr(
        ("src.alignment.prompt_engineering", "alignment.prompt_engineering",
         "prompt_engineering"),
        "PROMPT_TEMPLATES",
    )
    if module is None:
        raise ImportError(
            "src/alignment/prompt_engineering.py is required for run_prompt_eval.py"
        )
    import importlib

    for name in ("src.alignment.prompt_engineering", "alignment.prompt_engineering",
                 "prompt_engineering"):
        try:
            return importlib.import_module(name)
        except Exception:  # pragma: no cover
            continue
    raise ImportError("could not import prompt_engineering")


DEFAULT_TEMPLATES: Tuple[str, ...] = ("baseline", "stack_parent", "taxonomy_parent",
                                      "shuffle_parent")
DATASETS: Tuple[str, ...] = ("imagenet", "v2", "s", "r", "a", "objectnet")
OOD_DATASETS: Tuple[str, ...] = ("v2", "s", "r", "a", "objectnet")
SEVERE_SHIFT: Tuple[str, ...] = ("s", "r", "a", "objectnet")

DISPLAY = {
    "imagenet": "ImageNet",
    "v2": "ImgN-v2",
    "s": "ImgN-S",
    "r": "ImgN-R",
    "a": "ImgN-A",
    "objectnet": "ObjNet",
}

TEMPLATE_DISPLAY = {
    "baseline": "Baseline",
    "stack_parent": "Stack Parent",
    "taxonomy_parent": "Taxonomy Parent",
    "shuffle_parent": "Shuffle Parent",
    "ensemble_taxonomy": "Ensemble Taxonomy",
}

# Paper Table 14 reference values (CLIP-ViT32, Top-1).
TABLE14_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    "imagenet": {
        "baseline": {"top1": 0.589, "ce": 2.70},
        "stack_parent": {"top1": 0.592, "ce": 2.62},
        "taxonomy_parent": {"top1": 0.626, "ce": 2.26},
        "shuffle_parent": {"top1": 0.594, "ce": 2.64},
    },
    "v2": {
        "baseline": {"top1": 0.528, "ce": 2.95},
        "taxonomy_parent": {"top1": 0.561, "ce": 2.58},
    },
    "s": {
        "baseline": {"top1": 0.352, "ce": 4.06},
        "taxonomy_parent": {"top1": 0.384, "ce": 3.55},
    },
    "r": {
        "baseline": {"top1": 0.604, "ce": 2.47},
        "taxonomy_parent": {"top1": 0.632, "ce": 2.18},
    },
    "a": {
        "baseline": {"top1": 0.365, "ce": 4.47},
        "taxonomy_parent": {"top1": 0.400, "ce": 3.80},
    },
    "objectnet": {
        "baseline": {"top1": 0.336, "ce": 4.21},
        "taxonomy_parent": {"top1": 0.368, "ce": 3.58},
    },
}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file (``configs/config.yaml`` by default)."""
    candidates = [path] if path else []
    candidates += [os.path.join(REPO_ROOT, "configs", "config.yaml"),
                   os.path.join(os.getcwd(), "configs", "config.yaml")]
    for candidate in candidates:
        if not candidate or not os.path.isfile(candidate):
            continue
        try:
            import yaml

            with open(candidate, "r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            LOG.info("loaded config from %s", candidate)
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.debug("could not parse config %s: %s", candidate, exc)
    return {}


def _cfg(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Flat lookup across nested config sections."""
    for key in keys:
        if key in config:
            return config[key]
    for key in keys:
        head, _, tail = key.partition(".")
        if head in config and isinstance(config[head], dict) and tail in config[head]:
            return config[head][tail]
    return default


def resolve_cache_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    cache = getattr(args, "cache_dir", None) or _cfg(
        config, "cache_dir", "paths.cache_dir", "outputs.cache_dir", "output_dir", default=None
    )
    if not cache:
        cache = os.path.join(REPO_ROOT, "cache", "outputs")
    return os.path.abspath(cache)


def resolve_results_dir(args: argparse.Namespace, config: Dict[str, Any]) -> str:
    results = getattr(args, "results_dir", None) or _cfg(
        config, "results_dir", "paths.results_dir", default=None
    )
    if not results:
        results = os.path.join(REPO_ROOT, "results", "prompt_eval")
    return os.path.abspath(results)


def parse_ood_roots(values: Optional[Sequence[str]]) -> Dict[str, str]:
    """Parse repeated ``name=path`` CLI arguments."""
    roots: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            LOG.warning("ignoring malformed --ood-root %r (expected name=path)", value)
            continue
        name, _, path = value.partition("=")
        roots[name.strip().lower()] = path.strip()
    return roots


# ---------------------------------------------------------------------------
# hierarchy / class names
# ---------------------------------------------------------------------------
def build_hierarchy(csv_path: Optional[str] = None, allow_synthetic: bool = True) -> Any:
    builder = _import_attr(
        ("src.hierarchy.wordnet", "hierarchy.wordnet", "wordnet"),
        "build_wordnet_hierarchy",
    )
    if builder is None:
        raise ImportError("src/hierarchy/wordnet.py is required")
    return _call(builder, csv_path=csv_path, allow_synthetic=allow_synthetic)


def load_class_names(num_classes: int = 1000) -> List[str]:
    loader = _import_attr(
        ("src.data.imagenet", "data.imagenet", "imagenet"),
        "load_imagenet_class_names",
    )
    names: List[str] = []
    if loader is not None:
        try:
            names = list(_call(loader) or [])
        except Exception as exc:  # pragma: no cover - offline
            LOG.debug("could not load ImageNet class names: %s", exc)
    if len(names) < num_classes:
        names = list(names) + [f"class_{i}" for i in range(len(names), num_classes)]
    return names[:num_classes]


# ---------------------------------------------------------------------------
# dataset / feature plumbing
# ---------------------------------------------------------------------------
def build_dataset(
    name: str,
    dataset_root: Optional[str],
    ood_roots: Dict[str, str],
    transform: Any = None,
    resolution: int = 224,
    max_samples: Optional[int] = None,
    seed: int = 0,
    allow_synthetic: bool = False,
) -> Optional[Any]:
    """Build an ID or OOD dataset by canonical name."""
    key = (name or "").lower()
    try:
        if key in ("imagenet", "id", "imagenet-1k", "val", "validation"):
            builder = _import_attr(
                ("src.data.imagenet", "data.imagenet", "imagenet"),
                "build_imagenet_dataset",
            )
            if builder is None:
                return None
            return _call(
                builder,
                root=dataset_root,
                resolution=resolution,
                transform=transform,
                max_samples=max_samples,
                seed=seed,
                allow_synthetic=allow_synthetic,
            )
        builder = _import_attr(
            ("src.data.ood_datasets", "data.ood_datasets", "ood_datasets"),
            "build_ood_dataset",
        )
        if builder is None:
            return None
        return _call(
            builder,
            key,
            root=ood_roots.get(key) or dataset_root,
            transform=transform,
            resolution=resolution,
            max_samples=max_samples,
            seed=seed,
            allow_synthetic=allow_synthetic,
        )
    except Exception as exc:
        LOG.warning("could not build dataset %s: %s", name, exc)
        return None


def build_loader(dataset: Any, batch_size: int = 64, num_workers: int = 4) -> Any:
    if dataset is None:
        return None
    maker = _import_attr(
        ("src.data.imagenet", "data.imagenet", "imagenet"), "build_imagenet_loader"
    )
    if maker is not None:
        try:
            return _call(
                maker,
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
        except Exception as exc:  # pragma: no cover
            LOG.debug("build_imagenet_loader failed: %s", exc)
    try:  # pragma: no cover - fallback
        from torch.utils.data import DataLoader

        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                          shuffle=False, pin_memory=True)
    except Exception:
        return None


def default_encoder_name(backend: str) -> str:
    return "ViT-B/32" if backend == "clip" else "ViT-B-32"


def build_prompt_encoder(
    backend: str = "auto",
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    logit_scale: Optional[float] = None,
) -> Optional[Any]:
    """Resolve a text/image encoder through ``prompt_engineering.build_encoder``."""
    factory = _import_attr(
        ("src.alignment.prompt_engineering", "alignment.prompt_engineering",
         "prompt_engineering"),
        "build_encoder",
    )
    name = model_name or default_encoder_name("clip" if backend == "clip" else "open_clip")
    order = {
        "clip": ("clip",),
        "open_clip": ("open_clip",),
        "auto": ("auto",),
    }.get(backend, ("auto",))
    for candidate in order:
        if factory is None:
            break
        try:
            encoder = _call(
                factory,
                encoder=None,
                backbone=name,
                backend=candidate,
                device=device,
                logit_scale=logit_scale,
            )
            if encoder is not None:
                return encoder
        except Exception as exc:
            LOG.debug("encoder backend %s failed for %s: %s", candidate, name, exc)
    LOG.error(
        "could not build a CLIP encoder for %s; install `clip`/`open_clip_torch` or "
        "pass --features-json produced offline", name,
    )
    return None


def collect_image_features(
    encoder: Any,
    loader: Any,
    device: Optional[str] = None,
    max_batches: Optional[int] = None,
) -> Tuple[Any, Any]:
    """Stream a loader through an encoder -> (image_features, targets)."""
    collect = _import_attr(
        ("src.alignment.prompt_engineering", "alignment.prompt_engineering",
         "prompt_engineering"),
        "encode_image_loader",
    )
    if collect is not None:
        try:
            return _call(collect, encoder, loader, device=device, max_batches=max_batches)
        except Exception as exc:  # pragma: no cover
            LOG.debug("encode_image_loader failed (%s); falling back to manual loop", exc)
    # manual fallback
    import numpy as np

    feats: List[Any] = []
    targets: List[Any] = []
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        images, labels = batch[0], batch[1]
        out = encoder.encode_images(images)
        try:  # torch tensor
            feats.append(out.detach().cpu().numpy())
        except AttributeError:
            feats.append(np.asarray(out))
        try:
            targets.append(labels.detach().cpu().numpy())
        except AttributeError:
            targets.append(np.asarray(labels))
    if not feats:
        return np.zeros((0, 1), dtype=float), np.zeros((0,), dtype=int)
    return np.concatenate(feats, axis=0), np.concatenate(targets, axis=0).astype(int)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate_prompts(
    encoder: Any,
    image_features: Any,
    targets: Any,
    prompts: Sequence[str],
    logit_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate one prompt bank (delegates to prompt_engineering.evaluate_prompts)."""
    evaluator = _import_attr(
        ("src.alignment.prompt_engineering", "alignment.prompt_engineering",
         "prompt_engineering"),
        "evaluate_prompts",
    )
    if evaluator is not None:
        return dict(
            _call(
                evaluator,
                encoder,
                image_features,
                targets,
                prompts,
                logit_scale=logit_scale,
                compute_top5=True,
            )
        )
    # local fallback: only possible if the caller supplied text features
    raise RuntimeError("prompt_engineering.evaluate_prompts is unavailable")


def run_prompt_evaluation(
    encoder: Any,
    image_features_by_dataset: Dict[str, Any],
    targets_by_dataset: Dict[str, Any],
    hierarchy: Any = None,
    class_names: Optional[Sequence[str]] = None,
    templates: Sequence[str] = DEFAULT_TEMPLATES,
    num_classes: int = 1000,
    shuffle_seed: int = 0,
    max_ancestors: int = 2,
    logit_scale: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build prompt banks and evaluate each template on each dataset.

    Returns ``{dataset: {template: {"top1":..., "top5":..., "ce":..., "n":...}}}``.
    """
    import numpy as np

    pe = _pe()
    build_all = getattr(pe, "build_all_prompt_texts", None)
    prompt_banks: Dict[str, List[str]] = {}
    if build_all is not None:
        try:
            banks = _call(
                build_all,
                hierarchy=hierarchy,
                class_names=class_names,
                templates=tuple(templates),
                num_classes=num_classes,
                shuffle_seed=shuffle_seed,
                max_ancestors=max_ancestors,
            )
            if isinstance(banks, dict):
                prompt_banks = {k: list(v) for k, v in banks.items()}
        except Exception as exc:
            LOG.warning("build_all_prompt_texts failed: %s", exc)
    if not prompt_banks:
        builder = getattr(pe, "build_prompt_texts", None)
        if builder is not None:
            for template in templates:
                try:
                    prompt_banks[template] = list(
                        _call(
                            builder,
                            template,
                            hierarchy=hierarchy,
                            class_names=class_names,
                            num_classes=num_classes,
                            shuffle_seed=shuffle_seed,
                            max_ancestors=max_ancestors,
                        )
                    )
                except Exception as exc:
                    LOG.warning("build_prompt_texts(%s) failed: %s", template, exc)

    if not prompt_banks:
        raise RuntimeError("no prompt texts could be constructed")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name in DATASETS:
        features = image_features_by_dataset.get(dataset_name)
        targets = targets_by_dataset.get(dataset_name)
        if features is None or targets is None:
            LOG.warning("skipping %s: no image features available", dataset_name)
            continue
        features = np.asarray(features)
        targets = np.asarray(targets).astype(int).reshape(-1)
        if features.shape[0] != targets.shape[0]:
            LOG.warning(
                "skipping %s: feature/target length mismatch (%d vs %d)",
                dataset_name, features.shape[0], targets.shape[0],
            )
            continue
        dataset_results: Dict[str, Dict[str, float]] = {}
        for template in templates:
            prompts = prompt_banks.get(template)
            if not prompts:
                continue
            try:
                metrics = evaluate_prompts(
                    encoder, features, targets, prompts, logit_scale=logit_scale
                )
            except Exception as exc:
                LOG.warning("evaluation failed (%s / %s): %s", dataset_name, template, exc)
                continue
            dataset_results[template] = {
                "top1": float(metrics.get("top1", float("nan"))),
                "top5": float(metrics.get("top5", float("nan"))),
                "ce": float(metrics.get("ce", float("nan"))),
                "n": int(metrics.get("n", len(targets))),
            }
            if verbose:
                LOG.info(
                    "%-9s %-16s top1=%.4f top5=%.4f ce=%.4f",
                    DISPLAY.get(dataset_name, dataset_name), template,
                    dataset_results[template]["top1"],
                    dataset_results[template]["top5"],
                    dataset_results[template]["ce"],
                )
        if dataset_results:
            results[dataset_name] = dataset_results
    return results


# ---------------------------------------------------------------------------
# validation / reporting
# ---------------------------------------------------------------------------
def check_against_table14(
    table: Dict[str, Dict[str, Dict[str, float]]],
    tolerance: float = 0.05,
    require_improvement: bool = True,
) -> List[str]:
    """Compare observed results with the paper's Table 14 expectations."""
    checks: List[str] = []

    checker = _import_attr(
        ("src.alignment.prompt_engineering", "alignment.prompt_engineering",
         "prompt_engineering"),
        "check_against_table14",
    )
    if checker is not None:
        try:
            checks.extend(list(_call(checker, table, tolerance=tolerance,
                                     require_improvement=require_improvement) or []))
        except Exception as exc:  # pragma: no cover
            LOG.debug("prompt_engineering.check_against_table14 failed: %s", exc)

    if not table:
        checks.append("FAIL: no results were produced (encoder/datasets unavailable)")
        return checks

    # Top-1 agreement with the reference values
    for dataset_name, templates in TABLE14_REFERENCE.items():
        observed = table.get(dataset_name) or {}
        for template, reference in templates.items():
            got = observed.get(template)
            if not got:
                continue
            delta = abs(float(got.get("top1", float("nan"))) - reference["top1"])
            if delta > tolerance:
                checks.append(
                    f"WARN: {DISPLAY.get(dataset_name, dataset_name)}/"
                    f"{TEMPLATE_DISPLAY.get(template, template)} top1="
                    f"{got.get('top1'):.4f} vs reference {reference['top1']:.3f} "
                    f"(delta {delta:.4f} > {tolerance})"
                )

    # Taxonomy Parent must beat all ablations on every dataset
    if require_improvement:
        for dataset_name, templates in table.items():
            baseline = (templates.get("baseline") or {}).get("top1")
            taxonomy = (templates.get("taxonomy_parent") or {}).get("top1")
            if baseline is None or taxonomy is None:
                continue
            if taxonomy < baseline - 1e-6:
                checks.append(
                    f"FAIL: Taxonomy Parent ({taxonomy:.4f}) does not beat Baseline "
                    f"({baseline:.4f}) on {DISPLAY.get(dataset_name, dataset_name)}"
                )
            for ablation in ("stack_parent", "shuffle_parent"):
                other = (templates.get(ablation) or {}).get("top1")
                if other is None:
                    continue
                if taxonomy < other - 1e-6:
                    checks.append(
                        f"FAIL: Taxonomy Parent ({taxonomy:.4f}) does not beat "
                        f"{TEMPLATE_DISPLAY.get(ablation, ablation)} ({other:.4f}) on "
                        f"{DISPLAY.get(dataset_name, dataset_name)}"
                    )

    # Test CE must decrease for Taxonomy Parent vs Baseline
    for dataset_name, templates in table.items():
        base_ce = (templates.get("baseline") or {}).get("ce")
        tax_ce = (templates.get("taxonomy_parent") or {}).get("ce")
        if base_ce is None or tax_ce is None:
            continue
        if tax_ce > base_ce + 1e-6:
            checks.append(
                f"FAIL: test CE increased for Taxonomy Parent on "
                f"{DISPLAY.get(dataset_name, dataset_name)} "
                f"({base_ce:.4f} -> {tax_ce:.4f})"
            )

    if not checks:
        checks.append(
            "PASS: Taxonomy Parent improves Top-1 and test CE over Baseline, and "
            "beats the Stack/Shuffle ablations on all evaluated datasets"
        )
    return checks


def format_table14(
    table: Dict[str, Dict[str, Dict[str, float]]],
    templates: Sequence[str] = DEFAULT_TEMPLATES,
    metric: str = "top1",
    decimals: int = 4,
) -> str:
    """Render a Table-14-style text block (rows = dataset, columns = template)."""
    header = f"{'Dataset':<10}" + "".join(
        f"{TEMPLATE_DISPLAY.get(t, t):>18}" for t in templates
    )
    lines = [header, "-" * len(header)]
    for dataset_name in DATASETS:
        templates_obs = table.get(dataset_name)
        if not templates_obs:
            continue
        row = f"{DISPLAY.get(dataset_name, dataset_name):<10}"
        for template in templates:
            value = (templates_obs.get(template) or {}).get(metric)
            row += " " * 17 + "n/a" if value is None else f"{value:>18.{decimals}f}"
        lines.append(row)
    return "\n".join(lines)


def success_summary(
    table: Dict[str, Dict[str, Dict[str, float]]],
    datasets: Sequence[str] = DATASETS,
) -> Dict[str, Any]:
    """Machine-readable summary of the Table 14 gates."""
    summary: Dict[str, Any] = {"datasets": {}, "passes_table14": None}
    improved: List[bool] = []
    for dataset_name in datasets:
        observed = table.get(dataset_name)
        if not observed:
            continue
        baseline = (observed.get("baseline") or {}).get("top1")
        taxonomy = (observed.get("taxonomy_parent") or {}).get("top1")
        base_ce = (observed.get("baseline") or {}).get("ce")
        tax_ce = (observed.get("taxonomy_parent") or {}).get("ce")
        entry = {
            "baseline_top1": baseline,
            "taxonomy_top1": taxonomy,
            "delta_top1": None if baseline is None or taxonomy is None else taxonomy - baseline,
            "baseline_ce": base_ce,
            "taxonomy_ce": tax_ce,
            "delta_ce": None if base_ce is None or tax_ce is None else base_ce - tax_ce,
        }
        entry["improved_top1"] = bool(
            entry["delta_top1"] is not None and entry["delta_top1"] > 0
        )
        entry["improved_ce"] = bool(entry["delta_ce"] is not None and entry["delta_ce"] > 0)
        if entry["delta_top1"] is not None:
            improved.append(entry["improved_top1"])
        summary["datasets"][dataset_name] = entry
    if improved:
        summary["passes_table14"] = bool(all(improved))
    return summary


def save_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# offline mode (cached image features produced elsewhere)
# ---------------------------------------------------------------------------
def load_features_json(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load pre-computed image features/targets from a ``.npz`` file."""
    import numpy as np

    payload = np.load(path, allow_pickle=True)
    features = payload["features"] if "features" in payload else payload["image_features"]
    targets = payload["targets"] if "targets" in payload else payload["labels"]
    return np.asarray(features), np.asarray(targets).astype(int)


def synthetic_features(
    num_classes: int = 1000,
    samples_per_class: int = 2,
    feature_dim: int = 512,
    seed: int = 0,
) -> Tuple[Any, Any]:
    """Deterministic separable features for offline smoke tests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    class_centers = rng.normal(size=(num_classes, feature_dim)).astype(np.float32)
    class_centers /= np.linalg.norm(class_centers, axis=1, keepdims=True) + 1e-8
    features: List[Any] = []
    targets: List[Any] = []
    for class_index in range(num_classes):
        noise = rng.normal(scale=0.3, size=(samples_per_class, feature_dim))
        batch = class_centers[class_index][None, :] + noise
        batch /= np.linalg.norm(batch, axis=1, keepdims=True) + 1e-8
        features.append(batch.astype(np.float32))
        targets.append(np.full(samples_per_class, class_index, dtype=int))
    return np.concatenate(features, axis=0), np.concatenate(targets, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 14 (taxonomy-aware prompt engineering) of "
                    "LCA-on-the-Line."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to configs/config.yaml")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory with cached model outputs (.npz)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Where to write JSON/PNG artifacts")
    parser.add_argument("--dataset-root", type=str, default=None,
                        help="ImageNet-1k root (ID dataset)")
    parser.add_argument("--ood-root", type=str, action="append", default=None,
                        help="OOD dataset root as name=path (repeatable)")
    parser.add_argument("--hierarchy-csv", type=str, default=None,
                        help="Path to imagenet_fiveai.csv")
    parser.add_argument("--class-names", type=str, default=None,
                        help="Optional newline-separated class-name file")
    parser.add_argument("--encoder", type=str, default="ViT-B/32",
                        help="CLIP backbone name (default: ViT-B/32, i.e. CLIP-ViT32)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=("auto", "clip", "open_clip"),
                        help="Which CLIP implementation to use")
    parser.add_argument("--templates", type=str, nargs="+", default=list(DEFAULT_TEMPLATES),
                        help="Prompt templates to evaluate")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--max-ancestors", type=int, default=2,
                        help="Number of ancestor levels in the prompts")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="Seed for the Shuffle Parent ablation")
    parser.add_argument("--logit-scale", type=float, default=None,
                        help="Override the CLIP logit scale (default: model value)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per dataset (smoke tests)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit batches per dataset when extracting features")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--features-json", type=str, default=None,
                        help="Pre-computed single-dataset image features (.npz)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Offline smoke test with synthetic features")
    parser.add_argument("--allow-synthetic-datasets", action="store_true",
                        help="Use synthetic dataset fallbacks when unavailable")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Tolerance when checking against Table 14")
    parser.add_argument("--no-improvement-check", action="store_true",
                        help="Skip the Taxonomy>ablation improvement assertions")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    started = time.time()
    config = load_config(args.config)
    cache_dir = resolve_cache_dir(args, config)
    results_dir = resolve_results_dir(args, config)
    os.makedirs(results_dir, exist_ok=True)
    ood_roots = parse_ood_roots(args.ood_root) or dict(
        _cfg(config, "ood_roots", default={}) or {}
    )

    num_classes = int(_cfg(config, "num_classes", default=args.num_classes) or args.num_classes)
    hierarchy: Any = None
    try:
        hierarchy = build_hierarchy(
            csv_path=args.hierarchy_csv or _cfg(config, "hierarchy_csv", "hierarchy.csv_path"),
            allow_synthetic=True,
        )
        LOG.info("hierarchy ready (%s classes)", getattr(hierarchy, "num_classes", num_classes))
    except Exception as exc:
        LOG.warning("could not build WordNet hierarchy (%s); prompts will be class-name only", exc)

    class_names: Optional[List[str]] = None
    if args.class_names and os.path.isfile(args.class_names):
        with open(args.class_names, "r", encoding="utf-8") as handle:
            class_names = [line.strip() for line in handle if line.strip()]
    if not class_names:
        class_names = load_class_names(num_classes)

    # ---- image features -------------------------------------------------
    image_features: Dict[str, Any] = {}
    targets: Dict[str, Any] = {}

    if args.synthetic:
        LOG.warning("SYNTHETIC MODE: features are random -- results are smoke-test only")
        import numpy as np

        for dataset_name in DATASETS:
            features, labels = synthetic_features(
                num_classes=num_classes, samples_per_class=2, feature_dim=512,
                seed=abs(hash(dataset_name)) % (2 ** 31),
            )
            image_features[dataset_name] = features
            targets[dataset_name] = labels
    elif args.features_json:
        features, labels = load_features_json(args.features_json)
        image_features["imagenet"] = features
        targets["imagenet"] = labels
        LOG.info("loaded %d pre-computed features from %s", features.shape[0], args.features_json)

    encoder: Any = None
    if not image_features:
        encoder = build_prompt_encoder(
            backend=args.backend,
            model_name=args.encoder,
            device=args.device,
            logit_scale=args.logit_scale,
        )
        if encoder is None:
            LOG.error("no encoder available; rerun with --synthetic or --features-json")
            return 2
        for dataset_name in DATASETS:
            dataset = build_dataset(
                dataset_name,
                args.dataset_root,
                ood_roots,
                resolution=args.resolution,
                max_samples=args.max_samples,
                allow_synthetic=args.allow_synthetic_datasets,
            )
            if dataset is None:
                LOG.warning("dataset %s unavailable; skipping", dataset_name)
                continue
            loader = build_loader(dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
            if loader is None:
                continue
            try:
                features, labels = collect_image_features(
                    encoder, loader, device=args.device, max_batches=args.max_batches
                )
            except Exception as exc:
                LOG.warning("feature extraction failed for %s: %s", dataset_name, exc)
                continue
            image_features[dataset_name] = features
            targets[dataset_name] = labels
            LOG.info("extracted %d features for %s", len(labels), dataset_name)
    elif encoder is None:
        encoder = build_prompt_encoder(
            backend=args.backend,
            model_name=args.encoder,
            device=args.device,
            logit_scale=args.logit_scale,
        )

    if not image_features:
        LOG.error("no image features available; nothing to evaluate")
        return 3
    if encoder is None:
        LOG.error("no text encoder available for prompt evaluation")
        return 2

    # ---- prompt evaluation ---------------------------------------------
    table = run_prompt_evaluation(
        encoder,
        image_features,
        targets,
        hierarchy=hierarchy,
        class_names=class_names,
        templates=tuple(args.templates),
        num_classes=num_classes,
        shuffle_seed=args.shuffle_seed,
        max_ancestors=args.max_ancestors,
        logit_scale=args.logit_scale,
        verbose=True,
    )
    if not table:
        LOG.error("prompt evaluation produced no results")
        return 3

    print()
    print(format_table14(table, templates=tuple(args.templates), metric="top1"))
    print()
    print(format_table14(table, templates=tuple(args.templates), metric="ce"))

    checks = check_against_table14(
        table, tolerance=args.tol, require_improvement=not args.no_improvement_check
    )
    LOG.info("Table 14 checks:")
    for line in checks:
        LOG.info("  %s", line)

    summary = success_summary(table, datasets=DATASETS)
    summary["runtime_seconds"] = time.time() - started
    summary["encoder"] = getattr(encoder, "name", args.encoder)
    summary["templates"] = list(args.templates)
    summary["synthetic"] = bool(args.synthetic)

    save_json(os.path.join(results_dir, "table14_prompts.json"), table)
    save_json(os.path.join(results_dir, "checks.json"), checks)
    save_json(os.path.join(results_dir, "success_summary.json"), summary)
    LOG.info("artifacts written to %s", results_dir)

    if summary.get("passes_table14") is False and not args.synthetic:
        LOG.warning("some datasets did not show the expected Table 14 improvement")
    return 0


if __name__ == "__main__":
    sys.exit(main())
