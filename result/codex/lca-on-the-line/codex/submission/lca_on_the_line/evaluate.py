"""Main evaluation loop: ID LCA / Top-1 and OOD Top-1/Top-5 for all models.

This is the machinery behind Tables 1-3 and Figures 1/5/9: every model is run
once on ImageNet (ID) and once on each OOD dataset, the logits are cached to
disk, and the per-model metrics are written to ``metrics.csv``.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .data import OOD_DATASETS, load_dataset_by_name
from .hierarchy import WordNetHierarchy, load_wordnet_hierarchy
from .lca import dataset_lca, elca_from_logits, topk_accuracy
from .models import DEFAULT_CACHE_DIR, all_model_specs, build_classifier

METRIC_FIELDS = [
    "model",
    "family",
    "source",
    "dataset",
    "top1",
    "top5",
    "lca",
    "elca",
    "n_samples",
    "n_mistakes",
    "seconds",
]


def logits_path(out_dir: str, model: str, dataset: str) -> str:
    safe = model.replace("/", "_").replace("@", "_").replace(" ", "")
    return os.path.join(out_dir, "logits", "%s__%s.npy" % (safe, dataset))


def collect_logits(
    classifier,
    dataset,
    batch_size: int = 64,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Run the classifier over a dataset and stack the logits."""
    order = _sample_order(len(dataset), limit, indices)
    outs: List[np.ndarray] = []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        images = [dataset[i][0] for i in chunk]
        outs.append(classifier.logits(images))
    if not outs:
        return np.zeros((0, 1000), dtype=np.float32)
    return np.concatenate(outs, axis=0).astype(np.float32)


def collect_features(
    classifier,
    dataset,
    batch_size: int = 64,
    limit: Optional[int] = None,
    indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Extract ``M(X)`` features (last hidden layer / image embedding)."""
    order = _sample_order(len(dataset), limit, indices)
    outs: List[np.ndarray] = []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        images = [dataset[i][0] for i in chunk]
        outs.append(classifier.features(images))
    if not outs:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(outs, axis=0).astype(np.float32)


def _sample_order(n_total: int, limit: Optional[int],
                  indices: Optional[Sequence[int]]) -> np.ndarray:
    if indices is not None:
        return np.asarray(indices, dtype=np.int64)
    n = n_total if limit is None else min(limit, n_total)
    return np.arange(n, dtype=np.int64)


def stratified_subset_indices(targets: Sequence[int], per_class: int,
                              seed: int = 0) -> np.ndarray:
    """Pick up to ``per_class`` samples per class (for feature extraction)."""
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    chosen: List[int] = []
    for cls in np.unique(targets):
        idx = np.where(targets == cls)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        chosen.extend(idx.tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def dataset_targets(dataset, limit: Optional[int] = None) -> np.ndarray:
    n = len(dataset) if limit is None else min(limit, len(dataset))
    return np.asarray([dataset[i][1] for i in range(n)], dtype=np.int64)


def metrics_for_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    hierarchy: Optional[WordNetHierarchy] = None,
    compute_elca: bool = True,
) -> Dict[str, float]:
    preds = logits.argmax(axis=1)
    out = {
        "top1": topk_accuracy(logits, targets, 1),
        "top5": topk_accuracy(logits, targets, 5),
        "n_samples": int(len(targets)),
    }
    if hierarchy is not None:
        out["lca"] = dataset_lca(
            hierarchy=hierarchy, predictions=preds, targets=targets
        )
        out["n_mistakes"] = int((preds != targets).sum())
        if compute_elca:
            out["elca"] = elca_from_logits(logits, targets, hierarchy=hierarchy)
    return out


def run_benchmark(
    model_names: Sequence[str],
    data_root: str,
    out_dir: str,
    datasets: Sequence[str] = ("imagenet",) + OOD_DATASETS,
    limit: Optional[int] = None,
    batch_size: int = 64,
    device: str = "cpu",
    compute_elca: bool = True,
    overwrite: bool = False,
    hierarchy: Optional[WordNetHierarchy] = None,
    loader_overrides: Optional[Dict[str, object]] = None,
    templates: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
) -> str:
    """Evaluate ``model_names`` on ``datasets`` and write ``metrics.csv``."""
    os.makedirs(os.path.join(out_dir, "logits"), exist_ok=True)
    hierarchy = hierarchy if hierarchy is not None else load_wordnet_hierarchy()
    specs = {s.name: s for s in all_model_specs()}
    loader_overrides = loader_overrides or {}

    def load(dataset_name: str):
        if dataset_name in loader_overrides:
            return loader_overrides[dataset_name]
        return load_dataset_by_name(dataset_name, data_root)

    rows: List[Dict[str, object]] = []
    for name in model_names:
        spec = specs[name]
        print("[eval] building %s (%s)" % (name, spec.source))
        classifier = build_classifier(
            spec, device=device, batch_size=batch_size, cache_dir=cache_dir,
            templates=templates,
        )
        for dataset_name in datasets:
            path = logits_path(out_dir, name, dataset_name)
            start = time.time()
            if os.path.exists(path) and not overwrite:
                logits = np.load(path)
            else:
                dataset = load(dataset_name)
                logits = collect_logits(
                    classifier, dataset, batch_size=batch_size, limit=limit
                )
                np.save(path, logits.astype(np.float16))
            targets = _cached_targets(
                out_dir, dataset_name, data_root, limit,
                dataset=loader_overrides.get(dataset_name),
            )
            if len(targets) != len(logits):
                targets = targets[: len(logits)]
            metrics = metrics_for_logits(
                logits.astype(np.float64), targets, hierarchy, compute_elca
            )
            row = {
                "model": name,
                "family": spec.family,
                "source": spec.source,
                "dataset": dataset_name,
            }
            row.update(metrics)
            row["seconds"] = round(time.time() - start, 2)
            rows.append(row)
            print(
                "[eval] %-40s %-12s top1=%.4f lca=%s"
                % (
                    name,
                    dataset_name,
                    row.get("top1", float("nan")),
                    ("%.4f" % row["lca"]) if "lca" in row else "-",
                )
            )
        del classifier

    csv_path = os.path.join(out_dir, "metrics.csv")
    _write_rows(csv_path, rows)
    # also persist the canonical dataset order for the analysis scripts
    with open(os.path.join(out_dir, "dataset_order.txt"), "w") as fh:
        fh.write("\n".join(datasets))
    return csv_path


def _cached_targets(out_dir: str, dataset_name: str, data_root: str,
                    limit: Optional[int], dataset=None) -> np.ndarray:
    path = os.path.join(out_dir, "targets", "%s.npy" % dataset_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        targets = np.load(path)
    else:
        dataset = dataset if dataset is not None else load_dataset_by_name(
            dataset_name, data_root
        )
        targets = dataset_targets(dataset, limit)
        np.save(path, targets)
    if limit is not None:
        targets = targets[:limit]
    return targets


def _write_rows(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LCA-on-the-Line benchmark")
    parser.add_argument("--models", nargs="*", default=None,
                        help="model names (default: all 75)")
    parser.add_argument("--data-root", required=True,
                        help="directory holding the ImageNet + OOD datasets")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N images (smoke tests)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-templates", type=int, default=None,
                        help="number of CLIP prompt templates (default: all 80)")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-elca", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    models = args.models or [s.name for s in all_model_specs()]
    datasets = args.datasets or ["imagenet"] + list(OOD_DATASETS)
    templates = None
    if args.n_templates:
        from .prompt_engineering import IMAGENET_TEMPLATES

        templates = IMAGENET_TEMPLATES[: args.n_templates]
    path = run_benchmark(
        model_names=models,
        data_root=args.data_root,
        out_dir=args.out_dir,
        datasets=datasets,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
        compute_elca=not args.no_elca,
        overwrite=args.overwrite,
        templates=templates,
        cache_dir=args.cache_dir,
    )
    print("wrote", path)


if __name__ == "__main__":  # pragma: no cover
    main()
