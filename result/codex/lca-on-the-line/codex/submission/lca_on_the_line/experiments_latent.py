"""Section 4.3.1 / Table 4: latent hierarchies from K-means clustering.

For every one of the 75 pretrained *source* models we

1. extract ``M(X)`` features on ImageNet and average them per class,
2. build a latent LCA distance matrix with 9-level K-means clustering,
3. score *all* 75 models with that matrix (ID LCA on ImageNet) and correlate the
   result with their OOD Top-1 accuracy.

The table reports the mean / min / max / std of the 75 correlations, next to the
WordNet-hierarchy correlation and the ID Top-1 baseline.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Sequence

import numpy as np

from .data import OOD_DATASETS, load_dataset_by_name
from .evaluate import (
    collect_features,
    dataset_targets,
    stratified_subset_indices,
)
from .hierarchy import load_wordnet_hierarchy
from .latent import (
    class_features_from_logits_and_features,
    latent_lca_distance_matrix,
)
from .models import all_model_specs, build_classifier


def class_features_for_model(
    model_name: str,
    data_root: str,
    per_class: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    dataset_name: str = "imagenet",
    seed: int = 0,
) -> np.ndarray:
    """Average per-class ImageNet features of one pretrained model."""
    spec = next(s for s in all_model_specs() if s.name == model_name)
    dataset = load_dataset_by_name(dataset_name, data_root)
    targets = dataset_targets(dataset)
    indices = stratified_subset_indices(targets, per_class, seed=seed)
    classifier = build_classifier(
        spec, device=device, batch_size=batch_size, cache_dir=cache_dir
    )
    feats = collect_features(
        classifier, dataset, batch_size=batch_size, indices=indices
    )
    return class_features_from_logits_and_features(
        feats, targets[indices], n_classes=1000
    )


def build_latent_matrices(
    model_names: Sequence[str],
    data_root: str,
    feature_cache: Optional[str] = None,
    per_class: int = 20,
    n_levels: int = 9,
    seed: int = 0,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """Latent LCA distance matrix for every source model (cached on disk)."""
    matrices: Dict[str, np.ndarray] = {}
    for name in model_names:
        cache_path = None
        if feature_cache:
            safe = name.replace("/", "_").replace("@", "_")
            cache_path = os.path.join(feature_cache, "%s.npy" % safe)
        if cache_path and os.path.exists(cache_path):
            class_features = np.load(cache_path)
        else:
            class_features = class_features_for_model(
                name, data_root, per_class=per_class, seed=seed, device=device
            )
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                np.save(cache_path, class_features)
        matrices[name] = latent_lca_distance_matrix(
            class_features, n_levels=n_levels, seed=seed
        )
    return matrices


def table4(
    matrices: Dict[str, np.ndarray],
    id_predictions: Sequence[np.ndarray],
    ood_predictions: Dict[str, Sequence[np.ndarray]],
    targets: Sequence[int],
    wordnet_matrix: Optional[np.ndarray] = None,
    id_top1: Optional[np.ndarray] = None,
    ood_top1: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, float]]:
    """Aggregate the per-hierarchy correlations (Table 4)."""
    from .lca import dataset_lca_from_matrix
    from .metrics import pea

    targets = np.asarray(targets)
    if id_top1 is None:
        id_top1 = np.array(
            [float((np.asarray(p) == targets).mean()) for p in id_predictions]
        )
    out: Dict[str, Dict[str, float]] = {}
    for ood_name, ood_preds in ood_predictions.items():
        ood_acc = np.array(
            [float((np.asarray(p) == targets).mean()) for p in ood_preds]
        )
        peason_per_matrix = []
        for name, matrix in matrices.items():
            lca = np.array(
                [
                    dataset_lca_from_matrix(matrix, p, targets)
                    for p in id_predictions
                ]
            )
            peason_per_matrix.append(pea(lca, ood_acc))
        arr = np.array(peason_per_matrix)
        entry = {
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "std": float(arr.std()),
        }
        if wordnet_matrix is not None:
            lca_wn = np.array(
                [
                    dataset_lca_from_matrix(wordnet_matrix, p, targets)
                    for p in id_predictions
                ]
            )
            entry["WordNet"] = pea(lca_wn, ood_acc)
        entry["ID Top1 baseline"] = pea(id_top1, ood_acc)
        out[ood_name] = entry
    return out


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Latent hierarchy benchmark")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--results-dir", required=True,
                        help="output of evaluate.py (metrics + cached logits)")
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="limit the number of evaluated images (debugging)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):  # pragma: no cover
    args = _parse_args(argv)
    model_names = args.models or [s.name for s in all_model_specs()]
    matrices = build_latent_matrices(
        model_names,
        args.data_root,
        feature_cache=args.feature_cache,
        per_class=args.per_class,
        device=args.device,
        seed=args.seed,
    )

    from .analysis import load_logits, load_metrics

    df = load_metrics(args.results_dir)
    evaluated = list(df["model"].unique())
    targets = np.load(os.path.join(args.results_dir, "targets", "imagenet.npy"))
    id_predictions = [
        load_logits(args.results_dir, m, "imagenet").argmax(axis=1) for m in evaluated
    ]
    ood_predictions = {}
    for ds in OOD_DATASETS:
        try:
            ood_predictions[ds] = [
                load_logits(args.results_dir, m, ds).argmax(axis=1)
                for m in evaluated
            ]
        except FileNotFoundError:
            continue

    hierarchy = load_wordnet_hierarchy()
    result = table4(
        matrices,
        id_predictions,
        ood_predictions,
        targets,
        wordnet_matrix=hierarchy.lca_distance_matrix("information"),
    )
    for dataset, entry in result.items():
        print(dataset, {k: round(v, 4) for k, v in entry.items()})
    return result


if __name__ == "__main__":  # pragma: no cover
    main()
