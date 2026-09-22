"""Section 4.3.2 / Tables 5, 6: LCA soft labels for linear probing.

Trains a linear probe on top of frozen backbone features with

* the plain cross-entropy objective (baseline), and
* the paper's ``lambda * CE + soft_lca`` objective (Algorithm 1),

then interpolates in weight space (``W_interp = a W_ce + (1-a) W_soft``) and
evaluates on ImageNet plus the five OOD datasets.

The hierarchy used for the soft labels is either WordNet (Tables 5/9) or a
latent K-means hierarchy derived from a pretrained source model (Table 6).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .data import OOD_DATASETS, load_dataset_by_name
from .evaluate import collect_features, dataset_targets
from .hierarchy import load_wordnet_hierarchy
from .latent import latent_lca_distance_matrix
from .models import all_model_specs, build_classifier
from .soft_labels import (
    ProbeConfig,
    evaluate_probe,
    interpolate_weights,
    train_linear_probe,
)

EVAL_SPLITS = ["imagenet"] + list(OOD_DATASETS)


def extract_backbone_features(
    model_name: str,
    data_root: str,
    splits: Sequence[str] = ("train",) + tuple(EVAL_SPLITS),
    limit_train: Optional[int] = None,
    limit_eval: Optional[int] = None,
    batch_size: int = 64,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
    feature_cache: Optional[str] = None,
) -> Dict[str, tuple]:
    """Extract frozen backbone features for the probe train/test splits."""
    spec = next(s for s in all_model_specs() if s.name == model_name)
    classifier = build_classifier(
        spec, device=device, batch_size=batch_size, cache_dir=cache_dir
    )
    splits_out: Dict[str, tuple] = {}
    for split in splits:
        cache_path = None
        if feature_cache:
            cache_path = os.path.join(feature_cache, "%s__%s.npz" % (model_name, split))
        if cache_path and os.path.exists(cache_path):
            data = np.load(cache_path)
            splits_out[split] = (
                torch.from_numpy(data["x"]), torch.from_numpy(data["y"])
            )
            continue
        dataset_name = "imagenet" if split == "train" else split
        kwargs = {}
        if dataset_name == "imagenet":
            kwargs["split"] = "train" if split == "train" else "validation"
        dataset = load_dataset_by_name(dataset_name, data_root, **kwargs)
        limit = limit_train if split == "train" else limit_eval
        feats = collect_features(
            classifier, dataset, batch_size=batch_size, limit=limit
        )
        targets = dataset_targets(dataset, limit)
        splits_out[split] = (
            torch.from_numpy(feats), torch.from_numpy(targets)
        )
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, x=feats, y=targets)
    return splits_out


def soft_label_linear_probe(
    backbone: str,
    data_root: str,
    hierarchy_matrix: Optional[np.ndarray] = None,
    tree_prefix: str = "WordNet",
    config: Optional[ProbeConfig] = None,
    alphas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    limit_train: Optional[int] = None,
    limit_eval: Optional[int] = None,
    device: str = "cpu",
    feature_cache: Optional[str] = None,
) -> Dict[str, float]:
    """Table 5/6 row for one backbone and one hierarchy."""
    config = config or ProbeConfig()
    features = extract_backbone_features(
        backbone,
        data_root,
        limit_train=limit_train,
        limit_eval=limit_eval,
        device=device,
        feature_cache=feature_cache,
    )
    if hierarchy_matrix is None:
        hierarchy_matrix = load_wordnet_hierarchy().lca_distance_matrix("depth")

    from .soft_labels import build_soft_labels

    soft_matrix = build_soft_labels(
        hierarchy_matrix, temperature=config.temperature, tree_prefix=tree_prefix
    )
    train_x, train_y = features["train"]
    probe_ce = train_linear_probe(train_x, train_y, None, config, device=device)
    probe_soft = train_linear_probe(
        train_x, train_y, soft_matrix, config, device=device
    )

    results: Dict[str, float] = {}
    for split in EVAL_SPLITS:
        if split not in features:
            continue
        feat, targ = features[split]
        results["%s/baseline" % split] = evaluate_probe(
            probe_ce, feat, targ, device=device
        )["top1"]
        results["%s/soft" % split] = evaluate_probe(
            probe_soft, feat, targ, device=device
        )["top1"]
        for alpha in alphas:
            interp = interpolate_weights(probe_ce, probe_soft, alpha)
            results["%s/interp@%.2f" % (split, alpha)] = evaluate_probe(
                interp, feat, targ, device=device
            )["top1"]
    return results


def format_soft_label_table(results: Dict[str, float]) -> str:
    rows = {}
    for key, value in results.items():
        split, variant = key.split("/")
        rows.setdefault(split, {})[variant] = value
    lines = ["%-14s%-12s%-10s" % ("split", "variant", "top1")]
    for split, variants in rows.items():
        for variant, value in sorted(variants.items()):
            lines.append("%-14s%-12s%-10.4f" % (split, variant, value))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Table 10 / Figure 8: does the source model's quality improve the soft labels?
# --------------------------------------------------------------------------- #
def soft_label_quality_correlation(
    backbone: str,
    data_root: str,
    results_dir: str,
    source_models: Sequence[str],
    per_class: int = 20,
    config: Optional[ProbeConfig] = None,
    n_levels: int = 9,
    limit_train: Optional[int] = None,
    limit_eval: Optional[int] = None,
    device: str = "cpu",
    feature_cache: Optional[str] = None,
    seed: int = 0,
) -> Dict[str, object]:
    """Correlate source-model ID LCA with the OOD accuracy of its soft labels.

    For every pretrained source model we build a latent hierarchy from its
    ImageNet class-mean features, train a linear probe on ``backbone`` features
    with that hierarchy as the soft labels, and record the probe's OOD accuracy.
    We then correlate that accuracy with the *source model's* ID LCA distance on
    the WordNet hierarchy (loading the source predictions from the benchmark
    results directory).
    """
    from .experiments_latent import class_features_for_model
    from .lca import dataset_lca_from_matrix
    from .metrics import pea
    from .soft_labels import build_soft_labels, evaluate_probe, train_linear_probe

    config = config or ProbeConfig()
    features = extract_backbone_features(
        backbone,
        data_root,
        limit_train=limit_train,
        limit_eval=limit_eval,
        device=device,
        feature_cache=feature_cache,
    )
    train_x, train_y = features["train"]
    targets = np.load(os.path.join(results_dir, "targets", "imagenet.npy"))
    hierarchy = load_wordnet_hierarchy()
    wordnet_matrix = hierarchy.lca_distance_matrix("information")

    eval_splits = [ds for ds in OOD_DATASETS if ds in features]
    source_lca: List[float] = []
    soft_top1: Dict[str, List[float]] = {ds: [] for ds in eval_splits}
    base_top1: Dict[str, List[float]] = {ds: [] for ds in eval_splits}
    for name in source_models:
        # 1. the source model's own ID LCA on WordNet
        try:
            logits_path = os.path.join(
                results_dir, "logits", "%s__imagenet.npy"
                % name.replace("/", "_").replace("@", "_")
            )
            preds = np.load(logits_path).argmax(axis=1)
        except FileNotFoundError:
            print("[table10] skipping %s (no cached ImageNet logits)" % name)
            continue
        source_lca.append(
            dataset_lca_from_matrix(wordnet_matrix, preds, targets)
        )

        # 2. a probe trained with the latent hierarchy derived from that model
        class_features = class_features_for_model(
            name, data_root, per_class=per_class, device=device, seed=seed
        )
        latent_matrix = latent_lca_distance_matrix(
            class_features, n_levels=n_levels, seed=seed
        )
        soft_matrix = build_soft_labels(
            latent_matrix, temperature=config.temperature, tree_prefix="latent"
        )
        probe_ce = train_linear_probe(train_x, train_y, None, config, device=device)
        probe_soft = train_linear_probe(
            train_x, train_y, soft_matrix, config, device=device
        )
        for ds in eval_splits:
            feat, targ = features[ds]
            base_top1[ds].append(evaluate_probe(probe_ce, feat, targ, device=device)["top1"])
            soft_top1[ds].append(
                evaluate_probe(probe_soft, feat, targ, device=device)["top1"]
            )

    source_lca_arr = np.array(source_lca)
    return {
        "source_lca": source_lca_arr,
        "probe_soft_top1": {ds: np.array(v) for ds, v in soft_top1.items()},
        "probe_baseline_top1": {ds: np.array(v) for ds, v in base_top1.items()},
        "PEA": {
            "soft_labels": {
                ds: pea(source_lca_arr, np.array(soft_top1[ds]))
                for ds in eval_splits
            },
            "ce_baseline": {
                ds: pea(source_lca_arr, np.array(base_top1[ds]))
                for ds in eval_splits
            },
        },
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="LCA soft-label linear probing")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--hierarchy", default="WordNet",
                        help="WordNet or a path to a latent hierarchy matrix (.npy)")
    parser.add_argument("--source-model", default=None,
                        help="build the hierarchy from this model's features")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lambda-weight", type=float, default=0.03)
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-eval", type=int, default=None)
    parser.add_argument("--results-dir", default=None,
                        help="benchmark results dir; enables the Table 10 study")
    parser.add_argument("--table10", action="store_true",
                        help="run the soft-label-quality correlation study")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):  # pragma: no cover
    args = _parse_args(argv)
    if args.table10:
        if not args.results_dir:
            raise SystemExit("--table10 requires --results-dir")
        from .models import all_model_specs

        study = soft_label_quality_correlation(
            args.backbone,
            args.data_root,
            args.results_dir,
            source_models=[s.name for s in all_model_specs()],
            config=ProbeConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lambda_weight=args.lambda_weight,
                temperature=args.temperature,
            ),
            limit_train=args.limit_train,
            limit_eval=args.limit_eval,
            device=args.device,
        )
        print("Table 10 (PEA between source ID LCA and probe OOD Top-1)")
        for kind, values in study["PEA"].items():
            print(" ", kind, {k: round(v, 4) for k, v in values.items()})
        return study

    matrix = None
    tree_prefix = "WordNet"
    if args.source_model:
        from .experiments_latent import class_features_for_model

        class_features = class_features_for_model(
            args.source_model, args.data_root, device=args.device
        )
        matrix = latent_lca_distance_matrix(class_features)
        tree_prefix = "latent"
    elif args.hierarchy not in (None, "WordNet"):
        matrix = np.load(args.hierarchy)
        tree_prefix = "latent"
    results = soft_label_linear_probe(
        args.backbone,
        args.data_root,
        hierarchy_matrix=matrix,
        tree_prefix=tree_prefix,
        config=ProbeConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lambda_weight=args.lambda_weight,
            temperature=args.temperature,
        ),
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        device=args.device,
    )
    print(format_soft_label_table(results))
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
