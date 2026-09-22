#!/usr/bin/env python3
"""Validate the pipeline against numbers reported in the paper.

ImageNet-v2 (MatchedFrequency) is the cheapest solid check: it shares the 1000
ImageNet classes, so the same hierarchy and the same class-index mapping are
exercised, and the paper reports both Top-1 and LCA for several models in
Table 1 and Table 8.

    python scripts/validate_against_paper.py --imagenet-v2 /path/to/imagenetv2 \
        --models resnet18 resnet50 CLIP_RN50

Use ``--per-class N`` for a fast, *unbiased* subset (2 images per class on the
MatchedFrequency split reproduces the paper to within sampling noise); plain
``--limit`` instead takes the first N files, which only covers a few classes and
is therefore optimistic.

Reference values (paper):

================ ============= =============
model            ImgN-v2 Top-1 ImgN-v2 LCA
================ ============= =============
ResNet-18        0.573         6.918
ResNet-50        0.610         6.863
CLIP_RN50        0.511         6.538
CLIP_RN50x4      0.573         6.383
================ ============= =============
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.data import load_imagenet_v2  # noqa: E402
from lca_on_the_line.evaluate import (  # noqa: E402
    collect_logits,
    dataset_targets,
    metrics_for_logits,
    stratified_subset_indices,
)
from lca_on_the_line.hierarchy import load_wordnet_hierarchy  # noqa: E402
from lca_on_the_line.models import all_model_specs, build_classifier  # noqa: E402

REFERENCE = {
    "resnet18": {"top1": 0.573, "lca": 6.918},
    "resnet50": {"top1": 0.610, "lca": 6.863},
    "CLIP_RN50": {"top1": 0.511, "lca": 6.538},
    "CLIP_RN50x4": {"top1": 0.573, "lca": 6.383},
}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-v2", required=True,
                        help="directory holding imagenetv2-*-format-val-*")
    parser.add_argument("--models", nargs="*",
                        default=["resnet18", "resnet50", "CLIP_RN50"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None,
                        help="evaluate only N images per class (stratified)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    dataset = load_imagenet_v2(args.imagenet_v2)
    targets = dataset_targets(dataset, args.limit)
    hierarchy = load_wordnet_hierarchy()
    specs = {s.name: s for s in all_model_specs()}

    indices = None
    if args.per_class:
        indices = stratified_subset_indices(targets, args.per_class)
        targets = targets[indices]
        print("evaluating %d images (%d per class)"
              % (len(indices), args.per_class))

    print("%-14s %-12s %-12s %-12s %-12s" % (
        "model", "top1", "paper top1", "LCA", "paper LCA"))
    for name in args.models:
        classifier = build_classifier(
            specs[name], device=args.device, batch_size=args.batch_size
        )
        logits = collect_logits(
            classifier, dataset, batch_size=args.batch_size, limit=args.limit,
            indices=indices,
        )
        metrics = metrics_for_logits(
            logits.astype(np.float64), targets, hierarchy, compute_elca=False
        )
        ref = REFERENCE.get(name, {})
        print("%-14s %-12.4f %-12s %-12.4f %-12s" % (
            name,
            metrics["top1"],
            ("%.3f" % ref["top1"]) if "top1" in ref else "-",
            metrics["lca"],
            ("%.3f" % ref["lca"]) if "lca" in ref else "-",
        ))


if __name__ == "__main__":
    main()
