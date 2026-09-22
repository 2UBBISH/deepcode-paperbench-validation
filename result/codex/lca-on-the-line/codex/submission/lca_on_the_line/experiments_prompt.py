"""Section 4.3.3 / Table 14: taxonomy alignment through prompt engineering.

For a CLIP-style zero-shot model we compare four prompt protocols:

===============  =====================================================
``baseline``     ``<dalmatian>``
``stack_parent`` ``<dalmatian, dog, animal>``
``taxonomy_parent``  ``<dalmatian, which is a type of a dog, which is
                 a type of an animal>``
``shuffle_parent``   the ``is-a`` phrasing with *random* ancestors
===============  =====================================================

Only ``taxonomy_parent`` tells the model about the correct hierarchical
relationships, which is the effect the paper reports.
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .data import OOD_DATASETS, load_dataset_by_name
from .hierarchy import WordNetHierarchy, load_wordnet_hierarchy
from .models import all_model_specs, build_classifier
from .prompt_engineering import TAXONOMY_PROMPT_MODES, build_taxonomy_prompts


def evaluate_prompt_modes(
    model_name: str,
    data_root: str,
    datasets: Sequence[str] = ("imagenet",) + OOD_DATASETS,
    modes: Sequence[str] = TAXONOMY_PROMPT_MODES,
    depth: int = 2,
    template: str = "a photo of a {}.",
    limit: Optional[int] = None,
    batch_size: int = 64,
    device: str = "cpu",
    hierarchy: Optional[WordNetHierarchy] = None,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return ``{dataset: {mode: {top1, ce}}}``."""
    hierarchy = hierarchy or load_wordnet_hierarchy()
    spec = next(s for s in all_model_specs() if s.name == model_name)
    classifier = build_classifier(spec, device=device, batch_size=batch_size)

    text_features: Dict[str, torch.Tensor] = {}
    for mode in modes:
        # one prompt per class (the per-template ensemble is handled separately)
        per_class = [
            [prompt]
            for prompt in build_taxonomy_prompts(
                hierarchy, mode=mode, template=template, depth=depth, seed=seed
            )
        ]
        text_features[mode] = classifier.encode_prompts(per_class)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset_name in datasets:
        dataset = load_dataset_by_name(dataset_name, data_root)
        n = len(dataset) if limit is None else min(limit, len(dataset))
        images = [dataset[i][0] for i in range(n)]
        targets = np.array([dataset[i][1] for i in range(n)])
        results[dataset_name] = {}
        for mode in modes:
            logits = classifier.logits_with_text(images, text_features[mode])
            preds = logits.argmax(axis=1)
            probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
            ce = float(
                -np.log(np.clip(probs[np.arange(n), targets], 1e-12, 1.0)).mean()
            )
            results[dataset_name][mode] = {
                "top1": float((preds == targets).mean()),
                "ce": ce,
            }
    return results


def format_table14(results: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = ["%-16s%-14s%-10s%-10s" % ("dataset", "mode", "top1", "CE")]
    for dataset, modes in results.items():
        for mode, metrics in modes.items():
            lines.append(
                "%-16s%-14s%-10.4f%-10.4f"
                % (dataset, mode, metrics["top1"], metrics["ce"])
            )
    return "\n".join(lines)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Taxonomy prompt engineering")
    parser.add_argument("--model", default="CLIP_ViT-B_32")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):  # pragma: no cover
    args = _parse_args(argv)
    results = evaluate_prompt_modes(
        args.model,
        args.data_root,
        datasets=args.datasets or (("imagenet",) + OOD_DATASETS),
        depth=args.depth,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(format_table14(results))
    return results


if __name__ == "__main__":  # pragma: no cover
    main()
