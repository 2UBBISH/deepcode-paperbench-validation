#!/usr/bin/env python3
"""Run every few-shot task of the paper and print Tables 1 and 2.

The ten tasks of Section 5.2 are

    DDPM: FFHQ -> Babies / Sunglasses / Raphael's paintings
          LSUN Church -> Haunted houses / Landscape drawings
    LDM:  the same five target domains

each with the learning rate, ``C`` (adaptor bottleneck), ``gamma``, ``omega``,
``J`` and iteration count given in the supplementary material
(:mod:`dpms_ant.configs`).

Example
-------
python scripts/run_all_tasks.py \
    --data-root data --output-dir outputs \
    --classifier-root outputs/classifiers

The GAN-based and DDPM-PA baselines of Tables 1-2 come from their own
codebases (TGAN, TGAN+ADA, EWC, CDC, DCL, DDPM-PA) and are therefore not
re-run here; this script produces the two rows that this repository
implements (DDPM-ANT and LDM-ANT).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpms_ant.configs import TASK_CONFIGS
from dpms_ant.image_experiments import ExperimentArgs, run_task

TARGETS = ["babies", "sunglasses", "raphael", "haunted_houses", "landscape_drawings"]
TARGET_LABELS = {
    "babies": "FFHQ -> Babies",
    "sunglasses": "FFHQ -> Sunglasses",
    "raphael": "FFHQ -> Raphael's paintings",
    "haunted_houses": "LSUN Church -> Haunted houses",
    "landscape_drawings": "LSUN Church -> Landscape drawings",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--classifier-root", default=None,
                        help="folder with pre-trained <target>_classifier.pt files")
    parser.add_argument("--baseline", default="ant", choices=["ant", "vanilla", "full"])
    parser.add_argument("--backbones", default="ddpm,ldm",
                        help="comma separated subset of {ddpm,ldm}")
    parser.add_argument("--fid-root", default=None)
    parser.add_argument("--num-generated", type=int, default=1000)
    parser.add_argument("--num-fid-images", type=int, default=2500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    results = {}
    for name, task in TASK_CONFIGS.items():
        if task.backbone not in backbones:
            continue
        target_root = os.path.join(args.data_root, "few_shot", task.target)
        source_root = os.path.join(args.data_root, "ffhq256" if task.source == "ffhq" else "lsun_church")
        classifier_checkpoint = None
        if args.classifier_root:
            candidate = os.path.join(args.classifier_root, f"{task.target}_classifier.pt")
            classifier_checkpoint = candidate if os.path.exists(candidate) else None
        fid_root = os.path.join(args.fid_root, task.target) if args.fid_root else None
        experiment = ExperimentArgs(
            task=name,
            target_root=target_root,
            source_root=source_root,
            classifier_checkpoint=classifier_checkpoint,
            fid_real_root=fid_root,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            num_generated=args.num_generated,
            num_fid_images=args.num_fid_images,
            baseline=args.baseline,
        )
        print(f"\n=== {name} ({task.backbone}, {task.iterations} iterations, lr {task.learning_rate}) ===")
        if args.dry_run:
            continue
        if not os.path.isdir(target_root):
            print(f"[skip] missing target dataset at {target_root}")
            continue
        results[name] = run_task(experiment)

    if not results:
        return
    with open(os.path.join(args.output_dir, "all_tasks.json"), "w") as handle:
        json.dump(results, handle, indent=2)

    print("\nTable 1 (Intra-LPIPS, higher is better)")
    for backbone, label in (("ddpm", "DDPM-ANT (Ours)"), ("ldm", "LDM-ANT (Ours)")):
        row = {"Methods": label}
        for target in TARGETS:
            key = f"{backbone}_" + (
                "ffhq_" + target if target in ("babies", "sunglasses", "raphael") else "lsun_" + target
            )
            value = results.get(key, {}).get("intra_lpips")
            row[TARGET_LABELS[target]] = "n/a" if value is None else f"{value:.3f}"
        print(json.dumps(row, indent=2))

    if any("fid" in result for result in results.values()):
        print("\nTable 2 (FID, lower is better)")
        for backbone, label in (("ddpm", "DDPM-ANT (Ours)"), ("ldm", "LDM-ANT (Ours)")):
            for target in ("babies", "sunglasses"):
                key = f"{backbone}_ffhq_{target}"
                value = results.get(key, {}).get("fid")
                if value is not None:
                    print(f"  {label:16s} {target:12s} FID = {value:.2f}")


if __name__ == "__main__":
    main()
