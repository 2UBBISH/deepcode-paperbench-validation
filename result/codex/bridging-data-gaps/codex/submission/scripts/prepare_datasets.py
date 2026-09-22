#!/usr/bin/env python3
"""Check (and explain how to obtain) the datasets of Section 5.2.

The datasets are third-party artefacts and are therefore *not* redistributed
here.  This script validates the directory layout expected by the experiment
scripts and prints where each dataset comes from.

Expected layout (any folder name works, pass it with --data-root):

    <data-root>/ffhq256/                 # source: FFHQ 256x256 images
    <data-root>/lsun_church/             # source: LSUN Church outdoor images
    <data-root>/few_shot/babies/         # 10 target images
    <data-root>/few_shot/sunglasses/
    <data-root>/few_shot/raphael/
    <data-root>/few_shot/sketches/
    <data-root>/few_shot/amedeo/
    <data-root>/few_shot/haunted_houses/
    <data-root>/few_shot/landscape_drawings/
    <data-root>/fid/sunglasses/          # (optional) larger real sets for FID
    <data-root>/fid/babies/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpms_ant.data import DATASET_REGISTRY, list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {
        "ffhq": os.path.join(args.data_root, "ffhq256"),
        "lsun_church": os.path.join(args.data_root, "lsun_church"),
    }
    for name, spec in DATASET_REGISTRY.items():
        if spec.kind == "target":
            expected[name] = os.path.join(args.data_root, "few_shot", name)

    print("dataset registry (see paper Section 5.2)\n")
    for name, spec in DATASET_REGISTRY.items():
        path = expected[name]
        images = list_images(path) if os.path.isdir(path) else []
        status = f"{len(images):6d} images" if images else "MISSING"
        print(f"  {name:20s} [{spec.kind:6s}] {path:45s} {status}")
        print(f"      source: {spec.origin}")
    print(
        "\nSources (third party, not redistributed):\n"
        "  * FFHQ          https://github.com/NVlabs/ffhq-dataset\n"
        "  * LSUN          https://www.yf.io/p/lsun  (church_outdoor)\n"
        "  * Babies/Sunglasses/Haunted houses/Landscape drawings: the 10-shot\n"
        "    datasets released with CDC (Ojha et al., 2021),\n"
        "    https://github.com/utkarshojha/few-shot-gan-adaptation\n"
        "  * Raphael / Amedeo Modigliani: released with DCL (Zhao et al., 2022)\n"
        "  * the released guided-diffusion checkpoints are downloaded\n"
        "    automatically by dpms_ant.backbones (256x256_diffusion_uncond.pt,\n"
        "    256x256_classifier.pt).\n"
    )


if __name__ == "__main__":
    main()
