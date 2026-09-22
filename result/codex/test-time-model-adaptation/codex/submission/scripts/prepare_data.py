#!/usr/bin/env python3
"""Download the benchmarks used by the paper.

Usage::

    python scripts/prepare_data.py --data-root data --only all
    python scripts/prepare_data.py --data-root data --only imagenet_c

ImageNet-C (~65 GB, 5 tarballs) comes from Zenodo, ImageNet-R/V2/Sketch from their
official hosts and the clean ImageNet-1K validation set (used for the source statistics)
comes from HuggingFace, as recommended by the paper's addendum.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foa.data import download as DL  # noqa: E402
from foa.data.datasets import IMAGENET_C_CORRUPTIONS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--only",
        default="all",
        choices=["all", "imagenet_c", "imagenet_r", "imagenet_v2", "imagenet_sketch", "imagenet"],
    )
    parser.add_argument(
        "--hf-imagenet-c",
        action="store_true",
        help="materialise ImageNet-C from the HuggingFace webdataset mirror instead of "
        "downloading the 65 GB of official Zenodo tarballs (one-off, then the layout is "
        "identical)",
    )
    parser.add_argument("--corruptions", default=None, help="comma separated, for --hf-imagenet-c")
    parser.add_argument("--severities", default="5", help="comma separated, for --hf-imagenet-c")
    parser.add_argument("--limit", type=int, default=None, help="max images per corruption (debug)")
    args = parser.parse_args()
    root = args.data_root
    if args.hf_imagenet_c:
        from foa.data.hf import materialize_imagenet_c_from_hf

        corruptions = (
            args.corruptions.split(",") if args.corruptions else list(IMAGENET_C_CORRUPTIONS)
        )
        severities = [int(s) for s in args.severities.split(",")]
        materialize_imagenet_c_from_hf(
            os.path.join(root, "imagenet-c"),
            corruptions=corruptions,
            severities=severities,
            limit=args.limit,
        )
        return
    if args.only in ("all", "imagenet_c"):
        DL.download_imagenet_c(os.path.join(root, "imagenet-c"))
    if args.only in ("all", "imagenet_r"):
        DL.download_imagenet_r(os.path.join(root, "imagenet-r"))
    if args.only in ("all", "imagenet_v2"):
        DL.download_imagenet_v2(os.path.join(root, "imagenet-v2"))
    if args.only in ("all", "imagenet_sketch"):
        DL.download_imagenet_sketch(os.path.join(root, "imagenet-sketch"))
    if args.only in ("all", "imagenet"):
        DL.download_imagenet1k_val()


if __name__ == "__main__":
    main()
