#!/usr/bin/env python3
"""Compute Intra-LPIPS / FID for a folder of generated images.

python scripts/evaluate.py --generated outputs/ddpm_ffhq_sunglasses/samples.pt \
    --target-root data/few_shot/sunglasses --real-root data/sunglasses_full
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dpms_ant.data import ImageTensorDataset, list_images
from dpms_ant.metrics import FIDMetric, LPIPSMetric, compute_fid, intra_lpips
from dpms_ant.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", required=True, help=".pt tensor or image folder")
    parser.add_argument("--target-root", required=True, help="the 10-shot target images")
    parser.add_argument("--real-root", default=None, help="larger real set, for FID")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-fid-images", type=int, default=2500)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_images(path: str, image_size: int) -> torch.Tensor:
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    dataset = ImageTensorDataset(list_images(path), image_size=image_size)
    return torch.stack([dataset[index] for index in range(len(dataset))])


def main() -> None:
    args = parse_args()
    device = str(get_device(args.device))
    generated = load_images(args.generated, args.image_size)
    target = load_images(args.target_root, args.image_size)
    print(f"Intra-LPIPS = {intra_lpips(generated, target, metric=LPIPSMetric(device=device), device=device):.4f}")
    if args.real_root:
        real = load_images(args.real_root, args.image_size)[: args.num_fid_images]
        print(f"FID         = {compute_fid(generated, real, device=device, metric=FIDMetric(device=device)):.4f}")


if __name__ == "__main__":
    main()

