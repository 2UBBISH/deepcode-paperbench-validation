#!/usr/bin/env python3
"""Compare the mask generator size with Table 4 of the paper.

Table 4 reports 26,499 (5-layer CNN, ResNet-18/50) and 102,339 (6-layer CNN,
ViT-B/32) extra parameters, i.e. 17.60% / 23.13% of the reprogramming
parameters (the shared pattern ``delta``, ``3 x H x W``) and 0.23% / 0.10% /
0.12% of the pre-trained model parameters.

The per-layer widths are only shown in Figures 8/9 (not part of the provided
markdown), so this script also reports the closest configurations found by a
search over "nice" channel widths.

    python scripts/param_stats.py [--search]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smm.mask_generator import (  # noqa: E402
    DEFAULT_5_LAYER_CHANNELS,
    DEFAULT_6_LAYER_CHANNELS,
    MaskNet,
)

TABLE4 = {
    "resnet18": {
        "image_size": (224, 224),
        "num_layers": 5,
        "extra_params": 26499,
        "pattern_params": 3 * 224 * 224,
        "ratio_to_reprogramming": 0.1760,
        "model_params": 11689512,
        "ratio_to_model": 0.0023,
    },
    "resnet50": {
        "image_size": (224, 224),
        "num_layers": 5,
        "extra_params": 26499,
        "pattern_params": 3 * 224 * 224,
        "ratio_to_reprogramming": 0.1760,
        "model_params": 25557032,
        "ratio_to_model": 0.0010,
    },
    "vit_b32": {
        "image_size": (384, 384),
        "num_layers": 6,
        "extra_params": 102339,
        "pattern_params": 3 * 384 * 384,
        "ratio_to_reprogramming": 0.2313,
        "model_params": 88224232,
        "ratio_to_model": 0.0012,
    },
}


def count(hidden, in_channels: int = 3, out_channels: int = 3) -> int:
    total, prev = 0, in_channels
    for c in list(hidden) + [out_channels]:
        total += 3 * 3 * prev * c + c
        prev = c
    return total


def search(target: int, num_layers: int, limit: int = 5):
    values = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    hits = []
    for hidden in itertools.product(values, repeat=num_layers - 1):
        if count(hidden) == target:
            hits.append(hidden)
    return hits[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", action="store_true", help="search for exact matches")
    args = parser.parse_args()

    for name, spec in TABLE4.items():
        hidden = DEFAULT_6_LAYER_CHANNELS if spec["num_layers"] == 6 else DEFAULT_5_LAYER_CHANNELS
        net = MaskNet(spec["image_size"], hidden_channels=hidden,
                      num_pool_layers=3)
        params = sum(p.numel() for p in net.parameters())
        print(f"[{name}] mask generator channels={tuple(hidden)}")
        print(f"    parameters          : {params}")
        print(f"    Table 4             : {spec['extra_params']} "
              f"({100.0 * (params - spec['extra_params']) / spec['extra_params']:+.2f}%)")
        print(f"    delta parameters    : {spec['pattern_params']}")
        print(f"    params / delta      : {params / spec['pattern_params']:.4f} "
              f"(Table 4: {spec['ratio_to_reprogramming']:.4f})")
        print(f"    params / model      : {params / spec['model_params']:.5f} "
              f"(Table 4: {spec['ratio_to_model']:.5f})")
        if args.search:
            print(f"    exact matches found : {search(spec['extra_params'], spec['num_layers'])}")
        print()

    # Patch-wise interpolation statistics (Appendix A.3 / Table 5).
    x = torch.randn(1, 3, 224, 224)
    net = MaskNet((224, 224))
    with torch.no_grad():
        low = net.generator(x)
        up = net.interpolation(low)
    print("patch-wise interpolation:", tuple(low.shape), "->", tuple(up.shape),
          f"(patch size {net.patch_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
