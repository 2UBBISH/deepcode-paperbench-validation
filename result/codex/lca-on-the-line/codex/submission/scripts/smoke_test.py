#!/usr/bin/env python3
"""Small end-to-end smoke test of the evaluation + analysis pipeline.

It builds a synthetic ImageNet-shaped dataset (a handful of random images with
random ImageNet labels), runs two small torchvision models over it with the
real ``run_benchmark`` code path, and prints the resulting tables/figures.

    python scripts/smoke_test.py            # resnet18 + resnet34
    python scripts/smoke_test.py --clip     # additionally exercise CLIP

Nothing here needs a GPU and it finishes in under a minute.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.analysis import (  # noqa: E402
    compute_table1,
    compute_table2,
    figure1,
    figure5,
    format_table2,
    load_metrics,
)
from lca_on_the_line.data import IndexedImageDataset  # noqa: E402
from lca_on_the_line.evaluate import run_benchmark  # noqa: E402

DATASETS = ["imagenet", "imagenet_v2", "imagenet_s", "imagenet_r", "imagenet_a",
            "objectnet"]


def make_synthetic_dataset(tmp_dir: str, n_images: int = 24, seed: int = 0):
    """Random images with *pseudo labels* so the metrics are non-degenerate.

    The labels come from an off-the-shelf ResNet-18 (flipped for 30% of the
    images), which gives the different models genuinely different accuracies and
    lets the correlation code produce finite numbers.
    """
    from PIL import Image
    import torch

    rng = np.random.RandomState(seed)
    arrays = [(rng.rand(300, 300, 3) * 255).astype("uint8") for _ in range(n_images)]

    import torchvision.models as tvm

    weights = tvm.ResNet18_Weights.IMAGENET1K_V1
    model = tvm.resnet18(weights=weights).eval()
    transform = weights.transforms()
    stack = torch.stack([transform(Image.fromarray(a).convert("RGB")) for a in arrays])
    with torch.no_grad():
        pseudo = model(stack).argmax(dim=1).numpy()
    flip = rng.rand(n_images) < 0.3
    pseudo[flip] = rng.randint(0, 1000, size=int(flip.sum()))

    samples = []
    for index, array in enumerate(arrays):
        path = os.path.join(tmp_dir, "img_%03d.png" % index)
        Image.fromarray(array).save(path)
        samples.append((path, int(pseudo[index])))
    return IndexedImageDataset(samples, name="synthetic")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["resnet18", "resnet34"])
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--clip", action="store_true",
                        help="also exercise the CLIP zero-shot path")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-templates", type=int, default=1,
                        help="prompt templates to use for the zero-shot path")
    args = parser.parse_args(argv)

    tmp_data = tempfile.mkdtemp(prefix="lca-smoke-data-")
    out_dir = args.out_dir or tempfile.mkdtemp(prefix="lca-smoke-out-")
    dataset = make_synthetic_dataset(tmp_data, n_images=args.limit)
    overrides = {name: dataset for name in DATASETS}

    models = list(args.models)
    if args.clip:
        models.append("CLIP_RN50")
    templates = None
    if args.n_templates:
        from lca_on_the_line.prompt_engineering import IMAGENET_TEMPLATES

        templates = IMAGENET_TEMPLATES[: args.n_templates]

    path = run_benchmark(
        model_names=models,
        data_root=tmp_data,
        out_dir=out_dir,
        datasets=DATASETS,
        limit=args.limit,
        batch_size=4,
        device="cpu",
        loader_overrides=overrides,
        templates=templates,
    )
    print("\nmetrics written to", path)

    df = load_metrics(out_dir)
    print("\n== Table 1 (head) ==\n", compute_table1(df).head())
    table2 = compute_table2(df)
    print("\n== Table 2 ==\n", format_table2(table2))
    figure5(df, os.path.join(out_dir, "figure5.png"))
    figure1(df, os.path.join(out_dir, "figure1.png"))
    print("\nfigures written to", out_dir)


if __name__ == "__main__":
    main()
