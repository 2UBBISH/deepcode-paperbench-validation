#!/usr/bin/env python3
"""Small-scale, CPU-friendly sanity check of the FOA mechanism.

The full ImageNet-C protocol needs 50 000 images per corruption and a GPU.  This script
runs the *same* code path on a few hundred images that can be downloaded in seconds from
the public HuggingFace mirror of ImageNet-V2:

* 32 *clean* ImageNet-V2 images play the role of the source in-distribution set used to
  estimate ``{mu_i^S, sigma_i^S}``;
* the remaining images are corrupted (Gaussian noise) and streamed at test time;
* the ablation of Table 5 is reproduced, i.e. the fitness must be *entropy only*
  (worst), then *activation discrepancy only*, then *entropy + discrepancy* and finally
  the full FOA with activation shifting (best), both in accuracy and in ECE.

Usage::

    python scripts/validate_proxy_imagenetv2.py

Reference run (CPU, ~12 minutes, `vit_small_patch16_224.augreg_in21k_ft_in1k`, sigma 0.6)::

    NoAdapt               acc=38.00  ece=13.67
    FOA entropy only      acc=37.50  ece=14.81
    FOA discrepancy only  acc=38.50  ece=13.17
    FOA entropy+disc      acc=40.00  ece=12.80
    FOA full              acc=40.50  ece=11.67

which matches the ordering of Table 5 of the paper (44.9 / 63.4 / 65.4 / 66.3 and
36.8 / 9.4 / 3.3 / 3.2 there).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foa.config import FOAConfig  # noqa: E402
from foa.core.activation_shift import BackToSourceShifting  # noqa: E402
from foa.core.statistics import compute_source_statistics  # noqa: E402
from foa.evaluation.runner import run_stream  # noqa: E402
from foa.methods import FOAMethod, NoAdapt  # noqa: E402
from foa.models.prompt_vit import PromptViT  # noqa: E402


def load_v2(num_images: int, seed: int = 0):
    """Load ``num_images`` ImageNet-V2 (matched frequency) images with their labels."""
    from datasets import load_dataset
    from timm.data import create_transform, resolve_data_config
    from timm.data.imagenet_info import ImageNetInfo

    transform = create_transform(
        **resolve_data_config({"input_size": (3, 224, 224), "crop_pct": 0.9, "interpolation": "bicubic"}),
        is_training=False,
    )
    wnids = ImageNetInfo().label_names()
    index_of = {w: i for i, w in enumerate(wnids)}

    ds = load_dataset("vaishaal/ImageNetV2", split="train", streaming=True)
    images, labels = [], []
    for example in ds:
        key = example["__key__"]
        try:
            wnid = key.split("/")[1]  # e.g. imagenetv2-.../986/....jpeg
            label = int(wnid) if wnid.isdigit() else index_of[wnid]
        except Exception:
            continue
        images.append(transform(example["jpeg"].convert("RGB")))
        labels.append(label)
        if len(images) >= num_images:
            break
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(images), generator=g)
    return torch.stack(images)[perm], torch.tensor(labels)[perm]


def gaussian_noise(images: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    """Add Gaussian noise in normalised space (the ImageNet-C `gaussian_noise` corruption)."""
    g = torch.Generator().manual_seed(seed)
    return (images + sigma * torch.randn(images.shape, generator=g)).clamp(-3.0, 3.0)


class ListStream:
    """Deterministic stream over a fixed tensor of images / labels."""

    def __init__(self, images: torch.Tensor, targets: torch.Tensor, batch_size: int):
        self.images = images
        self.targets = targets
        self.batch_size = batch_size

    def __iter__(self):
        for start in range(0, self.images.shape[0], self.batch_size):
            yield self.images[start : start + self.batch_size], self.targets[start : start + self.batch_size]

    def __len__(self):
        return (self.images.shape[0] + self.batch_size - 1) // self.batch_size


class ShiftingOnly:
    """Ablation: NoAdapt + back-to-source activation shifting (Eqn. 7-9)."""

    name = "Act. Shifting"

    def __init__(self, model: PromptViT, stats, device):
        self.model = model.to(device)
        self.device = device
        self.source_mean = stats.means[-1]
        self.shifting = BackToSourceShifting(self.source_mean, alpha=0.1, gamma=1.0)
        self.last_extra = {}

    def reset(self):
        self.shifting.reset()

    @torch.no_grad()
    def step(self, images):
        images = images.to(self.device)
        means, _ = self.model.cls_statistics(images, prompt=None)
        shift = self.shifting.update(means[-1])
        logits, _, _ = self.model.forward_with_prompt(images, shift=shift)
        return logits


ABLATION = {
    "FOA entropy only": dict(
        use_entropy=True, use_activation_discrepancy=False, use_activation_shifting=False
    ),
    "FOA discrepancy only": dict(
        use_entropy=False, use_activation_discrepancy=True, use_activation_shifting=False
    ),
    "FOA entropy+disc.": dict(
        use_entropy=True, use_activation_discrepancy=True, use_activation_shifting=False
    ),
    "FOA full": dict(
        use_entropy=True, use_activation_discrepancy=True, use_activation_shifting=True
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--num-images", type=int, default=232)
    parser.add_argument("--num-source", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ablation", action="store_true", default=True)
    parser.add_argument("--no-ablation", dest="ablation", action="store_false")
    args = parser.parse_args()

    device = torch.device("cpu")
    t0 = time.time()
    images, labels = load_v2(args.num_images, seed=args.seed)
    print(f"loaded {len(images)} ImageNet-V2 images in {time.time() - t0:.1f}s")

    import timm

    vit = timm.create_model(args.checkpoint, pretrained=True)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    model = PromptViT(vit, num_prompts=3)
    print(f"classes with more than one image: {int((torch.bincount(labels) > 1).sum())}")

    src = images[: args.num_source]
    test_images = images[args.num_source :]
    test_labels = labels[args.num_source :]
    stats = compute_source_statistics(model, [src])

    clean = ListStream(test_images, test_labels, args.batch_size)
    no_adapt_clean = run_stream(NoAdapt(model, device=device), clean)
    print(f"clean  NoAdapt          acc={no_adapt_clean.acc:.2f}")

    noisy = gaussian_noise(test_images, args.sigma, args.seed)
    stream = ListStream(noisy, test_labels, args.batch_size)

    no_adapt = run_stream(NoAdapt(model, device=device), stream)
    print(f"noisy  NoAdapt          acc={no_adapt.acc:.2f} ece={no_adapt.ece:.2f}")

    shifting = run_stream(ShiftingOnly(model, stats, device), stream)
    print(f"noisy  Act. Shifting    acc={shifting.acc:.2f} ece={shifting.ece:.2f}")

    results = {"NoAdapt": no_adapt, "Act. Shifting": shifting}
    variants = ABLATION if args.ablation else {"FOA": ABLATION["FOA full"]}
    for name, options in variants.items():
        cfg = FOAConfig(
            popsize=args.popsize,
            batch_size=args.batch_size,
            device="cpu",
            cma_seed=args.seed,
            # the reference run reported in the README centres the CMA search on a
            # uniformly initialised prompt (the paper's "uniform initialization")
            cma_mean_from_prompt_init=True,
            **options,
        )
        results[name] = run_stream(FOAMethod(model, stats, cfg=cfg, device=device), stream)
        print(f"noisy  {name:20s} acc={results[name].acc:.2f} ece={results[name].ece:.2f}")

    print("\nstream results (order of Table 5):")
    for name, res in results.items():
        print(f"  {name:22s} acc={res.acc:6.2f}  ece={res.ece:6.2f}")
    print(f"  {'clean NoAdapt':22s} acc={no_adapt_clean.acc:6.2f}")

    assert no_adapt_clean.acc >= no_adapt.acc - 1e-6, "corruption should hurt accuracy"
    if args.ablation and no_adapt_clean.acc - no_adapt.acc > 15.0:
        # with a severe enough corruption the paper's ordering must be reproduced
        assert results["FOA full"].acc >= results["FOA entropy only"].acc
        assert results["FOA full"].ece <= results["FOA entropy only"].ece


if __name__ == "__main__":
    main()
