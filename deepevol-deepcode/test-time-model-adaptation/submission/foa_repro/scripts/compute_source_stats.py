#!/usr/bin/env python
"""Offline computation of the source (ImageNet-1K) CLS activation statistics for FOA.

Paper reference (§3.1 "Statistics calculation"):
    "Before TTA, we first collect a small set of source in-distribution samples
     D_S = {x_q}_{q=1..Q} and feed them into the model to obtain the corresponding CLS
     tokens {e_i^0}_{i=1..N}.  Then, we calculate the mean and standard deviations of
     CLS tokens {e_i^0}_{i=1..N} over all samples in D_S to obtain source in-distribution
     statistics {mu_i^S, sigma_i^S}_{i=0..N}. Note that we only need a small number of
     in-distribution samples **without labels** for calculation, e.g., 32 samples are
     sufficient for the ImageNet dataset."

Appendix B.2:
    "The source in-distribution statistics {mu_i^S, sigma_i^S}_{i=0..N} are calculated
     **without using the newly inserted prompt**. ... We use the validation set of
     ImageNet-1K to estimate source training statistics."

This script:
  * builds the frozen ViT-Base (timm augreg checkpoint),
  * forwards Q unlabeled ImageNet-1K validation images through it WITHOUT prompt
    injection (``prompt=None`` / ``num_prompts=0``),
  * accumulates per-layer (i = 0..N) mean/std of the [CLS] tokens,
  * persists the resulting bank as a ``.pt`` checkpoint consumed by
    ``src/method/fitness.py`` (activation discrepancy) and
    ``src/method/activation_shifting.py`` (mu_N^S).

Usage
-----
    python scripts/compute_source_stats.py --config configs/foa_imagenetc.yaml
    python scripts/compute_source_stats.py --num-samples 32 --dataset imagenet-1k
    python scripts/compute_source_stats.py --images-dir /path/to/imagenet/val \\
        --num-samples 32 --output checkpoints/source_stats_vit_base.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator, List, Optional, Sequence

import torch

# --------------------------------------------------------------------------------------
# Make ``import src...`` work when the script is run directly (python scripts/xxx.py).
# --------------------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.method.source_stats import (  # noqa: E402
    DEFAULT_NUM_SOURCE_SAMPLES,
    SourceStats,
    compute_source_statistics,
    compute_source_statistics_from_loader,
    load_source_stats,
    save_source_stats,
)
from src.models.vit_loader import FOA_CHECKPOINT_URL, build_vit  # noqa: E402
from src.utils.config import FOA_DEFAULTS, load_config  # noqa: E402


# --------------------------------------------------------------------------------------
# Fallback image sources (used when src/data/datasets.py is unavailable or when the user
# points the script at a plain directory of images).
# --------------------------------------------------------------------------------------
def _build_imagefolder_iter(
    root: str,
    num_samples: int,
    batch_size: int = 16,
    image_size: int = 224,
) -> Iterator[torch.Tensor]:
    """Yield normalized 224x224 batches from an ``ImageFolder``-style directory.

    Used for the ``--images-dir`` path (e.g. a local copy of ImageNet-1K ``val/``).
    Labels are discarded: the source statistics require no annotations.
    """
    from torchvision import datasets as tv_datasets
    from torchvision import transforms

    # Standard ViT preprocessing from timm / Appendix B.1:
    #   resize to 256 (bicubic) -> center crop 224 -> ImageNet normalization.
    tf = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    dataset = tv_datasets.ImageFolder(root, transform=tf)
    # Deterministic subset selection: first `num_samples` entries of the ordered dataset.
    limit = min(int(num_samples), len(dataset))
    subset = torch.utils.data.Subset(dataset, list(range(limit)))
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    for images, _labels in loader:
        yield images


def build_source_image_iter(
    cfg,
    num_samples: int,
    seed: int = 0,
) -> Iterator[torch.Tensor]:
    """Return an iterator of source (ID) image batches.

    Resolution order:
      1. explicit ``--images-dir`` / ``cfg.source_stats.images_dir`` (ImageFolder),
      2. the project dataset loader (``src/data/datasets.py``, HuggingFace ImageNet-1K),
      3. a clear error asking the user to provide one of the two.
    """
    images_dir = cfg.get("images_dir", None)
    if images_dir is None:
        images_dir = getattr(getattr(cfg, "source_stats", {}), "images_dir", None)

    if images_dir:
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"Source image directory not found: {images_dir}"
            )
        return _build_imagefolder_iter(
            images_dir,
            num_samples=num_samples,
            batch_size=int(cfg.data.batch_size),
            image_size=224,
        )

    # Preferred path: the project's dataset loader.
    try:
        from src.data import datasets as foa_datasets  # type: ignore

        if hasattr(foa_datasets, "build_source_stream"):
            return foa_datasets.build_source_stream(
                num_samples=num_samples,
                batch_size=int(cfg.data.batch_size),
                root=cfg.data.get("root", "./data"),
                seed=seed,
                num_workers=int(cfg.data.get("num_workers", 4)),
            )
        if hasattr(foa_datasets, "build_dataset_loader"):
            loader = foa_datasets.build_dataset_loader(
                dataset="imagenet-1k",
                split="validation",
                batch_size=int(cfg.data.batch_size),
                root=cfg.data.get("root", "./data"),
                shuffle=False,
                num_workers=int(cfg.data.get("num_workers", 4)),
            )
            return _take_batches(loader, num_samples)
    except Exception as exc:  # pragma: no cover - depends on optional HF datasets pkg
        raise RuntimeError(
            "Could not build the ImageNet-1K source stream automatically "
            f"({type(exc).__name__}: {exc}).\n"
            "Provide the validation images explicitly, e.g.\n"
            "    python scripts/compute_source_stats.py --images-dir /path/to/imagenet/val"
        ) from exc

    raise RuntimeError(
        "No source image source configured. Pass --images-dir /path/to/imagenet/val "
        "or install the HuggingFace `datasets` package (see requirements.txt)."
    )


def _take_batches(loader, num_samples: int) -> Iterator[torch.Tensor]:
    """Yield only enough batches from ``loader`` to cover ``num_samples`` images."""
    seen = 0
    for batch in loader:
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        yield images
        seen += int(images.shape[0])
        if seen >= num_samples:
            return


# --------------------------------------------------------------------------------------
# Main routine
# --------------------------------------------------------------------------------------
def compute_and_save(
    cfg,
    num_samples: int = DEFAULT_NUM_SOURCE_SAMPLES,
    output_path: Optional[str] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    unbiased: bool = False,
    dtype: torch.dtype = torch.float32,
    progress: bool = True,
    images_dir: Optional[str] = None,
) -> SourceStats:
    """Compute ``{mu_i^S, sigma_i^S}_{i=0..N}`` and persist them to ``output_path``.

    The forward pass never injects a prompt (Appendix B.2) and never computes gradients.
    """
    device = device or cfg.get("device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    seed = int(cfg.seed if seed is None else seed)
    torch.manual_seed(seed)

    if images_dir is not None:
        cfg = dict(cfg)
        cfg["images_dir"] = images_dir

    output_path = (
        output_path
        or cfg.source_stats.get("path", FOA_DEFAULTS["source_stats"]["path"])
    )

    print("=" * 78)
    print("FOA :: source in-distribution statistics calculation")
    print("=" * 78)
    print(f"  backbone          : {cfg.model.get('name', FOA_DEFAULTS['model']['name'])}")
    print(f"  checkpoint        : {cfg.model.get('checkpoint', None) or FOA_CHECKPOINT_URL}")
    print(f"  #source samples Q : {num_samples}   (paper default: 32)")
    print(f"  prompt injection  : NONE (Appendix B.2)")
    print(f"  unbiased std      : {unbiased}")
    print(f"  device            : {device}")
    print(f"  output            : {output_path}")
    print("=" * 78)

    # ---- frozen backbone (no gradients anywhere) --------------------------------------
    model = build_vit(
        model_name=cfg.model.get("name", FOA_DEFAULTS["model"]["name"]),
        checkpoint=cfg.model.get("checkpoint", None),
        pretrained=True,
        num_classes=int(cfg.model.get("num_classes", 1000)),
        device=device,
    )
    print(
        f"[model] ViT loaded: layers N={model.num_layers}, dim d={model.embed_dim}, "
        f"patches={model.num_patches}"
    )

    # ---- source stream ----------------------------------------------------------------
    image_iter = build_source_image_iter(cfg, num_samples=num_samples, seed=seed)

    # ---- statistics -------------------------------------------------------------------
    stats = compute_source_statistics(
        model=model,
        image_iter=image_iter,
        num_samples=num_samples,
        device=device,
        unbiased=unbiased,
        dtype=dtype,
        progress=progress,
    )
    stats.dataset = str(cfg.source_stats.get("images_dir", None) or cfg.data.get("dataset", "imagenet-1k"))
    stats.seed = seed
    stats.meta.update(
        {
            "prompt_injected": False,
            "source": "imagenet-1k validation",
            "script": "scripts/compute_source_stats.py",
        }
    )

    # ---- save -------------------------------------------------------------------------
    save_source_stats(stats, output_path)
    print(f"[save] wrote source statistics for {stats.num_layers + 1} layers to {output_path}")

    # ---- ID sanity check: the discrepancy term should be ~0 on held-in samples --------
    print("[check] ||mu_N(X) - mu_N^S||_2 on the Q source samples "
          "(expected ~0 up to sampling noise):")
    for i in (0, min(1, stats.num_layers), stats.num_layers):
        mu_i = stats.mu_i(i)
        sigma_i = stats.sigma_i(i)
        print(
            f"    layer {i:>3}: mean_norm={float(mu_i.norm()):.4f} "
            f"std_norm={float(sigma_i.norm()):.4f} "
            f"sigma_mean={float(sigma_i.mean()):.4f}"
        )
    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FOA source in-distribution CLS statistics (Eqn. 5 / 7)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config (e.g. configs/foa_imagenetc.yaml). Optional.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SOURCE_SAMPLES,
        help="Q: number of unlabeled source ID images (paper default 32).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to write the statistics checkpoint (.pt).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Directory of source images (ImageFolder layout, e.g. imagenet/val).",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--unbiased",
        action="store_true",
        help="Use the unbiased (Bessel-corrected) std instead of the population std.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Disable the progress bar."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    paths: List[str] = []
    if args.config:
        paths.append(args.config)
    cfg = load_config(*paths) if paths else load_config()

    if args.images_dir:
        cfg = dict(cfg)
        cfg["images_dir"] = args.images_dir
        cfg.setdefault("source_stats", {})
        cfg["source_stats"]["images_dir"] = args.images_dir

    compute_and_save(
        cfg,
        num_samples=args.num_samples,
        output_path=args.output,
        device=args.device,
        seed=args.seed,
        unbiased=args.unbiased,
        progress=not args.quiet,
        images_dir=args.images_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
