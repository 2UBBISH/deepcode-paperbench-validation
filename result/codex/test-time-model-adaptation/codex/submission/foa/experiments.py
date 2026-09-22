"""Experiment helpers shared by the CLI: dataset / method / stream construction."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .config import BaselineConfig, FOAConfig, foa_config_for_dataset
from .core.statistics import FeatureStatistics, compute_source_statistics
from .data import datasets as D
from .data import hf as HF
from .methods import BNAdapt, CoTTA, FOAMethod, LAME, NoAdapt, SAR, T3A, TENT
from .models.prompt_vit import PAPER_VIT_CHECKPOINT, PromptViT, build_prompt_vit
from .quantization import calibrate, quantize_vit


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def build_test_loader(
    dataset: str,
    data_root: str,
    transform,
    stream_cfg: D.StreamConfig,
    corruption: Optional[str] = None,
    severity: int = 5,
    hf_cache: Optional[str] = None,
    use_hf: bool = False,
):
    """Build the test stream of one domain."""
    name = dataset.lower()
    if name == "synthetic":
        return D.SyntheticStream(
            num_samples=stream_cfg.max_samples or 64,
            num_classes=getattr(stream_cfg, "num_classes", 1000),
            img_size=getattr(stream_cfg, "img_size", 224),
            batch_size=stream_cfg.batch_size,
            seed=stream_cfg.seed,
        )
    if use_hf:
        if name == "imagenet_c":
            if corruption is None:
                raise ValueError("ImageNet-C needs a corruption name")
            print(
                "[data] streaming ImageNet-C from HuggingFace filters the corpus client "
                "side and is much slower than the official tarballs; prefer "
                "`scripts/prepare_data.py --hf-imagenet-c` once and then read from disk"
            )
            return HF.hf_imagenet_c_loader(
                corruption,
                severity=severity,
                transform=transform,
                batch_size=stream_cfg.batch_size,
                max_samples=stream_cfg.max_samples,
            )
        hf_name = {
            "imagenet_r": "imagenet_r",
            "imagenet_v2": "imagenet_v2",
            "imagenet_sketch": "imagenet_sketch",
            "imagenet": "imagenet",
        }.get(name, name)
        return HF.hf_loader(
            hf_name,
            transform,
            batch_size=stream_cfg.batch_size,
            shuffle=stream_cfg.shuffle,
            seed=stream_cfg.seed,
            num_workers=stream_cfg.num_workers,
            max_samples=stream_cfg.max_samples,
            cache_dir=hf_cache,
        )
    if name == "imagenet_c":
        if corruption is None:
            raise ValueError("ImageNet-C needs a corruption name")
        records = D.imagenet_c_records(data_root, corruption, severity)
    elif name == "imagenet_r":
        records = D.imagenet_r_records(data_root)
    elif name == "imagenet_v2":
        records = D.imagenet_v2_records(data_root)
    elif name == "imagenet_sketch":
        records = D.imagenet_sketch_records(data_root)
    elif name == "imagenet":
        records = D.imagenet_val_records(data_root)
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return D.build_stream(records, transform, stream_cfg)


def source_statistics_batches(
    data_root: str,
    transform,
    num_samples: int = 32,
    batch_size: int = 32,
    seed: int = 0,
    hf_cache: Optional[str] = None,
    use_hf: bool = True,
    synthetic: bool = False,
    img_size: int = 224,
    num_classes: int = 1000,
) -> List[torch.Tensor]:
    """32 unlabelled ImageNet-1K validation images (Appendix B.2)."""
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        return [
            torch.randn(num_samples, 3, img_size, img_size, generator=g)
        ]
    if use_hf:
        try:
            ds = HF.load_hf_dataset("imagenet", split="validation", transform=transform, cache_dir=hf_cache)
            generator = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(ds), generator=generator)[:num_samples].tolist()
            images = [ds[i][0] for i in idx]
            return [torch.stack(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
        except Exception as exc:  # fall back to a local copy of ImageNet-1K
            print(f"[source-stats] HuggingFace loading failed ({exc}); using {data_root}")
    records = D.imagenet_val_records(data_root)
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(records), generator=generator)[:num_samples].tolist()
    ds = D.ImageRecordDataset([records[i] for i in idx], transform=transform)
    images = [ds[i][0] for i in range(len(ds))]
    return [torch.stack(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]


# --------------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------------
METHOD_NAMES = ("NoAdapt", "LAME", "T3A", "TENT", "CoTTA", "SAR", "BNAdapt", "FOA")


def build_method(
    name: str,
    model,
    device: torch.device,
    source_stats: Optional[FeatureStatistics] = None,
    foa_cfg: Optional[FOAConfig] = None,
    baseline_cfg: Optional[BaselineConfig] = None,
    num_classes: int = 1000,
    **kwargs,
):
    """Instantiate a method with the hyper-parameters quoted in the paper."""
    baseline_cfg = baseline_cfg or BaselineConfig()
    key = name.lower().replace("-", "").replace("_", "")
    if key == "noadapt":
        return NoAdapt(model, device=device)
    if key == "lame":
        return LAME(model, k=baseline_cfg.lame_knn_k, device=device)
    if key == "t3a":
        return T3A(
            model,
            num_classes=num_classes,
            num_supports=baseline_cfg.t3a_num_supports,
            device=device,
        )
    if key == "tent":
        return TENT(
            model,
            lr=baseline_cfg.sgd_lr,
            momentum=baseline_cfg.sgd_momentum,
            device=device,
        )
    if key == "cotta":
        return CoTTA(
            model,
            lr=baseline_cfg.cotta_lr,
            momentum=baseline_cfg.sgd_momentum,
            augmentation_threshold=baseline_cfg.cotta_aug_threshold,
            num_augmentations=baseline_cfg.cotta_num_aug,
            restore_prob=baseline_cfg.cotta_restore_prob,
            ema_alpha=baseline_cfg.cotta_ema_alpha,
            device=device,
        )
    if key == "sar":
        return SAR(
            model,
            lr=baseline_cfg.sgd_lr,
            momentum=baseline_cfg.sgd_momentum,
            device=device,
            num_classes=num_classes,
        )
    if key == "bnadapt":
        return BNAdapt(model, device=device)
    if key == "foa":
        if source_stats is None:
            raise ValueError("FOA needs the source in-distribution statistics")
        return FOAMethod(model, source_stats, cfg=foa_cfg or FOAConfig(), device=device)
    raise ValueError(f"unknown method: {name}")


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    checkpoint: str = PAPER_VIT_CHECKPOINT
    num_prompts: int = 3
    bits: Optional[int] = None          # None -> full precision
    calibration_batches: Optional[List[torch.Tensor]] = None
    prompt_pos: str = "zero"
    prompt_init: str = "uniform"
    num_classes: int = 1000
    backbone: str = "vit"           # vit | resnet50 | visionmamba

    def build(self, device: torch.device) -> PromptViT:
        if self.backbone == "resnet50":
            from .models.prompt_resnet import PromptResNet, build_prompt_resnet

            return build_prompt_resnet(pretrained=True).to(device)
        if self.backbone == "visionmamba":
            from .models.prompt_sequence import build_prompt_visionmamba

            return build_prompt_visionmamba(
                num_prompts=self.num_prompts, pretrained=True
            ).to(device)
        model = build_prompt_vit(
            checkpoint=self.checkpoint,
            num_prompts=self.num_prompts,
            prompt_pos=self.prompt_pos,
            prompt_init=self.prompt_init,
            num_classes=self.num_classes,
        )
        if self.bits is not None:
            quantize_vit(model.vit, bits=self.bits)
            if self.calibration_batches:
                calibrate(model.vit, self.calibration_batches, search=True)
        return model.to(device)
