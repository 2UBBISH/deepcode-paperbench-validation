"""Pre-trained backbones used by the paper (guided-diffusion DDPM + classifier).

Section 5.2: "we employ a pre-trained DDPM similar to DDPM-PA" and "We utilize
a model pre-trained on the ImageNet dataset, provided by (Dhariwal & Nichol,
2021), and subsequently fine-tune it with a new binary classifier head on a
limited set of 10 target domain images."

The supplementary material specifies the two checkpoints:

    DDPM:       256x256_diffusion_uncond.pt
    classifier: 256x256_classifier.pt

both released by OpenAI at ``https://openaipublic.blob.core.windows.net/diffusion/jul-2021/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .third_party.guided_diffusion.unet import EncoderUNetModel, UNetModel
from .utils import download_file

OPENAI_DIFFUSION_BASE = "https://openaipublic.blob.core.windows.net/diffusion/jul-2021"
DDPM_256_CHECKPOINT = f"{OPENAI_DIFFUSION_BASE}/256x256_diffusion_uncond.pt"
CLASSIFIER_256_CHECKPOINT = f"{OPENAI_DIFFUSION_BASE}/256x256_classifier.pt"


@dataclass
class DDPM256Config:
    """Configuration of the released 256x256 unconditional DDPM."""

    image_size: int = 256
    num_channels: int = 256
    num_res_blocks: int = 2
    attention_resolutions: str = "16,8"
    channel_mult: str = "1,1,2,2,4,4"
    num_head_channels: int = 64
    dropout: float = 0.0
    use_scale_shift_norm: bool = True
    resblock_updown: bool = True
    learn_sigma: bool = True
    num_classes: Optional[int] = None  # unconditional


@dataclass
class Classifier256Config:
    """Configuration of the released 256x256 ImageNet classifier."""

    image_size: int = 256
    classifier_width: int = 128
    classifier_depth: int = 2
    classifier_attention_resolutions: str = "32,16,8"
    classifier_use_scale_shift_norm: bool = True
    classifier_resblock_updown: bool = True
    classifier_pool: str = "attention"


def build_ddpm_unet(config: Optional[DDPM256Config] = None) -> nn.Module:
    """Instantiate the 256x256 unconditional guided-diffusion U-Net."""
    config = config or DDPM256Config()
    channel_mult = tuple(int(part) for part in config.channel_mult.split(","))
    attention_resolutions = tuple(
        config.image_size // int(part) for part in config.attention_resolutions.split(",")
    )
    return UNetModel(
        image_size=config.image_size,
        in_channels=3,
        model_channels=config.num_channels,
        out_channels=6 if config.learn_sigma else 3,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        num_classes=config.num_classes,
        dropout=config.dropout,
        num_heads=4,
        num_head_channels=config.num_head_channels,
        use_scale_shift_norm=config.use_scale_shift_norm,
        resblock_updown=config.resblock_updown,
        use_fp16=False,
    )


def load_ddpm_256(checkpoint: Optional[str] = None, download: bool = True, strict: bool = True):
    """Load the pre-trained source DDPM used for the few-shot experiments."""
    if checkpoint is None and download:
        checkpoint = os.path.join("checkpoints", "256x256_diffusion_uncond.pt")
        download_file(DDPM_256_CHECKPOINT, checkpoint)
    model = build_ddpm_unet()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            print(f"[backbone] missing={missing} unexpected={unexpected}")
    model.convert_to_fp32()
    return model


def build_classifier_unet(
    num_classes: int = 1000, config: Optional[Classifier256Config] = None
) -> nn.Module:
    """Instantiate the 256x256 ImageNet classifier architecture."""
    config = config or Classifier256Config()
    if config.image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif config.image_size == 128:
        channel_mult = (1, 1, 2, 3, 4)
    elif config.image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif config.image_size == 32:  # only used by the CPU smoke test
        channel_mult = (1, 2, 3, 4)
    elif config.image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    else:
        raise ValueError(f"unsupported image size: {config.image_size}")
    attention_resolutions = tuple(
        config.image_size // int(part)
        for part in config.classifier_attention_resolutions.split(",")
    )
    return EncoderUNetModel(
        image_size=config.image_size,
        in_channels=3,
        model_channels=config.classifier_width,
        out_channels=num_classes,
        num_res_blocks=config.classifier_depth,
        attention_resolutions=attention_resolutions,
        channel_mult=channel_mult,
        use_fp16=False,
        num_head_channels=64,
        use_scale_shift_norm=config.classifier_use_scale_shift_norm,
        resblock_updown=config.classifier_resblock_updown,
        pool=config.classifier_pool,
    )


def replace_classifier_head(
    classifier: nn.Module, num_classes: int = 2, init_std: float = 0.02
) -> nn.Module:
    """Replace the last layer so that it outputs ``num_classes`` logits.

    "These pre-trained models were fine-tuned by modifying the last layer to
    output two classes to classify whether images were coming from the source
    or the target dataset" (supplementary material, Section 5.2).
    """
    pool = getattr(classifier, "pool", None)
    if pool == "attention":
        pool_layer = classifier.out[-1]
        embed_dim = pool_layer.qkv_proj.in_channels
        new_head = torch.nn.Conv1d(embed_dim, num_classes, kernel_size=1)
    elif pool == "adaptive":
        conv = classifier.out[3]
        new_head = torch.nn.Conv2d(conv.in_channels, num_classes, kernel_size=1)
    elif pool == "spatial" or pool == "spatial_v2":
        linear = classifier.out[-1]
        new_head = torch.nn.Linear(linear.in_features, num_classes)
    else:
        raise NotImplementedError(f"unsupported pooling type: {pool}")

    nn.init.normal_(new_head.weight, mean=0.0, std=init_std)
    nn.init.zeros_(new_head.bias)
    if pool == "attention":
        pool_layer.c_proj = new_head
    elif pool == "adaptive":
        classifier.out[3] = new_head
    else:
        classifier.out[-1] = new_head
    classifier.out_channels = num_classes
    return classifier


def load_domain_classifier(
    checkpoint: Optional[str] = None,
    download: bool = True,
    num_classes: int = 2,
    device="cpu",
) -> nn.Module:
    """Load the ImageNet classifier, then attach a fresh binary head.

    The head is the only randomly initialised part; it is fine-tuned on the
    source/target images (10 target images are enough, Section 5.5).
    """
    if checkpoint is None and download:
        checkpoint = os.path.join("checkpoints", "256x256_classifier.pt")
        download_file(CLASSIFIER_256_CHECKPOINT, checkpoint)
    classifier = build_classifier_unet(num_classes=1000)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        classifier.load_state_dict(state)
    classifier.convert_to_fp32()
    replace_classifier_head(classifier, num_classes=num_classes)
    return classifier.to(device)


def base_model_and_adaptor_names(model: nn.Module) -> Tuple[str, ...]:
    """Names of adaptor sub-modules inside a wrapped model (used for bookkeeping)."""
    return tuple(name for name, _ in model.named_modules() if name.endswith("adaptor"))


def load_trained_domain_classifier(path: str, num_classes: int = 2, device="cpu") -> nn.Module:
    """Reload a classifier produced by :func:`dpms_ant.classifier.save_classifier`."""
    classifier = build_classifier_unet(num_classes=num_classes)
    payload = torch.load(path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    classifier.load_state_dict(state)
    classifier.convert_to_fp32()
    return classifier.to(device)
