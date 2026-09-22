"""Build the LVLM used for the evaluation from a parsed argument namespace."""
from __future__ import annotations

import torch

from .base import LVLM


def build_lvlm(args) -> LVLM:
    """``args.backend`` selects LLaVA-1.5 7B or OpenFlamingo-9B."""
    backend = getattr(args, "backend", "llava")
    dtype = torch.float16 if getattr(args, "precision", "fp16") == "fp16" else torch.float32
    if backend == "llava":
        from .llava import load_llava_1p5

        return load_llava_1p5(
            clip_checkpoint=getattr(args, "clip_checkpoint", None),
            clip_arch=getattr(args, "clip_arch", "ViT-L-14"),
            image_size=getattr(args, "image_size", 224),
            llava_path=getattr(args, "llava_path", "llava-hf/llava-1.5-7b-hf"),
            device=getattr(args, "device", "cuda"),
            dtype=dtype,
        )
    if backend in ("openflamingo", "of"):
        from .openflamingo import OpenFlamingoRunner

        return OpenFlamingoRunner.from_pretrained(
            clip_checkpoint=getattr(args, "clip_checkpoint", None),
            clip_arch=getattr(args, "clip_arch", "ViT-L-14"),
            image_size=getattr(args, "image_size", 224),
            device=getattr(args, "device", "cuda"),
            dtype=dtype,
        )
    raise ValueError(f"unknown LVLM backend '{backend}'")
