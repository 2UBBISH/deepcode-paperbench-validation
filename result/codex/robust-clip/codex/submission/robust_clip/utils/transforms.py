"""Normalization helpers.

Convention used throughout this repository: **all data pipelines hand images in
the unit range ``[0, 1]`` to the models, and the model wrappers apply the CLIP
normalization internally.**

This is a direct consequence of the addendum of the paper, which states that the
PGD implementation computes the ``l_inf`` ball *"around non-normalized inputs"*:
the perturbation budget ``eps`` is always defined on the raw pixel values, while
the (differentiable) normalization is part of the model.
"""

from __future__ import annotations

from typing import Sequence

import torch

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)


def _as_tensor(values: Sequence[float], like: torch.Tensor) -> torch.Tensor:
    tensor = torch.tensor(list(values), dtype=like.dtype, device=like.device)
    return tensor.view(1, -1, *([1] * (like.dim() - 2)))


def normalize_images(x: torch.Tensor, mean: Sequence[float] = MEAN, std: Sequence[float] = STD) -> torch.Tensor:
    """``[0, 1]`` image -> normalized model input (CLIP mean / std by default)."""
    mean_t = _as_tensor(mean, x)
    std_t = _as_tensor(std, x)
    return (x - mean_t) / std_t


def denormalize_images(x: torch.Tensor, mean: Sequence[float] = MEAN, std: Sequence[float] = STD) -> torch.Tensor:
    """Inverse of :func:`normalize_images`."""
    mean_t = _as_tensor(mean, x)
    std_t = _as_tensor(std, x)
    return x * std_t + mean_t
