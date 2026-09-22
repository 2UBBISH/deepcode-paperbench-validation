"""Down/up-sampling operators for the super-resolution experiments (Sec. 4.2).

The task is defined by  x_0 = U(D(x_1)) + sigma zeta,  with
D: R^{C x W x H} -> R^{C x W_low x H_low} the down-sampling operator and
U: R^{C x W_low x H_low} -> R^{C x W x H} the up-sampling operator.  The
up-sampled low-resolution image xi = U(D(x_1)) is given to the velocity model
as additional input channels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def downsample(x: Tensor, scale: int = 4, mode: str = "area") -> Tensor:
    """D: high-resolution image -> low-resolution image."""
    h, w = x.shape[-2:]
    kwargs = {"antialias": True} if mode in ("bilinear", "bicubic") else {}
    return F.interpolate(x, size=(h // scale, w // scale), mode=mode, **kwargs)


def upsample(x: Tensor, scale: int = 4, size=None, mode: str = "bicubic") -> Tensor:
    """U: low-resolution image -> high-resolution image."""
    if size is not None:
        return F.interpolate(x, size=size, mode=mode, align_corners=False)
    return F.interpolate(x, scale_factor=scale, mode=mode, align_corners=False)


def paired_lowres(high_res: Tensor, scale: int = 4, mode: str = "area") -> Tensor:
    """U(D(x_1)): the low-resolution image used as conditioning and base mean."""
    return upsample(downsample(high_res, scale, mode=mode), size=high_res.shape[-2:])


__all__ = ["downsample", "upsample", "paired_lowres"]
