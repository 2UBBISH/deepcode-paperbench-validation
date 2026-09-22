"""In-painting masks (Section 4.1).

The mask is exposed both as a stand-alone helper and through
:class:`si_couplings.couplings.InpaintingCoupling`, which uses it to build
the base sample x_0 = xi o x_1 + (1 - xi) o zeta.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..couplings import tile_mask


def observed_mask(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    n_tiles: int = 8,
    p_missing: float = 0.3,
    device=None,
) -> Tensor:
    """xi = 1 on observed pixels, 0 on the missing tiles (uniform over channels)."""
    return tile_mask(batch_size, channels, height, width, n_tiles, p_missing, device)


def apply_mask(image: Tensor, xi: Tensor, noise: Tensor | None = None) -> Tensor:
    """x_0 = xi o x_1 + (1 - xi) o zeta."""
    xi = xi.expand_as(image) if xi.shape[1] == 1 else xi
    if noise is None:
        noise = torch.randn_like(image)
    return xi * image + (1.0 - xi) * noise


def arbitrary_mask(image: Tensor, n_tiles: int = 8, p_missing: float = 0.3) -> Tensor:
    """Convenience: mask a batch of images with a freshly sampled tile mask."""
    b, c, h, w = image.shape
    xi = observed_mask(b, c, h, w, n_tiles, p_missing, image.device)
    return apply_mask(image, xi, torch.randn_like(image))


__all__ = ["tile_mask", "observed_mask", "apply_mask", "arbitrary_mask"]
