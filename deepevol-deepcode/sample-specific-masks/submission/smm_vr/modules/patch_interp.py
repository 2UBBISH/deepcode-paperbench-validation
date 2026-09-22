"""Patch-wise interpolation module (paper Section 3.3, Appendix A.3).

The patch-wise interpolation module upscales the CNN-generated masks from
``floor(H / 2**l) x floor(W / 2**l)`` back to the original size ``H x W`` per
channel (it is omitted when ``l == 0``).

Implementation follows the paper verbatim:

* a grid of ``floor(H / 2**l) x floor(W / 2**l)`` patches is used, each of size
  ``2**l x 2**l``, "ensuring the same values within each patch";
* "non-divisible cases mirroring the closest patches" -- i.e. when ``H`` (or
  ``W``) is not divisible by ``2**l`` the extra output rows/columns are filled
  by the *nearest* existing patch (block replication with index clipping);
* the operation is a pure index-copy / nearest-neighbour gather, so

  - no floating point interpolation weights are introduced, and
  - gradients still flow to the mask generator (``f_mask``) while the
    interpolation itself needs no gradient computation, exactly as the paper
    motivates ("this module exclusively involves copying operations, thus
    avoiding floating-point calculations ... during training").
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchWiseInterpolation(nn.Module):
    """Non-parametric patch-wise (block replication) upsampling.

    Parameters
    ----------
    patch_size:
        Edge length ``2**l`` of an individual patch. ``1`` is the identity
        (``l = 0``).
    """

    def __init__(self, patch_size: int = 8) -> None:
        super().__init__()
        patch_size = int(patch_size)
        if patch_size < 1 or (patch_size & (patch_size - 1)) != 0:
            raise ValueError(
                f"patch_size must be a power of two (2**l), got {patch_size}"
            )
        self.patch_size = patch_size

    @property
    def num_pooling_layers(self) -> int:
        """``l`` such that ``patch_size == 2**l``."""
        return self.patch_size.bit_length() - 1

    @staticmethod
    def _index_map(in_size: int, out_size: int, patch_size: int,
                   device: torch.device) -> torch.Tensor:
        """Row (or column) index map of length ``out_size`` into ``in_size`` entries.

        Output entry ``o`` reads from input entry ``floor(o / patch_size)``.
        Entries beyond ``in_size`` "mirror the closest patches", i.e. they clip
        to the last available patch.
        """
        out = torch.arange(out_size, device=device)
        idx = torch.div(out, patch_size, rounding_mode="floor")
        return idx.clamp_(max=max(in_size - 1, 0))

    def forward(self, mask: torch.Tensor, out_size=None) -> torch.Tensor:
        """Upsample ``mask`` of shape ``(B, C, h, w)`` to ``out_size = (H, W)``.

        When ``out_size`` is ``None`` the natural size
        ``(h * 2**l, w * 2**l)`` is used.
        """
        if mask.dim() != 4:
            raise ValueError(
                f"expected a 4D mask tensor, got shape {tuple(mask.shape)}"
            )

        if self.patch_size == 1:
            return mask

        _, _, h, w = mask.shape
        if out_size is None:
            h_out = h * self.patch_size
            w_out = w * self.patch_size
        else:
            h_out, w_out = int(out_size[0]), int(out_size[1])

        if h_out == h and w_out == w:
            return mask

        row_idx = self._index_map(h, h_out, self.patch_size, mask.device)
        col_idx = self._index_map(w, w_out, self.patch_size, mask.device)

        # ``index_select`` is a pure index copy: differentiable w.r.t. ``mask``
        # and it does not create gradient nodes for interpolation weights.
        out = mask.index_select(2, row_idx)
        out = out.index_select(3, col_idx)
        return out


def patch_wise_interpolate(mask: torch.Tensor, patch_size: int = 8,
                           out_size=None) -> torch.Tensor:
    """Functional wrapper around :class:`PatchWiseInterpolation`."""
    return PatchWiseInterpolation(patch_size=patch_size)(mask, out_size=out_size)
