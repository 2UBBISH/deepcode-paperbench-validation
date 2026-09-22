"""Lightweight mask generator ``f_mask`` and the patch-wise interpolation module.

Reproduced from Section 3.2 ("Lightweight Mask Generator Module") and Section 3.3
("Patch-wise Interpolation Module") of
*Sample-specific Masks for Visual Reprogramming-based Prompting*.

Design (as described in the paper)
---------------------------------
* ``f_mask`` is a CNN made of 3x3 convolutions (stride 1, padding 1) and 2x2
  max-pooling layers.  The paper instantiates two variants: a 5-layer CNN for
  ResNet-18/ResNet-50 and a 6-layer CNN for ViT-B/32.  Both use 3 max-pooling
  layers by default, so the spatial resolution of the produced mask is
  ``floor(H / 2**l) x floor(W / 2**l)`` with ``l`` the number of pooling layers.
* The last convolution has 3 output channels, i.e. the mask is *multi-channel*,
  and it is a plain (affine) convolution: no activation is applied on the last
  layer.  This matters for Proposition 4.3 where the output of the mask
  generator is written as ``W_last * f''(r(x)) + b_last``; with ``W_last = 0``
  and ``b_last = M`` any shared binary mask ``M`` is representable, which is
  exactly the inclusion ``F_shr(f_P') subseteq F_smm(f_P')``.
* The patch-wise interpolation module rescales the low-resolution mask back to
  the image resolution by copying every mask pixel over a ``2**l x 2**l`` patch
  ("the same values within each patch").  Non divisible cases replicate the
  closest patches (edge padding).  Because the operation is a pure copy it
  avoids the floating point derivations of bilinear/bicubic interpolation
  (Appendix A.3, Table 5); in an autograd implementation the copy is a
  ``repeat_interleave`` whose Jacobian is trivial, so gradients still reach the
  mask generator.

Parameter budget
----------------
Table 4 reports 26,499 trainable parameters for the 5-layer CNN and 102,339 for
the 6-layer CNN.  The per-layer channel widths are only given in Figures 8/9,
which are not part of the provided markdown, so the defaults below were chosen
to land on (or very near) the reported budget:

===========  ==================  ==============================
backbone     hidden channels     parameters / Table 4
===========  ==================  ==============================
ResNet-18/50 (24, 32, 32, 32)    26,979 / 26,499   (+1.8%)
ViT-B/32     (16, 64, 64, 64, 32) 102,915 / 102,339 (+0.6%)
===========  ==================  ==============================

``scripts/param_stats.py`` prints the statistic and also searches for channel
widths that reproduce the exact counts; the widths remain fully configurable in
``configs/*.yaml``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PatchWiseInterpolation",
    "MaskGenerator",
    "MaskNet",
    "DEFAULT_5_LAYER_CHANNELS",
    "DEFAULT_6_LAYER_CHANNELS",
]

# Channel widths used by default (see module docstring).
DEFAULT_5_LAYER_CHANNELS: Tuple[int, ...] = (24, 32, 32, 32)
DEFAULT_6_LAYER_CHANNELS: Tuple[int, ...] = (16, 64, 64, 64, 32)


class PatchWiseInterpolation(nn.Module):
    """Patch-wise interpolation module (Section 3.3).

    Upscales a mask of size ``floor(H / p) x floor(W / p)`` to ``(H, W)``
    (``p = 2 ** l``) by repeating every pixel over a ``p x p`` patch.  Values are
    constant inside a patch; when the low-resolution grid does not divide the
    image size the closest patches are mirrored (edge replication) and the
    result is cropped.

    Parameters
    ----------
    patch_size:
        ``2 ** l`` with ``l`` the number of max-pooling layers of the generator.
        ``patch_size == 1`` makes the module an identity (it is omitted when
        ``l == 0``, as stated in the paper).
    target_size:
        ``(H, W)`` of the image the mask has to match.  When ``None`` the mask
        is upsampled to the largest multiple of ``patch_size`` of its own size
        (useful for shape unit tests).
    """

    def __init__(self, patch_size: int = 8, target_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1")
        self.patch_size = int(patch_size)
        self.target_size = tuple(target_size) if target_size is not None else None

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"patch_size={self.patch_size}, target_size={self.target_size}"

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() != 4:
            raise ValueError(f"expected a (B, C, h, w) mask, got {tuple(mask.shape)}")
        p = self.patch_size
        if p == 1 and self.target_size is None:
            return mask

        x = mask
        if self.target_size is not None:
            h_t, w_t = self.target_size
            need_h = max(math.ceil(h_t / p), x.shape[-2])
            need_w = max(math.ceil(w_t / p), x.shape[-1])
            if need_h > x.shape[-2] or need_w > x.shape[-1]:
                # "non-divisible cases mirroring the closest patches"
                x = F.pad(
                    x,
                    (0, need_w - x.shape[-1], 0, need_h - x.shape[-2]),
                    mode="replicate",
                )

        if p > 1:
            x = x.repeat_interleave(p, dim=-2).repeat_interleave(p, dim=-1)

        if self.target_size is not None:
            x = x[..., : self.target_size[0], : self.target_size[1]]
        return x


class MaskGenerator(nn.Module):
    """The CNN part of ``f_mask`` (Section 3.2).

    Parameters
    ----------
    in_channels:
        Channels of the resized target image (3 for RGB).
    hidden_channels:
        Output channels of every layer except the last one.
    out_channels:
        3 by default: the mask is multi-channel (Section 3.1/3.2).
    num_pool_layers:
        Number of 2x2 max-pooling layers ``l`` (3 by default).  It defines the
        patch size ``2 ** l`` of :class:`PatchWiseInterpolation`.
    pool_after:
        Indices (0-based) of the convolution layers that are followed by a max
        pooling operation.  Defaults to the first ``num_pool_layers``
        convolutions; must have exactly ``num_pool_layers`` entries.
    batch_norm:
        The paper only mentions convolutions and pooling ("For simplicity, we
        only include 3x3 convolution layers and 2x2 Max-Pooling layers"), so
        this is disabled by default.
    spatial_bias:
        Formalisation detail used by the proof of Proposition 4.3.  Appendix B.2
        writes the last layer as ``W_last f''(r(x)) + b_last`` with
        ``b_last in R^{H*W*C}``, i.e. an *output-space* affine term.  A standard
        convolution bias is only per-channel, so a spatially varying mask is in
        general not exactly representable by the ``W_last = 0`` argument.  With
        ``spatial_bias=True`` the generator carries an explicit
        ``output_bias`` tensor of the mask resolution (matching the paper's
        formulation); the default ``False`` keeps the plain convolutional
        architecture, for which exactly the masks that are constant per channel
        (e.g. the full watermarking mask) are representable.
    image_size:
        Required only when ``spatial_bias=True`` (it defines the shape of the
        output bias).
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: Sequence[int] = DEFAULT_5_LAYER_CHANNELS,
        out_channels: int = 3,
        num_pool_layers: int = 3,
        pool_after: Optional[Sequence[int]] = None,
        batch_norm: bool = False,
        spatial_bias: bool = False,
        image_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        channels: List[int] = list(int(c) for c in hidden_channels) + [int(out_channels)]
        num_layers = len(channels)
        if num_layers < 2:
            raise ValueError("the mask generator needs at least 2 convolution layers")
        if num_pool_layers < 0 or num_pool_layers > num_layers:
            raise ValueError("num_pool_layers must be within [0, num layers]")

        if pool_after is None:
            pool_after = list(range(num_pool_layers))
        pool_after = [int(i) for i in pool_after]
        if len(pool_after) != num_pool_layers:
            raise ValueError("pool_after must contain exactly num_pool_layers entries")
        if any(i < 0 or i >= num_layers for i in pool_after):
            raise ValueError("pool_after indices must be valid layer indices")

        self.in_channels = int(in_channels)
        self.channels = channels
        self.out_channels = int(out_channels)
        self.num_layers = num_layers
        self.num_pool_layers = int(num_pool_layers)
        self.pool_after = sorted(pool_after)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev = self.in_channels
        for i, c in enumerate(channels):
            self.convs.append(nn.Conv2d(prev, c, kernel_size=3, stride=1, padding=1, bias=True))
            if batch_norm and i < num_layers - 1:
                self.norms.append(nn.BatchNorm2d(c))
            else:
                self.norms.append(nn.Identity())
            prev = c
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.spatial_bias = bool(spatial_bias)
        if self.spatial_bias:
            if image_size is None:
                raise ValueError("spatial_bias=True requires image_size")
            low = (int(image_size[0]) // self.patch_size, int(image_size[1]) // self.patch_size)
            self.output_bias = nn.Parameter(torch.zeros(1, self.out_channels, *low))
        else:
            self.register_parameter("output_bias", None)

        self.reset_parameters()

    # ------------------------------------------------------------------ utils
    def reset_parameters(self) -> None:
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        if self.output_bias is not None:
            nn.init.zeros_(self.output_bias)

    @property
    def patch_size(self) -> int:
        """``2 ** l`` with ``l`` the number of max-pooling layers."""
        return 2 ** self.num_pool_layers

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def output_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Spatial size of the produced mask: ``floor(H / 2**l) x floor(W / 2**l)``."""
        h, w = image_size
        return (h // self.patch_size, w // self.patch_size)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        return_penultimate: bool = False,
    ):
        """Map a resized image ``r(x)`` to a mask.

        Parameters
        ----------
        x:
            Image tensor of shape ``(B, in_channels, H, W)``.
        return_penultimate:
            Also return the (activated) output of the second-to-last layer; the
            single-channel ablation of Table 3 averages it over channels.
        """
        h = x
        penultimate = None
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h)
            if i < self.num_layers - 1:
                h = norm(h)
                h = F.relu(h, inplace=False)
                if i == self.num_layers - 2:
                    penultimate = h
                if i in self.pool_after:
                    h = self.pool(h)
        if self.output_bias is not None:
            h = h + self.output_bias
        if return_penultimate:
            return h, penultimate
        return h


class MaskNet(nn.Module):
    """``f_mask``: mask generator + patch-wise interpolation (Section 3.1-3.3).

    ``single_channel=True`` implements the ablation of Table 3
    (``f_mask^s``): the penultimate feature map of the generator is averaged
    over its channels to produce a single-channel mask which is then broadcast
    to three channels.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        in_channels: int = 3,
        hidden_channels: Sequence[int] = DEFAULT_5_LAYER_CHANNELS,
        out_channels: int = 3,
        num_pool_layers: int = 3,
        single_channel: bool = False,
        batch_norm: bool = False,
        pool_after: Optional[Sequence[int]] = None,
        spatial_bias: bool = False,
    ):
        super().__init__()
        self.image_size = tuple(int(s) for s in image_size)
        self.single_channel = bool(single_channel)
        self.generator = MaskGenerator(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_pool_layers=num_pool_layers,
            pool_after=pool_after,
            batch_norm=batch_norm,
            spatial_bias=spatial_bias,
            image_size=image_size,
        )
        self.interpolation = PatchWiseInterpolation(
            patch_size=self.generator.patch_size, target_size=self.image_size
        )

    @property
    def patch_size(self) -> int:
        return self.generator.patch_size

    @property
    def mask_size(self) -> Tuple[int, int]:
        return self.generator.output_size(self.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.single_channel:
            _, feats = self.generator(x, return_penultimate=True)
            if feats is None:
                raise RuntimeError("the generator needs at least 2 layers for the single-channel variant")
            low_res = feats.mean(dim=1, keepdim=True)  # (B, 1, h, w)
        else:
            low_res = self.generator(x)
        mask = self.interpolation(low_res)
        if mask.shape[1] != self.generator.out_channels:
            mask = mask.expand(-1, self.generator.out_channels, -1, -1)
        return mask


def build_mask_net(image_size: Tuple[int, int], num_layers: int = 5, **kwargs) -> MaskNet:
    """Convenience factory: 5-layer CNN for ResNets, 6-layer CNN for ViT."""
    if num_layers == 5:
        kwargs.setdefault("hidden_channels", DEFAULT_5_LAYER_CHANNELS)
    elif num_layers == 6:
        kwargs.setdefault("hidden_channels", DEFAULT_6_LAYER_CHANNELS)
    else:
        raise ValueError("num_layers must be 5 (ResNet) or 6 (ViT)")
    return MaskNet(image_size=image_size, **kwargs)
