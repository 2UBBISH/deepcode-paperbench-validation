"""Lightweight CNN mask generator :math:`f_{\\mathrm{mask}}` used by SMM.

Reference
---------
* Section 3.2 "Lightweight Mask Generator Module"
* Appendix A.2 "Architecture of the Mask Generator and Parameter Statistics"
  (Figures 8 and 9, Table 4)

Paper specification (verbatim points)
-------------------------------------
* The mask generator is a CNN with **five** layers for ResNet-18 / ResNet-50 and
  **six** layers for ViT-B32.
* Each CNN layer is a ``3 x 3`` convolution with ``padding = 1`` and ``stride = 1``;
  both models contain **three** ``2 x 2`` Max-Pooling layers.
* The number of channels of the last layer is set to ``3`` so that a three-channel
  mask is produced (Figure 8 / Figure 9).
* Input is the resized image :math:`r(x_i)`; the output mask has spatial size
  ``floor(H / 2**l) x floor(W / 2**l)`` where ``l`` is the number of pooling layers
  (``l = 3`` in the paper, i.e. an output of ``28 x 28`` for a ``224 x 224`` input
  and ``48 x 48`` for a ``384 x 384`` input).  The mask is up-sampled back to the
  input resolution by the non-parametric patch-wise interpolation module
  (:mod:`smm_vr.modules.patch_interp`).
* Table 4 parameter budgets (total trainable parameters of ``f_mask``):

  ===========  ============  ===========  ============
  backbone     input size    CNN layers   params
  ===========  ============  ===========  ============
  ResNet-18    224x224x3     5            26,499
  ResNet-50    224x224x3     5            26,499
  ViT-B32      384x384x3     6            102,339
  ===========  ============  ===========  ============

Design notes / documented defaults taken where the paper is silent
------------------------------------------------------------------
The paper reports the *exact* parameter count but omits the channel widths.  A
geometrically widening stack starting from a base width of ``8``
(``8, 16, 32, 64`` for the 5-layer net and ``8, 16, 32, 64, 128`` for the 6-layer
net) together with affine BatchNorm parameters on the hidden layers reproduces the
Table 4 counts *exactly*:

* 5-layer: conv weights/biases ``26,259`` + BatchNorm ``2 * (8+16+32+64) = 240``
  => **26,499**
* 6-layer: conv weights/biases ``101,843`` + BatchNorm ``2 * (8+16+32+64+128) = 496``
  => **102,339**

Consequently the hidden blocks are ``Conv2d -> BatchNorm2d -> ReLU`` and the final
layer is a plain (bias-only) ``3 x 3`` convolution producing the three mask
channels.  The paper does not state which layers carry the pooling; by default the
pooling layers are placed in the first ``l`` hidden blocks ("pool early"), which
keeps the cheap layers at full resolution and is the more economical choice for the
384x384 ViT input.

The module deliberately returns the **low resolution** mask; up-sampling is the
responsibility of :class:`smm_vr.modules.patch_interp.PatchWiseInterpolation`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = [
    "MaskGenerator",
    "build_mask_generator",
    "MASK_GENERATOR_CONFIGS",
    "EXPECTED_PARAMETERS",
    "verify_parameters",
    "count_parameters",
]


# --------------------------------------------------------------------------------------
# Table 4 of the paper: backbone -> total number of trainable parameters of f_mask
# --------------------------------------------------------------------------------------
EXPECTED_PARAMETERS: Dict[str, int] = {
    "resnet18": 26_499,
    "resnet50": 26_499,
    "vit_b32": 102_339,
    # Appendix E.1 also discusses a ViT-Large/384 backbone; the paper only tabulates
    # ResNet-18, ResNet-50 and ViT-B32, so the 6-layer mask generator is reused.
    "vit_large": 102_339,
}

#: Backbone -> mask generator keyword arguments (number of CNN layers).
MASK_GENERATOR_CONFIGS: Dict[str, Dict[str, object]] = {
    "resnet18": dict(num_layers=5, base_channels=8, num_pooling_layers=3),
    "resnet50": dict(num_layers=5, base_channels=8, num_pooling_layers=3),
    "vit_b32": dict(num_layers=6, base_channels=8, num_pooling_layers=3),
    "vit_large": dict(num_layers=6, base_channels=8, num_pooling_layers=3),
}

#: Default patch size (2**l) shared with the patch-wise interpolation module.
DEFAULT_PATCH_SIZE = 8


def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:
    """Number of (trainable) parameters held by ``module``."""
    if only_trainable:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in module.parameters()))


class MaskGenerator(nn.Module):
    """CNN that maps a resized image ``r(x_i)`` to a low-resolution 3-channel mask.

    Parameters
    ----------
    num_layers:
        Total number of convolution layers (``5`` for ResNet backbones, ``6`` for
        ViT backbones, c.f. Figures 8 and 9).
    base_channels:
        Width of the first hidden layer; each subsequent hidden layer doubles it.
    out_channels:
        Number of channels produced by the last layer (``3`` in the paper).
    num_pooling_layers:
        Number of ``2 x 2`` Max-Pooling layers (``l = 3`` in the paper).
    pool_after:
        1-based indices of the hidden convolution layers after which a pooling layer
        is inserted.  Defaults to the first ``num_pooling_layers`` hidden layers.
    use_bn:
        Insert affine ``BatchNorm2d`` after every hidden convolution; required to
        reproduce the Table 4 parameter counts (see the module docstring).
    use_relu:
        Insert ``ReLU`` after every hidden convolution (the paper does not name the
        activation; it is parameter-free and therefore does not affect Table 4).
    width_scale / channels:
        Used by the scaling study of Appendix D.3 (Table 11), which progressively
        doubles the intermediate channels.  ``width_scale`` multiplies the geometric
        channel widths; ``channels`` overrides them completely.
    in_channels:
        Channels of the input image (``3``).
    """

    def __init__(
        self,
        num_layers: int = 5,
        base_channels: int = 8,
        out_channels: int = 3,
        num_pooling_layers: int = 3,
        pool_after: Optional[Sequence[int]] = None,
        use_bn: bool = True,
        use_relu: bool = True,
        width_scale: float = 1.0,
        channels: Optional[Sequence[int]] = None,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError(f"a mask generator needs at least 2 CNN layers, got {num_layers}")
        if num_pooling_layers < 0:
            raise ValueError("num_pooling_layers must be non-negative")

        if channels is not None:
            hidden: List[int] = [int(c) for c in channels]
            if len(hidden) != num_layers - 1:
                raise ValueError(
                    "`channels` must give exactly num_layers - 1 hidden widths, "
                    f"got {len(hidden)} for num_layers={num_layers}"
                )
            if any(c <= 0 for c in hidden):
                raise ValueError("every channel width must be positive")
        else:
            if base_channels <= 0:
                raise ValueError("base_channels must be positive")
            if width_scale <= 0:
                raise ValueError("width_scale must be positive")
            hidden = [
                max(1, int(round(base_channels * width_scale * (2 ** i))))
                for i in range(num_layers - 1)
            ]

        if num_pooling_layers > len(hidden):
            raise ValueError(
                "cannot insert more pooling layers than hidden convolution layers "
                f"({num_pooling_layers} > {len(hidden)})"
            )

        if pool_after is None:
            pool_after_list = list(range(1, num_pooling_layers + 1))
        else:
            pool_after_list = [int(p) for p in pool_after]
            if len(pool_after_list) != num_pooling_layers:
                raise ValueError(
                    f"`pool_after` must contain {num_pooling_layers} entries, "
                    f"got {len(pool_after_list)}"
                )
            if any(p < 1 or p > len(hidden) for p in pool_after_list):
                raise ValueError(
                    f"`pool_after` indices must lie in [1, {len(hidden)}], got {pool_after_list}"
                )

        self.num_layers = int(num_layers)
        self.num_pooling_layers = int(num_pooling_layers)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_channels = int(base_channels)
        self.width_scale = float(width_scale)
        self.hidden_channels: List[int] = hidden
        self.pool_after: List[int] = sorted(pool_after_list)
        self.use_bn = bool(use_bn)
        self.use_relu = bool(use_relu)

        blocks: List[nn.Sequential] = []
        prev = self.in_channels
        for idx, width in enumerate(hidden, start=1):
            ops: List[nn.Module] = [
                nn.Conv2d(prev, width, kernel_size=3, stride=1, padding=1, bias=True)
            ]
            if self.use_bn:
                ops.append(nn.BatchNorm2d(width))
            if self.use_relu:
                ops.append(nn.ReLU(inplace=True))
            if idx in self.pool_after:
                # 2x2 max-pooling with stride 2 halves each spatial dimension.
                ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            blocks.append(nn.Sequential(*ops))
            prev = width
        self.blocks = nn.ModuleList(blocks)

        # Last layer: three output channels (the mask).
        self.head = nn.Conv2d(
            prev, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        self._init_weights()

    # ------------------------------------------------------------------- helpers
    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def num_pooling(self) -> int:
        """``l`` of Section 3.3 / Appendix A.2 (number of 2x2 max-pooling layers)."""
        return self.num_pooling_layers

    @property
    def patch_size(self) -> int:
        """Patch size ``2**l`` implied by the number of pooling layers."""
        return 2 ** self.num_pooling_layers

    def spatial_out_size(self, in_size: Iterable[int]) -> Tuple[int, int]:
        """Spatial size of the produced mask for an input of size ``(H, W)``.

        Follows Appendix A.2: ``floor(H / 2**l) x floor(W / 2**l)``.
        """
        h, w = tuple(int(v) for v in in_size)[:2]
        scale = 2 ** self.num_pooling_layers
        return (h // scale, w // scale)

    # ------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, return_penultimate: bool = False):
        """Compute the low-resolution mask for a batch of resized images.

        Parameters
        ----------
        x:
            Resized images ``r(x_i)`` of shape ``(B, 3, H, W)``.
        return_penultimate:
            When ``True`` also return the penultimate feature map (the output of the
            last hidden block, before the final 3-channel convolution).  The
            single-channel ablation ``f_mask^s`` of Section 5 averages this feature
            map over channels to obtain a one-channel mask.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            The mask of shape ``(B, 3, H // 2**l, W // 2**l)`` and, optionally, the
            penultimate feature map of shape ``(B, C, H // 2**l, W // 2**l)``.
        """
        if x.dim() != 4:
            raise ValueError(f"expected a 4D input tensor (B, C, H, W), got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")

        h = x
        for block in self.blocks:
            h = block(h)

        penultimate = h
        mask = self.head(penultimate)

        if return_penultimate:
            return mask, penultimate
        return mask

    # -------------------------------------------------------------- introspection
    def parameter_breakdown(self) -> Dict[str, int]:
        """Break the parameter budget down by component (mirrors Table 4 checks)."""
        conv = sum(
            p.numel()
            for m in self.modules()
            if isinstance(m, nn.Conv2d)
            for p in m.parameters()
        )
        bn = sum(
            p.numel()
            for m in self.modules()
            if isinstance(m, nn.BatchNorm2d)
            for p in m.parameters()
        )
        return {"conv": int(conv), "batchnorm": int(bn), "total": count_parameters(self)}

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"num_layers={self.num_layers}, hidden_channels={self.hidden_channels}, "
            f"pool_after={self.pool_after}, out_channels={self.out_channels}, "
            f"use_bn={self.use_bn}, patch_size={self.patch_size}"
        )


def build_mask_generator(backbone: str = "resnet18", **overrides) -> MaskGenerator:
    """Build the mask generator prescribed for a pre-trained backbone.

    Parameters
    ----------
    backbone:
        ``"resnet18"``, ``"resnet50"``, ``"vit_b32"`` or ``"vit_large"``
        (case-insensitive; ``-`` and ``/`` separators are accepted).
    overrides:
        Extra keyword arguments forwarded to :class:`MaskGenerator` (for instance
        ``width_scale=2`` for the scaling study of Table 11).
    """
    key = str(backbone).lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    if key in ("resnet_18",):
        key = "resnet18"
    elif key in ("resnet_50",):
        key = "resnet50"
    elif key in ("vit_b_32", "vitb32", "vit_b32_384", "vitb_32", "vit_b32"):
        key = "vit_b32"
    elif key in ("vit_l_16", "vitl16", "vit_large_384", "vitlarge"):
        key = "vit_large"
    if key not in MASK_GENERATOR_CONFIGS:
        raise ValueError(
            f"unknown backbone '{backbone}'; expected one of {sorted(MASK_GENERATOR_CONFIGS)}"
        )
    kwargs = dict(MASK_GENERATOR_CONFIGS[key])
    kwargs.update(overrides)
    return MaskGenerator(**kwargs)


def verify_parameters(verbose: bool = False) -> Dict[str, int]:
    """Check that each mask generator matches the Table 4 parameter budget.

    Returns a mapping ``backbone -> number of parameters``.  Raises ``AssertionError``
    if a configuration deviates from the paper.
    """
    counts: Dict[str, int] = {}
    for backbone, expected in EXPECTED_PARAMETERS.items():
        generator = build_mask_generator(backbone)
        total = count_parameters(generator)
        counts[backbone] = total
        if verbose:
            print(f"{backbone:>10}: {total:>8,} params (expected {expected:,})")
        assert total == expected, (
            f"mask generator for {backbone} has {total} parameters, expected {expected} "
            "(Table 4 of the paper)"
        )
    return counts


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    verify_parameters(verbose=True)

    for backbone, in_size in (("resnet18", 224), ("resnet50", 224), ("vit_b32", 384)):
        net = build_mask_generator(backbone)
        net.eval()
        dummy = torch.zeros(2, 3, in_size, in_size)
        with torch.no_grad():
            mask, penultimate = net(dummy, return_penultimate=True)
        print(
            f"{backbone}: input {tuple(dummy.shape)} -> mask {tuple(mask.shape)} "
            f"(expected {(in_size // 8, in_size // 8)}), penultimate {tuple(penultimate.shape)}"
        )
        assert tuple(mask.shape[2:]) == net.spatial_out_size((in_size, in_size))
