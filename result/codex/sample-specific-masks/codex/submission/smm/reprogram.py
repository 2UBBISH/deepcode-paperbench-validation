"""Input visual reprogramming modules ``f_in`` (SMM and shared-mask baselines).

Paper reference: Section 3.1 (framework of SMM), Section 3.4 (learning strategy)
and Section 5 "Baselines" (Pad / Narrow / Medium / Full).

SMM hypothesis (Eq. 4):

    f_in(x_i | phi, delta) = r(x_i) + delta * f_mask(r(x_i) | phi)

with ``r`` the resizing function (bilinear upsampling to the pre-trained model
input size), ``delta`` the *shared* noise pattern (initialised to zeros) and
``f_mask`` the sample-specific multi-channel mask generator.

Shared-mask baselines replace ``f_mask(r(x_i))`` by a fixed binary mask ``M``:

* ``full``   : ``M`` is an all-one matrix over the whole image (watermarking,
               Bahng et al. 2022; this is also the "Only delta" ablation).
* ``narrow`` : ``M`` is one only on a border of width 28 (= 1/8 of 224).
* ``medium`` : ``M`` is one only on a border of width 56 (a quarter of 224).
* ``pad``    : the resized image is centred on a zero canvas and ``M`` is one
               only on the padded border (padding-based reprogramming,
               Chen et al. 2023).

All variants share the same learnable pattern ``delta`` of shape
``(1, C, H, W)``; the masks modulate *where* the pattern is applied, which is
the object of the theoretical analysis (Theorem 4.2 / Proposition 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mask_generator import (
    DEFAULT_5_LAYER_CHANNELS,
    DEFAULT_6_LAYER_CHANNELS,
    MaskNet,
)

__all__ = [
    "InputReprogrammingConfig",
    "InputReprogramming",
    "build_input_reprogramming",
    "border_mask",
]


def border_mask(
    image_size: Tuple[int, int],
    width: int,
    channels: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Binary mask that is one only on a border of ``width`` pixels."""
    h, w = int(image_size[0]), int(image_size[1])
    m = torch.zeros(1, channels, h, w, device=device)
    if width > 0:
        m[..., :width, :] = 1.0
        m[..., -width:, :] = 1.0
        m[..., :, :width] = 1.0
        m[..., :, -width:] = 1.0
    return m


@dataclass
class InputReprogrammingConfig:
    """Configuration of the input transformation ``f_in``.

    Attributes
    ----------
    image_size:
        ``(H, W)`` fed to the pre-trained model (224 for ResNet, 384 for ViT).
    variant:
        ``"smm"`` (generative sample-specific mask), ``"shared"`` (fixed binary
        mask) or ``"sample_specific_pattern"`` (``f_in = r(x) + f_mask(r(x))``,
        the second ablation column of Table 3, i.e. no shared pattern).
    mask_kind:
        For ``variant="shared"``: ``"full"``, ``"narrow"``, ``"medium"`` or
        ``"pad"``.
    num_layers:
        Number of convolution layers of ``f_mask`` (5 for ResNet, 6 for ViT).
    hidden_channels:
        Channel widths of ``f_mask``.
    num_pool_layers:
        ``l``: number of 2x2 max-pooling layers, i.e. patch size ``2 ** l``.
        The patch-size study of Figure 4 sweeps ``l`` in ``{0, 1, 2, 3, 4}``.
    single_channel:
        Single-channel ablation ``f_mask^s`` (Table 3).
    use_pattern:
        Whether the shared pattern ``delta`` is used (ablation "Only f_mask").
    border_width / pad_width:
        Border widths of the Narrow / Medium masks (28 and 56) and of the Pad
        variant (the image is resized to ``image_size - 2 * pad_width``).
    output_channels:
        ``3``: masks are three-channel (Section 3.1).
    """

    image_size: Tuple[int, int] = (224, 224)
    variant: str = "smm"
    mask_kind: str = "full"
    num_layers: int = 5
    hidden_channels: Optional[Sequence[int]] = None
    num_pool_layers: int = 3
    single_channel: bool = False
    use_pattern: bool = True
    border_width: int = 28
    medium_width: int = 56
    pad_width: int = 28
    output_channels: int = 3
    batch_norm: bool = False
    pool_after: Optional[Sequence[int]] = None
    spatial_bias: bool = False

    def __post_init__(self) -> None:
        if self.variant not in {"smm", "shared", "sample_specific_pattern"}:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.mask_kind not in {"full", "narrow", "medium", "pad"}:
            raise ValueError(f"unknown mask_kind {self.mask_kind!r}")
        if self.hidden_channels is None:
            self.hidden_channels = (
                DEFAULT_6_LAYER_CHANNELS if self.num_layers == 6 else DEFAULT_5_LAYER_CHANNELS
            )


class InputReprogramming(nn.Module):
    """``f_in``: resize + masked learnable pattern (see module docstring)."""

    def __init__(self, config: InputReprogrammingConfig):
        super().__init__()
        self.config = config
        image_size = tuple(int(s) for s in config.image_size)
        self.image_size = image_size
        self.uses_mask_generator = config.variant in {"smm", "sample_specific_pattern"}

        if self.uses_mask_generator:
            self.mask_net: Optional[MaskNet] = MaskNet(
                image_size=image_size,
                in_channels=3,
                hidden_channels=config.hidden_channels,
                out_channels=config.output_channels,
                num_pool_layers=config.num_pool_layers,
                single_channel=config.single_channel,
                batch_norm=config.batch_norm,
                pool_after=config.pool_after,
                spatial_bias=config.spatial_bias,
            )
            self.register_buffer("shared_mask", torch.zeros(1, 1, 1, 1), persistent=False)
            self.pad_width = 0
        else:
            self.mask_net = None
            self.pad_width = int(config.pad_width) if config.mask_kind == "pad" else 0
            inner = (image_size[0] - 2 * self.pad_width, image_size[1] - 2 * self.pad_width)
            if config.mask_kind == "full":
                mask = torch.ones(1, 1, *image_size)
            elif config.mask_kind == "narrow":
                mask = border_mask(image_size, config.border_width)
            elif config.mask_kind == "medium":
                mask = border_mask(image_size, config.medium_width)
            else:  # pad
                mask = border_mask(image_size, self.pad_width)
                if self.pad_width == 0:  # degenerate: allow full pattern on the canvas
                    mask = torch.ones(1, 1, *image_size)
            self.register_buffer("shared_mask", mask)  # (1, 1, H, W) binary, fixed
            self._inner_size = inner

        # Shared learnable pattern delta, initialised to zero (Section 3.4).
        if config.use_pattern:
            self.delta = nn.Parameter(torch.zeros(1, config.output_channels, *image_size))
        else:
            self.register_parameter("delta", None)

    # ------------------------------------------------------------------ utils
    @property
    def pattern(self) -> Optional[nn.Parameter]:
        return self.delta

    @property
    def patch_size(self) -> int:
        if self.mask_net is None:
            return 1
        return self.mask_net.patch_size

    def mask_parameters(self):
        """Parameters of the mask generator ``phi`` (empty for shared masks)."""
        if self.mask_net is None:
            return []
        return list(self.mask_net.parameters())

    def resized_input(self, x: torch.Tensor) -> torch.Tensor:
        """``r(x)``: resize (and, for the Pad baseline, centre on a canvas)."""
        if self.pad_width > 0:
            inner = F.interpolate(
                x, size=self._inner_size, mode="bilinear", align_corners=False
            )
            return F.pad(inner, (self.pad_width,) * 4, mode="constant", value=0.0)
        return x

    def make_mask(self, r_x: torch.Tensor) -> torch.Tensor:
        """The (sample-specific or shared) mask used to place ``delta``."""
        if self.mask_net is not None:
            return self.mask_net(r_x)
        mask = self.shared_mask
        if mask.shape[-2:] != r_x.shape[-2:]:
            mask = F.interpolate(mask, size=r_x.shape[-2:], mode="nearest")
        return mask.expand(r_x.shape[0], self.config.output_channels, *r_x.shape[-2:])

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, return_mask: bool = False):
        r_x = self.resized_input(x)
        mask = self.make_mask(r_x)
        if self.delta is None:
            out = r_x + mask
        else:
            out = r_x + self.delta * mask
        if return_mask:
            return out, mask
        return out


def build_input_reprogramming(
    image_size: Tuple[int, int],
    method: str,
    num_layers: int = 5,
    num_pool_layers: int = 3,
    **kwargs,
) -> InputReprogramming:
    """Factory used by the training scripts.

    ``method`` is one of ``smm``, ``full``, ``narrow``, ``medium``, ``pad``
    (baselines), ``only_delta`` (= ``full``), ``only_mask`` (sample-specific
    pattern without ``delta``) and ``single_channel`` (``f_mask^s``).
    """

    method = method.lower()
    if method == "smm":
        cfg = InputReprogrammingConfig(
            image_size=image_size, variant="smm", num_layers=num_layers,
            num_pool_layers=num_pool_layers, **kwargs,
        )
    elif method in {"full", "watermark", "only_delta"}:
        cfg = InputReprogrammingConfig(image_size=image_size, variant="shared", mask_kind="full", **kwargs)
    elif method == "narrow":
        cfg = InputReprogrammingConfig(image_size=image_size, variant="shared", mask_kind="narrow", **kwargs)
    elif method == "medium":
        cfg = InputReprogrammingConfig(image_size=image_size, variant="shared", mask_kind="medium", **kwargs)
    elif method == "pad":
        cfg = InputReprogrammingConfig(image_size=image_size, variant="shared", mask_kind="pad", **kwargs)
    elif method == "only_mask":
        cfg = InputReprogrammingConfig(
            image_size=image_size, variant="sample_specific_pattern", num_layers=num_layers,
            num_pool_layers=num_pool_layers, use_pattern=False, **kwargs,
        )
    elif method == "single_channel":
        cfg = InputReprogrammingConfig(
            image_size=image_size, variant="smm", num_layers=num_layers,
            num_pool_layers=num_pool_layers, single_channel=True, **kwargs,
        )
    else:
        raise ValueError(f"unknown reprogramming method {method!r}")
    return InputReprogramming(cfg)
