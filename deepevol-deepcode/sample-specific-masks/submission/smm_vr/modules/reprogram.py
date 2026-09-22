"""SMM reprogramming wrapper (:math:`f_{\\text{in}}`).

This module implements the core reprogramming function of *Sample-specific
Multi-channel Masks for Visual Reprogramming* (SMM, ICML 2024):

.. math::

    f_{\\text{in}}\\left(x_{i} \\mid \\phi, \\delta\\right)
        = r\\left(x_{i}\\right)
          + \\delta \\odot f_{\\text{mask}}\\left(r\\left(x_{i}\\right) \\mid \\phi\\right)

(paper Eq. (1)/(4)), together with the training objective

.. math::

    \\arg\\min_{\\phi \\in \\Phi,\\, \\delta \\in \\mathbb{R}^{d_{P}}}
        \\mathbb{E}_{\\left(x_{i}, y_{i}\\right) \\sim \\mathcal{D}_{T}}
        \\left[ \\ell \\left( f_{\\text{out}}\\left(
            f_{P}\\left( r(x_{i}) + \\delta \\odot f_{\\text{mask}}(r(x_{i}) \\mid \\phi) \\right)
        \\right), y_{i} \\right) \\right]

and the Algorithm 1 initialization: :math:`\\phi` is drawn randomly while
:math:`\\delta \\leftarrow \\{0\\}^{d_{P}}` (an all-zero tensor with the shape of
the pre-trained model input).

Following the paper:

* :math:`\\delta` is **shared** by all images of the dataset (following Bahng et
  al., 2022 and Chen et al., 2023),
* :math:`f_{\\text{mask}}` produces a **sample-specific multi-channel** mask
  (3 channels, matching the RGB input),
* the low-resolution CNN output of :math:`f_{\\text{mask}}` is enlarged with the
  patch-wise interpolation module, which *"omits the derivation step in
  back-propagation"* (Section 3.3).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..models.mask_generator import MaskGenerator, build_mask_generator
from .patch_interp import PatchWiseInterpolation

__all__ = [
    "SMMReprogram",
    "SMMReprogramming",
    "SharedMaskReprogram",
    "build_smm_reprogram",
    "init_zero_pattern",
    "DEFAULT_PATCH_SIZE",
]


DEFAULT_PATCH_SIZE = 8
"""Default patch size :math:`2^{l}` for the patch-wise interpolation module."""


def init_zero_pattern(
    channels: int = 3,
    height: int = 224,
    width: int = 224,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Return the all-zero learnable noise pattern :math:`\\delta \\leftarrow \\{0\\}^{d_{P}}`.

    Algorithm 1 of the paper states: *"Initialize phi randomly; set delta <-
    {0}^{d_P}"*; Section 3.4 adds *"To mitigate the impact of initialization,
    delta is set to be a zero matrix before training"*.
    """
    return torch.zeros((channels, height, width), device=device, dtype=dtype)


def _resolve_mask_generator(mask_generator, backbone: str, patch_size: int, **kwargs):
    """Normalize the many accepted ways of specifying :math:`f_{\\text{mask}}`."""
    if mask_generator is None:
        mask_generator = build_mask_generator(backbone, **kwargs)
    elif isinstance(mask_generator, str):
        mask_generator = build_mask_generator(mask_generator, **kwargs)
    elif not isinstance(mask_generator, nn.Module):
        raise TypeError(
            "mask_generator must be None, a backbone name or an nn.Module, "
            f"got {type(mask_generator)!r}"
        )
    return mask_generator


class SMMReprogram(nn.Module):
    """Sample-specific Multi-channel Mask (SMM) reprogramming function.

    Given a *target-domain* image tensor :math:`x_i` (already resized by the
    dataset transform pipeline, i.e. :math:`r(x_i)` — see
    :mod:`smm_vr.data.transforms`) this module computes

    ``f_in(x_i) = r(x_i) + delta * f_mask(r(x_i))``

    which is the input handed to the frozen pre-trained classifier
    :math:`f_{P}`.

    Parameters
    ----------
    mask_generator:
        The lightweight CNN :math:`f_{\\text{mask}}`. If ``None`` (or a backbone
        name string), it is built with :func:`build_mask_generator`.
    input_size:
        Spatial size of the pre-trained model input, ``224`` for ResNet-18 /
        ResNet-50 and ``384`` for ViT-B/32.
    patch_size:
        Patch size :math:`2^l` used by the patch-wise interpolation module
        (default ``8``, i.e. :math:`l=3`).
    backbone:
        Backbone name used to build the default mask generator
        (``"resnet18"``, ``"resnet50"``, ``"vit_b32"``).
    delta_init:
        How the shared pattern is initialized. Algorithm 1 fixes ``"zero"``.
    normalize_mask:
        Optional switch; when ``True`` the multi-channel mask is passed through
        ``tanh`` so that the perturbation is bounded. The paper does not mention
        such a squashing function and the default is therefore ``False``.
    """

    def __init__(
        self,
        mask_generator=None,
        input_size: int = 224,
        patch_size: int = DEFAULT_PATCH_SIZE,
        backbone: str = "resnet18",
        in_channels: int = 3,
        delta_init: str = "zero",
        normalize_mask: bool = False,
        **mask_generator_kwargs,
    ) -> None:
        super().__init__()

        self.input_size = int(input_size)
        self.in_channels = int(in_channels)
        self.backbone = backbone
        self.normalize_mask = bool(normalize_mask)

        # --- f_mask ------------------------------------------------------
        self.mask_generator: MaskGenerator = _resolve_mask_generator(
            mask_generator, backbone, patch_size, **mask_generator_kwargs
        )
        if not isinstance(self.mask_generator, nn.Module):
            raise TypeError("mask_generator must be an nn.Module")

        # --- patch-wise interpolation ------------------------------------
        self.patch_interp = PatchWiseInterpolation(patch_size=int(patch_size))
        self.patch_size = int(patch_size)

        # --- shared learnable pattern delta (paper Eq. (4)) --------------
        delta_shape = (self.in_channels, self.input_size, self.input_size)
        if delta_init == "zero":
            delta = init_zero_pattern(*delta_shape)
        elif delta_init == "random":
            delta = torch.randn(*delta_shape) * 0.001
        else:
            raise ValueError(f"Unsupported delta_init={delta_init!r}")
        # delta is a single shared tensor for the whole dataset, not a batch of
        # per-sample patterns (paper Section 3.1).
        self.delta = nn.Parameter(delta)

        self._init_mask_generator_weights()

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _init_mask_generator_weights(self) -> None:
        """Random initialization of :math:`\\phi` (Algorithm 1: "Initialize phi randomly")."""
        for module in self.mask_generator.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def num_pooling_layers(self) -> int:
        """Number :math:`l` of 2x2 max-pooling layers inside :math:`f_{\\text{mask}}`."""
        return int(self.mask_generator.num_pooling)

    def low_res_mask_size(self, in_size: Optional[int] = None) -> int:
        """Spatial size :math:`\\lfloor H/2^{l} \\rfloor` of the CNN mask output."""
        size = self.input_size if in_size is None else int(in_size)
        return max(1, size // (2 ** self.num_pooling_layers))

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def mask(self, images: torch.Tensor) -> torch.Tensor:
        """Compute the sample-specific full-resolution mask for a batch.

        Returns a tensor with the same shape as ``images``.
        """
        low_res = self.mask_generator(images)
        mask = self.patch_interp(low_res, out_size=images.shape[-2:])
        if self.normalize_mask:
            mask = torch.tanh(mask)
        return mask

    def forward(
        self,
        images: torch.Tensor,
        return_mask: bool = False,
    ):
        """Apply :math:`f_{\\text{in}}` (paper Eq. (1)).

        Parameters
        ----------
        images:
            Resized target-domain batch :math:`r(x_i)` of shape ``(B, C, H, W)``.
        return_mask:
            When ``True`` also return the sample-specific masks, which is handy
            for the mask-visualization / ablation code paths.

        Returns
        -------
        torch.Tensor or (torch.Tensor, torch.Tensor)
            The reprogrammed batch and optionally its masks.
        """
        if images.dim() != 4:
            raise ValueError(
                f"SMMReprogram expects a 4D (B,C,H,W) batch, got shape {tuple(images.shape)}"
            )

        mask = self.mask(images)
        delta = self.delta.to(dtype=images.dtype, device=images.device)
        if delta.shape[-2:] != images.shape[-2:]:
            raise ValueError(
                "delta spatial shape "
                f"{tuple(delta.shape[-2:])} does not match the input "
                f"{tuple(images.shape[-2:])}; build the reprogrammer with the "
                "right input_size."
            )
        reprogrammed = images + delta.unsqueeze(0) * mask
        if return_mask:
            return reprogrammed, mask
        return reprogrammed

    def extra_repr(self) -> str:
        return (
            f"backbone={self.backbone}, input_size={self.input_size}, "
            f"patch_size={self.patch_size}, "
            f"mask_params={sum(p.numel() for p in self.mask_generator.parameters())}, "
            f"delta_shape={tuple(self.delta.shape)}"
        )


# Convenient alias matching the paper's ``f_in`` naming convention.
SMMReprogramming = SMMReprogram


class SharedMaskReprogram(nn.Module):
    """Shared-mask VR baseline reprogramming used for Pad / Narrow / Medium / Full.

    These baselines follow Section 5 *Baselines*: a pre-determined, *shared*
    binary mask :math:`m` still multiplies the trainable pattern, but the mask no
    longer depends on the sample:

    ``f_in(x_i) = r(x_i) + delta * m``

    Parameters
    ----------
    input_size:
        Spatial size of the pre-trained model input (224 for ResNets, 384 for ViT).
    mask:
        Optional ``(1, C, H, W)`` or ``(C, H, W)`` mask tensor. When ``None`` an
        all-one mask is used (which degenerates to the Full / only-delta case).
    """

    def __init__(
        self,
        input_size: int = 224,
        in_channels: int = 3,
        mask: Optional[torch.Tensor] = None,
        delta_init: str = "zero",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.in_channels = int(in_channels)

        if mask is None:
            mask = torch.ones((1, self.in_channels, self.input_size, self.input_size))
        mask = torch.as_tensor(mask, dtype=torch.float32)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        self.register_buffer("mask_indicator", mask)

        if delta_init == "zero":
            delta = init_zero_pattern(self.in_channels, self.input_size, self.input_size)
        elif delta_init == "random":
            delta = torch.randn(self.in_channels, self.input_size, self.input_size) * 0.001
        else:
            raise ValueError(f"Unsupported delta_init={delta_init!r}")
        self.delta = nn.Parameter(delta)

    def forward(self, images: torch.Tensor, return_mask: bool = False):
        delta = self.delta.to(dtype=images.dtype, device=images.device)
        mask = self.mask_indicator.to(dtype=images.dtype, device=images.device)
        reprogrammed = images + delta.unsqueeze(0) * mask
        if return_mask:
            return reprogrammed, mask
        return reprogrammed


def build_smm_reprogram(
    backbone: str = "resnet18",
    input_size: Optional[int] = None,
    patch_size: int = DEFAULT_PATCH_SIZE,
    **kwargs,
) -> SMMReprogram:
    """Build an :class:`SMMReprogram` with backbone-appropriate defaults.

    The input size defaults to ``384`` for ViT-B/32 and ``224`` otherwise
    (Section 5, *Pre-trained Models and Target Tasks* and Appendix E.1).
    """
    if input_size is None:
        input_size = 384 if str(backbone).lower().startswith("vit") else 224
    return SMMReprogram(
        mask_generator=None,
        input_size=int(input_size),
        patch_size=int(patch_size),
        backbone=backbone,
        **kwargs,
    )
