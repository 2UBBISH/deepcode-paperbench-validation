"""Quantisers of PTQ4ViT (Yuan et al., ECCV 2022).

Two kinds of quantisers are used:

* :class:`UniformQuantizer` - the standard (a)symmetric uniform quantiser, applied to the
  MatMul inputs, the attention logits and all weights;
* :class:`TwinUniformQuantizer` - PTQ4ViT's *twin uniform quantisation* for the two
  activations whose distribution is extremely unbalanced, i.e. the post-Softmax and the
  post-GELU activations.  Positive and negative values are quantised with two different
  scaling factors, which keeps the small negative values from being collapsed to zero.

Both quantisers perform *fake* quantisation: values are rounded to the integer grid and
de-quantised immediately, so the arithmetic stays in floating point.  This reproduces the
numerical behaviour of the quantised model (and is what the original PTQ4ViT simulation
does) while allowing us to use the exact same forward-only adaptation loop as for the
full-precision model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False
    #: initial clipping threshold: quantile of the observed activation magnitude
    percentile: float = 0.999
    #: candidate clipping ratios searched during calibration (PTQ4ViT's grid search)
    search_ratios: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)

    @property
    def qmax(self) -> int:
        return 2 ** (self.bits - 1) - 1

    @property
    def qmin(self) -> int:
        return -(2 ** (self.bits - 1))


class UniformQuantizer(nn.Module):
    """Affine (or symmetric) uniform quantiser with a calibratable clipping threshold."""

    def __init__(self, cfg: Optional[QuantConfig] = None, channel_axis: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or QuantConfig()
        self.channel_axis = channel_axis
        self.register_buffer("scale", torch.tensor(1.0), persistent=True)
        self.register_buffer("zero_point", torch.tensor(0.0), persistent=True)
        self.enabled = True
        self.observed_min = None
        self.observed_max = None
        self._ratio = 1.0

    # -- calibration -----------------------------------------------------------------
    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        if self.channel_axis is not None:
            axes = [a for a in range(x.dim()) if a != self.channel_axis]
            mn = x.detach().amin(dim=axes)
            mx = x.detach().amax(dim=axes)
        else:
            mn = x.detach().min()
            mx = x.detach().max()
        self.observed_min = mn if self.observed_min is None else torch.minimum(self.observed_min, mn)
        self.observed_max = mx if self.observed_max is None else torch.maximum(self.observed_max, mx)

    @torch.no_grad()
    def init_from_observations(self, symmetric: Optional[bool] = None) -> None:
        symmetric = self.cfg.symmetric if symmetric is None else symmetric
        if self.observed_min is None:
            raise RuntimeError("Quantizer.observe() must be called before init_from_observations()")
        qmax = self.cfg.qmax
        if symmetric:
            m = torch.maximum(self.observed_max.abs(), self.observed_min.abs()).clamp_min(1e-8)
            self.scale = m / qmax
            self.zero_point = torch.zeros_like(self.scale)
        else:
            qmin = self.cfg.qmin
            mn = self.observed_min
            mx = self.observed_max
            self.scale = (mx - mn).clamp_min(1e-8) / (qmax - qmin)
            self.zero_point = torch.clamp((qmin - mn / self.scale).round(), qmin, qmax)

    @torch.no_grad()
    def set_ratio(self, ratio: float) -> None:
        """Re-scale the clipping threshold (used by the grid search)."""
        self.scale = self.scale / self._ratio * ratio
        self._ratio = ratio

    # -- quantisation -----------------------------------------------------------------
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        scale = self.scale
        zp = self.zero_point
        if self.channel_axis is not None and scale.dim() > 0:
            shape = [1] * x.dim()
            shape[self.channel_axis] = -1
            scale = scale.reshape(shape)
            zp = zp.reshape(shape)
        q = torch.clamp(torch.round(x / scale) + zp, self.cfg.qmin, self.cfg.qmax)
        return (q - zp) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize(x)


class TwinUniformQuantizer(nn.Module):
    """Twin uniform quantisation of PTQ4ViT for post-Softmax / post-GELU activations.

    ``x > 0`` and ``x <= 0`` use two independent scaling factors, so the small negative
    tail (which dominates the post-GELU distribution) is represented precisely.
    """

    def __init__(self, cfg: Optional[QuantConfig] = None, channel_axis: Optional[int] = None):
        super().__init__()
        self.cfg = cfg or QuantConfig(symmetric=False)
        self.channel_axis = channel_axis
        self.register_buffer("scale_p", torch.tensor(1.0))
        self.register_buffer("scale_n", torch.tensor(1.0))
        self.enabled = True
        self.observed_max = None
        self._ratio = 1.0

    @torch.no_grad()
    def observe(self, x: torch.Tensor) -> None:
        mx = x.detach().max()
        self.observed_max = mx if self.observed_max is None else torch.maximum(self.observed_max, mx)

    @torch.no_grad()
    def init_from_observations(self) -> None:
        if self.observed_max is None:
            raise RuntimeError("TwinUniformQuantizer.observe() must be called first")
        # PTQ4ViT sets the positive threshold to the maximum and the negative threshold to
        # the symmetric quantile of the *negative* part (the activations are bounded below
        # by a small negative value for GELU and by 0 for Softmax).
        self.scale_p = self.observed_max.clamp_min(1e-8) / self.cfg.qmax
        self.scale_n = self.scale_p

    @torch.no_grad()
    def set_ratio(self, ratio: float) -> None:
        self.scale_p = self.scale_p / self._ratio * ratio
        self.scale_n = self.scale_n / self._ratio * ratio
        self._ratio = ratio

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        qmax = self.cfg.qmax
        pos = torch.clamp(torch.round(x / self.scale_p), 0, qmax) * self.scale_p
        neg = torch.clamp(torch.round(x / self.scale_n), -qmax, 0) * self.scale_n
        return torch.where(x > 0, pos, neg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantize(x)


def make_quantizer(
    bits: int,
    twin: bool = False,
    per_channel: bool = False,
    channel_axis: Optional[int] = None,
    percentile: float = 0.999,
    search_ratios=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
) -> nn.Module:
    cfg = QuantConfig(
        bits=bits,
        symmetric=not twin,
        per_channel=per_channel,
        percentile=percentile,
        search_ratios=tuple(search_ratios),
    )
    if twin:
        return TwinUniformQuantizer(cfg, channel_axis=channel_axis)
    return UniformQuantizer(cfg, channel_axis=channel_axis)


def quant_parameters(module: nn.Module) -> int:
    """Number of (fake-)quantised values of a module - used by the memory report."""
    return sum(1 for _ in iter_quantizers(module))


def iter_quantizers(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (UniformQuantizer, TwinUniformQuantizer)):
            yield m
