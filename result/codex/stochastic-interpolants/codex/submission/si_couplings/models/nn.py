"""Building blocks of the U-Net velocity model (Appendix B).

The architecture is the DDPM U-Net of (Ho et al., 2020b) as implemented in
the ``denoising-diffusion-pytorch`` repository, which is the implementation
the paper used ("For the velocity model we use the U-net from (Ho et al.,
2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository;
this variant of the architecture includes embeddings to condition on class
labels").  The hyper-parameters of Appendix B map onto this file as follows:

    Dim Mults: (1, 1, 2, 3, 4)          -> ``Unet(dim_mults=...)``
    Dim (channels): 256                -> ``Unet(dim=256)``
    Resnet block groups: 8             -> ``resnet_block_groups=8``
                                          (the number of groups of the
                                          GroupNorm inside the ResNet blocks)
    Learned Sinusoidal Cond: True      -> ``learned_sinusoidal_cond=True``
    Learned Sinusoidal Dim: 32         -> ``learned_sinusoidal_dim=32``
    Attention Dim Head: 64             -> ``attn_dim_head=64``
    Attention Heads: 4                 -> ``attn_heads=4``
    Random Fourier Features: False     -> ``random_fourier_features=False``
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d() if callable(d) else d


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module, groups: int = 8):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(groups, dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal time embedding."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """Learned (or random fixed) Fourier features of the time variable.

    ``is_random=False`` corresponds to "Learned Sinusoidal Cond: True" in
    Appendix B; the raw scalar t is concatenated to its Fourier features,
    which is why the output dimension is ``dim + 1``.
    """

    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        assert dim % 2 == 0, "dimension must be divisible by 2"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: Tensor) -> Tensor:
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class WeightStandardizedConv2d(nn.Conv2d):
    """Convolution with weight standardisation (as in the reference U-Net)."""

    def forward(self, x: Tensor) -> Tensor:
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        var = weight.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, scale_shift=None) -> Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """ResNet block with a (scale, shift) modulation from the time embedding."""

    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: Tensor, time_emb: Optional[Tensor] = None) -> Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)[:, :, None, None]
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    """Full (quadratic) multi-head self-attention over the spatial grid."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads
        self.norm = nn.GroupNorm(groups, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        def split(t: Tensor) -> Tensor:
            return t.reshape(b, self.heads, -1, h * w)

        q, k, v = map(split, (q, k, v))
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)  # (b, heads, hw, dim_head)
        dim_head = out.shape[-1]
        out = out.permute(0, 1, 3, 2).reshape(b, self.heads * dim_head, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """Linear attention, inserted at every resolution in the reference U-Net."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32, groups: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        hidden_dim = dim_head * heads
        self.norm = nn.GroupNorm(groups, dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        def split(t: Tensor) -> Tensor:
            return t.reshape(b, self.heads, -1, h * w)

        q, k, v = map(split, (q, k, v))
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, self.heads * out.shape[2], h, w)
        return self.to_out(out)


class Upsample(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, default(dim_out, dim), 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None):
        super().__init__()
        self.conv = nn.Conv2d(dim, default(dim_out, dim), 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


__all__ = [
    "exists",
    "default",
    "Residual",
    "PreNorm",
    "SinusoidalPosEmb",
    "RandomOrLearnedSinusoidalPosEmb",
    "WeightStandardizedConv2d",
    "Block",
    "ResnetBlock",
    "Attention",
    "LinearAttention",
    "Upsample",
    "Downsample",
]
