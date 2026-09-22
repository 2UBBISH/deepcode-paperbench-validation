"""
Latent Diffusion Model (LDM) backbone loading utilities for DPMs-ANT.

Paper reference (Rombach et al. 2022, "High-Resolution Image Synthesis with
Latent Diffusion Models"):

  * A frozen autoencoder (encoder E / decoder D) maps images to latents
    ``z = E(x)`` in a 64x64 (spatial) latent grid, and back to pixel space.
  * A frozen U-Net ``eps_theta(z_t, t)`` performs the diffusion denoising in
    latent space.  LDM-ANT inserts zero-initialised adaptors into the U-Net's
    residual "shift" module (the timestep-conditioned normalization stage),
    exactly as done for DDPM in :mod:`dpm_ant.models.unet_loader`, and only
    those adaptor parameters are trained.

This module provides two interchangeable code paths:

  1. If the official CompVis ``latent-diffusion`` / ``taming`` packages are
     importable (``ldm.modules.diffusionmodules.openaimodel`` and
     ``ldm.models.autoencoder``) they are used directly, so any released
     checkpoint loads without modification.
  2. Otherwise a faithful, self-contained replica of the same modules with
     identical parameter naming is used, so checkpoints still load.

The public interface mirrors :mod:`dpm_ant.models.unet_loader`:

    u = load_ldm_unet(ckpt, cfg, device)
    insert_adaptors_ldm(u.unet, adaptor_factory)
    u.freeze(); u.train_adaptors()
    eps = u.epsilon_theta(z_t, t)
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

__all__ = [
    # local replica modules
    "timestep_embedding",
    "nonlinearity",
    "zero_module",
    "conv_nd",
    "linear",
    "normalization",
    "avg_pool_nd",
    "GEGLU",
    "CrossAttention",
    "BasicTransformerBlock",
    "SpatialTransformer",
    "Upsample",
    "Downsample",
    "ResBlock",
    "AttentionBlock",
    "TimestepEmbedSequential",
    "UNetModel",
    "DiagonalGaussianDistribution",
    "Encoder",
    "Decoder",
    "AutoencoderKL",
    # wrappers / factories
    "FrozenLDMUNet",
    "FrozenLDMAutoencoder",
    "LDM_256_CONFIG",
    "LDM_AUTOENCODER_CONFIG",
    "build_ldm_unet",
    "load_ldm_unet",
    "load_autoencoder",
    "insert_adaptors_ldm",
    "iter_res_blocks",
    "count_parameters",
    "adaptor_parameters",
]


# ---------------------------------------------------------------------------
# Optional upstream CompVis modules (preferred when available).
# ---------------------------------------------------------------------------
_UPSTREAM_OPENAI = None
_UPSTREAM_AE = None
try:  # pragma: no cover - depends on environment
    from ldm.modules.diffusionmodules import openaimodel as _UPSTREAM_OPENAI  # type: ignore
except Exception:  # pragma: no cover
    _UPSTREAM_OPENAI = None
try:  # pragma: no cover - depends on environment
    from ldm.models import autoencoder as _UPSTREAM_AE  # type: ignore
except Exception:  # pragma: no cover
    _UPSTREAM_AE = None


# ---------------------------------------------------------------------------
# Basic helpers (identical semantics / naming to guided-diffusion & LDM).
# ---------------------------------------------------------------------------
def nonlinearity(x):
    """SiLU activation used throughout the LDM U-Net."""
    return x * torch.sigmoid(x)


def normalization(channels: int) -> nn.Module:
    """GroupNorm(32 groups, eps=1e-6, affine) as in LDM's ``Normalize``."""
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it (zero-init)."""
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def conv_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs) -> nn.Module:
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims: int, *args, **kwargs) -> nn.Module:
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Sinusoidal positional embedding of diffusion timesteps (1-indexed)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def _default(value, fallback):
    return fallback if value is None else value


# ---------------------------------------------------------------------------
# Local replica of the LDM spatial transformer stack
# ---------------------------------------------------------------------------
class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = _default(context_dim, query_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = _default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(*t.shape[:2], h, -1).permute(0, 2, 1, 3).reshape(-1, h, t.shape[-2], -1) if False else t, (q, k, v))

        q = q.reshape(q.shape[0], q.shape[1], h, -1).permute(0, 2, 1, 3)   # (b, h, n, d)
        k = k.reshape(k.shape[0], k.shape[1], h, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], h, -1).permute(0, 2, 1, 3)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        checkpoint: bool = True,
        disable_self_attn: bool = False,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, context_dim=None
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.disable_self_attn = disable_self_attn

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mult: int = 4, glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = _default(dim_out, dim)
        project_in = GEGLU(dim, inner_dim) if glu else nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class SpatialTransformer(nn.Module):
    """Transformer block operating on flattened spatial feature maps."""

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        disable_self_attn: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)
        if not use_linear_projection:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    disable_self_attn=disable_self_attn,
                )
                for _ in range(depth)
            ]
        )

        if not use_linear_projection:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear_projection = use_linear_projection

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear_projection:
            x = self.proj_in(x)
        else:
            x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context)
        if not self.use_linear_projection:
            x = self.proj_out(x)
        else:
            x = self.proj_out(x)
            x = x.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return x + x_in


# ---------------------------------------------------------------------------
# Local replica of the LDM U-Net
# ---------------------------------------------------------------------------
class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = _default(out_channels, self.channels)
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None):
        super().__init__()
        self.channels = channels
        self.out_channels = _default(out_channels, self.channels)
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class TimestepBlock(nn.Module):
    """Module taking ``(x, emb)`` (timestep embedding) as input."""

    def forward(self, x, emb):
        raise NotImplementedError


class ResBlock(TimestepBlock):
    """LDM residual block; with ``use_scale_shift_norm`` it owns the "shift"
    module into which DPMs-ANT adaptors are inserted."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
        use_fp16: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = _default(out_channels, channels)
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        if use_fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        # DPMs-ANT: zero-initialised adaptor attached to the shift module.
        self.adaptor: Optional[nn.Module] = None

    def forward(self, x, emb):
        h = x.type(self.dtype)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        out = self.skip_connection(x) + h
        if self.adaptor is not None:
            # psi^l(x^{l-1}) added to the frozen shift-module output.
            out = out + self.adaptor(x.to(self.adaptor_dtype()))
        return out

    def adaptor_dtype(self):
        for p in (self.adaptor.parameters() if self.adaptor is not None else []):
            return p.dtype
        return self.dtype


class AttentionBlock(nn.Module):
    """LDM self-attention block (supports both qkv-conv and linear-projection
    naming schemes so released checkpoints load)."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_new_attention_order: bool = False,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.use_linear_projection = use_linear_projection
        if use_linear_projection:
            self.to_q = linear(channels, channels, bias=False)
            self.to_k = linear(channels, channels, bias=False)
            self.to_v = linear(channels, channels, bias=False)
            self.proj_out = zero_module(linear(channels, channels))
        else:
            self.qkv = conv_nd(1, channels, channels * 3, 1)
            if use_new_attention_order:
                self.q, self.k, self.v = (
                    self.qkv.in_channels,
                    self.qkv.out_channels,
                    self.qkv.kernel_size,
                )
            self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.norm(x_flat)
        if self.use_linear_projection:
            q = self.to_q(qkv.permute(0, 2, 1))
            k = self.to_k(qkv.permute(0, 2, 1))
            v = self.to_v(qkv.permute(0, 2, 1))
            q, k, v = [t.permute(0, 2, 1) for t in (q, k, v)]
        else:
            qkv = self.qkv(qkv)
            q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, c // self.num_heads, -1).permute(0, 1, 3, 2)
        k = k.reshape(b, self.num_heads, c // self.num_heads, -1)
        v = v.reshape(b, self.num_heads, c // self.num_heads, -1).permute(0, 1, 3, 2)

        w = torch.einsum("bhdn,bhnm->bhdm", q, k) * (1.0 / math.sqrt(c // self.num_heads))
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        out = torch.einsum("bhdm,bhnm->bhdn", w, v)
        out = out.reshape(b, c, *spatial)
        if self.use_linear_projection:
            out = out.reshape(b, c, -1).permute(0, 2, 1)
            out = self.proj_out(out).permute(0, 2, 1).reshape(b, c, *spatial)
        else:
            out = self.proj_out(out.reshape(b, c, -1)).reshape(b, c, *spatial)
        return x + out


class TimestepEmbedSequential(nn.Sequential):
    """Sequential container passing the timestep embedding to ``TimestepBlock``
    and the (optional) context to spatial transformers."""

    def forward(  # type: ignore[override]
        self,
        x,
        emb=None,
        context=None,
    ):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    """LDM U-Net operating on latents (64x64 for the 256x256 f=4 autoencoder)."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        use_fp16: bool = False,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        use_linear_projection: bool = False,
        n_embed: Optional[int] = None,
        legacy: bool = True,
        disable_middle_self_attn: bool = False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = tuple(channel_mult)
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.use_spatial_transformer = use_spatial_transformer
        self.transformer_depth = transformer_depth
        self.context_dim = context_dim
        self.use_linear_projection = use_linear_projection
        self.legacy = legacy
        self.dtype = torch.float16 if use_fp16 else torch.float32

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: List[nn.Module] = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    if self.use_spatial_transformer and (
                        level in (0, 1, 2) or transformer_depth > 0
                    ):
                        dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads=num_heads_upsample if num_heads_upsample > 0 else max(1, num_heads),
                                d_head=dim_head,
                                depth=1,
                                dropout=dropout,
                                context_dim=context_dim,
                                use_linear_projection=use_linear_projection,
                                disable_self_attn=disable_middle_self_attn,
                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                use_linear_projection=use_linear_projection,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if self.use_spatial_transformer:
            dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                SpatialTransformer(
                    ch,
                    num_heads_upsample if num_heads_upsample > 0 else max(1, num_heads),
                    d_head=dim_head,
                    depth=1,
                    dropout=dropout,
                    context_dim=context_dim,
                    use_linear_projection=use_linear_projection,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
        else:
            self.middle_block = TimestepEmbedSequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads_upsample,
                    num_head_channels=num_head_channels,
                    use_new_attention_order=use_new_attention_order,
                    use_linear_projection=use_linear_projection,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                ),
            )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    if self.use_spatial_transformer:
                        dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
                        layers.append(
                            SpatialTransformer(
                                ch,
                                num_heads_upsample if num_heads_upsample > 0 else max(1, num_heads),
                                d_head=dim_head,
                                depth=1,
                                dropout=dropout,
                                context_dim=context_dim,
                                use_linear_projection=use_linear_projection,
                                disable_self_attn=disable_middle_self_attn,
                            )
                        )
                    else:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                use_linear_projection=use_linear_projection,
                            )
                        )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y is not None and y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)


# ---------------------------------------------------------------------------
# Local replica of the LDM autoencoder (AutoencoderKL)
# ---------------------------------------------------------------------------
class DiagonalGaussianDistribution:
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self) -> torch.Tensor:
        return self.mean + self.std * torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype)

    def mode(self) -> torch.Tensor:
        return self.mean


def _make_attn(in_channels, attn_type="vanilla", num_layers=1, num_heads=1):
    return AttentionBlock(in_channels, num_heads=num_heads)


def _make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


class Encoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (),
        z_channels: int = 4,
        resolution: int = 256,
        in_channels: int = 3,
        double_z: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = _make_conv(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_make_attn(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _make_attn(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.norm_out = normalization(block_in)
        self.conv_out = _make_conv(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (),
        z_channels: int = 4,
        resolution: int = 256,
        in_channels: int = 3,
        give_pre_end: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = _make_conv(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _make_attn(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(_make_attn(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = normalization(block_in)
        self.conv_out = _make_conv(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        out_channels = _default(out_channels, in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = normalization(in_channels)
        self.conv1 = _make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalization(out_channels)
        self.conv2 = _make_conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.nin_shortcut = _make_conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        ddconfig: Optional[Dict[str, Any]] = None,
        embed_dim: int = 4,
        double_z: bool = True,
    ):
        super().__init__()
        ddconfig = dict(ddconfig or {})
        double_z = ddconfig.pop("double_z", double_z)
        self.encoder = Encoder(double_z=double_z, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        self.ddconfig = ddconfig

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)


# ---------------------------------------------------------------------------
# Default configurations (256x256 pixel space, 64x64 latent grid, f=4).
# ---------------------------------------------------------------------------
LDM_256_CONFIG: Dict[str, Any] = dict(
    image_size=64,                 # latent spatial size
    in_channels=4,
    model_channels=320,
    out_channels=4,
    num_res_blocks=2,
    attention_resolutions=[4, 2, 1],
    dropout=0.0,
    channel_mult=[1, 2, 4],
    conv_resample=True,
    dims=2,
    num_classes=None,
    use_checkpoint=False,
    num_heads=8,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    resblock_updown=True,
    use_new_attention_order=False,
    use_fp16=False,
    use_spatial_transformer=True,
    transformer_depth=1,
    context_dim=None,
    use_linear_projection=False,
    n_embed=None,
    legacy=True,
)

LDM_AUTOCODER_CONFIG: Dict[str, Any] = dict(
    embed_dim=4,
    double_z=True,
    ddconfig=dict(
        double_z=True,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_resolutions=[],
        dropout=0.0,
    ),
)


# ---------------------------------------------------------------------------
# Building / loading helpers
# ---------------------------------------------------------------------------
def _merged_config(cfg: Optional[Dict[str, Any]] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = dict(LDM_256_CONFIG)
    if cfg:
        # Accept either the flat config or a nested {'models': {'ldm': {...}}} one.
        flat = cfg
        if "models" in cfg and isinstance(cfg["models"], dict):
            flat = dict(cfg["models"].get("ldm", {}))
        for k, v in flat.items():
            if k in merged and v is not None:
                merged[k] = v
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v
    return merged


def build_ldm_unet(cfg: Optional[Dict[str, Any]] = None, **overrides) -> nn.Module:
    """Instantiate an LDM U-Net (upstream class when available)."""
    merged = _merged_config(cfg, overrides)
    cls = UNetModel
    if _UPSTREAM_OPENAI is not None and hasattr(_UPSTREAM_OPENAI, "UNetModel"):
        cls = _UPSTREAM_OPENAI.UNetModel  # type: ignore[assignment]
    return cls(**merged)


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "state_dict."):
            while nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_ema", "weights"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _strip_prefixes(checkpoint[key])
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return _strip_prefixes(checkpoint)
    raise ValueError("could not locate a state dict inside the checkpoint")


def load_ldm_unet(
    checkpoint: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
    **overrides,
) -> UNetModel:
    """Build an LDM U-Net and optionally load a checkpoint.

    The returned module still owns its pretrained weights; freezing and adaptor
    handling is performed by :class:`FrozenLDMUNet`.
    """
    model = build_ldm_unet(cfg, **overrides)
    if checkpoint is not None and os.path.isfile(checkpoint):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
        except Exception:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _extract_state_dict(ckpt)
        # keep only U-Net keys (a full LDM checkpoint also stores the VAE).
        model_keys = set(model.state_dict().keys())
        state = {k: v for k, v in state.items() if k in model_keys or f"{k}" in model_keys}
        # drop adaptor keys (absent from the pretrained checkpoint anyway)
        state = {k: v for k, v in state.items() if ".adaptor." not in k}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if verbose:
            print(
                f"[ldm_loader] loaded U-Net checkpoint '{checkpoint}' "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if missing and strict:
                raise RuntimeError(f"missing keys while loading LDM U-Net: {missing[:8]}")
            if missing and verbose:
                print(f"[ldm_loader] first missing keys: {missing[:8]}")
    elif checkpoint is not None and verbose:
        print(f"[ldm_loader] checkpoint '{checkpoint}' not found; using randomly initialised U-Net.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def _resolve_autoencoder_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in LDM_AUTOCODER_CONFIG.items()}
    if cfg:
        flat = cfg.get("models", {}).get("ldm", cfg) if "models" in cfg else cfg
        for key in ("embed_dim", "double_z", "ddconfig"):
            if key in flat and flat[key] is not None:
                if key == "ddconfig" and isinstance(flat[key], dict):
                    merged["ddconfig"] = {**merged["ddconfig"], **flat[key]}
                else:
                    merged[key] = flat[key]
    return merged


def load_autoencoder(
    checkpoint: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> nn.Module:
    """Load the frozen LDM autoencoder (upstream ``AutoencoderKL`` if available)."""
    acfg = _resolve_autoencoder_cfg(cfg)
    cls = AutoencoderKL
    if _UPSTREAM_AE is not None and hasattr(_UPSTREAM_AE, "AutoencoderKL"):
        cls = _UPSTREAM_AE.AutoencoderKL  # type: ignore[assignment]
    model = cls(embed_dim=acfg["embed_dim"], ddconfig=acfg["ddconfig"])
    if checkpoint is not None and os.path.isfile(checkpoint):
        try:
            ckpt = torch.load(checkpoint, map_location="cpu")
        except Exception:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = _extract_state_dict(ckpt)
        # A full LDM checkpoint stores the VAE under 'first_stage_model.'.
        vae_state = {}
        for k, v in state.items():
            for prefix in ("first_stage_model.", "vae."):
                if k.startswith(prefix):
                    vae_state[k[len(prefix):]] = v
        if not vae_state:
            vae_state = state
        missing, unexpected = model.load_state_dict(vae_state, strict=False)
        if verbose:
            print(
                f"[ldm_loader] loaded autoencoder checkpoint '{checkpoint}' "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
    elif checkpoint is not None and verbose:
        print(f"[ldm_loader] autoencoder checkpoint '{checkpoint}' not found; using random init.")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


# ---------------------------------------------------------------------------
# Adaptor insertion into the U-Net "shift module"
# ---------------------------------------------------------------------------
def iter_res_blocks(module: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """Yield ``(name, resblock)`` for every residual block in the U-Net."""
    for name, child in module.named_modules():
        if isinstance(child, ResBlock) or child.__class__.__name__ == "ResBlock":
            yield name, child


def _is_shift_block(block: nn.Module) -> bool:
    return bool(getattr(block, "use_scale_shift_norm", False))


def insert_adaptors_ldm(
    unet: nn.Module,
    adaptor_factory,
    shift_only: bool = True,
    verbose: bool = True,
) -> List[nn.Module]:
    """Insert zero-initialised adaptors into the U-Net shift modules.

    Parameters
    ----------
    unet:
        The LDM U-Net (any module owning ``ResBlock``s).
    adaptor_factory:
        Callable ``(in_channels, out_channels, name) -> nn.Module`` returning a
        zero-initialised adaptor.  See :mod:`dpm_ant.models.adaptor`.
    shift_only:
        Only attach to timestep-conditioned scale/shift residual blocks
        (paper's "shift module"); otherwise attach to every block.
    """
    inserted: List[nn.Module] = []
    for name, block in iter_res_blocks(unet):
        if shift_only and not _is_shift_block(block):
            continue
        in_ch = int(block.in_layers[-1].in_channels)
        out_ch = int(getattr(block, "out_channels", in_ch))
        adaptor = adaptor_factory(in_ch, out_ch, name)
        # The adaptor acts on the block input; match output channel count.
        if getattr(adaptor, "out_channels", None) is None and hasattr(adaptor, "attn"):
            pass
        if out_ch != in_ch and not getattr(adaptor, "projects_channels", False):
            # Wrap with a 1x1 projection so shapes always match.
            proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            adaptor = nn.Sequential(adaptor, proj)
        block.adaptor = adaptor
        inserted.append(adaptor)
    if verbose:
        print(f"[ldm_loader] inserted {len(inserted)} adaptors into U-Net shift modules")
    return inserted


def count_parameters(module: nn.Module, only_trainable: bool = False) -> int:
    params = module.parameters()
    if only_trainable:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def adaptor_parameters(unet: nn.Module) -> List[nn.Parameter]:
    """Return adaptor-only parameters of an adapted U-Net."""
    return [p for n, p in unet.named_parameters() if ".adaptor." in n or n.endswith("adaptor")]


# ---------------------------------------------------------------------------
# Frozen wrappers
# ---------------------------------------------------------------------------
class FrozenLDMUNet(nn.Module):
    """Frozen LDM U-Net with a uniform ``epsilon_theta(x_t, t)`` interface.

    ``epsilon_theta`` behaves exactly like the DDPM counterpart in
    :mod:`dpm_ant.models.unet_loader`, so ANT training/sampling code is
    backbone-agnostic.
    """

    def __init__(
        self,
        unet: nn.Module,
        context_dim: Optional[int] = None,
        scale_factor: float = 0.18215,
        freeze: bool = True,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.unet = unet
        self.context_dim = context_dim
        self.scale_factor = scale_factor
        self.learn_sigma = learn_sigma
        if freeze:
            self.freeze()

    # -- freezing -----------------------------------------------------------
    def freeze(self) -> "FrozenLDMUNet":
        for n, p in self.unet.named_parameters():
            p.requires_grad_((".adaptor." in n or n.endswith("adaptor")))
        self.unet.eval()
        return self

    def train_adaptors(self) -> "FrozenLDMUNet":
        for n, p in self.unet.named_parameters():
            p.requires_grad_((".adaptor." in n or n.endswith("adaptor")))
        for m in self.unet.modules():
            if isinstance(m, nn.Dropout):
                m.eval()
        return self

    # -- forward ------------------------------------------------------------
    def _ctx(self, batch_size: int, device, context: Optional[torch.Tensor] = None):
        if context is not None:
            return context
        if self.context_dim is None:
            return None
        # unconditional: zero context (classifier-free style null embedding)
        return torch.zeros(batch_size, 1, self.context_dim, device=device)

    def epsilon_theta(self, x_t: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        t = t.reshape(-1).to(x_t.device)
        out = self.unet(x_t, t, context=self._ctx(x_t.shape[0], x_t.device, context))
        if self.learn_sigma:
            eps, _rest = torch.split(out, out.shape[1] // 2, dim=1)
            return eps
        return out

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.epsilon_theta(x_t, t, context=context)

    # -- statistics ---------------------------------------------------------
    def adaptor_parameters(self) -> List[nn.Parameter]:
        return adaptor_parameters(self.unet)

    def param_rate(self) -> float:
        total = count_parameters(self.unet)
        adapt = count_parameters(self.unet, only_trainable=True)
        return float(adapt) / float(max(total, 1))


class FrozenLDMAutoencoder(nn.Module):
    """Frozen LDM autoencoder providing pixel<->latent conversion."""

    def __init__(self, vae: nn.Module, scale_factor: float = 0.18215):
        super().__init__()
        self.vae = vae
        self.scale_factor = scale_factor
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor, sample: bool = False) -> torch.Tensor:
        posterior = self.vae.encode(images)
        z = posterior.sample() if sample else (posterior.mode() if hasattr(posterior, "mode") else posterior.mean)
        return z * self.scale_factor

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents / self.scale_factor)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)
