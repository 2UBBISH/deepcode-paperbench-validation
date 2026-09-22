"""Frozen DDPM (guided-diffusion) U-Net backbone for DPMs-ANT.

This module implements a self-contained replica of the OpenAI *guided-diffusion*
256x256 U-Net (Ho et al. 2020; Dhariwal & Nichol 2021) together with utilities
to

* build the architecture from a config dict (``build_unet_model``),
* load the released checkpoints with tolerant key remapping
  (``load_guided_diffusion_unet``),
* freeze every pre-trained parameter (``FrozenDDPMUNet.freeze``),
* insert zero-initialised adaptors into the U-Net *shift module* only
  (``insert_adaptors`` / ``attach_adaptors_via_hooks``), and
* expose the uniform interface ``epsilon_theta(x_t, t)`` required by the ANT
  training/sampling code.

Paper references
----------------
* §3 Preliminary        -- DDPM forward/reverse process, ``eps_theta(x_t, t)``.
* §5.2 Configurations   -- "We restrict our fine-tuning to the shift module of
  the U-Net, maintaining the pre-trained DPMs ... as they are."
* §4.3                  -- ``x_t^l = theta^l(x_t^{l-1}) + psi^l(x_t^{l-1})``.

The architecture reproduces the guided-diffusion ``UNetModel`` *exactly* (same
module names, same parameter shapes) so that released ``256x256_diffusion`` and
``256x256_classifier`` state dicts load without modification.  The only
addition is an optional ``adaptor`` attribute on every ``ResBlock`` which is
evaluated as ``h + psi(x)`` (zero-initialised, hence a no-op at init).
"""

from __future__ import annotations

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "UNetModel",
    "ResBlock",
    "AttentionBlock",
    "TimestepEmbedSequential",
    "FrozenDDPMUNet",
    "UNET_256_CONFIG",
    "build_unet_model",
    "load_guided_diffusion_unet",
    "insert_adaptors",
    "attach_adaptors_via_hooks",
    "iter_res_blocks",
    "count_parameters",
]


# ---------------------------------------------------------------------------
# guided-diffusion shared building blocks
# ---------------------------------------------------------------------------
def normalization(channels: int) -> nn.Module:
    """GroupNorm(32, channels) as used by guided-diffusion."""
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-5, affine=True)


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


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: float = 10000.0) -> th.Tensor:
    """Sinusoidal positional embedding of the diffusion timestep."""
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32, device=timesteps.device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class SiLU(nn.Module):
    """Export-friendly swish activation (``nn.SiLU``)."""

    @staticmethod
    def forward(x: th.Tensor) -> th.Tensor:
        return x * th.sigmoid(x)


class TimestepBlock(nn.Module):
    """Marker base class: module whose ``forward`` takes ``(x, emb)``."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential that dispatches ``emb`` to modules deriving TimestepBlock."""

    def forward(self, x: th.Tensor, emb: th.Tensor):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, dims: int = 2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x: th.Tensor) -> th.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """Residual block of the U-Net.

    Implements exactly the guided-diffusion block; the *shift module* is the
    ``use_scale_shift_norm`` conditioning branch.  When ``self.adaptor`` is set
    the block computes ``theta(x, emb) + psi(x)`` (§4.3).
    """

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
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.dims = dims
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32

        # `adaptor` is registered lazily by `insert_adaptors`; kept out of the
        # state dict so pre-trained checkpoints load cleanly.
        self.adaptor: Optional[nn.Module] = None

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
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
            SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        h = x.type(self.dtype)
        h = self.in_layers(h)
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            # ---- shift module ------------------------------------------------
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        # ---- adaptor branch: x^l = theta^l(x^{l-1}) + psi^l(x^{l-1}) --------
        if self.adaptor is not None:
            h = h + self.adaptor(x)
        return self.skip_connection(x) + h


class QKVAttentionLegacy(nn.Module):
    """Legacy attention order (``use_new_attention_order=False``)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    """New attention order (``use_new_attention_order=True``)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Self-attention over spatial positions (guided-diffusion version)."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"qkv_channels {channels} should be divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            self.attention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x).type(self.dtype))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
UNET_256_CONFIG: Dict[str, object] = {
    "image_size": 256,
    "in_channels": 3,
    "model_channels": 256,
    "out_channels": 6,
    "num_res_blocks": 2,
    "attention_resolutions": (32, 16, 8),
    "dropout": 0.0,
    "channel_mult": (1, 2, 4, 8),
    "conv_resample": True,
    "dims": 2,
    "num_classes": None,
    "use_checkpoint": False,
    "num_heads": 4,
    "num_head_channels": 64,
    "use_scale_shift_norm": True,
    "resblock_updown": True,
    "use_fp16": False,
    "use_new_attention_order": False,
}


class UNetModel(nn.Module):
    """Guided-diffusion U-Net predicting ``eps_theta(x_t, t)``."""

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
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_fp16: bool = False,
        use_new_attention_order: bool = False,
        **kwargs,
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
        self.use_scale_shift_norm = use_scale_shift_norm
        self.resblock_updown = resblock_updown
        self.use_fp16 = use_fp16
        self.dtype = th.float16 if use_fp16 else th.float32
        self.learn_sigma = out_channels == 2 * in_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
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
                        use_fp16=use_fp16,
                    )
                ]
                ch = int(mult * model_channels)
                if image_size // ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            use_fp16=use_fp16,
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
                            use_fp16=use_fp16,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_fp16=use_fp16,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                use_fp16=use_fp16,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_fp16=use_fp16,
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
                        use_fp16=use_fp16,
                    )
                ]
                ch = int(model_channels * mult)
                if image_size // ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            use_fp16=use_fp16,
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
                            use_fp16=use_fp16,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    # -- guided-diffusion API -------------------------------------------------
    def convert_to_fp16(self) -> None:
        self.use_fp16 = True
        self.dtype = th.float16
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        self.use_fp16 = False
        self.dtype = th.float32
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x: th.Tensor, timesteps: th.Tensor, y: Optional[th.Tensor] = None) -> th.Tensor:
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs: List[th.Tensor] = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y is not None and y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


def convert_module_to_f16(l: nn.Module) -> None:
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l: nn.Module) -> None:
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


# ---------------------------------------------------------------------------
# construction / loading
# ---------------------------------------------------------------------------
def build_unet_model(cfg: Optional[Dict] = None, **overrides) -> UNetModel:
    """Build a guided-diffusion U-Net from a (possibly nested) config dict."""
    cfg = dict(cfg or {})
    for key, value in cfg.pop("model", {}).items() if "model" in cfg else []:
        cfg[key] = value
    cfg.update(overrides)

    known = dict(UNET_256_CONFIG)
    for key in list(cfg.keys()):
        if key in known:
            known[key] = cfg[key]
    # tolerate string attention resolutions, e.g. "32,16,8"
    ar = known["attention_resolutions"]
    if isinstance(ar, str):
        known["attention_resolutions"] = tuple(int(v) for v in ar.replace(" ", "").split(",") if v)
    if known.get("out_channels") == 0:  # sentinel: derive from learn_sigma
        known["out_channels"] = 2 * known["in_channels"] if cfg.get("learn_sigma", True) else known["in_channels"]
    if "learn_sigma" in cfg:
        known["out_channels"] = 2 * known["in_channels"] if cfg["learn_sigma"] else known["in_channels"]
    return UNetModel(**known)


def _clean_state_dict(state_dict: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
    cleaned: Dict[str, th.Tensor] = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def load_guided_diffusion_unet(
    checkpoint: Optional[str] = None,
    cfg: Optional[Dict] = None,
    device: str = "cpu",
    strict: bool = False,
    verbose: bool = True,
    **overrides,
) -> UNetModel:
    """Build the 256x256 U-Net and (optionally) load a pre-trained checkpoint.

    ``checkpoint`` may point to a ``.pt`` file containing either a raw state
    dict or a dict under one of the keys ``{"model", "state_dict", "model_state_dict"}``.
    """
    model = build_unet_model(cfg, **overrides)
    if checkpoint is not None and os.path.isfile(checkpoint):
        blob = th.load(checkpoint, map_location="cpu")
        if isinstance(blob, dict):
            state_dict = None
            for key in ("model", "state_dict", "model_state_dict", "ema"):
                if key in blob and isinstance(blob[key], dict):
                    state_dict = blob[key]
                    break
            if state_dict is None:
                state_dict = blob
        else:
            state_dict = blob
        state_dict = _clean_state_dict(state_dict)
        # drop parameters belonging to inserted adaptors (never in checkpoints)
        state_dict = {k: v for k, v in state_dict.items() if ".adaptor." not in k}
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if verbose:
            n_loaded = len(state_dict) - len(unexpected)
            print(
                f"[unet_loader] loaded {n_loaded}/{len(state_dict)} tensors "
                f"from {checkpoint} (missing={len(missing)}, unexpected={len(unexpected)})"
            )
    elif checkpoint is not None and verbose:
        print(f"[unet_loader] checkpoint '{checkpoint}' not found -> using random init.")
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# adaptor insertion helpers
# ---------------------------------------------------------------------------
def iter_res_blocks(module: nn.Module) -> Iterable[Tuple[str, "ResBlock"]]:
    """Yield ``(name, ResBlock)`` for every ResBlock of a U-Net (in order)."""
    for name, sub in module.named_modules():
        if isinstance(sub, ResBlock):
            yield name, sub


class AdaptedResBlock(nn.Module):
    """Generic wrapper: ``base(x, emb) + adaptor(x)``.

    Used by :func:`attach_adaptors_via_hooks` to retrofit adaptors onto a
    vendored/third-party U-Net without editing its source.
    """

    def __init__(self, base: nn.Module, adaptor: nn.Module):
        super().__init__()
        self.base = base
        self.adaptor = adaptor

    def forward(self, x: th.Tensor, emb: Optional[th.Tensor] = None) -> th.Tensor:
        out = self.base(x, emb) if emb is not None else self.base(x)
        return out + self.adaptor(x)


def _resolve_children(container: nn.Module, index: int) -> Optional[nn.Module]:
    try:
        return container[index]
    except Exception:
        return None


def _set_child(container: nn.Module, index: int, module: nn.Module) -> bool:
    try:
        container[index] = module
        return True
    except Exception:
        return False


def insert_adaptors(
    unet: UNetModel,
    adaptor_factory,
    shift_only: bool = True,
    verbose: bool = True,
) -> List[nn.Module]:
    """Insert zero-initialised adaptor modules into the U-Net.

    ``adaptor_factory(in_channels, out_channels, name) -> nn.Module`` must return
    a module mapping the *block input* to the block's output space and must be
    zero-initialised (so the adapted model is identical to the pre-trained one
    before training, §4.3).

    When ``shift_only=True`` (paper §5.2) adaptors are inserted only in residual
    blocks using the shift (``scale_shift``) conditioning; if the model has no
    such block the filter degrades gracefully to "all residual blocks".
    """
    blocks = list(iter_res_blocks(unet))
    if not blocks:
        raise RuntimeError("insert_adaptors: no ResBlock found in the U-Net")
    targeted = [b for b in blocks if (b[1].use_scale_shift_norm if shift_only else True)]
    if shift_only and not targeted:
        targeted = blocks

    adaptors: List[nn.Module] = []
    for name, block in targeted:
        adaptor = adaptor_factory(block.channels, block.out_channels, name)
        # store outside state dict of the base checkpoint & register params
        block.add_module("adaptor", adaptor)
        adaptors.append(adaptor)
    if verbose:
        print(f"[unet_loader] inserted {len(adaptors)} adaptors into the shift module (of {len(blocks)} res blocks)")
    return adaptors


def attach_adaptors_via_hooks(
    unet: nn.Module,
    adaptor_factory,
    shift_only: bool = True,
    verbose: bool = True,
) -> List[nn.Module]:
    """Insert adaptors into a *foreign* U-Net by wrapping its ResBlocks.

    This works with the official ``guided_diffusion.unet.UNetModel`` (or any
    U-Net exposing ``input_blocks`` / ``middle_block`` / ``output_blocks`` of
    ``TimestepEmbedSequential``) without modifying its code.
    """
    containers: List[nn.Module] = []
    for attr in ("input_blocks", "middle_block", "output_blocks"):
        sub = getattr(unet, attr, None)
        if sub is not None:
            containers.append(sub)

    # resolve the torch module class name of a ResBlock in this code base
    try:
        from guided_diffusion.unet import ResBlock as GDResBlock  # type: ignore
    except Exception:  # pragma: no cover - guided_diffusion is optional
        GDResBlock = ResBlock  # type: ignore

    adaptors: List[nn.Module] = []
    for container in containers:
        for idx in range(len(container)):
            layer = _resolve_children(container, idx)
            candidates = list(layer.children()) if isinstance(layer, nn.Module) else []
            for sub_idx, child in enumerate(candidates):
                if not isinstance(child, GDResBlock):
                    continue
                if shift_only and not getattr(child, "use_scale_shift_norm", False):
                    continue
                channels = getattr(child, "channels")
                out_channels = getattr(child, "out_channels", channels)
                name = f"{type(container).__name__}[{idx}].{sub_idx}"
                adaptor = adaptor_factory(channels, out_channels, name)
                wrapped = AdaptedResBlock(child, adaptor)
                if isinstance(layer, nn.ModuleList):
                    layer[sub_idx] = wrapped
                elif isinstance(layer, nn.Sequential):
                    layer[sub_idx] = wrapped
                else:
                    setattr(layer, str(sub_idx), wrapped)
                adaptors.append(adaptor)
    if verbose:
        print(f"[unet_loader] (hook mode) inserted {len(adaptors)} adaptors")
    return adaptors


def count_parameters(module: nn.Module, only_trainable: bool = False) -> int:
    params = module.parameters()
    if only_trainable:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def adaptor_parameters(unet: nn.Module) -> List[nn.Parameter]:
    """Return only the adaptor parameters of an adapted U-Net."""
    out: List[nn.Parameter] = []
    for name, param in unet.named_parameters():
        if ".adaptor." in name or name.startswith("adaptor.") or ".adaptor" == name[-8:]:
            out.append(param)
    return out


# ---------------------------------------------------------------------------
# uniform frozen-backbone interface
# ---------------------------------------------------------------------------
class FrozenDDPMUNet(nn.Module):
    """Frozen DDPM U-Net exposing the uniform ``epsilon_theta(x_t, t)`` API.

    The wrapper

    * freezes ``theta`` (all pre-trained weights),
    * keeps adaptor parameters trainable when present,
    * splits the ``learn_sigma`` output into ``(eps, sigma)`` and returns ``eps``.
    """

    def __init__(self, unet: UNetModel, learn_sigma: Optional[bool] = None, freeze: bool = True):
        super().__init__()
        self.unet = unet
        self.learn_sigma = unet.learn_sigma if learn_sigma is None else learn_sigma
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        """Freeze every pre-trained parameter, leaving adaptors trainable."""
        for name, param in self.unet.named_parameters():
            param.requires_grad_(".adaptor." in name)

    def train_adaptors(self) -> None:
        for module in self.modules():
            module.eval()
        self.unet.eval()

    def epsilon_theta(self, x_t: th.Tensor, t: th.Tensor, y: Optional[th.Tensor] = None) -> th.Tensor:
        """Return ``eps_theta(x_t, t)`` for integer timesteps ``t`` of shape (B,)."""
        out = self.unet(x_t, t, y)
        if self.learn_sigma:
            eps, _rest = th.split(out, x_t.shape[1], dim=1)
            return eps
        return out

    def forward(self, x_t: th.Tensor, t: th.Tensor, y: Optional[th.Tensor] = None) -> th.Tensor:
        return self.epsilon_theta(x_t, t, y)

    def adaptor_parameters(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "adaptor" in n]

    def param_rate(self) -> float:
        total = count_parameters(self.unet)
        adaptor = sum(p.numel() for p in self.adaptor_parameters())
        return adaptor / max(total, 1)
