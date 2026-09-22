"""Binary source/target classifier ``p_phi`` used for similarity-guided training.

Paper reference
---------------
Section 4.1 ("Similarity-Guided Training"):
    "... we employ a fixed pre-trained binary classifier that differentiates
    between source and target images at time step :math:`t` ..."

Section 5.2 / Addendum ("Classifier Training (Section 5.2)"):
    * DDPM backbone classifier checkpoint:
      ``https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt``
    * LDM backbone classifier checkpoint:
      ``https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt``
    * "These pre-trained models were fine-tuned by modifying the last layer to
      output two classes to classify whether images where coming from the source
      or the target dataset."
    * "the authors used Adam as the optimizer with a learning rate of 1e-4, a
      batch size of 64, and trained for 300 iterations."

Section 5.5:
    "... the classifiers being trained on noised targeted images among T (1000
    steps) as Equation (1), ensuring a robust gradient for training."

What this module provides
-------------------------
1. ``EncoderUNetModel`` -- a self-contained replica of the OpenAI
   guided-diffusion classifier network (identical module/parameter names and
   tensor shapes), including every supported pooling head (``attention``,
   ``adaptive``, ``spatial``, ``spatial_v2``).
2. ``AttentionPool2d`` -- the attention pooling head used by the released
   ImageNet classifiers.
3. ``PretrainedClassifier`` -- thin wrapper exposing
   ``forward(x_t, t) -> logits``, ``log_prob_target(x_t, t)`` and the detached
   guidance gradient ``grad_log_target(x_t, t) = grad_{x_t} log p_phi(y=T|x_t)``
   which is the quantity appearing in Eq. (5) / Eq. (8) of the paper.
4. ``replace_head`` / ``build_classifier`` / ``load_pretrained_classifier`` --
   load a released (1000-way) ImageNet classifier, swap the final layer for a
   2-way head (source vs. target) and optionally fine-tune it on noised
   10-shot source/target images (see ``dpm_ant/training/classifier_train.py``).
5. ``infer_encoder_config_from_state_dict`` -- best-effort architecture
   recovery so that a checkpoint can be loaded even if its exact configuration
   is not written down in the config file.

Everything is written so the module still *runs* when the pretrained
checkpoints are unavailable (offline reproduction): in that case the network is
randomly initialised and a warning is emitted, which keeps the rest of the ANT
pipeline executable end-to-end.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "EncoderUNetModel",
    "AttentionPool2d",
    "PretrainedClassifier",
    "build_classifier",
    "load_pretrained_classifier",
    "infer_encoder_config_from_state_dict",
    "classifier_logit_grad",
    "ENCODER_IMAGE_256_CONFIG",
    "ENCODER_IMAGE_64_CONFIG",
]


# ---------------------------------------------------------------------------
# Low level building blocks.  We preferentially reuse the (already validated)
# guided-diffusion replica from ``unet_loader`` so that ModuleDict key names and
# tensor shapes are guaranteed to match the released checkpoints.  A local
# fallback with identical naming keeps this module importable on its own.
# ---------------------------------------------------------------------------
_HAS_UNET_LOADER = True
try:  # pragma: no cover - import path exercised at runtime
    from .unet_loader import (  # type: ignore
        AttentionBlock,
        ResBlock,
        TimestepEmbedSequential,
        conv_nd,
        linear,
        normalization,
        timestep_embedding,
        zero_module,
    )
except Exception:  # pragma: no cover - fallback replicas below
    _HAS_UNET_LOADER = False

    def normalization(channels: int) -> nn.Module:
        return nn.GroupNorm(32, channels)

    def zero_module(module: nn.Module) -> nn.Module:
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
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    class TimestepBlock(nn.Module):
        def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            raise NotImplementedError

    class TimestepEmbedSequential(nn.Sequential):
        def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if self.use_conv:
                x = self.conv(x)
            return x

    class Downsample(nn.Module):
        def __init__(self, channels: int, use_conv: bool, dims: int = 2, out_channels: Optional[int] = None,
                     padding: int = 1):
            super().__init__()
            self.channels = channels
            self.out_channels = out_channels or channels
            self.use_conv = use_conv
            self.dims = dims
            stride = 2 if dims != 3 else (1, 2, 2)
            if use_conv:
                self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=padding)
            else:
                assert self.channels == self.out_channels
                self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.op(x)

    class ResBlock(TimestepBlock):
        """Guided-diffusion residual block (identical parameter naming)."""

        def __init__(self, channels: int, emb_channels: int, dropout: float, out_channels: Optional[int] = None,
                     use_conv: bool = False, use_scale_shift_norm: bool = False, dims: int = 2,
                     use_checkpoint: bool = False, up: bool = False, down: bool = False, use_fp16: bool = False):
            super().__init__()
            self.channels = channels
            self.emb_channels = emb_channels
            self.dropout = dropout
            self.out_channels = out_channels or channels
            self.use_conv = use_conv
            self.use_checkpoint = use_checkpoint
            self.use_scale_shift_norm = use_scale_shift_norm
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
                self.h_upd = nn.Identity()
                self.x_upd = nn.Identity()
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

        def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
            if self.updown:
                h = self.in_layers[0](x)
                h = self.in_layers[1](h)
                h = self.h_upd(h)
                x_ = self.x_upd(x)
                h = self.in_layers[2](h)
            else:
                h = self.in_layers(x)
                x_ = x
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
            return self.skip_connection(x_) + h

    class QKVAttention(nn.Module):
        def __init__(self, n_heads: int):
            super().__init__()
            self.n_heads = n_heads

        def forward(self, qkv: torch.Tensor) -> torch.Tensor:
            bs, width, length = qkv.shape
            assert width % (3 * self.n_heads) == 0
            ch = width // (3 * self.n_heads)
            q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
            scale = 1 / math.sqrt(math.sqrt(ch))
            weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            a = torch.einsum("bts,bcs->bct", weight, v)
            return a.reshape(bs, -1, length)

    class AttentionBlock(nn.Module):
        def __init__(self, channels: int, num_heads: int = 1, num_head_channels: int = -1,
                     use_checkpoint: bool = False, use_fp16: bool = False, use_new_attention_order: bool = False):
            super().__init__()
            self.channels = channels
            if num_head_channels == -1:
                self.num_heads = num_heads
            else:
                assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
                self.num_heads = channels // num_head_channels
            self.use_checkpoint = use_checkpoint
            self.use_fp16 = use_fp16
            self.norm = normalization(channels)
            self.qkv = conv_nd(1, channels, channels * 3, 1)
            self.attention = QKVAttention(self.num_heads)
            self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, *spatial = x.shape
            x = x.reshape(b, c, -1)
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
            return (x + h).reshape(b, c, *spatial)


class AttentionPool2d(nn.Module):
    """Attention pooling head used by the released ImageNet classifiers.

    Parameter names (``positional_embedding``, ``k_proj``, ``q_proj``,
    ``v_proj``, ``c_proj``) match the original guided-diffusion implementation so
    released checkpoints load without remapping.
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: Optional[int] = None):
        super().__init__()
        self.spacial_dim = spacial_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim or embed_dim
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NCHW -> (HW)NC
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        # manual multi-head attention (avoids torch version differences in
        # ``F.multi_head_attention_forward`` while keeping the same params).
        heads = self.num_heads
        hd = x.shape[-1] // heads
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        n_tok = q.shape[0]
        q = q.reshape(n_tok, -1, heads, hd).permute(1, 2, 0, 3)  # N, H, T, hd
        k = k.reshape(n_tok, -1, heads, hd).permute(1, 2, 0, 3)
        v = v.reshape(n_tok, -1, heads, hd).permute(1, 2, 0, 3)
        out = F.scaled_dot_product_attention(q, k, v)  # N, H, T, hd
        out = out.permute(2, 0, 1, 3).reshape(n_tok, -1, heads * hd)
        out = self.c_proj(out)
        return out[0]


# ---------------------------------------------------------------------------
# Encoder U-Net (classifier network)
# ---------------------------------------------------------------------------
class EncoderUNetModel(nn.Module):
    """Guided-diffusion ``EncoderUNetModel`` replica used for ``p_phi``.

    The network maps a noised image ``x_t`` (and its timestep ``t``) to logits
    over ``out_channels`` classes.  Following the paper, the last layer is a
    linear/attentive head whose output dimension is the number of classes (2 for
    source vs. target).
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 2,
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (16,),
        dropout: float = 0.0,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = 64,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
        pool: str = "attention",
        **_unused,
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
        self.use_fp16 = use_fp16
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.pool = pool

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
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            use_fp16=use_fp16,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
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
                use_fp16=use_fp16,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
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

        self.pool_spacial_dim = image_size // ds
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(self.pool_spacial_dim, ch, num_head_channels, out_channels),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected pool type: {pool}")

    # -- utilities ---------------------------------------------------------
    def convert_to_fp16(self) -> None:
        self.input_blocks.apply(lambda m: m.half() if hasattr(m, "half") else m)
        self.middle_block.apply(lambda m: m.half() if hasattr(m, "half") else m)
        self.out.apply(lambda m: m.half() if hasattr(m, "half") else m)

    def convert_to_fp32(self) -> None:
        self.input_blocks.apply(lambda m: m.float() if hasattr(m, "float") else m)
        self.middle_block.apply(lambda m: m.float() if hasattr(m, "float") else m)
        self.out.apply(lambda m: m.float() if hasattr(m, "float") else m)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        hs: List[torch.Tensor] = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y is not None
            emb = emb + self.label_emb(y)

        h = x.type(self.input_blocks[0][0].weight.dtype) if self.use_fp16 else x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        return self.out(h)


# ---------------------------------------------------------------------------
# Wrapper providing the guidance gradient
# ---------------------------------------------------------------------------
class PretrainedClassifier(nn.Module):
    """Binary (source vs. target) classifier wrapper.

    Parameters
    ----------
    model : EncoderUNetModel
        The classifier network.  Its ``out_channels`` must equal ``num_classes``.
    num_classes : int
        Number of classes (2 for source/target).
    target_index : int
        Index of the *target* class in the logits (default ``1``), i.e. the
        class used by ``grad_log_target``.
    grad_scale : float
        Optional constant multiplier applied to the returned gradient (the
        paper's hyper-parameter :math:`\\gamma` is applied downstream).
    input_mode : str
        ``"pixel"`` (default) or ``"latent"``.  For ``"latent"`` the wrapper
        requires a ``decode_fn`` that maps latents to pixel space before
        classification (used for LDM backbones).
    """

    def __init__(
        self,
        model: EncoderUNetModel,
        num_classes: int = 2,
        target_index: int = 1,
        grad_scale: float = 1.0,
        input_mode: str = "pixel",
        decode_fn: Optional[Any] = None,
        freeze: bool = True,
    ):
        super().__init__()
        if model.out_channels != num_classes:
            raise ValueError(
                f"model.out_channels ({model.out_channels}) != num_classes ({num_classes}); "
                "call replace_head() first."
            )
        self.model = model
        self.num_classes = num_classes
        self.target_index = int(target_index)
        self.grad_scale = float(grad_scale)
        self.input_mode = input_mode
        self.decode_fn = decode_fn
        if freeze:
            self.freeze()

    # -- freezing ---------------------------------------------------------
    def freeze(self) -> "PretrainedClassifier":
        """Freeze ``phi`` (used during ANT adaptation)."""
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        return self

    def unfreeze(self) -> "PretrainedClassifier":
        """Unfreeze ``phi`` (used during classifier fine-tuning)."""
        for p in self.model.parameters():
            p.requires_grad_(True)
        return self

    @property
    def parameters_frozen(self) -> bool:
        return all(not p.requires_grad for p in self.model.parameters())

    def head_parameters(self) -> List[nn.Parameter]:
        """Parameters of the (replaced) classification head."""
        for name, module in self.model.named_modules():
            if name == "out":
                return list(module.parameters())
        return []

    # -- inference --------------------------------------------------------
    def _maybe_decode(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "latent" and self.decode_fn is not None:
            return self.decode_fn(x)
        return x

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return logits for a batch of noised images ``x_t`` at timestep ``t``."""
        if self.num_classes is not None and y is None:
            y = None
        return self.model(self._maybe_decode(x_t), t, y=y)

    @torch.no_grad()
    def predict_proba(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x_t, t), dim=-1)

    @torch.no_grad()
    def log_prob_target(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``log p_phi(y=T | x_t)`` per sample, shape ``(B,)``."""
        return F.log_softmax(self.forward(x_t, t), dim=-1)[:, self.target_index]

    @torch.no_grad()
    def log_prob_source(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        src = 1 - self.target_index if self.num_classes == 2 else 0
        return F.log_softmax(self.forward(x_t, t), dim=-1)[:, src]

    # -- guidance gradient -------------------------------------------------
    def grad_log_target(
        self,
        x_t: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        target_index: Optional[int] = None,
        detach: bool = True,
        create_graph: bool = False,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """``grad_{x_t} log p_phi(y=T | x_t)`` with the same shape as ``x_t``.

        The classifier parameters are frozen (``phi`` is not trained during ANT
        adaptation) and the returned gradient is detached by default, exactly as
        required by Eq. (5) / Eq. (8), where the correction term
        ``sigma_hat_t^2 * gamma * grad_{x_t} log p_phi(y=T | x_t)`` is added to
        the predicted noise without propagating into ``phi``.
        """
        idx = self.target_index if target_index is None else int(target_index)
        if t is None:
            t = torch.zeros(x_t.shape[0], device=x_t.device, dtype=torch.long)
        if t.ndim == 0:
            t = t.expand(x_t.shape[0])
        scale = self.grad_scale if scale is None else float(scale)

        with torch.enable_grad():
            inp = x_t if x_t.requires_grad else x_t.detach().requires_grad_(True)
            logits = self.forward(inp, t)
            logp = F.log_softmax(logits.float(), dim=-1)[:, idx].sum()
            (grad,) = torch.autograd.grad(logp, inp, create_graph=create_graph, retain_graph=create_graph)
        if scale != 1.0:
            grad = grad * scale
        return grad.detach() if detach else grad

    # -- head replacement --------------------------------------------------
    def replace_head(self, num_classes: int = 2, zero_init: bool = True, device: Optional[Any] = None) -> "PretrainedClassifier":
        """Swap the final layer for a fresh ``num_classes``-way head.

        Matches the addendum: *"These pre-trained models were fine-tuned by
        modifying the last layer to output two classes"*.  All upstream
        parameters are preserved, so the ImageNet classifier becomes a
        source-vs-target classifier with only the head re-initialised.
        """
        m = self.model
        pool = m.pool
        new_out: nn.Module
        if pool == "attention":
            head = AttentionPool2d(m.pool_spacial_dim, m.out[2].embed_dim, m.out[2].num_heads, num_classes)
            # preserve the backbone part of the head where shapes agree
            old = m.out[2]
            with torch.no_grad():
                head.positional_embedding.copy_(old.positional_embedding)
                for name in ("k_proj", "q_proj", "v_proj"):
                    getattr(head, name).load_state_dict(getattr(old, name).state_dict())
            if zero_init:
                nn.init.zeros_(head.c_proj.weight)
                nn.init.zeros_(head.c_proj.bias)
            new_out = nn.Sequential(m.out[0], m.out[1], head)
        elif pool == "adaptive":
            conv = conv_nd(1 if m.out[3].weight.ndim == 3 else 2, m.out[3].in_channels, num_classes, 1)
            if zero_init:
                zero_module(conv)
            new_out = nn.Sequential(m.out[0], m.out[1], m.out[2], conv, nn.Flatten())
        elif pool in ("spatial", "spatial_v2"):
            last = m.out[-1]
            lin = linear(last.in_features, num_classes)
            if zero_init:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            new_out = nn.Sequential(*list(m.out[:-1]), lin) if pool == "spatial_v2" else nn.Sequential(
                m.out[0], m.out[1], lin
            )
        else:  # pragma: no cover
            raise NotImplementedError(f"Unsupported pool for head replacement: {pool}")

        if device is not None:
            new_out = new_out.to(device)
        m.out = new_out
        m.out_channels = num_classes
        self.num_classes = num_classes
        return self

    # -- misc -------------------------------------------------------------
    def count_parameters(self, only_trainable: bool = False) -> int:
        return sum(p.numel() for p in self.model.parameters() if (p.requires_grad or not only_trainable))

    def extra_repr(self) -> str:  # pragma: no cover
        return (
            f"num_classes={self.num_classes}, target_index={self.target_index}, "
            f"input_mode={self.input_mode}, frozen={self.parameters_frozen}"
        )


# ---------------------------------------------------------------------------
# Config presets / state-dict based architecture inference
# ---------------------------------------------------------------------------
# Default ImageNet 256x256 classifier configuration used by guided-diffusion.
ENCODER_IMAGE_256_CONFIG: Dict[str, Any] = dict(
    image_size=256,
    in_channels=3,
    model_channels=128,
    out_channels=1000,
    num_res_blocks=2,
    attention_resolutions=(32, 16, 8),
    channel_mult=(1, 2, 4, 8),
    num_head_channels=64,
    use_scale_shift_norm=True,
    resblock_updown=True,
    pool="attention",
    dropout=0.0,
)

# Default ImageNet 64x64 classifier configuration (used for the LDM backbone).
ENCODER_IMAGE_64_CONFIG: Dict[str, Any] = dict(
    image_size=64,
    in_channels=3,
    model_channels=128,
    out_channels=1000,
    num_res_blocks=2,
    attention_resolutions=(16,),
    channel_mult=(1, 2, 4, 8),
    num_head_channels=64,
    use_scale_shift_norm=True,
    resblock_updown=True,
    pool="attention",
    dropout=0.0,
)


def _infer_levels(channels: Sequence[int]) -> Optional[Tuple[int, int]]:
    """Recover ``(len(channel_mult), num_res_blocks)`` from a channel sequence.

    In ``EncoderUNetModel`` every level contains ``num_res_blocks`` residual
    blocks plus (except the last level) one downsampling block with unchanged
    channel count.  Hence ``n_blocks = L * r + (L - 1)``.
    """
    n = len(channels)
    for L in range(2, 8):
        rem = n - (L - 1)
        if rem <= 0 or rem % L != 0:
            continue
        r = rem // L
        ok = True
        pos = 0
        level_chans: List[int] = []
        for level in range(L):
            span = r + (1 if level != L - 1 else 0)
            if pos + span > n:
                ok = False
                break
            group = channels[pos:pos + span]
            if len(set(group)) != 1:
                ok = False
                break
            level_chans.append(group[0])
            pos += span
        if ok and pos == n:
            if all(level_chans[i] <= level_chans[i + 1] for i in range(L - 1)):
                return L, r
    return None


def infer_encoder_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Best-effort recovery of the classifier architecture from a checkpoint.

    Returns a configuration dict suitable for :class:`EncoderUNetModel`.  Any
    field that cannot be inferred is taken from ``base`` (or the module
    defaults).  Exceptions never propagate: on failure ``base`` is returned.
    """
    cfg: Dict[str, Any] = dict(base or ENCODER_IMAGE_256_CONFIG)
    try:
        # ---- initial conv -------------------------------------------------
        if "input_blocks.0.0.weight" in state_dict:
            w = state_dict["input_blocks.0.0.weight"]
            cfg["model_channels"] = int(w.shape[0])
            cfg["in_channels"] = int(w.shape[1])

        # ---- parse input_blocks into {block_idx: {sub_idx: {suffix: param}}} --
        pat = re.compile(r"^input_blocks\.(\d+)\.(\d+)\.(.+)$")
        blocks: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
        for k, v in state_dict.items():
            m = pat.match(k)
            if m:
                blocks.setdefault(int(m.group(1)), {}).setdefault(int(m.group(2)), {})[m.group(3)] = v
        if not blocks:
            return cfg

        # channels in/out per encoder ResBlock
        ch_out: Dict[int, int] = {}
        attn_levels: List[int] = []
        has_downsample_module = False
        for i in sorted(blocks):
            subs = blocks[i]
            if "qkv.weight" in subs.get(0, {}) or "qkv.weight" in subs.get(1, {}):
                pass
            for j, sd in subs.items():
                if "in_layers.2.weight" in sd:
                    ch_out[i] = int(sd["in_layers.2.weight"].shape[0])
                if "qkv.weight" in sd:
                    attn_levels.append(i)
                if "op.weight" in sd:
                    has_downsample_module = True

        if 0 in ch_out:
            ordered = [ch_out[i] for i in sorted(ch_out)]
            cfg["channel_mult"] = tuple(dict.fromkeys(
                int(c // max(cfg["model_channels"], 1)) for c in ordered
            ))
            levels = _infer_levels(ordered)
            if levels is not None:
                L, r = levels
                cfg["channel_mult"] = tuple(
                    int(ordered[min(idx * (r + 1) + r, len(ordered) - 1)] // cfg["model_channels"])
                    for idx in range(L)
                )
                cfg["num_res_blocks"] = int(r)

        # ---- resblock_updown ---------------------------------------------
        cfg["resblock_updown"] = not has_downsample_module

        # ---- attention resolutions ---------------------------------------
        if attn_levels:
            base_ch = cfg["model_channels"]
            # level index of each attention block, from channel sequence
            ordered_idx = sorted(ch_out)
            chans = [ch_out[i] for i in ordered_idx]
            mults = cfg["channel_mult"]
            res: List[int] = []
            for i in sorted(set(attn_levels)):
                if i not in ch_out:
                    continue
                c = ch_out[i] // base_ch
                # level = index of first occurrence of this channel value
                try:
                    level = list(mults).index(c)
                except ValueError:
                    level = 0
                ds = 2 ** level
                img = cfg.get("image_size", 256)
                res.append(max(img // ds, 1))
            if res:
                cfg["attention_resolutions"] = tuple(sorted(set(res), reverse=True))

        # ---- scale/shift norm --------------------------------------------
        for i in sorted(blocks):
            for j, sd in blocks[i].items():
                if "emb_layers.1.weight" in sd and "out_layers.2.weight" in sd:
                    emb_out = int(sd["emb_layers.1.weight"].shape[0])
                    out_ch = int(sd["out_layers.2.weight"].shape[0])
                    cfg["use_scale_shift_norm"] = bool(emb_out == 2 * out_ch)
                    break
            else:
                continue
            break

        # ---- pooling head & image size -----------------------------------
        out_keys = [k for k in state_dict if k.startswith("out.")]
        if any(k.endswith("q_proj.weight") for k in out_keys):
            cfg["pool"] = "attention"
            for k in out_keys:
                if k.endswith("positional_embedding"):
                    n_tok = int(state_dict[k].shape[0])
                    spacial = int(round(math.sqrt(max(n_tok - 1, 1))))
                    # number of levels determines the total downsample factor
                    L = len(cfg["channel_mult"])
                    cfg["image_size"] = int(spacial * (2 ** (L - 1)))
                    break
            for k in out_keys:
                if k.endswith("c_proj.weight"):
                    cfg["out_channels"] = int(state_dict[k].shape[0])
                    break
        elif any(k == "out.3.weight" for k in out_keys):
            cfg["pool"] = "adaptive"
            cfg["out_channels"] = int(state_dict["out.3.weight"].shape[0])
        elif any(k == "out.2.weight" for k in out_keys) and state_dict["out.2.weight"].ndim == 2:
            cfg["pool"] = "spatial"
            cfg["out_channels"] = int(state_dict["out.2.weight"].shape[0])
        elif any(k == "out.3.weight" for k in out_keys) and state_dict["out.3.weight"].ndim == 2:
            cfg["pool"] = "spatial_v2"
            cfg["out_channels"] = int(state_dict["out.3.weight"].shape[0])
    except Exception:  # pragma: no cover - inference is best effort
        return dict(base or ENCODER_IMAGE_256_CONFIG)
    return cfg


# ---------------------------------------------------------------------------
# Factories / helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Optional[Dict[str, Any]], *names: str, default: Any = None) -> Any:
    """Look up a (possibly nested) config key, accepting dotted paths."""
    if cfg is None:
        return default
    for name in names:
        if name in cfg:
            return cfg[name]
        cur: Any = cfg
        ok = True
        for part in name.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok:
            return cur
    return default


def _extract_checkpoint_state(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "model_ema"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
        # strip common prefixes / DDP wrappers
        cleaned = {}
        for k, v in obj.items():
            if not isinstance(v, torch.Tensor):
                continue
            k2 = re.sub(r"^(module|model)\.", "", k)
            cleaned[k2] = v
        return cleaned
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def load_pretrained_classifier(
    checkpoint: Optional[str] = None,
    cfg: Optional[Dict[str, Any]] = None,
    backbone: str = "ddpm",
    num_classes: int = 2,
    target_index: int = 1,
    device: Any = "cpu",
    strict: bool = False,
    replace_head: bool = True,
    verbose: bool = True,
    **overrides: Any,
) -> PretrainedClassifier:
    """Build a (optionally pretrained) source/target classifier ``p_phi``.

    Parameters
    ----------
    checkpoint :
        Path to a released guided-diffusion classifier checkpoint
        (``256x256_classifier.pt`` for DDPM, ``64x64_classifier.pt`` for LDM).
        When missing/unreadable, weights are randomly initialised (a warning is
        emitted) so that the pipeline remains runnable offline.
    cfg :
        Config dict; both flat and ``{"models": {"ddpm": {...}}}`` layouts are
        accepted, matching ``configs/default.yaml``.
    backbone :
        ``"ddpm"`` (256x256 pixel space) or ``"ldm"`` (classifier applied on the
        decoded 64x64 image, or directly on the latent when
        ``input_mode="latent"``).
    """
    base = ENCODER_IMAGE_256_CONFIG if backbone == "ddpm" else ENCODER_IMAGE_64_CONFIG
    nested = _cfg_get(cfg, f"models.{backbone}", f"models.{backbone}.classifier", default=None)
    if isinstance(nested, dict):
        merged = dict(base)
        merged.update({k: v for k, v in nested.items() if k in base or k in ("image_size", "latent_size")})
        base = merged
    # explicit overrides from the config's classifier block
    expected_ckpt_keys = {"num_res_blocks", "attention_resolutions", "channel_mult", "pool",
                          "model_channels", "use_scale_shift_norm", "resblock_updown", "num_head_channels"}
    for k in expected_ckpt_keys:
        v = _cfg_get(cfg, f"classifier.{k}", default=None)
        if v is not None:
            base[k] = v

    if checkpoint is None:
        checkpoint = _cfg_get(
            cfg,
            f"models.{backbone}.classifier_ckpt",
            f"models.{backbone}.classifier_url",
            f"models.{backbone}.classifier_path",
            "classifier.ckpt",
            "classifier.checkpoint",
            default=None,
        )
    if checkpoint is None:
        checkpoint = (
            "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt"
            if backbone == "ddpm"
            else "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_classifier.pt"
        )

    state: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(checkpoint, str) and os.path.isfile(checkpoint):
        try:
            state = _extract_checkpoint_state(checkpoint)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to read classifier checkpoint {checkpoint}: {exc}")
    else:
        warnings.warn(
            f"Classifier checkpoint not found locally ({checkpoint}); "
            "using randomly initialised weights (fine-tune with "
            "scripts/train_classifier.py before running ANT)."
        )

    if state is not None:
        base = infer_encoder_config_from_state_dict(state, base)
    base = {**base, **{k: v for k, v in overrides.items() if v is not None}}
    # the head is re-initialised with `num_classes` outputs anyway
    build_cfg = dict(base)
    build_cfg["out_channels"] = num_classes if replace_head else int(base.get("out_channels", num_classes))

    model = EncoderUNetModel(**build_cfg)

    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if verbose:
            n_loaded = len(state) - len(unexpected)
            print(
                f"[classifier] loaded {n_loaded}/{len(state)} tensors from {checkpoint} "
                f"({len(missing)} missing, {len(unexpected)} unexpected)"
            )
            if unexpected:
                head = [k for k in unexpected if k.startswith("out.")]
                if head:
                    print(f"[classifier] skipped {len(head)} old head tensors: {head[:4]}")

    clf = PretrainedClassifier(
        model,
        num_classes=num_classes,
        target_index=target_index,
        grad_scale=float(_cfg_get(cfg, "classifier.grad_scale", default=1.0) or 1.0),
        input_mode=str(_cfg_get(cfg, "classifier.input_mode", default="pixel") or "pixel"),
        freeze=bool(_cfg_get(cfg, "classifier.freeze_during_ant", default=True)),
    ).to(device)
    return clf


def build_classifier(
    cfg: Optional[Dict[str, Any]] = None,
    backbone: str = "ddpm",
    num_classes: Optional[int] = None,
    target_index: Optional[int] = None,
    checkpoint: Optional[str] = None,
    device: Any = None,
    **overrides: Any,
) -> PretrainedClassifier:
    """Config-driven entry point used by the scripts and ``main.py``."""
    if num_classes is None:
        num_classes = int(_cfg_get(cfg, "classifier.num_classes", default=2) or 2)
    if target_index is None:
        target_index = int(_cfg_get(cfg, "classifier.target_index", default=1) or 1)
    if device is None:
        device = _cfg_get(cfg, "device", default="cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    return load_pretrained_classifier(
        checkpoint=checkpoint,
        cfg=cfg,
        backbone=backbone,
        num_classes=num_classes,
        target_index=target_index,
        device=device,
        **overrides,
    )


def classifier_logit_grad(
    classifier: PretrainedClassifier,
    x_t: torch.Tensor,
    t: torch.Tensor,
    gamma: float = 1.0,
    detach: bool = True,
    target_index: Optional[int] = None,
) -> torch.Tensor:
    """``gamma * grad_{x_t} log p_phi(y=T | x_t)`` (Eq. 5 / Eq. 8 correction).

    Convenience wrapper around :meth:`PretrainedClassifier.grad_log_target`
    returning the scaled, detached gradient used as the noise correction term
    ``sigma_hat_t^2 * gamma * grad_{x_t} log p_phi(...)``.
    """
    return classifier.grad_log_target(
        x_t, t, target_index=target_index, detach=detach, scale=float(gamma)
    )


def _selftest() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    clf = build_classifier(cfg=None, backbone="ddpm", num_classes=2, device="cpu")
    x = torch.randn(2, 3, 256, 256)
    t = torch.randint(1, 1000, (2,), dtype=torch.long)
    logits = clf(x, t)
    grad = clf.grad_log_target(x, t)
    print("logits", tuple(logits.shape), "grad", tuple(grad.shape), float(grad.abs().mean()))
    assert logits.shape == (2, 2) and grad.shape == x.shape


if __name__ == "__main__":  # pragma: no cover
    _selftest()
