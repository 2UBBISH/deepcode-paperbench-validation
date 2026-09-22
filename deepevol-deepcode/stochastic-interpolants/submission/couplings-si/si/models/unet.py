"""U-Net velocity model ``\\hat b_t(x, xi)`` for stochastic interpolants with data-dependent couplings.

This module implements the pixel-space U-Net used by the paper for both in-painting
(Section 4.1) and super-resolution (Section 4.2).  Following Appendix B we use the U-Net
from (Ho et al., 2020b) as implemented in lucidrain's ``denoising-diffusion-pytorch``
repository, which includes class-label conditioning, with the hyperparameters

    - Dim Mults: (1, 1, 2, 3, 4)
    - Dim (channels): 256
    - Resnet block groups: 8
    - Learned Sinusoidal Cond: True
    - Learned Sinusoidal Dim: 32
    - Attention Dim Head: 64
    - Attention Heads: 4
    - Random Fourier Features: False

Image-shaped conditioning (Appendix B, "Image-shaped conditioning in the Unet"): the
upsampled low-resolution image (SR) or the missingness mask (in-painting) is appended to
the input ``x_t`` **at each time step**, i.e. as extra input channels.

In-painting structural information (Section 4.1): because ``xi o I_t = xi o x_1`` for every
``t``, the velocity is identically zero on the unmasked (observed) pixels.  The network can
have this property built in via :meth:`VelocityUNet.mask_velocity` /
``mask_observed=True``, which zeroes the predicted velocity on observed pixels.

The model is written to be robust to the exact call conventions used by the training loop:
``xi``/``y`` may be passed positionally or through a number of keyword aliases, ``t`` may be
a scalar, a ``(B,)`` tensor or a broadcast-shaped ``(B, 1, 1, 1)`` tensor.

Reference: Stochastic Interpolants with Data-Dependent Couplings -- Appendix B, Sections
4.1 and 4.2.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import (
    ClassLabelEmbedding,
    ImageConditioningEmbedding,
    TimeEmbedding,
    combine_embeddings,
)

__all__ = [
    "Attention",
    "LinearAttention",
    "ResnetBlock",
    "VelocityUNet",
    "Unet",
    "UNet",
    "APPENDIX_B_CONFIG",
    "appendix_b_config",
    "unet_from_config",
]


# --------------------------------------------------------------------------------------
# Appendix-B hyperparameter block
# --------------------------------------------------------------------------------------
APPENDIX_B_CONFIG: Dict[str, Any] = {
    # (Ho et al., 2020b) U-Net as in lucidrain's denoising-diffusion-pytorch, Appendix B
    "dim_mults": (1, 1, 2, 3, 4),
    "channels": 256,
    "resnet_block_groups": 8,
    "learned_sinusoidal_cond": True,
    "learned_sinusoidal_dim": 32,
    "attention_dim_head": 64,
    "attention_heads": 4,
    "random_fourier_features": False,
}


def appendix_b_config(**overrides: Any) -> Dict[str, Any]:
    """Return a copy of the Appendix-B U-Net configuration with optional overrides."""
    cfg = dict(APPENDIX_B_CONFIG)
    cfg.update(overrides)
    return cfg


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _exists(val: Any) -> bool:
    return val is not None


def _default(val: Any, d: Any) -> Any:
    return val if val is not None else d


def _normalize_time(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """Reshape ``t`` into a ``(B,)`` float tensor suitable for the time embedding."""
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=device)
    t = t.to(device=device, dtype=torch.float32)
    if t.dim() == 0:
        return t.expand(batch_size)
    if t.dim() == 1:
        if t.shape[0] == 1 and batch_size != 1:
            return t.expand(batch_size)
        return t
    return t.reshape(t.shape[0], -1)[:, 0] if t.shape[0] == batch_size else t.reshape(-1)


class PreNorm(nn.Module):
    """Apply group norm before ``fn`` (lucidrain/U-Net convention)."""

    def __init__(self, dim: int, fn: nn.Module, groups: int = 32) -> None:
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(min(groups, dim), dim)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class Residual(nn.Module):
    """Residual (optionally re-scaled) wrapper."""

    def __init__(self, fn: nn.Module, scale: float = 1.0, trainable_scale: bool = True) -> None:
        super().__init__()
        self.fn = fn
        self.scale = scale
        if trainable_scale:
            self.res_weight = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.res_weight = None

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        out = self.fn(x, *args, **kwargs)
        weight = self.res_weight if self.res_weight is not None else self.scale
        return x + out * weight


# --------------------------------------------------------------------------------------
# ResNet block (dim_mults / resnet_block_groups hyperparameters)
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    """Conv -> GroupNorm -> (scale/shift from the time embedding) -> SiLU."""

    def __init__(self, dim: int, dim_out: int, groups: int = 8) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(min(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    """DDPM residual block with additive/scale-shift time-embedding injection."""

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if _exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        scale_shift = None
        if _exists(self.mlp) and _exists(time_emb):
            t = self.mlp(time_emb)
            t = t[:, :, None, None]
            scale_shift = t.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


# --------------------------------------------------------------------------------------
# Attention (attention_dim_head = 64, attention_heads = 4)
# --------------------------------------------------------------------------------------
class Attention(nn.Module):
    """Standard multi-head self-attention over spatial positions (lucidrain/U-Net)."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (
            t.reshape(b, self.heads, self.dim_head, h * w) for t in qkv
        )
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = out.reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """O(N) linear attention over spatial positions (default in the U-Net blocks)."""

    def __init__(self, dim: int, heads: int = 4, dim_head: int = 64) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        hidden = heads * dim_head
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.heads, self.dim_head, h * w) for t in qkv)

        q = q.softmax(dim=-2)  # softmax over feature dim
        k = k.softmax(dim=-1)  # softmax over spatial dim

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = out.reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)


# --------------------------------------------------------------------------------------
# Down / Up sampling blocks
# --------------------------------------------------------------------------------------
class Downsample(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mode: str = "conv") -> None:
        super().__init__()
        dim_out = _default(dim_out, dim)
        self.mode = mode
        if mode == "conv":
            self.conv = nn.Conv2d(dim, dim_out, 3, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode != "conv":
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, dim: int, dim_out: Optional[int] = None, mode: str = "nearest") -> None:
        super().__init__()
        dim_out = _default(dim_out, dim)
        self.mode = mode
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


# --------------------------------------------------------------------------------------
# The velocity U-Net
# --------------------------------------------------------------------------------------
class VelocityUNet(nn.Module):
    """U-Net velocity network ``\\hat b_t(x, xi)``.

    Parameters
    ----------
    in_channels:
        Number of image channels of ``x_t`` (3 for RGB ImageNet).
    out_channels:
        Number of velocity channels; defaults to ``in_channels``.
    channels:
        Base channel width ("Dim (channels)" in Appendix B, default 256).
    dim_mults:
        Channel multipliers per resolution level (Appendix B: ``(1, 1, 2, 3, 4)``).
    resnet_block_groups:
        GroupNorm groups inside the ResNet/attention blocks (Appendix B: 8).
    num_classes:
        Number of real class labels (``None`` disables class conditioning).  A null class
        (used for classifier-free guidance) is added internally.
    conditioning_channels:
        Number of image-shaped conditioning channels appended to ``x_t`` at every time
        step: 1 for the in-painting missingness mask ``xi``, ``in_channels`` for the
        upsampled low-resolution image ``U(D(x1))`` of super-resolution.
    mask_observed:
        If True, the returned velocity is multiplied by ``(1 - xi)``, i.e. it is
        structurally zero on the observed (unmasked) pixels.  Required for the in-painting
        task where ``xi o I_t = xi o x_1`` for all ``t`` (Section 4.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: Optional[int] = None,
        channels: int = 256,
        dim_mults: Sequence[int] = (1, 1, 2, 3, 4),
        resnet_block_groups: int = 8,
        num_classes: Optional[int] = None,
        class_dropout_prob: float = 0.1,
        conditioning_channels: int = 0,
        project_conditioning: bool = False,
        learned_sinusoidal_cond: bool = True,
        learned_sinusoidal_dim: int = 32,
        attention_dim_head: int = 64,
        attention_heads: int = 4,
        random_fourier_features: bool = False,
        time_emb_dim: Optional[int] = None,
        attn_resolution: int = 16,
        mid_attention: bool = True,
        mask_observed: bool = False,
        observed_value: float = 1.0,
        image_size: int = 256,
        out_activation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(_default(out_channels, in_channels))
        self.channels = int(channels)
        self.dim_mults = tuple(dim_mults)
        self.resnet_block_groups = int(resnet_block_groups)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.conditioning_channels = int(conditioning_channels)
        self.mask_observed = bool(mask_observed)
        self.observed_value = float(observed_value)
        self.attn_resolution = int(attn_resolution)
        self.mid_attention = bool(mid_attention)
        self.out_activation = out_activation

        init_dim = self.channels
        total_in = self.in_channels + self.conditioning_channels

        # Optional learned projector for image-shaped conditioning (identity by default,
        # i.e. plain channel concatenation as in Appendix B).
        if self.conditioning_channels > 0 and project_conditioning:
            self.cond_embed: Optional[nn.Module] = ImageConditioningEmbedding(
                self.conditioning_channels, out_channels=self.conditioning_channels
            )
        else:
            self.cond_embed = None

        self.init_conv = nn.Conv2d(total_in, init_dim, 7, padding=3)

        dims = [init_dim, *[self.channels * m for m in self.dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.num_resolutions = len(in_out)

        # ------------------------------ time (and class) embedding --------------------
        self.time_embed = TimeEmbedding(
            dim=self.channels,
            time_emb_dim=time_emb_dim,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            learned_sinusoidal_dim=learned_sinusoidal_dim,
            random_fourier_features=random_fourier_features,
        )
        with torch.no_grad():
            probe = self.time_embed(torch.zeros(2))
        self.time_emb_dim = int(probe.shape[-1])

        if self.num_classes is not None:
            self.class_embed: Optional[nn.Module] = ClassLabelEmbedding(
                self.num_classes, self.time_emb_dim, dropout_prob=class_dropout_prob
            )
            self._class_dim_proj: Optional[nn.Module] = None
            num_rows = self._class_embedding_rows()
            if num_rows != self.num_classes + 1:
                # The embedding module may or may not add a null row; keep track of the
                # actual number of rows so label values can always be clamped safely.
                self._null_label_index = num_rows - 1
            else:
                self._null_label_index = self.num_classes
        else:
            self.class_embed = None
            self._class_dim_proj = None
            self._null_label_index = 0

        # ------------------------------ down / mid / up -------------------------------
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        res = int(image_size)
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attn = res <= self.attn_resolution
            blocks = [
                ResnetBlock(
                    dim_in, dim_in, time_emb_dim=self.time_emb_dim, groups=self.resnet_block_groups
                ),
                ResnetBlock(
                    dim_in,
                    dim_in,
                    time_emb_dim=self.time_emb_dim,
                    groups=self.resnet_block_groups,
                ),
                (
                    Residual(
                        PreNorm(
                            dim_in,
                            LinearAttention(dim_in, heads=attention_heads, dim_head=attention_dim_head),
                            groups=self.resnet_block_groups,
                        ),
                        scale=0.0,
                    )
                    if use_attn
                    else nn.Identity()
                ),
                (
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ),
            ]
            self.downs.append(nn.ModuleList(blocks))
            if not is_last:
                res = max(res // 2, 1)

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=self.time_emb_dim, groups=self.resnet_block_groups
        )
        self.mid_attn = (
            Residual(
                PreNorm(
                    mid_dim,
                    Attention(mid_dim, heads=attention_heads, dim_head=attention_dim_head),
                    groups=self.resnet_block_groups,
                ),
                scale=0.0,
            )
            if self.mid_attention
            else nn.Identity()
        )
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=self.time_emb_dim, groups=self.resnet_block_groups
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            use_attn = res <= self.attn_resolution
            blocks = [
                ResnetBlock(
                    dim_out * 2,
                    dim_in,
                    time_emb_dim=self.time_emb_dim,
                    groups=self.resnet_block_groups,
                ),
                ResnetBlock(
                    dim_in,
                    dim_in,
                    time_emb_dim=self.time_emb_dim,
                    groups=self.resnet_block_groups,
                ),
                (
                    Residual(
                        PreNorm(
                            dim_in,
                            LinearAttention(dim_in, heads=attention_heads, dim_head=attention_dim_head),
                            groups=self.resnet_block_groups,
                        ),
                        scale=0.0,
                    )
                    if use_attn
                    else nn.Identity()
                ),
                (
                    Upsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ),
            ]
            self.ups.append(nn.ModuleList(blocks))
            if not is_last:
                res = min(res * 2, int(image_size))

        self.final_conv = nn.Sequential(
            ResnetBlock(
                self.channels,
                self.channels,
                time_emb_dim=self.time_emb_dim,
                groups=self.resnet_block_groups,
            ),
            nn.Conv2d(self.channels, self.out_channels, 1),
        )

    # ------------------------------------------------------------------ helpers ----
    def _class_embedding_rows(self) -> int:
        emb = getattr(self.class_embed, "embedding", None)
        if emb is not None and hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)
        return self.num_classes + 1

    def _sanitize_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Map negative labels to the null class and clamp to the valid embedding range."""
        num_rows = self._class_embedding_rows()
        labels = labels.long()
        labels = labels.clone()
        labels[labels < 0] = self._null_label_index
        return labels.clamp_(0, num_rows - 1)

    @staticmethod
    def _resolve_kwarg(kwargs: Dict[str, Any], names: Tuple[str, ...]) -> Optional[Any]:
        for name in names:
            if kwargs.get(name, None) is not None:
                return kwargs.pop(name)
        return None

    def mask_velocity(
        self,
        velocity: torch.Tensor,
        xi: Optional[torch.Tensor],
        observed_value: Optional[float] = None,
    ) -> torch.Tensor:
        """Zero the velocity on observed pixels (Section 4.1).

        With the paper's convention ``xi = 1`` marks *unmasked/observed* pixels, where
        ``xi o I_t = xi o x_1`` implies ``I_dot_t = 0``.  Hence we multiply the predicted
        velocity by ``(1 - xi)``.
        """
        if xi is None:
            return velocity
        obs = self.observed_value if observed_value is None else float(observed_value)
        keep = (obs - xi).abs()
        keep = (keep < 1e-6).to(velocity.dtype)
        # keep == 1 only where xi == obs (observed pixels)
        keep = 1.0 - keep
        return velocity * keep

    # ------------------------------------------------------------------ forward ----
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        mask_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        mask_observed: Optional[bool] = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Predict the velocity at ``x`` at time ``t``.

        ``xi`` (aliases ``cond``/``conditioning``/``context``/``mask``/``c``) is the
        image-shaped conditioning appended to the input channels; ``y`` (aliases
        ``label``/``labels``/``class_labels``/``cls``) are the ImageNet class labels.
        """
        if xi is None:
            xi = self._resolve_kwarg(kwargs, ("cond", "conditioning", "context", "mask", "c"))
        if y is None:
            y = self._resolve_kwarg(kwargs, ("label", "labels", "class_labels", "cls"))
        if mask_fn is None:
            mask_fn = kwargs.pop("mask_function", None)
        if kwargs.pop("train_mode", None) is not None or "train" in kwargs:
            pass

        b = x.shape[0]
        device = x.device

        # ---- image-shaped conditioning: append to x_t at each time step (App. B) ----
        if self.conditioning_channels > 0 and xi is not None:
            cond = xi.to(device=x.device, dtype=x.dtype)
            if cond.shape[1] != self.conditioning_channels:
                if cond.shape[1] == 1:
                    cond = cond.expand(-1, self.conditioning_channels, -1, -1)
                else:
                    cond = cond[:, : self.conditioning_channels]
            if self.cond_embed is not None:
                cond = self.cond_embed(cond)
            x = torch.cat([x, cond], dim=1)
        elif self.conditioning_channels > 0:
            raise ValueError(
                "This U-Net was built with conditioning_channels="
                f"{self.conditioning_channels} but no image-shaped conditioning was provided."
            )

        t = _normalize_time(t, b, device)
        time_emb = self.time_embed(t)
        if time_emb.shape[-1] != self.time_emb_dim:
            raise RuntimeError(
                "Time embedding dim mismatch: expected "
                f"{self.time_emb_dim}, got {time_emb.shape[-1]}"
            )

        if self.class_embed is not None and y is not None:
            labels = self._sanitize_labels(torch.as_tensor(y, device=device))
            class_emb = self.class_embed(labels, train=self.training)
            if class_emb.shape[-1] != time_emb.shape[-1]:
                if self._class_dim_proj is None:
                    self._class_dim_proj = nn.Linear(
                        class_emb.shape[-1], time_emb.shape[-1]
                    ).to(device=device, dtype=time_emb.dtype)
                class_emb = self._class_dim_proj(class_emb)
            time_emb = combine_embeddings(time_emb, class_emb)

        # ------------------------------------------------------------------ U-Net ----
        h = self.init_conv(x)
        h = h.to(dtype=time_emb.dtype)
        r = h

        hs: list = []
        for idx, (block1, block2, attn, downsample) in enumerate(self.downs):
            r = block1(r, time_emb)
            r = block2(r, time_emb)
            r = attn(r)
            hs.append(r)
            r = downsample(r)

        r = self.mid_block1(r, time_emb)
        r = self.mid_attn(r)
        r = self.mid_block2(r, time_emb)

        for block1, block2, attn, upsample in self.ups:
            skip = hs.pop()
            r = torch.cat([r, skip], dim=1)
            r = block1(r, time_emb)
            r = block2(r, time_emb)
            r = attn(r)
            r = upsample(r)

        out = self.final_conv(r)

        if self.out_activation == "tanh":
            out = torch.tanh(out)

        # ------------------------------------------- in-painting structural masking --
        if mask_fn is not None:
            out = mask_fn(out, xi)
        elif mask_observed if mask_observed is not None else self.mask_observed:
            out = self.mask_velocity(out, xi)

        if return_dict:
            return {"velocity": out, "t": t, "time_emb": time_emb}
        return out

    # ------------------------------------------------------------- introspection ----
    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return int(sum(p.numel() for p in params if p.requires_grad))
        return int(sum(p.numel() for p in params))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"channels={self.channels}, dim_mults={self.dim_mults}, "
            f"resnet_block_groups={self.resnet_block_groups}, "
            f"num_classes={self.num_classes}, "
            f"conditioning_channels={self.conditioning_channels}, "
            f"mask_observed={self.mask_observed}, "
            f"params={self.num_parameters():,}"
        )


# Convenience aliases used across the repo
Unet = VelocityUNet
UNet = VelocityUNet


# --------------------------------------------------------------------------------------
# Config-driven builder (Appendix B hyperparameters)
# --------------------------------------------------------------------------------------
def unet_from_config(config: Optional[Dict[str, Any]] = None, **overrides: Any) -> VelocityUNet:
    """Build the velocity U-Net from a flat config dict.

    Recognised keys (defaults from :data:`APPENDIX_B_CONFIG`):

    ``in_channels``, ``out_channels``, ``channels``/``dim``, ``dim_mults``,
    ``resnet_block_groups``, ``learned_sinusoidal_cond``, ``learned_sinusoidal_dim``,
    ``attention_dim_head``, ``attention_heads``/``attn_heads``,
    ``random_fourier_features``, ``num_classes``/``num_class``, ``class_dropout_prob``,
    ``conditioning_channels``, ``project_conditioning``, ``attn_resolution``,
    ``mask_observed``, ``image_size``, ``time_emb_dim``.
    """
    cfg: Dict[str, Any] = dict(APPENDIX_B_CONFIG)
    if config:
        cfg.update(config)
    cfg.update(overrides)

    if "dim" in cfg and "channels" not in cfg:
        cfg["channels"] = cfg.pop("dim")
    if "attn_heads" in cfg and "attention_heads" not in cfg:
        cfg["attention_heads"] = cfg.pop("attn_heads")
    if "num_class" in cfg and "num_classes" not in cfg:
        cfg["num_classes"] = cfg.pop("num_class")

    allowed = {
        "in_channels",
        "out_channels",
        "channels",
        "dim_mults",
        "resnet_block_groups",
        "num_classes",
        "class_dropout_prob",
        "conditioning_channels",
        "project_conditioning",
        "learned_sinusoidal_cond",
        "learned_sinusoidal_dim",
        "attention_dim_head",
        "attention_heads",
        "random_fourier_features",
        "time_emb_dim",
        "attn_resolution",
        "mid_attention",
        "mask_observed",
        "observed_value",
        "image_size",
        "out_activation",
    }
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    if isinstance(kwargs.get("dim_mults"), list):
        kwargs["dim_mults"] = tuple(kwargs["dim_mults"])
    return VelocityUNet(**kwargs)


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")

    # ---- tiny in-painting-like model (mask conditioning + structural masking) -----
    net = VelocityUNet(
        in_channels=3,
        channels=16,
        dim_mults=(1, 2),
        resnet_block_groups=8,
        num_classes=10,
        conditioning_channels=1,
        mask_observed=True,
        attention_dim_head=8,
        attention_heads=2,
        image_size=16,
    )
    x = torch.randn(2, 3, 16, 16, device=device, requires_grad=True)
    t = torch.rand(2, device=device)
    xi = (torch.rand(2, 1, 16, 16, device=device) > 0.3).float()
    y = torch.tensor([0, 5], device=device)
    out = net(x, t, xi=xi, y=y)
    assert out.shape == (2, 3, 16, 16), out.shape
    # velocity must be exactly zero on observed pixels (xi == 1)
    assert torch.allclose(out, out * (1 - xi)), "in-painting velocity not masked"
    out.sum().backward()
    assert x.grad is not None

    # ---- t given as a broadcast (B,1,1,1) tensor ----------------------------------
    out2 = net(x.detach(), t.reshape(-1, 1, 1, 1), xi=xi, y=y)
    assert out2.shape == out.shape

    # ---- super-resolution-like model (3 conditioning channels, no masking) ---------
    sr = VelocityUNet(
        in_channels=3,
        channels=16,
        dim_mults=(1, 2),
        num_classes=1000,
        conditioning_channels=3,
        image_size=16,
        attention_dim_head=8,
        attention_heads=2,
    )
    low = torch.randn(2, 3, 16, 16, device=device)
    out3 = sr(torch.randn(2, 3, 16, 16, device=device), torch.rand(2, device=device), xi=low, y=torch.tensor([7, 11]))
    assert out3.shape == (2, 3, 16, 16)

    # ---- keyword aliases and scalars ----------------------------------------------
    out4 = sr(torch.randn(2, 3, 16, 16, device=device), 0.5, cond=low, labels=torch.tensor([1, 2]))
    assert out4.shape == out3.shape

    # ---- unconditional model ------------------------------------------------------
    un = VelocityUNet(in_channels=3, channels=8, dim_mults=(1, 2), image_size=16,
                      attention_dim_head=4, attention_heads=2)
    out5 = un(torch.randn(1, 3, 16, 16, device=device), torch.rand(1, device=device))
    assert out5.shape == (1, 3, 16, 16)

    # ---- config builder uses Appendix-B defaults ----------------------------------
    cfg_net = unet_from_config({"num_classes": 1000, "conditioning_channels": 1})
    assert cfg_net.channels == 256
    assert cfg_net.dim_mults == (1, 1, 2, 3, 4)
    assert cfg_net.resnet_block_groups == 8

    print("unet.py self-test passed; params(small net) =", net.num_parameters())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
