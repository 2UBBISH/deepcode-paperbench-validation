"""DDPM U-Net with time / class / image-shaped conditioning (Appendix B).

Hyper-parameters used in the paper (Appendix B):

    dim=256, dim_mults=(1, 1, 2, 3, 4), resnet_block_groups=8, num_classes=1000,
    learned_sinusoidal_cond=True, learned_sinusoidal_dim=32,
    attn_dim_head=64, attn_heads=4, random_fourier_features=False.

Image-shaped conditioning follows (Ho et al., 2022a) as described in
Appendix B: the up-sampled low-resolution image (super-resolution) or the
missingness mask (in-painting) is appended to the channel dimension of x_t.
That concatenation is performed by
:class:`si_couplings.models.velocity.VelocityModel`, which keeps this file a
plain image-to-image U-Net.
"""

from __future__ import annotations

from functools import partial
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .nn import (
    Attention,
    Downsample,
    LinearAttention,
    PreNorm,
    RandomOrLearnedSinusoidalPosEmb,
    Residual,
    ResnetBlock,
    SinusoidalPosEmb,
    Upsample,
    default,
    exists,
)


class Unet(nn.Module):
    """DDPM U-Net conditional on the time, a class label and (optionally)
    extra input channels (used for image-shaped conditioning).
    """

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Sequence[int] = (1, 2, 4, 8),
        channels: int = 3,
        resnet_block_groups: int = 8,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        attn_dim_head: int = 64,
        attn_heads: int = 4,
        attn_resolutions: Sequence[int] = (16,),
        use_linear_attn: bool = True,
        num_classes: Optional[int] = None,
        image_size: int = 256,
    ):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *[dim * m for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # ---------------- time conditioning ----------------
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=10000)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # ---------------- class conditioning ----------------
        self.num_classes = num_classes
        if num_classes is not None:
            # index ``num_classes`` is the null label used for
            # classifier-free guidance at sampling time
            self.label_emb = nn.Embedding(num_classes + 1, time_dim)
            self.null_class = num_classes

        # resolution of every level of the encoder/decoder (the inner-most
        # resolution, reached by the mid block, is image_size / 2**(L-1))
        resolutions = [max(image_size >> i, 1) for i in range(num_resolutions)]
        self.resolutions = resolutions
        self.attn_resolutions = set(int(r) for r in attn_resolutions)
        attn_flags = [r in self.attn_resolutions for r in resolutions]

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        _local_attention(
                            dim_in,
                            use_linear_attn=use_linear_attn,
                            full=attn_flags[ind],
                            groups=resnet_block_groups,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                        ),
                        _DownsampleOrConv(dim_in, dim_out, is_last),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(
                mid_dim,
                Attention(mid_dim, heads=attn_heads, dim_head=attn_dim_head, groups=resnet_block_groups),
            )
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        _local_attention(
                            dim_out,
                            use_linear_attn=use_linear_attn,
                            full=attn_flags[len(in_out) - 1 - ind],
                            groups=resnet_block_groups,
                            heads=attn_heads,
                            dim_head=attn_dim_head,
                        ),
                        _UpsampleOrConv(dim_out, dim_in, is_last),
                    ]
                )
            )

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, default(out_dim, channels), 1)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, time: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        if exists(self.num_classes) and exists(labels):
            t = t + self.label_emb(labels)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class _DownsampleOrConv(nn.Module):
    """Downsample unless we are at the last resolution (reference behaviour)."""

    def __init__(self, dim_in: int, dim_out: int, is_last: bool):
        super().__init__()
        self.layer = nn.Conv2d(dim_in, dim_out, 3, padding=1) if is_last else Downsample(dim_in, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


def _local_attention(
    dim: int,
    *,
    use_linear_attn: bool,
    full: bool,
    groups: int = 8,
    heads: int = 4,
    dim_head: int = 64,
) -> nn.Module:
    """Attention block at a given resolution.

    The reference implementation inserts linear attention at every
    resolution and full attention at the resolutions listed in
    ``attn_resolutions``; the mid block always has full attention.
    """
    if full:
        return Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, groups=groups)))
    if use_linear_attn:
        return Residual(PreNorm(dim, LinearAttention(dim, heads=heads, dim_head=dim_head, groups=groups)))
    return nn.Identity()


class _UpsampleOrConv(nn.Module):
    """Upsample unless we are at the last resolution (reference behaviour)."""

    def __init__(self, dim_in: int, dim_out: int, is_last: bool):
        super().__init__()
        self.layer = nn.Conv2d(dim_in, dim_out, 3, padding=1) if is_last else Upsample(dim_in, dim_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


def unet_imagenet(num_classes: Optional[int] = 1000, channels: int = 3, **overrides) -> Unet:
    """The U-Net configuration of Appendix B.

    ``attn_resolutions`` is the only hyper-parameter that the paper does not
    list; the reference implementation inserts full attention at the
    innermost resolution (16x16 for 256x256 images, which in our
    parameterisation is the 32x32 level followed by one more down-sampling
    before the mid block).  We therefore default to ``(16, 32)``.
    """
    kwargs = dict(
        dim=256,
        dim_mults=(1, 1, 2, 3, 4),
        channels=channels,
        resnet_block_groups=8,
        learned_sinusoidal_cond=True,
        learned_sinusoidal_dim=32,
        random_fourier_features=False,
        attn_dim_head=64,
        attn_heads=4,
        attn_resolutions=(16, 32),
        num_classes=num_classes,
    )
    kwargs.update(overrides)
    return Unet(**kwargs)


__all__ = ["Unet", "unet_imagenet"]
