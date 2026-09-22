"""Embeddings used by the U-Net velocity model ``\\hat b_t(x, xi)``.

This module implements the conditioning components of the U-Net from
(Ho et al., 2020b) as implemented in lucidrain's ``denoising-diffusion-pytorch``
repository, which the paper (Appendix B) uses verbatim:

    - Dim Mults: (1,1,2,3,4)
    - Dim (channels): 256
    - Resnet block groups: 8
    - Learned Sinusoidal Cond: True
    - Learned Sinusoidal Dim: 32
    - Attention Dim Head: 64
    - Attention Heads: 4
    - Random Fourier Features: False

The paper additionally conditions the velocity model on *image-shaped* inputs
(Appendix B, "Image-shaped conditioning in the Unet"): for super-resolution the
upsampled low-resolution image :math:`\\xi = \\mathcal{U}(\\mathcal{D}(x_1))` is
appended to the input :math:`x_t` at each time step, and for in-painting the
missingness mask :math:`\\xi` is appended likewise.  Those concatenations change
the number of input channels, which :class:`ImageConditioningEmbedding` can
optionally project back to the model width (identity by default, matching the
paper's plain channel concatenation).  Both ImageNet tasks also include the
ImageNet class labels (Section 4.1, Section 4.2, Appendix B).

Provided pieces
---------------
* :class:`SinusoidalPositionEmbeddings` -- fixed (non-learned) sinusoidal time
  embeddings.
* :class:`RandomOrLearnedSinusoidalPosEmb` -- the lucidrain variant: a linear map
  of a scalar time through ``learned_sinusoidal_dim / 2`` frequencies.  When
  ``is_random=True`` the frequencies are frozen random Fourier features
  (``random_fourier_features=True``); when ``False`` they are learned.  The
  paper uses ``False`` with ``learned_sinusoidal_dim = 32``.
* :class:`TimeEmbedding` -- the full time-embedding MLP producing a
  ``time_emb_dim`` vector per batch item.
* :class:`ClassLabelEmbedding` -- class-label embedding with an extra null
  ("unconditional") class and optional label dropout for classifier-free
  guidance, mirroring the lucidrain variant.
* :class:`ImageConditioningEmbedding` -- optional projection of concatenated
  image-shaped conditioning channels.
* :func:`combine_embeddings` / :func:`sum_embeddings` -- additive fusion of
  time and class embeddings (standard in this architecture family).
* :func:`get_time_embedding` -- convenience helper used by the U-Net forward
  pass.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "SinusoidalPositionEmbeddings",
    "RandomOrLearnedSinusoidalPosEmb",
    "TimeEmbedding",
    "ClassLabelEmbedding",
    "ImageConditioningEmbedding",
    "combine_embeddings",
    "sum_embeddings",
    "get_time_embedding",
    "NULL_CLASS_INDEX",
]

#: Index reserved for the null / unconditional class in
#: :class:`ClassLabelEmbedding` (used for classifier-free guidance).
NULL_CLASS_INDEX = -1


class SinusoidalPositionEmbeddings(nn.Module):
    """Fixed sinusoidal embeddings of a scalar time, as in Ho et al. (2020b).

    Parameters
    ----------
    dim:
        Output embedding dimension (must be even).
    max_period:
        Controls the minimum frequency (10000 in the original code).

    Shape
    -----
    input  : ``(B,)`` or ``(B, 1)`` time tensor in ``[0, 1]``
    output : ``(B, dim)``
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPositionEmbeddings expects an even dim, got {dim}")
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1)
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.to(torch.float32)[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, max_period={self.max_period}"


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """lucidrain's random-or-learned sinusoidal positional embedding.

    Following @crowsonkb: a scalar ``x`` is featurized as
    ``cat([x, sin(2*pi*w*x), cos(2*pi*w*x)])`` where ``w`` has
    ``dim // 2`` entries.  ``is_random=True`` freezes ``w`` at random
    initialization (random Fourier features, disabled in the paper);
    ``is_random=False`` makes ``w`` learnable (used in the paper).

    Shape
    -----
    input  : ``(B,)``
    output : ``(B, dim + 1)``
    """

    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"expected a positive even dim, got {dim}")
        self.dim = int(dim)
        self.is_random = bool(is_random)
        half_dim = self.dim // 2
        self.weights = nn.Parameter(
            torch.randn(half_dim), requires_grad=not self.is_random
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(-1, 1)
        freqs = x * self.weights.reshape(1, -1) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, is_random={self.is_random}"


class TimeEmbedding(nn.Module):
    """Time-embedding MLP used by the U-Net velocity net.

    Mirrors the lucidrain ``Unet`` time embedding: a (learned or fixed)
    sinusoidal encoder followed by ``Linear -> GELU -> Linear`` producing a
    ``time_emb_dim``-dimensional vector.

    Parameters
    ----------
    dim:
        Model channel width (256 in the paper).
    time_emb_dim:
        Output embedding width; defaults to ``dim * 4``.
    learned_sinusoidal_cond:
        ``True`` in the paper (Appendix B: "Learned Sinusoidal Cond: True").
    learned_sinusoidal_dim:
        32 in the paper; the sinusoidal encoder outputs
        ``learned_sinusoidal_dim + 1`` features.
    random_fourier_features:
        ``False`` in the paper.  When ``True`` and
        ``learned_sinusoidal_cond=True`` the sinusoidal frequencies are frozen
        random features instead of being learned.
    max_period:
        Passed to the fixed sinusoidal encoder (unused when learned).
    """

    def __init__(
        self,
        dim: int,
        time_emb_dim: Optional[int] = None,
        learned_sinusoidal_cond: bool = True,
        learned_sinusoidal_dim: int = 32,
        random_fourier_features: bool = False,
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.time_emb_dim = int(time_emb_dim) if time_emb_dim is not None else int(dim) * 4
        self.learned_sinusoidal_cond = bool(learned_sinusoidal_cond)
        self.learned_sinusoidal_dim = int(learned_sinusoidal_dim)
        self.random_fourier_features = bool(random_fourier_features)

        if self.learned_sinusoidal_cond:
            self.sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                self.learned_sinusoidal_dim, is_random=self.random_fourier_features
            )
            fourier_dim = self.learned_sinusoidal_dim + 1
        else:
            self.sinu_pos_emb = SinusoidalPositionEmbeddings(self.dim, max_period=max_period)
            fourier_dim = self.dim

        self.time_mlp = nn.Sequential(
            nn.Linear(fourier_dim, self.time_emb_dim),
            nn.GELU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map ``t`` of shape ``(B,)`` (or a scalar) to ``(B, time_emb_dim)``."""
        if not torch.is_tensor(t):
            raise TypeError("t must be a torch.Tensor")
        t = t.to(dtype=torch.float32).reshape(-1)
        return self.time_mlp(self.sinu_pos_emb(t))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, time_emb_dim={self.time_emb_dim}, "
            f"learned_sinusoidal_cond={self.learned_sinusoidal_cond}, "
            f"learned_sinusoidal_dim={self.learned_sinusoidal_dim}, "
            f"random_fourier_features={self.random_fourier_features}"
        )


class ClassLabelEmbedding(nn.Module):
    """Class-label embedding with a null class for classifier-free guidance.

    The paper's U-Net variant "includes embeddings to condition on class
    labels" (Appendix B); both the in-painting and super-resolution ImageNet
    experiments "include the ImageNet class labels" (Sections 4.1, 4.2).  A
    single extra index (``num_classes``) is the null/unconditional class and is
    used for classifier-free-guidance-style label dropout (``dropout_prob``),
    matching the lucidrain variant.

    Parameters
    ----------
    num_classes:
        Number of real classes (1000 for ImageNet-1k).
    embed_dim:
        Embedding width (``time_emb_dim``, i.e. ``4 * channels`` for a 256-wide
        U-Net, so it can be additively fused with the time embedding).
    dropout_prob:
        Probability of replacing a label with the null class during training.
    """

    def __init__(self, num_classes: int, embed_dim: int, dropout_prob: float = 0.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dim = int(embed_dim)
        self.dropout_prob = float(dropout_prob)
        self.class_embedding = nn.Embedding(self.num_classes + 1, self.embed_dim)
        # Standard initialisation: small normal weights, with the null class
        # initialised to the mean of the real-class embeddings.
        nn.init.normal_(self.class_embedding.weight, std=0.02)
        if self.num_classes > 0:
            with torch.no_grad():
                self.class_embedding.weight[self.num_classes] = self.class_embedding.weight[
                    : self.num_classes
                ].mean(dim=0)

    def token_drop(
        self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Replace some labels with the null class index."""
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.reshape(-1).bool()
        return torch.where(drop_ids, torch.full_like(labels, self.num_classes), labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool = True,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``labels`` of shape ``(B,)`` (or ``(B, 1)``) -> ``(B, embed_dim)``."""
        labels = labels.reshape(-1).long()
        if force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        elif train and self.dropout_prob > 0:
            labels = self.token_drop(labels)
        if bool((labels < 0).any()):
            raise ValueError(
                "class labels must be non-negative; use num_classes as the null index"
            )
        return self.class_embedding(labels.clamp(max=self.num_classes))

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, embed_dim={self.embed_dim}, "
            f"dropout_prob={self.dropout_prob}"
        )


class ImageConditioningEmbedding(nn.Module):
    """Optional projector for image-shaped conditioning channels.

    Appendix B: "For image-shaped conditioning, we follow (Ho et al., 2022a) and
    append upsampled low-resolution images to the input ``x_t`` at each time step
    to the velocity model.  We also condition on the missingness masks for
    in-painting by appending them to ``x_t``."  The paper's conditioning is a
    plain channel concatenation, i.e. the identity here; this module exists so
    that an optional learned projection (``channels != out_channels`` or
    ``hidden_channels`` set) can be used without changing the rest of the code.

    Parameters
    ----------
    channels:
        Number of conditioning channels appended to ``x_t``.
    out_channels:
        Channels produced (defaults to ``channels``, i.e. plain concatenation).
    hidden_channels:
        If given with ``out_channels != channels``, a 1x1-conv bottleneck is
        inserted (``channels -> hidden_channels -> out_channels``).
    """

    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = int(channels)
        self.out_channels = int(out_channels) if out_channels is not None else int(channels)
        self.identity_projection = self.out_channels == self.channels and hidden_channels is None

        if self.identity_projection:
            self.project = nn.Identity()
        else:
            hidden = int(hidden_channels) if hidden_channels is not None else self.out_channels
            self.project = nn.Sequential(
                nn.Conv2d(self.channels, hidden, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(hidden, self.out_channels, kernel_size=1),
            )

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        if conditioning.dim() != 4:
            raise ValueError(
                "image-shaped conditioning must have shape (B, C, H, W), got "
                f"{tuple(conditioning.shape)}"
            )
        if conditioning.shape[1] != self.channels:
            raise ValueError(
                f"expected {self.channels} conditioning channels, got {conditioning.shape[1]}"
            )
        return self.project(conditioning)

    @staticmethod
    def concatenate(x_t: torch.Tensor, conditioning: Optional[torch.Tensor]) -> torch.Tensor:
        """Append image-shaped conditioning to ``x_t`` along the channel axis."""
        if conditioning is None:
            return x_t
        return torch.cat([x_t, conditioning], dim=1)

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, out_channels={self.out_channels}, "
            f"identity={self.identity_projection}"
        )


def combine_embeddings(
    time_emb: torch.Tensor,
    class_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Additively fuse the time and (optional) class embeddings."""
    if class_emb is None:
        return time_emb
    if class_emb.shape != time_emb.shape:
        raise ValueError(
            f"class embedding shape {tuple(class_emb.shape)} does not match "
            f"time embedding shape {tuple(time_emb.shape)}"
        )
    return time_emb + class_emb


#: Alias kept for readability at call sites in the U-Net.
sum_embeddings = combine_embeddings


def get_time_embedding(
    t: torch.Tensor,
    time_embed: nn.Module,
    class_labels: Optional[torch.Tensor] = None,
    class_embed: Optional[nn.Module] = None,
    train: bool = True,
    force_drop_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience helper: time embedding fused with an optional label embedding."""
    emb = time_embed(t)
    if class_labels is not None and class_embed is not None:
        emb = combine_embeddings(
            emb, class_embed(class_labels, train=train, force_drop_ids=force_drop_ids)
        )
    return emb


def _self_test() -> None:
    torch.manual_seed(0)
    B = 4
    t = torch.rand(B)
    labels = torch.randint(0, 1000, (B,))

    # Fixed sinusoidal embeddings.
    spe = SinusoidalPositionEmbeddings(256)
    e = spe(t)
    assert e.shape == (B, 256), e.shape
    assert torch.isfinite(e).all()

    # lucidrain random-or-learned embedding: dim + 1 features, learnable weights.
    learned = RandomOrLearnedSinusoidalPosEmb(32, is_random=False)
    el = learned(t)
    assert el.shape == (B, 33), el.shape
    assert learned.weights.requires_grad
    frozen = RandomOrLearnedSinusoidalPosEmb(32, is_random=True)
    _ = frozen(t)
    assert not frozen.weights.requires_grad

    # Time embedding MLP with the paper's hyperparameters.
    temb = TimeEmbedding(
        dim=256,
        learned_sinusoidal_cond=True,
        learned_sinusoidal_dim=32,
        random_fourier_features=False,
    )
    et = temb(t)
    assert et.shape == (B, 1024), et.shape  # time_emb_dim = dim * 4
    assert torch.isfinite(et).all()

    # Class-label embedding fused additively with the time embedding.
    cemb = ClassLabelEmbedding(1000, 1024, dropout_prob=0.1)
    ec = cemb(labels)
    assert ec.shape == (B, 1024), ec.shape
    fused = combine_embeddings(et, ec)
    assert fused.shape == (B, 1024)
    dropped = cemb(labels, force_drop_ids=torch.zeros(B, dtype=torch.bool))
    assert dropped.shape == (B, 1024)
    all_null = cemb(labels, train=False, force_drop_ids=torch.ones(B, dtype=torch.bool))
    # All entries equal the (shared) null-class embedding.
    assert torch.allclose(all_null, all_null[0:1].expand_as(all_null))
    assert torch.allclose(cemb.class_embedding.weight[1000], all_null[0])

    # Image-shaped conditioning: identity by default, projection when widths differ.
    x = torch.randn(2, 3, 8, 8)
    cond = torch.randn(2, 3, 8, 8)
    ice = ImageConditioningEmbedding(3)
    assert torch.allclose(ice(cond), cond)
    cat = ImageConditioningEmbedding.concatenate(x, ice(cond))
    assert cat.shape == (2, 6, 8, 8)
    proj = ImageConditioningEmbedding(3, out_channels=4)
    assert proj(cond).shape == (2, 4, 8, 8)

    # Scalar t must broadcast to a batch of one.
    assert temb(torch.tensor(0.5)).shape == (1, 1024)

    print("si/models/embeddings.py self-test passed")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
