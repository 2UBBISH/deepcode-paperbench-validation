"""Zero-initialized adaptor modules for DPMs-ANT (Adapting Pretrained Diffusion Models for Few-Shot Image Generation).

Paper specification
-------------------
Section 4.3 ("Optimization") links the pretrained U-Net with an extra adaptor layer
(Houlsby et al., 2019) as

    x_t^l = theta^l(x_t^{l-1}) + psi^l(x_t^{l-1}),

where ``theta^l`` is the ``l``-th layer of the frozen pretrained U-Net and ``psi^l`` is the
additional adaptor layer.  Only the adaptor parameters ``psi`` are updated during training.

Section 5.2 ("Configurations") gives the concrete parameterisation used in the paper:

    psi^l(x^{l-1}) = f(x^{l-1} W_down) W_up

The input is projected downward from ``R^{w x h x r}`` to a bottleneck of dimension
``R^{(w/c) x (h/c) x d}`` (spatial down-sampling by factor ``c`` and channel bottleneck
``d``), a non-linear activation ``f(.)`` is applied, and an upward projection ``W_up``
restores the original spatial/channel layout.  Defaults: ``c = 4, d = 8`` for DDPMs and
``c = 2, d = 8`` for LDMs.

The addendum additionally describes a richer composition (referring to Section 4 /
Algorithm 1):

    down-pooling -> normalization + 3x3 convolution -> 4-head attention
    -> MLP reducing the feature size to 8 or 16 -> up-sampling with factor 4
    -> normalization -> 3x3 convolution.

Both compositions are implemented here.  ``composition="linear"`` is the default
(the Eq. in Section 5.2), ``composition="bottleneck_full"`` (alias ``"attention"``)
implements the addendum stack, and ``composition="conv_bottleneck"`` adds the
normalisation + 3x3 convolutions of the addendum without the attention/MLP part.

Critical property: *all* extra parameters are initialized to zero, so at initialisation
the adapted U-Net is numerically identical to the pretrained one.

Adaptors are only inserted into the U-Net "shift module" (the timestep-conditioned
``use_scale_shift_norm`` ResBlocks); see :mod:`dpm_ant.models.unet_loader` and
:mod:`dpm_ant.models.ldm_loader` for the insertion routines.  The factory produced by
:func:`build_adaptor_factory` follows the contract expected by those loaders:

    adaptor_factory(in_channels, out_channels=None, name="") -> nn.Module
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AdaptorConfig",
    "Adaptor",
    "BottleneckDown",
    "BottleneckUp",
    "SpatialAttention",
    "build_adaptor",
    "build_adaptor_factory",
    "resolve_adaptor_config",
    "count_parameters",
    "adaptor_parameters",
    "zero_init_parameters",
]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _activation(name: str) -> nn.Module:
    """Non-linear activation ``f(.)`` used between the down and up projections."""
    name = (name or "silu").lower()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=False)
    if name in ("relu",):
        return nn.ReLU(inplace=False)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU(0.2, inplace=False)
    raise ValueError(f"Unknown adaptor activation '{name}'")


def _num_groups(channels: int, groups: int) -> int:
    """GroupNorm requires ``channels % num_groups == 0``; fall back to a valid divisor."""
    if groups is None or groups <= 0:
        groups = 1
    g = min(int(groups), int(channels))
    while g > 1 and channels % g != 0:
        g -= 1
    return max(1, g)


def _norm_layer(norm: str, channels: int, groups: int = 32) -> nn.Module:
    """Normalisation layer used inside the adaptor bottleneck."""
    norm = (norm or "group").lower()
    if norm in ("group", "gn", "groupnorm"):
        return nn.GroupNorm(_num_groups(channels, groups), channels)
    if norm in ("layer", "ln", "layernorm", "layer_norm"):
        return nn.GroupNorm(1, channels)  # per-sample norm over (C, H, W)
    if norm in ("batch", "bn", "batchnorm"):
        return nn.BatchNorm2d(channels)
    if norm in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unknown adaptor norm '{norm}'")


def zero_init_parameters(module: nn.Module, zero_bias: bool = True) -> nn.Module:
    """Set every parameter of ``module`` to zero (the paper's zero-initialisation)."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dtype.is_floating_point:
                p.zero_()
            if not zero_bias and name.endswith("bias"):
                # nothing to do: already zeroed (kept for readability of intent)
                pass
    return module


def count_parameters(module: nn.Module, only_trainable: bool = False) -> int:
    """Number of parameters (optionally only the ones receiving gradients)."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not only_trainable))


def adaptor_parameters(module: nn.Module) -> List[nn.Parameter]:
    """Collect adaptor-only parameters (names containing ``.adaptor.``)."""
    return [p for n, p in module.named_parameters() if ".adaptor." in n or n.startswith("adaptor.")]


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class AdaptorConfig:
    """Hyper-parameters of the adaptor module (Section 4.3 / 5.2 + addendum).

    Attributes
    ----------
    bottleneck_c:
        Spatial compression factor ``c`` (input ``R^{w x h x r}`` -> bottleneck
        ``R^{(w/c) x (h/c) x d}``).  Paper: 4 for DDPM, 2 for LDM.
    bottleneck_d:
        Bottleneck channel dimension ``d``.  Paper: 8 for both backbones.
        The addendum's per-task ``C`` (8 or 16) is handled by :meth:`for_task`.
    hidden_dims:
        MLP feature sizes of the addendum composition (``[8]``, ``[8, 16]``, ...).
        ``8/16`` in the addendum corresponds to the final MLP width.
    composition:
        ``"linear"`` (Section 5.2 equation, default), ``"conv_bottleneck"``
        (adds norm + 3x3 conv), or ``"bottleneck_full"`` / ``"attention"``
        (full addendum stack with 4-head attention + MLP + 4x up-sampling).
    num_heads:
        4-head attention as in the addendum.
    up_sample_factor:
        Up-sampling factor of the addendum composition (4).
    per_task_c:
        Optional per-task ``C`` override mapping ``task name -> C`` (addendum
        "Hyperparameters for Table 3").  Values are used as the bottleneck
        channel dimension ``d`` (8 or 16) and as the final MLP width.
    """

    bottleneck_c: int = 4
    bottleneck_d: int = 8
    hidden_dims: Sequence[int] = field(default_factory=lambda: (8, 16))
    composition: str = "linear"
    activation: str = "silu"
    norm: str = "group"
    groups: int = 32
    num_heads: int = 4
    up_sample_factor: int = 4
    zero_init: bool = True
    per_task_c: Dict[str, int] = field(default_factory=dict)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_dict(cls, cfg: Optional[dict] = None, **overrides) -> "AdaptorConfig":
        cfg = dict(cfg or {})
        # accept a nested ``{"adaptor": {...}}`` config as used by configs/default.yaml
        if "adaptor" in cfg and isinstance(cfg["adaptor"], dict):
            cfg = dict(cfg["adaptor"])
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in cfg.items() if k in allowed}
        if "hidden_dims" in kwargs and kwargs["hidden_dims"] is not None:
            kwargs["hidden_dims"] = tuple(int(h) for h in kwargs["hidden_dims"])
        if "per_task_c" in kwargs and kwargs["per_task_c"] is None:
            kwargs["per_task_c"] = {}
        return cls(**kwargs)

    # ------------------------------------------------------------------ helpers
    def for_backbone(self, backbone: str) -> "AdaptorConfig":
        """Return a copy with the paper's defaults for ``"ddpm"`` or ``"ldm"``."""
        b = (backbone or "ddpm").lower()
        out = AdaptorConfig(**{**self.__dict__})
        if b in ("ldm", "latent", "latent_diffusion"):
            if out.bottleneck_c == 4 and out.bottleneck_d == 8:  # untouched default
                out.bottleneck_c = 2
                out.bottleneck_d = 8
        else:
            if out.bottleneck_c == 2 and out.bottleneck_d == 8:  # untouched ldm default
                out.bottleneck_c = 4
                out.bottleneck_d = 8
        return out

    def for_task(self, task: Optional[str]) -> "AdaptorConfig":
        """Apply the addendum's per-task ``C`` override (bottleneck channels 8/16)."""
        out = AdaptorConfig(**{**self.__dict__})
        if task and out.per_task_c:
            c_val = out.per_task_c.get(task)
            if c_val is not None:
                out.bottleneck_d = int(c_val)
                out.hidden_dims = (int(c_val),)
        return out


def resolve_adaptor_config(cfg: Optional[dict] = None,
                           backbone: str = "ddpm",
                           task: Optional[str] = None,
                           **overrides) -> AdaptorConfig:
    """Build an :class:`AdaptorConfig` from a config dict, honouring backbone/task defaults."""
    a_cfg = AdaptorConfig.from_dict(cfg, **overrides)
    a_cfg = a_cfg.for_backbone(backbone)
    a_cfg = a_cfg.for_task(task)
    return a_cfg


# --------------------------------------------------------------------------------------
# projection building blocks
# --------------------------------------------------------------------------------------
class BottleneckDown(nn.Module):
    """Downward projection ``W_down``: ``R^{w x h x r} -> R^{(w/c) x (h/c) x d}``.

    Implemented as a strided convolution so that the spatial compression factor ``c``
    and the channel bottleneck ``d`` are learned.  When the spatial size is not divisible
    by ``c`` the input is first interpolated to a compatible size (robustness for
    non-power-of-two feature maps).
    """

    def __init__(self, in_channels: int, out_channels: int, factor: int = 4, norm: str = "none",
                 groups: int = 32, conv_before: bool = False):
        super().__init__()
        self.factor = max(1, int(factor))
        k = 2 * self.factor if self.factor > 1 else 3
        pad = self.factor // 2 if self.factor > 1 else 1
        self.need_resize = self.factor > 1
        self.pre_norm = _norm_layer(norm, in_channels, groups) if norm not in ("none", "", None) else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k,
                              stride=self.factor, padding=pad, bias=True)
        # optional extra 3x3 convolution (addendum: "normalization layer with 3x3 convolution")
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=True) if conv_before else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.need_resize:
            h, w = x.shape[-2:]
            th, tw = max(1, h // self.factor), max(1, w // self.factor)
            if th * self.factor != h or tw * self.factor != w:
                x = F.interpolate(x, size=(th * self.factor, tw * self.factor),
                                  mode="nearest")
                # trim the padding introduced by ceil interpolation
                x = x[..., : th * self.factor, : tw * self.factor]
        x = self.pre_norm(x)
        x = self.conv(x)
        if self.conv2 is not None:
            x = self.conv2(x)
        return x


class BottleneckUp(nn.Module):
    """Upward projection ``W_up`` restoring ``R^{(w/c) x (h/c) x d} -> R^{w x h x r}``."""

    def __init__(self, in_channels: int, out_channels: int, factor: int = 4,
                 norm: str = "none", groups: int = 32, conv_after: bool = False):
        super().__init__()
        self.factor = max(1, int(factor))
        k = 2 * self.factor if self.factor > 1 else 3
        pad = self.factor // 2 if self.factor > 1 else 1
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k,
                                       stride=self.factor, padding=pad, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=True) if conv_after else None
        self.post_norm = _norm_layer(norm, out_channels, groups) if norm not in ("none", "", None) else nn.Identity()

    def forward(self, x: torch.Tensor, size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self.conv(x)
        if size is not None and (x.shape[-2], x.shape[-1]) != tuple(size):
            x = F.interpolate(x, size=tuple(size), mode="nearest")
        if self.conv2 is not None:
            x = self.conv2(x)
        x = self.post_norm(x)
        return x


class SpatialAttention(nn.Module):
    """4-head spatial self-attention on the flattened bottleneck feature map."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = max(1, int(num_heads))
        if channels % self.num_heads != 0:
            self.num_heads = _num_groups(channels, self.num_heads)
        self.attn = nn.MultiheadAttention(channels, self.num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)               # (B, H*W, C)
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return out.transpose(1, 2).reshape(b, c, h, w)


# --------------------------------------------------------------------------------------
# adaptor
# --------------------------------------------------------------------------------------
class Adaptor(nn.Module):
    """Zero-initialized adaptor ``psi^l`` added to a frozen U-Net layer.

    Implements ``psi^l(x) = f(x W_down) W_up`` (Section 5.2) with optional
    bottleneck refinements (normalisation + 3x3 conv, 4-head attention, MLP, 4x
    up-sampling) exactly as described in the addendum.

    All parameters are initialised to zero so the adapted network reproduces the
    pretrained network's outputs before any training step (verified in the
    validation checklist: "Zero-init adaptor must reproduce the pretrained model output").
    """

    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 bottleneck_c: int = 4, bottleneck_d: int = 8,
                 hidden_dims: Sequence[int] = (8, 16),
                 composition: str = "linear", activation: str = "silu",
                 norm: str = "group", groups: int = 32, num_heads: int = 4,
                 up_sample_factor: int = 4, zero_init: bool = True,
                 name: str = ""):
        super().__init__()
        out_channels = int(out_channels if out_channels is not None else in_channels)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.bottleneck_c = int(bottleneck_c)
        self.bottleneck_d = int(bottleneck_d)
        self.hidden_dims = tuple(int(h) for h in (hidden_dims or ()))
        self.composition = (composition or "linear").lower()
        self.name = name

        conv_norm = norm if self.composition in ("conv_bottleneck", "bottleneck_full", "attention") else "none"
        conv_before = self.composition in ("conv_bottleneck", "bottleneck_full", "attention")
        conv_after = self.composition in ("conv_bottleneck", "bottleneck_full", "attention")

        # ---- downward projection: R^{w x h x r} -> R^{(w/c) x (h/c) x d}
        self.down = BottleneckDown(
            self.in_channels, self.bottleneck_d, factor=self.bottleneck_c,
            norm=conv_norm, groups=groups, conv_before=conv_before,
        )
        if conv_before:
            self.norm_down = _norm_layer(norm, self.bottleneck_d, groups)
        self.act = _activation(activation)

        bottleneck_ch = self.bottleneck_d

        # ---- addendum composition: 4-head attention + MLP to 8/16
        if self.composition in ("bottleneck_full", "attention"):
            self.attn = SpatialAttention(bottleneck_ch, num_heads=num_heads)
            dims = list(self.hidden_dims) or [bottleneck_ch]
            mlp: List[nn.Module] = []
            prev = bottleneck_ch
            for i, h in enumerate(dims):
                mlp.append(nn.Conv2d(prev, int(h), kernel_size=1, bias=True))
                if i < len(dims) - 1:
                    mlp.append(_activation(activation))
                prev = int(h)
            self.mlp = nn.Sequential(*mlp)
            bottleneck_ch = prev
            # addendum: up-sampling layer with a factor of 4 (spatial, inside the bottleneck)
            self.up_sample = nn.Upsample(scale_factor=float(max(1, int(up_sample_factor))),
                                         mode="nearest")
            self.norm_up = _norm_layer(norm, bottleneck_ch, groups)
            self.conv_up = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3,
                                     stride=1, padding=1, bias=True)
        else:
            self.attn = None
            self.mlp = None
            self.up_sample = None
            self.norm_up = None
            self.conv_up = None

        # ---- upward projection back to R^{w x h x r}
        self.up = BottleneckUp(
            bottleneck_ch, self.out_channels, factor=self.bottleneck_c,
            norm=conv_norm, groups=groups, conv_after=conv_after,
        )

        # channel mismatch between the adaptor output and the residual stream
        self.out_proj = None
        if self.out_channels != self.in_channels:
            self.out_proj = nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1, bias=True)

        if zero_init:
            self.zero_init()
        self._init_final_bias()

    # ------------------------------------------------------------------ utilities
    def zero_init(self) -> "Adaptor":
        """Set every adaptor parameter to zero (Section 5.2: "all the extra layer
        parameters to zero")."""
        zero_init_parameters(self)
        return self

    def _init_final_bias(self) -> None:
        """Keep the output layer zero-initialised (it is the last op applied)."""
        if self.out_proj is not None:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    @property
    def output_channels(self) -> int:
        """Number of channels produced by the adaptor (used by the loaders)."""
        return self.in_channels if self.out_proj is not None else self.out_channels

    def extra_repr(self) -> str:  # pragma: no cover - debugging aid
        return (f"in={self.in_channels}, out={self.out_channels}, c={self.bottleneck_c}, "
                f"d={self.bottleneck_d}, composition={self.composition}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``psi^l(x)``: zero output before training, residual correction afterwards."""
        size = (x.shape[-2], x.shape[-1])
        h = self.down(x)
        if self.composition in ("conv_bottleneck", "bottleneck_full", "attention"):
            h = self.norm_down(h)
        h = self.act(h)
        if self.attn is not None:
            h = h + self.attn(h)
        if self.mlp is not None:
            h = self.mlp(h)
        if self.up_sample is not None:
            h = self.up_sample(h)
            h = self.norm_up(h)
            h = self.act(h)
            h = self.conv_up(h)
        h = self.up(h, size=size)
        if self.out_proj is not None:
            h = self.out_proj(h)
        return h


# --------------------------------------------------------------------------------------
# factories / builders
# --------------------------------------------------------------------------------------
def build_adaptor(in_channels: int, out_channels: Optional[int] = None,
                  config: Optional[AdaptorConfig] = None, name: str = "",
                  **overrides) -> Adaptor:
    """Build a single zero-initialized adaptor module."""
    if config is None:
        config = resolve_adaptor_config(None, **overrides)
    elif overrides:
        config = AdaptorConfig(**{**config.__dict__, **overrides})
    return Adaptor(
        in_channels=in_channels,
        out_channels=out_channels,
        bottleneck_c=config.bottleneck_c,
        bottleneck_d=config.bottleneck_d,
        hidden_dims=config.hidden_dims,
        composition=config.composition,
        activation=config.activation,
        norm=config.norm,
        groups=config.groups,
        num_heads=config.num_heads,
        up_sample_factor=config.up_sample_factor,
        zero_init=config.zero_init,
        name=name,
    )


def build_adaptor_factory(cfg: Optional[dict] = None, backbone: str = "ddpm",
                          task: Optional[str] = None, **overrides):
    """Return an ``adaptor_factory(in_channels, out_channels, name)`` callable.

    The returned factory matches the contract required by
    :func:`dpm_ant.models.unet_loader.insert_adaptors` and
    :func:`dpm_ant.models.ldm_loader.insert_adaptors_ldm`.
    """
    config = resolve_adaptor_config(cfg, backbone=backbone, task=task, **overrides)

    def factory(in_channels: int, out_channels: Optional[int] = None, name: str = "") -> Adaptor:
        return build_adaptor(in_channels, out_channels, config=config, name=name)

    factory.config = config          # type: ignore[attr-defined]
    return factory


if __name__ == "__main__":  # pragma: no cover - smoke test
    torch.manual_seed(0)
    for comp, c, d, chans in (("linear", 4, 8, 256),
                              ("conv_bottleneck", 4, 8, 256),
                              ("bottleneck_full", 2, 8, 320)):
        ad = build_adaptor(chans, chans,
                           config=AdaptorConfig(bottleneck_c=c, bottleneck_d=d,
                                                composition=comp, hidden_dims=(8, 16)))
        x = torch.randn(2, chans, 32, 32)
        y = ad(x)
        print(f"[{comp}] c={c} d={d} out={tuple(y.shape)} "
              f"zero_at_init={(y.abs().max().item() == 0.0)} "
              f"params={count_parameters(ad)}")
