"""The adaptor module and its injection into a pre-trained denoiser.

Section 4.3 of the paper:

    "To save training time and memory, we implement an additional adaptor
    module (Noguchi & Harada, 2019) to learn the shift gap (i.e. Equation (4))
    based on x_t in practice.  During the training, we freeze the parameters
    theta and only update the adaptor parameters psi."

and the residual formulation

    x_t^l = theta^l(x_t^{l-1}) + psi^l(x_t^{l-1})

where ``theta^l`` is the ``l``-th layer of the frozen pre-trained U-Net and
``psi^l`` the parallel adaptor.  Section 5.2 gives the concrete projection

    psi^l(x^{l-1}) = f(x^{l-1} W_down) W_up,
    R^{w x h x r} -> R^{w/c x h/c x d},   c = 4, d = 8   (DDPM)
                                          c = 2, d = 8   (LDM)

and the supplementary description of the module composition:

    down-pooling layer -> normalization + 3x3 convolution -> 4-head attention
    -> MLP that reduces the feature size to 8 (or 16) -> 4x up-sampling ->
    normalization -> 3x3 convolution

All extra parameters are zero-initialised, so at the start of transfer learning
the model is *exactly* the pre-trained source model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .third_party.guided_diffusion.unet import TimestepBlock


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that computes in float32 (as in guided-diffusion)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch's GroupNorm refuses batches with a single value per (sample,
        # group) while in training mode; this happens when the adaptor pools a
        # very small feature map (e.g. a 32x32 U-Net with a pooling factor of
        # 4).  Fall back to instance-style normalisation in that case.
        if self.training:
            channels_per_group = x.shape[1] // self.num_groups
            spatial = math.prod(x.shape[2:])
            if channels_per_group * spatial <= 1:
                dims = tuple(range(1, x.ndim))
                mean = x.float().mean(dim=dims, keepdim=True)
                variance = x.float().var(dim=dims, keepdim=True, unbiased=False)
                normalized = (x.float() - mean) / torch.sqrt(variance + self.eps)
                if self.affine:
                    shape = (1, -1) + (1,) * (x.ndim - 2)
                    normalized = normalized * self.weight.reshape(shape) + self.bias.reshape(shape)
                return normalized.type(x.dtype)
        return super().forward(x.float()).type(x.dtype)


def normalization(channels: int) -> nn.Module:
    return GroupNorm32(math.gcd(32, channels), channels)


class FourHeadAttention(nn.Module):
    """The 4-head attention layer of the adaptor (Section 4.3 / addendum)."""

    def __init__(self, channels: int, num_heads: int = 4, zero_init: bool = False):
        super().__init__()
        assert channels % num_heads == 0, (channels, num_heads)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        length = height * width
        qkv = self.qkv(x).reshape(batch, 3, self.num_heads, self.head_dim, length)
        query, key, value = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = 1.0 / math.sqrt(self.head_dim)
        attention = torch.einsum("bhdn,bhdm->bhnm", query * scale, key)
        attention = attention.softmax(dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attention, value)
        out = out.reshape(batch, channels, height, width)
        return self.proj(out)


class MLP(nn.Module):
    """Small channel-mixing MLP (the adaptor's bottleneck, size ``d = 8/16``)."""

    def __init__(self, in_channels: int, hidden_channels: int, zero_init: bool = False):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@dataclass
class AdaptorConfig:
    """Hyper-parameters of the adaptor module.

    bottleneck_factor (``c``): spatial down/up-sampling factor (4 for DDPM,
        2 for LDM) -- "an up-sampling layer with a factor of 4".
    bottleneck_dim (``d``): dimension the features are projected down to
        (8 or 16 in the paper).
    num_heads: number of attention heads (4).
    zero_init: zero-initialise the whole adaptor (the paper sets "all the extra
        layer parameters to zero").
    """

    bottleneck_factor: int = 4
    bottleneck_dim: int = 8
    num_heads: int = 4
    zero_init: bool = True


class Adaptor(nn.Module):
    """``psi^l``: maps a layer input ``x^{l-1}`` to a residual for layer ``l``."""

    def __init__(self, in_channels: int, out_channels: int, config: AdaptorConfig):
        super().__init__()
        self.config = config
        dim = config.bottleneck_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # "a down-pooling layer followed by a normalization layer with 3x3
        # convolution"  (W_down)
        self.down_pool = (
            nn.AvgPool2d(config.bottleneck_factor, config.bottleneck_factor)
            if config.bottleneck_factor > 1
            else nn.Identity()
        )
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)

        # "a 4 head attention layer followed by an MLP layer reducing feature
        # size to 8 or 16"
        self.attention = FourHeadAttention(dim, num_heads=config.num_heads)
        self.mlp = MLP(dim, dim)

        # "an up-sampling layer with a factor of 4, a normalization layer, and
        # 3x3 convolutions"  (W_up)
        self.upsample = (
            nn.Upsample(scale_factor=config.bottleneck_factor, mode="nearest")
            if config.bottleneck_factor > 1
            else nn.Identity()
        )
        self.norm2 = normalization(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)

        if config.zero_init:
            # zero output => identity at initialisation
            for module in (self.conv3,):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        factor = self.config.bottleneck_factor
        if factor <= 1:
            return x
        height, width = x.shape[-2:]
        if height % factor == 0 and width % factor == 0:
            return self.down_pool(x)
        # odd feature maps (e.g. the 3x3-conv levels of some U-Nets): fall back
        # to an adaptive pool so that any resolution is supported.
        target = (max(1, height // factor), max(1, width // factor))
        return F.adaptive_avg_pool2d(x, target)

    def forward(self, x: torch.Tensor, out_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
        """Return the adaptor residual for the layer input ``x``.

        ``out_shape`` is the shape of the *base layer's* output; the residual is
        resampled to it when the wrapped block changes resolution/channels.
        """
        h = self.conv1(self.norm1(self._downsample(x)))
        h = self.attention(h)
        h = self.mlp(h)
        h = self.upsample(h)
        if h.shape[-2:] != x.shape[-2:]:
            h = F.interpolate(h, size=x.shape[-2:], mode="nearest")
        h = self.conv2(self.norm2(h))
        h = self.conv3(h)
        if out_shape is not None and tuple(h.shape[-2:]) != tuple(out_shape[-2:]):
            h = F.interpolate(h, size=tuple(out_shape[-2:]), mode="nearest")
        return h


class AdaptorWrapper(TimestepBlock):
    """``x^l = theta^l(x^{l-1}) + psi^l(x^{l-1})`` (Section 4.3).

    Subclasses the vendored ``TimestepBlock`` so that the residual can also be
    attached *inside* a ``TimestepEmbedSequential`` (which dispatches on that
    type); ``forward`` accepts and forwards any extra arguments.
    """

    def __init__(self, base: nn.Module, adaptor: Adaptor):
        super().__init__()
        self.base = base
        self.adaptor = adaptor

    def __getattr__(self, name: str):
        """Expose the wrapped module's attributes (diffusers inspects them,
        e.g. ``upsample_block.resnets``)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = self.__dict__.get("_modules", {}).get("base")
            if base is None:
                raise
            return getattr(base, name)

    def forward(self, x: Optional[torch.Tensor] = None, *args, **kwargs):
        if torch.is_tensor(x):
            out = self.base(x, *args, **kwargs)
        else:
            # diffusers passes the activation by keyword (hidden_states=...)
            out = self.base(*args, **kwargs)
            x = first_tensor_argument(None, kwargs)
        if isinstance(out, tuple):  # e.g. diffusers blocks return (hidden, residuals)
            hidden = out[0]
            delta = self.adaptor(x, out_shape=hidden.shape)
            return (hidden + delta,) + tuple(out[1:])
        delta = self.adaptor(x, out_shape=out.shape)
        return out + delta


def first_tensor_argument(x: Optional[torch.Tensor], kwargs: dict) -> torch.Tensor:
    """Return the tensor a U-Net block was called with.

    guided-diffusion passes it positionally, diffusers usually by keyword
    (``hidden_states=`` / ``sample=``).
    """
    if torch.is_tensor(x):
        return x
    for key in ("hidden_states", "sample", "x", "inputs"):
        value = kwargs.get(key)
        if torch.is_tensor(value):
            return value
    for value in kwargs.values():
        if torch.is_tensor(value):
            return value
    raise TypeError("could not find the input tensor of a U-Net block")


# ---------------------------------------------------------------------- #
# injection helpers
# ---------------------------------------------------------------------- #
def _direct_children(
    model: nn.Module, container_names: Iterable[str], granularity: str = "block"
) -> List[Tuple[str, nn.Module]]:
    """Locate the modules that receive an adaptor.

    ``granularity="block"`` (default) attaches one ``psi`` to every top-level
    U-Net block: the elements of ``input_blocks``/``output_blocks`` (which are
    ``nn.ModuleList``s) and ``middle_block`` itself (which *is* a
    ``TimestepEmbedSequential``, i.e. an ``nn.Sequential`` -- it must therefore
    not be expanded).  ``granularity="layer"`` additionally covers the layers
    inside each block.
    """
    targets: List[Tuple[str, nn.Module]] = []
    for name in container_names:
        container = getattr(model, name, None)
        if container is None:
            continue
        if isinstance(container, nn.ModuleList):
            for index, child in enumerate(container):
                targets.append((f"{name}.{index}", child))
                if granularity == "layer" and isinstance(child, nn.Sequential):
                    for inner_index, inner in enumerate(child):
                        targets.append((f"{name}.{index}.{inner_index}", inner))
        else:
            targets.append((name, container))
            if granularity == "layer" and isinstance(container, nn.Sequential):
                for inner_index, inner in enumerate(container):
                    targets.append((f"{name}.{inner_index}", inner))
    return targets


def find_adaptor_targets(
    model: nn.Module,
    containers: Sequence[str] = ("input_blocks", "middle_block", "output_blocks"),
    granularity: str = "layer",
) -> List[Tuple[str, nn.Module]]:
    """The layers ``psi^l`` is attached to.

    Covers the guided-diffusion / LDM U-Net (``input_blocks`` / ``middle_block``
    / ``output_blocks``) as well as the diffusers U-Net used by the LDM backend
    (``down_blocks`` / ``mid_block`` / ``up_blocks``).
    """
    targets = _direct_children(model, containers, granularity=granularity)
    if not targets:
        # diffusers style
        targets = _direct_children(
            model, ("down_blocks", "mid_block", "up_blocks"), granularity=granularity
        )
    if not targets:
        raise ValueError(
            "could not locate U-Net blocks to attach adaptors to; pass explicit "
            "container names via `containers=`"
        )
    return targets


def _infer_io_channels(
    model: nn.Module,
    targets: List[Tuple[str, nn.Module]],
    example_input: torch.Tensor,
    forward_kwargs: Optional[dict] = None,
) -> Dict[str, Tuple[int, int]]:
    """Probe the model once to read each target block's in/out channel counts."""
    forward_kwargs = forward_kwargs or {}
    records: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    handles = []

    def pre_hook(name):
        def hook(module, args, kwargs):
            try:
                tensor = first_tensor_argument(args[0] if args else None, kwargs or {})
            except TypeError:
                return
            records.setdefault(name, {})["in"] = tuple(tensor.shape)
        return hook

    def post_hook(name):
        def hook(module, args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                records.setdefault(name, {})["out"] = tuple(tensor.shape)
        return hook

    for name, module in targets:
        handles.append(module.register_forward_pre_hook(pre_hook(name), with_kwargs=True))
        handles.append(module.register_forward_hook(post_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_input, **forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    channels: Dict[str, Tuple[int, int]] = {}
    for name, _ in targets:
        record = records.get(name)
        if record is None or "in" not in record or "out" not in record:
            raise RuntimeError(f"could not infer shapes for block {name!r}")
        channels[name] = (record["in"][1], record["out"][1])
    return channels


def add_adaptors(
    model: nn.Module,
    config: Optional[AdaptorConfig] = None,
    example_input: Optional[torch.Tensor] = None,
    forward_kwargs: Optional[dict] = None,
    containers: Sequence[str] = ("input_blocks", "middle_block", "output_blocks"),
    granularity: str = "layer",
    verbose: bool = False,
) -> nn.Module:
    """Attach zero-initialised adaptors to every U-Net block, in place.

    The pre-trained weights are frozen: only ``psi`` (the adaptors) will be
    optimised, which is what keeps the "Parameter Rate" of Table 1 at 1.3%
    (DDPM) / 1.6% (LDM).
    """
    config = config or AdaptorConfig()
    targets = find_adaptor_targets(model, containers=containers, granularity=granularity)
    if example_input is None:
        raise ValueError("`example_input` is required to infer block channel counts")
    channels = _infer_io_channels(model, targets, example_input, forward_kwargs=forward_kwargs)

    for name, module in targets:
        in_channels, out_channels = channels[name]
        adaptor = Adaptor(in_channels, out_channels, config)
        parent_path, _, attribute = name.rpartition(".")
        if parent_path:
            parent = model.get_submodule(parent_path)
        else:
            parent = model
        if attribute.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            setattr(parent, attribute, AdaptorWrapper(module, adaptor))
        else:
            setattr(parent, attribute, AdaptorWrapper(module, adaptor))
        if verbose:
            print(
                f"[adaptor] {name}: {in_channels} -> {out_channels} "
                f"(d={config.bottleneck_dim}, c={config.bottleneck_factor})",
                flush=True,
            )
    freeze_base(model)
    return model


def freeze_base(model: nn.Module) -> nn.Module:
    """Freeze everything except the adaptor parameters (Algorithm 1)."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(".adaptor." in name)
    return model


def adaptor_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for name, p in model.named_parameters() if ".adaptor." in name and p.requires_grad]


def base_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for name, p in model.named_parameters() if ".adaptor." not in name]


def set_adaptor_training(model: nn.Module, training: bool = True) -> None:
    """Toggle the adaptors (the frozen backbone always stays in eval mode)."""
    model.eval()
    for module in model.modules():
        if isinstance(module, Adaptor):
            module.train(training)
