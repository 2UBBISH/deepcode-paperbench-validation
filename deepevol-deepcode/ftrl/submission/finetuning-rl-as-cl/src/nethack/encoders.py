"""Observation encoders for the NetHack joint actor/critic model (Appendix B.1).

The paper (Section B.1 "Architecture") describes a joint actor/critic backbone that
consumes three observation components:

* **main dungeon screen** -- each character and each colour is looked up in an
  embedding table; the resulting grid of embedding vectors is processed by a small
  ResNet (see Tuyls et al. 2023 for the original details),
* **blstats** -- the player's status information (health, hunger, ...) processed by a
  two-layer MLP,
* **message** -- the textual message displayed to the player, also processed by a
  two-layer MLP.

The three encoded components are merged (concatenated) before being fed to the joint
LSTM of ``hidden_dim = 1738`` (Table 1 of the paper).  This module implements exactly
those three encoders plus the small convenience helpers used by :mod:`src.nethack.model`
and by the fine-tuning runners (freezing the encoders during fine-tuning, parameter
counting, output-dimension bookkeeping).

Everything torch-related is imported defensively so that the module can be imported
(and statically analysed) in environments without PyTorch; constructing or running an
encoder requires torch and raises a clear ``RuntimeError`` otherwise.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - exercised only when torch is unavailable
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False


# --------------------------------------------------------------------------------------
# Defaults (mirroring Appendix B.1 and Table 1)
# --------------------------------------------------------------------------------------

#: Shape of the NLE main screen (rows, columns).
MAIN_SCREEN_SHAPE: Tuple[int, int] = (24, 80)
#: Number of distinct characters in the NLE tty char array (``tty_chars``).
NUM_CHARS: int = 128  # np.uint8 values, padded from the observed 0..127 range
#: Number of distinct colours in the NLE tty colour array (``tty_colors``).
NUM_COLORS: int = 32  # NLE uses 0..15 plus BL_MASK; 32 leaves headroom
#: Dimensionality of ``blstats`` in NLE.
BLSTATS_DIM: int = 25
#: Number of characters in the NLE message buffer.
MESSAGE_LENGTH: int = 256
#: Number of distinct characters appearing inside messages.
NUM_MESSAGE_CHARS: int = 128
#: Width of the joint LSTM hidden state (Table 1: ``hidden_dim = 1738``).
HIDDEN_DIM: int = 1738

#: Default embedding widths for the character / colour lookup tables.
DEFAULT_CHAR_EMBED_DIM: int = 32
DEFAULT_COLOR_EMBED_DIM: int = 16
#: Default number of channels produced by the embedding grid before the ResNet.
DEFAULT_SCREEN_CHANNELS: int = 48
#: Default hidden width of the blstats / message MLPs, and their output width.
DEFAULT_MLP_HIDDEN: int = 512
DEFAULT_MLP_OUT: int = 256
#: Default number of residual blocks inside the screen encoder.
DEFAULT_RESNET_BLOCKS: int = 2

#: Parameter-name prefixes used to locate encoders inside the joint model; used by the
#: fine-tuning runners when freezing the encoders (Appendix B.1: "we froze the encoders
#: during the course of the training").
ENCODER_NAMES: Tuple[str, ...] = (
    "main_screen_encoder",
    "screen_encoder",
    "blstats_encoder",
    "message_encoder",
    "encoder",
    "encoders",
)


def _require_torch() -> None:
    """Raise a helpful error when torch is not installed."""

    if not _HAS_TORCH:
        raise RuntimeError(
            "PyTorch is required to build NetHack encoders. Install torch and retry."
        )


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class EncoderConfig:
    """Hyperparameters of the three observation encoders (Appendix B.1)."""

    # main screen encoder ------------------------------------------------------------
    main_screen_shape: Tuple[int, int] = MAIN_SCREEN_SHAPE
    num_chars: int = NUM_CHARS
    num_colors: int = NUM_COLORS
    char_embed_dim: int = DEFAULT_CHAR_EMBED_DIM
    color_embed_dim: int = DEFAULT_COLOR_EMBED_DIM
    screen_channels: int = DEFAULT_SCREEN_CHANNELS
    screen_out_dim: int = DEFAULT_MLP_OUT
    num_blocks: int = DEFAULT_RESNET_BLOCKS
    use_color: bool = True

    # blstats encoder ----------------------------------------------------------------
    blstats_dim: int = BLSTATS_DIM
    blstats_hidden_dim: int = DEFAULT_MLP_HIDDEN
    blstats_out_dim: int = DEFAULT_MLP_OUT

    # message encoder ----------------------------------------------------------------
    message_shape: Tuple[int, int] = (MESSAGE_LENGTH, NUM_MESSAGE_CHARS)
    message_hidden_dim: int = DEFAULT_MLP_HIDDEN
    message_out_dim: int = DEFAULT_MLP_OUT
    message_mode: str = "one_hot"  # "one_hot" | "embedding"
    message_embed_dim: int = 16

    # shared -------------------------------------------------------------------------
    activation: str = "relu"

    def with_overrides(self, **overrides: Any) -> "EncoderConfig":
        """Return a copy of this config with the given fields replaced."""

        kwargs = {k: v for k, v in overrides.items() if v is not None and hasattr(self, k)}
        data = {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        data.update(kwargs)
        return EncoderConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "EncoderConfig":
        """Build an :class:`EncoderConfig` from a YAML/dict/dataclass config object.

        Recognised blocks (whichever exist) are ``encoder``/``encoders``/``model``, plus
        the top-level keys ``hidden_dim``/``activation``.  Unknown keys are ignored.
        """

        base = cls()
        values: Dict[str, Any] = {}

        if cfg is not None:
            for block in ("encoder", "encoders", "model", "model_kwargs"):
                sub = _cfg_get(cfg, block)
                if isinstance(sub, dict):
                    values.update(sub)
            act = _cfg_get(cfg, "activation_function") or _cfg_get(cfg, "activation")
            if act is not None and "activation" not in values:
                values["activation"] = act
            screen = _cfg_get(cfg, "main_screen_shape")
            if screen is not None:
                values["main_screen_shape"] = tuple(screen)  # type: ignore[arg-type]
            msg = _cfg_get(cfg, "message_shape")
            if msg is not None:
                values["message_shape"] = tuple(msg)  # type: ignore[arg-type]

        values.update({k: v for k, v in overrides.items() if v is not None})
        for key in list(values):
            if key == "hidden_dim":
                # ``hidden_dim`` in the configs is the LSTM width; do not use it here.
                values.pop(key, None)
        return base.with_overrides(**values)

    @property
    def num_message_tokens(self) -> int:
        return int(self.message_shape[0])

    @property
    def num_message_chars(self) -> int:
        return int(self.message_shape[1])

    @property
    def flat_message_dim(self) -> int:
        return int(self.message_shape[0] * self.message_shape[1])


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Tolerant lookup on dict / Config / object-style configs."""

    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:  # pragma: no cover - exotic mapping
            pass
    return getattr(cfg, key, default)


def resolve_activation(name: Union[str, Any, None] = "relu") -> Any:
    """Map an activation name onto an ``nn.Module`` class (or instance-callable)."""

    _require_torch()
    if name is None:
        return nn.ReLU
    if not isinstance(name, str):
        return name
    key = name.strip().lower()
    table = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "leakyrelu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
        "linear": nn.Identity,
    }
    return table.get(key, nn.ReLU)


def make_activation(name: Union[str, Any, None] = "relu") -> Any:
    """Instantiate an activation module (paper: ``activation_function = relu``)."""

    act_cls = resolve_activation(name)
    try:
        return act_cls()
    except Exception:  # pragma: no cover - custom callable
        return act_cls


# --------------------------------------------------------------------------------------
# Main screen encoder: embedding lookup + ResNet  (Appendix B.1)
# --------------------------------------------------------------------------------------


if _HAS_TORCH:

    class ResidualBlock(nn.Module):
        """Basic pre-activation residual block used by the screen ResNet."""

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            activation: Any = None,
            stride: int = 1,
        ) -> None:
            super().__init__()
            act = activation if activation is not None else nn.ReLU
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.norm1 = nn.GroupNorm(_num_groups(out_channels), out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
            self.act = _activation_factory(act)
            self.downsample: Optional[nn.Module] = None
            if stride != 1 or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(_num_groups(out_channels), out_channels),
                )

        def forward(self, x: Any) -> Any:
            identity = x if self.downsample is None else self.downsample(x)
            out = self.act(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            return self.act(out + identity)

    class ScreenResNet(nn.Module):
        """Small ResNet operating on the embedded main-screen grid.

        The input is ``(B, C, H, W)`` where ``C`` is the sum of the character and colour
        embedding widths; the output is a flattened ``screen_out_dim`` feature vector.
        Adaptive pooling keeps the module robust to the exact screen resolution.
        """

        def __init__(
            self,
            in_channels: int,
            out_dim: int = DEFAULT_MLP_OUT,
            activation: Any = None,
            num_blocks: int = DEFAULT_RESNET_BLOCKS,
            channels: Optional[Sequence[int]] = None,
            pool_size: Tuple[int, int] = (3, 6),
        ) -> None:
            super().__init__()
            act = activation if activation is not None else nn.ReLU
            if channels is None:
                channels = (in_channels, max(in_channels * 2, 64))
            channels = tuple(int(c) for c in channels)
            if not channels:
                channels = (in_channels,)

            layers: List[nn.Module] = []
            prev = int(in_channels)
            for i, ch in enumerate(channels):
                stride = 2 if (i > 0 and ch != prev) else 1
                layers.append(ResidualBlock(prev, ch, activation=act, stride=stride))
                prev = ch
            self.blocks = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self.out_dim = int(out_dim)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(prev * int(pool_size[0]) * int(pool_size[1]), self.out_dim),
                _activation_factory(act),
            )

        def forward(self, x: Any) -> Any:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            return self.head(self.pool(self.blocks(x)))

    class MainScreenEncoder(nn.Module):
        """Character + colour embedding lookup followed by a ResNet (Appendix B.1)."""

        def __init__(
            self,
            num_chars: int = NUM_CHARS,
            num_colors: int = NUM_COLORS,
            char_embed_dim: int = DEFAULT_CHAR_EMBED_DIM,
            color_embed_dim: int = DEFAULT_COLOR_EMBED_DIM,
            screen_channels: int = DEFAULT_SCREEN_CHANNELS,
            out_dim: int = DEFAULT_MLP_OUT,
            activation: Any = "relu",
            num_blocks: int = DEFAULT_RESNET_BLOCKS,
            screen_shape: Tuple[int, int] = MAIN_SCREEN_SHAPE,
            use_color: bool = True,
        ) -> None:
            super().__init__()
            self.num_chars = int(num_chars)
            self.num_colors = int(num_colors)
            self.char_embed_dim = int(char_embed_dim)
            self.color_embed_dim = int(color_embed_dim)
            self.screen_shape = tuple(screen_shape)
            self.use_color = bool(use_color)

            self.char_embedding = nn.Embedding(self.num_chars, self.char_embed_dim)
            self.color_embedding = (
                nn.Embedding(self.num_colors, self.color_embed_dim) if self.use_color else None
            )
            in_channels = self.char_embed_dim + (self.color_embed_dim if self.use_color else 0)
            # Optional 1x1 projection to ``screen_channels`` keeps the ResNet small.
            self.proj = (
                nn.Conv2d(in_channels, int(screen_channels), kernel_size=1)
                if int(screen_channels) > 0
                else None
            )
            resnet_in = int(screen_channels) if self.proj is not None else in_channels
            self.resnet = ScreenResNet(
                resnet_in,
                out_dim=int(out_dim),
                activation=activation,
                num_blocks=int(num_blocks),
            )
            self.out_dim = int(out_dim)

        # -- helpers ------------------------------------------------------------------
        def _index_grid(self, tensor: Any) -> Any:
            """Coerce a screen tensor to a long ``(B, H, W)`` index grid."""

            if tensor.dim() == 4:
                # Accept ``(B, 1, H, W)`` or ``(B, C, H, W)`` (take the first channel).
                tensor = tensor[:, 0] if tensor.shape[1] == 1 else tensor[:, 0]
            t = tensor.long()
            if t.dim() == 2:  # single (H, W) grid
                t = t.unsqueeze(0)
            t = t.clamp(min=0, max=self.num_chars - 1)
            return t

        def forward(self, tty_chars: Any, tty_colors: Optional[Any] = None) -> Any:
            chars = self._index_grid(tty_chars)
            embeds = [self.char_embedding(chars)]  # (B, H, W, E)
            if self.color_embedding is not None and tty_colors is not None:
                colors = self._index_grid(tty_colors).clamp(
                    min=0, max=max(self.num_colors - 1, 0)
                )
                embeds.append(self.color_embedding(colors))
            grid = torch.cat(embeds, dim=-1)
            grid = grid.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
            if self.proj is not None:
                grid = F.relu(self.proj(grid))
            return self.resnet(grid)

    class BlstatsEncoder(nn.Module):
        """Two-layer MLP over the 25-dimensional ``blstats`` vector (Appendix B.1)."""

        def __init__(
            self,
            in_dim: int = BLSTATS_DIM,
            hidden_dim: int = DEFAULT_MLP_HIDDEN,
            out_dim: int = DEFAULT_MLP_OUT,
            activation: Any = None,
        ) -> None:
            super().__init__()
            act = activation if activation is not None else nn.ReLU
            self.in_dim = int(in_dim)
            self.out_dim = int(out_dim)
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, int(hidden_dim)),
                _activation_factory(act),
                nn.Linear(int(hidden_dim), self.out_dim),
            )

        def forward(self, blstats: Any) -> Any:
            x = blstats.float() if getattr(blstats, "dtype", None) is not None else blstats
            if x.dim() > 2:
                x = x.reshape(x.shape[0], -1)
            if x.shape[-1] < self.in_dim:
                x = F.pad(x, (0, self.in_dim - x.shape[-1]))
            elif x.shape[-1] > self.in_dim:
                x = x[..., : self.in_dim]
            return self.net(x)

    class MessageEncoder(nn.Module):
        """Two-layer MLP over the message buffer (Appendix B.1).

        ``mode="one_hot"`` flattens the ``(message_length, num_chars)`` one-hot buffer and
        feeds it through the MLP.  ``mode="embedding"`` first embeds the character indices
        (and may be given a boolean mask) and then feeds the flattened embeddings through
        the MLP, mirroring the original Sample-Factory implementation.
        """

        def __init__(
            self,
            in_dim: int = MESSAGE_LENGTH * NUM_MESSAGE_CHARS,
            hidden_dim: int = DEFAULT_MLP_HIDDEN,
            out_dim: int = DEFAULT_MLP_OUT,
            activation: Any = None,
            mode: str = "one_hot",
            num_chars: int = NUM_MESSAGE_CHARS,
            embed_dim: int = 16,
            message_length: int = MESSAGE_LENGTH,
        ) -> None:
            super().__init__()
            act = activation if activation is not None else nn.ReLU
            self.mode = str(mode)
            self.num_chars = int(num_chars)
            self.message_length = int(message_length)
            self.embed_dim = int(embed_dim)
            self.out_dim = int(out_dim)

            if self.mode == "embedding":
                self.char_embedding: Optional[nn.Module] = nn.Embedding(
                    self.num_chars, self.embed_dim
                )
                effective_in = self.message_length * self.embed_dim
            else:
                self.char_embedding = None
                effective_in = int(in_dim)

            self.in_dim = int(effective_in)
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, int(hidden_dim)),
                _activation_factory(act),
                nn.Linear(int(hidden_dim), self.out_dim),
            )

        def _flatten(self, message: Any) -> Any:
            x = message.float() if getattr(message, "dtype", None) is not None else message
            if x.dim() == 1:
                x = x.unsqueeze(0)
            if x.dim() > 2:
                x = x.reshape(x.shape[0], -1)
            return x

        def forward(self, message: Any, mask: Optional[Any] = None) -> Any:
            x = self._flatten(message)
            if self.mode == "embedding" and self.char_embedding is not None:
                idx = x.long().clamp(min=0, max=max(self.num_chars - 1, 0))
                if mask is not None:
                    idx = idx * mask.long()
                emb = self.char_embedding(idx)  # (B, L, E)
                if mask is not None:
                    emb = emb * mask.float().unsqueeze(-1)
                x = emb.reshape(emb.shape[0], -1)
            if x.shape[-1] < self.in_dim:
                x = F.pad(x, (0, self.in_dim - x.shape[-1]))
            elif x.shape[-1] > self.in_dim:
                x = x[..., : self.in_dim]
            return self.net(x)

    class NetHackEncoders(nn.Module):
        """Bundle of the three observation encoders (main screen, blstats, message)."""

        def __init__(self, config: Optional[EncoderConfig] = None, **overrides: Any) -> None:
            super().__init__()
            self.config = (config or EncoderConfig()).with_overrides(**overrides)
            cfg = self.config
            self.main_screen_encoder = MainScreenEncoder(
                num_chars=cfg.num_chars,
                num_colors=cfg.num_colors,
                char_embed_dim=cfg.char_embed_dim,
                color_embed_dim=cfg.color_embed_dim,
                screen_channels=cfg.screen_channels,
                out_dim=cfg.screen_out_dim,
                activation=cfg.activation,
                num_blocks=cfg.num_blocks,
                screen_shape=cfg.main_screen_shape,
                use_color=cfg.use_color,
            )
            self.blstats_encoder = BlstatsEncoder(
                in_dim=cfg.blstats_dim,
                hidden_dim=cfg.blstats_hidden_dim,
                out_dim=cfg.blstats_out_dim,
                activation=cfg.activation,
            )
            self.message_encoder = MessageEncoder(
                in_dim=cfg.flat_message_dim,
                hidden_dim=cfg.message_hidden_dim,
                out_dim=cfg.message_out_dim,
                activation=cfg.activation,
                mode=cfg.message_mode,
                num_chars=cfg.num_message_chars,
                embed_dim=cfg.message_embed_dim,
                message_length=cfg.num_message_tokens,
            )
            self.out_dim = int(cfg.screen_out_dim + cfg.blstats_out_dim + cfg.message_out_dim)

        def forward(self, obs: Any) -> Any:
            """Encode a (batched) observation dict and return merged features."""

            parts = self.encode_parts(obs)
            return torch.cat([parts["screen"], parts["blstats"], parts["message"]], dim=-1)

        def encode_parts(self, obs: Any) -> Dict[str, Any]:
            """Return the three encoded components separately (useful for analysis)."""

            chars, colors, blstats, message, mask = split_observation(obs)
            return {
                "screen": self.main_screen_encoder(chars, colors),
                "blstats": self.blstats_encoder(blstats),
                "message": self.message_encoder(message, mask),
            }

    # Aliases used by the joint model / external checkpoints ---------------------------
    EncoderResNet = ScreenResNet
    ScreenEncoder = MainScreenEncoder
    ObservationEncoders = NetHackEncoders

else:  # pragma: no cover - torch-less fallbacks keep the module importable

    class _TorchMissing:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()

    ResidualBlock = _TorchMissing  # type: ignore[assignment]
    ScreenResNet = _TorchMissing  # type: ignore[assignment]
    EncoderResNet = _TorchMissing  # type: ignore[assignment]
    MainScreenEncoder = _TorchMissing  # type: ignore[assignment]
    ScreenEncoder = _TorchMissing  # type: ignore[assignment]
    BlstatsEncoder = _TorchMissing  # type: ignore[assignment]
    MessageEncoder = _TorchMissing  # type: ignore[assignment]
    NetHackEncoders = _TorchMissing  # type: ignore[assignment]
    ObservationEncoders = _TorchMissing  # type: ignore[assignment]


def _num_groups(channels: int, max_groups: int = 8) -> int:
    """Pick a valid ``GroupNorm`` group count for ``channels``."""

    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _activation_factory(activation: Any) -> Any:
    """Return a callable producing a fresh activation module instance."""

    if activation is None:
        return nn.ReLU()
    if isinstance(activation, type):
        try:
            return activation()
        except Exception:  # pragma: no cover
            return activation
    if isinstance(activation, str):
        return make_activation(activation)
    return activation


# --------------------------------------------------------------------------------------
# Observation handling
# --------------------------------------------------------------------------------------


def split_observation(obs: Any) -> Tuple[Any, Any, Any, Any, Optional[Any]]:
    """Extract ``(tty_chars, tty_colors, blstats, message, message_mask)`` from ``obs``.

    Supports the NLE/Sample-Factory observation dictionary as well as tuples/lists in the
    canonical order ``(tty_chars, tty_colors, blstats, message)``.  ``message_mask`` is
    optional and only used by the embedding-based message encoder.
    """

    if obs is None:  # pragma: no cover - defensive
        raise ValueError("Observation is None; expected a NetHack observation dict.")

    if isinstance(obs, dict) or hasattr(obs, "get"):
        get = obs.get if isinstance(obs, dict) else obs.get  # type: ignore[assignment]
        chars = get("tty_chars", None)
        if chars is None:
            chars = get("main_screen", None)
        if chars is None:
            chars = get("screen", None)
        colors = get("tty_colors", None)
        if colors is None:
            colors = get("colors", None)
        blstats = get("blstats", None)
        message = get("message", None)
        if message is None:
            message = get("tty_message", None)
        mask = get("message_mask", None)
        return chars, colors, blstats, message, mask

    if isinstance(obs, (tuple, list)):
        items = list(obs)
        while len(items) < 4:
            items.append(None)
        mask = items[4] if len(items) > 4 else None
        return items[0], items[1], items[2], items[3], mask

    raise TypeError(f"Unsupported observation type: {type(obs)!r}")


def observation_shapes(obs: Any) -> Dict[str, Any]:
    """Best-effort shape report for a (possibly batched) NetHack observation."""

    chars, colors, blstats, message, mask = split_observation(obs)
    report: Dict[str, Any] = {}

    def _shape(x: Any) -> Optional[Tuple[int, ...]]:
        if x is None:
            return None
        if hasattr(x, "shape"):
            return tuple(int(s) for s in x.shape)
        try:  # pragma: no cover - nested sequences
            return (len(x),)  # type: ignore[arg-type]
        except Exception:
            return None

    report["tty_chars"] = _shape(chars)
    report["tty_colors"] = _shape(colors)
    report["blstats"] = _shape(blstats)
    report["message"] = _shape(message)
    report["message_mask"] = _shape(mask)
    return report


# --------------------------------------------------------------------------------------
# Construction / freezing helpers (used by the fine-tuning runners)
# --------------------------------------------------------------------------------------


def build_encoders(
    config: Union[EncoderConfig, Any, None] = None,
    *,
    from_model_config: bool = True,
    **overrides: Any,
) -> Any:
    """Build the :class:`NetHackEncoders` bundle from a config object or kwargs."""

    _require_torch()
    if isinstance(config, EncoderConfig) or config is None:
        cfg = config or EncoderConfig()
        cfg = cfg.with_overrides(**overrides)
    elif from_model_config:
        cfg = EncoderConfig.from_config(config, **overrides)
    else:  # pragma: no cover - explicit EncoderConfig expected
        cfg = EncoderConfig().with_overrides(**overrides)
    return NetHackEncoders(cfg)


def encoder_output_dim(config: Union[EncoderConfig, Any, None] = None, **overrides: Any) -> int:
    """Merged feature width produced by the three encoders (input to the LSTM)."""

    if isinstance(config, EncoderConfig) or config is None:
        cfg = (config or EncoderConfig()).with_overrides(**overrides)
    else:
        cfg = EncoderConfig.from_config(config, **overrides)
    return int(cfg.screen_out_dim + cfg.blstats_out_dim + cfg.message_out_dim)


def encoder_state_dicts(model: Any) -> Dict[str, Dict[str, Any]]:
    """Collect the ``state_dict`` of every encoder submodule of ``model``."""

    out: Dict[str, Dict[str, Any]] = {}
    for name in ENCODER_NAMES:
        module = getattr(model, name, None)
        if module is None:
            continue
        if hasattr(module, "state_dict"):
            out[name] = module.state_dict()
    return out


def load_encoder_state_dicts(model: Any, state: Dict[str, Any], strict: bool = False) -> List[str]:
    """Load encoder parameters previously captured by :func:`encoder_state_dicts`.

    Returns the list of encoder names that were successfully loaded.
    """

    loaded: List[str] = []
    for name, sd in (state or {}).items():
        module = getattr(model, name, None)
        if module is None or not hasattr(module, "load_state_dict"):
            continue
        try:
            module.load_state_dict(sd, strict=strict)
            loaded.append(name)
        except Exception:  # pragma: no cover - checkpoint mismatch
            continue
    return loaded


def freeze_encoders(model: Any, names: Optional[Iterable[str]] = None) -> List[str]:
    """Freeze the observation encoders of the joint model.

    Appendix B.1: "To improve the stability of the models we froze the encoders during
    the course of the training."  Returns the names of the frozen submodules.
    """

    candidates = list(names) if names is not None else list(ENCODER_NAMES)
    frozen: List[str] = []
    for name in candidates:
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in getattr(module, "parameters", lambda: [])():
            param.requires_grad_(False)
        if hasattr(module, "eval"):
            try:
                module.train(False)
            except Exception:  # pragma: no cover
                pass
        frozen.append(name)
    return frozen


def unfreeze_encoders(model: Any, names: Optional[Iterable[str]] = None) -> List[str]:
    """Undo :func:`freeze_encoders` (used by the from-scratch baseline)."""

    candidates = list(names) if names is not None else list(ENCODER_NAMES)
    thawed: List[str] = []
    for name in candidates:
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in getattr(module, "parameters", lambda: [])():
            param.requires_grad_(True)
        thawed.append(name)
    return thawed


def count_parameters(module: Any, trainable_only: bool = False) -> int:
    """Number of parameters of ``module`` (optionally only trainable ones)."""

    total = 0
    for param in getattr(module, "parameters", lambda: [])():
        if trainable_only and not getattr(param, "requires_grad", True):
            continue
        total += int(param.numel())
    return total


def encoder_summary(model: Any) -> Dict[str, Dict[str, Any]]:
    """Parameter counts / trainability report for every encoder of ``model``."""

    report: Dict[str, Dict[str, Any]] = {}
    for name in ENCODER_NAMES:
        module = getattr(model, name, None)
        if module is None:
            continue
        report[name] = {
            "parameters": count_parameters(module),
            "trainable": count_parameters(module, trainable_only=True),
        }
    return report


def summary(encoders: Any) -> Dict[str, Any]:
    """Compact summary of an encoder bundle (or joint model)."""

    return {
        "output_dim": int(getattr(encoders, "out_dim", encoder_output_dim())),
        "parameters": count_parameters(encoders),
        "modules": encoder_summary(encoders) or {
            name: {"parameters": count_parameters(getattr(encoders, name))}
            for name in ("main_screen_encoder", "blstats_encoder", "message_encoder")
            if getattr(encoders, name, None) is not None
        },
    }


# --------------------------------------------------------------------------------------
# Smoke test / CLI
# --------------------------------------------------------------------------------------


def smoke_test(batch_size: int = 2, device: str = "cpu") -> Dict[str, Any]:
    """Instantiate the encoders and run a dummy forward pass (CPU-friendly)."""

    _require_torch()
    encoders = build_encoders().to(device)
    t, h, w = 1, MAIN_SCREEN_SHAPE[0], MAIN_SCREEN_SHAPE[1]
    obs = {
        "tty_chars": torch.randint(0, NUM_CHARS, (batch_size, t, h, w), dtype=torch.long),
        "tty_colors": torch.randint(0, NUM_COLORS, (batch_size, t, h, w), dtype=torch.long),
        "blstats": torch.randn(batch_size, t, BLSTATS_DIM),
        "message": torch.randn(batch_size, t, MESSAGE_LENGTH, NUM_MESSAGE_CHARS),
    }
    # Flatten the time dimension into the batch dimension for a single-step forward.
    flat = {k: v.reshape(batch_size * t, *v.shape[2:]) for k, v in obs.items()}
    with torch.no_grad():
        merged = encoders(flat)
    return {
        "output_shape": tuple(merged.shape),
        "output_dim": int(merged.shape[-1]),
        "expected_dim": encoder_output_dim(),
        "parameters": count_parameters(encoders),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NetHack observation encoders (Appendix B.1)")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[], help="key=value")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--no-smoke-test", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = None
    if args.config:
        try:
            from src.common.config import load_config, apply_overrides  # pragma: no cover

            cfg = load_config(args.config)
            if args.overrides:
                apply_overrides(cfg, args.overrides)
        except Exception:  # pragma: no cover - config optional
            cfg = None

    enc_cfg = EncoderConfig.from_config(cfg)
    print(f"Encoder config: {enc_cfg.to_dict()}")
    print(f"Merged encoder output dim: {encoder_output_dim(enc_cfg)}")

    if not args.no_smoke_test:
        try:
            report = smoke_test(batch_size=args.batch_size, device=args.device)
            for key, value in report.items():
                print(f"{key}: {value}")
        except RuntimeError as exc:  # torch missing
            print(f"Skipping smoke test: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
