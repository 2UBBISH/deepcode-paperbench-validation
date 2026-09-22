"""NetHack actor/critic model (Appendix B.1, Architecture).

Reproduces the joint actor/critic backbone used for the NetHack Human Monk
experiments of Wolczyk et al. (2024):

    "This model utilizes an LSTM architecture that incorporates representations
     from three encoders, which take observations as inputs. The LSTM network's
     output is then fed into two separate heads: a policy head and a baseline
     head. The model architecture used both in online and offline settings
     consists of a joint backbone for both actor and critic. It takes as an
     input three components: main observation of the dungeon screen, blstats,
     and message. ... blstats and message are processed using two layer MLP. The
     main observation of the dungeon screen is processed by embedding each
     character and color in an embedding lookup table which is later put into a
     grid processed by ResNet. ... The components are encoded, and are merged
     before passing to LSTM."                                            (§B.1)

Layout implemented here::

    tty_chars  (24, 80) --char embedding--\\
                                          >-- concat -> ResNet -> GAP -\\
    tty_colors (24, 80) --color embedding-/                          \\
                                                                      \\
    blstats     (25,)  ------------------ 2-layer MLP ---------------- >-- concat
                                                                      /      |
    message  (256, 128) one-hot ---------- 2-layer MLP --------------/       |
                                                                            v
                                              Linear(enc, hidden_dim) -> LSTM(hidden_dim)
                                                                            |
                                              +-----------------------------+
                                              v                             v
                                        policy head (120)            baseline head (1)

Model hyperparameters used by the paper (Table 1): activation ``relu``,
``hidden_dim = 1738``, Adam lr ``1e-4`` with weight decay ``1e-4``,
``batch_size = 128``, ``unroll_length = 32``.

The pre-trained ``pi_*`` (Tuyls et al., 2023, 30M-parameter LSTM) is loaded into
this module; the baseline head is additionally pre-trained for 500M environment
steps with the rest of the network frozen, and the encoders are frozen during
fine-tuning (both handled by :func:`freeze_encoders` / :meth:`NetHackModel.set_phase`).

Torch is imported defensively so that the module can be imported (and the stub
model exercised) in CPU-only / dependency-free environments.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is optional at import time
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


# --------------------------------------------------------------------------------------
# Constants (NLE observation / action spaces and Table 1 defaults)
# --------------------------------------------------------------------------------------

NUM_ACTIONS: int = 120                 # NLE action space size (§B.1)
HIDDEN_DIM: int = 1738                 # Table 1 "hidden_dim"
MAIN_SCREEN_SHAPE: Tuple[int, int] = (24, 80)
BLSTATS_SHAPE: Tuple[int, ...] = (25,)
MESSAGE_SHAPE: Tuple[int, ...] = (256, 128)   # one-hot message (256 tokens x 128 chars)
MESSAGE_DIM: int = 256 * 128
NUM_CHARS: int = 256                   # tty_chars alphabet size
NUM_COLORS: int = 16                   # tty_colors palette size

TABLE1_MODEL_DEFAULTS: Dict[str, Any] = {
    "activation_function": "relu",
    "hidden_dim": HIDDEN_DIM,
    "adam_learning_rate": 1e-4,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-7,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "unroll_length": 32,
    "discounting": 0.999999,
    "entropy_cost": 0.001,
    "grad_norm_clipping": 4.0,
    "appo_clip_policy": 0.1,
    "appo_clip_baseline": 1.0,
    "baseline_cost": 1.0,
    "reward_clip": 10.0,
    "reward_scale": 1.0,
}

# Architecture constants not enumerated in the paper (Appendix: "ResNet/MLP encoder
# widths not specified -> reuse the released Tuyls et al. checkpoint"); these are the
# defaults used when a session is constructed from scratch rather than loaded.
DEFAULT_CHAR_EMBED_DIM: int = 32
DEFAULT_COLOR_EMBED_DIM: int = 16
DEFAULT_SCREEN_CHANNELS: Tuple[int, ...] = (32, 32, 32)
DEFAULT_SCREEN_OUT_DIM: int = 128
DEFAULT_MLP_HIDDEN: int = 128
DEFAULT_MLP_OUT: int = 128
DEFAULT_LSTM_LAYERS: int = 1
DEFAULT_POLICY_HIDDEN: int = 0   # 0 == linear head straight off the LSTM

__all__ = [
    "NetHackModelConfig",
    "ModelOutput",
    "MainScreenEncoder",
    "MessageEncoder",
    "BlstatsEncoder",
    "ResidualBlock",
    "ScreenResNet",
    "NetHackModel",
    "NetHackActor",
    "MiniNetHackModel",
    "build_model",
    "build_nethack_model",
    "build_actor",
    "load_lstm_checkpoint",
    "load_pretrained",
    "freeze_encoders",
    "freeze_module",
    "set_requires_grad",
    "parameter_groups",
    "count_parameters",
    "parameter_names",
    "summary",
    "save_checkpoint",
    "load_checkpoint",
    "NUM_ACTIONS",
    "HIDDEN_DIM",
    "MAIN_SCREEN_SHAPE",
    "BLSTATS_SHAPE",
    "MESSAGE_SHAPE",
    "TABLE1_MODEL_DEFAULTS",
]


def _require_torch() -> None:
    if not _HAS_TORCH:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for src.nethack.model but is not installed. "
            "Install torch (CUDA build for the NetHack experiments)."
        )


# --------------------------------------------------------------------------------------
# Small containers / helpers
# --------------------------------------------------------------------------------------


class ModelOutput(dict):
    """Dict with attribute access, mirroring Sample Factory's ``ModelOutput``.

    Standard keys: ``policy_logits``, ``baseline`` (alias ``value``), ``state``
    (LSTM ``(h, c)``), ``lstm_state``, ``features``/``embeddings``.
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    @property
    def policy_logits(self) -> Any:
        return self.get("policy_logits")

    @property
    def value(self) -> Any:
        return self.get("baseline", self.get("value"))

    @property
    def state(self) -> Any:
        return self.get("state", self.get("lstm_state"))

    def squeeze_value(self) -> Any:
        value = self.value
        if value is not None and hasattr(value, "squeeze"):
            return value.squeeze(-1)
        return value


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Tolerant getter working with ``Config``/dict/dataclass/argparse objects."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        if key in cfg:
            return cfg[key]
        cur: Any = cfg
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            elif hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                return default
        return cur
    cur = cfg
    for part in key.split("."):
        if isinstance(cur, Mapping) and part in cur:  # pragma: no cover - mixed
            cur = cur[part]
        elif hasattr(cur, part):
            cur = getattr(cur, part)
        else:
            return default
    return cur


def activation_from_name(name: Optional[str]) -> Any:
    """Resolve the Table 1 ``activation_function`` (default ``relu``)."""
    _require_torch()
    name = (name or "relu").lower()
    mapping = {
        "relu": nn.ReLU,
        "leaky_relu": lambda: nn.LeakyReLU(0.01),
        "gelu": nn.GELU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }
    factory = mapping.get(name, nn.ReLU)
    return factory()


def _as_int_tuple(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    return (int(value),)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class NetHackModelConfig:
    """Model / optimiser hyperparameters for the NetHack joint actor-critic.

    Defaults follow Table 1 of the paper; architecture widths not tabulated in
    the paper keep the values of the released Tuyls et al. (2023) checkpoint
    ("ResNet/MLP encoder widths not specified -> reuse the released checkpoint").
    """

    # Observation space
    main_screen_shape: Tuple[int, int] = MAIN_SCREEN_SHAPE
    blstats_shape: Tuple[int, ...] = BLSTATS_SHAPE
    message_shape: Tuple[int, ...] = MESSAGE_SHAPE
    num_chars: int = NUM_CHARS
    num_colors: int = NUM_COLORS
    num_actions: int = NUM_ACTIONS
    message_encoding: str = "one_hot"       # "one_hot" | "embedding"

    # Encoders
    char_embed_dim: int = DEFAULT_CHAR_EMBED_DIM
    color_embed_dim: int = DEFAULT_COLOR_EMBED_DIM
    screen_channels: Tuple[int, ...] = DEFAULT_SCREEN_CHANNELS
    screen_out_dim: int = DEFAULT_SCREEN_OUT_DIM
    screen_blocks: int = 2
    mlp_hidden: int = DEFAULT_MLP_HIDDEN
    mlp_out: int = DEFAULT_MLP_OUT
    message_out_dim: Optional[int] = None    # defaults to mlp_out
    blstats_out_dim: Optional[int] = None    # defaults to mlp_out

    # Backbone / heads (Table 1)
    hidden_dim: int = HIDDEN_DIM
    lstm_layers: int = DEFAULT_LSTM_LAYERS
    policy_hidden: int = DEFAULT_POLICY_HIDDEN
    activation: str = "relu"
    activation_function: Optional[str] = None    # Table 1 alias
    use_layernorm: bool = False

    # Optimiser settings from Table 1 (kept here for convenience of the runners)
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-7
    weight_decay: float = 1e-4
    batch_size: int = 128
    unroll_length: int = 32

    device: str = "cpu"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.activation_function:
            self.activation = self.activation_function
        self.main_screen_shape = tuple(int(v) for v in self.main_screen_shape)
        self.blstats_shape = tuple(int(v) for v in self.blstats_shape)
        self.message_shape = tuple(int(v) for v in self.message_shape)
        self.screen_channels = tuple(int(v) for v in self.screen_channels)
        if self.message_out_dim is None:
            self.message_out_dim = self.mlp_out
        if self.blstats_out_dim is None:
            self.blstats_out_dim = self.mlp_out

    # -- construction ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for f in fields(self):
            out[f.name] = getattr(self, f.name)
        return out

    def with_overrides(self, **overrides: Any) -> "NetHackModelConfig":
        data = self.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            if key in data:
                data[key] = value
        return NetHackModelConfig(**data)

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "NetHackModelConfig":
        """Build from a YAML config object, a mapping, or another config dataclass.

        Recognises ``model``/``nethack``/``obs``/``env`` sub-blocks as well as the
        Table 1 hyperparameter names (``hidden_dim``, ``adam_learning_rate``,
        ``activation_function``, ``unroll_length``, ...).
        """
        kwargs: Dict[str, Any] = {}

        def put(name: str, *paths: str) -> None:
            for path in paths:
                value = _cfg_get(cfg, path, None)
                if value is not None:
                    kwargs.setdefault(name, value)
                    return

        put("num_actions", "model.num_actions", "env.num_actions", "num_actions")
        put("hidden_dim", "model.hidden_dim", "nethack.hidden_dim", "hidden_dim")
        put("lstm_layers", "model.lstm_layers", "model.num_lstm_layers", "lstm_layers")
        put("activation_function", "model.activation_function",
            "model.activation", "activation_function")
        put("screen_out_dim", "model.screen_out_dim", "model.main_screen_out_dim")
        put("screen_channels", "model.screen_channels")
        put("char_embed_dim", "model.char_embed_dim", "model.character_embedding_dim")
        put("color_embed_dim", "model.color_embed_dim", "model.color_embedding_dim")
        put("mlp_hidden", "model.mlp_hidden", "model.encoder_hidden_dim")
        put("mlp_out", "model.mlp_out", "model.encoder_out_dim")
        put("message_encoding", "model.message_encoding")
        put("learning_rate", "model.adam_learning_rate", "learning_rate", "lr")
        put("adam_beta1", "model.adam_beta1")
        put("adam_beta2", "model.adam_beta2")
        put("adam_eps", "model.adam_eps")
        put("weight_decay", "model.weight_decay")
        put("batch_size", "model.batch_size", "batch_size")
        put("unroll_length", "model.unroll_length", "unroll_length", "unroll")
        put("device", "compute.device", "device")
        put("seed", "seed")

        for key, value in overrides.items():
            if value is not None:
                kwargs[key] = value
        return cls(**kwargs)

    # -- shape helpers -----------------------------------------------------------------

    @property
    def num_message_tokens(self) -> int:
        if len(self.message_shape) >= 2:
            return int(self.message_shape[0])
        return 256

    @property
    def num_message_chars(self) -> int:
        if len(self.message_shape) >= 2:
            return int(self.message_shape[1])
        return 128

    @property
    def flat_message_dim(self) -> int:
        if self.message_encoding == "embedding":
            return int(self.num_message_tokens)
        dim = 1
        for size in self.message_shape:
            dim *= int(size)
        return dim

    @property
    def blstats_dim(self) -> int:
        dim = 1
        for size in self.blstats_shape:
            dim *= int(size)
        return dim


# --------------------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Standard pre-activation-free residual block (conv3x3 -> act -> conv3x3)."""

    def __init__(self, in_channels: int, out_channels: int, activation: Any = None,
                 stride: int = 1) -> None:
        super().__init__()
        _require_torch()
        act = activation if activation is not None else nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, out_channels) if out_channels > 1 else nn.Identity()
        self.act1 = act
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                               bias=False)
        self.bn2 = nn.GroupNorm(1, out_channels) if out_channels > 1 else nn.Identity()
        self.act2 = nn.ReLU() if not isinstance(act, nn.ReLU) else nn.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                 stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Any) -> Any:
        identity = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + identity)


class ScreenResNet(nn.Module):
    """Small ResNet applied to the embedded main-screen grid."""

    def __init__(self, in_channels: int, channels: Sequence[int] = DEFAULT_SCREEN_CHANNELS,
                 out_dim: int = DEFAULT_SCREEN_OUT_DIM, activation: Any = None,
                 num_blocks: int = 2) -> None:
        super().__init__()
        _require_torch()
        channels = [int(c) for c in channels] or [in_channels]
        act = activation if activation is not None else nn.ReLU()
        self.stem = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False)
        self.stem_norm = nn.GroupNorm(1, channels[0]) if channels[0] > 1 else nn.Identity()
        blocks: List[nn.Module] = []
        prev = channels[0]
        for i, ch in enumerate(channels):
            n = num_blocks if i == 0 else 1
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(ResidualBlock(prev, ch, activation=act, stride=stride))
                prev = ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(prev, int(out_dim))
        self.out_dim = int(out_dim)
        self.activation = act

    def forward(self, x: Any) -> Any:
        x = self.activation(self.stem_norm(self.stem(x)))
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.activation(self.proj(x))


class MainScreenEncoder(nn.Module):
    """Main-screen encoder: character + colour embedding lookups -> ResNet (§B.1)."""

    def __init__(self, num_chars: int = NUM_CHARS, num_colors: int = NUM_COLORS,
                 char_embed_dim: int = DEFAULT_CHAR_EMBED_DIM,
                 color_embed_dim: int = DEFAULT_COLOR_EMBED_DIM,
                 screen_channels: Sequence[int] = DEFAULT_SCREEN_CHANNELS,
                 out_dim: int = DEFAULT_SCREEN_OUT_DIM, activation: Any = None,
                 num_blocks: int = 2, screen_shape: Tuple[int, int] = MAIN_SCREEN_SHAPE,
                 use_color: bool = True) -> None:
        super().__init__()
        _require_torch()
        self.num_chars = int(num_chars)
        self.num_colors = int(num_colors)
        self.char_embed_dim = int(char_embed_dim)
        self.color_embed_dim = int(color_embed_dim)
        self.screen_shape = tuple(int(v) for v in screen_shape)
        self.use_color = bool(use_color)
        self.char_embedding = nn.Embedding(self.num_chars, self.char_embed_dim)
        self.color_embedding = (
            nn.Embedding(self.num_colors, self.color_embed_dim) if self.use_color else None
        )
        in_channels = self.char_embed_dim + (self.color_embed_dim if self.use_color else 0)
        self.resnet = ScreenResNet(
            in_channels, channels=screen_channels, out_dim=out_dim,
            activation=activation, num_blocks=num_blocks,
        )
        self.out_dim = int(out_dim)

    def forward(self, tty_chars: Any, tty_colors: Any = None) -> Any:
        chars = tty_chars.long().clamp_(0, self.num_chars - 1)
        if chars.dim() == 2:  # (H, W) unbatched
            chars = chars.unsqueeze(0)
        embeds = self.char_embedding(chars)                 # (B, H, W, C_c)
        embeds = embeds.permute(0, 3, 1, 2)                 # (B, C_c, H, W)
        if self.use_color and self.color_embedding is not None:
            if tty_colors is None:
                colors = torch.zeros_like(chars)
            else:
                colors = tty_colors.long()
                if colors.dim() == 2:
                    colors = colors.unsqueeze(0)
                colors = colors.clamp_(0, self.num_colors - 1)
            color_embeds = self.color_embedding(colors).permute(0, 3, 1, 2)
            embeds = torch.cat([embeds, color_embeds], dim=1)
        return self.resnet(embeds)


class BlstatsEncoder(nn.Module):
    """Two-layer MLP over the player status vector (``blstats``, 25 dims)."""

    def __init__(self, in_dim: int = 25, hidden_dim: int = DEFAULT_MLP_HIDDEN,
                 out_dim: int = DEFAULT_MLP_OUT, activation: Any = None) -> None:
        super().__init__()
        _require_torch()
        act = activation if activation is not None else nn.ReLU()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden_dim)),
            act,
            nn.Linear(int(hidden_dim), self.out_dim),
            act if not isinstance(act, nn.ReLU) else nn.ReLU(),
        )

    def forward(self, blstats: Any) -> Any:
        x = blstats
        if x.dim() > 2:
            x = x.flatten(1)
        x = x.float()
        return self.net(x)


class MessageEncoder(nn.Module):
    """Two-layer MLP over the one-hot message (or an embedding over char ids)."""

    def __init__(self, in_dim: int = MESSAGE_DIM, hidden_dim: int = DEFAULT_MLP_HIDDEN,
                 out_dim: int = DEFAULT_MLP_OUT, activation: Any = None,
                 mode: str = "one_hot", num_chars: int = 128,
                 embed_dim: int = 16) -> None:
        super().__init__()
        _require_torch()
        act = activation if activation is not None else nn.ReLU()
        self.mode = str(mode)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if self.mode == "embedding":
            self.embedding: Optional[nn.Module] = nn.Embedding(int(num_chars), int(embed_dim))
            mlp_in = 0  # resolved lazily from the input shape
            self.net = None
            self.embed_dim = int(embed_dim)
        else:
            self.embedding = None
            self.net = nn.Sequential(
                nn.Linear(self.in_dim, int(hidden_dim)),
                act,
                nn.Linear(int(hidden_dim), self.out_dim),
                act if not isinstance(act, nn.ReLU) else nn.ReLU(),
            )
            self.embed_dim = 0
        self.hidden_dim = int(hidden_dim)
        self.activation = act

    def _build_embedding_mlp(self, in_dim: int) -> None:
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), self.hidden_dim),
            self.activation,
            nn.Linear(self.hidden_dim, self.out_dim),
            self.activation if not isinstance(self.activation, nn.ReLU) else nn.ReLU(),
        )
        if self.net is not None and getattr(self, "device", None) is not None:
            self.net = self.net.to(self.device)

    def forward(self, message: Any) -> Any:
        x = message
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if self.mode == "embedding" and self.embedding is not None:
            ids = x.long().clamp_(0, self.embedding.num_embeddings - 1)
            emb = self.embedding(ids)                     # (B, T, E)
            flat = emb.flatten(1)
            if self.net is None:
                self._build_embedding_mlp(flat.shape[-1])
            return self.net(flat)
        x = x.float()
        if x.dim() > 2:
            x = x.flatten(1)
        if self.net is None:  # pragma: no cover - one-hot MLP always pre-built
            self._build_embedding_mlp(x.shape[-1])
        return self.net(x)


# --------------------------------------------------------------------------------------
# Main model
# --------------------------------------------------------------------------------------


class NetHackActor(nn.Module):
    """Actor-only view of the joint network (backbone + policy head, no baseline head).

    Retention losses (EWC / BC / Kickstarting) are applied to the actor only —
    the critic coefficient is always 0 (§2, §C) — so runners can simply pass
    ``model.actor`` to the retention objects.
    """

    def __init__(self, model: "NetHackModel") -> None:
        super().__init__()
        _require_torch()
        self.model = model

    def forward(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.model.policy_logits(obs, state=state, **kwargs)

    def distribution(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        logits = self.forward(obs, state=state, **kwargs)
        return torch.distributions.Categorical(logits=logits)

    def log_prob(self, obs: Any, actions: Any = None, state: Any = None,
                 **kwargs: Any) -> Any:
        if actions is None and isinstance(obs, Mapping):
            actions = obs.get("actions", obs.get("action"))
        dist = self.distribution(obs, state=state, **kwargs)
        return dist.log_prob(actions.long())

    def act(self, obs: Any, deterministic: bool = False, **kwargs: Any) -> Any:
        logits = self.forward(obs, **kwargs)
        if deterministic:
            return logits.argmax(-1)
        return torch.distributions.Categorical(logits=logits).sample()


class NetHackModel(nn.Module):
    """Joint actor/critic backbone for NLE (Appendix B.1, Architecture).

    Forward accepts a single-step observation dict (``tty_chars``, ``tty_colors``,
    ``blstats``, ``message``) or a whole unroll when ``sequence=True`` (time-major
    by default, matching Sample Factory's rollout buffers). It returns a
    :class:`ModelOutput` with ``policy_logits`` (B, 120), ``baseline`` (B, 1) and the
    new LSTM ``state``.
    """

    def __init__(self, config: Optional[NetHackModelConfig] = None, **overrides: Any) -> None:
        super().__init__()
        _require_torch()
        self.config = (config or NetHackModelConfig()).with_overrides(**overrides)
        cfg = self.config

        act = activation_from_name(cfg.activation)
        self.activation_name = cfg.activation

        self.main_screen_encoder = MainScreenEncoder(
            num_chars=cfg.num_chars,
            num_colors=cfg.num_colors,
            char_embed_dim=cfg.char_embed_dim,
            color_embed_dim=cfg.color_embed_dim,
            screen_channels=cfg.screen_channels,
            out_dim=cfg.screen_out_dim,
            activation=act,
            num_blocks=cfg.screen_blocks,
            screen_shape=cfg.main_screen_shape,
        )
        self.blstats_encoder = BlstatsEncoder(
            in_dim=cfg.blstats_dim, hidden_dim=cfg.mlp_hidden,
            out_dim=int(cfg.blstats_out_dim or cfg.mlp_out), activation=act,
        )
        self.message_encoder = MessageEncoder(
            in_dim=cfg.flat_message_dim, hidden_dim=cfg.mlp_hidden,
            out_dim=int(cfg.message_out_dim or cfg.mlp_out), activation=act,
            mode=cfg.message_encoding,
            num_chars=cfg.num_message_chars,
        )

        merged_dim = (
            self.main_screen_encoder.out_dim
            + self.blstats_encoder.out_dim
            + self.message_encoder.out_dim
        )
        self.merged_dim = merged_dim
        self.merge = nn.Sequential(
            nn.Linear(merged_dim, cfg.hidden_dim),
            nn.ReLU() if cfg.activation == "relu" else activation_from_name(cfg.activation),
        )
        if cfg.use_layernorm:
            self.merge.add_module("norm", nn.LayerNorm(cfg.hidden_dim))

        self.lstm = nn.LSTM(
            input_size=cfg.hidden_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.lstm_layers,
        )
        self.hidden_dim = cfg.hidden_dim
        self.num_layers = cfg.lstm_layers

        if cfg.policy_hidden and cfg.policy_hidden > 0:
            self.policy_head: nn.Module = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.policy_hidden),
                nn.ReLU() if cfg.activation == "relu" else activation_from_name(cfg.activation),
                nn.Linear(cfg.policy_hidden, cfg.num_actions),
            )
        else:
            self.policy_head = nn.Linear(cfg.hidden_dim, cfg.num_actions)
        self.baseline_head = nn.Linear(cfg.hidden_dim, 1)

        self.actor = NetHackActor(self)
        self._init_parameters()

    # -- initialisation ----------------------------------------------------------------

    def _init_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LSTM):  # pragma: no cover - stable init
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.orthogonal_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)
        # Small policy-head initialisation (standard PPO trick) and zero baseline bias.
        last = self.policy_head[-1] if isinstance(self.policy_head, nn.Sequential) else self.policy_head
        if isinstance(last, nn.Linear):
            nn.init.orthogonal_(last.weight, gain=0.01)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        nn.init.orthogonal_(self.baseline_head.weight, gain=1.0)
        nn.init.zeros_(self.baseline_head.bias)

    # -- observation handling ----------------------------------------------------------

    @staticmethod
    def _field(obs: Any, *keys: str, default: Any = None) -> Any:
        if obs is None:
            return default
        if isinstance(obs, Mapping):
            for key in keys:
                if key in obs:
                    return obs[key]
            return default
        for key in keys:
            if hasattr(obs, key):
                return getattr(obs, key)
        return default

    @staticmethod
    def _to_tensor(x: Any, device: Any = None, dtype: Any = None) -> Any:
        if x is None:
            return None
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if dtype is not None:
            x = x.to(dtype)
        if device is not None and x.device != device:
            x = x.to(device)
        return x

    def _split_sequence(self, x: Any, time_major: bool) -> Tuple[Any, bool]:
        """Flatten a possible (T, B, ...) / (B, T, ...) observation into (T*B, ...).

        Returns the flattened tensor and whether a sequence dimension was present.
        """
        if x is None or not torch.is_tensor(x):
            return x, False
        if x.dim() == 3 and x.shape[-1] == 2 and x.shape[-2] == 1:
            return x, False
        # (T, B, H) or (B, T, H): H is the last dim; screens are 5D (T, B, H, W).
        if x.dim() == 5:
            time_major_ = True if time_major else False
            if time_major_:
                t, b = x.shape[0], x.shape[1]
                x = x.reshape(t * b, *x.shape[2:])
            else:
                b, t = x.shape[0], x.shape[1]
                x = x.transpose(0, 1).reshape(t * b, *x.shape[2:])
            return x, True
        if x.dim() == 4 and x.shape[-1] > 4 and x.shape[-2] > 4:
            # Ambiguous 4D: (T, B, H) vs (B, T, H). Batch-first containers produced by
            # APPO store unrolls as (T, B, H); treat dim 0 as time unless told otherwise.
            if time_major:
                t, b = x.shape[0], x.shape[1]
                x = x.reshape(t * b, *x.shape[2:])
            else:
                b, t = x.shape[0], x.shape[1]
                x = x.transpose(0, 1).reshape(t * b, *x.shape[2:])
            return x, True
        return x, False

    # -- forward -----------------------------------------------------------------------

    def encode(self, obs: Any, time_major: bool = True) -> Dict[str, Any]:
        """Encode an observation (or a full unroll) into merged features."""
        tty_chars = self._field(obs, "tty_chars", "screen", "main_screen")
        tty_colors = self._field(obs, "tty_colors", "colors", "screen_colors")
        blstats = self._field(obs, "blstats", "stats")
        message = self._field(obs, "message", "tty_message", "msg")

        device = next(self.parameters()).device
        tty_chars = self._to_tensor(tty_chars, device=device)
        tty_colors = self._to_tensor(tty_colors, device=device)
        blstats = self._to_tensor(blstats, device=device)
        message = self._to_tensor(message, device=device)

        sequence = False
        for name, tensor in (("tty_chars", tty_chars), ("blstats", blstats), ("message", message)):
            if tensor is not None and tensor.dim() >= 4 and name != "tty_chars":
                _, sequence = self._split_sequence(tensor, time_major)
            if tensor is not None and name == "tty_chars" and tensor.dim() == 4:
                _, sequence = self._split_sequence(tensor, time_major)

        tty_chars, _ = self._split_sequence(tty_chars, time_major)
        tty_colors, _ = self._split_sequence(tty_colors, time_major)
        blstats, _ = self._split_sequence(blstats, time_major)
        message, _ = self._split_sequence(message, time_major)

        if tty_chars is None:  # pragma: no cover - defensive
            raise ValueError("NetHackModel expects an observation with 'tty_chars'.")

        screen_feat = self.main_screen_encoder(tty_chars, tty_colors)
        blstats_feat = self.blstats_encoder(blstats) if blstats is not None else \
            torch.zeros(screen_feat.shape[0], self.blstats_encoder.out_dim, device=device)
        if message is not None:
            message_feat = self.message_encoder(message)
        else:
            message_feat = torch.zeros(screen_feat.shape[0], self.message_encoder.out_dim,
                                       device=device)

        merged = torch.cat([screen_feat, blstats_feat, message_feat], dim=-1)
        features = self.merge(merged)
        return {
            "features": features,
            "screen": screen_feat,
            "blstats": blstats_feat,
            "message": message_feat,
            "batch_size": int(features.shape[0]),
            "sequence": sequence,
        }

    def forward(self, obs: Any, state: Any = None, sequence: Optional[bool] = None,
                time_major: bool = True, return_state: bool = True,
                keep_sequence: bool = False) -> ModelOutput:
        """Run the full backbone.

        Args:
            obs: observation mapping (single step or an unroll).
            state: optional LSTM ``(h, c)`` tuple of shape ``(num_layers, B, hidden)``.
            sequence: force sequence mode on/off (auto-detected otherwise).
            time_major: sequence layout is ``(T, B, ...)`` when True.
            return_state: include the new LSTM state in the output.
            keep_sequence: return ``(T, B, ...)`` shaped heads instead of ``(T*B, ...)``.
        """
        # Resolve the sequence layout from a single representative field so that all
        # observation fields are reshaped consistently.
        rep = self._field(obs, "blstats", "tty_chars", "message")
        if sequence is None and rep is not None and torch.is_tensor(rep):
            if rep.dim() >= 4:
                sequence = True
            elif rep.dim() == 3 and rep.shape[-1] not in (2,):
                sequence = False
            else:
                sequence = False
        sequence = bool(sequence)
        t0 = b0 = None
        if sequence:
            t0, b0 = (int(rep.shape[0]), int(rep.shape[1])) if time_major else \
                (int(rep.shape[1]), int(rep.shape[0]))

        encoded = self.encode(obs, time_major=time_major)
        features = encoded["features"]
        n = features.shape[0]
        batch = b0 if (sequence and b0) else n

        if sequence and t0:
            features = features.reshape(t0, batch, self.hidden_dim)

        if state is None:
            zeros = torch.zeros(self.num_layers, batch, self.hidden_dim,
                                device=features.device, dtype=features.dtype)
            state = (zeros, zeros)
        else:
            state = self._check_state(state, batch, features.device, features.dtype)

        outputs, new_state = self.lstm(features, state)
        logits = self.policy_head(outputs)
        baseline = self.baseline_head(outputs)

        if sequence and not keep_sequence:
            logits = logits.reshape(t0 * batch, self.config.num_actions)
            baseline = baseline.reshape(t0 * batch, 1)

        out = ModelOutput(
            policy_logits=logits,
            baseline=baseline,
            value=baseline,
            state=new_state if return_state else None,
            lstm_state=new_state,
            features=outputs,
            embeddings=encoded["screen"],
            sequence_mask=None,
        )
        return out

    def _check_state(self, state: Any, batch: int, device: Any, dtype: Any) -> Tuple[Any, Any]:
        if isinstance(state, (list, tuple)) and len(state) == 2:
            h, c = state
        else:  # single tensor -> split as (2, L, B, H)
            h = state[0]
            c = state[1]
        h = self._to_tensor(h, device=device, dtype=dtype)
        c = self._to_tensor(c, device=device, dtype=dtype)
        if h.dim() == 2:  # (L, H) -> (L, 1, H)
            h = h.unsqueeze(1)
            c = c.unsqueeze(1)
        if h.shape[1] != batch and h.shape[1] == 1:
            h = h.expand(-1, batch, -1).contiguous()
            c = c.expand(-1, batch, -1).contiguous()
        return (h.contiguous(), c.contiguous())

    # -- convenience accessors ---------------------------------------------------------

    def policy_logits(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.forward(obs, state=state, **kwargs).policy_logits

    def baseline(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.forward(obs, state=state, **kwargs).squeeze_value()

    def value(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.baseline(obs, state=state, **kwargs)

    def distribution(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return torch.distributions.Categorical(logits=self.policy_logits(obs, state=state,
                                                                        **kwargs))

    def log_prob(self, obs: Any, actions: Any, state: Any = None, **kwargs: Any) -> Any:
        dist = self.distribution(obs, state=state, **kwargs)
        return dist.log_prob(actions.long())

    def entropy(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.distribution(obs, state=state, **kwargs).entropy()

    def evaluate_actions(self, obs: Any, actions: Any, state: Any = None,
                         **kwargs: Any) -> Dict[str, Any]:
        """Return log-probs, entropy and baseline for a batch of observations."""
        out = self.forward(obs, state=state, **kwargs)
        dist = torch.distributions.Categorical(logits=out.policy_logits)
        return {
            "log_prob": dist.log_prob(actions.long()),
            "entropy": dist.entropy(),
            "baseline": out.baseline.squeeze(-1),
            "state": out.state,
        }

    @torch.no_grad() if _HAS_TORCH else (lambda f: f)
    def act(self, obs: Any, deterministic: bool = False, state: Any = None,
            return_state: bool = False, **kwargs: Any) -> Any:
        out = self.forward(obs, state=state, **kwargs)
        dist = torch.distributions.Categorical(logits=out.policy_logits)
        action = out.policy_logits.argmax(-1) if deterministic else dist.sample()
        if action.dim() > 0 and action.shape[0] == 1:
            action_out: Any = int(action.item())
        else:
            action_out = action
        if return_state:
            return action_out, out.state
        return action_out

    def reset_state(self) -> None:
        return None

    def init_state(self, batch_size: int = 1, device: Any = None, dtype: Any = None) -> Tuple[Any, Any]:
        device = device or next(self.parameters()).device
        dtype = dtype or next(self.parameters()).dtype
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device, dtype=dtype)
        return (zeros, zeros.clone())

    # -- phases (pre-training / fine-tuning) -------------------------------------------

    def set_phase(self, phase: str = "finetune") -> List[str]:
        """Configure which sub-modules are trainable.

        * ``"baseline"`` — freeze everything except the baseline head (used for the
          500M-step baseline-head pre-training described in §B.1).
        * ``"finetune"`` — train everything except the (frozen) encoders, matching
          "To improve the stability of the models we froze the encoders during the
          course of the training" (§B.1).
        * ``"full"``/``"scratch"`` — everything trainable.
        """
        phase = (phase or "full").lower()
        if phase in ("baseline", "critic", "value"):
            freeze_module(self, freeze=True)
            freeze_module(self.baseline_head, freeze=False)
        elif phase in ("finetune", "fine_tune", "online"):
            freeze_module(self, freeze=False)
            freeze_encoders(self)
        else:
            freeze_module(self, freeze=False)
        return self.trainable_parameter_names()

    def freeze_encoders(self) -> List[str]:
        return freeze_encoders(self)

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, p in self.named_parameters() if p.requires_grad]

    def actor_module(self) -> "NetHackActor":
        return self.actor

    def critic_module(self) -> Any:
        return self.baseline_head

    # -- persistence -------------------------------------------------------------------

    def save(self, path: str, step: Optional[int] = None, extra: Optional[Mapping[str, Any]] = None) -> str:
        return save_checkpoint(path, self, step=step, extra=extra)

    def load(self, path: str, strict: bool = False, map_location: str = "cpu") -> Dict[str, Any]:
        return load_pretrained(self, path, strict=strict, map_location=map_location)

    def describe(self) -> Dict[str, Any]:
        return summary(self)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NetHackModel(hidden_dim={self.hidden_dim}, num_actions={self.config.num_actions}, "
            f"screen_out={self.main_screen_encoder.out_dim}, "
            f"blstats={self.blstats_encoder.in_dim}, merged={self.merged_dim}, "
            f"params={count_parameters(self):,})"
        )


# --------------------------------------------------------------------------------------
# Stub model (CPU smoke tests without NLE-sized inputs)
# --------------------------------------------------------------------------------------


class MiniNetHackModel(nn.Module):
    """Tiny actor-critic with the same interface as :class:`NetHackModel`.

    Used for CPU smoke tests / `--stub` runs where the real NLE observation stack
    is unavailable. Accepts either dict observations or flat vectors.
    """

    def __init__(self, obs_dim: int = 64, num_actions: int = 8, hidden_dim: int = 64,
                 **_: Any) -> None:
        super().__init__()
        _require_torch()
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = 1
        self.backbone = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=False)
        self.policy_head = nn.Linear(self.hidden_dim, self.num_actions)
        self.baseline_head = nn.Linear(self.hidden_dim, 1)
        self.config = NetHackModelConfig(num_actions=self.num_actions, hidden_dim=self.hidden_dim)
        self.actor = NetHackActor(self)

    def _flatten(self, obs: Any) -> Any:
        if isinstance(obs, Mapping):
            parts = []
            for key in sorted(obs.keys()):
                value = obs[key]
                t = value if torch.is_tensor(value) else torch.as_tensor(value)
                parts.append(t.float().flatten(1) if t.dim() > 1 else t.float().unsqueeze(0))
            return torch.cat(parts, dim=-1)
        x = obs if torch.is_tensor(obs) else torch.as_tensor(obs)
        x = x.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def forward(self, obs: Any, state: Any = None, **kwargs: Any) -> ModelOutput:
        x = self._flatten(obs)
        if x.shape[-1] != self.obs_dim:
            if x.shape[-1] < self.obs_dim:
                pad = torch.zeros(x.shape[0], self.obs_dim - x.shape[-1], device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[..., : self.obs_dim]
        features = self.backbone(x).unsqueeze(0)   # (1, B, H)
        batch = features.shape[1]
        if state is None:
            zeros = torch.zeros(1, batch, self.hidden_dim, device=x.device)
            state = (zeros, zeros.clone())
        outputs, new_state = self.lstm(features, state)
        outputs = outputs.squeeze(0)
        return ModelOutput(
            policy_logits=self.policy_head(outputs),
            baseline=self.baseline_head(outputs),
            value=self.baseline_head(outputs),
            state=new_state,
            lstm_state=new_state,
            features=outputs,
        )

    def policy_logits(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.forward(obs, state=state, **kwargs).policy_logits

    def baseline(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.forward(obs, state=state, **kwargs).baseline.squeeze(-1)

    value = baseline

    def distribution(self, obs: Any, state: Any = None, **kwargs: Any) -> Any:
        return torch.distributions.Categorical(logits=self.policy_logits(obs, state=state))

    def log_prob(self, obs: Any, actions: Any, state: Any = None, **kwargs: Any) -> Any:
        return self.distribution(obs, state=state, **kwargs).log_prob(actions.long())

    def evaluate_actions(self, obs: Any, actions: Any, state: Any = None, **kwargs: Any) -> Dict[str, Any]:
        out = self.forward(obs, state=state, **kwargs)
        dist = torch.distributions.Categorical(logits=out.policy_logits)
        return {
            "log_prob": dist.log_prob(actions.long()),
            "entropy": dist.entropy(),
            "baseline": out.baseline.squeeze(-1),
            "state": out.state,
        }

    def act(self, obs: Any, deterministic: bool = False, state: Any = None,
            return_state: bool = False, **kwargs: Any) -> Any:
        out = self.forward(obs, state=state, **kwargs)
        action = out.policy_logits.argmax(-1) if deterministic else \
            torch.distributions.Categorical(logits=out.policy_logits).sample()
        action_out: Any = int(action.item()) if action.numel() == 1 else action
        return (action_out, out.state) if return_state else action_out

    def init_state(self, batch_size: int = 1, device: Any = None, dtype: Any = None) -> Tuple[Any, Any]:
        device = device or next(self.parameters()).device
        zeros = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (zeros, zeros.clone())

    def reset_state(self) -> None:
        return None

    def set_phase(self, phase: str = "full") -> List[str]:
        if phase in ("baseline", "critic", "value"):
            freeze_module(self, freeze=True)
            freeze_module(self.baseline_head, freeze=False)
        else:
            freeze_module(self, freeze=False)
        return self.trainable_parameter_names()

    def freeze_encoders(self) -> List[str]:
        return freeze_encoders(self)

    def trainable_parameter_names(self) -> List[str]:
        return [n for n, p in self.named_parameters() if p.requires_grad]

    def actor_module(self) -> Any:
        return self.actor

    def save(self, path: str, step: Optional[int] = None,
             extra: Optional[Mapping[str, Any]] = None) -> str:
        return save_checkpoint(path, self, step=step, extra=extra)

    def load(self, path: str, strict: bool = False, map_location: str = "cpu") -> Dict[str, Any]:
        return load_pretrained(self, path, strict=strict, map_location=map_location)

    def describe(self) -> Dict[str, Any]:
        return summary(self)


# --------------------------------------------------------------------------------------
# Freezing / parameter utilities
# --------------------------------------------------------------------------------------


def freeze_module(module: Any, freeze: bool = True) -> int:
    """Toggle ``requires_grad`` on every parameter of ``module``; returns the count."""
    if module is None:
        return 0
    count = 0
    for param in module.parameters():
        param.requires_grad = not freeze
        count += 1
    return count


def set_requires_grad(module: Any, requires_grad: bool = True) -> int:
    return freeze_module(module, freeze=not requires_grad)


ENCODER_NAMES: Tuple[str, ...] = (
    "main_screen_encoder", "blstats_encoder", "message_encoder", "encoder",
)


def freeze_encoders(model: Any, names: Optional[Sequence[str]] = None) -> List[str]:
    """Freeze the observation encoders (§B.1: "we froze the encoders during ... training").

    Returns the list of frozen module names.
    """
    candidates = tuple(names) if names else ENCODER_NAMES
    frozen: List[str] = []
    for name in candidates:
        module = getattr(model, name, None)
        if module is None:
            continue
        freeze_module(module, freeze=True)
        frozen.append(name)
    return frozen


def parameter_groups(model: Any, weight_decay: float = 1e-4) -> List[Dict[str, Any]]:
    """Adam parameter groups: no weight decay on biases/norms (standard practice)."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad],
                       "weight_decay": float(weight_decay)})
    return groups


def count_parameters(module: Any, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def parameter_names(module: Any, trainable_only: bool = False) -> List[str]:
    return [n for n, p in module.named_parameters() if (p.requires_grad or not trainable_only)]


def summary(model: Any) -> Dict[str, Any]:
    """Report parameter counts per sub-module (used in logs / checkpoints)."""
    info: Dict[str, Any] = {
        "params_total": count_parameters(model),
        "params_trainable": count_parameters(model, trainable_only=True),
    }
    for name in ("main_screen_encoder", "blstats_encoder", "message_encoder", "merge",
                 "lstm", "policy_head", "baseline_head"):
        module = getattr(model, name, None)
        if module is not None:
            info[f"params_{name}"] = count_parameters(module)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        info["hidden_dim"] = getattr(cfg, "hidden_dim", None)
        info["num_actions"] = getattr(cfg, "num_actions", None)
    return info


# --------------------------------------------------------------------------------------
# Builders / checkpoint loading
# --------------------------------------------------------------------------------------


def build_model(config: Any = None, checkpoint: Optional[str] = None,
                device: Optional[str] = None, stub: bool = False,
                **overrides: Any) -> Any:
    """Build a NetHack actor-critic, optionally loading the released π* checkpoint."""
    _require_torch()
    if stub:
        obs_dim = int(overrides.pop("obs_dim", _cfg_get(config, "model.stub_obs_dim", 64) or 64))
        num_actions = int(overrides.pop("num_actions",
                                        _cfg_get(config, "model.num_actions", 8) or 8))
        model: Any = MiniNetHackModel(obs_dim=obs_dim, num_actions=num_actions)
        device = device or _cfg_get(config, "compute.device", "cpu") or "cpu"
        model.to(torch.device(device))
        return model

    if isinstance(config, NetHackModelConfig):
        model_cfg = config.with_overrides(**overrides)
    else:
        model_cfg = NetHackModelConfig.from_config(config, **overrides)
    device = device or model_cfg.device or "cpu"
    model = NetHackModel(model_cfg)
    model.to(torch.device(device))
    if checkpoint:
        load_lstm_checkpoint(model, checkpoint, map_location=str(device))
    return model


def build_nethack_model(config: Any = None, checkpoint: Optional[str] = None, **kwargs: Any) -> Any:
    """Alias used by the APPO runner/evaluation code."""
    return build_model(config, checkpoint=checkpoint, **kwargs)


def build_actor(model: Any) -> Any:
    """Return the actor-only view used by the retention losses (critic coef is 0)."""
    actor = getattr(model, "actor", None)
    if actor is None:
        actor = getattr(model, "actor_module", lambda: model)()
    return actor


# -- checkpoint key remapping ----------------------------------------------------------

_PREFIX_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("actor_encoder.main_screen_encoder.", "main_screen_encoder."),
    ("actor_encoder.", ""),
    ("actor_head.", "policy_head."),
    ("actor_core.", "lstm."),
    ("critic_encoder.", ""),
    ("critic_head.", "baseline_head."),
    ("critic_core.", "lstm."),
    ("encoder.", ""),
    ("core.", "lstm."),
    ("model.", ""),
    ("module.", ""),
    ("policy.", ""),
    ("net.", ""),
)

STATE_DICT_CONTAINER_KEYS: Tuple[str, ...] = (
    "model_state_dict", "state_dict", "model", "policy", "actor_critic",
    "model_state", "agent_state_dict", "weights",
)


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    """Pull the parameter mapping out of an arbitrary checkpoint payload."""
    if isinstance(payload, Mapping):
        for key in STATE_DICT_CONTAINER_KEYS:
            inner = payload.get(key)
            if isinstance(inner, Mapping) and inner and all(
                isinstance(v, (dict,)) or hasattr(v, "shape") or hasattr(v, "numel")
                for v in list(inner.values())[:3]
            ):
                return dict(inner)
        values = list(payload.values())
        if values and all(hasattr(v, "shape") or hasattr(v, "numel") for v in values[:3]):
            return dict(payload)
        # Nested containers (e.g. {"model": {"state_dict": {...}}})
        for key in STATE_DICT_CONTAINER_KEYS:
            inner = payload.get(key)
            if isinstance(inner, Mapping):
                return _extract_state_dict(inner)
    return {}


def remap_state_dict(state: Mapping[str, Any], model: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Remap external checkpoint keys onto this model's parameter names.

    Handles Sample-Factory style naming (``actor_encoder.*``, ``actor_head.*``,
    ``critic_head.*``) as well as plain layouts. Returns ``(remapped, unmatched)``.
    """
    target = set(model.state_dict().keys())
    remapped: Dict[str, Any] = {}
    unmatched: List[str] = []
    for key, value in state.items():
        candidates = [key]
        for prefix, replacement in _PREFIX_ALIASES:
            if key.startswith(prefix):
                candidates.append(replacement + key[len(prefix):])
        if key.startswith("lstm."):
            candidates.append("lstm." + key[len("lstm."):])
        hit = next((c for c in candidates if c in target), None)
        if hit is not None:
            remapped.setdefault(hit, value)
        else:
            # Try ignoring an unexpected extra nesting level.
            parts = key.split(".")
            if len(parts) > 2:
                joined = ".".join(parts[1:])
                if joined in target:
                    remapped.setdefault(joined, value)
                    continue
            unmatched.append(key)
    return remapped, unmatched


def load_pretrained(model: Any, path_or_state: Any, strict: bool = False,
                    map_location: str = "cpu", verbose: bool = False) -> Dict[str, Any]:
    """Load pre-trained weights (path or state dict) into ``model``.

    Works with the released Tuyls et al. (2023) 30M LSTM checkpoint as well as with
    checkpoints written by :func:`save_checkpoint`. Because the released checkpoint
    predates the baseline head (BC has no critic, §B.1), missing keys default to the
    freshly initialised parameters and are reported in the result.
    """
    _require_torch()
    if isinstance(path_or_state, (str, os.PathLike)):
        payload = torch.load(str(path_or_state), map_location=map_location)
    else:
        payload = path_or_state
    state = _extract_state_dict(payload)
    if not state and isinstance(payload, Mapping):
        state = dict(payload)
    remapped, unmatched = remap_state_dict(state, model)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    result = {
        "loaded": len(remapped),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "unmatched_source_keys": unmatched,
        "path": str(path_or_state) if isinstance(path_or_state, (str, os.PathLike)) else None,
    }
    if strict and (missing or unmatched):
        raise RuntimeError(
            f"Strict load failed: missing={len(missing)} unmatched={len(unmatched)}"
        )
    if verbose:  # pragma: no cover - logging convenience
        print(f"[nethack.model] loaded {result['loaded']} tensors "
              f"(missing={len(result['missing'])}, unmatched={len(unmatched)})")
    return result


def load_lstm_checkpoint(model: Any, path: str, map_location: str = "cpu",
                         verbose: bool = False) -> Dict[str, Any]:
    """Load the released 30M-parameter LSTM π* checkpoint into ``model`` (§B.1)."""
    return load_pretrained(model, path, strict=False, map_location=map_location,
                           verbose=verbose)


# --------------------------------------------------------------------------------------
# Persistence helpers (compatible with src.common.checkpointing)
# --------------------------------------------------------------------------------------


def save_checkpoint(path: str, model: Any, step: Optional[int] = None,
                    optimizer: Any = None, extra: Optional[Mapping[str, Any]] = None) -> str:
    """Save model (and optionally optimiser) state, merging in ``extra`` payload."""
    _require_torch()
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    cfg = getattr(model, "config", None)
    if cfg is not None:
        try:
            payload["config"] = cfg.to_dict()
        except AttributeError:  # pragma: no cover - defensive
            payload["config"] = dict(cfg)
    if extra:
        payload.update(dict(extra))
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    if step is not None and not path.endswith(f"step_{int(step)}.pt"):
        base, ext = os.path.splitext(path)
        path = f"{base}_step_{int(step)}{ext or '.pt'}"
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, model: Any, optimizer: Any = None,
                    map_location: str = "cpu") -> Dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint` into ``model``."""
    _require_torch()
    payload = torch.load(path, map_location=map_location)
    result = load_pretrained(model, payload, strict=False, map_location=map_location)
    if optimizer is not None and isinstance(payload, Mapping) and \
            payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        result["optimizer_loaded"] = True
    return result


# --------------------------------------------------------------------------------------
# CLI (smoke test / parameter report)
# --------------------------------------------------------------------------------------


def _build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="NetHack model (Appendix B.1) smoke test")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="config overrides key.subkey=value")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--stub", action="store_true", help="build the mini model instead")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=0,
                        help="if > 0, run a sequence forward of this length")
    parser.add_argument("--phase", type=str, default=None,
                        help="baseline | finetune | full")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg: Any = None
    if args.config:
        try:
            from src.common.config import load_config, apply_overrides  # type: ignore
        except Exception:  # pragma: no cover - fallback for direct execution
            from ..common.config import load_config, apply_overrides  # type: ignore
        cfg = load_config(args.config)
        if args.overrides:
            cfg = apply_overrides(cfg, args.overrides)

    model = build_model(cfg, checkpoint=args.checkpoint, device=args.device,
                        stub=args.stub)
    print(model.describe())
    if args.phase:
        print("frozen encoders / trainable:", model.set_phase(args.phase))

    if _HAS_TORCH:
        t = int(args.sequence_length) if args.sequence_length else 0
        if args.stub:
            obs: Any = torch.randn(args.batch_size, 64)
        else:
            B = int(args.batch_size)
            obs = {
                "tty_chars": torch.randint(0, 128, (B, 2, 24, 80) if t else (B, 24, 80)),
                "tty_colors": torch.randint(0, 16, (B, 2, 24, 80) if t else (B, 24, 80)),
                "blstats": torch.randn(B, 2, 25) if t else torch.randn(B, 25),
                "message": torch.zeros(B, 2, 256, 128) if t else torch.zeros(B, 256, 128),
            }
        out = model(obs, sequence=bool(t), time_major=True if t else False)
        print("policy_logits:", tuple(out.policy_logits.shape),
              "baseline:", tuple(out.baseline.shape))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
