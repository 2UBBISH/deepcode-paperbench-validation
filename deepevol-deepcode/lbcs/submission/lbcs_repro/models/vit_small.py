"""ViT-small target network for the Section 6 cross-architecture evaluation.

Paper reference
---------------
Section 6 ("More Justifications and Analyses" -> "Cross network architecture
evaluation")::

    Here we demonstrate that the proposed method is not limited to specific
    network architectures. We employ SVHN and use ViTsmall (Dosovitskiy et al.,
    2021) and WideResNet (abbreviated as W-NET) (Zagoruyko & Komodakis, 2016)
    for training on the constructed coreset. The other experimental settings are
    not changed. Results are provided in Table 6. As can be seen, with ViT, our
    method is still superior to the competitors with respect to test accuracy and
    coreset sizes (the exact coreset sizes of our method can be checked in
    Table 2).

So ViT-small is used as the *post-selection target model* on SVHN (32x32, 10
classes) while the coreset selection itself is unchanged (Section 5.2 settings:
``k in {1000, 2000, 3000, 4000}``, ``eps=0.2``, ``T=500``, inner loop Adam with
learning rate 0.001, and 100 epochs for training F-MNIST / SVHN target models).

Paper-stated vs SUGGESTED
-------------------------
The paper names the architecture ("ViT-small") and fixes the SVHN training
protocol through the "other experimental settings are not changed" statement of
Section 5.2 (Adam, lr = 0.001, 100 epochs).  Appendix D.2 points to Table 7 for
the detailed network structures, but Table 7 is not recoverable from the paper
text, so every architectural width (embedding dimension, depth, heads, patch
size, ...) is exposed as an explicitly **SUGGESTED** default that can be
overridden from YAML.  No architecture field below is claimed to be
paper-stated.

This module deliberately mirrors the conventions of ``models/convnet.py``,
``models/lenet.py``, ``models/svhn_cnn.py``, ``models/cifar_cnn.py`` and
``models/resnet18.py``: a dataclass config, a ``torch.nn.Module`` returning raw
logits, a *factory* that produces a fresh network (required because every mask
evaluation in Algorithm 1 trains a new ``theta(m)``), an evaluation helper and a
``_selftest``.  PyTorch is a soft dependency: mask-only unit tests still import
this module successfully.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - torch is a soft dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

# Optional timm integration (used to cross-check / swap in the reference
# "vit_small" implementation).  The self-contained implementation below is the
# default because it works without timm installed.
try:  # pragma: no cover
    import timm  # type: ignore

    _TIMM_AVAILABLE = True
except Exception:  # pragma: no cover
    timm = None  # type: ignore
    _TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants: SUGGESTED defaults (NOT paper-stated)
# ---------------------------------------------------------------------------

#: SUGGESTED ViT-small geometry for 32x32 inputs (CIFAR-style patch size).
SUGGESTED_PATCH_SIZE = 4
SUGGESTED_EMBED_DIM = 384
SUGGESTED_DEPTH = 12
SUGGESTED_NUM_HEADS = 6
SUGGESTED_MLP_RATIO = 4.0
SUGGESTED_DROPOUT = 0.0
SUGGESTED_ATTENTION_DROPOUT = 0.0
SUGGESTED_DROP_PATH = 0.0
SUGGESTED_QKV_BIAS = True
SUGGESTED_LAYER_NORM_EPS = 1e-6
SUGGESTED_WEIGHT_DECAY = 5e-4
SUGGESTED_BATCH_SIZE = 128

#: Paper-stated Section 5.2 target-training protocol for SVHN
#: ("for F-MNIST and SVHN, an Adam optimizer is used with a learning rate of
#: 0.001 and 100 epochs").
SECTION52_TARGET_OPTIMIZER = "adam"
SECTION52_TARGET_LR = 0.001
SECTION52_TARGET_EPOCHS = 100

#: timm names that correspond to ViT-small (used when timm is available).
TIMM_VIT_SMALL_NAMES = (
    "vit_small_patch4_32",
    "vit_small_patch16_224",
    "vit_small_patch32_224",
)

SVHN_NUM_CLASSES = 10
SVHN_IN_CHANNELS = 3
SVHN_INPUT_SHAPE = (3, 32, 32)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ViTSmallConfig:
    """Static description of the ViT-small target network.

    All fields are SUGGESTED defaults (Table 7 is not extractable from the
    paper) and are overridable from YAML via :meth:`with_overrides` /
    :meth:`from_dict`.
    """

    # data / task
    in_channels: int = SVHN_IN_CHANNELS
    num_classes: int = SVHN_NUM_CLASSES
    input_shape: Tuple[int, int, int] = field(default_factory=lambda: tuple(SVHN_INPUT_SHAPE))

    # patch embedding
    patch_size: int = SUGGESTED_PATCH_SIZE
    embed_dim: int = SUGGESTED_EMBED_DIM

    # transformer body
    depth: int = SUGGESTED_DEPTH
    num_heads: int = SUGGESTED_NUM_HEADS
    mlp_ratio: float = SUGGESTED_MLP_RATIO
    qkv_bias: bool = SUGGESTED_QKV_BIAS
    norm_eps: float = SUGGESTED_LAYER_NORM_EPS
    class_token: bool = True
    global_pool: bool = True  # GAP over tokens (in addition to / instead of cls)

    # regularization
    dropout: float = SUGGESTED_DROPOUT
    attention_dropout: float = SUGGESTED_ATTENTION_DROPOUT
    drop_path: float = SUGGESTED_DROP_PATH

    # post-selection target training (Section 5.2 for SVHN; SUGGESTED defaults)
    optimizer: str = SECTION52_TARGET_OPTIMIZER
    lr: float = SECTION52_TARGET_LR
    epochs: int = SECTION52_TARGET_EPOCHS
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    batch_size: int = SUGGESTED_BATCH_SIZE
    scheduler: Optional[str] = None
    augment: bool = False
    num_workers: int = 0

    # ------------------------------------------------------------------ #
    # presets / helpers
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        self.input_shape = tuple(int(v) for v in self.input_shape)  # type: ignore[assignment]
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads "
                f"({self.num_heads})"
            )

    @classmethod
    def for_svhn(cls, num_classes: int = SVHN_NUM_CLASSES, **overrides: Any) -> "ViTSmallConfig":
        """ViT-small preset for SVHN (Section 6 cross-architecture evaluation)."""
        base = dict(
            in_channels=SVHN_IN_CHANNELS,
            num_classes=num_classes,
            input_shape=SVHN_INPUT_SHAPE,
        )
        base.update(overrides)
        return cls(**base)

    @classmethod
    def for_inner(cls, **overrides: Any) -> "ViTSmallConfig":
        """Preset matching the inner-loop (proxy) configuration of Section 5.2.

        The paper uses simple CNNs for the inner loop; this preset only exists so
        that a ViT can be substituted for ablation, with Adam lr = 0.001.
        """
        base = dict(
            optimizer="adam",
            lr=0.001,
            epochs=100,
            scheduler=None,
            augment=False,
        )
        base.update(overrides)
        return cls.for_svhn(**base)

    @classmethod
    def for_target(cls, **overrides: Any) -> "ViTSmallConfig":
        """Preset for post-selection target training on SVHN (Section 5.2)."""
        base = dict(
            optimizer=SECTION52_TARGET_OPTIMIZER,
            lr=SECTION52_TARGET_LR,
            epochs=SECTION52_TARGET_EPOCHS,
            scheduler=None,
            augment=False,
        )
        base.update(overrides)
        return cls.for_svhn(**base)

    def with_overrides(self, **overrides: Any) -> "ViTSmallConfig":
        """Return a copy with ``overrides`` applied (ignores ``None`` values)."""
        overrides = {k: v for k, v in overrides.items() if v is not None}
        overrides.pop("input_shape", None) if not overrides else None
        if "input_shape" in overrides:
            overrides["input_shape"] = tuple(overrides["input_shape"])
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["input_shape"] = tuple(data["input_shape"])
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ViTSmallConfig":
        """Build a config from a YAML/dict block, ignoring unknown keys."""
        if not data:
            return cls()
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in dict(data).items() if k in allowed}
        return cls(**kwargs)

    # ---- derived quantities ------------------------------------------- #
    def num_patches(self) -> int:
        """Number of patch tokens for ``input_shape`` (no class token)."""
        _, h, w = self.input_shape
        ph = h // self.patch_size
        pw = w // self.patch_size
        return int(ph * pw)

    def seq_len(self) -> int:
        """Token sequence length actually consumed by the transformer."""
        return self.num_patches() + (1 if self.class_token else 0)

    def head_dim(self) -> int:
        return int(self.embed_dim // self.num_heads)

    def hidden_dim(self) -> int:
        return int(round(self.embed_dim * self.mlp_ratio))

    def summary(self) -> Dict[str, Any]:
        return {
            "name": "ViT-small",
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim(),
            "mlp_ratio": self.mlp_ratio,
            "hidden_dim": self.hidden_dim(),
            "num_patches": self.num_patches(),
            "seq_len": self.seq_len(),
            "num_classes": self.num_classes,
            "input_shape": tuple(self.input_shape),
        }


# ---------------------------------------------------------------------------
# Building blocks (self-contained, no timm required)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:  # pragma: no cover - exercised through _selftest

    class PatchEmbed(nn.Module):
        """Split an image into patches and linearly embed them."""

        def __init__(self, config: ViTSmallConfig) -> None:
            super().__init__()
            self.config = config
            self.patch_size = int(config.patch_size)
            self.num_patches = config.num_patches()
            self.proj = nn.Conv2d(
                config.in_channels,
                config.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = self.proj(x)  # (N, embed_dim, H/P, W/P)
            x = x.flatten(2).transpose(1, 2)  # (N, num_patches, embed_dim)
            return x


    class MultiheadSelfAttention(nn.Module):
        """Standard multi-head self attention with an optional qkv bias."""

        def __init__(self, config: ViTSmallConfig) -> None:
            super().__init__()
            self.num_heads = int(config.num_heads)
            self.head_dim = config.head_dim()
            self.scale = self.head_dim ** -0.5
            self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim, bias=config.qkv_bias)
            self.proj = nn.Linear(config.embed_dim, config.embed_dim)
            self.attn_drop = nn.Dropout(config.attention_dropout)
            self.proj_drop = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            n, tokens, dim = x.shape
            qkv = self.qkv(x).reshape(n, tokens, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, heads, tokens, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v  # (N, heads, tokens, head_dim)
            out = out.transpose(1, 2).reshape(n, tokens, dim)
            out = self.proj(out)
            return self.proj_drop(out)


    class MLP(nn.Module):
        """Transformer MLP block (Linear -> GELU -> Dropout -> Linear -> Dropout)."""

        def __init__(self, config: ViTSmallConfig) -> None:
            super().__init__()
            hidden = config.hidden_dim()
            self.fc1 = nn.Linear(config.embed_dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, config.embed_dim)
            self.drop = nn.Dropout(config.dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.drop(self.act(self.fc1(x)))
            x = self.drop(self.fc2(x))
            return x


    def _drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
        """Per-sample stochastic depth (identity when ``drop_prob == 0``)."""
        if drop_prob <= 0.0 or not training:
            return x
        keep = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


    class Block(nn.Module):
        """Pre-norm transformer encoder block."""

        def __init__(self, config: ViTSmallConfig, drop_path: float = 0.0) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
            self.attn = MultiheadSelfAttention(config)
            self.norm2 = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
            self.mlp = MLP(config)
            self.drop_path = float(drop_path)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + _drop_path(self.attn(self.norm1(x)), self.drop_path, self.training)
            x = x + _drop_path(self.mlp(self.norm2(x)), self.drop_path, self.training)
            return x


    class ViTSmall(nn.Module):
        """Self-contained ViT-small classifier returning raw logits.

        Used in Section 6 as the *target* network trained on the constructed
        SVHN coreset.
        """

        def __init__(
            self,
            config: Optional[ViTSmallConfig] = None,
            **kwargs: Any,
        ) -> None:
            super().__init__()
            if config is None:
                config = ViTSmallConfig(**kwargs) if kwargs else ViTSmallConfig()
            elif kwargs:
                config = config.with_overrides(**kwargs)
            self.config = config

            self.patch_embed = PatchEmbed(config)
            seq_len = config.seq_len()
            self.cls_token = (
                nn.Parameter(torch.zeros(1, 1, config.embed_dim)) if config.class_token else None
            )
            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, config.embed_dim))
            self.pos_drop = nn.Dropout(config.dropout)

            # linearly increasing stochastic depth (SUGGESTED schedule)
            dpr = (
                [config.drop_path * i / max(config.depth - 1, 1) for i in range(config.depth)]
                if config.drop_path > 0
                else [0.0] * config.depth
            )
            self.blocks = nn.ModuleList([Block(config, dpr[i]) for i in range(config.depth)])
            self.norm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)
            self.head = nn.Linear(config.embed_dim, config.num_classes)
            self.num_features = int(config.embed_dim)

            self._init_weights()

        # -- helpers ---------------------------------------------------- #
        def _init_weights(self) -> None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if self.cls_token is not None:
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            x = self.patch_embed(x)
            n = x.shape[0]
            if self.cls_token is not None:
                cls = self.cls_token.expand(n, -1, -1)
                x = torch.cat([cls, x], dim=1)
            x = x + self.pos_embed[:, : x.shape[1]]
            x = self.pos_drop(x)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return x

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return raw logits of shape ``(N, num_classes)``.

            Plug-compatible with ``nn.CrossEntropyLoss`` so the network can be
            used for both ``f1(m)`` / ``L(m, theta)`` and target training.
            """
            tokens = self.forward_features(x)
            if self.config.global_pool:
                pooled = tokens.mean(dim=1)
            elif self.cls_token is not None:
                pooled = tokens[:, 0]
            else:
                pooled = tokens.mean(dim=1)
            return self.head(pooled)

        def features(self, x: torch.Tensor) -> torch.Tensor:
            """Pooled penultimate representation (for distance-based baselines)."""
            tokens = self.forward_features(x)
            if self.config.global_pool or self.cls_token is None:
                return tokens.mean(dim=1)
            return tokens[:, 0]

        def num_parameters(self, trainable_only: bool = True) -> int:
            if trainable_only:
                return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
            return int(sum(p.numel() for p in self.parameters()))

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return ", ".join(f"{k}={v}" for k, v in self.config.summary().items())

else:  # pragma: no cover - no torch available

    class ViTSmall(object):  # type: ignore[no-redef]
        """Placeholder raising a clear error when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required to instantiate ViTSmall. "
                "Install torch to use the Section 6 cross-architecture model."
            )

    class PatchEmbed(object):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for PatchEmbed.")

    class MultiheadSelfAttention(object):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for MultiheadSelfAttention.")

    class MLP(object):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for MLP.")

    class Block(object):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for Block.")


# ---------------------------------------------------------------------------
# Constructors / factories
# ---------------------------------------------------------------------------


def build_vit_small(
    num_classes: int = SVHN_NUM_CLASSES,
    in_channels: int = SVHN_IN_CHANNELS,
    input_shape: Optional[Sequence[int]] = None,
    config: Optional[ViTSmallConfig] = None,
    **kwargs: Any,
) -> "ViTSmall":
    """Instantiate a (fresh) ViT-small network.

    Parameters mirror :func:`models.convnet.build_convnet` /
    :func:`models.resnet18.build_resnet18` so experiment drivers can treat the
    model zoo uniformly.
    """
    base = config or ViTSmallConfig.for_svhn(num_classes=num_classes)
    overrides: Dict[str, Any] = {}
    if in_channels is not None:
        overrides["in_channels"] = in_channels
    if input_shape is not None:
        overrides["input_shape"] = tuple(input_shape)
    if num_classes is not None:
        overrides["num_classes"] = num_classes
    overrides.update({k: v for k, v in kwargs.items() if k not in ("config",)})
    cfg = base.with_overrides(**overrides) if overrides else base
    return ViTSmall(cfg)


#: Short alias, matching the naming convention of the other model modules.
vit_small = build_vit_small


def vit_small_factory(
    config: Optional[ViTSmallConfig] = None,
    **overrides: Any,
) -> Callable[..., "ViTSmall"]:
    """Return a builder producing **fresh** ViT-small instances.

    ``lbcs.bilevel.LBCS`` requires a new ``theta(m)`` for every mask evaluation;
    the returned callable therefore constructs a brand-new network on each call
    and carries the merged ``.config`` as an attribute for introspection.
    """
    base_config = config or ViTSmallConfig.for_svhn()
    if overrides:
        base_config = base_config.with_overrides(**overrides)

    def _build(**kwargs: Any) -> "ViTSmall":
        cfg = base_config
        if kwargs:
            cfg = cfg.with_overrides(**{k: v for k, v in kwargs.items() if v is not None})
        return ViTSmall(cfg)

    _build.config = base_config  # type: ignore[attr-defined]
    return _build


def svhn_vit_small(num_classes: int = SVHN_NUM_CLASSES, **kwargs: Any) -> "ViTSmall":
    """Section 6 preset: ViT-small target network for SVHN."""
    return build_vit_small(num_classes=num_classes, config=ViTSmallConfig.for_target(), **kwargs)


def build_timm_vit_small(
    pretrained: bool = False,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Optional helper returning a ``timm`` ViT-small (when timm is installed).

    Kept as a cross-check against the reference implementation.  The paper does
    not state which ViT code base was used, so the self-contained
    :class:`ViTSmall` above remains the default.
    """
    if not _TIMM_AVAILABLE:
        raise ImportError(
            "timm is not installed. Install `timm` to use build_timm_vit_small(), "
            "or use the self-contained ViTSmall implementation."
        )
    names: List[str] = [model_name] if model_name else list(TIMM_VIT_SMALL_NAMES)
    last_err: Optional[Exception] = None
    for name in names:
        try:
            return timm.create_model(name, pretrained=pretrained, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on timm version
            last_err = exc
    raise RuntimeError(f"No timm ViT-small variant could be created: {last_err}")


# ---------------------------------------------------------------------------
# Target-model training / evaluation (Section 5.2 protocol for SVHN)
# ---------------------------------------------------------------------------


def target_train_config(**kwargs: Any) -> Dict[str, Any]:
    """Return the Section 5.2 target-training configuration for SVHN.

    Paper-stated: Adam, learning rate 0.001, 100 epochs ("for F-MNIST and SVHN,
    an Adam optimizer is used with a learning rate of 0.001 and 100 epochs").
    Weight decay and batch size are SUGGESTED.
    """
    cfg: Dict[str, Any] = {
        "optimizer": SECTION52_TARGET_OPTIMIZER,
        "lr": SECTION52_TARGET_LR,
        "epochs": SECTION52_TARGET_EPOCHS,
        "weight_decay": SUGGESTED_WEIGHT_DECAY,
        "batch_size": SUGGESTED_BATCH_SIZE,
        "scheduler": None,
    }
    cfg.update({k: v for k, v in kwargs.items() if v is not None})
    return cfg


def build_target_optimizer(model: "nn.Module", config: Optional[Dict[str, Any]] = None, **kwargs: Any):
    """Build the Adam/SGD/AdamW optimizer used for target training."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to build an optimizer.")
    merged = target_train_config(**(config or {}))
    merged.update({k: v for k, v in kwargs.items() if v is not None})
    name = str(merged.get("optimizer", "adam")).lower()
    lr = float(merged.get("lr", SECTION52_TARGET_LR))
    wd = float(merged.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name in ("sgd", "momentum"):
        return torch.optim.SGD(
            params, lr=lr, momentum=float(merged.get("momentum", 0.9)), weight_decay=wd
        )
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return torch.optim.Adam(params, lr=lr, weight_decay=wd)


def build_target_scheduler(optimizer, config: Optional[Dict[str, Any]] = None, steps_per_epoch: int = 1):
    """Optional LR scheduler for target training (``None`` by default for SVHN)."""
    merged = target_train_config(**(config or {}))
    name = merged.get("scheduler")
    if not name:
        return None
    epochs = int(merged.get("epochs", SECTION52_TARGET_EPOCHS))
    if str(name).lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * max(1, steps_per_epoch))
        )
    if str(name).lower() == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    return None


def evaluate(model: "nn.Module", loader, device: Optional[Any] = None) -> float:
    """Top-1 test accuracy in percent over ``loader``."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for evaluation.")
    device = device or next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive
                inputs, targets = batch, None
            if targets is None:
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


#: Alias matching the convention used by ``models/resnet18.py``.
accuracy = evaluate


def train_target_model(
    model: "nn.Module",
    train_loader,
    test_loader=None,
    epochs: int = SECTION52_TARGET_EPOCHS,
    lr: float = SECTION52_TARGET_LR,
    optimizer: str = SECTION52_TARGET_OPTIMIZER,
    weight_decay: float = SUGGESTED_WEIGHT_DECAY,
    scheduler: Optional[str] = None,
    device: Optional[Any] = None,
    verbose: bool = False,
    log_every: int = 0,
    **kwargs: Any,
) -> Tuple["nn.Module", List[float]]:
    """Train a target model on the constructed coreset (Section 5.2, SVHN).

    Paper-stated settings: Adam with learning rate 0.001 for 100 epochs.  The
    same routine is used for both ViT-small and WideResNet in Section 6 ("the
    other experimental settings are not changed").
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for target training.")
    device = device or next(model.parameters()).device
    model.to(device)
    cfg = target_train_config(
        epochs=epochs,
        lr=lr,
        optimizer=optimizer,
        weight_decay=weight_decay,
        scheduler=scheduler,
        **kwargs,
    )
    opt = build_target_optimizer(model, cfg)
    sched = build_target_scheduler(opt, cfg, steps_per_epoch=max(1, len(train_loader)))
    criterion = nn.CrossEntropyLoss()
    history: List[float] = []
    for epoch in range(int(cfg["epochs"])):
        model.train()
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive
                continue
            inputs = inputs.to(device)
            targets = targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
        if test_loader is not None:
            acc = evaluate(model, test_loader, device=device)
            history.append(acc)
            if verbose and (log_every <= 0 or (epoch + 1) % log_every == 0):
                logger.info("[ViT-small] epoch %d/%d test acc %.2f", epoch + 1, cfg["epochs"], acc)
    return model, history


def train_svhn_vit_small(
    train_loader,
    test_loader=None,
    num_classes: int = SVHN_NUM_CLASSES,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Tuple["nn.Module", List[float]]:
    """Build + train a ViT-small on a constructed SVHN coreset (Section 6)."""
    model = svhn_vit_small(num_classes=num_classes)
    return train_target_model(model, train_loader, test_loader=test_loader, device=device, **kwargs)


# ---------------------------------------------------------------------------
# Registry / exports
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "ViTSmall": vit_small_factory,
    "ViT-small": vit_small_factory,
    "vit_small": vit_small_factory,
    "ViT": vit_small_factory,
    "SVHNVITS": vit_small_factory,
    "ViTSVHN": vit_small_factory,
}

__all__ = [
    "ViTSmallConfig",
    "ViTSmall",
    "PatchEmbed",
    "MultiheadSelfAttention",
    "MLP",
    "Block",
    "build_vit_small",
    "vit_small",
    "vit_small_factory",
    "svhn_vit_small",
    "build_timm_vit_small",
    "target_train_config",
    "build_target_optimizer",
    "build_target_scheduler",
    "evaluate",
    "accuracy",
    "train_target_model",
    "train_svhn_vit_small",
    "MODEL_REGISTRY",
    "TIMM_VIT_SMALL_NAMES",
    "SECTION52_TARGET_OPTIMIZER",
    "SECTION52_TARGET_LR",
    "SECTION52_TARGET_EPOCHS",
    "SUGGESTED_PATCH_SIZE",
    "SUGGESTED_EMBED_DIM",
    "SUGGESTED_DEPTH",
    "SUGGESTED_NUM_HEADS",
    "SUGGESTED_MLP_RATIO",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline structural checks (``python -m lbcs_repro.models.vit_small``)."""
    report: Dict[str, Any] = {}

    # ---- config algebra (torch-free) ---------------------------------- #
    cfg = ViTSmallConfig.for_svhn()
    report["config"] = cfg.summary()
    assert cfg.embed_dim % cfg.num_heads == 0
    assert cfg.num_patches() == (32 // cfg.patch_size) ** 2
    assert cfg.seq_len() == cfg.num_patches() + 1
    rt = ViTSmallConfig.from_dict(cfg.to_dict())
    assert rt.to_dict() == cfg.to_dict(), "config round-trip failed"
    ov = cfg.with_overrides(embed_dim=192, num_heads=6, nested=False if False else None)
    assert ov.embed_dim == 192 and ov.num_heads == 6
    assert "nested" not in ov.to_dict()
    report["config_roundtrip"] = True
    report["overrides_ok"] = True

    if not _TORCH_AVAILABLE:
        report["torch"] = False
        if verbose:
            print("[vit_small selftest] torch unavailable - config checks only")
            print(report)
        return report

    report["torch"] = True

    # ---- small model for fast checks ---------------------------------- #
    small = ViTSmallConfig.for_svhn(
        patch_size=8,
        embed_dim=48,
        depth=2,
        num_heads=3,
        mlp_ratio=2.0,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path=0.1,
    )
    model = ViTSmall(small)
    model.train()
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    report["logits_shape"] = tuple(logits.shape)
    assert logits.shape == (4, 10), logits.shape

    feats = model.features(x)
    report["features_shape"] = tuple(feats.shape)
    assert feats.shape == (4, small.embed_dim), feats.shape

    # patch count for patch_size=8 -> 4x4 = 16 tokens + cls
    assert model.patch_embed.num_patches == 16
    assert model.pos_embed.shape[1] == 17, model.pos_embed.shape
    assert model.num_parameters() > 0

    loss = nn.CrossEntropyLoss()(logits, torch.arange(4) % 10)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    report["params_without_grad"] = missing
    assert not missing, f"some parameters received no gradient: {missing}"

    # ---- numpy input tolerance ---------------------------------------- #
    np_out = model(torch.as_tensor(np.zeros((2, 3, 32, 32), dtype=np.float32)))
    assert np_out.shape == (2, 10)

    # ---- eval mode stability (drop path off) -------------------------- #
    model.eval()
    with torch.no_grad():
        a = model(x)
        b = model(x)
    report["eval_deterministic"] = bool(torch.allclose(a, b))
    assert torch.allclose(a, b)

    # ---- factory produces fresh instances ----------------------------- #
    factory = vit_small_factory(patch_size=8, embed_dim=24, depth=1, num_heads=3)
    m1, m2 = factory(), factory()
    assert m1 is not m2
    assert factory.config.embed_dim == 24
    report["factory_fresh"] = True
    report["registry_keys"] = sorted(MODEL_REGISTRY)
    assert "ViTSmall" in MODEL_REGISTRY

    # ---- build helpers ------------------------------------------------- #
    mb = build_vit_small(config=small)
    assert mb.config.depth == small.depth
    report["build_ok"] = True

    # ---- target training config (Section 5.2 protocol) ----------------- #
    tcfg = target_train_config()
    report["target_train_config"] = tcfg
    assert tcfg["optimizer"] == "adam" and tcfg["lr"] == 0.001 and tcfg["epochs"] == 100

    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,)))
    loader = DataLoader(ds, batch_size=8)
    trained, hist = train_target_model(
        ViTSmall(small), loader, test_loader=loader, epochs=1, verbose=False
    )
    report["train_one_epoch_acc"] = hist[-1] if hist else None
    assert len(hist) == 1
    report["train_ok"] = True

    if verbose:
        print("[vit_small selftest] OK")
        for key, value in report.items():
            print(f"  {key}: {value}")
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest(verbose=True)
