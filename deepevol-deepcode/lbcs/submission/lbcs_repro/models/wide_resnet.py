"""WideResNet (W-NET) target network for the Section 6 cross-architecture study.

Paper reference
---------------
Section 6 ("Cross network architecture evaluation"):

    "We employ SVHN and use ViTsmall (Dosovitskiy et al., 2021) and WideResNet
     (abbreviated as W-NET) (Zagoruyko & Komodakis, 2016) for training on the
     constructed coreset. The other experimental settings are not changed.
     Results are provided in Table 6."

Table 6 reports mean/std test accuracy (%) on SVHN for
``k in {1000, 2000, 3000, 4000}`` with two networks. Section 6 says the other
settings are unchanged, so the post-selection protocol is Section 5.2's: for
SVHN "an Adam optimizer is used with a learning rate of 0.001 and 100 epochs".
The exact coreset sizes are those LBCS achieved in Table 2.

Nothing else about W-NET's size is stated in the paper (Section D.2 defers to
Table 7, whose cell contents are not recoverable from the text). Following the
paper's convention for unspecified architecture details, every architectural
field is labelled SUGGESTED, exposed on :class:`WideResNetConfig`, and
overridable from YAML, so paper-exact widths can be injected without touching
algorithm code.

Design mirrors the rest of the model zoo (``models/convnet.py``,
``models/lenet.py``, ``models/resnet18.py``, ``models/vit_small.py``):

* returns raw logits, so it plugs straight into ``CrossEntropyLoss`` for
  ``f1(m)`` / ``L(m, theta)``;
* exposes a *factory* (:func:`wide_resnet_factory`) producing a brand-new
  ``theta(m)`` for every mask evaluation of Algorithm 1;
* ``features()`` returns the pooled penultimate representation for
  distance-based baselines (Moderate / CCS);
* PyTorch is a *soft* dependency so mask-only unit tests still import cleanly.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # PyTorch is a soft dependency (mask-only unit tests must import cleanly).
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in torch-free envs
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# SVHN constants (Section 5.2 / Section 6: SVHN, 32x32, 10 classes)
# --------------------------------------------------------------------------- #
SVHN_NUM_CLASSES: int = 10
SVHN_IN_CHANNELS: int = 3
SVHN_INPUT_SHAPE: Tuple[int, int, int] = (3, 32, 32)


# --------------------------------------------------------------------------- #
# Section 5.2 post-selection (target) training protocol, re-used by Section 6.
# These three values ARE paper-stated for SVHN ("an Adam optimizer is used with
# a learning rate of 0.001 and 100 epochs"); Section 6 keeps them unchanged.
# --------------------------------------------------------------------------- #
SECTION52_TARGET_OPTIMIZER: str = "adam"
SECTION52_TARGET_LR: float = 0.001
SECTION52_TARGET_EPOCHS: int = 100

# --------------------------------------------------------------------------- #
# SUGGESTED defaults (NOT stated numerically in the paper).
# --------------------------------------------------------------------------- #
SUGGESTED_DEPTH: int = 16
SUGGESTED_WIDEN_FACTOR: int = 4
SUGGESTED_STEM_WIDTH: int = 16
SUGGESTED_KERNEL_SIZE: int = 3
SUGGESTED_DROPOUT: float = 0.0
SUGGESTED_WEIGHT_DECAY: float = 5e-4
SUGGESTED_BATCH_SIZE: int = 128
SUGGESTED_STRIDE: int = 1
SUGGESTED_NUM_WORKERS: int = 0
SUGGESTED_BOTTLENECK: bool = False
SUGGESTED_ETA_MIN: float = 0.0

_CONFIG_FIELDS: Tuple[str, ...] = (
    "in_channels", "num_classes", "input_shape", "depth", "widen_factor",
    "stem_width", "kernel_size", "padding", "dropout", "bottleneck",
    "optimizer", "lr", "epochs", "weight_decay", "batch_size", "scheduler",
    "augment", "num_workers", "eta_min", "momentum", "use_bias",
)


def _resolve_padding(kernel_size: int, padding: Optional[int]) -> int:
    """``padding=None`` means "same" convolution (``kernel_size // 2``)."""
    return kernel_size // 2 if padding is None else int(padding)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class WideResNetConfig:
    """Static description of W-NET plus its post-selection training recipe.

    Architecture fields are SUGGESTED defaults: the paper only states that
    WideResNet (Zagoruyko & Komodakis, 2016) is used on SVHN in Section 6 and
    that "the other experimental settings are not changed". The training fields
    ``optimizer``/``lr``/``epochs`` are paper-stated (Section 5.2, SVHN:
    Adam, 0.001, 100 epochs); the rest are SUGGESTED.
    """

    # --- architecture (SUGGESTED) ---
    in_channels: int = SVHN_IN_CHANNELS
    num_classes: int = SVHN_NUM_CLASSES
    input_shape: Tuple[int, int, int] = SVHN_INPUT_SHAPE
    depth: int = SUGGESTED_DEPTH
    widen_factor: int = SUGGESTED_WIDEN_FACTOR
    stem_width: int = SUGGESTED_STEM_WIDTH
    kernel_size: int = SUGGESTED_KERNEL_SIZE
    padding: Optional[int] = None  # -> kernel_size // 2 ("same")
    dropout: float = SUGGESTED_DROPOUT
    bottleneck: bool = SUGGESTED_BOTTLENECK
    use_bias: bool = True

    # --- post-selection training (paper-stated for SVHN) ---
    optimizer: str = SECTION52_TARGET_OPTIMIZER
    lr: float = SECTION52_TARGET_LR
    epochs: int = SECTION52_TARGET_EPOCHS

    # --- post-selection training (SUGGESTED) ---
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    batch_size: int = SUGGESTED_BATCH_SIZE
    scheduler: Optional[str] = None
    eta_min: float = SUGGESTED_ETA_MIN
    momentum: float = 0.9
    augment: bool = False
    num_workers: int = SUGGESTED_NUM_WORKERS

    # ---------------------------------------------------------------- helpers
    def __post_init__(self) -> None:
        self.input_shape = tuple(int(v) for v in self.input_shape)  # type: ignore[assignment]
        self.depth = int(self.depth)
        self.widen_factor = int(self.widen_factor)
        self.stem_width = int(self.stem_width)
        self.num_classes = int(self.num_classes)
        self.in_channels = int(self.in_channels)
        if self.depth < 4 or (self.depth - 4) % 6 != 0:
            raise ValueError(
                "WideResNet depth must satisfy depth >= 4 and (depth - 4) % 6 == 0 "
                f"(e.g. 16, 28, 40); got depth={self.depth}."
            )

    # -- presets -----------------------------------------------------------
    @classmethod
    def for_svhn(cls, num_classes: int = SVHN_NUM_CLASSES, **overrides: Any) -> "WideResNetConfig":
        """SVHN preset (Section 6 cross-architecture target model)."""
        cfg = cls(in_channels=SVHN_IN_CHANNELS, num_classes=num_classes,
                  input_shape=SVHN_INPUT_SHAPE)
        return cfg.with_overrides(**overrides)

    @classmethod
    def for_inner(cls, num_classes: int = SVHN_NUM_CLASSES, **overrides: Any) -> "WideResNetConfig":
        """Alias retained for symmetry: W-NET is *not* the paper's proxy model.

        Section 5.2 uses "simple convolutional neural networks (CNNs)" for the
        inner loop; W-NET appears only as a Section 6 target model.
        """
        return cls.for_svhn(num_classes=num_classes, **overrides)

    @classmethod
    def for_target(cls, num_classes: int = SVHN_NUM_CLASSES, **overrides: Any) -> "WideResNetConfig":
        """Target preset: Adam, lr=0.001, 100 epochs (Section 5.2 / Section 6)."""
        return cls.for_svhn(
            num_classes=num_classes,
            optimizer=SECTION52_TARGET_OPTIMIZER,
            lr=SECTION52_TARGET_LR,
            epochs=SECTION52_TARGET_EPOCHS,
            **overrides,
        )

    @classmethod
    def cifar_variant(cls, num_classes: int = 10, **overrides: Any) -> "WideResNetConfig":
        """CIFAR-sized W-NET preset (CIFAR-10's paper target is ResNet-18)."""
        return cls.for_svhn(num_classes=num_classes, **overrides)

    # -- generic plumbing ---------------------------------------------------
    def with_overrides(self, **overrides: Any) -> "WideResNetConfig":
        """Return a copy with selected fields replaced."""
        clean = {k: v for k, v in overrides.items() if k in _CONFIG_FIELDS}
        unknown = set(overrides) - set(_CONFIG_FIELDS)
        if unknown:
            LOGGER.debug("WideResNetConfig ignoring unknown override(s): %s", sorted(unknown))
        return replace(self, **clean)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WideResNetConfig":
        data = dict(data or {})
        return cls(**{k: v for k, v in data.items() if k in _CONFIG_FIELDS})

    # -- derived geometry ---------------------------------------------------
    def blocks_per_group(self) -> int:
        """Number of residual blocks ``n`` per group: ``(depth - 4) / 6``."""
        return (self.depth - 4) // 6

    def num_blocks(self) -> int:
        """Total number of residual blocks (``3n``) plus the stem."""
        return 3 * self.blocks_per_group() + 1

    def group_widths(self) -> Tuple[int, int, int]:
        """Output widths of the three residual groups: ``stem * w * {1,2,4}``."""
        w = self.stem_width * self.widen_factor
        return (w, 2 * w, 4 * w)

    def conv_output_shape(self) -> Tuple[int, int, int]:
        """Shape (C, H, W) after the stem (= input resolution of group 1)."""
        _, h, w = self.input_shape
        k = int(self.kernel_size)
        p = _resolve_padding(k, self.padding)
        out = (h + 2 * p - k) // SUGGESTED_STRIDE + 1
        return (self.stem_width, out, out)

    def feature_dim(self) -> int:
        """Dimension of :meth:`WideResNet.features` (global average pooled)."""
        return self.group_widths()[2]

    def pool_output_shape(self) -> Tuple[int, int, int]:
        """Shape after the two stride-2 transitions (before global pooling)."""
        _, h, _ = self.input_shape
        k = int(self.kernel_size)
        p = _resolve_padding(k, self.padding)
        out = (h + 2 * p - k) // SUGGESTED_STRIDE + 1
        for _ in range(2):
            out = out // 2
        return (self.group_widths()[2], out, out)

    def summary(self) -> Dict[str, Any]:
        d = self.to_dict()
        d.update(
            blocks_per_group=self.blocks_per_group(),
            num_blocks=self.num_blocks(),
            group_widths=list(self.group_widths()),
            feature_dim=self.feature_dim(),
            pool_output_shape=list(self.pool_output_shape()),
        )
        return d


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
if _TORCH_AVAILABLE:

    class WideBasicBlock(nn.Module):
        """Residual block: BN-ReLU-Conv(-BN-ReLU-Dropout-Conv) + shortcut."""

        def __init__(
            self,
            in_planes: int,
            out_planes: int,
            stride: int = 1,
            dropout: float = 0.0,
            kernel_size: int = 3,
            padding: Optional[int] = None,
            use_bias: bool = False,
        ) -> None:
            super().__init__()
            pad = _resolve_padding(kernel_size, padding)
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=pad, bias=use_bias,
            )
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.conv2 = nn.Conv2d(
                out_planes, out_planes, kernel_size=kernel_size, stride=1,
                padding=pad, bias=use_bias,
            )
            self.dropout = float(dropout)

            self.equal_in_out = (stride == 1 and in_planes == out_planes)
            if not self.equal_in_out:
                self.shortcut = nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=use_bias
                )
            else:
                self.shortcut = None

        def forward(self, x):
            if self.equal_in_out:
                out = x
            else:
                out = F.relu(self.bn1(x), inplace=True)
            out = self.conv1(out)
            out = F.relu(self.bn2(out), inplace=True)
            if self.dropout > 0.0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = self.conv2(out)
            identity = x if self.shortcut is None else self.shortcut(x)
            return out + identity

    class WideResNet(nn.Module):
        """Wide residual network (Zagoruyko & Komodakis, 2016) returning logits.

        Topology: ``conv1 -> group1 -> group2 -> group3 -> BN -> ReLU -> GAP -> FC``
        with ``n = (depth - 4) / 6`` blocks per group, widening factor ``w`` and
        group widths ``stem * w * {1, 2, 4}``. Group 1 keeps the input resolution;
        groups 2 and 3 downsample by stride 2 at their first block.
        """

        def __init__(self, config: Optional[WideResNetConfig] = None, **kwargs: Any) -> None:
            super().__init__()
            if config is None:
                config = WideResNetConfig(
                    **{k: v for k, v in kwargs.items() if k in _CONFIG_FIELDS}
                )
            elif kwargs:
                config = config.with_overrides(**kwargs)
            self.config = config

            n = config.blocks_per_group()
            k = int(config.kernel_size)
            pad = _resolve_padding(k, config.padding)
            stem = int(config.stem_width)
            widths = config.group_widths()

            self.conv1 = nn.Conv2d(
                config.in_channels, stem, kernel_size=k, stride=1, padding=pad, bias=False
            )

            stages: List[nn.Module] = []
            in_planes = stem
            for gi, (width, stride) in enumerate(zip(widths, (1, 2, 2))):
                planes = in_planes
                for bi in range(n):
                    stages.append(
                        WideBasicBlock(
                            in_planes=planes,
                            out_planes=width,
                            stride=stride if bi == 0 else 1,
                            dropout=config.dropout,
                            kernel_size=k,
                            padding=config.padding,
                            use_bias=config.use_bias,
                        )
                    )
                    planes = width
                in_planes = width
            self.blocks = nn.Sequential(*stages)
            self.num_blocks = len(stages)

            self.bn1 = nn.BatchNorm2d(widths[-1])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(widths[-1], config.num_classes)

            self._init_weights()

        # -- init -----------------------------------------------------------
        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0.0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # -- forward --------------------------------------------------------
        def forward_features(self, x):
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x))
            if x.dtype == torch.uint8:
                x = x.float()
            out = self.conv1(x)
            out = self.blocks(out)
            out = F.relu(self.bn1(out), inplace=True)
            return out

        def forward(self, x):
            feats = self.forward_features(x)
            pooled = self.pool(feats).flatten(1)
            return self.linear(pooled)

        def features(self, x):
            """Penultimate (globally pooled) representation for score baselines."""
            feats = self.forward_features(x)
            return self.pool(feats).flatten(1)

        def num_parameters(self, trainable_only: bool = True) -> int:
            params = self.parameters()
            if trainable_only:
                params = (p for p in self.parameters() if p.requires_grad)
            return int(sum(p.numel() for p in params))

else:  # pragma: no cover - torch-free fallback

    class WideBasicBlock:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required to build a WideBasicBlock.")

    class WideResNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required to build a WideResNet.")


# --------------------------------------------------------------------------- #
# Builders / factories (mirror convnet / lenet / resnet18 / vit_small APIs)
# --------------------------------------------------------------------------- #
def build_wide_resnet(
    num_classes: int = SVHN_NUM_CLASSES,
    in_channels: int = SVHN_IN_CHANNELS,
    input_shape: Optional[Sequence[int]] = None,
    config: Optional[WideResNetConfig] = None,
    **kwargs: Any,
) -> "WideResNet":
    """Instantiate a W-NET from a config and/or explicit overrides."""
    if config is None:
        config = WideResNetConfig.for_svhn(
            num_classes=num_classes, in_channels=in_channels
        )
        if input_shape is not None:
            config = config.with_overrides(input_shape=tuple(int(v) for v in input_shape))
        if kwargs:
            config = config.with_overrides(**kwargs)
    elif kwargs:
        config = config.with_overrides(**kwargs)
    return WideResNet(config=config)


wide_resnet = build_wide_resnet
wnet = build_wide_resnet


def wide_resnet_factory(
    config: Optional[WideResNetConfig] = None, **overrides: Any
) -> Callable[..., "WideResNet"]:
    """Return a builder that yields a *fresh* W-NET on every call.

    Algorithm 1 trains a new ``theta(m)`` for every mask evaluation, so the
    experiment drivers need a factory rather than a single instance (same
    contract as ``convnet_factory`` / ``lenet_factory`` / ``resnet18_factory`` /
    ``vit_small_factory``).
    """
    base = WideResNetConfig.from_dict(
        (config or WideResNetConfig.for_svhn()).with_overrides(**overrides).to_dict()
    )

    def _build(**kwargs: Any) -> "WideResNet":
        cfg = base.with_overrides(**kwargs) if kwargs else base
        return WideResNet(config=cfg)

    _build.config = base  # type: ignore[attr-defined]
    _build.config_dict = base.to_dict()  # type: ignore[attr-defined]
    return _build


def svhn_wide_resnet(num_classes: int = SVHN_NUM_CLASSES, **kwargs: Any) -> "WideResNet":
    """Section 6 SVHN W-NET preset."""
    cfg = WideResNetConfig.for_svhn(num_classes=num_classes, **kwargs)
    return WideResNet(config=cfg)


def wide_resnet_16_4(num_classes: int = SVHN_NUM_CLASSES, **kwargs: Any) -> "WideResNet":
    """Canonical WRN-16-4 instance (the default SUGGESTED configuration)."""
    cfg = WideResNetConfig.for_svhn(num_classes=num_classes, depth=16, widen_factor=4)
    cfg = cfg.with_overrides(**kwargs)
    return WideResNet(config=cfg)


def svhn_wide_resnet_factory(
    num_classes: int = SVHN_NUM_CLASSES, **overrides: Any
) -> Callable[..., "WideResNet"]:
    """Factory preset for the Section 6 SVHN target model."""
    return wide_resnet_factory(WideResNetConfig.for_target(num_classes=num_classes, **overrides))


# Aliases used by configs / experiment drivers.
WideResNetTarget = wide_resnet_factory
WNetSVHN = svhn_wide_resnet
WRN = svhn_wide_resnet


# --------------------------------------------------------------------------- #
# Section 5.2 / Section 6 target training and evaluation helpers
# --------------------------------------------------------------------------- #
def target_train_config(**kwargs: Any) -> Dict[str, Any]:
    """Post-selection training config (Adam, lr 0.001, 100 epochs for SVHN).

    The optimizer/lr/epochs are paper-stated ("an Adam optimizer is used with a
    learning rate of 0.001 and 100 epochs" for F-MNIST and SVHN, Section 5.2,
    kept unchanged in Section 6). Weight decay and batch size are SUGGESTED.
    """
    cfg: Dict[str, Any] = {
        "optimizer": SECTION52_TARGET_OPTIMIZER,
        "lr": SECTION52_TARGET_LR,
        "epochs": SECTION52_TARGET_EPOCHS,
        "weight_decay": SUGGESTED_WEIGHT_DECAY,
        "batch_size": SUGGESTED_BATCH_SIZE,
        "scheduler": None,
    }
    cfg.update(kwargs)
    return cfg


def build_target_optimizer(model, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
    """Build the post-selection optimizer (Adam by default, per Section 5.2)."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to build an optimizer.")
    cfg = target_train_config()
    cfg.update(config or {})
    cfg.update(kwargs)
    name = str(cfg.get("optimizer", SECTION52_TARGET_OPTIMIZER)).lower()
    lr = float(cfg.get("lr", SECTION52_TARGET_LR))
    wd = float(cfg.get("weight_decay", SUGGESTED_WEIGHT_DECAY))
    params = [p for p in model.parameters() if p.requires_grad]
    if name in ("adam", "adamw"):
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd
        )
    raise ValueError(f"Unsupported target optimizer: {cfg.get('optimizer')!r}")


def build_target_scheduler(
    optimizer,
    config: Optional[Dict[str, Any]] = None,
    steps_per_epoch: int = 1,
    **kwargs: Any,
):
    """Optional cosine / step / exponential scheduler.

    Defaults to ``None``: the paper states a cosine scheduler only for CIFAR-10
    (ResNet-18, SGD), not for the SVHN target models.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to build a scheduler.")
    cfg = target_train_config()
    cfg.update(config or {})
    cfg.update(kwargs)
    name = cfg.get("scheduler")
    if not name:
        return None
    name = str(name).lower()
    if name == "cosine":
        total = int(cfg.get("epochs", SECTION52_TARGET_EPOCHS)) * max(1, int(steps_per_epoch))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total), eta_min=float(cfg.get("eta_min", SUGGESTED_ETA_MIN))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 30)),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(cfg.get("gamma", 0.95))
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")


def evaluate(model, loader, device: Optional[str] = None) -> float:
    """Top-1 test accuracy (%) of ``model`` on ``loader``.

    Unpacks both ``(x, y)`` pairs and ``(x, y, index)`` triples produced by the
    index-aware loaders in :mod:`lbcs_repro.data.datasets`.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to evaluate a model.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            logits = model(inputs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    return 100.0 * correct / max(1, total)


accuracy = evaluate


def train_target_model(
    model,
    train_loader,
    test_loader=None,
    epochs: int = SECTION52_TARGET_EPOCHS,
    lr: float = SECTION52_TARGET_LR,
    optimizer: str = SECTION52_TARGET_OPTIMIZER,
    weight_decay: float = SUGGESTED_WEIGHT_DECAY,
    scheduler: Optional[str] = None,
    device: Optional[str] = None,
    criterion=None,
    verbose: bool = False,
    log_every: int = 0,
    **kwargs: Any,
) -> Tuple[Any, List[float]]:
    """Train a W-NET on a constructed coreset (Section 5.2 / 6 protocol).

    For SVHN: Adam, lr = 0.001, 100 epochs, with the other settings unchanged
    from Section 5.2. Returns ``(model, accuracy_history)``; the history holds
    the test accuracy after each epoch (``nan`` when ``test_loader is None``).
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to train a target model.")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model.to(dev)

    cfg = target_train_config(
        optimizer=optimizer, lr=lr, epochs=epochs,
        weight_decay=weight_decay, scheduler=scheduler,
    )
    steps_per_epoch = max(1, len(train_loader))
    opt = build_target_optimizer(model, cfg)
    sched = build_target_scheduler(opt, cfg, steps_per_epoch=steps_per_epoch)
    crit = criterion if criterion is not None else nn.CrossEntropyLoss()

    history: List[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(inputs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            loss = crit(logits, targets)
            loss.backward()
            opt.step()
            running += float(loss.item()) * int(targets.numel())
            seen += int(targets.numel())
        if sched is not None:
            sched.step()
        acc = evaluate(model, test_loader, device=device) if test_loader is not None else float("nan")
        history.append(acc)
        if verbose and log_every and epoch % log_every == 0:
            LOGGER.info(
                "W-NET epoch %d/%d loss=%.4f test_acc=%.2f%%",
                epoch, int(epochs), running / max(1, seen), acc,
            )
    return model, history


def train_svhn_wide_resnet(
    train_loader,
    test_loader=None,
    num_classes: int = SVHN_NUM_CLASSES,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[Any, List[float]]:
    """Build a fresh SVHN W-NET and train it on the given (coreset) loader."""
    model = svhn_wide_resnet(num_classes=num_classes)
    return train_target_model(
        model, train_loader, test_loader=test_loader, device=device, **kwargs
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "WideResNet": wide_resnet_factory,
    "wide_resnet": wide_resnet_factory,
    "WideResNet-16-4": wide_resnet_factory,
    "WRN": wide_resnet_factory,
    "W-NET": wide_resnet_factory,
    "wnet": wide_resnet_factory,
    "SVHNWideResNet": wide_resnet_factory,
    "WideResNetTarget": wide_resnet_factory,
}


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline structural / behavioural checks (config part runs without torch)."""
    results: Dict[str, Any] = {}

    # --- config algebra (no torch needed) ---------------------------------
    cfg = WideResNetConfig.for_svhn()
    assert cfg.blocks_per_group() == 2, cfg.blocks_per_group()
    assert cfg.num_blocks() == 7, cfg.num_blocks()
    assert cfg.group_widths() == (64, 128, 256), cfg.group_widths()
    assert cfg.feature_dim() == 256
    assert cfg.pool_output_shape() == (256, 8, 8), cfg.pool_output_shape()
    assert WideResNetConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()

    over = cfg.with_overrides(depth=28, widen_factor=2)
    assert over.blocks_per_group() == 4
    assert over.group_widths() == (32, 64, 128)
    assert over.summary()["num_blocks"] == 13
    results["config"] = cfg.summary()

    try:
        WideResNetConfig(depth=15)
        raise AssertionError("invalid depth should raise")
    except ValueError:
        results["depth_validation"] = "ok"

    # paper-stated target protocol for SVHN
    tc = target_train_config()
    assert tc["optimizer"] == "adam" and tc["lr"] == 0.001 and tc["epochs"] == 100
    results["target_config"] = tc

    if not _TORCH_AVAILABLE:  # pragma: no cover
        results["torch"] = "unavailable"
        if verbose:
            print("[wide_resnet selftest] torch unavailable; config checks passed.")
        return results

    x = torch.randn(4, SVHN_IN_CHANNELS, 32, 32)
    model = svhn_wide_resnet()
    model.eval()
    with torch.no_grad():
        logits = model(x)
        feats = model.features(x)
    assert tuple(logits.shape) == (4, SVHN_NUM_CLASSES), tuple(logits.shape)
    assert tuple(feats.shape) == (4, 256), tuple(feats.shape)
    assert model.num_blocks == 6, model.num_blocks
    results["logits_shape"] = list(logits.shape)
    results["feature_dim"] = int(feats.shape[1])

    # numpy input tolerance
    with torch.no_grad():
        out_np = model(np.random.randn(2, 3, 32, 32).astype("float32"))
    assert tuple(out_np.shape) == (2, SVHN_NUM_CLASSES)

    # backward pass reaches every parameter
    model.train()
    model(x).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing
    results["parameters"] = model.num_parameters()

    # factory freshness + config round trip
    factory = svhn_wide_resnet_factory()
    assert factory() is not factory()
    assert getattr(factory, "config").depth == SUGGESTED_DEPTH
    results["factory_depth"] = getattr(factory, "config_dict", {}).get("depth")

    # registry exposure
    for key in ("WideResNet", "W-NET", "SVHNWideResNet"):
        assert key in MODEL_REGISTRY, key
    results["registry"] = sorted(MODEL_REGISTRY)

    # training smoke test: history length matches the epoch count
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, torch.randint(0, SVHN_NUM_CLASSES, (4,))),
        batch_size=2,
    )
    _, hist = train_target_model(model, loader, test_loader=loader, epochs=1)
    assert len(hist) == 1
    results["train_smoke"] = hist

    if verbose:
        print("[wide_resnet selftest]", results)
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)


__all__ = [
    "WideResNetConfig",
    "WideResNet",
    "WideBasicBlock",
    "build_wide_resnet",
    "wide_resnet",
    "wnet",
    "wide_resnet_factory",
    "svhn_wide_resnet",
    "svhn_wide_resnet_factory",
    "wide_resnet_16_4",
    "WideResNetTarget",
    "WNetSVHN",
    "WRN",
    "train_target_model",
    "train_svhn_wide_resnet",
    "build_target_optimizer",
    "build_target_scheduler",
    "target_train_config",
    "evaluate",
    "accuracy",
    "MODEL_REGISTRY",
    "SECTION52_TARGET_OPTIMIZER",
    "SECTION52_TARGET_LR",
    "SECTION52_TARGET_EPOCHS",
    "SUGGESTED_DEPTH",
    "SUGGESTED_WIDEN_FACTOR",
    "SUGGESTED_STEM_WIDTH",
    "SUGGESTED_DROPOUT",
    "SUGGESTED_WEIGHT_DECAY",
    "SUGGESTED_BATCH_SIZE",
    "SVHN_NUM_CLASSES",
    "SVHN_IN_CHANNELS",
    "SVHN_INPUT_SHAPE",
]
