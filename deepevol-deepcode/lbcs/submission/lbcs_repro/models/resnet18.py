"""ResNet-18 target network for CIFAR-10.

Paper references
----------------
* Section 5.2 (Datasets and implementation):
  "After coreset selection, for training on the constructed coreset, we utilize a
   LeNet (LeCun et al., 1998) for F-MNIST, a CNN for SVHN, and a ResNet-18 network
   for CIFAR-10 respectively. ... For CIFAR-10, an SGD optimizer is exploited with
   an initial learning rate of 0.1 and a cosine rate scheduler. 200 epochs are set
   totally."
* Appendix D.2 (Details of Network Structures): "We provide the detailed network
  structures of the used models in our main paper, which can be checked in Table 7."
  (Table 7 cells are not machine-extractable from the paper PDF, so the remaining
  architectural choices below are marked SUGGESTED and are fully overridable.)
* Section 2.1: the trivial solutions ``min_m f_1(m) s.t. theta(m) in argmin L``
  (Eq. 3) and ``min_m (1-lambda) f_1(m) + lambda f_2(m)`` (Eq. 4) motivate the
  lexicographic formulation this target model serves (coreset sizes and
  accuracies are reported before/after LBCS in Table 2 / Table 3).

This module implements the *post-coreset-selection target model* used for CIFAR-10,
together with the paper-stated optimizer schedule (SGD lr 0.1, momentum 0.9, cosine
scheduler, 200 epochs).

The network is the standard CIFAR-10 ResNet-18 variant (two 3x3 convolutions per
basic block, four residual stages with two blocks each, widths 64/128/256/512,
global average pooling and a single fully-connected classifier).  Raw logits are
returned so the model plugs directly into ``torch.nn.CrossEntropyLoss`` (used for
``f_1(m)`` and ``L(m, theta)``).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # torch is a soft dependency: mask-only unit tests run without PyTorch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in torch-less environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


__all__ = [
    # configuration
    "ResNet18Config",
    # model
    "BasicBlock",
    "ResNet18",
    "build_resnet18",
    "resnet18",
    "resnet18_factory",
    "cifar_resnet18",
    "cifar10_resnet18",
    # Section 5.2 target-training protocol
    "SECTION52_TARGET_OPTIMIZER",
    "SECTION52_TARGET_LR",
    "SECTION52_TARGET_MOMENTUM",
    "SECTION52_TARGET_EPOCHS",
    "SECTION52_TARGET_SCHEDULER",
    "SECTION52_TARGET_WEIGHT_DECAY",
    "SUGGESTED_BATCH_SIZE",
    "target_train_config",
    "build_target_optimizer",
    "build_target_scheduler",
    "train_target_model",
    "train_cifar_resnet18",
    "evaluate",
    "accuracy",
    # registry
    "MODEL_REGISTRY",
]


# --------------------------------------------------------------------------- #
# Paper-stated Section 5.2 target-training hyper-parameters
# --------------------------------------------------------------------------- #
SECTION52_TARGET_OPTIMIZER = "sgd"
SECTION52_TARGET_LR = 0.1
SECTION52_TARGET_MOMENTUM = 0.9
SECTION52_TARGET_EPOCHS = 200
SECTION52_TARGET_SCHEDULER = "cosine"
# Weight decay and batch size are NOT stated by the paper (SUGGESTED defaults,
# standard for ResNet-18 on CIFAR-10).
SECTION52_TARGET_WEIGHT_DECAY = 5e-4  # SUGGESTED
SUGGESTED_BATCH_SIZE = 128  # SUGGESTED
SUGGESTED_COSINE_ETA_MIN = 0.0

CIFAR_NUM_CLASSES = 10
CIFAR_IN_CHANNELS = 3
CIFAR_INPUT_SHAPE = (3, 32, 32)
SUGGESTED_STAGE_CHANNELS: Tuple[int, int, int, int] = (64, 128, 256, 512)
SUGGESTED_BLOCKS_PER_STAGE = (2, 2, 2, 2)
SUGGESTED_STEM_WIDTH = 64


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class ResNet18Config:
    """Static description of the CIFAR-10 ResNet-18.

    Only the *optimizer* schedule (SGD, lr ``0.1``, momentum ``0.9``, cosine
    scheduler, 200 epochs) is stated by the paper (Section 5.2).  Every
    architectural field below carries a SUGGESTED value because Appendix D.2
    Table 7 is not machine-extractable from the paper text; all fields are
    overridable from YAML without editing algorithm code.
    """

    in_channels: int = CIFAR_IN_CHANNELS
    num_classes: int = CIFAR_NUM_CLASSES
    input_shape: Tuple[int, int, int] = CIFAR_INPUT_SHAPE
    # SUGGESTED: standard CIFAR ResNet-18 topology
    stage_channels: Tuple[int, int, int, int] = SUGGESTED_STAGE_CHANNELS
    blocks_per_stage: Tuple[int, int, int, int] = SUGGESTED_BLOCKS_PER_STAGE
    stem_width: int = SUGGESTED_STEM_WIDTH
    kernel_size: int = 3
    padding: int = 1
    zero_init_residual: bool = True
    groups: int = 1
    width_per_group: int = 64
    norm_layer: str = "batchnorm"
    dropout: float = 0.0
    # SUGGESTED training defaults (the paper states the schedule but not these)
    batch_size: int = SUGGESTED_BATCH_SIZE
    weight_decay: float = SECTION52_TARGET_WEIGHT_DECAY
    optimizer: str = SECTION52_TARGET_OPTIMIZER
    lr: float = SECTION52_TARGET_LR
    momentum: float = SECTION52_TARGET_MOMENTUM
    epochs: int = SECTION52_TARGET_EPOCHS
    scheduler: str = SECTION52_TARGET_SCHEDULER
    augment: bool = False
    num_workers: int = 0

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def for_cifar10(cls, num_classes: int = CIFAR_NUM_CLASSES, **overrides: Any) -> "ResNet18Config":
        """Canonical CIFAR-10 target-model configuration (Section 5.2)."""
        return cls(num_classes=num_classes).with_overrides(**overrides)

    @classmethod
    def for_section52(cls, **overrides: Any) -> "ResNet18Config":
        """Alias of :meth:`for_cifar10` named after the paper section."""
        return cls.for_cifar10(**overrides)

    # -- utilities --------------------------------------------------------- #
    def with_overrides(self, **overrides: Any) -> "ResNet18Config":
        known = set(self.__dataclass_fields__)  # type: ignore[attr-defined]
        clean: Dict[str, Any] = {}
        for key, value in overrides.items():
            if key in known and value is not None:
                clean[key] = value
        for tuple_key in ("stage_channels", "blocks_per_stage", "input_shape"):
            if isinstance(clean.get(tuple_key), list):
                clean[tuple_key] = tuple(clean[tuple_key])
        return replace(self, **clean)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key in ("stage_channels", "blocks_per_stage", "input_shape"):
            if key in data:
                data[key] = list(data[key])
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ResNet18Config":
        if not data:
            return cls()
        return cls().with_overrides(**dict(data))

    # -- derived quantities ------------------------------------------------ #
    def num_blocks(self) -> int:
        return int(sum(self.blocks_per_stage))

    def num_stages(self) -> int:
        return len(self.blocks_per_stage)

    def feature_dim(self) -> int:
        return int(self.stage_channels[-1])

    def conv_output_shape(self, input_shape: Optional[Sequence[int]] = None) -> Tuple[int, int]:
        """Spatial size entering the global average pool (CIFAR stem: stride 1, no max-pool)."""
        shape = tuple(input_shape) if input_shape is not None else tuple(self.input_shape)
        h, w = int(shape[1]), int(shape[2])
        return max(h // 32, 1), max(w // 32, 1)

    def summary(self) -> str:
        return (
            f"ResNet18(target, CIFAR-10) stages={self.stage_channels} "
            f"blocks={self.blocks_per_stage} classes={self.num_classes} "
            f"optimizer={self.optimizer}(lr={self.lr}, momentum={self.momentum}) "
            f"epochs={self.epochs} scheduler={self.scheduler}"
        )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
if _TORCH_AVAILABLE:

    def _get_norm_layer(name: str) -> Callable[[int], "nn.Module"]:
        name = (name or "batchnorm").lower()
        if name in ("batchnorm", "bn", "batch_norm"):
            return nn.BatchNorm2d
        if name in ("groupnorm", "gn"):
            return lambda channels: nn.GroupNorm(32 if channels % 32 == 0 else 1, channels)
        if name in ("none", "identity"):
            return lambda channels: nn.Identity()
        raise ValueError(f"Unknown normalization layer: {name!r}")

    class BasicBlock(nn.Module):
        """Standard CIFAR ResNet basic residual block (two 3x3 convolutions)."""

        expansion = 1

        def __init__(
            self,
            in_planes: int,
            planes: int,
            stride: int = 1,
            norm_layer: Optional[Callable[[int], "nn.Module"]] = None,
            kernel_size: int = 3,
            padding: int = 1,
            groups: int = 1,
            base_width: int = 64,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            norm_layer = norm_layer or nn.BatchNorm2d
            width = int(planes * (base_width / 64.0)) * groups
            self.conv1 = nn.Conv2d(
                in_planes, width, kernel_size, stride=stride, padding=padding, bias=False
            )
            self.bn1 = norm_layer(width)
            self.conv2 = nn.Conv2d(
                width, planes, kernel_size, stride=1, padding=padding, groups=groups, bias=False
            )
            self.bn2 = norm_layer(planes)
            self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

            self.downsample: Optional["nn.Sequential"]
            if stride != 1 or in_planes != planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes),
                )
            else:
                self.downsample = None

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            identity = x
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            if self.downsample is not None:
                identity = self.downsample(x)
            out = out + identity
            return F.relu(out, inplace=True)

    class ResNet18(nn.Module):
        """CIFAR-10 ResNet-18 target network returning raw logits."""

        def __init__(self, config: Optional[ResNet18Config] = None, **kwargs: Any) -> None:
            super().__init__()
            if config is not None:
                self.config = config.with_overrides(**kwargs)
            else:
                self.config = ResNet18Config.from_dict(kwargs)
            cfg = self.config
            norm_layer = _get_norm_layer(cfg.norm_layer)

            self.in_planes = int(cfg.stem_width)
            self.conv1 = nn.Conv2d(
                cfg.in_channels,
                self.in_planes,
                cfg.kernel_size,
                stride=1,
                padding=cfg.padding,
                bias=False,
            )
            self.bn1 = norm_layer(self.in_planes)
            # CIFAR variant: identity instead of the ImageNet 3x3/2 max-pool stem,
            # kept as an attribute so callers expecting that interface still work.
            self.maxpool = nn.Identity()

            stages: List[nn.Module] = []
            for stage_idx, (planes, num_blocks) in enumerate(
                zip(cfg.stage_channels, cfg.blocks_per_stage)
            ):
                stride = 1 if stage_idx == 0 else 2
                stages.append(
                    self._make_layer(
                        norm_layer,
                        int(planes),
                        int(num_blocks),
                        stride=stride,
                        kernel_size=cfg.kernel_size,
                        padding=cfg.padding,
                        groups=cfg.groups,
                        base_width=cfg.width_per_group,
                        dropout=cfg.dropout,
                    )
                )
            self.layer1, self.layer2, self.layer3, self.layer4 = stages
            self.layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(int(cfg.stage_channels[-1]), cfg.num_classes)

            self._init_weights(cfg.zero_init_residual)

        # -- construction helpers ----------------------------------------- #
        def _make_layer(
            self,
            norm_layer: Callable[[int], "nn.Module"],
            planes: int,
            num_blocks: int,
            stride: int,
            kernel_size: int,
            padding: int,
            groups: int,
            base_width: int,
            dropout: float,
        ) -> "nn.Sequential":
            blocks: List[nn.Module] = []
            for block_idx in range(num_blocks):
                blocks.append(
                    BasicBlock(
                        self.in_planes,
                        planes,
                        stride=stride if block_idx == 0 else 1,
                        norm_layer=norm_layer,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=groups,
                        base_width=base_width,
                        dropout=dropout,
                    )
                )
                self.in_planes = planes
            return nn.Sequential(*blocks)

        def _init_weights(self, zero_init_residual: bool = True) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0.0, 0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)
            if zero_init_residual:
                for module in self.modules():
                    if isinstance(module, BasicBlock):
                        nn.init.constant_(module.bn2.weight, 0.0)

        # -- forward ------------------------------------------------------ #
        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.maxpool(out)
            out = self.layers(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            return self.fc(out)

        def features(self, x: "torch.Tensor") -> "torch.Tensor":
            """Penultimate (post-pool) features, used by distance-based baselines."""
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.maxpool(out)
            out = self.layers(out)
            out = self.avgpool(out)
            return torch.flatten(out, 1)

        # -- introspection ------------------------------------------------- #
        def num_parameters(self, trainable_only: bool = True) -> int:
            if trainable_only:
                return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
            return int(sum(p.numel() for p in self.parameters()))

        def feature_dim(self) -> int:
            return self.config.feature_dim()

else:  # pragma: no cover - torch-less fallback

    class BasicBlock(object):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required to instantiate BasicBlock.")

    class ResNet18(object):  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required to instantiate ResNet18.")


# --------------------------------------------------------------------------- #
# Factories (contract expected by lbcs.bilevel.LBCS: fresh theta(m) per mask)
# --------------------------------------------------------------------------- #
def build_resnet18(
    num_classes: int = CIFAR_NUM_CLASSES,
    in_channels: int = CIFAR_IN_CHANNELS,
    input_shape: Optional[Sequence[int]] = None,
    config: Optional[ResNet18Config] = None,
    **kwargs: Any,
) -> "ResNet18":
    """Instantiate a CIFAR-10 ResNet-18 (the Section 5.2 target model)."""
    base = config or ResNet18Config()
    overrides: Dict[str, Any] = dict(kwargs)
    overrides["num_classes"] = int(num_classes)
    overrides["in_channels"] = int(in_channels)
    if input_shape is not None:
        overrides["input_shape"] = tuple(input_shape)
    return ResNet18(base.with_overrides(**overrides))


resnet18 = build_resnet18


def resnet18_factory(
    config: Optional[ResNet18Config] = None, **overrides: Any
) -> Callable[..., "ResNet18"]:
    """Return a builder producing *fresh* ResNet-18 instances.

    ``lbcs.bilevel.LBCS`` needs a new ``theta(m)`` for every mask evaluation;
    this mirrors the convention of ``convnet_factory`` / ``lenet_factory``.
    """
    base = (config or ResNet18Config()).with_overrides(**overrides)
    known = set(ResNet18Config.__dataclass_fields__)  # type: ignore[attr-defined]

    def _build(**kwargs: Any) -> "ResNet18":
        cfg = base.with_overrides(**{k: v for k, v in kwargs.items() if k in known})
        return ResNet18(cfg)

    _build.config = base  # type: ignore[attr-defined]
    _build.__name__ = "resnet18_builder"
    return _build


def cifar_resnet18(num_classes: int = CIFAR_NUM_CLASSES, **kwargs: Any) -> "ResNet18":
    """CIFAR-10 target-model preset (SGD lr 0.1, momentum 0.9, cosine, 200 epochs)."""
    return build_resnet18(num_classes=num_classes, **kwargs)


cifar10_resnet18 = cifar_resnet18


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "ResNet18": resnet18_factory,
    "resnet18": resnet18_factory,
    "ResNet-18": resnet18_factory,
    "CIFARResNet18": resnet18_factory,
    "cifar_resnet18": resnet18_factory,
}


# --------------------------------------------------------------------------- #
# Section 5.2 target-training harness
# --------------------------------------------------------------------------- #
def target_train_config(
    epochs: int = SECTION52_TARGET_EPOCHS,
    lr: float = SECTION52_TARGET_LR,
    momentum: float = SECTION52_TARGET_MOMENTUM,
    weight_decay: float = SECTION52_TARGET_WEIGHT_DECAY,
    optimizer: str = SECTION52_TARGET_OPTIMIZER,
    scheduler: str = SECTION52_TARGET_SCHEDULER,
    batch_size: int = SUGGESTED_BATCH_SIZE,
    **extra: Any,
) -> Dict[str, Any]:
    """Paper-stated Section 5.2 target-training configuration as a plain dict."""
    cfg: Dict[str, Any] = {
        "optimizer": optimizer,
        "lr": float(lr),
        "momentum": float(momentum),
        "weight_decay": float(weight_decay),
        "epochs": int(epochs),
        "scheduler": scheduler,
        "batch_size": int(batch_size),
    }
    cfg.update(extra)
    return cfg


def build_target_optimizer(
    model: Any, config: Optional[Dict[str, Any]] = None, **kwargs: Any
) -> Any:
    """Build the Section 5.2 target optimizer (SGD, lr 0.1, momentum 0.9)."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build optimizers.")
    cfg = target_train_config(**{**(config or {}), **kwargs})
    name = str(cfg.get("optimizer", "sgd")).lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg["lr"],
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
            nesterov=bool(cfg.get("nesterov", False)),
        )
    if name in ("adam", "adamw"):
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    raise ValueError(f"Unsupported target optimizer: {cfg.get('optimizer')!r}")


def build_target_scheduler(
    optimizer: Any,
    config: Optional[Dict[str, Any]] = None,
    steps_per_epoch: int = 1,
    **kwargs: Any,
) -> Optional[Any]:
    """Cosine LR scheduler over the full Section 5.2 budget (200 epochs)."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to build schedulers.")
    cfg = target_train_config(**{**(config or {}), **kwargs})
    name = str(cfg.get("scheduler") or "none").lower()
    if name in ("none", "", "constant"):
        return None
    total_steps = max(int(cfg.get("epochs", 1)) * max(int(steps_per_epoch), 1), 1)
    if name in ("cosine", "cosineannealing", "cosineannealinglr"):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(cfg.get("eta_min", SUGGESTED_COSINE_ETA_MIN)),
        )
    if name in ("step", "steplr"):
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", max(total_steps // 3, 1))),
            gamma=float(cfg.get("gamma", 0.1)),
        )
    if name in ("exponential", "explr"):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(cfg.get("gamma", 0.99))
        )
    raise ValueError(f"Unsupported target scheduler: {cfg.get('scheduler')!r}")


def evaluate(model: Any, loader: Any, device: Any = None) -> float:
    """Top-1 accuracy (%) of ``model`` on ``loader`` (deterministic, shuffle=False)."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to evaluate models.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive
                raise ValueError("Loader must yield (inputs, targets[, index]) batches.")
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
    return float(100.0 * correct / total) if total else float("nan")


accuracy = evaluate


def train_target_model(
    model: Any,
    train_loader: Any,
    test_loader: Optional[Any] = None,
    epochs: int = SECTION52_TARGET_EPOCHS,
    lr: float = SECTION52_TARGET_LR,
    momentum: float = SECTION52_TARGET_MOMENTUM,
    weight_decay: float = SECTION52_TARGET_WEIGHT_DECAY,
    optimizer: str = SECTION52_TARGET_OPTIMIZER,
    scheduler: str = SECTION52_TARGET_SCHEDULER,
    device: Any = None,
    verbose: bool = False,
    log_every: int = 0,
    **kwargs: Any,
) -> Tuple[Any, List[float]]:
    """Train a target network on the constructed coreset (Section 5.2 protocol).

    Defaults are exactly the paper's CIFAR-10 settings: "an SGD optimizer is
    exploited with an initial learning rate of 0.1 and a cosine rate scheduler.
    200 epochs are set totally."  Returns ``(model, test_accuracy_per_epoch)``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train target models.")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cfg = target_train_config(
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    opt = build_target_optimizer(model, cfg)
    sched = build_target_scheduler(opt, cfg, steps_per_epoch=max(len(train_loader), 1))
    criterion = nn.CrossEntropyLoss()

    history: List[float] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for batch in train_loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()

        if test_loader is not None:
            acc = evaluate(model, test_loader, device)
            history.append(acc)
            if verbose and log_every and epoch % log_every == 0:
                LOGGER.info("[resnet18] epoch %d/%d test acc %.2f", epoch, epochs, acc)

    return model, history


def train_cifar_resnet18(
    train_loader: Any,
    test_loader: Optional[Any] = None,
    num_classes: int = CIFAR_NUM_CLASSES,
    device: Any = None,
    **kwargs: Any,
) -> Tuple[Any, List[float]]:
    """Convenience: build a ResNet-18 and train it on a CIFAR-10 coreset."""
    model = build_resnet18(num_classes=num_classes)
    return train_target_model(model, train_loader, test_loader=test_loader, device=device, **kwargs)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline structural/forward/backward checks (no dataset needed)."""
    report: Dict[str, Any] = {"torch_available": _TORCH_AVAILABLE}
    if not _TORCH_AVAILABLE:  # pragma: no cover
        if verbose:
            print("[resnet18] PyTorch unavailable - skipping model checks.")
        return report

    torch.manual_seed(0)
    cfg = ResNet18Config.for_cifar10()
    model = ResNet18(cfg)
    x = torch.randn(4, CIFAR_IN_CHANNELS, 32, 32)
    logits = model(x)
    assert tuple(logits.shape) == (4, CIFAR_NUM_CLASSES), logits.shape
    feats = model.features(x)
    assert tuple(feats.shape) == (4, cfg.feature_dim()), feats.shape

    loss = F.cross_entropy(logits, torch.randint(0, CIFAR_NUM_CLASSES, (4,)))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)

    assert cfg.num_blocks() == sum(SUGGESTED_BLOCKS_PER_STAGE)
    assert len(list(model.layer4)) == SUGGESTED_BLOCKS_PER_STAGE[-1]

    # the factory must return *fresh* instances and expose .config
    factory = resnet18_factory()
    a, b = factory(), factory()
    assert a is not b
    assert hasattr(factory, "config")

    # Section 5.2 schedule round-trip (SGD lr 0.1, cosine, 200 epochs)
    tc = target_train_config()
    assert tc["optimizer"] == "sgd" and tc["lr"] == 0.1 and tc["epochs"] == 200
    opt = build_target_optimizer(model, tc)
    sched = build_target_scheduler(opt, tc, steps_per_epoch=10)
    assert sched is not None and int(sched.T_max) == 2000

    report.update(
        {
            "logits_shape": tuple(logits.shape),
            "feature_dim": cfg.feature_dim(),
            "num_parameters": model.num_parameters(),
            "conv_output_shape": cfg.conv_output_shape(),
            "summary": cfg.summary(),
        }
    )
    if verbose:
        print("[resnet18] self-test passed:", report)
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest(verbose=True)
