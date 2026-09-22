"""CIFAR-10 convolutional networks for the LBCS reproduction (Section 5.2).

The paper states (Section 5.2, "Datasets and implementation")::

    "In the procedure of coreset selection, we employ a LeNet for F-MNIST and
     simple convolutional neural networks (CNNs) for SVHN and CIFAR-10. An Adam
     optimizer is used with a learning rate of 0.001 for the inner loop."
    "After coreset selection, for training on the constructed coreset, we
     utilize a LeNet for F-MNIST, a CNN for SVHN, and a ResNet-18 network for
     CIFAR-10 respectively."
    "For CIFAR-10, an SGD optimizer is exploited with an initial learning rate
     of 0.1 and a cosine rate scheduler. 200 epochs are set totally."

Appendix D.2 states that the "detailed network structures of the used models in
our main paper [...] can be checked in Table 7"; however, Table 7's cell
contents are not recoverable from the paper text that is available to this
reproduction.  Therefore every layer width below is an explicitly labelled
**SUGGESTED** default (not a paper-stated value), exposed through
:class:`CIFARCNNConfig` so exact Table 7 values can be injected from YAML
without touching code.

Two presets are provided:

* :func:`cifar_inner_cnn` -- the simple CNN used as the inner-loop (proxy)
  model ``theta(m)`` during coreset selection (Adam, lr 0.001).
* :func:`cifar_target_cnn` -- a slightly wider CNN preset that can be trained
  on the constructed coreset.  Section 5.2/Appendix D.2 specify a *ResNet-18*
  for CIFAR-10 target training (see :mod:`lbcs_repro.models.resnet18`); this
  preset is kept for ablation / non-ResNet comparisons and is *not* the
  paper's primary CIFAR-10 target network.

Both presets return raw logits so they plug directly into
``torch.nn.CrossEntropyLoss`` for ``f1(m)`` and ``L(m, theta)``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace, asdict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # torch is a soft dependency: mask-only code paths must work without it.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without PyTorch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paper-stated constants for the CIFAR-10 benchmark.
# ---------------------------------------------------------------------------
CIFAR_INPUT_SHAPE: Tuple[int, int, int] = (3, 32, 32)
CIFAR_NUM_CLASSES: int = 10
CIFAR_IN_CHANNELS: int = 3

# Inner-loop optimizer settings stated in Section 5.2.
SECTION52_INNER_OPTIMIZER: str = "adam"
SECTION52_INNER_LR: float = 0.001

# Target-model optimizer settings stated in Section 5.2 / addendum.
SECTION52_TARGET_OPTIMIZER: str = "sgd"
SECTION52_TARGET_LR: float = 0.1
SECTION52_TARGET_MOMENTUM: float = 0.9
SECTION52_TARGET_EPOCHS: int = 200
SECTION52_TARGET_SCHEDULER: str = "cosine"

# ---------------------------------------------------------------------------
# SUGGESTED (NOT paper-stated) architectural defaults.
# ---------------------------------------------------------------------------
SUGGESTED_CONV_CHANNELS: Tuple[int, ...] = (32, 64, 128)
SUGGESTED_CONV_CHANNELS_INNER: Tuple[int, ...] = (32, 32, 64, 64)
SUGGESTED_CONV_CHANNELS_TARGET: Tuple[int, ...] = (64, 64, 128, 128)
SUGGESTED_KERNEL_SIZE: int = 3
SUGGESTED_STRIDE: int = 1
SUGGESTED_POOL_EVERY: int = 2
SUGGESTED_POOL_KERNEL: int = 2
SUGGESTED_DROPOUT: float = 0.25
SUGGESTED_HIDDEN_DIMS: Tuple[int, ...] = (256,)
SUGGESTED_HIDDEN_DIMS_TARGET: Tuple[int, ...] = (512, 256)
SUGGESTED_BATCHNORM: bool = True
SUGGESTED_WEIGHT_DECAY: float = 5e-4


def _resolve_padding(padding: Optional[int], kernel_size: int) -> int:
    """``None`` means "same" padding (``kernel_size // 2``)."""
    if padding is None:
        return kernel_size // 2
    return int(padding)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CIFARCNNConfig:
    """Static description of a CIFAR-10 CNN.

    Every field is overridable, so the exact Table 7 architecture (Appendix D.2)
    can be supplied from a YAML config once it is known.  The defaults are
    SUGGESTED values, not paper-stated ones.
    """

    in_channels: int = CIFAR_IN_CHANNELS
    num_classes: int = CIFAR_NUM_CLASSES
    conv_channels: Sequence[int] = field(default_factory=lambda: tuple(SUGGESTED_CONV_CHANNELS))
    kernel_size: int = SUGGESTED_KERNEL_SIZE
    stride: int = SUGGESTED_STRIDE
    padding: Optional[int] = None  # None -> kernel_size // 2 ("same")
    pool_every: int = SUGGESTED_POOL_EVERY
    pool_kernel: int = SUGGESTED_POOL_KERNEL
    pool_stride: Optional[int] = None  # None -> pool_kernel
    hidden_dims: Sequence[int] = field(default_factory=lambda: tuple(SUGGESTED_HIDDEN_DIMS))
    dropout: float = SUGGESTED_DROPOUT
    batchnorm: bool = SUGGESTED_BATCHNORM
    input_shape: Tuple[int, int, int] = CIFAR_INPUT_SHAPE

    # -- constructors -------------------------------------------------------
    def __post_init__(self) -> None:
        self.conv_channels = tuple(int(c) for c in self.conv_channels)
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        self.input_shape = tuple(int(s) for s in self.input_shape)  # type: ignore[assignment]
        if self.padding is None:
            self.padding = _resolve_padding(None, self.kernel_size)
        if self.pool_stride is None:
            self.pool_stride = self.pool_kernel
        self.pool_every = max(1, int(self.pool_every))

    @classmethod
    def for_cifar(cls, num_classes: int = CIFAR_NUM_CLASSES, **overrides: Any) -> "CIFARCNNConfig":
        """Default CIFAR-10 config (SUGGESTED widths)."""
        cfg = cls(num_classes=num_classes, input_shape=CIFAR_INPUT_SHAPE)
        return cfg.with_overrides(**overrides) if overrides else cfg

    @classmethod
    def for_inner(cls, num_classes: int = CIFAR_NUM_CLASSES, **overrides: Any) -> "CIFARCNNConfig":
        """Inner-loop (proxy) preset used during coreset selection."""
        cfg = cls(
            num_classes=num_classes,
            conv_channels=SUGGESTED_CONV_CHANNELS_INNER,
            hidden_dims=SUGGESTED_HIDDEN_DIMS,
            dropout=SUGGESTED_DROPOUT,
            batchnorm=SUGGESTED_BATCHNORM,
            input_shape=CIFAR_INPUT_SHAPE,
        )
        return cfg.with_overrides(**overrides) if overrides else cfg

    @classmethod
    def for_target(cls, num_classes: int = CIFAR_NUM_CLASSES, **overrides: Any) -> "CIFARCNNConfig":
        """Wider CNN preset for post-selection training (ResNet-18 is the
        paper's specified CIFAR-10 target model; see :mod:`resnet18`)."""
        cfg = cls(
            num_classes=num_classes,
            conv_channels=SUGGESTED_CONV_CHANNELS_TARGET,
            hidden_dims=SUGGESTED_HIDDEN_DIMS_TARGET,
            dropout=SUGGESTED_DROPOUT,
            batchnorm=SUGGESTED_BATCHNORM,
            input_shape=CIFAR_INPUT_SHAPE,
        )
        return cfg.with_overrides(**overrides) if overrides else cfg

    def with_overrides(self, **overrides: Any) -> "CIFARCNNConfig":
        allowed = set(self.to_dict().keys())
        payload = {k: v for k, v in overrides.items() if k in allowed and v is not None}
        return replace(self, **payload)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["conv_channels"] = list(self.conv_channels)
        data["hidden_dims"] = list(self.hidden_dims)
        data["input_shape"] = list(self.input_shape)
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CIFARCNNConfig":
        if not data:
            return cls()
        payload = {k: v for k, v in dict(data).items() if k in cls.__dataclass_fields__}
        for key in ("conv_channels", "hidden_dims", "input_shape"):
            if key in payload and payload[key] is not None:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    # -- derived quantities -------------------------------------------------
    def num_conv_layers(self) -> int:
        return len(self.conv_channels)

    def num_pools(self) -> int:
        """Number of max-pooling stages (one every ``pool_every`` convs)."""
        return len(range(1, len(self.conv_channels) + 1, self.pool_every))

    def conv_output_shape(self, input_shape: Optional[Sequence[int]] = None) -> Tuple[int, int, int]:
        """Spatial shape of the last conv feature map (channels, H, W)."""
        shape = tuple(input_shape) if input_shape is not None else tuple(self.input_shape)
        h, w = int(shape[-2]), int(shape[-1])
        for i in range(1, len(self.conv_channels) + 1):
            if i % self.pool_every == 0:
                h = (h - self.pool_kernel) // int(self.pool_stride) + 1
                w = (w - self.pool_kernel) // int(self.pool_stride) + 1
        return int(self.conv_channels[-1]), max(h, 1), max(w, 1)

    def feature_dim(self, input_shape: Optional[Sequence[int]] = None) -> int:
        c, h, w = self.conv_output_shape(input_shape)
        return int(c * h * w)

    def summary(self) -> str:
        c, h, w = self.conv_output_shape()
        return (
            f"CIFARCNNConfig(convs={tuple(self.conv_channels)}, k={self.kernel_size}, "
            f"pool_every={self.pool_every}, hidden={tuple(self.hidden_dims)}, "
            f"dropout={self.dropout}, bn={self.batchnorm}, feature_map={c}x{h}x{w}, "
            f"feature_dim={self.feature_dim()})"
        )


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class CIFARCNN(nn.Module):
        """Simple CNN for CIFAR-10 with configurable depth/width.

        Structure: ``[Conv -> (BatchNorm) -> ReLU -> (MaxPool every pool_every)
        -> Dropout] * len(conv_channels)`` followed by
        ``[Linear -> ReLU -> Dropout] * len(hidden_dims) -> Linear``.
        Returns raw logits of shape ``(N, num_classes)``.
        """

        def __init__(self, config: Optional[CIFARCNNConfig] = None, **kwargs: Any) -> None:
            super().__init__()
            self.config = (config or CIFARCNNConfig()).with_overrides(**kwargs) if kwargs else (
                config or CIFARCNNConfig()
            )
            cfg = self.config
            pad = _resolve_padding(cfg.padding, cfg.kernel_size)

            blocks: list = []
            in_ch = int(cfg.in_channels)
            for i, out_ch in enumerate(cfg.conv_channels, start=1):
                blocks.append(
                    nn.Conv2d(
                        in_ch,
                        int(out_ch),
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=pad,
                        bias=not cfg.batchnorm,
                    )
                )
                if cfg.batchnorm:
                    blocks.append(nn.BatchNorm2d(int(out_ch)))
                blocks.append(nn.ReLU(inplace=True))
                if i % cfg.pool_every == 0:
                    blocks.append(
                        nn.MaxPool2d(kernel_size=cfg.pool_kernel, stride=int(cfg.pool_stride))
                    )
                if cfg.dropout and cfg.dropout > 0:
                    blocks.append(nn.Dropout(p=float(cfg.dropout)))
                in_ch = int(out_ch)
            self.features_extractor = nn.Sequential(*blocks)

            feat_dim = cfg.feature_dim()
            head: list = []
            prev = feat_dim
            for hidden in cfg.hidden_dims:
                head.append(nn.Linear(prev, int(hidden)))
                head.append(nn.ReLU(inplace=True))
                if cfg.dropout and cfg.dropout > 0:
                    head.append(nn.Dropout(p=float(cfg.dropout)))
                prev = int(hidden)
            head.append(nn.Linear(prev, int(cfg.num_classes)))
            self.classifier = nn.Sequential(*head)

            self._feature_dim = int(prev)
            self._init_weights()

        # -- forward -------------------------------------------------------
        def forward(self, x: Any) -> Any:  # returns torch.Tensor
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return self.classifier(torch.flatten(self.features_extractor(x), start_dim=1))

        def features(self, x: Any) -> Any:
            """Flattened penultimate features (for Moderate/CCS-style scores)."""
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return torch.flatten(self.features_extractor(x), start_dim=1)

        # -- utilities -----------------------------------------------------
        def num_parameters(self, trainable_only: bool = True) -> int:
            return int(
                sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))
            )

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _selftest(verbose: bool = True) -> Dict[str, Any]:
        """Offline structural + forward/backward checks."""
        results: Dict[str, Any] = {}
        cfg = CIFARCNNConfig.for_inner()
        model = CIFARCNN(cfg)
        results["config"] = cfg.summary()
        results["num_conv_layers"] = cfg.num_conv_layers()
        results["num_pools"] = cfg.num_pools()
        results["feature_dim"] = cfg.feature_dim()

        x = torch.randn(4, CIFAR_IN_CHANNELS, 32, 32)
        y = torch.randint(0, CIFAR_NUM_CLASSES, (4,))
        logits = model(x)
        feats = model.features(x)
        results["logits_shape"] = tuple(logits.shape)
        results["feature_shape"] = tuple(feats.shape)

        assert tuple(logits.shape) == (4, CIFAR_NUM_CLASSES), results["logits_shape"]
        assert feats.shape[1] == cfg.feature_dim(), (
            feats.shape[1],
            cfg.feature_dim(),
        )
        assert cfg.num_pools() == len(range(1, len(cfg.conv_channels) + 1, cfg.pool_every))
        n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
        n_pool = sum(1 for m in model.modules() if isinstance(m, nn.MaxPool2d))
        results["counted_conv"], results["counted_pool"] = n_conv, n_pool
        assert n_conv == cfg.num_conv_layers(), (n_conv, cfg.num_conv_layers())
        assert n_pool == cfg.num_pools(), (n_pool, cfg.num_pools())
        assert cfg.num_conv_layers() >= 2, "inner CNN must have at least two conv blocks"

        loss = F.cross_entropy(logits, y)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        results["params_without_grad"] = missing
        assert not missing, missing

        # config round-trip + factory contract
        rt = CIFARCNNConfig.from_dict(cfg.to_dict())
        assert rt.conv_channels == cfg.conv_channels and rt.feature_dim() == cfg.feature_dim()
        factory = cifar_cnn_factory(cfg)
        m1, m2 = factory(), factory()
        results["factory_distinct_instances"] = m1 is not m2
        assert m1 is not m2

        # target preset
        tcfg = CIFARCNNConfig.for_target()
        tmodel = CIFARCNN(tcfg)
        results["target_summary"] = tcfg.summary()
        assert tuple(tmodel(x).shape) == (4, CIFAR_NUM_CLASSES)

        results["ok"] = True
        if verbose:
            for key, value in results.items():
                LOGGER.info("  %-28s %s", key, value)
        return results

else:  # pragma: no cover - stub used when PyTorch is unavailable

    class CIFARCNN:  # type: ignore[no-redef]
        """Placeholder that raises a clear error when PyTorch is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required to instantiate CIFARCNN. Install torch/torchvision to use "
                "the CIFAR-10 models of the LBCS reproduction."
            )

    def _selftest(verbose: bool = True) -> Dict[str, Any]:  # type: ignore[no-redef]
        raise ImportError("PyTorch is required to run the CIFARCNN self-test.")


# ---------------------------------------------------------------------------
# Factories (contract expected by lbcs.bilevel.LBCS: fresh theta(m) per mask)
# ---------------------------------------------------------------------------
def build_cifar_cnn(
    num_classes: int = CIFAR_NUM_CLASSES,
    in_channels: int = CIFAR_IN_CHANNELS,
    input_shape: Optional[Sequence[int]] = None,
    config: Optional[CIFARCNNConfig] = None,
    **kwargs: Any,
) -> "CIFARCNN":
    """Instantiate a :class:`CIFARCNN` from a config plus overrides."""
    cfg = config or CIFARCNNConfig()
    overrides: Dict[str, Any] = dict(kwargs)
    overrides.setdefault("num_classes", num_classes)
    if in_channels is not None:
        overrides.setdefault("in_channels", int(in_channels))
    if input_shape is not None:
        overrides.setdefault("input_shape", tuple(int(s) for s in input_shape))
    return CIFARCNN(cfg.with_overrides(**overrides))


#: Short alias mirroring the other model modules.
cifar_cnn = build_cifar_cnn


def cifar_cnn_factory(
    config: Optional[CIFARCNNConfig] = None, **overrides: Any
) -> Callable[..., "CIFARCNN"]:
    """Return a builder producing *fresh* CIFARCNN instances.

    ``lbcs.bilevel.LBCS`` needs one ``theta(m)`` per mask evaluation, so each
    call to the returned builder must construct a new network rather than reuse
    a trained one.  The returned callable carries a ``.config`` attribute.
    """
    base = config or CIFARCNNConfig()

    def _build(**kwargs: Any) -> "CIFARCNN":
        cfg = base
        merged = {**overrides, **kwargs}
        if merged:
            cfg = base.with_overrides(**merged)
        return CIFARCNN(cfg)

    _build.config = base  # type: ignore[attr-defined]
    return _build


def cifar_inner_cnn(num_classes: int = CIFAR_NUM_CLASSES, **kwargs: Any) -> "CIFARCNN":
    """Inner-loop (proxy) CIFAR-10 CNN: simple CNN, Adam lr 0.001 (Section 5.2)."""
    return CIFARCNN(CIFARCNNConfig.for_inner(num_classes=num_classes, **kwargs))


def cifar_target_cnn(num_classes: int = CIFAR_NUM_CLASSES, **kwargs: Any) -> "CIFARCNN":
    """Wider CNN preset usable for training on the constructed coreset.

    Note: the paper specifies a ResNet-18 for CIFAR-10 target training
    (Section 5.2, Appendix D.2 / addendum: SGD lr 0.1, momentum 0.9, cosine
    scheduler, 200 epochs).  Use :mod:`lbcs_repro.models.resnet18` for the
    paper's reported CIFAR-10 numbers.
    """
    return CIFARCNN(CIFARCNNConfig.for_target(num_classes=num_classes, **kwargs))


def cifar_inner_cnn_factory(num_classes: int = CIFAR_NUM_CLASSES, **overrides: Any):
    """Factory for the inner-loop preset (fresh instance per call)."""
    return cifar_cnn_factory(CIFARCNNConfig.for_inner(num_classes=num_classes, **overrides))


def cifar_target_cnn_factory(num_classes: int = CIFAR_NUM_CLASSES, **overrides: Any):
    """Factory for the target preset (fresh instance per call)."""
    return cifar_cnn_factory(CIFARCNNConfig.for_target(num_classes=num_classes, **overrides))


#: Aliases kept for naming symmetry with the SVHN module.
CIFARInnerCNN = cifar_inner_cnn
CIFARCNNTarget = cifar_target_cnn
CIFARTargetCNN = cifar_target_cnn


def train_cifar_target(
    model: Any,
    loader: Any,
    epochs: int = SECTION52_TARGET_EPOCHS,
    lr: float = SECTION52_TARGET_LR,
    momentum: float = SECTION52_TARGET_MOMENTUM,
    weight_decay: float = SUGGESTED_WEIGHT_DECAY,
    scheduler: Optional[str] = SECTION52_TARGET_SCHEDULER,
    device: Any = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Any:
    """Train a CIFAR-10 target model per Section 5.2.

    SGD with initial lr 0.1, momentum 0.9, cosine rate scheduler, 200 epochs
    (paper-stated for CIFAR-10 target training; the paper uses ResNet-18).
    Returns the trained model.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to train the CIFAR-10 target model.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs
    )
    sched = None
    if scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(int(epochs)):
        total, seen = 0.0, 0
        for batch in loader:
            inputs, targets = batch[0], batch[1]
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * inputs.size(0)
            seen += inputs.size(0)
        if sched is not None:
            sched.step()
        if verbose:
            LOGGER.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, total / max(1, seen))
    return model


#: Name -> builder mapping for name-driven experiment drivers.
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "CIFARCNN": cifar_cnn_factory,
    "cifar_cnn": cifar_cnn_factory,
    "cifar": cifar_cnn_factory,
    "CIFARInnerCNN": cifar_inner_cnn_factory,
    "CIFARCNNTarget": cifar_target_cnn_factory,
    "CIFARTargetCNN": cifar_target_cnn_factory,
}


__all__ = [
    "CIFARCNN",
    "CIFARCNNConfig",
    "CIFAR_INPUT_SHAPE",
    "CIFAR_NUM_CLASSES",
    "CIFAR_IN_CHANNELS",
    "SECTION52_INNER_OPTIMIZER",
    "SECTION52_INNER_LR",
    "SECTION52_TARGET_OPTIMIZER",
    "SECTION52_TARGET_LR",
    "SECTION52_TARGET_MOMENTUM",
    "SECTION52_TARGET_EPOCHS",
    "SECTION52_TARGET_SCHEDULER",
    "SUGGESTED_CONV_CHANNELS",
    "SUGGESTED_CONV_CHANNELS_INNER",
    "SUGGESTED_CONV_CHANNELS_TARGET",
    "SUGGESTED_HIDDEN_DIMS",
    "SUGGESTED_HIDDEN_DIMS_TARGET",
    "SUGGESTED_DROPOUT",
    "SUGGESTED_BATCHNORM",
    "SUGGESTED_WEIGHT_DECAY",
    "build_cifar_cnn",
    "cifar_cnn",
    "cifar_cnn_factory",
    "cifar_inner_cnn",
    "cifar_target_cnn",
    "cifar_inner_cnn_factory",
    "cifar_target_cnn_factory",
    "CIFARInnerCNN",
    "CIFARCNNTarget",
    "CIFARTargetCNN",
    "train_cifar_target",
    "MODEL_REGISTRY",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    _selftest()
    LOGGER.info("cifar_cnn self-test passed.")
