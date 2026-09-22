"""SVHN convolutional networks for the LBCS reproduction.

Section 5.2 of the paper states:

    "In the procedure of coreset selection, we employ a LeNet for F-MNIST and
     simple convolutional neural networks (CNNs) for SVHN and CIFAR-10. An Adam
     optimizer is used with a learning rate of 0.001 for the inner loop. [...]
     After coreset selection, for training on the constructed coreset, we
     utilize a LeNet for F-MNIST, a CNN for SVHN, and a ResNet-18 network for
     CIFAR-10 respectively. In addition, for F-MNIST and SVHN, an Adam optimizer
     is used with a learning rate of 0.001 and 100 epochs."

So there are two distinct networks for SVHN:

* the *inner-loop* (coreset-selection) CNN -- the proxy model :math:`\\theta(m)`
  trained on the selected coreset, and
* the *target* CNN -- retrained from scratch on the constructed coreset after
  selection, whose test accuracy is the reported measurement.

Appendix D.2 says "We provide the detailed network structures of the used models
in our main paper, which can be checked in Table 7." The table itself is not
extractable from the PDF, therefore all layer widths below are exposed as
configuration fields and explicitly labelled **SUGGESTED** defaults, so the exact
Table 7 values can be supplied from YAML without touching any algorithm code.

Both networks consume the standard SVHN split (3 x 32 x 32 inputs, 10 classes),
normalised with the SVHN statistics used by ``lbcs_repro.data.datasets``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # torch is a soft dependency: mask-only unit tests must run without it
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when torch is missing
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# SUGGESTED (not paper-stated) architectural defaults.
# The paper only says "simple convolutional neural networks (CNNs) for SVHN".
# --------------------------------------------------------------------------- #
SUGGESTED_CONV_CHANNELS: Tuple[int, ...] = (32, 32, 64, 64)
SUGGESTED_KERNEL_SIZE = 3
SUGGESTED_STRIDE = 1
SUGGESTED_POOL_EVERY = 2
SUGGESTED_POOL_KERNEL = 2
SUGGESTED_DROPOUT = 0.25
SUGGESTED_HIDDEN_DIMS: Tuple[int, ...] = (256,)
SUGGESTED_HIDDEN_DIMS_TARGET: Tuple[int, ...] = (512, 256)
SUGGESTED_BATCHNORM = True

SVHN_INPUT_SHAPE: Tuple[int, int, int] = (3, 32, 32)
SVHN_NUM_CLASSES = 10

__all__ = [
    "SVHNCNNConfig",
    "SVHNCNN",
    "build_svhn_cnn",
    "svhn_cnn",
    "svhn_cnn_factory",
    "svhn_inner_cnn",
    "svhn_target_cnn",
    "SVHNInnerCNN",
    "SVHNCNNTarget",
    "SVHNTargetCNN",
    "MODEL_REGISTRY",
    "SUGGESTED_CONV_CHANNELS",
    "SUGGESTED_KERNEL_SIZE",
    "SUGGESTED_POOL_KERNEL",
    "SUGGESTED_POOL_EVERY",
    "SUGGESTED_DROPOUT",
    "SUGGESTED_HIDDEN_DIMS",
    "SUGGESTED_HIDDEN_DIMS_TARGET",
    "SUGGESTED_BATCHNORM",
    "SVHN_INPUT_SHAPE",
    "SVHN_NUM_CLASSES",
]

_CONFIG_FIELDS = (
    "in_channels",
    "num_classes",
    "conv_channels",
    "kernel_size",
    "stride",
    "padding",
    "pool_every",
    "pool_kernel",
    "pool_stride",
    "hidden_dims",
    "dropout",
    "batchnorm",
    "input_shape",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SVHNCNNConfig:
    """Static description of a simple SVHN CNN.

    Layer pattern (one conv per entry of ``conv_channels``, max-pool every
    ``pool_every`` convolutions, dropout after every conv block)::

        for i, out_c in enumerate(conv_channels):
            Conv -> [BatchNorm] -> ReLU
            if (i + 1) % pool_every == 0: MaxPool
            Dropout

        Flatten -> [Linear -> ReLU -> Dropout] * len(hidden_dims) -> Linear

    ``forward`` returns raw logits so the module plugs directly into
    ``torch.nn.CrossEntropyLoss`` for the full-data objective :math:`f_1(m)` and
    the inner coreset loss :math:`L(m, \\theta)`.

    All numeric values are **SUGGESTED** defaults: the paper refers to Table 7
    (Appendix D.2) which is not extractable, so every field is overridable via
    ``with_overrides`` / YAML.
    """

    in_channels: int = 3
    num_classes: int = SVHN_NUM_CLASSES
    conv_channels: Tuple[int, ...] = SUGGESTED_CONV_CHANNELS
    kernel_size: int = SUGGESTED_KERNEL_SIZE
    stride: int = SUGGESTED_STRIDE
    padding: Optional[int] = None  # None -> kernel_size // 2 ("same" padding)
    pool_every: int = SUGGESTED_POOL_EVERY
    pool_kernel: int = SUGGESTED_POOL_KERNEL
    pool_stride: Optional[int] = None  # None -> pool_kernel
    hidden_dims: Tuple[int, ...] = SUGGESTED_HIDDEN_DIMS
    dropout: float = SUGGESTED_DROPOUT
    batchnorm: bool = SUGGESTED_BATCHNORM
    input_shape: Tuple[int, int, int] = SVHN_INPUT_SHAPE

    def __post_init__(self) -> None:
        self.conv_channels = tuple(int(c) for c in self.conv_channels)
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        self.input_shape = tuple(int(v) for v in self.input_shape)
        if self.padding is None:
            self.padding = self.kernel_size // 2
        if self.pool_stride is None:
            self.pool_stride = self.pool_kernel
        if self.pool_every < 1:
            raise ValueError("pool_every must be >= 1")

    # ---------------------------------------------------------------- presets
    @classmethod
    def for_svhn(cls, num_classes: int = SVHN_NUM_CLASSES, **overrides: Any) -> "SVHNCNNConfig":
        """Inner-loop (coreset-selection) SVHN CNN preset."""
        base = cls(num_classes=num_classes)
        return base.with_overrides(**overrides)

    @classmethod
    def for_inner(cls, **overrides: Any) -> "SVHNCNNConfig":
        """Alias of :meth:`for_svhn` (proxy model :math:`\\theta(m)`)."""
        return cls.for_svhn(**overrides)

    @classmethod
    def for_target(cls, num_classes: int = SVHN_NUM_CLASSES, **overrides: Any) -> "SVHNCNNConfig":
        """Post-selection *target* CNN preset (retrained on the coreset).

        Slightly wider than the proxy network (SUGGESTED), because the target
        model is trained once on the fixed coreset with Adam, lr=0.001, 100
        epochs (Section 5.2).
        """
        base = cls(
            num_classes=num_classes,
            conv_channels=SUGGESTED_CONV_CHANNELS,
            hidden_dims=SUGGESTED_HIDDEN_DIMS_TARGET,
            dropout=SUGGESTED_DROPOUT,
            batchnorm=SUGGESTED_BATCHNORM,
        )
        return base.with_overrides(**overrides)

    @classmethod
    def cifar_variant(cls, num_classes: int = 10, **overrides: Any) -> "SVHNCNNConfig":
        """Same topology with 3x32x32 CIFAR-10-style inputs (convenience)."""
        base = cls(num_classes=num_classes, input_shape=(3, 32, 32))
        return base.with_overrides(**overrides)

    # ------------------------------------------------------------- utilities
    def with_overrides(self, **overrides: Any) -> "SVHNCNNConfig":
        unknown = set(overrides) - set(_CONFIG_FIELDS)
        if unknown:
            raise TypeError(f"Unknown SVHNCNNConfig fields: {sorted(unknown)}")
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["conv_channels"] = list(self.conv_channels)
        data["hidden_dims"] = list(self.hidden_dims)
        data["input_shape"] = list(self.input_shape)
        if "input_shape" in overrides_placeholder():  # pragma: no cover
            pass
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SVHNCNNConfig":
        if not data:
            return cls()
        allowed = {k: v for k, v in dict(data).items() if k in _CONFIG_FIELDS}
        return cls(**allowed)

    # -------------------------------------------------------------- geometry
    def num_conv_layers(self) -> int:
        return len(self.conv_channels)

    def num_pools(self) -> int:
        return len(self.conv_channels) // self.pool_every

    def _spatial_after_convs(self) -> int:
        size = int(self.input_shape[-1])
        for i in range(len(self.conv_channels)):
            size = (size + 2 * int(self.padding or 0) - self.kernel_size) // self.stride + 1
            if (i + 1) % self.pool_every == 0:
                size = (size - self.pool_kernel) // int(self.pool_stride or self.pool_kernel) + 1
        return max(int(size), 1)

    def conv_output_shape(self) -> Tuple[int, int, int]:
        """(C, H, W) of the tensor entering the classifier head."""
        return (self.conv_channels[-1], self._spatial_after_convs(), self._spatial_after_convs())

    def feature_dim(self) -> int:
        """Flattened penultimate feature dimension."""
        c, h, w = self.conv_output_shape()
        return int(c * h * w)

    def summary(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "conv_channels": list(self.conv_channels),
            "num_pools": self.num_pools(),
            "conv_output_shape": list(self.conv_output_shape()),
            "feature_dim": self.feature_dim(),
            "hidden_dims": list(self.hidden_dims),
            "dropout": self.dropout,
            "batchnorm": self.batchnorm,
            "input_shape": list(self.input_shape),
        }


def overrides_placeholder() -> Dict[str, Any]:  # pragma: no cover - trivial
    """Kept for structural symmetry with sibling model modules."""
    return {}


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
if _TORCH_AVAILABLE:

    class SVHNCNN(nn.Module):
        """Simple convolutional network for SVHN (inner-loop and/or target).

        Outputs *raw logits* of shape ``(N, num_classes)``.
        """

        def __init__(self, config: Optional[SVHNCNNConfig] = None, **kwargs: Any) -> None:
            super().__init__()
            if config is None:
                cfg = SVHNCNNConfig()
            elif isinstance(config, SVHNCNNConfig):
                cfg = config
            else:  # allow passing a plain dict / namespace
                cfg = SVHNCNNConfig.from_dict(dict(config))
            if kwargs:
                cfg = cfg.with_overrides(**kwargs)
            self.config = cfg

            convs = []
            in_ch = cfg.in_channels
            for i, out_ch in enumerate(cfg.conv_channels):
                convs.append(
                    nn.Conv2d(
                        in_ch,
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=int(cfg.padding or 0),
                        bias=not cfg.batchnorm,
                    )
                )
                if cfg.batchnorm:
                    convs.append(nn.BatchNorm2d(out_ch))
                convs.append(nn.ReLU(inplace=True))
                if (i + 1) % cfg.pool_every == 0:
                    convs.append(nn.MaxPool2d(kernel_size=cfg.pool_kernel, stride=cfg.pool_stride))
                if cfg.dropout and cfg.dropout > 0:
                    convs.append(nn.Dropout(cfg.dropout))
                in_ch = out_ch
            self.features_net = nn.Sequential(*convs)

            head = []
            prev = cfg.feature_dim()
            for h in cfg.hidden_dims:
                head.append(nn.Linear(prev, h))
                head.append(nn.ReLU(inplace=True))
                if cfg.dropout and cfg.dropout > 0:
                    head.append(nn.Dropout(cfg.dropout))
                prev = h
            head.append(nn.Linear(prev, cfg.num_classes))
            self.classifier = nn.Sequential(*head)

            self._init_weights()

        # ------------------------------------------------------------ forward
        def features(self, x: Any) -> "torch.Tensor":
            """Flattened penultimate features ``(N, feature_dim)``.

            Used by distance-based baselines (Moderate / CCS) for the SVHN
            inner-loop network.
            """
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return self.features_net(x).flatten(1)

        def forward(self, x: Any) -> "torch.Tensor":
            if not isinstance(x, torch.Tensor):
                x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
            return self.classifier(self.features(x))

        # ----------------------------------------------------------- helpers
        def num_parameters(self, trainable_only: bool = True) -> int:
            params = self.parameters()
            if trainable_only:
                return int(sum(p.numel() for p in params if p.requires_grad))
            return int(sum(p.numel() for p in params))

        def _init_weights(self) -> None:  # deterministic init
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

else:  # pragma: no cover - torch-less environments

    class SVHNCNN:  # type: ignore
        """Placeholder raising ``ImportError`` when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required to instantiate SVHNCNN. Install torch/torchvision."
            )


# --------------------------------------------------------------------------- #
# Factories (LBCS needs a *fresh* theta(m) for every mask evaluation)
# --------------------------------------------------------------------------- #
def build_svhn_cnn(
    num_classes: int = SVHN_NUM_CLASSES,
    in_channels: int = 3,
    input_shape: Optional[Sequence[int]] = None,
    config: Optional[SVHNCNNConfig] = None,
    **kwargs: Any,
) -> "SVHNCNN":
    """Instantiate an SVHN CNN from a config / overrides."""
    cfg = config if isinstance(config, SVHNCNNConfig) else SVHNCNNConfig.from_dict(config)
    overrides: Dict[str, Any] = {"num_classes": num_classes, "in_channels": in_channels}
    if input_shape is not None:
        overrides["input_shape"] = tuple(int(v) for v in input_shape)
    overrides.update(kwargs)
    cfg = cfg.with_overrides(**overrides)
    return SVHNCNN(cfg)


svhn_cnn = build_svhn_cnn


def svhn_cnn_factory(
    config: Optional[SVHNCNNConfig] = None, **overrides: Any
) -> Callable[..., "SVHNCNN"]:
    """Return a builder producing fresh :class:`SVHNCNN` instances.

    The signature expected by ``lbcs.bilevel.LBCS`` (a new network per inner-loop
    run). The returned callable carries a ``.config`` attribute.
    """
    base_cfg = config if isinstance(config, SVHNCNNConfig) else SVHNCNNConfig.from_dict(config)
    if overrides:
        base_cfg = base_cfg.with_overrides(**overrides)

    def _build(**kwargs: Any) -> "SVHNCNN":
        cfg = base_cfg
        if kwargs:
            cfg = base_cfg.with_overrides(**kwargs)
        return SVHNCNN(cfg)

    _build.config = base_cfg  # type: ignore[attr-defined]
    _build.__name__ = "svhn_cnn_factory"
    return _build


def svhn_inner_cnn(num_classes: int = SVHN_NUM_CLASSES, **kwargs: Any) -> "SVHNCNN":
    """Inner-loop (coreset-selection) SVHN CNN, Section 5.2."""
    cfg = SVHNCNNConfig.for_inner(num_classes=num_classes, **kwargs)
    return SVHNCNN(cfg)


def svhn_target_cnn(num_classes: int = SVHN_NUM_CLASSES, **kwargs: Any) -> "SVHNCNN":
    """Post-selection target SVHN CNN, Section 5.2 / Appendix D.2."""
    cfg = SVHNCNNConfig.for_target(num_classes=num_classes, **kwargs)
    return SVHNCNN(cfg)


# Convenience aliases used by experiment drivers / configs.
SVHNInnerCNN = svhn_inner_cnn
SVHNCNNTarget = svhn_target_cnn
SVHNTargetCNN = svhn_target_cnn


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "SVHNCNN": svhn_cnn_factory,
    "svhn_cnn": svhn_cnn_factory,
    "svhn": svhn_cnn_factory,
    "SVHNInnerCNN": svhn_cnn_factory,
    "SVHNCNNTarget": svhn_cnn_factory,
    "SVHNTargetCNN": svhn_cnn_factory,
}


# --------------------------------------------------------------------------- #
# Offline self-test
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Structural / forward / backward checks (no dataset required)."""
    results: Dict[str, Any] = {}
    cfg = SVHNCNNConfig.for_svhn()
    results["feature_dim"] = cfg.feature_dim()
    results["conv_output_shape"] = list(cfg.conv_output_shape())
    results["num_parameters_cfg"] = cfg.summary()

    assert cfg.feature_dim() == 64 * 8 * 8, cfg.feature_dim()
    assert cfg.num_pools() == 2

    if not _TORCH_AVAILABLE:
        results["torch"] = False
        if verbose:
            print("[svhn_cnn] torch unavailable -- geometry checks only", results)
        return results

    torch.manual_seed(0)
    model = svhn_inner_cnn()
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)
    assert tuple(logits.shape) == (4, SVHN_NUM_CLASSES), logits.shape
    feats = model.features(x)
    assert tuple(feats.shape) == (4, cfg.feature_dim()), feats.shape

    loss = torch.nn.functional.cross_entropy(logits, torch.randint(0, SVHN_NUM_CLASSES, (4,)))
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    assert all(grads), "some parameters received no gradient"

    n_params = model.num_parameters()

    # Target preset must also build and forward.
    target = svhn_target_cnn()
    assert tuple(target(x).shape) == (4, SVHN_NUM_CLASSES)

    # Factory contract used by lbcs.bilevel.LBCS.
    factory = svhn_cnn_factory()
    m1, m2 = factory(), factory()
    assert m1 is not m2, "factory must return fresh instances"
    assert hasattr(factory, "config")

    results.update(
        {
            "torch": True,
            "logits_shape": list(logits.shape),
            "features_shape": list(feats.shape),
            "num_parameters": n_params,
            "num_parameters_target": target.num_parameters(),
            "loss": float(loss.detach()),
        }
    )
    if verbose:
        print("[svhn_cnn] selftest OK:", results)
    return results


if __name__ == "__main__":  # pragma: no cover
    _selftest(verbose=True)
