"""ConvNet used in Figure 1 / Section 5.1 of the LBCS paper.

Paper specification
-------------------
Section 5.1 (verbatim):

    "Staying with previous work (Borsos et al., 2020), we use a convolutional
     neural network stacked with two blocks of convolution, dropout,
     max-pooling, and ReLU activation."

Appendix C.3 (verbatim):

    "For the experiments in Figure 1, we employ a subset of MNIST.  A
     convolutional neural network stacked with two blocks of convolution,
     dropout, max-pooling, and ReLU activation is used.  Following (Zhou et
     al., 2022), for the inner loop, the model is trained for 100 epochs using
     SGD with a learning rate of 0.1 and momentum of 0.9.  For the outer loop,
     the probabilities are optimized by Adam with a learning rate of 2.5 and a
     cosine scheduler."

This is the ``ConvNet`` class of Zhou et al. (ICML 2022), *Probabilistic
Bilevel Coreset Selection in One Forward Pass*, which the paper follows
explicitly, so the topology is:

    block 1: Conv2d -> ReLU -> Dropout -> MaxPool2d
    block 2: Conv2d -> ReLU -> Dropout -> MaxPool2d
    head   : Flatten -> Linear -> ReLU -> Dropout -> Linear

Appendix D.2 only points at Table 7 ("We provide the detailed network
structures of the used models ... which can be checked in Table 7"); the
per-layer widths in that table are not extractable from the PDF text, so they
are exposed as configuration (:class:`ConvNetConfig`) with SUGGESTED defaults
that match the reference implementation's MNIST setting (channel progression
16/32, 3x3 kernels, single hidden layer).  Every such choice is marked
SUGGESTED and can be overridden from a YAML config without touching code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # soft dependency: mask-only unit tests run without torch
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


__all__ = [
    "ConvNetConfig",
    "ConvNet",
    "convnet_factory",
    "build_convnet",
    "mnist_convnet",
    "MODEL_REGISTRY",
    "SUGGESTED_CONV_CHANNELS",
    "SUGGESTED_DROPOUT",
]


# ---------------------------------------------------------------------------
# SUGGESTED defaults (NOT paper-stated; see module docstring)
# ---------------------------------------------------------------------------
SUGGESTED_CONV_CHANNELS: Tuple[int, ...] = (16, 32)
SUGGESTED_KERNEL_SIZE: int = 3
SUGGESTED_STRIDE: int = 1
SUGGESTED_PADDING: int = 1
SUGGESTED_POOL_KERNEL: int = 2
SUGGESTED_DROPOUT: float = 0.25
SUGGESTED_HIDDEN: int = 128

_CONFIG_FIELDS = {
    "in_channels",
    "num_classes",
    "conv_channels",
    "kernel_size",
    "stride",
    "padding",
    "pool_kernel",
    "pool_stride",
    "dropout",
    "hidden_dim",
    "input_shape",
    "batchnorm",
}


@dataclass
class ConvNetConfig:
    """Static description of the two-block ConvNet.

    Parameters
    ----------
    in_channels, num_classes:
        Input/output sizes (MNIST and MNIST-S use ``1`` / ``10``).
    conv_channels:
        One entry per convolution block (the paper specifies exactly two).
    kernel_size, stride, padding, pool_kernel, pool_stride:
        Convolution / max-pooling hyper-parameters.
    dropout:
        Rate used after every convolution block and after the hidden ReLU.
    hidden_dim:
        Width of the fully connected hidden layer.
    input_shape:
        Spatial input shape ``(C, H, W)``, used to size the classifier head.
    batchnorm:
        Optional BatchNorm after each convolution.  Off by default because the
        paper lists *only* convolution, dropout, max-pooling and ReLU.
    """

    in_channels: int = 1
    num_classes: int = 10
    conv_channels: Sequence[int] = field(default_factory=lambda: tuple(SUGGESTED_CONV_CHANNELS))
    kernel_size: int = SUGGESTED_KERNEL_SIZE
    stride: int = SUGGESTED_STRIDE
    padding: int = SUGGESTED_PADDING
    pool_kernel: int = SUGGESTED_POOL_KERNEL
    pool_stride: Optional[int] = None
    dropout: float = SUGGESTED_DROPOUT
    hidden_dim: int = SUGGESTED_HIDDEN
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    batchnorm: bool = False

    # ------------------------------------------------------------------
    @classmethod
    def for_mnist(cls, num_classes: int = 10, in_channels: int = 1, **overrides: Any) -> "ConvNetConfig":
        """Config for the MNIST / MNIST-S Figure 1 and Section 5.1 setup."""
        cfg = cls(in_channels=in_channels, num_classes=num_classes, input_shape=(in_channels, 28, 28))
        return cfg.with_overrides(**overrides) if overrides else cfg

    @classmethod
    def for_fashion_mnist(cls, num_classes: int = 10, **overrides: Any) -> "ConvNetConfig":
        return cls.for_mnist(num_classes=num_classes, in_channels=1, **overrides)

    def with_overrides(self, **overrides: Any) -> "ConvNetConfig":
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if k in _CONFIG_FIELDS and v is not None})
        return ConvNetConfig(
            in_channels=int(data["in_channels"]),
            num_classes=int(data["num_classes"]),
            conv_channels=tuple(data["conv_channels"]),
            kernel_size=int(data["kernel_size"]),
            stride=int(data["stride"]),
            padding=int(data["padding"]),
            pool_kernel=int(data["pool_kernel"]),
            pool_stride=data.get("pool_stride"),
            dropout=float(data["dropout"]),
            hidden_dim=int(data["hidden_dim"]),
            input_shape=tuple(data.get("input_shape", (data["in_channels"], 28, 28))),
            batchnorm=bool(data.get("batchnorm", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "conv_channels": list(self.conv_channels),
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "pool_kernel": self.pool_kernel,
            "pool_stride": self.pool_stride,
            "dropout": self.dropout,
            "hidden_dim": self.hidden_dim,
            "input_shape": tuple(self.input_shape),
            "batchnorm": self.batchnorm,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ConvNetConfig":
        if not data:
            return cls()
        data = {k: v for k, v in dict(data).items() if k in _CONFIG_FIELDS}
        return cls(
            in_channels=int(data.get("in_channels", 1)),
            num_classes=int(data.get("num_classes", 10)),
            conv_channels=tuple(data.get("conv_channels", SUGGESTED_CONV_CHANNELS)),
            kernel_size=int(data.get("kernel_size", SUGGESTED_KERNEL_SIZE)),
            stride=int(data.get("stride", SUGGESTED_STRIDE)),
            padding=int(data.get("padding", SUGGESTED_PADDING)),
            pool_kernel=int(data.get("pool_kernel", SUGGESTED_POOL_KERNEL)),
            pool_stride=data.get("pool_stride"),
            dropout=float(data.get("dropout", SUGGESTED_DROPOUT)),
            hidden_dim=int(data.get("hidden_dim", SUGGESTED_HIDDEN)),
            input_shape=tuple(data.get("input_shape", (1, 28, 28))),
            batchnorm=bool(data.get("batchnorm", False)),
        )

    def num_blocks(self) -> int:
        """Number of convolution blocks (paper: two)."""
        return len(self.conv_channels)

    def feature_dim(self) -> int:
        """Flattened size of the final convolutional feature map."""
        _, h, w = self.input_shape
        for _ in self.conv_channels:
            h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
            w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
            ps = self.pool_stride or self.pool_kernel
            h = (h - self.pool_kernel) // ps + 1
            w = (w - self.pool_kernel) // ps + 1
        return int(self.conv_channels[-1] * h * w)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:

    class ConvNet(nn.Module):
        """Two blocks of {conv, ReLU, dropout, max-pool} followed by an MLP head.

        Matches Section 5.1 / Appendix C.3 of the LBCS paper and the ``ConvNet``
        of Zhou et al. (2022) that the paper follows.

        ``forward`` returns raw logits of shape ``(N, num_classes)`` so the
        module plugs straight into ``nn.CrossEntropyLoss`` and into the
        objective functions of :mod:`lbcs_repro.lbcs.objectives`.  The flattened
        penultimate representation is available via :meth:`features` for the
        distance-based baselines (Moderate, CCS).
        """

        def __init__(self, config: Optional[ConvNetConfig] = None, **kwargs: Any) -> None:
            super().__init__()
            if config is None:
                config = ConvNetConfig(**{k: v for k, v in kwargs.items() if k in _CONFIG_FIELDS})
            self.config = config
            cfg = self.config

            channels = [cfg.in_channels] + list(cfg.conv_channels)
            blocks: List[nn.Module] = []
            for i in range(len(cfg.conv_channels)):
                # One "block of convolution, dropout, max-pooling, ReLU".
                block: List[nn.Module] = [
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        kernel_size=cfg.kernel_size,
                        stride=cfg.stride,
                        padding=cfg.padding,
                    )
                ]
                if cfg.batchnorm:
                    block.append(nn.BatchNorm2d(channels[i + 1]))
                block.append(nn.ReLU(inplace=True))
                if cfg.dropout and cfg.dropout > 0:
                    block.append(nn.Dropout(cfg.dropout))
                block.append(nn.MaxPool2d(kernel_size=cfg.pool_kernel, stride=cfg.pool_stride or cfg.pool_kernel))
                blocks.append(nn.Sequential(*block))
            self.blocks = nn.ModuleList(blocks)

            self.feature_dim = cfg.feature_dim()
            head: List[nn.Module] = [
                nn.Flatten(),
                nn.Linear(self.feature_dim, cfg.hidden_dim),
                nn.ReLU(inplace=True),
            ]
            if cfg.dropout and cfg.dropout > 0:
                head.append(nn.Dropout(cfg.dropout))
            head.append(nn.Linear(cfg.hidden_dim, cfg.num_classes))
            self.classifier = nn.Sequential(*head)

            self._init_weights()

        # --------------------------------------------------------------
        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # --------------------------------------------------------------
        def features(self, x: "torch.Tensor") -> "torch.Tensor":
            """Flattened penultimate features (for distance-based baselines)."""
            for block in self.blocks:
                x = block(x)
            return torch.flatten(x, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.features(x)
            return self.classifier(x)

        # --------------------------------------------------------------
        def num_parameters(self, trainable_only: bool = True) -> int:
            params = self.parameters()
            if trainable_only:
                params = (p for p in params if p.requires_grad)
            return int(sum(p.numel() for p in params))

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return f"ConvNet({self.config.to_dict()})"

else:  # pragma: no cover - torch missing

    class ConvNet:  # type: ignore
        """Placeholder raising a clear error when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("ConvNet requires PyTorch; install torch to use the model zoo.")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def convnet_factory(config: Optional[ConvNetConfig] = None, **overrides: Any):
    """Return a callable building fresh ``ConvNet`` instances.

    This is the form consumed by :class:`lbcs_repro.lbcs.bilevel.LBCS`, which
    needs a new ``theta(m)`` for every inner-loop training run.
    """
    cfg = ConvNetConfig.from_dict(config.to_dict()) if isinstance(config, ConvNetConfig) else ConvNetConfig.from_dict(config)
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    def _build(**kwargs: Any):
        return ConvNet(cfg.with_overrides(**kwargs) if kwargs else cfg)

    _build.config = cfg  # type: ignore[attr-defined]
    _build.__name__ = "convnet_factory"
    return _build


def build_convnet(
    num_classes: int = 10,
    in_channels: int = 1,
    input_shape: Optional[Tuple[int, int, int]] = None,
    **kwargs: Any,
) -> "ConvNet":
    """Instantiate a ``ConvNet`` directly (Figure 1 / Section 5.1 default)."""
    config = ConvNetConfig(
        in_channels=in_channels,
        num_classes=num_classes,
        input_shape=input_shape or (in_channels, 28, 28),
        **{k: v for k, v in kwargs.items() if k in _CONFIG_FIELDS},
    )
    return ConvNet(config)


def mnist_convnet(num_classes: int = 10, **kwargs: Any) -> "ConvNet":
    """The MNIST / MNIST-S two-block ConvNet of Section 5.1 and Figure 1."""
    return build_convnet(num_classes=num_classes, in_channels=1, input_shape=(1, 28, 28), **kwargs)


MODEL_REGISTRY: Dict[str, Any] = {"ConvNet": convnet_factory, "convnet": convnet_factory}


# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: two conv blocks, two max-pools, dropout, forward/backward."""
    info: Dict[str, Any] = {"torch": _TORCH_AVAILABLE}
    if not _TORCH_AVAILABLE:
        if verbose:
            print("[convnet] torch unavailable; skipping forward-pass checks")
        return info

    model = mnist_convnet()
    n_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
    n_pool = sum(1 for m in model.modules() if isinstance(m, nn.MaxPool2d))
    n_drop = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    info.update(num_conv=n_conv, num_pool=n_pool, num_dropout=n_drop, params=model.num_parameters())

    assert n_conv == 2, f"expected two convolution blocks, got {n_conv}"
    assert n_pool == 2, f"expected two max-pooling blocks, got {n_pool}"
    assert n_drop >= 2, f"expected dropout inside the blocks, got {n_drop}"

    model.eval()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        logits = model(x)
        feats = model.features(x)
    info["logits_shape"] = tuple(logits.shape)
    info["feature_shape"] = tuple(feats.shape)
    assert tuple(logits.shape) == (4, 10), info["logits_shape"]
    assert feats.shape[1] == model.feature_dim

    model.train()
    loss = F.cross_entropy(model(x), torch.randint(0, 10, (4,)))
    loss.backward()
    info["all_params_get_grad"] = bool(all(p.grad is not None for p in model.parameters()))

    if verbose:
        print(f"[convnet] selftest passed: {info}")
    return info


if __name__ == "__main__":  # pragma: no cover
    _selftest()
