"""LeNet for Fashion-MNIST (F-MNIST).

The paper (Section 5.2 "Datasets and implementation") states that in the procedure of
coreset selection a *LeNet* is employed for F-MNIST, and that after coreset selection
"we utilize a LeNet (LeCun et al., 1998) for F-MNIST" for training on the constructed
coreset.  Appendix D.2 points at Table 7 for the exact layer widths, which are not
extractable from the provided paper text.

This module therefore implements the canonical LeNet family:

    conv -> ReLU (-> optional dropout) -> max-pool      (x2)
    flatten -> FC -> ReLU -> FC(logits)

with two presets exposed through :class:`LeNetConfig`:

* ``LeNetConfig.for_fashion_mnist()`` -- the two-conv LeNet used by the paper's proxy
  (inner loop) and post-selection target model.  Widths follow the widely used
  F-MNIST LeNet (``conv 1->32->64`` with ``fc 128``) and are labelled SUGGESTED because
  Table 7 is not readable from the paper text.
* ``LeNetConfig.lenet5()`` -- the classical LeNet-5 layout of LeCun et al. (1998)
  (``conv 1->6->16``, ``fc 120 -> 84``) for reference / ablation.

All widths are configurable so the exact Table 7 values can be supplied from YAML
without editing algorithm code.

``torch`` is a *soft* dependency: mask-only unit tests keep working without PyTorch
installed, while model construction raises a clear error if torch is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # soft dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - numpy-only environments
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False

__all__ = [
    "LeNetConfig",
    "LeNet",
    "build_lenet",
    "lenet",
    "lenet_factory",
    "fashion_mnist_lenet",
    "lenet5",
    "MODEL_REGISTRY",
    "SUGGESTED_CONV_CHANNELS",
    "SUGGESTED_KERNEL_SIZE",
    "SUGGESTED_HIDDEN_DIMS",
    "SUGGESTED_DROPOUT",
]

# ---------------------------------------------------------------------------
# SUGGESTED defaults (NOT paper-stated: Appendix D.2 delegates to Table 7)
# ---------------------------------------------------------------------------
SUGGESTED_CONV_CHANNELS: Tuple[int, ...] = (32, 64)
SUGGESTED_KERNEL_SIZE: int = 3
SUGGESTED_HIDDEN_DIMS: Tuple[int, ...] = (128,)
SUGGESTED_DROPOUT: float = 0.0
LENET5_CONV_CHANNELS: Tuple[int, ...] = (6, 16)
LENET5_KERNEL_SIZE: int = 5
LENET5_HIDDEN_DIMS: Tuple[int, ...] = (120, 84)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LeNetConfig:
    """Static description of a LeNet-style network.

    Attributes
    ----------
    in_channels:
        Input image channels (1 for MNIST / F-MNIST).
    num_classes:
        Number of output logits.
    conv_channels:
        Output channels of the successive convolution blocks.
    kernel_size / padding:
        Convolution kernel and padding; padding defaults to ``kernel_size // 2`` so the
        spatial size is preserved until pooling.
    pool_kernel / pool_stride:
        Max-pooling kernel (default 2) and stride (defaults to ``pool_kernel``).
    hidden_dims:
        Sizes of the fully connected hidden layers (appended before the classifier).
    dropout:
        Dropout probability applied after each conv block and after hidden FC layers
        (0.0 disables it).
    input_shape:
        ``(C, H, W)`` of the inputs; used to compute the flattened feature dimension.
    classic:
        When True the activation is ``tanh`` (classical LeNet-5) instead of ReLU.
    bias:
        Whether convolution/linear layers use a bias term.
    """

    in_channels: int = 1
    num_classes: int = 10
    conv_channels: Tuple[int, ...] = SUGGESTED_CONV_CHANNELS
    kernel_size: int = SUGGESTED_KERNEL_SIZE
    padding: Optional[int] = None
    stride: int = 1
    pool_kernel: int = 2
    pool_stride: Optional[int] = None
    hidden_dims: Tuple[int, ...] = SUGGESTED_HIDDEN_DIMS
    dropout: float = SUGGESTED_DROPOUT
    input_shape: Tuple[int, int, int] = (1, 28, 28)
    classic: bool = False
    bias: bool = True

    # -- constructors -------------------------------------------------------
    def __post_init__(self) -> None:
        self.conv_channels = tuple(int(c) for c in self.conv_channels)
        self.hidden_dims = tuple(int(h) for h in self.hidden_dims)
        self.input_shape = tuple(int(s) for s in self.input_shape)
        if self.padding is None:
            self.padding = self.kernel_size // 2

    @classmethod
    def for_fashion_mnist(cls, num_classes: int = 10, **overrides: Any) -> "LeNetConfig":
        """LeNet used for F-MNIST selection (inner loop) and target training (SUGGESTED)."""
        cfg = cls(
            in_channels=1,
            num_classes=int(num_classes),
            conv_channels=SUGGESTED_CONV_CHANNELS,
            kernel_size=SUGGESTED_KERNEL_SIZE,
            hidden_dims=SUGGESTED_HIDDEN_DIMS,
            dropout=SUGGESTED_DROPOUT,
            input_shape=(1, 28, 28),
        )
        return cfg.with_overrides(**overrides)

    # paper-symbol alias
    for_fmnist = for_fashion_mnist

    @classmethod
    def lenet5(cls, num_classes: int = 10, **overrides: Any) -> "LeNetConfig":
        """Classical LeNet-5 (LeCun et al., 1998): 6/16 convs, 120/84 FCs, tanh."""
        cfg = cls(
            in_channels=1,
            num_classes=int(num_classes),
            conv_channels=LENET5_CONV_CHANNELS,
            kernel_size=LENET5_KERNEL_SIZE,
            hidden_dims=LENET5_HIDDEN_DIMS,
            dropout=0.0,
            input_shape=(1, 28, 28),
            classic=True,
        )
        return cfg.with_overrides(**overrides)

    def with_overrides(self, **overrides: Any) -> "LeNetConfig":
        """Return a copy with the given fields replaced (unknown keys raise)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        for key in clean:
            if key not in self.__dataclass_fields__:  # type: ignore[attr-defined]
                raise TypeError(f"LeNetConfig has no field {key!r}")
        return replace(self, **clean)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view (YAML friendly)."""
        return {
            "in_channels": int(self.in_channels),
            "num_classes": int(self.num_classes),
            "conv_channels": list(self.conv_channels),
            "kernel_size": int(self.kernel_size),
            "padding": int(self.padding if self.padding is not None else 0),
            "stride": int(self.stride),
            "pool_kernel": int(self.pool_kernel),
            "pool_stride": int(self.pool_stride or self.pool_kernel),
            "hidden_dims": list(self.hidden_dims),
            "dropout": float(self.dropout),
            "input_shape": list(self.input_shape),
            "classic": bool(self.classic),
            "bias": bool(self.bias),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LeNetConfig":
        """Rebuild from :meth:`to_dict` output (or a ``configs/*.yaml`` block)."""
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(data).items() if k in known}
        if payload.get("conv_channels") is not None:
            payload["conv_channels"] = tuple(payload["conv_channels"])
        if payload.get("hidden_dims") is not None:
            payload["hidden_dims"] = tuple(payload["hidden_dims"])
        if payload.get("input_shape") is not None:
            payload["input_shape"] = tuple(payload["input_shape"])
        return cls(**payload)

    # -- geometry -----------------------------------------------------------
    def num_blocks(self) -> int:
        """Number of convolution blocks."""
        return len(self.conv_channels)

    def conv_output_shape(self) -> Tuple[int, int]:
        """Spatial size ``(H, W)`` after the convolution/pooling stack."""
        h, w = int(self.input_shape[1]), int(self.input_shape[2])
        pool_stride = int(self.pool_stride or self.pool_kernel)
        pad = int(self.padding if self.padding is not None else 0)
        for _ in self.conv_channels:
            h = (h + 2 * pad - self.kernel_size) // self.stride + 1
            w = (w + 2 * pad - self.kernel_size) // self.stride + 1
            h = (h - self.pool_kernel) // pool_stride + 1
            w = (w - self.pool_kernel) // pool_stride + 1
        return int(h), int(w)

    def feature_dim(self) -> int:
        """Flattened dimension entering the first fully connected layer."""
        h, w = self.conv_output_shape()
        return int(self.conv_channels[-1]) * h * w


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _LeNetStub:
    """Placeholder base when torch is unavailable (keeps import-time API stable)."""


if _TORCH_AVAILABLE:

    class LeNet(nn.Module):  # type: ignore[misc]
        """LeNet for F-MNIST coreset selection and target training.

        ``forward`` returns raw logits of shape ``(N, num_classes)`` so the module plugs
        directly into ``nn.CrossEntropyLoss`` for ``f1(m)`` / ``L(m, theta)``.
        """

        def __init__(self, config: Optional[LeNetConfig] = None, **kwargs: Any) -> None:
            super().__init__()
            if config is None:
                config = LeNetConfig(**kwargs) if kwargs else LeNetConfig.for_fashion_mnist()
            elif kwargs:
                config = config.with_overrides(**kwargs)
            self.config = config

            convs = []
            in_ch = int(config.in_channels)
            for out_ch in config.conv_channels:
                convs.append(
                    nn.Conv2d(
                        in_ch,
                        int(out_ch),
                        kernel_size=int(config.kernel_size),
                        stride=int(config.stride),
                        padding=int(config.padding if config.padding is not None else 0),
                        bias=bool(config.bias),
                    )
                )
                convs.append(nn.Tanh() if config.classic else nn.ReLU(inplace=True))
                if float(config.dropout) > 0:
                    convs.append(nn.Dropout(float(config.dropout)))
                convs.append(
                    nn.MaxPool2d(
                        kernel_size=int(config.pool_kernel),
                        stride=int(config.pool_stride or config.pool_kernel),
                    )
                )
                in_ch = int(out_ch)
            self.features_module = nn.Sequential(*convs)

            dim = config.feature_dim()
            fcs = []
            for hid in config.hidden_dims:
                fcs.append(nn.Linear(dim, int(hid), bias=bool(config.bias)))
                fcs.append(nn.Tanh() if config.classic else nn.ReLU(inplace=True))
                if float(config.dropout) > 0:
                    fcs.append(nn.Dropout(float(config.dropout)))
                dim = int(hid)
            self.classifier = nn.Sequential(*fcs) if fcs else nn.Identity()
            self.fc_out = nn.Linear(dim, int(config.num_classes), bias=bool(config.bias))

            self._init_weights()

        # -- weight init ----------------------------------------------------
        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        # -- forward --------------------------------------------------------
        def forward(self, x: Any) -> Any:
            if isinstance(x, np.ndarray):
                x = torch.as_tensor(x, dtype=torch.float32)
            h = self.features_module(x)
            h = torch.flatten(h, start_dim=1)
            h = self.classifier(h)
            return self.fc_out(h)

        # -- feature extraction (Moderate / CCS baselines) ------------------
        def features(self, x: Any) -> Any:
            """Flattened penultimate representation used for distance-based scores."""
            if isinstance(x, np.ndarray):
                x = torch.as_tensor(x, dtype=torch.float32)
            h = self.features_module(x)
            h = torch.flatten(h, start_dim=1)
            return self.classifier(h)

        def num_parameters(self, trainable_only: bool = True) -> int:
            params = [p for p in self.parameters() if (p.requires_grad or not trainable_only)]
            return int(sum(int(p.numel()) for p in params))

        def __repr__(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"LeNet(conv={tuple(self.config.conv_channels)}, "
                f"fc={tuple(self.config.hidden_dims)}, classes={self.config.num_classes}, "
                f"params={self.num_parameters()})"
            )

else:  # pragma: no cover - numpy-only environments

    class LeNet(_LeNetStub):  # type: ignore[misc]
        """Stub that raises a clear error when PyTorch is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required to instantiate LeNet")


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_lenet(
    num_classes: int = 10,
    in_channels: int = 1,
    input_shape: Optional[Sequence[int]] = None,
    config: Optional[LeNetConfig] = None,
    **kwargs: Any,
) -> "LeNet":
    """Instantiate a LeNet directly.

    Parameters mirror the other model modules so generic drivers can construct any
    architecture from ``(num_classes, in_channels, input_shape)``.
    """
    if config is None:
        config = LeNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            input_shape=(
                tuple(int(s) for s in input_shape)
                if input_shape is not None
                else (int(in_channels), 28, 28)
            ),
        )
    overrides = dict(kwargs)
    for key in ("conv_channels", "hidden_dims", "input_shape"):
        if overrides.get(key) is not None:
            overrides[key] = tuple(overrides[key])
    if overrides:
        config = config.with_overrides(**overrides)
    return LeNet(config)


# short alias
lenet = build_lenet


def lenet_factory(config: Optional[LeNetConfig] = None, **overrides: Any) -> Callable[..., "LeNet"]:
    """Return a callable building *fresh* LeNet instances.

    ``lbcs.bilevel.LBCS`` needs a new ``theta(m)`` for every mask evaluation, so the
    factory signature mirrors ``models.convnet.convnet_factory``.
    """
    base = config if config is not None else LeNetConfig.for_fashion_mnist()
    if overrides:
        base = base.with_overrides(**overrides)

    def _build(**kwargs: Any) -> "LeNet":
        cfg = base
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in ("conv_channels", "hidden_dims", "input_shape"):
                value = tuple(value)
            cfg = cfg.with_overrides(**{key: value})
        return LeNet(cfg)

    _build.config = base  # type: ignore[attr-defined]
    return _build


def fashion_mnist_lenet(num_classes: int = 10, **kwargs: Any) -> "LeNet":
    """LeNet configured for F-MNIST (§5.2 proxy model and post-selection target model)."""
    return LeNet(LeNetConfig.for_fashion_mnist(num_classes=num_classes, **kwargs))


def lenet5(num_classes: int = 10, **kwargs: Any) -> "LeNet":
    """Classical LeNet-5 (LeCun et al., 1998) instance."""
    return LeNet(LeNetConfig.lenet5(num_classes=num_classes, **kwargs))


MODEL_REGISTRY: Dict[str, Callable[..., "LeNet"]] = {
    "LeNet": lenet_factory,
    "lenet": lenet_factory,
    "LeNet5": lenet_factory,
}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline structural / forward / backward checks (``python -m ...lenet``)."""
    report: Dict[str, Any] = {"torch_available": _TORCH_AVAILABLE}

    cfg = LeNetConfig.for_fashion_mnist()
    report["conv_output_shape"] = tuple(cfg.conv_output_shape())
    report["feature_dim"] = int(cfg.feature_dim())
    assert cfg.feature_dim() == 64 * 7 * 7, report["feature_dim"]
    assert cfg.padding == 1

    cfg5 = LeNetConfig.lenet5()
    report["lenet5_feature_dim"] = int(cfg5.feature_dim())
    assert cfg5.feature_dim() == 16 * 5 * 5, report["lenet5_feature_dim"]

    if not _TORCH_AVAILABLE:
        report["status"] = "numpy-only"
        if verbose:
            print("[lenet selftest] torch unavailable; config checks passed.")
        return report

    for name, config in (("fmnist", cfg), ("lenet5", cfg5)):
        model = LeNet(config)
        model.eval()
        x = torch.randn(4, config.in_channels, *config.input_shape[1:])
        with torch.no_grad():
            logits = model(x)
            feats = model.features(x)
        assert tuple(logits.shape) == (4, config.num_classes), (name, tuple(logits.shape))
        assert tuple(feats.shape) == (4, config.feature_dim()), (name, tuple(feats.shape))

        model.train()
        loss = F.cross_entropy(model(x), torch.randint(0, config.num_classes, (4,)))
        loss.backward()
        assert all(p.grad is not None for p in model.parameters() if p.requires_grad), name
        report[f"{name}_logits_shape"] = tuple(logits.shape)
        report[f"{name}_num_params"] = model.num_parameters()

    # factory produces a *fresh* model each call (needed: one theta(m) per evaluation)
    factory = lenet_factory()
    m1, m2 = factory(), factory()
    assert m1 is not m2
    assert m1.num_parameters() == m2.num_parameters()
    report["factory_params"] = m1.num_parameters()

    report["roundtrip_dict"] = LeNetConfig.from_dict(cfg.to_dict()) == cfg
    assert report["roundtrip_dict"]

    report["status"] = "ok"
    if verbose:
        print(f"[lenet selftest] {report}")
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest(verbose=True)
