"""Model zoo for the LBCS reproduction.

This package aggregates the network architectures required by the paper:

* :mod:`lbcs_repro.models.convnet`      -- ConvNet of Zhou et al. (2022), Figure 1 / Section 5.1
* :mod:`lbcs_repro.models.lenet`        -- LeNet proxy/target model for F-MNIST (Section 5.2)
* :mod:`lbcs_repro.models.svhn_cnn`     -- SVHN inner-loop CNN and SVHN target CNN (Table 7)
* :mod:`lbcs_repro.models.cifar_cnn`    -- CIFAR-10 inner-loop CNN (Table 7)
* :mod:`lbcs_repro.models.resnet18`     -- CIFAR-10 ResNet-18 target model (Section 5.2)
* :mod:`lbcs_repro.models.vit_small`    -- ViT-small target model (Section 6)
* :mod:`lbcs_repro.models.wide_resnet`  -- WideResNet target model (Section 6)

Every architecture module follows the same contract expected by the Algorithm 1
inner loop (:class:`lbcs_repro.lbcs.bilevel.LBCS`):

1. a ``*Config`` dataclass describing the topology (all paper-unstated widths are
   exposed as clearly labelled SUGGESTED defaults so they can be overridden from
   ``configs/*.yaml`` without editing algorithm code),
2. a ``*_factory(...)`` callable returning a *fresh* network instance per call
   (Algorithm 1 trains a new ``theta(m)`` for every mask evaluation),
3. a ``MODEL_REGISTRY`` mapping names -> builder callables, and
4. ``features(x)`` / raw-logit ``forward(x)`` so distance-based baselines
   (Moderate, CCS) and the cross-entropy objectives ``f1(m)`` / ``L(m, theta)``
   can both consume the same model.

The aggregator also exposes helper functions from the model modules, and a
``get_model`` / ``build_model`` name-based factory so experiment drivers and
configuration dispatch can instantiate architectures without hard-coded imports.

Scope note: ImageNet-1k (Section 5.4) is out of scope for this reproduction, so
no ImageNet architectures are registered here.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Type

__all__: List[str] = []

LOGGER = logging.getLogger(__name__)

# Set to True if any of the architecture modules could be imported.  Architectures
# themselves treat PyTorch as a *soft* dependency, so the package still imports in
# a numpy-only environment (mask algebra / unit tests).
_TORCH_AVAILABLE = False


def _extend_public(names: List[str]) -> None:
    for name in names:
        if name not in __all__:
            __all__.append(name)


# ---------------------------------------------------------------------------
# ConvNet (Figure 1 / Section 5.1; Zhou et al. 2022)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .convnet import (  # noqa: F401
        ConvNet,
        ConvNetConfig,
        build_convnet,
        convnet_factory,
        mnist_convnet,
    )

    _CONVNET_AVAILABLE = True
    _extend_public(
        [
            "ConvNet",
            "ConvNetConfig",
            "build_convnet",
            "convnet_factory",
            "mnist_convnet",
        ]
    )
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("ConvNet unavailable: %s", exc)
    _CONVNET_AVAILABLE = False
    ConvNet = None  # type: ignore[assignment]
    ConvNetConfig = None  # type: ignore[assignment]
    build_convnet = None  # type: ignore[assignment]
    convnet_factory = None  # type: ignore[assignment]
    mnist_convnet = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# LeNet (F-MNIST proxy + target, Section 5.2)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .lenet import (  # noqa: F401
        LeNet,
        LeNetConfig,
        build_lenet,
        fashion_mnist_lenet,
        lenet_factory,
        lenet5,
    )

    _LENET_AVAILABLE = True
    _extend_public(
        [
            "LeNet",
            "LeNetConfig",
            "build_lenet",
            "lenet",
            "lenet_factory",
            "fashion_mnist_lenet",
            "lenet5",
        ]
    )
    try:  # `lenet` is an alias of `build_lenet`
        from .lenet import lenet  # noqa: F401
    except Exception:  # pragma: no cover
        lenet = build_lenet  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("LeNet unavailable: %s", exc)
    _LENET_AVAILABLE = False
    LeNet = None  # type: ignore[assignment]
    LeNetConfig = None  # type: ignore[assignment]
    build_lenet = None  # type: ignore[assignment]
    lenet = None  # type: ignore[assignment]
    lenet_factory = None  # type: ignore[assignment]
    fashion_mnist_lenet = None  # type: ignore[assignment]
    lenet5 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# SVHN CNN (inner proxy + target, Table 7 / Section 5.2)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .svhn_cnn import (  # noqa: F401
        SVHNCNN,
        SVHNCNNConfig,
        build_svhn_cnn,
        svhn_cnn_factory,
        svhn_inner_cnn,
        svhn_target_cnn,
    )

    _SVHN_CNN_AVAILABLE = True
    _extend_public(
        [
            "SVHNCNN",
            "SVHNCNNConfig",
            "build_svhn_cnn",
            "svhn_cnn",
            "svhn_cnn_factory",
            "svhn_inner_cnn",
            "svhn_target_cnn",
            "SVHNInnerCNN",
            "SVHNCNNTarget",
            "SVHNTargetCNN",
        ]
    )
    try:  # optional aliases
        from .svhn_cnn import (  # noqa: F401
            SVHNCNNTarget,
            SVHNInnerCNN,
            SVHNTargetCNN,
            svhn_cnn,
        )
    except Exception:  # pragma: no cover
        pass
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("SVHN CNN unavailable: %s", exc)
    _SVHN_CNN_AVAILABLE = False
    SVHNCNN = None  # type: ignore[assignment]
    SVHNCNNConfig = None  # type: ignore[assignment]
    build_svhn_cnn = None  # type: ignore[assignment]
    svhn_cnn_factory = None  # type: ignore[assignment]
    svhn_inner_cnn = None  # type: ignore[assignment]
    svhn_target_cnn = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# CIFAR-10 CNN (inner proxy, Table 7 / Section 5.2)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .cifar_cnn import (  # noqa: F401
        CIFARCNN,
        CIFARCNNConfig,
        build_cifar_cnn,
        cifar_cnn_factory,
        cifar_inner_cnn,
        cifar_inner_cnn_factory,
        cifar_target_cnn,
        cifar_target_cnn_factory,
        train_cifar_target,
    )

    _CIFAR_CNN_AVAILABLE = True
    _extend_public(
        [
            "CIFARCNN",
            "CIFARCNNConfig",
            "build_cifar_cnn",
            "cifar_cnn",
            "cifar_cnn_factory",
            "cifar_inner_cnn",
            "cifar_inner_cnn_factory",
            "cifar_target_cnn",
            "cifar_target_cnn_factory",
            "CIFARInnerCNN",
            "CIFARCNNTarget",
            "CIFARTargetCNN",
            "train_cifar_target",
        ]
    )
    try:  # optional aliases
        from .cifar_cnn import (  # noqa: F401
            CIFARCNNTarget,
            CIFARInnerCNN,
            CIFARTargetCNN,
            cifar_cnn,
        )
    except Exception:  # pragma: no cover
        pass
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("CIFAR CNN unavailable: %s", exc)
    _CIFAR_CNN_AVAILABLE = False
    CIFARCNN = None  # type: ignore[assignment]
    CIFARCNNConfig = None  # type: ignore[assignment]
    build_cifar_cnn = None  # type: ignore[assignment]
    cifar_cnn_factory = None  # type: ignore[assignment]
    cifar_inner_cnn = None  # type: ignore[assignment]
    cifar_target_cnn = None  # type: ignore[assignment]
    train_cifar_target = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ResNet-18 (CIFAR-10 target model, Section 5.2)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .resnet18 import (  # noqa: F401
        BasicBlock as ResNetBasicBlock,
    )
    from .resnet18 import (  # noqa: F401
        ResNet18,
        ResNet18Config,
        build_resnet18,
        cifar_resnet18,
        evaluate as resnet_evaluate,
        resnet18,
        resnet18_factory,
        target_train_config as resnet_target_train_config,
        train_cifar_resnet18,
        train_target_model as train_resnet_target,
    )

    _RESNET18_AVAILABLE = True
    _extend_public(
        [
            "ResNet18",
            "ResNet18Config",
            "ResNetBasicBlock",
            "build_resnet18",
            "resnet18",
            "resnet18_factory",
            "cifar_resnet18",
            "cifar10_resnet18",
            "train_cifar_resnet18",
            "train_resnet_target",
            "resnet_target_train_config",
            "resnet_evaluate",
        ]
    )
    try:  # optional alias
        from .resnet18 import cifar10_resnet18  # noqa: F401

    except Exception:  # pragma: no cover
        cifar10_resnet18 = cifar_resnet18  # type: ignore[assignment]
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("ResNet-18 unavailable: %s", exc)
    _RESNET18_AVAILABLE = False
    ResNet18 = None  # type: ignore[assignment]
    ResNet18Config = None  # type: ignore[assignment]
    build_resnet18 = None  # type: ignore[assignment]
    resnet18 = None  # type: ignore[assignment]
    resnet18_factory = None  # type: ignore[assignment]
    cifar_resnet18 = None  # type: ignore[assignment]
    cifar10_resnet18 = None  # type: ignore[assignment]
    train_cifar_resnet18 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ViT-small (Section 6 cross-architecture)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .vit_small import (  # noqa: F401
        ViTSmall,
        ViTSmallConfig,
        build_vit_small,
        svhn_vit_small,
        train_svhn_vit_small,
        vit_small,
        vit_small_factory,
    )

    _VIT_SMALL_AVAILABLE = True
    _extend_public(
        [
            "ViTSmall",
            "ViTSmallConfig",
            "build_vit_small",
            "vit_small",
            "vit_small_factory",
            "svhn_vit_small",
            "train_svhn_vit_small",
        ]
    )
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("ViT-small unavailable: %s", exc)
    _VIT_SMALL_AVAILABLE = False
    ViTSmall = None  # type: ignore[assignment]
    ViTSmallConfig = None  # type: ignore[assignment]
    build_vit_small = None  # type: ignore[assignment]
    vit_small = None  # type: ignore[assignment]
    vit_small_factory = None  # type: ignore[assignment]
    svhn_vit_small = None  # type: ignore[assignment]
    train_svhn_vit_small = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# WideResNet (Section 6 cross-architecture)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .wide_resnet import (  # noqa: F401
        WideBasicBlock,
        WideResNet,
        WideResNetConfig,
        build_wide_resnet,
        svhn_wide_resnet,
        svhn_wide_resnet_factory,
        train_svhn_wide_resnet,
        train_target_model as train_wide_resnet_target,
        wide_resnet,
        wide_resnet_16_4,
        wide_resnet_factory,
        wnet,
    )

    _WIDE_RESNET_AVAILABLE = True
    _extend_public(
        [
            "WideResNet",
            "WideResNetConfig",
            "WideBasicBlock",
            "build_wide_resnet",
            "wide_resnet",
            "wnet",
            "wide_resnet_factory",
            "wide_resnet_16_4",
            "svhn_wide_resnet",
            "svhn_wide_resnet_factory",
            "train_svhn_wide_resnet",
            "train_wide_resnet_target",
            "WideResNetTarget",
            "WNetSVHN",
            "WRN",
        ]
    )
    try:  # optional aliases
        from .wide_resnet import (  # noqa: F401
            WNetSVHN,
            WRN,
            WideResNetTarget,
        )
    except Exception:  # pragma: no cover
        pass
except Exception as exc:  # pragma: no cover - defensive
    LOGGER.debug("WideResNet unavailable: %s", exc)
    _WIDE_RESNET_AVAILABLE = False
    WideResNet = None  # type: ignore[assignment]
    WideResNetConfig = None  # type: ignore[assignment]
    build_wide_resnet = None  # type: ignore[assignment]
    wide_resnet = None  # type: ignore[assignment]
    wide_resnet_factory = None  # type: ignore[assignment]
    svhn_wide_resnet = None  # type: ignore[assignment]
    svhn_wide_resnet_factory = None  # type: ignore[assignment]


_TORCH_AVAILABLE = any(
    [
        _CONVNET_AVAILABLE,
        _LENET_AVAILABLE,
        _SVHN_CNN_AVAILABLE,
        _CIFAR_CNN_AVAILABLE,
        _RESNET18_AVAILABLE,
        _VIT_SMALL_AVAILABLE,
        _WIDE_RESNET_AVAILABLE,
    ]
)


# ---------------------------------------------------------------------------
# Unified registry: name -> fresh-instance factory
# ---------------------------------------------------------------------------
def _build_registry() -> Dict[str, Callable[..., Any]]:
    """Aggregate the per-module ``MODEL_REGISTRY`` dicts.

    Name lookups are case-insensitive; the first registration for a canonical
    key wins so that e.g. ``"resnet18"`` maps to the ResNet-18 factory rather
    than being shadowed by another module.
    """
    registry: Dict[str, Callable[..., Any]] = {}
    module_names = [
        "convnet",
        "lenet",
        "svhn_cnn",
        "cifar_cnn",
        "resnet18",
        "vit_small",
        "wide_resnet",
    ]
    for module_name in module_names:
        try:
            module = __import__(f"{__name__}.{module_name}", fromlist=["MODEL_REGISTRY"])
        except Exception:  # pragma: no cover - module already guarded above
            continue
        module_registry = getattr(module, "MODEL_REGISTRY", None)
        if not isinstance(module_registry, dict):
            continue
        for key, factory in module_registry.items():
            registry.setdefault(str(key), factory)
            registry.setdefault(str(key).lower(), factory)
    return registry


MODEL_REGISTRY: Dict[str, Callable[..., Any]] = _build_registry()

# Human-readable architecture names grouped by benchmark role.
ARCHITECTURES: Dict[str, str] = {
    # Figure 1 / Section 5.1 (MNIST-S)
    "convnet": "ConvNet (Zhou et al. 2022): two conv blocks + MLP head",
    "mnist": "ConvNet configured for 1x28x28 MNIST / MNIST-S",
    # Section 5.2 F-MNIST
    "lenet": "LeNet proxy and target model for Fashion-MNIST",
    # Section 5.2 SVHN
    "svhn_inner": "SVHN inner-loop CNN (coreset selection proxy)",
    "svhn_target": "SVHN target CNN (post-selection accuracy)",
    # Section 5.2 CIFAR-10
    "cifar_inner": "CIFAR-10 inner-loop CNN (coreset selection proxy)",
    "cifar_target": "CIFAR-10 ResNet-18 target model",
    # Section 6 cross-architecture
    "vit_small": "ViT-small target model (SVHN)",
    "wide_resnet": "WideResNet target model (SVHN)",
}

# Default (proxy, target) architecture pair per benchmark dataset, mirroring the
# paper: the proxy network theta(m) is trained on the coreset, while the reported
# test accuracy is measured with the target network retrained on the coreset.
DEFAULT_MODELS: Dict[str, Dict[str, str]] = {
    "MNIST-S": {"inner": "ConvNet", "target": "ConvNet"},
    "MNIST": {"inner": "ConvNet", "target": "ConvNet"},
    "F-MNIST": {"inner": "LeNet", "target": "LeNet"},
    "SVHN": {"inner": "SVHNCNN", "target": "SVHNCNNTarget"},
    "CIFAR-10": {"inner": "CIFARCNN", "target": "ResNet18"},
}


def available_models() -> List[str]:
    """Return the sorted list of registered architecture names."""
    return sorted(MODEL_REGISTRY.keys())


def get_model(name: str) -> Callable[..., Any]:
    """Return the builder callable registered under ``name``.

    Raises
    ------
    KeyError
        If no architecture is registered under the given name.
    """
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    lowered = str(name).lower()
    if lowered in MODEL_REGISTRY:
        return MODEL_REGISTRY[lowered]
    raise KeyError(
        f"Unknown model '{name}'. Available: {sorted(set(MODEL_REGISTRY.keys()))}"
    )


def build_model(name: str, **kwargs: Any) -> Any:
    """Instantiate a fresh network of type ``name`` (Algorithm 1 contract).

    Each call returns a brand-new ``theta(m)`` so that the inner loop can train a
    model from scratch for every evaluated mask.
    """
    factory = get_model(name)
    return factory(**kwargs)


def model_factory(name: str, **factory_kwargs: Any) -> Callable[..., Any]:
    """Return a callable that builds *fresh* networks of type ``name``.

    This is the signature consumed by :class:`lbcs_repro.lbcs.bilevel.LBCS`,
    which requires a new ``theta(m)`` for each inner-loop training run.
    """
    factory = get_model(name)

    def _build(**kwargs: Any) -> Any:
        merged = dict(factory_kwargs)
        merged.update(kwargs)
        return factory(**merged)

    _build.name = name  # type: ignore[attr-defined]
    return _build


def default_model_for(dataset: str, role: str = "inner") -> str:
    """Return the paper-default architecture name for ``dataset`` and ``role``.

    ``role`` is ``"inner"`` (proxy ``theta(m)``) or ``"target"`` (post-selection
    model whose test accuracy is reported).
    """
    key = str(dataset)
    if key not in DEFAULT_MODELS:
        # tolerate aliases / different capitalisation
        for candidate, value in DEFAULT_MODELS.items():
            if candidate.lower() == key.lower():
                return value.get(role, value["inner"])
        raise KeyError(f"Unknown dataset '{dataset}'. Known: {sorted(DEFAULT_MODELS)}")
    entry = DEFAULT_MODELS[key]
    return entry.get(role, entry["inner"])


def model_config(name: str, **overrides: Any) -> Any:
    """Build the ``*Config`` dataclass for a registered architecture.

    Falls back to a plain dict of overrides when the module does not expose a
    config class (should not happen for the bundled architectures).
    """
    lowered = str(name).lower()
    config_cls: Optional[Type[Any]] = None
    if lowered in {"convnet", "conv"}:
        config_cls = ConvNetConfig
    elif lowered in {"lenet", "lenet5"}:
        config_cls = LeNetConfig
    elif lowered in {"svhncnn", "svhn_cnn", "svhn", "svhninnercnn"}:
        config_cls = SVHNCNNConfig
    elif lowered in {"cifarcnn", "cifar_cnn", "cifar", "cifarinnercnn"}:
        config_cls = CIFARCNNConfig
    elif lowered in {"resnet18", "resnet-18", "cifarresnet18"}:
        config_cls = ResNet18Config
    elif lowered in {"vitsmall", "vit-small", "vit_small", "vit"}:
        config_cls = ViTSmallConfig
    elif lowered in {"wideresnet", "wide_resnet", "wrn", "w-net", "wnet"}:
        config_cls = WideResNetConfig

    if config_cls is None:
        return dict(overrides)
    if hasattr(config_cls, "with_overrides"):
        return config_cls.with_overrides(**overrides)
    return config_cls(**overrides)


# ---------------------------------------------------------------------------
# Self-test (offline, no dataset download required)
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Structural checks of the registry; safe to run without datasets."""
    report: Dict[str, Any] = {
        "torch_available": _TORCH_AVAILABLE,
        "models": available_models(),
        "num_models": len(available_models()),
        "defaults": dict(DEFAULT_MODELS),
    }

    # registry lookups must be case-insensitive and resolve to callables
    for key in list(MODEL_REGISTRY.keys()):
        assert callable(MODEL_REGISTRY[key]), f"registry entry '{key}' not callable"
    assert callable(get_model("ConvNet")) or not _CONVNET_AVAILABLE
    assert callable(get_model("convnet")) or not _CONVNET_AVAILABLE

    # default model resolution
    assert default_model_for("F-MNIST", "inner").lower().startswith("lenet")
    assert default_model_for("CIFAR-10", "target").lower().startswith("resnet")
    report["default_lookup_ok"] = True

    if _TORCH_AVAILABLE:
        try:  # cheap forward-pass smoke test on the registry's ConvNet
            import torch

            builder = model_factory("ConvNet")
            model = builder()
            x = torch.zeros(2, 1, 28, 28)
            out = model(x)
            report["convnet_logits_shape"] = tuple(out.shape)
            assert tuple(out.shape) == (2, 10), out.shape
            report["forward_ok"] = True
        except Exception as exc:  # pragma: no cover - environment dependent
            report["forward_ok"] = False
            report["forward_error"] = repr(exc)

    if verbose:
        print("models.__init__ self-test")
        for key, value in report.items():
            print(f"  {key}: {value}")
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest(verbose=True)
