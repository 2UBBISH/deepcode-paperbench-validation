"""BBox-Adapter energy adapter package.

Exposes the scalar energy model ``g_theta(x, y)`` (Section 3.1) and the
``alpha * E[g_theta^2]`` energy regularizer used by the ranking-based NCE
objective (Section 3.2 / Eq. 3).

The adapter is a small *open* pretrained encoder (DeBERTa-v3-base/large, or
BERT-base-cased for TruthfulQA); the black-box LLM is never touched here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .energy_model import (
    ADAPTER_BACKBONES,
    SIZE_BACKBONES,
    EnergyModel,
    EnergyModelConfig,
    batch_pairs,
    build_energy_model,
    format_pair,
    resolve_backbone,
)

__all__ = [
    # energy model
    "ADAPTER_BACKBONES",
    "SIZE_BACKBONES",
    "EnergyModel",
    "EnergyModelConfig",
    "batch_pairs",
    "build_energy_model",
    "format_pair",
    "resolve_backbone",
    # regularizer
    "DEFAULT_ALPHA",
    "ALPHA_SWEEP",
    "EnergyRegularizer",
    "RegularizerConfig",
    "alpha_schedule",
    "check_energy_scales",
    "energy_regularizer",
    "positive_negative_penalty",
    "squared_energy_penalty",
    # factory
    "get_adapter",
]

# The regularizer lives in ``adapter/regularizer.py`` and is optional at import
# time so that the adapter package remains usable if only the energy model file
# is present.
try:  # pragma: no cover - trivial import guard
    from .regularizer import (  # type: ignore
        ALPHA_SWEEP,
        DEFAULT_ALPHA,
        EnergyRegularizer,
        RegularizerConfig,
        alpha_schedule,
        check_energy_scales,
        energy_regularizer,
        positive_negative_penalty,
        squared_energy_penalty,
    )
except Exception:  # pragma: no cover
    pass


def get_adapter(dataset: Optional[str] = None,
                size: Optional[str] = None,
                backbone: Optional[str] = None,
                **kwargs: Any) -> EnergyModel:
    """Convenience factory mirroring :func:`build_energy_model`.

    Resolves the paper's per-dataset backbone (DeBERTa-v3-base / -large, or
    bert-base-cased for TruthfulQA) and returns a freshly initialized
    :class:`~bbox_adapter.adapter.energy_model.EnergyModel` whose scalar head is
    randomly initialized before online adaptation (Section 4.1 Implementations).
    """
    return build_energy_model(dataset=dataset, size=size, backbone=backbone, **kwargs)


def describe() -> Dict[str, Any]:
    """Return a small metadata dict (used by scripts/run headers)."""
    try:
        from ..utils.logging import describe_environment  # local import

        env = describe_environment()
    except Exception:  # pragma: no cover
        env = {}
    return {
        "module": "bbox_adapter.adapter",
        "paper": "Lightweight Adapting for Black-Box Large Language Models",
        "section": "3.1 / 3.2",
        "backbones": dict(ADAPTER_BACKBONES),
        "sizes": dict(SIZE_BACKBONES),
        "environment": env,
    }


def _self_test() -> Dict[str, Any]:
    """Dependency-free sanity check for the adapter package surface."""
    import inspect

    assert callable(get_adapter)
    assert issubclass(EnergyModel, object)
    assert "deberta-v3" in resolve_backbone(dataset="strategyqa", size="0.1b")
    assert resolve_backbone(dataset="truthfulqa", size="0.1b") == "bert-base-cased"
    assert "EnergyRegularizer" in __all__
    return {
        "ok": True,
        "backbone_strategyqa": resolve_backbone(dataset="strategyqa", size="0.1b"),
        "backbone_truthfulqa": resolve_backbone(dataset="truthfulqa", size="0.1b"),
        "has_get_adapter": inspect.isfunction(get_adapter),
    }


if __name__ == "__main__":  # pragma: no cover
    print(_self_test())
