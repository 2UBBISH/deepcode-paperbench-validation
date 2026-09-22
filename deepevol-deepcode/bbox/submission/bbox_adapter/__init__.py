"""BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models.

This package implements the *BBox-Adapter* method: a small, trainable energy
adapter ``g_theta`` (0.1B / 0.3B) is attached to a **frozen black-box** LLM
(gpt-3.5-turbo, davinci-002, Mixtral-8x7B-v0.1).  The adapter never observes the
black-box model's parameters, hidden states or output token probabilities; it
only sees plain text proposals and ranks them.

High level mapping to the paper:

===========================================  =================================
Paper component                              Module
===========================================  =================================
Sec. 3.1  g_theta(x, y) energy adapter       ``bbox_adapter.adapter``
Sec. 3.2  Ranking-based NCE (Eq. 2/3)        ``bbox_adapter.losses``
Sec. 3.3  Adapted inference / beam search    ``bbox_adapter.inference``
Sec. 3.4  Online adaptation (Algorithm 1)    ``bbox_adapter.training``
Sec. 4.1  Black-box clients + prompts        ``bbox_adapter.llm``
Sec. 3.4  GPT-4 AI feedback (SEL)            ``bbox_adapter.feedback``
Sec. 4.2+ Metrics, cost, VRAM                ``bbox_adapter.eval``
Sec. F.1  StrategyQA/GSM8K/.../ToxiGen       ``bbox_adapter.data``
===========================================  =================================

The submodules import heavy third-party dependencies (``torch``,
``transformers``, ``datasets``, ``requests``).  To keep ``import bbox_adapter``
cheap and robust (e.g. for config inspection on a machine without a GPU) the
submodules are exposed lazily through :func:`__getattr__` and a handful of
light helpers are re-exported eagerly.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

__version__ = "1.0.0"
__paper__ = "Lightweight Adapting for Black-Box Large Language Models"
__abbreviation__ = "BBox-Adapter"

__all__ = [
    "__version__",
    "__paper__",
    "__abbreviation__",
    "SUBMODULES",
    "get_config",
    "list_datasets",
    "get_default_config",
    "build_adapter",
    "build_blackbox",
    "describe",
    # re-exported helpers (utils is dependency-light)
    "load_config",
    "deep_merge",
    "resolve_config_path",
    "default_config_path",
    "set_seed",
    "seed_everything",
    "get_logger",
]

# ---------------------------------------------------------------------------
# lazy submodule registry
# ---------------------------------------------------------------------------

#: public subpackage -> short description (used by ``describe()`` and docs).
SUBMODULES: Dict[str, str] = {
    "data": "Dataset specs, loaders and answer extraction (Appendix F.1).",
    "llm": "Text-only black-box clients, prompts and token cost (Sec. 4.1, App. J).",
    "adapter": "Scalar energy adapter g_theta and its L2 regularizer (Sec. 3.1/3.2).",
    "losses": "Ranking-based NCE loss Eq.(2)/Eq.(3) and the MLM ablation (Sec. 4.5).",
    "inference": "Sentence-level beam search and final answer selection (Sec. 3.3).",
    "training": "Positive/negative buffers and the online adaptation loop (Algorithm 1).",
    "feedback": "GPT-4 AI feedback used as the SEL function (Sec. 3.4, App. G).",
    "eval": "Accuracy / True+Info / toxicity metrics, cost and VRAM accounting.",
    "utils": "Seeding, logging, artifact I/O and YAML configuration helpers.",
    "configs": "Bundled YAML hyperparameter configurations (App. H.2).",
}

_LAZY_MODULES = (
    "data",
    "llm",
    "adapter",
    "losses",
    "inference",
    "training",
    "feedback",
    "eval",
    "utils",
    "configs",
)

_LAZY_CACHE: Dict[str, Any] = {}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin import shim
    """Import ``bbox_adapter.<name>`` on first attribute access (PEP 562)."""
    if name in _LAZY_MODULES:
        if name not in _LAZY_CACHE:
            import importlib

            _LAZY_CACHE[name] = importlib.import_module(f"{__name__}.{name}")
        return _LAZY_CACHE[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - introspective helper
    return sorted(list(globals().keys()) + list(_LAZY_MODULES))


# ---------------------------------------------------------------------------
# light-weight eager helpers (no torch / transformers required)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - utils is dependency-light but stay defensive
    from .utils import (  # noqa: F401
        deep_merge,
        default_config_path,
        get_logger,
        load_config,
        resolve_config_path,
        seed_everything,
        set_seed,
    )
    from .utils.logging import RunLogger  # noqa: F401

    _UTILS_AVAILABLE = True
except Exception as _exc:  # pragma: no cover
    logging.getLogger(__name__).debug("bbox_adapter.utils unavailable: %s", _exc)
    _UTILS_AVAILABLE = False

    def load_config(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[misc]
        raise ImportError(
            "bbox_adapter.utils could not be imported; install pyyaml to use "
            "load_config()."
        )

    def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:  # type: ignore[misc]
        out = dict(base)
        for key, value in (override or {}).items():
            if (
                key in out
                and isinstance(out[key], Mapping)
                and isinstance(value, Mapping)
            ):
                out[key] = deep_merge(out[key], value)
            else:
                out[key] = value
        return out

    def resolve_config_path(name_or_path: str) -> str:  # type: ignore[misc]
        raise ImportError("bbox_adapter.utils is unavailable.")

    def default_config_path() -> str:  # type: ignore[misc]
        raise ImportError("bbox_adapter.utils is unavailable.")

    def set_seed(seed: int = 0, **kwargs: Any) -> int:  # type: ignore[misc]
        raise ImportError("bbox_adapter.utils is unavailable.")

    seed_everything = set_seed  # type: ignore[assignment]

    def get_logger(name: str = "bbox_adapter", **kwargs: Any) -> logging.Logger:  # type: ignore[misc]
        return logging.getLogger(name)

    RunLogger = None  # type: ignore[assignment]


def get_default_config(**kwargs: Any) -> Dict[str, Any]:
    """Return ``configs/default.yaml`` (optionally merged with overrides).

    Mirrors Appendix H.2: ``eta=5e-6``, ``batch_size=64``, ``6000`` training
    steps, AdamW ``weight_decay=0.01``, beam size 3, ``max_len=512`` and
    sampling ``temperature=1.0``.
    """
    return load_config(**kwargs)


def get_config(
    dataset: Optional[str] = None,
    *,
    path: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Load the default config deep-merged with a per-dataset YAML.

    ``dataset`` may be one of ``strategyqa``, ``gsm8k``, ``truthfulqa``,
    ``scienceqa`` or ``toxigen`` (or a literal path to a YAML file).
    """
    if dataset is not None:
        return load_config(dataset=dataset, overrides=overrides, **kwargs)
    return load_config(path=path, overrides=overrides, **kwargs)


def list_datasets() -> List[str]:
    """Names of the datasets reproduced by the paper (Sec. F.1 / App. E)."""
    try:
        from .data.dataset_specs import ALL_SPECS

        return [spec.name for spec in ALL_SPECS]
    except Exception:  # pragma: no cover
        return ["strategyqa", "gsm8k", "truthfulqa", "scienceqa", "toxigen"]


def build_adapter(
    dataset: Optional[str] = None,
    size: Optional[str] = None,
    backbone: Optional[str] = None,
    **kwargs: Any,
):
    """Convenience factory for the energy adapter ``g_theta`` (Sec. 3.1).

    Prefer calling :func:`bbox_adapter.adapter.build_energy_model` directly in
    library code; this wrapper only exists so ``bbox_adapter.build_adapter``
    works as a one-liner in scripts and notebooks.
    """
    from .adapter.energy_model import build_energy_model

    return build_energy_model(
        dataset=dataset, size=size, backbone=backbone, **kwargs
    )


def build_blackbox(model: str = "gpt-3.5-turbo", **kwargs: Any):
    """Convenience factory for a text-only black-box LLM client (Sec. 4.1)."""
    from .llm.blackbox_client import build_client

    return build_client(model=model, **kwargs)


def describe() -> Dict[str, Any]:
    """Return a small dict describing the package and its submodules."""
    return {
        "name": __name__,
        "version": __version__,
        "paper": __paper__,
        "abbreviation": __abbreviation__,
        "utils_available": _UTILS_AVAILABLE,
        "modules": dict(SUBMODULES),
        "datasets": list_datasets(),
    }
