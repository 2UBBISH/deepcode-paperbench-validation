"""LCA-on-the-Line: In-Distribution Taxonomic Distance (LCA) Predicts
Out-of-Distribution Generalization.

Top-level package initializer.

This module is intentionally *dependency free*: importing :mod:`src` must never
trigger a heavy import (``torch``, ``torchvision``, ``open_clip``, ``datasets``,
``numpy`` ...).  Sub-packages are exposed lazily through :pep:`562` module
``__getattr__`` so that ``import src`` stays cheap and works even in a minimal
environment (e.g. metadata-only tooling, unit tests that only need the taxonomy
code).

Sub-packages
------------
``src.hierarchy``
    WordNet / latent K-means class taxonomies, information content, pairwise
    ``D_LCA^I`` / ``D_LCA^P`` distances and the ``n x n`` LCA matrix pipeline.
``src.metrics``
    Dataset-level ``D_LCA`` / ``D_ELCA`` measurement and the correlation /
    regression suite (R^2, PEA, KEN, SPE, MAE) used by Table 2 / Table 3.
``src.models``
    The 36 torchvision vision models (VM) and 39 vision-language models (VLM)
    of Appendix A, exposing penultimate features ``M(X)`` and class logits.
``src.data``
    ImageNet-1k (in-distribution) and ImageNet-v2/Sketch/R/A/ObjectNet loaders
    with label alignment to the canonical 1000-class WordNet ordering.
``src.eval``
    The evaluation driver (ID/OOD Top-1/Top-5/LCA/ELCA) and the OOD
    performance-prediction baselines (ID Top-1, AC, Aline-D/S).
``src.alignment``
    Taxonomy-alignment soft loss (Algorithm 1), linear probing with weight
    interpolation, and taxonomy-aware prompt engineering.
``src.simulation``
    The Section C / Table 7 Gaussian-mixture study.
"""

from typing import Any, List

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "hierarchy",
    "metrics",
    "models",
    "data",
    "eval",
    "alignment",
    "simulation",
]

# Sub-package names that are importable on demand.
_SUBPACKAGES = (
    "hierarchy",
    "metrics",
    "models",
    "data",
    "eval",
    "alignment",
    "simulation",
)


def _import_subpackage(name: str) -> Any:
    """Import one of the sub-packages, tolerating both layout variants.

    The repository can be used either as ``import src.<name>`` (package at the
    repository root) or with ``<repo>/src`` on ``sys.path`` (flat layout, where
    the sub-package is simply ``<name>``).
    """
    import importlib

    errors: List[Exception] = []
    for candidate in (f"{__name__}.{name}", name):
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(exc)
    if errors:
        raise errors[-1]
    raise ImportError(name)  # pragma: no cover - unreachable


def __getattr__(name: str) -> Any:
    """Lazily resolve sub-package attributes (:pep:`562`)."""
    if name in _SUBPACKAGES:
        return _import_subpackage(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(_SUBPACKAGES)))
