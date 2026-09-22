"""Evaluation utilities for the stochastic-interpolant couplings repo.

This package bundles the quantitative (FID-50k) and qualitative
(base / model-sample / ground-truth triples + probability-flow slices)
evaluation code used to reproduce the numbers and figures in the paper
"Stochastic Interpolants with Data-Dependent Couplings".

The sub-modules are imported lazily so that heavy optional dependencies
(``torch-fidelity`` / ``clean-fid`` / ``scipy`` for FID, ``matplotlib`` for
figures) are only required when the corresponding function is actually used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    # --- FID -------------------------------------------------------------
    "compute_fid",
    "compute_fid_from_features",
    "InceptionFeatureExtractor",
    "FIDStats",
    "frechet_distance",
    "get_reference_stats",
    "fid_for_images",
    # --- qualitative -----------------------------------------------------
    "TripleFigure",
    "make_triples",
    "save_triples_grid",
    "make_probability_flow_figure",
    # --- convenience -----------------------------------------------------
    "EvalResult",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy re-export
    """Lazily re-export the public evaluation API.

    Importing :mod:`eval` must stay cheap (training scripts should not pay the
    cost of importing ``torch_fidelity``/``scipy``/``matplotlib``), therefore
    the symbols are resolved on first attribute access.
    """
    from . import fid as _fid
    from . import qualitative as _qualitative

    _fid_names = {
        "compute_fid",
        "compute_fid_from_features",
        "InceptionFeatureExtractor",
        "FIDStats",
        "frechet_distance",
        "get_reference_stats",
        "fid_for_images",
        "EvalResult",
    }
    _qualitative_names = {
        "TripleFigure",
        "make_triples",
        "save_triples_grid",
        "make_probability_flow_figure",
    }

    if name in _fid_names:
        return getattr(_fid, name)
    if name in _qualitative_names:
        return getattr(_qualitative, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover - introspection helper
    return sorted(set(list(globals().keys()) + __all__))


def describe() -> Dict[str, Optional[str]]:  # pragma: no cover
    """Return a small dictionary describing the evaluation entry points."""
    return {
        "fid": "FID-50k via Inception feature Gaussians (Section 4.1 / 4.2)",
        "qualitative": "base/model/GT triples and probability-flow slices (Figs. 3-6)",
    }
