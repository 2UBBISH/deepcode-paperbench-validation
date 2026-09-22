"""Mechanistic-analysis subpackage for the DPO/toxicity reproduction.

This package groups the analysis utilities that reproduce Figures 1-5 and the
mechanistic claims (Sections 5 and 6) of:

    "A Mechanistic Understanding of Alignment Algorithms:
     A Case Study on DPO and Toxicity"

Submodules
----------
logit_lens
    Applies the unembedding to every intermittent residual stream (including
    ``l-mid`` states after attention, before the MLP) to reproduce Figure 1.
activations
    Mean MLP activation ``m_i`` (Eq. 4) and the activation-region / activation
    strength analysis behind Figure 2.
residual_shift
    ``delta_x`` residual-stream shift analysis: PCA projection (Figures 3-4) and
    cosine-similarity histograms against ``delta_MLP.v`` (Figure 5).
parameter_diff
    Parameter-delta checks: cosine similarity and norm difference between
    GPT2 and GPT2_DPO weights (Appendix C/D).
plots
    Shared matplotlib/seaborn helpers used by the analysis scripts.

Design notes
------------
As with :mod:`src`, this package defers all heavy imports (``torch``,
``transformers``, ``matplotlib``) to the individual submodules via PEP 562
module-level ``__getattr__``.  Consequently ``import src.analysis`` is cheap and
side-effect free, which keeps smoke tests and CI fast.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "__version__",
    "logit_lens",
    "activations",
    "residual_shift",
    "parameter_diff",
    "plots",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazily import analysis submodules on first attribute access (PEP 562)."""
    if name in __all__ and name != "__version__":
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
