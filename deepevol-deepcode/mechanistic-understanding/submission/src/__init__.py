"""Core library for the mechanistic study of DPO and toxicity (GPT2-medium).

This package implements the components used to reproduce:

    "A Mechanistic Understanding of Alignment Algorithms:
     A Case Study on DPO and Toxicity"

Sub-packages / modules
----------------------
``src.model_utils``
    GPT2 loading, architecture introspection, residual-stream and MLP hooks,
    key/value vector access, vocabulary projection and the out-of-scope GLU stub.
``src.probe``
    The linear toxicity probe ``W_Toxic`` (shape ``[d_model, 2]``).
``src.toxic_vectors``
    ``MLP.v_Toxic`` / ``MLP.k_Toxic`` / ``SVD.U_Toxic`` extraction and
    vocabulary-space projection.
``src.interventions``
    Residual-stream subtraction interventions (Table 2).
``src.pplm_generate``
    PPLM toxic-continuation generation used to build the preference pairs.
``src.dpo_trainer``
    DPO loss, batching and training loop (Table 8 settings).
``src.unalign``
    GPT2 un-alignment by scaling ``MLP.k_Toxic`` key vectors (Table 4).
``src.analysis``
    Logit lens, mean activations, residual-shift / parameter-delta analyses
    and shared plotting utilities (Figures 1-5).
``src.eval``
    Toxicity (unbiased-toxic-roberta), Wikitext-2 perplexity and F1 metrics.

All heavy third-party imports (``torch``, ``transformers``, ``datasets``) are
deferred to the individual modules so that ``import src`` stays cheap and does
not touch the network.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "model_utils",
    "probe",
    "toxic_vectors",
    "interventions",
    "pplm_generate",
    "dpo_trainer",
    "unalign",
    "analysis",
    "eval",
]


def __getattr__(name: str):  # pragma: no cover - convenience lazy import
    """Lazily import submodules on attribute access (PEP 562)."""
    import importlib

    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover - intro
    return sorted(list(globals().keys()) + __all__)
