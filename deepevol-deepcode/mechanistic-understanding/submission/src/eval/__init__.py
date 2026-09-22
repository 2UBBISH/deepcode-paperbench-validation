"""Evaluation metrics for the DPO / toxicity mechanistic reproduction (Section 3.3).

This subpackage implements the three metrics used throughout the paper:

* :mod:`src.eval.toxicity`    -- mean toxicity of generations on the 1,199
  RealToxicityPrompts "challenge" prompts, scored with
  ``unitary/unbiased-toxic-roberta`` (documented substitution for the
  Perspective API).
* :mod:`src.eval.perplexity`  -- Wikitext-2 perplexity via sliding-window
  causal-LM scoring, plus the alpha-selection helper that matches a target
  (post-DPO) perplexity for intervention experiments.
* :mod:`src.eval.f1`          -- token-overlap F1 against the original
  Wikipedia continuation on 2,000 Wikitext-2 sentence prompts.

All three modules are deliberately *cheap to import*: the heavy third-party
dependencies (``torch``, ``transformers``) are imported lazily inside their
functions, so ``import src.eval`` never touches the network or requires a GPU.

Access the submodules lazily::

    import src.eval
    src.eval.toxicity            # <module 'src.eval.toxicity'>
    src.eval.perplexity          # <module 'src.eval.perplexity'>
    src.eval.f1                  # <module 'src.eval.f1'>

    # convenience flat aliases (collision-free only):
    src.eval.mean_toxicity       # -> src.eval.toxicity.evaluate_toxicity
    src.eval.wikitext_perplexity # -> src.eval.perplexity.evaluate_perplexity
    src.eval.wikipedia_f1        # -> src.eval.f1.evaluate_f1
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

# Public submodule names exposed by this package (resolved lazily).
__all__ = [
    "__version__",
    "toxicity",
    "perplexity",
    "f1",
    # flat convenience aliases
    "mean_toxicity",
    "wikitext_perplexity",
    "wikipedia_f1",
    "ToxicityResult",
    "PerplexityResult",
    "F1Result",
    "ToxicityScorer",
    "generate_continuations",
    "match_alpha_for_target_ppl",
]

_SUBMODULES = ("toxicity", "perplexity", "f1")

# Flat alias -> (submodule, attribute).  Only collision-free, high-level entry
# points are re-exported here; everything else stays namespaced under the
# submodule to avoid clashing helper names (``default_path``, ``save_result``,
# ``ARTIFACT_DIR``, ``DEFAULT_BATCH_SIZE``, ... appear in all three modules).
_FLAT_ALIASES = {
    "mean_toxicity": ("toxicity", "evaluate_toxicity"),
    "wikitext_perplexity": ("perplexity", "evaluate_perplexity"),
    "wikipedia_f1": ("f1", "evaluate_f1"),
    "ToxicityResult": ("toxicity", "ToxicityResult"),
    "PerplexityResult": ("perplexity", "PerplexityResult"),
    "F1Result": ("f1", "F1Result"),
    "ToxicityScorer": ("toxicity", "ToxicityScorer"),
    "generate_continuations": ("toxicity", "generate_continuations"),
    "match_alpha_for_target_ppl": ("perplexity", "match_alpha_for_target_ppl"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute hook (keeps ``import src.eval`` side-effect free)."""
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _FLAT_ALIASES:
        submodule, attribute = _FLAT_ALIASES[name]
        module = import_module(f"{__name__}.{submodule}")
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
