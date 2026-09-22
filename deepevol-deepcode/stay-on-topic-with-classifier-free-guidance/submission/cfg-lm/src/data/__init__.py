"""Data layer for the CFG-LM reproduction.

This package bundles the two data-side modules of the *Stay on Topic with
Classifier-Free Guidance* reproduction:

* :mod:`src.data.prompts` -- every prompt string / exemplar / answer marker /
  shared sweep constant used across the paper's experiments (Self-Consistency
  8-shot GSM8K, standard AQuA few-shot, the Section 3.4 default & edited system
  prompts, and the gamma / temperature / unconditional-mode grids).
* :mod:`src.data.p3_sampler` -- the Section 5 P3 sampling protocol (~50 examples
  per subset, drop inputs > 200 tokens, 32,902-datapoint target, deterministic
  seed) including a cache and an offline synthetic fallback.

Both modules are deliberately *stdlib-only* where possible so plotting / eval
scripts can import this package cheaply; :mod:`p3_sampler` reaches for
``datasets``/``transformers`` lazily and degrades to an offline path when they
(或 the network) are unavailable.

The package uses a name-checked lazy re-export: module attributes are copied up
only when they actually exist, so partial or dependency-restricted installs
still import cleanly and ``from src.data import *`` stays consistent with what
is really available.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__: List[str] = []

# ---------------------------------------------------------------------------
# Prompts (no heavy dependencies)
# ---------------------------------------------------------------------------

_PROMPT_EXPORTS = (
    # builders / helpers
    "PromptExample",
    "build_gsm8k_prompt",
    "build_aqua_prompt",
    "build_chat_prompt",
    "build_cot_prompt",
    "negative_prompt_pair",
    "get_prompt",
    # few-shot prefixes and exemplars
    "GSM8K_EXEMPLARS",
    "AQUA_EXEMPLARS",
    "GSM8K_8SHOT",
    "GSM8K_PROMPT",
    "AQUA_4SHOT",
    "AQUA_PROMPT",
    # answer markers / formats
    "GSM8K_ANSWER_MARKER",
    "GSM8K_ANSWER_FORMAT",
    "AQUA_ANSWER_MARKER",
    # token budgets
    "HUMANEVAL_MAX_NEW_TOKENS",
    "COT_MAX_NEW_TOKENS",
    "HUMANEVAL_INSTRUCTION",
    # system prompts (Section 3.4; out of scope to *run* but kept for completeness)
    "DEFAULT_SYSTEM_PROMPT",
    "EDITED_SYSTEM_PROMPTS",
    "SYSTEM_PROMPTS",
    "N_SYSTEM_PROMPTS",
    "N_USER_PROMPTS",
    "N_PROMPT_PAIRS",
    "NEGATIVE_PROMPT_GAMMAS",
    "USER_PROMPTS",
    # shared sweep grids / conventions
    "CFG_GAMMAS",
    "HUMANEVAL_TEMPERATURES",
    "ANALYSIS_GAMMA",
    "UNCONDITIONAL_MODE_DEFAULT",
    "UNCONDITIONAL_MODE_ZERO_SHOT",
    "UNCONDITIONAL_MODES",
    "HARNESS_DEFAULT_TEMPERATURE",
    "HARNESS_DEFAULT_TOP_P",
    "ZERO_SHOT_MAX_LENGTH",
    # registries
    "PROMPT_REGISTRY",
    "COT_PROMPTS",
)

try:  # pragma: no cover - import guard
    from . import prompts as _prompts
except Exception as exc:  # noqa: BLE001 - keep the package importable
    logger.debug("src.data.prompts unavailable: %s", exc)
    _prompts = None  # type: ignore[assignment]

if _prompts is not None:
    _missing = []
    for _name in _PROMPT_EXPORTS:
        if hasattr(_prompts, _name):
            globals()[_name] = getattr(_prompts, _name)
            __all__.append(_name)
        else:
            _missing.append(_name)
    if _missing:
        logger.debug("src.data.prompts missing expected symbols: %s", _missing)

# ---------------------------------------------------------------------------
# P3 sampler (optional `datasets` / network)
# ---------------------------------------------------------------------------

_P3_EXPORTS = (
    # containers
    "P3Sample",
    "WhitespaceTokenCounter",
    # token counting
    "make_token_counter",
    "count_tokens",
    # sampling core
    "subset_seed",
    "sample_from_records",
    "list_p3_subsets",
    "sample_dataset",
    "sample_p3",
    "synthetic_records",
    "sample_p3_synthetic",
    # persistence / inspection
    "save_samples",
    "load_samples",
    "dataset_histogram",
    "summary_stats",
    "iter_batches",
    "split_samples",
    "get_p3_sample",
    # protocol constants
    "P3_DATASET_NAME",
    "SAMPLES_PER_DATASET",
    "N_P3_DATASETS",
    "TARGET_N_DATAPOINTS",
    "MAX_INPUT_TOKENS",
    "DEFAULT_SEED",
    "DEFAULT_SPLIT",
    "FALLBACK_SPLIT",
    "DEFAULT_CACHE_PATH",
    "INPUT_FIELD",
    "TARGET_FIELD",
    "TOKENIZED_INPUT_FIELD",
    "TOKENIZED_TARGET_FIELD",
)

try:  # pragma: no cover - import guard
    from . import p3_sampler as _p3_sampler
except Exception as exc:  # noqa: BLE001 - keep the package importable
    logger.debug("src.data.p3_sampler unavailable: %s", exc)
    _p3_sampler = None  # type: ignore[assignment]

if _p3_sampler is not None:
    _missing = []
    for _name in _P3_EXPORTS:
        if hasattr(_p3_sampler, _name):
            globals()[_name] = getattr(_p3_sampler, _name)
            __all__.append(_name)
        else:
            _missing.append(_name)
    if _missing:
        logger.debug("src.data.p3_sampler missing expected symbols: %s", _missing)


# ---------------------------------------------------------------------------
# Convenience aggregate
# ---------------------------------------------------------------------------


def available_exports() -> Dict[str, bool]:
    """Report which submodules imported successfully.

    Useful for scripts that want to warn the user instead of crashing when the
    optional data dependencies (``datasets``/``transformers``) are missing.

    Returns:
        ``{"prompts": bool, "p3_sampler": bool}``
    """
    return {
        "prompts": _prompts is not None,
        "p3_sampler": _p3_sampler is not None,
    }


def get_cot_prompt_spec(task: str = "gsm8k") -> Dict[str, object]:
    """Return the ``{prefix, builder, answer_marker}`` triple for a CoT task.

    Thin wrapper over :data:`COT_PROMPTS` so the CoT script (Figure 2 / 17) can
    fetch its prompt spec from the data package without knowing the prompts
    module layout.
    """
    if _prompts is None:  # pragma: no cover - defensive
        raise RuntimeError("src.data.prompts failed to import; cannot fetch CoT prompt spec")
    return COT_PROMPTS[task]  # type: ignore[name-defined]


__all__ += ["available_exports", "get_cot_prompt_spec"]
