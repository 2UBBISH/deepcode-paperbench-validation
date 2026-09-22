"""Classifier-Free Guidance (CFG) for autoregressive language models.

Reference paper: "Stay on Topic with Classifier-Free Guidance".

This package implements the paper's inference-time, training-free CFG technique
for causal language models.  At every decoding step two forward passes are run
through the *same* LM weights:

    * a conditional pass with the prompt ``c``,
    * an unconditional pass with the prompt dropped (or replaced by a negative
      prompt ``c_bar``),

and the next-token logits are combined in log space (paper Eq. 5 / Eq. 7):

    guided = logits_uncond + gamma * (logits_cond - logits_uncond)

The package is split into four core modules:

``logits``
    Pure logit-space math: Eq. 7 combination, negative prompting (Eq. 5),
    log-softmax / softmax, and top-p (nucleus) filtering.
``model_wrapper``
    ``CFGModelWrapper`` -- a HuggingFace ``AutoModelForCausalLM`` wrapper that
    returns the paired raw (pre-softmax) logits from both contexts.
``sampler``
    ``CFGSampler`` -- applies CFG -> temperature -> top-p/top-k -> softmax ->
    multinomial, with the EleutherAI LM-Evaluation-Harness defaults.
``generator``
    ``CFGGenerator`` -- the autoregressive CFG decoding loop with EOS /
    stop-string handling and kv-cache reuse.

Import strategy
---------------
The mathematical core (``logits``) only needs :mod:`numpy`, so it is always
imported eagerly.  The modules that require :mod:`torch` (and, for the wrapper,
:mod:`transformers`) are imported lazily inside ``try``/``except`` blocks so
that a minimal environment can still use the logit math and run the unit tests.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Pure logit-space math (no torch required).
# ---------------------------------------------------------------------------
from .logits import (  # noqa: E402
    cfg_combine,
    guided_logits,
    negative_prompt_logits,
    log_softmax,
    top_p_filter,
    softmax,
)

__all__: List[str] = [
    # logits.py
    "cfg_combine",
    "guided_logits",
    "negative_prompt_logits",
    "log_softmax",
    "top_p_filter",
    "softmax",
]

# ---------------------------------------------------------------------------
# Torch-dependent modules: import best-effort.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when torch is installed
    from .sampler import (  # noqa: E402
        CFGSampler,
        SamplingConfig,
        cfg_sample,
        greedy_token,
        guided_logits_from_pair,
        apply_temperature,
        apply_top_p,
        apply_top_k,
        grid_configs,
        get_generator,
        cfG_overhead_factor,
        numpy_distribution,
        CFG_GAMMAS,
        HUMANEVAL_TEMPERATURES,
        HARNESS_DEFAULT_TEMPERATURE,
        HARNESS_DEFAULT_TOP_P,
        ANALYSIS_GAMMA,
    )

    __all__.extend(
        [
            "CFGSampler",
            "SamplingConfig",
            "cfg_sample",
            "greedy_token",
            "guided_logits_from_pair",
            "apply_temperature",
            "apply_top_p",
            "apply_top_k",
            "grid_configs",
            "get_generator",
            "cfG_overhead_factor",
            "numpy_distribution",
            "CFG_GAMMAS",
            "HUMANEVAL_TEMPERATURES",
            "HARNESS_DEFAULT_TEMPERATURE",
            "HARNESS_DEFAULT_TOP_P",
            "ANALYSIS_GAMMA",
        ]
    )
except Exception as _exc:  # pragma: no cover
    logger.debug("CFG sampler unavailable (torch missing?): %s", _exc)

try:  # pragma: no cover - exercised only when torch+transformers installed
    from .model_wrapper import (  # noqa: E402
        CFGModelWrapper,
        DualLogits,
        resolve_dtype,
        UNCONDITIONAL_MODES,
    )

    __all__.extend(
        ["CFGModelWrapper", "DualLogits", "resolve_dtype", "UNCONDITIONAL_MODES"]
    )
except Exception as _exc:  # pragma: no cover
    logger.debug("CFG model wrapper unavailable (torch/transformers missing?): %s", _exc)

try:  # pragma: no cover
    from .generator import (  # noqa: E402
        CFGGenerator,
        GenerationConfig,
        GenerationOutput,
        generate,
        truncate_at_stop_strings,
        HUMANEVAL_MAX_NEW_TOKENS,
        COT_MAX_NEW_TOKENS,
        ANSWER_MARKERS,
    )

    __all__.extend(
        [
            "CFGGenerator",
            "GenerationConfig",
            "GenerationOutput",
            "generate",
            "truncate_at_stop_strings",
            "HUMANEVAL_MAX_NEW_TOKENS",
            "COT_MAX_NEW_TOKENS",
            "ANSWER_MARKERS",
        ]
    )
except Exception as _exc:  # pragma: no cover
    logger.debug("CFG generator unavailable (torch missing?): %s", _exc)
