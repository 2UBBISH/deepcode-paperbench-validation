"""Evaluation layer for the CFG-LM reproduction.

This package groups every scoring / benchmarking utility used to reproduce the
paper's evaluation tables and figures:

* :mod:`src.eval.harness_cfg`     -- EleutherAI LM Evaluation Harness shim that
  applies the Classifier-Free Guidance combination (Eq. 7) to the next-token
  logits before scoring multiple-choice / QA tasks (Section 3.1, Appendix C.1).
* :mod:`src.eval.triviaqa_match`  -- substring-match scoring for TriviaQA
  (Appendix C.1 clarification: no exact match, containment instead).
* :mod:`src.eval.cot_eval`        -- chain-of-thought valid-answer parsing and
  task accuracy (Section 3.2, Appendix C.5; Figure 2 / Figure 17).
* :mod:`src.eval.pass_at_k`       -- unbiased pass@k estimator
  (Chen et al. 2021, footnote 4; Tables 2 / 7 / 8 / 9).
* :mod:`src.eval.humaneval_eval`  -- HumanEval code-generation and execution
  harness used by the pass@k sweep.

Heavy dependencies (``transformers``, ``datasets``, the LM Evaluation Harness)
are optional: every submodule is imported lazily inside a ``try/except`` block so
that the pure-Python scoring helpers (substring match, pass@k estimator, CoT
answer parsing) remain usable in a minimal CPU-only environment.
"""

from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__: List[str] = []

# --------------------------------------------------------------------------- #
# TriviaQA substring-match scoring (pure python -- always available).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from .triviaqa_match import (  # noqa: F401
        normalize_answer,
        substring_match,
        triviaqa_score,
        TriviaQAScorer,
    )

    __all__ += [
        "normalize_answer",
        "substring_match",
        "triviaqa_score",
        "TriviaQAScorer",
    ]
except Exception as exc:  # pragma: no cover - import guard
    logger.debug("src.eval.triviaqa_match unavailable: %s", exc)

# --------------------------------------------------------------------------- #
# CoT evaluation (pure python parsing + accuracy aggregation).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from .cot_eval import (  # noqa: F401
        extract_answer,
        parse_chain,
        is_valid_chain,
        evaluate_cot,
        CoTResult,
    )

    __all__ += [
        "extract_answer",
        "parse_chain",
        "is_valid_chain",
        "evaluate_cot",
        "CoTResult",
    ]
except Exception as exc:  # pragma: no cover - import guard
    logger.debug("src.eval.cot_eval unavailable: %s", exc)

# --------------------------------------------------------------------------- #
# pass@k estimator (numpy only).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from .pass_at_k import (  # noqa: F401
        estimate_pass_at_k,
        pass_at_k,
        compute_pass_at_k,
    )

    __all__ += [
        "estimate_pass_at_k",
        "pass_at_k",
        "compute_pass_at_k",
    ]
except Exception as exc:  # pragma: no cover - import guard
    logger.debug("src.eval.pass_at_k unavailable: %s", exc)

# --------------------------------------------------------------------------- #
# LM Evaluation Harness shim (needs torch / the harness package).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from .harness_cfg import (  # noqa: F401
        CFGHarnessLM,
        HARNESS_GAMMAS,
        HARNESS_TASKS,
    )

    __all__ += [
        "CFGHarnessLM",
        "HARNESS_GAMMAS",
        "HARNESS_TASKS",
    ]
except Exception as exc:  # pragma: no cover - import guard
    logger.debug("src.eval.harness_cfg unavailable: %s", exc)

# --------------------------------------------------------------------------- #
# HumanEval execution harness (needs code execution utilities).
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from .humaneval_eval import (  # noqa: F401
        run_humaneval_problem,
        evaluate_humaneval,
        HUMANEVAL_PASS_AT_K,
    )

    __all__ += [
        "run_humaneval_problem",
        "evaluate_humaneval",
        "HUMANEVAL_PASS_AT_K",
    ]
except Exception as exc:  # pragma: no cover - import guard
    logger.debug("src.eval.humaneval_eval unavailable: %s", exc)
