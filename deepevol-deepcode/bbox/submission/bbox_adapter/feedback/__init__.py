"""BBox-Adapter feedback package.

Exposes the GPT-4-simulated human-preference rater used as the positive-sample
selection function ``SEL(\\cdot)`` in BBox-Adapter's online adaptation loop
(Section 3.4, Appendix G/J).

Nothing in this package ever requests token probabilities, hidden states, or
gradients from the black-box LLM: only prompt strings go out and raw text comes
back.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .ai_feedback import (
    BEST_ANSWER_MARKER,
    DEFAULT_N_RANKED,
    RANKED_DATASETS,
    RANKED_MARKER,
    AIFeedback,
    AIFeedbackConfig,
    FeedbackSelection,
    build_ai_feedback,
    criteria_summary,
    extract_explanation,
    heuristic_select,
    parse_best_answer,
    parse_feedback_index,
    parse_ranked_answers,
    select_by_ai_feedback,
    split_positives_negatives,
)

__all__ = [
    "AIFeedback",
    "AIFeedbackConfig",
    "FeedbackSelection",
    "build_ai_feedback",
    "select_by_ai_feedback",
    "parse_feedback_index",
    "parse_best_answer",
    "parse_ranked_answers",
    "extract_explanation",
    "heuristic_select",
    "split_positives_negatives",
    "criteria_summary",
    "RANKED_DATASETS",
    "DEFAULT_N_RANKED",
    "BEST_ANSWER_MARKER",
    "RANKED_MARKER",
    "get_selector",
]


def get_selector(
    mode: str = "ai",
    *,
    dataset: Optional[str] = None,
    rater: Any = None,
    config: Optional[Union[AIFeedbackConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Optional[AIFeedback]:
    """Factory returning a positive-sample selector for a named SEL mode.

    Parameters
    ----------
    mode:
        One of ``{"ai", "ai_feedback", "feedback", "gpt4", "gpt-4", "default"}``
        which returns an :class:`AIFeedback` instance.  The ``"ground_truth"``,
        ``"ground-truth"``, ``"gt"`` and ``"combined"`` modes return ``None``
        because those selection strategies are implemented directly in
        ``training.buffers`` (they need dataset gold answers, not a rater).
    dataset:
        Dataset name forwarded to the selector (controls ranked vs best mode).
    rater:
        Optional pre-built rater client (duck-typed); if ``None`` the selector
        lazily builds a GPT-4 rater via ``llm.blackbox_client.build_rater``.
    config / kwargs:
        Forwarded to :class:`AIFeedback`.
    """
    key = (mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    ai_modes = {"", "ai", "ai_feedback", "feedback", "gpt4", "gpt_4", "default"}
    if key in ai_modes:
        if isinstance(config, AIFeedbackConfig):
            return AIFeedback(rater=rater, config=config, dataset=dataset, **kwargs)
        return build_ai_feedback(dataset=dataset, rater=rater, config=config, **kwargs)
    if key in {"ground_truth", "groundtruth", "gt", "combined"}:
        # Handled by training.buffers (needs gold answers); no rater required.
        return None
    raise ValueError(
        f"Unknown SEL mode {mode!r}; expected one of "
        f"{sorted(ai_modes | {'ground_truth', 'combined'})}"
    )
