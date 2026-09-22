"""BBox-Adapter adapted-inference package.

This package implements the *inference-time* half of BBox-Adapter: the black-box
LLM is treated purely as a text-only proposal generator, while the trained scalar
energy adapter ``g_theta`` acts as the evaluator that ranks/prunes proposals.

Contents
--------
``beam_search``  -- sentence-level beam search (Section 3.3, Eq. 4) with an
                    adapter-driven top-k pruning, plus the cheaper single-step
                    variant in which the black-box model emits complete answers
                    once and the adapter only ranks them (Table 4).
``selector``     -- final answer selection: return the surface answer of the
                    highest-scoring hypothesis/answer group.

Nothing in this package ever requests logprobs, hidden states, or gradients from
the black-box LLM; only ``prompt``, ``n``, ``temperature`` and ``max_len`` are
forwarded to the generator.

The symbols are re-exported here so the rest of the codebase can simply do::

    from bbox_adapter.inference import (
        SentenceBeamSearch, BeamSearchConfig, beam_search, single_step_search,
        AnswerSelector, select_answer,
    )

Both modules are imported defensively: ``selector`` depends on
``beam_search`` for ``BeamSearchResult``/``resolve_adapter_scores``, so the
selector import is guarded to keep the beam-search path usable on its own.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sentence-level beam search (adapted inference)
# ---------------------------------------------------------------------------
from .beam_search import (  # noqa: F401
    DEFAULT_MAX_STEPS,
    STOP_SIGNAL,
    BeamSearchConfig,
    BeamSearchResult,
    Hypothesis,
    SentenceBeamSearch,
    beam_search,
    resolve_adapter_scores,
    score_candidates,
    single_step_rank,
    single_step_search,
    topk_indices,
)

__all__ = [
    # beam search
    "SentenceBeamSearch",
    "BeamSearchConfig",
    "Hypothesis",
    "BeamSearchResult",
    "beam_search",
    "single_step_search",
    "single_step_rank",
    "resolve_adapter_scores",
    "score_candidates",
    "topk_indices",
    "STOP_SIGNAL",
    "DEFAULT_MAX_STEPS",
]

# ---------------------------------------------------------------------------
# Final answer selection (defensive import: keeps beam search usable alone)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    from .selector import (  # noqa: F401
        DEFAULT_AGGREGATION,
        SCORE_AGGREGATIONS,
        AnswerSelector,
        Candidate,
        SelectionResult,
        SelectorConfig,
        aggregate_scores,
        best_by_answer,
        best_by_score,
        logsumexp,
        normalize_candidates,
        rank_candidates,
        score_tie_break,
        select_answer,
        select_from_result,
    )

    _SELECTOR_EXPORTS = [
        "AnswerSelector",
        "SelectorConfig",
        "Candidate",
        "SelectionResult",
        "select_answer",
        "select_from_result",
        "normalize_candidates",
        "rank_candidates",
        "aggregate_scores",
        "best_by_score",
        "best_by_answer",
        "score_tie_break",
        "logsumexp",
        "SCORE_AGGREGATIONS",
        "DEFAULT_AGGREGATION",
    ]
    __all__.extend(_SELECTOR_EXPORTS)
except Exception:  # pragma: no cover - selector is optional for beam search only
    pass


def get_inference(mode: str = "beam", **kwargs):
    """Factory selecting the adapted-inference strategy.

    Parameters
    ----------
    mode:
        ``"beam"``/``"full"``/``"sentence"``  -> :class:`SentenceBeamSearch`
        ``"single_step"``/``"single"``/``"rank"`` -> single-step ranking helper
            (returns ``None``; callers should use ``single_step_search`` /
            ``single_step_rank`` directly).
    kwargs:
        Forwarded to the constructed object/config.

    Returns
    -------
    SentenceBeamSearch | None
    """
    normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"beam", "full", "sentence", "beam_search", "default", ""}:
        return SentenceBeamSearch(**kwargs)
    if normalized in {"single_step", "single", "rank", "ranking"}:
        # Single-step inference is a functional wrapper; expose it for symmetry.
        return None
    raise ValueError(
        f"Unknown inference mode {mode!r}; expected one of "
        "{'beam','full','sentence','single_step','rank'}."
    )
