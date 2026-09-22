"""``bbox_adapter.data`` — datasets, splits, and answer extraction.

This package is the data layer of the BBox-Adapter reproduction
("Lightweight Adapting for Black-Box Large Language Models").  It provides:

* :mod:`bbox_adapter.data.dataset_specs` -- static, paper-faithful metadata for
  the five evaluation datasets (StrategyQA 2059/229, GSM8K 7473/1319,
  TruthfulQA 717/100, ScienceQA 2000/500, ToxiGen 2000/500), including the
  HuggingFace identifiers, answer-type tags, prompt keys and metrics.
* :mod:`bbox_adapter.data.loaders` -- uniform ``Example`` records and
  deterministic train/test splits.
* :mod:`bbox_adapter.data.answer_extraction` -- parsing of the paper's ``####``
  terminator, Yes/No (StrategyQA), numeric (GSM8K), multiple-choice (ScienceQA)
  and free-form / TruthfulQA-True+Info answers.

Only *text* is ever handled here: nothing in this package touches logprobs,
hidden states, or gradients of the black-box LLM.

Example
-------
>>> from bbox_adapter.data import load_dataset, accuracy, get_spec   # doctest: +SKIP
>>> spec = get_spec("strategyqa")
>>> train, test = load_dataset("strategyqa")
>>> accuracy(["#### Yes."], ["Yes"], spec.answer_type)
100.0
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # --- dataset specs -------------------------------------------------
    "DatasetSpec",
    "get_spec",
    "ALL_SPECS",
    "QA_SPECS",
    "SPEC_BY_NAME",
    "STRATEGYQA",
    "GSM8K",
    "TRUTHFULQA",
    "SCIENCEQA",
    "TOXIGEN",
    "PAPER_TABLE2",
    "PAPER_TABLE3",
    "PAPER_TABLE5",
    "PAPER_TABLE7",
    # --- loaders -------------------------------------------------------
    "Example",
    "load_split",
    "load_train",
    "load_test",
    "load_dataset",
    "split_sizes",
    "final_numeric_answer",
    "iter_questions",
    "DATASET_LOADERS",
    # --- answer extraction --------------------------------------------
    "ANSWER_TERMINATOR",
    "ANSWER_TYPE_YESNO",
    "ANSWER_TYPE_NUMERIC",
    "ANSWER_TYPE_MCQ",
    "ANSWER_TYPE_TRUTHFULQA",
    "ANSWER_TYPE_TOXIC",
    "ANSWER_TYPE_FREE",
    "extract_after_terminator",
    "contains_terminator",
    "split_steps",
    "extract_yesno",
    "extract_numeric",
    "extract_choice_index",
    "extract_final_answer",
    "normalize_answer",
    "is_correct",
    "grade_generation",
    "extract_answers",
    "accuracy",
    "format_answer",
    "truthfulqa_score",
    "truthfulqa_score_lexical",
    "true_info_rate",
    "toxigen_prompt_text",
    "toxicity_is_toxic",
    # --- helpers -------------------------------------------------------
    "get_dataset",
    "describe",
    "list_datasets",
]


# ---------------------------------------------------------------------------
# dataset specs
# ---------------------------------------------------------------------------
from .dataset_specs import (  # noqa: E402
    ALL_SPECS,
    GSM8K,
    PAPER_TABLE2,
    PAPER_TABLE3,
    PAPER_TABLE5,
    PAPER_TABLE7,
    QA_SPECS,
    SCIENCEQA,
    SPEC_BY_NAME,
    STRATEGYQA,
    TOXIGEN,
    TRUTHFULQA,
    DatasetSpec,
    get_spec,
)

# ---------------------------------------------------------------------------
# loaders (optional: requires the ``datasets`` package at call time only)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - defensive, the module itself is import-light
    from .loaders import (  # noqa: E402
        DATASET_LOADERS,
        Example,
        final_numeric_answer,
        iter_questions,
        load_dataset,
        load_split,
        load_test,
        load_train,
        split_sizes,
    )
except Exception:  # pragma: no cover
    DATASET_LOADERS = {}  # type: ignore[assignment]

    def load_split(*args: Any, **kwargs: Any):  # type: ignore[misc]
        raise ImportError("bbox_adapter.data.loaders is unavailable")

    load_train = load_split  # type: ignore[assignment]
    load_test = load_split  # type: ignore[assignment]
    load_dataset = load_split  # type: ignore[assignment]

    def split_sizes(name: str) -> Dict[str, int]:  # type: ignore[misc]
        spec = get_spec(name)
        return {"train": spec.n_train, "test": spec.n_test}

    def iter_questions(examples: Any):  # type: ignore[misc]
        for ex in examples:
            yield ex.question if hasattr(ex, "question") else str(ex)

    def final_numeric_answer(answer_text: Any) -> Optional[str]:  # type: ignore[misc]
        from .answer_extraction import extract_numeric

        return extract_numeric(str(answer_text) if answer_text is not None else "")

    Example = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# answer extraction
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .answer_extraction import (  # noqa: E402
        ANSWER_TERMINATOR,
        ANSWER_TYPE_FREE,
        ANSWER_TYPE_MCQ,
        ANSWER_TYPE_NUMERIC,
        ANSWER_TYPE_TOXIC,
        ANSWER_TYPE_TRUTHFULQA,
        ANSWER_TYPE_YESNO,
        accuracy,
        contains_terminator,
        extract_after_terminator,
        extract_answers,
        extract_choice_index,
        extract_final_answer,
        extract_numeric,
        extract_yesno,
        format_answer,
        grade_generation,
        is_correct,
        normalize_answer,
        split_steps,
        toxicity_is_toxic,
        toxigen_prompt_text,
        true_info_rate,
        truthfulqa_score,
        truthfulqa_score_lexical,
    )
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# convenience helpers
# ---------------------------------------------------------------------------
def get_dataset(name: str, *, split: str = "test", limit: Optional[int] = None, seed: int = 0):
    """Load a single split of a named dataset.

    Thin wrapper around :func:`bbox_adapter.data.loaders.load_split` that
    accepts the loose dataset aliases handled by ``get_spec``.

    Parameters
    ----------
    name:
        Dataset name (``strategyqa``/``gsm8k``/``truthfulqa``/``scienceqa``/``toxigen``).
    split:
        ``"train"`` or ``"test"``.
    limit:
        Optional debug limit on the number of examples.
    seed:
        Sub-sampling seed.
    """
    return load_split(name, split=split, limit=limit, seed=seed)


def list_datasets() -> List[str]:
    """Return the names of the five paper datasets."""
    return [spec.name for spec in ALL_SPECS]


def describe() -> Dict[str, Any]:
    """Return a metadata dict describing the datasets handled here."""
    return {
        "module": "bbox_adapter.data",
        "paper": "Lightweight Adapting for Black-Box Large Language Models",
        "sections": ["F.1", "4.1", "Appendix E", "Appendix J"],
        "n_datasets": len(ALL_SPECS),
        "datasets": {
            spec.name: {
                "hf_path": spec.hf_path,
                "n_train": spec.n_train,
                "n_test": spec.n_test,
                "answer_type": spec.answer_type,
                "prompt_key": spec.prompt_key,
                "metric": spec.metric,
            }
            for spec in ALL_SPECS
        },
    }


def __getattr__(name: str):  # pragma: no cover - PEP 562 lazy access
    """Lazily expose loader functions that may be missing in bare installs."""
    if name in ("loaders", "dataset_specs", "answer_extraction"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _self_test() -> Dict[str, Any]:
    """Dependency-free smoke test (``python -m bbox_adapter.data``)."""
    spec = get_spec("gsm8k")
    assert spec.n_train == 7473 and spec.n_test == 1319, spec
    assert get_spec("truthfulqa").metric == "true_info"
    sizes = split_sizes("strategyqa")
    assert sizes["train"] == 2059 and sizes["test"] == 229, sizes
    assert "#####" not in (ANSWER_TERMINATOR or "")
    assert format_answer(7, ANSWER_TYPE_NUMERIC) != ""
    assert accuracy(["#### Yes."], ["Yes"], ANSWER_TYPE_YESNO) == 100.0
    return {
        "ok": True,
        "datasets": list_datasets(),
        "splits": {name: split_sizes(name) for name in list_datasets()},
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(_self_test(), indent=2, default=str))
