"""Dataset loaders for the five BBox-Adapter evaluation datasets.

This module turns the static metadata in :mod:`bbox_adapter.data.dataset_specs`
into concrete, uniformly shaped Python examples.

Paper references
----------------
* §F.1 Additional Dataset Details
    - GSM8K: 7473 train / 1319 test samples.
    - StrategyQA: 2059 train / 229 test samples.
    - TruthfulQA: 100 randomly sampled questions as test set, remaining 717 as
      training set.
    - ScienceQA: questions requiring image input are *excluded*; 2000 questions
      are randomly selected for training and 500 for testing, each drawn from the
      dataset's original training / testing subsets respectively.
* §E (ToxiGen): 2000 samples for training and 500 for testing; data are prompts
  containing hateful statements about demographic groups.
* §4.1 Experiment Setup: the four QA tasks + the three labelled-data settings.
* §J Prompt Design: question/answer field conventions (``####`` terminators,
  ``Yes./No.``, multiple-choice indices).

Every loader returns :class:`Example` records so that downstream components
(prompt builder, black-box client, adapter, buffers, evaluator) never need to
know about HuggingFace field names.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .dataset_specs import DatasetSpec, get_spec

__all__ = [
    "Example",
    "load_split",
    "load_train",
    "load_test",
    "load_dataset",
    "split_sizes",
    "final_numeric_answer",
    "DATASET_LOADERS",
]


# --------------------------------------------------------------------------- #
# Example container
# --------------------------------------------------------------------------- #
@dataclass
class Example:
    """A single question/answer record, dataset agnostic.

    Attributes
    ----------
    uid:
        Stable identifier (hash of dataset name + question text) used to key the
        positive/negative sample buffers.
    question:
        The question (or ToxiGen prompt) shown to the black-box LLM.
    answer:
        The *normalized* reference answer used for scoring. ``None`` when the
        dataset provides no single ground truth (AI-feedback setting).
    answer_raw:
        The raw dataset answer string (e.g. the full GSM8K solution including
        the ``####`` terminator).
    choices:
        Multiple-choice options (ScienceQA), else ``None``.
    steps:
        Reference reasoning steps, if the dataset provides them (GSM8K's
        solution is the only one that does; StrategyQA provides none).
    meta:
        Free-form per-dataset extras (``correct_answers``/``incorrect_answers``
        for TruthfulQA, ``target_group`` for ToxiGen, ...).
    """

    uid: str
    question: str
    answer: Optional[str] = None
    answer_raw: Optional[str] = None
    choices: Optional[List[str]] = None
    steps: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------- #
    def prompt_fields(self) -> Dict[str, Any]:
        """Fields consumed by :mod:`bbox_adapter.llm.prompts`."""
        return {"question": self.question, "choices": self.choices}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        q = self.question if len(self.question) <= 60 else self.question[:57] + "..."
        return f"Example(uid={self.uid[:8]}, q={q!r}, answer={self.answer!r})"


def _make_uid(dataset: str, key: str) -> str:
    digest = hashlib.sha1(f"{dataset}::{key}".encode("utf-8")).hexdigest()
    return f"{dataset}-{digest[:16]}"


# --------------------------------------------------------------------------- #
# Answer helpers (dataset-local; full parsing lives in answer_extraction.py)
# --------------------------------------------------------------------------- #
def final_numeric_answer(answer_text: Any) -> Optional[str]:
    """Extract the value after the final ``####`` marker (GSM8K convention).

    Returns the trailing string with commas removed, or ``None`` if no marker is
    present.
    """
    if answer_text is None:
        return None
    text = str(answer_text)
    if "####" not in text:
        return None
    tail = text.split("####")[-1].strip()
    tail = tail.replace(",", "").replace("$", "").strip()
    # Strip a trailing period, e.g. "#### 18." -> "18"
    tail = tail.rstrip(".").strip()
    return tail or None


def _as_yes_no(value: Any) -> Optional[str]:
    """Normalize a StrategyQA boolean answer to ``"Yes"`` / ``"No"``."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1"}:
        return "Yes"
    if text in {"false", "no", "n", "0"}:
        return "No"
    return None


# --------------------------------------------------------------------------- #
# HuggingFace plumbing
# --------------------------------------------------------------------------- #
_HF_CACHE: Dict[tuple, Any] = {}


def _hf_load(path: str, config: Optional[str], split: str):
    """Load a HuggingFace dataset split, cached in-process.

    Mirrors the paper's HF usage (§4.1 Implementations, §F.1) without pinning a
    revision, so any available snapshot of the dataset can be used.
    """
    key = (path, config, split)
    if key in _HF_CACHE:
        return _HF_CACHE[key]
    try:
        from datasets import load_dataset  # imported lazily: optional at import time
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The `datasets` package is required to load "
            f"{path!r}. Install it with `pip install datasets`."
        ) from exc

    if config:
        try:
            ds = load_dataset(path, config, split=split)
        except Exception:
            ds = load_dataset(path, split=split)
    else:
        ds = load_dataset(path, split=split)
    _HF_CACHE[key] = ds
    return ds


def _first_present(row: Dict[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _subsample(rows: List[Any], n: int, seed: int) -> List[Any]:
    """Deterministic random subsample (paper: 'randomly selected' splits)."""
    if n is None or n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


# --------------------------------------------------------------------------- #
# Per-dataset builders
# --------------------------------------------------------------------------- #
def _build_strategyqa(spec: DatasetSpec, split: str, seed: int, limit: Optional[int]):
    ds = _hf_load(spec.hf_path, spec.hf_config, spec.train_split if split == "train" else spec.test_split)
    out: List[Example] = []
    for row in ds:
        row = dict(row)
        question = _first_present(row, ["question", "Question", "text"])
        if question is None:
            continue
        # wics/strategy-qa stores `answer` as a bool; some mirrors expose a
        # string. Either way we normalize to Yes/No.
        ans = _first_present(row, ["answer", "Answer", "label"])
        norm = _as_yes_no(ans)
        raw = str(ans) if ans is not None else None
        out.append(
            Example(
                uid=_make_uid("strategyqa", str(question)),
                question=str(question).strip(),
                answer=norm,
                answer_raw=raw,
                steps=None,
                meta={"qid": _first_present(row, ["qid", "id"])},
            )
        )
    if limit:
        out = out[:limit]
    return out


def _build_gsm8k(spec: DatasetSpec, split: str, seed: int, limit: Optional[int]):
    ds = _hf_load(spec.hf_path, spec.hf_config, spec.train_split if split == "train" else spec.test_split)
    out: List[Example] = []
    for row in ds:
        row = dict(row)
        question = _first_present(row, ["question", "problem", "text"])
        answer_raw = _first_present(row, ["answer", "solution", "target"])
        if question is None or answer_raw is None:
            continue
        answer_raw = str(answer_raw)
        steps = [
            ln.strip()
            for ln in answer_raw.split("####")[0].split("\n")
            if ln.strip()
        ]
        out.append(
            Example(
                uid=_make_uid("gsm8k", str(question)),
                question=str(question).strip(),
                answer=final_numeric_answer(answer_raw),
                answer_raw=answer_raw,
                steps=steps or None,
                meta={},
            )
        )
    if limit:
        out = out[:limit]
    return out


def _build_truthfulqa(spec: DatasetSpec, split: str, seed: int, limit: Optional[int]):
    """717 train / 100 test, both carved from the 817-sample `validation` split.

    §F.1: "We randomly sample 100 questions from the dataset as a test set and
    use the remaining 717 samples as the training set."
    """
    ds = _hf_load(spec.hf_path, spec.hf_config, spec.train_split)
    rows = [dict(r) for r in ds]
    n_test = spec.n_test or 100
    rng = random.Random(spec.extra.get("random_test_seed", seed))
    idx = list(range(len(rows)))
    test_idx = set(rng.sample(idx, min(n_test, len(rows))))
    if split == "test":
        chosen = [rows[i] for i in range(len(rows)) if i in test_idx]
    else:
        chosen = [rows[i] for i in range(len(rows)) if i not in test_idx]

    out: List[Example] = []
    for row in chosen:
        question = _first_present(row, ["question", "Question"])
        if question is None:
            continue
        best = _first_present(row, ["best_answer", "Best Answer", "answer"])
        correct = _first_present(row, ["correct_answers", "Correct Answers"]) or []
        incorrect = _first_present(row, ["incorrect_answers", "Incorrect Answers"]) or []
        if isinstance(correct, str):
            correct = [correct]
        if isinstance(incorrect, str):
            incorrect = [incorrect]
        out.append(
            Example(
                uid=_make_uid("truthfulqa", str(question)),
                question=str(question).strip(),
                answer=str(best).strip() if best is not None else None,
                answer_raw=str(best).strip() if best is not None else None,
                choices=None,
                # TruthfulQA reference solutions double as the "step-wise"
                # solution used for the positive buffer.
                steps=[str(best)] if best is not None else None,
                meta={
                    "correct_answers": [str(a) for a in correct],
                    "incorrect_answers": [str(a) for a in incorrect],
                    "category": row.get("category"),
                },
            )
        )
    if limit:
        out = out[:limit]
    return out


def _build_scienceqa(spec: DatasetSpec, split: str, seed: int, limit: Optional[int]):
    """Image-free ScienceQA subset: 2000 train / 500 test, random selection."""
    ds = _hf_load(spec.hf_path, spec.hf_config, spec.train_split if split == "train" else spec.test_split)
    rows = []
    for r in ds:
        r = dict(r)
        # §F.1: "We excluded questions requiring image input".
        if spec.extra.get("exclude_image", True) and r.get("image") is not None:
            continue
        question = _first_present(r, ["question", "Question"])
        choices = _first_present(r, ["choices", "Choices"])
        ans_idx = _first_present(r, ["answer", "Answer", "label"])
        # Some ScienceQA configurations fold the choices into the question text.
        if choices is None and r.get("choices_text"):
            choices = r["choices_text"]
        if isinstance(choices, str):
            choices = [c.strip() for c in choices.split("|")]
        if question is None or ans_idx is None or not choices:
            continue
        try:
            ans_idx = int(ans_idx)
        except (TypeError, ValueError):
            continue
        if ans_idx < 0 or ans_idx >= len(choices):
            continue
        rows.append((r, str(question).strip(), [str(c).strip() for c in choices], ans_idx))

    n_target = spec.n_train if split == "train" else spec.n_test
    rows = _subsample(rows, n_target or 0, spec.extra.get("random_subset_seed", seed))

    out: List[Example] = []
    for r, question, choices, ans_idx in rows:
        out.append(
            Example(
                uid=_make_uid("scienceqa", r.get("id", question) if r.get("id") is not None else question),
                question=question,
                # The answer is the multiple-choice *index* (Appendix J: "#### 1").
                answer=str(ans_idx),
                answer_raw=str(ans_idx),
                choices=choices,
                steps=[str(ans_idx)],
                meta={
                    "lecture": r.get("lecture"),
                    "solution": r.get("solution"),
                    "hint": r.get("hint"),
                    "subject": r.get("subject"),
                    "topic": r.get("topic"),
                    "choice_text": choices[ans_idx],
                },
            )
        )
    if limit:
        out = out[:limit]
    return out


def _build_toxigen(spec: DatasetSpec, split: str, seed: int, limit: Optional[int]):
    """ToxiGen prompts (§E): 2000 train / 500 test prompt statements."""
    ds = _hf_load(spec.hf_path, spec.hf_config, spec.train_split)
    rows = [dict(r) for r in ds]
    rows = _subsample(rows, spec.n_train or 0, spec.extra.get("random_subset_seed", seed))

    out: List[Example] = []
    for i, row in enumerate(rows):
        text = _first_present(row, ["text", "prompt", "Text"])
        if text is None:
            continue
        target = _first_present(row, ["target_group", "target", "group", "Target"])
        # Reference answer: a neutral continuation. ToxiGen itself has no gold
        # continuation, so the adapter relies on AI feedback / outcome
        # supervision; the raw annotated toxicity label is kept for reference.
        out.append(
            Example(
                uid=_make_uid("toxigen", f"{i}-{str(text)[:64]}"),
                question=str(text).strip(),
                answer=None,
                answer_raw=None,
                steps=None,
                meta={
                    "target_group": target,
                    "toxicity_human": _first_present(row, ["toxicity_human", "toxicity_annotated"]),
                    "split": split,
                },
            )
        )
    if limit:
        out = out[:limit]
    return out


DATASET_LOADERS = {
    "strategyqa": _build_strategyqa,
    "gsm8k": _build_gsm8k,
    "truthfulqa": _build_truthfulqa,
    "scienceqa": _build_scienceqa,
    "toxigen": _build_toxigen,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_split(
    name: str,
    split: str = "train",
    limit: Optional[int] = None,
    seed: int = 0,
) -> List[Example]:
    """Load ``split`` (``"train"`` or ``"test"``) of dataset ``name``.

    Parameters
    ----------
    name:
        One of ``strategyqa``, ``gsm8k``, ``truthfulqa``, ``scienceqa``,
        ``toxigen`` (separator/case insensitive).
    split:
        ``"train"``, ``"test"``, or a raw HF split name which is passed through.
    limit:
        Keep only the first ``limit`` examples (debugging / smoke tests).
    seed:
        Seed for the deterministic random subsampling of the QA datasets.
    """
    spec = get_spec(name)
    builder = DATASET_LOADERS[spec.name]
    if split not in {"train", "test"}:
        # Allow raw HF split names (e.g. "validation") to fall through.
        if split == spec.train_split:
            split = "train"
        elif split == spec.test_split:
            split = "test"
        else:
            raise ValueError(
                f"{name}: unknown split {split!r}; expected 'train' or 'test'."
            )
    return builder(spec, split, seed, limit)


def load_train(name: str, limit: Optional[int] = None, seed: int = 0) -> List[Example]:
    """Training split (labels available: used for the positive buffers)."""
    return load_split(name, "train", limit=limit, seed=seed)


def load_test(name: str, limit: Optional[int] = None, seed: int = 0) -> List[Example]:
    """Held-out evaluation split."""
    return load_split(name, "test", limit=limit, seed=seed)


def load_dataset(name: str, limit: Optional[int] = None, seed: int = 0):
    """Convenience helper returning ``(train_examples, test_examples)``."""
    return (
        load_train(name, limit=limit, seed=seed),
        load_test(name, limit=limit, seed=seed),
    )


def split_sizes(name: str) -> Dict[str, int]:
    """Paper-reported split sizes (§F.1, §E) for dataset ``name``."""
    spec = get_spec(name)
    return {"train": spec.n_train, "test": spec.n_test}


def iter_questions(examples: Iterable[Example]) -> Iterable[str]:
    """Yield only the question strings (handy for prompt construction)."""
    for ex in examples:
        yield ex.question


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import argparse

    parser = argparse.ArgumentParser(description="Inspect BBox-Adapter dataset splits.")
    parser.add_argument("dataset", nargs="?", default="strategyqa")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    for split in ("train", "test"):
        try:
            exs = load_split(args.dataset, split, limit=args.limit)
        except Exception as exc:  # pragma: no cover
            print(f"{args.dataset}/{split}: FAILED -> {exc}")
            continue
        print(f"{args.dataset}/{split}: {len(exs)} examples")
        for ex in exs[: args.limit]:
            print("   ", ex)
