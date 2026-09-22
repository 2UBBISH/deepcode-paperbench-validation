"""MMLU as the refinement dataset ``D_R`` of the FLAN-T5 experiments.

Sec. 4.1: "For FLAN-T5, we use MMLU, since the P3 dataset (including the test
split) is involved in pretraining the model."  The addendum specifies that the
*validation* split of the original MMLU release
(https://people.eecs.berkeley.edu/~hendrycks/data.tar) is used, and that the 57
subjects of that release are all used.

Zero-shot multiple-choice prompt (the paper does not spell the prompt out; a
letter-answer prompt is used so that Exact Match against the gold letter is the
same metric as in the P3 experiments):

    Question: <question>
    A. <choice A>
    B. <choice B>
    C. <choice C>
    D. <choice D>
    Answer:

The target is the gold choice letter.
"""
from __future__ import annotations

import csv
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

from .registry import MMLU_SUBJECTS
from .types import Dataset, Example

LETTERS = ["A", "B", "C", "D"]

PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def format_prompt(question: str, choices: Sequence[str]) -> str:
    choices = list(choices) + [""] * (4 - len(choices))
    return PROMPT_TEMPLATE.format(question=question, a=choices[0], b=choices[1], c=choices[2], d=choices[3])


def extract_answer_letter(text: str) -> str:
    """Map a free-form generation to an MMLU choice letter (A-D)."""
    if not text:
        return ""
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)
    stripped = text.strip().upper()
    return stripped[0] if stripped[:1] in LETTERS else ""


# --------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------
def _read_original_release(root: str, split: str) -> Optional[Dataset]:
    """Read the original MMLU release (``<root>/<split>/<subject>_<split>.csv``)."""
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        return None
    examples: Dataset = []
    for subject in MMLU_SUBJECTS:
        path = os.path.join(split_dir, f"{subject}_{split}.csv")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf8") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                choices = [row.get(l, "") for l in LETTERS]
                gold = str(row.get("answer", "")).strip().upper()[:1]
                examples.append(
                    Example(
                        input=format_prompt(row.get("question", ""), choices),
                        target=gold,
                        task=subject,
                        config=f"mmlu/{subject}",
                        idx=i,
                        meta={"subject": subject, "choices": choices},
                    )
                )
    return examples or None


def _read_hf_release(subjects: Sequence[str], split: str) -> Dataset:
    """Fallback: the mirror of MMLU on the HF hub (``cais/mmlu``)."""
    from datasets import load_dataset

    hf_split = {"val": "validation", "dev": "dev", "test": "test"}.get(split, split)
    examples: Dataset = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split=hf_split)
        for i, row in enumerate(ds):
            choices = list(row["choices"])
            gold = LETTERS[int(row["answer"])]
            examples.append(
                Example(
                    input=format_prompt(row["question"], choices),
                    target=gold,
                    task=subject,
                    config=f"mmlu/{subject}",
                    idx=i,
                    meta={"subject": subject, "choices": choices},
                )
            )
    return examples


def load_mmlu(
    split: str = "val",
    subjects: Optional[Sequence[str]] = None,
    root: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Dataset:
    """Load the MMLU validation split used as ``D_R`` for the FLAN-T5 experiments."""
    subjects = list(subjects) if subjects is not None else list(MMLU_SUBJECTS)
    examples: Optional[Dataset] = None
    if root and os.path.isdir(root):
        examples = _read_original_release(root, split)
    if examples is None:
        examples = _read_hf_release(subjects, split)
    if subjects and root:
        wanted = set(subjects)
        examples = [e for e in examples if e.task in wanted]
    if max_examples is not None and len(examples) > max_examples:
        examples = examples[:max_examples]
    return examples


def subject_histogram(examples: Iterable[Example]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for ex in examples:
        hist[ex.task] = hist.get(ex.task, 0) + 1
    return hist
