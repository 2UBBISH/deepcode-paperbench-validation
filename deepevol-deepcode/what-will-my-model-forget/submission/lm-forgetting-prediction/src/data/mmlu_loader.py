"""MMLU (Massive Multitask Language Understanding) loader.

This module builds the mispredicted-example pool ``D_R`` for the FLAN-T5
experiments of the paper *"What Will My Model Forget? Forecasting Forgotten
Examples in Language Model Refinement"*.

For FLAN-T5 models the refinement (online) examples are taken from the MMLU
validation split: all 57 subjects (all variants of the multiple-choice
questions).  Examples are rendered in the canonical A/B/C/D multiple-choice
format and carry a single-reference ``target`` (the letter answer), so that the
SQuAD-2.0-style exact-match metric of :mod:`src.data.em_eval` can be used both
for collecting ``D_R`` (mispredicted examples) and for the Edit Success Rate.

Returned examples follow the exact same schema as :mod:`src.data.p3_loader`::

    {
        "id":      "<subject>_<split>_<index>",
        "task":    "<subject>",
        "split":   "validation",
        "template_name": "mmlu_abc",
        "input":   "Question: ...\nA. ...\nB. ...\nC. ...\nD. ...\nAnswer:",
        "target":  "B",
        "references": ["B"],
        "raw":     {...},          # the original MMLU fields, when available
    }

Three loading back-ends are tried, in order:

1. the local original Berkeley MMLU release (``data/val/*.csv`` or
   ``data/test/*.csv`` as distributed in ``hendrycks/test``),
2. a HuggingFace ``datasets`` build of MMLU (``cais/mmlu`` / ``lukaemon/mmlu`` /
   ``hails/mmlu_no_train``),
3. any pre-tokenized JSON/JSONL file placed under ``<data_dir>/<subject>.json``.
"""

from __future__ import annotations

import csv
import glob
import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "MMLU_SUBJECTS",
    "MMLU_ANSWER_CHOICES",
    "DEFAULT_MMLU_DATASET_IDS",
    "format_mmlu_example",
    "load_mmlu_task",
    "load_mmlu_tasks",
    "load_mmlu_dataset",
    "flatten_task_dict",
]


# --------------------------------------------------------------------------- #
# 57 MMLU subjects (validation split used for FLAN-T5's D_R)
# --------------------------------------------------------------------------- #
MMLU_SUBJECTS: List[str] = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

MMLU_ANSWER_CHOICES: List[str] = ["A", "B", "C", "D"]

DEFAULT_MMLU_DATASET_IDS: Sequence[str] = (
    "cais/mmlu",
    "hails/mmlu_no_train",
    "lukaemon/mmlu",
)


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #
def format_mmlu_example(
    question: str,
    choices: Sequence[str],
    answer: Optional[str] = None,
    question_prefix: str = "Question: ",
    answer_prefix: str = "\nAnswer:",
) -> Dict[str, Any]:
    """Render one MMLU multiple-choice question in the canonical A/B/C/D form.

    ``choices`` may either be the raw answer texts (4 entries) or already-formatted
    lines such as ``"A. 42"`` / ``"B) 42"``; in the latter case the option letters
    are re-derived.  ``answer`` should be the option letter (``"A"``-``"D"``); a
    raw answer text is also accepted and mapped back to its letter.
    """
    letters = MMLU_ANSWER_CHOICES
    rendered_choices: List[str] = []
    raw_texts: List[str] = []

    for idx, choice in enumerate(choices):
        text = str(choice).strip()
        # Strip a leading option marker ("A.", "A)", "A -") if present.
        if len(text) >= 2 and text[0].upper() in letters and text[1] in ".)-:":
            text = text[2:].strip()
        rendered_choices.append(f"{letters[idx]}. {text}")
        raw_texts.append(text)

    input_ids = (
        f"{question_prefix}{str(question).strip()}\n"
        + "\n".join(rendered_choices)
        + answer_prefix
    )

    target: Optional[str] = None
    if answer is not None:
        ans = str(answer).strip()
        if len(ans) == 1 and ans.upper() in letters:
            target = ans.upper()
        else:
            # Fall back to matching the answer text against the choices.
            for idx, text in enumerate(raw_texts):
                if text.lower() == ans.lower():
                    target = letters[idx]
                    break
    if target is None and answer is not None:
        logger.debug("Could not map MMLU answer %r to an option letter.", answer)

    return {
        "input": input_ids,
        "target": target,
        "choices": rendered_choices,
        "answer_texts": raw_texts,
    }


# --------------------------------------------------------------------------- #
# Back-ends
# --------------------------------------------------------------------------- #
def _examples_from_local_csv(
    task_name: str, split: str, data_dir: str
) -> List[Dict[str, Any]]:
    """Load the original Berkeley MMLU CSVs (``data/{val,test,dev}/<subject>_test.csv``)."""
    candidates = []
    subdirs = {
        "validation": ["val", "validation", "dev"],
        "test": ["test"],
        "train": ["auxiliary_train", "train"],
    }.get(split, [split])
    for sub in subdirs:
        candidates.extend(
            glob.glob(os.path.join(data_dir, sub, f"{task_name}_*.csv"))
        )
        candidates.extend(glob.glob(os.path.join(data_dir, sub, f"{task_name}.csv")))
    if not candidates:
        return []

    # Prefer the shortest file name (generally "<subject>_test.csv").
    path = sorted(candidates, key=lambda p: (len(os.path.basename(p)), p))[0]
    examples: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for idx, row in enumerate(reader):
            if len(row) < 6:
                continue
            question = row[0]
            choices = row[1:5]
            answer = row[5]
            if idx == 0 and str(answer).strip().lower() in {
                "answer",
                "answer_key",
                "answers",
            }:
                continue  # header row
            rendered = format_mmlu_example(question, choices, answer)
            examples.append(_wrap(task_name, split, idx, rendered, path))
    logger.info("Loaded %d MMLU examples for '%s' from %s", len(examples), task_name, path)
    return examples


def _examples_from_local_json(
    task_name: str, split: str, data_dir: str, path: str
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        parsed = json.loads(text)
        rows = parsed if isinstance(parsed, list) else parsed.get("data", [])
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = row.get("question", row.get("input", ""))
        choices = row.get("choices", row.get("options", []))
        answer = row.get("answer", row.get("target"))
        rendered = format_mmlu_example(question, choices, answer)
        ex = _wrap(task_name, split, idx, rendered, path)
        ex["raw"] = row
        examples.append(ex)
    logger.info("Loaded %d MMLU examples for '%s' from %s", len(examples), task_name, path)
    return examples


def _examples_from_hf(task_name: str, split: str) -> List[Dict[str, Any]]:
    """Load MMLU through HuggingFace ``datasets`` (config ``task_name``)."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return []

    hf_split = "validation" if split in {"validation", "val", "dev"} else split
    last_err: Optional[Exception] = None
    for dataset_id in DEFAULT_MMLU_DATASET_IDS:
        for name in (task_name, task_name.replace("_", "-")):
            try:
                ds = load_dataset(dataset_id, name, split=hf_split)
            except Exception as err:  # noqa: BLE001 - try the next id/name
                last_err = err
                continue
            examples: List[Dict[str, Any]] = []
            for idx, row in enumerate(ds):
                choices = row.get("choices") or [
                    row.get("A"),
                    row.get("B"),
                    row.get("C"),
                    row.get("D"),
                ]
                answer = row.get("answer", row.get("target"))
                if isinstance(answer, int):
                    answer = MMLU_ANSWER_CHOICES[answer]
                rendered = format_mmlu_example(row.get("question", ""), choices, answer)
                ex = _wrap(task_name, split, idx, rendered, f"{dataset_id}:{name}")
                ex["raw"] = {k: v for k, v in row.items() if k != "choices"}
                examples.append(ex)
            logger.info(
                "Loaded %d MMLU examples for '%s' from HF %s (%s)",
                len(examples),
                task_name,
                dataset_id,
                name,
            )
            return examples
    logger.warning("Could not load MMLU subject '%s' from HF: %s", task_name, last_err)
    return []


def _wrap(
    task_name: str, split: str, idx: int, rendered: Dict[str, Any], source: str
) -> Dict[str, Any]:
    """Wrap a rendered MMLU question into the shared example schema."""
    example: Dict[str, Any] = {
        "id": f"{task_name}_{split}_{idx}",
        "task": task_name,
        "split": split,
        "template_name": "mmlu_abc",
        "input": rendered["input"],
        "target": rendered["target"],
        "references": [] if rendered["target"] is None else [rendered["target"]],
        "source_path": source,
    }
    return example


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_mmlu_task(
    task_name: str,
    split: str = "validation",
    data_dir: Optional[str] = None,
    max_examples: Optional[int] = None,
    seed: int = 42,
    prefer_local: bool = True,
) -> List[Dict[str, Any]]:
    """Load one MMLU subject and return plain example dictionaries.

    Args:
        task_name: MMLU subject, e.g. ``"abstract_algebra"``.
        split: ``"validation"`` (the paper's D_R pool) is the usual choice.
        data_dir: optional directory of the original Berkeley MMLU release
            (looking for ``<data_dir>/val/<subject>_test.csv`` etc.).
        max_examples: deterministic (seeded) subsample size, if given.
        seed: seed used for the subsample.
        prefer_local: try the local CSV/JSON back-ends before HuggingFace.

    Returns:
        List of example dicts (possibly empty if the subject cannot be loaded).
    """
    examples: List[Dict[str, Any]] = []

    if data_dir:
        json_paths = [
            os.path.join(data_dir, f"{task_name}.json"),
            os.path.join(data_dir, f"{task_name}.jsonl"),
            os.path.join(data_dir, split, f"{task_name}.json"),
            os.path.join(data_dir, split, f"{task_name}.jsonl"),
        ]
        for path in json_paths:
            if os.path.isfile(path):
                examples = _examples_from_local_json(task_name, split, data_dir, path)
                break

    if not examples and data_dir and (prefer_local or not examples):
        examples = _examples_from_local_csv(task_name, split, data_dir)

    if not examples:
        examples = _examples_from_hf(task_name, split)

    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        examples = rng.sample(examples, max_examples)
        examples.sort(key=lambda ex: ex["id"])

    return examples


def load_mmlu_tasks(
    task_names: Optional[Iterable[str]] = None,
    split: str = "validation",
    per_task: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load several MMLU subjects, returning ``{subject: [examples]}``."""
    names = list(task_names) if task_names is not None else list(MMLU_SUBJECTS)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        exs = load_mmlu_task(name, split=split, max_examples=per_task, **kwargs)
        if exs:
            out[name] = exs
    logger.info("Loaded %d/%d MMLU subjects", len(out), len(names))
    return out


def flatten_task_dict(task_dict: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten ``{task: [examples]}`` into a task-sorted flat example list."""
    flat: List[Dict[str, Any]] = []
    for task in sorted(task_dict):
        flat.extend(task_dict[task])
    return flat


def load_mmlu_dataset(
    task_names: Optional[Iterable[str]] = None,
    split: str = "validation",
    per_task: Optional[int] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load the full MMLU validation pool (all 57 subjects) as a flat list."""
    return flatten_task_dict(
        load_mmlu_tasks(task_names, split=split, per_task=per_task, **kwargs)
    )
