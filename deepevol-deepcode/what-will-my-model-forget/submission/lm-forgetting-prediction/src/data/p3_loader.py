"""P3 (Public Pool of Prompts) data loading utilities.

Paper references
----------------
* Sec. 4.1 "Base PTLMs and Datasets (f0, D_PT)": "We evaluate forgetting over 36
  tasks from the training split of the Public Pool of Prompts (P3) dataset ...
  We use a balanced sample of 100 examples per task to form our D_PT."
* Clarifications: the 36 tasks are the intersection of the T0-Train tasks and the
  tasks in the BART0 (ReCross) data repo, and "balanced" simply means "an equal
  number (100) of examples per task".
* Clarifications: for BART0, D_R is built from the *test* split of the P3 dataset
  (the 8 ReCross test tasks); all prompt variants of a task are used
  ("All task variants are used.").

This module knows how to obtain, for one P3 task:
  * the prompt templates that promptsource exposes (preferring "score_eval"-style
    evaluation templates, matching the templates used when BART0/T0 were trained),
  * a list of graded ``{input, references}`` examples.

Three back ends are supported, tried in order:
  1. ``promptsource.templates.DatasetTemplates`` + ``datasets.load_dataset`` on the
     P3 hub dataset (canonical path, gives every template variant).
  2. Plain ``datasets.load_dataset(p3_dataset_id, task)`` using the pretokenized
     columns that ship with the hub copy of P3 (single, "chosen" template).
  3. Local ReCross-style JSON files (``<json_dir>/<task>*.json``) which carry
     ``{"inputs": ..., "targets": ...}`` records.

The loaders return plain Python dicts so that downstream code (dataset_builders,
models, caches) never has to touch promptsource/datasets objects.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)

# Default P3 hub identifier and prompt-template preferences.  Both can be
# overridden from ``config/config.yaml`` (keys ``data.p3_dataset_id`` and
# ``prefer_template_substrings`` in ``config/tasks.yaml``).
DEFAULT_P3_DATASET_ID = "bigscience/P3"
DEFAULT_PREFER_TEMPLATE_SUBSTRINGS: Sequence[str] = ("score_eval", "eval")

# Prefixes used by some releases of the P3 hub dataset.
_TASK_NAME_PREFIXES = ("", "task_")


# ---------------------------------------------------------------------------
# Template handling
# ---------------------------------------------------------------------------
def _template_names(dataset_templates: Any) -> List[str]:
    """Return the template names exposed by a promptsource ``DatasetTemplates``."""
    templates = getattr(dataset_templates, "templates", dataset_templates)
    if isinstance(templates, dict):
        return list(templates.keys())
    # Older promptsource versions expose a dict-like object only.
    return [t.name for t in templates]


def pick_template_name(
    template_names: Sequence[str],
    prefer_substrings: Sequence[str] = DEFAULT_PREFER_TEMPLATE_SUBSTRINGS,
) -> Optional[str]:
    """Pick the template whose name contains the earliest preferred substring.

    Preference order follows ``prefer_substrings`` (default: "score_eval" first,
    then any "eval" template).  If nothing matches, the first available template
    is returned so that loading degrades gracefully instead of failing.  Returns
    ``None`` when no templates are available at all.
    """
    if not template_names:
        return None
    for needle in prefer_substrings:
        for name in template_names:
            if needle in name:
                return name
    return template_names[0]


def _apply_template(template: Any, example: Dict[str, Any]) -> Optional[List[str]]:
    """Apply a promptsource template, returning ``[input_text, target_text]``."""
    try:
        out = template.apply(example)
    except Exception as exc:  # pragma: no cover - template-specific failures
        LOGGER.debug("Template %s failed on an example: %s", getattr(template, "name", "?"), exc)
        return None
    if not out or len(out) < 2:
        return None
    inp, tgt = out[0], out[1]
    if inp is None or tgt is None:
        return None
    return [str(inp), str(tgt)]


def _templates_for_task(task_name: str) -> Optional[Any]:
    """Return the promptsource ``DatasetTemplates`` for a task, or ``None``."""
    try:
        from promptsource.templates import DatasetTemplates  # type: ignore
    except Exception:  # pragma: no cover - promptsource not installed
        LOGGER.debug("promptsource is not available; falling back to hub columns")
        return None
    try:
        return DatasetTemplates(task_name)
    except Exception as exc:  # pragma: no cover - unknown task name
        LOGGER.debug("promptsource has no templates for %s: %s", task_name, exc)
        return None


# ---------------------------------------------------------------------------
# Back end 1/2: HuggingFace ``datasets``
# ---------------------------------------------------------------------------
def _load_hf_split(dataset_id: str, task_name: str, split: str, cache_dir: Optional[str]) -> Any:
    """Load one P3 task split via ``datasets.load_dataset`` (with name fallbacks)."""
    from datasets import load_dataset  # local import: heavy dependency

    last_err: Optional[Exception] = None
    for prefix in _TASK_NAME_PREFIXES:
        name = f"{prefix}{task_name}" if prefix else task_name
        try:
            return load_dataset(dataset_id, name, split=split, cache_dir=cache_dir)
        except Exception as exc:  # pragma: no cover - depends on remote data
            last_err = exc
    raise RuntimeError(f"Could not load P3 task '{task_name}' (split={split})") from last_err


def _raw_example_to_pair(
    example: Dict[str, Any],
    template: Optional[Any],
    output_mode: str = "inputs",
) -> Optional[Dict[str, Any]]:
    """Convert one raw dataset row into ``{"input", "target"}`` (or ``None``)."""
    if template is not None:
        pair = _apply_template(template, example)
        if pair is not None:
            return {"input": pair[0], "target": pair[1], "raw": example}

    # Fall back to the pretokenized columns that ship with the hub copy of P3.
    inp = example.get("inputs_pretokenized") or example.get("inputs")
    tgt = example.get("targets_pretokenized") or example.get("targets")
    # Some releases store per-template lists; take the first template.
    if isinstance(inp, (list, tuple)):
        inp = inp[0] if len(inp) else None
    if isinstance(tgt, (list, tuple)):
        tgt = tgt[0] if len(tgt) else None
    if inp is None or tgt is None:
        return None
    return {"input": str(inp), "target": str(tgt), "raw": example}


def _load_task_hf(
    task_name: str,
    split: str,
    dataset_id: str,
    cache_dir: Optional[str],
    prefer_substrings: Sequence[str],
    template_name: Optional[str] = None,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load (a subset of) one P3 task using the HuggingFace back ends."""
    dataset = _load_hf_split(dataset_id, task_name, split, cache_dir)

    dt = _templates_for_task(task_name)
    template = None
    chosen_name: Optional[str] = None
    if dt is not None:
        names = _template_names(dt)
        chosen_name = template_name or pick_template_name(names, prefer_substrings)
        if chosen_name is not None:
            try:
                template = dt[chosen_name]
            except Exception:  # pragma: no cover
                template = None

    # Deterministic subsampling *before* rendering keeps the cost low; when the
    # caller wants every example we simply walk the whole split.
    indices = list(range(len(dataset)))
    if max_examples is not None and max_examples < len(indices):
        rng = random.Random(seed)
        indices = sorted(rng.sample(indices, max_examples))

    examples: List[Dict[str, Any]] = []
    for idx in indices:
        pair = _raw_example_to_pair(dataset[idx], template)
        if pair is None:
            continue
        examples.append(
            {
                "id": f"{task_name}-{idx}",
                "task": task_name,
                "split": split,
                "template_name": chosen_name,
                "input": pair["input"],
                "target": pair["target"],
                "references": [pair["target"]],
            }
        )
    return examples


# ---------------------------------------------------------------------------
# Back end 3: local ReCross-style JSON
# ---------------------------------------------------------------------------
def _json_candidates(task_name: str, json_dir: str, split: Optional[str] = None) -> List[str]:
    """Locate JSON files for a task inside a ReCross-style data directory."""
    patterns = [
        os.path.join(json_dir, f"{task_name}.json"),
        os.path.join(json_dir, f"{task_name}_*.json"),
        os.path.join(json_dir, "*", f"{task_name}.json"),
        os.path.join(json_dir, "*", f"{task_name}_*.json"),
    ]
    if split:
        patterns = [os.path.join(json_dir, split, f"{task_name}.json")] + patterns
    found: List[str] = []
    for pattern in patterns:
        found.extend(sorted(glob.glob(pattern)))
    # de-duplicate, preserving order
    seen = set()
    unique = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _normalise_json_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a ReCross/P3 JSON record into ``{"input", "target"}``."""
    inp = (
        record.get("input")
        or record.get("inputs")
        or record.get("inputs_pretokenized")
        or record.get("question")
    )
    tgt = (
        record.get("target")
        or record.get("targets")
        or record.get("targets_pretokenized")
        or record.get("answer")
    )
    if isinstance(inp, (list, tuple)):
        inp = inp[0] if len(inp) else None
    if isinstance(tgt, (list, tuple)):
        tgt = tgt[0] if len(tgt) else None
    if inp is None or tgt is None:
        return None
    references = record.get("references")
    if references is None:
        references = [str(tgt)]
    elif isinstance(references, str):
        references = [references]
    return {"input": str(inp), "target": str(tgt), "references": list(references)}


def _load_task_json(
    task_name: str,
    split: str,
    json_dir: str,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load one task from local ReCross-style JSON files."""
    paths = _json_candidates(task_name, json_dir, split=None)
    if not paths:
        return []

    records: List[Dict[str, Any]] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not read %s: %s", path, exc)
            continue
        if isinstance(payload, dict):
            payload = payload.get("data", payload.get("examples", []))
        if not isinstance(payload, list):
            continue
        # Keep the split that the file name advertises when it advertises one.
        file_split = os.path.basename(path).split("_")[-1].replace(".json", "")
        if split and file_split in {"train", "test", "validation", "dev"} and file_split != split:
            continue
        for i, record in enumerate(payload):
            if not isinstance(record, dict):
                continue
            norm = _normalise_json_record(record)
            if norm is None:
                continue
            norm.update(
                {
                    "id": f"{task_name}-{os.path.basename(path)}-{i}",
                    "task": task_name,
                    "split": split,
                    "template_name": os.path.basename(path).replace(".json", ""),
                    "source_path": path,
                }
            )
            records.append(norm)

    if max_examples is not None and max_examples < len(records):
        rng = random.Random(seed)
        records = [records[i] for i in sorted(rng.sample(range(len(records)), max_examples))]
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_p3_task(
    task_name: str,
    split: str = "train",
    dataset_id: str = DEFAULT_P3_DATASET_ID,
    cache_dir: Optional[str] = None,
    prefer_substrings: Sequence[str] = DEFAULT_PREFER_TEMPLATE_SUBSTRINGS,
    template_name: Optional[str] = None,
    max_examples: Optional[int] = None,
    seed: int = 42,
    json_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load one P3 task as a list of ``{input, target, references, ...}`` dicts.

    Args:
        task_name: P3 task name without the template suffix (e.g. ``glue-mrpc``).
        split: dataset split (``train`` for D_PT, ``test`` for BART0 D_R).
        dataset_id: hub id of the P3 copy to use.
        cache_dir: HuggingFace datasets cache directory.
        prefer_substrings: template preference order (see :func:`pick_template_name`).
        template_name: force one specific promptsource template.
        max_examples: deterministic cap on the number of loaded examples.
        seed: seed of the (seeded) subsampling.
        json_dir: optional directory of local ReCross-style JSON files, tried last.

    Returns:
        A list of example dicts.  Empty list (with a warning) if the task could
        not be loaded from any back end.
    """
    try:
        examples = _load_task_hf(
            task_name=task_name,
            split=split,
            dataset_id=dataset_id,
            cache_dir=cache_dir,
            prefer_substrings=prefer_substrings,
            template_name=template_name,
            max_examples=max_examples,
            seed=seed,
        )
        if examples:
            LOGGER.info("Loaded %d examples for P3 task %s (%s)", len(examples), task_name, split)
            return examples
    except Exception as exc:
        LOGGER.warning("HF loading failed for P3 task %s: %s", task_name, exc)

    if json_dir:
        examples = _load_task_json(task_name, split, json_dir, max_examples=max_examples, seed=seed)
        if examples:
            LOGGER.info("Loaded %d examples for P3 task %s from %s", len(examples), task_name, json_dir)
            return examples

    LOGGER.error("No data could be loaded for P3 task %s (split=%s)", task_name, split)
    return []


def load_p3_tasks(
    task_names: Iterable[str],
    split: str = "train",
    per_task: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load several P3 tasks; returns ``{task_name: [examples]}``.

    ``per_task`` is passed through as ``max_examples`` (the paper's *balanced*
    sampling of 100 examples per task for D_PT).
    """
    if per_task is not None:
        kwargs["max_examples"] = per_task
    out: Dict[str, List[Dict[str, Any]]] = {}
    for task in task_names:
        out[task] = load_p3_task(task, split=split, **kwargs)
    return out


def flatten_task_dict(task_dict: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Flatten ``{task: [examples]}`` into a single list of example dicts."""
    flat: List[Dict[str, Any]] = []
    for task in sorted(task_dict):
        flat.extend(task_dict[task])
    return flat


def load_p3_dataset(
    task_names: Iterable[str],
    split: str = "train",
    per_task: Optional[int] = 100,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Load several P3 tasks and return a single flat list (e.g. D_PT)."""
    return flatten_task_dict(load_p3_tasks(task_names, split=split, per_task=per_task, **kwargs))


def load_bart0_r_task(task_name: str, **kwargs: Any) -> List[Dict[str, Any]]:
    """Load one BART0 D_R task (P3 *test* split / ReCross data repo)."""
    kwargs.setdefault("split", "test")
    return load_p3_task(task_name, **kwargs)


__all__ = [
    "DEFAULT_P3_DATASET_ID",
    "DEFAULT_PREFER_TEMPLATE_SUBSTRINGS",
    "pick_template_name",
    "load_p3_task",
    "load_p3_tasks",
    "load_p3_dataset",
    "load_bart0_r_task",
    "flatten_task_dict",
]
