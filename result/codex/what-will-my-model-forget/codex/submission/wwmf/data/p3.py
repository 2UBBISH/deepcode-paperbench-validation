"""Loading the Public Pool of Prompts (P3) dataset.

The paper builds ``D_PT`` from the *training split of 36 P3 tasks* and, for
BART0, ``D_R`` from the *test split of P3* (Sec. 4.1).

Two data sources are supported:

1. ``bigscience/P3`` on the HF hub (per-template parquet configs, e.g.
   ``glue_mrpc_equivalent``).  Template configs are mapped to T0 task names with
   :mod:`wwmf.data.registry`.
2. A local directory in the layout of the BART0/ReCross data dump
   (``<data_root>/<task>/{train,validation,test}.jsonl``).  This is the dump the
   authors used; the released ``bigscience/P3`` snapshot misses a couple of
   tasks (``paws_x-en``, ``storycloze``) which are present there.
"""
from __future__ import annotations

import glob
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence

from ..utils import ensure_dir, read_json
from .registry import find_p3_configs, p3_config_prefixes
from .types import Dataset, Example

P3_HF_DATASET = "bigscience/P3"

#: fallback order when a requested split does not exist for a template config.
_SPLIT_FALLBACKS: Dict[str, List[str]] = {
    "test": ["test", "validation", "train"],
    "validation": ["validation", "test", "train"],
    "train": ["train", "validation"],
}


def list_p3_configs(cache_dir: str = "cache") -> List[str]:
    """All template config names of ``bigscience/P3`` (cached on disk)."""
    cache_file = os.path.join(cache_dir, "p3_configs.json")
    if os.path.exists(cache_file):
        return read_json(cache_file)
    from huggingface_hub import list_repo_files

    files = list_repo_files(P3_HF_DATASET, repo_type="dataset")
    configs = sorted({f.split("/")[0] for f in files if "/" in f})
    ensure_dir(cache_dir)
    with open(cache_file, "w", encoding="utf8") as fh:
        json.dump(configs, fh)
    return configs


def _rows_to_examples(rows: Iterable[dict], task: str, config: str) -> Dataset:
    out: Dataset = []
    for i, row in enumerate(rows):
        x = row.get("inputs_pretokenized", row.get("input", row.get("inputs", "")))
        y = row.get("targets_pretokenized", row.get("target", row.get("targets", "")))
        # P3 stores the answer choices separately for some templates; the
        # pretokenized input already contains the rendered prompt.
        meta = {"dataset": row.get("dataset", ""), "template": row.get("template", "")}
        out.append(Example(input=str(x), target=str(y), task=task, config=config, idx=int(row.get("idx", i)), meta=meta))
    return out


def load_p3_config(
    config: str,
    split: str,
    task: Optional[str] = None,
    hf_dataset: str = P3_HF_DATASET,
) -> Dataset:
    """Load one template config of P3 (parquet-backed, no loading script needed)."""
    from datasets import load_dataset

    last_error: Optional[Exception] = None
    for candidate in _SPLIT_FALLBACKS.get(split, [split]):
        try:
            ds = load_dataset(hf_dataset, name=config, split=candidate)
        except Exception as exc:  # split missing / network issue
            last_error = exc
            continue
        task = task or config
        return _rows_to_examples(ds, task=task, config=config)
    raise RuntimeError(f"could not load P3 config {config!r} split {split!r}: {last_error}")


def load_p3_task(
    task: str,
    split: str,
    max_examples: Optional[int] = None,
    seed: int = 0,
    configs: Optional[Sequence[str]] = None,
) -> Dataset:
    """Load one T0 task (all of its P3 template variants).

    The addendum specifies that all variants of a task are used.
    """
    configs = list(configs) if configs is not None else find_p3_configs(task, list_p3_configs())
    if not configs:
        raise ValueError(f"no P3 template configs found for task {task!r}")
    pool: Dataset = []
    for config in configs:
        try:
            pool.extend(load_p3_config(config, split=split, task=task))
        except Exception as exc:  # pragma: no cover - depends on the release snapshot
            print(f"[p3] skipping {config}: {exc}")
    if max_examples is not None and len(pool) > max_examples:
        rng = random.Random(seed)
        pool = rng.sample(pool, max_examples)
    return pool


def load_p3_tasks(
    tasks: Sequence[str],
    split: str,
    max_examples_per_task: Optional[int] = None,
    seed: int = 0,
) -> Dataset:
    """Load several T0 tasks, keeping a *balanced* sample per task.

    "We use a balanced sample of 100 examples per task to form our D_PT"
    (Sec. 4.1) -- i.e. an equal number of examples per task (see addendum).
    """
    out: Dataset = []
    available = list_p3_configs()
    for k, task in enumerate(tasks):
        out.extend(
            load_p3_task(
                task,
                split=split,
                max_examples=max_examples_per_task,
                seed=seed + k,
                configs=find_p3_configs(task, available),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Local data dump (BART0 / ReCross layout)
# --------------------------------------------------------------------------------------
def _find_split_file(data_root: str, task: str, split: str) -> Optional[str]:
    aliases = {"validation": ["validation", "valid", "dev", "val"], "test": ["test"], "train": ["train"]}
    for name in aliases.get(split, [split]):
        for pattern in (f"{name}.jsonl", f"{name}.json", f"{name}.csv", f"{name}.tsv"):
            hits = glob.glob(os.path.join(data_root, task, pattern))
            if hits:
                return hits[0]
    return None


def _read_records(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.endswith(".json"):
        with open(path, encoding="utf8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, list) else list(obj.values())
    sep = "," if path.endswith(".csv") else "\t"
    import csv

    with open(path, encoding="utf8") as fh:
        return list(csv.DictReader(fh, delimiter=sep))


def load_local_task(
    data_root: str,
    task: str,
    split: str,
    max_examples: Optional[int] = None,
    seed: int = 0,
) -> Dataset:
    """Load a task from a local BART0/ReCross-style data dump."""
    path = _find_split_file(data_root, task, split)
    if path is None:
        raise FileNotFoundError(f"no {split!r} split for task {task!r} under {data_root!r}")
    rows = _read_records(path)
    examples: Dataset = []
    for i, row in enumerate(rows):
        x = row.get("inputs_pretokenized", row.get("input", row.get("inputs", row.get("question", ""))))
        y = row.get("targets_pretokenized", row.get("target", row.get("targets", row.get("answer", ""))))
        examples.append(Example(input=str(x), target=str(y), task=task, config=task, idx=i))
    if max_examples is not None and len(examples) > max_examples:
        rng = random.Random(seed)
        examples = rng.sample(examples, max_examples)
    return examples


def load_tasks_auto(
    tasks: Sequence[str],
    split: str,
    data_root: Optional[str] = None,
    max_examples_per_task: Optional[int] = None,
    seed: int = 0,
) -> Dataset:
    """Load tasks from a local dump if available, otherwise from the HF hub."""
    local_root = data_root
    if local_root and os.path.isdir(os.path.join(local_root, tasks[0])) is False and not any(
        os.path.isdir(os.path.join(local_root, t)) for t in tasks
    ):
        local_root = None
    if local_root:
        out: Dataset = []
        for k, task in enumerate(tasks):
            try:
                out.extend(load_local_task(local_root, task, split, max_examples_per_task, seed + k))
                continue
            except FileNotFoundError:
                pass
            out.extend(
                load_p3_task(task, split, max_examples=max_examples_per_task, seed=seed + k,
                             configs=find_p3_configs(task, list_p3_configs()))
            )
        return out
    return load_p3_tasks(tasks, split, max_examples_per_task=max_examples_per_task, seed=seed)


def task_config_summary(tasks: Sequence[str]) -> Dict[str, int]:
    """How many P3 template configs each task maps to (useful for sanity checks)."""
    available = list_p3_configs()
    return {t: len(find_p3_configs(t, available)) for t in tasks}


__all__ = [
    "P3_HF_DATASET",
    "list_p3_configs",
    "load_p3_config",
    "load_p3_task",
    "load_p3_tasks",
    "load_local_task",
    "load_tasks_auto",
    "p3_config_prefixes",
]
