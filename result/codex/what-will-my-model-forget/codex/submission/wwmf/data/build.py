"""Construction of ``D_PT``, ``D_R`` and their splits (Sec. 4.1).

* ``D_PT``      -- 36 upstream P3 training tasks, 100 examples per task.
* ``D_hat_PT``  -- the subset of ``D_PT`` that the base PTLM ``f_0`` answers correctly.
* ``D_R``       -- the mispredicted examples of the refinement task (P3-Test for
  BART0, MMLU validation for FLAN-T5), randomly split 60% / 40% into
  ``D_R^Train`` and ``D_R^Test``.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import DR_TRAIN_FRACTION, ExperimentConfig
from ..evaluation.metrics import exact_match
from ..utils import ensure_dir
from . import mmlu as mmlu_module
from . import p3 as p3_module
from .registry import (
    MMLU_SUBJECTS,
    P3_TEST_ID_TASKS,
    P3_TEST_OOD_TASKS,
    P3_TEST_TASKS,
    UPSTREAM_TASKS,
)
from .types import Dataset, Example, save_dataset


@dataclass
class RefinementSplits:
    """``D_R`` split into the train / test subsets used to train and evaluate g."""

    train: Dataset
    test: Dataset
    candidates: Dataset
    em: float = 0.0


# --------------------------------------------------------------------------------------
# D_PT
# --------------------------------------------------------------------------------------
def build_dpt(cfg: ExperimentConfig, tasks: Optional[Sequence[str]] = None) -> Dataset:
    """Build ``D_PT``: a balanced sample of ``cfg.examples_per_upstream_task`` per task."""
    tasks = list(tasks) if tasks is not None else list(UPSTREAM_TASKS)
    cache_path = os.path.join(cfg.data_root, "processed", "dpt.jsonl")
    if os.path.exists(cache_path):
        from .types import load_dataset_file

        cached = load_dataset_file(cache_path)
        if tasks == list(UPSTREAM_TASKS):
            return cached
    local_root = os.path.join(cfg.data_root, "p3") if cfg.data_root else None
    dpt = p3_module.load_tasks_auto(
        tasks,
        split="train",
        data_root=local_root if local_root and os.path.isdir(local_root) else None,
        max_examples_per_task=cfg.examples_per_upstream_task,
        seed=cfg.seed,
    )
    ensure_dir(os.path.dirname(cache_path))
    save_dataset(cache_path, dpt)
    return dpt


def upstream_task_histogram(dpt: Dataset) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for ex in dpt:
        hist[ex.task] = hist.get(ex.task, 0) + 1
    return hist


# --------------------------------------------------------------------------------------
# D_R
# --------------------------------------------------------------------------------------
def refinement_candidates(cfg: ExperimentConfig, tasks: Optional[Sequence[str]] = None) -> Dataset:
    """All examples of the refinement dataset, before filtering mispredictions."""
    if cfg.refinement_data == "mmlu":
        root = os.path.join(cfg.data_root, "mmlu") if cfg.data_root else None
        return mmlu_module.load_mmlu(split="val", subjects=tasks or MMLU_SUBJECTS, root=root)
    if cfg.refinement_data == "p3_test":
        tasks = list(tasks) if tasks is not None else list(P3_TEST_TASKS)
        local_root = os.path.join(cfg.data_root, "p3") if cfg.data_root else None
        return p3_module.load_tasks_auto(
            tasks,
            split="test",
            data_root=local_root if local_root and os.path.isdir(local_root) else None,
            seed=cfg.seed,
        )
    raise ValueError(f"unknown refinement dataset {cfg.refinement_data!r}")


def split_dr(
    examples: Dataset,
    seed: int = 0,
    train_fraction: float = DR_TRAIN_FRACTION,
) -> RefinementSplits:
    """Random 60 / 40 split of ``D_R`` (addendum)."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    cut = int(round(len(shuffled) * train_fraction))
    return RefinementSplits(train=shuffled[:cut], test=shuffled[cut:], candidates=list(examples))


def build_dr(
    cfg: ExperimentConfig,
    model,
    tasks: Optional[Sequence[str]] = None,
    max_candidates: Optional[int] = None,
    save_name: str = "dr",
) -> RefinementSplits:
    """Collect the mispredicted examples of the base model on the refinement dataset.

    "We evaluate the LM f_0 on a new task and collect all the mispredicted
    examples, noted as D_R" (Sec. 2).  Predictions are graded with Exact Match.
    """
    candidates = refinement_candidates(cfg, tasks)
    if max_candidates is not None:
        rng = random.Random(cfg.seed)
        candidates = rng.sample(candidates, min(max_candidates, len(candidates)))
    predictions = model.predict([e.input for e in candidates], batch_size=max(cfg.eval_batch_size, 8))
    mispredicted = [
        ex for ex, pred in zip(candidates, predictions) if exact_match([pred], [ex.target]) <= 0.5
    ]
    em = exact_match(predictions, [e.target for e in candidates])
    splits = split_dr(mispredicted, seed=cfg.seed)
    splits.em = em
    out_dir = os.path.join(cfg.data_root, "processed")
    ensure_dir(out_dir)
    save_dataset(os.path.join(out_dir, f"{save_name}_train.jsonl"), splits.train)
    save_dataset(os.path.join(out_dir, f"{save_name}_test.jsonl"), splits.test)
    return splits


def load_dr(cfg: ExperimentConfig, name: str = "dr") -> Optional[RefinementSplits]:
    """Load a previously built ``D_R`` split, if it exists."""
    from .types import load_dataset_file

    base = os.path.join(cfg.data_root, "processed")
    train_path = os.path.join(base, f"{name}_train.jsonl")
    test_path = os.path.join(base, f"{name}_test.jsonl")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return None
    train = load_dataset_file(train_path)
    test = load_dataset_file(test_path)
    return RefinementSplits(train=train, test=test, candidates=list(train) + list(test))


# --------------------------------------------------------------------------------------
# Table 2: in-domain / out-of-domain split of P3-Test
# --------------------------------------------------------------------------------------
def build_ood_split(cfg: ExperimentConfig, model) -> Tuple[RefinementSplits, RefinementSplits]:
    """In-domain and out-of-domain ``D_R`` splits used for Table 2."""
    id_splits = build_dr(cfg, model, tasks=P3_TEST_ID_TASKS, save_name="dr_id")
    ood_splits = build_dr(cfg, model, tasks=P3_TEST_OOD_TASKS, save_name="dr_ood")
    return id_splits, ood_splits
