"""Dataset builders for the "What Will My Model Forget?" reproduction.

This module assembles the three data artifacts used by the paper's pipeline:

* ``D_PT``     -- 100 examples sampled from each of the 36 P3 *train* tasks
                 (36 x 100 = 3600 upstream pretraining examples).
* ``D_PT_hat`` -- the subset of ``D_PT`` that the base model ``f0`` answers
                 correctly (:math:`\\hat{D}_{PT}` in the paper, §2).  ``D_PT_hat``
                 is the candidate set of *forgettable* upstream examples and the
                 set on which the forecasters are evaluated.
* ``D_R``      -- the *refinement* / mispredicted examples: for BART0 they come
                 from the 8 P3 *test* tasks (ReCross data repo), for FLAN-T5 they
                 come from the MMLU validation split (57 subjects).  ``D_R`` is
                 randomly split 60% / 40% into ``D_R^Train`` / ``D_R^Test``.

The builders are deliberately agnostic of the concrete LM implementation: any
``predictor`` that maps an example dict to a *string* prediction is enough.  Two
convenient adapter contracts are supported (see :func:`predict_examples`):

* a plain callable ``predictor(example) -> str``;
* an object exposing ``predict(example) -> str``, ``predict_batch(examples)`` or
  ``generate(example) -> str`` (the ``src.modeling.base_lm`` wrappers in this
  repo use ``generate``).

If no predictor is supplied (e.g. the LM weights are not available in the
environment) the builders fall back to a *pass-through* mode that keeps every
example and records ``prediction=None``.  This lets the data pipeline be smoke
tested, and it is always written into the manifest so that a run can never
silently report a filtered dataset that was in fact not filtered.

Source: §2 (definitions of D_PT, D_PT_hat, D_R, EM); §4.1; Addendum (task lists,
SQuAD-2.0 EM, 60/40 split, ID/OOD split).  This file itself is glue: it only
composes ``p3_loader``, ``mmlu_loader`` and ``em_eval`` according to the plan.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .em_eval import extract_references, is_correct
from .mmlu_loader import MMLU_SUBJECTS, load_mmlu_tasks
from .p3_loader import (
    flatten_task_dict,
    load_p3_task,
    load_p3_tasks,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PT_FORWARD_TASKS",
    "DEFAULT_PT_CLASSIFICATION_TASKS",
    "DEFAULT_BART0_R_TASKS",
    "DEFAULT_ID_TASKS",
    "DEFAULT_OOD_TASKS",
    "load_tasks_yaml",
    "save_jsonl",
    "load_jsonl",
    "predict_examples",
    "build_d_pt",
    "filter_d_pt_hat",
    "collect_d_r",
    "split_d_r",
    "split_id_ood",
    "build_all",
]

# ---------------------------------------------------------------------------
# Default task lists (mirror of config/tasks.yaml, used when the YAML cannot be
# located -- e.g. import from another working directory).
# ---------------------------------------------------------------------------
DEFAULT_PT_FORWARD_TASKS: List[str] = [
    "glue-mrpc",
    "glue-qqp",
    "paws_x-en",
    "kilt_tasks-hotpotqa",
    "wiki_qa",
    "adversarial_qa-dbert",
    "adversarial_qa-dbidaf",
    "adversarial_qa-droberta",
    "duorc-SelfRC",
    "duorc-ParaphraseRC",
    "ropes",
    "quoref",
    "cos_e-v1.11",
    "cosmos_qa",
    "dream",
    "qasc",
    "quail",
    "quartz",
    "sciq",
    "social_i_qa",
    "wiki_hop-original",
    "wiqa",
]

DEFAULT_PT_CLASSIFICATION_TASKS: List[str] = [
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail-3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
]

DEFAULT_BART0_R_TASKS: List[str] = [
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    "super_glue-rte",
    "anli",
    "super_glue-copa",
    "hellaswag",
    "super_glue-wic",
]

# Appendix B ID/OOD partition of the BART0 D_R tasks.  Note: "anli" is listed
# among the BART0 D_R tasks but Appendix B places it in the *out-of-domain*
# bucket, so it belongs to OOD here.
DEFAULT_ID_TASKS: List[str] = [
    "super_glue-cb",
    "super_glue-rte",
    "super_glue-wsc.fixed",
    "super_glue-copa",
    "super_glue-wic",
]

DEFAULT_OOD_TASKS: List[str] = [
    "storycloze",
    "hellaswag",
    "anli",
    "winogrande-winogrande_xl",
]


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def load_tasks_yaml(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config/tasks.yaml`` with a few candidate-path fallbacks.

    Returns an empty dict if PyYAML or the file is unavailable; callers then use
    the module-level ``DEFAULT_*`` task lists.
    """
    candidates: List[str] = []
    if path:
        candidates.append(path)
    here = os.path.dirname(os.path.abspath(__file__))
    # src/data -> repo root is three levels up for lm-forgetting-prediction/,
    # but config/ lives next to the top-level workspace, so probe both.
    candidates += [
        os.path.join(here, "..", "..", "config", "tasks.yaml"),
        os.path.join(here, "..", "..", "..", "config", "tasks.yaml"),
        os.path.join(os.getcwd(), "config", "tasks.yaml"),
    ]
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - PyYAML is in requirements
        logger.warning("PyYAML unavailable; using built-in default task lists.")
        return {}
    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            logger.info("Loaded task registry from %s", os.path.abspath(cand))
            return data
    logger.warning("config/tasks.yaml not found; using built-in default task lists.")
    return {}


def _task_lists(tasks_yaml: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Resolve the task lists, preferring the YAML registry over the defaults."""
    ty = tasks_yaml if tasks_yaml is not None else load_tasks_yaml()
    pt_block = ty.get("pt_tasks", {}) or {}
    forward = list(pt_block.get("forward_tasks") or DEFAULT_PT_FORWARD_TASKS)
    classification = list(
        pt_block.get("classification_tasks") or DEFAULT_PT_CLASSIFICATION_TASKS
    )
    return {
        "pt_forward": forward,
        "pt_classification": classification,
        "pt": forward + classification,
        "bart0_r": list(ty.get("bart0_r_tasks") or DEFAULT_BART0_R_TASKS),
        "mmlu": list(ty.get("mmlu_tasks") or MMLU_SUBJECTS),
        "id": list(ty.get("id_tasks") or DEFAULT_ID_TASKS),
        "ood": list(ty.get("ood_tasks") or DEFAULT_OOD_TASKS),
        "prefer_template_substrings": list(
            ty.get("prefer_template_substrings") or ["score_eval", "eval"]
        ),
    }


def save_jsonl(examples: Sequence[Dict[str, Any]], path: str) -> str:
    """Persist a list of example dicts as JSON Lines (creates parent dirs)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Wrote %d examples -> %s", len(examples), path)
    return path


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSON Lines file produced by :func:`save_jsonl`."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Prediction adapters
# ---------------------------------------------------------------------------
def _single_prediction(predictor: Any, example: Dict[str, Any]) -> Optional[str]:
    """Obtain one string prediction from any supported predictor contract."""
    if predictor is None:
        return None
    if callable(predictor) and not hasattr(predictor, "generate"):
        try:
            return predictor(example)
        except TypeError:
            return None
    for attr in ("predict", "generate", "answer"):
        fn = getattr(predictor, attr, None)
        if callable(fn):
            for arg in (example, example.get("input", ""), example):
                try:
                    out = fn(arg) if not isinstance(arg, dict) else fn(arg)
                except TypeError:
                    continue
                if out is not None:
                    return out
            return None
    return None


def predict_examples(
    examples: Sequence[Dict[str, Any]],
    predictor: Any = None,
    predictions: Optional[Sequence[str]] = None,
    batch_size: int = 8,
) -> List[Optional[str]]:
    """Return one prediction string per example.

    Priority: an explicit ``predictions`` sequence wins; otherwise ``predictor``
    is queried either in batch (``predict_batch`` / ``generate_batch``) or one
    example at a time; otherwise ``None`` for every example (pass-through mode).
    """
    if predictions is not None:
        preds = list(predictions)
        if len(preds) != len(examples):
            raise ValueError(
                "predictions length %d != examples length %d"
                % (len(preds), len(examples))
            )
        return [None if p is None else str(p) for p in preds]

    if predictor is None:
        return [None] * len(examples)

    # Batch contract first (much cheaper for LM generation).
    for attr in ("predict_batch", "generate_batch"):
        fn = getattr(predictor, attr, None)
        if callable(fn):
            outputs: List[Optional[str]] = []
            try:
                for start in range(0, len(examples), max(1, batch_size)):
                    chunk = list(examples[start : start + batch_size])
                    res = fn(chunk)
                    if isinstance(res, str):
                        res = [res]
                    res = list(res)
                    if len(res) != len(chunk):
                        raise ValueError("batch predictor returned wrong length")
                    outputs.extend(None if r is None else str(r) for r in res)
                return outputs
            except Exception as exc:  # pragma: no cover - fall back to per-example
                logger.warning("Batch prediction failed (%s); falling back.", exc)
                outputs = []

    return [_single_prediction(predictor, ex) for ex in examples]


# ---------------------------------------------------------------------------
# D_PT / D_PT_hat
# ---------------------------------------------------------------------------
def build_d_pt(
    examples_per_task: int = 100,
    tasks: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    dataset_id: Optional[str] = None,
    split: str = "train",
    seed: int = 42,
    tasks_yaml: Optional[Dict[str, Any]] = None,
    prefer_template_substrings: Optional[Sequence[str]] = None,
    json_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build ``D_PT``: ``examples_per_task`` examples from each P3 *train* task.

    The 36 tasks are the P3 train tasks listed in the addendum; with the default
    ``examples_per_task=100`` this yields the paper's 3600-example ``D_PT``.
    """
    tl = _task_lists(tasks_yaml)
    task_names = list(tasks) if tasks is not None else tl["pt"]
    prefer = list(prefer_template_substrings or tl["prefer_template_substrings"])

    kwargs: Dict[str, Any] = {"split": split, "prefer_substrings": prefer}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if dataset_id is not None:
        kwargs["dataset_id"] = dataset_id
    if json_dir is not None:
        kwargs["json_dir"] = json_dir

    per_task_map = load_p3_tasks(
        task_names,
        per_task=None,
        **kwargs,
    )
    # Re-load with a hard per-task cap so that a task that yields more than
    # `examples_per_task` raw rows is deterministically subsampled.
    d_pt: List[Dict[str, Any]] = []
    for tname in task_names:
        rows = per_task_map.get(tname) or []
        rows = _subsample(rows, examples_per_task, seed=seed, key=tname)
        for ex in rows:
            ex.setdefault("split", split)
            ex.setdefault("task", tname)
        d_pt.extend(rows)

    logger.info(
        "D_PT: %d examples from %d tasks (requested %d/task).",
        len(d_pt),
        len([t for t in task_names if per_task_map.get(t)]),
        examples_per_task,
    )
    return d_pt


def _subsample(
    rows: Sequence[Dict[str, Any]], n: int, seed: int, key: str = ""
) -> List[Dict[str, Any]]:
    """Deterministically take at most ``n`` rows, shuffled with a stable seed."""
    rows = list(rows)
    if n is None or n <= 0 or len(rows) <= n:
        return rows
    rng = random.Random("%s|%d|%d" % (key, seed, n))
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [rows[i] for i in idx]


def filter_d_pt_hat(
    d_pt: Sequence[Dict[str, Any]],
    predictor: Any = None,
    predictions: Optional[Sequence[str]] = None,
    batch_size: int = 8,
    keep_predictions: bool = True,
    target_key: str = "target",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter ``D_PT`` down to the examples ``f0`` answers correctly (``D_PT_hat``).

    ``z``-free: an upstream example is retained iff its EM score against its gold
    reference(s) is 1 (SQuAD-2.0-style grading, §2).  In pass-through mode
    (``predictor is None`` and ``predictions is None``) behaviour is controlled
    by ``keep_predictions``: all examples are kept and the manifest flags
    ``filtered=False``.
    """
    preds = predict_examples(
        d_pt, predictor=predictor, predictions=predictions, batch_size=batch_size
    )
    pass_through = all(p is None for p in preds) and len(preds) > 0

    kept: List[Dict[str, Any]] = []
    n_correct = 0
    for ex, pred in zip(d_pt, preds):
        refs = extract_references(ex, target_key=target_key)
        if pass_through:
            ok = bool(keep_predictions)
        else:
            ok = is_correct(pred or "", refs)
        if ok:
            n_correct += 1
            out = dict(ex)
            if pass_through:
                out.setdefault("prediction", None)
            elif keep_predictions:
                out["prediction"] = pred
            out["correct"] = True
            kept.append(out)

    manifest = {
        "n_input": len(d_pt),
        "n_kept": len(kept),
        "n_correct": n_correct,
        "filtered": not pass_through,
        "base_em": (n_correct / len(d_pt)) if d_pt and not pass_through else None,
    }
    logger.info(
        "D_PT_hat: kept %d/%d (filtered=%s, base EM=%s).",
        len(kept),
        len(d_pt),
        manifest["filtered"],
        manifest["base_em"],
    )
    return kept, manifest


# ---------------------------------------------------------------------------
# D_R
# ---------------------------------------------------------------------------
def collect_d_r(
    examples: Sequence[Dict[str, Any]],
    predictor: Any = None,
    predictions: Optional[Sequence[str]] = None,
    batch_size: int = 8,
    keep_predictions: bool = True,
    target_key: str = "target",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect mispredicted examples ``D_R`` (``f0(x) != y``) from a pool.

    Used with the 8 P3-test tasks for BART0 and with the MMLU validation split
    (57 subjects) for FLAN-T5.  In pass-through mode every example is returned
    and the manifest flags ``filtered=False``.
    """
    preds = predict_examples(
        examples, predictor=predictor, predictions=predictions, batch_size=batch_size
    )
    pass_through = all(p is None for p in preds) and len(preds) > 0

    d_r: List[Dict[str, Any]] = []
    for ex, pred in zip(examples, preds):
        refs = extract_references(ex, target_key=target_key)
        if pass_through:
            wrong = True
        else:
            wrong = not is_correct(pred or "", refs)
        if wrong:
            out = dict(ex)
            if pass_through:
                out.setdefault("prediction", None)
            elif keep_predictions:
                out["prediction"] = pred
            out["correct"] = False
            d_r.append(out)

    manifest = {
        "n_input": len(examples),
        "n_mispredicted": len(d_r),
        "filtered": not pass_through,
        "error_rate": (len(d_r) / len(examples)) if examples and not pass_through else None,
    }
    logger.info(
        "D_R: collected %d mispredicted examples from %d (filtered=%s).",
        len(d_r),
        len(examples),
        manifest["filtered"],
    )
    return d_r, manifest


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def split_d_r(
    d_r: Sequence[Dict[str, Any]],
    r_train_ratio: float = 0.6,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Randomly split ``D_R`` into ``D_R^Train`` / ``D_R^Test`` (60% / 40%).

    The split is performed over a shuffled copy with a fixed seed so that the
    train/test membership is reproducible across runs.
    """
    rows = list(d_r)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_train = int(round(len(rows) * r_train_ratio))
    n_train = max(0, min(len(rows), n_train))
    train_idx = sorted(idx[:n_train])
    test_idx = sorted(idx[n_train:])
    train = [rows[i] for i in train_idx]
    test = [rows[i] for i in test_idx]
    for i, ex in enumerate(train):
        ex["r_split"] = "train"
        ex.setdefault("id", "r_train_%d" % i)
    for i, ex in enumerate(test):
        ex["r_split"] = "test"
        ex.setdefault("id", "r_test_%d" % i)
    logger.info(
        "D_R split: %d train / %d test (ratio=%.2f).", len(train), len(test), r_train_ratio
    )
    return train, test


def split_id_ood(
    examples: Sequence[Dict[str, Any]],
    id_tasks: Optional[Sequence[str]] = None,
    ood_tasks: Optional[Sequence[str]] = None,
    tasks_yaml: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition a BART0 example pool by task into ID / OOD (Appendix B).

    Returns ``(id_examples, ood_examples, other_examples)``; ``other`` holds
    examples whose task is in neither list (kept for completeness, never dropped
    silently).
    """
    tl = _task_lists(tasks_yaml)
    id_set = set(id_tasks if id_tasks is not None else tl["id"])
    ood_set = set(ood_tasks if ood_tasks is not None else tl["ood"])

    id_ex: List[Dict[str, Any]] = []
    ood_ex: List[Dict[str, Any]] = []
    other_ex: List[Dict[str, Any]] = []
    for ex in examples:
        t = ex.get("task")
        if t in id_set:
            ex["bucket"] = "id"
            id_ex.append(ex)
        elif t in ood_set:
            ex["bucket"] = "ood"
            ood_ex.append(ex)
        else:
            ex["bucket"] = "other"
            other_ex.append(ex)
    logger.info(
        "ID/OOD split: %d id / %d ood / %d other.", len(id_ex), len(ood_ex), len(other_ex)
    )
    return id_ex, ood_ex, other_ex


# ---------------------------------------------------------------------------
# One-shot driver
# ---------------------------------------------------------------------------
def load_bart0_r_pool(
    tasks: Optional[Sequence[str]] = None,
    seed: int = 42,
    tasks_yaml: Optional[Dict[str, Any]] = None,
    prefer_template_substrings: Optional[Sequence[str]] = None,
    cache_dir: Optional[str] = None,
    json_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the *pool* of BART0 D_R candidate examples (P3 test tasks).

    The pool is the set of examples the base model is asked about; only its
    mispredicted subset becomes ``D_R``.
    """
    tl = _task_lists(tasks_yaml)
    task_names = list(tasks) if tasks is not None else tl["bart0_r"]
    prefer = list(prefer_template_substrings or tl["prefer_template_substrings"])
    kwargs: Dict[str, Any] = {"split": "test", "prefer_substrings": prefer}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if json_dir is not None:
        kwargs["json_dir"] = json_dir
    pool: List[Dict[str, Any]] = []
    for tname in task_names:
        try:
            rows = load_p3_task(tname, **kwargs)
        except Exception as exc:
            logger.warning("Failed to load BART0 R task %s: %s", tname, exc)
            rows = []
        pool.extend(rows)
    logger.info("BART0 D_R pool: %d examples from %d tasks.", len(pool), len(task_names))
    return pool


def load_mmlu_r_pool(
    task_names: Optional[Sequence[str]] = None,
    seed: int = 42,
    tasks_yaml: Optional[Dict[str, Any]] = None,
    mmlu_data_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load the FLAN-T5 D_R candidate pool (MMLU validation, all 57 subjects)."""
    tl = _task_lists(tasks_yaml)
    subjects = list(task_names) if task_names is not None else tl["mmlu"]
    mapping = load_mmlu_tasks(
        subjects, split="validation", per_task=None, data_dir=mmlu_data_dir
    )
    pool = flatten_task_dict(mapping)
    logger.info("MMLU D_R pool: %d examples from %d subjects.", len(pool), len(subjects))
    return pool


def build_all(
    out_dir: str,
    model_key: str = "BART0_L",
    predictor: Any = None,
    r_predictor: Any = None,
    examples_per_task: int = 100,
    r_train_ratio: float = 0.6,
    seed: int = 42,
    tasks_yaml: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    json_dir: Optional[str] = None,
    mmlu_data_dir: Optional[str] = None,
    allow_passthrough: bool = True,
) -> Dict[str, Any]:
    """Build every dataset artifact and persist them under ``out_dir``.

    ``predictor`` grades ``D_PT`` (and typically is the same base LM as
    ``r_predictor``, which grades the D_R pool).  When ``allow_passthrough`` is
    False a missing predictor raises instead of silently keeping everything.

    Returns a manifest dict (also written to ``out_dir/manifest.json``).
    """
    if not allow_passthrough and (predictor is None or r_predictor is None):
        raise ValueError(
            "build_all(allow_passthrough=False) requires both `predictor` and "
            "`r_predictor` (a model or explicit predictions)."
        )
    os.makedirs(out_dir, exist_ok=True)

    # ---- D_PT -------------------------------------------------------------
    d_pt = build_d_pt(
        examples_per_task=examples_per_task,
        seed=seed,
        tasks_yaml=tasks_yaml,
        cache_dir=cache_dir,
        json_dir=json_dir,
    )
    save_jsonl(d_pt, os.path.join(out_dir, "d_pt.jsonl"))

    # ---- D_PT_hat ---------------------------------------------------------
    d_pt_hat, pt_manifest = filter_d_pt_hat(d_pt, predictor=predictor)

    # ---- D_R --------------------------------------------------------------
    model_key = (model_key or "").upper()
    if model_key.startswith("BART0"):
        pool = load_bart0_r_pool(
            seed=seed,
            tasks_yaml=tasks_yaml,
            cache_dir=cache_dir,
            json_dir=json_dir,
        )
        pool_name = "P3-test (BART0)"
    else:
        pool = load_mmlu_r_pool(
            seed=seed, tasks_yaml=tasks_yaml, mmlu_data_dir=mmlu_data_dir
        )
        pool_name = "MMLU validation (FLAN-T5)"
    save_jsonl(pool, os.path.join(out_dir, "r_pool.jsonl"))

    d_r, r_manifest = collect_d_r(pool, predictor=r_predictor)
    d_r_train, d_r_test = split_d_r(d_r, r_train_ratio=r_train_ratio, seed=seed)
    save_jsonl(d_r, os.path.join(out_dir, "d_r.jsonl"))
    save_jsonl(d_r_train, os.path.join(out_dir, "d_r_train.jsonl"))
    save_jsonl(d_r_test, os.path.join(out_dir, "d_r_test.jsonl"))
    save_jsonl(d_pt_hat, os.path.join(out_dir, "d_pt_hat.jsonl"))

    # ---- ID / OOD (BART0 only; Appendix B) --------------------------------
    id_manifest: Dict[str, Any] = {}
    if model_key.startswith("BART0"):
        tl = _task_lists(tasks_yaml)
        id_ex, ood_ex, other_ex = split_id_ood(
            d_r_test, id_tasks=tl["id"], ood_tasks=tl["ood"], tasks_yaml=tasks_yaml
        )
        save_jsonl(id_ex, os.path.join(out_dir, "d_r_test_id.jsonl"))
        save_jsonl(ood_ex, os.path.join(out_dir, "d_r_test_ood.jsonl"))
        id_manifest = {
            "n_id": len(id_ex),
            "n_ood": len(ood_ex),
            "n_other": len(other_ex),
        }

    manifest: Dict[str, Any] = {
        "model_key": model_key,
        "seed": seed,
        "examples_per_task": examples_per_task,
        "r_train_ratio": r_train_ratio,
        "d_pt": {"n": len(d_pt)},
        "d_pt_hat": pt_manifest,
        "r_pool": {"n": len(pool), "name": pool_name},
        "d_r": r_manifest,
        "d_r_train": {"n": len(d_r_train)},
        "d_r_test": {"n": len(d_r_test)},
        "id_ood": id_manifest,
    }
    # Positive (forgotten) prevalence sanity: the paper expects 1%-10% for
    # D_PT_hat, which is a property of the ground-truth stage, not of this
    # builder.  We still expose the pieces needed to check it downstream.
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    logger.info("Dataset manifest: %s", json.dumps(manifest, ensure_ascii=False))
    return manifest
