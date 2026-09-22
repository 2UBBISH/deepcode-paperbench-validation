"""Test suite for the *What Will My Model Forget?* reproduction.

This package collects the unit tests that guard the pieces of the pipeline which
the paper specifies analytically and which are easy to get subtly wrong:

* ``test_em_eval`` / ``test_metrics``
    SQuAD-2.0 exact match (normalization + max-over-references), binary
    forgetting-forecast F1 / precision / recall, Edit Success Rate and
    EM Drop Ratio.

* ``test_datasets``
    ``D_PT``/``D_PT_hat``/``D_R`` construction helpers, the reproducible 60/40
    ``D_R^Train``/``D_R^Test`` split and the BART0 ID/OOD partition
    (Appendix B).

* ``test_ground_truth``
    The forgetting label ``z_ij = 1[f_i(x_j) != y_j]`` (Sec. 2 definition; the
    Appendix F spelling with ``x_i`` is treated as a typo) plus brute-force
    verification helpers.

* ``test_prior``
    The frequency prior ``b_j = log P(z_ij = 1) - log P(z_ij = 0)`` (Sec. 3.3,
    Algorithms 3 & 4) and its JSON (de)serialization.

* ``test_forecasters``
    The threshold baseline (Eq. 1), the logit-change-transfer kernel /
    Eq. 2 prediction / Eq. 3 margin loss (Sec. 3.2) and the representation
    model ``sigmoid(<h_j, h_i> + b_j)`` with its ``w/o Prior`` ablation
    (Sec. 3.3, Table 1).

* ``test_caches``
    The persistent top-k logit / ``h`` / prior caches that make inference
    ``O(|D_PT_hat|)`` without re-running the PTLM (Sec. 3.2 "Efficient
    Inference", Sec. 3.3).

* ``test_replay``
    Replay selection strategies and the replay schedule (8 examples every 10
    steps for BART0_L / FLAN-T5_L, 4 every 5 steps for FLAN-T5_3B;
    Sec. 4.2, Appendix D.2), plus the continual-stream bookkeeping of Figure 3.

Design notes
------------
The tests intentionally avoid downloading the paper's checkpoints: everything is
exercised against small dummy models/tensors so the suite runs on CPU in a few
seconds.  The shared helpers below (``set_seed``, ``make_example``,
``make_pair_records``, ``tiny_topk_logits``) keep the individual test modules
terse and free of duplicated scaffolding.

Run the whole suite from the repository root with either

    python -m pytest lm-forgetting-prediction/tests -q

or, without pytest installed,

    python -m unittest discover -s lm-forgetting-prediction/tests -t .

Every leaf module is importable on its own (heavy dependencies such as
``torch``/``transformers``/``peft`` are imported lazily inside functions), so
partial environments can still execute the pure-Python tests.
"""

from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "set_seed",
    "make_example",
    "make_examples",
    "make_pair_record",
    "make_pair_records",
    "tiny_topk_logits",
    "TASK_A",
    "TASK_B",
]


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
#: Two fabricated task names used to exercise the ID/OOD and per-task bucketing
#: code paths without touching the real P3/MMLU registries.
TASK_A = "unit-test-task-a"
TASK_B = "unit-test-task-b"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> int:
    """Seed ``random`` (and torch/numpy when available) for reproducibility.

    The paper does not publish seeds, so every experiment module in this
    repository uses a fixed default of ``42``; tests mirror that so they behave
    identically across machines.
    """
    random.seed(seed)
    try:  # numpy is optional in the light/test environment
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy not installed
        pass
    try:  # torch is optional in the light/test environment
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - needs a GPU
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch not installed
        pass
    return seed


# ---------------------------------------------------------------------------
# Example factories (schema shared by p3_loader / mmlu_loader)
# ---------------------------------------------------------------------------
def make_example(
    input_text: str = "Question: what is 2 + 2?\nAnswer:",
    target: str = "4",
    task: str = TASK_A,
    index: Optional[int] = None,
    references: Optional[Sequence[str]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build one dataset example dict matching ``src.data``'s loader schema.

    Keys mirror :mod:`src.data.p3_loader`/:mod:`src.data.mmlu_loader`:
    ``input``, ``target``, ``references``, ``id``, ``task``, ``split`` and
    ``template_name``.  Extra keyword arguments are merged verbatim, which lets
    individual tests attach cached predictions (``f0_prediction``,
    ``fi_prediction``, ...) to an example.
    """
    example: Dict[str, Any] = {
        "id": f"{task}-{index}" if index is not None else task,
        "task": task,
        "split": "test",
        "template_name": "unit_test",
        "input": input_text,
        "target": target,
        "references": list(references) if references is not None else [target],
    }
    if index is not None:
        example["index"] = index
    example.update(extra)
    return example


def make_examples(
    n: int,
    task: str = TASK_A,
    target_pattern: str = "4",
    start_index: int = 0,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Create ``n`` synthetic examples with deterministic inputs/targets."""
    out: List[Dict[str, Any]] = []
    for offset in range(n):
        idx = start_index + offset
        out.append(
            make_example(
                input_text=kwargs.pop("input_text", "Question: q%d\nAnswer:" % idx),
                target=target_pattern,
                task=task,
                index=idx,
                **dict(kwargs),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Ground-truth pair factories (z_ij records)
# ---------------------------------------------------------------------------
def make_pair_record(
    i: int,
    j: int,
    z: int,
    f0_j_token_logits: Any = None,
    fi_j_token_logits: Any = None,
    f0_i_token_logits: Any = None,
    fi_i_token_logits: Any = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Create one plain-dict ``PairRecord``-shaped ground-truth row.

    ``z = 1`` means the upstream example ``x_j`` was *forgotten* after refining
    on the online example ``x_i``, i.e. ``f_i(x_j) != y_j`` (Sec. 2).  The
    accessors in the forecasters/eval modules all tolerate dicts as well as the
    dataclass form, so dicts keep the tests dependency-free.
    """
    record: Dict[str, Any] = {
        "i": i,
        "j": j,
        "z": int(z),
        "i_id": f"online-{i}",
        "j_id": f"upstream-{j}",
        "i_task": TASK_A,
        "j_task": TASK_A,
        "f0_i_token_logits": f0_i_token_logits,
        "fi_i_token_logits": fi_i_token_logits,
        "f0_j_token_logits": f0_j_token_logits,
        "fi_j_token_logits": fi_j_token_logits,
    }
    record.update(extra)
    return record


def make_pair_records(
    pairs: Iterable[Tuple[int, int, int]],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Materialize an iterable of ``(i, j, z)`` triples into pair records."""
    return [make_pair_record(i, j, z, **dict(kwargs)) for (i, j, z) in pairs]


# ---------------------------------------------------------------------------
# Top-k logit helpers
# ---------------------------------------------------------------------------
class _DictTopKLogits(dict):
    """dict subclass exposing ``.indices`` / ``.values`` attribute access.

    ``ground_truth.TopKLogits`` is a lightweight container; several downstream
    modules read it either as a mapping or via attributes.  This shim satisfies
    both access patterns so the pure-Python tests need not import torch.
    """

    @property
    def indices(self) -> Any:
        return self.get("indices")

    @property
    def values(self) -> Any:
        return self.get("values")


def tiny_topk_logits(
    indices: Sequence[Sequence[int]],
    values: Sequence[Sequence[float]],
) -> _DictTopKLogits:
    """Build a minimal top-k logits container ``{indices, values}``.

    ``indices`` and ``values`` are lists (one per output token) of equal length;
    this mirrors the cache format produced by
    :mod:`src.modeling.caches` (top ``k = 100`` logits per output token).
    """
    if len(indices) != len(values):
        raise ValueError("indices and values must have the same number of tokens")
    for row_i, row_v in zip(indices, values):
        if len(row_i) != len(row_v):
            raise ValueError("each token row must have matching indices/values")
    return _DictTopKLogits(indices=[list(r) for r in indices], values=[list(r) for r in values])
