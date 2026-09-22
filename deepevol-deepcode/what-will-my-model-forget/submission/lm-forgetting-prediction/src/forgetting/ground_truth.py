"""Ground-truth forgetting labels ``z_ij`` and cached logit streams.

Paper reference
---------------
* Sec. 2 (*Forecasting Forgotten Examples*)::

      D_R                = mispredicted examples of f0 on a new task
      f_i                = f0 refined for K steps on <x_i, y_i>
      D_PT^Fgt,i         = {<x_j, y_j> in D_PT_hat | f_i(x_j) != y_j}
      z_ij = 1           iff <x_j, y_j> is forgotten upon learning <x_i, y_i>
      EM_{D,f}           = |{<x, y> in D | f(x) = y}| / |D|
      Edit Success Rate  = |{<x_i,y_i> in D_R | f_i(x_i) = y_i}| / |D_R|

* Algorithms 1 & 3 (Appendix F) describe *training* of the forecasters: one online
  example ``<x_i, y_i>`` and one upstream example ``<x_j, y_j>`` are sampled per
  iteration, ``f_i`` is obtained by updating ``f_0`` with ``<x_i, y_i>``, and the
  logit streams (``f0(x_i)``, ``f_i(x_i)``, ``f0(x_j)``, ``f_i(x_j)``) are obtained.

IMPORTANT DISCREPANCY (flagged in the reproduction plan)
--------------------------------------------------------
Appendix F writes ``z_ij <- 1 if f_0(x_i) != f_i(x_i) else 0``, i.e. it labels the
*online* example with its own correctness change.  That is a typo: every
``<x_i, y_i> in D_R`` satisfies ``f_0(x_i) != y_i`` by construction, while the
quantity Sec. 2 defines and the forecasters predict is whether the *upstream*
example ``<x_j, y_j>`` is forgotten::

      z_ij = 1[f_i(x_j) != y_j]        # Sec. 2 -- USED HERE

``1[f_0(x_i) != f_i(x_i)]`` is recorded separately as ``f0_correct``/``fi_correct``
on the online record for completeness, but never used as ``z_ij``.

Cheap-forecasting contract (Sec. 2): ``f_i(x_j)`` is evaluated *only* while
generating the supervision (this module).  At forecast time the forecasters consume
the persisted caches (``f0(x_j)`` top-k logits, ``h(x_j, y_j)``) and must not re-run
the LM over ``D_PT_hat``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "TopKLogits",
    "OnlineRecord",
    "PairRecord",
    "topk_from_logits",
    "predict_examples",
    "token_logits",
    "iter_token_logits",
    "ground_truth_label",
    "edit_success",
    "sample_online_subset",
    "sample_upstream_subset",
    "sample_upstream_indices",
    "generate_forgetting_pairs",
    "sample_pairs_with_model",
    "save_ground_truth_jsonl",
    "load_ground_truth_jsonl",
    "positive_prevalence",
    "summarize_records",
    "verify_labels_bruteforce",
    "main",
]

logger = logging.getLogger("forgetting.ground_truth")

# --------------------------------------------------------------------------------------
# Small containers
# --------------------------------------------------------------------------------------


@dataclass
class TopKLogits:
    """Top-``k`` logit values/indices of the decoder output distribution.

    ``values[t][v]`` is the ``v``-th largest logit at output position ``t`` and
    ``indices[t][v]`` is the corresponding vocabulary index.  The paper caches only
    the top ``k = 100`` logits per output token of ``y_j`` (Sec. 3.2, "Efficient
    Inference"; Algorithm 4).
    """

    indices: List[List[int]] = field(default_factory=list)
    values: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"indices": self.indices, "values": self.values}

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "TopKLogits":
        d = d or {}
        return cls(indices=d.get("indices", []) or [], values=d.get("values", []) or [])

    def __len__(self) -> int:  # number of cached output positions
        return len(self.indices)


@dataclass
class OnlineRecord:
    """Diagnostics for one online (model-refinement) example ``<x_i, y_i>``."""

    index: int
    id: str
    task: str
    input: str
    target: str
    f0_prediction: str
    fi_prediction: str
    f0_correct: bool
    fi_correct: bool
    edit_success: bool
    n_refine_steps: int = 0
    f0_token_logits: TopKLogits = field(default_factory=TopKLogits)
    fi_token_logits: TopKLogits = field(default_factory=TopKLogits)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["f0_token_logits"] = self.f0_token_logits.to_dict()
        d["fi_token_logits"] = self.fi_token_logits.to_dict()
        return d


@dataclass
class PairRecord:
    """One ``(online i, upstream j)`` pair with its ground-truth forgetting label."""

    i: int
    j: int
    z: int
    f0_j_prediction: str
    fi_j_prediction: str
    f0_j_correct: bool
    fi_j_correct: bool
    f0_j_token_logits: TopKLogits = field(default_factory=TopKLogits)
    fi_j_token_logits: TopKLogits = field(default_factory=TopKLogits)
    i_id: str = ""
    i_task: str = ""
    j_id: str = ""
    j_task: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["f0_j_token_logits"] = self.f0_j_token_logits.to_dict()
        d["fi_j_token_logits"] = self.fi_j_token_logits.to_dict()
        return d


# --------------------------------------------------------------------------------------
# Model-interface adapters (tolerant to the API exposed by base_lm / refinement)
# --------------------------------------------------------------------------------------


def _as_list_of_str(seq: Any) -> List[str]:
    if isinstance(seq, str):
        return [seq]
    return ["" if s is None else str(s) for s in seq]


def predict_examples(model: Any, inputs: Sequence[str], batch_size: int = 8, **gen_kwargs: Any) -> List[str]:
    """Generate predictions for ``inputs`` with any supported model contract.

    Accepts an object exposing ``generate`` / ``predict`` / ``predict_batch`` /
    ``batch_generate`` (or a plain callable).
    """
    inputs = _as_list_of_str(inputs)
    if not inputs:
        return []
    for name in ("predict", "predict_batch", "generate", "batch_generate"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                out = fn(list(inputs), **gen_kwargs)
            except TypeError:
                out = fn(list(inputs))
            out = _as_list_of_str(out)
            if len(out) != len(inputs):
                raise ValueError(
                    "predictor %r returned %d outputs for %d inputs" % (name, len(out), len(inputs))
                )
            return out
    if callable(model):
        out = _as_list_of_str(model(list(inputs)))
        if len(out) != len(inputs):
            raise ValueError("callable predictor returned %d outputs for %d inputs" % (len(out), len(inputs)))
        return out
    raise TypeError(
        "model %r does not expose any of generate/predict/predict_batch/batch_generate" % type(model)
    )


def _extract_token_logits(raw: Any) -> List[Any]:
    """Normalize the return value of a logits-producing call into a list of tensors."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        if hasattr(raw, "dim") and raw.dim() > 1:
            return [raw[k] for k in range(raw.shape[0])]
    except Exception:  # pragma: no cover - defensive
        pass
    return [raw]


def _call_logits(model: Any, inputs: Sequence[str], targets: Sequence[str], **kw: Any) -> List[Any]:
    for name in ("token_logits", "sequence_logits", "logits", "get_token_logits", "forward_logits"):
        fn = getattr(model, name, None)
        if callable(fn):
            return _extract_token_logits(fn(list(inputs), list(targets), **kw))
    raise TypeError(
        "model %r does not expose a token-logit API (token_logits / sequence_logits / logits)" % type(model)
    )


def topk_from_logits(logits: Any, k: int = 100) -> TopKLogits:
    """Top-``k`` values/indices of a per-position logit tensor.

    ``logits`` may be a ``[T, V]`` tensor (one sequence), a ``[B, T, V]`` tensor
    (batch element 0 is taken) or an already materialized nested list.
    """
    k = max(1, int(k))
    if hasattr(logits, "dim"):
        if logits.dim() == 3:
            logits = logits[0]
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        try:
            import torch  # local import: torch stays optional for pure-python use

            vals, idx = torch.topk(logits.float(), min(k, logits.shape[-1]), dim=-1)
            return TopKLogits(
                indices=[[int(v) for v in row] for row in idx.tolist()],
                values=[[float(v) for v in row] for row in vals.tolist()],
            )
        except Exception:  # pragma: no cover - fall back to python
            logits = logits.tolist()
    if isinstance(logits, (list, tuple)):
        if logits and isinstance(logits[0], (list, tuple)) and logits[0] and isinstance(logits[0][0], (list, tuple)):
            logits = logits[0]
        if logits and not isinstance(logits[0], (list, tuple)):
            logits = [logits]
        indices: List[List[int]] = []
        values: List[List[float]] = []
        for row in logits:
            pairs = sorted(range(len(row)), key=lambda v: float(row[v]), reverse=True)[:k]
            indices.append([int(v) for v in pairs])
            values.append([float(row[v]) for v in pairs])
        return TopKLogits(indices=indices, values=values)
    return TopKLogits()


def token_logits(
    model: Any, inputs: Sequence[str], targets: Sequence[str], k: int = 100, **kw: Any
) -> List[TopKLogits]:
    """Top-``k`` teacher-forced output logits (on ``targets``) for each input."""
    inputs = _as_list_of_str(inputs)
    targets = _as_list_of_str(targets)
    if not inputs:
        return []
    raw = _call_logits(model, inputs, targets, **kw)
    if len(raw) != len(inputs):
        raise ValueError("logit API returned %d tensors for %d inputs" % (len(raw), len(inputs)))
    return [topk_from_logits(t, k=k) for t in raw]


def iter_token_logits(
    model: Any,
    inputs: Sequence[str],
    targets: Sequence[str],
    k: int = 100,
    batch_size: int = 8,
    **kw: Any,
) -> Iterator[TopKLogits]:
    """Batched generator over :func:`token_logits` (keeps peak memory bounded)."""
    inputs = _as_list_of_str(inputs)
    targets = _as_list_of_str(targets)
    batch_size = max(1, int(batch_size))
    for start in range(0, len(inputs), batch_size):
        chunk_in = inputs[start : start + batch_size]
        chunk_tg = targets[start : start + batch_size]
        for tl in token_logits(model, chunk_in, chunk_tg, k=k, **kw):
            yield tl


# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------


def ground_truth_label(fi_prediction_xj: str, y_j: str, references: Optional[Sequence[str]] = None) -> int:
    """``z_ij = 1[f_i(x_j) != y_j]`` (Sec. 2).

    Grading uses the SQuAD-2.0 style exact match of :mod:`src.data.em_eval` so that
    labels are consistent with the EM used to build ``D_R`` and ``D_PT_hat``
    (multi-reference examples are graded max-over-references).
    """
    from ..data.em_eval import is_correct

    refs = list(references) if references else [y_j]
    return 0 if is_correct(fi_prediction_xj, refs) else 1


def edit_success(fi_prediction_xi: str, y_i: str, references: Optional[Sequence[str]] = None) -> bool:
    """``f_i(x_i) == y_i`` (Sec. 2, Edit Success Rate)."""
    from ..data.em_eval import is_correct

    refs = list(references) if references else [y_i]
    return bool(is_correct(fi_prediction_xi, refs))


# --------------------------------------------------------------------------------------
# Sampling helpers
# --------------------------------------------------------------------------------------


def _rng(seed: int) -> random.Random:
    return random.Random(seed)


def sample_online_subset(examples: Sequence[Dict[str, Any]], n: Optional[int] = None, seed: int = 42) -> List[int]:
    """Indices of online examples to refine on (all of them when ``n`` is None)."""
    idx = list(range(len(examples)))
    if n is None or n >= len(idx):
        return idx
    return sorted(_rng(seed).sample(idx, int(n)))


def sample_upstream_subset(examples: Sequence[Dict[str, Any]], n: Optional[int] = None, seed: int = 42) -> List[int]:
    """Indices of upstream examples to label (all of ``D_PT_hat`` when ``n`` is None)."""
    idx = list(range(len(examples)))
    if n is None or n >= len(idx):
        return idx
    return sorted(_rng(seed).sample(idx, int(n)))


def sample_upstream_indices(
    pool_size: int,
    n_pos: int = 8,
    n_neg: int = 8,
    labels: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> List[int]:
    """Sample ``n_pos`` forgotten + ``n_neg`` non-forgotten upstream indices.

    Appendix B / Algorithms 1 & 3: each mini-batch of size 16 holds 8 positive and 8
    negative pairs.  Because positive prevalence is only ~1-10%, upstream examples
    must be sampled *within* each class once the label vector is known; when
    ``labels`` is None a plain random sample of ``n_pos + n_neg`` indices is returned
    (labels are unknown before ``f_i`` is run).
    """
    rng = _rng(seed)
    if labels is None:
        total = min(pool_size, n_pos + n_neg)
        return sorted(rng.sample(range(pool_size), total)) if total < pool_size else list(range(pool_size))
    positives = [k for k, v in enumerate(labels) if int(v) == 1]
    negatives = [k for k, v in enumerate(labels) if int(v) == 0]
    out: List[int] = []
    if positives:
        out += rng.sample(positives, n_pos) if n_pos < len(positives) else list(positives)
    if negatives:
        out += rng.sample(negatives, n_neg) if n_neg < len(negatives) else list(negatives)
    return sorted(out)


# --------------------------------------------------------------------------------------
# Core generation
# --------------------------------------------------------------------------------------


def _example_input(ex: Dict[str, Any]) -> str:
    return str(ex.get("input", ex.get("prompt", "")) or "")


def _example_target(ex: Dict[str, Any]) -> str:
    return str(ex.get("target", ex.get("label", "")) or "")


def _example_refs(ex: Dict[str, Any]) -> List[str]:
    try:
        from ..data.em_eval import extract_references

        return extract_references(ex)
    except Exception:
        refs = ex.get("references") or ex.get("targets")
        if refs:
            return [str(r) for r in refs]
        return [_example_target(ex)]


def generate_forgetting_pairs(
    f0: Any,
    refine_fn: Callable[[Dict[str, Any]], Any],
    online_examples: Sequence[Dict[str, Any]],
    upstream_examples: Sequence[Dict[str, Any]],
    online_indices: Optional[Sequence[int]] = None,
    upstream_indices: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    topk: int = 100,
    collect_logits: bool = True,
    progress: bool = True,
) -> Tuple[List[OnlineRecord], List[PairRecord]]:
    """Ground-truth ``z_ij`` labels plus the logit streams (Algorithm 1/3 data).

    For each online example ``<x_i, y_i>``, one at a time (the paper's "fix one error
    at a time" protocol):

    1. ``f0(x_i)`` prediction and top-``k`` logits;
    2. ``f_i <- refine_fn(<x_i, y_i>)`` -- K gradient steps from ``f_0``;
    3. ``f_i(x_i)`` prediction and top-``k`` logits (Edit Success member);
    4. for every ``<x_j, y_j>`` in ``D_PT_hat``: ``f_i(x_j)`` prediction giving
       ``z_ij = 1[f_i(x_j) != y_j]``, with ``f0(x_j)`` and ``f_i(x_j)`` top-``k``
       logits recorded for the caches.

    ``refine_fn`` maps an online example dict to the refined model ``f_i`` (see
    :mod:`src.modeling.refinement`).  ``f0(x_j)`` is computed once outside the loop
    since it does not depend on ``i``.

    Returns ``(online_records, pair_records)``.
    """
    online_indices = list(range(len(online_examples))) if online_indices is None else list(online_indices)
    upstream_indices = list(range(len(upstream_examples))) if upstream_indices is None else list(upstream_indices)

    up_inputs = [_example_input(upstream_examples[j]) for j in upstream_indices]
    up_targets = [_example_target(upstream_examples[j]) for j in upstream_indices]
    up_refs = [_example_refs(upstream_examples[j]) for j in upstream_indices]

    logger.info("computing f0 predictions/top-%d logits on %d upstream examples", topk, len(up_inputs))
    f0_up_predictions = predict_examples(f0, up_inputs, batch_size=batch_size)
    if collect_logits:
        f0_up_logits = list(iter_token_logits(f0, up_inputs, up_targets, k=topk, batch_size=batch_size))
    else:
        f0_up_logits = [TopKLogits() for _ in up_inputs]

    online_records: List[OnlineRecord] = []
    pair_records: List[PairRecord] = []

    for n, i in enumerate(online_indices):
        ex_i = online_examples[i]
        x_i, y_i = _example_input(ex_i), _example_target(ex_i)
        refs_i = _example_refs(ex_i)
        if progress:
            logger.info(
                "[%d/%d] refining on online example %s (%s)",
                n + 1,
                len(online_indices),
                ex_i.get("id", i),
                ex_i.get("task", ""),
            )

        f0_pred_i = predict_examples(f0, [x_i], batch_size=1)[0]
        f0_i_logits = token_logits(f0, [x_i], [y_i], k=topk)[0] if collect_logits else TopKLogits()

        f_i = refine_fn(ex_i)

        fi_pred_i = predict_examples(f_i, [x_i], batch_size=1)[0]
        fi_i_logits = token_logits(f_i, [x_i], [y_i], k=topk)[0] if collect_logits else TopKLogits()
        edit_ok = edit_success(fi_pred_i, y_i, refs_i)

        online_records.append(
            OnlineRecord(
                index=int(i),
                id=str(ex_i.get("id", i)),
                task=str(ex_i.get("task", "")),
                input=x_i,
                target=y_i,
                f0_prediction=f0_pred_i,
                fi_prediction=fi_pred_i,
                f0_correct=bool(edit_success(f0_pred_i, y_i, refs_i)),
                fi_correct=bool(edit_ok),
                edit_success=bool(edit_ok),
                n_refine_steps=int(getattr(f_i, "n_steps", 0) or 0),
                f0_token_logits=f0_i_logits,
                fi_token_logits=fi_i_logits,
            )
        )

        # ---- upstream behaviour of f_i -------------------------------------------------
        fi_up_predictions = predict_examples(f_i, up_inputs, batch_size=batch_size)
        if collect_logits:
            fi_up_logits = list(iter_token_logits(f_i, up_inputs, up_targets, k=topk, batch_size=batch_size))
        else:
            fi_up_logits = [TopKLogits() for _ in up_inputs]

        for pos, j in enumerate(upstream_indices):
            ex_j = upstream_examples[j]
            z = ground_truth_label(fi_up_predictions[pos], _example_target(ex_j), up_refs[pos])
            pair_records.append(
                PairRecord(
                    i=int(i),
                    j=int(j),
                    z=int(z),
                    f0_j_prediction=f0_up_predictions[pos],
                    fi_j_prediction=fi_up_predictions[pos],
                    f0_j_correct=bool(edit_success(f0_up_predictions[pos], _example_target(ex_j), up_refs[pos])),
                    fi_j_correct=bool(edit_success(fi_up_predictions[pos], _example_target(ex_j), up_refs[pos])),
                    f0_j_token_logits=f0_up_logits[pos],
                    fi_j_token_logits=fi_up_logits[pos] if pos < len(fi_up_logits) else TopKLogits(),
                    i_id=str(ex_i.get("id", i)),
                    i_task=str(ex_i.get("task", "")),
                    j_id=str(ex_j.get("id", j)),
                    j_task=str(ex_j.get("task", "")),
                )
            )

        del f_i  # release GPU memory before the next refinement

    return online_records, pair_records


def sample_pairs_with_model(
    f0: Any,
    refine_fn: Callable[[Dict[str, Any]], Any],
    online_examples: Sequence[Dict[str, Any]],
    upstream_examples: Sequence[Dict[str, Any]],
    n_pairs: int = 16,
    n_pos: int = 8,
    n_neg: int = 8,
    online_index: Optional[int] = None,
    batch_size: int = 8,
    topk: int = 100,
    seed: int = 42,
    collect_logits: bool = True,
) -> Tuple[List[OnlineRecord], List[PairRecord]]:
    """Sampler used inside the forecaster training loop (Algorithms 1 & 3).

    Samples one online example (or uses ``online_index``), refines ``f_0`` once and
    then samples ``n_pos`` forgotten + ``n_neg`` non-forgotten upstream examples
    *from the resulting label vector* (a plain random upstream sample would almost
    never contain positives at 1-10% prevalence).  Falls back to a plain random
    sample when a class split is impossible.
    """
    rng = _rng(seed + (0 if online_index is None else int(online_index)))
    if online_index is None:
        online_index = rng.randrange(len(online_examples))

    ex_i = online_examples[online_index]
    x_i, y_i = _example_input(ex_i), _example_target(ex_i)
    refs_i = _example_refs(ex_i)

    f0_pred_i = predict_examples(f0, [x_i], batch_size=1)[0]
    f0_i_logits = token_logits(f0, [x_i], [y_i], k=topk)[0] if collect_logits else TopKLogits()
    f_i = refine_fn(ex_i)
    fi_pred_i = predict_examples(f_i, [x_i], batch_size=1)[0]
    fi_i_logits = token_logits(f_i, [x_i], [y_i], k=topk)[0] if collect_logits else TopKLogits()

    up_inputs = [_example_input(e) for e in upstream_examples]
    up_targets = [_example_target(e) for e in upstream_examples]
    f0_up_predictions = predict_examples(f0, up_inputs, batch_size=batch_size)
    f0_up_logits = (
        list(iter_token_logits(f0, up_inputs, up_targets, k=topk, batch_size=batch_size))
        if collect_logits
        else [TopKLogits() for _ in up_inputs]
    )
    fi_up_predictions = predict_examples(f_i, up_inputs, batch_size=batch_size)
    fi_up_logits = (
        list(iter_token_logits(f_i, up_inputs, up_targets, k=topk, batch_size=batch_size))
        if collect_logits
        else [TopKLogits() for _ in up_inputs]
    )

    labels = [
        ground_truth_label(fi_up_predictions[p], _example_target(upstream_examples[p]), _example_refs(upstream_examples[p]))
        for p in range(len(upstream_examples))
    ]
    keep = sample_upstream_indices(len(upstream_examples), n_pos=n_pos, n_neg=n_neg, labels=labels, seed=seed)
    if not keep:
        keep = list(range(min(len(upstream_examples), n_pairs)))

    online = OnlineRecord(
        index=int(online_index),
        id=str(ex_i.get("id", online_index)),
        task=str(ex_i.get("task", "")),
        input=x_i,
        target=y_i,
        f0_prediction=f0_pred_i,
        fi_prediction=fi_pred_i,
        f0_correct=bool(edit_success(f0_pred_i, y_i, refs_i)),
        fi_correct=bool(edit_success(fi_pred_i, y_i, refs_i)),
        edit_success=bool(edit_success(fi_pred_i, y_i, refs_i)),
        n_refine_steps=int(getattr(f_i, "n_steps", 0) or 0),
        f0_token_logits=f0_i_logits,
        fi_token_logits=fi_i_logits,
    )

    pairs: List[PairRecord] = []
    for j in keep:
        ex_j = upstream_examples[j]
        refs_j = _example_refs(ex_j)
        pairs.append(
            PairRecord(
                i=int(online_index),
                j=int(j),
                z=int(labels[j]),
                f0_j_prediction=f0_up_predictions[j],
                fi_j_prediction=fi_up_predictions[j],
                f0_j_correct=bool(edit_success(f0_up_predictions[j], _example_target(ex_j), refs_j)),
                fi_j_correct=bool(edit_success(fi_up_predictions[j], _example_target(ex_j), refs_j)),
                f0_j_token_logits=f0_up_logits[j],
                fi_j_token_logits=fi_up_logits[j],
                i_id=online.id,
                i_task=online.task,
                j_id=str(ex_j.get("id", j)),
                j_task=str(ex_j.get("task", "")),
            )
        )
    del f_i
    return [online], pairs


# --------------------------------------------------------------------------------------
# Persistence / statistics
# --------------------------------------------------------------------------------------


def save_ground_truth_jsonl(
    online_records: Sequence[OnlineRecord],
    pair_records: Sequence[PairRecord],
    out_dir: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Persist online records, pair records and meta; returns the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    online_path = os.path.join(out_dir, "online.jsonl")
    pair_path = os.path.join(out_dir, "pairs.jsonl")
    meta_path = os.path.join(out_dir, "meta.json")

    with open(online_path, "w", encoding="utf-8") as fh:
        for rec in online_records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
    with open(pair_path, "w", encoding="utf-8") as fh:
        for rec in pair_records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    meta = dict(meta or {})
    meta.update(
        {
            "n_online": len(online_records),
            "n_pairs": len(pair_records),
            "positive_prevalence": positive_prevalence(pair_records),
            "edit_success_rate": (sum(1 for r in online_records if r.edit_success) / len(online_records))
            if online_records
            else 0.0,
        }
    )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    return {"online": online_path, "pairs": pair_path, "meta": meta_path}


def load_ground_truth_jsonl(path: str, cls: Any = PairRecord) -> List[Any]:
    """Load records written by :func:`save_ground_truth_jsonl`.

    ``cls`` is :class:`OnlineRecord` or :class:`PairRecord`; nested
    ``*_token_logits`` dicts are converted back into :class:`TopKLogits`.
    """
    out: List[Any] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if cls is OnlineRecord:
                raw["f0_token_logits"] = TopKLogits.from_dict(raw.get("f0_token_logits"))
                raw["fi_token_logits"] = TopKLogits.from_dict(raw.get("fi_token_logits"))
            else:
                raw["f0_j_token_logits"] = TopKLogits.from_dict(raw.get("f0_j_token_logits"))
                raw["fi_j_token_logits"] = TopKLogits.from_dict(raw.get("fi_j_token_logits"))
            out.append(cls(**raw))
    return out


def positive_prevalence(pair_records: Sequence[PairRecord]) -> float:
    """Fraction of forgotten pairs (expected to land in the 1%-10% range)."""
    if not pair_records:
        return 0.0
    return sum(1 for r in pair_records if int(r.z) == 1) / float(len(pair_records))


def summarize_records(online_records: Sequence[OnlineRecord], pair_records: Sequence[PairRecord]) -> Dict[str, Any]:
    """Aggregate diagnostics: prevalence, per-online positive counts, edit success."""
    per_online: Dict[str, Dict[str, int]] = {}
    for r in pair_records:
        d = per_online.setdefault(str(int(r.i)), {"n": 0, "n_fgt": 0})
        d["n"] += 1
        d["n_fgt"] += int(r.z)
    return {
        "n_online": len(online_records),
        "n_pairs": len(pair_records),
        "positive_prevalence": positive_prevalence(pair_records),
        "n_positive": sum(1 for r in pair_records if int(r.z) == 1),
        "edit_success_rate": (sum(1 for r in online_records if r.edit_success) / len(online_records))
        if online_records
        else 0.0,
        "mean_forgotten_per_online": (
            sum(d["n_fgt"] for d in per_online.values()) / len(per_online) if per_online else 0.0
        ),
        "per_online": per_online,
    }


def verify_labels_bruteforce(
    f_i: Any,
    upstream_examples: Sequence[Dict[str, Any]],
    pair_records: Sequence[PairRecord],
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Brute-force re-check of ``z_ij`` for one refined model (test helper).

    Re-runs ``f_i`` over the upstream examples of the given pairs and compares the
    recomputed labels / predictions against the stored ones.  ``tests/`` uses this to
    catch label-construction bugs.
    """
    pairs = list(pair_records)
    if not pairs:
        return {"n": 0, "n_mismatch": 0, "mismatch_rate": 0.0, "ok": True}
    j_idx = [int(r.j) for r in pairs]
    inputs = [_example_input(upstream_examples[j]) for j in j_idx]
    preds = predict_examples(f_i, inputs, batch_size=batch_size)
    mismatches = 0
    for r, pred in zip(pairs, preds):
        ex_j = upstream_examples[int(r.j)]
        z = ground_truth_label(pred, _example_target(ex_j), _example_refs(ex_j))
        if int(z) != int(r.z) or pred != r.fi_j_prediction:
            mismatches += 1
    rate = mismatches / float(len(pairs))
    return {"n": len(pairs), "n_mismatch": mismatches, "mismatch_rate": rate, "ok": rate == 0.0}


# --------------------------------------------------------------------------------------
# CLI (thin driver; scripts/generate_ground_truth.py is the main entry point)
# --------------------------------------------------------------------------------------


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "..", "config", "config.yaml")
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("could not load config %s (%s)", path, exc)
        return {}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ground-truth forgetting labels z_ij and logit streams.")
    p.add_argument("--config", default=None, help="path to config/config.yaml")
    p.add_argument("--model", default="BART0_L", help="base LM key (models section of the config)")
    p.add_argument("--mode", default="full_ft", choices=["head", "lora", "full_ft"], help="refinement setup")
    p.add_argument("--d-r", default=None, help="D_R (or D_R^Train) JSONL; default artifacts/<model>/d_r_train.jsonl")
    p.add_argument("--d-pt-hat", default=None, help="D_PT_hat JSONL; default artifacts/<model>/d_pt_hat.jsonl")
    p.add_argument("--out", default=None, help="output dir; default artifacts/<model>/ground_truth/<mode>")
    p.add_argument("--n-online", type=int, default=None, help="limit the number of refined online examples")
    p.add_argument("--pool-size", type=int, default=None, help="limit the number of labelled upstream examples")
    p.add_argument("--topk", type=int, default=None, help="cached logits per output token (default 100)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-logits", action="store_true", help="skip logit caching (labels only)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    cfg = _load_config(args.config)

    def cfg_get(*keys: str, default: Any = None) -> Any:
        cur: Any = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    out_base = cfg_get("output_dir", default="artifacts")
    if not os.path.isabs(out_base):
        out_base = os.path.join(repo_root, out_base)
    seed = int(args.seed if args.seed is not None else cfg_get("seed", default=42))
    topk = int(args.topk if args.topk is not None else cfg_get("forecaster", "cache_topk", default=100))

    d_r_path = args.d_r or os.path.join(out_base, args.model, "d_r_train.jsonl")
    d_pt_hat_path = args.d_pt_hat or os.path.join(out_base, args.model, "d_pt_hat.jsonl")
    out_dir = args.out or os.path.join(out_base, args.model, "ground_truth", args.mode)

    if not os.path.exists(d_r_path) or not os.path.exists(d_pt_hat_path):
        logger.error("missing inputs (%s, %s); run scripts/build_datasets.py first", d_r_path, d_pt_hat_path)
        return 2

    online_examples = _load_jsonl(d_r_path)
    upstream_examples = _load_jsonl(d_pt_hat_path)
    logger.info("loaded %d online and %d upstream examples", len(online_examples), len(upstream_examples))

    # Heavy imports are late so the module stays importable without torch installed.
    from ..modeling.base_lm import load_base_lm
    from ..modeling.refinement import build_refinement_engine

    f0 = load_base_lm(
        args.model,
        device=cfg_get("device", default="cuda"),
        dtype=cfg_get("dtype", default="float32"),
        cache_dir=cfg_get("cache_dir", default=None),
        max_input_len=int(cfg_get("data", "max_input_len", default=512)),
        max_output_len=int(cfg_get("data", "max_output_len", default=64)),
    )
    engine = build_refinement_engine(f0, model_key=args.model, mode=args.mode, config=cfg, seed=seed)

    online_idx = sample_online_subset(online_examples, args.n_online, seed=seed)
    pool_idx = sample_upstream_subset(upstream_examples, args.pool_size, seed=seed)

    online_records, pair_records = generate_forgetting_pairs(
        f0,
        engine.refine,
        online_examples,
        upstream_examples,
        online_indices=online_idx,
        upstream_indices=pool_idx,
        batch_size=args.batch_size,
        topk=topk,
        collect_logits=not args.no_logits,
    )
    paths = save_ground_truth_jsonl(
        online_records,
        pair_records,
        out_dir,
        meta={
            "model": args.model,
            "mode": args.mode,
            "seed": seed,
            "topk": topk,
            "n_online_requested": len(online_idx),
            "n_pool_requested": len(pool_idx),
            "z_definition": "z_ij = 1[f_i(x_j) != y_j]  (Sec. 2)",
            "note": (
                "Appendix F writes z_ij = 1[f0(x_i) != f_i(x_i)]; treated as a typo per the "
                "reproduction plan, Sec. 2 definition used."
            ),
        },
    )
    logger.info("summary: %s", json.dumps(summarize_records(online_records, pair_records), indent=2)[:2000])
    logger.info("wrote %s", paths)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
