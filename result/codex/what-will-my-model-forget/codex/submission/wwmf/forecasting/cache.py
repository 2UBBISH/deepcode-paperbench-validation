"""Pre-computation of everything the forecasting methods reuse.

The paper stresses computational efficiency (Sec. 3.2 / Sec. 5.3): the logits and
representations of the upstream examples are computed *once* and cached, and are
then reused for every online learning example.
"""
from __future__ import annotations

import collections
import os
import pickle
from typing import Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from ..config import ExperimentConfig, TOPK_CACHED_LOGITS
from ..data.types import Dataset, Example
from ..evaluation.metrics import exact_match
from ..models.lm import Seq2SeqLM
from ..models.tuning import fix_single_error
from ..utils import ensure_dir
from .types import CandidateVocab, OnlineArtifact, UpstreamCache, UpstreamExampleCache


def build_candidate_vocab(
    reduced_logits: Sequence[np.ndarray],
    gold_ids: Sequence[np.ndarray],
    topk: int = TOPK_CACHED_LOGITS,
    max_size: int = 512,
) -> CandidateVocab:
    """Union of the per-position top-k vocabulary ids of all upstream examples."""
    top_ids = []
    for logits, gold in zip(reduced_logits, gold_ids):
        k = min(topk, logits.shape[1])
        top = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        top_ids.append(np.concatenate([top, np.asarray(gold).reshape(-1, 1)], axis=1))
    return build_candidate_vocab_from_ids(top_ids, topk=topk, max_size=max_size)


def build_candidate_vocab_from_ids(
    top_ids: Sequence[np.ndarray],
    topk: int = TOPK_CACHED_LOGITS,
    max_size: int = 512,
) -> CandidateVocab:
    """Union of per-example top-k vocabulary ids (memory-light first pass).
    Ids that appear most often among the top-k are kept when pruning to
    ``max_size`` columns.
    """
    counts: "collections.Counter" = collections.Counter()
    for arr in top_ids:
        counts.update(int(v) for v in np.asarray(arr).reshape(-1).tolist())
    ordered = sorted(vid for vid, _ in counts.most_common(max_size))
    return CandidateVocab(np.asarray(ordered, dtype=np.int64), topk=topk, max_size=max_size)


def build_upstream_cache(
    base_model: Seq2SeqLM,
    upstream: Dataset,
    *,
    topk: int = TOPK_CACHED_LOGITS,
    max_vocab: int = 512,
    batch_size: int = 8,
    chunk_size: int = 256,
    verbose: bool = True,
    cache_path: Optional[str] = None,
) -> UpstreamCache:
    """Cache logits, gold ids and representations of every example of ``D_PT``.

    Also records whether ``f_0`` answers the example correctly; the forecasting
    task is evaluated on ``D_hat_PT`` (see the addendum to Sec. 3.1).

    The upstream set is processed in chunks and the logits are stored *reduced to
    the candidate vocabulary* (``[T, C]`` with ``C <= max_vocab``): caching the full
    ``[T, V]`` logits of 3,600 examples would need tens of gigabytes
    (``3600 * 32 * 32128`` floats).  The candidate vocabulary is discovered in a
    first pass that only keeps the top-k ids of every example.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    inputs = [e.input for e in upstream]
    targets = [e.target for e in upstream]
    n = len(upstream)
    pad_id = base_model._pad_token_id()

    # ---- pass 1: discover the candidate vocabulary (top-k ids of every example)
    top_ids: List[np.ndarray] = []
    order = np.arange(n)
    for start in tqdm(range(0, n, chunk_size), desc="candidate vocabulary", disable=not verbose):
        idx = order[start:start + chunk_size]
        logits = base_model.teacher_forced_logits(
            [inputs[i] for i in idx], [targets[i] for i in idx], batch_size=batch_size
        )
        for example_logits, gold in logits:
            k = min(topk, example_logits.shape[1])
            top = np.argpartition(-example_logits, kth=k - 1, axis=1)[:, :k]
            top_ids.append(np.concatenate([top, np.asarray(gold).reshape(-1, 1)], axis=1))
    vocab = build_candidate_vocab_from_ids(top_ids, topk=topk, max_size=max_vocab)

    # ---- pass 2: reduced logits, gold ids, representations and correctness
    items: List[UpstreamExampleCache] = []
    preds_all: List[str] = []
    for start in tqdm(range(0, n, chunk_size), desc="cache logits (f_0)", disable=not verbose):
        idx = order[start:start + chunk_size]
        chunk_x = [inputs[i] for i in idx]
        chunk_y = [targets[i] for i in idx]
        logits = base_model.teacher_forced_logits(chunk_x, chunk_y, batch_size=batch_size)
        reps = base_model.encode(chunk_x, chunk_y, batch_size=batch_size)
        preds = base_model.predict(chunk_x, batch_size=max(batch_size, 16))
        preds_all.extend(preds)
        for i, (example_logits, gold), (enc_states, dec_states), pred in zip(idx, logits, reps, preds):
            gold = np.asarray(gold, dtype=np.int64)
            reduced = example_logits[:, vocab.vocab_ids]
            pooled = np.concatenate([enc_states, dec_states], axis=0).mean(axis=0)
            items.append(
                UpstreamExampleCache(
                    example=upstream[i],
                    reduced_logits=reduced.astype(np.float16),
                    gold_ids=gold,
                    mask=gold != pad_id,
                    decoder_reps=dec_states.astype(np.float16),
                    pooled_rep=pooled.astype(np.float16),
                    correct=exact_match([pred], [upstream[i].target]) > 0.5,
                )
            )
    em = exact_match(preds_all, targets)
    cache = UpstreamCache(examples=list(upstream), vocab=vocab, items=items, base_em=em)
    if cache_path:
        ensure_dir(os.path.dirname(cache_path) or ".")
        with open(cache_path, "wb") as fh:
            pickle.dump(cache, fh)
    return cache


def build_online_artifacts(
    base_model: Seq2SeqLM,
    online_examples: Sequence[Example],
    upstream_cache: UpstreamCache,
    *,
    steps: int,
    lr: float,
    mode: str,
    collect_labels: bool = True,
    cache_frozen_reps: bool = True,
    batch_size: int = 8,
    verbose: bool = True,
    cache_path: Optional[str] = None,
) -> List[OnlineArtifact]:
    """Fix every online error separately with ``f_i`` and record what changed.

    For each ``<x_i, y_i>`` in ``D_R``:

    1. ``f_i`` is obtained by updating ``f_0`` for ``steps`` steps (Sec. 4.1);
    2. the logit change ``f_i(x_i) - f_0(x_i)`` is cached (Sec. 3.2);
    3. optionally, the ground-truth forgetting labels ``z_ij`` are obtained by
       running inference with ``f_i`` over ``D_PT`` (Sec. 2).  This is the
       expensive quantity the forecasting methods are supposed to replace.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    base_snapshot = base_model.snapshot()
    base_logits_x = [
        logits[:, upstream_cache.vocab.vocab_ids]
        for logits, _ in base_model.teacher_forced_logits(
            [e.input for e in online_examples],
            [e.target for e in online_examples],
            # batch size 1: the online logits are later compared with the logits
            # of the updated model computed for the same example one at a time.
            batch_size=1,
        )
    ]
    upstream_inputs = [e.input for e in upstream_cache.examples]
    upstream_targets = [e.target for e in upstream_cache.examples]
    frozen_reps: List[Optional[np.ndarray]] = [None] * len(online_examples)
    if cache_frozen_reps:
        # h(x_i, y_i) of the frozen base PTLM, required by the fixed logit-based
        # forecaster (Sec. 4.2).  Computed once, before any model update.
        reps = base_model.encode(
            [e.input for e in online_examples], [e.target for e in online_examples], batch_size=batch_size
        )
        frozen_reps = [dec.astype(np.float16) for _, dec in reps]

    artifacts: List[OnlineArtifact] = []
    iterator = tqdm(
        list(zip(online_examples, base_logits_x, frozen_reps)), desc="collect forgetting", disable=not verbose
    )
    for example, logits_0, frozen_rep in iterator:
        base_model.restore(base_snapshot)
        updated = base_model.clone()
        fix_single_error(updated, example, steps=steps, lr=lr, mode=mode)
        logits_i, gold_i = updated.teacher_forced_logits([example.input], [example.target], batch_size=1)[0]
        delta = logits_i[:, upstream_cache.vocab.vocab_ids] - logits_0
        pred_self = updated.predict([example.input], batch_size=1)[0]
        labels = None
        if collect_labels:
            preds = updated.predict(upstream_inputs, batch_size=max(batch_size, 16))
            labels = np.asarray(
                [0 if exact_match([p], [t]) > 0.5 else 1 for p, t in zip(preds, upstream_targets)],
                dtype=np.int8,
            )
        artifacts.append(
            OnlineArtifact(
                example=example,
                delta_logits=delta.astype(np.float16),
                gold_ids=np.asarray(gold_i, dtype=np.int64),
                mask=np.asarray(gold_i) != updated._pad_token_id(),
                labels=labels,
                edit_success=exact_match([pred_self], [example.target]) > 0.5,
                n_steps=steps,
                frozen_token_reps=frozen_rep,
            )
        )
    base_model.restore(base_snapshot)
    if cache_path:
        ensure_dir(os.path.dirname(cache_path) or ".")
        with open(cache_path, "wb") as fh:
            pickle.dump(artifacts, fh)
    return artifacts


def label_matrix(artifacts: Sequence[OnlineArtifact], n_upstream: Optional[int] = None) -> np.ndarray:
    """Stack the ground-truth forgetting labels into a ``n_online x n_upstream`` matrix."""
    rows = []
    for art in artifacts:
        if art.labels is None:
            raise ValueError("artifacts were built without ground-truth labels")
        rows.append(art.labels)
    matrix = np.stack(rows, axis=0)
    if n_upstream is not None:
        matrix = matrix[:, :n_upstream]
    return matrix


def default_cache_path(
    cfg: ExperimentConfig, kind: str, n_upstream: int, n_online: int, extra: str = ""
) -> str:
    """Deterministic cache filename for the expensive ground-truth computations."""
    name = f"{kind}_{cfg.model}_{cfg.tuning_mode}_up{n_upstream}_on{n_online}_seed{cfg.seed}"
    if extra:
        name += f"_{extra}"
    name += ".pkl"
    return os.path.join(cfg.cache_root, name)
