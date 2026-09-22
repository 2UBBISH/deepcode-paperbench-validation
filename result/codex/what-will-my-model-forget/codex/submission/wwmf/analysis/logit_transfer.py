"""Figure 2(a) -- inspection of the logit change transfer.

Figure 2(a) shows that, after fixing an error in the online learned example
``<x_i, y_i>``, the logits of tokens such as "not" and "duplicates" change a lot
although their probabilities are close to zero, and that part of this change
transfers to the upstream example ``<x_j, y_j>`` and flips its prediction.

This module reproduces that analysis: it needs one pair of examples, the base
model and the model updated on ``<x_i, y_i>``, and reports the largest logit
changes of both examples as well as the top-2 candidates of ``x_j`` before and
after the update.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.types import Example
from ..models.lm import Seq2SeqLM


def _top_changes(logits_before: np.ndarray, logits_after: np.ndarray, tokenizer, top_k: int = 10):
    delta = logits_after - logits_before
    flat = np.argsort(-np.abs(delta), axis=None)[:top_k]
    out = []
    for idx in flat:
        t, v = np.unravel_index(idx, delta.shape)
        out.append(
            {
                "position": int(t),
                "token": tokenizer.decode([int(v)]),
                "logit_before": float(logits_before[t, v]),
                "logit_after": float(logits_after[t, v]),
                "delta": float(delta[t, v]),
            }
        )
    return out


def _top_candidates(logits: np.ndarray, position: int, tokenizer, k: int = 3):
    row = logits[position]
    best = np.argsort(-row)[:k]
    return [{"token": tokenizer.decode([int(v)]), "logit": float(row[v])} for v in best]


def logit_change_report(
    base_model: Seq2SeqLM,
    updated_model: Seq2SeqLM,
    online_example: Example,
    upstream_example: Example,
    top_k: int = 10,
    position: int = 0,
) -> Dict:
    """Logit changes of both examples and the top candidates of ``x_j``."""
    before_i, _ = base_model.teacher_forced_logits([online_example.input], [online_example.target], batch_size=1)[0]
    after_i, _ = updated_model.teacher_forced_logits([online_example.input], [online_example.target], batch_size=1)[0]
    before_j, gold_j = base_model.teacher_forced_logits(
        [upstream_example.input], [upstream_example.target], batch_size=1
    )[0]
    after_j, _ = updated_model.teacher_forced_logits(
        [upstream_example.input], [upstream_example.target], batch_size=1
    )[0]
    tokenizer = base_model.tokenizer
    report = {
        "online_example": {"input": online_example.input, "target": online_example.target},
        "upstream_example": {"input": upstream_example.input, "target": upstream_example.target},
        "online_top_changes": _top_changes(before_i, after_i, tokenizer, top_k),
        "upstream_top_changes": _top_changes(before_j, after_j, tokenizer, top_k),
        "upstream_top_candidates_before": _top_candidates(before_j, position, tokenizer),
        "upstream_top_candidates_after": _top_candidates(after_j, position, tokenizer),
        "upstream_prediction_flipped": bool(
            int(before_j[position].argmax()) != int(after_j[position].argmax())
        ),
        "gold_token": tokenizer.decode([int(gold_j[position])]),
    }
    return report


def plot_logit_changes(report: Dict, path: str) -> str:
    """Bar plot of the online/upstream logit changes (Figure 2a style)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in (
        (axes[0], "online_top_changes", "online learned example <x_i, y_i>"),
        (axes[1], "upstream_top_changes", "upstream example <x_j, y_j>"),
    ):
        entries = report[key]
        labels = [f"{e['token']}@{e['position']}" for e in entries]
        deltas = [e["delta"] for e in entries]
        ax.barh(range(len(entries)), deltas)
        ax.set_yticks(range(len(entries)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("logit change")
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
