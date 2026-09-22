"""Sequential model refinement (Sec. 5.2, Table 3, Table 4, Figure 3).

* ``sequential_refinement``          -- continually fix errors of a stream and
  measure the edit success rate / EM drop ratio at the end (Table 3).
* ``evaluate_single_error_replay``   -- fix every error separately, starting from
  the same base model (Table 4).
* ``continual_forecasting_curves``   -- F1 / precision / recall of the forecasting
  models while the LM is continually refined (Figure 3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from ..config import STREAM_FRACTION, ExperimentConfig
from ..data.types import Dataset, Example
from ..evaluation.metrics import (
    binary_f1_scores,
    edit_success_rate as edit_success_rate_metric,
    em_drop_ratio as em_drop_ratio_metric,
    exact_match,
)
from ..forecasting.cache import build_online_artifacts, label_matrix
from ..models.lm import Seq2SeqLM
from ..models.tuning import fix_single_error, refinement_optimizer
from ..refinement.replay import ReplayPool, make_replay_callback


@dataclass
class RefinementResult:
    method: str
    tuning_mode: str
    edit_success_rate: float
    em_drop_ratio: float          # in percent
    base_em: float
    final_em: float
    n_online: int = 0
    extra: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        def _clean(value):
            # keep the JSON valid: NaN is not a JSON number
            return None if isinstance(value, float) and value != value else value

        return {
            "method": self.method,
            "tuning_mode": self.tuning_mode,
            "Succ.": round(100 * self.edit_success_rate, 1),
            "EM Drop %": round(self.em_drop_ratio, 3),
            "base_em": _clean(round(self.base_em, 4)),
            "final_em": _clean(round(self.final_em, 4)),
            "n_online": self.n_online,
            **self.extra,
        }


def shuffle_stream(examples: Sequence[Example], seed: int = 0) -> List[Example]:
    """The examples in each stream are randomly shuffled (addendum, Figure 3)."""
    stream = list(examples)
    random.Random(seed).shuffle(stream)
    return stream


def sequential_refinement(
    cfg: ExperimentConfig,
    base_model: Seq2SeqLM,
    online_examples: Sequence[Example],
    upstream_eval: Dataset,
    *,
    method: str = "vanilla",
    replay_strategy=None,
    replay_pool: Optional[ReplayPool] = None,
    distill_weight: float = 1.0,
    seed: int = 0,
    verbose: bool = True,
) -> RefinementResult:
    """Continually fix the errors of ``online_examples`` (Sec. 5.2)."""
    base_em = base_model.exact_match(upstream_eval, batch_size=cfg.eval_batch_size)
    snapshot = base_model.snapshot()
    optimizer = refinement_optimizer(base_model, cfg, sequential=True)
    schedule = cfg.replay_schedule()
    stream = shuffle_stream(online_examples, seed=seed)
    successes = 0
    predictions: List[str] = []
    iterator = tqdm(stream, desc=f"sequential refinement ({method})", disable=not verbose)
    for example in iterator:
        if replay_strategy is not None and replay_pool is not None:
            callback = make_replay_callback(replay_strategy, replay_pool, example, schedule["batch_size"])
        else:
            callback = None
        fix_single_error(
            base_model,
            example,
            steps=cfg.steps_single(),
            lr=cfg.lr_sequential(),
            mode=cfg.tuning_mode,
            replay_pool=callback,
            replay_every_n_steps=schedule["every_n_steps"],
            distill_weight=distill_weight,
            optimizer=optimizer,
        )
        pred = base_model.predict([example.input], batch_size=1)[0]
        predictions.append(pred)
        if exact_match([pred], [example.target]) > 0.5:
            successes += 1
    final_em = base_model.exact_match(upstream_eval, batch_size=cfg.eval_batch_size)
    result = RefinementResult(
        method=method,
        tuning_mode=cfg.tuning_mode,
        edit_success_rate=successes / max(1, len(stream)),
        em_drop_ratio=100.0 * em_drop_ratio_metric(final_em, base_em),
        base_em=base_em,
        final_em=final_em,
        n_online=len(stream),
    )
    base_model.restore(snapshot)
    return result


def evaluate_single_error_replay(
    cfg: ExperimentConfig,
    base_model: Seq2SeqLM,
    online_examples: Sequence[Example],
    upstream_eval: Dataset,
    *,
    method: str = "vanilla",
    replay_strategy=None,
    replay_pool: Optional[ReplayPool] = None,
    distill_weight: float = 1.0,
    verbose: bool = True,
) -> RefinementResult:
    """Fix every error separately, always starting from ``f_0`` (Table 4)."""
    base_em = base_model.exact_match(upstream_eval, batch_size=cfg.eval_batch_size)
    snapshot = base_model.snapshot()
    schedule = cfg.replay_schedule()
    drops: List[float] = []
    successes = 0
    iterator = tqdm(online_examples, desc=f"single-error refinement ({method})", disable=not verbose)
    for example in iterator:
        base_model.restore(snapshot)
        callback = None
        if replay_strategy is not None and replay_pool is not None:
            callback = make_replay_callback(replay_strategy, replay_pool, example, schedule["batch_size"])
        fix_single_error(
            base_model,
            example,
            steps=cfg.steps_single(),
            lr=cfg.lr_single(),
            mode=cfg.tuning_mode,
            replay_pool=callback,
            replay_every_n_steps=schedule["every_n_steps"],
            distill_weight=distill_weight,
        )
        em = base_model.exact_match(upstream_eval, batch_size=cfg.eval_batch_size)
        drops.append(100.0 * em_drop_ratio_metric(em, base_em))
        pred = base_model.predict([example.input], batch_size=1)[0]
        successes += int(exact_match([pred], [example.target]) > 0.5)
    base_model.restore(snapshot)
    return RefinementResult(
        method=method,
        tuning_mode=cfg.tuning_mode,
        edit_success_rate=successes / max(1, len(online_examples)),
        em_drop_ratio=float(np.mean(drops)) if drops else float("nan"),
        base_em=base_em,
        final_em=float("nan"),
        n_online=len(online_examples),
    )


def continual_forecasting_curves(
    cfg: ExperimentConfig,
    base_model: Seq2SeqLM,
    forecasters: Dict[str, object],
    online_examples: Sequence[Example],
    upstream_cache,
    *,
    stream_fraction: float = STREAM_FRACTION,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[str, List[Dict[str, float]]]:
    """Figure 3: F1 / precision / recall while the LM is continually refined.

    Per the addendum the forecasted forgetting indicator is computed *once*, at
    the start of the stream (with ``f_0``), and reused at every time step, while
    the ground truth is re-measured with the continually updated LM.
    """
    stream = shuffle_stream(online_examples, seed=seed)
    n_stream = max(1, int(round(len(stream) * stream_fraction)))
    stream = stream[:n_stream]

    artifacts = build_online_artifacts(
        base_model,
        stream,
        upstream_cache,
        steps=cfg.steps_single(),
        lr=cfg.lr_sequential(),
        mode=cfg.tuning_mode,
        collect_labels=True,
        verbose=verbose,
    )
    predictions = {name: [np.asarray(f.predict(art)) for art in artifacts] for name, f in forecasters.items()}

    snapshot = base_model.snapshot()
    optimizer = refinement_optimizer(base_model, cfg, sequential=True)
    eval_mask = upstream_cache.correct_mask
    upstream_inputs = [e.input for e in upstream_cache.examples]
    upstream_targets = [e.target for e in upstream_cache.examples]
    curves: Dict[str, List[Dict[str, float]]] = {name: [] for name in forecasters}
    y_true: List[int] = []
    y_pred: Dict[str, List[int]] = {name: [] for name in forecasters}

    iterator = tqdm(stream, desc="continual refinement", disable=not verbose)
    for step, example in enumerate(iterator, start=1):
        fix_single_error(
            base_model,
            example,
            steps=cfg.steps_single(),
            lr=cfg.lr_sequential(),
            mode=cfg.tuning_mode,
            optimizer=optimizer,
        )
        preds = base_model.predict(upstream_inputs, batch_size=cfg.eval_batch_size)
        labels = np.asarray(
            [0 if exact_match([p], [t]) > 0.5 else 1 for p, t in zip(preds, upstream_targets)], dtype=np.int8
        )
        y_true.extend(labels[eval_mask].tolist())
        for name in forecasters:
            y_pred[name].extend(predictions[name][step - 1][eval_mask].tolist())
            metrics = binary_f1_scores(y_true, y_pred[name])
            metrics["step"] = float(step)
            curves[name].append(metrics)
    base_model.restore(snapshot)
    return curves
