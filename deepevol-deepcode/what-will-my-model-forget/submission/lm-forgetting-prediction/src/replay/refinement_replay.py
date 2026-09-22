"""Replay-based model refinement (Sec. 4.2 / Sec. 5.2 / Appendix D.2).

The paper fixes errors in an instruction-tuned seq2seq LM one online example at a
time and observes that upstream pretraining examples are catastrophically
forgotten.  Two experimental protocols are implemented here:

* **Sequential / continual refinement (Table 3)** -- errors from ``D_R^Test`` are
  fixed one after another, the refined model keeps moving forward
  (``sequential=True``).
* **Separately fixing single errors (Table 4)** -- every error is fixed starting
  from the pristine base PTLM ``f_0`` (``sequential=False``).

both times the same recipe is used (Sec. 4.2, Appendix D.2)::

    For all variants of replay, we sparsely replay a mini-batch of 8 examples
    every 10 training steps on BART0_Large and FLAN-T5_Large, and 4 examples
    every 5 steps on FLAN-T5_3B.

The replayed examples come from ``D_PT_hat`` and are selected with one of the
strategies compared in the paper:

=========================  ====================================================
selector                   description
=========================  ====================================================
``vanilla``                no replay (Vanilla FT)
``random``                 uniformly random upstream examples
``threshold``              frequency-threshold forecaster (Eq. 1)
``logit``                  trainable / fixed logit-change-transfer forecaster
``representation``         black-box representation forecaster (Eq. 4) + prior b_j
``gt``                     ground-truth forgotten examples (upper bound)
=========================  ====================================================

The refinement step itself is delegated to
:class:`src.modeling.refinement.RefinementEngine`; a self-contained fallback
(HF forward/backward with a KL/MSE distillation loss against a frozen copy of the
base PTLM, cf. Buzzega et al. 2020a) is used when the engine does not expose
replay hooks.  Everything the script needs in order to remain cheap at selection
time (``f_0`` top-k logits, ``h(x_j, y_j)`` representations, frequency priors)
is read from ``src.modeling.caches`` -- no PTLM inference on ``D_PT_hat`` is
performed to *forecast*, only to *measure* forgetting after each refinement.

Metrics reported (see :mod:`src.eval.metrics`):

* ``em_drop_percent``   -- EM Drop Ratio on ``D_PT_hat`` (magnitude, Table 3/4)
* ``edit_success_rate`` -- Edit Success Rate on ``D_R^Test`` measured at the end
  of the stream, plus the immediate (per-step) rate which stays above 95%.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

REPLAY_METHODS: Tuple[str, ...] = (
    "vanilla",
    "random",
    "threshold",
    "logit",
    "representation",
    "gt",
)

METHOD_ALIASES: Dict[str, str] = {
    "": "vanilla",
    "none": "vanilla",
    "no_replay": "vanilla",
    "noreplay": "vanilla",
    "vanilla": "vanilla",
    "vanilla_ft": "vanilla",
    "vanillaft": "vanilla",
    "ft": "vanilla",
    "random": "random",
    "rand": "random",
    "random_replay": "random",
    "threshold": "threshold",
    "freq": "threshold",
    "frequency": "threshold",
    "logit": "logit",
    "trainable_logit": "logit",
    "logit_based": "logit",
    "fixed": "logit",
    "fixed_logit": "logit",
    "representation": "representation",
    "repr": "representation",
    "representation_based": "representation",
    "blackbox": "representation",
    "gt": "gt",
    "ground_truth": "gt",
    "gt_forget": "gt",
    "oracle": "gt",
}

# Sec. 4.2 / Appendix D.2 replay schedule.
DEFAULT_REPLAY_BATCH = 8
DEFAULT_REPLAY_EVERY = 10
DEFAULT_REPLAY_BATCH_LARGE = 4
DEFAULT_REPLAY_EVERY_LARGE = 5
LARGE_MODEL_KEYS: Tuple[str, ...] = ("FLAN-T5_3B", "FLAN_T5_3B", "flan-t5-3b", "FLAN-T5-3b")

DEFAULT_DISTILL_MODE = "kl"
DEFAULT_DISTILL_TEMPERATURE = 1.0
DEFAULT_DISTILL_WEIGHT = 1.0
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_INPUT_LEN = 512
DEFAULT_MAX_OUTPUT_LEN = 64
DEFAULT_FILENAME = "replay_summary.json"
DEFAULT_HISTORY_FILENAME = "replay_history.jsonl"

__all__ = [
    "REPLAY_METHODS",
    "ReplaySelector",
    "NoReplaySelector",
    "RandomReplaySelector",
    "ThresholdReplaySelector",
    "ForecastReplaySelector",
    "LogitReplaySelector",
    "RepresentationReplaySelector",
    "GroundTruthReplaySelector",
    "build_selector",
    "normalize_method",
    "replay_schedule_for",
    "lookup_labels",
    "StepRecord",
    "ReplayRefinementResult",
    "SequentialReplayRefinement",
    "run_replay_refinement",
    "manual_replay_refinement",
    "parse_args",
    "main",
]


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #

def normalize_method(name: Optional[str]) -> str:
    """Map a user supplied replay-method string onto one of :data:`REPLAY_METHODS`."""
    if name is None:
        return "vanilla"
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    if key.startswith("replay_"):
        key = key[len("replay_"):]
    if key.startswith("w/"):
        key = key[len("w/"):]
    if key not in METHOD_ALIASES:
        raise ValueError(
            "unknown replay method %r (known: %s)" % (name, ", ".join(REPLAY_METHODS))
        )
    return METHOD_ALIASES[key]


def _model_is_large(model_key: Optional[str]) -> bool:
    if not model_key:
        return False
    key = str(model_key)
    for cand in LARGE_MODEL_KEYS:
        if key.lower().replace("_", "-") == cand.lower().replace("_", "-"):
            return True
    return "3b" in key.lower()


def replay_schedule_for(
    model_key: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None,
    batch_size: Optional[int] = None,
    every_n_steps: Optional[int] = None,
) -> Tuple[int, int]:
    """Return ``(replay_batch_size, every_n_steps)`` for ``model_key``.

    Appendix D.2: 8 examples every 10 steps for BART0_L / FLAN-T5_L and 4
    examples every 5 steps for FLAN-T5_3B.
    """
    cfg_batch = None
    cfg_every = None
    cfg_batch_large = None
    cfg_every_large = None
    if isinstance(config, Mapping):
        replay_cfg = config.get("replay", config)
        if isinstance(replay_cfg, Mapping):
            cfg_batch = replay_cfg.get("batch_size")
            cfg_every = replay_cfg.get("every_n_steps")
            cfg_batch_large = replay_cfg.get("batch_size_large")
            cfg_every_large = replay_cfg.get("every_n_steps_large")
    if _model_is_large(model_key):
        batch = batch_size if batch_size is not None else (cfg_batch_large or DEFAULT_REPLAY_BATCH_LARGE)
        every = every_n_steps if every_n_steps is not None else (cfg_every_large or DEFAULT_REPLAY_EVERY_LARGE)
    else:
        batch = batch_size if batch_size is not None else (cfg_batch or DEFAULT_REPLAY_BATCH)
        every = every_n_steps if every_n_steps is not None else (cfg_every or DEFAULT_REPLAY_EVERY)
    return int(batch), int(every)


def lookup_labels(pair_records: Iterable[Any], aggregate: str = "any") -> Dict[Tuple[int, int], int]:
    """Build ``{(online_index i, upstream_index j): z_ij}`` from pair records.

    ``aggregate``: ``"any"`` (max over duplicates -- "was ever forgotten"),
    ``"last"`` (last occurrence wins) or ``"mean"`` (rounded mean).
    """
    out: Dict[Tuple[int, int], int] = {}
    acc: Dict[Tuple[int, int], List[int]] = {}
    for rec in pair_records:
        i = _rec_get(rec, "i", "online_index", "online", "i_idx")
        j = _rec_get(rec, "j", "upstream_index", "upstream", "j_idx")
        if i is None or j is None:
            continue
        z = _rec_get(rec, "z", "label", "z_ij", "forgotten")
        if z is None:
            continue
        key = (int(i), int(j))
        z = int(bool(z))
        if aggregate == "last":
            out[key] = z
        else:
            acc.setdefault(key, []).append(z)
    if aggregate == "mean":
        for key, vals in acc.items():
            out[key] = int(round(float(sum(vals)) / max(1, len(vals))))
    elif aggregate == "any":
        for key, vals in acc.items():
            out[key] = int(max(vals))
    return out


def _rec_get(record: Any, *names: str, default: Any = None) -> Any:
    """Tolerant attribute / mapping accessor (mirrors ``src.eval.evaluate.rec_get``)."""
    if record is None:
        return default
    if isinstance(record, Mapping):
        for name in names:
            if name in record and record[name] is not None:
                return record[name]
        return default
    for name in names:
        if hasattr(record, name):
            value = getattr(record, name)
            if value is not None:
                return value
    if hasattr(record, "to_dict"):
        try:
            return _rec_get(record.to_dict(), *names, default=default)
        except Exception:  # pragma: no cover - defensive
            return default
    return default


def target_of(example: Mapping[str, Any]) -> str:
    """Target string of an example (``target`` > ``targets`` > ``label``)."""
    if example is None:
        return ""
    for key in ("target", "targets", "label", "output", "answer"):
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, (list, tuple)):
                return str(value[0]) if value else ""
            return str(value)
    return ""


def filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments that ``fn`` actually accepts."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return accepted


def _predict_examples(model: Any, inputs: Sequence[str], batch_size: int = 8, **gen_kwargs: Any) -> List[str]:
    """Generation adapter tolerant to the project's several prediction APIs."""
    if model is None:
        raise ValueError("cannot predict: model is None")
    inputs = [("" if x is None else str(x)) for x in inputs]
    for name in ("predict", "predict_batch"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                return list(fn(inputs, batch_size=batch_size, **gen_kwargs))
            except TypeError:
                try:
                    return list(fn(inputs, **gen_kwargs))
                except TypeError:
                    continue
    for name in ("batch_generate", "generate"):
        fn = getattr(model, name, None)
        if callable(fn):
            try:
                return list(fn(inputs, batch_size=batch_size, **gen_kwargs))
            except TypeError:
                try:
                    return list(fn(inputs, **gen_kwargs))
                except TypeError:
                    continue
    if callable(model):
        return [str(p) for p in model(inputs)]
    raise TypeError("object %r exposes no prediction API" % (type(model).__name__,))


def _get_hf_model(obj: Any) -> Any:
    """Extract the underlying ``transformers`` model from wrappers/engines."""
    if obj is None:
        return None
    if hasattr(obj, "model") and hasattr(getattr(obj, "model"), "forward"):
        return obj.model
    if hasattr(obj, "base_lm"):
        return _get_hf_model(obj.base_lm)
    if hasattr(obj, "forward"):
        return obj
    return None


def _get_tokenizer(obj: Any) -> Any:
    if obj is None:
        return None
    for name in ("tokenizer", "tok"):
        tok = getattr(obj, name, None)
        if tok is not None:
            return tok
    for name in ("base_lm", "engine"):
        inner = getattr(obj, name, None)
        if inner is not None:
            tok = _get_tokenizer(inner)
            if tok is not None:
                return tok
    return None


# --------------------------------------------------------------------------- #
# replay selectors
# --------------------------------------------------------------------------- #

class ReplaySelector:
    """Base class for replay-example selectors.

    Sub-classes implement :meth:`select`.  ``begin_online`` is called by the
    refinement driver before the online example is fixed and ``observe`` after
    the resulting forgetting labels are known, which lets frequency-based
    selectors accumulate statistics as the stream progresses.
    """

    name = "base"

    def __init__(self, n_upstream: Optional[int] = None, seed: int = 42, default_n: int = DEFAULT_REPLAY_BATCH):
        self.n_upstream = int(n_upstream) if n_upstream is not None else None
        self.seed = int(seed)
        self.default_n = int(default_n)
        self._rng = random.Random(self.seed)
        self._online_index: Optional[int] = None
        self._step: int = 0
        self.observations: int = 0

    # -- hooks ------------------------------------------------------------- #
    def begin_online(self, online_index: Optional[int], step: int = 0) -> None:
        self._online_index = None if online_index is None else int(online_index)
        self._step = int(step)

    def observe(self, online_index: Optional[int], labels: Mapping[int, int], step: int = 0) -> None:
        """Feed the ground-truth forgetting labels for one online example."""
        self.observations += int(len(labels or {}))

    def fit(self, *args: Any, **kwargs: Any) -> "ReplaySelector":  # pragma: no cover - optional
        return self

    # -- selection --------------------------------------------------------- #
    def pool(self, exclude: Sequence[int] = ()) -> List[int]:
        if self.n_upstream is None:
            return []
        excluded = {int(x) for x in (exclude or ())}
        return [j for j in range(self.n_upstream) if j not in excluded]

    def select(
        self,
        n: int,
        step: Optional[int] = None,
        online_index: Optional[int] = None,
        exclude: Sequence[int] = (),
    ) -> List[int]:
        raise NotImplementedError

    def ranked(self, *args: Any, **kwargs: Any) -> List[int]:  # pragma: no cover - optional
        return []

    # -- callable adapter used as the engine's ``replay_selector`` ---------- #
    def __call__(self, *args: Any, **kwargs: Any) -> List[int]:
        n: Optional[int] = kwargs.get("n")
        step: Optional[int] = kwargs.get("step")
        online_index: Optional[int] = kwargs.get("online_index")
        n = kwargs.get("batch_size", kwargs.get("replay_batch_size", n))
        if step is None:
            step = kwargs.get("global_step", kwargs.get("t"))
        positional = list(args)
        if positional:
            first = positional.pop(0)
            if isinstance(first, (int, float)) and n is None and 0 < int(first) <= max(2, self.default_n * 4):
                n = int(first)
            elif isinstance(first, (int, float)):
                # a larger integer is interpreted as the current step
                if step is None:
                    step = int(first)
                else:
                    n = n if n is not None else int(first)
        if positional:
            second = positional.pop(0)
            if isinstance(second, (int, float)) and step is None:
                step = int(second)
        if n is None:
            n = self.default_n
        if online_index is None:
            online_index = self._online_index
        if step is None:
            step = self._step
        return self.select(int(n), step=int(step) if step is not None else None, online_index=online_index)

    # -- persistence ------------------------------------------------------- #
    def state(self) -> Dict[str, Any]:
        return {"name": self.name, "n_upstream": self.n_upstream, "seed": self.seed,
                "default_n": self.default_n, "observations": self.observations}

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "%s(n_upstream=%s, default_n=%d)" % (self.name, self.n_upstream, self.default_n)


class NoReplaySelector(ReplaySelector):
    """Vanilla FT -- never replays anything."""

    name = "vanilla"

    def select(self, n: int, step: Optional[int] = None, online_index: Optional[int] = None,
               exclude: Sequence[int] = ()) -> List[int]:
        return []


class RandomReplaySelector(ReplaySelector):
    """Uniformly random upstream examples (``Replay w/ Random``)."""

    name = "random"

    def __init__(self, n_upstream: Optional[int] = None, seed: int = 42,
                 default_n: int = DEFAULT_REPLAY_BATCH, sample_without_replacement: bool = True):
        super().__init__(n_upstream=n_upstream, seed=seed, default_n=default_n)
        self.sample_without_replacement = bool(sample_without_replacement)

    def select(self, n: int, step: Optional[int] = None, online_index: Optional[int] = None,
               exclude: Sequence[int] = ()) -> List[int]:
        pool = self.pool(exclude)
        if not pool or n <= 0:
            return []
        if self.sample_without_replacement:
            n = min(int(n), len(pool))
            return self._rng.sample(pool, n)
        return [self._rng.choice(pool) for _ in range(int(n))]

    def state(self) -> Dict[str, Any]:
        st = super().state()
        st["sample_without_replacement"] = self.sample_without_replacement
        return st


class ThresholdReplaySelector(ReplaySelector):
    """Frequency-threshold forecaster (Sec. 3.1, Eq. 1) used as a selector.

    ``g(x_j) = 1[#past forgettings of x_j >= gamma]``.  For a *mini-batch* of
    ``n`` replay examples the upstream pool is ranked by the number of past
    forgettings and the top-``n`` examples whose count passes ``gamma`` are
    returned.  Before any count information is available (step 0) the selector
    falls back to random sampling, which is documented behaviour of the baseline
    described in the paper (Eq. 1 needs at least one previous observation).
    """

    name = "threshold"

    def __init__(self, n_upstream: Optional[int] = None, seed: int = 42,
                 default_n: int = DEFAULT_REPLAY_BATCH, gamma: float = 1.0,
                 use_frequency: bool = False, n_online: Optional[int] = None):
        super().__init__(n_upstream=n_upstream, seed=seed, default_n=default_n)
        self.gamma = float(gamma)
        self.use_frequency = bool(use_frequency)
        self.n_online = int(n_online) if n_online is not None else None
        self.counts: Dict[int, int] = {}
        self.totals: Dict[int, int] = {}
        self._fallback = RandomReplaySelector(n_upstream=n_upstream, seed=seed + 1,
                                              default_n=default_n)

    def observe(self, online_index: Optional[int], labels: Mapping[int, int], step: int = 0) -> None:
        super().observe(online_index, labels, step=step)
        for j, z in (labels or {}).items():
            j = int(j)
            self.totals[j] = self.totals.get(j, 0) + 1
            if int(bool(z)):
                self.counts[j] = self.counts.get(j, 0) + 1

    def score(self, j: int) -> float:
        if self.use_frequency:
            total = float(self.totals.get(int(j), 0))
            return (float(self.counts.get(int(j), 0)) / total) if total > 0 else 0.0
        return float(self.counts.get(int(j), 0))

    def ranked(self, exclude: Sequence[int] = ()) -> List[int]:
        pool = self.pool(exclude)
        # Descending count, ties broken by a deterministic seeded permutation.
        order = {j: self._rng.random() for j in pool}
        return sorted(pool, key=lambda j: (-self.score(j), order[j]))

    def select(self, n: int, step: Optional[int] = None, online_index: Optional[int] = None,
               exclude: Sequence[int] = ()) -> List[int]:
        if n <= 0:
            return []
        ranked = self.ranked(exclude=exclude)
        passing = [j for j in ranked if self.score(j) >= self.gamma]
        if len(passing) >= n:
            return passing[:n]
        if not passing:
            return self._fallback.select(n, step=step, online_index=online_index, exclude=exclude)
        # fewer than ``n`` examples pass the threshold: pad with the best remaining
        out = list(passing)
        seen = set(out)
        for j in ranked:
            if len(out) >= n:
                break
            if j not in seen:
                out.append(j)
                seen.add(j)
        return out[:n]

    def state(self) -> Dict[str, Any]:
        st = super().state()
        st.update({"gamma": self.gamma, "use_frequency": self.use_frequency,
                   "n_online": self.n_online, "n_counted": len(self.counts)})
        return st


class ForecastReplaySelector(ReplaySelector):
    """Common machinery for forecast-driven selectors (logit / representation).

    ``score_fn`` is a callable ``score_fn(online_index) -> Sequence[float]`` that
    returns one score per upstream example (higher score = more likely to be
    forgotten).  Scores are computed from the *cached* ``f_0`` / ``h`` artifacts,
    so selection never re-runs the PTLM over ``D_PT_hat`` (Sec. 3.2, 3.3).
    """

    def __init__(self, n_upstream: Optional[int] = None, seed: int = 42,
                 default_n: int = DEFAULT_REPLAY_BATCH, score_fn: Optional[Callable[[int], Sequence[float]]] = None,
                 min_score: Optional[float] = None, fill_with_random: bool = True):
        super().__init__(n_upstream=n_upstream, seed=seed, default_n=default_n)
        self.score_fn = score_fn
        self.min_score = min_score
        self.fill_with_random = bool(fill_with_random)
        self._fallback = RandomReplaySelector(n_upstream=n_upstream, seed=seed + 7,
                                              default_n=default_n)
        self._cache: Dict[int, List[float]] = {}
        self._failure_logged = False

    def scores(self, online_index: Optional[int]) -> Optional[List[float]]:
        if self.score_fn is None or online_index is None:
            return None
        key = int(online_index)
        if key in self._cache:
            return self._cache[key]
        try:
            values = list(self.score_fn(key))
        except Exception as exc:  # pragma: no cover - defensive
            if not self._failure_logged:
                logger.warning("[%s] score computation failed (%s); falling back to random replay",
                               self.name, exc)
                self._failure_logged = True
            return None
        self._cache[key] = values
        return values

    def ranked(self, online_index: Optional[int] = None, exclude: Sequence[int] = ()) -> List[int]:
        scores = self.scores(online_index)
        pool = self.pool(exclude)
        order = {j: self._rng.random() for j in pool}
        if scores is None:
            return sorted(pool, key=lambda j: order[j])
        return sorted(pool, key=lambda j: (-float(scores[j]) if j < len(scores) else 0.0, order[j]))

    def select(self, n: int, step: Optional[int] = None, online_index: Optional[int] = None,
               exclude: Sequence[int] = ()) -> List[int]:
        if n <= 0:
            return []
        if online_index is None:
            online_index = self._online_index
        ranked = self.ranked(online_index=online_index, exclude=exclude)
        scores = self.scores(online_index)
        if scores is not None and self.min_score is not None:
            passing = [j for j in ranked if j < len(scores) and float(scores[j]) >= self.min_score]
            if len(passing) >= n:
                return passing[:n]
        if scores is None:
            return self._fallback.select(n, step=step, online_index=online_index, exclude=exclude)
        out = ranked[:n]
        if len(out) < n and self.fill_with_random:
            pad = self._fallback.select(n - len(out), step=step, online_index=online_index,
                                        exclude=list(exclude) + out)
            out = out + pad
        return out[:n]

    def state(self) -> Dict[str, Any]:
        st = super().state()
        st.update({"min_score": self.min_score, "has_score_fn": self.score_fn is not None,
                   "n_scored": len(self._cache)})
        return st


class LogitReplaySelector(ForecastReplaySelector):
    """Trainable / fixed logit-change-transfer forecaster as a selector (Sec. 3.2)."""

    name = "logit"


class RepresentationReplaySelector(ForecastReplaySelector):
    """Black-box representation forecaster (Eq. 4) as a selector (Sec. 3.3)."""

    name = "representation"


class GroundTruthReplaySelector(ReplaySelector):
    """Upper bound -- replays the examples that *are* forgotten (Sec. 4.2 / 5.2).

    Ground-truth labels may either be supplied up-front (``labels`` mapping
    ``(i, j) -> z``) or accumulated online via :meth:`observe`.  Whenever fewer
    than ``n`` positives exist, the mini-batch is padded with random upstream
    examples so that all replay variants replay an equal number of examples
    ("All replay-based methods replay an equal number of examples from D_PT").
    """

    name = "gt"

    def __init__(self, n_upstream: Optional[int] = None, seed: int = 42,
                 default_n: int = DEFAULT_REPLAY_BATCH,
                 labels: Optional[Mapping[Tuple[int, int], int]] = None,
                 positives_only: bool = True):
        super().__init__(n_upstream=n_upstream, seed=seed, default_n=default_n)
        self.labels: Dict[Tuple[int, int], int] = {tuple(k): int(v) for k, v in (labels or {}).items()}  # type: ignore[misc]
        self.positives_only = bool(positives_only)
        self._online_labels: Dict[int, int] = {}
        self._fallback = RandomReplaySelector(n_upstream=n_upstream, seed=seed + 13,
                                              default_n=default_n)

    def observe(self, online_index: Optional[int], labels: Mapping[int, int], step: int = 0) -> None:
        super().observe(online_index, labels, step=step)
        online_index = 0 if online_index is None else int(online_index)
        for j, z in (labels or {}).items():
            z = int(bool(z))
            self.labels[(online_index, int(j))] = z
            self._online_labels[int(j)] = max(self._online_labels.get(int(j), 0), z)

    def positive_indices(self, online_index: Optional[int], exclude: Sequence[int] = ()) -> List[int]:
        excluded = {int(x) for x in (exclude or ())}
        if online_index is not None:
            positives = [j for (i, j), z in self.labels.items() if i == int(online_index) and z == 1]
        else:  # fall back to "ever forgotten"
            positives = [j for j, z in self._online_labels.items() if z == 1]
        out = sorted({j for j in positives if j not in excluded})
        if self.n_upstream is not None:
            out = [j for j in out if 0 <= j < self.n_upstream]
        return out

    def select(self, n: int, step: Optional[int] = None, online_index: Optional[int] = None,
               exclude: Sequence[int] = ()) -> List[int]:
        if n <= 0:
            return []
        if online_index is None:
            online_index = self._online_index
        positives = self.positive_indices(online_index, exclude=exclude)
        self._rng.shuffle(positives)
        out = positives[:n]
        if len(out) < n:
            pad = self._fallback.select(n - len(out), step=step, online_index=online_index,
                                        exclude=list(exclude) + out)
            out = out + pad
        return out[:n]

    def state(self) -> Dict[str, Any]:
        st = super().state()
        st.update({"n_labels": len(self.labels), "positives_only": self.positives_only})
        return st


def build_selector(
    method: str,
    *,
    n_upstream: Optional[int] = None,
    default_n: int = DEFAULT_REPLAY_BATCH,
    seed: int = 42,
    gamma: float = 1.0,
    use_frequency: bool = False,
    score_fn: Optional[Callable[[int], Sequence[float]]] = None,
    min_score: Optional[float] = None,
    labels: Optional[Mapping[Tuple[int, int], int]] = None,
    n_online: Optional[int] = None,
) -> ReplaySelector:
    """Factory returning the selector implementing ``method``."""
    name = normalize_method(method)
    if name == "vanilla":
        return NoReplaySelector(n_upstream=n_upstream, default_n=default_n)
    if name == "random":
        return RandomReplaySelector(n_upstream=n_upstream, seed=seed, default_n=default_n)
    if name == "threshold":
        return ThresholdReplaySelector(n_upstream=n_upstream, seed=seed, default_n=default_n,
                                       gamma=gamma, use_frequency=use_frequency, n_online=n_online)
    if name == "logit":
        return LogitReplaySelector(n_upstream=n_upstream, seed=seed, default_n=default_n,
                                   score_fn=score_fn, min_score=min_score)
    if name == "representation":
        return RepresentationReplaySelector(n_upstream=n_upstream, seed=seed, default_n=default_n,
                                            score_fn=score_fn, min_score=min_score)
    if name == "gt":
        return GroundTruthReplaySelector(n_upstream=n_upstream, seed=seed, default_n=default_n,
                                         labels=labels)
    raise ValueError("unsupported replay method: %r" % (method,))  # pragma: no cover


# --------------------------------------------------------------------------- #
# optional score providers built from the cached forecasting artifacts
# --------------------------------------------------------------------------- #

def build_representation_score_provider(
    forecaster: Any,
    encoder: Any,
    cache: Any,
    online_examples: Sequence[Mapping[str, Any]],
    prior: Any = None,
    upstream_indices: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    use_prior: bool = True,
    device: Any = "cpu",
) -> Callable[[int], List[float]]:
    """Return ``f(i) -> [score_j]`` using Eq. 4 and *cached* upstream features."""
    indices = list(upstream_indices) if upstream_indices is not None else list(
        cache.upstream_indices() if hasattr(cache, "upstream_indices") else range(len(online_examples))
    )
    h_upstream = cache.stack_mean(indices) if hasattr(cache, "stack_mean") else None
    prior_vec = None
    if use_prior:
        try:
            from ..forecasters.representation_based import prior_vector_from

            prior_vec = prior_vector_from(prior, indices, device=device)
        except Exception:  # pragma: no cover - optional
            prior_vec = None

    def provider(online_index: int) -> List[float]:
        import torch

        example = online_examples[int(online_index)]
        if encoder is None:
            raise RuntimeError("representation selector requires an encoder h")
        h_i = encoder.encode([example.get("input", "")], [target_of(example)], mean_pool=True)
        h_i = torch.as_tensor(h_i, dtype=torch.float32).reshape(1, -1).to(device)
        h_j = h_upstream
        if h_j is None:
            raise RuntimeError("representation selector requires a cached upstream h")
        h_j = torch.as_tensor(h_j, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = torch.matmul(h_j, h_i.t()).reshape(-1)
            if prior_vec is not None:
                logits = logits + torch.as_tensor(prior_vec, dtype=torch.float32).to(device).reshape(-1)
            probs = torch.sigmoid(logits)
        return [float(v) for v in probs.detach().cpu().reshape(-1)]

    return provider


def build_logit_score_provider(
    forecaster: Any,
    encoder: Any,
    cache: Any,
    online_deltas: Mapping[int, Tuple[Any, Any]],
    upstream_indices: Optional[Sequence[int]] = None,
    topk: int = 100,
    device: Any = "cpu",
) -> Callable[[int], List[float]]:
    """Return ``f(i) -> [margin_j]`` from cached ``f_0(x_j)`` and ``f_i(x_i)-f_0(x_i)``.

    ``online_deltas`` maps online index ``i`` to ``(f0_topk, fi_topk)`` -- exactly
    the logit streams persisted in the ground-truth pairs (Appendix F,
    Algorithms 1-2), which makes selection free of any extra LM inference.
    """
    indices = list(upstream_indices) if upstream_indices is not None else list(
        cache.upstream_indices() if hasattr(cache, "upstream_indices") else range(0)
    )
    upstream_topk = []
    for j in indices:
        entry = cache.f0_topk(j) if hasattr(cache, "f0_topk") else None
        upstream_topk.append(entry)

    def provider(online_index: int) -> List[float]:
        import torch

        from ..forecasters.logit_based import (
            build_candidate_indices,
            densify_topk,
            forecast_pair_from_cache,
            make_delta_matrix,
        )

        pair = online_deltas.get(int(online_index))
        if pair is None:
            raise KeyError("no cached logit delta for online example %d" % int(online_index))
        f0_i, fi_i = pair
        f0_idx = torch.as_tensor(_topk_field(f0_i, "indices"), dtype=torch.long)
        f0_val = torch.as_tensor(_topk_field(f0_i, "values"), dtype=torch.float32)
        fi_idx = torch.as_tensor(_topk_field(fi_i, "indices"), dtype=torch.long)
        fi_val = torch.as_tensor(_topk_field(fi_i, "values"), dtype=torch.float32)

        scores: List[float] = []
        for j, entry in zip(indices, upstream_topk):
            try:
                cj_idx = torch.as_tensor(_topk_field(entry, "indices"), dtype=torch.long)
                cj_val = torch.as_tensor(_topk_field(entry, "values"), dtype=torch.float32)
                candidates = build_candidate_indices(cj_idx, max_candidates=None, vocab_size=None)
                delta = make_delta_matrix(f0_idx, f0_val, fi_idx, fi_val, candidates)
                z_hat, _cands, pred = forecast_pair_from_cache(
                    forecaster,
                    h_j=None, h_i=None,
                    f0_xi_topk=f0_i, fi_xi_topk=fi_i, f0_xj_topk=entry,
                    return_logits=False,  # only z_hat needed below (kept for API symmetry)
                )
                scores.append(float(z_hat) if not isinstance(z_hat, (list, tuple)) else float(z_hat[0]))
            except Exception:
                scores.append(0.0)
        return scores

    return provider


def _topk_field(entry: Any, field: str) -> Any:
    if entry is None:
        return []
    if isinstance(entry, Mapping):
        return entry.get(field, [])
    return getattr(entry, field, [])


def build_online_delta_lookup(records: Iterable[Any]) -> Dict[int, Tuple[Any, Any]]:
    """Extract ``{i: (f0(x_i) top-k, f_i(x_i) top-k)}`` from ground-truth records."""
    out: Dict[int, Tuple[Any, Any]] = {}
    for rec in records or []:
        i = _rec_get(rec, "i", "online_index", "index")
        f0 = _rec_get(rec, "f0_token_logits", "f0_i_token_logits", "f0_topk")
        fi = _rec_get(rec, "fi_token_logits", "fi_i_token_logits", "fi_topk")
        if i is None or f0 is None or fi is None:
            continue
        out[int(i)] = (f0, fi)
    return out


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #

@dataclass
class StepRecord:
    """Book-keeping for one fixed online example (one row of a stream)."""

    step: int
    online_index: int
    online_id: str = ""
    f0_correct: Optional[bool] = None
    edit_success: Optional[bool] = None
    loss: Optional[float] = None
    n_replay: int = 0
    replayed: List[int] = field(default_factory=list)
    n_upstream_evaluated: int = 0
    n_forgotten: int = 0
    forgetting_rate: float = 0.0
    edit_success_rate_at_step: float = 0.0
    em_drop_percent_at_step: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplayRefinementResult:
    """Aggregated outcome of a replay-refinement experiment (Tables 3 / 4)."""

    method: str
    model_key: Optional[str] = None
    tuning: Optional[str] = None
    sequential: bool = True
    replay_batch_size: int = DEFAULT_REPLAY_BATCH
    replay_every_n_steps: int = DEFAULT_REPLAY_EVERY
    steps: List[StepRecord] = field(default_factory=list)
    pair_records: List[Dict[str, Any]] = field(default_factory=list)
    online_records: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "model_key": self.model_key,
            "tuning": self.tuning,
            "sequential": bool(self.sequential),
            "replay_batch_size": int(self.replay_batch_size),
            "replay_every_n_steps": int(self.replay_every_n_steps),
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "online_records": self.online_records,
            "pair_records": self.pair_records,
            "meta": self.meta,
        }

    def save(self, out_dir: str, filename: str = DEFAULT_FILENAME,
             history_filename: str = DEFAULT_HISTORY_FILENAME) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        summary_path = os.path.join(out_dir, filename)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=_json_default)
        history_path = os.path.join(out_dir, history_filename)
        with open(history_path, "w", encoding="utf-8") as fh:
            for row in self.steps:
                fh.write(json.dumps(row.to_dict(), default=_json_default) + "\n")
        pairs_path = os.path.join(out_dir, "replay_pairs.jsonl")
        with open(pairs_path, "w", encoding="utf-8") as fh:
            for row in self.pair_records:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        online_path = os.path.join(out_dir, "replay_online.jsonl")
        with open(online_path, "w", encoding="utf-8") as fh:
            for row in self.online_records:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        return {"summary": summary_path, "history": history_path,
                "pairs": pairs_path, "online": online_path}


def _json_default(obj: Any) -> Any:  # pragma: no cover - defensive
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


# --------------------------------------------------------------------------- #
# manual (engine-free) replay refinement
# --------------------------------------------------------------------------- #

def _tokenize_examples(tokenizer: Any, examples: Sequence[Mapping[str, Any]],
                       max_input_len: int, max_output_len: int, device: Any) -> Tuple[Any, Any]:
    inputs = [str(e.get("input", "")) for e in examples]
    targets = [target_of(e) for e in examples]
    enc = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_input_len)
    try:
        dec = tokenizer(text_target=targets, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_output_len)
    except TypeError:  # older transformers
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(targets, return_tensors="pt", padding=True, truncation=True,
                            max_length=max_output_len)
    enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    labels = dec["input_ids"].to(device)
    return enc, labels


def _forward_loss_and_logits(model: Any, tokenizer: Any, examples: Sequence[Mapping[str, Any]],
                             max_input_len: int, max_output_len: int, device: Any,
                             no_grad: bool = False) -> Tuple[Any, Any, Any]:
    import torch

    enc, labels = _tokenize_examples(tokenizer, examples, max_input_len, max_output_len, device)
    context = torch.no_grad() if no_grad else _nullcontext()
    with context:
        out = model(**enc, labels=labels)
    return out.loss, out.logits, labels


class _nullcontext:  # minimal local null context (avoids importing contextlib for typing noise)
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return False


def manual_replay_refinement(
    base_lm: Any,
    example: Mapping[str, Any],
    *,
    selector: ReplaySelector,
    upstream_examples: Sequence[Mapping[str, Any]],
    steps: int = 30,
    lr: float = 1e-5,
    replay_batch_size: int = DEFAULT_REPLAY_BATCH,
    replay_every_n_steps: int = DEFAULT_REPLAY_EVERY,
    distill_mode: str = DEFAULT_DISTILL_MODE,
    distill_temperature: float = DEFAULT_DISTILL_TEMPERATURE,
    distill_weight: float = DEFAULT_DISTILL_WEIGHT,
    teacher: Any = None,
    device: Any = "cuda",
    max_input_len: int = DEFAULT_MAX_INPUT_LEN,
    max_output_len: int = DEFAULT_MAX_OUTPUT_LEN,
    grad_clip: float = DEFAULT_GRAD_CLIP,
    online_index: Optional[int] = None,
    step_offset: int = 0,
) -> Tuple[Any, Dict[str, Any]]:
    """Fine-tune ``base_lm`` on one online example with scheduled replay.

    This is the paper's refinement objective: cross-entropy on the online example
    plus a distillation loss against the frozen base PTLM for the replayed
    examples (Sec. 4.2, Buzzega et al. 2020a).  It is only used when the
    :class:`~src.modeling.refinement.RefinementEngine` does not expose replay
    hooks; the engine's native implementation is preferred.
    """
    import torch
    from torch.nn.utils import clip_grad_norm_

    model = _get_hf_model(base_lm)
    tokenizer = _get_tokenizer(base_lm)
    if model is None or tokenizer is None:
        raise RuntimeError("manual_replay_refinement requires an HF model + tokenizer")

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:  # head-only engines may keep every parameter frozen but the head
        params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(lr), betas=(0.9, 0.999), weight_decay=0.0)

    teacher_model = _get_hf_model(teacher) if teacher is not None else None
    if teacher_model is not None:
        teacher_model.eval()

    replay_events: List[List[int]] = []
    losses: List[float] = []
    for step in range(int(steps)):
        loss, _logits, _labels = _forward_loss_and_logits(
            model, tokenizer, [example], max_input_len, max_output_len, device)
        replayed: List[int] = []
        if replay_every_n_steps and replay_batch_size and (step % int(replay_every_n_steps) == 0):
            replayed = selector.select(int(replay_batch_size), step=step + int(step_offset),
                                       online_index=online_index)
            if replayed:
                replay_examples = [upstream_examples[j] for j in replayed if 0 <= j < len(upstream_examples)]
                if replay_examples:
                    if teacher_model is not None:
                        try:
                            _t_loss, t_logits, t_labels = _forward_loss_and_logits(
                                teacher_model, tokenizer, replay_examples,
                                max_input_len, max_output_len, device, no_grad=True)
                        except Exception:  # pragma: no cover - defensive
                            t_logits, t_labels = None, None
                    else:
                        t_logits, t_labels = None, None
                    s_loss, s_logits, s_labels = _forward_loss_and_logits(
                        model, tokenizer, replay_examples, max_input_len, max_output_len, device)
                    if t_logits is not None and t_logits.shape == s_logits.shape:
                        try:
                            from ..forecasters.losses import distillation_loss

                            pad_id = getattr(tokenizer, "pad_token_id", None)
                            mask = None
                            if pad_id is not None:
                                mask = (s_labels != int(pad_id)).float()
                            d_loss = distillation_loss(s_logits, t_logits,
                                                       temperature=float(distill_temperature),
                                                       mode=distill_mode, mask=mask)
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.warning("distillation unavailable (%s); replay falls back to CE", exc)
                            d_loss = s_loss
                        loss = loss + float(distill_weight) * d_loss
                    else:  # no teacher: plain supervised replay
                        loss = loss + float(distill_weight) * s_loss
                replay_events.append(list(replayed))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            clip_grad_norm_(params, float(grad_clip))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    info = {"loss": (sum(losses) / len(losses)) if losses else None,
            "final_loss": losses[-1] if losses else None,
            "replay_events": replay_events,
            "n_replay": int(sum(len(e) for e in replay_events))}
    return base_lm, info


# --------------------------------------------------------------------------- #
# main driver
# --------------------------------------------------------------------------- #

class SequentialReplayRefinement:
    """Sequentially fix errors with optional replay of ``D_PT_hat`` examples.

    Parameters
    ----------
    engine:
        A :class:`~src.modeling.refinement.RefinementEngine` (or any object
        exposing ``refine`` / ``refine_on_example`` / ``fit_on_example``).
    online_examples, upstream_examples:
        ``D_R^Test`` errors and the ``D_PT_hat`` upstream pool.
    method:
        one of :data:`REPLAY_METHODS`.
    selector:
        pre-built selector; by default :func:`build_selector` is used.
    sequential:
        ``True`` -> Table 3 stream (the model keeps drifting), ``False`` ->
        Table 4 (every error fixed starting from the pristine ``f_0``).
    eval_upstream_indices:
        upstream indices used to *measure* forgetting (default: all).  Measuring
        is the only operation that runs LM inference over ``D_PT_hat``.
    """

    def __init__(
        self,
        engine: Any,
        online_examples: Sequence[Mapping[str, Any]],
        upstream_examples: Sequence[Mapping[str, Any]],
        *,
        method: str = "vanilla",
        selector: Optional[ReplaySelector] = None,
        model_key: Optional[str] = None,
        tuning: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        sequential: bool = True,
        replay_batch_size: Optional[int] = None,
        replay_every_n_steps: Optional[int] = None,
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        eval_every: int = 1,
        eval_upstream_indices: Optional[Sequence[int]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 42,
        device: Any = "cuda",
        distill_mode: str = DEFAULT_DISTILL_MODE,
        distill_temperature: float = DEFAULT_DISTILL_TEMPERATURE,
        distill_weight: float = DEFAULT_DISTILL_WEIGHT,
        teacher: Any = None,
        shuffle: bool = False,
        max_online: Optional[int] = None,
        predict_batch_size: int = DEFAULT_BATCH_SIZE,
        progress: bool = True,
        verbose: bool = False,
        refine_hook: Optional[Callable[..., Any]] = None,
        predict_hook: Optional[Callable[..., Any]] = None,
        use_precomputed_labels: Optional[Mapping[Tuple[int, int], int]] = None,
        base_em_percent: Optional[float] = None,
    ):
        self.engine = engine
        self.online_examples = list(online_examples)
        self.upstream_examples = list(upstream_examples)
        self.method = normalize_method(method)
        self.model_key = model_key
        self.tuning = tuning
        self.config = config or {}
        self.sequential = bool(sequential)
        self.seed = int(seed)
        self.device = device
        self.batch_size = int(batch_size)
        self.predict_batch_size = int(predict_batch_size)
        self.eval_every = max(1, int(eval_every))
        self.max_online = None if max_online is None else int(max_online)
        self.shuffle = bool(shuffle)
        self.progress = bool(progress)
        self.verbose = bool(verbose)
        self.distill_mode = distill_mode
        self.distill_temperature = float(distill_temperature)
        self.distill_weight = float(distill_weight)
        self.teacher = teacher
        self.refine_hook = refine_hook
        self.predict_hook = predict_hook
        self.use_precomputed_labels = dict(use_precomputed_labels or {})
        self.base_em_percent = base_em_percent
        self._rng = random.Random(self.seed)

        # replay schedule (Sec. 4.2 / Appendix D.2)
        self.replay_batch_size, self.replay_every_n_steps = replay_schedule_for(
            model_key, config, replay_batch_size, replay_every_n_steps)

        self.steps = steps
        self.lr = lr
        self.upstream_indices: List[int] = (
            list(eval_upstream_indices) if eval_upstream_indices is not None
            else list(range(len(self.upstream_examples)))
        )
        self.selector: ReplaySelector = selector or build_selector(
            self.method,
            n_upstream=len(self.upstream_examples),
            default_n=self.replay_batch_size,
            seed=self.seed,
            n_online=len(self.online_examples),
        )
        self.selector.default_n = int(self.replay_batch_size)

    # -- model access ------------------------------------------------------ #
    def _model_ref(self) -> Any:
        for attr in ("base_lm", "model_ref"):
            obj = getattr(self.engine, attr, None)
            if obj is not None:
                return obj
        return self.engine

    def _resolve_steps(self) -> int:
        if self.steps is not None:
            return int(self.steps)
        for name in ("steps", "n_steps", "num_steps"):
            value = getattr(self.engine, name, None)
            if isinstance(value, (int, float)) and value > 0:
                return int(value)
        try:
            from ..modeling.refinement import resolve_steps as _resolve

            return int(_resolve(getattr(self.engine, "mode", "full_ft"), None, self.config))
        except Exception:  # pragma: no cover - defensive
            return 30

    def _resolve_lr(self) -> float:
        if self.lr is not None:
            return float(self.lr)
        for name in ("lr", "learning_rate"):
            value = getattr(self.engine, name, None)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
        try:
            from ..modeling.refinement import resolve_lr as _resolve_lr

            return float(_resolve_lr(self.model_key or "BART0_L",
                                     getattr(self.engine, "mode", "full_ft"),
                                     sequential=self.sequential, lr=None, config=self.config))
        except Exception:  # pragma: no cover - defensive
            return 1e-5

    # -- refinement -------------------------------------------------------- #
    def _selector_callable(self, online_index: int, step: int) -> Callable[..., List[int]]:
        selector = self.selector

        def _call(*args: Any, **kwargs: Any) -> List[int]:
            kwargs.setdefault("online_index", online_index)
            kwargs.setdefault("step", step)
            return selector(*args, **kwargs)

        return _call

    def refine_one(self, online_index: int, step: int) -> Dict[str, Any]:
        """Run ``K`` gradient steps on online example ``online_index``."""
        example = self.online_examples[online_index]
        if self.refine_hook is not None:
            out = self.refine_hook(example, online_index=online_index, step=step,
                                   selector=self._selector_callable(online_index, step))
            return out if isinstance(out, dict) else {"result": out}

        if not self.sequential:
            reset = getattr(self.engine, "reset", None)
            if callable(reset):
                try:
                    reset()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("engine.reset() failed: %s", exc)

        steps = self._resolve_steps()
        lr = self._resolve_lr()
        kwargs = {
            "replay_selector": self._selector_callable(online_index, step),
            "replay_batch_size": self.replay_batch_size,
            "replay_every_n_steps": self.replay_every_n_steps,
            "replay_examples": None,
            "sequential": self.sequential,
            "steps": steps,
            "lr": lr,
            "online_index": online_index,
            "distill_mode": self.distill_mode,
            "distill_temperature": self.distill_temperature,
            "distill_weight": self.distill_weight,
            "teacher": self.teacher,
        }
        for name in ("refine", "refine_on_example", "fit_on_example", "update_online_example", "__call__"):
            fn = getattr(self.engine, name, None)
            if not callable(fn):
                continue
            accepted = filter_kwargs(fn, kwargs)
            if not any(k in accepted for k in ("replay_selector", "replay_examples", "replay_batch_size")):
                continue
            try:
                result = fn(example, **accepted)
            except TypeError:
                try:
                    result = fn(example, online_index=online_index, **accepted)
                except TypeError as exc:
                    logger.warning("engine.%s() rejected replay kwargs (%s)", name, exc)
                    continue
            info = result[1] if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict) else {}
            return info

        # ---- fallback: engine without replay support ------------------- #
        logger.warning("RefinementEngine exposes no replay hooks; using the local "
                       "replay-refinement implementation")
        _lm, info = manual_replay_refinement(
            self._model_ref(), example,
            selector=self.selector,
            upstream_examples=self.upstream_examples,
            steps=steps, lr=lr,
            replay_batch_size=self.replay_batch_size,
            replay_every_n_steps=self.replay_every_n_steps,
            distill_mode=self.distill_mode,
            distill_temperature=self.distill_temperature,
            distill_weight=self.distill_weight,
            teacher=self.teacher,
            device=self.device,
            online_index=online_index,
            step_offset=step,
        )
        return info

    # -- evaluation -------------------------------------------------------- #
    def _predict(self, inputs: Sequence[str]) -> List[str]:
        if self.predict_hook is not None:
            return list(self.predict_hook(inputs))
        model = self._model_ref()
        return _predict_examples(model, inputs, batch_size=self.predict_batch_size)

    def predict_upstream(self, indices: Optional[Sequence[int]] = None) -> Tuple[List[str], List[int]]:
        idx = list(self.upstream_indices if indices is None else indices)
        inputs = [str(self.upstream_examples[j].get("input", "")) for j in idx]
        return self._predict(inputs), idx

    def upstream_labels(self, indices: Optional[Sequence[int]] = None) -> Dict[int, int]:
        """Ground-truth forgetting labels ``z_ij`` for the current model."""
        idx = list(self.upstream_indices if indices is None else indices)
        preds, idx = self.predict_upstream(idx)
        from ..data.em_eval import is_correct

        labels: Dict[int, int] = {}
        for j, pred in zip(idx, preds):
            example = self.upstream_examples[j]
            refs = example.get("references") or example.get("target") or example.get("targets")
            labels[int(j)] = 0 if is_correct(pred, refs) else 1
        return labels

    # -- main loop --------------------------------------------------------- #
    def run(self) -> ReplayRefinementResult:
        order = list(range(len(self.online_examples)))
        if self.max_online is not None:
            order = order[: self.max_online]
        if self.shuffle:
            self._rng.shuffle(order)

        result = ReplayRefinementResult(
            method=self.method,
            model_key=self.model_key,
            tuning=self.tuning,
            sequential=self.sequential,
            replay_batch_size=self.replay_batch_size,
            replay_every_n_steps=self.replay_every_n_steps,
        )
        immediate_successes = 0
        immediate_total = 0
        first_success_rate: Optional[float] = None
        last_labels: Dict[int, int] = {}
        union_forgotten: Dict[int, int] = {}
        eval_upstream_indices = list(self.upstream_indices)
        iterator = order
        if self.progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(order, desc="replay:%s" % self.method)
            except Exception:  # pragma: no cover - tqdm optional
                pass

        for t, online_index in enumerate(iterator):
            example = self.online_examples[online_index]
            self.selector.begin_online(online_index, step=t)
            t0 = time.time()
            info = self.refine_one(online_index, t) or {}

            # -- immediate edit success (paper: > 95 % right after fixing) -- #
            pred_i = self._predict([str(example.get("input", ""))])
            from ..data.em_eval import is_correct

            refs_i = example.get("references") or example.get("target") or example.get("targets")
            edit_ok = bool(is_correct(pred_i[0] if pred_i else "", refs_i))
            immediate_total += 1
            immediate_successes += int(edit_ok)
            if first_success_rate is None:
                first_success_rate = float(immediate_successes) / max(1, immediate_total)

            # -- measure forgetting on D_PT_hat ---------------------------- #
            labels: Dict[int, int] = {}
            if self.use_precomputed_labels:
                labels = {int(j): int(z) for (i, j), z in self.use_precomputed_labels.items()
                          if int(i) == int(online_index)}
                labels = {j: labels.get(j, 0) for j in eval_upstream_indices}
            elif (t % self.eval_every == 0) or (t == len(order) - 1):
                try:
                    labels = self.upstream_labels(eval_upstream_indices)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("forgetting measurement failed at step %d: %s", t, exc)
                    labels = {}
            n_eval = len(labels)
            n_forgotten = int(sum(int(bool(v)) for v in labels.values()))
            forgetting_rate = (float(n_forgotten) / n_eval) if n_eval else 0.0
            if labels:
                last_labels = labels
                for j, z in labels.items():
                    union_forgotten[int(j)] = max(union_forgotten.get(int(j), 0), int(bool(z)))

            self.selector.observe(online_index, labels, step=t)
            self.selector.fit(online_index=online_index, labels=labels, step=t)

            step_record = StepRecord(
                step=t,
                online_index=int(online_index),
                online_id=str(example.get("id", "")),
                f0_correct=None,
                edit_success=edit_ok,
                loss=float(info["loss"]) if isinstance(info.get("loss"), (int, float)) else None,
                n_replay=int(info.get("n_replay", 0) or (info.get("replayed_count") or 0)),
                replayed=list(info.get("replayed", []) or []),
                n_upstream_evaluated=n_eval,
                n_forgotten=n_forgotten,
                forgetting_rate=forgetting_rate,
                edit_success_rate_at_step=float(immediate_successes) / max(1, immediate_total),
                em_drop_percent_at_step=forgetting_rate * 100.0,
                seconds=time.time() - t0,
            )
            result.steps.append(step_record)
            result.online_records.append({
                "i": int(online_index),
                "step": t,
                "id": step_record.online_id,
                "edit_success": edit_ok,
                "loss": step_record.loss,
                "n_replay": step_record.n_replay,
            })
            for j, z in labels.items():
                result.pair_records.append({
                    "i": int(online_index),
                    "j": int(j),
                    "z": int(z),
                    "step": t,
                    "i_id": str(example.get("id", "")),
                    "j_id": str(self.upstream_examples[int(j)].get("id", "")),
                    "i_task": str(example.get("task", "")),
                    "j_task": str(self.upstream_examples[int(j)].get("task", "")),
                })
            if self.verbose:
                logger.info("[%s] step %d/%d online=%s edit=%s replay=%d forgotten=%d (%.2f%%)",
                            self.method, t + 1, len(order), online_index, edit_ok,
                            step_record.n_replay, n_forgotten, forgetting_rate * 100.0)

        # -- end-of-stream metrics ---------------------------------------- #
        final_edit_success: Optional[float] = None
        if self.online_examples:
            try:
                preds = self._predict([str(e.get("input", "")) for e in self.online_examples])
                ok = 0
                for example, pred in zip(self.online_examples, preds):
                    refs = example.get("references") or example.get("target") or example.get("targets")
                    ok += int(is_correct(pred, refs))
                final_edit_success = 100.0 * ok / max(1, len(self.online_examples))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("end-of-stream edit-success evaluation failed: %s", exc)

        n_upstream_eval = len(eval_upstream_indices)
        forgetting_final = (sum(last_labels.values()) / float(len(last_labels))) if last_labels else 0.0
        forgetting_union = (len(union_forgotten) / float(n_upstream_eval)) if n_upstream_eval else 0.0
        base_em = self.base_em_percent
        em_after_final = None if base_em is None else base_em * (1.0 - forgetting_final)
        em_drop_final = forgetting_final * 100.0
        em_drop_union = forgetting_union * 100.0

        summary: Dict[str, Any] = {
            "method": self.method,
            "model_key": self.model_key,
            "tuning": self.tuning,
            "sequential": self.sequential,
            "n_steps": len(result.steps),
            "replay_batch_size": self.replay_batch_size,
            "replay_every_n_steps": self.replay_every_n_steps,
            "n_replay_events": sum(1 for s in result.steps if s.n_replay > 0),
            "edit_success_rate_immediate": 100.0 * immediate_successes / max(1, immediate_total),
            "edit_success_rate": final_edit_success if final_edit_success is not None
                                 else 100.0 * immediate_successes / max(1, immediate_total),
            "edit_success_rate_final": final_edit_success,
            "em_drop": -em_drop_final,
            "em_drop_percent": em_drop_final,
            "em_drop_union_percent": em_drop_union,
            "forgetting_rate_final": forgetting_final,
            "forgetting_rate_union": forgetting_union,
            "n_upstream_evaluated": n_upstream_eval,
            "base_em_percent": base_em,
            "em_after_percent": em_after_final,
            "mean_loss": (sum(s.loss for s in result.steps if s.loss is not None)
                          / max(1, sum(1 for s in result.steps if s.loss is not None)))
                         if any(s.loss is not None for s in result.steps) else None,
        }
        selector_state = self.selector.state() if hasattr(self.selector, "state") else {"name": self.method}
        result.meta = {
            "selector": selector_state,
            "seed": self.seed,
            "eval_every": self.eval_every,
            "steps_per_online_example": self._resolve_steps(),
            "lr": self._resolve_lr(),
            "distill_mode": self.distill_mode,
            "distill_temperature": self.distill_temperature,
            "distill_weight": self.distill_weight,
            "schedule_source": "Sec. 4.2 / Appendix D.2",
        }
        result.summary = summary
        return result

    # convenience alias
    __call__ = run


def run_replay_refinement(
    engine: Any,
    online_examples: Sequence[Mapping[str, Any]],
    upstream_examples: Sequence[Mapping[str, Any]],
    *,
    method: str = "vanilla",
    selector: Optional[ReplaySelector] = None,
    **kwargs: Any,
) -> ReplayRefinementResult:
    """One-shot helper mirroring :class:`SequentialReplayRefinement`."""
    runner = SequentialReplayRefinement(
        engine, online_examples, upstream_examples, method=method, selector=selector, **kwargs)
    return runner.run()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_config(path: Optional[str] = None) -> Dict[str, Any]:
    candidates = [path] if path else []
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidates += [os.path.join(root, "config", "config.yaml"), "config/config.yaml"]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                import yaml

                with open(cand, "r", encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or {}
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("could not read config %s: %s", cand, exc)
    return {}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path or not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def artifact_root(config: Mapping[str, Any], model_key: str, tuning: Optional[str] = None) -> str:
    out = config.get("output_dir", "artifacts") if isinstance(config, Mapping) else "artifacts"
    root = os.path.join(str(out), str(model_key))
    return os.path.join(root, str(tuning)) if tuning else root


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay-based refinement of a PTLM on D_R^Test (Tables 3 / 4).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-key", default="BART0_L")
    parser.add_argument("--tuning", default="full_ft", choices=["full_ft", "lora", "head", "none"])
    parser.add_argument("--method", default="vanilla", choices=list(REPLAY_METHODS))
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None, help="gradient steps per online example")
    parser.add_argument("--replay-batch-size", type=int, default=None)
    parser.add_argument("--replay-every-n-steps", type=int, default=None)
    parser.add_argument("--distill-mode", default=DEFAULT_DISTILL_MODE, choices=["kl", "mse", "both"])
    parser.add_argument("--distill-temperature", type=float, default=DEFAULT_DISTILL_TEMPERATURE)
    parser.add_argument("--distill-weight", type=float, default=DEFAULT_DISTILL_WEIGHT)
    parser.add_argument("--online-file", default=None, help="D_R^Test jsonl")
    parser.add_argument("--upstream-file", default=None, help="D_PT_hat jsonl")
    parser.add_argument("--gt-dir", default=None, help="directory with pairs_train.jsonl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--no-prior", action="store_true")
    parser.add_argument("--fixed", action="store_true", help="fixed-logit selector")
    parser.add_argument("--sequential", dest="sequential", action="store_true", default=True)
    parser.add_argument("--single-error", dest="sequential", action="store_false",
                        help="Table 4 protocol: reset to f_0 for every online example")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-online", type=int, default=None)
    parser.add_argument("--max-upstream", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="run an offline smoke test")
    return parser.parse_args(argv)


def _self_test() -> int:
    """Offline check of the selectors and the stream bookkeeping (no models)."""
    n_up = 64
    upstream = [{"input": "u%d" % j, "target": "t%d" % j} for j in range(n_up)]
    online = [{"input": "o%d" % i, "target": "a%d" % i} for i in range(12)]

    # --- selectors -------------------------------------------------------- #
    rnd = build_selector("random", n_upstream=n_up, default_n=8, seed=1)
    picks = rnd.select(8)
    assert len(picks) == 8 and len(set(picks)) == 8, picks

    thr = build_selector("threshold", n_upstream=n_up, default_n=8, seed=1, gamma=1.0)
    first = thr.select(8)
    assert len(first) == 8, "threshold falls back to random before any observation"
    thr.observe(0, {0: 1, 1: 1, 2: 0}, step=0)
    picks = thr.select(3)
    assert 0 in picks and 1 in picks, "counted positives must be selected first"

    gt = build_selector("gt", n_upstream=n_up, default_n=8, seed=1,
                        labels={(0, 5): 1, (0, 6): 1, (0, 7): 0})
    picks = gt.select(8, online_index=0)
    assert 5 in picks and 6 in picks and len(picks) == 8

    scores = {3: 1.0, 4: 0.9}
    rep = build_selector("representation", n_upstream=n_up, default_n=2, seed=1,
                         score_fn=lambda i: [scores.get(j, 0.0) for j in range(n_up)])
    assert rep.select(2, online_index=0) == [3, 4]

    vanilla = build_selector("vanilla", n_upstream=n_up, default_n=8)
    assert vanilla.select(8) == []

    # schedule (Appendix D.2)
    assert replay_schedule_for("BART0_L") == (8, 10)
    assert replay_schedule_for("FLAN-T5_L") == (8, 10)
    assert replay_schedule_for("FLAN-T5_3B") == (4, 5)

    # --- stream bookkeeping with injected hooks --------------------------- #
    def refine_hook(example, online_index=None, step=None, selector=None, **kwargs):
        selector(8)
        return {"loss": 0.5, "n_replay": 8}

    success_online = set()
    def predict_hook(inputs):
        outs = []
        for text in inputs:
            if text.startswith("o"):
                outs.append(text.replace("o", "a"))          # edit succeeded
            else:
                idx = int(text[1:])
                outs.append("t%d" % idx if idx not in success_online else "wrong")
        return outs

    runner = SequentialReplayRefinement(
        engine=None, online_examples=online, upstream_examples=upstream,
        method="random", sequential=False, replay_batch_size=4, replay_every_n_steps=5,
        model_key="BART0_L", eval_every=1, progress=False, batch_size=4,
        refine_hook=refine_hook, predict_hook=predict_hook, base_em_percent=100.0,
    )
    result = runner.run()
    assert result.summary["n_steps"] == len(online)
    assert result.summary["edit_success_rate_immediate"] > 95.0, result.summary
    assert result.summary["em_drop_percent"] == 0.0  # no upstream example forgotten
    assert len(result.pair_records) == len(online) * len(upstream)

    # forgetting path -- upstream examples 0..9 are forgotten after step 1
    def predict_hook2(inputs):
        outs = []
        for text in inputs:
            if text.startswith("o"):
                outs.append(text.replace("o", "a"))
            else:
                idx = int(text[1:])
                outs.append("t%d" % idx if idx >= 10 else "wrong")
        return outs

    runner2 = SequentialReplayRefinement(
        engine=None, online_examples=online, upstream_examples=upstream,
        method="gt", sequential=False, replay_batch_size=4, replay_every_n_steps=5,
        model_key="BART0_L", eval_every=1, progress=False, batch_size=4,
        refine_hook=refine_hook, predict_hook=predict_hook2, base_em_percent=100.0,
    )
    result2 = runner2.run()
    assert abs(result2.summary["em_drop_percent"] - 100.0 * 10 / n_up) < 1e-6, result2.summary
    assert result2.summary["forgetting_rate_union"] > 0

    print("self-test OK: selectors, schedule, stream bookkeeping, EM Drop")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return _self_test()

    config = _load_config(args.config)
    root = artifact_root(config, args.model_key, args.tuning if args.tuning != "none" else None)
    dataset_dir = os.path.join(str(config.get("output_dir", "artifacts")), args.model_key)

    online_path = args.online_file or os.path.join(dataset_dir, "d_r_test.jsonl")
    upstream_path = args.upstream_file or os.path.join(dataset_dir, "d_pt_hat.jsonl")
    online_examples = _load_jsonl(online_path)
    upstream_examples = _load_jsonl(upstream_path)
    if not online_examples or not upstream_examples:
        logger.error("missing artifacts: online=%s (%d), upstream=%s (%d)",
                     online_path, len(online_examples), upstream_path, len(upstream_examples))
        return 2
    if args.max_online:
        online_examples = online_examples[: args.max_online]
    if args.max_upstream:
        upstream_examples = upstream_examples[: args.max_upstream]

    # base PTLM f_0 + refinement engine (Sec. 2)
    try:
        from ..modeling.base_lm import load_base_lm
        from ..modeling.refinement import build_refinement_engine

        base_lm = load_base_lm(args.model_key, device=args.device, dtype=args.dtype,
                               cache_dir=config.get("cache_dir"))
        engine = build_refinement_engine(base_lm, model_key=args.model_key,
                                         mode=args.tuning if args.tuning != "none" else "full_ft",
                                         config=config)
    except Exception as exc:
        logger.error("could not build the refinement engine: %s", exc)
        return 2

    # optional forecast-driven selectors (cache-only inference, Sec. 3.2/3.3)
    selector = None
    if args.method in ("representation", "logit"):
        try:
            from ..modeling.caches import load_caches

            cache_dir = os.path.join(root, "ground_truth")
            cache = load_caches(cache_dir)
            gt_dir = args.gt_dir or os.path.join(root, "ground_truth")
            gt_records = _load_jsonl(os.path.join(gt_dir, "pairs_train.jsonl"))
            if args.method == "representation":
                from ..forecasters.representation_based import RepresentationBasedForecaster
                from ..modeling.encoder_h import load_encoder_h

                forecaster = RepresentationBasedForecaster(dim=768, use_prior=not args.no_prior)
                ckpt = os.path.join(root, "representation_forecaster.pt")
                if os.path.isfile(ckpt):
                    forecaster.load_state(ckpt)
                encoder = load_encoder_h(args.model_key, config=config)
                prior = None
                score_fn = build_representation_score_provider(
                    forecaster, encoder, cache, online_examples, prior=prior,
                    use_prior=not args.no_prior, device=args.device)
            else:
                from ..forecasters.logit_based import FixedLogitForecaster, LogitChangeTransferForecaster
                from ..modeling.encoder_h import load_encoder_h

                encoder = load_encoder_h(args.model_key, config=config)
                forecaster = (FixedLogitForecaster.from_base_lm(base_lm) if args.fixed
                              else LogitChangeTransferForecaster(encoder=encoder))
                ckpt = os.path.join(root, "logit_forecaster.pt")
                if os.path.isfile(ckpt) and not args.fixed:
                    forecaster.load_state(ckpt)
                online_deltas = build_online_delta_lookup(gt_records)
                score_fn = build_logit_score_provider(
                    forecaster, encoder, cache, online_deltas,
                    upstream_indices=list(range(len(upstream_examples))))
            selector = build_selector(args.method,
                                     n_upstream=len(upstream_examples),
                                     default_n=replay_schedule_for(args.model_key, config)[0],
                                     seed=args.seed, score_fn=score_fn)
        except Exception as exc:
            logger.warning("cache-based selection unavailable (%s); falling back to random replay", exc)
            selector = None
    if selector is None and args.method in ("threshold", "gt"):
        gt_dir = args.gt_dir or os.path.join(root, "ground_truth")
        gt_records = _load_jsonl(os.path.join(gt_dir, "pairs_train.jsonl"))
        if args.method == "gt":
            selector = build_selector("gt", n_upstream=len(upstream_examples),
                                      default_n=replay_schedule_for(args.model_key, config)[0],
                                      seed=args.seed, labels=lookup_labels(gt_records, aggregate="last"))
        else:
            selector = build_selector("threshold", n_upstream=len(upstream_examples),
                                      default_n=replay_schedule_for(args.model_key, config)[0],
                                      seed=args.seed, gamma=args.gamma)

    runner = SequentialReplayRefinement(
        engine, online_examples, upstream_examples,
        method=args.method, selector=selector, model_key=args.model_key, tuning=args.tuning,
        config=config, sequential=args.sequential, replay_batch_size=args.replay_batch_size,
        replay_every_n_steps=args.replay_every_n_steps, steps=args.steps, lr=args.lr,
        eval_every=args.eval_every, batch_size=args.batch_size, seed=args.seed,
        device=args.device, distill_mode=args.distill_mode,
        distill_temperature=args.distill_temperature, distill_weight=args.distill_weight,
        shuffle=args.shuffle, verbose=args.verbose,
    )
    result = runner.run()
    out_dir = args.out_dir or os.path.join(root, "replay", args.method)
    paths = result.save(out_dir)
    logger.info("replay refinement done: Edit Success=%.2f%% EM Drop=%.3f%% -> %s",
                result.summary.get("edit_success_rate") or float("nan"),
                result.summary.get("em_drop_percent") or float("nan"), paths["summary"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
