"""Frequency-threshold based forecasting baseline.

Paper: "What Will My Model Forget? Forecasting Forgotten Examples in Language
Model Refinement".

Source: Sec. 3.1 ("Frequency-Threshold based Forcasting"), Eq. 1, plus the
Addendum clarifications (the baseline is tuned on ``D_R^Train`` and evaluated
on ``D_PT_hat``).

The paper's baseline is

    g(<x_i, y_i>, <x_j, y_j>) = 1[ |{ i : z_ij = 1 }| >= gamma ]            (Eq. 1)

i.e. an upstream (pretraining) example ``x_j`` is predicted to be forgotten by
the current online edit iff it has been forgotten at least ``gamma`` times in
the past.  The paper's set-comprehension is written with the index ``j`` on the
left of Eq. 1, but the count is over the *online* examples ``i`` that caused
``x_j`` to be forgotten (``z_ij`` is defined in Sec. 2 as the indicator that
upstream example ``j`` is forgotten after refining on online example ``i``).
Following the reproduction plan we count over the online examples:

    count_j = |{ i in D_R^Train : z_ij = 1 }|                              (*)

``gamma`` is tuned to maximise the F1 of the baseline on ``D_R^Train``.

Notes / reconciliation of ambiguities
-------------------------------------
* Eq. 1 uses an *absolute* count (not a frequency).  We therefore tune over
  integer thresholds.  A *normalised* variant (count / number of online
  examples) is also provided for convenience (``use_frequency=True``), but the
  default -- and the one reported in the paper -- is the absolute count.
* The threshold baseline is static with respect to the online example: for a
  pair ``(i, j)`` the prediction depends only on ``j``.  This is intentional:
  the baseline "does not capture or interpret how interactions between the
  online learned example and the pretraining example contribute to
  forgetting" (Sec. 3.1).
* The paper does not specify the sign/edge behaviour at ``gamma = 0`` (which
  would predict everything positive).  We include it in the grid because it
  is a legitimate degenerate point; F1 maximisation simply will not select it
  unless the positive class dominates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "DEFAULT_GAMMA_GRID",
    "forget_counts",
    "forget_frequencies",
    "counts_by_upstream",
    "threshold_predict",
    "predict_pairs_with_counts",
    "tune_gamma",
    "evaluate_gamma_grid",
    "ThresholdForecaster",
    "DEFAULT_THRESHOLD_FILENAME",
    "main",
    "parse_args",
]

logger = logging.getLogger("forecasters.threshold")

# Default grid used when the config file does not provide one.  The paper does
# not publish the grid; a geometric-ish sweep up to large counts is a sensible
# default because the expected number of forgettings per upstream example is
# small (positive prevalence in D_PT_hat is ~1-10%).
DEFAULT_GAMMA_GRID: Tuple[int, ...] = (
    1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100,
)

DEFAULT_THRESHOLD_FILENAME = "threshold_forecaster.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _pair_index(pair: Any, name: str) -> Optional[Any]:
    """Fetch attribute or dict key ``name`` from a pair-like object."""
    if pair is None:
        return None
    if isinstance(pair, Mapping):
        return pair.get(name)
    return getattr(pair, name, None)


def _pair_z(pair: Any) -> int:
    z = _pair_index(pair, "z")
    if z is None:
        z = _pair_index(pair, "label")
    if z is None:
        raise ValueError("pair record has no 'z' / 'label' field")
    return int(z)


def _pair_j(pair: Any) -> Any:
    j = _pair_index(pair, "j")
    if j is None:
        j = _pair_index(pair, "upstream_index")
    if j is None:
        raise ValueError("pair record has no 'j' / 'upstream_index' field")
    return j


def counts_by_upstream(
    pair_records: Iterable[Any],
    n_upstream: Optional[int] = None,
    n_online: Optional[int] = None,
) -> Tuple[Dict[Any, int], Dict[Any, int]]:
    """Count positives and negatives per upstream example ``j``.

    Returns ``(n_positive, n_negative)`` mappings keyed by upstream index.
    Following Sec. 3.1 / Eq. 1 the positive count is the number of online
    examples for which ``z_ij = 1``.
    """
    n_pos: Dict[Any, int] = {}
    n_neg: Dict[Any, int] = {}
    seen_online: set = set()
    for pair in pair_records:
        j = _pair_j(pair)
        z = _pair_z(pair)
        if z:
            n_pos[j] = n_pos.get(j, 0) + 1
        else:
            n_neg[j] = n_neg.get(j, 0) + 1
        i = _pair_index(pair, "i")
        if i is None:
            i = _pair_index(pair, "online_index")
        if i is not None:
            seen_online.add(i)

    if n_upstream is not None and n_upstream > 0:
        for j in range(n_upstream):
            n_pos.setdefault(j, 0)
            n_neg.setdefault(j, 0)

    n_online_eff = n_online if n_online is not None else (len(seen_online) or None)
    if n_online_eff:
        return n_pos, n_neg
    return n_pos, n_neg


def forget_counts(
    pair_records: Iterable[Any],
    n_upstream: Optional[int] = None,
    n_online: Optional[int] = None,
) -> Dict[Any, float]:
    """Return ``count_j`` for every upstream example (Eq. 1 lhs count)."""
    n_pos, _ = counts_by_upstream(pair_records, n_upstream=n_upstream, n_online=n_online)
    return {j: float(c) for j, c in n_pos.items()}


def forget_frequencies(
    pair_records: Iterable[Any],
    n_upstream: Optional[int] = None,
    n_online: Optional[int] = None,
) -> Dict[Any, float]:
    """Return ``count_j / n_online`` for every upstream example.

    Only used for the (non-default) normalised variant of the baseline.
    """
    n_pos, n_neg = counts_by_upstream(pair_records, n_upstream=n_upstream)
    freqs: Dict[Any, float] = {}
    for j in set(n_pos) | set(n_neg):
        total = n_pos.get(j, 0) + n_neg.get(j, 0)
        freqs[j] = (float(n_pos.get(j, 0)) / float(total)) if total > 0 else 0.0
    if n_online and n_online > 0:
        # Absolute-frequency variant: normalize by the number of online examples
        freqs = {j: float(n_pos.get(j, 0)) / float(n_online) for j in freqs}
    return freqs


def threshold_predict(score: float, gamma: float) -> int:
    """``1[score >= gamma]`` (Eq. 1)."""
    return int(float(score) >= float(gamma))


def predict_pairs_with_counts(
    pair_records: Sequence[Any],
    scores: Mapping[Any, float],
    gamma: float,
    default_score: float = 0.0,
) -> List[int]:
    """Predict for each pair using pre-computed per-upstream ``scores``."""
    preds: List[int] = []
    for pair in pair_records:
        j = _pair_j(pair)
        preds.append(threshold_predict(scores.get(j, default_score), gamma))
    return preds


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _counts_ternary(preds: Sequence[int], labels: Sequence[int]) -> Tuple[int, int, int]:
    tp = fp = fn = 0
    for p, y in zip(preds, labels):
        if y == 1 and p == 1:
            tp += 1
        elif y == 0 and p == 1:
            fp += 1
        elif y == 1 and p == 0:
            fn += 1
    return tp, fp, fn


def _f1_from_metrics(preds: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
    """F1/precision/recall, delegating to :mod:`src.eval.metrics` when possible."""
    try:  # pragma: no cover - import is exercised in the full pipeline
        from ..eval import metrics as _metrics  # type: ignore

        for fn_name in ("binary_f1", "f1_score_binary", "forecast_f1"):
            fn = getattr(_metrics, fn_name, None)
            if callable(fn):
                out = fn(preds, labels)
                if isinstance(out, Mapping) and "f1" in out:
                    return {
                        "f1": float(out["f1"]),
                        "precision": float(out.get("precision", 0.0)),
                        "recall": float(out.get("recall", 0.0)),
                    }
    except Exception:  # pragma: no cover - fall back to local implementation
        pass

    tp, fp, fn = _counts_ternary(preds, labels)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    return {"f1": _f1(tp, fp, fn), "precision": precision, "recall": recall}


# --------------------------------------------------------------------------- #
# gamma tuning
# --------------------------------------------------------------------------- #
def evaluate_gamma_grid(
    pair_records: Sequence[Any],
    gammas: Iterable[float],
    scores: Optional[Mapping[Any, float]] = None,
    n_upstream: Optional[int] = None,
    default_score: float = 0.0,
) -> List[Dict[str, float]]:
    """Evaluate every ``gamma`` on ``pair_records`` (F1/precision/recall)."""
    labels = [_pair_z(p) for p in pair_records]
    js = [_pair_j(p) for p in pair_records]

    if scores is None:
        scores = forget_counts(pair_records, n_upstream=n_upstream)
    # `scores` was derived from these very pairs, so a transductive evaluation
    # (as done by the paper when tuning on D_R^Train) can use the counts of the
    # same pairs without extra cost.

    out: List[Dict[str, float]] = []
    for gamma in gammas:
        preds = [threshold_predict(scores.get(j, default_score), gamma) for j in js]
        m = _f1_from_metrics(preds, labels)
        m["gamma"] = float(gamma)
        out.append(m)
    return out


def tune_gamma(
    pair_records: Sequence[Any],
    gammas: Optional[Iterable[float]] = None,
    n_upstream: Optional[int] = None,
    use_frequency: bool = False,
    default_score: float = 0.0,
) -> Dict[str, Any]:
    """Tune ``gamma`` to maximise F1 on the given (training) pairs.

    Returns ``{"gamma", "f1", "precision", "recall", "curve"}``.
    """
    labels = [_pair_z(p) for p in pair_records]
    n_pos = sum(labels)
    if n_pos == 0:
        # Degenerate: nothing is ever forgotten.  Predicting all-negative gives
        # undefined F1; pick the smallest gamma and report zeros.
        return {
            "gamma": 1.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "n_pairs": len(labels),
            "n_positive": 0,
            "curve": [],
            "warning": "no positive pairs in tuning set",
        }

    scores = (
        forget_frequencies(pair_records, n_upstream=n_upstream)
        if use_frequency
        else forget_counts(pair_records, n_upstream=n_upstream)
    )

    if gammas is None:
        max_count = max(scores.values()) if scores else 1.0
        grid = sorted({g for g in DEFAULT_GAMMA_GRID if g <= max(1.0, max_count)})
        if use_frequency:
            grid = [i / 20.0 for i in range(1, 21)]
        else:
            grid = grid or [1]
    else:
        grid = sorted({float(g) for g in gammas})

    curve = evaluate_gamma_grid(
        pair_records, grid, scores=scores, n_upstream=n_upstream,
        default_score=default_score,
    )

    best = None
    for row in curve:
        if best is None:
            best = row
            continue
        # tie-break: higher F1, then higher precision, then smaller gamma
        key = (row["f1"], row["precision"], -row["gamma"])
        best_key = (best["f1"], best["precision"], -best["gamma"])
        if key > best_key:
            best = row

    assert best is not None
    return {
        "gamma": float(best["gamma"]),
        "f1": float(best["f1"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "n_pairs": len(labels),
        "n_positive": int(n_pos),
        "curve": curve,
    }


# --------------------------------------------------------------------------- #
# forecaster
# --------------------------------------------------------------------------- #
class ThresholdForecaster:
    """Frequency-threshold baseline forecaster (Sec. 3.1, Eq. 1).

    The forecaster is *static*: for a pair ``(i, j)`` the prediction depends
    only on the accumulated forget-count of upstream example ``j``.

    Parameters
    ----------
    gamma:
        Decision threshold (Eq. 1).  Tuned on ``D_R^Train`` when ``fit`` is
        called with ``tune=True``.
    use_frequency:
        Use the normalised forget-frequency instead of the absolute count.
        Default ``False`` (matches Eq. 1).
    default_score:
        Score assigned to upstream examples never observed in the fit set.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        use_frequency: bool = False,
        default_score: float = 0.0,
        n_upstream: Optional[int] = None,
        max_gamma: Optional[float] = None,
    ) -> None:
        self.gamma = float(gamma)
        self.use_frequency = bool(use_frequency)
        self.default_score = float(default_score)
        self.n_upstream = n_upstream
        self.max_gamma = max_gamma
        self.scores: Dict[Any, float] = {}
        self.curve: List[Dict[str, float]] = []
        self.meta: Dict[str, Any] = {}

    # -- fitting ---------------------------------------------------------- #
    def fit(
        self,
        pair_records: Sequence[Any],
        gammas: Optional[Iterable[float]] = None,
        n_upstream: Optional[int] = None,
        tune: bool = True,
    ) -> "ThresholdForecaster":
        """Accumulate forget statistics from ``D_R^Train`` and tune ``gamma``."""
        n_upstream = n_upstream if n_upstream is not None else self.n_upstream
        if self.use_frequency:
            self.scores = forget_frequencies(pair_records, n_upstream=n_upstream)
        else:
            self.scores = forget_counts(pair_records, n_upstream=n_upstream)

        if tune:
            res = tune_gamma(
                pair_records,
                gammas=gammas,
                n_upstream=n_upstream,
                use_frequency=self.use_frequency,
                default_score=self.default_score,
            )
            self.gamma = float(res["gamma"])
            self.curve = res.get("curve", [])
            self.meta = {
                "tune_f1": res.get("f1", 0.0),
                "tune_precision": res.get("precision", 0.0),
                "tune_recall": res.get("recall", 0.0),
                "n_pairs": res.get("n_pairs", 0),
                "n_positive": res.get("n_positive", 0),
            }
        self.n_upstream = n_upstream
        return self

    # -- scoring / prediction --------------------------------------------- #
    def score(self, upstream_index: Any) -> float:
        return float(self.scores.get(upstream_index, self.default_score))

    def predict_pairs(self, pair_records: Sequence[Any]) -> List[int]:
        """Predict ``z_hat_ij`` for every pair using the fitted counts."""
        return predict_pairs_with_counts(
            pair_records, self.scores, self.gamma, default_score=self.default_score
        )

    def predict_upstream(self, upstream_indices: Sequence[Any]) -> List[int]:
        """Predict for bare upstream indices (no online example needed)."""
        return [threshold_predict(self.score(j), self.gamma) for j in upstream_indices]

    def pairwise_matrix(self, online_indices: Sequence[Any], upstream_indices: Sequence[Any]):
        """Broadcast prediction to a ``[len(online), len(upstream)]`` list of lists."""
        row = self.predict_upstream(upstream_indices)
        return [list(row) for _ in online_indices]

    def predict(self, pairs: Sequence[Any]) -> List[int]:
        """Alias of :meth:`predict_pairs` for the generic forecaster contract."""
        return self.predict_pairs(pairs)

    # -- persistence ------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "threshold",
            "gamma": self.gamma,
            "use_frequency": self.use_frequency,
            "default_score": self.default_score,
            "n_upstream": self.n_upstream,
            "max_gamma": self.max_gamma,
            "scores": {str(k): float(v) for k, v in self.scores.items()},
            "curve": self.curve,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ThresholdForecaster":
        obj = cls(
            gamma=d.get("gamma", 1.0),
            use_frequency=d.get("use_frequency", False),
            default_score=d.get("default_score", 0.0),
            n_upstream=d.get("n_upstream"),
            max_gamma=d.get("max_gamma"),
        )
        scores = d.get("scores", {}) or {}
        # keys were stringified on save; restore ints when the upstream indices
        # were integers (the common case: upstream index in D_PT_hat).
        restored: Dict[Any, float] = {}
        for k, v in scores.items():
            try:
                restored[int(k)] = float(v)
            except (TypeError, ValueError):
                restored[k] = float(v)
        obj.scores = restored
        obj.curve = list(d.get("curve", []) or [])
        obj.meta = dict(d.get("meta", {}) or {})
        return obj

    def save(self, path: str) -> str:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ThresholdForecaster":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune the frequency-threshold baseline.")
    p.add_argument("--train-pairs", required=True,
                   help="pairs.jsonl for D_R^Train (ground-truth labels).")
    p.add_argument("--eval-pairs", default=None,
                   help="optional pairs.jsonl to report F1 on (e.g. D_R^Test).")
    p.add_argument("--out", default=None,
                   help="output JSON path for the fitted forecaster.")
    p.add_argument("--use-frequency", action="store_true",
                   help="use normalised forget-frequency instead of absolute count.")
    p.add_argument("--n-upstream", type=int, default=None)
    p.add_argument("--config", default=None)
    return p.parse_args(argv)


def _load_pairs(path: str) -> List[Any]:
    from ..forgetting.ground_truth import load_ground_truth_jsonl

    return load_ground_truth_jsonl(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    train_pairs = _load_pairs(args.train_pairs)
    gammas = None
    if args.config:
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            gammas = (cfg.get("forecaster", {}) or {}).get("threshold_gamma_grid")
        except Exception as exc:  # pragma: no cover
            logger.warning("could not read config %s: %s", args.config, exc)

    fc = ThresholdForecaster(
        use_frequency=args.use_frequency, n_upstream=args.n_upstream
    ).fit(train_pairs, gammas=gammas)
    logger.info(
        "tuned gamma=%.4g  F1=%.4f  P=%.4f  R=%.4f  (n_pairs=%d)",
        fc.gamma, fc.meta.get("tune_f1", 0.0), fc.meta.get("tune_precision", 0.0),
        fc.meta.get("tune_recall", 0.0), fc.meta.get("n_pairs", 0),
    )

    if args.eval_pairs:
        eval_pairs = _load_pairs(args.eval_pairs)
        preds = fc.predict_pairs(eval_pairs)
        labels = [_pair_z(p) for p in eval_pairs]
        m = _f1_from_metrics(preds, labels)
        logger.info(
            "eval F1=%.4f  P=%.4f  R=%.4f  (n_pairs=%d)",
            m["f1"], m["precision"], m["recall"], len(labels),
        )

    if args.out:
        fc.save(args.out)
        logger.info("saved threshold forecaster -> %s", args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
