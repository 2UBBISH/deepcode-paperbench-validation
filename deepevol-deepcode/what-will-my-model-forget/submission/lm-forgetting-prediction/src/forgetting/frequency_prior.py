"""Frequency priors b_j for representation-based forecasting.

Paper: "What Will My Model Forget? Forecasting Forgotten Examples in Language
Model Refinement".

Source: Sec. 3.3 ("Forecasting with Frequency Priors") and Appendix F
Algorithms 3 & 4.

The paper defines, for every upstream (pretraining) example
<x_j, y_j> in D_PT, the frequency prior as the log odds that the example is
forgotten by the online (refinement) examples of the training split::

    b_j = log( |{<x_i, y_i> in D_R^train | z_ij = 1}| / |D_R^train| )
        - log( |{<x_i, y_i> in D_R^train | z_ij = 0}| / |D_R^train| )

i.e. ``b_j = log(P(z_ij = 1)) - log(P(z_ij = 0))``, where the probabilities are
estimated empirically over the online examples of D_R^train.  The prior is used
as an additive bias inside the representation-based forecaster

    z_tilde_ij = sigma( h(x_j, y_j) h(x_i, y_i)^T + b_j )      (Eq. 4)

so that the learned encoder only has to fit the *residual* interactions that
threshold-based forecasting (Sec. 3.1) cannot capture, and it is cached per
upstream example for cheap inference (Appendix F, Algorithm 4).

Notes on ambiguity
------------------
* The paper does not state how to handle the degenerate case of an upstream
  example that is *never* (or *always*) forgotten within D_R^train, where the
  empirical probability is 0 (log -> -inf).  We expose ``eps`` (default
  ``1e-8``) which clips the two frequencies before taking logs, and an optional
  symmetric Laplace-style ``smoothing`` parameter (default ``0.0``, i.e. the
  paper's plain empirical estimate).  Both are documented in the manifest.
* Indexing is by the *upstream* example index ``j`` into D_PT_hat, matching the
  caches used by the forecasters.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_EPS = 1e-8
DEFAULT_PRIOR_FILENAME = "frequency_prior.json"

__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_PRIOR_FILENAME",
    "FrequencyPrior",
    "prior_from_log_odds",
    "log_odds",
    "counts_from_pairs",
    "estimate_frequency_prior",
    "estimate_frequency_prior_with_model",
    "estimate_prior_from_ground_truth_dir",
    "prior_for_upstream",
    "load_prior",
    "save_prior",
    "main",
    "parse_args",
]


# ---------------------------------------------------------------------------
# scalar helpers
# ---------------------------------------------------------------------------
def log_odds(p_positive: float, p_negative: float, eps: float = DEFAULT_EPS) -> float:
    """``log(p_positive) - log(p_negative)`` with numerical guards.

    Parameters
    ----------
    p_positive, p_negative:
        Empirical frequencies (in [0, 1]) of ``z_ij = 1`` / ``z_ij = 0``.
    eps:
        Lower clip applied before the logarithm to avoid ``-inf`` for
        degenerate upstream examples.  ``eps`` is *small*, so an example that is
        never forgotten receives a large positive prior (matching the paper's
        intent that such examples are very unlikely to be forgotten).
    """
    p_positive = max(float(p_positive), float(eps))
    p_negative = max(float(p_negative), float(eps))
    return math.log(p_positive) - math.log(p_negative)


def prior_from_log_odds(n_pos: float, n_neg: float, eps: float = DEFAULT_EPS) -> float:
    """Log-odds prior computed from raw counts ``n_pos`` / ``n_neg``."""
    total = float(n_pos) + float(n_neg)
    if total <= 0.0:
        return 0.0
    return log_odds(float(n_pos) / total, float(n_neg) / total, eps=eps)


# ---------------------------------------------------------------------------
# FrequencyPrior container
# ---------------------------------------------------------------------------
class FrequencyPrior:
    """Mapping ``upstream index j -> b_j`` with caching / (de)serialization.

    The prior is *per upstream example*, exactly as in Algorithm 3/4 of the
    paper, and is therefore cached once and reused at inference time.
    """

    def __init__(
        self,
        priors: Optional[Mapping[Any, float]] = None,
        n_online: Optional[int] = None,
        n_upstream: Optional[int] = None,
        default: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.priors: Dict[int, float] = {
            int(k): float(v) for k, v in dict(priors or {}).items()
        }
        self.n_online = n_online
        self.n_upstream = n_upstream
        self.default = float(default)
        self.meta: Dict[str, Any] = dict(meta or {})
        # optional raw counts kept for inspection / table reproduction
        self.counts: Dict[int, Tuple[int, int]] = {}

    # -- construction -------------------------------------------------------
    @classmethod
    def from_counts(
        cls,
        counts: Mapping[Any, Tuple[int, int]],
        n_online: Optional[int] = None,
        eps: float = DEFAULT_EPS,
        smoothing: float = 0.0,
        n_upstream: Optional[int] = None,
        default: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "FrequencyPrior":
        """Build a prior from ``{j: (n_pos, n_neg)}`` forgetting counts.

        ``smoothing`` (default 0.0) adds the constant to both counts, i.e. a
        symmetric Laplace correction.  ``eps`` only guards the logarithm.
        """
        priors: Dict[int, float] = {}
        clean_counts: Dict[int, Tuple[int, int]] = {}
        for key, value in counts.items():
            j = int(key)
            n_pos = float(value[0]) + float(smoothing)
            n_neg = float(value[1]) + float(smoothing)
            priors[j] = prior_from_log_odds(n_pos, n_neg, eps=eps)
            clean_counts[j] = (int(value[0]), int(value[1]))
        obj = cls(
            priors=priors,
            n_online=n_online,
            n_upstream=n_upstream,
            default=default,
            meta=meta,
        )
        obj.counts = clean_counts
        obj.meta.setdefault("eps", float(eps))
        obj.meta.setdefault("smoothing", float(smoothing))
        return obj

    @classmethod
    def from_pairs(
        cls,
        pair_records: Iterable[Any],
        n_online: Optional[int] = None,
        n_upstream: Optional[int] = None,
        eps: float = DEFAULT_EPS,
        smoothing: float = 0.0,
        default: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
        online_indices: Optional[Iterable[int]] = None,
    ) -> "FrequencyPrior":
        """Estimate ``b_j`` from labelled ``(i, j, z_ij)`` pair records.

        ``pair_records`` may contain :class:`src.forgetting.ground_truth.PairRecord`
        objects or plain dicts exposing the keys ``i``, ``j`` and ``z``.
        ``online_indices`` optionally restricts the online examples that the
        probabilities are averaged over (i.e. selecting the D_R^train rows).
        """
        keep: Optional[set] = None
        if online_indices is not None:
            keep = {int(i) for i in online_indices}

        counts: Dict[int, List[int]] = {}
        seen_online: set = set()
        for record in pair_records:
            i, j, z = _pair_fields(record)
            if keep is not None and i not in keep:
                continue
            seen_online.add(i)
            bucket = counts.setdefault(j, [0, 0])
            bucket[1 if z else 0] += 1

        n_online_eff = len(keep) if keep is not None else len(seen_online)
        if n_online is None:
            n_online = n_online_eff
        if n_upstream is None:
            n_upstream = len(counts)
        return cls.from_counts(
            counts,
            n_online=n_online,
            eps=eps,
            smoothing=smoothing,
            n_upstream=n_upstream,
            default=default,
            meta=meta,
        )

    # -- access -------------------------------------------------------------
    def get(self, j: int, default: Optional[float] = None) -> float:
        if default is None:
            default = self.default
        return float(self.priors.get(int(j), default))

    def __getitem__(self, j: int) -> float:
        return self.get(j)

    def __contains__(self, j: Any) -> bool:
        return int(j) in self.priors

    def __len__(self) -> int:
        return len(self.priors)

    def __iter__(self):
        return iter(self.priors)

    def items(self):
        return self.priors.items()

    def keys(self):
        return self.priors.keys()

    def values(self):
        return self.priors.values()

    def as_list(self, n: Optional[int] = None, fill: Optional[float] = None) -> List[float]:
        """Dense ``b_j`` list indexed by upstream position (``fill`` for unseen)."""
        if fill is None:
            fill = self.default
        if n is None:
            n = (max(self.priors) + 1) if self.priors else 0
        return [float(self.priors.get(j, fill)) for j in range(int(n))]

    def subset(self, indices: Sequence[int]) -> "FrequencyPrior":
        return FrequencyPrior(
            {int(j): self.get(int(j)) for j in indices},
            n_online=self.n_online,
            n_upstream=len(indices),
            default=self.default,
            meta=dict(self.meta),
        )

    # -- serialization ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "priors": {str(j): v for j, v in sorted(self.priors.items())},
            "counts": {str(j): [int(c[0]), int(c[1])] for j, c in sorted(self.counts.items())},
            "n_online": self.n_online,
            "n_upstream": self.n_upstream,
            "default": self.default,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrequencyPrior":
        obj = cls(
            priors=data.get("priors", {}),
            n_online=data.get("n_online"),
            n_upstream=data.get("n_upstream"),
            default=data.get("default", 0.0),
            meta=data.get("meta", {}),
        )
        obj.counts = {
            int(k): (int(v[0]), int(v[1]))
            for k, v in dict(data.get("counts", {})).items()
        }
        return obj

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        logger.info("saved frequency priors (%d entries) to %s", len(self.priors), path)
        return path

    @classmethod
    def load(cls, path: str) -> "FrequencyPrior":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    # -- statistics ---------------------------------------------------------
    def positive_ratio(self) -> float:
        """Fraction of upstream examples whose prior favours 'forgotten'."""
        if not self.priors:
            return 0.0
        return sum(1 for v in self.priors.values() if v > 0.0) / float(len(self.priors))


# ---------------------------------------------------------------------------
# estimation drivers
# ---------------------------------------------------------------------------
def _pair_fields(record: Any) -> Tuple[int, int, int]:
    """Extract ``(i, j, z)`` from a PairRecord-like object or mapping."""
    if isinstance(record, Mapping):
        return int(record["i"]), int(record["j"]), int(record["z"])
    return int(record.i), int(record.j), int(record.z)


def counts_from_pairs(
    pair_records: Iterable[Any],
    n_upstream: Optional[int] = None,
    online_indices: Optional[Iterable[int]] = None,
) -> Dict[int, Tuple[int, int]]:
    """Count ``(n_forgotten, n_not_forgotten)`` per upstream index ``j``."""
    keep: Optional[set] = None
    if online_indices is not None:
        keep = {int(i) for i in online_indices}
    counts: Dict[int, List[int]] = {}
    for record in pair_records:
        i, j, z = _pair_fields(record)
        if keep is not None and i not in keep:
            continue
        bucket = counts.setdefault(j, [0, 0])
        bucket[1 if z else 0] += 1
    if n_upstream is not None:
        for j in range(int(n_upstream)):
            counts.setdefault(j, [0, 0])
    return {j: (c[0], c[1]) for j, c in counts.items()}


def estimate_frequency_prior(
    pair_records: Iterable[Any],
    n_online: Optional[int] = None,
    n_upstream: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    smoothing: float = 0.0,
    online_indices: Optional[Iterable[int]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> FrequencyPrior:
    """Estimate ``b_j`` exactly as in Sec. 3.3 / Algorithm 3.

    ``pair_records`` must be the ground-truth pairs produced for the *training*
    split of online examples D_R^train (see
    :func:`src.forgetting.ground_truth.generate_forgetting_pairs`).
    """
    pairs = list(pair_records)
    counts = counts_from_pairs(pairs, n_upstream=n_upstream, online_indices=online_indices)
    if n_online is None:
        if online_indices is not None:
            n_online = len({int(i) for i in online_indices})
        else:
            n_online = len({_pair_fields(r)[0] for r in pairs})
    prior = FrequencyPrior.from_counts(
        counts,
        n_online=n_online,
        eps=eps,
        smoothing=smoothing,
        n_upstream=n_upstream if n_upstream is not None else len(counts),
        meta=meta,
    )
    prior.meta.setdefault("source", "ground_truth_pairs")
    prior.meta.setdefault("n_pairs", len(pairs))
    return prior


def estimate_frequency_prior_with_model(
    f0: Any,
    refine_fn: Any,
    online_examples: Sequence[Mapping[str, Any]],
    upstream_examples: Sequence[Mapping[str, Any]],
    n_online: Optional[int] = None,
    online_indices: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    topk: int = 100,
    eps: float = DEFAULT_EPS,
    smoothing: float = 0.0,
    seed: int = 42,
    progress: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[FrequencyPrior, List[Any]]:
    """Estimate ``b_j`` by actually refining ``f_0`` on D_R^train examples.

    This mirrors Algorithm 3's inner bookkeeping: for each online example
    ``i in D_R^train`` we refine the base LM once and evaluate it on every
    upstream example of D_PT, yielding the ``z_ij`` needed for the counts.  The
    heavy lifting (batched forward passes + logit caching) is delegated to
    :func:`src.forgetting.ground_truth.generate_forgetting_pairs` so the labels
    are exactly the ones used to train the forecasters.

    Returns ``(prior, pair_records)``.
    """
    from .ground_truth import generate_forgetting_pairs  # local import (heavy)

    online_records, pair_records = generate_forgetting_pairs(
        f0=f0,
        refine_fn=refine_fn,
        online_examples=list(online_examples),
        upstream_examples=list(upstream_examples),
        online_indices=list(online_indices) if online_indices is not None else None,
        batch_size=batch_size,
        topk=topk,
        collect_logits=False,
        progress=progress,
    )
    if n_online is None:
        n_online = len(online_records)
    prior = estimate_frequency_prior(
        pair_records,
        n_online=n_online,
        n_upstream=len(upstream_examples),
        eps=eps,
        smoothing=smoothing,
        meta=meta,
    )
    prior.meta.setdefault("seed", int(seed))
    prior.meta.setdefault("source", "model_ground_truth")
    return prior, pair_records


def estimate_prior_from_ground_truth_dir(
    gt_dir: str,
    n_upstream: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    smoothing: float = 0.0,
    meta: Optional[Dict[str, Any]] = None,
) -> FrequencyPrior:
    """Load ``pairs.jsonl`` from ``gt_dir`` and estimate the prior from it."""
    from .ground_truth import load_ground_truth_jsonl  # local import (heavy)

    path = os.path.join(gt_dir, "pairs.jsonl")
    pairs = load_ground_truth_jsonl(path)
    meta = dict(meta or {})
    meta.setdefault("pairs_path", os.path.abspath(path))
    if n_upstream is None:
        try:
            with open(os.path.join(gt_dir, "meta.json"), "r", encoding="utf-8") as handle:
                summary = json.load(handle)
            n_upstream = (summary.get("summary") or {}).get("n_upstream") or summary.get(
                "n_upstream"
            )
        except Exception:  # pragma: no cover - optional metadata
            n_upstream = None
    return estimate_frequency_prior(
        pairs,
        n_upstream=n_upstream,
        eps=eps,
        smoothing=smoothing,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# convenience wrappers used by the forecasters / caches
# ---------------------------------------------------------------------------
def prior_for_upstream(
    prior: Any,
    upstream_index: int,
    default: float = 0.0,
) -> float:
    """Fetch ``b_j`` for one upstream index, tolerating dict / list inputs."""
    if prior is None:
        return float(default)
    if isinstance(prior, FrequencyPrior):
        return prior.get(int(upstream_index), default)
    if isinstance(prior, Mapping):
        value = prior.get(int(upstream_index), prior.get(str(int(upstream_index)), default))
        return float(value)
    if isinstance(prior, Sequence):
        idx = int(upstream_index)
        if 0 <= idx < len(prior):
            return float(prior[idx])
        return float(default)
    return float(default)


def save_prior(prior: FrequencyPrior, path: str) -> str:
    return prior.save(path)


def load_prior(path: str) -> FrequencyPrior:
    return FrequencyPrior.load(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate the frequency priors b_j (Sec. 3.3, Algorithm 3)."
    )
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory produced by generate_ground_truth.py (pairs.jsonl).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output json path (default: <gt-dir>/frequency_prior.json).")
    parser.add_argument("--n-upstream", type=int, default=None,
                        help="Number of upstream examples in D_PT_hat.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help="Clip applied to frequencies before taking logs.")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="Symmetric Laplace-style smoothing added to counts (default 0.0).")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    prior = estimate_prior_from_ground_truth_dir(
        args.gt_dir,
        n_upstream=args.n_upstream,
        eps=args.eps,
        smoothing=args.smoothing,
    )
    out = args.out or os.path.join(args.gt_dir, DEFAULT_PRIOR_FILENAME)
    prior.save(out)
    logger.info(
        "priors=%d | n_online=%s | frac(b_j>0)=%.4f",
        len(prior), prior.n_online, prior.positive_ratio(),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
