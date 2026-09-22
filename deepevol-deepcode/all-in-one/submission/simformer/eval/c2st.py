"""Classifier two-sample test (C2ST) evaluation for Simformer.

Implements the primary metric used in the paper (Sec. 4.1, Fig. 4):

    "For each task, samples for ten ground-truth posteriors are available
     (Appendix Sec. A2.2), and we assessed performance as classifier
     two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5
     signifies perfect alignment with the ground truth posterior, and 1.0
     indicates that a classifier can completely distinguish between the
     approximation and the ground truth."

Protocol (Appendix A3.1 / Addendum):
  * classifier: random forest with 100 trees,
  * metric: held-out classification accuracy between the two sample sets,
  * ``0.5`` -> the approximation is indistinguishable from the ground truth
    (ideal), ``1.0`` -> perfectly separable (bad).

The module is NumPy-only and degrades gracefully to a small pure-NumPy
decision-tree/forest fallback when ``scikit-learn`` is unavailable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]

__all__ = [
    "DEFAULT_N_TREES",
    "DEFAULT_TEST_FRACTION",
    "DEFAULT_SEED",
    "C2STConfig",
    "C2STResult",
    "c2st_accuracy",
    "c2st",
    "classifier_two_sample_test",
    "two_sample_test",
    "c2st_per_target",
    "c2st_accuracy_many",
    "aggregate_c2st",
    "evaluate_c2st",
    "c2st_from_sampler",
    "paired_c2st_table",
    "summarize",
    "RandomForestFallback",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_N_TREES = 100
DEFAULT_TEST_FRACTION = 0.3
DEFAULT_SEED = 0
DEFAULT_MAX_DEPTH: Optional[int] = None
CHANCE_LEVEL = 0.5
_HAS_SKLEARN = False
try:  # pragma: no cover - availability probe
    from sklearn.ensemble import RandomForestClassifier as _SKRandomForest

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _SKRandomForest = None  # type: ignore


# --------------------------------------------------------------------------- #
# Configuration / results
# --------------------------------------------------------------------------- #


@dataclass
class C2STConfig:
    """Configuration of the C2ST evaluation protocol.

    Attributes
    ----------
    n_trees:
        Number of trees of the random forest classifier (paper: 100).
    test_fraction:
        Fraction of the pooled samples held out for the accuracy estimate.
    max_depth:
        Optional maximum tree depth (``None`` = unlimited).
    seed:
        Seed for the train/test split and the random forest.
    n_parameters, n_data:
        Optional dimensionality metadata (used for logging only).
    standardize:
        Whether to z-score the samples before classification.  The paper does
        not specify any preprocessing; default ``False`` (random forests are
        scale invariant anyway).
    backend:
        ``"auto"``, ``"sklearn"`` or ``"numpy"``.
    normalize:
        Alias kept for symmetry with other configurations.
    """

    n_trees: int = DEFAULT_N_TREES
    test_fraction: float = DEFAULT_TEST_FRACTION
    max_depth: Optional[int] = DEFAULT_MAX_DEPTH
    seed: int = DEFAULT_SEED
    n_parameters: Optional[int] = None
    n_data: Optional[int] = None
    standardize: bool = False
    backend: str = "auto"
    normalize: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_trees": int(self.n_trees),
            "test_fraction": float(self.test_fraction),
            "max_depth": self.max_depth,
            "seed": int(self.seed),
            "standardize": bool(self.standardize),
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, cfg: Any = None, **kwargs: Any) -> "C2STConfig":
        if cfg is None:
            return cls(**kwargs)
        if isinstance(cfg, C2STConfig):
            base = cfg.to_dict()
            base.update(kwargs)
            return cls(**base)
        if isinstance(cfg, dict):
            merged = dict(cfg)
            merged.update(kwargs)
            known = {
                k: v
                for k, v in merged.items()
                if k in cls.__dataclass_fields__  # type: ignore[attr-defined]
            }
            extra = {
                k: v
                for k, v in merged.items()
                if k not in cls.__dataclass_fields__  # type: ignore[attr-defined]
            }
            obj = cls(**known)
            obj.extra.update(extra)
            return obj
        return cls(**kwargs)


@dataclass
class C2STResult:
    """Outcome of one C2ST comparison.

    Attributes
    ----------
    accuracy:
        Random-forest accuracy (``0.5`` = perfect match, ``1.0`` = fully
        separable).
    n_samples_approx, n_samples_reference:
        Sizes of the two compared sample sets.
    n_train, n_test:
        Split sizes actually used.
    backend:
        Which classifier implementation was used.
    std_error:
        Binomial standard error of the accuracy estimate.
    """

    accuracy: float
    n_samples_approx: int = 0
    n_samples_reference: int = 0
    n_train: int = 0
    n_test: int = 0
    backend: str = "sklearn"
    std_error: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Alias of :attr:`accuracy` (lower is better)."""
        return float(self.accuracy)

    @property
    def is_perfect(self) -> bool:
        return bool(self.accuracy <= CHANCE_LEVEL + 1e-8)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": float(self.accuracy),
            "std_error": float(self.std_error),
            "n_test": int(self.n_test),
            "backend": self.backend,
        }

    def __float__(self) -> float:  # convenience
        return float(self.accuracy)


# --------------------------------------------------------------------------- #
# Pure NumPy random-forest fallback
# --------------------------------------------------------------------------- #


class _DecisionTree:
    """Minimal CART classifier (Gini impurity) used when sklearn is missing."""

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        n_features: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.tree: Any = None

    # -- fitting ---------------------------------------------------------- #

    def _gini(self, y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        p = float(np.mean(y))
        return 2.0 * p * (1.0 - p)

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        n, d = X.shape
        best_gain = 0.0
        best_feature: Optional[int] = None
        best_threshold: Optional[float] = None
        parent = self._gini(y)
        if self.n_features is not None and self.n_features < d:
            feats = self.rng.choice(d, size=self.n_features, replace=False)
        else:
            feats = np.arange(d)
        for j in feats:
            col = X[:, j]
            order = np.argsort(col)
            col_sorted = col[order]
            y_sorted = y[order]
            # candidate thresholds = midpoints of distinct consecutive values
            distinct = np.nonzero(np.diff(col_sorted) > 0)[0]
            if distinct.size == 0:
                continue
            if distinct.size > 32:
                distinct = distinct[
                    self.rng.choice(
                        distinct.size, size=32, replace=False
                    )
                ]
            cum = np.cumsum(y_sorted)
            for i in distinct:
                n_left = i + 1
                n_right = n - n_left
                if n_left < 1 or n_right < 1:
                    continue
                p_left = cum[i] / n_left
                p_right = (cum[-1] - cum[i]) / n_right
                gini = (
                    (n_left / n) * 2.0 * p_left * (1.0 - p_left)
                    + (n_right / n) * 2.0 * p_right * (1.0 - p_right)
                )
                gain = parent - gini
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_feature = int(j)
                    best_threshold = 0.5 * (col_sorted[i] + col_sorted[i + 1])
        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Dict[str, Any]:
        node: Dict[str, Any] = {"leaf": float(np.mean(y)) if y.size else 0.5}
        if (
            y.size < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.all(y == y[0])
        ):
            return node
        feature, threshold, gain = self._best_split(X, y)
        if feature is None or gain <= 1e-12:
            return node
        left = X[:, feature] <= threshold
        node["feature"] = feature
        node["threshold"] = float(threshold)
        node["left"] = self._build(X[left], y[left], depth + 1)
        node["right"] = self._build(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_DecisionTree":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.tree = self._build(X, y, 0)
        return self

    # -- prediction -------------------------------------------------------- #

    def _predict_one(self, node: Dict[str, Any], x: np.ndarray) -> float:
        while "feature" in node:
            node = (
                node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
            )
        return float(node["leaf"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_one(self.tree, x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = np.clip(self.predict(X), 0.0, 1.0)
        return np.stack([1.0 - p1, p1], axis=-1)


class RandomForestFallback:
    """Bagged ensemble of :class:`_DecisionTree` with the same API subset."""

    def __init__(
        self,
        n_estimators: int = DEFAULT_N_TREES,
        max_depth: Optional[int] = None,
        seed: int = DEFAULT_SEED,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.seed = int(seed)
        self.trees: List[_DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestFallback":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for b in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = _DecisionTree(
                max_depth=self.max_depth,
                n_features=max(1, int(math.sqrt(d))),
                rng=np.random.default_rng(self.seed + 1000 + b),
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise RuntimeError("RandomForestFallback must be fitted first")
        proba = np.mean([t.predict_proba(X) for t in self.trees], axis=0)
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.float64)


def _build_classifier(config: C2STConfig) -> Tuple[Any, str]:
    """Instantiate the random forest (sklearn when available)."""
    backend = (config.backend or "auto").lower()
    if backend not in {"auto", "sklearn", "numpy"}:
        backend = "auto"
    if backend in {"auto", "sklearn"} and _HAS_SKLEARN:
        clf = _SKRandomForest(
            n_estimators=int(config.n_trees),
            max_depth=config.max_depth,
            random_state=int(config.seed),
            n_jobs=-1,
        )
        return clf, "sklearn"
    return (
        RandomForestFallback(
            n_estimators=int(config.n_trees),
            max_depth=config.max_depth,
            seed=int(config.seed),
        ),
        "numpy",
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_2d(samples: ArrayLike, name: str = "samples") -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_dims), got {arr.shape}")
    return arr


def _standardize_pair(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([a, b], axis=0)
    mean = pooled.mean(axis=0, keepdims=True)
    std = pooled.std(axis=0, keepdims=True) + 1e-8
    return (a - mean) / std, (b - mean) / std


def _balanced_subsample(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Truncate the larger set so both classes have equal size."""
    n = min(a.shape[0], b.shape[0])
    if a.shape[0] > n:
        a = a[rng.choice(a.shape[0], size=n, replace=False)]
    if b.shape[0] > n:
        b = b[rng.choice(b.shape[0], size=n, replace=False)]
    return a, b


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #


def c2st_accuracy(
    samples_a: ArrayLike,
    samples_b: ArrayLike,
    *,
    n_trees: int = DEFAULT_N_TREES,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    max_depth: Optional[int] = None,
    standardize: bool = False,
    backend: str = "auto",
    config: Optional[C2STConfig] = None,
    return_result: bool = False,
    balance: bool = True,
    train_samples_a: Optional[ArrayLike] = None,
    train_samples_b: Optional[ArrayLike] = None,
) -> Union[float, C2STResult]:
    """Classifier two-sample test accuracy between two sample sets.

    Parameters
    ----------
    samples_a, samples_b:
        ``(n, d)`` arrays of samples (e.g. Simformer posterior samples and
        ground-truth posterior samples).  If ``train_samples_*`` is given,
        these are used purely as the held-out test set.
    n_trees:
        Number of trees (paper: 100).
    test_fraction:
        Fraction of the pooled data used for the held-out accuracy.
    seed:
        Seed for split / forest.
    return_result:
        Return a :class:`C2STResult` instead of a bare float.

    Returns
    -------
    float or C2STResult
        Accuracy in ``[0, 1]``; ``0.5`` means the two distributions are
        indistinguishable given the sample size, ``1.0`` means perfectly
        separable.
    """
    if config is None:
        config = C2STConfig(
            n_trees=n_trees,
            test_fraction=test_fraction,
            seed=seed,
            max_depth=max_depth,
            standardize=standardize,
            backend=backend,
        )

    a = _as_2d(samples_a, "samples_a")
    b = _as_2d(samples_b, "samples_b")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"dimension mismatch: {a.shape[1]} vs {b.shape[1]} "
            "(both sets must live in the same space)"
        )

    rng = np.random.default_rng(int(config.seed))
    if balance:
        a, b = _balanced_subsample(a, b, rng)

    if train_samples_a is not None and train_samples_b is not None:
        X_train = np.concatenate(
            [_as_2d(train_samples_a, "train_samples_a"), _as_2d(train_samples_b, "train_samples_b")],
            axis=0,
        )
        y_train = np.concatenate(
            [
                np.zeros(_as_2d(train_samples_a).shape[0], dtype=np.float64),
                np.ones(_as_2d(train_samples_b).shape[0], dtype=np.float64),
            ]
        )
        X_test = np.concatenate([a, b], axis=0)
        y_test = np.concatenate(
            [np.zeros(a.shape[0], dtype=np.float64), np.ones(b.shape[0], dtype=np.float64)]
        )
    else:
        X = np.concatenate([a, b], axis=0)
        y = np.concatenate(
            [np.zeros(a.shape[0], dtype=np.float64), np.ones(b.shape[0], dtype=np.float64)]
        )
        perm = rng.permutation(X.shape[0])
        X, y = X[perm], y[perm]
        n_test = int(round(float(config.test_fraction) * X.shape[0]))
        n_test = max(1, min(n_test, X.shape[0] - 1)) if X.shape[0] > 1 else X.shape[0]
        X_test, y_test = X[:n_test], y[:n_test]
        X_train, y_train = X[n_test:], y[n_test:]

    if config.standardize:
        mean = X_train.mean(axis=0, keepdims=True)
        std = X_train.std(axis=0, keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    clf, used_backend = _build_classifier(config)

    if X_train.shape[0] == 0:  # degenerate: no training split
        accuracy = CHANCE_LEVEL
        n_train = 0
    else:
        clf.fit(X_train, y_train)
        predictions = np.asarray(clf.predict(X_test), dtype=np.float64).ravel()
        accuracy = float(np.mean((predictions >= 0.5).astype(np.float64) == y_test))
        n_train = int(X_train.shape[0])

    n_test = int(X_test.shape[0])
    std_error = (
        math.sqrt(max(accuracy * (1.0 - accuracy), 1e-12) / n_test) if n_test else 0.0
    )
    result = C2STResult(
        accuracy=accuracy,
        n_samples_approx=int(a.shape[0]),
        n_samples_reference=int(b.shape[0]),
        n_train=n_train,
        n_test=n_test,
        backend=used_backend,
        std_error=std_error,
        extra={"seed": int(config.seed), "n_trees": int(config.n_trees)},
    )
    return result if return_result else float(result.accuracy)


def c2st(samples_a: ArrayLike, samples_b: ArrayLike, **kwargs: Any) -> float:
    """Alias of :func:`c2st_accuracy` returning a plain float."""
    return float(c2st_accuracy(samples_a, samples_b, **kwargs))


def classifier_two_sample_test(
    samples_a: ArrayLike, samples_b: ArrayLike, **kwargs: Any
) -> float:
    """Alias of :func:`c2st_accuracy` (explicit spelling, sbi-compatible)."""
    return float(c2st_accuracy(samples_a, samples_b, **kwargs))


def two_sample_test(samples_a: ArrayLike, samples_b: ArrayLike, **kwargs: Any) -> float:
    """Short alias of :func:`c2st_accuracy`."""
    return float(c2st_accuracy(samples_a, samples_b, **kwargs))


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #


def c2st_per_target(
    approx_samples: Sequence[ArrayLike],
    reference_samples: Sequence[ArrayLike],
    *,
    n_trees: int = DEFAULT_N_TREES,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
    backend: str = "auto",
    return_result: bool = False,
) -> Union[List[float], List[C2STResult]]:
    """C2ST accuracy for each of several conditional targets.

    Matches the Sec. 4.1 protocol, where accuracy is computed separately for
    each of the (ten ground-truth posteriors / 100 random conditionals) and
    then averaged (see :func:`aggregate_c2st`).
    """
    if len(approx_samples) != len(reference_samples):
        raise ValueError("approx_samples and reference_samples must have equal length")
    results: List[Any] = []
    for i, (a, b) in enumerate(zip(approx_samples, reference_samples)):
        results.append(
            c2st_accuracy(
                a,
                b,
                n_trees=n_trees,
                test_fraction=test_fraction,
                seed=int(seed) + i,
                backend=backend,
                return_result=return_result,
            )
        )
    return results


def c2st_accuracy_many(
    approx_samples: Sequence[ArrayLike],
    reference_samples: Sequence[ArrayLike],
    **kwargs: Any,
) -> List[float]:
    """Convenience wrapper returning a list of float accuracies."""
    out = c2st_per_target(approx_samples, reference_samples, **kwargs)
    return [float(r) for r in out]


def aggregate_c2st(
    accuracies: Union[Sequence[float], Sequence[C2STResult]],
    *,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Mean/std/median of per-target C2ST accuracies (Fig. 4 error bars)."""
    vals = np.asarray(
        [float(a.accuracy) if isinstance(a, C2STResult) else float(a) for a in accuracies],
        dtype=np.float64,
    )
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "n": 0}
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else w
        mean = float(np.sum(w * vals[: w.size]))
    else:
        mean = float(np.mean(vals))
    return {
        "mean": mean,
        "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "median": float(np.median(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "chance_level": CHANCE_LEVEL,
        "n": int(vals.size),
    }


# --------------------------------------------------------------------------- #
# High level evaluation entry points
# --------------------------------------------------------------------------- #


def evaluate_c2st(
    samples_a: ArrayLike,
    samples_b: ArrayLike,
    *,
    n_trees: int = DEFAULT_N_TREES,
    seed: int = DEFAULT_SEED,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, C2STResult]:
    """Paper-default C2ST evaluation (100 random-forest trees).

    Accepts either two sample arrays, or two *sequences* of sample arrays
    (one per conditional target), in which case the per-target accuracies are
    averaged.
    """
    is_sequence = (
        isinstance(samples_a, (list, tuple))
        and len(samples_a) > 0
        and np.asarray(samples_a[0]).ndim >= 2
    )
    if is_sequence:
        accs = c2st_accuracy_many(
            samples_a, samples_b, n_trees=n_trees, seed=seed, **kwargs
        )
        summary = aggregate_c2st(accs)
        if return_result:
            return C2STResult(
                accuracy=summary["mean"],
                n_samples_approx=int(sum(np.asarray(a).shape[0] for a in samples_a)),
                n_samples_reference=int(sum(np.asarray(b).shape[0] for b in samples_b)),
                backend="ensemble",
                std_error=summary["std"] / math.sqrt(max(summary["n"], 1)),
                extra={"per_target": accs, **summary},
            )
        return float(summary["mean"])
    return c2st_accuracy(
        samples_a,
        samples_b,
        n_trees=n_trees,
        seed=seed,
        return_result=return_result,
        **kwargs,
    )


def c2st_from_sampler(
    sampler_fn: Callable[[int, np.random.Generator], np.ndarray],
    reference_samples: ArrayLike,
    *,
    n_samples: int = 1000,
    seed: int = DEFAULT_SEED,
    n_trees: int = DEFAULT_N_TREES,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, C2STResult]:
    """C2ST of a *callable* approximate sampler against reference samples.

    ``sampler_fn(n_samples, rng) -> (n_samples, d)`` draws from the
    approximated conditional (e.g. ``ConditionalSampler.posterior``).
    """
    rng = np.random.default_rng(int(seed))
    approx = sampler_fn(int(n_samples), rng)
    return c2st_accuracy(
        approx,
        reference_samples,
        n_trees=n_trees,
        seed=seed,
        return_result=return_result,
        **kwargs,
    )


def paired_c2st_table(
    results: Dict[str, Sequence[float]],
    *,
    reference_key: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Summarize named method -> per-target accuracies (benchmark table)."""
    table: Dict[str, Dict[str, float]] = {}
    for name, accs in results.items():
        table[name] = aggregate_c2st(accs)
    return table


def summarize(accuracies: Sequence[float]) -> Dict[str, float]:
    """Alias of :func:`aggregate_c2st` for plain accuracy lists."""
    return aggregate_c2st(accuracies)
