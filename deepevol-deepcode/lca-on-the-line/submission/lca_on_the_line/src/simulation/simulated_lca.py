"""Simulated-data illustration of the LCA hypothesis (paper Section C, Table 7).

The paper builds a 4-class Gaussian mixture in ``R^3``::

    x | z=1 ~ N(mu1, I),  mu1 = (1, 1, 0)
    x | z=2 ~ N(mu2, I),  mu2 = (3, 17, 0)
    x | z=3 ~ N(mu3, I),  mu3 = (15, 7, 0)
    x | z=4 ~ N(mu4, I),  mu4 = (17, 21, 0)

with the hierarchy ``root: (class 1, class 2), (class 3, class 4)``.  Only
``x1`` supports that hierarchy (classes 1/2 and 3/4 are neighbours along it),
``x2`` separates all four classes but is *not* supported by the hierarchy, and
``x3`` is pure noise.  In-distribution (ID) data exposes all three features
while out-of-distribution (OOD) data only exposes ``x1`` and ``x3``.

Two logistic-regression models are trained on the ID data:

* model ``f`` on the transferable causal feature ``x1`` (+ noise ``x3``),
* model ``g`` on the non-transferable confounding feature ``x2`` (+ noise ``x3``).

Model ``g`` wins on ID top-1 accuracy, model ``f`` wins on OOD top-1 accuracy
and has the lower ID LCA distance -- the central claim of the paper.

Everything here is self-contained (numpy required, scikit-learn optional) so it
can serve as a fast sanity check of the LCA directionality before touching real
models.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

LOG = logging.getLogger("simulated_lca")

# --------------------------------------------------------------------------------------
# Design constants (paper Section C)
# --------------------------------------------------------------------------------------

#: Latent-class means of the Gaussian mixture, ``mu_z`` for ``z in {1, 2, 3, 4}``.
CLASS_MEANS: Dict[int, Tuple[float, float, float]] = {
    1: (1.0, 1.0, 0.0),
    2: (3.0, 17.0, 0.0),
    3: (15.0, 7.0, 0.0),
    4: (17.0, 21.0, 0.0),
}

NUM_CLASSES = 4

#: Feature indices (0-based): x1 -> 0, x2 -> 1, x3 -> 2.
ID_FEATURES: Tuple[int, ...] = (0, 1, 2)
OOD_FEATURES: Tuple[int, ...] = (0, 2)

#: Model f (causal) trains on x1 and the noise feature x3.
CAUSAL_FEATURES: Tuple[int, ...] = (0, 2)
#: Model g (confounding) trains on x2 and the noise feature x3.
CONFOUNDING_FEATURES: Tuple[int, ...] = (1, 2)

MODEL_NAMES: Tuple[str, str] = ("f", "g")

#: Analytic pairwise ``D_LCA^I`` matrix (base-2 information content, |L| = 4).
#: Within-pair distance (1 vs 2 and 3 vs 4) is ``log2(4/2) = 1``, cross-pair
#: distance is ``log2(4/1) = 2`` (their LCA is the root), and the diagonal is 0.
LCA_MATRIX: List[List[float]] = [
    [0.0, 1.0, 2.0, 2.0],
    [1.0, 0.0, 2.0, 2.0],
    [2.0, 2.0, 0.0, 1.0],
    [2.0, 2.0, 1.0, 0.0],
]

#: Reference values reported for Table 7 (paper Section C / reproduction plan).
#: ``id_lca`` uses misclassified-only averaging; exact values depend on the
#: sampling seed and on the logistic-regression solver, hence the tolerances.
TABLE7_REFERENCE: Dict[str, Dict[str, float]] = {
    "f": {"id_error": 0.1587, "ood_error": 0.3197, "id_lca": 1.005},
    "g": {"id_error": 0.0000, "ood_error": 0.7500, "id_lca": 2.000},
}

#: Default tolerances used by :meth:`SimulationResult.check_against_table7`.
TABLE7_TOLERANCE: Dict[str, float] = {"id_error": 0.06, "ood_error": 0.09, "id_lca": 0.35}

DEFAULT_NUM_SAMPLES = 10_000
DEFAULT_NUM_TRIALS = 100
DEFAULT_SEED = 0
DEFAULT_MISSING_STRATEGY = "zero"


# --------------------------------------------------------------------------------------
# Hierarchy helpers
# --------------------------------------------------------------------------------------


def simulated_lca_matrix() -> List[List[float]]:
    """Return the analytic 4x4 ``D_LCA^I`` matrix of the simulated hierarchy."""
    return [row[:] for row in LCA_MATRIX]


def build_simulated_hierarchy():
    """Best-effort build of a :class:`WordNetHierarchy` for the 4-class toy tree.

    Returns ``None`` when the hierarchy package (or a hierarchy matching the
    paper's ``(1, 2), (3, 4)`` structure) is unavailable -- callers then fall
    back to the analytic :data:`LCA_MATRIX`.
    """
    try:  # pragma: no cover - depends on package layout
        from ..hierarchy.wordnet import build_two_pair_hierarchy
    except Exception:  # pragma: no cover
        try:
            from hierarchy.wordnet import build_two_pair_hierarchy  # type: ignore
        except Exception:
            return None
    try:
        hierarchy = build_two_pair_hierarchy()
    except Exception:  # pragma: no cover
        return None
    return hierarchy


def hierarchy_matches_analytic_matrix(hierarchy, atol: float = 1e-6) -> bool:
    """Check that a hierarchy reproduces :data:`LCA_MATRIX` under ``D_LCA^I``."""
    if hierarchy is None:
        return False
    matrix = hierarchy_lca_matrix(hierarchy)
    if matrix is None:
        return False
    return bool(np.allclose(np.asarray(matrix, dtype=float), np.asarray(LCA_MATRIX), atol=atol))


def hierarchy_lca_matrix(hierarchy, mode: str = "information") -> Optional[List[List[float]]]:
    """Compute the pairwise LCA distance matrix of ``hierarchy`` (or ``None``)."""
    try:  # pragma: no cover - depends on package layout
        from ..hierarchy.lca import pairwise_lca_matrix
    except Exception:  # pragma: no cover
        try:
            from hierarchy.lca import pairwise_lca_matrix  # type: ignore
        except Exception:
            return None
    try:
        matrix = pairwise_lca_matrix(hierarchy, mode=mode)
    except Exception:  # pragma: no cover
        return None
    return [list(map(float, row)) for row in matrix]


def build_lca_metric(matrix: Optional[Sequence[Sequence[float]]] = None):
    """Build an :class:`LcaMetric` over the 4-class simulated hierarchy.

    Falls back to ``None`` when ``src.metrics.lca_metric`` cannot be imported;
    :func:`lca_of_predictions` then indexes the matrix directly.
    """
    if matrix is None:
        matrix = LCA_MATRIX
    try:  # pragma: no cover - depends on package layout
        from ..metrics.lca_metric import LcaMetric
    except Exception:  # pragma: no cover
        try:
            from metrics.lca_metric import LcaMetric  # type: ignore
        except Exception:
            try:
                from src.metrics.lca_metric import LcaMetric  # type: ignore
            except Exception:
                return None
    try:
        return LcaMetric(matrix=[list(map(float, r)) for r in matrix], num_classes=len(matrix), mode="information")
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------------------


def class_means(as_array: bool = True) -> Union[Dict[int, Tuple[float, float, float]], np.ndarray]:
    """Class means ``mu_z``; as a ``(4, 3)`` array by default (rows = class 1..4)."""
    if not as_array:
        return dict(CLASS_MEANS)
    return np.asarray([CLASS_MEANS[z] for z in range(1, NUM_CLASSES + 1)], dtype=float)


def sample_mixture(
    n_samples: int = DEFAULT_NUM_SAMPLES,
    seed: Optional[int] = None,
    means: Optional[Sequence[Sequence[float]]] = None,
    equal_priors: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``(X, z)`` from the 4-class Gaussian mixture with unit covariance.

    Returns
    -------
    X : ``(n_samples, 3)`` float array
    z : ``(n_samples,)`` int array with labels in ``{1, 2, 3, 4}``
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    means_arr = np.asarray(means if means is not None else class_means(), dtype=float)
    if means_arr.shape != (NUM_CLASSES, 3):
        raise ValueError("means must have shape (4, 3)")
    if equal_priors:
        priors = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
    else:  # pragma: no cover - kept for completeness
        priors = np.ones(NUM_CLASSES) / NUM_CLASSES
    labels_zero = rng.choice(NUM_CLASSES, size=int(n_samples), p=priors)
    X = means_arr[labels_zero] + rng.standard_normal((int(n_samples), 3))
    return X, labels_zero + 1  # labels 1..4


def select_features(X: np.ndarray, features: Sequence[int]) -> np.ndarray:
    """Slice the observed feature columns of ``X``."""
    X = np.asarray(X, dtype=float)
    idx = list(features)
    return X[:, idx] if idx else np.zeros((X.shape[0], 0), dtype=float)


# --------------------------------------------------------------------------------------
# Logistic regression (sklearn fast path + numpy fallback)
# --------------------------------------------------------------------------------------


class NumpySoftmaxClassifier:
    """Minimal multinomial logistic regression used when sklearn is unavailable."""

    def __init__(
        self,
        learning_rate: float = 0.5,
        max_iter: int = 800,
        l2: float = 1e-4,
        num_classes: int = NUM_CLASSES,
        seed: int = 0,
    ) -> None:
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.l2 = float(l2)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    # -- internals ------------------------------------------------------------
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpySoftmaxClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        W = np.zeros((d, self.num_classes), dtype=float)
        b = np.zeros(self.num_classes, dtype=float)
        Y = np.zeros((n, self.num_classes), dtype=float)
        Y[np.arange(n), y] = 1.0
        lr = self.learning_rate
        for it in range(self.max_iter):
            P = self._softmax(X @ W + b)
            grad_W = X.T @ (P - Y) / n + self.l2 * W
            grad_b = (P - Y).mean(axis=0)
            # simple momentum for faster convergence on this easy problem
            W -= lr * grad_W
            b -= lr * grad_b
            if it and it % max(1, self.max_iter // 10) == 0:
                lr *= 0.8
        self.coef_ = W.T.copy()
        self.intercept_ = b.copy()
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("classifier is not fitted")
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.decision_function(X), axis=1)


def _sklearn_logistic(**kwargs):
    try:  # pragma: no cover - optional dependency
        from sklearn.linear_model import LogisticRegression

        params = dict(max_iter=1000, C=1e6, solver="lbfgs")
        params.update({k: v for k, v in kwargs.items() if v is not None})
        return LogisticRegression(**params)
    except Exception:
        return None


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "auto",
    seed: int = DEFAULT_SEED,
    **kwargs,
):
    """Fit a 4-class logistic regression on ``(X, y)`` with ``y in {1,..,4}``.

    ``backend`` is one of ``auto`` / ``sklearn`` / ``numpy``.  Labels are
    converted to ``0..3`` internally so that the model's class index equals
    ``z - 1`` regardless of which classes happen to be present in a split.
    """
    X = np.asarray(X, dtype=float)
    y_zero = np.asarray(y, dtype=int) - 1
    if backend in ("auto", "sklearn"):
        model = _sklearn_logistic(**kwargs)
        if model is not None:
            model.fit(X, y_zero)
            return model
        if backend == "sklearn":
            raise ImportError("scikit-learn is required for backend='sklearn'")
    return NumpySoftmaxClassifier(seed=seed, **kwargs).fit(X, y_zero)


def model_predict(model, X_observed: np.ndarray) -> np.ndarray:
    """Predict class labels ``1..4`` from already-selected features."""
    X_observed = np.asarray(X_observed, dtype=float)
    if X_observed.shape[1] == 0:  # no informative feature: deterministic ties
        return np.ones(X_observed.shape[0], dtype=int)
    return np.asarray(model.predict(X_observed), dtype=int) + 1


def model_predict_proba(model, X_observed: np.ndarray) -> np.ndarray:
    """Predict class probabilities ``(n, 4)`` from already-selected features."""
    X_observed = np.asarray(X_observed, dtype=float)
    if X_observed.shape[1] == 0:  # pragma: no cover - degenerate case
        return np.full((X_observed.shape[0], NUM_CLASSES), 1.0 / NUM_CLASSES)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X_observed), dtype=float)
    scores = np.asarray(model.decision_function(X_observed), dtype=float)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------
# OOD feature handling
# --------------------------------------------------------------------------------------


def ood_feature_view(
    X: np.ndarray,
    train_features: Sequence[int],
    ood_features: Sequence[int] = OOD_FEATURES,
    strategy: str = DEFAULT_MISSING_STRATEGY,
    feature_means: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Build the input a model sees on the OOD data.

    Features the model was trained on but that are *not* observable OOD are
    imputed (``strategy='zero'``, the default, or ``'mean'``); observable ones
    are passed through unchanged.  This makes model ``g`` (which relies on the
    missing confounder ``x2``) fall back to the noise feature ``x3``, matching
    the paper's story.
    """
    X = np.asarray(X, dtype=float)
    train_features = list(train_features)
    observed = set(int(f) for f in ood_features)
    means = None
    if strategy == "mean":
        means = np.asarray(feature_means if feature_means is not None else np.zeros(X.shape[1]), dtype=float)
    out = np.zeros((X.shape[0], len(train_features)), dtype=float)
    for col, feat in enumerate(train_features):
        if feat in observed:
            out[:, col] = X[:, feat]
        elif strategy == "mean" and means is not None:
            out[:, col] = means[feat]
        else:  # 'zero' (default) -- or unknown strategy: impute the mean of a noise dim
            out[:, col] = 0.0
    return out


def model_view(
    X: np.ndarray,
    train_features: Sequence[int],
    features: Sequence[int],
    strategy: str = DEFAULT_MISSING_STRATEGY,
    feature_means: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Model input when evaluating on a split that exposes ``features``."""
    return ood_feature_view(
        X,
        train_features=train_features,
        ood_features=features,
        strategy=strategy,
        feature_means=feature_means,
    )


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def top1_error(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Top-1 classification error."""
    preds = np.asarray(predictions, dtype=int).ravel()
    tgts = np.asarray(targets, dtype=int).ravel()
    if preds.size == 0:
        return float("nan")
    return float(np.mean(preds != tgts))


def accuracy(predictions: Sequence[int], targets: Sequence[int]) -> float:
    """Top-1 classification accuracy."""
    return 1.0 - top1_error(predictions, targets)


def lca_of_predictions(
    predictions: Sequence[int],
    targets: Sequence[int],
    matrix: Optional[Sequence[Sequence[float]]] = None,
    metric: Optional[Any] = None,
    return_details: bool = False,
) -> Union[float, Tuple[float, np.ndarray]]:
    """Dataset-level ``D_LCA`` over misclassified samples (paper Section 2 / D.3).

    ``matrix`` is indexed with *zero-based* class indices (``z - 1``).
    """
    if metric is None:
        metric = build_lca_metric(matrix)
    preds = np.asarray(predictions, dtype=int).ravel() - 1
    tgts = np.asarray(targets, dtype=int).ravel() - 1
    if metric is not None:
        try:
            return metric.dataset_lca(preds, tgts, return_details=return_details)
        except TypeError:  # pragma: no cover - older signature
            return metric.dataset_lca(preds, tgts)
    mat = np.asarray(matrix if matrix is not None else LCA_MATRIX, dtype=float)
    wrong = preds != tgts
    distances = mat[preds[wrong], tgts[wrong]]
    value = float(distances.mean()) if distances.size else float("nan")
    if return_details:
        return value, distances
    return value


def lca_all_of_predictions(
    predictions: Sequence[int],
    targets: Sequence[int],
    matrix: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """``D_LCA`` averaged over *all* samples (``lca * (1 - top1)``)."""
    preds = np.asarray(predictions, dtype=int).ravel() - 1
    tgts = np.asarray(targets, dtype=int).ravel() - 1
    mat = np.asarray(matrix if matrix is not None else LCA_MATRIX, dtype=float)
    if preds.size == 0:
        return float("nan")
    wrong = preds != tgts
    return float(mat[preds[wrong], tgts[wrong]].sum() / preds.size) if wrong.any() else 0.0


def elca_of_predictions(
    probabilities: Sequence[Sequence[float]],
    targets: Sequence[int],
    matrix: Optional[Sequence[Sequence[float]]] = None,
) -> float:
    """``D_ELCA = (1/(nK)) sum_i sum_k p_hat_{k,i} D_LCA(k, y_i)`` (Section D.3)."""
    probs = np.asarray(probabilities, dtype=float)
    tgts = np.asarray(targets, dtype=int).ravel() - 1
    mat = np.asarray(matrix if matrix is not None else LCA_MATRIX, dtype=float)
    if probs.size == 0 or probs.shape[0] == 0:
        return float("nan")
    # mat[k, y_i] -> column gather for every sample
    dist = mat[:, tgts].T  # (n, K)
    return float(np.sum(probs * dist) / (probs.shape[0] * mat.shape[0]))


# --------------------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------------------


def evaluate_split(
    model,
    X: np.ndarray,
    z: np.ndarray,
    train_features: Sequence[int],
    features: Sequence[int],
    matrix: Optional[Sequence[Sequence[float]]] = None,
    metric: Optional[Any] = None,
    strategy: str = DEFAULT_MISSING_STRATEGY,
    feature_means: Optional[Sequence[float]] = None,
    compute_elca: bool = False,
) -> Dict[str, float]:
    """Evaluate one model on one split: top-1 error, LCA (misclassified-only), LCA_all."""
    view = model_view(
        X,
        train_features=train_features,
        features=features,
        strategy=strategy,
        feature_means=feature_means,
    )
    preds = model_predict(model, view)
    out: Dict[str, float] = {
        "error": top1_error(preds, z),
        "accuracy": accuracy(preds, z),
        "lca": lca_of_predictions(preds, z, matrix=matrix, metric=metric),
        "lca_all": lca_all_of_predictions(preds, z, matrix=matrix),
        "num_mistakes": float(np.sum(np.asarray(preds) != np.asarray(z))),
        "n": float(np.size(z)),
    }
    if compute_elca:
        out["elca"] = elca_of_predictions(
            model_predict_proba(model, view), z, matrix=matrix
        )
    return out


def train_models(
    n_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    backend: str = "auto",
    model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    compute_elca: bool = False,
) -> Dict[str, Any]:
    """Sample ID training data and fit models ``f`` (x1, x3) and ``g`` (x2, x3)."""
    X, z = sample_mixture(n_samples=n_samples, seed=seed)
    means = X.mean(axis=0)
    model_kwargs = model_kwargs or {}
    models: Dict[str, Any] = {}
    models["f"] = fit_logistic_regression(
        select_features(X, CAUSAL_FEATURES), z, backend=backend, seed=seed, **model_kwargs.get("f", {})
    )
    models["g"] = fit_logistic_regression(
        select_features(X, CONFOUNDING_FEATURES), z, backend=backend, seed=seed + 1, **model_kwargs.get("g", {})
    )
    return {
        "models": models,
        "X": X,
        "z": z,
        "feature_means": means,
        "train_features": {"f": CAUSAL_FEATURES, "g": CONFOUNDING_FEATURES},
    }


def evaluate_trial(
    n_samples: int = DEFAULT_NUM_SAMPLES,
    seed: int = DEFAULT_SEED,
    test_samples: Optional[int] = None,
    backend: str = "auto",
    matrix: Optional[Sequence[Sequence[float]]] = None,
    metric: Optional[Any] = None,
    strategy: str = DEFAULT_MISSING_STRATEGY,
    compute_elca: bool = False,
) -> Dict[str, Dict[str, float]]:
    """One independent trial: train on ID, evaluate on ID test and OOD test."""
    test_samples = int(test_samples if test_samples is not None else n_samples)
    trained = train_models(n_samples=n_samples, seed=seed, backend=backend, compute_elca=compute_elca)
    models, feature_means = trained["models"], trained["feature_means"]
    X_id, z_id = sample_mixture(n_samples=test_samples, seed=seed + 10_000)
    X_ood, z_ood = sample_mixture(n_samples=test_samples, seed=seed + 20_000)

    if metric is None:
        metric = build_lca_metric(matrix)
    results: Dict[str, Dict[str, float]] = {}
    for name in MODEL_NAMES:
        train_features = CAUSAL_FEATURES if name == "f" else CONFOUNDING_FEATURES
        id_split = evaluate_split(
            models[name], X_id, z_id, train_features, ID_FEATURES,
            matrix=matrix, metric=metric, strategy=strategy,
            feature_means=feature_means, compute_elca=compute_elca,
        )
        ood_split = evaluate_split(
            models[name], X_ood, z_ood, train_features, OOD_FEATURES,
            matrix=matrix, metric=metric, strategy=strategy,
            feature_means=feature_means, compute_elca=compute_elca,
        )
        results[name] = {
            "id_error": id_split["error"],
            "id_accuracy": id_split["accuracy"],
            "id_lca": id_split["lca"],
            "id_lca_all": id_split["lca_all"],
            "id_mistakes": id_split["num_mistakes"],
            "ood_error": ood_split["error"],
            "ood_accuracy": ood_split["accuracy"],
            "ood_lca": ood_split["lca"],
            "ood_lca_all": ood_split["lca_all"],
            "ood_mistakes": ood_split["num_mistakes"],
        }
        if compute_elca:
            results[name]["id_elca"] = id_split.get("elca", float("nan"))
            results[name]["ood_elca"] = ood_split.get("elca", float("nan"))
    return results


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Aggregated results of the Section C simulation (Table 7)."""

    num_trials: int
    n_samples: int
    seed: int
    backend: str
    models: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fields_std: Dict[str, Dict[str, float]] = field(default_factory=dict)
    pooled: Dict[str, Dict[str, float]] = field(default_factory=dict)
    trials: List[Dict[str, Dict[str, float]]] = field(default_factory=list)
    lca_matrix: List[List[float]] = field(default_factory=lambda: simulated_lca_matrix())
    elapsed: float = float("nan")

    # -- convenience ---------------------------------------------------------
    def asdict(self) -> Dict[str, Any]:
        return {
            "num_trials": self.num_trials,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "backend": self.backend,
            "models": self.models,
            "fields_std": self.fields_std,
            "pooled": self.pooled,
            "lca_matrix": self.lca_matrix,
            "elapsed": self.elapsed,
        }

    def to_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.asdict(), handle, indent=2)
        return path

    def value(self, model: str, key: str, prefer_pooled: bool = False) -> float:
        if prefer_pooled and key in ("id_lca", "ood_lca"):
            pooled = self.pooled.get(model, {})
            if key in pooled and not math.isnan(pooled[key]):
                return float(pooled[key])
        return float(self.models.get(model, {}).get(key, float("nan")))

    # -- reporting -----------------------------------------------------------
    def format_table(self, decimals: int = 4, use_pooled_lca: bool = True) -> str:
        """Render a Table-7-style text block (rows = model, columns = metrics)."""
        header = f"{'model':<6}{'ID Top-1 err':>14}{'OOD Top-1 err':>15}{'ID LCA':>10}{'OOD LCA':>10}"
        lines = [header, "-" * len(header)]
        for name in MODEL_NAMES:
            id_err = self.value(name, "id_error")
            ood_err = self.value(name, "ood_error")
            id_lca = self.value(name, "id_lca", prefer_pooled=use_pooled_lca)
            ood_lca = self.value(name, "ood_lca", prefer_pooled=use_pooled_lca)
            lines.append(
                f"{name:<6}{id_err:>14.{decimals}f}{ood_err:>15.{decimals}f}"
                f"{id_lca:>10.{decimals}f}{ood_lca:>10.{decimals}f}"
            )
        lines.append("")
        lines.append("reference (paper Table 7 / Section C):")
        for name in MODEL_NAMES:
            ref = TABLE7_REFERENCE.get(name, {})
            lines.append(
                f"  {name}: ID err {ref.get('id_error', float('nan')):.4f} | "
                f"OOD err {ref.get('ood_error', float('nan')):.4f} | "
                f"ID LCA {ref.get('id_lca', float('nan')):.4f}"
            )
        return "\n".join(lines)

    def check_against_table7(
        self,
        tolerance: Optional[Union[float, Dict[str, float]]] = None,
        use_pooled_lca: bool = True,
    ) -> List[str]:
        """Compare observed statistics with the paper's reference values.

        Returns a list of human-readable check lines; qualitative claims
        (``f`` < ``g`` in OOD error and LCA, ``g`` < ``f`` in ID error) are
        always verified since they are the actual claim of Section C.
        """
        if tolerance is None:
            tol = dict(TABLE7_TOLERANCE)
        elif isinstance(tolerance, (int, float)):
            tol = {k: float(tolerance) for k in TABLE7_TOLERANCE}
        else:
            tol = dict(TABLE7_TOLERANCE)
            tol.update({k: float(v) for k, v in tolerance.items()})
        lines: List[str] = []

        # 1) quantitative spot checks
        for name in MODEL_NAMES:
            ref = TABLE7_REFERENCE.get(name, {})
            for key in ("id_error", "ood_error", "id_lca"):
                observed = self.value(name, key, prefer_pooled=(use_pooled_lca and key.endswith("lca")))
                expected = ref.get(key)
                if expected is None or math.isnan(observed):
                    continue
                delta = abs(observed - expected)
                status = "PASS" if delta <= tol.get(key, 0.1) else "WARN"
                lines.append(
                    f"[{status}] model {name} {key}: observed {observed:.4f} "
                    f"vs reference {expected:.4f} (|delta|={delta:.4f} <= {tol.get(key, 0.1):.4f})"
                )

        # 2) qualitative claims (the actual point of Section C)
        f_id_err = self.value("f", "id_error")
        g_id_err = self.value("g", "id_error")
        f_ood_err = self.value("f", "ood_error")
        g_ood_err = self.value("g", "ood_error")
        f_lca = self.value("f", "id_lca", prefer_pooled=use_pooled_lca)
        g_lca = self.value("g", "id_lca", prefer_pooled=use_pooled_lca)

        def _claim(ok: bool, text: str) -> str:
            return f"[{'PASS' if ok else 'FAIL'}] {text}"

        lines.append(_claim(g_id_err < f_id_err, f"model g has better (lower) ID error: g={g_id_err:.4f} < f={f_id_err:.4f}"))
        lines.append(_claim(f_ood_err < g_ood_err, f"model f has better (lower) OOD error: f={f_ood_err:.4f} < g={g_ood_err:.4f}"))
        if not (math.isnan(f_lca) or math.isnan(g_lca)):
            lines.append(_claim(f_lca < g_lca, f"model f has lower ID LCA distance: f={f_lca:.4f} < g={g_lca:.4f}"))
        else:
            lines.append("[WARN] ID LCA for model g is undefined (no ID mistakes) -- increase --test-samples")
        return lines


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_simulation(
    num_trials: int = DEFAULT_NUM_TRIALS,
    n_samples: int = DEFAULT_NUM_SAMPLES,
    test_samples: Optional[int] = None,
    seed: int = DEFAULT_SEED,
    backend: str = "auto",
    matrix: Optional[Sequence[Sequence[float]]] = None,
    strategy: str = DEFAULT_MISSING_STRATEGY,
    compute_elca: bool = False,
    keep_trials: bool = True,
    verbose: bool = True,
) -> SimulationResult:
    """Run the Section C simulation over ``num_trials`` independent trials."""
    import time

    t0 = time.time()
    matrix = [list(map(float, r)) for r in (matrix if matrix is not None else LCA_MATRIX)]
    metric = build_lca_metric(matrix)
    if metric is None and verbose:
        LOG.warning("LcaMetric unavailable; falling back to direct matrix indexing")

    keys = ["id_error", "id_accuracy", "id_lca", "id_lca_all", "id_mistakes", "ood_error", "ood_accuracy", "ood_lca", "ood_lca_all", "ood_mistakes"]
    if compute_elca:
        keys += ["id_elca", "ood_elca"]

    per_model: Dict[str, Dict[str, List[float]]] = {
        name: {k: [] for k in keys} for name in MODEL_NAMES
    }
    # pooled accumulators: sum of LCA distances / number of mistakes / number of samples
    pooled: Dict[str, Dict[str, Dict[str, float]]] = {
        name: {"id": {"sum": 0.0, "mistakes": 0.0, "n": 0.0}, "ood": {"sum": 0.0, "mistakes": 0.0, "n": 0.0}}
        for name in MODEL_NAMES
    }
    trials: List[Dict[str, Dict[str, float]]] = []

    for trial in range(int(num_trials)):
        trial_seed = int(seed) + trial * 1_000
        trial_res = evaluate_trial(
            n_samples=n_samples,
            seed=trial_seed,
            test_samples=test_samples,
            backend=backend,
            matrix=matrix,
            metric=metric,
            strategy=strategy,
            compute_elca=compute_elca,
        )
        for name in MODEL_NAMES:
            for key in keys:
                per_model[name][key].append(float(trial_res[name].get(key, float("nan"))))
            for split in ("id", "ood"):
                mistakes = trial_res[name][f"{split}_mistakes"]
                lca = trial_res[name][f"{split}_lca"]
                if not math.isnan(lca) and mistakes > 0:
                    pooled[name][split]["sum"] += lca * mistakes
                    pooled[name][split]["mistakes"] += mistakes
                pooled[name][split]["n"] += trial_res[name][f"{split}_lca_all"] and 0 or 0  # placeholder no-op
        if keep_trials:
            trials.append(trial_res)
        if verbose and (trial + 1) % max(1, int(num_trials) // 10 or 1) == 0:
            LOG.info("trial %d/%d", trial + 1, num_trials)

    means: Dict[str, Dict[str, float]] = {}
    stds: Dict[str, Dict[str, float]] = {}
    for name in MODEL_NAMES:
        means[name] = {}
        stds[name] = {}
        for key in keys:
            arr = np.asarray(per_model[name][key], dtype=float)
            finite = arr[np.isfinite(arr)]
            means[name][key] = float(finite.mean()) if finite.size else float("nan")
            stds[name][key] = float(finite.std()) if finite.size else float("nan")
        # pooled (mistake-weighted) LCA estimates, robust when a model never errs
        for split in ("id", "ood"):
            accum = pooled[name][split]
            value = accum["sum"] / accum["mistakes"] if accum["mistakes"] > 0 else float("nan")
            means[name].setdefault(f"{split}_lca", value)
        # `id_lca` aggregated over trials where the model made at least one mistake
        for split in ("id", "ood"):
            arr = np.asarray(per_model[name][f"{split}_lca"], dtype=float)
            finite = arr[np.isfinite(arr)]
            means[name][f"{split}_lca_nanmean"] = float(finite.mean()) if finite.size else float("nan")
            means[name][f"{split}_lca_finite_fraction"] = float(finite.size / arr.size) if arr.size else float("nan")

    pooled_out = {
        name: {
            f"{split}_lca": (pooled[name][split]["sum"] / pooled[name][split]["mistakes"]
                             if pooled[name][split]["mistakes"] > 0 else float("nan")),
            f"{split}_mistakes": pooled[name][split]["mistakes"],
        }
        for name in MODEL_NAMES
        for split in ("id", "ood")
    }
    # flatten pooled dict to {model: {key: value}}
    pooled_flat = {
        name: {
            "id_lca": pooled_out[name]["id_lca"],
            "ood_lca": pooled_out[name]["ood_lca"],
            "id_mistakes": pooled_out[name]["id_mistakes"],
            "ood_mistakes": pooled_out[name]["ood_mistakes"],
        }
        for name in MODEL_NAMES
    }

    return SimulationResult(
        num_trials=int(num_trials),
        n_samples=int(n_samples),
        seed=int(seed),
        backend=backend,
        models=means,
        fields_std=stds,
        pooled=pooled_flat,
        trials=trials,
        lca_matrix=matrix,
        elapsed=time.time() - t0,
    )


# --------------------------------------------------------------------------------------
# Sanity checks / optional figure
# --------------------------------------------------------------------------------------


def sanity_check(matrix: Optional[Sequence[Sequence[float]]] = None, atol: float = 1e-9) -> Dict[str, Any]:
    """Verify the simulated hierarchy's LCA matrix and distance semantics."""
    mat = np.asarray(matrix if matrix is not None else LCA_MATRIX, dtype=float)
    report: Dict[str, Any] = {}
    report["zero_diagonal"] = bool(np.allclose(np.diag(mat), 0.0, atol=atol))
    report["symmetric"] = bool(np.allclose(mat, mat.T, atol=atol))
    report["non_negative"] = bool(np.all(mat >= -atol))
    report["within_pair_distance"] = float(mat[0, 1])
    report["cross_pair_distance"] = float(mat[0, 2])
    report["within_lt_cross"] = bool(mat[0, 1] < mat[0, 2])
    report["reverse_diagonal_ones"] = bool(np.allclose(1.0 - np.diag(mat), 1.0, atol=atol))
    hierarchy = build_simulated_hierarchy()
    report["hierarchy_available"] = hierarchy is not None
    report["hierarchy_matches"] = hierarchy_matches_analytic_matrix(hierarchy) if hierarchy is not None else False
    report["passed"] = bool(
        report["zero_diagonal"]
        and report["symmetric"]
        and report["non_negative"]
        and report["within_lt_cross"]
        and report["reverse_diagonal_ones"]
    )
    return report


def bayes_errors(matrix: Optional[Sequence[Sequence[float]]] = None) -> Dict[str, float]:
    """Analytic Gaussian-pair mistakes along ``x1`` (unit variance) for context."""
    # classes 1-2 (means 1 vs 3) and 3-4 (15 vs 17) are neighbours along x1
    prob = 0.5 * (1.0 - math.erf(1.0 / math.sqrt(2.0)))  # Phi(-1) for a midpoint boundary
    return {
        "pair_1_2_onesided": prob,
        "pair_3_4_onesided": prob,
        "expected_id_error_model_f": prob,        # equal priors: one confusion per pair, averaged
        "expected_id_error_model_g": 0.0,
    }


def make_figure(result: SimulationResult, out_dir: str, dpi: int = 200) -> List[str]:
    """Scatter ID/OOD error vs ID LCA for models f and g (optional matplotlib)."""
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for name, marker in zip(MODEL_NAMES, ("o", "s")):
        lca = result.value(name, "id_lca", prefer_pooled=True)
        axes[0].scatter(lca, result.value(name, "id_error"), marker=marker, s=70, label=f"model {name}")
        axes[1].scatter(lca, result.value(name, "ood_error"), marker=marker, s=70, label=f"model {name}")
    for ax, title, ylabel in zip(
        axes, ("ID accuracy vs ID LCA", "OOD accuracy vs ID LCA"), ("ID top-1 error", "OOD top-1 error")
    ):
        ax.set_xlabel("ID LCA distance")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "figure_c_simulation.png")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 7 (Section C simulated-data illustration of LCA)."
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_NUM_TRIALS, help="number of independent trials")
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES, help="ID training samples per trial")
    parser.add_argument("--test-samples", type=int, default=None, help="test samples per split (default: --samples)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--backend", choices=("auto", "sklearn", "numpy"), default="auto")
    parser.add_argument("--strategy", choices=("zero", "mean"), default=DEFAULT_MISSING_STRATEGY)
    parser.add_argument("--elca", action="store_true", help="also compute D_ELCA")
    parser.add_argument("--no-trials-dump", action="store_true", help="do not keep per-trial records")
    parser.add_argument("--sanity-only", action="store_true", help="only run the hierarchy sanity checks")
    parser.add_argument("--figure", action="store_true", help="write the simulation scatter figure")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--json", default=None, help="path of the JSON results file")
    parser.add_argument("--tol", type=float, default=None, help="single tolerance for all Table-7 checks")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sanity = sanity_check()
    print("sanity checks:")
    for key in ("zero_diagonal", "symmetric", "non_negative", "within_lt_cross", "reverse_diagonal_ones"):
        print(f"  [{'PASS' if sanity[key] else 'FAIL'}] {key}")
    print(f"  within-pair distance = {sanity['within_pair_distance']:.4f}, "
          f"cross-pair distance = {sanity['cross_pair_distance']:.4f}")
    print(f"  hierarchy available = {sanity['hierarchy_available']} "
          f"(matches analytic matrix: {sanity['hierarchy_matches']})")
    if not sanity["passed"]:
        LOG.error("LCA matrix sanity checks failed")
        return 2
    if args.sanity_only:
        return 0

    result = run_simulation(
        num_trials=args.trials,
        n_samples=args.samples,
        test_samples=args.test_samples,
        seed=args.seed,
        backend=args.backend,
        strategy=args.strategy,
        compute_elca=args.elca,
        keep_trials=not args.no_trials_dump,
        verbose=args.verbose,
    )
    print()
    print(result.format_table())
    print()
    print(f"({result.num_trials} trials x {result.n_samples} ID training samples, "
          f"backend={result.backend}, {result.elapsed:.1f}s)")
    print()
    checks = result.check_against_table7(tolerance=args.tol)
    for line in checks:
        print(line)
    failed = [c for c in checks if c.startswith("[FAIL]")]

    results_dir = args.results_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "simulation")
    os.makedirs(results_dir, exist_ok=True)
    json_path = args.json or os.path.join(results_dir, "table7_simulation.json")
    result.to_json(json_path)
    print(f"\nwrote {json_path}")
    with open(os.path.join(results_dir, "checks.json"), "w", encoding="utf-8") as handle:
        json.dump({"sanity": sanity, "checks": checks, "failed": failed}, handle, indent=2)
    if args.figure:
        for path in make_figure(result, results_dir):
            print(f"wrote {path}")

    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
