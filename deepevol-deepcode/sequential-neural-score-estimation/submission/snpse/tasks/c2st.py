"""Classifier two-sample test (C2ST) used to evaluate posterior approximations.

Paper reference
---------------
Section 5.2 states that *"In all cases, we report the classification-based
two-sample test (C2ST) score (Lopez-Paz & Oquab, 2017), which varies between
0.5 and 1 (lower is better), with a score of 0.5 indicating perfect posterior
estimation."*  The score is computed with the Python toolkit ``sbibm``
(Lueckmann et al., 2021) with its default settings.

This module is a *thin wrapper* around ``sbibm.metrics.c2st``:

* if ``sbibm`` is installed, its ``c2st`` implementation (with default
  hyper-parameters) is used directly;
* otherwise a dependency-light, behaviourally equivalent fallback is used:
  a scikit-learn ``MLPClassifier`` trained on concatenated samples from the
  reference posterior and the estimated posterior, evaluated with stratified
  k-fold cross validation, exactly as described in Lopez-Paz & Oquab (2017)
  and implemented in ``sbibm``.

Public API
----------
``C2STResult``            -- dataclass with mean/std and per-fold scores
``c2st``                  -- raw metric on two sample tensors/arrays
``c2st_scores``           -- convenience alias returning the fold vector
``c2st_accuracy``         -- alias of ``c2st`` (default scoring is accuracy)
``task_c2st``             -- C2ST of posterior samples against a task's reference posterior
``reference_posterior``   -- pull reference posterior samples from a task (sbibm or fallback)
``evaluate_posterior``    -- generic evaluation entry point used by experiments
``to_numpy``              -- tensor/list -> float64 numpy helper
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when sbibm is installed
    import sbibm  # noqa: F401

    try:
        from sbibm.metrics.c2st import c2st as _sbibm_c2st
    except Exception:  # pragma: no cover - older/newer sbibm layouts
        from sbibm.metrics import c2st as _sbibm_c2st  # type: ignore

    SBIBM_AVAILABLE = True
except Exception:  # pragma: no cover
    _sbibm_c2st = None  # type: ignore
    SBIBM_AVAILABLE = False

try:  # pragma: no cover
    from sklearn.model_selection import KFold
    from sklearn.neural_network import MLPClassifier

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    KFold = None  # type: ignore
    MLPClassifier = None  # type: ignore
    SKLEARN_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# sbibm / sbibm-adjacent defaults (kept identical to the reference settings)
DEFAULT_N_FOLDS = 10
DEFAULT_SCORING = "accuracy"
DEFAULT_HIDDEN_LAYER_SIZES: Tuple[int, ...] = (50, 50)
DEFAULT_MAX_ITER = 1000
C2ST_IDEAL = 0.5
C2ST_WORST = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_numpy(values: Any) -> np.ndarray:
    """Convert torch tensors / lists / arrays to a 2-D float64 numpy array."""
    if values is None:
        raise ValueError("Cannot convert None to array.")
    if TORCH_AVAILABLE and isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(values, np.ndarray):
        array = np.asarray(values)
    elif isinstance(values, (list, tuple)):
        array = np.asarray([to_numpy(v) if hasattr(v, "__len__") else [v] for v in values])
    elif isinstance(values, (int, float, np.floating, np.integer)):
        array = np.asarray([[float(values)]])
    else:  # numpy arrays with __array__ (e.g. pandas)
        array = np.asarray(values)

    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        array = array.reshape(array.shape[0], -1)
    return array


def _as_2d_pair(
    samples_a: Any,
    samples_b: Any,
    min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and coerce two sample sets to matching 2-D arrays."""
    a = to_numpy(samples_a)
    b = to_numpy(samples_b)
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Sample dimensions differ: {a.shape[1]} vs {b.shape[1]}. "
            "Both sample sets must live in the same parameter space."
        )
    if a.shape[0] < min_samples or b.shape[0] < min_samples:
        raise ValueError(
            f"Need at least {min_samples} samples per set, got {a.shape[0]} and {b.shape[0]}."
        )
    return a, b


def _format_samples(values: Any) -> np.ndarray:
    """Accepts (samples, log_prob) tuples and returns just the samples."""
    if isinstance(values, (tuple, list)) and len(values) == 2 and not isinstance(values[0], (int, float)):
        candidate = values[0]
        if TORCH_AVAILABLE and isinstance(candidate, torch.Tensor):
            return to_numpy(candidate)
        if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
            return to_numpy(candidate)
    return to_numpy(values)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class C2STResult:
    """Result of a classifier two-sample test.

    Attributes:
        mean: averaged score across folds (0.5 = perfect, 1.0 = worst).
        std: standard deviation across folds.
        scores: per-fold scores.
        n_folds: number of cross-validation folds.
        scoring: metric used ('accuracy' by default).
        backend: 'sbibm', 'sklearn' or 'torch'.
        n_samples: number of samples per class.
        dim: parameter dimension.
    """

    mean: float
    std: float
    scores: List[float] = field(default_factory=list)
    n_folds: int = DEFAULT_N_FOLDS
    scoring: str = DEFAULT_SCORING
    backend: str = "sklearn"
    n_samples: int = 0
    dim: int = 0

    # -- convenience -------------------------------------------------------
    @property
    def accuracy(self) -> float:
        """Alias for the mean score (sbibm default scoring is accuracy)."""
        return self.mean

    @property
    def score(self) -> float:
        return self.mean

    def is_perfect(self, tol: float = 0.05) -> bool:
        return abs(self.mean - C2ST_IDEAL) <= tol

    def to_dict(self) -> Dict[str, Any]:
        return {
            "c2st": float(self.mean),
            "c2st_mean": float(self.mean),
            "c2st_std": float(self.std),
            "c2st_scores": [float(s) for s in self.scores],
            "n_folds": int(self.n_folds),
            "scoring": self.scoring,
            "backend": self.backend,
            "n_samples": int(self.n_samples),
            "dim": int(self.dim),
        }

    def __float__(self) -> float:  # allows float(result) / np.mean([...])
        return float(self.mean)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"C2STResult(mean={self.mean:.4f}, std={self.std:.4f}, "
            f"n_folds={self.n_folds}, backend='{self.backend}')"
        )


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------
def c2st(
    samples_a: Any,
    samples_b: Any,
    seed: Optional[int] = 0,
    n_folds: int = DEFAULT_N_FOLDS,
    scoring: str = DEFAULT_SCORING,
    z_score: bool = True,
    backend: str = "auto",
    classifier: Optional[Any] = None,
    return_result: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> Union[float, C2STResult]:
    """Classification-based two-sample test (Lopez-Paz & Oquab, 2017).

    Args:
        samples_a: ``(n, d)`` samples from the reference/target distribution.
        samples_b: ``(n, d)`` samples from the approximate distribution.
        seed: RNG seed for the classifier (reproducibility of the metric).
        n_folds: number of cross-validation folds (sbibm default: 10).
        scoring: ``'accuracy'`` (sbibm default) or ``'auc'``.
        z_score: standardise the pooled samples before classification.
        backend: ``'auto'`` (prefer sbibm), ``'sbibm'``, ``'sklearn'``, ``'torch'``.
        classifier: optional pre-instantiated sklearn-compatible classifier.
        return_result: return a :class:`C2STResult` instead of a float.
        verbose: print fold scores.

    Returns:
        The mean C2ST score (0.5 = perfect posterior estimation, 1.0 = worst)
        or, when ``return_result=True``, a :class:`C2STResult`.
    """
    a, b = _as_2d_pair(_format_samples(samples_a), _format_samples(samples_b))

    requested = (backend or "auto").lower()
    if requested == "auto":
        requested = "sbibm" if SBIBM_AVAILABLE else ("sklearn" if SKLEARN_AVAILABLE else "torch")

    result: Optional[C2STResult] = None

    if requested == "sbibm" and SBIBM_AVAILABLE:
        try:
            mean, scores = _sbibm_c2st_backend(
                a, b, seed=seed, n_folds=n_folds, scoring=scoring, z_score=z_score, verbose=verbose
            )
            result = C2STResult(
                mean=float(mean),
                std=float(np.std(scores)) if len(scores) else 0.0,
                scores=[float(s) for s in scores],
                n_folds=n_folds,
                scoring=scoring,
                backend="sbibm",
                n_samples=int(a.shape[0]),
                dim=int(a.shape[1]),
            )
        except Exception as exc:  # pragma: no cover - fall through to sklearn
            LOGGER.warning("sbibm c2st failed (%s); falling back to sklearn implementation.", exc)

    if result is None and requested in ("auto", "sklearn", "sbibm") and SKLEARN_AVAILABLE:
        mean, scores = _sklearn_c2st(
            a,
            b,
            seed=seed,
            n_folds=n_folds,
            scoring=scoring,
            z_score=z_score,
            classifier=classifier,
            verbose=verbose,
            **kwargs,
        )
        result = C2STResult(
            mean=float(mean),
            std=float(np.std(scores)) if len(scores) else 0.0,
            scores=[float(s) for s in scores],
            n_folds=n_folds,
            scoring=scoring,
            backend="sklearn",
            n_samples=int(a.shape[0]),
            dim=int(a.shape[1]),
        )

    if result is None and requested in ("auto", "torch") and TORCH_AVAILABLE:
        mean, scores = _torch_c2st(
            a, b, seed=seed, n_folds=n_folds, z_score=z_score, verbose=verbose
        )
        result = C2STResult(
            mean=float(mean),
            std=float(np.std(scores)) if len(scores) else 0.0,
            scores=[float(s) for s in scores],
            n_folds=n_folds,
            scoring=scoring,
            backend="torch",
            n_samples=int(a.shape[0]),
            dim=int(a.shape[1]),
        )

    if result is None:  # pragma: no cover - no backend at all
        raise RuntimeError(
            "C2ST requires either 'sbibm' or 'scikit-learn' (or torch). "
            "Install one of them, e.g. `pip install sbibm`."
        )

    return result if return_result else result.mean


def c2st_scores(
    samples_a: Any, samples_b: Any, return_result: bool = False, **kwargs: Any
) -> Union[np.ndarray, C2STResult]:
    """Return the per-fold C2ST scores (or the full result object)."""
    result = c2st(samples_a, samples_b, return_result=True, **kwargs)
    if return_result:
        return result
    return np.asarray(result.scores, dtype=np.float64)


# Alias: the sbibm default scoring is accuracy, so both names are equivalent.
c2st_accuracy = c2st


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _sbibm_c2st_backend(
    a: np.ndarray,
    b: np.ndarray,
    seed: Optional[int],
    n_folds: int,
    scoring: str,
    z_score: bool,
    verbose: bool,
) -> Tuple[float, List[float]]:
    """Call ``sbibm.metrics.c2st`` with sbibm's default settings."""
    assert _sbibm_c2st is not None
    sig = _call_signature(_sbibm_c2st)
    options: Dict[str, Any] = {}
    if "seed" in sig:
        options["seed"] = seed
    if "n_folds" in sig:
        options["n_folds"] = n_folds
    if "scoring" in sig:
        options["scoring"] = scoring
    if "z_score" in sig:
        options["z_score"] = z_score
    if "verbosity" in sig:
        options["verbosity"] = int(verbose)

    out = _sbibm_c2st(a, b, **options)

    scores: List[float]
    if isinstance(out, (tuple, list)) and len(out) == 2:
        # sbibm returns (mean, per-fold scores) in some versions
        first, second = out
        if np.isscalar(second):
            mean, scores = float(first), _repeat(float(second), n_folds)
        else:
            scores = [float(s) for s in np.asarray(second).ravel().tolist()]
            mean = float(first)
    elif isinstance(out, dict):
        scores = [float(s) for s in out.get("scores", [])]
        mean = float(out.get("mean", np.mean(scores) if scores else float("nan")))
    else:
        if np.isscalar(out):
            mean = float(out)
            scores = _repeat(mean, 1)
        else:
            scores = [float(s) for s in np.asarray(out).ravel().tolist()]
            mean = float(np.mean(scores)) if scores else float("nan")

    if verbose:
        LOGGER.info("sbibm C2ST folds: %s (mean %.4f)", np.round(scores, 4), mean)
    return float(mean), scores


def _repeat(value: float, n: int) -> List[float]:
    return [float(value)] * max(int(n), 1)


def _fallback_classifier(seed: Optional[int], **overrides: Any) -> Any:
    """Default classifier: sklearn MLP, matching sbibm-style defaults."""
    assert MLPClassifier is not None
    params: Dict[str, Any] = {
        "hidden_layer_sizes": DEFAULT_HIDDEN_LAYER_SIZES,
        "max_iter": DEFAULT_MAX_ITER,
        "random_state": seed,
    }
    # User-provided classifier hyper-parameters (e.g. hidden_layer_sizes)
    for key in ("hidden_layer_sizes", "max_iter", "alpha", "batch_size", "learning_rate_init"):
        if key in overrides and overrides[key] is not None:
            params[key] = overrides[key]
    return MLPClassifier(**params)


def _sklearn_c2st(
    a: np.ndarray,
    b: np.ndarray,
    seed: Optional[int] = None,
    n_folds: int = DEFAULT_N_FOLDS,
    scoring: str = DEFAULT_SCORING,
    z_score: bool = True,
    classifier: Optional[Any] = None,
    verbose: bool = False,
    **classifier_kwargs: Any,
) -> Tuple[float, List[float]]:
    """k-fold cross-validated C2ST with a scikit-learn classifier."""
    assert KFold is not None
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    x = np.concatenate([a, b], axis=0)
    # Class labels: 0 = reference samples, 1 = estimated samples
    y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

    if z_score:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        x = (x - mean) / std

    effective_folds = int(min(max(n_folds, 2), 2 * n))
    seed_int = 0 if seed is None else int(seed)
    kf = KFold(n_splits=effective_folds, shuffle=True, random_state=seed_int)

    scores: List[float] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
        clf = classifier if classifier is not None else _fallback_classifier(seed_int + fold, **classifier_kwargs)
        # Re-seed folds for reproducibility (mirrors sbibm's seeded folds)
        if hasattr(clf, "random_state"):
            try:
                clf.set_params(random_state=seed_int + fold)
            except Exception:
                pass
        clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
        if scoring == "auc" and hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(x[test_idx])[:, 1]
            scores.append(float(_auc(y[test_idx], proba)))
        else:
            scores.append(float(np.mean(pred == y[test_idx])))
        if verbose:
            LOGGER.info("fold %d/%d: %.4f", fold + 1, effective_folds, scores[-1])

    mean = float(np.mean(scores)) if scores else float("nan")
    return mean, scores


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (no sklearn dependency)."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = ranks[labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _torch_c2st(
    a: np.ndarray,
    b: np.ndarray,
    seed: Optional[int] = None,
    n_folds: int = DEFAULT_N_FOLDS,
    z_score: bool = True,
    verbose: bool = False,
    max_epochs: int = 200,
) -> Tuple[float, List[float]]:
    """Pure-torch fallback classifier two-sample test (used only if sklearn is absent)."""
    assert TORCH_AVAILABLE
    torch.manual_seed(0 if seed is None else int(seed))
    n = min(a.shape[0], b.shape[0])
    x = np.concatenate([a[:n], b[:n]], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(n)])
    if z_score:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        x = (x - mean) / std

    x_t = torch.as_tensor(x, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    dim = x_t.shape[1]
    effective_folds = int(min(max(n_folds, 2), 2 * n))

    # deterministic fold assignment
    perm = torch.randperm(x_t.shape[0], generator=torch.Generator().manual_seed(0 if seed is None else int(seed)))
    fold_ids = torch.arange(x_t.shape[0]) % effective_folds
    fold_ids = fold_ids[perm]

    scores: List[float] = []
    for fold in range(effective_folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        net = torch.nn.Sequential(
            torch.nn.Linear(dim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 1),
        )
        optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        for _ in range(max_epochs):
            optimiser.zero_grad()
            logits = net(x_t[train_mask]).squeeze(-1)
            loss = loss_fn(logits, y_t[train_mask])
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            logits = net(x_t[test_mask]).squeeze(-1)
            pred = (logits > 0).float()
            scores.append(float((pred == y_t[test_mask]).float().mean().item()))
        if verbose:
            LOGGER.info("fold %d/%d: %.4f", fold + 1, effective_folds, scores[-1])
    return float(np.mean(scores)), scores


# ---------------------------------------------------------------------------
# Task-level evaluation
# ---------------------------------------------------------------------------
def reference_posterior(
    task: Any,
    num_samples: int = 10_000,
    observation_index: Optional[int] = None,
    generator: Optional[Any] = None,
    seed: Optional[int] = None,
) -> Any:
    """Obtain reference posterior samples for a benchmark task.

    Works with:
      * this repository's :class:`snpse.tasks.benchmarks.BenchmarkTask`
        (``reference_posterior_samples``),
      * an ``sbibm`` task (``get_reference_posterior_samples``) with either
        ``num_observation`` or ``observation`` keyword,
      * a plain callable / already-materialised sample array.

    Returns:
        Reference posterior samples as a tensor/array of shape ``(num_samples, d)``.
    """
    if TORCH_AVAILABLE and isinstance(task, torch.Tensor):
        return task
    if isinstance(task, np.ndarray):
        return task
    if callable(task) and not hasattr(task, "reference_posterior_samples"):
        return task(num_samples)

    obs_index = observation_index
    if obs_index is None:
        obs_index = getattr(task, "observation_index", 1)

    # 1) repository BenchmarkTask
    for attr in ("reference_posterior_samples", "reference_samples", "get_reference_posterior_samples"):
        fn = getattr(task, attr, None)
        if fn is None:
            continue
        for kwargs in (
            {"num_samples": num_samples, "generator": generator},
            {"num_samples": num_samples},
            {"n": num_samples},
            {"num_samples": num_samples, "num_observation": obs_index},
            {"num_samples": num_samples, "observation": obs_index},
            {},
        ):
            try:
                out = fn(**kwargs)
                if out is not None:
                    return torch.as_tensor(out) if TORCH_AVAILABLE and not isinstance(out, np.ndarray) else out
            except TypeError:
                continue
            except Exception:  # pragma: no cover - sbibm needs data on disk
                continue

    # 2) sbibm task object
    sbibm_task = getattr(task, "sbibm_task", None)
    if sbibm_task is not None and hasattr(sbibm_task, "get_reference_posterior_samples"):
        for kwargs in (
            {"num_samples": num_samples, "num_observation": obs_index},
            {"num_samples": num_samples},
            {"num_observation": obs_index},
            {},
        ):
            try:
                return sbibm_task.get_reference_posterior_samples(**kwargs)
            except TypeError:
                continue
            except Exception:  # pragma: no cover
                continue

    if SKLEARN_AVAILABLE and hasattr(task, "prior") and hasattr(task, "simulate"):
        raise RuntimeError(
            "No reference posterior samples available for this task. Install `sbibm` "
            "and download the reference posteriors (`sbibm.get_task(...)`), or pass "
            "reference samples explicitly to `c2st`."
        )
    raise RuntimeError(f"Could not obtain reference posterior samples from object {type(task)!r}.")


def task_c2st(
    task: Any,
    posterior_samples: Any,
    num_reference_samples: Optional[int] = None,
    observation_index: Optional[int] = None,
    seed: Optional[int] = 0,
    n_folds: int = DEFAULT_N_FOLDS,
    scoring: str = DEFAULT_SCORING,
    generator: Optional[Any] = None,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, C2STResult]:
    """C2ST between posterior samples and a task's reference posterior.

    Args:
        task: ``BenchmarkTask``/``sbibm`` task (see :func:`reference_posterior`).
        posterior_samples: approximate posterior samples ``(n, d)``.
        num_reference_samples: defaults to ``posterior_samples.shape[0]``.
        observation_index: sbibm observation index (defaults to the task's).
        seed: metric seed (fixed seeds give a reproducible C2ST comparison).
        n_folds: number of cross-validation folds.
        scoring: ``'accuracy'`` (default) or ``'auc'``.
        return_result: return :class:`C2STResult` instead of a float.

    Returns:
        Mean C2ST score (0.5 = perfect).
    """
    samples = _format_samples(posterior_samples)
    n_samples = int(samples.shape[0])
    if num_reference_samples is None:
        num_reference_samples = n_samples
    reference = _format_samples(
        reference_posterior(
            task,
            num_samples=int(num_reference_samples),
            observation_index=observation_index,
            generator=generator,
            seed=seed,
        )
    )
    return c2st(
        reference,
        samples,
        seed=seed,
        n_folds=n_folds,
        scoring=scoring,
        return_result=return_result,
        **kwargs,
    )


def evaluate_posterior(
    task: Any,
    posterior_samples: Any,
    methods: Optional[Sequence[str]] = None,
    metrics: Optional[Sequence[str]] = None,
    seed: Optional[int] = 0,
    n_folds: int = DEFAULT_N_FOLDS,
    num_reference_samples: Optional[int] = None,
    observation_index: Optional[int] = None,
    generator: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compute the paper's evaluation metrics for a posterior approximation.

    Returns a dict with at least ``{"c2st": float, "c2st_std": float}`` so that
    experiment scripts can log/aggregate results uniformly.
    """
    metrics = list(metrics) if metrics else ["c2st"]
    out: Dict[str, Any] = {}

    result = task_c2st(
        task,
        posterior_samples,
        num_reference_samples=num_reference_samples,
        observation_index=observation_index,
        seed=seed,
        n_folds=n_folds,
        generator=generator,
        return_result=True,
    )
    out.update(result.to_dict())
    if "c2st" not in metrics:  # always keep the c2st fields as they are the paper metric
        out["c2st_extra"] = True
    if methods:
        out["methods"] = list(methods)
    return out


# ---------------------------------------------------------------------------
# Small helper for the experiments scripts
# ---------------------------------------------------------------------------
def summarise_c2st(values: Sequence[float]) -> Dict[str, float]:
    """Mean/std/min/max of a list of C2ST scores."""
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _call_signature(fn: Any) -> set:
    """Best-effort inspection of a callable's accepted keyword names."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return set()
    names = {p.name for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return names


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:  # pragma: no cover - manual diagnostic
    """Sanity checks: identical distributions -> ~0.5, separated -> ~1.0."""
    rng = np.random.default_rng(0)
    n, d = 2000, 2
    a = rng.normal(size=(n, d))
    b = rng.normal(size=(n, d))
    same = c2st(a, b, seed=0, n_folds=5, return_result=True)
    shift = c2st(a, b + 3.0, seed=0, n_folds=5, return_result=True)
    print(f"[c2st] identical  samples: {same}")
    print(f"[c2st] separated  samples: {shift}")
    assert 0.4 <= same.mean <= 0.6, f"expected ~0.5 for identical distributions, got {same.mean}"
    assert shift.mean >= 0.95, f"expected ~1.0 for separated distributions, got {shift.mean}"
    if TORCH_AVAILABLE:
        res_t = c2st(a, b, seed=0, n_folds=3, backend="torch", return_result=True)
        print(f"[c2st] torch backend check: {res_t}")
    print("[c2st] OK (backend:", same.backend, ")")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest()
