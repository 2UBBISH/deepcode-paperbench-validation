"""Unbiased pass@k estimator for code-generation evaluation (HumanEval).

The paper (Section 3.3.1, footnote 4) follows the definition of Chen et al. (2021),
"Evaluating Large Language Models Trained on Code":

    "k code samples are generated per problem, a problem is considered solved if any
     sample passes the unit tests, and the total fraction of problems solved is reported."

Because a naive ``1 - (1 - p)^k`` estimate from a fixed budget of ``n`` samples is
biased high for small ``n``, Chen et al. (2021) use the unbiased estimator

    pass@k = E_problems[ 1 - C(n - c, k) / C(n, k) ]                          (1)

where ``n`` samples are generated per problem, ``c`` of them are correct (pass the
unit tests), and ``C`` is the binomial coefficient.  The expectation is over problems.

Numerical note
--------------
``C(n - c, k) / C(n, k)`` is computed as a product of ratios in log-space
(or as an exact product of ``(n - c - i) / (n - i)``) to avoid overflow of the huge
binomial coefficients for ``n = 200`` and ``k = 100``.  With ``n - c < k`` the
numerator is 0 by convention and pass@k is exactly 1.0 (some correct sample must be
within the first ``k`` draws).

The module is dependency-light (numpy only, ``scipy``/``pandas`` optional) so that it
can be unit-tested without torch, transformers, or a code-execution sandbox.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "PASS_AT_K_VALUES",
    "HUMANEVAL_N_SAMPLES",
    "HUMANEVAL_N_PROBLEMS",
    "estimate_pass_at_k",
    "pass_at_k",
    "compute_pass_at_k",
    "pass_at_k_from_counts",
    "pass_at_k_from_matrix",
    "aggregate_pass_at_k",
    "pass_at_k_table",
    "per_problem_pass_at_k",
    "count_correct",
    "n_samples_needed",
    "make_seeds",
    "sanitize_counts",
]

#: k values reported throughout the paper (Table 2 / Tables 7, 8, 9).
PASS_AT_K_VALUES: Tuple[int, ...] = (1, 10, 100)

#: HumanEval contains 164 Python coding tasks (Chen et al. 2021).
HUMANEVAL_N_PROBLEMS: int = 164

#: Number of samples per problem used in the CodeGen HumanEval evaluations.  The paper
#: evaluates pass@k for k in {1, 10, 100}; the Chen et al. estimator requires n >= k and
#: is conventionally run with n = 200 (and at least n >= 100 here).
HUMANEVAL_N_SAMPLES: int = 200

#: Guidance strengths swept over HumanEval (footnote 3): gamma = 1.0 ... 2.0.
CFG_GAMMAS: Tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)

#: Temperatures swept over HumanEval (Section 3.3.1 / Figure 11).
HUMANEVAL_TEMPERATURES: Tuple[float, ...] = (0.2, 0.6, 0.8)


# --------------------------------------------------------------------------------------
# Core estimator
# --------------------------------------------------------------------------------------
def _validate(n: int, c: int) -> None:
    if n < 0:
        raise ValueError(f"num_samples must be non-negative, got {n}")
    if c < 0:
        raise ValueError(f"num_correct must be non-negative, got {c}")
    if c > n:
        raise ValueError(f"num_correct ({c}) cannot exceed num_samples ({n})")


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Unbiased pass@k for a *single* problem (Chen et al. 2021, Eq. 1).

    ``pass@k = 1 - C(n - c, k) / C(n, k)`` where ``n = num_samples`` and
    ``c = num_correct`` samples out of ``n`` pass the unit tests.

    Edge cases:

    * ``n - c < k``  -> ``1.0`` (at least one correct sample is guaranteed within k).
    * ``c == 0``     -> ``0.0``.
    * ``c == n``     -> ``1.0``.
    * ``k > n``      -> ``1.0`` if ``c > 0`` else ``0.0`` (cannot estimate honestly, but
      the usual convention clamps to what is observable).

    Parameters
    ----------
    num_samples:
        Number of generated samples ``n`` for this problem.
    num_correct:
        Number of those samples that passed the unit tests ``c``.
    k:
        Number of samples we are allowed to draw from (``k <= n`` normally).

    Returns
    -------
    float
        Estimated probability that at least one of ``k`` samples is correct.
    """
    n = int(num_samples)
    c = int(num_correct)
    k = int(k)
    _validate(n, c)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n == 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        # C(n - c, k) == 0 -> the whole probability mass is covered.
        return 1.0
    if c == n:
        return 1.0
    # 1 - prod_{i=0}^{k-1} (n - c - i) / (n - i)
    # Computed as a product of ratios so no large binomial coefficient is materialised.
    ratio = 1.0
    for i in range(k):
        ratio *= (n - c - i) / (n - i)
    # Guard tiny negative rounding artefacts.
    return float(min(1.0, max(0.0, 1.0 - ratio)))


def pass_at_k(
    num_samples: Union[int, Sequence[int], np.ndarray],
    num_correct: Union[int, Sequence[int], np.ndarray],
    k: Union[int, Sequence[int], np.ndarray],
) -> np.ndarray:
    """Vectorised unbiased pass@k over problems and/or k values.

    The Chen et al. (2021) estimator is

        pass@k = 1 - C(n - c, k) / C(n, k)

    evaluated per problem and then averaged (arithmetic mean) over problems.

    Parameters
    ----------
    num_samples:
        ``n`` per problem.  Either a scalar (same budget for all problems) or an array
        of shape ``[n_problems]``.
    num_correct:
        ``c`` per problem, shape ``[n_problems]`` (or scalar).
    k:
        A single ``k`` or a sequence of k values, e.g. ``(1, 10, 100)``.

    Returns
    -------
    np.ndarray
        Shape ``[n_problems]`` when ``k`` is scalar, else ``[len(k), n_problems]``.
    """
    n_arr = np.atleast_1d(np.asarray(num_samples, dtype=np.int64)).astype(np.float64)
    c_arr = np.atleast_1d(np.asarray(num_correct, dtype=np.int64)).astype(np.float64)
    if n_arr.size == 1 and c_arr.size > 1:
        n_arr = np.full_like(c_arr, n_arr.item())
    if c_arr.size == 1 and n_arr.size > 1:
        c_arr = np.full_like(n_arr, c_arr.item())
    if n_arr.shape != c_arr.shape:
        raise ValueError(
            f"num_samples shape {n_arr.shape} != num_correct shape {c_arr.shape}"
        )

    if np.any(c_arr > n_arr):
        raise ValueError("num_correct cannot exceed num_samples")
    if np.any(c_arr < 0):
        raise ValueError("num_correct must be non-negative")

    k_values = np.atleast_1d(np.asarray(k, dtype=np.int64))
    if np.any(k_values <= 0):
        raise ValueError("all k must be positive")

    results = np.zeros((len(k_values), len(n_arr)), dtype=np.float64)
    for i, kk in enumerate(k_values):
        kk = int(kk)
        n, c = n_arr, c_arr
        # NumPy broadcasting: a single formula handles both uniform and per-problem n.
        num = np.ones_like(n)
        den = np.ones_like(n)
        for j in range(kk):
            num = num * (n - c - j)
            den = den * (n - j)
        ratio = np.where((n - c) < kk, 0.0, num / np.where(den == 0.0, 1.0, den))
        ratio = np.clip(ratio, 0.0, 1.0)
        results[i] = 1.0 - ratio
        # Force exact values where the closed form is defined by convention.
        results[i] = np.where(c <= 0, 0.0, results[i])
        results[i] = np.where(n - c < kk, 1.0, results[i])
        results[i] = np.where(c >= n, 1.0, results[i])

    if np.isscalar(k) or (np.asarray(k).ndim == 0):
        return results[0]
    return results


def compute_pass_at_k(
    num_samples: Union[int, Sequence[int], np.ndarray],
    num_correct: Union[int, Sequence[int], np.ndarray],
    k: Union[int, Sequence[int], np.ndarray] = PASS_AT_K_VALUES,
) -> Dict[int, float]:
    """Mean unbiased pass@k over problems, returned as a ``{k: value}`` dict.

    This is the headline number reported per (model, temperature, gamma) cell of
    Table 2, and Tables 7/8/9 of Appendix C.4.
    """
    k_values = [int(kk) for kk in np.atleast_1d(np.asarray(k, dtype=np.int64))]
    matrix = np.atleast_2d(pass_at_k(num_samples, num_correct, k_values))
    if matrix.shape[0] != len(k_values):
        matrix = matrix.reshape(len(k_values), -1)
    return {kk: float(np.mean(matrix[i])) for i, kk in enumerate(k_values)}


# --------------------------------------------------------------------------------------
# Results-matrix utilities
# --------------------------------------------------------------------------------------
def sanitize_counts(
    num_samples: Union[int, Sequence[int], np.ndarray],
    num_correct: Union[int, Sequence[int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce ``(n, c)`` to equal-length int arrays and validate ``0 <= c <= n``."""
    n_arr = np.atleast_1d(np.asarray(num_samples, dtype=np.int64)).astype(np.float64)
    c_arr = np.atleast_1d(np.asarray(num_correct, dtype=np.int64)).astype(np.float64)
    if n_arr.size == 1 and c_arr.size > 1:
        n_arr = np.full_like(c_arr, n_arr.item())
    if c_arr.size == 1 and n_arr.size > 1:
        c_arr = np.full_like(n_arr, c_arr.item())
    if n_arr.shape != c_arr.shape:
        raise ValueError("num_samples and num_correct must have the same length")
    if np.any(c_arr > n_arr):
        raise ValueError("num_correct cannot exceed num_samples")
    return n_arr, c_arr


def pass_at_k_from_counts(
    counts: Dict[Any, Tuple[int, int]],
    k: Union[int, Sequence[int]] = PASS_AT_K_VALUES,
    num_samples: Optional[int] = None,
) -> Dict[int, float]:
    """Compute pass@k from a mapping ``problem_id -> (n_samples, n_correct)``.

    Problems with ``n_samples == 0`` are skipped with a warning.
    """
    k_values = [int(kk) for kk in np.atleast_1d(np.asarray(k, dtype=np.int64))]
    n_list: List[int] = []
    c_list: List[int] = []
    for pid, (n, c) in counts.items():
        n_i = int(n if num_samples is None else min(int(n), int(num_samples)))
        c_i = int(c)
        if n_i <= 0:
            logger.warning("Skipping problem %s with zero samples", pid)
            continue
        if c_i > n_i:
            logger.warning(
                "Problem %s reports more correct (%d) than samples (%d); clamping",
                pid,
                c_i,
                n_i,
            )
            c_i = n_i
        if c_i < 0:
            c_i = 0
        n_list.append(n_i)
        c_list.append(c_i)
    if not n_list:
        return {kk: 0.0 for kk in k_values}
    return compute_pass_at_k(np.asarray(n_list), np.asarray(c_list), k_values)


def pass_at_k_from_matrix(
    correctness: np.ndarray,
    k: Union[int, Sequence[int]] = PASS_AT_K_VALUES,
    problem_ids: Optional[Sequence[Any]] = None,
) -> Dict[int, float]:
    """Compute pass@k from a boolean ``[n_problems, n_samples]`` correctness matrix."""
    mat = np.asarray(correctness)
    if mat.ndim != 2:
        raise ValueError(f"correctness matrix must be 2-D, got shape {mat.shape}")
    n_arr = np.full(mat.shape[0], mat.shape[1], dtype=np.float64)
    c_arr = mat.astype(np.float64).sum(axis=1)
    k_values = [int(kk) for kk in np.atleast_1d(np.asarray(k, dtype=np.int64))]
    matrix = np.atleast_2d(pass_at_k(n_arr, c_arr, k_values))
    if matrix.shape[0] != len(k_values):
        matrix = matrix.reshape(len(k_values), -1)
    return {kk: float(np.mean(matrix[i])) for i, kk in enumerate(k_values)}


def per_problem_pass_at_k(
    num_samples: Union[int, Sequence[int], np.ndarray],
    num_correct: Union[int, Sequence[int], np.ndarray],
    k: int,
    problem_ids: Optional[Sequence[Any]] = None,
) -> Dict[Any, float]:
    """Per-problem pass@k values (used for the Figure 11/12/13 scatter plots)."""
    n_arr, c_arr = sanitize_counts(num_samples, num_correct)
    values = np.atleast_1d(pass_at_k(n_arr, c_arr, int(k)))
    if problem_ids is None:
        problem_ids = list(range(len(n_arr)))
    return {pid: float(v) for pid, v in zip(problem_ids, values)}


def count_correct(results: Iterable[Union[bool, int, float, str]]) -> int:
    """Count how many samples in ``results`` count as passing.

    Accepts booleans, numeric results (``> 0`` means passed), and strings from an
    execution harness (``"passed"``/``"failed"``/``"timed out"``...).
    """
    n_ok = 0
    for r in results:
        if isinstance(r, str):
            if r.strip().lower() in {"passed", "pass", "ok", "success", "true", "1"}:
                n_ok += 1
        elif isinstance(r, bool):
            n_ok += int(r)
        elif isinstance(r, (int, float, np.integer, np.floating)):
            n_ok += int(r > 0)
        else:
            n_ok += int(bool(r))
    return n_ok


def aggregate_pass_at_k(
    per_problem_results: Dict[Any, Iterable[Any]],
    k: Union[int, Sequence[int]] = PASS_AT_K_VALUES,
) -> Dict[int, float]:
    """Convenience: ``problem_id -> [sample results]`` -> mean pass@k."""
    counts = {
        pid: (len(list(res)), count_correct(res))
        for pid, res in per_problem_results.items()
    }
    return pass_at_k_from_counts(counts, k)


# --------------------------------------------------------------------------------------
# Sweep support (Table 2 / Tables 7-9)
# --------------------------------------------------------------------------------------
def pass_at_k_table(
    results: Dict[float, float],
    num_samples: Union[int, Sequence[int], np.ndarray],
    num_correct: Union[int, Sequence[int], np.ndarray],
) -> Dict[Tuple[Any, Any], Dict[int, float]]:
    """Build a ``{(temperature, gamma): {k: pass@k}}`` table.

    Parameters
    ----------
    results:
        Mapping ``(temperature, gamma) -> per-problem sample results`` where each value
        is either a boolean matrix ``[n_problems, n_samples]`` or a sequence of
        per-problem correctness counts.
    num_samples, num_correct:
        Optional overrides.  When ``results`` holds boolean matrices these are ignored.

    Returns
    -------
    Dict
        Nested ``{(T, gamma): {k: value}}`` dict, plus the flattened ``"(T, gamma)"``
        keys used by the reporting script.
    """
    table: Dict[Tuple[Any, Any], Dict[int, float]] = {}
    for key, value in results.items():
        if isinstance(key, (tuple, list)) and len(key) == 2:
            temp, gamma = key
        else:
            temp, gamma = None, key
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.dtype == bool:
            table[(temp, gamma)] = pass_at_k_from_matrix(arr)
        else:
            flat = np.asarray(list(value), dtype=np.float64)
            if flat.ndim == 0:
                flat = flat.reshape(1)
            table[(temp, gamma)] = compute_pass_at_k(
                np.asarray(num_samples), np.asarray(flat), PASS_AT_K_VALUES
            )
    return table


def n_samples_needed(k: int) -> int:
    """Minimum ``n`` for an unbiased pass@k estimate: ``n >= k``."""
    return int(k)


def make_seeds(base_seed: int, n_problems: int, n_samples: int) -> np.ndarray:
    """Deterministic per-(problem, sample) seeds for reproducible pass@k runs.

    Generating ``n`` samples per problem with an explicit seed per sample makes the
    estimator insensitive to batching/scheduling order (footnote 4 reproducibility).
    """
    rng = np.random.RandomState(int(base_seed))
    return rng.randint(0, 2 ** 31 - 1, size=(int(n_problems), int(n_samples)), dtype=np.int64)


def binomial(n: int, k: int) -> int:
    """Exact binomial coefficient (helper for tests / small-n reference values)."""
    return math.comb(int(n), int(k))
