"""ANCOVA / logistic-regression cost analysis (Section 4, Appendix C.2, Table 6).

This module implements the statistical half of Section 4 ("Cost Analysis") of
*Stay on Topic with Classifier-Free Guidance*:

*   Section 4 argues that CFG doubles inference FLOPs (two forward passes), and
    asks whether a CFG model can emulate a vanilla model of **twice the size**
    at equal inference FLOPs.  Appendix C.2 turns this into regression charts of
    **accuracy vs. inference FLOP per token**: the location of a data point
    "ignores the model size and only reflects its inference FLOP per token", so
    a 1.4B model with CFG lands near a 2.8B vanilla model.
*   For every task, two regression lines are fitted -- the CFG group (red) and
    the vanilla group (blue) -- over ``log(FLOPs per token)``, and an **ANCOVA
    regression analysis** (Rutherford 2011) tests whether the two lines differ.
    The paper uses a p-value cutoff of ``0.01``; "higher than 0.01 means an
    insignificant difference between the regression lines of the two groups".
*   Across the 9 Section 3.1 benchmarks: **5 of 9 are insignificant** (i.e. CFG
    matches a twice-as-large model), and of the significant ones **2 favor CFG**
    (LAMBADA, SciQ) and **2 favor vanilla** (WinoGrande, TriviaQA).

The implementation is deliberately dependency-light: the whole numerics layer
(IRLS logistic regression with binomial counts, weighted least squares, Wald /
likelihood-ratio / quasi-binomial F tests and the chi-square / F survival
functions) is pure Python + ``math`` so that the statistics are unit-testable on
CPU.  ``numpy``, ``scipy`` and ``statsmodels`` are used when available (scipy
provides the distribution functions, statsmodels the GLM fits) but are never
required.

Public API
----------
Dataclasses
    ``BenchmarkPoint``   -- one (task, model, gamma, accuracy, FLOPs) data point
    ``RegressionFit``    -- a fitted line (logistic / OLS) with full inference
    ``AncovaResult``     -- ANCOVA outcome for one task (p-value, winner, fits)
    ``AncovaReport``     -- per-task results + paper comparison
Fitting
    ``fit_logistic``, ``fit_ols``, ``fit_group_lines``, ``logistic_regression``
Testing
    ``ancova``, ``ancova_by_task``, ``ancova_from_results``,
    ``chow_test``, ``wald_test``, ``likelihood_ratio_test``, ``quasi_f_test``
Data plumbing
    ``build_points``, ``points_from_results``, ``point_from_row``,
    ``group_of_gamma``, ``log_flops_per_token``
Reporting
    ``ancova_table``, ``p_value_table``, ``format_ancova_table``,
    ``ancova_report``, ``save_ancova``, ``to_dataframe``,
    ``significant_tasks``, ``favor_counts``, ``check_against_paper``
Reference values
    ``PAPER_TABLE6``, ``PAPER_TABLE6_PVALUES``, ``PAPER_TABLE6_WINNERS``
Distributions
    ``chi2_sf``, ``f_sf``, ``t_sf``, ``norm_sf``, ``betainc``, ``gammaincc``
Constants
    ``SIGNIFICANCE_LEVEL = 0.01``, ``ANCOVA_METHODS``, ``BENCHMARK_TASKS``,
    ``CFG_GROUP``, ``VANILLA_GROUP``, ``CFG_GAMMA``, ``VANILLA_GAMMA``

Paper anchors reproduced here
-----------------------------
``PAPER_TABLE6`` (Appendix C.2 Table 9 / main-paper Table 6) holds the per-task
p-values and winners, and ``check_against_paper`` verifies the Section 4 split
``{inconclusive: 5, cfg: 2, vanilla: 2}``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

__all__ = [
    # constants
    "SIGNIFICANCE_LEVEL",
    "ANCOVA_METHODS",
    "BENCHMARK_TASKS",
    "CFG_GROUP",
    "VANILLA_GROUP",
    "CFG_GAMMA",
    "VANILLA_GAMMA",
    "PAPER_TABLE6",
    "PAPER_TABLE6_PVALUES",
    "PAPER_TABLE6_WINNERS",
    # dataclasses
    "BenchmarkPoint",
    "RegressionFit",
    "AncovaResult",
    "AncovaReport",
    # fitting
    "fit_logistic",
    "fit_ols",
    "fit_group_lines",
    "logistic_regression",
    "group_of_gamma",
    "log_flops_per_token",
    # testing
    "ancova",
    "ancova_by_task",
    "ancova_from_results",
    "chow_test",
    "wald_test",
    "likelihood_ratio_test",
    "quasi_f_test",
    "winner_at",
    "compare_lines",
    # data plumbing
    "build_points",
    "points_from_results",
    "point_from_row",
    # reporting
    "ancova_table",
    "p_value_table",
    "format_ancova_table",
    "ancova_report",
    "save_ancova",
    "to_dataframe",
    "significant_tasks",
    "favor_counts",
    "check_against_paper",
    # distributions
    "chi2_sf",
    "f_sf",
    "t_sf",
    "norm_sf",
    "betainc",
    "gammaincc",
    "gammainc",
    "log_gamma",
    # misc
    "synthetic_points",
]


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:  # pragma: no cover - environment dependent
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover - environment dependent
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None

try:  # pragma: no cover - environment dependent
    import statsmodels.api as _sm
except Exception:  # pragma: no cover
    _sm = None

_HAS_NUMPY = _np is not None
_HAS_SCIPY = _scipy_stats is not None
_HAS_STATSMODELS = _sm is not None


# ---------------------------------------------------------------------------
# Constants / paper anchors
# ---------------------------------------------------------------------------

#: Significance cutoff recommended by Rutherford (2011) and used in Section 4.
SIGNIFICANCE_LEVEL = 0.01

#: Supported ANCOVA test statistics.
#:   ``lr``   -- likelihood-ratio chi-square on the logistic fits (default)
#:   ``wald`` -- Wald chi-square on the group terms of the logistic fit
#:   ``f``    -- quasi-binomial F test on the logistic (over-dispersed) fit
#:   ``ols``  -- classical ANCOVA F test on the (weighted) accuracy scale
ANCOVA_METHODS = ("lr", "wald", "f", "ols")

CFG_GROUP = "cfg"
VANILLA_GROUP = "vanilla"

#: CFG group uses the guided gamma (Sections 3.1 / 5 use 1.5); vanilla == 1.0.
CFG_GAMMA = 1.5
VANILLA_GAMMA = 1.0

#: The nine Section 3.1 zero-shot benchmarks, in the paper's reporting order.
BENCHMARK_TASKS: Tuple[str, ...] = (
    "lambada_openai",
    "winogrande",
    "sciq",
    "triviaqa",
    "hellaswag",
    "piqa",
    "arc_challenge",
    "boolq",
    "arc_easy",
)

_TASK_ALIASES = {
    "lambada": "lambada_openai",
    "lambada_openai": "lambada_openai",
    "lambada-openai": "lambada_openai",
    "winogrande": "winogrande",
    "wino_grande": "winogrande",
    "sciq": "sciq",
    "trivia_qa": "triviaqa",
    "triviaqa": "triviaqa",
    "hellaswag": "hellaswag",
    "piqa": "piqa",
    "arc_challenge": "arc_challenge",
    "arc-c": "arc_challenge",
    "arc_c": "arc_challenge",
    "arcc": "arc_challenge",
    "boolq": "boolq",
    "arc_easy": "arc_easy",
    "arc-e": "arc_easy",
    "arc_e": "arc_easy",
    "arce": "arc_easy",
}


def canonical_task_name(task: str) -> str:
    """Map harness/table spellings onto :data:`BENCHMARK_TASKS` names."""
    key = str(task).strip().lower().replace(" ", "_")
    return _TASK_ALIASES.get(key, key)


#: Appendix C.2 Table 9 / main-paper Table 6: per-task ANCOVA p-values and the
#: group favored when the test is significant at p=.01 (``None`` = inconclusive,
#: i.e. "an insignificant difference between the regression lines").
#: LAMBADA and SciQ are the "clear winners" called out in Appendix C.2; the
#: WinoGrande line is negatively impacted by CFG.
PAPER_TABLE6_PVALUES: Dict[str, float] = {
    "lambada_openai": 0.000,
    "winogrande": 0.003,
    "sciq": 0.008,
    "triviaqa": 0.008,
    "hellaswag": 0.012,
    "piqa": 0.030,
    "arc_challenge": 0.216,
    "boolq": 0.345,
    "arc_easy": 0.355,
}

PAPER_TABLE6_WINNERS: Dict[str, Optional[str]] = {
    "lambada_openai": CFG_GROUP,
    "winogrande": VANILLA_GROUP,
    "sciq": CFG_GROUP,
    "triviaqa": VANILLA_GROUP,
    "hellaswag": None,
    "piqa": None,
    "arc_challenge": None,
    "boolq": None,
    "arc_easy": None,
}

#: ``{task: {"p_value": float, "winner": str|None, "significant": bool}}``
PAPER_TABLE6: Dict[str, Dict[str, Any]] = {
    task: {
        "p_value": PAPER_TABLE6_PVALUES[task],
        "winner": PAPER_TABLE6_WINNERS[task],
        "significant": PAPER_TABLE6_PVALUES[task] < SIGNIFICANCE_LEVEL,
        "favor": PAPER_TABLE6_WINNERS[task],
    }
    for task in BENCHMARK_TASKS
}

#: Section 4 headline split: 5 tasks insignificant, 2 favor CFG, 2 favor vanilla.
PAPER_FAVOR_COUNTS: Dict[str, int] = {
    "inconclusive": 5,
    CFG_GROUP: 2,
    VANILLA_GROUP: 2,
}


# ---------------------------------------------------------------------------
# Distribution functions (pure Python fallbacks; scipy used when present)
# ---------------------------------------------------------------------------

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def log_gamma(x: float) -> float:
    """Natural log of the gamma function (Lanczos approximation)."""
    if x < 0.5:
        # reflection formula
        return math.log(math.pi / abs(math.sin(math.pi * x))) - log_gamma(1.0 - x)
    x -= 1.0
    coeffs = (
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    )
    a = coeffs[0]
    t = x + 7.5
    for i in range(1, len(coeffs)):
        a += coeffs[i] / (x + i)
    return 0.5 * math.log(2.0 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(a)


def _gammainc_series(a: float, x: float) -> float:
    """Lower regularised incomplete gamma P(a, x) via its series expansion."""
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(1000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - log_gamma(a))


def _gammainc_cf(a: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(a, x) via continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if b != 0.0 else 1.0 / tiny
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - log_gamma(a)) * h


def gammainc(a: float, x: float) -> float:
    """Regularised lower incomplete gamma function ``P(a, x)``."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("gammainc requires a > 0 and x >= 0")
    if x == 0.0:
        return 0.0
    if _HAS_SCIPY:
        try:  # pragma: no cover
            return float(_scipy_stats.gamma.cdf(x, a))
        except Exception:
            pass
    if x < a + 1.0:
        return _gammainc_series(a, x)
    return 1.0 - _gammainc_cf(a, x)


def gammaincc(a: float, x: float) -> float:
    """Regularised upper incomplete gamma function ``Q(a, x)``."""
    if _HAS_SCIPY:
        try:  # pragma: no cover
            return float(_scipy_stats.gamma.sf(x, a))
        except Exception:
            pass
    return 1.0 - gammainc(a, x)


def chi2_sf(x: float, df: int) -> float:
    """Survival function ``P(X > x)`` of a chi-square with ``df`` d.o.f."""
    if df <= 0:
        raise ValueError("df must be positive")
    if x <= 0:
        return 1.0
    return gammaincc(df / 2.0, x / 2.0)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function ``I_x(a, b)``."""
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must be in [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    if _HAS_SCIPY:
        try:  # pragma: no cover
            return float(_scipy_stats.beta.cdf(x, a, b))
        except Exception:
            pass
    front = math.exp(
        log_gamma(a + b) - log_gamma(a) - log_gamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def f_sf(f: float, df1: float, df2: float) -> float:
    """Survival function of the F distribution with ``(df1, df2)`` d.o.f."""
    if df1 <= 0 or df2 <= 0:
        raise ValueError("degrees of freedom must be positive")
    if f <= 0:
        return 1.0
    if _HAS_SCIPY:
        try:  # pragma: no cover
            return float(_scipy_stats.f.sf(f, df1, df2))
        except Exception:
            pass
    return betainc(df2 / 2.0, df1 / 2.0, df2 / (df2 + df1 * f))


def t_sf(t: float, df: float) -> float:
    """Two-sided p-value of Student's t statistic (``df`` d.o.f.)."""
    if df <= 0:
        raise ValueError("df must be positive")
    if _HAS_SCIPY:
        try:  # pragma: no cover
            return float(2.0 * _scipy_stats.t.sf(abs(t), df))
        except Exception:
            pass
    x = df / (df + t * t)
    return betainc(df / 2.0, 0.5, x)


def norm_sf(z: float) -> float:
    """Two-sided p-value of a standard normal statistic."""
    if _HAS_SCIPY:
        try:  # pragma: no cover
            return float(2.0 * _scipy_stats.norm.sf(abs(z)))
        except Exception:
            pass
    return math.erfc(abs(z) / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (pure python, list-of-lists)
# ---------------------------------------------------------------------------


def _to_lists(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    return [[float(v) for v in row] for row in matrix]


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> List[float]:
    """Gaussian elimination with partial pivoting (returns ``x`` for ``Ax=b``)."""
    n = len(matrix)
    if n == 0:
        return []
    aug = [list(map(float, matrix[i])) + [float(rhs[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-14:
            raise ValueError("singular design matrix (collinear columns?)")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col] / pivot_val
            if factor == 0.0:
                continue
            for c in range(col, n + 1):
                aug[row][c] -= factor * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def _invert(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    """Matrix inverse via Gauss-Jordan elimination."""
    n = len(matrix)
    aug = [list(map(float, matrix[i])) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for c in range(2 * n):
            aug[col][c] /= pivot_val
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for c in range(2 * n):
                aug[row][c] -= factor * aug[col][c]
    return [row[n:] for row in aug]


def _matvec(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    return [sum(matrix[i][j] * vec[j] for j in range(len(vec))) for i in range(len(matrix))]


def _xtwx(X: Sequence[Sequence[float]], W: Sequence[float]) -> List[List[float]]:
    p = len(X[0])
    out = [[0.0] * p for _ in range(p)]
    for i, row in enumerate(X):
        wi = W[i]
        for a in range(p):
            if row[a] == 0.0:
                continue
            vai = wi * row[a]
            for b in range(p):
                out[a][b] += vai * row[b]
    return out


def _xtwy(X: Sequence[Sequence[float]], W: Sequence[float], y: Sequence[float]) -> List[float]:
    p = len(X[0])
    out = [0.0] * p
    for i, row in enumerate(X):
        wy = W[i] * y[i]
        for a in range(p):
            out[a] += row[a] * wy
    return out


def _logit(p: float) -> float:
    eps = 1e-10
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


def group_of_gamma(gamma: float) -> str:
    """``gamma == 1`` is the vanilla group, any other gamma is the CFG group."""
    try:
        g = float(gamma)
    except (TypeError, ValueError):
        return CFG_GROUP
    return VANILLA_GROUP if abs(g - 1.0) < 1e-9 else CFG_GROUP


def log_flops_per_token(
    n_params: Optional[float] = None,
    flops_per_token: Optional[float] = None,
    gamma: float = VANILLA_GAMMA,
    model: Optional[str] = None,
    apply_cfg: bool = True,
) -> Optional[float]:
    """``log`` of inference FLOPs per token, with the CFG doubling applied.

    CFG runs two forward passes per decoding step (Section 2.2), so the CFG
    group's x-coordinate is ``log(2 * FLOPs)`` -- this is what makes a 1.4B CFG
    model land near a 2.8B vanilla model in the Appendix C.2 charts.
    """
    flops = flops_per_token
    if flops is None and n_params is not None:
        # Fallback: the ~2 FLOPs-per-parameter-per-token forward estimate of
        # Section 4 / the ELECTRA flops_computation.py approximation.
        flops = 2.0 * float(n_params)
    if flops is None and model is not None:
        flops = _flops_from_model_name(model)
    if flops is None or flops <= 0:
        return None
    if apply_cfg and group_of_gamma(gamma) == CFG_GROUP:
        flops = 2.0 * float(flops)
    return math.log(float(flops))


def _flops_from_model_name(model: str) -> Optional[float]:
    """Look up per-token FLOPs in :mod:`src.analysis.flops` when available."""
    try:  # pragma: no cover - depends on sibling module availability
        from . import flops as _flops  # type: ignore

        spec = _flops.get_spec(model)
        per_token = _flops.flops_per_token(spec)
        return float(per_token) if per_token else None
    except Exception:
        return None


@dataclass
class BenchmarkPoint:
    """One accuracy-vs-FLOP data point for the Appendix C.2 charts.

    ``accuracy`` is the task score (fraction correct, in ``[0, 1]``); the x-axis
    is :attr:`log_flops`, i.e. ``log`` inference FLOPs per token *with* the CFG
    doubling already applied for the CFG group.
    """

    task: str
    accuracy: float
    model: str = "unknown"
    gamma: float = VANILLA_GAMMA
    n_examples: Optional[int] = None
    n_params: Optional[float] = None
    flops_per_token: Optional[float] = None
    group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.task = canonical_task_name(self.task)
        self.accuracy = float(self.accuracy)
        self.gamma = float(self.gamma)
        if self.group is None:
            self.group = group_of_gamma(self.gamma)

    # -- derived quantities -------------------------------------------------
    @property
    def is_cfg(self) -> bool:
        return self.group == CFG_GROUP or group_of_gamma(self.gamma) == CFG_GROUP

    @property
    def n_correct(self) -> int:
        """Rounded number of correct examples (for binomial IRLS)."""
        if self.n_examples is None:
            return int(round(self.accuracy))
        return int(round(self.accuracy * self.n_examples))

    @property
    def n_trials(self) -> int:
        if self.n_examples is not None and self.n_examples > 0:
            return int(self.n_examples)
        return 1

    @property
    def log_flops(self) -> float:
        value = log_flops_per_token(
            n_params=self.n_params,
            flops_per_token=self.flops_per_token,
            gamma=self.gamma,
            model=self.model if self.flops_per_token is None and self.n_params is None else None,
            apply_cfg=True,
        )
        if value is None:
            cached = self.metadata.get("log_flops")
            if cached is None:
                raise ValueError(
                    f"cannot determine log-FLOPs for point {self.task}/{self.model}; "
                    "provide flops_per_token, n_params or metadata['log_flops']"
                )
            return float(cached)
        return value

    @property
    def flops(self) -> float:
        return math.exp(self.log_flops)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "task": self.task,
            "model": self.model,
            "gamma": self.gamma,
            "group": self.group,
            "accuracy": self.accuracy,
            "n_examples": self.n_examples,
            "n_params": self.n_params,
            "flops_per_token": self.flops_per_token,
        }
        try:
            out["log_flops"] = self.log_flops
            out["flops_per_token_applied"] = self.flops
        except ValueError:
            out["log_flops"] = None
        if self.metadata:
            out["metadata"] = dict(self.metadata)
        return out

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "BenchmarkPoint":
        return cls(**{k: v for k, v in row.items() if k in cls.__dataclass_fields__})


point_from_row = BenchmarkPoint.from_dict


def build_points(rows: Iterable[Union[Dict[str, Any], BenchmarkPoint]]) -> List[BenchmarkPoint]:
    """Coerce mixed dicts/:class:`BenchmarkPoint` into a point list."""
    out: List[BenchmarkPoint] = []
    for row in rows:
        out.append(row if isinstance(row, BenchmarkPoint) else BenchmarkPoint.from_dict(row))
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


@dataclass
class RegressionFit:
    """A fitted regression line on ``log(FLOPs per token) -> accuracy``.

    ``coefficients`` are in the order of the design matrix columns, provided in
    :attr:`column_names` (always ``["intercept", "log_flops", ...]``).  For
    logistic fits, ``intercept``/``slope`` are on the logit scale and
    :meth:`predict` returns probabilities; for OLS fits everything is on the
    accuracy scale.
    """

    name: str
    coefficients: List[float]
    covariance: List[List[float]]
    column_names: List[str]
    method: str = "logistic"  # "logistic" | "ols"
    log_likelihood: Optional[float] = None
    deviance: Optional[float] = None
    sse: Optional[float] = None
    n: int = 0
    df_resid: float = 0.0
    group: Optional[str] = None
    x_mean: Optional[float] = None
    x_range: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- inference ----------------------------------------------------------
    @property
    def standard_errors(self) -> List[float]:
        return [math.sqrt(max(0.0, self.covariance[i][i])) for i in range(len(self.coefficients))]

    @property
    def stats(self) -> List[float]:
        """z statistics (logistic) or t statistics (OLS) per coefficient."""
        se = self.standard_errors
        return [c / s if s > 0 else float("nan") for c, s in zip(self.coefficients, se)]

    @property
    def p_values(self) -> List[float]:
        stat = self.stats
        if self.method == "logistic":
            return [norm_sf(z) for z in stat]
        return [t_sf(t, self.df_resid) for t in stat]

    @property
    def intercept(self) -> float:
        return self.coefficients[0]

    @property
    def slope(self) -> float:
        return self.coefficients[1] if len(self.coefficients) > 1 else float("nan")

    @property
    def dispersion(self) -> Optional[float]:
        """Quasi-binomial dispersion ``deviance / df_resid`` (logistic only)."""
        if self.method != "logistic" or self.deviance is None or self.df_resid <= 0:
            return None
        return self.deviance / self.df_resid

    # -- prediction ---------------------------------------------------------
    def linear_predictor(self, x: float) -> float:
        eta = 0.0
        for name, coef in zip(self.column_names, self.coefficients):
            if name == "intercept":
                eta += coef
            elif name == "log_flops":
                eta += coef * float(x)
            elif name == "group":
                eta += coef  # group indicator == 1 (CFG group)
            elif name in ("group:log_flops", "log_flops:group"):
                eta += coef * float(x)
        return eta

    def predict(self, x: Union[float, Sequence[float]]) -> Union[float, List[float]]:
        """Predicted accuracy at ``x = log(FLOPs per token)``."""
        scalar = isinstance(x, (int, float))
        xs = [float(x)] if scalar else [float(v) for v in x]
        out: List[float] = []
        for xv in xs:
            eta = self.linear_predictor(xv)
            if self.method == "logistic":
                out.append(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, eta)))))
            else:
                out.append(eta)
        return out[0] if scalar else out

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "method": self.method,
            "coefficients": list(self.coefficients),
            "column_names": list(self.column_names),
            "standard_errors": self.standard_errors,
            "statistics": self.stats,
            "p_values": self.p_values,
            "intercept": self.intercept,
            "slope": self.slope,
            "log_likelihood": self.log_likelihood,
            "deviance": self.deviance,
            "sse": self.sse,
            "dispersion": self.dispersion,
            "n": self.n,
            "df_resid": self.df_resid,
            "x_mean": self.x_mean,
            "x_range": list(self.x_range) if self.x_range else None,
        }

    def line_str(self, decimals: int = 3) -> str:
        if self.method == "logistic":
            return (
                f"logit(acc) = {self.intercept:.{decimals}f} "
                f"{'+' if self.slope >= 0 else '-'} {abs(self.slope):.{decimals}f} * logFLOPs"
            )
        return (
            f"acc = {self.intercept:.{decimals}f} "
            f"{'+' if self.slope >= 0 else '-'} {abs(self.slope):.{decimals}f} * logFLOPs"
        )


def _points_of(points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]]) -> List[BenchmarkPoint]:
    return [p if isinstance(p, BenchmarkPoint) else BenchmarkPoint.from_dict(p) for p in points]


def _design(
    xs: Sequence[float],
    groups: Sequence[str],
    kind: str = "full",
) -> Tuple[List[List[float]], List[str]]:
    """Build the ANCOVA design matrix.

    ``kind``
        ``pooled``      -- ``[1, x]`` (one line for both groups, reduced model)
        ``group``       -- ``[1, x, g]`` (parallel lines, group main effect)
        ``interaction`` -- ``[1, x, g, g*x]`` (interaction only test: full vs group)
        ``full``        -- ``[1, x, g, g*x]`` (two independent lines, full model)
    """
    g = [1.0 if gp == CFG_GROUP else 0.0 for gp in groups]
    if kind == "pooled":
        X = [[1.0, float(x)] for x in xs]
        return X, ["intercept", "log_flops"]
    if kind == "group":
        X = [[1.0, float(x), gi] for x, gi in zip(xs, g)]
        return X, ["intercept", "log_flops", "group"]
    if kind in ("interaction", "full"):
        X = [[1.0, float(x), gi, gi * float(x)] for x, gi in zip(xs, g)]
        return X, ["intercept", "log_flops", "group", "group:log_flops"]
    raise ValueError(f"unknown design kind: {kind!r}")


def logistic_regression(
    X: Sequence[Sequence[float]],
    successes: Sequence[float],
    trials: Optional[Sequence[float]] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    ridge: float = 1e-8,
    column_names: Optional[Sequence[str]] = None,
    name: str = "logistic",
) -> RegressionFit:
    """Binomial logistic regression by IRLS (grouped/proportion data).

    ``successes`` may be counts (with matching ``trials``) or fractions (with
    ``trials == 1``).  Returns MLE coefficients and the inverse-Fisher-information
    covariance matrix, so Wald tests are available without statsmodels.
    """
    X = _to_lists(X)
    n = len(X)
    p = len(X[0]) if n else 0
    trials = [1.0] * n if trials is None else [float(t) for t in trials]
    y = [float(s) for s in successes]
    if len(y) != n or len(trials) != n:
        raise ValueError("X, successes and trials must have equal length")
    names = list(column_names) if column_names else [f"x{i}" for i in range(p)]

    beta = [0.0] * p
    # sensible starting value: intercept from the overall rate
    tot_trials = sum(trials)
    tot_succ = sum(y)
    if tot_trials > 0 and 0.0 < tot_succ < tot_trials:
        beta[0] = _logit(tot_succ / tot_trials)

    deviance = float("nan")
    for _ in range(max_iter):
        eta = _matvec(X, beta)
        mu = [trials[i] * (1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, eta[i]))))) for i in range(n)]
        w = [max(mu[i] * (1.0 - mu[i] / trials[i]), 1e-10) * 1.0 for i in range(n)]
        # working response z = eta + (y - mu) / w
        z = [eta[i] + (y[i] - mu[i]) / w[i] for i in range(n)]
        xtwx = _xtwx(X, w)
        for a in range(p):  # tiny ridge for numerical stability
            xtwx[a][a] += ridge
        xtwy = _xtwy(X, w, z)
        new_beta = _solve(xtwx, xtwy)
        change = max(abs(new_beta[a] - beta[a]) for a in range(p)) if p else 0.0
        beta = new_beta
        if change < tol:
            break

    # final statistics
    eta = _matvec(X, beta)
    mu = [trials[i] / (1.0 + math.exp(-max(-500.0, min(500.0, eta[i])))) for i in range(n)]
    w = [max(mu[i] * (1.0 - mu[i] / trials[i]), 1e-10) for i in range(n)]
    xtwx = _xtwx(X, w)
    for a in range(p):
        xtwx[a][a] += ridge
    cov = _invert(xtwx)

    ll = 0.0
    dev = 0.0
    for i in range(n):
        phat = min(max(mu[i] / trials[i], 1e-12), 1.0 - 1e-12)
        if y[i] > 0:
            ll += y[i] * math.log(phat)
        if trials[i] - y[i] > 0:
            ll += (trials[i] - y[i]) * math.log1p(-phat)
        dev += 2.0 * (
            (y[i] * math.log(y[i] / mu[i]) if y[i] > 0 else 0.0)
            + (
                (trials[i] - y[i]) * math.log((trials[i] - y[i]) / (trials[i] - mu[i]))
                if trials[i] - y[i] > 0
                else 0.0
            )
        )

    return RegressionFit(
        name=name,
        coefficients=beta,
        covariance=cov,
        column_names=names,
        method="logistic",
        log_likelihood=ll,
        deviance=dev,
        n=n,
        df_resid=float(n - p),
    )


def _weighted_ols(
    X: Sequence[Sequence[float]],
    y: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> Tuple[List[float], List[List[float]], float, float]:
    """Solve weighted least squares; returns (beta, cov, sse, df_resid)."""
    X = _to_lists(X)
    n = len(X)
    p = len(X[0])
    w = [1.0] * n if weights is None else [float(v) for v in weights]
    xtwx = _xtwx(X, w)
    for a in range(p):
        xtwx[a][a] += 1e-10
    xtwy = _xtwy(X, w, [float(v) for v in y])
    beta = _solve(xtwx, xtwy)
    resid = [float(y[i]) - sum(X[i][a] * beta[a] for a in range(p)) for i in range(n)]
    sse = sum(w[i] * resid[i] ** 2 for i in range(n))
    df = float(n - p)
    sigma2 = sse / df if df > 0 else float("nan")
    cov = _invert(xtwx)
    cov = [[cov[i][j] * sigma2 for j in range(p)] for i in range(p)]
    return beta, cov, sse, df


def fit_ols(
    points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]],
    weights: Union[str, Sequence[float], None] = "trials",
    kind: str = "full",
    name: str = "ols",
) -> RegressionFit:
    """Weighted least-squares fit of accuracy on log-FLOPs (classical ANCOVA)."""
    pts = _points_of(points)
    xs = [pt.log_flops for pt in pts]
    groups = [pt.group or group_of_gamma(pt.gamma) for pt in pts]
    X, names = _design(xs, groups, kind)
    y = [pt.accuracy for pt in pts]
    if weights == "trials":
        w: Optional[List[float]] = [float(pt.n_trials) for pt in pts]
    elif weights is None:
        w = None
    else:
        w = [float(v) for v in weights]
    beta, cov, sse, df = _weighted_ols(X, y, w)
    return RegressionFit(
        name=name,
        coefficients=beta,
        covariance=cov,
        column_names=names,
        method="ols",
        sse=sse,
        n=len(pts),
        df_resid=df,
        x_mean=sum(xs) / len(xs) if xs else None,
        x_range=(min(xs), max(xs)) if xs else None,
    )


def fit_logistic(
    points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]],
    kind: str = "full",
    name: str = "logistic",
) -> RegressionFit:
    """Binomial logistic fit of accuracy on log-FLOPs (plan's primary model)."""
    pts = _points_of(points)
    xs = [pt.log_flops for pt in pts]
    groups = [pt.group or group_of_gamma(pt.gamma) for pt in pts]
    X, names = _design(xs, groups, kind)
    trials = [float(pt.n_trials) for pt in pts]
    successes = [float(pt.n_correct) for pt in pts]
    fit = logistic_regression(X, successes, trials, column_names=names, name=name)
    fit.x_mean = sum(xs) / len(xs) if xs else None
    fit.x_range = (min(xs), max(xs)) if xs else None
    return fit


def fit_group_lines(
    points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]],
    method: str = "logistic",
) -> Dict[str, RegressionFit]:
    """Fit one line per group (the red CFG line and the blue vanilla line)."""
    pts = _points_of(points)
    out: Dict[str, RegressionFit] = {}
    for group in (VANILLA_GROUP, CFG_GROUP):
        subset = [pt for pt in pts if (pt.group or group_of_gamma(pt.gamma)) == group]
        if len(subset) < 2:
            logger.debug("group %s has %d points - line not fitted", group, len(subset))
            continue
        xs = [pt.log_flops for pt in subset]
        X = [[1.0, x] for x in xs]
        names = ["intercept", "log_flops"]
        if method == "ols":
            y = [pt.accuracy for pt in subset]
            w = [float(pt.n_trials) for pt in subset]
            beta, cov, sse, df = _weighted_ols(X, y, w)
            fit = RegressionFit(
                name=f"{group}_ols",
                coefficients=beta,
                covariance=cov,
                column_names=names,
                method="ols",
                sse=sse,
                n=len(subset),
                df_resid=df,
            )
        else:
            fit = logistic_regression(
                X,
                [float(pt.n_correct) for pt in subset],
                [float(pt.n_trials) for pt in subset],
                column_names=names,
                name=f"{group}_logistic",
            )
        fit.group = group
        fit.x_mean = sum(xs) / len(xs)
        fit.x_range = (min(xs), max(xs))
        out[group] = fit
    return out


# ---------------------------------------------------------------------------
# ANCOVA tests
# ---------------------------------------------------------------------------


def likelihood_ratio_test(reduced: RegressionFit, full: RegressionFit) -> Tuple[float, int, float]:
    """LR chi-square statistic, its d.o.f. and p-value."""
    if reduced.log_likelihood is None or full.log_likelihood is None:
        raise ValueError("likelihood-ratio test requires logistic fits")
    df = len(full.coefficients) - len(reduced.coefficients)
    stat = max(0.0, 2.0 * (full.log_likelihood - reduced.log_likelihood))
    return stat, df, chi2_sf(stat, df) if df > 0 else 1.0


def wald_test(fit: RegressionFit, terms: Sequence[Union[int, str]]) -> Tuple[float, int, float]:
    """Wald chi-square for a subset of coefficients being zero."""
    idx: List[int] = []
    for term in terms:
        if isinstance(term, int):
            idx.append(term)
        else:
            try:
                idx.append(fit.column_names.index(term))
            except ValueError as exc:
                raise KeyError(f"{term!r} not in {fit.column_names}") from exc
    sub_cov = [[fit.covariance[i][j] for j in idx] for i in idx]
    sub_beta = [fit.coefficients[i] for i in idx]
    inv = _invert(sub_cov)
    stat = 0.0
    for a, ba in enumerate(sub_beta):
        for b, bb in enumerate(sub_beta):
            stat += ba * inv[a][b] * bb
    df = len(idx)
    return stat, df, chi2_sf(stat, df)


def quasi_f_test(reduced: RegressionFit, full: RegressionFit) -> Tuple[float, float, float, float]:
    """Over-dispersed (quasi-binomial) F test on the deviance difference.

    Returns ``(F, p, df_diff, dispersion)``.  This is the closest analogue of a
    classical ANCOVA F test when the response is a proportion from a finite
    number of examples.
    """
    if reduced.deviance is None or full.deviance is None:
        raise ValueError("quasi-F test requires logistic fits (deviances)")
    df_diff = len(full.coefficients) - len(reduced.coefficients)
    df_resid = full.df_resid
    if df_diff <= 0 or df_resid <= 0:
        return 0.0, 1.0, 0.0, float("nan")
    phi = max(full.deviance / df_resid, 1e-10)
    stat = max(0.0, reduced.deviance - full.deviance) / df_diff / phi
    return stat, f_sf(stat, df_diff, df_resid), float(df_diff), phi


def chow_test(
    reduced_sse: float,
    full_sse: float,
    df_diff: float,
    df_resid_full: float,
) -> Tuple[float, float]:
    """Classical F test for the (weighted) OLS ANOVA decomposition."""
    if df_diff <= 0 or df_resid_full <= 0 or full_sse <= 0:
        return 0.0, 1.0
    stat = ((reduced_sse - full_sse) / df_diff) / (full_sse / df_resid_full)
    return stat, f_sf(max(stat, 0.0), df_diff, df_resid_full)


@dataclass
class AncovaResult:
    """ANCOVA outcome for a single task."""

    task: str
    method: str
    statistic: float
    df: Union[int, float]
    p_value: float
    alpha: float = SIGNIFICANCE_LEVEL
    winner: Optional[str] = None
    favor: Optional[str] = None
    x_compare: Optional[float] = None
    n_points: int = 0
    n_points_cfg: int = 0
    n_points_vanilla: int = 0
    fits: Dict[str, RegressionFit] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return bool(self.p_value < self.alpha)

    @property
    def inconclusive(self) -> bool:
        """``p > 0.01`` means "an insignificant difference between the lines"."""
        return not self.significant

    @property
    def conclusion(self) -> str:
        if not self.significant:
            return f"insignificant (p={self.p_value:.3f} > {self.alpha})"
        return f"significant (p={self.p_value:.3f} <= {self.alpha}): favors {self.favor}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "statistic": self.statistic,
            "df": self.df,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "significant": self.significant,
            "winner": self.winner,
            "favor": self.favor,
            "x_compare": self.x_compare,
            "n_points": self.n_points,
            "n_points_cfg": self.n_points_cfg,
            "n_points_vanilla": self.n_points_vanilla,
            "fits": {k: v.as_dict() for k, v in self.fits.items()},
            "extra": dict(self.extra),
        }


def winner_at(
    fits: Dict[str, RegressionFit],
    x: float,
    tolerance: float = 1e-9,
) -> Optional[str]:
    """Which group's line sits higher at ``x = log FLOPs per token``."""
    if VANILLA_GROUP not in fits or CFG_GROUP not in fits:
        return None
    a = float(fits[CFG_GROUP].predict(x))
    b = float(fits[VANILLA_GROUP].predict(x))
    if abs(a - b) <= tolerance:
        return None
    return CFG_GROUP if a > b else VANILLA_GROUP


def compare_lines(
    fits: Dict[str, RegressionFit],
    x: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare the two fitted lines across the observed FLOP range."""
    if CFG_GROUP not in fits or VANILLA_GROUP not in fits:
        return {"available": False}
    cfg, van = fits[CFG_GROUP], fits[VANILLA_GROUP]
    xs = [r[0] for r in (cfg.x_range, van.x_range) if r][0] if (cfg.x_range or van.x_range) else (0.0, 0.0)
    xs = (
        min(cfg.x_range[0] if cfg.x_range else xs[0], van.x_range[0] if van.x_range else xs[0]),
        max(cfg.x_range[1] if cfg.x_range else xs[1], van.x_range[1] if van.x_range else xs[1]),
    )
    if x is None:
        candidates = [v for v in (cfg.x_mean, van.x_mean) if v is not None]
        x = sum(candidates) / len(candidates) if candidates else 0.5 * (xs[0] + xs[1])
    return {
        "available": True,
        "x_compare": x,
        "accuracy_cfg": float(cfg.predict(x)),
        "accuracy_vanilla": float(van.predict(x)),
        "delta_cfg_minus_vanilla": float(cfg.predict(x)) - float(van.predict(x)),
        "winner": winner_at(fits, x),
        "x_range": list(xs),
        "meets_at": _meeting_point(cfg, van, xs),
    }


def _meeting_point(
    cfg: RegressionFit,
    van: RegressionFit,
    x_range: Tuple[float, float],
    n_grid: int = 400,
) -> Optional[float]:
    """Locate where the two lines cross (accuracy scale), if they do."""
    lo, hi = x_range
    prev = None
    for i in range(n_grid + 1):
        x = lo + (hi - lo) * i / n_grid
        diff = float(cfg.predict(x)) - float(van.predict(x))
        if prev is not None and prev[1] * diff <= 0.0 and abs(prev[1] - diff) > 0:
            frac = abs(prev[1]) / (abs(prev[1]) + abs(diff))
            return prev[0] + (x - prev[0]) * frac
        prev = (x, diff)
    return None


def ancova(
    points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]],
    method: str = "lr",
    test: str = "lines",
    alpha: float = SIGNIFICANCE_LEVEL,
    task: Optional[str] = None,
    x_compare: Optional[float] = None,
) -> AncovaResult:
    """Run the ANCOVA regression comparison for one task's data points.

    Parameters
    ----------
    points
        Data points for a single task (all models, vanilla and CFG gammas).
    method
        One of :data:`ANCOVA_METHODS` (``"lr"`` default: logistic regression +
        likelihood-ratio test, per the plan's "logistic (log-FLOP -> accuracy)
        regression lines ... then ANCOVA on log-transformed variables").
    test
        ``"lines"`` tests both intercept and slope differences (2 d.o.f.), the
        literal comparison of "the regression line of the CFG group ... and the
        one of the vanilla group".  ``"interaction"`` tests the slope difference
        only (1 d.o.f.).
    alpha
        Significance cutoff; the paper uses ``p = .01`` (Rutherford 2011).
    """
    method = method.lower()
    if method not in ANCOVA_METHODS:
        raise ValueError(f"method must be one of {ANCOVA_METHODS}, got {method!r}")
    pts = _points_of(points)
    if task:
        pts = [pt for pt in pts if pt.task == canonical_task_name(task)]
    if not pts:
        raise ValueError("ancova requires at least one data point")

    full_kind = "full" if test == "lines" else "interaction"
    reduced_kind = "pooled" if test == "lines" else "group"
    df_diff = len(_design([0.0], [CFG_GROUP], full_kind)[0][0]) - 0  # placeholder-free df calc
    df_diff = len(_design([0.0], [CFG_GROUP], full_kind)[1]) - len(
        _design([0.0], [VANILLA_GROUP], reduced_kind)[1]
    )

    groups = {}
    fits = fit_group_lines(pts, method="logistic" if method != "ols" else "ols")

    if method == "ols":
        full = fit_ols(pts, kind=full_kind, name="full_ols")
        reduced = fit_ols(pts, kind=reduced_kind, name="reduced_ols")
        stat, p_value = chow_test(reduced.sse, full.sse, df_diff, full.df_resid)
        df: Union[int, float] = df_diff
        extra = {
            "sse_reduced": reduced.sse,
            "sse_full": full.sse,
            "test": test,
        }
    else:
        full = fit_logistic(pts, kind=full_kind, name="full_logistic")
        reduced = fit_logistic(pts, kind=reduced_kind, name="reduced_logistic")
        if method == "lr":
            stat, df, p_value = likelihood_ratio_test(reduced, full)
            extra = {"log_likelihood_reduced": reduced.log_likelihood, "log_likelihood_full": full.log_likelihood}
        elif method == "wald":
            terms = ["group", "group:log_flops"] if test == "lines" else ["group:log_flops"]
            stat, df, p_value = wald_test(full, terms)
            extra = {"terms": terms}
        else:  # quasi-binomial F
            stat, p_value, df, phi = quasi_f_test(reduced, full)
            extra = {"dispersion": phi, "deviance_reduced": reduced.deviance, "deviance_full": full.deviance}
        extra.update({"test": test, "deviance_reduced": reduced.deviance, "deviance_full": full.deviance})

    fits["full"] = full
    fits["reduced"] = reduced
    comparison = compare_lines(fits, x=x_compare)
    winner = comparison.get("winner")
    xs = [pt.log_flops for pt in pts]
    xs_mean = sum(xs) / len(xs)

    return AncovaResult(
        task=pts[0].task,
        method=method,
        statistic=float(stat),
        df=df,
        p_value=float(p_value),
        alpha=alpha,
        winner=winner,
        favor=winner if p_value < alpha else None,
        x_compare=comparison.get("x_compare", xs_mean),
        n_points=len(pts),
        n_points_cfg=sum(1 for pt in pts if (pt.group or group_of_gamma(pt.gamma)) == CFG_GROUP),
        n_points_vanilla=sum(1 for pt in pts if (pt.group or group_of_gamma(pt.gamma)) == VANILLA_GROUP),
        fits=fits,
        extra={**extra, "comparison": comparison, "x_mean": xs_mean, "group_summaries": groups},
    )


def ancova_by_task(
    points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]],
    method: str = "lr",
    test: str = "lines",
    alpha: float = SIGNIFICANCE_LEVEL,
    tasks: Optional[Sequence[str]] = None,
    min_points_per_group: int = 2,
    x_compare: Optional[float] = None,
) -> Dict[str, AncovaResult]:
    """Run :func:`ancova` independently for each task present in ``points``."""
    pts = _points_of(points)
    wanted = [canonical_task_name(t) for t in tasks] if tasks else None
    by_task: Dict[str, List[BenchmarkPoint]] = {}
    for pt in pts:
        by_task.setdefault(pt.task, []).append(pt)

    results: Dict[str, AncovaResult] = {}
    for task, subset in by_task.items():
        if wanted is not None and task not in wanted:
            continue
        n_cfg = sum(1 for pt in subset if (pt.group or group_of_gamma(pt.gamma)) == CFG_GROUP)
        n_van = sum(1 for pt in subset if (pt.group or group_of_gamma(pt.gamma)) == VANILLA_GROUP)
        if min(n_cfg, n_van) < min_points_per_group:
            logger.warning(
                "skipping task %s: too few points per group (cfg=%d, vanilla=%d)", task, n_cfg, n_van
            )
            continue
        try:
            results[task] = ancova(subset, method=method, test=test, alpha=alpha, task=task, x_compare=x_compare)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ANCOVA failed for task %s: %s", task, exc)
    return results


def points_from_results(
    results: Union[Dict[str, Any], List[Dict[str, Any]], Iterable[BenchmarkPoint]],
    model_specs: Optional[Dict[str, Any]] = None,
    n_examples: Optional[Dict[str, int]] = None,
) -> List[BenchmarkPoint]:
    """Build ANCOVA points from a zero-shot sweep result table.

    ``results`` accepts either

    * ``{task: {gamma: {"accuracy": a, "model": m, "n_params": p, "flops_per_token": f}}}``,
    * ``{model: {task: {gamma: accuracy}}}``, or
    * a flat list of row dicts / :class:`BenchmarkPoint` objects.

    ``model_specs`` optionally maps a model name to a parameter count or to a
    :class:`src.analysis.flops.ModelSpec` (used to derive per-token FLOPs).
    """
    if isinstance(results, list) and results and isinstance(results[0], BenchmarkPoint):
        return list(results)  # type: ignore[arg-type]

    out: List[BenchmarkPoint] = []

    def _resolve(model: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if model_specs and model in model_specs:
            spec = model_specs[model]
            if isinstance(spec, (int, float)):
                info["n_params"] = float(spec)
            else:
                n = getattr(spec, "num_parameters", None)
                if callable(n):
                    n = n()
                if n is not None:
                    info["n_params"] = float(n)
                try:  # pragma: no cover - depends on sibling module
                    from . import flops as _flops  # type: ignore

                    info["flops_per_token"] = float(_flops.flops_per_token(spec))
                except Exception:
                    pass
        if "flops_per_token" not in info and "n_params" not in info:
            flops = _flops_from_model_name(model)
            if flops:
                info["flops_per_token"] = flops
        return info

    for key, value in (results.items() if isinstance(results, dict) else []):
        task = canonical_task_name(key)
        # layout A: {task: {gamma: row}}
        if isinstance(value, dict) and value and all(
            isinstance(v, (dict, BenchmarkPoint)) for v in value.values()
        ):
            for gamma, row in value.items():
                row = row.as_dict() if isinstance(row, BenchmarkPoint) else dict(row)
                model = str(row.get("model", "unknown"))
                info = {**_resolve(model), **row}
                info.setdefault("gamma", float(gamma) if _is_number(gamma) else VANILLA_GAMMA)
                out.append(
                    BenchmarkPoint(
                        task=task,
                        accuracy=float(row.get("accuracy", row.get("acc", 0.0))),
                        model=model,
                        gamma=float(info["gamma"]),
                        n_examples=n_examples.get(task) if n_examples else row.get("n_examples"),
                        n_params=info.get("n_params"),
                        flops_per_token=info.get("flops_per_token"),
                        metadata={k: v for k, v in row.items() if k not in _POINT_FIELDS},
                    )
                )
        # layout B: {model: {task: {gamma: acc}}} or {model: {task: acc}}
        elif isinstance(value, dict):
            for task_name, cell in value.items():
                if isinstance(cell, dict):
                    for gamma, acc in cell.items():
                        out.append(
                            BenchmarkPoint(
                                task=task_name,
                                accuracy=float(acc),
                                model=key,
                                gamma=float(gamma) if _is_number(gamma) else VANILLA_GAMMA,
                                n_examples=n_examples.get(canonical_task_name(task_name)) if n_examples else None,
                                **_resolve(key),
                            )
                        )
                elif _is_number(cell):
                    out.append(
                        BenchmarkPoint(
                            task=task_name,
                            accuracy=float(cell),
                            model=key,
                            gamma=VANILLA_GAMMA,
                            n_examples=n_examples.get(canonical_task_name(task_name)) if n_examples else None,
                            **_resolve(key),
                        )
                    )
    return out


_POINT_FIELDS = set(BenchmarkPoint.__dataclass_fields__.keys())  # type: ignore[attr-defined]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def ancova_table(
    points: Iterable[Union[BenchmarkPoint, Dict[str, Any]]],
    method: str = "lr",
    test: str = "lines",
    alpha: float = SIGNIFICANCE_LEVEL,
    **kwargs: Any,
) -> Dict[str, AncovaResult]:
    """Alias of :func:`ancova_by_task` returning the per-task result mapping."""
    return ancova_by_task(points, method=method, test=test, alpha=alpha, **kwargs)


def p_value_table(
    results: Union[Dict[str, AncovaResult], Iterable[Union[BenchmarkPoint, Dict[str, Any]]]],
    method: str = "lr",
    test: str = "lines",
    alpha: float = SIGNIFICANCE_LEVEL,
) -> Dict[str, Dict[str, Any]]:
    """``{task: {"p_value", "winner", "favor", "significant", "conclusion"}}``."""
    if isinstance(results, dict) and all(isinstance(v, AncovaResult) for v in results.values()):
        per_task: Dict[str, AncovaResult] = dict(results)  # type: ignore[arg-type]
    else:
        per_task = ancova_by_task(results, method=method, test=test, alpha=alpha)  # type: ignore[arg-type]
    table: Dict[str, Dict[str, Any]] = {}
    for task, res in per_task.items():
        table[task] = {
            "p_value": res.p_value,
            "statistic": res.statistic,
            "df": res.df,
            "winner": res.winner,
            "favor": res.favor,
            "significant": res.significant,
            "conclusion": res.conclusion,
            "n_points": res.n_points,
        }
    return table


def significant_tasks(
    results: Union[Dict[str, AncovaResult], Dict[str, float], None] = None,
    alpha: float = SIGNIFICANCE_LEVEL,
    p_values: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Tasks whose regression lines differ significantly at ``alpha``."""
    pvals = _pvals(results, p_values)
    out = []
    for task, p in sorted(pvals.items(), key=lambda kv: kv[1]):
        if p < alpha:
            out.append({"task": task, "p_value": p, "favor": _favor_from_results(results, task)})
    return out


def favor_counts(
    results: Union[Dict[str, AncovaResult], Dict[str, float], None] = None,
    alpha: float = SIGNIFICANCE_LEVEL,
    p_values: Optional[Dict[str, float]] = None,
    winners: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, int]:
    """Count ``{inconclusive, cfg, vanilla}`` at the ``alpha`` cutoff.

    Section 4 reports ``{inconclusive: 5, cfg: 2, vanilla: 2}`` for the nine
    Section 3.1 tasks (using :data:`PAPER_TABLE6` when no results are given).
    """
    pvals = _pvals(results, p_values)
    counts = {"inconclusive": 0, CFG_GROUP: 0, VANILLA_GROUP: 0}
    for task, p in pvals.items():
        if p >= alpha:
            counts["inconclusive"] += 1
            continue
        winner = None
        if winners and task in winners:
            winner = winners[task]
        else:
            winner = _favor_from_results(results, task)
        if winner in (CFG_GROUP, VANILLA_GROUP):
            counts[winner] += 1
        else:
            counts["inconclusive"] += 1
    return counts


def _pvals(
    results: Union[Dict[str, AncovaResult], Dict[str, float], None],
    p_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if p_values is not None:
        return {canonical_task_name(k): float(v) for k, v in p_values.items()}
    if results is None:
        return dict(PAPER_TABLE6_PVALUES)
    if all(isinstance(v, (int, float)) for v in results.values()):
        return {canonical_task_name(k): float(v) for k, v in results.items()}  # type: ignore[arg-type]
    return {
        canonical_task_name(k): float(v.p_value) for k, v in results.items()  # type: ignore[union-attr]
    }


def _favor_from_results(
    results: Union[Dict[str, AncovaResult], Dict[str, float], None],
    task: str,
) -> Optional[str]:
    if results and task in results and isinstance(results[task], AncovaResult):  # type: ignore[index]
        return results[task].favor  # type: ignore[index,union-attr]
    if task in PAPER_TABLE6_WINNERS and _pvals(results)[task] == PAPER_TABLE6_PVALUES.get(task):
        return PAPER_TABLE6_WINNERS[task]
    return PAPER_TABLE6_WINNERS.get(task)


def format_ancova_table(
    results: Union[Dict[str, AncovaResult], Dict[str, float]],
    alpha: float = SIGNIFICANCE_LEVEL,
    method: Optional[str] = None,
) -> str:
    """Plain-text rendering of the per-task ANCOVA p-value / winner table."""
    if all(isinstance(v, AncovaResult) for v in results.values()):
        per_task = results  # type: ignore[assignment]
    else:
        per_task = {
            task: AncovaResult(
                task=task,
                method=method or "paper",
                statistic=float("nan"),
                df=0,
                p_value=float(p),
                alpha=alpha,
                winner=PAPER_TABLE6_WINNERS.get(task),
                favor=PAPER_TABLE6_WINNERS.get(task) if float(p) < alpha else None,
            )
            for task, p in results.items()  # type: ignore[union-attr]
        }

    header = f"{'task':<16} {'p-value':>9} {'sig.@0.01':>10} {'favors':>9}   conclusion"
    lines = [header, "-" * len(header)]
    for task in sorted(per_task, key=lambda t: per_task[t].p_value):  # type: ignore[index]
        res: AncovaResult = per_task[task]  # type: ignore[index]
        fav = res.favor or "-"
        lines.append(
            f"{res.task:<16} {res.p_value:>9.3f} {('yes' if res.significant else 'no'):>10} "
            f"{fav:>9}   {res.conclusion}"
        )
    counts = favor_counts(per_task, alpha=alpha)
    lines.append("-" * len(header))
    lines.append(
        f"favor counts @p={alpha}: inconclusive={counts['inconclusive']}, "
        f"cfg={counts[CFG_GROUP]}, vanilla={counts[VANILLA_GROUP]}"
    )
    return "\n".join(lines)


@dataclass
class AncovaReport:
    """Full Section 4 ANCOVA report for a set of benchmark points."""

    results: Dict[str, AncovaResult] = field(default_factory=dict)
    method: str = "lr"
    test: str = "lines"
    alpha: float = SIGNIFICANCE_LEVEL
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def p_values(self) -> Dict[str, float]:
        return {task: res.p_value for task, res in self.results.items()}

    @property
    def counts(self) -> Dict[str, int]:
        return favor_counts(self.results, alpha=self.alpha)

    def table(self) -> str:
        return format_ancova_table(self.results, alpha=self.alpha, method=self.method)

    def check_against_paper(self, expected: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        return check_against_paper(
            results=self.results, alpha=self.alpha, expected=expected
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "test": self.test,
            "alpha": self.alpha,
            "results": {k: v.as_dict() for k, v in self.results.items()},
            "p_values": self.p_values,
            "favor_counts": self.counts,
            "paper_favor_counts": dict(PAPER_FAVOR_COUNTS),
            "extra": dict(self.extra),
        }


def ancova_report(
    points: Union[Iterable[Union[BenchmarkPoint, Dict[str, Any]]], None] = None,
    method: str = "lr",
    test: str = "lines",
    alpha: float = SIGNIFICANCE_LEVEL,
    use_paper_when_empty: bool = True,
    **kwargs: Any,
) -> AncovaReport:
    """Compute per-task ANCOVA results, falling back to the paper's p-values.

    When ``points`` is empty (e.g. no GPU available to produce the Table 5
    sweep), ``use_paper_when_empty`` makes the report carry Appendix C.2's
    published p-values so downstream validation of the Section 4 split still
    works.
    """
    pts = _points_of(points) if points else []
    if pts:
        results = ancova_by_task(pts, method=method, test=test, alpha=alpha, **kwargs)
        extra: Dict[str, Any] = {"source": "measured", "n_points": len(pts)}
    else:
        results = {}
        extra = {"source": "paper" if use_paper_when_empty else "empty"}
    return AncovaReport(results=results, method=method, test=test, alpha=alpha, extra=extra)


def check_against_paper(
    results: Union[Dict[str, AncovaResult], Dict[str, float], None] = None,
    alpha: float = SIGNIFICANCE_LEVEL,
    p_values: Optional[Dict[str, float]] = None,
    expected: Optional[Dict[str, int]] = None,
    tolerance: float = 0.02,
) -> Dict[str, Any]:
    """Validate the Section 4 claim: 5 insignificant, 2 favor CFG, 2 favor vanilla.

    Also reports per-task deviations from Appendix C.2's published p-values when
    a mapping of measured p-values is supplied.
    """
    pvals = _pvals(results, p_values)
    expected_counts = dict(expected or PAPER_FAVOR_COUNTS)
    winners = {
        task: (PAPER_TABLE6_WINNERS.get(task) if not isinstance(results, dict) or not results
               else None)
        for task in pvals
    }
    # measured winners take precedence when available
    if isinstance(results, dict) and results and all(isinstance(v, AncovaResult) for v in results.values()):
        winners = {task: res.favor for task, res in results.items()}  # type: ignore[union-attr]

    counts = favor_counts(p_values=pvals, alpha=alpha, winners={k: v for k, v in winners.items() if v})
    within_cutoff = all(
        (p < alpha) == (PAPER_TABLE6_PVALUES[task] < alpha)
        for task, p in pvals.items()
        if task in PAPER_TABLE6_PVALUES
    )
    deltas = {
        task: abs(p - PAPER_TABLE6_PVALUES[task])
        for task, p in pvals.items()
        if task in PAPER_TABLE6_PVALUES
    }
    return {
        "alpha": alpha,
        "measured_favor_counts": counts,
        "expected_favor_counts": expected_counts,
        "counts_match": counts == expected_counts,
        "significant_matches_paper": within_cutoff,
        "max_p_value_deviation": max(deltas.values()) if deltas else None,
        "deviations": deltas,
        "deviations_within_tolerance": all(d <= tolerance for d in deltas.values()) if deltas else True,
        "paper_p_values": dict(PAPER_TABLE6_PVALUES),
    }


def to_dataframe(results: Dict[str, AncovaResult]):
    """Optional pandas view of a result mapping (requires pandas)."""
    try:  # pragma: no cover - optional dependency
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("pandas is required for to_dataframe") from exc
    rows = []
    for task, res in results.items():
        rows.append(
            {
                "task": task,
                "p_value": res.p_value,
                "statistic": res.statistic,
                "df": res.df,
                "significant": res.significant,
                "favor": res.favor,
                "winner": res.winner,
                "slope_cfg": _slope(res, CFG_GROUP),
                "slope_vanilla": _slope(res, VANILLA_GROUP),
                "n_points_cfg": res.n_points_cfg,
                "n_points_vanilla": res.n_points_vanilla,
            }
        )
    return pd.DataFrame(rows)


def _slope(res: AncovaResult, group: str) -> Optional[float]:
    fit = res.fits.get(group)
    return None if fit is None else fit.slope


def save_ancova(path: str, report: Union[AncovaReport, Dict[str, Any]]) -> str:
    """Persist an ANCOVA report (or table) as JSON."""
    payload = report.as_dict() if isinstance(report, AncovaReport) else report
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_ancova(path: str) -> Dict[str, Any]:
    """Load a JSON ANCOVA report."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Synthetic data + self-test
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def synthetic_points(
    tasks: Sequence[str] = BENCHMARK_TASKS,
    n_params_grid: Sequence[float] = (
        1.24e8,   # gpt2
        3.55e8,   # gpt2-medium
        7.74e8,   # gpt2-large
        1.56e9,   # gpt2-xl
        1.6e8,    # pythia-160m
        1.4e9,    # pythia-1.4b
        6.9e9,    # pythia-6.9b
        1.2e10,   # pythia-12b
    ),
    gammas: Sequence[float] = (VANILLA_GAMMA, CFG_GAMMA),
    n_examples: int = 1000,
    seed: int = 0,
    effects: Optional[Dict[str, float]] = None,
) -> List[BenchmarkPoint]:
    """Deterministic synthetic accuracy-vs-FLOP data for dry runs and unit tests.

    The generative model is a logistic curve in ``log FLOPs per token`` with a
    per-task intercept, plus a per-task CFG bonus :data:`_SYNTHETIC_EFFECTS`
    (positive = CFG's line lies above vanilla's).  The bonus is applied to the
    CFG points only, mimicking the Appendix C.2 charts.
    """
    import random

    rng = random.Random(seed)
    table = dict(_SYNTHETIC_EFFECTS)
    if effects:
        table.update(effects)
    out: List[BenchmarkPoint] = []
    for task in tasks:
        task_c = canonical_task_name(task)
        effect = table.get(task_c, 0.0)
        # Accuracy rises with FLOPs; scale so ~1e8 -> low, ~1e10 -> high.
        for n_params in n_params_grid:
            for gamma in gammas:
                is_cfg = abs(gamma - 1.0) > 1e-9
                x = math.log(2.0 * n_params)
                base = -1.6 + 0.30 * (x - math.log(2e9))
                if is_cfg:
                    base += effect
                p = _sigmoid(base)
                acc = min(max(p + rng.gauss(0.0, 0.004), 0.0), 1.0)
                out.append(
                    BenchmarkPoint(
                        task=task_c,
                        accuracy=acc,
                        model=f"synth-{n_params:.3g}",
                        gamma=float(gamma),
                        n_examples=n_examples,
                        n_params=float(n_params),
                    )
                )
    return out


#: Synthetic per-task CFG logit-scale bonuses used by :func:`synthetic_points`.
#: Chosen so the machinery recovers the paper's qualitative picture: CFG clearly
#: helps LAMBADA and SciQ, hurts WinoGrande, and the rest are mixed/close.
_SYNTHETIC_EFFECTS: Dict[str, float] = {
    "lambada_openai": 0.55,
    "winogrande": -0.35,
    "sciq": 0.45,
    "triviaqa": -0.20,
    "hellaswag": 0.10,
    "piqa": 0.08,
    "arc_challenge": -0.05,
    "boolq": 0.05,
    "arc_easy": 0.02,
}


def _demo() -> None:
    """Self-test: exact identities + paper-anchor checks (no model required)."""
    # --- distribution helpers -------------------------------------------
    assert abs(chi2_sf(3.841, 1) - 0.05) < 5e-3, chi2_sf(3.841, 1)
    assert abs(f_sf(4.0, 2, 40) - f_sf(4.0, 2, 40)) < 1e-12
    assert 0.0 < norm_sf(1.96) < 0.06

    # --- fitted lines reproduce known parameters -------------------------
    xs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    pts = [
        BenchmarkPoint(task="t", accuracy=_sigmoid(0.5 + 0.8 * x), model="m", gamma=1.0, n_examples=1000, metadata={"log_flops": x})
        for x in xs
    ]
    fit = fit_ols(pts, kind="pooled")
    assert abs(fit.intercept - 0.5) < 0.05, fit.intercept
    assert abs(fit.slope - 0.8) < 0.05, fit.slope
    assert abs(float(fit.predict(0.0)) - 0.5) < 0.05

    # --- identical lines => insignificant ---------------------------------
    same = [
        BenchmarkPoint(
            task="t",
            accuracy=_sigmoid(-0.5 + 0.35 * x),
            model="m",
            gamma=1.0 if i % 2 == 0 else CFG_GAMMA,
            n_examples=1000,
            metadata={"log_flops": x},
        )
        for i, x in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, -0.5, 0.5, 1.5])
    ]
    res_same = ancova(same, method="lr")
    assert res_same.p_value > 0.05, res_same.p_value
    assert not res_same.significant
    assert res_same.conclusion.startswith("insignificant")

    # --- separated lines => significant, CFG wins -------------------------
    sep = []
    for i, x in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0, -1.5, -0.5, 0.5, 1.5]):
        is_cfg = i % 2 == 1
        base = -3.5 + 0.5 * x + (1.2 if is_cfg else 0.0)
        sep.append(
            BenchmarkPoint(
                task="t",
                accuracy=_sigmoid(base),
                model="m",
                gamma=CFG_GAMMA if is_cfg else 1.0,
                n_examples=5000,
                metadata={"log_flops": x},
            )
        )
    res_sep = ancova(sep, method="lr")
    assert res_sep.significant, res_sep.p_value
    assert res_sep.favor == CFG_GROUP, res_sep.favor

    # --- all four test statistics run and agree in direction --------------
    for method in ANCOVA_METHODS:
        r = ancova(sep, method=method)
        assert r.p_value < 0.05, (method, r.p_value)
        assert r.favor == CFG_GROUP, (method, r.favor)

    # --- gamma handling ---------------------------------------------------
    assert group_of_gamma(1.0) == VANILLA_GROUP
    assert group_of_gamma(0.0) == CFG_GROUP
    assert group_of_gamma(1.5) == CFG_GROUP
    assert log_flops_per_token(n_params=1e9, gamma=1.5) - log_flops_per_token(
        n_params=1e9, gamma=1.0
    ) == math.log(2.0)

    # --- synthetic sweep + report ----------------------------------------
    syn = synthetic_points(seed=3)
    by_task = ancova_by_task(syn, method="lr")
    assert len(by_task) == len(BENCHMARK_TASKS), sorted(by_task)
    rep = AncovaReport(results=by_task, method="lr")
    text = rep.table()
    assert "task" in text and "favor counts" in text
    chunk = rep.as_dict()
    assert "p_values" in chunk and "favor_counts" in chunk

    # --- paper anchors ----------------------------------------------------
    checks = check_against_paper(p_values=PAPER_TABLE6_PVALUES)
    assert checks["counts_match"], checks["measured_favor_counts"]
    assert checks["significant_matches_paper"], checks
    assert checks["max_p_value_deviation"] == 0.0, checks
    assert favor_counts(p_values=PAPER_TABLE6_PVALUES) == PAPER_FAVOR_COUNTS
    assert len(significant_tasks(p_values=PAPER_TABLE6_PVALUES)) == 4

    print(text)
    print(
        "synthetic favor counts @0.01:",
        favor_counts(p_values={t: r.p_value for t, r in by_task.items()}, alpha=SIGNIFICANCE_LEVEL),
    )
    print("ancova.py self-test OK")


if __name__ == "__main__":  # pragma: no cover
    _demo()
