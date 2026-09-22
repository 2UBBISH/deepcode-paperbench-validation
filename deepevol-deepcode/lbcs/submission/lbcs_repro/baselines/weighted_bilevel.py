"""Trivial bilevel coreset-selection baselines of §2.1 (Eq. (3) and Eq. (4)).

The paper motivates Refined Coreset Selection by showing that straightforward
adaptations of existing bilevel coreset selection are *non-trivial*:

* Eq. (3) -- "O1 only", fixed/upper-bounded coreset size::

      min_m f1(m)  s.t.  theta(m) in arg min_theta L(m, theta)

  With no optimization pressure on the coreset size, ``f1(m)`` is minimized
  effectively (Figure 1(a)) while ``f2(m)`` remains close to the predefined
  size ``k`` (Figure 1(b)) -- not the desideratum of RCS.

* Eq. (4) -- weighted combination of both objectives::

      min_m (1 - lambda) f1(m) + lambda f2(m)
      s.t.  theta(m) in arg min_theta L(m, theta)

  With equal weights (``lambda = 1/2``) the optimization does not implicitly
  favour ``f1(m)``: the minimization of ``f2(m)`` is salient and after all
  iterations ``f2(m)`` is too small while ``f1(m)`` is still large
  (Figure 1(c) and 1(d)).  Tuning ``lambda < 1/2`` is intractable because the
  two objectives have data/network/optimizer-dependent magnitudes.

Both formulations are realized the way Figure 1's reference method does it
(Zhou et al., 2022): the binary mask is relaxed with a Bernoulli
reparameterization ``m_i ~ Bern(s_i)`` and the outer probabilities ``s`` are
optimized with the unbiased score-function (policy-gradient) estimator
``f(m) * grad_s log p(m|s) = f(m) * (m - s) / (s (1 - s))`` (Appendix C.1-C.2).
The outer optimizer is Adam with learning rate ``2.5`` and a cosine scheduler;
the inner loop trains the model for 100 epochs with SGD (lr 0.1, momentum 0.9)
-- see Appendix C.3.

This module provides:

* the (non-differentiable) objective combinations of Eq. (3) / Eq. (4),
* a self-contained numpy outer loop ``weighted_bilevel`` implementing both
  formulations (it reuses ``baselines.probabilistic`` when available),
* convenience runners ``eq3_curves`` / ``eq4_curves`` / ``figure1_comparison``
  returning the ``f1``/``f2``-versus-iteration curves of Figure 1,
* a small diagnosis helper detecting the two failure modes,
* ``ScoreBaseline``-compatible selectors for §5.2-style comparisons.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import (
    BaselineSelector,
    ScoreBaseline,
    indices_to_mask,
    resolve_seed,
    set_seed,
    topk_indices,
)

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Soft import of the probabilistic (Zhou et al., 2022) machinery
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import guard
    from . import probabilistic as _probabilistic  # type: ignore

    _PROBABILISTIC_AVAILABLE = True
except Exception:  # pragma: no cover - keeps module importable standalone
    _probabilistic = None  # type: ignore
    _PROBABILISTIC_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Paper-stated + suggested defaults
# --------------------------------------------------------------------------- #
#: Equal weighting of the two objectives in Eq. (4) (§2.1 discussion).
EQ4_LAMBDA = 0.5
#: ``lambda`` used for Figure 1 (paper states ``lambda = 1/2``).
DEFAULT_LAMBDA = 0.5

# Figure 1 settings (Appendix C.3) -- paper-stated numbers.
FIGURE1_INNER_EPOCHS = 100
FIGURE1_INNER_LR = 0.1
FIGURE1_INNER_MOMENTUM = 0.9
FIGURE1_OUTER_LR = 2.5
FIGURE1_OUTER_ITERS = 1000
FIGURE1_OUTER_OPTIMIZER = "adam"
FIGURE1_OUTER_SCHEDULER = "cosine"

# SUGGESTED defaults for values the paper does not specify for this baseline.
SUGGESTED_OUTER_ITERS = 500
SUGGESTED_SAMPLES_PER_ITER = 1
SUGGESTED_S_MIN = 1e-3
SUGGESTED_S_MAX = 1.0 - 1e-3
SUGGESTED_ADAM_BETA1 = 0.9
SUGGESTED_ADAM_BETA2 = 0.999
SUGGESTED_ADAM_EPS = 1e-8

#: Variant identifiers.
VARIANTS = ("auto", "probabilistic", "internal")

__all__ = [
    "EQ4_LAMBDA",
    "DEFAULT_LAMBDA",
    "FIGURE1_INNER_EPOCHS",
    "FIGURE1_INNER_LR",
    "FIGURE1_INNER_MOMENTUM",
    "FIGURE1_OUTER_LR",
    "FIGURE1_OUTER_ITERS",
    "FIGURE1_OUTER_OPTIMIZER",
    "FIGURE1_OUTER_SCHEDULER",
    "SUGGESTED_OUTER_ITERS",
    "VARIANTS",
    "eq3_value",
    "eq4_value",
    "combined_value",
    "TrivialBilevelResult",
    "WeightedBilevelBaseline",
    "Eq3Baseline",
    "Eq4Baseline",
    "weighted_bilevel",
    "trivial_bilevel_curves",
    "eq3_curves",
    "eq4_curves",
    "figure1_comparison",
    "diagnose_failure_mode",
    "WeightedBilevelSelector",
    "Eq3Selector",
    "Eq4Selector",
]


# --------------------------------------------------------------------------- #
# Objective combinations of Eq. (3) and Eq. (4)
# --------------------------------------------------------------------------- #
def eq3_value(f1: float, f2: Optional[float] = None) -> float:
    """Eq. (3): only the performance objective ``f1`` is optimized."""
    return float(f1)


def eq4_value(f1: float, f2: float, lambda_: float = DEFAULT_LAMBDA) -> float:
    """Eq. (4): ``(1 - lambda) f1 + lambda f2``.

    Note that the paper keeps the *raw* magnitudes of the two objectives (the
    magnitudes are data/network/optimizer dependent, which is precisely why
    ``lambda < 1/2`` is intractable).  Therefore no normalization is applied by
    default; pass ``size_norm`` to :func:`combined_value` if a normalized
    variant is wanted for diagnostics.
    """
    return float((1.0 - float(lambda_)) * float(f1) + float(lambda_) * float(f2))


def combined_value(
    f1: float,
    f2: float,
    lambda_: float = DEFAULT_LAMBDA,
    use_size_objective: bool = True,
    size_norm: Optional[float] = None,
) -> float:
    """Scalar objective of Eq. (3) or Eq. (4).

    Args:
        f1: full-data objective of the trained proxy network.
        f2: coreset size ``||m||_0``.
        lambda_: trade-off of Eq. (4) (ignored when ``use_size_objective`` is False).
        use_size_objective: ``False`` reproduces Eq. (3) (``f1`` only).
        size_norm: optional divisor applied to ``f2`` before weighting
            (SUGGESTED, used only for numerical diagnostics).
    """
    if not use_size_objective:
        return eq3_value(f1, f2)
    f2_scaled = float(f2) / float(size_norm) if size_norm else float(f2)
    return eq4_value(f1, f2_scaled, lambda_)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _cosine_lr(step: int, total: int, base_lr: float, eta_min: float = 0.0) -> float:
    """Cosine schedule used by Figure 1's outer loop (Appendix C.3)."""
    if total <= 1:
        return float(base_lr)
    t = min(max(int(step), 0), int(total) - 1)
    return float(eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * t / float(total - 1))))


def _score_function_gradient(
    mask: np.ndarray,
    s: np.ndarray,
    eps: float = SUGGESTED_S_MIN,
) -> np.ndarray:
    """``grad_s log p(m | s) = (m - s) / (s (1 - s))`` (Appendix C.1)."""
    s = np.clip(np.asarray(s, dtype=np.float64), eps, 1.0 - eps)
    m = np.asarray(mask, dtype=np.float64)
    return (m - s) / (s * (1.0 - s))


def _bernoulli_sample(s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(np.shape(s)) < s).astype(np.float64)


class _Adam:
    """Minimal numpy Adam used by the self-contained outer loop.

    The paper optimizes the outer probabilities with Adam (lr 2.5, cosine
    scheduler) following Zhou et al. (2022); betas/eps are SUGGESTED defaults.
    """

    def __init__(
        self,
        lr: float = FIGURE1_OUTER_LR,
        beta1: float = SUGGESTED_ADAM_BETA1,
        beta2: float = SUGGESTED_ADAM_BETA2,
        eps: float = SUGGESTED_ADAM_EPS,
    ) -> None:
        self.lr = float(lr)
        self.b1 = float(beta1)
        self.b2 = float(beta2)
        self.eps = float(eps)
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray, lr: Optional[float] = None) -> np.ndarray:
        params = np.asarray(params, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * (grad * grad)
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        v_hat = self.v / (1.0 - self.b2 ** self.t)
        step_lr = self.lr if lr is None else float(lr)
        return params - step_lr * m_hat / (np.sqrt(v_hat) + self.eps)


def _project_expected_size(
    s: np.ndarray,
    k: float,
    s_min: float = SUGGESTED_S_MIN,
    s_max: float = SUGGESTED_S_MAX,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> np.ndarray:
    """Shift ``logit(s)`` by a constant so that ``sum(s) == k`` (bisection).

    This enforces the "predefined coreset size" of the fixed-size methods
    (Eq. (3) as used by Borsos et al. 2020 / Zhou et al. 2022).
    """
    s = np.clip(np.asarray(s, dtype=np.float64), s_min, s_max)
    if k is None:
        return s
    k = float(min(max(k, 0.0), float(s.size)))
    z = np.log(s / (1.0 - s))
    lo, hi = -60.0, 60.0
    target_tol = tol * max(1.0, k)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        val = float(np.sum(_sigmoid(z + mid)))
        if abs(val - k) <= target_tol:
            break
        if val > k:
            hi = mid
        else:
            lo = mid
    return np.clip(_sigmoid(z + 0.5 * (lo + hi)), s_min, s_max)


def _extract_size(objective: Any, mask: np.ndarray) -> float:
    """``f2(m) = ||m||_0`` computed directly from the (binary) mask."""
    m = np.asarray(mask).reshape(-1)
    return float(np.count_nonzero(m > 0.5) if m.dtype.kind == "f" else np.count_nonzero(m))


def _evaluate_objective(
    objective: Any,
    mask: np.ndarray,
    train_kwargs: Optional[Dict[str, Any]] = None,
    iteration: int = -1,
) -> Tuple[float, float, Any]:
    """Evaluate ``f1(m)`` (and ``f2(m)`` when the objective reports it).

    ``objective`` may be a ``lbcs.objectives.MaskObjectiveEvaluator``, a
    ``MaskEvaluation``-returning callable, or a plain ``mask -> f1`` callable.
    Returns ``(f1, f2_or_nan, raw)``.
    """
    kwargs = dict(train_kwargs or {})
    raw: Any = None
    if hasattr(objective, "evaluate"):
        try:
            raw = objective.evaluate(mask, iteration=iteration, **kwargs)
        except TypeError:
            raw = objective.evaluate(mask, **kwargs)
    elif callable(objective):
        raw = objective(mask, **kwargs)

    f1: Optional[float] = None
    f2: Optional[float] = None
    if raw is not None:
        if hasattr(raw, "f1"):
            f1 = float(getattr(raw, "f1"))
            if hasattr(raw, "f2"):
                f2 = float(getattr(raw, "f2"))
        elif isinstance(raw, dict):
            for key in ("f1", "f_1", "objective", "value", "loss"):
                if key in raw:
                    f1 = float(raw[key])
                    break
            for key in ("f2", "f_2", "size", "coreset_size", "num_selected"):
                if key in raw:
                    f2 = float(raw[key])
                    break
        elif isinstance(raw, (tuple, list)) and len(raw) >= 1:
            f1 = float(raw[0])
            if len(raw) >= 2 and not isinstance(raw[1], (bool, str)):
                try:
                    f2 = float(raw[1])
                except (TypeError, ValueError):
                    f2 = None
        elif np.isscalar(raw):
            f1 = float(raw)
    if f1 is None:
        raise ValueError(
            "Could not extract f1 from the objective; pass a MaskObjectiveEvaluator "
            "or a callable returning MaskEvaluation / (f1, f2) / a scalar."
        )
    if f2 is None or not np.isfinite(f2):
        f2 = _extract_size(objective, mask)
    return float(f1), float(f2), raw


def _infer_n(objective: Any, k: Optional[int] = None) -> int:
    for attr in ("n", "num_examples", "dataset_size", "dimension", "num_data"):
        val = getattr(objective, attr, None)
        if isinstance(val, (int, np.integer)) and val > 0:
            return int(val)
    for attr in ("dataset", "data", "targets", "labels"):
        val = getattr(objective, attr, None)
        try:
            if val is not None and len(val) > 0:  # type: ignore[arg-type]
                return int(len(val))  # type: ignore[arg-type]
        except TypeError:
            continue
    for attr in ("cache", "_cache"):
        cache = getattr(objective, attr, None)
        if cache is not None and hasattr(cache, "keys"):
            try:
                keys = list(cache.keys())
                if keys:
                    return int(len(cache))
            except Exception:  # pragma: no cover - defensive
                pass
    if k is not None:
        LOGGER.warning("Could not infer n from the objective; assuming n = 2 * k.")
        return int(2 * k)
    raise ValueError("Please provide `n` (number of candidate examples).")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TrivialBilevelResult:
    """Curves and final values of one Eq. (3) / Eq. (4) run."""

    formulation: str
    lambda_: float
    use_size_objective: bool
    k: int
    n: int
    f1: np.ndarray
    f2: np.ndarray
    objective_values: np.ndarray
    iterations: int
    best_f1: float
    best_f2: float
    final_f1: float
    final_f2: float
    best_mask: Any = None
    final_mask: Any = None
    s: Any = None
    expected_size: float = float("nan")
    wall_time: float = 0.0
    stopped_reason: str = "completed"
    num_evaluations: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    raw: Any = None

    # -- access ------------------------------------------------------------ #
    def curve(self, key: str) -> np.ndarray:
        key = str(key).lower()
        if key in ("f1", "f_1", "loss", "objective"):
            return np.asarray(self.f1)
        if key in ("f2", "f_2", "size", "coreset_size", "num_selected"):
            return np.asarray(self.f2)
        if key in ("obj", "combined", "value"):
            return np.asarray(self.objective_values)
        raise KeyError(f"Unknown curve key {key!r}")

    def __len__(self) -> int:
        return int(len(self.f1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formulation": self.formulation,
            "lambda": float(self.lambda_),
            "use_size_objective": bool(self.use_size_objective),
            "k": int(self.k),
            "n": int(self.n),
            "iterations": int(self.iterations),
            "num_evaluations": int(self.num_evaluations),
            "best_f1": float(self.best_f1),
            "best_f2": float(self.best_f2),
            "final_f1": float(self.final_f1),
            "final_f2": float(self.final_f2),
            "expected_size": float(self.expected_size),
            "wall_time": float(self.wall_time),
            "stopped_reason": self.stopped_reason,
            "f1_curve": np.asarray(self.f1).tolist(),
            "f2_curve": np.asarray(self.f2).tolist(),
        }

    # -- diagnosis of the trivial failure modes (§2.1, Figure 1) ----------- #
    def diagnostics(
        self,
        fixed_size_tol: float = 0.10,
        shrink_threshold: float = 0.75,
        f1_worse_tol: float = 1.0,
    ) -> Dict[str, Any]:
        return diagnose_failure_mode(
            self,
            fixed_size_tol=fixed_size_tol,
            shrink_threshold=shrink_threshold,
            f1_worse_tol=f1_worse_tol,
        )


def diagnose_failure_mode(
    result: TrivialBilevelResult,
    fixed_size_tol: float = 0.10,
    shrink_threshold: float = 0.75,
    f1_worse_tol: float = 1.0,
) -> Dict[str, Any]:
    """Detect the two trivial failure modes reported in §2.1 / Figure 1.

    * ``fixed_size``: Eq. (3) -- ``f2`` stays close to the predefined ``k``
      while ``f1`` decreases (Figure 1(a)-(b)).
    * ``over_minimized``: Eq. (4) with ``lambda = 1/2`` -- ``f2`` becomes too
      small while ``f1`` stays large (Figure 1(c)-(d)).
    """
    f1 = np.asarray(result.f1, dtype=np.float64)
    f2 = np.asarray(result.f2, dtype=np.float64)
    k = float(result.k) if result.k else float("nan")
    f1_init = float(f1[0]) if f1.size else float("nan")
    f1_final = float(f1[-1]) if f1.size else float("nan")
    f2_final = float(f2[-1]) if f2.size else float("nan")
    rel_dev = abs(f2_final - k) / k if k and np.isfinite(f2_final) and k > 0 else float("nan")
    info = {
        "formulation": result.formulation,
        "k": float(k),
        "f1_init": f1_init,
        "f1_final": f1_final,
        "f2_final": f2_final,
        "f1_improved": bool(np.isfinite(f1_init) and np.isfinite(f1_final) and f1_final < f1_init),
        "relative_size_deviation": float(rel_dev),
        "size_fraction_of_k": float(f2_final / k) if k else float("nan"),
    }
    info["fixed_size"] = bool(
        np.isfinite(rel_dev) and rel_dev <= float(fixed_size_tol) and info["f1_improved"]
    )
    info["over_minimized"] = bool(
        (not result.use_size_objective) is False
        and np.isfinite(f2_final)
        and k
        and (f2_final / k) <= float(shrink_threshold)
        and abs(f1_final - f1_init) <= float(f1_worse_tol) * max(1.0, abs(f1_init))
    )
    return info


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #
class WeightedBilevelBaseline:
    """Runner for the trivial bilevel baselines Eq. (3) and Eq. (4).

    Args:
        objective: mask-evaluating objective (typically
            ``lbcs.objectives.MaskObjectiveEvaluator``); must train the inner
            loop (``theta(m) <- arg min L(m, theta)``) and return ``f1(m)``.
        n: number of candidate examples (inferred from ``objective`` when None).
        k: predefined coreset size (fixed-size upper bound of Eq. (3)).
        lambda_: trade-off in Eq. (4); ``1/2`` reproduces Figure 1(c)-(d).
        formulation: ``"eq3"`` or ``"eq4"``; derived from
            ``use_size_objective``/``lambda_`` when omitted.
        use_size_objective: ``False`` -> Eq. (3) (``f1`` only), ``True`` -> Eq. (4).
        constrain_size: keep ``1^T s = k`` (Eq. (3) fixes the coreset size).
        final_selection: how the reported mask is derived from ``s`` --
            ``"topk"`` (Eq. (3)) or ``"threshold"`` (Eq. (4)).
        variant: ``"auto"`` (use ``baselines.probabilistic`` when importable),
            ``"probabilistic"`` (require it) or ``"internal"`` (self-contained).
        train_kwargs: forwarded to the objective's inner trainer
            (e.g. ``inner_epochs=100, inner_lr=0.1, inner_optimizer="sgd"``).
    """

    def __init__(
        self,
        objective: Any,
        n: Optional[int] = None,
        k: Optional[int] = None,
        lambda_: float = DEFAULT_LAMBDA,
        formulation: Optional[str] = None,
        use_size_objective: Optional[bool] = None,
        outer_iters: int = FIGURE1_OUTER_ITERS,
        outer_lr: float = FIGURE1_OUTER_LR,
        cosine: bool = True,
        optimizer: str = FIGURE1_OUTER_OPTIMIZER,
        samples_per_iter: int = SUGGESTED_SAMPLES_PER_ITER,
        constrain_size: Optional[bool] = None,
        final_selection: Optional[str] = None,
        variant: str = "auto",
        s_init: Optional[np.ndarray] = None,
        init_noise: float = 0.0,
        s_min: float = SUGGESTED_S_MIN,
        s_max: float = SUGGESTED_S_MAX,
        train_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 0,
        device: Any = None,
        log_every: int = 0,
        verbose: bool = False,
    ) -> None:
        self.objective = objective
        self.n = int(n) if n is not None else _infer_n(objective, k)
        self.k = int(k) if k is not None else int(max(1, self.n // 2))
        self.lambda_ = float(lambda_)
        self.variant = str(variant).lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

        if formulation is not None:
            form = str(formulation).lower().replace("equation", "eq").replace("(", "").replace(")", "")
            if form in ("3", "eq3", "eq_3"):
                form = "eq3"
            elif form in ("4", "eq4", "eq_4"):
                form = "eq4"
            if form not in ("eq3", "eq4"):
                raise ValueError(f"formulation must be 'eq3' or 'eq4', got {formulation!r}")
            self.formulation = form
        else:
            self.formulation = "eq4" if bool(use_size_objective) and self.lambda_ > 0 else "eq3"

        if use_size_objective is None:
            use_size_objective = self.formulation == "eq4"
        self.use_size_objective = bool(use_size_objective)
        # Eq. (3) optimizes f1 only; if the caller asks for eq3 we also set
        # lambda = 0 so the size term carries no weight.
        if self.formulation == "eq3" and not self.use_size_objective:
            self.effective_lambda = 0.0
        else:
            self.effective_lambda = self.lambda_

        self.outer_iters = int(outer_iters)
        self.outer_lr = float(outer_lr)
        self.cosine = bool(cosine)
        self.optimizer = str(optimizer).lower()
        self.samples_per_iter = max(1, int(samples_per_iter))
        if constrain_size is None:
            constrain_size = self.formulation == "eq3"  # fixed-size methods
        self.constrain_size = bool(constrain_size)
        if final_selection is None:
            final_selection = "topk" if (self.constrain_size or self.formulation == "eq3") else "threshold"
        self.final_selection = str(final_selection).lower()
        self.s_init = None if s_init is None else np.asarray(s_init, dtype=np.float64).reshape(-1)
        self.init_noise = float(init_noise)
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.train_kwargs = dict(train_kwargs or {})
        self.seed = 0 if seed is None else int(seed)
        self.device = device
        self.log_every = int(log_every)
        self.verbose = bool(verbose)
        self.result_: Optional[TrivialBilevelResult] = None

    # ------------------------------------------------------------------ #
    # probability initialization
    # ------------------------------------------------------------------ #
    def initial_probabilities(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if self.s_init is not None and self.s_init.size == self.n:
            s = self.s_init.astype(np.float64).copy()
        elif self.s_init is not None and self.s_init.size == 1:
            s = np.full(self.n, float(self.s_init[0]), dtype=np.float64)
        else:
            s = np.full(self.n, float(self.k) / float(max(1, self.n)), dtype=np.float64)
        if self.init_noise > 0:
            gen = rng if rng is not None else np.random.default_rng(self.seed)
            s = s + float(self.init_noise) * gen.standard_normal(self.n)
        s = np.clip(s, self.s_min, self.s_max)
        if self.constrain_size:
            s = _project_expected_size(s, self.k, self.s_min, self.s_max)
        return s

    # ------------------------------------------------------------------ #
    # running
    # ------------------------------------------------------------------ #
    def run(self) -> TrivialBilevelResult:
        """Run the outer loop and return the f1/f2 curves of Figure 1."""
        set_seed(self.seed)
        start = time.time()
        if self.variant in ("auto", "probabilistic") and _PROBABILISTIC_AVAILABLE:
            try:
                res = self._run_probabilistic()
                res.wall_time = time.time() - start
                self.result_ = res
                return res
            except Exception as exc:  # pragma: no cover - fall back to internal loop
                if self.variant == "probabilistic":
                    raise
                LOGGER.warning(
                    "baselines.probabilistic path failed (%s); using the internal "
                    "self-contained outer loop instead.",
                    exc,
                )
        res = self._run_internal()
        res.wall_time = time.time() - start
        self.result_ = res
        return res

    fit = run
    __call__ = run

    # -- path A: reuse baselines/probabilistic.py ------------------------ #
    def _run_probabilistic(self) -> TrivialBilevelResult:  # pragma: no cover - heavy
        prob = _probabilistic  # type: ignore[assignment]
        res = prob.probabilistic_bilevel(
            self.objective,
            self.n,
            k=self.k,
            lambda_=self.effective_lambda,
            use_size_objective=self.use_size_objective,
            outer_iters=self.outer_iters,
            outer_lr=self.outer_lr,
            cosine=self.cosine,
            samples_per_iter=self.samples_per_iter,
            s_init=self.s_init,
            init_noise=self.init_noise,
            s_min=self.s_min,
            s_max=self.s_max,
            constrain_size=self.constrain_size,
            final_selection=self.final_selection,
            train_kwargs=self.train_kwargs,
            seed=self.seed,
            verbose=self.verbose,
            log_every=self.log_every,
            track_history=True,
        )
        return self._from_probabilistic(result=res)

    def _from_probabilistic(self, result: Any) -> TrivialBilevelResult:
        history = list(getattr(result, "history", []) or [])
        f1 = np.full(len(history), np.nan)
        f2 = np.full(len(history), np.nan)
        for i, entry in enumerate(history):
            if isinstance(entry, dict):
                for key in ("f1", "f_1", "loss", "objective"):
                    if key in entry and entry[key] is not None:
                        f1[i] = float(entry[key])
                        break
                for key in ("f2", "f_2", "size", "coreset_size", "num_selected"):
                    if key in entry and entry[key] is not None:
                        f2[i] = float(entry[key])
                        break
        if history and np.isnan(f1).all() and hasattr(result, "curve"):
            try:
                f1 = np.asarray(result.curve("f1"), dtype=np.float64)
                f2 = np.asarray(result.curve("f2"), dtype=np.float64)
            except Exception:
                pass
        if not history:
            f1 = np.array([float(getattr(result, "final_f1", np.nan))])
            f2 = np.array([float(getattr(result, "final_f2", np.nan))])
        obj_vals = np.array(
            [
                combined_value(f1[i] if np.isfinite(f1[i]) else np.nan,
                               f2[i] if np.isfinite(f2[i]) else np.nan,
                               self.effective_lambda, self.use_size_objective)
                for i in range(len(f1))
            ]
        )
        finite_best = int(np.argmin(f1)) if np.isfinite(f1).any() else 0
        return TrivialBilevelResult(
            formulation=self.formulation,
            lambda_=self.effective_lambda,
            use_size_objective=self.use_size_objective,
            k=int(self.k),
            n=int(self.n),
            f1=np.asarray(f1, dtype=np.float64),
            f2=np.asarray(f2, dtype=np.float64),
            objective_values=obj_vals,
            iterations=int(getattr(result, "num_iterations", len(f1))),
            best_f1=float(np.nanmin(f1)) if np.isfinite(f1).any() else float("nan"),
            best_f2=float(f2[finite_best]) if f2.size else float("nan"),
            final_f1=float(getattr(result, "final_f1", f1[-1] if f1.size else np.nan)),
            final_f2=float(getattr(result, "final_f2", f2[-1] if f2.size else np.nan)),
            best_mask=getattr(result, "best_mask", None),
            final_mask=getattr(result, "final_mask", None),
            s=getattr(result, "s", None),
            expected_size=float(getattr(result, "expected_size", float("nan"))),
            stopped_reason=str(getattr(result, "stopped_reason", "completed")),
            num_evaluations=int(getattr(result, "num_evaluations", len(f1))),
            history=history,
            raw=result,
        )

    # -- path B: self-contained REINFORCE outer loop --------------------- #
    def _run_internal(self) -> TrivialBilevelResult:
        rng = np.random.default_rng(self.seed)
        s = self.initial_probabilities(rng)
        adam = _Adam(lr=self.outer_lr)
        n_eval = 0
        f1_hist: List[float] = []
        f2_hist: List[float] = []
        history: List[Dict[str, Any]] = []
        best_f1, best_mask = float("inf"), None

        for t in range(self.outer_iters):
            lr = _cosine_lr(t, self.outer_iters, self.outer_lr) if self.cosine else self.outer_lr
            grad = np.zeros(self.n, dtype=np.float64)
            step_f1: List[float] = []
            step_f2: List[float] = []
            for _ in range(self.samples_per_iter):
                mask = _bernoulli_sample(s, rng)
                f1_val, f2_val, _raw = _evaluate_objective(
                    self.objective, mask, self.train_kwargs, iteration=t
                )
                n_eval += 1
                value = combined_value(
                    f1_val, f2_val, self.effective_lambda, self.use_size_objective
                )
                grad += value * _score_function_gradient(mask, s, self.s_min)
                step_f1.append(float(f1_val))
                step_f2.append(float(f2_val))
                if f1_val < best_f1:
                    best_f1 = float(f1_val)
                    best_mask = mask.copy()
            grad /= float(self.samples_per_iter)

            s = adam.step(s, grad, lr=lr)
            if self.constrain_size:
                s = _project_expected_size(s, self.k, self.s_min, self.s_max)
            else:
                s = np.clip(s, self.s_min, self.s_max)

            f1_step = float(np.mean(step_f1)) if step_f1 else float("nan")
            f2_step = float(np.mean(step_f2)) if step_f2 else float("nan")
            f1_hist.append(f1_step)
            f2_hist.append(f2_step)
            history.append(
                {
                    "iteration": int(t),
                    "f1": f1_step,
                    "f2": f2_step,
                    "objective": combined_value(
                        f1_step, f2_step, self.effective_lambda, self.use_size_objective
                    ),
                    "expected_size": float(np.sum(s)),
                    "lr": float(lr),
                }
            )
            if self.log_every and (t + 1) % self.log_every == 0:
                LOGGER.info(
                    "[%s] iter %d/%d  f1=%.4f  f2=%.1f  E|m|=%.1f  lr=%.3g",
                    self.formulation, t + 1, self.outer_iters, f1_step, f2_step,
                    float(np.sum(s)), lr,
                )

        final_mask = self.mask_from_probabilities(s)
        final_f1 = f1_hist[-1] if f1_hist else float("nan")
        final_f2 = float(np.count_nonzero(final_mask))
        try:
            if np.isfinite(final_f1):
                final_f2 = final_f2
        except Exception:  # pragma: no cover - defensive
            pass

        return TrivialBilevelResult(
            formulation=self.formulation,
            lambda_=self.effective_lambda,
            use_size_objective=self.use_size_objective,
            k=int(self.k),
            n=int(self.n),
            f1=np.asarray(f1_hist, dtype=np.float64),
            f2=np.asarray(f2_hist, dtype=np.float64),
            objective_values=np.asarray(
                [
                    combined_value(f1_hist[i], f2_hist[i], self.effective_lambda, self.use_size_objective)
                    for i in range(len(f1_hist))
                ],
                dtype=np.float64,
            ),
            iterations=int(self.outer_iters),
            best_f1=float(best_f1) if np.isfinite(best_f1) else float("nan"),
            best_f2=float(np.count_nonzero(best_mask)) if best_mask is not None else float("nan"),
            final_f1=float(final_f1),
            final_f2=float(final_f2),
            best_mask=best_mask,
            final_mask=final_mask,
            s=s,
            expected_size=float(np.sum(s)),
            stopped_reason="completed",
            num_evaluations=int(n_eval),
            history=history,
        )

    # ------------------------------------------------------------------ #
    def mask_from_probabilities(self, s: Optional[np.ndarray] = None) -> np.ndarray:
        """Project the outer probabilities onto a binary coreset mask."""
        s = np.asarray(self.initial_probabilities() if s is None else s, dtype=np.float64).reshape(-1)
        if self.final_selection == "threshold":
            mask = (s >= 0.5).astype(np.float64)
            if not mask.any():  # degenerate collapse: keep the most probable entry
                mask = np.zeros_like(s)
                mask[int(np.argmax(s))] = 1.0
            return mask
        return indices_to_mask(topk_indices(s, min(int(self.k), s.size), largest=True), s.size)

    # ------------------------------------------------------------------ #
    def result(self) -> TrivialBilevelResult:
        if self.result_ is None:
            return self.run()
        return self.result_


# --------------------------------------------------------------------------- #
# Formulation-specific convenience classes
# --------------------------------------------------------------------------- #
class Eq3Baseline(WeightedBilevelBaseline):
    """Eq. (3): ``min_m f1(m)`` -- size fixed near the predefined ``k``."""

    def __init__(self, objective: Any, n: Optional[int] = None, k: Optional[int] = None, **kwargs: Any) -> None:
        kwargs.setdefault("formulation", "eq3")
        kwargs.setdefault("use_size_objective", False)
        kwargs.setdefault("constrain_size", True)
        kwargs.setdefault("final_selection", "topk")
        super().__init__(objective, n=n, k=k, **kwargs)


class Eq4Baseline(WeightedBilevelBaseline):
    """Eq. (4): ``min_m (1 - lambda) f1(m) + lambda f2(m)`` (``lambda = 1/2``)."""

    def __init__(
        self,
        objective: Any,
        n: Optional[int] = None,
        k: Optional[int] = None,
        lambda_: float = EQ4_LAMBDA,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("formulation", "eq4")
        kwargs.setdefault("use_size_objective", True)
        kwargs.setdefault("constrain_size", False)
        kwargs.setdefault("final_selection", "threshold")
        super().__init__(objective, n=n, k=k, lambda_=lambda_, **kwargs)


# --------------------------------------------------------------------------- #
# Functional entry points (Figure 1 curves)
# --------------------------------------------------------------------------- #
def weighted_bilevel(
    objective: Any,
    n: Optional[int] = None,
    k: Optional[int] = None,
    lambda_: float = DEFAULT_LAMBDA,
    formulation: str = "eq4",
    **kwargs: Any,
) -> TrivialBilevelResult:
    """Run Eq. (3) (``formulation="eq3"``) or Eq. (4) (``formulation="eq4"``)."""
    cls = Eq3Baseline if str(formulation).lower() in ("eq3", "3") else Eq4Baseline
    if cls is Eq4Baseline:
        return cls(objective, n=n, k=k, lambda_=lambda_, **kwargs).run()
    return cls(objective, n=n, k=k, **kwargs).run()


def trivial_bilevel_curves(
    objective: Any,
    formulation: str = "eq3",
    n: Optional[int] = None,
    k: Optional[int] = None,
    lambda_: float = DEFAULT_LAMBDA,
    outer_iters: int = FIGURE1_OUTER_ITERS,
    outer_lr: float = FIGURE1_OUTER_LR,
    train_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    **kwargs: Any,
) -> TrivialBilevelResult:
    """Figure 1 curves for the requested trivial formulation.

    Defaults follow Appendix C.3: inner loop 100 epochs SGD (lr 0.1, momentum
    0.9); outer loop Adam (lr 2.5) with a cosine scheduler; ``T = 1000``
    outer iterations.
    """
    default_train = {
        "inner_epochs": FIGURE1_INNER_EPOCHS,
        "inner_lr": FIGURE1_INNER_LR,
        "inner_optimizer": "sgd",
        "inner_momentum": FIGURE1_INNER_MOMENTUM,
    }
    merged_train = dict(default_train)
    merged_train.update(train_kwargs or {})
    return weighted_bilevel(
        objective,
        n=n,
        k=k,
        lambda_=lambda_,
        formulation=formulation,
        outer_iters=outer_iters,
        outer_lr=outer_lr,
        train_kwargs=merged_train,
        seed=seed,
        **kwargs,
    )


def eq3_curves(objective: Any, **kwargs: Any) -> TrivialBilevelResult:
    """``f1``/``f2`` vs. outer iterations with Eq. (3) (Figure 1(a)-(b))."""
    kwargs.pop("formulation", None)
    return trivial_bilevel_curves(objective, formulation="eq3", **kwargs)


def eq4_curves(
    objective: Any,
    lambda_: float = EQ4_LAMBDA,
    **kwargs: Any,
) -> TrivialBilevelResult:
    """``f1``/``f2`` vs. outer iterations with Eq. (4) (Figure 1(c)-(d))."""
    kwargs.pop("formulation", None)
    return trivial_bilevel_curves(objective, formulation="eq4", lambda_=lambda_, **kwargs)


def figure1_comparison(
    objective: Any,
    n: Optional[int] = None,
    k: Optional[int] = None,
    lambda_: float = EQ4_LAMBDA,
    outer_iters: int = FIGURE1_OUTER_ITERS,
    seed: int = 0,
    train_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, TrivialBilevelResult]:
    """Run both trivial formulations and return the four Figure 1 curves."""
    out: Dict[str, TrivialBilevelResult] = {}
    for i, form in enumerate(("eq3", "eq4")):
        out[form] = trivial_bilevel_curves(
            objective,
            formulation=form,
            n=n,
            k=k,
            lambda_=lambda_,
            outer_iters=outer_iters,
            seed=resolve_seed(seed, i),
            train_kwargs=train_kwargs,
            **kwargs,
        )
    return out


# --------------------------------------------------------------------------- #
# §5.2-style selector wrappers
# --------------------------------------------------------------------------- #
class WeightedBilevelSelector(ScoreBaseline):
    """``ScoreBaseline`` wrapper around :class:`WeightedBilevelBaseline`.

    The learned outer probabilities ``s`` are exposed as scores (higher is
    better), so the inherited top-``k`` machinery yields the Eq. (3) mask.
    """

    name = "WeightedBilevel"
    abbreviation = "Eq.(4)"
    requires_model = True
    higher_is_better = True

    def __init__(
        self,
        model: Any = None,
        objective: Any = None,
        k: Optional[int] = None,
        lambda_: float = DEFAULT_LAMBDA,
        formulation: Optional[str] = None,
        use_size_objective: Optional[bool] = None,
        outer_iters: int = FIGURE1_OUTER_ITERS,
        outer_lr: float = FIGURE1_OUTER_LR,
        cosmetic: bool = True,  # unused; kept for signature compatibility
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=kwargs.pop("seed", None), device=kwargs.pop("device", None))
        self.model = model
        self.objective = objective
        self.k_ = k
        self.lambda_ = float(lambda_)
        self.formulation = formulation
        self.use_size_objective = use_size_objective
        self.outer_iters = int(outer_iters)
        self.outer_lr = float(outer_lr)
        self.extra_kwargs = dict(kwargs)
        self._result: Optional[TrivialBilevelResult] = None
        self._s: Optional[np.ndarray] = None

    # -- internal ------------------------------------------------------------ #
    def _run(
        self,
        n: int,
        k: Optional[int] = None,
        objective: Any = None,
        dataset: Any = None,
        targets: Any = None,
        **kwargs: Any,
    ) -> TrivialBilevelResult:
        obj = objective if objective is not None else self.objective
        if obj is None:
            raise ValueError(
                "WeightedBilevelSelector requires a mask-evaluating `objective` "
                "(e.g. lbcs.objectives.MaskObjectiveEvaluator)."
            )
        k_eff = int(k if k is not None else (self.k_ if self.k_ is not None else max(1, n // 2)))
        runner_cls = Eq3Baseline if self.formulation == "eq3" else (
            Eq4Baseline if self.formulation == "eq4" else WeightedBilevelBaseline
        )
        runner_kwargs: Dict[str, Any] = dict(self.extra_kwargs)
        runner_kwargs.update(kwargs)
        runner_kwargs.setdefault("outer_iters", self.outer_iters)
        runner_kwargs.setdefault("outer_lr", self.outer_lr)
        runner_kwargs.setdefault("seed", self.seed)
        if runner_cls is Eq4Baseline:
            runner = runner_cls(obj, n=n, k=k_eff, lambda_=self.lambda_, **runner_kwargs)
        elif runner_cls is Eq3Baseline:
            runner = runner_cls(obj, n=n, k=k_eff, **runner_kwargs)
        else:
            runner = runner_cls(
                obj,
                n=n,
                k=k_eff,
                lambda_=self.lambda_,
                formulation=self.formulation,
                use_size_objective=self.use_size_objective,
                **runner_kwargs,
            )
        self._result = runner.run()
        s = self._result.s
        self._s = None if s is None else np.asarray(s, dtype=np.float64).reshape(-1)
        return self._result

    # -- ScoreBaseline API --------------------------------------------------- #
    def compute_scores(
        self,
        dataset: Any = None,
        targets: Any = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        model: Any = None,
        loader: Any = None,
        train_loader: Any = None,
        num_classes: Optional[int] = None,
        objective: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        n_eff = int(n) if n is not None else (dataset if isinstance(dataset, int) else None)
        if n_eff is None:
            n_eff = _infer_n(objective if objective is not None else self.objective, self.k_)
        res = self._run(n_eff, k=kwargs.pop("k", None), objective=objective, **kwargs)
        if self._s is not None and self._s.size == n_eff:
            return self._s
        return np.asarray(res.final_mask, dtype=np.float64)

    def select_indices(
        self,
        n: int,
        k: int,
        dataset: Any = None,
        targets: Any = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        objective: Any = None,
        return_scores: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        if scores is None:
            scores = self.compute_scores(
                dataset=dataset, targets=targets, n=n, seed=seed, objective=objective, **kwargs
            )
        if kwargs.get("final_selection") == "threshold" or (
            self.formulation == "eq4" and kwargs.get("threshold_mask", False)
        ):
            idx = np.flatnonzero(np.asarray(scores).reshape(-1) >= 0.5)
            if idx.size == 0:
                idx = topk_indices(scores, min(int(k), n), largest=True)
            return (idx, scores) if return_scores else idx
        idx = topk_indices(scores, min(int(k), int(n)), largest=True)
        return (idx, scores) if return_scores else idx

    def select_mask(
        self,
        n: int,
        k: int,
        dataset: Any = None,
        targets: Any = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        scores: Optional[np.ndarray] = None,
        objective: Any = None,
        return_indices: bool = False,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> np.ndarray:
        idx = self.select_indices(
            n, k, dataset=dataset, targets=targets, num_classes=num_classes,
            seed=seed, scores=scores, objective=objective, **kwargs
        )
        mask = indices_to_mask(idx, int(n), dtype=dtype)
        return (mask, idx) if return_indices else mask

    mask = select_mask

    @classmethod
    def figure1_config(cls) -> Dict[str, Any]:
        """Paper-stated Figure 1 settings (Appendix C.3)."""
        return {
            "outer_iters": FIGURE1_OUTER_ITERS,
            "outer_lr": FIGURE1_OUTER_LR,
            "outer_optimizer": FIGURE1_OUTER_OPTIMIZER,
            "outer_scheduler": FIGURE1_OUTER_SCHEDULER,
            "inner_epochs": FIGURE1_INNER_EPOCHS,
            "inner_lr": FIGURE1_INNER_LR,
            "inner_momentum": FIGURE1_INNER_MOMENTUM,
            "lambda": EQ4_LAMBDA,
        }


class Eq3Selector(WeightedBilevelSelector):
    """Eq. (3) as a fixed-size selector for §5.2-style comparisons."""

    name = "Eq.(3)"
    abbreviation = "Eq.(3)"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formulation", "eq3")
        kwargs.setdefault("use_size_objective", False)
        super().__init__(*args, **kwargs)


class Eq4Selector(WeightedBilevelSelector):
    """Eq. (4) as a selector (``lambda = 1/2``)."""

    name = "Eq.(4)"
    abbreviation = "Eq.(4)"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formulation", "eq4")
        kwargs.setdefault("use_size_objective", True)
        super().__init__(*args, **kwargs)


# --------------------------------------------------------------------------- #
# Offline self-test
# --------------------------------------------------------------------------- #
class _SyntheticObjective:
    """Tiny stand-in objective: f1 decreases with more/``better`` selected data.

    ``f1(m) = base - alpha * (sum_{i in m} w_i) / max(1, ||m||_0)`` with fixed
    example ``w_i`` and ``base`` chosen so that fewer selected examples degrade
    performance; this lets the outer loop be exercised without any network.
    """

    def __init__(self, n: int, k: int, seed: int = 0) -> None:
        self.n = int(n)
        self.k = int(k)
        rng = np.random.default_rng(seed)
        self.w = rng.random(n)
        self.base = 3.0
        self.alpha = 2.0
        self.evaluations = 0

    def evaluate(self, mask: np.ndarray, **kwargs: Any) -> Dict[str, float]:
        m = np.asarray(mask, dtype=np.float64).reshape(-1)
        sel = np.flatnonzero(m > 0.5)
        self.evaluations += 1
        if sel.size == 0:
            return {"f1": float(self.base * 2.0), "f2": 0.0, "size": 0.0}
        quality = float(np.mean(self.w[sel]))
        f1 = float(self.base - self.alpha * quality)
        return {"f1": f1, "f2": float(sel.size), "size": float(sel.size)}


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks for the Eq. (3) / Eq. (4) trivial baselines."""
    info: Dict[str, Any] = {}

    # -- objective combinations -------------------------------------------- #
    assert eq3_value(2.0, 100.0) == 2.0
    assert abs(eq4_value(2.0, 10.0, 0.5) - 6.0) < 1e-12
    assert abs(combined_value(2.0, 10.0, 0.5, True) - 6.0) < 1e-12
    assert abs(combined_value(2.0, 10.0, 0.5, False) - 2.0) < 1e-12
    info["objective_combinations"] = True

    # -- score-function gradient (Appendix C.1) ---------------------------- #
    s = np.array([0.3, 0.7])
    m = np.array([1.0, 0.0])
    g = _score_function_gradient(m, s)
    assert np.allclose(g, (m - s) / (s * (1.0 - s)))
    info["score_function_gradient"] = True

    # -- expected-size projection ------------------------------------------ #
    s0 = np.full(10, 0.5)
    sp = _project_expected_size(s0, 4)
    assert abs(sp.sum() - 4.0) < 1e-3, sp.sum()
    info["expected_size_projection"] = float(sp.sum())

    # -- mask projection --------------------------------------------------- #
    runner = Eq3Baseline(_SyntheticObjective(20, 6), n=20, k=6, outer_iters=1)
    mask = runner.mask_from_probabilities(np.full(20, 0.5))
    assert mask.sum() == 6
    runner4 = Eq4Baseline(_SyntheticObjective(20, 6), n=20, k=6, outer_iters=1)
    mask4 = runner4.mask_from_probabilities(np.array([0.9] + [0.1] * 19))
    assert mask4.sum() == 1
    info["mask_projection"] = True

    # -- Eq. (3): fixed size, f1 improves ---------------------------------- #
    obj3 = _SyntheticObjective(30, 10, seed=1)
    res3 = Eq3Baseline(
        obj3, n=30, k=10, outer_iters=40, outer_lr=0.5, cosine=True, variant="internal", seed=0
    ).run()
    assert np.asarray(res3.f2).size == 40
    assert res3.f2[-1] <= 12, res3.f2[-1]  # size stays near the predefined k
    assert res3.f1[-1] <= res3.f1[0] + 1e-6  # f1 decreases (Figure 1(a))
    info["eq3_curves"] = {"f1_first": float(res3.f1[0]), "f1_last": float(res3.f1[-1]), "f2_last": float(res3.f2[-1])}

    # -- Eq. (4) with lambda = 1/2: size collapses ------------------------- #
    obj4 = _SyntheticObjective(30, 10, seed=1)
    res4 = Eq4Baseline(
        obj4, n=30, k=10, outer_iters=40, outer_lr=0.2, cosine=True, variant="internal", seed=0
    ).run()
    final_size = float(np.count_nonzero(res4.final_mask))
    assert final_size < 10, final_size  # f2 over-minimized (Figure 1(d))
    info["eq4_curves"] = {"final_size": final_size, "k": 10, "f1_last": float(res4.f1[-1])}

    # -- diagnostics -------------------------------------------------------- #
    diag3 = res3.diagnostics()
    diag4 = res4.diagnostics()
    info["diagnostics_eq3"] = diag3
    info["diagnostics_eq4"] = diag4
    assert diag3["fixed_size"] is True, diag3
    assert diag4["over_minimized"] is True, diag4

    # -- functional wrappers ------------------------------------------------ #
    res_w = weighted_bilevel(_SyntheticObjective(12, 4, seed=2), n=12, k=4, formulation="eq4",
                             outer_iters=3, variant="internal", seed=0)
    assert isinstance(res_w, TrivialBilevelResult)
    d = res_w.to_dict()
    assert d["formulation"] == "eq4" and d["k"] == 4
    curves = figure1_comparison(_SyntheticObjective(12, 4, seed=3), n=12, k=4, outer_iters=2,
                                variant="internal", seed=0)
    assert set(curves) == {"eq3", "eq4"}
    info["functional_api"] = True

    if verbose:
        print("[weighted_bilevel] self-test passed")
        for key, value in info.items():
            print(f"  - {key}: {value}")
    return info


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
