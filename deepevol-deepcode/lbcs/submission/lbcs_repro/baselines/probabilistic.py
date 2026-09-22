"""Probabilistic bilevel coreset selection baseline (Zhou et al., ICML 2022).

This module implements the probabilistic (continuous) bilevel coreset baseline that the
paper uses both

  * as the comparator of Figure 1 (the trivial-solution study of §2.1), and
  * as one of the competitors in the §5.2 comparison (Appendix D.1).

Paper specification (Appendix C.1, "Method Description")
--------------------------------------------------------
The binary mask entry ``m_i`` is reparameterized as a Bernoulli random variable

    m_i ~ Bern(s_i),      s_i in [0, 1],

and, assuming independence of the ``m_i``,

    p(m | s) = prod_i s_i^{m_i} (1 - s_i)^{1 - m_i},
    E_{m ~ p(m|s)} ||m||_0 = sum_i s_i = 1^T s .

Two outer problems are considered:

    (1)  min_s E_{p(m|s)} f_1(m)                     s.t. theta(m) in argmin L(m, theta)
    (2)  min_s E_{p(m|s)} f_1(m) + E_{p(m|s)} f_2(m) s.t. theta(m) in argmin L(m, theta)

Eq. (2) is exactly the ``lambda = 1/2`` instance of the weighted problem (4) of §2.1,
i.e. ``(1 - lambda) f_1(m) + lambda f_2(m)`` with ``lambda = 1/2``.

Paper specification (Appendix C.2, "Gradient Analysis")
-------------------------------------------------------
The outer gradient of (2) is

    grad_s [ E f_1 + E ||m||_0 ]
        = E_{p(m|s)} f_1(m) grad_s ln p(m|s)  +  1 ,

and the score-function (unbiased policy-gradient) term factorizes as

    f_1(m) grad_s ln p(m|s)
        = f_1(m) ( m/s - (1-m)/(1-s) )
        = f_1(m) ( m - s ) / ( s (1 - s) ) .

Following Zhou et al. (2022) the estimator ``f_1(m) grad_s ln p(m|s)`` is used directly
because it is an unbiased stochastic gradient of ``grad_s E f_1(m)``.

The corresponding gradient norms (eq. (30) of Appendix C.2) are

    zeta_1(lambda) = (1 - lambda) || f_1(m) (m - s) / (s (1 - s)) ||_2
    zeta_2(lambda) = lambda * sqrt(n)

so that for ``lambda = 1/2`` the size term has norm ``sqrt(n)/2``, which is huge for
coreset-selection sized problems -- this is the paper's explanation of why the equal
weighting over-minimizes the coreset size (Figure 1(c)-(d)).

Paper specification (Appendix C.3, Figure 1 settings)
-----------------------------------------------------
Inner loop: 100 epochs SGD, lr = 0.1, momentum = 0.9.
Outer loop: Adam, lr = 2.5, cosine scheduler.  (T = 1000 in Figure 1.)

Paper specification (Appendix D.1)
----------------------------------
"Probabilistic coreset (ICML 2022) (Zhou et al., 2022). The method proposes continuous
probabilistic bilevel optimization for coreset selection. A solver is developed for the
bilevel optimization problem via unbiased policy gradient without the trouble of implicit
differentiation."

SUGGESTED (non paper-stated) defaults are explicitly labelled below.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .base import (
    ArrayLike,
    ScoreBaseline,
    indices_to_mask,
)

try:  # torch is a soft dependency (mask algebra works without it)
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch
    _TORCH_AVAILABLE = False


LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Paper-stated / suggested constants
# --------------------------------------------------------------------------------------

#: Appendix C.3 (Figure 1): outer-loop optimizer settings.
FIGURE1_OUTER_LR = 2.5
FIGURE1_OUTER_ITERS = 1000
FIGURE1_OUTER_OPTIMIZER = "adam"
FIGURE1_OUTER_SCHEDULER = "cosine"

#: Appendix C.3 (Figure 1) / §2.1: inner-loop settings (Zhou et al. 2022).
FIGURE1_INNER_EPOCHS = 100
FIGURE1_INNER_LR = 0.1
FIGURE1_INNER_MOMENTUM = 0.9

#: §5.2: inner loop uses Adam with learning rate 0.001.
SECTION52_INNER_LR = 0.001
SECTION52_INNER_OPTIMIZER = "adam"

#: §2.1 eq. (4): the equal weight that makes the size minimization salient.
DEFAULT_LAMBDA = 0.5

#: SUGGESTED defaults (the paper does not state these numerically).
SUGGESTED_OUTER_ITERS = 500
SUGGESTED_SAMPLES_PER_ITER = 1
SUGGESTED_S_MIN = 1e-3
SUGGESTED_S_MAX = 1.0 - 1e-3
SUGGESTED_ADAM_BETA1 = 0.9
SUGGESTED_ADAM_BETA2 = 0.999
SUGGESTED_ADAM_EPS = 1e-8
SUGGESTED_INIT_NOISE = 0.0


# --------------------------------------------------------------------------------------
# Core probability algebra (Appendix C.1 / C.2)
# --------------------------------------------------------------------------------------


def _as_float_array(x: Any, dtype: float = np.float64) -> np.ndarray:
    """Coerce masks / probabilities to a 1-D ``float`` numpy array (no copy if possible)."""
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=dtype).reshape(-1)
    if np.isscalar(x):
        return np.asarray([x], dtype=dtype)
    if _TORCH_AVAILABLE:
        try:  # pragma: no cover - container dependent
            import torch as _torch

            if isinstance(x, _torch.Tensor):
                return x.detach().cpu().numpy().astype(dtype, copy=False).reshape(-1)
        except Exception:
            pass
    return np.asarray(list(x), dtype=dtype).reshape(-1)


def clip_probabilities(
    s: ArrayLike,
    s_min: float = SUGGESTED_S_MIN,
    s_max: float = SUGGESTED_S_MAX,
) -> np.ndarray:
    """Numerically safe projection of probabilities onto ``[s_min, s_max]``.

    ``s_min``/``s_max`` are SUGGESTED (the paper only requires ``s_i in [0, 1]``); they
    keep ``s(1-s)`` away from zero in the score-function gradient of eq. (31).
    """
    return np.clip(_as_float_array(s), float(s_min), float(s_max))


def bernoulli_probabilities(n: int, k: Optional[int] = None, seed: Optional[int] = None,
                            init_noise: float = SUGGESTED_INIT_NOISE,
                            s_min: float = SUGGESTED_S_MIN,
                            s_max: float = SUGGESTED_S_MAX,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Initialize ``s`` with expected coreset size ``k`` (uniform ``s_i = k / n``).

    Corresponds to initializing the Bernoulli probabilities so that
    ``E ||m||_0 = 1^T s = k`` (Appendix C.1).
    """
    n = int(n)
    k = int(n // 2) if k is None else int(k)
    p = float(np.clip(k / max(n, 1), 0.0, 1.0))
    gen = rng if rng is not None else np.random.default_rng(seed)
    s = np.full(n, p, dtype=np.float64)
    if init_noise and float(init_noise) > 0.0:
        s = s + gen.normal(0.0, float(init_noise), size=n)
    return clip_probabilities(s, s_min=s_min, s_max=s_max)


def sample_mask(s: ArrayLike, seed: Optional[int] = None,
                rng: Optional[np.random.Generator] = None,
                ensure_nonempty: bool = True,
                s_min: float = SUGGESTED_S_MIN,
                s_max: float = SUGGESTED_S_MAX) -> np.ndarray:
    """Sample a binary mask ``m ~ p(m | s) = prod_i s_i^{m_i}(1-s_i)^{1-m_i}``."""
    s_arr = clip_probabilities(s, s_min=s_min, s_max=s_max)
    gen = rng if rng is not None else np.random.default_rng(seed)
    m = (gen.random(s_arr.shape[0]) < s_arr).astype(np.float64)
    if ensure_nonempty and not m.any() and s_arr.size:
        m[int(np.argmax(s_arr))] = 1.0
    return m


def sample_masks(s: ArrayLike, num_samples: int = 1, seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None,
                 **kwargs: Any) -> np.ndarray:
    """Sample ``num_samples`` masks from ``p(m | s)``; returns an array of shape ``(S, n)``."""
    gen = rng if rng is not None else np.random.default_rng(seed)
    num_samples = max(1, int(num_samples))
    return np.stack([sample_mask(s, rng=gen, **kwargs) for _ in range(num_samples)], axis=0)


def mask_probability(m: ArrayLike, s: ArrayLike, eps: float = 0.0) -> float:
    """``p(m | s) = prod_i s_i^{m_i} (1 - s_i)^{1 - m_i}`` (Appendix C.1)."""
    m_arr = _as_float_array(m)
    s_arr = np.clip(_as_float_array(s), eps, 1.0 - eps) if eps else _as_float_array(s)
    return float(np.prod(np.where(m_arr > 0, s_arr, 1.0 - s_arr)))


def log_prob(m: ArrayLike, s: ArrayLike, eps: float = 1e-12) -> float:
    """``ln p(m | s) = sum_i [ m_i ln s_i + (1 - m_i) ln(1 - s_i) ]`` (Appendix C.1)."""
    m_arr = _as_float_array(m)
    s_arr = np.clip(_as_float_array(s), eps, 1.0 - eps)
    return float(np.sum(m_arr * np.log(s_arr) + (1.0 - m_arr) * np.log(1.0 - s_arr)))


def score_function_log_prob_gradient(m: ArrayLike, s: ArrayLike,
                                     eps: float = 1e-12) -> np.ndarray:
    """``grad_s ln p(m | s) = m/s - (1-m)/(1-s) = (m - s) / (s (1 - s))`` (eq. 31)."""
    m_arr = _as_float_array(m)
    s_arr = np.clip(_as_float_array(s), eps, 1.0 - eps)
    return (m_arr - s_arr) / (s_arr * (1.0 - s_arr))


def score_function_gradient(f1_value: float, m: ArrayLike, s: ArrayLike,
                            eps: float = 1e-12) -> np.ndarray:
    """``f_1(m) grad_s ln p(m | s) = f_1(m) (m - s) / (s (1 - s))`` (eq. 31).

    This is the unbiased policy-gradient estimator of ``grad_s E f_1(m)`` used by
    Zhou et al. (2022) (Appendix C.2).
    """
    return float(f1_value) * score_function_log_prob_gradient(m, s, eps=eps)


def objective_score_gradient(f1_value: float, m: ArrayLike, s: ArrayLike,
                             lambda_: float = DEFAULT_LAMBDA,
                             with_size_term: bool = True,
                             eps: float = 1e-12) -> np.ndarray:
    """Unbiased gradient estimator of eq. (2) / eq. (4) of the paper.

    ``grad_s [ (1 - lambda) E f_1(m) + lambda E ||m||_0 ]
          = (1 - lambda) f_1(m) (m - s)/(s(1-s)) + lambda * 1``

    With ``with_size_term=False`` the estimator reduces to the original Zhou et al.
    (2022) objective ``E f_1(m)`` (eq. (1) of Appendix C.1).
    """
    grad = (1.0 - float(lambda_)) * score_function_gradient(f1_value, m, s, eps=eps)
    if with_size_term:
        grad = grad + float(lambda_) * np.ones_like(grad)
    return grad


def expected_coreset_size(s: ArrayLike) -> float:
    """``E_{m ~ p(m|s)} ||m||_0 = 1^T s`` (Appendix C.1)."""
    return float(np.sum(_as_float_array(s)))


def zeta1(f1_value: float, m: ArrayLike, s: ArrayLike, lambda_: float = DEFAULT_LAMBDA,
          eps: float = 1e-12) -> float:
    """``zeta_1(lambda) = (1 - lambda) || f_1(m) (m - s)/(s (1 - s)) ||_2`` (eq. 30)."""
    grad = score_function_gradient(f1_value, m, s, eps=eps)
    return float(1.0 - float(lambda_)) * float(np.linalg.norm(grad))


def zeta2(n: int, lambda_: float = DEFAULT_LAMBDA) -> float:
    """``zeta_2(lambda) = lambda * sqrt(n)`` (eq. 30); ``= sqrt(n)/2`` at ``lambda=1/2``."""
    return float(lambda_) * math.sqrt(float(n))


def gradient_norm_analysis(n: int, f1_value: float = 0.0, m: ArrayLike = None,
                           s: ArrayLike = None, lambda_: float = DEFAULT_LAMBDA,
                           eps: float = 1e-12) -> Dict[str, float]:
    """Bundle ``zeta_1``, ``zeta_2`` and their ratio (Appendix C.2 explanation).

    ``m``/``s`` are optional: when omitted only the data-dependent size term is returned,
    which is the quantity that already explains the over-minimization at ``lambda = 1/2``.
    """
    out: Dict[str, float] = {
        "lambda": float(lambda_),
        "zeta2": zeta2(n, lambda_),
        "zeta2_lambda_half": zeta2(n, 0.5),
    }
    if m is not None and s is not None:
        out["zeta1"] = zeta1(f1_value, m, s, lambda_=lambda_, eps=eps)
        out["ratio_zeta2_over_zeta1"] = (
            out["zeta2"] / out["zeta1"] if out["zeta1"] > 1e-12 else float("inf")
        )
    return out


def project_to_size(s: ArrayLike, k: int, s_min: float = 0.0, s_max: float = 1.0,
                    tol: float = 1e-9, max_iter: int = 200) -> np.ndarray:
    """Euclidean projection of ``s`` onto ``{s : 1^T s = k, s_min <= s <= s_max}``.

    Solved by bisection on the shift ``tau``: the projector is
    ``clip(s - tau, s_min, s_max)`` because the feasible set is a box intersected with a
    hyperplane.  Used for the fixed-``k`` (Zhou et al.) variant where the coreset size
    must remain at the predefined ``k`` (SUGGESTED mechanism; the paper only states that
    the size can be *controlled* by ``1^T s``).
    """
    s_arr = _as_float_array(s)
    n = s_arr.size
    k = int(np.clip(int(k), 0, n))
    if n == 0:
        return s_arr
    lo, hi = float(s_arr.min() - s_max), float(s_arr.max() - s_min)
    for _ in range(int(max_iter)):
        tau = 0.5 * (lo + hi)
        total = float(np.sum(np.clip(s_arr - tau, s_min, s_max)))
        if total > k:
            lo = tau
        else:
            hi = tau
        if abs(total - k) < tol:
            break
    return np.clip(s_arr - 0.5 * (lo + hi), s_min, s_max)


def cosine_lr(step: int, total: int, base_lr: float, eta_min: float = 0.0) -> float:
    """Cosine annealing from ``base_lr`` (Appendix C.3: "cosine scheduler")."""
    total = max(1, int(total))
    step = int(np.clip(step, 0, total))
    return float(eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * step / total)))


class AdamState:
    """Minimal NumPy Adam optimizer with optional cosine schedule."""

    def __init__(self, lr: float = FIGURE1_OUTER_LR, beta1: float = SUGGESTED_ADAM_BETA1,
                 beta2: float = SUGGESTED_ADAM_BETA2, eps: float = SUGGESTED_ADAM_EPS,
                 cosine: bool = True, total_steps: int = 1, eta_min: float = 0.0) -> None:
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.cosine = bool(cosine)
        self.total_steps = max(1, int(total_steps))
        self.eta_min = float(eta_min)
        self.m: Optional[np.ndarray] = None
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        grad = _as_float_array(grad)
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        lr = (cosine_lr(self.t, self.total_steps, self.lr, self.eta_min)
              if self.cosine else self.lr)
        return _as_float_array(params) - lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def state_dict(self) -> Dict[str, Any]:
        return {"m": self.m, "v": self.v, "t": self.t, "lr": self.lr}


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class ProbabilisticResult:
    """Outcome of a probabilistic bilevel coreset-selection run."""

    s: np.ndarray                       # final Bernoulli probabilities
    best_mask: np.ndarray               # best binary mask found (by f1)
    best_f1: float                      # f1 of best_mask (full-data objective)
    best_f2: float                      # f2 of best_mask = ||m||_0
    final_mask: Optional[np.ndarray] = None   # mask induced by s (top-k / thresholded)
    final_f1: float = float("nan")
    final_f2: float = float("nan")
    expected_size: float = float("nan")  # 1^T s
    num_iterations: int = 0
    num_evaluations: int = 0
    objective: str = "eq1"              # "eq1" (Zhou et al.) or "eq2" (with f2 / lambda)
    lambda_: float = DEFAULT_LAMBDA
    wall_time: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)
    stopped_reason: str = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_f1": float(self.best_f1),
            "best_f2": float(self.best_f2),
            "final_f1": float(self.final_f1),
            "final_f2": float(self.final_f2),
            "expected_size": float(self.expected_size),
            "num_iterations": int(self.num_iterations),
            "num_evaluations": int(self.num_evaluations),
            "objective": self.objective,
            "lambda": float(self.lambda_),
            "wall_time": float(self.wall_time),
            "stopped_reason": self.stopped_reason,
        }

    def curve(self, key: str = "f1") -> np.ndarray:
        """Extract one curve (``f1``, ``f1_best``, ``f2``, ``expected_size``) from history."""
        return np.asarray([h[key] for h in self.history], dtype=float)


# --------------------------------------------------------------------------------------
# Objective plumbing
# --------------------------------------------------------------------------------------


def _extract_f1(value: Any) -> float:
    """Extract ``f_1`` from whatever the provided objective callable returns."""
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        for key in ("f1", "f_1", "loss", "value"):
            if key in value:
                return float(value[key])
        raise ValueError("objective dict has no 'f1' entry")
    if hasattr(value, "f1"):
        return float(value.f1)
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) > 0:
        return float(value[0])
    return float(value)


def evaluate_mask(objective: Callable[..., Any], mask: np.ndarray,
                  train_kwargs: Optional[Dict[str, Any]] = None) -> Tuple[float, Any]:
    """Evaluate ``f_1(m)`` for a binary mask using a user-supplied objective callable.

    Three calling conventions are supported:

    * ``objective(mask)`` (or ``objective(mask, **train_kwargs)``) returning ``f1`` or an
      object exposing ``.f1`` (e.g. ``lbcs.objectives.MaskEvaluation``);
    * an object exposing ``.evaluate(mask, **kwargs)`` such as
      ``lbcs.objectives.MaskObjectiveEvaluator``.

    Returns ``(f1, raw_result)``.
    """
    kwargs = dict(train_kwargs or {})
    try:
        raw = objective(mask, **kwargs)
    except TypeError:
        raw = objective(mask)
    return _extract_f1(raw), raw


# --------------------------------------------------------------------------------------
# The outer loop (Zhou et al. 2022; eq. (1) / eq. (2) of Appendix C.1)
# --------------------------------------------------------------------------------------


def probabilistic_bilevel(
    objective: Callable[..., Any],
    n: int,
    k: Optional[int] = None,
    *,
    lambda_: float = DEFAULT_LAMBDA,
    use_size_objective: bool = True,
    outer_iters: int = SUGGESTED_OUTER_ITERS,
    outer_lr: float = FIGURE1_OUTER_LR,
    cosine: bool = True,
    samples_per_iter: int = SUGGESTED_SAMPLES_PER_ITER,
    s_init: Optional[ArrayLike] = None,
    init_noise: float = SUGGESTED_INIT_NOISE,
    s_min: float = SUGGESTED_S_MIN,
    s_max: float = SUGGESTED_S_MAX,
    constrain_size: bool = False,
    final_selection: str = "topk",
    train_kwargs: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
    log_every: int = 0,
    track_history: bool = True,
    extra_objective: Optional[Callable[..., Any]] = None,
) -> ProbabilisticResult:
    """Probabilistic bilevel coreset selection with an unbiased policy gradient.

    Implements the outer optimization of Appendix C.1 eq. (1) (``use_size_objective=False``,
    the Zhou et al. 2022 objective) and eq. (2) (``use_size_objective=True``, ``lambda=1/2``,
    i.e. the modified objective (4) of §2.1).

    Parameters
    ----------
    objective:
        Callable returning ``f_1(m)`` for a binary mask ``m`` (typically a
        :class:`lbcs_repro.lbcs.objectives.MaskObjectiveEvaluator`).  The inner problem
        ``theta(m) in argmin_theta L(m, theta)`` is solved inside this callable.
    n:
        Number of candidate examples (``|D|``).
    k:
        Predefined coreset size used to initialize ``s`` (``1^T s = k``).
    lambda_:
        Weight of the size term in eq. (4); ``0.5`` is the value studied in §2.1.
    use_size_objective:
        ``False`` -> eq. (1) of Appendix C.1 (pure ``E f_1``, no ``+1`` in the gradient).
    outer_iters:
        ``T``: 1000 for Figure 1 (Appendix C.3), 500 suggested for §5.2.
    outer_lr:
        Adam learning rate for the outer loop: 2.5 for Figure 1 (Appendix C.3).
    cosine:
        Use the cosine scheduler of Appendix C.3.
    constrain_size:
        Keep ``1^T s = k`` after every step (fixed-size variant); SUGGESTED, since the
        paper only states the size is controllable via ``1^T s``.
    final_selection:
        ``"topk"`` (deterministic top-``k`` of ``s``) or ``"threshold"`` (``s >= 0.5``),
        used to emit the mask handed to downstream target-model training.
    extra_objective:
        Optional callable returning the secondary objective ``f_2(m)`` when the provided
        ``objective`` only returns ``f_1``.  By default ``f_2(m) = ||m||_0``.
    """
    n = int(n)
    gen = np.random.default_rng(seed)
    t0 = time.time()

    s = (clip_probabilities(s_init, s_min=s_min, s_max=s_max) if s_init is not None
         else bernoulli_probabilities(n, k=k, rng=gen, init_noise=init_noise,
                                      s_min=s_min, s_max=s_max))
    if constrain_size and k is not None:
        s = project_to_size(s, int(k), s_min=s_min, s_max=s_max)

    opt = AdamState(lr=outer_lr, cosine=cosine, total_steps=max(1, int(outer_iters)))

    history: List[Dict[str, float]] = []
    best_f1 = float("inf")
    best_mask: Optional[np.ndarray] = None
    best_f2 = float("nan")
    num_evals = 0
    stopped_reason = "completed"
    obj_name = "eq2" if use_size_objective else "eq1"
    batch = max(1, int(samples_per_iter))

    for t in range(int(outer_iters)):
        grad_accum = np.zeros(n, dtype=np.float64)
        f1_batch: List[float] = []
        f2_batch: List[float] = []
        for _ in range(batch):
            m = sample_mask(s, rng=gen, s_min=s_min, s_max=s_max)
            f1_val, raw = evaluate_mask(objective, m, train_kwargs=train_kwargs)
            num_evals += 1
            if extra_objective is not None:
                f2_val = float(_extract_f1(extra_objective(m, raw=raw)))
            else:
                f2_val = float(np.count_nonzero(m))
            f1_batch.append(f1_val)
            f2_batch.append(f2_val)
            grad_accum += objective_score_gradient(
                f1_val, m, s, lambda_=lambda_, with_size_term=use_size_objective
            )
            if np.isfinite(f1_val) and f1_val < best_f1:
                best_f1 = float(f1_val)
                best_mask = m.copy()
                best_f2 = float(f2_val)

        grad = grad_accum / float(batch)
        s = clip_probabilities(opt.step(s, grad), s_min=s_min, s_max=s_max)
        if constrain_size and k is not None:
            s = project_to_size(s, int(k), s_min=s_min, s_max=s_max)

        if track_history:
            history.append({
                "iteration": float(t + 1),
                "f1": float(np.mean(f1_batch)),
                "f1_best": float(best_f1),
                "f2": float(np.mean(f2_batch)),
                "expected_size": expected_coreset_size(s),
                "outer_lr": (cosine_lr(t + 1, outer_iters, outer_lr) if cosine else outer_lr),
            })
        if log_every and (t + 1) % int(log_every) == 0:
            LOGGER.info("[probabilistic] iter %d/%d f1=%.4f f2=%.2f E||m||_0=%.2f",
                        t + 1, outer_iters, float(np.mean(f1_batch)),
                        float(np.mean(f2_batch)), expected_coreset_size(s))
        if verbose:
            print(f"[probabilistic] {t + 1}/{outer_iters} "
                  f"f1={np.mean(f1_batch):.4f} f2={np.mean(f2_batch):.2f} "
                  f"E||m||_0={expected_coreset_size(s):.2f}", flush=True)

    # ---- final mask induced by the learned probabilities -----------------------------
    final_mask: np.ndarray
    if final_selection == "threshold":
        final_mask = (s >= 0.5).astype(np.float64)
        if not final_mask.any():
            final_mask = sample_mask(s, rng=gen, s_min=s_min, s_max=s_max)
    else:
        kk = int(k) if k is not None else int(round(expected_coreset_size(s)))
        kk = int(np.clip(kk, 1, n))
        order = np.lexsort((np.arange(n), -s))
        final_mask = np.zeros(n, dtype=np.float64)
        final_mask[order[:kk]] = 1.0

    final_f1, _ = evaluate_mask(objective, final_mask, train_kwargs=train_kwargs)
    num_evals += 1
    final_f2 = float(np.count_nonzero(final_mask))
    if np.isfinite(final_f1) and final_f1 < best_f1:
        best_f1, best_mask, best_f2 = float(final_f1), final_mask.copy(), final_f2

    if best_mask is None:  # pragma: no cover - only if every evaluation was non-finite
        best_mask = final_mask.copy()
        best_f1 = float(final_f1)
        best_f2 = final_f2
        stopped_reason = "no_finite_evaluation"

    return ProbabilisticResult(
        s=s,
        best_mask=best_mask,
        best_f1=float(best_f1),
        best_f2=float(best_f2),
        final_mask=final_mask,
        final_f1=float(final_f1),
        final_f2=float(final_f2),
        expected_size=expected_coreset_size(s),
        num_iterations=int(outer_iters),
        num_evaluations=int(num_evals),
        objective=obj_name,
        lambda_=float(lambda_),
        wall_time=float(time.time() - t0),
        history=history,
        stopped_reason=stopped_reason,
    )


# --------------------------------------------------------------------------------------
# Baseline-selector interface (§5.2 / Appendix D.1 comparator)
# --------------------------------------------------------------------------------------


class ProbabilisticSelector(ScoreBaseline):
    """Probabilistic bilevel coreset selection (Zhou et al., ICML 2022).

    Two usage modes:

    * :meth:`run` / :meth:`fit` -- full continuous probabilistic outer optimization with
      the unbiased policy gradient of Appendix C.2; produces the Figure 1 curves
      (``f_1(m)`` and ``f_2(m)`` versus outer iterations) through
      :attr:`ProbabilisticResult.history`.
    * :meth:`select_mask` / :meth:`select_indices` -- fixed-``k`` comparator for the §5.2
      tables: ``s`` is optimized and the top-``k`` probabilities are emitted as a binary
      mask, like every other baseline.

    Class attributes: ``name="Probabilistic"``, ``abbreviation="Prob."``,
    ``requires_model=True``.
    """

    name = "Probabilistic"
    abbreviation = "Prob."
    requires_model = True
    higher_is_better = True  # top-k over the probabilities s

    def __init__(
        self,
        model: Any = None,
        k: Optional[int] = None,
        lambda_: float = DEFAULT_LAMBDA,
        use_size_objective: bool = True,
        outer_iters: int = SUGGESTED_OUTER_ITERS,
        outer_lr: float = FIGURE1_OUTER_LR,
        cosine: bool = True,
        samples_per_iter: int = SUGGESTED_SAMPLES_PER_ITER,
        constrain_size: bool = True,
        final_selection: str = "topk",
        s_min: float = SUGGESTED_S_MIN,
        s_max: float = SUGGESTED_S_MAX,
        init_noise: float = SUGGESTED_INIT_NOISE,
        # inner-loop settings forwarded to the objective callable
        inner_epochs: int = FIGURE1_INNER_EPOCHS,
        inner_lr: float = FIGURE1_INNER_LR,
        inner_optimizer: str = "sgd",
        inner_momentum: float = FIGURE1_INNER_MOMENTUM,
        weight_decay: float = 0.0,
        batch_size: int = 128,
        num_classes: Optional[int] = None,
        device: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, device=device, num_classes=num_classes, **kwargs)
        self.model = model
        self.k = k
        self.lambda_ = float(lambda_)
        self.use_size_objective = bool(use_size_objective)
        self.outer_iters = int(outer_iters)
        self.outer_lr = float(outer_lr)
        self.cosine = bool(cosine)
        self.samples_per_iter = int(samples_per_iter)
        self.constrain_size = bool(constrain_size)
        self.final_selection = final_selection
        self.s_min = float(s_min)
        self.s_max = float(s_max)
        self.init_noise = float(init_noise)
        self.inner_epochs = int(inner_epochs)
        self.inner_lr = float(inner_lr)
        self.inner_optimizer = inner_optimizer
        self.inner_momentum = float(inner_momentum)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.extra = dict(kwargs)

        self._s: Optional[np.ndarray] = None
        self._result: Optional[ProbabilisticResult] = None
        self._scores: Optional[np.ndarray] = None

    # -- config helpers ---------------------------------------------------------------

    def train_kwargs(self) -> Dict[str, Any]:
        """Inner-loop kwargs forwarded to the objective callable (Appendix C.3 / §5.2)."""
        kw: Dict[str, Any] = {
            "epochs": self.inner_epochs,
            "lr": self.inner_lr,
            "momentum": self.inner_momentum,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "optimizer": self.inner_optimizer,
        }
        if self.model is not None:
            kw["model"] = self.model
        return kw

    def figure1_config(self) -> Dict[str, Any]:
        """The Figure 1 configuration of Appendix C.3 (inner SGD, outer Adam + cosine)."""
        return {
            "inner_epochs": FIGURE1_INNER_EPOCHS,
            "inner_lr": FIGURE1_INNER_LR,
            "inner_momentum": FIGURE1_INNER_MOMENTUM,
            "inner_optimizer": "sgd",
            "outer_iters": FIGURE1_OUTER_ITERS,
            "outer_lr": FIGURE1_OUTER_LR,
            "cosine": True,
        }

    # -- main entry points ------------------------------------------------------------

    def run(self, objective: Callable[..., Any], n: int, k: Optional[int] = None,
            seed: Optional[int] = None, **overrides: Any) -> ProbabilisticResult:
        """Run the probabilistic outer optimization (eq. (1) or eq. (2), Appendix C.1)."""
        k = k if k is not None else self.k
        params: Dict[str, Any] = dict(
            lambda_=self.lambda_,
            use_size_objective=self.use_size_objective,
            outer_iters=self.outer_iters,
            outer_lr=self.outer_lr,
            cosine=self.cosine,
            samples_per_iter=self.samples_per_iter,
            s_min=self.s_min,
            s_max=self.s_max,
            init_noise=self.init_noise,
            constrain_size=self.constrain_size,
            final_selection=self.final_selection,
            train_kwargs=self.train_kwargs(),
            seed=seed if seed is not None else self.seed,
        )
        params.update(overrides)
        self._result = probabilistic_bilevel(objective, n=n, k=k, **params)
        self._s = self._result.s
        self._scores = self._s.copy()
        return self._result

    # Alias kept for symmetry with the model-training style APIs used in the drivers.
    fit = run

    def probabilities(self) -> np.ndarray:
        if self._s is None:
            raise RuntimeError("call `run`/`select_mask` before requesting probabilities")
        return self._s.copy()

    @property
    def result(self) -> Optional[ProbabilisticResult]:
        return self._result

    # -- ScoreBaseline interface ------------------------------------------------------

    def compute_scores(self, dataset: Any = None, targets: Any = None,
                       n: Optional[int] = None, seed: Optional[int] = None,
                       objective: Optional[Callable[..., Any]] = None,
                       k: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """Return the Bernoulli probabilities ``s`` (higher = more likely selected).

        If no ``objective`` callable is supplied the probabilities are simply initialized
        to ``k / n`` (uniform), which makes the probabilistic selector degrade to uniform
        sampling -- the honest behaviour of the method without an outer optimizer.
        """
        if n is None:
            n = self._infer_n(dataset, targets)
        k = k if k is not None else self.k
        seed = seed if seed is not None else self.seed
        if objective is None:
            self._s = bernoulli_probabilities(
                int(n), k=k, seed=seed, init_noise=self.init_noise,
                s_min=self.s_min, s_max=self.s_max,
            )
        else:
            res = self.run(objective, n=int(n), k=k, seed=seed, **kwargs)
            self._s = res.s
        self._scores = self._s.copy()
        return self._s

    def select_indices(self, n: Optional[int] = None, k: Optional[int] = None,
                       dataset: Any = None, targets: Any = None,
                       num_classes: Optional[int] = None, seed: Optional[int] = None,
                       scores: Optional[ArrayLike] = None, objective: Any = None,
                       **kwargs: Any) -> np.ndarray:
        """Deterministic top-``k`` selection over the learned probabilities ``s``."""
        if n is None:
            n = self._infer_n(dataset, targets)
        n = int(n)
        k = int(np.clip(int(self.k if k is None else k), 0, n))
        if scores is None:
            scores = self._scores if self._scores is not None else self.compute_scores(
                dataset=dataset, targets=targets, n=n, seed=seed, objective=objective, k=k
            )
        s_arr = _as_float_array(scores)
        if s_arr.size != n:
            raise ValueError(f"scores length {s_arr.size} != n {n}")
        order = np.lexsort((np.arange(n), -s_arr))
        return np.sort(order[:k])

    def select_mask(self, n: Optional[int] = None, k: Optional[int] = None,
                    dataset: Any = None, targets: Any = None,
                    num_classes: Optional[int] = None, seed: Optional[int] = None,
                    scores: Optional[ArrayLike] = None, objective: Any = None,
                    return_indices: bool = False, dtype: Any = np.float32,
                    **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Binary mask ``m in {0,1}^n`` with ``||m||_0 = k`` (fixed-size comparator)."""
        if n is None:
            n = self._infer_n(dataset, targets)
        n = int(n)
        k = int(np.clip(int(self.k if k is None else k), 0, n))
        idx = self.select_indices(n=n, k=k, dataset=dataset, targets=targets,
                                  num_classes=num_classes, seed=seed, scores=scores,
                                  objective=objective, **kwargs)
        mask = np.asarray(indices_to_mask(idx, n, dtype=float), dtype=dtype)
        if return_indices:
            return mask, idx
        return mask

    def mask(self, n: Optional[int] = None, k: Optional[int] = None,
             **kwargs: Any) -> np.ndarray:
        return self.select_mask(n=n, k=k, **kwargs)

    # -- internal ---------------------------------------------------------------------

    @staticmethod
    def _infer_n(dataset: Any = None, targets: Any = None) -> int:
        if targets is not None:
            return len(np.asarray(targets))
        if dataset is not None:
            if hasattr(dataset, "targets"):
                return len(dataset.targets)
            if hasattr(dataset, "labels"):
                return len(dataset.labels)
            try:
                return len(dataset)
            except TypeError:  # pragma: no cover
                pass
        raise ValueError("cannot infer n: pass `n`, `dataset` or `targets`")


# Aliases used by the experiment drivers / baseline registry.
Probabilistic = ProbabilisticSelector
ProbabilisticCoreset = ProbabilisticSelector
ProbabilisticBilevelSelector = ProbabilisticSelector


# --------------------------------------------------------------------------------------
# Figure 1 helper
# --------------------------------------------------------------------------------------


def figure1_curves(objective: Callable[..., Any], n: int, k: int,
                   lambda_: float = DEFAULT_LAMBDA, seed: int = 0,
                   outer_iters: int = FIGURE1_OUTER_ITERS,
                   outer_lr: float = FIGURE1_OUTER_LR,
                   use_size_objective: Optional[bool] = None,
                   **kwargs: Any) -> Dict[str, Any]:
    """Produce the ``f_1`` / ``f_2`` versus outer-iteration curves of Figure 1.

    ``use_size_objective=None`` selects the objective from ``lambda_``: ``lambda_ == 0``
    gives eq. (1) of Appendix C.1 (the fixed-size phenomenon of Figure 1(a)-(b)), while
    ``lambda_ > 0`` gives the weighted problem (4) of §2.1 (``lambda = 1/2`` is the case
    discussed in the paper and reproduced in Figure 1(c)-(d)).
    """
    if use_size_objective is None:
        use_size_objective = lambda_ > 0.0
    res = probabilistic_bilevel(
        objective, n=n, k=k, lambda_=lambda_, use_size_objective=use_size_objective,
        outer_iters=outer_iters, outer_lr=outer_lr, seed=seed, **kwargs
    )
    return {
        "iterations": res.curve("iteration") if res.history else np.asarray([]),
        "f1": res.curve("f1") if res.history else np.asarray([]),
        "f2": res.curve("f2") if res.history else np.asarray([]),
        "expected_size": res.curve("expected_size") if res.history else np.asarray([]),
        "final_s": res.s,
        "final_mask": res.final_mask,
        "best_f1": res.best_f1,
        "best_f2": res.best_f2,
        "result": res,
    }


# --------------------------------------------------------------------------------------
# Offline self-test
# --------------------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of the probability algebra and of the unbiased estimator.

    Verifies, without PyTorch or a dataset:

    1. ``p(m|s)`` sums to one over all ``2^n`` masks and ``1^T s`` equals
       ``E ||m||_0`` under that distribution (Appendix C.1).
    2. the closed-form policy gradient (eq. 31) equals a finite difference of
       ``E f_1(m)`` (Appendix C.2).
    3. the Monte-Carlo estimator of ``E_p f_1 grad ln p`` approximates the exact sum
       ``sum_m f_1(m) grad p(m|s)`` for a small ``n`` where enumeration is feasible.
    4. ``zeta_2(1/2) = sqrt(n)/2`` (eq. 30) and the size projection keeps ``1^T s = k``.
    5. a tiny end-to-end outer loop runs with a synthetic objective.
    """
    out: Dict[str, Any] = {}
    rng = np.random.default_rng(0)

    # --- enumerate all masks for n = 3 -------------------------------------------
    n_small = 3
    s = np.array([0.2, 0.5, 0.8])
    masks: List[Tuple[np.ndarray, float]] = []
    for bits in range(2 ** n_small):
        m = np.array([(bits >> i) & 1 for i in range(n_small)], dtype=float)
        p = mask_probability(m, s)
        masks.append((m, p))

    total_p = sum(p for _, p in masks)
    expected_l0 = sum(p * float(np.count_nonzero(m)) for m, p in masks)
    out["prob_sums_to_one"] = bool(abs(total_p - 1.0) < 1e-12)
    out["expected_l0_matches_sum_s"] = bool(abs(expected_l0 - float(np.sum(s))) < 1e-12)
    out["log_prob_matches_log_mask_prob"] = bool(
        all(abs(math.exp(log_prob(m, s)) - p) < 1e-12 for m, p in masks if p > 1e-12)
    )

    def f1_of_mask(m: np.ndarray) -> float:
        return 1.0 + 0.5 * float(m[0]) + 0.25 * float(m[1]) + 0.125 * float(m[2])

    def expected_f1(s_vec: np.ndarray) -> float:
        return sum(mask_probability(m, s_vec) * f1_of_mask(m) for m, _ in masks)

    # --- 2. exact policy gradient vs. finite difference of E f_1 ------------------
    grad_closed = np.zeros(n_small)
    for m, p in masks:
        grad_closed += p * score_function_gradient(f1_of_mask(m), m, s)
    eps = 1e-6
    grad_fd = np.zeros(n_small)
    for i in range(n_small):
        s_p = s.copy(); s_p[i] += eps
        s_m = s.copy(); s_m[i] -= eps
        grad_fd[i] = (expected_f1(s_p) - expected_f1(s_m)) / (2.0 * eps)
    out["closed_form_grad"] = grad_closed.tolist()
    out["finite_diff_grad"] = grad_fd.tolist()
    out["closed_form_grad_matches_fd"] = bool(np.allclose(grad_closed, grad_fd, atol=1e-5))

    # --- 3. Monte-Carlo score-function estimator is unbiased ---------------------
    mc_grads = np.empty((20000, n_small))
    for i in range(20000):
        m = sample_mask(s, rng=rng)
        mc_grads[i] = score_function_gradient(f1_of_mask(m), m, s)
    mc_mean = mc_grads.mean(axis=0)
    out["mc_grad"] = mc_mean.tolist()
    out["mc_estimator_close_to_exact"] = bool(np.allclose(mc_mean, grad_closed, atol=5e-2))

    # --- 4. gradient norms and projection ----------------------------------------
    out["zeta2_half_equals_sqrt_n_over_2"] = bool(abs(zeta2(100, 0.5) - 5.0) < 1e-12)
    out["zeta1_half_definition"] = float(
        zeta1(2.0, np.array([1.0, 0.0]), np.array([0.5, 0.5]), lambda_=0.5)
    )
    s_proj = project_to_size(np.full(10, 0.5), 4)
    out["projection_size"] = float(np.sum(s_proj))
    out["projection_matches_k"] = bool(abs(float(np.sum(s_proj)) - 4.0) < 1e-6)
    out["projection_respects_box"] = bool(np.all(s_proj >= 0.0) and np.all(s_proj <= 1.0))

    # --- 5. tiny end-to-end run with a synthetic objective ------------------------
    def synthetic_objective(m: np.ndarray) -> float:
        return 3.0 - 2.0 * float(np.mean(m))

    res = probabilistic_bilevel(synthetic_objective, n=5, k=2, outer_iters=10,
                                seed=1, use_size_objective=False, lambda_=0.0)
    out["end_to_end_runs"] = bool(np.isfinite(res.best_f1))
    out["end_to_end_expected_size"] = float(res.expected_size)
    out["end_to_end_num_evaluations"] = int(res.num_evaluations)
    out["end_to_end_history_len"] = int(len(res.history))

    # fixed-size selector smoke test (no objective -> uniform probabilities)
    sel = ProbabilisticSelector(k=3, seed=0)
    mask = sel.select_mask(n=12, k=3)
    out["selector_mask_size"] = int(np.count_nonzero(mask))
    out["selector_mask_l0_equals_k"] = bool(int(np.count_nonzero(mask)) == 3)

    if verbose:
        for key, value in out.items():
            print(f"  {key}: {value}")
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
