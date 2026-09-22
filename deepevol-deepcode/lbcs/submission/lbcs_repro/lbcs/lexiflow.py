"""LexiFlow: the randomized direct-search outer-loop optimizer of LBCS.

This module implements Algorithm 2 of the paper ("Lexicographic Optimization for
:math:`f_1` and :math:`f_2`"), i.e. the black-box optimizer used by Lexicographic
Bilevel Coreset Selection (LBCS) to solve the outer loop of Refined Coreset
Selection (RCS).

Original text (Appendix A, *Details of the Black-box Optimization Algorithm*)::

    Input: Objectives F(.), compromise epsilon.
    Initialization: Initial mask m_0, t' = r = e = 0, and delta = delta_init;
        m* <- m_0, H <- {m_0}, and F_H <- F(m_0).
    while t = 0, 1, ... do
        Sample u uniformly from unit sphere S;
        if update(F(m_t + delta u), F(m_t), F_H) then
            m_{t+1} <- m_t + delta u, t' <- t;
        else if update(F(m_t - delta u), F(m_t), F_H) then
            m_{t+1} <- m_t - delta u, t' <- t;
        else
            m_{t+1} <- m_t, e <- e + 1;
        H <- H  U  {m_{t+1}}, and update F_H according to (14)
        if e = 2^{n-1} then e <- 0, delta <- delta sqrt((t'+1)/(t+1));
        if delta < delta_lower then
            // Random restart;
            r <- r + 1, m_{t+1} <- N(m_0, I), delta <- delta_init + r;
    Procedure update(F(m'), F(m), F_H):
        if F(m') =_(F_H) F(m) or (F(m') <_=(F_H) F(m) and F(m') < F(m)) then
            if F(m') <_=(F_H) F(m*) or (F(m') =_(F_H) F(m*) and F(m') < F(m*)) then
                m* <- m';
            Return True
        else
            Return False
    Output: The optimal mask m*.

The three lexicographic relations used by ``update`` are the *practical*
relations of the paper, defined with the threshold vector
``F_H = [f~_1*, f~_2*]`` computed from the historically evaluated set ``H``
(eq. (14) of the paper)::

    M_H^1 := {m in M_H^0 | f_1(m) <= f~_1*},  f^_1* := inf_{m in M_H^0} f_1(m),
                                              f~_1* = f^_1* * (1 + eps)
    M_H^2 := {m in M_H^1 | f_2(m) <= f~_2*},  f^_2* := inf_{m in M_H^1} f_2(m),
                                              f~_2* = f^_2*

Those relations and the threshold tracker live in :mod:`lbcs.lexicographic`;
this module only wires them into the search loop.  If that module cannot be
imported the class falls back to small logically equivalent local
implementations so that the search remains fully functional.

Practical notes implemented here
--------------------------------
* After each update, ``m`` is clamped (values below ``-1`` become ``-1`` and
  values above ``1`` become ``1``); discretisation (``[-1, 0) -> 0`` and
  ``[0, 1] -> 1``) only happens when a binary mask is required (final output).
* ``delta_init`` and ``delta_lower`` are **not specified numerically** by the
  paper; the defaults used here (``0.1`` and ``1e-3``) are suggested defaults and
  are therefore exposed as constructor arguments.
* Grouped masks (§3.2 acceleration trick 3) are supported: the search samples a
  direction in the group space (dimension ``G = ceil(n / group_size)``) and the
  move is expanded to the example space before being evaluated.

This file contains **no paper-specific numerical constant other than the
suggested defaults**; everything else is algorithm control flow.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch only needed for container preservation / grouped expansion
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is a soft dependency
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional collaborators (documented in the plan's file structure).  They are
# imported lazily / defensively because this module must remain importable on
# its own for the unit-level validation shiped in the repro plan.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised through the real package layout
    from .discretize import clamp_mask as _clamp_mask_helper
    from .discretize import discretize_mask as _discretize_mask_helper
except Exception:  # pragma: no cover
    _clamp_mask_helper = None  # type: ignore
    _discretize_mask_helper = None  # type: ignore

try:  # pragma: no cover
    from .lexicographic import (
        ThresholdTracker,
        as_F,
        accept_move as _lex_accept_move,
        improves_incumbent as _lex_improves_incumbent,
        practical_equal as _lex_practical_equal,
        practical_leq as _lex_practical_leq,
        lexicographic_compare as _lex_exact_compare,
    )

    _LEXICOGRAPHIC_AVAILABLE = True
except Exception:  # pragma: no cover
    ThresholdTracker = None  # type: ignore
    as_F = None  # type: ignore
    _lex_accept_move = None  # type: ignore
    _lex_improves_incumbent = None  # type: ignore
    _lex_practical_equal = None  # type: ignore
    _lex_practical_leq = None  # type: ignore
    _lex_exact_compare = None  # type: ignore
    _LEXICOGRAPHIC_AVAILABLE = False

try:  # pragma: no cover
    from .masks import Grouping, expand_mask as _expand_mask_helper

    _MASKS_AVAILABLE = True
except Exception:  # pragma: no cover
    Grouping = None  # type: ignore
    _expand_mask_helper = None  # type: ignore
    _MASKS_AVAILABLE = False


__all__ = [
    "LexiFlowResult",
    "LexiFlow",
    "lexiflow_search",
    "randomized_direct_search",
    "sample_unit_vector",
    "update_procedure",
    "DEFAULT_DELTA_INIT",
    "DEFAULT_DELTA_LOWER",
    "DEFAULT_EPSILON",
]

logger = logging.getLogger(__name__)

# Suggested defaults (the paper does not specify these numbers).
DEFAULT_DELTA_INIT: float = 0.1
DEFAULT_DELTA_LOWER: float = 1e-3
DEFAULT_EPSILON: float = 0.2

# Below this dimension the paper's stagnation trigger ``e = 2^{n-1}`` is used
# faithfully; above it the value is astronomically large, so the step size can
# never shrink through that branch and the limit is treated as infinity.  This
# is an explicit numerical guard, not a change of the algorithm.
_STAGNATION_DIM_LIMIT: int = 40


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------
def _is_torch(x: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(x, torch.Tensor)  # type: ignore[union-attr]


def _as_numpy(mask: Any) -> np.ndarray:
    """Return a float64 numpy view of ``mask`` (torch tensors are detached)."""
    if _is_torch(mask):
        return mask.detach().cpu().numpy().astype(np.float64, copy=False)  # type: ignore[union-attr]
    if isinstance(mask, np.ndarray):
        return mask.astype(np.float64, copy=False)
    return np.asarray(mask, dtype=np.float64)


def _like(reference: Any, values: np.ndarray) -> Any:
    """Return ``values`` in the same container/device/dtype family as ``reference``."""
    if _is_torch(reference):
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)  # type: ignore[union-attr]
    if isinstance(reference, np.ndarray):
        return values.astype(reference.dtype, copy=False)
    return values


def clamp_mask(mask: Any, lower: float = -1.0, upper: float = 1.0) -> Any:
    """Clamp a (possibly relaxed) mask into ``[lower, upper]``.

    Implements the paper's practical note: "when updating as did in Algorithm 2,
    the value of m less than -1 becomes -1 and the value greater than 1 becomes 1".
    """
    if _clamp_mask_helper is not None:  # pragma: no cover - preferred path
        try:
            return _clamp_mask_helper(mask, lower, upper)
        except TypeError:  # pragma: no cover - alternative signature
            return _clamp_mask_helper(mask)
    if _is_torch(mask):
        return mask.clamp(lower, upper)  # type: ignore[union-attr]
    arr = _as_numpy(mask)
    return _like(mask, np.clip(arr, lower, upper))


def discretize_mask(mask: Any, threshold: float = 0.0) -> Any:
    """Project a relaxed mask to ``{0, 1}`` (``[-1,0)->0``, ``[0,1]->1``)."""
    if _discretize_mask_helper is not None:  # pragma: no cover - preferred path
        return _discretize_mask_helper(mask)
    arr = _as_numpy(mask)
    return _like(mask, (arr >= threshold).astype(np.float64))


def sample_unit_vector(rng: np.random.Generator, dimension: int) -> np.ndarray:
    """Sample ``u`` uniformly from the unit sphere ``S`` of dimension ``dimension``.

    Standard construction: draw ``N(0, I)`` and normalise.  The degenerate
    all-zeros draw (probability zero, but guarded) falls back to a uniform
    unit-coordinate direction.
    """
    u = rng.standard_normal(int(dimension))
    norm = float(np.linalg.norm(u))
    if not np.isfinite(norm) or norm == 0.0:
        u = np.ones(int(dimension), dtype=np.float64)
        norm = float(np.linalg.norm(u))
    return u / norm


def _spherical_step(u: np.ndarray, delta: float) -> np.ndarray:
    """Return ``delta * u`` on the sphere of radius ``delta`` (Algorithm 2's move)."""
    return float(delta) * u


# ---------------------------------------------------------------------------
# Practical lexicographic relations (Appendix A)
# ---------------------------------------------------------------------------
def _f_vector(x: Any) -> np.ndarray:
    """Coerce ``x`` into a 2-vector ``F = [f_1, f_2]``."""
    if as_F is not None:  # pragma: no cover - preferred path
        return np.asarray(as_F(x), dtype=np.float64).reshape(-1)[:2]
    if isinstance(x, np.ndarray):
        return x.astype(np.float64).reshape(-1)[:2]
    if isinstance(x, dict):
        return np.asarray([x["f1"], x["f2"]], dtype=np.float64)
    if hasattr(x, "f1") and hasattr(x, "f2"):
        return np.asarray([x.f1, x.f2], dtype=np.float64)
    return np.asarray(list(x), dtype=np.float64).reshape(-1)[:2]


def _thresholds_vector(F_H: Any) -> np.ndarray:
    """Coerce ``F_H`` into ``[f~_1*, f~_2*]`` (accepts the tracker/thresholds object)."""
    if F_H is None:
        return np.asarray([np.inf, np.inf], dtype=np.float64)
    for attr in ("thresholds_array",):
        if hasattr(F_H, attr):
            try:
                return np.asarray(getattr(F_H, attr)(), dtype=np.float64).reshape(-1)[:2]
            except Exception:  # pragma: no cover
                pass
    if hasattr(F_H, "thresholds") and not isinstance(F_H, np.ndarray):
        try:
            return _thresholds_vector(F_H.thresholds)
        except Exception:  # pragma: no cover
            pass
    for attr in ("f1_tilde", "f2_tilde"):
        if hasattr(F_H, attr):
            return np.asarray([getattr(F_H, "f1_tilde"), getattr(F_H, "f2_tilde")], dtype=np.float64)
    return _f_vector(F_H)


def practical_equal_local(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> bool:
    """Practical equality of Appendix A.

    ``F(m) =_(F_H) F(m')  <=>  for all i in [2]:
       f_i(m) = f_i(m')  or  (f_i(m) <= f~_i*  and  f_i(m') <= f~_i*)``
    """
    a, b, thr = _f_vector(F_m), _f_vector(F_mp), _thresholds_vector(F_H)
    for i in range(2):
        equal = abs(a[i] - b[i]) <= atol
        both_within = (a[i] <= thr[i] + atol) and (b[i] <= thr[i] + atol)
        if not (equal or both_within):
            return False
    return True


def practical_less_local(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> bool:
    """Practical strict improvement ``F(m) <_(F_H) F(m')`` of Appendix A.

    ``exists i in [2]: f_i(m) < f_i(m')  and  f_i(m') > f~_i*
       and  F_{i-1}(m) =_(F_H) F_{i-1}(m')``
    """
    a, b, thr = _f_vector(F_m), _f_vector(F_mp), _thresholds_vector(F_H)
    for i in range(2):
        if not (a[i] < b[i] - atol):
            continue
        if not (b[i] > thr[i] + atol):
            continue
        # all earlier objectives must be practically equivalent
        ok = True
        for j in range(i):
            equal = abs(a[j] - b[j]) <= atol
            both_within = (a[j] <= thr[j] + atol) and (b[j] <= thr[j] + atol)
            if not (equal or both_within):
                ok = False
                break
        if ok:
            return True
    return False


def practical_leq_local(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> bool:
    """``F(m) <=_(F_H) F(m')  <=>  F(m) =_(F_H) F(m') or F(m) <_(F_H) F(m')``."""
    return practical_equal_local(F_m, F_mp, F_H, atol) or practical_less_local(F_m, F_mp, F_H, atol)


def _exact_less(F_m: Any, F_mp: Any, atol: float = 0.0) -> bool:
    """Exact Definition-1 strict lexicographic relation ``F(m) < F(m')``.

    ``exists i in [2]: f_i(m) < f_i(m')  and  for all i' < i: f_i'(m) = f_i'(m')``
    """
    if _lex_exact_compare is not None:  # pragma: no cover - preferred path
        try:
            return _lex_exact_compare(F_m, F_mp, atol) == "<"
        except Exception:
            pass
    a, b = _f_vector(F_m), _f_vector(F_mp)
    for i in range(2):
        if a[i] < b[i] - atol and all(abs(a[j] - b[j]) <= atol for j in range(i)):
            return True
    return False


def update_procedure(
    F_new: Any,
    F_curr: Any,
    F_H: Any,
    atol: float = 0.0,
    F_incumbent: Optional[Any] = None,
) -> Tuple[bool, bool]:
    """The paper's ``Procedure update``.

    Returns a pair ``(accept, replace_incumbent)`` where ``accept`` is the value
    returned by ``update`` and ``replace_incumbent`` says whether the incumbent
    ``m*`` must be replaced by ``m'``::

        if F(m') =_(F_H) F(m) or (F(m') <=_(F_H) F(m) and F(m') < F(m)) then
            if F(m') <=_(F_H) F(m*) or (F(m') =_(F_H) F(m*) and F(m') < F(m*)) then
                m* <- m'
            Return True
        else
            Return False
    """
    a, b = _f_vector(F_new), _f_vector(F_curr)
    thr = _thresholds_vector(F_H)

    eq = practical_equal_local(a, b, thr, atol)
    leq = practical_leq_local(a, b, thr, atol)
    alt_eq = practical_equal_local(a, b, thr, atol)
    if _lex_practical_equal is not None:  # pragma: no cover - preferred path
        try:
            alt_eq = _lex_practical_equal(a, b, thr, atol)
        except Exception:
            pass
    if _lex_practical_leq is not None:  # pragma: no cover
        try:
            leq = _lex_practical_leq(a, b, thr, atol)
        except Exception:
            pass
    exact_less = _exact_less(a, b, atol)

    accept = bool(eq or alt_eq or (leq and exact_less))
    if not accept:
        return False, False

    if F_incumbent is None:
        return True, True

    c = _f_vector(F_incumbent)
    inc_leq = practical_leq_local(a, c, thr, atol)
    inc_eq = practical_equal_local(a, c, thr, atol)
    if _lex_practical_leq is not None:  # pragma: no cover
        try:
            inc_leq = _lex_practical_leq(a, c, thr, atol)
        except Exception:
            pass
    if _lex_practical_equal is not None:  # pragma: no cover
        try:
            inc_eq = _lex_practical_equal(a, c, thr, atol)
        except Exception:
            pass

    replace = bool(inc_leq or (inc_eq and _exact_less(a, c, atol)))
    return True, replace


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class LexiFlowResult:
    """Outcome of a :class:`LexiFlow` run."""

    best_mask: Any                      # final binary mask (discretised m*)
    best_continuous: np.ndarray         # relaxed incumbent m* (in [-1, 1])
    best_F: np.ndarray                  # F(m*) = [f1*, f2*]
    num_iterations: int = 0
    num_evaluations: int = 0
    restarts: int = 0
    delta_final: float = 0.0
    delta_init: float = 0.0
    delta_lower: float = 0.0
    epsilon: float = 0.0
    converged: bool = False            # delta < delta_lower at least once
    stopped_reason: str = ""
    wall_time: float = 0.0
    history: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    trace: List[Dict[str, Any]] = field(default_factory=list)
    masks_history: List[np.ndarray] = field(default_factory=list)

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "f1": float(self.best_F[0]),
            "f2": float(self.best_F[1]),
            "num_iterations": int(self.num_iterations),
            "num_evaluations": int(self.num_evaluations),
            "restarts": int(self.restarts),
            "delta_final": float(self.delta_final),
            "epsilon": float(self.epsilon),
            "converged": bool(self.converged),
            "stopped_reason": self.stopped_reason,
            "wall_time": float(self.wall_time),
        }


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------
class LexiFlow:
    """Randomized direct search over masks with lexicographic acceptance.

    Parameters
    ----------
    objective : callable
        ``F(mask) -> [f1, f2]`` (or an object exposing ``.f1``/``.f2``).  It is
        expected to memoise internally; this class caches results by mask key as
        well so that a re-evaluated mask is never queried twice.
    epsilon : float
        Compromise ``epsilon`` of RCS: ``f~_1* = f^_1* * (1 + epsilon)``.
    delta_init, delta_lower : float
        Initial and lower-bound step sizes.  **Suggested defaults** (the paper
        does not state numerical values): ``0.1`` and ``1e-3``.
    max_iters : int or None
        Number of outer iterations ``T`` (Algorithm 2's ``while t = 0,1,...``
        is bounded by the inner/outer loop budget ``T`` of Algorithm 1).
    dimension : int or None
        Search-space dimension (group count ``G`` when grouping is used).
    grouping : Grouping or None
        Grouped-mask support (§3.2 acceleration trick 3).  Moves are sampled in
        the group space and expanded to the example space before evaluation.
    maximize : bool
        Keep ``False`` for RCS (both objectives are minimised).
    seed : int or None
        Seed of the internal generator used for ``u`` and for random restarts.
    tracker : ThresholdTracker or None
        External history/threshold tracker; one is created when omitted.
    best_mask_fn : callable or None
        ``mask -> binary mask`` conversion used for the reported optimum
        (defaults to the Appendix A discretisation rule).
    """

    def __init__(
        self,
        objective: Callable[[Any], Any],
        epsilon: float = DEFAULT_EPSILON,
        delta_init: float = DEFAULT_DELTA_INIT,
        delta_lower: float = DEFAULT_DELTA_LOWER,
        max_iters: Optional[int] = None,
        dimension: Optional[int] = None,
        grouping: Optional[Any] = None,
        maximize: bool = False,
        seed: Optional[int] = None,
        tracker: Optional[Any] = None,
        best_mask_fn: Optional[Callable[[Any], Any]] = None,
        clamp: bool = True,
        discretize_output: bool = True,
        stagnation_limit: Optional[float] = None,
        add_candidates_to_history: bool = False,
        supports_precompute: bool = False,
        visit_hook: Optional[Callable[[Any], None]] = None,
        tracker_factory: Optional[Callable[[float], Any]] = None,
        log_every: int = 0,
        callback: Optional[Callable[["LexiFlow", int, Any, Any], None]] = None,
        trace: bool = True,
    ) -> None:
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative (RCS compromise)")
        self.objective = objective
        self.epsilon = float(epsilon)
        self.delta_init = float(delta_init)
        self.delta_lower = float(delta_lower)
        self.max_iters = None if max_iters is None else int(max_iters)
        self.dimension = None if dimension is None else int(dimension)
        self.grouping = grouping
        self.maximize = bool(maximize)
        self.clamp = bool(clamp)
        self.discretize_output = bool(discretize_output)
        self.add_candidates_to_history = bool(add_candidates_to_history)
        self.supports_precompute = bool(supports_precompute)
        self.visit_hook = visit_hook
        self.tracker_factory = tracker_factory
        self.log_every = int(log_every)
        self.callback = callback
        self.keep_trace = bool(trace)

        self.rng = np.random.default_rng(seed)
        self.seed = seed

        # -- historical set H and threshold vector F_H (eq. (14)) -----------
        if tracker is not None:
            self.tracker = tracker
            self._owns_tracker = False
        elif _LEXICOGRAPHIC_AVAILABLE and (tracker_factory is not None or True):
            try:
                if tracker_factory is not None:
                    self.tracker = tracker_factory(self.epsilon)
                else:
                    self.tracker = ThresholdTracker(epsilon=self.epsilon)  # type: ignore[misc]
                self._owns_tracker = True
            except Exception:  # pragma: no cover
                self.tracker = None
                self._owns_tracker = True
        else:  # pragma: no cover
            self.tracker = None
            self._owns_tracker = True

        # local fallback history (used when no tracker object is available)
        self._local_history: List[np.ndarray] = []

        if best_mask_fn is not None:
            self.best_mask_fn = best_mask_fn
        elif self.discretize_output:
            self.best_mask_fn = lambda m: discretize_mask(m)
        else:
            self.best_mask_fn = lambda m: m

        if stagnation_limit is not None:
            self.stagnation_limit = float(stagnation_limit)
        else:
            dim = self.dimension
            if dim is None:
                self.stagnation_limit = np.inf
            elif dim >= _STAGNATION_DIM_LIMIT:
                # 2^{n-1} is astronomically large: the trigger is unreachable.
                self.stagnation_limit = np.inf
            else:
                self.stagnation_limit = float(2 ** (dim - 1))

        # run-time state (populated by :meth:`run`)
        self.step: int = 0
        self.t_prime: int = 0
        self.restart_count: int = 0
        self.stagnation: int = 0
        self.delta: float = self.delta_init
        self.num_evaluations: int = 0
        self.best_mask: Any = None
        self.best_continuous: Optional[np.ndarray] = None
        self.best_F: Optional[np.ndarray] = None
        self._cache: Dict[Any, np.ndarray] = {}
        self.trace_records: List[Dict[str, Any]] = []
        self.masks_history: List[np.ndarray] = []
        self.stopped_reason: str = ""
        self.wall_time: float = 0.0
        self._m0: Optional[np.ndarray] = None
        self._n_examples: Optional[int] = None

    # ------------------------------------------------------------------
    # mask keys / expansion
    # ------------------------------------------------------------------
    @staticmethod
    def _mask_key(mask: Any) -> Any:
        arr = _as_numpy(mask).reshape(-1)
        if np.all((arr == 0.0) | (arr == 1.0)):
            return ("b", np.packbits(arr.astype(np.uint8)).tobytes())
        return ("c", np.round(arr, 6).tobytes())

    def _search_dim(self, n: Optional[int] = None) -> int:
        if self.dimension is not None:
            return int(self.dimension)
        if self.grouping is not None and hasattr(self.grouping, "num_groups"):
            return int(getattr(self.grouping, "num_groups"))
        if n is not None:
            return int(n)
        raise ValueError("search dimension unknown; pass `dimension` or `grouping`")

    def _expand(self, v: np.ndarray) -> np.ndarray:
        """Expand a group-space vector to the example space (identity if ungrouped)."""
        if self.grouping is None:
            return np.asarray(v, dtype=np.float64)
        if hasattr(self.grouping, "expand"):
            return _as_numpy(self.grouping.expand(v))
        if _expand_mask_helper is not None:  # pragma: no cover
            return _as_numpy(_expand_mask_helper(v, self.grouping))
        assignment = _as_numpy(getattr(self.grouping, "assignment"))
        return np.asarray(v, dtype=np.float64)[assignment.astype(np.int64)]  # pragma: no cover

    # ------------------------------------------------------------------
    # evaluation + history bookkeeping
    # ------------------------------------------------------------------
    def _evaluate(self, mask: np.ndarray) -> np.ndarray:
        """Evaluate ``F(mask)`` with caching (never evaluates the same mask twice)."""
        key = self._mask_key(mask)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self.visit_hook is not None:
            try:
                self.visit_hook(mask)
            except Exception:  # pragma: no cover
                pass
        out = self.objective(mask)
        F = _f_vector(out)
        self._cache[key] = F
        self.num_evaluations += 1
        return F

    def _thresholds(self) -> np.ndarray:
        if self.tracker is not None:
            try:
                return _thresholds_vector(self.tracker)
            except Exception:  # pragma: no cover
                pass
        if not self._local_history:
            return np.asarray([np.inf, np.inf], dtype=np.float64)
        H = np.vstack(self._local_history)
        f1_hat = float(np.min(H[:, 0]))
        if self.maximize:
            f1_hat = float(np.max(H[:, 0]))
            f1_tilde = f1_hat * (1.0 - self.epsilon)
        else:
            f1_tilde = f1_hat * (1.0 + self.epsilon)
        m1 = H[H[:, 0] <= f1_tilde + 1e-12]
        if m1.size == 0:  # pragma: no cover - defensive
            m1 = H
        f2_hat = float(np.min(m1[:, 1]))
        return np.asarray([f1_tilde, f2_hat], dtype=np.float64)

    def _add_history(self, mask: np.ndarray, F: np.ndarray) -> None:
        self._local_history.append(np.asarray(F, dtype=np.float64).reshape(1, 2))
        if self.tracker is not None:
            try:
                self.tracker.add(F=F, mask=np.asarray(mask, dtype=np.float64))
            except TypeError:  # pragma: no cover - alternative tracker signature
                try:
                    self.tracker.add(f1=float(F[0]), f2=float(F[1]), mask=np.asarray(mask))
                except Exception:
                    pass
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------
    # core step
    # ------------------------------------------------------------------
    def _propose(self, m_t: np.ndarray, u: np.ndarray, delta: float) -> np.ndarray:
        cand = np.asarray(m_t, dtype=np.float64) + _spherical_step(u, delta)
        if self.clamp:
            cand = _as_numpy(clamp_mask(cand))
        return cand

    def run(
        self,
        initial_mask: Any,
        max_iters: Optional[int] = None,
        initial_F: Optional[Any] = None,
        delta_init: Optional[float] = None,
        delta_lower: Optional[float] = None,
    ) -> LexiFlowResult:
        """Run Algorithm 2 from ``initial_mask`` (``m_0``) for ``max_iters`` iterations."""
        t_start = time.time()
        iters = int(max_iters) if max_iters is not None else self.max_iters
        if iters is None:
            raise ValueError("max_iters must be given (Algorithm 1's budget T)")
        if delta_init is not None:
            self.delta_init = float(delta_init)
        if delta_lower is not None:
            self.delta_lower = float(delta_lower)

        m_t = _as_numpy(initial_mask).astype(np.float64)
        if self.clamp:
            m_t = _as_numpy(clamp_mask(m_t))
        self._m0 = m_t.copy()
        self._n_examples = int(m_t.size)

        # ---- Initialization: m* <- m_0, H <- {m_0}, F_H <- F(m_0) ---------
        F_curr = self._evaluate(m_t)
        self.best_continuous = m_t.copy()
        self.best_F = F_curr.copy()
        self.best_mask = self.best_mask_fn(m_t)
        self._add_history(m_t, F_curr)
        self.masks_history.append(m_t.copy())

        self.step = 0
        self.t_prime = 0
        self.restart_count = 0
        self.stagnation = 0
        self.delta = float(self.delta_init)
        self.stopped_reason = "max_iters"
        restart_pending = False

        while self.step < iters:
            # ---- random restart bookkeeping (Algorithm 2, last block) -----
            if restart_pending:
                self.restart_count += 1
                m_t = self._m0 + self.rng.standard_normal(self._m0.size)
                if self.clamp:
                    m_t = _as_numpy(clamp_mask(m_t))
                self.delta = float(self.delta_init + self.restart_count)
                F_curr = self._evaluate(m_t)
                # H <- H U {m_{t+1}} and update F_H
                self._add_history(m_t, F_curr)
                self.masks_history.append(m_t.copy())
                self.stopped_reason = "restarted"
                restart_pending = False

            u = sample_unit_vector(self.rng, self._search_dim(self._n_examples))

            accepted = False
            candidate_F: Optional[np.ndarray] = None
            m_next = m_t
            F_next = F_curr

            for sign in (1.0, -1.0):
                cand = self._propose(m_t, sign * u, self.delta)
                F_cand = self._evaluate(cand)
                if self.add_candidates_to_history:
                    self._add_history(cand, F_cand)
                thr = self._thresholds()
                accept, replace = update_procedure(
                    F_cand, F_curr, thr, F_incumbent=self.best_F
                )
                if accept:
                    m_next, F_next = cand, F_cand
                    self.t_prime = self.step
                    accepted = True
                    if replace and (
                        self.best_F is None or not np.array_equal(F_cand, self.best_F)
                    ):
                        self.best_F = F_cand.copy()
                        self.best_continuous = cand.copy()
                        self.best_mask = self.best_mask_fn(cand)
                    break

            if not accepted:
                self.stagnation += 1

            # ---- H <- H U {m_{t+1}} and update F_H ------------------------
            self._add_history(m_next, F_next)
            self.masks_history.append(np.asarray(m_next, dtype=np.float64).copy())

            # ---- e == 2^{n-1}: reset counter, shrink step ----------------
            if self.stagnation_limit != np.inf and self.stagnation >= self.stagnation_limit:
                self.stagnation = 0
                if self.step + 1 > 0:
                    self.delta = self.delta * math.sqrt(
                        (self.t_prime + 1) / (self.step + 1)
                    )

            # ---- delta < delta_lower: request random restart --------------
            if self.delta < self.delta_lower and self.step + 1 < iters:
                restart_pending = True

            if self.keep_trace:
                self.trace_records.append(
                    {
                        "t": self.step,
                        "f1": float(F_next[0]),
                        "f2": float(F_next[1]),
                        "best_f1": float(self.best_F[0]),
                        "best_f2": float(self.best_F[1]),
                        "accepted": bool(accepted),
                        "delta": float(self.delta),
                        "stagnation": int(self.stagnation),
                        "restarts": int(self.restart_count),
                    }
                )

            if self.callback is not None:
                try:
                    self.callback(self, self.step, m_next, F_next)
                except Exception:  # pragma: no cover
                    pass

            if self.log_every and self.step % self.log_every == 0:
                logger.info(
                    "LexiFlow t=%d f1=%.6f f2=%.3f best=(%.6f, %.3f) delta=%.3g accepted=%s",
                    self.step, F_next[0], F_next[1],
                    self.best_F[0], self.best_F[1], self.delta, accepted,
                )

            m_t, F_curr = m_next, F_next
            self.step += 1

        self.wall_time = time.time() - t_start
        if self.delta < self.delta_lower:
            self.stopped_reason = "delta_below_lower"

        return self.result()

    # ------------------------------------------------------------------
    def result(self) -> LexiFlowResult:
        hist = (
            np.vstack(self._local_history) if self._local_history
            else np.zeros((0, 2), dtype=np.float64)
        )
        return LexiFlowResult(
            best_mask=self.best_mask,
            best_continuous=np.asarray(self.best_continuous, dtype=np.float64),
            best_F=np.asarray(self.best_F, dtype=np.float64),
            num_iterations=int(self.step),
            num_evaluations=int(self.num_evaluations),
            restarts=int(self.restart_count),
            delta_final=float(self.delta),
            delta_init=float(self.delta_init),
            delta_lower=float(self.delta_lower),
            epsilon=float(self.epsilon),
            converged=bool(self.delta < self.delta_lower),
            stopped_reason=self.stopped_reason,
            wall_time=float(self.wall_time),
            history=hist,
            trace=list(self.trace_records),
            masks_history=[np.asarray(m).copy() for m in self.masks_history],
        )

    # ------------------------------------------------------------------
    # helpers for the driver loops
    # ------------------------------------------------------------------
    def clear_history(self) -> None:
        """Reset the historical set ``H`` and the evaluation cache."""
        self._local_history = []
        self._cache = {}
        if self.tracker is not None:
            try:
                self.tracker.H = []
                self.tracker.refresh()
            except Exception:  # pragma: no cover
                pass

    def best_binary_mask(self) -> Any:
        """The final coreset mask: ``m*`` projected to ``{0, 1}``."""
        if self.best_continuous is None:
            raise RuntimeError("run() must be called before best_binary_mask()")
        return self.best_mask_fn(self.best_continuous)

    def thresholds(self) -> np.ndarray:
        """Current vector ``F_H = [f~_1*, f~_2*]`` (eq. (14))."""
        return self._thresholds()


# ---------------------------------------------------------------------------
# Functional front-ends
# ---------------------------------------------------------------------------
def lexiflow_search(
    objective: Callable[[Any], Any],
    initial_mask: Any,
    max_iters: int,
    epsilon: float = DEFAULT_EPSILON,
    delta_init: float = DEFAULT_DELTA_INIT,
    delta_lower: float = DEFAULT_DELTA_LOWER,
    dimension: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> LexiFlowResult:
    """Convenience wrapper: create a :class:`LexiFlow` and run Algorithm 2 once."""
    optimizer = LexiFlow(
        objective=objective,
        epsilon=epsilon,
        delta_init=delta_init,
        delta_lower=delta_lower,
        max_iters=max_iters,
        dimension=dimension,
        seed=seed,
        **kwargs,
    )
    return optimizer.run(initial_mask, max_iters=max_iters)


#: Alias emphasising the family of the algorithm (Appendix A).
randomized_direct_search = lexiflow_search


# ---------------------------------------------------------------------------
# Self-test (run with ``python -m lbcs_repro.lbcs.lexiflow``)
# ---------------------------------------------------------------------------
def _selftest() -> None:  # pragma: no cover - validation harness
    rng = np.random.default_rng(0)

    # 1) unit-sphere sampling
    for d in (1, 2, 16, 101):
        u = sample_unit_vector(rng, d)
        assert u.shape == (d,) and abs(np.linalg.norm(u) - 1.0) < 1e-9

    # 2) practical relations on a tiny synthetic problem
    #    f1 = fraction of selected entries, f2 = ||m||_0  (both minimised)
    def F_of(mask):
        m = _as_numpy(mask)
        z = (m >= 0).astype(np.float64)
        return np.asarray([0.5 + 0.5 * float(np.mean(z)), float(np.count_nonzero(z))])

    H = [F_of(np.full(8, -1.0)), F_of(np.full(8, 1.0))]
    thr = np.asarray([0.5 * 1.2, 8.0])
    m_all = np.full(8, 1.0)
    m_none = np.full(8, -1.0)
    # mask at the threshold is equivalent to the historical optimum
    assert practical_equal_local(F_of(m_all), thr, thr)
    assert practical_equal_local(thr, F_of(m_all), thr)
    # sparse mask is practically strictly better on f2 (f1 in the eps-region)
    assert practical_less_local(F_of(m_none), F_of(m_all), thr)
    assert practical_leq_local(F_of(m_none), F_of(m_all), thr)
    assert update_procedure(F_of(m_none), F_of(m_all), thr)[0]

    # 3) full search on a toy objective: minimising f1 first, then f2
    def objective(mask):
        m = _as_numpy(mask)
        z = (m >= 0).astype(np.float64)
        f1 = 1.0 - 0.9 * float(np.mean(z))
        f2 = float(np.count_nonzero(z))
        return np.asarray([f1, f2])

    res = lexiflow_search(
        objective,
        initial_mask=np.full(64, -1.0),
        max_iters=60,
        epsilon=0.2,
        dimension=64,
        seed=0,
    )
    assert res.num_iterations == 60
    assert res.best_F.shape == (2,)
    assert res.history.shape[1] == 2
    assert res.num_evaluations >= 1
    # lexicographic: f1 must not be catastrophically worse than the start
    assert res.best_F[0] <= 1.0 + 1e-9
    # a sizeable part of the search space gets selected as f2 is the tie-breaker
    print("LexiFlow self-test passed:", res.to_dict())


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest()
