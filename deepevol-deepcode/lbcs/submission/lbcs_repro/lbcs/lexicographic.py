"""Lexicographic relations for Refined Coreset Selection (RCS).

This module implements

* **Definition 1** of the paper (main text, Section 3.2): the exact
  lexicographic relations ``=``, ``\prec`` and ``\preceq`` over the objective
  vector ``F(m) = [f_1(m), f_2(m)]``::

      F(m) = F(m')            <=>  f_i(m) = f_i(m')             for all i in [2]
      F(m) < F(m') (prec)     <=>  exists i in [2]:
                                      f_i(m) < f_i(m')
                                      and for all i' < i: f_i'(m) = f_i'(m')
      F(m) <= F(m') (preceq)  <=>  F(m) = F(m') or F(m) < F(m')

* The **practical lexicographic relations** of Appendix A, which replace the
  (theoretically achievable but practically inaccessible) infima of
  Definition 1 by the minima over the historical set of already evaluated
  points ``H``::

      F(m) =_(F_H) F(m')  <=>  for all i in [2]:
                                   f_i(m) = f_i(m')
                                   or (f_i(m) <= f~_i*  and  f_i(m') <= f~_i*)

      F(m) <_(F_H) F(m')  <=>  exists i in [2]:
                                   f_i(m) < f_i(m')
                                   and f_i(m') > f~_i*
                                   and F_{i-1}(m) =_(F_H) F_{i-1}(m')

      F(m) <=_(F_H) F(m') <=>  F(m) =_(F_H) F(m')  or  F(m) <_(F_H) F(m')

  where ``F_{i-1}(m) = [f_1(m), ..., f_{i-1}(m)]`` (an empty vector for
  ``i = 1``, which is vacuously "practically equal").

* The **threshold tracker** ``F_H = [f~_1*, f~_2*]`` computed from the
  historical set ``H`` (Appendix A, eq. (14)), with ``M_H^0 = H``::

      M_H^1 := { m in M_H^0 | f_1(m) <= f~_1* },  f^_1* := inf_{m in M_H^0} f_1(m),
                                                 f~_1*  = f^_1* * (1 + eps)
      M_H^2 := { m in M_H^1 | f_2(m) <= f~_2* },  f^_2* := inf_{m in M_H^1} f_2(m),
                                                 f~_2*  = f^_2*

  ``eps`` (``epsilon``) is the *relative* compromise of ``f_1(m)`` accepted in
  order to obtain a better value of ``f_2(m)`` (the paper removed the optional
  input targets and changed the compromise from an absolute to a relative
  value, see footnote 1 of Appendix A).  ``F_H`` means that any masks reaching
  these thresholds can be considered equivalent with respect to that objective.

* The ``Procedure update`` of Algorithm 2 expressed with the practical
  relations, used to decide whether a sampled point replaces the current point.

Everything operates on plain NumPy arrays / floats so the relations can be
unit-tested without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    # representation
    "FVector",
    "as_F",
    "as_thresholds",
    # Definition 1 (exact relations)
    "definition1_equal",
    "definition1_less",
    "definition1_leq",
    "lexicographic_compare",
    "compare_many",
    # threshold computation (eq. 14)
    "LexicographicThresholds",
    "compute_thresholds",
    "select_M1",
    "select_M2",
    # practical relations
    "practical_equal",
    "practical_less",
    "practical_leq",
    "practical_compare",
    # Procedure update / incumbent logic
    "accept_move",
    "improves_incumbent",
    "ThresholdTracker",
    "PracticalLexicographic",
    "HistoricalPoint",
    # small validation helpers
    "relations_are_reflexive",
    "relations_are_transitive",
    "validate_relations",
]

# ---------------------------------------------------------------------------
# Types / small helpers
# ---------------------------------------------------------------------------

FVector = Sequence[float]

#: comparison markers used throughout the module
EQ = "="
LESS = "<"
GREATER = ">"
INCOMPARABLE = "?"

_INF = float("inf")


def as_F(F: Any) -> np.ndarray:
    """Coerce anything sequence-like into a length-2 float vector ``[f1, f2]``.

    Accepts a 2-element sequence, an object exposing ``f1``/``f2`` attributes
    (e.g. ``objectives.MaskEvaluation``), or a mapping with ``"f1"``/``"f2"``.
    """
    if F is None:
        raise ValueError("F(m) cannot be None")
    if hasattr(F, "f1") and hasattr(F, "f2"):
        return np.asarray(
            [float(getattr(F, "f1")), float(getattr(F, "f2"))], dtype=np.float64
        )
    if isinstance(F, dict):
        return np.asarray([float(F["f1"]), float(F["f2"])], dtype=np.float64)
    arr = np.asarray(F, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError(
            f"F(m) must have exactly 2 entries [f1, f2], got shape {arr.shape}"
        )
    return arr


def as_thresholds(F_H: Any) -> np.ndarray:
    """Coerce ``F_H`` into ``[f~_1*, f~_2*]`` (accepts a tracker or thresholds)."""
    if hasattr(F_H, "thresholds"):
        return np.asarray(F_H.thresholds, dtype=np.float64)
    if hasattr(F_H, "thresholds_array"):
        return np.asarray(F_H.thresholds_array(), dtype=np.float64)
    if hasattr(F_H, "f1_tilde") and hasattr(F_H, "f2_tilde"):
        return np.asarray(
            [float(F_H.f1_tilde), float(F_H.f2_tilde)], dtype=np.float64
        )
    arr = np.asarray(F_H, dtype=np.float64).reshape(-1)
    if arr.size != 2:
        raise ValueError("F_H must have exactly 2 entries [f~1*, f~2*]")
    return arr


def _close(a: float, b: float, atol: float) -> bool:
    return bool(np.abs(float(a) - float(b)) <= atol)


# ---------------------------------------------------------------------------
# Definition 1 -- exact lexicographic relations
# ---------------------------------------------------------------------------


def definition1_equal(F_m: Any, F_mp: Any, atol: float = 0.0) -> bool:
    """``F(m) = F(m')  <=>  f_i(m) = f_i(m') for all i in [2]`` (Definition 1)."""
    a, b = as_F(F_m), as_F(F_mp)
    return bool(np.all(np.abs(a - b) <= atol))


def definition1_less(F_m: Any, F_mp: Any, atol: float = 0.0) -> bool:
    """``F(m) < F(m')`` (Definition 1): the first strictly better objective decides.

    ``exists i in [2]: f_i(m) < f_i(m') and (for all i' < i: f_i'(m) = f_i'(m'))``
    """
    a, b = as_F(F_m), as_F(F_mp)
    for i in range(2):
        prefix_equal = bool(np.all(np.abs(a[:i] - b[:i]) <= atol))
        if (a[i] < b[i] - atol) and prefix_equal:
            return True
    return False


def definition1_leq(F_m: Any, F_mp: Any, atol: float = 0.0) -> bool:
    """``F(m) <= F(m')  <=>  F(m) = F(m') or F(m) < F(m')`` (Definition 1)."""
    return definition1_equal(F_m, F_mp, atol=atol) or definition1_less(
        F_m, F_mp, atol=atol
    )


def lexicographic_compare(F_m: Any, F_mp: Any, atol: float = 0.0) -> str:
    """Return ``"="``, ``"<"``, ``">"`` or ``"?"`` for two exact objective vectors.

    The exact lexicographic relation is both reflexive and transitive
    (Zhang et al., 2023b, cited by the paper), so the comparison of any two
    feasible masks is conclusive; ``"?"`` only shows up for ``NaN`` inputs.
    """
    if definition1_equal(F_m, F_mp, atol=atol):
        return EQ
    if definition1_less(F_m, F_mp, atol=atol):
        return LESS
    if definition1_less(F_mp, F_m, atol=atol):
        return GREATER
    return INCOMPARABLE


def compare_many(F: Any, F_ref: Any, atol: float = 0.0) -> np.ndarray:
    """Vectorised ``lexicographic_compare`` against ``F_ref``; returns codes ±1/0."""
    arr = np.atleast_2d(np.asarray(F, dtype=np.float64)).reshape(-1, 2)
    other = np.atleast_2d(np.asarray(F_ref, dtype=np.float64)).reshape(-1, 2)
    if other.shape[0] == 1 and arr.shape[0] > 1:
        other = np.repeat(other, arr.shape[0], axis=0)
    out = np.zeros(arr.shape[0], dtype=np.int64)
    for i in range(arr.shape[0]):
        c = lexicographic_compare(arr[i], other[i], atol=atol)
        out[i] = -1 if c == LESS else (1 if c == GREATER else 0)
    return out


# ---------------------------------------------------------------------------
# Thresholds F_H = [f~_1*, f~_2*]  (Appendix A, eq. (14))
# ---------------------------------------------------------------------------


@dataclass
class LexicographicThresholds:
    """Thresholds ``F_H = [f~_1*, f~_2*]`` and the sets that define them.

    ``f1_hat``/``f2_hat`` are ``f^_1*``/``f^_2*`` (the infima over the
    historical sets ``M_H^0``/``M_H^1``), while ``f1_tilde``/``f2_tilde`` are
    the thresholds ``f~_1* = f^_1* (1 + eps)`` and ``f~_2* = f^_2*``.
    """

    epsilon: float = 0.0
    f1_hat: float = _INF
    f2_hat: float = _INF
    f1_tilde: float = _INF
    f2_tilde: float = _INF
    size_H: int = 0
    size_M1: int = 0
    size_M2: int = 0
    #: indices (into the history the thresholds were computed from) of M_H^1 / M_H^2
    M1_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    M2_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # -- convenience -------------------------------------------------------
    @property
    def thresholds(self) -> np.ndarray:
        """``F_H`` as a length-2 array ``[f~_1*, f~_2*]``."""
        return np.asarray([self.f1_tilde, self.f2_tilde], dtype=np.float64)

    def thresholds_array(self) -> np.ndarray:
        return self.thresholds

    def __getitem__(self, i: int) -> float:
        return float(self.thresholds[i])

    def __len__(self) -> int:
        return 2

    def within_f1(self, f1: float) -> bool:
        """Is ``f1`` inside the compromise region ``M_H^1`` (``f1 <= f~_1*``)?"""
        return float(f1) <= self.f1_tilde

    def within_f2(self, f2: float) -> bool:
        """Is ``f2`` inside ``M_H^2`` (``f2 <= f~_2*``)?"""
        return float(f2) <= self.f2_tilde

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": float(self.epsilon),
            "f1_hat": float(self.f1_hat),
            "f2_hat": float(self.f2_hat),
            "f1_tilde": float(self.f1_tilde),
            "f2_tilde": float(self.f2_tilde),
            "size_H": int(self.size_H),
            "size_M1": int(self.size_M1),
            "size_M2": int(self.size_M2),
        }


def compute_thresholds(F_history: Any, epsilon: float = 0.0) -> LexicographicThresholds:
    """Compute ``F_H`` from the historically evaluated points ``H`` (eq. (14)).

    ``F_history`` is an ``(N, 2)`` array (or anything coercible to one) holding
    ``[f_1(m), f_2(m)]`` for every ``m`` in ``H = M_H^0``.  Steps exactly as in
    Appendix A:

    1. ``f^_1* = inf_{m in M_H^0} f_1(m)``
    2. ``f~_1* = f^_1* * (1 + eps)``
    3. ``M_H^1 = {m in M_H^0 | f_1(m) <= f~_1*}``
    4. ``f^_2* = inf_{m in M_H^1} f_2(m)`` and ``f~_2* = f^_2*``
    5. ``M_H^2 = {m in M_H^1 | f_2(m) <= f~_2*}``
    """
    F = np.asarray(F_history, dtype=np.float64)
    if F.size == 0:
        return LexicographicThresholds(epsilon=float(epsilon))
    F = F.reshape(-1, 2)
    n = F.shape[0]

    f1_hat = float(np.min(F[:, 0]))
    f1_tilde = f1_hat * (1.0 + float(epsilon))

    M1 = np.flatnonzero(F[:, 0] <= f1_tilde)
    if M1.size == 0:  # numerically impossible (argmin is always inside); safety net
        M1 = np.asarray([int(np.argmin(F[:, 0]))], dtype=np.int64)

    f2_hat = float(np.min(F[M1, 1]))
    f2_tilde = f2_hat

    M2 = M1[F[M1, 1] <= f2_tilde]

    return LexicographicThresholds(
        epsilon=float(epsilon),
        f1_hat=f1_hat,
        f2_hat=f2_hat,
        f1_tilde=float(f1_tilde),
        f2_tilde=float(f2_tilde),
        size_H=int(n),
        size_M1=int(M1.size),
        size_M2=int(M2.size),
        M1_indices=np.asarray(M1, dtype=np.int64),
        M2_indices=np.asarray(M2, dtype=np.int64),
    )


def select_M1(F_history: Any, thresholds: Any) -> np.ndarray:
    """Indices of ``M_H^1 = {m in M_H^0 | f_1(m) <= f~_1*}``."""
    F = np.asarray(F_history, dtype=np.float64).reshape(-1, 2)
    thr = as_thresholds(thresholds)
    return np.flatnonzero(F[:, 0] <= thr[0])


def select_M2(F_history: Any, thresholds: Any) -> np.ndarray:
    """Indices of ``M_H^2 = {m in M_H^1 | f_2(m) <= f~_2*}``."""
    F = np.asarray(F_history, dtype=np.float64).reshape(-1, 2)
    thr = as_thresholds(thresholds)
    in_M1 = F[:, 0] <= thr[0]
    return np.flatnonzero(in_M1 & (F[:, 1] <= thr[1]))


# ---------------------------------------------------------------------------
# Practical lexicographic relations (Appendix A)
# ---------------------------------------------------------------------------


def _prefix_practically_equal(
    a: np.ndarray, b: np.ndarray, k: int, thr: np.ndarray, atol: float
) -> bool:
    """``F_k(m) =_(F_H) F_k(m')`` for the first ``k`` objectives.

    ``k == 0`` is the empty vector, which is (vacuously) practically equal.
    """
    for i in range(k):
        if not _close(a[i], b[i], atol) and not (a[i] <= thr[i] and b[i] <= thr[i]):
            return False
    return True


def practical_equal(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> bool:
    """Practical equality ``F(m) =_(F_H) F(m')`` (Appendix A).

    ``for all i in [2]: f_i(m) = f_i(m') or (f_i(m) <= f~_i* and f_i(m') <= f~_i*)``
    """
    a, b = as_F(F_m), as_F(F_mp)
    thr = as_thresholds(F_H)
    for i in range(2):
        if _close(a[i], b[i], atol):
            continue
        if a[i] <= thr[i] and b[i] <= thr[i]:
            continue
        return False
    return True


def practical_less(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> bool:
    """Practical strict improvement ``F(m) <_(F_H) F(m')`` (Appendix A).

    ``exists i in [2]: f_i(m) < f_i(m') and f_i(m') > f~_i*
                       and F_{i-1}(m) =_(F_H) F_{i-1}(m')``

    The extra condition ``f_i(m') > f~_i*`` prevents "improving" an objective
    whose value for the compared mask already sits inside the equivalence
    region defined by the thresholds.
    """
    a, b = as_F(F_m), as_F(F_mp)
    thr = as_thresholds(F_H)
    for i in range(2):
        if (a[i] < b[i] - atol) and (b[i] > thr[i]):
            if _prefix_practically_equal(a, b, i, thr, atol):
                return True
    return False


def practical_leq(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> bool:
    """``F(m) <=_(F_H) F(m')  <=>  F(m) =_(F_H) F(m') or F(m) <_(F_H) F(m')``."""
    return practical_equal(F_m, F_mp, F_H, atol=atol) or practical_less(
        F_m, F_mp, F_H, atol=atol
    )


def practical_compare(F_m: Any, F_mp: Any, F_H: Any, atol: float = 0.0) -> str:
    """Return ``"="``, ``"<"``, ``">"`` or ``"?"`` under the practical relations."""
    if practical_equal(F_m, F_mp, F_H, atol=atol):
        return EQ
    if practical_less(F_m, F_mp, F_H, atol=atol):
        return LESS
    if practical_less(F_mp, F_m, F_H, atol=atol):
        return GREATER
    return INCOMPARABLE


# ---------------------------------------------------------------------------
# Procedure update / incumbent update (Algorithm 2)
# ---------------------------------------------------------------------------


def accept_move(F_new: Any, F_curr: Any, F_H: Any, atol: float = 0.0) -> bool:
    """``Procedure update(F(m'), F(m), F_H)`` of Algorithm 2.

    The sampled point ``m'`` (here ``F_new``; ``F_curr`` is ``F(m)``) is
    accepted if

    ``F(m') =_(F_H) F(m)``  or  (``F(m') <_(F_H) F(m)`` and ``F(m') < F(m)``),

    i.e. either the two points are practically equivalent (which lets the
    search drift inside the ``eps`` region while ``f_2`` improves), or the new
    point is a genuine practical *and* exact lexicographic improvement.
    """
    if practical_equal(F_new, F_curr, F_H, atol=atol):
        return True
    return practical_less(F_new, F_curr, F_H, atol=atol) and definition1_less(
        F_new, F_curr, atol=atol
    )


def improves_incumbent(F_new: Any, F_star: Any, F_H: Any, atol: float = 0.0) -> bool:
    """Does ``F(m')`` beat the incumbent ``F(m*)``? (inner test of ``update``)

    ``F(m') <_(F_H) F(m*)``  or  (``F(m') =_(F_H) F(m*)`` and ``F(m') < F(m*)``).
    """
    if practical_less(F_new, F_star, F_H, atol=atol):
        return True
    return practical_equal(F_new, F_star, F_H, atol=atol) and definition1_less(
        F_new, F_star, atol=atol
    )


# ---------------------------------------------------------------------------
# Historical set / threshold tracker
# ---------------------------------------------------------------------------


@dataclass
class HistoricalPoint:
    """One element of ``H``: an evaluated mask together with ``F(m)``."""

    mask: Any = None
    f1: float = _INF
    f2: float = _INF
    key: Optional[Any] = None
    tag: Optional[str] = None

    @property
    def F(self) -> np.ndarray:
        return np.asarray([self.f1, self.f2], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "f1": float(self.f1),
            "f2": float(self.f2),
            "key": self.key,
            "tag": self.tag,
        }


class ThresholdTracker:
    """Maintains the historical set ``H`` and the thresholds ``F_H``.

    Every call to :meth:`add` inserts a newly evaluated point; :meth:`refresh`
    (automatic by default) recomputes ``F_H`` from the *whole* accumulated
    history according to eq. (14) of Appendix A.  The tracker also exposes the
    practical relations bound to the current ``F_H``, so that the outer
    optimiser can query them without passing thresholds around.

    Parameters
    ----------
    epsilon:
        Relative compromise of ``f_1(m)`` (``epsilon >= 0``).
    store_masks:
        Keep the mask objects in ``H`` (needed to retrieve the masks of
        ``M_H^1``/``M_H^2`` in experiments); disable for very long searches.
    atol:
        Absolute tolerance for the "``f_i(m) = f_i(m')``" tests.
    """

    def __init__(
        self,
        epsilon: float = 0.0,
        *,
        store_masks: bool = True,
        atol: float = 0.0,
    ) -> None:
        if epsilon < 0:
            raise ValueError("epsilon (compromise of f1) must be non-negative")
        self.epsilon = float(epsilon)
        self.store_masks = bool(store_masks)
        self.atol = float(atol)
        self.H: List[HistoricalPoint] = []
        self._keys: set = set()
        self.thresholds_info: LexicographicThresholds = LexicographicThresholds(
            epsilon=self.epsilon
        )
        #: snapshots of ``F_H`` after each update (for logging / plots)
        self.trace: List[Dict[str, Any]] = []

    # -- population --------------------------------------------------------
    @staticmethod
    def _key(mask: Any) -> Any:
        if mask is None:
            return None
        if isinstance(mask, (int, np.integer)):
            return int(mask)
        try:  # pragma: no cover - depends on torch availability
            import torch

            if isinstance(mask, torch.Tensor):
                arr = mask.detach().cpu().double().numpy().reshape(-1)
                return (arr.size, int(np.rint(np.abs(arr).sum() * 1e6)), float(arr.sum()))
        except Exception:  # pragma: no cover
            pass
        arr = np.asarray(mask, dtype=np.float64).reshape(-1)
        return (arr.size, int(np.rint(np.abs(arr).sum() * 1e6)), float(arr.sum()))

    def add(
        self,
        F: Any = None,
        *,
        mask: Any = None,
        key: Any = None,
        tag: Optional[str] = None,
        f1: Optional[float] = None,
        f2: Optional[float] = None,
        refresh: bool = True,
    ) -> HistoricalPoint:
        """Insert an evaluated point into ``H`` and (by default) refresh ``F_H``."""
        if F is None:
            if f1 is None or f2 is None:
                raise ValueError("provide either F or both f1 and f2")
            vec = np.asarray([float(f1), float(f2)], dtype=np.float64)
        else:
            vec = as_F(F)
        pt = HistoricalPoint(
            mask=mask if self.store_masks else None,
            f1=float(vec[0]),
            f2=float(vec[1]),
            key=key if key is not None else self._key(mask),
            tag=tag,
        )
        if pt.key is not None:
            self._keys.add(pt.key)
        self.H.append(pt)
        if refresh:
            self.refresh()
        return pt

    def extend(self, points: Iterable[Any], **kw: Any) -> None:
        """Add several points at once (one single refresh at the end)."""
        for p in points:
            self.add(p, refresh=False, **kw)
        self.refresh()

    def seen(self, key: Any) -> bool:
        return key in self._keys

    # -- thresholds --------------------------------------------------------
    def refresh(self) -> LexicographicThresholds:
        """Recompute ``F_H = [f~_1*, f~_2*]`` from the current ``H`` (eq. (14))."""
        self.thresholds_info = compute_thresholds(self.F_history(), self.epsilon)
        self.trace.append(self.thresholds_info.to_dict())
        return self.thresholds_info

    @property
    def thresholds(self) -> np.ndarray:
        return self.thresholds_info.thresholds

    @property
    def F_H(self) -> np.ndarray:
        return self.thresholds_info.thresholds

    def F_history(self) -> np.ndarray:
        if not self.H:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray([[p.f1, p.f2] for p in self.H], dtype=np.float64)

    def M1_points(self) -> List[HistoricalPoint]:
        """``M_H^1 = {m in M_H^0 | f_1(m) <= f~_1*}``."""
        idx = self.thresholds_info.M1_indices
        return [self.H[int(i)] for i in idx]

    def M2_points(self) -> List[HistoricalPoint]:
        """``M_H^2 = {m in M_H^1 | f_2(m) <= f~_2*}``."""
        idx = self.thresholds_info.M2_indices
        return [self.H[int(i)] for i in idx]

    def M1_masks(self) -> List[Any]:
        return [p.mask for p in self.M1_points()]

    def M2_masks(self) -> List[Any]:
        return [p.mask for p in self.M2_points()]

    def best_lexicographic(self) -> Optional[HistoricalPoint]:
        """Best point of ``H`` under the *exact* Definition 1 relation."""
        if not self.H:
            return None
        best = self.H[0]
        for p in self.H[1:]:
            if definition1_less(p.F, best.F, atol=self.atol):
                best = p
        return best

    def best_f1(self) -> Optional[HistoricalPoint]:
        if not self.H:
            return None
        return min(self.H, key=lambda p: p.f1)

    def best_f2(self) -> Optional[HistoricalPoint]:
        if not self.H:
            return None
        return min(self.H, key=lambda p: p.f2)

    def best_in_M2(self) -> Optional[HistoricalPoint]:
        """Point of ``M_H^2`` with the smallest ``f_2`` (the search target)."""
        pts = self.M2_points()
        if not pts:
            return None
        return min(pts, key=lambda p: (p.f2, p.f1))

    # -- relations bound to the current F_H --------------------------------
    def eq(self, F_a: Any, F_b: Any) -> bool:
        return practical_equal(F_a, F_b, self.thresholds_info, atol=self.atol)

    def lt(self, F_a: Any, F_b: Any) -> bool:
        return practical_less(F_a, F_b, self.thresholds_info, atol=self.atol)

    def leq(self, F_a: Any, F_b: Any) -> bool:
        return practical_leq(F_a, F_b, self.thresholds_info, atol=self.atol)

    def compare(self, F_a: Any, F_b: Any) -> str:
        return practical_compare(F_a, F_b, self.thresholds_info, atol=self.atol)

    # -- Procedure update (Algorithm 2) -----------------------------------
    def accept_move(self, F_new: Any, F_curr: Any) -> bool:
        return accept_move(F_new, F_curr, self.thresholds_info, atol=self.atol)

    def improves_incumbent(self, F_new: Any, F_star: Any) -> bool:
        return improves_incumbent(F_new, F_star, self.thresholds_info, atol=self.atol)

    def __len__(self) -> int:
        return len(self.H)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "size_H": len(self.H),
            "F_H": self.thresholds.tolist(),
            **self.thresholds_info.to_dict(),
        }


class PracticalLexicographic:
    """Facade for the outer loop: threshold tracking + practical relation queries.

    This is the object the outer optimiser (Algorithm 2, ``lexiflow.py``) and
    Algorithm 1 use to decide mask updates.  It owns a :class:`ThresholdTracker`
    and mirrors its API::

        lex = PracticalLexicographic(epsilon=0.2)
        lex.add(mask=m, f1=f1, f2=f2)
        if lex.accept_move(F_new, F_curr):
            ...
        if lex.improves_incumbent(F_new, F_star):
            ...
    """

    def __init__(self, epsilon: float = 0.0, **kw: Any) -> None:
        self.epsilon = float(epsilon)
        self.tracker = ThresholdTracker(epsilon=epsilon, **kw)

    # -- population --------------------------------------------------------
    def add(self, *args: Any, **kw: Any) -> HistoricalPoint:
        return self.tracker.add(*args, **kw)

    def extend(self, points: Iterable[Any], **kw: Any) -> None:
        self.tracker.extend(points, **kw)

    def seen(self, key: Any) -> bool:
        return self.tracker.seen(key)

    # -- thresholds --------------------------------------------------------
    @property
    def thresholds(self) -> np.ndarray:
        return self.tracker.thresholds

    @property
    def F_H(self) -> np.ndarray:
        return self.tracker.F_H

    @property
    def thresholds_info(self) -> LexicographicThresholds:
        return self.tracker.thresholds_info

    def refresh(self) -> LexicographicThresholds:
        return self.tracker.refresh()

    def F_history(self) -> np.ndarray:
        return self.tracker.F_history()

    # -- relations ---------------------------------------------------------
    def eq(self, F_a: Any, F_b: Any) -> bool:
        return self.tracker.eq(F_a, F_b)

    def lt(self, F_a: Any, F_b: Any) -> bool:
        return self.tracker.lt(F_a, F_b)

    def leq(self, F_a: Any, F_b: Any) -> bool:
        return self.tracker.leq(F_a, F_b)

    def compare(self, F_a: Any, F_b: Any) -> str:
        return self.tracker.compare(F_a, F_b)

    # -- Algorithm 2 logic -------------------------------------------------
    def accept_move(self, F_new: Any, F_curr: Any) -> bool:
        return self.tracker.accept_move(F_new, F_curr)

    def improves_incumbent(self, F_new: Any, F_star: Any) -> bool:
        return self.tracker.improves_incumbent(F_new, F_star)

    # -- introspection -----------------------------------------------------
    def M1_points(self) -> List[HistoricalPoint]:
        return self.tracker.M1_points()

    def M2_points(self) -> List[HistoricalPoint]:
        return self.tracker.M2_points()

    def best_lexicographic(self) -> Optional[HistoricalPoint]:
        return self.tracker.best_lexicographic()

    def best_in_M2(self) -> Optional[HistoricalPoint]:
        return self.tracker.best_in_M2()

    def __len__(self) -> int:
        return len(self.tracker)

    def to_dict(self) -> Dict[str, Any]:
        return self.tracker.to_dict()


# ---------------------------------------------------------------------------
# Validation helpers (used by the unit checks of `validation_approach` item 1)
# ---------------------------------------------------------------------------


def relations_are_reflexive(F_list: Any, F_H: Any = None, atol: float = 0.0) -> bool:
    """Check ``F(m) = F(m)`` (and ``=_(F_H)`` when thresholds are given)."""
    for F in np.atleast_2d(np.asarray(F_list, dtype=np.float64)).reshape(-1, 2):
        if not definition1_equal(F, F, atol=atol):
            return False
        if F_H is not None and not practical_equal(F, F, F_H, atol=atol):
            return False
    return True


def relations_are_transitive(
    F_list: Any, F_H: Any = None, relation: str = "lt", atol: float = 0.0
) -> bool:
    """Brute-force transitivity check on a (small) set of objective vectors.

    ``relation`` is ``"lt"`` (strict), ``"leq"`` or ``"eq"``.
    """
    F = np.atleast_2d(np.asarray(F_list, dtype=np.float64)).reshape(-1, 2)
    if F_H is not None:
        thr = as_thresholds(F_H)
        fn = {"eq": practical_equal, "lt": practical_less, "leq": practical_leq}[relation]

        def rel(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(fn(a, b, thr, atol=atol))

    else:
        fn = {
            "eq": definition1_equal,
            "lt": definition1_less,
            "leq": definition1_leq,
        }[relation]

        def rel(a: np.ndarray, b: np.ndarray) -> bool:
            return bool(fn(a, b, atol=atol))

    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            if not rel(F[i], F[j]):
                continue
            for k in range(F.shape[0]):
                if rel(F[j], F[k]) and not rel(F[i], F[k]):
                    return False
    return True


def validate_relations(
    F_list: Any, epsilon: float = 0.2, atol: float = 0.0
) -> Dict[str, Any]:
    """Run the relation sanity checks used by the validation harness."""
    F = np.atleast_2d(np.asarray(F_list, dtype=np.float64)).reshape(-1, 2)
    thr = compute_thresholds(F, epsilon)
    return {
        "exact_reflexive": relations_are_reflexive(F, None, atol=atol),
        "exact_transitive_lt": relations_are_transitive(F, None, "lt", atol=atol),
        "exact_transitive_leq": relations_are_transitive(F, None, "leq", atol=atol),
        "practical_reflexive": relations_are_reflexive(F, thr, atol=atol),
        "practical_transitive_lt": relations_are_transitive(F, thr, "lt", atol=atol),
        "practical_transitive_leq": relations_are_transitive(F, thr, "leq", atol=atol),
        "thresholds": thr.to_dict(),
    }
