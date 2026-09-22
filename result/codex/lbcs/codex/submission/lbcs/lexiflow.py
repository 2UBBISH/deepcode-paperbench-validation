"""LexiFlow: the randomized direct-search optimiser used for the outer loop.

This is Algorithm 2 of the paper with the "practical lexicographic relations"
of Appendix A.  The outer loop of lexicographic bilevel coreset selection is
treated as a black-box optimisation problem: the optimiser only needs to be
able to *compare* two masks through their objective values ``F(m) = [f1, f2]``
and it never needs gradients of ``f1`` / ``f2`` with respect to the mask.

Definitions (Appendix A).  With the history of evaluated masks ``H`` and

    f_hat_1* = inf_{m in H} f1(m),          f_tilde_1* = f_hat_1* * (1 + eps)
    f_hat_2* = inf_{m in H, f1 <= f_tilde_1*} f2(m),  f_tilde_2* = f_hat_2*

the practical lexicographic relations between ``F(m)`` and ``F(m')`` are

    F(m) =_(F_H) F(m')  <=>  for all i:  f_i(m) = f_i(m') or
                                          (f_i(m) <= f_tilde_i* and
                                           f_i(m') <= f_tilde_i*)
    F(m) <_(F_H) F(m')  <=>  exists i:   f_i(m) < f_i(m'), f_i(m') > f_tilde_i*
                                          and F_{i-1}(m) =_(F_H) F_{i-1}(m')
    F(m) <=_(F_H) F(m') <=>  F(m) =_(F_H) F(m') or F(m) <_(F_H) F(m')
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch


ArrayLike = Sequence[float]


def practical_eq(a: ArrayLike, b: ArrayLike, thr: ArrayLike) -> bool:
    """``F(m) =_(F_H) F(m')`` (Appendix A, first relation)."""
    return all(
        (float(a[i]) == float(b[i])) or
        (float(a[i]) <= float(thr[i]) and float(b[i]) <= float(thr[i]))
        for i in range(len(a))
    )


def practical_less(a: ArrayLike, b: ArrayLike, thr: ArrayLike) -> bool:
    """``F(m) <_(F_H) F(m')`` (Appendix A, second relation).

    ``a`` is better than ``b`` when the first objective on which the two
    disagree favours ``a`` -- provided ``b`` has not already reached the
    optimising threshold on that objective and the leading objectives are
    practically equivalent.
    """
    for i in range(len(a)):
        ai, bi, ti = float(a[i]), float(b[i]), float(thr[i])
        if ai == bi or (ai <= ti and bi <= ti):
            continue                      # practically equal at this level
        return ai < bi and bi > ti
    return False


def true_less(a: ArrayLike, b: ArrayLike) -> bool:
    """Plain (exact) lexicographic comparison, used as a tie-break."""
    for i in range(len(a)):
        ai, bi = float(a[i]), float(b[i])
        if ai != bi:
            return ai < bi
    return False


def thresholds_from_history(values: Sequence[ArrayLike],
                            epsilon: float) -> Tuple[float, float]:
    """``F_H = [f_tilde_1*, f_tilde_2*]`` computed from evaluated points."""
    if values is None or len(values) == 0:
        return (float("inf"), float("inf"))
    f1_min = min(float(v[0]) for v in values)
    f1_thr = f1_min * (1.0 + epsilon)
    feasible = [float(v[1]) for v in values if float(v[0]) <= f1_thr]
    f2_thr = min(feasible) if feasible else float("inf")
    return (f1_thr, f2_thr)


class _HistoryBuffer:
    """Growing ``(f_1, f_2)`` buffer with vectorised threshold computation."""

    def __init__(self, capacity: int = 1024):
        self._values = np.empty((max(capacity, 8), 2), dtype=np.float64)
        self._size = 0

    def append(self, value: ArrayLike) -> None:
        if self._size == self._values.shape[0]:
            bigger = np.empty((self._size * 2, 2), dtype=np.float64)
            bigger[:self._size] = self._values[:self._size]
            self._values = bigger
        self._values[self._size] = (float(value[0]), float(value[1]))
        self._size += 1

    @property
    def size(self) -> int:
        return self._size

    def array(self) -> np.ndarray:
        return self._values[:self._size]

    def thresholds(self, epsilon: float) -> Tuple[float, float]:
        if self._size == 0:
            return (float("inf"), float("inf"))
        values = self._values[:self._size]
        f1_thr = float(values[:, 0].min()) * (1.0 + epsilon)
        mask = values[:, 0] <= f1_thr
        f2_thr = float(values[mask, 1].min()) if mask.any() else float("inf")
        return (f1_thr, f2_thr)


@dataclass
class LexiFlowConfig:
    """Hyper-parameters of the black-box search."""

    epsilon: float = 0.2
    delta_init: float = 1.0
    delta_lower: float = 1e-3
    max_steps: int = 500
    # Appendix A states the step size is shrunk when ``e = 2^(n-1)``, where
    # ``e`` counts consecutive non-improving steps and ``n`` is the number of
    # search variables.  That is a theoretical bound which is never reached in
    # finite time, so the companion randomized direct search implementations
    # use a finite patience.  ``None`` reproduces the literal paper condition.
    step_decay_patience: Optional[int] = None
    # Algorithm 2 samples a direction ``u`` uniformly from the unit sphere.
    # A unit vector has per-coordinate magnitude O(1/sqrt(n)), so with n
    # examples the step delta*u could never change a coordinate of a mask whose
    # entries live in {-1, +1}: the mask would be frozen forever.  Searching
    # the *discrete* mask therefore requires a sampling scale that is O(1) per
    # coordinate.  ``u_mode='gaussian'`` (the default) draws u ~ N(0, I) so
    # that a step of size delta moves each coordinate by O(delta);
    # ``u_mode='sphere'`` implements the literal unit-sphere sampling of the
    # paper, for which ``delta_init`` should be of order sqrt(n) (equivalently,
    # a uniform direction on the sphere of radius sqrt(n)).
    #   "sparse"   (default) -- only ``sparse_size`` randomly chosen
    #     coordinates are perturbed, each by +-2 in the continuous mask, so a
    #     step of size delta flips at most ``sparse_size`` examples of the
    #     coreset.  This is the local search over *masks* that the paper's
    #     experiments perform: a dense direction (see below) changes
    #     O(n) coordinates at once, which makes the walker drift to the full
    #     data set instead of refining the coreset.
    #   "gaussian" -- u ~ N(0, I), i.e. O(1) movement on every coordinate.
    #   "sphere"   -- the literal unit-sphere sampling of Algorithm 2, for
    #     which ``delta_init`` should be of order sqrt(n).
    u_mode: str = "sparse"
    sparse_size: Optional[int] = 10
    sparse_step: float = 2.0           # +-sparse_step on the chosen coordinates
    # Algorithm 2 appends the new point to the history *after* the ``update``
    # procedure, but the optimising thresholds ``F_H`` are the best values
    # available to the algorithm.  Two readings are possible and both are
    # provided:
    #   True  (default) -- the thresholds include the candidate being
    #     evaluated, i.e. ``thr = thresholds(H + {F(m')})``.  This matches
    #     Remark 3: inside the compromise region the incumbent is updated
    #     whenever the candidate has a better value of f2 while f1 stays in
    #     M_1*, and it is what makes the voluntary compromise ``eps``
    #     meaningful (a larger eps allows a smaller coreset).
    #   False -- the literal ordering of the pseudocode (``F_H`` is the
    #     threshold of the history *before* the candidate is appended).
    thresholds_include_candidate: bool = True
    seed: int = 0
    log_every: int = 0


class LexiFlow:
    """Lexicographic randomised direct search over continuous masks.

    Parameters
    ----------
    objective:
        Callable ``F(m) -> tensor([f1, f2])`` accepting a continuous mask.
        It is expected to cache repeated queries (see
        :class:`lbcs.objectives.BilevelObjective`).
    config:
        :class:`LexiFlowConfig`.
    """

    def __init__(self, objective: Callable[[torch.Tensor], torch.Tensor],
                 config: Optional[LexiFlowConfig] = None):
        self.objective = objective
        self.cfg = config or LexiFlowConfig()
        self.history_values: List[Tuple[float, float]] = []
        self.history_masks: List[torch.Tensor] = []
        self._buffer = _HistoryBuffer()
        self.best_value: Optional[torch.Tensor] = None
        self.best_mask: Optional[torch.Tensor] = None

    # -- bookkeeping -------------------------------------------------------
    def _thresholds(self, candidate: Optional[torch.Tensor] = None
                    ) -> torch.Tensor:
        """Optimising thresholds ``F_H = [f_tilde_1*, f_tilde_2*]``.

        When ``thresholds_include_candidate`` is set (default) the candidate
        being evaluated is part of the history the thresholds are derived from.
        """
        if candidate is not None and self.cfg.thresholds_include_candidate:
            values = np.vstack([self._buffer.array(),
                                np.asarray([[float(candidate[0]),
                                             float(candidate[1])]])])
            thr = thresholds_from_history(values, self.cfg.epsilon)
        else:
            thr = self._buffer.thresholds(self.cfg.epsilon)
        return torch.tensor(thr, dtype=torch.float64)

    def _direction(self, n: int, gen: torch.Generator,
                   device) -> torch.Tensor:
        if self.cfg.u_mode == "sparse":
            size = self.cfg.sparse_size or max(1, n // 1000)
            size = max(1, min(int(size), n))
            idx = torch.randperm(n, generator=gen)[:size]
            u = torch.zeros(n)
            signs = torch.where(torch.rand(size, generator=gen) < 0.5, -1.0, 1.0)
            u[idx] = signs * self.cfg.sparse_step
            return u.to(device)
        u = torch.randn(n, generator=gen).to(device)
        if self.cfg.u_mode == "sphere":
            u = u / (u.norm() + 1e-12)
        return u

    def _record(self, mask: torch.Tensor, value: torch.Tensor) -> None:
        self.history_masks.append(mask.detach().clone())
        self.history_values.append((float(value[0]), float(value[1])))
        self._buffer.append((float(value[0]), float(value[1])))

    def _update(self, mask: torch.Tensor, value: torch.Tensor,
                incumbent_value: torch.Tensor, thr: torch.Tensor) -> bool:
        """The ``update`` procedure of Algorithm 2.

        Returns ``True`` when the candidate mask should replace the incumbent
        walker position; the global best mask ``m*`` is maintained as well.
        """
        cur, new = incumbent_value, value
        accept = (practical_less(new, cur, thr) or
                  (practical_eq(new, cur, thr) and true_less(new, cur)))
        if not accept:
            return False
        best = self.best_value
        improves_best = (practical_less(new, best, thr) or
                         (practical_eq(new, best, thr) and true_less(new, best)))
        if improves_best:
            self.best_value = new.detach().clone()
            self.best_mask = mask.detach().clone()
        return True

    # -- main loop ---------------------------------------------------------
    def optimize(self, mask0: torch.Tensor,
                 max_steps: Optional[int] = None) -> torch.Tensor:
        cfg = self.cfg
        device = mask0.device
        n = mask0.numel()
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        patience = (cfg.step_decay_patience
                    if cfg.step_decay_patience is not None else 2 ** (n - 1))

        m = mask0.detach().clone().clamp(-1.0, 1.0)
        self.best_mask = m.clone()
        value = self.objective(m)
        self.best_value = value.clone()
        self._record(m, value)

        t_prime = 0
        e = 0
        r = 0
        delta = cfg.delta_init
        steps = cfg.max_steps if max_steps is None else max_steps

        for t in range(steps):
            u = self._direction(n, gen, device)
            accepted = False
            for sign in (1.0, -1.0):
                cand = (m + sign * delta * u).clamp(-1.0, 1.0)
                cand_value = self.objective(cand)
                # Algorithm 2 evaluates ``update(F(m_t +- delta u), F(m_t),
                # F_H)``; the new point is appended to H only after the walker
                # has moved, but the thresholds are the best values available
                # to the algorithm (see ``thresholds_include_candidate``).
                thr = self._thresholds(cand_value)
                if self._update(cand, cand_value, value, thr):
                    m, value = cand, cand_value
                    t_prime = t
                    accepted = True
                    break
            self._record(m, value)
            if not accepted:
                e += 1
            if e >= patience:
                e = 0
                delta = delta * math.sqrt((t_prime + 1.0) / (t + 1.0))
            if delta < cfg.delta_lower:
                # random restart around the initial mask
                r += 1
                m = (mask0.detach().clone()
                     + torch.randn(n, generator=gen).to(device)).clamp(-1.0, 1.0)
                value = self.objective(m)
                delta = cfg.delta_init + r
            if cfg.log_every and (t + 1) % cfg.log_every == 0:
                print(f"[LexiFlow] t={t + 1} delta={delta:.4g} "
                      f"best_f1={float(self.best_value[0]):.4f} "
                      f"best_f2={float(self.best_value[1]):.1f} "
                      f"queries={len(self.history_values)}")

        return self.best_mask.clone()
