"""Spectral density of the L-BFGS-preconditioned Hessian (Appendix C).

L-BFGS takes the step ``w_{k+1} = w_k - eta H_k grad L(w_k)`` where ``H_k``
approximates the inverse Hessian.  Appendix C.1 shows this is equivalent to
preconditioning ``L`` and Appendix C.2 derives the factorisation

    H_k = Htilde_k Htilde_k^T,
    Htilde_k = [ sqrt(gamma_k) (I - Ytilde Vtilde^T)^T ,  Stilde ]   (p x (p + m))

with ``Ytilde``, ``Vtilde`` and ``Stilde`` obtained from ``Algorithm 2``.  The
non-zero eigenvalues of ``H_k H_L(w)`` coincide with the eigenvalues of

    M = Htilde_k^T H_L(w) Htilde_k,

which is symmetric and can therefore be fed to SLQ.  ``Algorithm 3`` gives the
matrix-vector product with ``M``: it only needs the stacked vectors of
Algorithm 2 and Hessian-vector products.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import torch


@dataclass
class LBFGSHistory:
    """The ``m`` most recent L-BFGS correction pairs, newest first.

    ``s[i] = w_{i+1} - w_i``, ``y[i] = grad_{i+1} - grad_i`` and
    ``rho[i] = 1 / (y_i^T s_i)``.  ``gamma`` is the scaling factor used for the
    initial inverse-Hessian approximation ``H_k^0 = gamma I``.

    Use :meth:`from_torch_state` to build the history from
    ``torch.optim.LBFGS(..., history_size=m, line_search_fn="strong_wolfe")``.
    Any L-BFGS implementation can be used as long as the correction pairs are
    recorded (see ``pinn/optim/lbfgs.py``).
    """

    s: List[torch.Tensor]
    y: List[torch.Tensor]
    rho: List[float]
    gamma: float

    def __len__(self) -> int:
        return len(self.s)

    @property
    def m(self) -> int:
        return len(self.s)

    @classmethod
    def from_torch_state(cls, state: dict) -> "LBFGSHistory":
        """Extract the history stored by ``torch.optim.LBFGS``."""
        old_dirs = state["old_dirs"]  # y_i, oldest first
        old_stps = state["old_stps"]  # s_i, oldest first
        ro = state["ro"]  # 1 / (y_i . s_i)
        H_diag = state["H_diag"]  # gamma
        s = [t.detach().clone() for t in reversed(old_stps)]
        y = [t.detach().clone() for t in reversed(old_dirs)]
        rho = [float(r) for r in reversed(ro)]
        return cls(s=s, y=y, rho=rho, gamma=float(H_diag))

    def unroll(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Algorithm 2: unrolling the L-BFGS update.

        Returns ``(Ytilde, Vtilde, Stilde)`` with one column per correction
        pair, newest first (paper's ordering ``k-1, ..., k-m``).
        """
        m = self.m
        assert m > 0, "the L-BFGS history is empty"
        yt = [self.rho[0] * self.y[0]]
        vt = [self.s[0].clone()]
        st = [self.s[0] * (self.rho[0] ** 0.5)]
        for i in range(1, m):
            yt.append(self.rho[i] * self.y[i])
            alpha = torch.zeros_like(self.s[i])
            for j in range(i):
                alpha = alpha + torch.dot(yt[j], self.s[i]) * vt[j]
            v = self.s[i] - alpha
            vt.append(v)
            st.append(v * (self.rho[i] ** 0.5))
        Y = torch.stack(yt, dim=1)  # p x m
        V = torch.stack(vt, dim=1)  # p x m
        S = torch.stack(st, dim=1)  # p x m
        return Y, V, S

    def two_loop(self, g: torch.Tensor) -> torch.Tensor:
        """Apply ``H_k`` to ``g`` with the standard two-loop recursion."""
        q = g.to(self.s[0].dtype).clone()
        m = self.m
        alpha = [0.0] * m
        for i in range(m):
            alpha[i] = self.rho[i] * torch.dot(self.s[i], q)
            q = q - alpha[i] * self.y[i]
        r = q * self.gamma
        for i in range(m - 1, -1, -1):
            beta = self.rho[i] * torch.dot(self.y[i], r)
            r = r + (alpha[i] - beta) * self.s[i]
        return r


class LBFGSPreconditioner:
    """Matrix-vector products with ``M = Htilde^T H Htilde`` (Algorithm 3)."""

    def __init__(self, history: LBFGSHistory):
        self.history = history
        self.Y, self.V, self.S = history.unroll()
        self.gamma = history.gamma
        self.p, self.m = self.Y.shape
        self.dim = self.p + self.m

    # ------------------------------------------------------------------ #
    def htilde(self, v: torch.Tensor) -> torch.Tensor:
        """``Htilde_k v`` (maps ``R^{p+m} -> R^p``)."""
        v = v.to(self.Y.dtype)
        v1, v2 = v[: self.p], v[self.p :]
        return (self.gamma**0.5) * (v1 - self.V @ (self.Y.T @ v1)) + self.S @ v2

    def htilde_t(self, u: torch.Tensor) -> torch.Tensor:
        """``Htilde_k^T u`` (maps ``R^p -> R^{p+m}``)."""
        u = u.to(self.Y.dtype)
        top = (self.gamma**0.5) * (u - self.Y @ (self.V.T @ u))
        bottom = self.S.T @ u
        return torch.cat([top, bottom])

    def matvec(self, v: torch.Tensor, hessian_matvec: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """``M v`` (Algorithm 3 of the paper)."""
        v = v.to(self.Y.dtype)
        v1, v2 = v[: self.p], v[self.p :]
        vp = (self.gamma**0.5) * (v1 - self.V @ (self.Y.T @ v1)) + self.S @ v2
        vpp = hessian_matvec(vp)
        vpp = vpp.to(self.Y.dtype)
        top = (self.gamma**0.5) * (vpp - self.Y @ (self.V.T @ vpp))
        bottom = self.S.T @ vpp
        return torch.cat([top, bottom])

    # ------------------------------------------------------------------ #
    def apply_H(self, u: torch.Tensor) -> torch.Tensor:
        """Apply the inverse-Hessian approximation ``H_k`` to a ``p``-vector.

        Implemented through the factorisation (rather than the two-loop
        recursion) so that it is guaranteed to be consistent with ``matvec``.
        """
        return self.htilde(self.htilde_t(u))
