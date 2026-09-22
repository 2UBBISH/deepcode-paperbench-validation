"""NysNewton-CG (NNCG) optimizer -- Algorithm 4 of the paper.

NysNewton-CG is a second-order optimizer for PINNs that combines:

  * a randomized Nystroem approximation of the Hessian (Algorithm 5), refreshed
    every ``F`` iterations,
  * a Nystroem-preconditioned conjugate gradient solve (Algorithm 6) of the
    damped Newton system ``(H_L(w_k) + mu I) d = grad L(w_k)``, warm-started
    with the previous Newton step ``d_{k-1}``,
  * an Armijo backtracking line search (Algorithm 7) to pick the step size.

The Hessian is never materialized: all products ``H v`` are computed with the
Pearlmutter double-backward trick (see ``src/hessian/hvp.py``).

Algorithm 4 (paper):
    Inputs: w_0, eta=1, K=2000, s=60, F=20, mu, eps=1e-16, M=1000,
            alpha=0.1, beta=0.5
    d_{-1} = 0
    for k = 0 .. K-1:
        if k % F == 0:
            [U, Lambda_hat] = RandomizedNystromApproximation(H_L(w_k), s)
        d_k = NystromPCG(H_L(w_k), grad L(w_k), d_{k-1}, U, Lambda_hat,
                         s, mu, eps, M)
        eta_k = Armijo(L, w_k, grad L(w_k), -d_k, eta)
        w_{k+1} = w_k - eta_k * d_k
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..hessian.hvp import hvp as _hvp
from ..model import flatten_parameters, set_flat_parameters
from .armijo import armijo_line_search_with_setter
from .nystrom import randomized_nystrom_approximation
from .pcg import nystrom_pcg

__all__ = ["NysNewtonCG", "nncg_minimize"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_hvp_fn(loss_fn: Callable[[], torch.Tensor], model: nn.Module):
    """Return a closure ``v -> H_L(w) v`` for the current model parameters.

    The closure recomputes the gradient graph on every call (the parameters may
    have changed since the last call), then applies the Pearlmutter trick.
    """

    def matvec(v: torch.Tensor) -> torch.Tensor:
        params = [p for p in model.parameters() if p.requires_grad]
        loss = loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        # First-order product g^T v
        gv = torch.dot(flat_grad, v)
        # Second-order product: d/dw (g^T v) = H v
        hvp_list = torch.autograd.grad(
            gv, params, retain_graph=False, create_graph=False, allow_unused=True
        )
        pieces = []
        for p, h in zip(params, hvp_list):
            if h is None:
                pieces.append(torch.zeros_like(p).reshape(-1))
            else:
                pieces.append(h.reshape(-1))
        return torch.cat(pieces)

    return matvec


def _flat_grad(loss_fn: Callable[[], torch.Tensor], model: nn.Module) -> torch.Tensor:
    """Compute the flat gradient of ``loss_fn`` w.r.t. model parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=False, retain_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------
class NysNewtonCG:
    """NysNewton-CG optimizer (Algorithm 4).

    Parameters
    ----------
    model : nn.Module
        The PINN model whose parameters are optimized.
    loss_fn : Callable[[], torch.Tensor]
        Zero-argument closure returning the scalar loss, reading the *live*
        model parameters (see ``src.loss.make_loss_fn``).
    mu : float
        Damping / regularization added to the Hessian (tuned in {1e-2, 1e-1}).
    s : int
        Nystroem sketch size (rank of the low-rank approximation).
    F : int
        Refresh the Nystroem approximation every ``F`` iterations.
    K : int
        Maximum number of outer iterations.
    M : int
        Maximum number of PCG iterations per outer step.
    eps : float
        PCG tolerance / Nystroem numerical floor.
    eta : float
        Initial Armijo step size.
    alpha, beta : float
        Armijo sufficient-decrease and backtracking parameters.
    grad_tol : float or None
        Optional early-stopping threshold on ``||grad L(w)||_2``.
    verbose : bool
        Print progress every ``log_every`` iterations.
    log_every : int
        Logging frequency.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        mu: float = 1e-2,
        s: int = 60,
        F: int = 20,
        K: int = 2000,
        M: int = 1000,
        eps: float = 1e-16,
        eta: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.5,
        grad_tol: Optional[float] = None,
        verbose: bool = False,
        log_every: int = 100,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.mu = float(mu)
        self.s = int(s)
        self.F = int(F)
        self.K = int(K)
        self.M = int(M)
        self.eps = float(eps)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.grad_tol = grad_tol
        self.verbose = verbose
        self.log_every = int(log_every)

        # History for diagnostics / plotting (Figure 4).
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "grad_norm": [],
            "step": [],
            "pcg_iters": [],
        }

    # -- parameter helpers -------------------------------------------------
    def _get_params(self) -> torch.Tensor:
        return flatten_parameters(self.model)

    def _set_params(self, flat: torch.Tensor) -> None:
        set_flat_parameters(self.model, flat)

    # -- main loop ---------------------------------------------------------
    def step(self) -> Dict[str, float]:
        """Run the full NNCG optimization loop. Returns a summary dict."""
        model = self.model
        p = sum(q.numel() for q in model.parameters() if q.requires_grad)

        d_prev: Optional[torch.Tensor] = None
        U: Optional[torch.Tensor] = None
        Lambda_hat: Optional[torch.Tensor] = None

        for k in range(self.K):
            # --- (1) refresh Nystroem approximation every F iterations -----
            if k % self.F == 0 or U is None:
                hvp_fn = _make_hvp_fn(self.loss_fn, model)
                U, Lambda_hat = randomized_nystrom_approximation(
                    hvp_fn, s=self.s, p=p
                )

            # --- (2) compute gradient --------------------------------------
            grad = _flat_grad(self.loss_fn, model)
            grad_norm = float(torch.linalg.norm(grad).item())

            loss_val = float(self.loss_fn().detach().item())
            self.history["loss"].append(loss_val)
            self.history["grad_norm"].append(grad_norm)

            if self.verbose and (k % self.log_every == 0 or k == self.K - 1):
                print(
                    f"[NNCG] iter {k:5d} | loss {loss_val:.6e} | "
                    f"||grad|| {grad_norm:.6e}"
                )

            if self.grad_tol is not None and grad_norm < self.grad_tol:
                if self.verbose:
                    print(f"[NNCG] converged at iter {k} (grad_norm < {self.grad_tol})")
                break

            # --- (3) Nystroem-preconditioned CG solve ----------------------
            hvp_fn = _make_hvp_fn(self.loss_fn, model)
            d_k, info = nystrom_pcg(
                hvp_fn,
                grad,
                U,
                Lambda_hat,
                mu=self.mu,
                eps=self.eps,
                max_iters=self.M,
                x0=d_prev,
                return_info=True,
            )
            self.history["pcg_iters"].append(float(info.get("iters", 0)))

            # --- (4) Armijo line search along -d_k -------------------------
            step, _ = armijo_line_search_with_setter(
                f=self.loss_fn,
                set_params=self._set_params,
                get_params=self._get_params,
                grad=grad,
                d=-d_k,
                t=self.eta,
                alpha=self.alpha,
                beta=self.beta,
            )
            self.history["step"].append(float(step))

            # --- (5) update parameters -------------------------------------
            w_k = self._get_params()
            w_next = w_k - step * d_k
            self._set_params(w_next)

            # Warm-start next PCG with the (scaled) Newton direction.
            d_prev = d_k

        # Final metrics
        final_loss = float(self.loss_fn().detach().item())
        final_grad = _flat_grad(self.loss_fn, model)
        final_grad_norm = float(torch.linalg.norm(final_grad).item())
        return {
            "loss": final_loss,
            "grad_norm": final_grad_norm,
            "iters": len(self.history["loss"]),
        }

    # Convenience alias
    def minimize(self) -> Dict[str, float]:
        return self.step()


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------
def nncg_minimize(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    mu: float = 1e-2,
    s: int = 60,
    F: int = 20,
    K: int = 2000,
    M: int = 1000,
    eps: float = 1e-16,
    eta: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    grad_tol: Optional[float] = None,
    verbose: bool = False,
    log_every: int = 100,
) -> Tuple[nn.Module, Dict[str, float], Dict[str, List[float]]]:
    """Functional wrapper around :class:`NysNewtonCG`.

    Returns
    -------
    (model, summary, history)
    """
    opt = NysNewtonCG(
        model=model,
        loss_fn=loss_fn,
        mu=mu,
        s=s,
        F=F,
        K=K,
        M=M,
        eps=eps,
        eta=eta,
        alpha=alpha,
        beta=beta,
        grad_tol=grad_tol,
        verbose=verbose,
        log_every=log_every,
    )
    summary = opt.step()
    return model, summary, opt.history
