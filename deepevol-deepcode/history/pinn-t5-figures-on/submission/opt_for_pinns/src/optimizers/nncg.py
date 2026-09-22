"""NysNewton-CG (NNCG) optimizer -- Algorithm 4 from the paper.

NNCG is a second-order optimizer for PINN training that combines:
  * a damped Newton step computed with NystromPCG (Algorithm 6),
  * a randomized Nystrom approximation of the Hessian (Algorithm 5),
  * an Armijo backtracking line search (Algorithm 7).

Algorithm 4 (NNCG):
    for k = 0, ..., K-1:
        if k mod F == 0:
            [U, Lambda_hat] = RandomizedNystrom(H_L(w_k), s)
        d_k = NystromPCG(H_L(w_k), grad L(w_k), d_{k-1}, U, Lambda_hat, s, mu, eps, M)
        eta_k = Armijo(L, w_k, grad L(w_k), -d_k, eta)
        w_{k+1} = w_k - eta_k * d_k

The Hessian is accessed only through Hessian-vector products (Pearlmutter's
double backprop), so the optimizer is matrix-free.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

from .armijo import armijo_line_search, make_armijo_objective
from .nystrom import randomized_nystrom_approximation
from .pcg import nystrom_pcg

__all__ = ["NNCG", "flatten_params", "unflatten_params"]


# ---------------------------------------------------------------------------
# Parameter <-> flat vector helpers
# ---------------------------------------------------------------------------
def flatten_params(params) -> torch.Tensor:
    """Concatenate a list of parameter tensors into a single flat vector."""
    return torch.cat([p.detach().reshape(-1) for p in params])


def unflatten_params(params, vec: torch.Tensor) -> None:
    """Copy a flat vector back into the parameter tensors (in place)."""
    offset = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].view_as(p))
        offset += n


class NNCG:
    """NysNewton-CG optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        The PINN model whose parameters are optimized.
    loss_fn : Callable[[], torch.Tensor]
        Zero-argument callable returning the (scalar) training loss at the
        current parameter values.  Must build a fresh graph on each call.
    s : int
        Nystrom sketch size (rank of the low-rank approximation).
    F : int
        Frequency (in iterations) at which the Nystrom approximation is
        recomputed.
    mu : float
        Damping / regularization parameter for the Newton system.
    eps : float
        PCG tolerance.
    M : int
        Maximum number of PCG iterations.
    eta : float
        Initial Armijo step size.
    alpha, beta : float
        Armijo sufficient-decrease and backtracking parameters.
    dtype : torch.dtype
        Working dtype for the linear algebra (float64 recommended).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        s: int = 60,
        F: int = 20,
        mu: float = 1e-2,
        eps: float = 1e-16,
        M: int = 1000,
        eta: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.5,
        dtype: torch.dtype = torch.float64,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.s = int(s)
        self.F = int(F)
        self.mu = float(mu)
        self.eps = float(eps)
        self.M = int(M)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.dtype = dtype
        self.verbose = verbose

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.p = sum(p.numel() for p in self.params)

        # Cached Nystrom factors.
        self.U: Optional[torch.Tensor] = None
        self.lam: Optional[torch.Tensor] = None

        # Warm-start direction.
        self.d_prev: Optional[torch.Tensor] = None

        # Diagnostics.
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Hessian-vector product (Pearlmutter)
    # ------------------------------------------------------------------
    def hvp(self, vec: torch.Tensor) -> torch.Tensor:
        """Compute H_L(w) @ vec via double backprop.

        `vec` is a flat vector in the working dtype; the result is returned in
        the same dtype.
        """
        params = self.params
        # Ensure grads are enabled for the graph construction.
        loss = self.loss_fn()
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        flat_grad = torch.cat([g.reshape(-1) for g in grads])

        # Cast the direction to the parameter dtype for the dot product.
        param_dtype = flat_grad.dtype
        v = vec.to(param_dtype)
        dot = torch.dot(flat_grad, v)

        hvp = torch.autograd.grad(dot, params, retain_graph=False, allow_unused=True)
        hvp = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp, params)]
        flat_hvp = torch.cat([h.reshape(-1) for h in hvp])
        return flat_hvp.to(self.dtype)

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------
    def grad(self) -> torch.Tensor:
        """Return the flat gradient of the loss at the current parameters."""
        loss = self.loss_fn()
        grads = torch.autograd.grad(loss, self.params, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params)]
        return torch.cat([g.reshape(-1) for g in grads]).to(self.dtype)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def step(self, k: int) -> Dict[str, float]:
        """Perform a single NNCG iteration (indexed by k)."""
        # 1. Recompute the Nystrom approximation every F iterations.
        if (k % self.F == 0) or (self.U is None):
            self.U, self.lam = randomized_nystrom_approximation(
                self.hvp, self.s, p=self.p, dtype=self.dtype
            )

        # 2. Compute the Newton direction via NystromPCG (warm-started).
        g = self.grad()
        res = nystrom_pcg(
            self.hvp,
            g,
            self.U,
            self.lam,
            mu=self.mu,
            tol=self.eps,
            max_iters=self.M,
            x0=self.d_prev,
        )
        d = res.x

        # 3. Armijo line search along -d.
        x = flatten_params(self.params).to(self.dtype)
        f, set_params = make_armijo_objective(self.model, self.loss_fn, x, d)
        # f(t) evaluates loss at x + t*d; we search along direction -d, so we
        # pass d = -d to the objective.
        f_neg, set_neg = make_armijo_objective(self.model, self.loss_fn, x, -d)
        eta = armijo_line_search(
            f_neg,
            x,
            g,
            -d,
            t=self.eta,
            alpha=self.alpha,
            beta=self.beta,
        )

        # 4. Update parameters: w_{k+1} = w_k - eta * d.
        new_x = x - eta * d
        unflatten_params(self.params, new_x.to(self.params[0].dtype))

        # Cache direction for warm-starting the next PCG solve.
        self.d_prev = d.detach().clone()

        info = {
            "iter": float(k),
            "eta": float(eta),
            "pcg_iters": float(res.iters),
            "pcg_residual": float(res.residual_norm),
            "pcg_converged": float(res.converged),
            "grad_norm": float(torch.linalg.norm(g)),
            "step_norm": float(torch.linalg.norm(eta * d)),
        }
        self.history.append(info)
        if self.verbose:
            print(
                f"[NNCG] k={k} eta={eta:.3e} pcg_iters={res.iters} "
                f"||g||={info['grad_norm']:.3e}"
            )
        return info

    def run(self, n_steps: int = 2000, log_every: int = 0) -> List[Dict[str, float]]:
        """Run `n_steps` NNCG iterations."""
        for k in range(n_steps):
            self.step(k)
            if log_every and ((k + 1) % log_every == 0):
                print(f"[NNCG] completed {k + 1}/{n_steps} steps")
        return self.history
