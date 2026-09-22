"""Armijo backtracking line search (Algorithm 7 of the paper, Appendix E.2).

The Armijo subroutine used inside NysNewton-CG (Algorithm 4) is a simple
backtracking line search

    eta_k = Armijo(L, w_k, grad L(w_k), d_k, eta)

which repeatedly shrinks the trial step size ``t`` by the factor ``beta`` until
the sufficient-decrease (Armijo) condition holds:

    L(w + t d) <= L(w) + alpha * t * <grad L(w), d>.

In the paper the direction passed by NNCG is ``-d_k`` (i.e. already a descent
direction) and the parameter update is ``w_{k+1} = w_k - eta_k d_k``; both
conventions are supported here since only the inner product with the gradient
matters.

Hyper-parameters used in the paper (Appendix E.2): ``alpha = 0.1``,
``beta = 0.5`` and the maximum learning rate ``eta = 1``.

This module works with *flat* parameter vectors (a single 1-D ``torch.Tensor``
holding all trainable parameters), which is the representation used by the
Hessian-vector-product / Nyström preconditioner code, while also exposing thin
adapters for ``torch.nn.Module`` objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor

__all__ = [
    "ArmijoConfig",
    "ArmijoResult",
    "ArmijoLineSearch",
    "armijo_backtracking",
    "armijo",
    "flatten_tensors",
    "flatten_params",
    "flat_grad",
    "set_flat_params",
    "make_line_search_oracle",
    "Armijo",
]


# ---------------------------------------------------------------------------
# flat parameter helpers
# ---------------------------------------------------------------------------
ModuleOrParams = Union[torch.nn.Module, Sequence[Tensor]]
LossOracle = Callable[[Tensor], Tensor]


def _parameter_list(model_or_params: ModuleOrParams) -> List[Tensor]:
    if isinstance(model_or_params, torch.nn.Module):
        return [p for p in model_or_params.parameters()]
    return list(model_or_params)


def flatten_tensors(tensors: Sequence[Tensor], dtype: Optional[torch.dtype] = None) -> Tensor:
    """Concatenate a sequence of tensors into a single 1-D tensor."""
    flats = []
    for t in tensors:
        out_dtype = dtype if dtype is not None else t.dtype
        flats.append(t.detach().reshape(-1).to(out_dtype))
    if not flats:
        return torch.zeros(0, dtype=dtype or torch.get_default_dtype())
    return torch.cat(flats)


def flatten_params(model_or_params: ModuleOrParams, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Return all parameters flattened into one vector (detached copy)."""
    return flatten_tensors(_parameter_list(model_or_params), dtype=dtype)


def flat_grad(model_or_params: ModuleOrParams, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Return ``grad`` of every parameter flattened into one vector.

    Parameters without a gradient contribute zeros, so the result always has the
    same shape as :func:`flatten_params`.
    """
    params = _parameter_list(model_or_params)
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    return flatten_tensors(grads, dtype=dtype)


@torch.no_grad()
def set_flat_params(model_or_params: ModuleOrParams, flat: Tensor) -> None:
    """In-place copy of ``flat`` into the parameters of ``model_or_params``."""
    params = _parameter_list(model_or_params)
    offset = 0
    for p in params:
        numel = p.numel()
        chunk = flat[offset : offset + numel].reshape(p.shape).to(p.dtype)
        p.copy_(chunk)
        offset += numel


def make_line_search_oracle(
    model: torch.nn.Module,
    loss_fn: Callable[[], Tensor],
    needs_grad: bool = False,
    zero_grad: bool = True,
    dtype: torch.dtype = torch.float64,
) -> LossOracle:
    """Build a scalar loss oracle ``f(flat_params) -> float64 scalar tensor``.

    ``loss_fn`` is any zero-argument callable returning the loss (typically the
    closure produced by :func:`src.pinns.loss.make_loss_fn`).  When
    ``needs_grad`` is True the closure is called with gradients enabled (some
    loss implementations need ``loss.backward()`` internally); the returned
    value is always detached.  Otherwise the loss is evaluated under
    ``torch.no_grad()``, which is what a pure line search needs.
    """

    def oracle(flat: Tensor) -> Tensor:
        set_flat_params(model, flat)
        if needs_grad:
            if zero_grad:
                model.zero_grad(set_to_none=True)
            value = loss_fn()
            detached = value.detach().reshape(())
            if torch.is_grad_enabled():
                # Recompute-free detach. The closure may have called backward()
                # itself; if not we do not force it (line search needs values).
                pass
            return detached.to(dtype)
        with torch.no_grad():
            value = loss_fn()
        return value.detach().reshape(()).to(dtype)

    return oracle


# ---------------------------------------------------------------------------
# configuration / result containers
# ---------------------------------------------------------------------------
@dataclass
class ArmijoConfig:
    """Backtracking parameters (paper Appendix E.2: alpha=0.1, beta=0.5)."""

    alpha: float = 0.1
    beta: float = 0.5
    eta: float = 1.0
    max_backtracks: int = 100
    min_step: float = 1e-16
    reduce_beta: float = 0.5  # = beta, kept explicit for readability
    verbose: bool = False

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must lie in (0, 1)")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must lie in (0, 1)")
        if self.eta <= 0:
            raise ValueError("eta (max learning rate) must be positive")

    @classmethod
    def from_dict(cls, cfg: Optional[Dict] = None) -> "ArmijoConfig":
        cfg = dict(cfg or {})
        return cls(
            alpha=float(cfg.get("alpha", 0.1)),
            beta=float(cfg.get("beta", 0.5)),
            eta=float(cfg.get("eta", 1.0)),
            max_backtracks=int(cfg.get("max_backtracks", 100)),
            min_step=float(cfg.get("min_step", 1e-16)),
            verbose=bool(cfg.get("verbose", False)),
        )


@dataclass
class ArmijoResult:
    """Outcome of one Armijo line search."""

    eta: float                       # accepted step size
    f0: float                        # loss at the current iterate
    f_new: float                     # loss at w + eta * d (after the search)
    slope: float                     # <grad, d>
    n_backtracks: int = 0
    accepted: bool = True            # sufficient decrease satisfied
    descent: bool = True             # slope < 0
    f_history: List[float] = field(default_factory=list)
    eta_history: List[float] = field(default_factory=list)

    @property
    def armijo_bound(self) -> float:
        """Right-hand side ``f0 + alpha * eta * slope`` of the Armijo condition."""
        return self.f0  # placeholder overwritten by caller if needed

    def as_dict(self) -> Dict[str, float]:
        return {
            "eta": self.eta,
            "f0": self.f0,
            "f_new": self.f_new,
            "slope": self.slope,
            "n_backtracks": self.n_backtracks,
            "accepted": self.accepted,
            "descent": self.descent,
        }


# ---------------------------------------------------------------------------
# core backtracking routine
# ---------------------------------------------------------------------------
def armijo_backtracking(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    grad: Tensor,
    direction: Tensor,
    eta: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_backtracks: int = 100,
    min_step: float = 1e-16,
    return_info: bool = False,
    verbose: bool = False,
):
    """Classic Armijo backtracking line search (Algorithm 7).

    Parameters
    ----------
    f : callable
        Scalar loss oracle mapping a flat parameter vector to a scalar tensor.
    x : Tensor
        Current (flat) parameters ``w_k``.
    grad : Tensor
        Gradient ``grad L(w_k)`` (flat).
    direction : Tensor
        Search direction ``d`` (flat). Use a descent direction such as
        ``-d_k`` for NNCG.
    eta : float
        Initial / maximum trial step size.
    alpha, beta : float
        Sufficient-decrease parameter and shrinking factor.

    Returns
    -------
    float or (float, ArmijoResult)
        The accepted step size ``eta_k`` (and diagnostics if requested).
    """
    x = x.detach().reshape(-1)
    grad = grad.detach().reshape(-1).to(x.dtype)
    direction = direction.detach().reshape(-1).to(x.dtype)

    f0_tensor = f(x)
    f0 = float(f0_tensor.detach())
    slope = float(torch.dot(grad, direction))

    info = ArmijoResult(
        eta=float(eta),
        f0=f0,
        f_new=f0,
        slope=slope,
        n_backtracks=0,
        descent=slope < 0.0,
        f_history=[f0],
        eta_history=[float(eta)],
    )

    if not math.isfinite(f0):
        # Nothing sensible to do with a non-finite current loss.
        info.accepted = False
        return (eta, info) if return_info else float(eta)

    if slope >= 0.0:
        # Not a descent direction: the Armijo condition cannot be satisfied for
        # small steps; return the (numerically safe) minimal step.
        info.accepted = False
        info.eta = float(min_step)
        if verbose:
            print(f"[armijo] non-descent direction (slope={slope:.3e})")
        return (info.eta, info) if return_info else info.eta

    step = float(eta)
    n_backtracks = 0
    f_new = f0
    accepted = False

    while n_backtracks < max_backtracks:
        candidate = x + step * direction
        f_new_tensor = f(candidate)
        f_new = float(f_new_tensor.detach())
        info.f_history.append(f_new)
        info.eta_history.append(step)

        if math.isfinite(f_new) and f_new <= f0 + alpha * step * slope:
            accepted = True
            break

        step *= beta
        n_backtracks += 1
        if verbose:
            print(f"[armijo] backtrack {n_backtracks}: t={step:.3e}, f={f_new:.6e}")
        if step < min_step:
            break

    # If we never accepted, fall back to the smallest tried step (never larger
    # than the original eta); the caller can detect this via `accepted`.
    info.eta = float(step)
    info.f_new = float(f_new)
    info.n_backtracks = int(n_backtracks)
    info.accepted = bool(accepted)

    return (info.eta, info) if return_info else info.eta


class ArmijoLineSearch:
    """Callable Armijo line search bound to a loss oracle.

    Examples
    --------
    >>> ls = ArmijoLineSearch(loss_oracle, alpha=0.1, beta=0.5)   # doctest: +SKIP
    >>> eta_k = ls(w_k, grad_k, -d_k, eta=1.0)                    # doctest: +SKIP
    """

    def __init__(
        self,
        f: LossOracle,
        alpha: float = 0.1,
        beta: float = 0.5,
        eta: float = 1.0,
        max_backtracks: int = 100,
        min_step: float = 1e-16,
        verbose: bool = False,
        config: Optional[ArmijoConfig] = None,
    ) -> None:
        if config is None:
            config = ArmijoConfig(
                alpha=alpha,
                beta=beta,
                eta=eta,
                max_backtracks=max_backtracks,
                min_step=min_step,
                verbose=verbose,
            )
        self.config = config
        self.f = f
        self.last_result: Optional[ArmijoResult] = None

    # -- paper signature: Armijo(L, w_k, grad L(w_k), d, eta) -----------------
    def __call__(
        self,
        x: Tensor,
        grad: Tensor,
        direction: Tensor,
        eta: Optional[float] = None,
        return_info: bool = False,
    ):
        cfg = self.config
        eta_val, info = armijo_backtracking(
            self.f,
            x,
            grad,
            direction,
            eta=cfg.eta if eta is None else float(eta),
            alpha=cfg.alpha,
            beta=cfg.beta,
            max_backtracks=cfg.max_backtracks,
            min_step=cfg.min_step,
            return_info=True,
            verbose=cfg.verbose,
        )
        self.last_result = info
        return (eta_val, info) if return_info else eta_val

    # -- convenience: oracle built from a model + closure --------------------
    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        loss_fn: Callable[[], Tensor],
        needs_grad: bool = False,
        dtype: torch.dtype = torch.float64,
        **kwargs,
    ) -> "ArmijoLineSearch":
        oracle = make_line_search_oracle(model, loss_fn, needs_grad=needs_grad, dtype=dtype)
        return cls(oracle, **kwargs)


def armijo(
    f: LossOracle,
    x: Tensor,
    grad: Tensor,
    direction: Tensor,
    eta: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_backtracks: int = 100,
    min_step: float = 1e-16,
    return_info: bool = False,
    verbose: bool = False,
):
    """Functional interface mirroring ``Armijo(L, w_k, grad, d, eta)``."""
    return armijo_backtracking(
        f,
        x,
        grad,
        direction,
        eta=eta,
        alpha=alpha,
        beta=beta,
        max_backtracks=max_backtracks,
        min_step=min_step,
        return_info=return_info,
        verbose=verbose,
    )


# Alias matching the paper's algorithm name.
Armijo = armijo


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # Simple quadratic: f(x) = 0.5 x^T A x with A = diag(1, 1000)
    dtype = torch.float64
    A = torch.diag(torch.tensor([1.0, 1000.0], dtype=dtype))

    def f(x: Tensor) -> Tensor:
        return 0.5 * x @ (A @ x)

    x = torch.tensor([1.0, 1.0], dtype=dtype)
    g = A @ x
    d = -g  # steepest descent (exact Newton direction here)

    eta_k, info = armijo(f, x, g, d, eta=1.0, alpha=0.1, beta=0.5, return_info=True)
    assert info.descent
    assert info.accepted, info
    assert info.f_new <= info.f0 + 0.1 * eta_k * info.slope + 1e-12

    # Non-descent direction is detected.
    _, bad = armijo(f, x, g, +g, eta=1.0, return_info=True)
    assert not bad.descent and not bad.accepted

    # Oracle built from a model + closure behaves like the armijo routine.
    model = torch.nn.Linear(2, 1, bias=False).double()
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, -2.0]], dtype=dtype))

    def closure() -> Tensor:
        return (model.weight ** 2).sum()

    ls = ArmijoLineSearch.from_model(model, closure)
    w = flatten_params(model)
    gr = flat_grad(model)
    eta_ls, info_ls = ls(w, gr, -gr, eta=1.0, return_info=True)
    assert info_ls.accepted and eta_ls > 0
    # Applying the step really decreases the loss.
    set_flat_params(model, w + eta_ls * (-gr))
    assert float(closure()) < info_ls.f0

    print("armijo.py self-test passed "
          f"(eta={eta_ls:.4f}, backtracks={info_ls.n_backtracks}, f0={info_ls.f0:.4e}).")


if __name__ == "__main__":
    _self_test()
