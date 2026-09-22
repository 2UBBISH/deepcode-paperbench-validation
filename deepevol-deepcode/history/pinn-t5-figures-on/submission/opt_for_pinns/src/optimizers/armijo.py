"""Armijo backtracking line search (Algorithm 7 in the paper).

Given a function ``f``, current point ``x``, current gradient ``grad`` and a
descent direction ``d`` (so that we step ``x + t * d``), find a step size
``t`` satisfying the sufficient-decrease (Armijo) condition::

    f(x + t d) <= f(x) + alpha * t * grad^T d

The step size is reduced geometrically by ``beta`` until the condition holds.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch


def armijo_line_search(
    f: Callable[[], torch.Tensor],
    x: torch.Tensor,
    grad: torch.Tensor,
    d: torch.Tensor,
    t: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_iter: int = 100,
    f0: Optional[torch.Tensor] = None,
) -> float:
    """Backtracking Armijo line search.

    Parameters
    ----------
    f : callable
        Zero-argument callable returning the (scalar) objective value at the
        *current* parameter values.  The caller is responsible for writing the
        candidate point ``x + t d`` into the model parameters before calling
        ``f`` (see :func:`make_armijo_objective`).
    x : torch.Tensor
        Flattened current parameter vector (used only for the directional
        derivative ``grad^T d``).
    grad : torch.Tensor
        Flattened gradient at ``x``.
    d : torch.Tensor
        Flattened descent direction.
    t : float
        Initial step size.
    alpha : float
        Sufficient decrease parameter (default 0.1).
    beta : float
        Backtracking factor (default 0.5).
    max_iter : int
        Maximum number of backtracking iterations.
    f0 : torch.Tensor, optional
        Pre-computed ``f(x)``.  If ``None`` it is evaluated.

    Returns
    -------
    float
        The accepted step size ``t``.
    """
    if f0 is None:
        f0 = f()

    # Directional derivative grad^T d (should be negative for a descent dir).
    dir_deriv = torch.dot(grad.reshape(-1), d.reshape(-1))

    for _ in range(max_iter):
        # The caller has already set params = x + t*d before calling f().
        f_new = f()
        if torch.isfinite(f_new) and f_new <= f0 + alpha * t * dir_deriv:
            return t
        t *= beta

    return t


def make_armijo_objective(
    model: torch.nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    x: torch.Tensor,
    d: torch.Tensor,
):
    """Build a callable ``f(t)`` that sets ``params = x + t*d`` and evaluates loss.

    Returns a function ``f(t)`` returning the scalar loss tensor.
    """
    params = [p for p in model.parameters()]

    def set_params(vec: torch.Tensor) -> None:
        offset = 0
        for p in params:
            numel = p.numel()
            p.data.copy_(vec[offset:offset + numel].view_as(p))
            offset += numel

    def f(t: float) -> torch.Tensor:
        set_params(x + t * d)
        return loss_fn()

    return f, set_params
