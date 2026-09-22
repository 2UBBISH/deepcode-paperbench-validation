"""Armijo backtracking line search (Algorithm 7 of the paper).

Given a current iterate ``x``, a descent direction ``d`` and an initial step
size ``t``, the Armijo (sufficient decrease) condition is

    f(x + t d) <= f(x) + alpha * t * grad f(x)^T d

The step size is repeatedly shrunk by a factor ``beta`` until the condition
holds (or a maximum number of backtracking steps is reached).

The loss oracle is provided as a zero-argument closure (see
``src.loss.make_loss_fn``) so that the line search is agnostic to how the loss
is computed.  The gradient ``grad f(x)`` must be supplied by the caller
(typically already available from the current optimisation step).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

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
    min_step: float = 1e-16,
) -> float:
    """Backtracking Armijo line search.

    Parameters
    ----------
    f : Callable[[], torch.Tensor]
        Zero-argument closure returning the (scalar) loss at the *current*
        parameters.  The parameters are mutated in place by this routine, so
        the closure must read the live parameter values.
    x : torch.Tensor
        Current parameter vector (flat, shape ``(p,)``).  Used only to compute
        the trial points; the caller is responsible for actually setting the
        parameters to ``x + t d``.
    grad : torch.Tensor
        Gradient of ``f`` at ``x`` (flat, shape ``(p,)``).
    d : torch.Tensor
        Descent direction (flat, shape ``(p,)``).
    t : float
        Initial step size (``eta`` in Algorithm 4).
    alpha : float
        Sufficient-decrease parameter (``alpha`` in Algorithm 7).
    beta : float
        Backtracking shrink factor (``beta`` in Algorithm 7).
    max_iter : int
        Maximum number of backtracking iterations.
    min_step : float
        Lower bound on the step size; the loop stops once ``t < min_step``.

    Returns
    -------
    float
        The accepted step size ``t``.
    """
    # Directional derivative  grad f(x)^T d  (scalar).
    with torch.no_grad():
        f0 = float(f().detach())
        dir_deriv = float(torch.dot(grad.detach().flatten(), d.detach().flatten()))

    # If the direction is not a descent direction, return the initial step
    # unchanged (the caller may still accept it or handle it separately).
    if dir_deriv >= 0:
        return t

    step = float(t)
    for _ in range(max_iter):
        if step < min_step:
            break
        # Evaluate f(x + step * d).  The caller's closure reads the live
        # parameters, so we temporarily set them here via the provided setter.
        # To keep this routine self-contained we rely on the closure being
        # bound to a parameter-setting helper through ``x`` mutation.
        with torch.no_grad():
            x.add_(d, alpha=step)
        try:
            f_new = float(f().detach())
        finally:
            with torch.no_grad():
                x.sub_(d, alpha=step)

        if f_new <= f0 + alpha * step * dir_deriv:
            return step
        step *= beta

    return step


def armijo_line_search_with_setter(
    f: Callable[[], torch.Tensor],
    set_params: Callable[[torch.Tensor], None],
    get_params: Callable[[], torch.Tensor],
    grad: torch.Tensor,
    d: torch.Tensor,
    t: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    max_iter: int = 100,
    min_step: float = 1e-16,
) -> Tuple[float, float]:
    """Armijo line search operating on a model through getter/setter callbacks.

    This is the variant used by NysNewton-CG (Algorithm 4): the parameters live
    inside an ``nn.Module`` and are accessed through ``get_params`` /
    ``set_params`` rather than a raw flat tensor.

    Returns
    -------
    (step, f_new) : Tuple[float, float]
        The accepted step size and the loss value at the accepted point.  The
        model is left at the accepted point ``x + step * d``.
    """
    with torch.no_grad():
        x0 = get_params().detach().clone()
        f0 = float(f().detach())
        dir_deriv = float(torch.dot(grad.detach().flatten(), d.detach().flatten()))

    if dir_deriv >= 0:
        return t, f0

    step = float(t)
    f_new = f0
    for _ in range(max_iter):
        if step < min_step:
            break
        with torch.no_grad():
            set_params(x0 + step * d)
        f_new = float(f().detach())
        if f_new <= f0 + alpha * step * dir_deriv:
            return step, f_new
        step *= beta

    # No step satisfied the condition: restore the original point.
    with torch.no_grad():
        set_params(x0)
    return 0.0, f0


# Alias matching the name referenced in ``optimizers/__init__.py``.
armijo = armijo_line_search


__all__ = [
    "armijo_line_search",
    "armijo_line_search_with_setter",
    "armijo",
]
