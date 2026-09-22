"""Hessian-vector products (Pearlmutter, 1994) for the PINN loss."""

from __future__ import annotations

from typing import Callable, List, Sequence

import torch


def parameters_of(net: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in net.parameters() if p.requires_grad]


def flatten_params(params: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def split_like(vec: torch.Tensor, params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Split a flat vector into chunks shaped like ``params``."""
    out, i = [], 0
    for p in params:
        n = p.numel()
        out.append(vec[i : i + n].reshape(p.shape))
        i += n
    return out


class HessianOperator:
    """Matrix-free Hessian of a scalar function of the network parameters.

    ``loss_fn`` is a callable returning a *scalar* tensor (typically the sum of
    the PINN loss components of interest).  It is re-evaluated on every
    ``matvec`` call so that the graph is always fresh.
    """

    def __init__(self, loss_fn: Callable[[], torch.Tensor], params: Sequence[torch.Tensor]):
        self.loss_fn = loss_fn
        self.params = list(params)
        self.n = sum(p.numel() for p in self.params)

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        return hvp(self.loss_fn, self.params, v)

    # convenience ------------------------------------------------------ #
    def __matmul__(self, v: torch.Tensor) -> torch.Tensor:
        return self.matvec(v)


def hvp(
    loss_fn: Callable[[], torch.Tensor],
    params: Sequence[torch.Tensor],
    v: torch.Tensor,
) -> torch.Tensor:
    """``H v`` for the Hessian of ``loss_fn()`` w.r.t. ``params``.

    Parameters that do not enter the graph contribute a zero row/column to the
    Hessian.  This happens for instance to the output bias of the network when
    ``loss_fn`` is the *residual* term only, since the residual involves
    derivatives of ``u`` and therefore drops any constant offset.
    """
    loss = loss_fn()
    grads = torch.autograd.grad(
        loss, list(params), create_graph=True, retain_graph=True, allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    flat = torch.cat([g.reshape(-1) for g in grads])
    out = torch.autograd.grad(
        flat,
        list(params),
        grad_outputs=v.reshape(-1).to(flat.dtype),
        retain_graph=False,
        allow_unused=True,
    )
    out = [o if o is not None else torch.zeros_like(p) for o, p in zip(out, params)]
    return torch.cat([o.reshape(-1) for o in out]).detach().to(v.dtype)


def loss_hessian_matrix(loss_fn: Callable[[], torch.Tensor], params: Sequence[torch.Tensor]) -> torch.Tensor:
    """Dense Hessian (only used in tests / for small networks)."""
    n = sum(p.numel() for p in params)
    dtype = params[0].dtype if len(params) else torch.get_default_dtype()
    H = torch.zeros(n, n, dtype=dtype)
    for i in range(n):
        e = torch.zeros(n, dtype=dtype)
        e[i] = 1.0
        H[:, i] = hvp(loss_fn, params, e)
    return 0.5 * (H + H.T)
