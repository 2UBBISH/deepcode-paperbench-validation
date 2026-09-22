"""Hessian-vector products via the Pearlmutter trick (Pearlmutter 1994).

The PINN loss ``L(w)`` is a scalar function of the flat parameter vector ``w``.
Given a vector ``v`` of the same shape as ``w``, we want ``H_L(w) v`` without
ever materialising the Hessian.  The Pearlmutter trick computes this with a
single double-backward pass:

    H v = d/dw [ (grad_w L)^T v ]

i.e. first compute ``g = grad_w L`` with ``create_graph=True``, then compute the
gradient of the scalar ``g^T v`` w.r.t. ``w``.  The result is exactly ``H v``.

This module exposes:

* :func:`hvp` -- functional Hessian-vector product for an ``nn.Module`` and a
  zero-argument loss closure (the closure must read the *live* parameters of the
  module, as produced by :func:`src.loss.make_loss_fn`).
* :class:`HVPOperator` -- a small callable wrapper that caches the loss closure
  and model so it can be passed directly to routines such as
  :func:`src.optimizers.nystrom.randomized_nystrom_approximation` and
  :func:`src.optimizers.pcg.nystrom_pcg`, which expect a matvec callable.
* :func:`hvp_from_grad` -- lower-level helper that takes an already-computed
  gradient tensor (with graph) and a vector, avoiding a redundant forward pass.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn


__all__ = ["hvp", "hvp_from_grad", "HVPOperator", "flat_grad"]


def _flat_params(model: nn.Module) -> torch.Tensor:
    """Return the concatenation of all model parameters (flat vector)."""
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def flat_grad(
    loss_fn: Callable[[], torch.Tensor],
    model: nn.Module,
    create_graph: bool = True,
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute the flat gradient of ``loss_fn`` w.r.t. the model parameters.

    Parameters
    ----------
    loss_fn:
        Zero-argument closure returning the scalar loss.  It must read the live
        parameters of ``model`` (e.g. produced by ``make_loss_fn``).
    model:
        The ``nn.Module`` whose parameters are differentiated.
    create_graph:
        If ``True`` the returned gradient keeps the autograd graph so that a
        second backward pass (the HVP) is possible.
    retain_graph:
        Passed through to ``torch.autograd.grad``.

    Returns
    -------
    torch.Tensor
        Flat gradient vector of shape ``(p,)``.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn()
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = []
    for p, g in zip(params, grads):
        if g is None:
            g = torch.zeros_like(p)
        flat.append(g.reshape(-1))
    return torch.cat(flat)


def hvp_from_grad(
    grad: torch.Tensor,
    params: List[torch.Tensor],
    v: torch.Tensor,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Hessian-vector product given a pre-computed flat gradient with graph.

    Computes ``d/dw [ grad^T v ]`` which equals ``H v``.

    Parameters
    ----------
    grad:
        Flat gradient tensor of shape ``(p,)`` that was produced with
        ``create_graph=True`` (so it is connected to ``params``).
    params:
        List of leaf parameter tensors the gradient was taken w.r.t.
    v:
        Flat vector of shape ``(p,)``.
    retain_graph:
        Whether to retain the graph of the second backward pass.

    Returns
    -------
    torch.Tensor
        Flat Hessian-vector product of shape ``(p,)``.
    """
    if v.shape != grad.shape:
        v = v.reshape(grad.shape)
    # grad^T v  -> scalar
    grad_v = torch.dot(grad, v)
    hvp_grads = torch.autograd.grad(
        grad_v,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flat = []
    for p, g in zip(params, hvp_grads):
        if g is None:
            g = torch.zeros_like(p)
        flat.append(g.reshape(-1))
    return torch.cat(flat)


def hvp(
    loss_fn: Callable[[], torch.Tensor],
    model: nn.Module,
    v: torch.Tensor,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Hessian-vector product ``H_L(w) v`` via the Pearlmutter trick.

    Parameters
    ----------
    loss_fn:
        Zero-argument closure returning the scalar loss, reading live params.
    model:
        The ``nn.Module`` whose parameters define ``w``.
    v:
        Flat vector of shape ``(p,)`` (or any shape that reshapes to ``(p,)``).
    retain_graph:
        Whether to retain the graph of the second backward pass.

    Returns
    -------
    torch.Tensor
        Flat Hessian-vector product of shape ``(p,)``.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grad = flat_grad(loss_fn, model, create_graph=True, retain_graph=True)
    return hvp_from_grad(grad, params, v, retain_graph=retain_graph)


class HVPOperator:
    """Callable matvec operator wrapping the Hessian of a PINN loss.

    Instances can be passed directly to routines that expect a matrix-free
    operator ``M`` (e.g. the randomized Nyström approximation and PCG).

    Example
    -------
    >>> op = HVPOperator(loss_fn, model)
    >>> Hv = op(v)
    """

    def __init__(
        self,
        loss_fn: Callable[[], torch.Tensor],
        model: nn.Module,
        retain_graph: bool = False,
    ) -> None:
        self.loss_fn = loss_fn
        self.model = model
        self.retain_graph = retain_graph
        self._params = [p for p in model.parameters() if p.requires_grad]
        self._p = sum(p.numel() for p in self._params)

    @property
    def shape(self):
        return (self._p, self._p)

    @property
    def numel(self) -> int:
        return self._p

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return hvp(
            self.loss_fn,
            self.model,
            v,
            retain_graph=self.retain_graph,
        )

    def matvec(self, v: torch.Tensor) -> torch.Tensor:
        """Alias for ``__call__`` for API compatibility."""
        return self(v)
