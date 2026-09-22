"""A flat-parameter view of the PINN objective and its derivatives."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..data import DataSet
from ..hessian.hvp import HessianOperator, flatten_params, split_like
from ..losses import component_terms
from ..problems import Problem


class Objective:
    """``L(w)`` of Eq. (2) together with gradients and Hessian-vector products.

    All optimizers operate on the flattened parameter vector ``w`` produced by
    :meth:`flat_params`.
    """

    def __init__(
        self,
        problem: Problem,
        net: nn.Module,
        ds: DataSet,
        aggregation: str = "combined",
    ):
        self.problem = problem
        self.net = net
        self.ds = ds
        self.aggregation = aggregation
        self.params: List[nn.Parameter] = [p for p in net.parameters() if p.requires_grad]
        self.n_params = sum(p.numel() for p in self.params)
        self._hessian = HessianOperator(self.loss, self.params)
        self._call_count = 0

    # ------------------------------------------------------------------ #
    # parameter access
    # ------------------------------------------------------------------ #
    def flat_params(self) -> torch.Tensor:
        return flatten_params(self.params)

    def set_flat_params(self, vec: torch.Tensor) -> None:
        with torch.no_grad():
            for p, chunk in zip(self.params, split_like(vec, self.params)):
                p.copy_(chunk)

    # ------------------------------------------------------------------ #
    # loss and derivatives
    # ------------------------------------------------------------------ #
    def terms(self) -> Dict[str, torch.Tensor]:
        self._call_count += 1
        return component_terms(self.problem, self.net, self.ds, aggregation=self.aggregation)

    def loss(self) -> torch.Tensor:
        t = self.terms()
        return t["residual"] + t["boundary"]

    def component_loss(self, name: str) -> torch.Tensor:
        """Loss of one component; ``name`` in {residual, ic, bc}."""
        t = self.terms()
        if name == "residual":
            return t["residual"]
        prefix = "ic:" if name == "ic" else "bc:"
        out = None
        for k, v in t.items():
            if k.startswith(prefix):
                out = v if out is None else out + v
        assert out is not None
        return out

    def loss_and_grad(self) -> Tuple[float, torch.Tensor]:
        loss = self.loss()
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params)]
        return float(loss.detach()), flatten_params([g.detach() for g in grads])

    def grad(self) -> torch.Tensor:
        loss = self.loss()
        grads = torch.autograd.grad(loss, self.params, retain_graph=False, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params)]
        return flatten_params([g.detach() for g in grads])

    def hvp(self, v: torch.Tensor) -> torch.Tensor:
        return self._hessian.matvec(v)

    # ------------------------------------------------------------------ #
    def evaluate(self) -> float:
        # the PDE residual contains derivatives of the network, so even a plain
        # forward evaluation of L(w) has to run with autograd enabled
        with torch.enable_grad():
            return float(self.loss().detach())
