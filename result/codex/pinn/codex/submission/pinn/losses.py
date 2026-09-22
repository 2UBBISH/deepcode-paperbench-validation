"""The PINN objective of Eq. (2) and its individual components.

Eq. (2) of the paper reads

    L(w) = 1/(2 n_res) sum_i (D[u(x_r^i; w), x_r^i])^2
           + 1/(2 n_bc) sum_j (B[u(x_b^j; w), x_b^j])^2

so the residual term and the (combined) initial/boundary term are weighted
equally.  ``aggregation="combined"`` reproduces Eq. (2) literally (a single
mean over all initial *and* boundary points); ``aggregation="per_condition"``
averages each initial/boundary condition separately, which is the convention
used by much of the PINN literature.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .data import DataSet
from .problems import Problem

def component_terms(
    problem: Problem, net: nn.Module, ds: DataSet, aggregation: str = "combined"
) -> Dict[str, torch.Tensor]:
    """Mean squared loss of each component (residual, IC, BC).

    The returned tensors are already the *full* contributions to Eq. (2), i.e.
    they include the ``1/(2 n)`` prefactor.
    """
    dtype = next(net.parameters()).dtype

    def cast(X: torch.Tensor) -> torch.Tensor:
        return X if X.dtype == dtype else X.to(dtype)

    # ---- residual ---------------------------------------------------- #
    r = problem.residual(net, cast(ds.X_res))
    n_res = ds.X_res.shape[0]

    ic_res = {t.name: t.fn(net, cast(ds.X_ic[t.name])) for t in problem.ic_terms()}
    bc_res = {}
    for t in problem.bc_terms():
        bc_res[t.name] = t.fn(net, cast(ds.X_bc[t.name]))

    terms: Dict[str, torch.Tensor] = {}
    terms["residual"] = r.pow(2).sum() / (2.0 * n_res)

    if aggregation == "combined":
        total_n = sum(v.shape[0] for v in ds.X_ic.values()) + sum(
            v.shape[0] for v in ds.X_bc.values()
        )
        s = torch.zeros((), dtype=r.dtype, device=r.device)
        for v in ic_res.values():
            s = s + v.pow(2).sum()
        for v in bc_res.values():
            s = s + v.pow(2).sum()
        combined = s / (2.0 * total_n)
        # distribute proportionally so that the individual components remain
        # available (their sum equals the combined term)
        for name, v in ic_res.items():
            terms[f"ic:{name}"] = v.pow(2).sum() / (2.0 * total_n)
        for name, v in bc_res.items():
            terms[f"bc:{name}"] = v.pow(2).sum() / (2.0 * total_n)
        terms["boundary"] = combined
    elif aggregation == "per_condition":
        s = torch.zeros((), dtype=r.dtype, device=r.device)
        for name, v in ic_res.items():
            t = v.pow(2).sum() / (2.0 * v.shape[0])
            terms[f"ic:{name}"] = t
            s = s + t
        for name, v in bc_res.items():
            t = v.pow(2).sum() / (2.0 * v.shape[0])
            terms[f"bc:{name}"] = t
            s = s + t
        terms["boundary"] = s
    else:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    return terms


def pinn_loss(problem: Problem, net: nn.Module, ds: DataSet, aggregation: str = "combined"):
    """Total PINN loss, ``L(w)`` of Eq. (2)."""
    terms = component_terms(problem, net, ds, aggregation=aggregation)
    return terms["residual"] + terms["boundary"]


def component_losses(problem: Problem, net: nn.Module, ds: DataSet, aggregation: str = "combined"):
    """The three building blocks used in Figures 3 and 7.

    Returns a dict with the keys ``"residual"``, ``"ic"`` and ``"bc"``.
    """
    terms = component_terms(problem, net, ds, aggregation=aggregation)
    residual = terms["residual"]
    ic = torch.zeros((), dtype=residual.dtype, device=residual.device)
    bc = torch.zeros((), dtype=residual.dtype, device=residual.device)
    for k, v in terms.items():
        if k.startswith("ic:"):
            ic = ic + v
        elif k.startswith("bc:"):
            bc = bc + v
    return {"residual": residual, "ic": ic, "bc": bc}
