"""L2 relative error (L2RE), Section 2.2."""

from __future__ import annotations

import torch
import torch.nn as nn

from .problems import Problem


def _net_dtype(net) -> torch.dtype | None:
    if isinstance(net, nn.Module):
        try:
            return next(net.parameters()).dtype
        except StopIteration:  # pragma: no cover - parameterless module
            return None
    return None


@torch.no_grad()
def l2_relative_error(net: nn.Module, problem: Problem, X: torch.Tensor | None = None) -> float:
    """``||y - y'||_2 / ||y'||_2`` on the evaluation points."""
    X = problem.evaluation_points() if X is None else X
    dtype = _net_dtype(net)
    if dtype is not None and X.dtype != dtype:
        X = X.to(dtype)
    y = net(X)
    y_true = problem.exact(X)
    return float(torch.linalg.norm(y - y_true) / torch.linalg.norm(y_true))


@torch.no_grad()
def mean_absolute_error(net: nn.Module, problem: Problem, X: torch.Tensor | None = None) -> float:
    X = problem.evaluation_points() if X is None else X
    dtype = _net_dtype(net)
    if dtype is not None and X.dtype != dtype:
        X = X.to(dtype)
    return float((net(X) - problem.exact(X)).abs().mean())
