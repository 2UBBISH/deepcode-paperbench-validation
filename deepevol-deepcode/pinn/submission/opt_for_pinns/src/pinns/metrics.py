"""L2RE evaluation metric for PINNs (Sec. 2.2, Eq. (3)).

Let ``y = (y_i)_{i=1}^n`` be the PINN prediction and ``y' = (y'_i)_{i=1}^n`` the
ground truth.  The paper defines the *l2 relative error* (L2RE) as

    L2RE = sqrt( sum_i (y_i - y'_i)^2 / sum_i (y'_i)^2 )
         = sqrt( ||y - y'||_2^2 / ||y'||_2^2 ).                        (Eq. 3)

The evaluation set consists of **all** points of the ``255 x 100`` interior grid
of the problem domain, together with the ``257`` initial-condition points and the
``101`` points used for each boundary condition.

This module exposes

* :func:`l2re` -- the metric itself on two flat tensors,
* :func:`region_predictions` / :func:`region_targets` -- model/analytical values
  on the full evaluation set, split by region (interior / initial / boundary),
* :func:`compute_l2re` -- the headline number used for the paper's tables,
* :func:`evaluate` -- a richer report with per-region L2RE (useful for debugging
  and for the per-component analysis of Sec. 5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

try:  # pragma: no cover - package-relative import when used as a library
    from .problems import PDEProblem, get_problem
    from .sampling import PINNSampler, build_sampler, full_evaluation_points
except ImportError:  # pragma: no cover
    from src.pinns.problems import PDEProblem, get_problem  # type: ignore
    from src.pinns.sampling import (  # type: ignore
        PINNSampler,
        build_sampler,
        full_evaluation_points,
    )

__all__ = [
    "REGION_INTERIOR",
    "REGION_INITIAL",
    "REGION_BOUNDARY",
    "L2REReport",
    "l2re",
    "l2re_from_tensors",
    "predict",
    "analytic",
    "region_predictions",
    "region_targets",
    "evaluation_points",
    "compute_l2re",
    "evaluate",
    "solution_error",
]

REGION_INTERIOR = "interior"
REGION_INITIAL = "initial"
REGION_BOUNDARY = "boundary"


# ---------------------------------------------------------------------------
# core metric
# ---------------------------------------------------------------------------
def l2re(y: Tensor, y_true: Tensor, eps: float = 0.0) -> float:
    """L2 relative error of Eq. (3).

    Parameters
    ----------
    y:
        PINN prediction (any shape; flattened internally).
    y_true:
        Ground truth of the same shape.
    eps:
        Optional ridge added to the denominator for numerical safety (the paper
        uses 0.0, i.e. a plain relative error).

    Returns
    -------
    float
        ``sqrt(||y - y'||_2^2 / ||y'||_2^2)``.
    """
    diff = (y.reshape(-1) - y_true.reshape(-1)).to(torch.float64)
    ref = y_true.reshape(-1).to(torch.float64)
    num = torch.sum(diff * diff)
    den = torch.sum(ref * ref)
    if eps > 0.0:
        den = den + eps
    if float(den) <= 0.0:
        # Degenerate reference: fall back to the absolute error.
        return float(torch.sqrt(num))
    return float(torch.sqrt(num / den))


# American-spelling alias, kept because the paper alternates between the two.
l2re_from_tensors = l2re


# ---------------------------------------------------------------------------
# model / analytical evaluation
# ---------------------------------------------------------------------------
def predict(
    model: torch.nn.Module,
    points: Tensor,
    batch_size: Optional[int] = None,
) -> Tensor:
    """Evaluate ``u(x; w)`` on ``points`` and return a flat ``(n,)`` tensor.

    ``batch_size`` splits the evaluation into chunks to bound memory usage on the
    large ``255 x 100`` grid (no effect on the result).
    """
    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if batch_size is None or points.shape[0] <= batch_size:
                out = model(points)
            else:
                chunks: List[Tensor] = []
                for start in range(0, points.shape[0], batch_size):
                    chunks.append(model(points[start : start + batch_size]))
                out = torch.cat(chunks, dim=0)
    finally:
        if model_was_training:
            model.train()
    return out.reshape(-1)


def analytic(problem: PDEProblem, points: Tensor, batch_size: Optional[int] = None) -> Tensor:
    """Evaluate the analytical (ground-truth) solution on ``points``."""
    with torch.no_grad():
        if batch_size is None or points.shape[0] <= batch_size:
            out = problem.exact(points)
        else:
            chunks: List[Tensor] = []
            for start in range(0, points.shape[0], batch_size):
                chunks.append(problem.exact(points[start : start + batch_size]))
            out = torch.cat(chunks, dim=0)
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# evaluation set assembly
# ---------------------------------------------------------------------------
def _interior_points(problem: PDEProblem, sampler: Optional[PINNSampler], device, dtype) -> Tensor:
    if sampler is not None and getattr(sampler, "eval_interior_points", None) is not None:
        return sampler.eval_interior_points.to(device=device, dtype=dtype)
    points, _, _ = full_evaluation_points(problem, device=device, dtype=dtype)
    return points


def _condition_groups(problem: PDEProblem, sampler: Optional[PINNSampler], device, dtype) -> List[Tensor]:
    if sampler is not None and hasattr(sampler, "condition_point_groups"):
        groups = list(sampler.condition_point_groups)
        return [g.to(device=device, dtype=dtype) for g in groups]
    _, groups = full_evaluation_points(problem, device=device, dtype=dtype)
    return [g.to(device=device, dtype=dtype) for g in groups]


def _is_initial(problem: PDEProblem, index: int) -> bool:
    """Whether condition ``index`` is an initial condition (vs. a boundary one)."""
    try:
        conds = problem.conditions()
    except Exception:  # pragma: no cover - defensive
        return False
    if index >= len(conds):
        return False
    kind = getattr(conds[index], "kind", "")
    name = str(getattr(conds[index], "name", "")).lower()
    if kind in ("ic", "ic_derivative", "initial"):
        return True
    return "ic" in name or "initial" in name


def evaluation_points(
    problem: PDEProblem,
    sampler: Optional[PINNSampler] = None,
    device="cpu",
    dtype=None,
) -> Tuple[Tensor, Tensor, List[Tensor], List[bool]]:
    """Assemble the full L2RE evaluation set.

    Returns
    -------
    interior, interior_target, condition_groups, condition_is_initial
        * ``interior``: ``(n_int, 2)`` interior points of the ``255 x 100`` grid.
        * ``interior_target``: analytical values on ``interior``.
        * ``condition_groups``: list of point tensors, one per boundary/initial
          condition (periodic conditions contribute two groups).
        * ``condition_is_initial``: flag per group.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    interior = _interior_points(problem, sampler, device, dtype)
    groups = _condition_groups(problem, sampler, device, dtype)
    flags = [_is_initial(problem, i) for i in range(len(groups))]
    interior_target = analytic(problem, interior)
    return interior, interior_target, groups, flags


def region_predictions(
    model: torch.nn.Module,
    problem: PDEProblem,
    sampler: Optional[PINNSampler] = None,
    device="cpu",
    dtype=None,
    batch_size: Optional[int] = 50000,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """Predictions and ground truth on the full evaluation set, per region.

    The concatenation of the returned regions is exactly the point set the paper
    uses for L2RE: the ``255 x 100`` interior grid plus the 257 IC points plus the
    101 points of each boundary condition.
    """
    interior, interior_target, groups, flags = evaluation_points(
        problem, sampler, device=device, dtype=dtype
    )
    preds: Dict[str, Tensor] = {REGION_INTERIOR: predict(model, interior, batch_size)}
    targets: Dict[str, Tensor] = {REGION_INTERIOR: interior_target}

    init_pred: List[Tensor] = []
    init_true: List[Tensor] = []
    bnd_pred: List[Tensor] = []
    bnd_true: List[Tensor] = []
    for points, is_initial in zip(groups, flags):
        p = predict(model, points, batch_size)
        t = analytic(problem, points)
        if is_initial:
            init_pred.append(p)
            init_true.append(t)
        else:
            bnd_pred.append(p)
            bnd_true.append(t)
    if init_pred:
        preds[REGION_INITIAL] = torch.cat(init_pred)
        targets[REGION_INITIAL] = torch.cat(init_true)
    if bnd_pred:
        preds[REGION_BOUNDARY] = torch.cat(bnd_pred)
        targets[REGION_BOUNDARY] = torch.cat(bnd_true)
    return preds, targets


def region_targets(
    problem: PDEProblem,
    sampler: Optional[PINNSampler] = None,
    device="cpu",
    dtype=None,
) -> Dict[str, Tensor]:
    """Ground-truth values on the full evaluation set, per region."""
    interior, interior_target, groups, flags = evaluation_points(
        problem, sampler, device=device, dtype=dtype
    )
    targets: Dict[str, Tensor] = {REGION_INTERIOR: interior_target}
    init_true = [analytic(problem, g) for g, f in zip(groups, flags) if f]
    bnd_true = [analytic(problem, g) for g, f in zip(groups, flags) if not f]
    if init_true:
        targets[REGION_INITIAL] = torch.cat(init_true)
    if bnd_true:
        targets[REGION_BOUNDARY] = torch.cat(bnd_true)
    return targets


# ---------------------------------------------------------------------------
# report objects
# ---------------------------------------------------------------------------
@dataclass
class L2REReport:
    """L2RE computed over the paper's full evaluation set, with a breakdown."""

    total: float
    n_points: int
    per_region: Dict[str, float] = field(default_factory=dict)
    n_per_region: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        out = {"l2re": self.total, "l2re_num_points": float(self.n_points)}
        for name, value in self.per_region.items():
            out[f"l2re_{name}"] = value
        return out

    def __float__(self) -> float:  # convenience: ``float(report)``
        return float(self.total)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        regs = ", ".join(f"{k}={v:.4e}" for k, v in self.per_region.items())
        return f"L2REReport(total={self.total:.4e}, n={self.n_points}, {regs})"


# ---------------------------------------------------------------------------
# high-level entry points
# ---------------------------------------------------------------------------
def compute_l2re(
    model: torch.nn.Module,
    problem,
    sampler: Optional[PINNSampler] = None,
    device="cpu",
    dtype=None,
    batch_size: Optional[int] = 50000,
) -> float:
    """L2RE of the PINN solution over the paper's full evaluation set.

    ``problem`` may be a :class:`~src.pinns.problems.PDEProblem` instance or the
    name of a problem (``"convection"``/``"reaction"``/``"wave"``).
    """
    if isinstance(problem, str):
        problem = get_problem(problem)
    if sampler is None:
        sampler = build_sampler(problem, device=device, dtype=dtype)
    preds, targets = region_predictions(
        model, problem, sampler=sampler, device=device, dtype=dtype, batch_size=batch_size
    )
    y = torch.cat([preds[k] for k in _ordered_regions(preds)])
    y_true = torch.cat([targets[k] for k in _ordered_regions(targets)])
    return l2re(y, y_true)


def evaluate(
    model: torch.nn.Module,
    problem,
    sampler: Optional[PINNSampler] = None,
    device="cpu",
    dtype=None,
    batch_size: Optional[int] = 50000,
) -> L2REReport:
    """Full L2RE report (headline metric + per-region breakdown)."""
    if isinstance(problem, str):
        problem = get_problem(problem)
    if sampler is None:
        sampler = build_sampler(problem, device=device, dtype=dtype)
    preds, targets = region_predictions(
        model, problem, sampler=sampler, device=device, dtype=dtype, batch_size=batch_size
    )
    order = _ordered_regions(preds)
    y = torch.cat([preds[k] for k in order])
    y_true = torch.cat([targets[k] for k in order])
    total = l2re(y, y_true)
    per_region = {k: l2re(preds[k], targets[k]) for k in order}
    n_per_region = {k: int(preds[k].numel()) for k in order}
    return L2REReport(
        total=total,
        n_points=int(y.numel()),
        per_region=per_region,
        n_per_region=n_per_region,
    )


def solution_error(
    model: torch.nn.Module,
    problem,
    sampler: Optional[PINNSampler] = None,
    device="cpu",
    dtype=None,
    batch_size: Optional[int] = 50000,
) -> Dict[str, float]:
    """Convenience dict of ``{"l2re": ..., "l2re_interior": ..., ...}``."""
    return evaluate(
        model, problem, sampler=sampler, device=device, dtype=dtype, batch_size=batch_size
    ).as_dict()


def _ordered_regions(mapping: Dict[str, Tensor]) -> List[str]:
    order = [REGION_INTERIOR, REGION_INITIAL, REGION_BOUNDARY]
    return [k for k in order if k in mapping] + [k for k in mapping if k not in order]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="L2RE of an (untrained) PINN.")
    parser.add_argument("problem", nargs="?", default="convection")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from .model import make_pinn  # type: ignore

    prob = get_problem(args.problem)
    net = make_pinn(in_dim=2, out_dim=1, width=args.width, seed=args.seed)
    print(evaluate(net, prob))
