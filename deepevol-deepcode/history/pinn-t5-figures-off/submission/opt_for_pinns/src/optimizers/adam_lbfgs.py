"""Adam + L-BFGS switching optimizer wrapper.

This module implements the "Adam + L-BFGS" baseline used throughout the paper
(see Section 4 / Table 1).  The idea is simple: run Adam for a fixed number of
iterations, then hand the parameters over to L-BFGS (with a strong-Wolfe line
search) for the remainder of the budget.

The paper uses a total budget of 41,000 iterations and switches from Adam to
L-BFGS at one of {1k, 11k, 31k} iterations.  The default switch point used for
the main results is 11,000 iterations.

In addition to the plain training loop we record the L-BFGS curvature pairs
``{s_k, y_k, rho_k}`` so that the *preconditioned* spectral density analysis of
Appendix C.2 (Figures 3 & 7) can be reproduced.  The pairs are stored in an
:class:`~src.hessian.lbfgs_unroll.LBFGSHistory` object.

Public interface
----------------
- :class:`AdamLBFGS` -- stateful optimizer wrapper.
- :func:`train_adam_lbfgs` -- convenience functional training routine.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..hessian.lbfgs_unroll import LBFGSHistory
from ..model import flatten_parameters, set_flat_parameters

__all__ = ["AdamLBFGS", "train_adam_lbfgs"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flat_grad_from_model(model: nn.Module) -> torch.Tensor:
    """Return the concatenated ``.grad`` buffers of ``model`` as a flat vector."""
    grads: List[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    return torch.cat(grads)


class _CurvatureRecorder:
    """Wraps a PyTorch ``LBFGS`` optimizer to record curvature pairs.

    PyTorch's L-BFGS implementation does not expose the ``(s_k, y_k)`` pairs it
    computes internally, so we reconstruct them from consecutive parameter /
    gradient snapshots.  The reconstruction is exact for the pairs that are
    actually used by the two-loop recursion (up to the memory truncation).
    """

    def __init__(self, memory: int = 100) -> None:
        self.history = LBFGSHistory(memory=memory)
        self._prev_params: Optional[torch.Tensor] = None
        self._prev_grad: Optional[torch.Tensor] = None

    def observe(self, model: nn.Module) -> None:
        """Record a new ``(s, y)`` pair from the current model state."""
        params = flatten_parameters(model).detach().clone()
        grad = _flat_grad_from_model(model).detach().clone()

        if self._prev_params is not None and self._prev_grad is not None:
            s = params - self._prev_params
            y = grad - self._prev_grad
            sy = torch.dot(s, y)
            if sy > 1e-12:
                self.history.add(s, y, rho=1.0 / sy.item())

        self._prev_params = params
        self._prev_grad = grad


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------
class AdamLBFGS:
    """Adam followed by L-BFGS.

    Parameters
    ----------
    model:
        The PINN model whose parameters are optimised in place.
    loss_fn:
        Zero-argument closure returning the scalar loss (reads live params).
    adam_lr:
        Learning rate for the Adam phase.
    switch_iter:
        Number of Adam iterations before switching to L-BFGS.
    total_iters:
        Total optimisation budget (Adam + L-BFGS iterations).
    lbfgs_memory:
        L-BFGS history size (paper uses 100).
    lbfgs_lr:
        Learning rate passed to PyTorch's L-BFGS (paper uses 1.0).
    max_iter_per_call:
        Maximum number of L-BFGS iterations per ``step`` call.
    record_curvature:
        Whether to record L-BFGS curvature pairs for spectral analysis.
    verbose:
        Print progress every ``log_every`` iterations.
    log_every:
        Logging frequency.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        adam_lr: float = 1e-3,
        switch_iter: int = 11000,
        total_iters: int = 41000,
        lbfgs_memory: int = 100,
        lbfgs_lr: float = 1.0,
        max_iter_per_call: int = 1,
        record_curvature: bool = False,
        verbose: bool = False,
        log_every: int = 1000,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.adam_lr = float(adam_lr)
        self.switch_iter = int(switch_iter)
        self.total_iters = int(total_iters)
        self.lbfgs_memory = int(lbfgs_memory)
        self.lbfgs_lr = float(lbfgs_lr)
        self.max_iter_per_call = int(max_iter_per_call)
        self.record_curvature = bool(record_curvature)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)

        self.adam = torch.optim.Adam(model.parameters(), lr=self.adam_lr)
        self.lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=self.lbfgs_lr,
            max_iter=self.max_iter_per_call,
            max_eval=max(5, 5 * self.max_iter_per_call),
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=self.lbfgs_memory,
            line_search_fn="strong_wolfe",
        )

        self.recorder = _CurvatureRecorder(memory=self.lbfgs_memory) if record_curvature else None

        self.iter = 0
        self.history: Dict[str, List[float]] = {
            "iter": [],
            "loss": [],
            "grad_norm": [],
            "phase": [],  # 0 = Adam, 1 = L-BFGS
        }

    # -- internals ---------------------------------------------------------
    def _closure(self) -> torch.Tensor:
        self.lbfgs.zero_grad()
        loss = self.loss_fn()
        loss.backward()
        return loss

    def _log(self, loss: float, phase: int) -> None:
        grad_norm = float(_flat_grad_from_model(self.model).norm().item())
        self.history["iter"].append(self.iter)
        self.history["loss"].append(float(loss))
        self.history["grad_norm"].append(grad_norm)
        self.history["phase"].append(phase)
        if self.verbose and (self.iter % self.log_every == 0 or self.iter == self.total_iters):
            print(
                f"[AdamLBFGS] iter={self.iter:6d} phase={'adam' if phase == 0 else 'lbfgs'} "
                f"loss={loss:.6e} grad_norm={grad_norm:.6e}"
            )

    # -- public API --------------------------------------------------------
    def step(self) -> float:
        """Perform a single optimisation iteration. Returns the current loss."""
        if self.iter >= self.total_iters:
            return float(self.history["loss"][-1]) if self.history["loss"] else float("nan")

        if self.iter < self.switch_iter:
            # ---- Adam phase ----
            self.adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.adam.step()
            loss_val = float(loss.detach().item())
            self._log(loss_val, phase=0)
        else:
            # ---- L-BFGS phase ----
            if self.recorder is not None:
                # snapshot the state *before* the L-BFGS step so that the
                # curvature pair (s, y) can be reconstructed afterwards.
                self.recorder.observe(self.model)
            loss = self.lbfgs.step(self._closure)
            loss_val = float(loss.detach().item())
            self._log(loss_val, phase=1)

        self.iter += 1
        return loss_val

    def minimize(self) -> Dict[str, List[float]]:
        """Run the full optimisation budget. Returns the history dict."""
        while self.iter < self.total_iters:
            self.step()
        return self.history

    # convenience alias
    run = minimize


# ---------------------------------------------------------------------------
# Functional wrapper
# ---------------------------------------------------------------------------
def train_adam_lbfgs(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    adam_lr: float = 1e-3,
    switch_iter: int = 11000,
    total_iters: int = 41000,
    lbfgs_memory: int = 100,
    lbfgs_lr: float = 1.0,
    record_curvature: bool = False,
    verbose: bool = False,
    log_every: int = 1000,
) -> Tuple[nn.Module, Dict[str, List[float]], Optional[LBFGSHistory]]:
    """Train ``model`` with Adam then L-BFGS.

    Returns
    -------
    (model, history, lbfgs_history)
        ``lbfgs_history`` is ``None`` unless ``record_curvature`` is True.
    """
    opt = AdamLBFGS(
        model=model,
        loss_fn=loss_fn,
        adam_lr=adam_lr,
        switch_iter=switch_iter,
        total_iters=total_iters,
        lbfgs_memory=lbfgs_memory,
        lbfgs_lr=lbfgs_lr,
        record_curvature=record_curvature,
        verbose=verbose,
        log_every=log_every,
    )
    history = opt.minimize()
    lbfgs_history = opt.recorder.history if opt.recorder is not None else None
    return model, history, lbfgs_history


# ---------------------------------------------------------------------------
# Plain Adam / plain L-BFGS baselines (used for Table 1)
# ---------------------------------------------------------------------------
def train_adam(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    lr: float = 1e-3,
    total_iters: int = 41000,
    verbose: bool = False,
    log_every: int = 1000,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Plain Adam baseline."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"iter": [], "loss": [], "grad_norm": []}
    for it in range(total_iters):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        loss_val = float(loss.detach().item())
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["grad_norm"].append(float(_flat_grad_from_model(model).norm().item()))
        if verbose and (it % log_every == 0 or it == total_iters - 1):
            print(f"[Adam] iter={it:6d} loss={loss_val:.6e}")
    return model, history


def train_lbfgs(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    lr: float = 1.0,
    total_iters: int = 41000,
    memory: int = 100,
    max_iter_per_call: int = 1,
    verbose: bool = False,
    log_every: int = 1000,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Plain L-BFGS baseline (strong-Wolfe line search)."""
    opt = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter_per_call,
        max_eval=max(5, 5 * max_iter_per_call),
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=memory,
        line_search_fn="strong_wolfe",
    )
    history: Dict[str, List[float]] = {"iter": [], "loss": [], "grad_norm": []}

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    for it in range(total_iters):
        loss = opt.step(closure)
        loss_val = float(loss.detach().item())
        history["iter"].append(it)
        history["loss"].append(loss_val)
        history["grad_norm"].append(float(_flat_grad_from_model(model).norm().item()))
        if verbose and (it % log_every == 0 or it == total_iters - 1):
            print(f"[LBFGS] iter={it:6d} loss={loss_val:.6e}")
    return model, history
