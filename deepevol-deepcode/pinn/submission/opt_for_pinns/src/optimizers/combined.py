"""Adam + L-BFGS combined optimizer (paper Section 2.2 / 6.1).

The paper trains PINNs with Adam for a warm-up phase and then switches to
L-BFGS for the remainder of the iteration budget.  The switches studied in
Section 6.1 are at 1k / 11k / 31k iterations, always with a total budget of
41,000 iterations.

This module provides

* :class:`CombinedOptimizer` -- an optimizer-like object that runs Adam for the
  first ``switch_iteration`` steps and L-BFGS afterwards.  Both phases share the
  same loss closure.
* :func:`run_adam_lbfgs` -- a standalone driver returning the training history
  **and** the L-BFGS curvature history ``(s_k, y_k, rho_k, gamma_k)`` recorded
  after the switch.  Those buffers are the input of the unrolled L-BFGS
  preconditioner used by the spectral-density experiments (Appendix C.2).
* :func:`sweep_switch_points` -- convenience helper that runs the three switch
  points from the paper.

The loss closure contract follows ``src/pinns/loss.py`` /
``src/optimizers/first_order.py``: the closure takes no arguments and returns an
autograd-tracked scalar loss tensor.  Both optimizers perform the backward pass
internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .first_order import AdamOptimizer, TrainingHistory
from .lbfgs_wrapper import LBFGS_DEFAULTS, LBFGSHistory, LBFGSOptimizer

__all__ = [
    "SWITCH_POINTS",
    "DEFAULT_SWITCH_POINT",
    "COMBINED_TOTAL_ITERATIONS",
    "CombinedOptimizer",
    "CombinedResult",
    "run_adam_lbfgs",
    "run_combined",
    "sweep_switch_points",
]


# --------------------------------------------------------------------------- #
# Constants from the paper (Section 6.1)
# --------------------------------------------------------------------------- #
#: Switch iterations evaluated in Section 6.1 of the paper.
SWITCH_POINTS: Tuple[int, ...] = (1000, 11000, 31000)

#: Switch point used by the spectral-density / NNCG experiments (Section 5, 7).
DEFAULT_SWITCH_POINT: int = 11000

#: Total number of optimizer iterations used everywhere in the paper.
COMBINED_TOTAL_ITERATIONS: int = 41000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion of a loss tensor to a python float."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _grad_norm(model: nn.Module) -> float:
    """L2 norm of the flattened gradient currently stored on ``model``."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().double().pow(2).sum().item())
    return math.sqrt(total)


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).double() for p in model.parameters()])


def parameter_distance(model: nn.Module, reference: torch.Tensor) -> float:
    """Euclidean distance between current parameters and ``reference``."""
    return float((_flat_params(model) - reference.double().reshape(-1)).norm().item())


# --------------------------------------------------------------------------- #
# Combined optimizer
# --------------------------------------------------------------------------- #
class CombinedOptimizer:
    """Adam for ``switch_iteration`` steps, then L-BFGS.

    Parameters
    ----------
    model:
        The PINN module whose parameters are optimised.
    adam_lr:
        Adam learning rate (paper grid ``{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}``).
    switch_iteration:
        Number of Adam iterations before switching to L-BFGS
        (``1k``, ``11k`` or ``31k`` in Section 6.1).
    total_iterations:
        Total iteration budget (41,000 in the paper).
    record:
        Record the L-BFGS curvature pairs after the switch (needed by the
        spectral-density pipeline).
    """

    def __init__(
        self,
        model: nn.Module,
        adam_lr: float = 1e-3,
        switch_iteration: int = DEFAULT_SWITCH_POINT,
        total_iterations: int = COMBINED_TOTAL_ITERATIONS,
        *,
        adam_kwargs: Optional[Dict[str, Any]] = None,
        lbfgs_kwargs: Optional[Dict[str, Any]] = None,
        record: bool = True,
        record_limit: Optional[int] = None,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.model = model
        self.adam_lr = float(adam_lr)
        self.switch_iteration = int(switch_iteration)
        self.total_iterations = int(total_iterations)
        self.grad_clip = grad_clip
        self.record = bool(record)

        adam_kwargs = dict(adam_kwargs or {})
        adam_kwargs.setdefault("lr", self.adam_lr)
        adam_kwargs.setdefault("grad_clip", grad_clip)
        # ``grad_clip`` is a wrapper-level argument; do not duplicate it.
        adam_kwargs.pop("grad_clip", None)
        self._adam_kwargs = adam_kwargs

        lbfgs_kwargs = dict(lbfgs_kwargs or {})
        for key, value in LBFGS_DEFAULTS.items():
            lbfgs_kwargs.setdefault(key, value)
        lbfgs_kwargs.setdefault("record", record)
        lbfgs_kwargs.setdefault("record_limit", record_limit)
        self._lbfgs_kwargs = lbfgs_kwargs

        self.adam = AdamOptimizer(model, grad_clip=grad_clip, **self._adam_kwargs)
        self.lbfgs = LBFGSOptimizer(model, **self._lbfgs_kwargs)

        self.iteration = 0
        self.n_adam_steps = 0
        self.n_lbfgs_steps = 0
        self.last_loss: Optional[float] = None
        self.switched = False
        self._params_at_switch: Optional[torch.Tensor] = None

    # -- introspection ----------------------------------------------------- #
    @property
    def phase(self) -> str:
        """``"adam"`` while warming up, ``"lbfgs"`` afterwards."""
        return "adam" if self.iteration < self.switch_iteration else "lbfgs"

    @property
    def history(self) -> LBFGSHistory:
        """Recorded L-BFGS curvature buffers (possibly empty)."""
        return self.lbfgs.history

    @property
    def lr(self) -> float:
        return self.adam_lr

    @lr.setter
    def lr(self, value: float) -> None:
        self.adam_lr = float(value)
        self.adam.lr = float(value)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adam.zero_grad(set_to_none=set_to_none)
        self.lbfgs.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "n_adam_steps": self.n_adam_steps,
            "n_lbfgs_steps": self.n_lbfgs_steps,
            "adam_lr": self.adam_lr,
            "switch_iteration": self.switch_iteration,
            "phase": self.phase,
        }

    # -- switching --------------------------------------------------------- #
    def _maybe_switch(self) -> None:
        """Mark parameters at the switch point and reset stale gradients."""
        if not self.switched and self.iteration >= self.switch_iteration:
            self.switched = True
            self._params_at_switch = _flat_params(self.model)
            # A switch with stale Adam momentum / grads is pathological; drop them.
            self.adam.zero_grad(set_to_none=True)
            self.lbfgs.zero_grad(set_to_none=True)

    @property
    def parameters_at_switch(self) -> Optional[torch.Tensor]:
        return self._params_at_switch

    # -- optimizer interface ---------------------------------------------- #
    def step(self, closure: Callable[[], torch.Tensor]) -> float:
        """Perform one optimizer iteration.

        ``closure`` must return an autograd-tracked scalar loss.
        """
        self._maybe_switch()
        if self.phase == "adam":
            loss = self.adam.step(closure)
            self.n_adam_steps += 1
        else:
            loss = self.lbfgs.step(closure)
            self.n_lbfgs_steps += 1
        value = _to_float(loss)
        if value is not None:
            self.last_loss = value
        self.iteration += 1
        return self.last_loss if self.last_loss is not None else float("nan")


# Alias kept for readability in downstream code.
AdamLBFGS = CombinedOptimizer


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
@dataclass
class CombinedResult:
    """Container returned by :func:`run_adam_lbfgs`."""

    history: Any
    lbfgs_history: LBFGSHistory
    switch_iteration: int
    total_iterations: int
    adam_lr: float
    switch_step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_loss(self) -> float:
        return getattr(self.history, "best_loss", float("nan"))

    @property
    def final_loss(self) -> float:
        return getattr(self.history, "final_loss", float("nan"))

    @property
    def best_l2re(self) -> Optional[float]:
        return getattr(self.history, "best_l2re", None)

    def as_dict(self) -> Dict[str, Any]:
        out = {
            "adam_lr": self.adam_lr,
            "switch_iteration": self.switch_iteration,
            "total_iterations": self.total_iterations,
            "best_loss": self.best_loss,
            "final_loss": self.final_loss,
            "best_l2re": self.best_l2re,
            "n_history_pairs": len(self.lbfgs_history),
        }
        out.update(self.extra)
        return out


def run_adam_lbfgs(
    model: nn.Module,
    closure: Callable[[], torch.Tensor],
    *,
    adam_lr: float = 1e-3,
    switch_iteration: int = DEFAULT_SWITCH_POINT,
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    eval_fn: Optional[Callable[[nn.Module], float]] = None,
    eval_every: int = 500,
    log_every: Optional[int] = None,
    record: bool = True,
    record_limit: Optional[int] = None,
    grad_clip: Optional[float] = None,
    adam_kwargs: Optional[Dict[str, Any]] = None,
    lbfgs_kwargs: Optional[Dict[str, Any]] = None,
    history: Optional[TrainingHistory] = None,
    callback: Optional[Callable[[int, "CombinedOptimizer", float], None]] = None,
    verbose: bool = False,
) -> CombinedResult:
    """Train ``model`` with Adam for ``switch_iteration`` steps, then L-BFGS.

    The total iteration count (Adam + L-BFGS) equals ``total_iterations``
    (41,000 in the paper).  Loss / L2RE / gradient-norm histories are recorded
    every ``eval_every`` iterations and at the final iteration.
    """
    opt = CombinedOptimizer(
        model,
        adam_lr=adam_lr,
        switch_iteration=switch_iteration,
        total_iterations=total_iterations,
        adam_kwargs=adam_kwargs,
        lbfgs_kwargs=lbfgs_kwargs,
        record=record,
        record_limit=record_limit,
        grad_clip=grad_clip,
    )
    hist = history if history is not None else TrainingHistory()

    last_loss = float("nan")
    switch_step = 0

    for it in range(int(total_iterations)):
        was_adam = opt.phase == "adam"
        loss = opt.step(closure)
        if was_adam and opt.phase == "lbfgs":
            switch_step = it + 1

        if loss is not None and not math.isnan(loss):
            last_loss = loss

        done = it + 1
        is_eval = bool(eval_every) and (done % int(eval_every) == 0)
        if done == int(total_iterations):
            is_eval = True
        if is_eval:
            gnorm = _grad_norm(model)
            l2 = None
            if eval_fn is not None:
                l2 = _to_float(eval_fn(model))
            try:
                hist.append(done, last_loss, l2, gnorm)
            except TypeError:  # very defensive: positional-only signature
                hist.append(done, last_loss, l2)  # type: ignore[misc]

        if log_every and done % int(log_every) == 0 and verbose:
            print(
                f"[combined-{adam_lr:g}] it={done} phase={opt.phase} loss={last_loss:.6e}",
                flush=True,
            )
        if callback is not None:
            callback(done, opt, last_loss)

    return CombinedResult(
        history=hist,
        lbfgs_history=opt.lbfgs.history,
        switch_iteration=int(switch_iteration),
        total_iterations=int(total_iterations),
        adam_lr=float(adam_lr),
        switch_step=switch_step,
        extra={
            "n_adam_steps": opt.n_adam_steps,
            "n_lbfgs_steps": opt.n_lbfgs_steps,
            "lbfgs_skipped": getattr(opt.lbfgs, "n_skipped", None),
        },
    )


# Friendly alias.
run_combined = run_adam_lbfgs


def sweep_switch_points(
    make_setup: Callable[[int], Tuple[nn.Module, Callable[[], torch.Tensor]]],
    switch_points: Sequence[int] = SWITCH_POINTS,
    *,
    total_iterations: int = COMBINED_TOTAL_ITERATIONS,
    eval_fn_factory: Optional[Callable[[nn.Module], Callable[[nn.Module], float]]] = None,
    eval_every: int = 500,
    verbose: bool = False,
    **kwargs: Any,
) -> List[CombinedResult]:
    """Run Adam+L-BFGS for each switch point of Section 6.1.

    ``make_setup(switch)`` must build a *freshly initialised* ``(model, closure)``
    pair so that the switch points are compared from identical initialisations.
    """
    results: List[CombinedResult] = []
    for switch in switch_points:
        model, closure = make_setup(int(switch))
        eval_fn = eval_fn_factory(model) if eval_fn_factory is not None else None
        results.append(
            run_adam_lbfgs(
                model,
                closure,
                switch_iteration=int(switch),
                total_iterations=int(total_iterations),
                eval_fn=eval_fn,
                eval_every=eval_every,
                verbose=verbose,
                **kwargs,
            )
        )
    return results


if __name__ == "__main__":  # pragma: no cover - tiny smoke test
    torch.manual_seed(0)
    lin = nn.Linear(2, 1, dtype=torch.float64)
    m = nn.Sequential(lin)

    def closure() -> torch.Tensor:
        x = torch.randn(64, 2, dtype=torch.float64)
        return (m(x) ** 2).mean()

    res = run_adam_lbfgs(m, closure, adam_lr=1e-2, switch_iteration=20,
                         total_iterations=40, eval_every=10, verbose=True)
    print(res.as_dict())
