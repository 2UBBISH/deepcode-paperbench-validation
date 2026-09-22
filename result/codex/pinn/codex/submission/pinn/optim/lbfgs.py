"""L-BFGS exactly as configured in Section 2.2 of the paper.

"For L-BFGS, we use the default learning rate 1.0, memory size 100, and strong
Wolfe line search."

One call to :meth:`step` performs a *single* L-BFGS iteration (``max_iter=1``),
so that iteration counts are comparable with Adam's gradient steps and so that
a per-iteration trace can be recorded (Figure 4).  The correction pairs
``(s_k, y_k)`` are kept in ``torch.optim.LBFGS.state`` and are exposed through
:meth:`history`, which is what the preconditioned spectral density computation
needs (Appendix C.2).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from ..hessian.lbfgs_precond import LBFGSHistory
from .objective import Objective


class LBFGSOptimizer:
    def __init__(
        self,
        objective: Objective,
        lr: float = 1.0,
        history_size: int = 100,
        max_iter: int = 1,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search_fn: str = "strong_wolfe",
    ):
        self.objective = objective
        self.opt = torch.optim.LBFGS(
            objective.params,
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval if max_eval is not None else max(25, int(max_iter * 1.25)),
            history_size=history_size,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            line_search_fn=line_search_fn,
        )
        self.n_step_calls = 0

    def closure(self):
        self.opt.zero_grad(set_to_none=True)
        loss = self.objective.loss()
        loss.backward()
        return loss

    def step(self) -> float:
        """One L-BFGS iteration; returns the loss after the update."""
        self.opt.step(self.closure)
        self.n_step_calls += 1
        return self.objective.evaluate()

    # ------------------------------------------------------------------ #
    def history(self) -> LBFGSHistory:
        """The stored correction pairs (newest first) and ``gamma_k``."""
        return LBFGSHistory.from_torch_state(self.opt.state[self.objective.params[0]])

    @property
    def state(self) -> dict:
        return self.opt.state[self.objective.params[0]]

    def last_step_size(self) -> float:
        """Step size ``t`` chosen by the most recent strong-Wolfe line search."""
        return float(self.state.get("t", 0.0))

    def loss_evals(self) -> int:
        return int(self.state.get("func_evals", 0))
