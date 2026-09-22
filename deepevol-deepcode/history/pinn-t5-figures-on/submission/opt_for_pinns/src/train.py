"""Training loops for PINNs.

Implements the baseline optimizers studied in the paper (Section 2.2, 6.1):

  * Adam
  * L-BFGS (strong Wolfe line search, memory 100)
  * Adam + L-BFGS (switch at a fixed iteration)

and the fine-tuning procedures used in Section 7.3:

  * NNCG (NysNewton-CG) for K steps after Adam+L-BFGS
  * Gradient descent for K steps after Adam+L-BFGS

All training is full-batch: every step evaluates the loss on all residual,
IC and BC points.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .data import PINNData
from .loss import evaluate, pinn_loss
from .pdes import PDE


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """Container for the outcome of a training run."""

    loss: float = float("nan")
    l2re: float = float("nan")
    grad_norm: float = float("nan")
    residual: float = float("nan")
    ic: float = float("nan")
    bc: float = float("nan")
    n_iters: int = 0
    wallclock: float = 0.0
    history: List[Dict[str, float]] = field(default_factory=list)
    # per-iteration wall clock (seconds) for the last phase run
    per_iter_time: float = float("nan")

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss": self.loss,
            "l2re": self.l2re,
            "grad_norm": self.grad_norm,
            "residual": self.residual,
            "ic": self.ic,
            "bc": self.bc,
            "n_iters": self.n_iters,
            "wallclock": self.wallclock,
            "per_iter_time": self.per_iter_time,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _closure_factory(pde: PDE, model, data: PINNData) -> Callable[[], torch.Tensor]:
    """Build a zero-arg closure returning the total PINN loss (for L-BFGS)."""

    def closure() -> torch.Tensor:
        loss = pinn_loss(
            pde,
            model,
            data.x_res,
            data.t_res,
            data.x_ic,
            data.t_ic,
            data.x_bc,
            data.t_bc,
        )
        return loss

    return closure


def _record(pde: PDE, model, data: PINNData) -> Dict[str, float]:
    """Evaluate the model and return the metric dict."""
    return evaluate(
        pde,
        model,
        data.x_res,
        data.t_res,
        data.x_ic,
        data.t_ic,
        data.x_bc,
        data.t_bc,
        data.x_eval,
        data.t_eval,
        data.y_eval,
    )


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------
def train_adam(
    pde: PDE,
    model,
    data: PINNData,
    lr: float = 1e-3,
    n_iters: int = 41000,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
    log_every: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Train with Adam for ``n_iters`` full-batch steps."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    result = TrainResult()
    history: List[Dict[str, float]] = []

    t0 = time.time()
    for it in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        loss = pinn_loss(
            pde,
            model,
            data.x_res,
            data.t_res,
            data.x_ic,
            data.t_ic,
            data.x_bc,
            data.t_bc,
        )
        loss.backward()
        optimizer.step()

        if log_every and (it % log_every == 0 or it == n_iters - 1):
            metrics = _record(pde, model, data)
            metrics["iter"] = it
            history.append(metrics)
            if verbose:
                print(
                    f"[Adam] it={it:6d} loss={metrics['loss']:.3e} "
                    f"l2re={metrics['l2re']:.3e}"
                )

    result.wallclock = time.time() - t0
    result.n_iters = n_iters
    result.per_iter_time = result.wallclock / max(1, n_iters)
    result.history = history

    final = _record(pde, model, data)
    for k, v in final.items():
        setattr(result, k, v)
    return result


# ---------------------------------------------------------------------------
# L-BFGS
# ---------------------------------------------------------------------------
def train_lbfgs(
    pde: PDE,
    model,
    data: PINNData,
    lr: float = 1.0,
    n_iters: int = 41000,
    history_size: int = 100,
    line_search_fn: str = "strong_wolfe",
    log_every: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Train with L-BFGS (strong Wolfe line search) for ``n_iters`` steps."""
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=1,
        max_eval=1,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )
    closure = _closure_factory(pde, model, data)
    result = TrainResult()
    history: List[Dict[str, float]] = []

    t0 = time.time()
    for it in range(n_iters):
        optimizer.step(closure)

        if log_every and (it % log_every == 0 or it == n_iters - 1):
            metrics = _record(pde, model, data)
            metrics["iter"] = it
            history.append(metrics)
            if verbose:
                print(
                    f"[L-BFGS] it={it:6d} loss={metrics['loss']:.3e} "
                    f"l2re={metrics['l2re']:.3e}"
                )

    result.wallclock = time.time() - t0
    result.n_iters = n_iters
    result.per_iter_time = result.wallclock / max(1, n_iters)
    result.history = history

    final = _record(pde, model, data)
    for k, v in final.items():
        setattr(result, k, v)
    return result


# ---------------------------------------------------------------------------
# Adam + L-BFGS
# ---------------------------------------------------------------------------
def train_adam_lbfgs(
    pde: PDE,
    model,
    data: PINNData,
    adam_lr: float = 1e-3,
    switch_iter: int = 11000,
    total_iters: int = 41000,
    lbfgs_lr: float = 1.0,
    history_size: int = 100,
    log_every: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Adam for ``switch_iter`` steps, then L-BFGS for the remainder."""
    result = TrainResult()
    history: List[Dict[str, float]] = []

    t0 = time.time()

    # --- Phase 1: Adam ---
    adam = torch.optim.Adam(model.parameters(), lr=adam_lr)
    for it in range(switch_iter):
        adam.zero_grad(set_to_none=True)
        loss = pinn_loss(
            pde,
            model,
            data.x_res,
            data.t_res,
            data.x_ic,
            data.t_ic,
            data.x_bc,
            data.t_bc,
        )
        loss.backward()
        adam.step()

        if log_every and (it % log_every == 0 or it == switch_iter - 1):
            metrics = _record(pde, model, data)
            metrics["iter"] = it
            metrics["phase"] = "adam"
            history.append(metrics)
            if verbose:
                print(
                    f"[A+L/Adam] it={it:6d} loss={metrics['loss']:.3e} "
                    f"l2re={metrics['l2re']:.3e}"
                )

    # --- Phase 2: L-BFGS ---
    lbfgs_iters = max(0, total_iters - switch_iter)
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=1,
        max_eval=1,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    closure = _closure_factory(pde, model, data)
    for it in range(lbfgs_iters):
        lbfgs.step(closure)

        if log_every and (it % log_every == 0 or it == lbfgs_iters - 1):
            metrics = _record(pde, model, data)
            metrics["iter"] = switch_iter + it
            metrics["phase"] = "lbfgs"
            history.append(metrics)
            if verbose:
                print(
                    f"[A+L/LBFGS] it={switch_iter + it:6d} "
                    f"loss={metrics['loss']:.3e} l2re={metrics['l2re']:.3e}"
                )

    result.wallclock = time.time() - t0
    result.n_iters = total_iters
    result.per_iter_time = result.wallclock / max(1, total_iters)
    result.history = history

    final = _record(pde, model, data)
    for k, v in final.items():
        setattr(result, k, v)
    return result


# ---------------------------------------------------------------------------
# Fine-tuning: NNCG / GD after Adam+L-BFGS
# ---------------------------------------------------------------------------
def finetune_nncg(
    pde: PDE,
    model,
    data: PINNData,
    n_steps: int = 2000,
    s: int = 60,
    F: int = 20,
    mu: float = 1e-2,
    eps: float = 1e-16,
    M: int = 1000,
    eta: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    log_every: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Run NNCG (Algorithm 4) for ``n_steps`` steps starting from the current model."""
    from .optimizers.nncg import NNCG

    result = TrainResult()
    history: List[Dict[str, float]] = []

    def loss_fn() -> torch.Tensor:
        return pinn_loss(
            pde,
            model,
            data.x_res,
            data.t_res,
            data.x_ic,
            data.t_ic,
            data.x_bc,
            data.t_bc,
        )

    optimizer = NNCG(
        model.parameters(),
        loss_fn=loss_fn,
        s=s,
        F=F,
        mu=mu,
        eps=eps,
        M=M,
        eta=eta,
        alpha=alpha,
        beta=beta,
    )

    t0 = time.time()
    for it in range(n_steps):
        optimizer.step()

        if log_every and (it % log_every == 0 or it == n_steps - 1):
            metrics = _record(pde, model, data)
            metrics["iter"] = it
            history.append(metrics)
            if verbose:
                print(
                    f"[NNCG] it={it:6d} loss={metrics['loss']:.3e} "
                    f"l2re={metrics['l2re']:.3e}"
                )

    result.wallclock = time.time() - t0
    result.n_iters = n_steps
    result.per_iter_time = result.wallclock / max(1, n_steps)
    result.history = history

    final = _record(pde, model, data)
    for k, v in final.items():
        setattr(result, k, v)
    return result


def finetune_gd(
    pde: PDE,
    model,
    data: PINNData,
    n_steps: int = 2000,
    lr: float = 1e-3,
    log_every: int = 0,
    verbose: bool = False,
) -> TrainResult:
    """Run plain gradient descent for ``n_steps`` steps (Section 7.3 baseline)."""
    result = TrainResult()
    history: List[Dict[str, float]] = []

    t0 = time.time()
    for it in range(n_steps):
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        loss = pinn_loss(
            pde,
            model,
            data.x_res,
            data.t_res,
            data.x_ic,
            data.t_ic,
            data.x_bc,
            data.t_bc,
        )
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-lr)

        if log_every and (it % log_every == 0 or it == n_steps - 1):
            metrics = _record(pde, model, data)
            metrics["iter"] = it
            history.append(metrics)
            if verbose:
                print(
                    f"[GD] it={it:6d} loss={metrics['loss']:.3e} "
                    f"l2re={metrics['l2re']:.3e}"
                )

    result.wallclock = time.time() - t0
    result.n_iters = n_steps
    result.per_iter_time = result.wallclock / max(1, n_steps)
    result.history = history

    final = _record(pde, model, data)
    for k, v in final.items():
        setattr(result, k, v)
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def train(
    pde: PDE,
    model,
    data: PINNData,
    optimizer_name: str = "adam_lbfgs",
    **kwargs,
) -> TrainResult:
    """Dispatch to the requested training routine.

    ``optimizer_name`` in {"adam", "lbfgs", "adam_lbfgs", "nncg", "gd"}.
    """
    name = optimizer_name.lower()
    if name == "adam":
        return train_adam(pde, model, data, **kwargs)
    if name in ("lbfgs", "l-bfgs"):
        return train_lbfgs(pde, model, data, **kwargs)
    if name in ("adam_lbfgs", "adam+lbfgs", "adam-lbfgs"):
        return train_adam_lbfgs(pde, model, data, **kwargs)
    if name in ("nncg", "nncg_finetune"):
        return finetune_nncg(pde, model, data, **kwargs)
    if name in ("gd", "gradient_descent"):
        return finetune_gd(pde, model, data, **kwargs)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")
