"""First-order optimizers used for the PINN experiments.

Implements:

* :class:`AdamOptimizer` -- a thin wrapper around :class:`torch.optim.Adam`
  with the learning-rate grid ``{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}`` used for the
  grid search described in Section 2.2 of "Challenges in Training PINNs: A Loss
  Landscape Perspective" (ICML 2024).
* :func:`adam_lr_grid` / :func:`sweep_adam_lr` -- helpers implementing the grid
  search ("For Adam, we tune the learning rate by a grid search on
  {1e-5, ..., 1e-1}").
* :class:`GradientDescent` -- plain gradient descent with a small fixed learning
  rate, used as the *control* method in the NNCG fine-tuning experiments
  (Figure 4: "GD fails").
* :func:`run_first_order` -- a generic driver that runs a first-order optimizer
  for a fixed number of steps while recording loss / L2RE trajectories.

All optimizers operate directly on the raw ``loss`` closure (which must return a
PyTorch scalar tensor that keeps the autograd graph, as produced by
``src.pinns.loss.make_loss_fn``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Learning-rate grid (Section 2.2)
# ---------------------------------------------------------------------------

#: The Adam learning-rate grid searched over in the paper (Section 2.2).
ADAM_LR_GRID: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def adam_lr_grid(lrs: Optional[Iterable[float]] = None) -> List[float]:
    """Return the Adam learning-rate grid (Section 2.2)."""
    if lrs is None:
        return list(ADAM_LR_GRID)
    return [float(lr) for lr in lrs]


# ---------------------------------------------------------------------------
# Parameter flattening helpers
# ---------------------------------------------------------------------------

def _parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _flatten_grads(model: nn.Module) -> torch.Tensor:
    """Concatenate ``.grad`` for all trainable parameters into one vector."""
    grads = []
    for p in _parameters(model):
        if p.grad is None:
            grads.append(torch.zeros_like(p).reshape(-1))
        else:
            grads.append(p.grad.detach().reshape(-1))
    if not grads:
        return torch.zeros(0)
    return torch.cat(grads)


def _set_grads_from_flat(model: nn.Module, flat: torch.Tensor) -> None:
    """Scatter a flat gradient vector back into ``param.grad``."""
    offset = 0
    for p in _parameters(model):
        n = p.numel()
        chunk = flat[offset:offset + n].reshape(p.shape)
        p.grad = chunk.clone()
        offset += n


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in _parameters(model)])


@torch.no_grad()
def _set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in _parameters(model):
        n = p.numel()
        p.copy_(flat[offset:offset + n].reshape(p.shape))
        offset += n


# ---------------------------------------------------------------------------
# Abstract first-order base
# ---------------------------------------------------------------------------

class FirstOrderOptimizer:
    """Base class shared by the first-order optimizers.

    Subclasses must implement :meth:`step` and :attr:`lr`.
    """

    name: str = "first-order"

    def __init__(self, model: nn.Module):
        self.model = model

    # -- interface ---------------------------------------------------------
    def step(self, closure: Callable[[], torch.Tensor]) -> float:  # pragma: no cover
        raise NotImplementedError

    def zero_grad(self) -> None:
        for p in _parameters(self.model):
            p.grad = None

    @property
    def lr(self) -> float:  # pragma: no cover
        raise NotImplementedError

    @lr.setter
    def lr(self, value: float) -> None:  # pragma: no cover
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def state_dict(self) -> Dict:  # pragma: no cover - overridden
        return {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(lr={self.lr})"


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class AdamOptimizer(FirstOrderOptimizer):
    """Adam wrapper with the paper's defaults.

    Parameters
    ----------
    model:
        The PINN whose parameters are optimized.
    lr:
        Learning rate (tuned via the grid ``{1e-5, 1e-4, 1e-3, 1e-2, 1e-1}``).
    betas:
        Standard Adam momentum parameters ``(0.9, 0.999)``.
    eps:
        Numerical stability term.
    weight_decay:
        L2 penalty (0.0 in the paper).
    grad_clip:
        Optional global-norm gradient clipping (disabled by default, ``None``).
    """

    name = "adam"

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = None,
    ):
        super().__init__(model)
        self._lr = float(lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self._optimizer = torch.optim.Adam(
            _parameters(model),
            lr=self._lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.last_loss: Optional[float] = None
        self.n_steps = 0

    # -- properties --------------------------------------------------------
    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        self._lr = float(value)
        for group in self._optimizer.param_groups:
            group["lr"] = self._lr

    # -- interface ---------------------------------------------------------
    def zero_grad(self) -> None:
        self._optimizer.zero_grad(set_to_none=True)

    def step(self, closure: Callable[[], torch.Tensor]) -> float:
        """Perform one Adam step.

        ``closure`` must return the scalar loss tensor with ``requires_grad``
        set (the graph is used for the backward pass).
        """
        self.zero_grad()
        loss = closure()
        if loss.dim() != 0:
            loss = loss.mean()
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(_parameters(self.model), self.grad_clip)
        self._optimizer.step()
        self.n_steps += 1
        self.last_loss = float(loss.detach())
        return self.last_loss

    def state_dict(self) -> Dict:
        return {
            "lr": self._lr,
            "n_steps": self.n_steps,
            "last_loss": self.last_loss,
            "torch": self._optimizer.state_dict(),
        }


# ---------------------------------------------------------------------------
# Gradient descent (control method for the NNCG experiments)
# ---------------------------------------------------------------------------

class GradientDescent(FirstOrderOptimizer):
    """Plain gradient descent with a (small) fixed learning rate.

    Used as the *control* optimizer in the NNCG fine-tuning study: starting from
    the same Adam+L-BFGS checkpoint, we run the same number of extra steps of GD
    and of NNCG.  GD is not required to make progress in this regime; "GD fails"
    is the expected outcome (Figure 4).
    """

    name = "gd"

    def __init__(self, model: nn.Module, lr: float = 1e-4):
        super().__init__(model)
        self._lr = float(lr)
        self.n_steps = 0
        self.last_loss: Optional[float] = None
        self.last_grad_norm: Optional[float] = None

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        self._lr = float(value)

    def zero_grad(self) -> None:
        for p in _parameters(self.model):
            p.grad = None

    @torch.no_grad()
    def _update(self, flat_grad: torch.Tensor) -> None:
        flat = _flat_params(self.model) - self._lr * flat_grad
        _set_flat_params(self.model, flat)

    def step(self, closure: Callable[[], torch.Tensor]) -> float:
        self.zero_grad()
        loss = closure()
        if loss.dim() != 0:
            loss = loss.mean()
        loss.backward()
        flat_grad = _flatten_grads(self.model)
        self.last_grad_norm = float(flat_grad.norm())
        self._update(flat_grad)
        self.n_steps += 1
        self.last_loss = float(loss.detach())
        return self.last_loss

    def state_dict(self) -> Dict:
        return {
            "lr": self._lr,
            "n_steps": self.n_steps,
            "last_loss": self.last_loss,
            "last_grad_norm": self.last_grad_norm,
        }


# ---------------------------------------------------------------------------
# Generic first-order training driver
# ---------------------------------------------------------------------------

@dataclass
class TrainingHistory:
    """History recorded while running an optimizer."""

    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    l2re: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    extra: Dict[str, List[float]] = field(default_factory=dict)

    def append(self, step: int, loss: float, l2re_value: Optional[float] = None,
               grad_norm: Optional[float] = None) -> None:
        self.steps.append(int(step))
        self.losses.append(float(loss))
        if l2re_value is not None:
            self.l2re.append(float(l2re_value))
        if grad_norm is not None:
            self.grad_norms.append(float(grad_norm))

    @property
    def best_loss(self) -> float:
        return min(self.losses) if self.losses else float("nan")

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def best_l2re(self) -> float:
        return min(self.l2re) if self.l2re else float("nan")

    def to_dict(self) -> Dict[str, List[float]]:
        out = {"steps": self.steps, "losses": self.losses}
        if self.l2re:
            out["l2re"] = self.l2re
        if self.grad_norms:
            out["grad_norms"] = self.grad_norms
        out.update(self.extra)
        return out


def run_first_order(
    model: nn.Module,
    closure: Callable[[], torch.Tensor],
    optimizer: FirstOrderOptimizer,
    n_steps: int,
    eval_fn: Optional[Callable[[], float]] = None,
    eval_every: int = 500,
    log_every: Optional[int] = None,
    history: Optional[TrainingHistory] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> TrainingHistory:
    """Run ``optimizer`` for ``n_steps`` iterations.

    Parameters
    ----------
    model:
        PINN module (used only for reporting).
    closure:
        Zero-argument callable returning the scalar training loss tensor.
    optimizer:
        Any :class:`FirstOrderOptimizer`.
    n_steps:
        Number of optimizer steps (e.g. 41000 for the full comparison runs).
    eval_fn:
        Optional callable returning the current L2RE (evaluation only).
    eval_every:
        Evaluate ``eval_fn`` every this many steps (and always at the end).
    log_every:
        If given, print the loss at this cadence.
    history:
        Optional pre-existing history to append to.
    callback:
        Optional ``callback(step, loss)`` hook called after every step.
    """
    hist = history if history is not None else TrainingHistory()
    if n_steps <= 0:
        return hist

    for step in range(1, n_steps + 1):
        loss_value = optimizer.step(closure)

        if callback is not None:
            callback(step, loss_value)

        needs_eval = (eval_fn is not None) and (step % eval_every == 0 or step == n_steps)
        l2re_value = None
        if needs_eval:
            l2re_value = float(eval_fn())

        grad_norm = None
        flat = _flatten_grads(model)
        if flat.numel() > 0:
            grad_norm = float(flat.norm())

        if log_every is not None and (step % log_every == 0 or step == n_steps):
            msg = f"[{optimizer.name}] step {step}: loss={loss_value:.6e}"
            if l2re_value is not None:
                msg += f", L2RE={l2re_value:.6e}"
            print(msg, flush=True)

        # Record at the evaluation cadence (keeps the history compact).
        if step % eval_every == 0 or step == n_steps:
            hist.append(step, loss_value, l2re_value, grad_norm)

    return hist


# ---------------------------------------------------------------------------
# Adam learning-rate grid search
# ---------------------------------------------------------------------------

@dataclass
class AdamSweepResult:
    """Result of an Adam learning-rate grid search."""

    lrs: Tuple[float, ...]
    losses: Tuple[float, ...]
    best_lr: float
    best_loss: float
    histories: Dict[float, TrainingHistory] = field(default_factory=dict)
    l2re: Tuple[float, ...] = ()
    best_l2re: float = float("nan")

    def as_dict(self) -> Dict:
        return {
            "lrs": list(self.lrs),
            "losses": list(self.losses),
            "l2re": list(self.l2re),
            "best_lr": self.best_lr,
            "best_loss": self.best_loss,
            "best_l2re": self.best_l2re,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AdamSweepResult(best_lr={self.best_lr:g}, best_loss={self.best_loss:.4e}, "
            f"best_l2re={self.best_l2re:.4e})"
        )


def sweep_adam_lr(
    make_setup: Callable[[float], Tuple[nn.Module, Callable[[], torch.Tensor]]],
    n_steps: int,
    lrs: Optional[Sequence[float]] = None,
    eval_fn_factory: Optional[Callable[[nn.Module], Callable[[], float]]] = None,
    eval_every: int = 500,
    select_by: str = "loss",
    verbose: bool = False,
) -> AdamSweepResult:
    """Grid-search the Adam learning rate.

    ``make_setup(lr)`` must return ``(model, closure)`` freshly constructed for
    that learning rate (so that each grid point starts from an identical
    initialization).  ``select_by`` is ``"loss"`` or ``"l2re"`` and controls
    which metric decides the best learning rate -- the paper selects by the
    lowest loss for the optimizer comparison (Section 6.1) and by the smallest
    L2RE for the spectral/NNCG experiments (Sections 4 and 5).
    """
    grid = adam_lr_grid(lrs)
    losses: List[float] = []
    l2res: List[float] = []
    histories: Dict[float, TrainingHistory] = {}

    for lr in grid:
        model, closure = make_setup(lr)
        eval_fn = eval_fn_factory(model) if eval_fn_factory is not None else None
        optimizer = AdamOptimizer(model, lr=lr)
        hist = run_first_order(
            model, closure, optimizer, n_steps,
            eval_fn=eval_fn, eval_every=eval_every,
            log_every=None,
        )
        histories[lr] = hist
        losses.append(hist.best_loss)
        l2res.append(hist.best_l2re)
        if verbose:
            print(
                f"  Adam lr={lr:g}: best loss={hist.best_loss:.6e}, "
                f"best L2RE={hist.best_l2re:.6e}",
                flush=True,
            )

    if select_by == "l2re" and any(math.isfinite(v) for v in l2res):
        best_idx = int(min(range(len(l2res)), key=lambda i: l2res[i]))
    else:
        best_idx = int(min(range(len(losses)), key=lambda i: losses[i]))

    return AdamSweepResult(
        lrs=tuple(grid),
        losses=tuple(losses),
        best_lr=grid[best_idx],
        best_loss=losses[best_idx],
        histories=histories,
        l2re=tuple(l2res),
        best_l2re=l2res[best_idx],
    )


__all__ = [
    "ADAM_LR_GRID",
    "adam_lr_grid",
    "FirstOrderOptimizer",
    "AdamOptimizer",
    "GradientDescent",
    "TrainingHistory",
    "run_first_order",
    "AdamSweepResult",
    "sweep_adam_lr",
]
