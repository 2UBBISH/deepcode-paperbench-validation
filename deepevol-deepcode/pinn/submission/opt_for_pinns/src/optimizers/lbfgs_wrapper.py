"""L-BFGS wrapper for PINN training (``opt_for_pinns``).

This module implements the L-BFGS baseline of Section 2.2 of *Challenges in
Training PINNs: A Loss Landscape Perspective* (ICML 2024, PMLR 235):

    "For L-BFGS, we use the default learning rate 1.0, memory size 100, and
     strong Wolfe line search. ... All three methods are run for a total of
     41000 iterations."

Besides being a drop-in replacement for :class:`AdamOptimizer` (same
``step(closure)`` / ``zero_grad()`` / ``state_dict()`` surface), the wrapper
**records the L-BFGS history buffers** ``(s_k, y_k, rho_k)`` -- and the curvature
scalar ``gamma_k = s_k^T y_k / y_k^T y_k`` -- for every accepted iteration.
Those buffers are the input of the *unrolled* L-BFGS formulation of the paper's
Appendix C.2:

* Algorithm 2 -- ``Unroll-LBFGSSteps`` (see :mod:`src.spectral.lbfgs_unroll`),
* Algorithm 3 -- ``LBFGSHessianMVP``   (see :mod:`src.spectral.preconditioned_mvp`),

which build ``Ytilde_k``, ``Vtilde_k``, ``Stilde_k`` such that the
preconditioned Hessian ``Htilde_k^T H_L(w) Htilde_k`` can be probed with
matrix--vector products only.

Public interface
----------------
* :class:`LBFGSHistory`        -- recorded ``(s, y, rho, gamma)`` buffers.
* :class:`LBFGSOptimizer`      -- ``torch.optim.LBFGS`` wrapper with recording.
* :func:`run_lbfgs`            -- small standalone training driver.
* :func:`lbfgs_recording_state`-- fetch the buffers of a trained model.

Notes
-----
* ``history_size=m=100``, ``lr=1.0`` and ``line_search_fn="strong_wolfe"`` are
  the paper's defaults (Section 2.2).
* Each call to :meth:`LBFGSOptimizer.step` performs **one** L-BFGS iteration
  (``max_iter=1``, ``max_eval`` bounded) so that "iterations" are comparable
  with Adam steps when the total budget of 41000 iterations is split.
* Gradients must be supplied by a ``closure`` that returns an autograd-tracked
  scalar loss (as produced by ``src.pinns.loss.make_loss_fn``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


__all__ = [
    "LBFGS_DEFAULTS",
    "LBFGSHistory",
    "LBFGSOptimizer",
    "run_lbfgs",
    "lbfgs_recording_state",
]


# --------------------------------------------------------------------------------------
# Paper defaults (Section 2.2)
# --------------------------------------------------------------------------------------

#: L-BFGS hyper-parameters used throughout the paper (Section 2.2).
LBFGS_DEFAULTS: Dict[str, Any] = {
    # "default learning rate 1.0"
    "lr": 1.0,
    # "memory size 100"
    "history_size": 100,
    # "strong Wolfe line search"
    "line_search_fn": "strong_wolfe",
    # exactly one L-BFGS iteration per ``step`` call (41000 total iterations)
    "max_iter": 1,
    "max_eval": 25,
    "tolerance_grad": 1e-7,
    "tolerance_change": 1e-9,
}


# --------------------------------------------------------------------------------------
# Flattening helpers (private)
# --------------------------------------------------------------------------------------


def _parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return the trainable parameters of ``model`` as a list."""
    if isinstance(model, (list, tuple)):
        params: List[nn.Parameter] = []
        for m in model:
            params.extend([p for p in m.parameters() if p.requires_grad])
        return params
    return [p for p in model.parameters() if p.requires_grad]


def _flatten(tensors: Sequence[Optional[Tensor]], like: Sequence[Tensor]) -> Tensor:
    """Flatten ``tensors`` into a single 1-D vector (missing entries -> zeros)."""
    parts: List[Tensor] = []
    for t, ref in zip(tensors, like):
        if t is None:
            parts.append(torch.zeros_like(ref).reshape(-1))
        else:
            parts.append(t.reshape(-1))
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _flat_params(params: Sequence[nn.Parameter]) -> Tensor:
    """Flatten the current parameter values into a single detached 1-D vector."""
    with torch.no_grad():
        return _flatten([p.detach() for p in params], params).detach().clone()


def _flat_grads(params: Sequence[nn.Parameter]) -> Tensor:
    """Flatten the current gradients into a single detached 1-D vector."""
    with torch.no_grad():
        return _flatten([p.grad for p in params], params).detach().clone()


# --------------------------------------------------------------------------------------
# History recording
# --------------------------------------------------------------------------------------


@dataclass
class LBFGSHistory:
    """Recorded L-BFGS history buffers for the unrolled formulation (Appendix C.2).

    Attributes
    ----------
    s : list of Tensor
        Curvature pairs ``s_k = w_{k+1} - w_k`` (flattened, length ``n_params``).
    y : list of Tensor
        Gradient differences ``y_k = g_{k+1} - g_k``.
    rho : list of float
        ``rho_k = 1 / (y_k^T s_k)`` (``0.0`` if the curvature is non-positive).
    gamma : list of float
        Diagonal scaling ``gamma_k = (s_k^T y_k) / (y_k^T y_k)``.
    losses : list of float
        Loss value observed at every *accepted* iteration.
    n_params : int
        Number of trainable parameters (dimension of ``s_k`` / ``y_k``).
    """

    s: List[Tensor] = field(default_factory=list)
    y: List[Tensor] = field(default_factory=list)
    rho: List[float] = field(default_factory=list)
    gamma: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    n_params: int = 0

    # -- basic accessors -----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.s)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return len(self.s) > 0

    @property
    def last_gamma(self) -> float:
        """Most recent curvature scalar ``gamma_k`` (1.0 when empty)."""
        return float(self.gamma[-1]) if self.gamma else 1.0

    def append(self, s: Tensor, y: Tensor, rho: float, gamma: float, loss: Optional[float] = None) -> None:
        """Record one curvature pair together with its derived scalars."""
        self.s.append(s)
        self.y.append(y)
        self.rho.append(float(rho))
        self.gamma.append(float(gamma))
        if loss is not None:
            self.losses.append(float(loss))

    # -- truncation / memory -------------------------------------------------------
    def truncate(self, memory: int) -> "LBFGSHistory":
        """Keep only the most recent ``memory`` curvature pairs (in place)."""
        if memory is None or memory <= 0 or len(self.s) <= memory:
            return self
        self.s = self.s[-memory:]
        self.y = self.y[-memory:]
        self.rho = self.rho[-memory:]
        self.gamma = self.gamma[-memory:]
        return self

    # -- conversions ---------------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """Plain-python/dict description (buffers kept as tensors)."""
        return {
            "s": list(self.s),
            "y": list(self.y),
            "rho": list(self.rho),
            "gamma": list(self.gamma),
            "losses": list(self.losses),
            "n_params": self.n_params,
        }

    def stacked(self, memory: Optional[int] = None) -> Dict[str, Tensor]:
        """Stack the *last* ``memory`` pairs into dense tensors.

        Returns a dict with keys ``"S"``, ``"Y"`` ``(m, n_params)``, ``"rho"``,
        ``"gamma"`` ``(m,)``.  Empty history yields zero-row tensors.
        """
        s_list = self.s if memory is None else self.s[-memory:]
        y_list = self.y if memory is None else self.y[-memory:]
        rho_list = self.rho if memory is None else self.rho[-memory:]
        gamma_list = self.gamma if memory is None else self.gamma[-memory:]
        n = self.n_params
        if len(s_list) == 0:
            return {
                "S": torch.zeros(0, n),
                "Y": torch.zeros(0, n),
                "rho": torch.zeros(0),
                "gamma": torch.zeros(0),
            }
        return {
            "S": torch.stack([t.reshape(-1) for t in s_list], dim=0),
            "Y": torch.stack([t.reshape(-1) for t in y_list], dim=0),
            "rho": torch.tensor(rho_list, dtype=torch.float64),
            "gamma": torch.tensor(gamma_list, dtype=torch.float64),
        }


# Module-level side table: id(model) -> LBFGSHistory
_RECORDED_HISTORY: Dict[int, LBFGSHistory] = {}


def lbfgs_recording_state(model: nn.Module) -> Optional[LBFGSHistory]:
    """Return the :class:`LBFGSHistory` recorded for ``model`` (or ``None``).

    The spectral-density pipeline of §5 uses this to recompute the unrolled
    L-BFGS preconditioners without re-running the optimizer.
    """
    return _RECORDED_HISTORY.get(id(model))


def clear_recording_state(model: Optional[nn.Module] = None) -> None:
    """Drop recorded buffers for ``model`` (or for every model)."""
    if model is None:
        _RECORDED_HISTORY.clear()
    else:
        _RECORDED_HISTORY.pop(id(model), None)


# --------------------------------------------------------------------------------------
# Optimizer
# --------------------------------------------------------------------------------------


class LBFGSOptimizer:
    """``torch.optim.LBFGS`` wrapper with paper defaults and buffer recording.

    Parameters
    ----------
    model : nn.Module
        PINN whose ``parameters()`` are optimized.
    lr : float
        Learning rate (paper: ``1.0``).
    history_size : int
        L-BFGS memory ``m`` (paper: ``100``).
    line_search_fn : str or None
        ``"strong_wolfe"`` (paper default) or ``None``.
    max_iter : int
        Inner L-BFGS iterations per :meth:`step` call (default ``1`` so that one
        ``step`` == one of the paper's 41000 iterations).
    record : bool
        If ``True`` (default), record ``(s_k, y_k, rho_k, gamma_k)``.
    record_limit : int or None
        Keep at most this many curvature pairs (defaults to ``history_size`` to
        mimic the optimizer's own memory); ``None`` keeps everything.
    dtype : torch.dtype
        dtype used for the recorded buffers (default ``float64`` for stability).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = LBFGS_DEFAULTS["lr"],
        history_size: int = LBFGS_DEFAULTS["history_size"],
        line_search_fn: Optional[str] = LBFGS_DEFAULTS["line_search_fn"],
        max_iter: int = LBFGS_DEFAULTS["max_iter"],
        max_eval: Optional[int] = LBFGS_DEFAULTS["max_eval"],
        tolerance_grad: float = LBFGS_DEFAULTS["tolerance_grad"],
        tolerance_change: float = LBFGS_DEFAULTS["tolerance_change"],
        record: bool = True,
        record_limit: Optional[int] = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.model = model
        self.params: List[nn.Parameter] = _parameters(model)
        self.n_params = int(sum(p.numel() for p in self.params))
        self.lr = float(lr)
        self.history_size = int(history_size)
        self.line_search_fn = line_search_fn
        self.max_iter = int(max_iter)
        self.max_eval = max_eval
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.record = bool(record)
        self.record_limit = self.history_size if record_limit is None else int(record_limit)
        self.dtype = dtype

        self.optimizer = torch.optim.LBFGS(
            self.params,
            lr=self.lr,
            max_iter=self.max_iter,
            max_eval=self.max_eval,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            history_size=self.history_size,
            line_search_fn=self.line_search_fn,
        )

        self.history: LBFGSHistory = LBFGSHistory(n_params=self.n_params)
        _RECORDED_HISTORY[id(model)] = self.history

        # State used to build (s_k, y_k) across consecutive *accepted* steps.
        self._prev_w: Optional[Tensor] = None
        self._prev_g: Optional[Tensor] = None
        self._last_w: Optional[Tensor] = None
        self._last_g: Optional[Tensor] = None

        self.last_loss: Optional[float] = None
        self.n_steps: int = 0
        self.n_skipped: int = 0

    # -- bookkeeping ---------------------------------------------------------------
    def _buffer_dtype(self, ref: Tensor) -> torch.dtype:
        return self.dtype if self.dtype is not None else ref.dtype

    def _record_pair(self, w_old: Tensor, g_old: Tensor, w_new: Tensor, g_new: Tensor) -> None:
        """Record ``(s, y, rho, gamma)`` for one *accepted* L-BFGS iteration."""
        s = (w_new - w_old).to(self._buffer_dtype(w_new))
        y = (g_new - g_old).to(self._buffer_dtype(g_new))
        denom = float(torch.dot(y, s).item())
        rho = 1.0 / denom if denom > 1e-300 else 0.0
        yy = float(torch.dot(y, y).item())
        gamma = (denom / yy) if yy > 1e-300 else 1.0
        self.history.append(s, y, rho, gamma, loss=self.last_loss)
        if self.record_limit and self.record_limit > 0:
            self.history.truncate(self.record_limit)

    # -- optimizer protocol --------------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero the gradients of the optimized parameters."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Serialisable optimizer state (including the recorded buffers)."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr": self.lr,
            "history_size": self.history_size,
            "line_search_fn": self.line_search_fn,
            "max_iter": self.max_iter,
            "n_steps": self.n_steps,
            "n_params": self.n_params,
            "history": self.history.as_dict(),
        }

    @property
    def grad_norm(self) -> Optional[float]:
        """L2 norm of the flattened gradient at the last closure evaluation."""
        if self._last_g is None:
            return None
        return float(torch.linalg.vector_norm(self._last_g).item())

    def step(self, closure: Callable[[], Tensor]) -> Optional[float]:
        """Perform one L-BFGS iteration (strong-Wolfe line search).

        ``closure`` must return an autograd-tracked scalar loss; it is evaluated
        once outside PyTorch's line search to capture the state used for the
        ``(s_k, y_k)`` bookkeeping, and then handed to ``torch.optim.LBFGS``.

        Returns the loss value produced by the inner optimizer (or the captured
        loss, as a float).
        """
        if self.params and all(p.grad is None for p in self.params):
            self.zero_grad(set_to_none=True)

        # -- pre-step state (used for the recorded curvature pair) -----------------
        loss_before = closure()
        if not torch.is_tensor(loss_before):
            loss_before = torch.as_tensor(loss_before, dtype=torch.float32)
        w_before = _flat_params(self.params)
        g_before = _flat_grads(self.params)
        self.last_loss = float(loss_before.detach().item())

        # -- run one L-BFGS iteration ---------------------------------------------
        if self.max_iter > 1:
            # let ``torch.optim.LBFGS`` perform the inner loop itself
            def _closure() -> Tensor:
                self.zero_grad(set_to_none=True)
                loss = closure()
                if not torch.is_tensor(loss):
                    loss = torch.as_tensor(loss)
                return loss

            out = self.optimizer.step(_closure)
            loss_value = float(out.detach().item()) if torch.is_tensor(out) else self.last_loss
        else:
            # a single iteration: reuse the already-computed forward/backward pass
            def _closure() -> Tensor:
                return loss_before

            out = self.optimizer.step(_closure)
            loss_value = float(out.detach().item()) if torch.is_tensor(out) else self.last_loss

        # -- post-step state -------------------------------------------------------
        w_after = _flat_params(self.params)
        g_after = _flat_grads(self.params)
        self._last_w, self._last_g = w_after, g_after

        moved = float(torch.linalg.vector_norm(w_after - w_before).item())
        if moved > 0.0 and g_after.abs().sum().item() > 0.0:
            self._record_pair(w_before, g_before, w_after, g_after)
        else:
            # line search rejected the step -> no new curvature pair
            self.n_skipped += 1

        self.n_steps += 1
        self._prev_w, self._prev_g = w_before, g_before
        return loss_value

    # -- convenience ---------------------------------------------------------------
    @property
    def gamma(self) -> float:
        """Last recorded curvature scalar ``gamma_k`` (1.0 if unavailable)."""
        return self.history.last_gamma


# --------------------------------------------------------------------------------------
# Standalone driver
# --------------------------------------------------------------------------------------


def run_lbfgs(
    model: nn.Module,
    closure: Callable[[], Tensor],
    n_steps: int = 41000,
    eval_fn: Optional[Callable[[], float]] = None,
    eval_every: int = 500,
    log_every: Optional[int] = None,
    history: Optional[Any] = None,
    callback: Optional[Callable[[int, Any], None]] = None,
    record: bool = True,
    optimizer: Optional[LBFGSOptimizer] = None,
    **optimizer_kwargs: Any,
) -> Any:
    """Train ``model`` with L-BFGS for ``n_steps`` iterations.

    Parameters
    ----------
    model : nn.Module
        PINN to optimize.
    closure : callable
        Returns an autograd-tracked scalar loss (see ``src.pinns.loss.make_loss_fn``).
    n_steps : int
        Number of L-BFGS iterations (paper: 41000).
    eval_fn : callable, optional
        Zero-argument callable returning the L2RE at the current parameters.
    eval_every : int
        Cadence (in iterations) at which ``eval_fn`` is called.
    log_every : int, optional
        Print the loss every ``log_every`` iterations.
    history : TrainingHistory, optional
        ``src.optimizers.first_order.TrainingHistory`` instance to fill in.  A
        compatible object is created locally when ``None`` and returned.
    callback : callable, optional
        ``callback(step, training_history)`` hook called at every step.
    record : bool
        Whether the returned :class:`LBFGSOptimizer` should record the buffers.
    optimizer : LBFGSOptimizer, optional
        Pre-built optimizer (e.g. to continue a run); otherwise one is created
        from ``**optimizer_kwargs``.

    Returns
    -------
    The populated training-history object; the recording state can be retrieved
    via :func:`lbfgs_recording_state` on the model (or from ``optimizer.history``).
    """
    opt = optimizer if optimizer is not None else LBFGSOptimizer(model, record=record, **optimizer_kwargs)

    if history is None:
        # local import to avoid a circular dependency at module import time
        try:
            from .first_order import TrainingHistory  # type: ignore

            history = TrainingHistory()
        except Exception:  # pragma: no cover - standalone fallback
            history = _FallbackHistory()

    for step in range(1, int(n_steps) + 1):
        loss_value = opt.step(closure)

        do_eval = (eval_fn is not None) and (step % eval_every == 0 or step == n_steps)
        l2re_value = float(eval_fn()) if do_eval else None
        grad_norm = None
        if step == n_steps or (log_every is not None and step % log_every == 0):
            grad_norm = opt.grad_norm

        history.append(step=step, loss=loss_value, l2re_value=l2re_value, grad_norm=grad_norm)

        if log_every is not None and step % log_every == 0:
            msg = f"[lbfgs] step {step:>6d} | loss {loss_value:.6e}"
            if l2re_value is not None:
                msg += f" | L2RE {l2re_value:.6e}"
            if grad_norm is not None:
                msg += f" | ||g|| {grad_norm:.6e}"
            print(msg, flush=True)

        if callback is not None:
            callback(step, history)

    return history


class _FallbackHistory:
    """Minimal stand-in for :class:`TrainingHistory` (used if the import fails)."""

    def __init__(self) -> None:
        self.steps: List[int] = []
        self.losses: List[float] = []
        self.l2re: List[float] = []
        self.grad_norms: List[float] = []
        self.extra: Dict[str, Any] = {}

    def append(self, step=None, loss=None, l2re_value=None, grad_norm=None, **kwargs: Any) -> None:
        if step is not None:
            self.steps.append(int(step))
        if loss is not None:
            self.losses.append(float(loss))
        if l2re_value is not None:
            self.l2re.append(float(l2re_value))
        if grad_norm is not None:
            self.grad_norms.append(float(grad_norm))

    @property
    def best_loss(self) -> Optional[float]:
        return min(self.losses) if self.losses else None

    @property
    def final_loss(self) -> Optional[float]:
        return self.losses[-1] if self.losses else None

    @property
    def best_l2re(self) -> Optional[float]:
        return min(self.l2re) if self.l2re else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "losses": self.losses,
            "l2re": self.l2re,
            "grad_norms": self.grad_norms,
        }


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 3.0])

    model = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, 0.5, 0.5]]))

    def closure() -> Tensor:
        pred = model(target.reshape(1, 3)).squeeze()
        return (pred - 4.0) ** 2

    opt = LBFGSOptimizer(model)
    hist = run_lbfgs(model, closure, n_steps=25, eval_fn=lambda: 1.0, eval_every=10, log_every=5)
    print("final loss:", hist.final_loss)
    print("recorded pairs:", len(opt.history), "| gamma:", opt.gamma)
    print("rho[0]:", opt.history.rho[0] if opt.history.rho else None)
