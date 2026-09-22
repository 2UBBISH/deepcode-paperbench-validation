"""NysNewton-CG (NNCG) optimizer -- Algorithm 4 of "Challenges in Training PINNs:
A Loss Landscape Perspective" (ICML 2024), Appendix E.2.

The paper describes NNCG as a *damped* Newton method preconditioned with
NyströmPCG, with an Armijo backtracking line search guaranteeing loss decrease::

    Algorithm 4  NysNewton-CG (NNCG)
    input  w0, max learning rate eta, #iterations K, sketch size s,
           preconditioner update frequency F, damping mu, CG tolerance eps,
           CG max iterations M, backtracking parameters alpha, beta
        d_{-1} = 0
        for k = 0, ..., K-1 do
            if k mod F == 0 then
                [U, Lambda_hat] = RandomizedNystromApproximation(H_L(w_k), s)
            end if
            d_k = NystromPCG(H_L(w_k), grad L(w_k), d_{k-1}, U, Lambda_hat, s, mu, eps, M)
            eta_k = Armijo(L, w_k, grad L(w_k), -d_k, eta)
            w_{k+1} = w_k - eta_k d_k
        end for

Experiments (§7.2, Appendix E.2): ``eta = 1``, ``K = 2000``, ``s = 60``,
``F = 20``, ``epsilon = 1e-16``, ``M = 1000``, ``alpha = 0.1``, ``beta = 0.5``
and ``mu`` tuned over ``[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]`` (``mu = 1e-2`` and
``1e-1`` work best in practice).

The implementation operates on *flat* parameter vectors (float64), accesses the
Hessian only through Hessian-vector products (``src/spectral/hvp.py``) and reuses
``randomized_nystrom_approximation`` / ``nystrom_pcg`` (``nystrom.py``) and
``ArmijoLineSearch`` (``armijo.py``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# imports with both relative and absolute fall-backs
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - import shim
    from .armijo import ArmijoLineSearch
except ImportError:  # pragma: no cover
    from src.optimizers.armijo import ArmijoLineSearch  # type: ignore

try:  # pragma: no cover - import shim
    from .nystrom import (
        NystromPCGInfo,
        NystromPreconditioner,
        RandomizedNystromInfo,
        nystrom_pcg,
        randomized_nystrom_approximation,
    )
except ImportError:  # pragma: no cover
    from src.optimizers.nystrom import (  # type: ignore
        NystromPCGInfo,
        NystromPreconditioner,
        RandomizedNystromInfo,
        nystrom_pcg,
        randomized_nystrom_approximation,
    )

try:  # pragma: no cover - import shim
    from ..spectral.hvp import (
        DEFAULT_DTYPE,
        HessianOperator,
        flatten_params as _flatten_params,
        hvp as _hvp_apply,
        loss_and_grad as _loss_and_grad,
        set_flat_params as _set_flat_params,
    )
except ImportError:  # pragma: no cover
    from src.spectral.hvp import (  # type: ignore
        DEFAULT_DTYPE,
        HessianOperator,
        flatten_params as _flatten_params,
        hvp as _hvp_apply,
        loss_and_grad as _loss_and_grad,
        set_flat_params as _set_flat_params,
    )

__all__ = [
    "NNCGConfig",
    "NNCGHistory",
    "NNCGResult",
    "MUTuningResult",
    "NNCGOptimizer",
    "NNCG",
    "NysNewtonCG",
    "run_nncg",
    "tune_mu",
    "NNCG_MU_GRID",
    "NNCG_DEFAULTS",
]

# Paper defaults (Appendix E.2).
NNCG_MU_GRID: Tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
NNCG_DEFAULTS: Dict[str, Any] = dict(
    eta=1.0,
    K=2000,
    s=60,
    F=20,
    mu=1e-2,
    epsilon=1e-16,
    M=1000,
    alpha=0.1,
    beta=0.5,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_flat(v: Any) -> torch.Tensor:
    """Coerce a matvec output (tensor / list of tensors) into a flat vector."""
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)
    if isinstance(v, (list, tuple)):
        return torch.cat([x.reshape(-1) for x in v])
    raise TypeError(f"cannot interpret {type(v)} as a flat vector")


def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return DEFAULT_DTYPE
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).replace("torch.", "")
    return getattr(torch, name)


def _mu_grid(mus: Optional[Sequence[float]] = None) -> List[float]:
    if mus is None:
        return list(NNCG_MU_GRID)
    out: List[float] = []
    for mu in mus:
        if isinstance(mu, (list, tuple)) and len(mu) == 2:  # (lo, hi) range
            lo, hi, n = float(mu[0]), float(mu[1]), 6
            out.extend([lo * (hi / lo) ** (i / (n - 1)) for i in range(n)])
        else:
            out.append(float(mu))
    return out


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass
class NNCGConfig:
    """Hyper-parameters of Algorithm 4 (defaults = paper Appendix E.2)."""

    eta: float = 1.0
    K: int = 2000
    s: int = 60
    F: int = 20
    mu: float = 1e-2
    epsilon: float = 1e-16
    M: int = 1000
    alpha: float = 0.1
    beta: float = 0.5
    dtype: Union[str, torch.dtype] = "float64"
    max_backtracks: int = 100
    seed: Optional[int] = None
    cast_dtype: bool = True
    fallback_to_gradient: bool = True
    label: str = ""

    def __post_init__(self) -> None:
        self.eta = float(self.eta)
        self.K = int(self.K)
        self.s = int(self.s)
        self.F = max(1, int(self.F))
        self.mu = float(self.mu)
        self.epsilon = float(self.epsilon)
        self.M = int(self.M)
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None) -> "NNCGConfig":
        """Build from a (possibly nested) config dict, e.g. ``cfg["nncg"]``."""
        if cfg is None:
            return cls()
        data: Dict[str, Any] = {}
        if isinstance(cfg, dict):
            for key in ("nncg", "nncg_finetune", "NNCG"):
                sub = cfg.get(key)
                if isinstance(sub, dict):
                    data.update(sub)
            for key in NNCG_DEFAULTS:
                if key in cfg:
                    data[key] = cfg[key]
            for key in (
                "dtype",
                "max_backtracks",
                "seed",
                "cast_dtype",
                "fallback_to_gradient",
                "label",
            ):
                if key in cfg:
                    data[key] = cfg[key]
            # allow mus list to define the default mu when present
            if "mu" not in data and isinstance(cfg.get("mus"), (list, tuple)) and cfg["mus"]:
                data["mu"] = float(cfg["mus"][-2] if len(cfg["mus"]) > 1 else cfg["mus"][0])
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in valid})

    @property
    def torch_dtype(self) -> torch.dtype:
        return _resolve_dtype(self.dtype)


# --------------------------------------------------------------------------- #
# history / results
# --------------------------------------------------------------------------- #
@dataclass
class NNCGHistory:
    """Per-iteration diagnostic history of an NNCG run."""

    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    step_sizes: List[float] = field(default_factory=list)
    pcg_iterations: List[int] = field(default_factory=list)
    refreshes: List[bool] = field(default_factory=list)
    armijo_backtracks: List[int] = field(default_factory=list)
    fallbacks: List[bool] = field(default_factory=list)
    iteration_times: List[float] = field(default_factory=list)
    l2re: Dict[int, float] = field(default_factory=dict)
    extra: Dict[str, List[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def append(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        step_size: Optional[float] = None,
        pcg_iterations: Optional[int] = None,
        refreshed: bool = False,
        armijo_backtracks: Optional[int] = None,
        fell_back: bool = False,
        elapsed: Optional[float] = None,
        l2re_value: Optional[float] = None,
    ) -> None:
        self.steps.append(int(step))
        self.losses.append(float(loss))
        self.grad_norms.append(float("nan") if grad_norm is None else float(grad_norm))
        self.step_sizes.append(float("nan") if step_size is None else float(step_size))
        self.pcg_iterations.append(-1 if pcg_iterations is None else int(pcg_iterations))
        self.refreshes.append(bool(refreshed))
        self.armijo_backtracks.append(
            -1 if armijo_backtracks is None else int(armijo_backtracks)
        )
        self.fallbacks.append(bool(fell_back))
        self.iteration_times.append(float("nan") if elapsed is None else float(elapsed))
        if l2re_value is not None:
            self.l2re[int(step)] = float(l2re_value)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.losses)

    @property
    def n_steps(self) -> int:
        return len(self.losses)

    @property
    def best_loss(self) -> float:
        return min(self.losses) if self.losses else float("inf")

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    @property
    def best_l2re(self) -> float:
        return min(self.l2re.values()) if self.l2re else float("nan")

    @property
    def final_l2re(self) -> float:
        if not self.l2re:
            return float("nan")
        return self.l2re[max(self.l2re.keys())]

    @property
    def total_time(self) -> float:
        vals = [t for t in self.iteration_times if not math.isnan(t)]
        return float(sum(vals))

    @property
    def time_per_iteration(self) -> float:
        vals = [t for t in self.iteration_times if not math.isnan(t)]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    @property
    def mean_pcg_iterations(self) -> float:
        vals = [i for i in self.pcg_iterations if i >= 0]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    @property
    def n_refreshes(self) -> int:
        return int(sum(1 for r in self.refreshes if r))

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            steps=list(self.steps),
            losses=list(self.losses),
            grad_norms=list(self.grad_norms),
            step_sizes=list(self.step_sizes),
            pcg_iterations=list(self.pcg_iterations),
            refreshes=list(self.refreshes),
            armijo_backtracks=list(self.armijo_backtracks),
            fallbacks=list(self.fallbacks),
            iteration_times=list(self.iteration_times),
            l2re=dict(self.l2re),
            best_loss=self.best_loss,
            final_loss=self.final_loss,
            best_l2re=self.best_l2re,
            mean_pcg_iterations=self.mean_pcg_iterations,
            time_per_iteration=self.time_per_iteration,
            total_time=self.total_time,
            n_refreshes=self.n_refreshes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.as_dict()


@dataclass
class NNCGResult:
    """Container returned by :func:`run_nncg`."""

    history: NNCGHistory
    config: NNCGConfig
    mu: float
    n_steps: int
    model: Optional[nn.Module] = None
    optimizer: Optional["NNCGOptimizer"] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_loss(self) -> float:
        return self.history.best_loss

    @property
    def final_loss(self) -> float:
        return self.history.final_loss

    @property
    def best_l2re(self) -> float:
        return self.history.best_l2re

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            mu=self.mu,
            n_steps=self.n_steps,
            best_loss=self.best_loss,
            final_loss=self.final_loss,
            best_l2re=self.best_l2re,
            config={k: getattr(self.config, k) for k in NNCG_DEFAULTS},
            history=self.history.as_dict(),
            **self.extra,
        )


@dataclass
class MUTuningResult:
    """Outcome of the ``mu`` grid search prescribed in Appendix E.2."""

    mus: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    l2re: List[float] = field(default_factory=list)
    results: List[NNCGResult] = field(default_factory=list)
    best_mu: float = float("nan")
    best_loss: float = float("inf")
    best_l2re: float = float("nan")
    select_by: str = "loss"

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            mus=list(self.mus),
            losses=list(self.losses),
            l2re=list(self.l2re),
            best_mu=self.best_mu,
            best_loss=self.best_loss,
            best_l2re=self.best_l2re,
            select_by=self.select_by,
        )


# --------------------------------------------------------------------------- #
# plain (unpreconditioned) CG fall-back
# --------------------------------------------------------------------------- #
def _plain_cg(
    A: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    x0: Optional[torch.Tensor] = None,
    mu: float = 0.0,
    epsilon: float = 1e-16,
    M: int = 1000,
    shift: bool = True,
) -> Tuple[torch.Tensor, int, bool]:
    """CG for ``(A + mu I) x = b``; used only if NyströmPCG fails."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()

    def mv(v: torch.Tensor) -> torch.Tensor:
        out = _as_flat(A(v))
        if shift:
            out = out + float(mu) * v
        return out

    r = b - mv(x)
    p = r.clone()
    rs = torch.dot(r, r)
    tol = float(epsilon) * float(torch.linalg.norm(b))
    converged = False
    it = 0
    for it in range(int(M)):
        if float(torch.linalg.norm(r)) <= tol:
            converged = True
            break
        Ap = mv(p)
        denom = torch.dot(p, Ap)
        if not torch.isfinite(denom) or float(denom) <= 0.0:
            break
        alpha = rs / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if float(rs_new) <= tol * tol:
            converged = True
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, it, converged


# --------------------------------------------------------------------------- #
# the optimizer
# --------------------------------------------------------------------------- #
class NNCGOptimizer:
    """NysNewton-CG (Algorithm 4), operating on flat float64 parameter vectors.

    Parameters
    ----------
    model:
        PINN ``nn.Module``; its parameters are updated in place.
    loss_fn:
        Zero-argument closure returning the autograd-tracked scalar PINN loss
        (e.g. from :func:`src.pinns.loss.make_loss_fn`).  The closure must
        re-evaluate the loss at the model's *current* parameters.
    eta, K, s, F, mu, epsilon, M, alpha, beta:
        Algorithm-4 hyper-parameters (paper defaults).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[], torch.Tensor],
        *,
        eta: float = 1.0,
        K: int = 2000,
        s: int = 60,
        F: int = 20,
        mu: float = 1e-2,
        epsilon: float = 1e-16,
        M: int = 1000,
        alpha: float = 0.1,
        beta: float = 0.5,
        dtype: Union[str, torch.dtype] = "float64",
        max_backtracks: int = 100,
        seed: Optional[int] = None,
        cast_dtype: bool = True,
        fallback_to_gradient: bool = True,
        record: bool = True,
        verbose: bool = False,
        config: Optional[NNCGConfig] = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        cfg = config or NNCGConfig(
            eta=eta,
            K=K,
            s=s,
            F=F,
            mu=mu,
            epsilon=epsilon,
            M=M,
            alpha=alpha,
            beta=beta,
            dtype=dtype,
            max_backtracks=max_backtracks,
            seed=seed,
            cast_dtype=cast_dtype,
            fallback_to_gradient=fallback_to_gradient,
        )
        self.config = cfg
        self.dtype = cfg.torch_dtype
        self.eta = cfg.eta
        self.K = cfg.K
        self.s = cfg.s
        self.F = cfg.F
        self.mu = cfg.mu
        self.epsilon = cfg.epsilon
        self.M = cfg.M
        self.alpha = cfg.alpha
        self.beta = cfg.beta
        self.record = record
        self.verbose = verbose
        self.fallback_to_gradient = cfg.fallback_to_gradient
        self.max_backtracks = cfg.max_backtracks

        # state
        self.n_steps = 0
        self.U: Optional[torch.Tensor] = None
        self.Lambda_hat: Optional[torch.Tensor] = None
        self.d_prev: Optional[torch.Tensor] = None  # d_{-1} = 0 at initialisation
        self.last_pcg_info: Optional[Any] = None
        self.last_nystrom_info: Optional[Any] = None
        self.last_armijo_info: Optional[Any] = None
        self.history = NNCGHistory()

        self._generator: Optional[torch.Generator] = None
        if cfg.seed is not None:
            self._generator = torch.Generator(device="cpu")
            self._generator.manual_seed(int(cfg.seed))

        self._line_search: Optional[Any] = None
        self._line_search_err: Optional[str] = None

        if cfg.cast_dtype and self.dtype != next(model.parameters()).dtype:
            try:
                model.to(self.dtype)
            except Exception:  # pragma: no cover - best effort
                pass

    # ------------------------------------------------------------------ #
    # basic plumbing
    # ------------------------------------------------------------------ #
    @property
    def parameters(self) -> List[nn.Parameter]:
        return list(self.model.parameters())

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    def state_dict(self) -> Dict[str, Any]:
        return dict(
            step=self.n_steps,
            mu=self.mu,
            s=self.s,
            F=self.F,
            has_preconditioner=self.U is not None,
            d_prev=None if self.d_prev is None else self.d_prev.detach().clone(),
            history=self.history.as_dict() if self.record else None,
        )

    def flat_params(self) -> torch.Tensor:
        return _flatten_params(self.model, dtype=self.dtype)

    def set_flat_params(self, flat: torch.Tensor) -> None:
        _set_flat_params(self.model, flat.to(self.dtype))

    # ------------------------------------------------------------------ #
    # loss / gradient / Hessian access
    # ------------------------------------------------------------------ #
    def _loss_and_flat_grad(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            loss, grad = _loss_and_grad(self.loss_fn, self.model)
            return loss, _as_flat(grad).to(self.dtype)
        except Exception:  # pragma: no cover - defensive fall-back
            loss = self.loss_fn()
            if not isinstance(loss, torch.Tensor):
                loss = torch.as_tensor(loss)
            params = self.parameters
            grads = torch.autograd.grad(
                loss, params, retain_graph=True, allow_unused=True
            )
            flat = torch.cat(
                [
                    (torch.zeros_like(p) if g is None else g).reshape(-1)
                    for g, p in zip(grads, params)
                ]
            )
            return loss, flat.to(self.dtype)

    def _make_matvec(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return ``v -> H_L(w_k) v`` (Hessian-vector product, Pearlmutter)."""
        try:
            op = HessianOperator(self.loss_fn, self.model, dtype=self.dtype)
            mv = getattr(op, "matvec", None)
            if mv is None:  # pragma: no cover
                raise AttributeError("HessianOperator has no matvec")
            return mv
        except Exception:  # pragma: no cover - defensive fall-back
            def matvec(v: torch.Tensor) -> torch.Tensor:
                return _as_flat(_hvp_apply(self.loss_fn, v, self.model, dtype=self.dtype))

            return matvec

    def _loss_oracle(self) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
        """Flat-parameter loss oracle for the Armijo line search."""
        if self._line_search is not None:
            return None
        try:
            self._line_search = ArmijoLineSearch.from_model(
                self.model,
                self.loss_fn,
                needs_grad=False,
                dtype=self.dtype,
                alpha=self.alpha,
                beta=self.beta,
                max_backtracks=self.max_backtracks,
            )
        except Exception as exc:  # pragma: no cover - defensive fall-back
            self._line_search_err = repr(exc)
            self._line_search = None
        return None

    # ------------------------------------------------------------------ #
    # Algorithm 4 building blocks
    # ------------------------------------------------------------------ #
    def refresh_preconditioner(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """``[U, Lambda_hat] = RandomizedNystromApproximation(H_L(w_k), s)``."""
        n = self.n_parameters
        matvec = self._make_matvec()
        try:
            out = randomized_nystrom_approximation(
                matvec,
                self.s,
                n=n,
                dtype=self.dtype,
                generator=self._generator,
                return_info=True,
            )
        except TypeError:
            out = randomized_nystrom_approximation(matvec, self.s, n=n, dtype=self.dtype)
        if isinstance(out, tuple) and len(out) == 3:
            U, lam, info = out
            self.last_nystrom_info = info
        else:
            U, lam = out
            self.last_nystrom_info = None
        U = U.to(self.dtype)
        lam = lam.reshape(-1).to(self.dtype)
        return U, lam

    def compute_newton_step(
        self, grad: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Any], bool]:
        """``d_k = NystromPCG(H_L(w_k), grad L(w_k), d_{k-1}, U, Lambda_hat, ...)``."""
        matvec = self._make_matvec()
        n = grad.numel()
        x0 = None
        if self.d_prev is not None and self.d_prev.numel() == n:
            x0 = self.d_prev.to(self.dtype)
        U, lam = self.U, self.Lambda_hat
        fell_back = False

        d: Optional[torch.Tensor] = None
        info: Optional[Any] = None
        if U is not None and lam is not None:
            try:
                out = nystrom_pcg(
                    matvec,
                    grad,
                    x0=x0,
                    U=U,
                    Lambda_hat=lam,
                    s=int(U.shape[1]),
                    mu=self.mu,
                    epsilon=self.epsilon,
                    M=self.M,
                    return_info=True,
                )
                if isinstance(out, tuple) and len(out) == 2:
                    d, info = out
                else:  # pragma: no cover - API variation
                    d = out
            except Exception:  # pragma: no cover - fall back to plain CG
                d = None

        if d is None:
            fell_back = True
            try:
                d, it, conv = _plain_cg(
                    matvec,
                    grad,
                    x0=x0,
                    mu=self.mu,
                    epsilon=self.epsilon,
                    M=self.M,
                    shift=True,
                )
                info = NystromPCGInfo(
                    iterations=it,
                    residual_norm=float("nan"),
                    r0_norm=float(torch.linalg.norm(grad)),
                    rel_residual=float("nan"),
                    converged=bool(conv),
                    breakdown=False,
                )
            except Exception:  # pragma: no cover - last resort
                d = grad.clone()

        d = _as_flat(d).to(self.dtype)
        if not torch.isfinite(d).all():
            d = grad.clone()
            fell_back = True
        self.last_pcg_info = info
        return d, info, fell_back

    def line_search(
        self, grad: torch.Tensor, direction: torch.Tensor, f0: float
    ) -> Tuple[float, Optional[Any], bool]:
        """``eta_k = Armijo(L, w_k, grad L(w_k), -d_k, eta)``."""
        self._loss_oracle()
        if self._line_search is not None:
            try:
                out = self._line_search(
                    self.flat_params(),
                    grad,
                    direction,
                    eta=self.eta,
                    return_info=True,
                )
                if isinstance(out, tuple):
                    eta_k, info = out
                    self.last_armijo_info = info
                else:  # pragma: no cover - API variation
                    eta_k, info = float(out), None
                    self.last_armijo_info = None
                eta_k = float(eta_k)
                if math.isfinite(eta_k) and eta_k > 0.0:
                    return eta_k, info, False
            except Exception as exc:  # pragma: no cover
                self._line_search_err = repr(exc)

        # fall-back: simple halving backtracking using the model's own closure
        t = float(self.eta)
        slack = float(self.alpha) * float(torch.dot(grad, direction))
        for _ in range(int(self.max_backtracks)):
            if not math.isfinite(t):
                break
            with torch.no_grad():
                x_new = self.flat_params() + t * direction
                self.set_flat_params(x_new)
                try:
                    f_new = float(self.loss_fn().detach())
                except Exception:  # pragma: no cover
                    f_new = float("inf")
            if math.isfinite(f_new) and f_new <= f0 + t * slack:
                return t, None, False
            t *= float(self.beta)
        # restore iterate
        self.set_flat_params(self.flat_params())
        return float(t), None, True

    # ------------------------------------------------------------------ #
    # one iteration
    # ------------------------------------------------------------------ #
    def step(self) -> Dict[str, Any]:
        """Perform one NNCG iteration (Algorithm 4 loop body)."""
        t0 = time.perf_counter()
        k = self.n_steps

        loss, grad = self._loss_and_flat_grad()
        loss_value = float(loss.detach())
        grad_norm = float(torch.linalg.norm(grad))
        w_k = self.flat_params()

        refreshed = False
        if k % self.F == 0 or self.U is None or self.Lambda_hat is None:
            U, lam = self.refresh_preconditioner()
            if U is not None and lam is not None:
                self.U, self.Lambda_hat = U, lam
                refreshed = True

        d, pcg_info, fell_back = self.compute_newton_step(grad)
        if (
            self.fallback_to_gradient
            and fell_back
            and float(torch.dot(grad, d)) <= 0.0
        ):
            d = grad.clone()

        eta_k, armijo_info, ls_failed = self.line_search(grad, -d, loss_value)

        w_new = w_k - eta_k * d
        if not torch.isfinite(w_new).all():  # pragma: no cover - safety
            w_new = w_k
            eta_k = 0.0
        self.set_flat_params(w_new)

        with torch.no_grad():
            try:
                loss_new = float(self.loss_fn().detach())
            except Exception:  # pragma: no cover
                loss_new = float("nan")
        if math.isfinite(loss_new) and loss_new > loss_value:
            # guarantee monotone decrease (Armijo is approximate in fall-backs)
            self.set_flat_params(w_k)
            with torch.no_grad():
                loss_new = float(self.loss_fn().detach())
            eta_k = 0.0
            ls_failed = True

        self.d_prev = d.detach().clone()
        self.n_steps = k + 1
        elapsed = time.perf_counter() - t0

        info: Dict[str, Any] = dict(
            step=k,
            loss=loss_new,
            loss_prev=loss_value,
            grad_norm=grad_norm,
            step_size=eta_k,
            pcg_iterations=getattr(pcg_info, "iterations", None),
            pcg_converged=getattr(pcg_info, "converged", None),
            preconditioner_refreshed=refreshed,
            armijo_backtracks=getattr(armijo_info, "n_backtracks", None),
            line_search_failed=bool(ls_failed),
            fell_back=bool(fell_back),
            elapsed=elapsed,
        )
        if self.record:
            self.history.append(
                step=k,
                loss=loss_new,
                grad_norm=grad_norm,
                step_size=eta_k,
                pcg_iterations=info["pcg_iterations"],
                refreshed=refreshed,
                armijo_backtracks=info["armijo_backtracks"],
                fell_back=fell_back,
                elapsed=elapsed,
            )
        if self.verbose:
            print(
                f"[NNCG] k={k:5d} loss={loss_new:.6e} |g|={grad_norm:.3e} "
                f"eta={eta_k:.3e} pcg={info['pcg_iterations']} "
                f"refresh={int(refreshed)} fallback={int(fell_back)}"
            )
        return info

    # ------------------------------------------------------------------ #
    def run(
        self,
        n_steps: Optional[int] = None,
        eval_fn: Optional[Callable[[], float]] = None,
        eval_every: int = 50,
        log_every: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any], NNCGHistory], None]] = None,
        history: Optional[NNCGHistory] = None,
    ) -> NNCGHistory:
        """Run ``n_steps`` NNCG iterations (default ``K`` from the config)."""
        if history is not None:
            self.history = history
        n = int(n_steps if n_steps is not None else self.K)
        for _ in range(n):
            info = self.step()
            step = int(info["step"])
            if eval_fn is not None and (
                step % max(1, int(eval_every)) == 0 or step == n - 1 or step == 0
            ):
                try:
                    with torch.no_grad():
                        val = float(eval_fn())
                except Exception:  # pragma: no cover
                    val = float("nan")
                if math.isfinite(val):
                    self.history.l2re[step] = val
            if callback is not None:
                callback(step, info, self.history)
            if log_every and step % int(log_every) == 0:
                print(
                    f"  [NNCG] step {step}/{n} loss {info['loss']:.6e} "
                    f"|g| {info['grad_norm']:.3e}"
                )
        return self.history


# paper-cased aliases
NNCG = NNCGOptimizer
NysNewtonCG = NNCGOptimizer


# --------------------------------------------------------------------------- #
# drivers
# --------------------------------------------------------------------------- #
def run_nncg(
    model: nn.Module,
    loss_fn: Callable[[], torch.Tensor],
    *,
    mu: float = 1e-2,
    n_steps: int = 2000,
    s: int = 60,
    F: int = 20,
    eta: float = 1.0,
    epsilon: float = 1e-16,
    M: int = 1000,
    alpha: float = 0.1,
    beta: float = 0.5,
    dtype: Union[str, torch.dtype] = "float64",
    eval_fn: Optional[Callable[[], float]] = None,
    eval_every: int = 50,
    log_every: Optional[int] = None,
    cast_dtype: bool = True,
    seed: Optional[int] = None,
    history: Optional[NNCGHistory] = None,
    callback: Optional[Callable[[int, Dict[str, Any], NNCGHistory], None]] = None,
    verbose: bool = False,
    config: Optional[NNCGConfig] = None,
    optimizer: Optional[NNCGOptimizer] = None,
    **kwargs: Any,
) -> NNCGResult:
    """Fine-tune ``model`` with NNCG (Algorithm 4) for ``n_steps`` iterations.

    ``loss_fn`` must be a zero-argument closure returning the scalar PINN loss at
    the model's current parameters (see ``src.pinns.loss.make_loss_fn``).
    """
    if config is None:
        config = NNCGConfig(
            eta=eta,
            K=n_steps,
            s=s,
            F=F,
            mu=mu,
            epsilon=epsilon,
            M=M,
            alpha=alpha,
            beta=beta,
            dtype=dtype,
            seed=seed,
            cast_dtype=cast_dtype,
        )
    opt = optimizer
    if opt is None:
        opt = NNCGOptimizer(
            model,
            loss_fn,
            config=config,
            record=True,
            verbose=verbose,
            **{
                k: v
                for k, v in kwargs.items()
                if k in NNCGConfig.__dataclass_fields__  # type: ignore[attr-defined]
            },
        )
    opt.mu = float(mu)
    hist = opt.run(
        n_steps=n_steps,
        eval_fn=eval_fn,
        eval_every=eval_every,
        log_every=log_every,
        callback=callback,
        history=history,
    )
    return NNCGResult(
        history=hist,
        config=config,
        mu=float(mu),
        n_steps=int(n_steps),
        model=model,
        optimizer=opt,
    )


def tune_mu(
    make_setup: Callable[[float], Any],
    mus: Optional[Sequence[float]] = None,
    n_steps: int = 2000,
    *,
    eval_fn_factory: Optional[Callable[[nn.Module], Callable[[], float]]] = None,
    eval_every: int = 50,
    select_by: str = "loss",
    verbose: bool = False,
    **kwargs: Any,
) -> MUTuningResult:
    """Grid-search the damping parameter ``mu`` as in Appendix E.2.

    ``make_setup(mu)`` must return a freshly-initialised ``(model, loss_fn)``
    (or ``(model, loss_fn, eval_fn)``) so that each ``mu`` starts from the same
    Adam+L-BFGS iterate.  ``select_by`` is ``"loss"`` or ``"l2re"``.
    """
    grid = _mu_grid(mus)
    out = MUTuningResult(mus=grid, select_by=select_by)
    best_key = float("inf")
    for mu in grid:
        setup = make_setup(mu)
        eval_fn = None
        if isinstance(setup, (tuple, list)):
            if len(setup) == 2:
                model, loss_fn = setup
            else:
                model, loss_fn, eval_fn = setup[:3]
        else:
            raise TypeError("make_setup(mu) must return (model, loss_fn[, eval_fn])")
        if eval_fn is None and eval_fn_factory is not None:
            eval_fn = eval_fn_factory(model)
        if verbose:
            print(f"[NNCG] tuning mu={mu:g} over {n_steps} steps")
        res = run_nncg(
            model,
            loss_fn,
            mu=mu,
            n_steps=n_steps,
            eval_fn=eval_fn,
            eval_every=eval_every,
            verbose=verbose,
            **kwargs,
        )
        out.results.append(res)
        out.losses.append(res.best_loss)
        out.l2re.append(res.best_l2re)
        key = res.best_l2re if select_by == "l2re" else res.best_loss
        if not math.isfinite(key):
            key = res.final_loss
        if key < best_key:
            best_key = key
            out.best_mu = mu
            out.best_loss = res.best_loss
            out.best_l2re = res.best_l2re
    return out


def _self_test() -> None:  # pragma: no cover - manual smoke test
    """Small quadratic smoke test of Algorithm 4 (no PINN required)."""
    torch.manual_seed(0)
    n = 40
    Q = torch.randn(n, n, dtype=torch.float64)
    A = Q @ Q.T / n + 0.5 * torch.eye(n, dtype=torch.float64)
    b = torch.randn(n, dtype=torch.float64)
    x_star = torch.linalg.solve(A, b)

    class Quad(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.x = nn.Parameter(torch.zeros(n, dtype=torch.float64))

    model = Quad()

    def loss_fn() -> torch.Tensor:
        r = A @ model.x - b
        return 0.5 * torch.dot(r, r)

    opt = NNCGOptimizer(model, loss_fn, mu=1e-2, s=10, F=5)
    hist = opt.run(n_steps=10)
    print("loss:", hist.losses[0], "->", hist.final_loss)
    print("||x - x*||:", float(torch.linalg.norm(model.x.detach() - x_star)))


if __name__ == "__main__":  # pragma: no cover
    _self_test()
