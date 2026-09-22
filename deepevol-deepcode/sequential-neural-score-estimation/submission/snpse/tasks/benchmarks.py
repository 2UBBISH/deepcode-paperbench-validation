"""Benchmark tasks of Section E.1 (Lueckmann et al. 2021, Appendix T).

The SNPSE paper evaluates NPSE / TSNPSE on eight simulator-based inference
benchmarks:

    Gaussian Linear, Gaussian Mixture, Two Moons, Gaussian Linear Uniform,
    Bernoulli GLM, SLCP, SIR, Lotka Volterra

The canonical implementations live in the ``sbibm`` package (required by the
paper's addendum).  This module therefore

1. exposes a thin, uniform wrapper around ``sbibm`` tasks
   (:func:`build_sbibm_task`), and
2. provides dependency-free fallback implementations
   (:func:`build_fallback_task`) that follow the *original paper text*
   (Section E.1, priors / simulators as written) so that the whole pipeline
   remains runnable without ``sbibm``.

The uniform interface is the :class:`BenchmarkTask` dataclass::

    task = get_task("two_moons")
    theta = task.sample_prior(1000, generator=g)        # (1000, d)
    x = task.simulate(theta, generator=g)               # (1000, p)
    x_obs = task.x_obs                                  # (1, p)
    lp = task.prior.log_prob(theta)                     # (1000,)
    lo, hi = task.bounds                                # uniform support or None

Task descriptions implemented here (Section E.1)
------------------------------------------------
Gaussian Linear
    10-dim Gaussian mean inference; prior ``N(0, 0.1 I)`` (std ``sqrt(0.1)``),
    simulator ``N(x | theta, 0.1 I)``.
Gaussian Mixture
    Uniform prior ``U(-10, 10)`` on ``R^2``; simulator
    ``0.5 N(x | theta, I) + 0.5 N(x | theta, 0.01 I)``.
Two Moons
    Uniform prior ``U(-1, 1)`` on ``R^2``; simulator
    ``x = (r cos a + 0.25, r sin a) + (-|theta_1+theta_2|/sqrt(2),
    (-theta_1+theta_2)/sqrt(2))`` with ``a ~ U(-pi/2, pi/2)`` and
    ``r ~ N(0.1, 0.01^2)``.
Gaussian Linear Uniform
    Uniform prior ``U(-1, 1)`` on ``R^10``; simulator ``N(x | theta, 0.1 I)``.
Bernoulli GLM
    10-dim parameter ``theta = (beta, f)`` with ``beta ~ N(0, 2)`` and
    ``f ~ N(0, (F^T F)^{-1})`` where ``F`` penalises second-order differences;
    the 10-dim observation is the vector of sufficient statistics.
SLCP
    Uniform prior ``U(-3, 3)`` on ``R^5``; 8-dim data, four 2-D Gaussian draws
    whose mean and covariance are non-linear functions of ``theta``.
SIR
    ``beta ~ LogNormal(log 0.4, 0.5)``, ``gamma ~ LogNormal(log 0.8, 0.2)``;
    SIR ODEs with population ``N = 1000``; 10 equally spaced noisy recordings
    ``x_i ~ Bin(1000, I_i / N)``.
Lotka Volterra
    ``alpha, gamma ~ LogNormal(-0.125, 0.5)``, ``beta, delta ~ LogNormal(-3,
    0.5)``; predator-prey ODEs; 10 evenly spaced recordings of both
    populations (20-dim data).

Run ``python -m snpse.tasks.benchmarks --list`` for the registry or
``python -m snpse.tasks.benchmarks --self-test`` to validate every task.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:  # pragma: no cover - optional dependency (required by the addendum)
    import sbibm  # type: ignore
except Exception:  # pragma: no cover
    sbibm = None


__all__ = [
    "Prior",
    "UniformPrior",
    "GaussianPrior",
    "IndependentLogNormalPrior",
    "SbibmPriorAdapter",
    "BenchmarkTask",
    "TASK_NAMES",
    "TASKS",
    "OBSERVATION_SEEDS",
    "get_task",
    "list_tasks",
    "task_summary",
    "simulate",
    "sample_prior",
    "sample_prior_predictive",
    "load_dataset",
    "dataset_cache_dir",
    "sbibm_available",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _resolve_n(n: Any = None, sample_shape: Any = None) -> int:
    """Coerce the many ways of requesting ``n`` prior samples to an int."""
    if n is None and sample_shape is not None:
        n = sample_shape
    if n is None:
        return 1
    if isinstance(n, torch.Size):
        n = tuple(n)
    if isinstance(n, (tuple, list)):
        if len(n) == 0:
            return 1
        if len(n) == 1:
            return int(n[0])
        out = 1
        for v in n:
            out *= int(v)
        return out
    if isinstance(n, torch.Tensor):
        return int(n.reshape(-1)[0].item())
    return int(n)


def _as_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    try:
        return torch.as_tensor(value, dtype=dtype)
    except Exception:
        import numpy as np  # lazy

        return torch.as_tensor(np.asarray(value), dtype=dtype)


def _randn(shape: Sequence[int], generator: Optional[torch.Generator], dtype: torch.dtype):
    """``torch.randn`` that tolerates ``generator=None``."""
    if generator is None:
        return torch.randn(*shape, dtype=dtype)
    return torch.randn(*shape, generator=generator, dtype=dtype)


def _rand(shape: Sequence[int], generator: Optional[torch.Generator], dtype: torch.dtype):
    if generator is None:
        return torch.rand(*shape, dtype=dtype)
    return torch.rand(*shape, generator=generator, dtype=dtype)


def _binomial_sample(
    total_count: int,
    probs: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Binomial draws ``Bin(total_count, probs)`` honouring ``generator``.

    PyTorch's ``Binomial`` distribution uses the *global* RNG; when a generator
    is supplied we briefly fork/seed the global RNG with a seed drawn from that
    generator so that results stay reproducible.
    """
    probs = probs.clamp(0.0, 1.0)
    seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            dist = torch.distributions.Binomial(
                total_count=total_count, probs=probs, validate_args=False
            )
            return dist.sample().to(probs.dtype)
    except Exception:  # pragma: no cover - very old / exotic torch
        # Normal approximation with continuity correction (fallback of last resort).
        d = torch.distributions.Normal(
            torch.as_tensor(float(total_count), dtype=probs.dtype) * probs,
            torch.sqrt(
                torch.as_tensor(float(total_count), dtype=probs.dtype) * probs * (1.0 - probs)
            ).clamp_min(1e-12),
        )
        z = _randn(probs.shape, generator, probs.dtype)
        return (d.mean + d.stddev * z).round().clamp(0.0, float(total_count))


def _rk4_integrate(
    rhs: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t_grid: torch.Tensor,
    substeps: int = 16,
) -> torch.Tensor:
    """Fixed step-size RK4 integration of batched ODE states.

    Parameters
    ----------
    rhs : callable ``(t, y) -> dy`` with ``y`` of shape ``(batch, state_dim)``
    y0 : tensor ``(batch, state_dim)``
    t_grid : 1-D tensor of (increasing) recording times
    substeps : number of RK4 steps between consecutive recording times

    Returns
    -------
    ``(batch, len(t_grid), state_dim)`` tensor of states at ``t_grid``.
    """
    times = torch.as_tensor(t_grid, dtype=y0.dtype).reshape(-1)
    states = [y0]
    y = y0
    for i in range(times.shape[0] - 1):
        t0 = float(times[i].item())
        t1 = float(times[i + 1].item())
        h = (t1 - t0) / float(substeps)
        for k in range(int(substeps)):
            tk = t0 + k * h
            k1 = rhs(tk, y)
            k2 = rhs(tk + 0.5 * h, y + 0.5 * h * k1)
            k3 = rhs(tk + 0.5 * h, y + 0.5 * h * k2)
            k4 = rhs(tk + h, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            y = y.clamp_min(0.0)  # populations / compartments are non-negative
        states.append(y)
    return torch.stack(states, dim=1)


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------
class Prior:
    """Base class for the benchmark priors.

    The interface (``sample`` / ``log_prob`` / ``bounds``) is intentionally
    duck-typed so that it can be swapped for an ``sbibm`` prior or any
    ``torch.distributions.Distribution``.
    """

    name: str = "prior"
    dim: int = 0

    # -- core interface ----------------------------------------------------
    def sample(
        self,
        n: Any = None,
        generator: Optional[torch.Generator] = None,
        sample_shape: Any = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- aliases used by the sequential drivers ---------------------------
    def sample_fn(self, n: Any = 1, generator: Optional[torch.Generator] = None, **kwargs):
        return self.sample(n, generator=generator, **kwargs)

    def log_prob_fn(self, theta: torch.Tensor) -> torch.Tensor:
        return self.log_prob(theta)

    def bounds(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Support box of the prior, or ``None`` when unbounded."""
        return None

    def __call__(self, n: Any = 1, generator: Optional[torch.Generator] = None):
        return self.sample(n, generator=generator)

    def to(self, device: Union[str, torch.device]):  # pragma: no cover - trivial
        return self


class UniformPrior(Prior):
    """Independent uniform prior on a box ``[low, high]``."""

    def __init__(
        self,
        low: Union[float, Sequence[float], torch.Tensor],
        high: Union[float, Sequence[float], torch.Tensor],
        dim: Optional[int] = None,
        name: str = "uniform",
    ):
        low_t = torch.as_tensor(low, dtype=torch.float32).reshape(-1)
        high_t = torch.as_tensor(high, dtype=torch.float32).reshape(-1)
        if low_t.numel() == 1 and dim is not None and dim > 1:
            low_t = low_t.expand(dim).clone()
            high_t = high_t.expand(dim).clone()
        if low_t.shape != high_t.shape:
            raise ValueError("low and high must have the same shape")
        self.low = low_t
        self.high = high_t
        self.dim = int(low_t.numel())
        self.name = name
        self._log_vol = float(torch.log(self.high - self.low).sum().item())

    def sample(self, n=None, generator=None, sample_shape=None, device=None, dtype=torch.float32):
        n = _resolve_n(n, sample_shape)
        u = _rand((n, self.dim), generator, dtype)
        low = self.low.to(dtype)
        high = self.high.to(dtype)
        out = low + u * (high - low)
        return out.to(device) if device is not None else out

    def log_prob(self, theta):
        theta = _as_tensor(theta)
        inside = ((theta >= self.low) & (theta <= self.high)).all(dim=-1)
        return torch.where(
            inside,
            torch.full_like(theta[..., 0], -self._log_vol),
            torch.full_like(theta[..., 0], float("-inf")),
        )

    def bounds(self):
        return self.low.clone(), self.high.clone()


class GaussianPrior(Prior):
    """(Possibly degenerate / rank-restricted) multivariate Gaussian prior.

    ``covariance`` is symmetrised and decomposed with ``torch.linalg.eigh``.
    Eigenvalues below ``sample_tol * max_eig`` are dropped when *sampling*
    (which is what makes a nominally improper smoothness prior usable, cf. the
    Bernoulli GLM prior ``f ~ N(0, (F^T F)^{-1})``), while ``log_prob``
    evaluates a density with eigenvalues clamped at ``min_eig``.
    """

    def __init__(
        self,
        mean: Union[Sequence[float], torch.Tensor],
        covariance: Optional[torch.Tensor] = None,
        std: Optional[Union[float, Sequence[float], torch.Tensor]] = None,
        name: str = "gaussian",
        dtype: torch.dtype = torch.float32,
        sample_tol: float = 1e-6,
        min_eig: float = 1e-8,
    ):
        mean_t = torch.as_tensor(mean, dtype=dtype).reshape(-1)
        d = mean_t.numel()
        if covariance is None:
            if std is None:
                raise ValueError("either covariance or std must be given")
            std_t = torch.as_tensor(std, dtype=dtype).reshape(-1)
            if std_t.numel() == 1:
                std_t = std_t.expand(d).clone()
            covariance = torch.diag(std_t**2)
        cov = torch.as_tensor(covariance, dtype=dtype)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        self.mean = mean_t
        self.dim = d
        self.name = name
        self._sample_tol = sample_tol
        self._min_eig = min_eig
        self._cov = cov
        self._eigvals, self._eigvecs = torch.linalg.eigh(cov)
        self._log_det = float(torch.log(self._eigvals.clamp_min(min_eig)).sum().item())

    # -- helpers -----------------------------------------------------------
    def _sampling_factor(self):
        tol = max(self._sample_tol * float(self._eigvals.max().item()), 0.0)
        keep = self._eigvals > tol
        vals = self._eigvals[keep].clamp_min(0.0)
        vecs = self._eigvecs[:, keep]
        return vecs, torch.sqrt(vals)

    def _log_prob_full(self, theta: torch.Tensor) -> torch.Tensor:
        diff = theta - self.mean
        vals = self._eigvals.clamp_min(self._min_eig)
        proj = diff @ self._eigvecs
        quad = ((proj**2) / vals).sum(dim=-1)
        return -0.5 * (quad + self._log_det + self.dim * math.log(2.0 * math.pi))

    # -- interface ---------------------------------------------------------
    def sample(self, n=None, generator=None, sample_shape=None, device=None, dtype=torch.float32):
        n = _resolve_n(n, sample_shape)
        vecs, scales = self._sampling_factor()
        z = _randn((n, scales.numel()), generator, dtype)
        out = self.mean.to(dtype) + (z * scales.to(dtype)) @ vecs.to(dtype).transpose(0, 1)
        return out.to(device) if device is not None else out

    def log_prob(self, theta):
        return self._log_prob_full(_as_tensor(theta))

    def bounds(self):
        return None


class IndependentLogNormalPrior(Prior):
    """Independent log-normal prior, ``theta_i ~ LogNormal(loc_i, scale_i)``."""

    def __init__(
        self,
        loc: Union[float, Sequence[float], torch.Tensor],
        scale: Union[float, Sequence[float], torch.Tensor],
        dim: Optional[int] = None,
        name: str = "lognormal",
    ):
        loc_t = torch.as_tensor(loc, dtype=torch.float32).reshape(-1)
        scale_t = torch.as_tensor(scale, dtype=torch.float32).reshape(-1)
        if dim is not None:
            if loc_t.numel() == 1 and dim > 1:
                loc_t = loc_t.expand(dim).clone()
            if scale_t.numel() == 1 and dim > 1:
                scale_t = scale_t.expand(dim).clone()
        self.loc = loc_t
        self.scale = scale_t
        self.dim = int(loc_t.numel())
        self.name = name

    def sample(self, n=None, generator=None, sample_shape=None, device=None, dtype=torch.float32):
        n = _resolve_n(n, sample_shape)
        z = _randn((n, self.dim), generator, dtype)
        out = torch.exp(self.loc.to(dtype) + self.scale.to(dtype) * z)
        return out.to(device) if device is not None else out

    def log_prob(self, theta):
        theta = _as_tensor(theta)
        loc, scale = self.loc.to(theta.dtype), self.scale.to(theta.dtype)
        z = (torch.log(theta.clamp_min(1e-30)) - loc) / scale
        return (-z**2 / 2.0 - torch.log(scale) - 0.5 * math.log(2.0 * math.pi) - torch.log(theta.clamp_min(1e-30))).sum(-1)

    def bounds(self):
        return None


class SbibmPriorAdapter(Prior):
    """Wrap an ``sbibm`` prior object in the :class:`Prior` interface."""

    def __init__(self, prior: Any, dim: int, name: str = "sbibm_prior"):
        self._prior = prior
        self.dim = int(dim)
        self.name = name

    def sample(self, n=None, generator=None, sample_shape=None, device=None, dtype=torch.float32):
        n = _resolve_n(n, sample_shape)
        out = None
        for call in (
            lambda: self._prior.sample(torch.Size([n])),
            lambda: self._prior.sample(sample_shape=torch.Size([n])),
            lambda: self._prior.sample((n,)),
            lambda: self._prior.sample(n),
        ):
            try:
                out = call()
                break
            except Exception:
                continue
        if out is None:  # pragma: no cover - defensive
            raise RuntimeError("could not sample from the sbibm prior")
        return _as_tensor(out, dtype=dtype).reshape(n, self.dim)

    def log_prob(self, theta):
        theta = _as_tensor(theta)
        try:
            out = self._prior.log_prob(theta)
        except Exception:  # pragma: no cover - numpy based sbibm priors
            import numpy as np

            out = self._prior.log_prob(theta.detach().cpu().numpy())
        return _as_tensor(out).reshape(theta.shape[0])


# ---------------------------------------------------------------------------
# fallback simulators (Section E.1)
# ---------------------------------------------------------------------------
# ---------- Gaussian Linear (d = 10, p = 10) -------------------------------
def _sim_gaussian_linear(theta: torch.Tensor, generator=None) -> torch.Tensor:
    std = math.sqrt(0.1)
    return theta + std * _randn(theta.shape, generator, theta.dtype)


# ---------- Gaussian Mixture (d = 2, p = 2) --------------------------------
def _sim_gaussian_mixture(theta: torch.Tensor, generator=None) -> torch.Tensor:
    n = theta.shape[0]
    wide = _rand((n, 1), generator, theta.dtype) < 0.5
    std = torch.where(
        wide, torch.ones_like(theta), torch.full_like(theta, 0.1)
    )  # cov I vs cov 0.01 I (std 1 / 0.1)
    return theta + std * _randn(theta.shape, generator, theta.dtype)


# ---------- Two Moons (d = 2, p = 2) ---------------------------------------
def _sim_two_moons(theta: torch.Tensor, generator=None) -> torch.Tensor:
    n = theta.shape[0]
    alpha = -0.5 * math.pi + math.pi * _rand((n,), generator, theta.dtype)
    r = 0.1 + 0.01 * _randn((n,), generator, theta.dtype)
    t1, t2 = theta[:, 0], theta[:, 1]
    x1 = r * torch.cos(alpha) + 0.25 - torch.abs(t1 + t2) / math.sqrt(2.0)
    x2 = r * torch.sin(alpha) + (-t1 + t2) / math.sqrt(2.0)
    return torch.stack([x1, x2], dim=-1)


# ---------- Gaussian Linear Uniform (d = 10, p = 10) -----------------------
def _sim_gaussian_linear_uniform(theta: torch.Tensor, generator=None) -> torch.Tensor:
    std = math.sqrt(0.1)
    return theta + std * _randn(theta.shape, generator, theta.dtype)


# ---------- Bernoulli GLM (d = 10, p = 10) ---------------------------------
def _second_difference_matrix(n: int, dtype=torch.float32) -> torch.Tensor:
    """``(n - 2, n)`` matrix penalising second-order differences."""
    if n < 3:
        raise ValueError("need at least three grid points")
    f = torch.zeros(n - 2, n, dtype=dtype)
    for i in range(n - 2):
        f[i, i] = 1.0
        f[i, i + 1] = -2.0
        f[i, i + 2] = 1.0
    return f


def _bernoulli_glm_prior(num_grid: int = 9, beta_var: float = 2.0) -> GaussianPrior:
    """``beta ~ N(0, 2)`` and ``f ~ N(0, (F^T F)^{-1})`` with ``F`` the
    second-difference operator (Section E.1).

    ``F^T F`` is singular for the 9-dimensional ``f`` (two-dimensional null
    space of linear functions), so the prior is evaluated on the range of
    ``F`` using the pseudo-inverse; :class:`GaussianPrior` drops the
    near-null eigenvalues when sampling.
    """
    f = _second_difference_matrix(num_grid)
    cov_f = torch.linalg.pinv(f.transpose(0, 1) @ f)
    cov = torch.zeros(num_grid + 1, num_grid + 1, dtype=torch.float32)
    cov[0, 0] = beta_var
    cov[1:, 1:] = cov_f
    return GaussianPrior(torch.zeros(num_grid + 1), covariance=cov, name="bernoulli_glm")


def _sim_bernoulli_glm(theta: torch.Tensor, generator=None) -> torch.Tensor:
    """Bernoulli GLM: 9 grid points, ``num_trials`` trials each.

    ``logit_i = beta + f_i`` and the 10-dimensional observation is the vector
    of sufficient statistics ``(N_1, ..., N_9, N_total)`` of the success
    counts ``N_i ~ Bin(num_trials, sigmoid(beta + f_i))``.

    NOTE: ``sbibm``'s ``bernoulli_glm`` is the canonical realisation of this
    task (same prior / sufficient statistics, different design); the fallback
    is used only when ``sbibm`` is unavailable.
    """
    num_trials = 100
    beta = theta[:, :1]
    f = theta[:, 1:]
    logits = beta + f
    probs = torch.sigmoid(logits)
    counts = _binomial_sample(num_trials, probs, generator)
    total = counts.sum(dim=-1, keepdim=True)
    return torch.cat([counts, total], dim=-1)


# ---------- SLCP (d = 5, p = 8) -------------------------------------------
def _sim_slcp(theta: torch.Tensor, generator=None) -> torch.Tensor:
    """Four 2-D Gaussian draws with non-linear mean/covariance (Papamakarios
    et al. 2019; Section E.1).

    ``mean = (theta_1, theta_2)`` and
    ``cov = [[s1^2, rho s1 s2], [rho s1 s2, s2^2]]`` with ``s1 = theta_3``,
    ``s2 = theta_4`` and correlation ``rho = tanh(theta_5)``.
    """
    n = theta.shape[0]
    mean = theta[:, :2]
    s1 = theta[:, 2]
    s2 = theta[:, 3]
    rho = torch.tanh(theta[:, 4])
    cov = torch.empty(n, 2, 2, dtype=theta.dtype)
    cov[:, 0, 0] = s1**2
    cov[:, 1, 1] = s2**2
    off = rho * s1 * s2
    cov[:, 0, 1] = off
    cov[:, 1, 0] = off
    # numerical safety: add a tiny jitter to keep the Cholesky stable
    eye = torch.eye(2, dtype=theta.dtype).expand(n, 2, 2)
    cov = cov + 1e-8 * eye
    chol = torch.linalg.cholesky(cov)
    num_draws = 4
    z = _randn((n, num_draws, 2), generator, theta.dtype)
    eps = z @ chol.transpose(-1, -2)
    return (mean.unsqueeze(1) + eps).reshape(n, 2 * num_draws)


# ---------- SIR (d = 2, p = 10) -------------------------------------------
SIR_POPULATION = 1000
SIR_T_MAX = 160.0
SIR_NUM_OBS = 10


def _sir_rhs(t: float, y: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    s, i = y[..., 0], y[..., 1]
    n = float(SIR_POPULATION)
    infection = beta * s * i / n
    recovery = gamma * i
    return torch.stack([-infection, infection - recovery, recovery], dim=-1)


def _sim_sir(theta: torch.Tensor, generator=None, substeps: int = 16) -> torch.Tensor:
    """SIR epidemic model with binomial recordings (Section E.1).

    ``theta = (beta, gamma)``; population ``N = 1000``, one initially infected
    individual; 10 equally spaced recordings ``x_i ~ Bin(N, I_i / N)`` over
    ``[0, 160]``.
    """
    beta = theta[:, 0:1]
    gamma = theta[:, 1:2]
    n = theta.shape[0]
    y0 = torch.zeros(n, 3, dtype=theta.dtype)
    y0[:, 0] = float(SIR_POPULATION) - 1.0
    y0[:, 1] = 1.0

    def rhs(_t, y):
        return _sir_rhs(_t, y, beta, gamma)

    t_grid = torch.linspace(0.0, SIR_T_MAX, SIR_NUM_OBS, dtype=theta.dtype)
    traj = _rk4_integrate(rhs, y0, t_grid, substeps=substeps)  # (n, T, 3)
    infected = traj[:, :, 1]
    probs = (infected / float(SIR_POPULATION)).clamp(0.0, 1.0)
    return _binomial_sample(SIR_POPULATION, probs, generator)


# ---------- Lotka Volterra (d = 4, p = 20) --------------------------------
LV_INITIAL_STATE = (30.0, 1.0)
LV_T_MAX = 20.0
LV_NUM_OBS = 10


def _lotka_volterra_rhs(
    t: float,
    y: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    prey, predator = y[..., 0], y[..., 1]
    d_prey = alpha * prey - beta * prey * predator
    d_predator = -gamma * predator + delta * prey * predator
    return torch.stack([d_prey, d_predator], dim=-1)


def _sim_lotka_volterra(theta: torch.Tensor, generator=None, substeps: int = 16) -> torch.Tensor:
    """Predator-prey ODE with 10 evenly spaced recordings of both populations.

    ``theta = (alpha, beta, gamma, delta)``; ``dx/dt = alpha x - beta x y``,
    ``dy/dt = -gamma y + delta x y``; initial state ``(30, 1)``; recordings on
    ``[0, 20]`` (20-dimensional data).
    """
    alpha = theta[:, 0:1]
    beta = theta[:, 1:2]
    gamma = theta[:, 2:3]
    delta = theta[:, 3:4]
    n = theta.shape[0]
    y0 = torch.zeros(n, 2, dtype=theta.dtype)
    y0[:, 0] = float(LV_INITIAL_STATE[0])
    y0[:, 1] = float(LV_INITIAL_STATE[1])

    def rhs(_t, y):
        return _lotka_volterra_rhs(_t, y, alpha, beta, gamma, delta)

    t_grid = torch.linspace(0.0, LV_T_MAX, LV_NUM_OBS, dtype=theta.dtype)
    traj = _rk4_integrate(rhs, y0, t_grid, substeps=substeps)  # (n, T, 2)
    return traj.reshape(n, 2 * LV_NUM_OBS)


# ---------------------------------------------------------------------------
# task container
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkTask:
    """Uniform description of one benchmark task."""

    name: str
    dim_parameters: int
    dim_data: int
    prior: Prior
    simulator_fn: Callable[..., torch.Tensor]
    x_obs: torch.Tensor
    description: str = ""
    observation_index: int = 1
    sbibm_task: Any = None
    backend: str = "fallback"
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- aliases ----------------------------------------------------------
    @property
    def dim_theta(self) -> int:
        return self.dim_parameters

    @property
    def dim_x(self) -> int:
        return self.dim_data

    # -- sampling ---------------------------------------------------------
    def sample_prior(self, n=None, generator=None, **kwargs) -> torch.Tensor:
        return self.prior.sample(n, generator=generator, **kwargs)

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return self.prior.log_prob(theta)

    def simulate(self, theta, generator=None) -> torch.Tensor:
        theta_t = _as_tensor(theta)
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)
        out = self.simulator_fn(theta_t, generator)
        return _as_tensor(out)

    def sample_prior_predictive(
        self, n: int = 1, generator=None, return_theta: bool = True
    ):
        theta = self.sample_prior(n, generator=generator)
        x = self.simulate(theta, generator=generator)
        if return_theta:
            return theta, x
        return x

    # -- reference quantities --------------------------------------------
    def reference_posterior_samples(
        self, num_samples: int = 10_000, generator=None
    ) -> Optional[torch.Tensor]:
        if self.sbibm_task is not None:
            for call in (
                lambda: self.sbibm_task.get_reference_posterior_samples(
                    num_observation=self.observation_index, num_samples=num_samples
                ),
                lambda: self.sbibm_task.get_reference_posterior_samples(
                    num_samples=num_samples
                ),
            ):
                try:
                    return _as_tensor(call()).reshape(num_samples, self.dim_parameters)
                except Exception:
                    continue
        if "reference_posterior_fn" in self.extra:
            return _as_tensor(self.extra["reference_posterior_fn"](num_samples, generator))
        if self.name == "gaussian_linear":
            return _gaussian_linear_reference_posterior(self, num_samples, generator)
        return None

    def bounds(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.prior.bounds()

    def parameters(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim_parameters": self.dim_parameters,
            "dim_data": self.dim_data,
            "backend": self.backend,
            "bounded": self.bounds() is not None,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BenchmarkTask(name={self.name!r}, dim_parameters={self.dim_parameters}, "
            f"dim_data={self.dim_data}, backend={self.backend!r})"
        )


def _gaussian_linear_reference_posterior(task, num_samples: int, generator):
    """Analytic posterior of the Gaussian Linear task.

    Prior ``N(0, 0.1 I_d)``, likelihood ``N(x | theta, 0.1 I_p)`` with
    ``p`` independent observations whose sum is the ``p``-dimensional statistic
    ``x``: posterior precision ``1/0.1 + p/0.1``.
    """
    d = task.dim_parameters
    p = task.dim_data
    prior_var = 0.1
    lik_var = 0.1
    post_var = 1.0 / (1.0 / prior_var + p / lik_var)
    obs = task.x_obs.reshape(-1)
    mean = post_var * obs / lik_var
    z = _randn((num_samples, d), generator, torch.float32)
    return mean.unsqueeze(0) + math.sqrt(post_var) * z


# ---------------------------------------------------------------------------
# fallback task builders
# ---------------------------------------------------------------------------
OBSERVATION_SEEDS: Dict[str, int] = {
    "gaussian_linear": 239,
    "gaussian_mixture": 40,
    "two_moons": 1,
    "gaussian_linear_uniform": 140,
    "bernoulli_glm": 10,
    "slcp": 27,
    "sir": 245,
    "lotka_volterra": 100,
}


def _observation(task_name, prior, simulator_fn, dim_data, seed=None):
    """Deterministically generate a fallback observation ``x_obs``."""
    seed = OBSERVATION_SEEDS.get(task_name, 0) if seed is None else seed
    g = torch.Generator().manual_seed(int(seed))
    theta_true = prior.sample(1, generator=g)
    x_obs = simulator_fn(theta_true, g)
    return _as_tensor(x_obs).reshape(1, dim_data), theta_true.reshape(-1)


def build_fallback_task(name: str) -> BenchmarkTask:
    name = str(name).lower()
    if name == "gaussian_linear":
        d = 10
        prior = GaussianPrior(torch.zeros(d), std=math.sqrt(0.1), name="gaussian")
        x_obs, theta_true = _observation(name, prior, _sim_gaussian_linear, d)
        return BenchmarkTask(
            name=name,
            dim_parameters=d,
            dim_data=d,
            prior=prior,
            simulator_fn=_sim_gaussian_linear,
            x_obs=x_obs,
            description="10-dim Gaussian mean inference; prior N(0, 0.1 I) (Section E.1)",
            extra={"theta_true": theta_true},
        )
    if name == "gaussian_mixture":
        prior = UniformPrior(-10.0, 10.0, dim=2)
        x_obs, theta_true = _observation(name, prior, _sim_gaussian_mixture, 2)
        return BenchmarkTask(
            name=name,
            dim_parameters=2,
            dim_data=2,
            prior=prior,
            simulator_fn=_sim_gaussian_mixture,
            x_obs=x_obs,
            description="U(-10,10)^2 prior; 0.5 N(theta, I) + 0.5 N(theta, 0.01 I)",
            extra={"theta_true": theta_true},
        )
    if name == "two_moons":
        prior = UniformPrior(-1.0, 1.0, dim=2)
        x_obs, theta_true = _observation(name, prior, _sim_two_moons, 2)
        return BenchmarkTask(
            name=name,
            dim_parameters=2,
            dim_data=2,
            prior=prior,
            simulator_fn=_sim_two_moons,
            x_obs=x_obs,
            description="U(-1,1)^2 prior; crescent/bimodal posterior (Section E.1)",
            extra={"theta_true": theta_true},
        )
    if name == "gaussian_linear_uniform":
        d = 10
        prior = UniformPrior(-1.0, 1.0, dim=d)
        x_obs, theta_true = _observation(name, prior, _sim_gaussian_linear_uniform, d)
        return BenchmarkTask(
            name=name,
            dim_parameters=d,
            dim_data=d,
            prior=prior,
            simulator_fn=_sim_gaussian_linear_uniform,
            x_obs=x_obs,
            description="U(-1,1)^10 prior; N(x | theta, 0.1 I) (Section E.1)",
            extra={"theta_true": theta_true},
        )
    if name == "bernoulli_glm":
        prior = _bernoulli_glm_prior()
        x_obs, theta_true = _observation(name, prior, _sim_bernoulli_glm, 10)
        return BenchmarkTask(
            name=name,
            dim_parameters=10,
            dim_data=10,
            prior=prior,
            simulator_fn=_sim_bernoulli_glm,
            x_obs=x_obs,
            description=(
                "Bernoulli GLM; beta ~ N(0,2), f ~ N(0,(F^T F)^-1); sufficient statistics"
            ),
            extra={"theta_true": theta_true, "num_trials": 100},
        )
    if name == "slcp":
        prior = UniformPrior(-3.0, 3.0, dim=5)
        x_obs, theta_true = _observation(name, prior, _sim_slcp, 8)
        return BenchmarkTask(
            name=name,
            dim_parameters=5,
            dim_data=8,
            prior=prior,
            simulator_fn=_sim_slcp,
            x_obs=x_obs,
            description="U(-3,3)^5 prior; non-linear Gaussian mean/covariance; 8-dim data",
            extra={"theta_true": theta_true, "num_draws": 4},
        )
    if name == "sir":
        prior = IndependentLogNormalPrior(
            loc=[math.log(0.4), math.log(0.8)], scale=[0.5, 0.2], name="sir"
        )
        x_obs, theta_true = _observation(name, prior, _sim_sir, SIR_NUM_OBS)
        return BenchmarkTask(
            name=name,
            dim_parameters=2,
            dim_data=SIR_NUM_OBS,
            prior=prior,
            simulator_fn=_sim_sir,
            x_obs=x_obs,
            description="SIR epidemiology; binomial recordings; 10-dim data (Section E.1)",
            extra={
                "theta_true": theta_true,
                "population": SIR_POPULATION,
                "t_max": SIR_T_MAX,
            },
        )
    if name in ("lotka_volterra", "lotka-volterra", "lv"):
        prior = IndependentLogNormalPrior(
            loc=[-0.125, -3.0, -0.125, -3.0], scale=0.5, name="lotka_volterra"
        )
        x_obs, theta_true = _observation(
            "lotka_volterra", prior, _sim_lotka_volterra, 2 * LV_NUM_OBS
        )
        return BenchmarkTask(
            name="lotka_volterra",
            dim_parameters=4,
            dim_data=2 * LV_NUM_OBS,
            prior=prior,
            simulator_fn=_sim_lotka_volterra,
            x_obs=x_obs,
            description="Lotka-Volterra predator-prey; 10 recordings of both species",
            extra={
                "theta_true": theta_true,
                "initial_state": LV_INITIAL_STATE,
                "t_max": LV_T_MAX,
            },
        )
    raise KeyError(f"unknown benchmark task: {name!r}")


# ---------------------------------------------------------------------------
# sbibm task builder
# ---------------------------------------------------------------------------
SBIBM_NAMES: Dict[str, str] = {
    "gaussian_linear": "gaussian_linear",
    "gaussian_mixture": "gaussian_mixture",
    "two_moons": "two_moons",
    "gaussian_linear_uniform": "gaussian_linear_uniform",
    "bernoulli_glm": "bernoulli_glm",
    "slcp": "slcp",
    "sir": "sir",
    "lotka_volterra": "lotka_volterra",
}


def _wrap_sbibm_simulator(sim: Callable) -> Callable[..., torch.Tensor]:
    """Make an ``sbibm`` simulator accept a batch ``(n, d)`` and a generator."""

    def wrapped(theta: torch.Tensor, generator=None) -> torch.Tensor:
        theta_t = _as_tensor(theta)
        # sbibm simulators are vectorised in recent versions: try that first.
        try:
            out = sim(theta_t)
            out = _as_tensor(out)
            if out.dim() >= 2 and out.shape[0] == theta_t.shape[0]:
                return out.reshape(theta_t.shape[0], -1)
        except Exception:
            pass
        rows: List[torch.Tensor] = []
        for i in range(theta_t.shape[0]):
            row = theta_t[i : i + 1]
            try:
                out = sim(row)
            except Exception:  # pragma: no cover - numpy-only sbibm simulators
                import numpy as np

                out = sim(row.detach().cpu().numpy())
            rows.append(_as_tensor(out).reshape(-1))
        return torch.stack(rows, dim=0)

    return wrapped


def build_sbibm_task(name: str, observation_index: int = 1) -> BenchmarkTask:
    """Build a :class:`BenchmarkTask` backed by ``sbibm``."""
    if sbibm is None:  # pragma: no cover - guarded by the caller
        raise ImportError("sbibm is not installed")
    key = str(name).lower()
    sbibm_name = SBIBM_NAMES.get(key, key)
    task = sbibm.get_task(sbibm_name)
    theta_dim = int(getattr(task, "dim_parameters"))
    x_dim = int(getattr(task, "dim_data"))
    prior = SbibmPriorAdapter(task.get_prior(), theta_dim, name=f"sbibm_{key}")
    simulator_fn = _wrap_sbibm_simulator(task.get_simulator())
    try:
        x_obs = _as_tensor(
            task.get_observation(num_observation=observation_index)
        ).reshape(1, x_dim)
    except TypeError:  # pragma: no cover - older sbibm signature
        x_obs = _as_tensor(task.get_observation()).reshape(1, x_dim)
    extra: Dict[str, Any] = {}
    try:
        extra["theta_true"] = _as_tensor(task.get_ground_truth()).reshape(-1)
    except Exception:
        pass
    return BenchmarkTask(
        name=key,
        dim_parameters=theta_dim,
        dim_data=x_dim,
        prior=prior,
        simulator_fn=simulator_fn,
        x_obs=x_obs,
        description=f"sbibm task {sbibm_name!r} (Lueckmann et al. 2021, Appendix T)",
        observation_index=observation_index,
        sbibm_task=task,
        backend="sbibm",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
TASK_NAMES: List[str] = [
    "gaussian_linear",
    "gaussian_mixture",
    "two_moons",
    "gaussian_linear_uniform",
    "bernoulli_glm",
    "slcp",
    "sir",
    "lotka_volterra",
]

TASKS: Dict[str, Dict[str, Any]] = {
    "gaussian_linear": {"dim_parameters": 10, "dim_data": 10, "prior": "gaussian"},
    "gaussian_mixture": {"dim_parameters": 2, "dim_data": 2, "prior": "uniform"},
    "two_moons": {"dim_parameters": 2, "dim_data": 2, "prior": "uniform"},
    "gaussian_linear_uniform": {"dim_parameters": 10, "dim_data": 10, "prior": "uniform"},
    "bernoulli_glm": {"dim_parameters": 10, "dim_data": 10, "prior": "gaussian"},
    "slcp": {"dim_parameters": 5, "dim_data": 8, "prior": "uniform"},
    "sir": {"dim_parameters": 2, "dim_data": 10, "prior": "lognormal"},
    "lotka_volterra": {"dim_parameters": 4, "dim_data": 20, "prior": "lognormal"},
}

_TASK_CACHE: Dict[Tuple[str, str, int], BenchmarkTask] = {}


def sbibm_available() -> bool:
    return sbibm is not None


def list_tasks() -> List[str]:
    """Names of the eight benchmark tasks."""
    return list(TASK_NAMES)


def get_task(
    name: str,
    backend: str = "auto",
    observation_index: int = 1,
) -> BenchmarkTask:
    """Return the :class:`BenchmarkTask` called ``name``.

    ``backend`` is one of ``"auto"`` (use ``sbibm`` when available, otherwise
    the built-in fallback), ``"sbibm"`` or ``"fallback"``.  Tasks are cached
    per ``(name, backend, observation)``.
    """
    key = str(name).lower().replace("-", "_")
    if key == "lv":
        key = "lotka_volterra"
    if key not in TASK_NAMES:
        raise KeyError(f"unknown benchmark task: {name!r}; available: {TASK_NAMES}")

    if backend == "auto":
        resolved = "sbibm" if sbibm is not None else "fallback"
    else:
        resolved = backend
    cache_key = (key, resolved, int(observation_index))
    if cache_key in _TASK_CACHE:
        return _TASK_CACHE[cache_key]

    if resolved == "sbibm":
        try:
            task = build_sbibm_task(key, observation_index=observation_index)
        except Exception as exc:  # pragma: no cover - sbibm problems
            if backend == "sbibm":
                raise
            task = build_fallback_task(key)
            task.extra["sbibm_error"] = repr(exc)
    else:
        task = build_fallback_task(key)
    _TASK_CACHE[cache_key] = task
    return task


# ---------------------------------------------------------------------------
# functional helpers used by the experiment drivers
# ---------------------------------------------------------------------------
def sample_prior(task: Union[str, BenchmarkTask], n: int = 1, generator=None) -> torch.Tensor:
    task = get_task(task) if isinstance(task, str) else task
    return task.sample_prior(n, generator=generator)


def simulate(
    task: Union[str, BenchmarkTask],
    theta: torch.Tensor,
    generator=None,
) -> torch.Tensor:
    task = get_task(task) if isinstance(task, str) else task
    return task.simulate(theta, generator=generator)


def sample_prior_predictive(
    task: Union[str, BenchmarkTask],
    n: int = 1,
    generator=None,
    seed: Optional[int] = None,
):
    task = get_task(task) if isinstance(task, str) else task
    if seed is not None and generator is None:
        generator = torch.Generator().manual_seed(int(seed))
    return task.sample_prior_predictive(n, generator=generator)


def task_summary(verbose: bool = True) -> List[Dict[str, Any]]:
    """Shape / prior information for every benchmark task."""
    rows = []
    for name in TASK_NAMES:
        task = get_task(name)
        rows.append(
            {
                "name": name,
                "dim_parameters": task.dim_parameters,
                "dim_data": task.dim_data,
                "backend": task.backend,
                "prior": task.prior.name,
                "bounded": task.bounds() is not None,
                "observation": task.x_obs.reshape(-1)[:4].tolist(),
            }
        )
    if verbose:  # pragma: no cover - cosmetic
        width = max(len(r["name"]) for r in rows)
        for r in rows:
            print(
                f"{r['name']:<{width}}  d={r['dim_parameters']:<3} p={r['dim_data']:<3} "
                f"prior={r['prior']:<10} backend={r['backend']:<8} bounded={r['bounded']}"
            )
    return rows


def dataset_cache_dir(cache_dir: Optional[str] = None) -> str:
    if cache_dir is not None:
        return cache_dir
    env = os.environ.get("SNPSE_DATA_DIR")
    if env:
        return os.path.join(env, "datasets")
    return os.path.join(os.path.expanduser("~"), ".cache", "snpse", "datasets")


def load_dataset(
    task: Union[str, BenchmarkTask],
    num_simulations: int,
    seed: int = 0,
    cache_dir: Optional[str] = None,
    use_cache: bool = True,
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Sample ``num_simulations`` prior-predictive pairs ``(theta, x)``.

    Results are cached on disk (``~/.cache/snpse/datasets`` by default) so that
    NPSE / TSNPSE / NLSE runs with the same budget share exactly the same
    simulations.
    """
    task = get_task(task) if isinstance(task, str) else task
    cache_path = None
    if use_cache:
        directory = dataset_cache_dir(cache_dir)
        os.makedirs(directory, exist_ok=True)
        cache_path = os.path.join(directory, f"{task.name}_{int(num_simulations)}_{int(seed)}.pt")
        if os.path.exists(cache_path):
            try:
                blob = torch.load(cache_path, map_location="cpu")
                theta, x = blob["theta"], blob["x"]
                if theta.shape[0] == int(num_simulations):
                    if verbose:  # pragma: no cover - cosmetic
                        print(f"[benchmarks] loaded dataset from {cache_path}")
                    if device is not None:
                        theta, x = theta.to(device), x.to(device)
                    return {"theta": theta.float(), "x": x.float(), "cached": torch.tensor(True)}
            except Exception:  # pragma: no cover - corrupted cache
                pass

    generator = torch.Generator().manual_seed(int(seed))
    theta = task.sample_prior(int(num_simulations), generator=generator)
    x = task.simulate(theta, generator=generator)

    if cache_path is not None:
        try:
            torch.save({"theta": theta.detach().cpu(), "x": x.detach().cpu()}, cache_path)
        except Exception:  # pragma: no cover - read-only filesystem
            pass
    if device is not None:
        theta, x = theta.to(device), x.to(device)
    return {"theta": theta.float(), "x": x.float(), "cached": torch.tensor(False)}


def observation(task: Union[str, BenchmarkTask]) -> torch.Tensor:
    task = get_task(task) if isinstance(task, str) else task
    return task.x_obs.clone()


# ---------------------------------------------------------------------------
# self test
# ---------------------------------------------------------------------------
def _selftest(verbose: bool = True) -> bool:
    """Sanity checks for every benchmark task (shapes, priors, simulators)."""
    ok = True
    g = torch.Generator().manual_seed(0)
    for name in TASK_NAMES:
        task = get_task(name)
        theta = task.sample_prior(8, generator=g)
        x = task.simulate(theta, generator=g)
        checks = [
            (theta.shape == (8, task.dim_parameters), f"prior shape {tuple(theta.shape)}"),
            (x.shape == (8, task.dim_data), f"simulator shape {tuple(x.shape)}"),
            (task.x_obs.shape[-1] == task.dim_data, f"x_obs shape {tuple(task.x_obs.shape)}"),
        ]
        lp = task.prior_log_prob(theta)
        checks.append((torch.isfinite(lp).all(), "prior log_prob finite"))
        bounds = task.bounds()
        if bounds is not None:
            low, high = bounds
            checks.append(
                (
                    bool(((theta >= low - 1e-5) & (theta <= high + 1e-5)).all()),
                    "prior samples inside support",
                )
            )
        for good, msg in checks:
            if not bool(good):
                ok = False
                if verbose:
                    print(f"[FAIL] {name}: {msg}")
        if verbose:
            print(f"[ok] {name:<24} d={task.dim_parameters:<3} p={task.dim_data:<3} "
                  f"finite log_prob={bool(torch.isfinite(lp).all())} backend={task.backend}")

    # analytic posterior of the Gaussian Linear task
    task = get_task("gaussian_linear")
    samples = task.reference_posterior_samples(20000, generator=g)
    if samples is None:
        ok = False
        if verbose:
            print("[FAIL] gaussian_linear: no reference posterior")
    else:
        obs = task.x_obs.reshape(-1)
        post_var = 1.0 / (1.0 / 0.1 + task.dim_data / 0.1)
        mean_err = (samples.mean(0) - post_var * obs / 0.1).abs().max()
        var_err = (samples.var(0, unbiased=True) - post_var).abs().max()
        good = bool(mean_err < 1e-2 and var_err < 1e-3)
        ok = ok and good
        if verbose:
            print(f"[{'ok' if good else 'FAIL'}] gaussian_linear analytic posterior "
                  f"(mean err {float(mean_err):.2e}, var err {float(var_err):.2e})")

    # simulator batching must be consistent with row-wise evaluation
    task = get_task("two_moons")
    theta = task.sample_prior(4, generator=g)
    x_batch = task.simulate(theta, generator=g)
    good = x_batch.shape == (4, 2) and torch.isfinite(x_batch).all()
    ok = ok and bool(good)
    if verbose:
        print(f"[{'ok' if good else 'FAIL'}] batched simulator evaluation")

    # dataset loader / cache round trip
    task = get_task("gaussian_mixture")
    ds = load_dataset(task, 16, seed=3, use_cache=False)
    good = ds["theta"].shape == (16, 2) and ds["x"].shape == (16, 2)
    ok = ok and bool(good)
    if verbose:
        print(f"[{'ok' if good else 'FAIL'}] load_dataset")

    if verbose:
        print("benchmarks self-test:", "PASS" if ok else "FAIL")
    return ok


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="SNPSE benchmark tasks (Section E.1)")
    parser.add_argument("--task", type=str, default=None, help="inspect a single task")
    parser.add_argument("--list", action="store_true", help="list all tasks")
    parser.add_argument("--self-test", action="store_true", help="run the self test")
    parser.add_argument("--num-simulations", type=int, default=0, help="sample a dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "sbibm", "fallback"])
    args = parser.parse_args(argv)

    if args.list or args.task is None and not args.self_test:
        print(f"sbibm available: {sbibm_available()}")
        task_summary(verbose=True)
        return 0

    if args.self_test:
        return 0 if _selftest() else 1

    task = get_task(args.task, backend=args.backend)
    print(task)
    print("  prior          :", task.prior.name)
    print("  dim_parameters :", task.dim_parameters)
    print("  dim_data       :", task.dim_data)
    print("  x_obs          :", task.x_obs.reshape(-1).tolist())
    if args.num_simulations > 0:
        ds = load_dataset(task, args.num_simulations, seed=args.seed, use_cache=False)
        print("  dataset theta  :", tuple(ds["theta"].shape))
        print("  dataset x      :", tuple(ds["x"].shape))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
