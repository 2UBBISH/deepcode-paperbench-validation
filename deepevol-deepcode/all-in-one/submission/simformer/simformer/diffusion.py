"""Stochastic differential equations (SDEs) used by Simformer.

Implements the two SDEs proposed by Song et al. (2021b) and used in all of the
paper's experiments (Appendix A2.1):

    VESDE:  f(x, t) = 0,
            g(t) = sigma_min * (sigma_max / sigma_min)^t
                   * sqrt(2 * log(sigma_max / sigma_min))

    VPSDE:  f(x, t) = -0.5 * (beta_min + t * (beta_max - beta_min)) * x,
            g(t) = sqrt(beta_min + t * (beta_max - beta_min))

with ``sigma_max = 15``, ``sigma_min = 1e-4``, ``beta_min = 0.01``,
``beta_max = 10`` and the time interval ``[1e-5, 1]`` (Appendix A2.1).

The reverse-time SDE of Anderson (1982)

    d x_t = [f(x_t, t) - g(t)^2 * s(x_t, t)] dt + g(t) d w~

is solved with an Euler-Maruyama discretization (500 steps by default, the
paper notes that ``>~50`` steps already give near-optimal performance,
Appendix A3.1 / Fig. A7).  The probability-flow ODE (Song et al., 2021b) is
provided for log-likelihood evaluation (Appendix A3.1).

The module is backend agnostic: the coefficients are evaluated with whichever
of NumPy / PyTorch the input arrays belong to, so the same SDE objects can be
used inside the (PyTorch) neural network training loop and inside the (NumPy)
MCMC / evaluation code.

Note on the drift of the VPSDE: the paper writes the drift coefficient without
making the ``x`` dependence explicit; the standard (and only sensible) reading
is ``f(x, t) = -0.5 * beta(t) * x``, which is what is implemented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # constants
    "DEFAULT_SIGMA_MAX",
    "DEFAULT_SIGMA_MIN",
    "DEFAULT_BETA_MIN",
    "DEFAULT_BETA_MAX",
    "DEFAULT_T_MIN",
    "DEFAULT_T_MAX",
    "DEFAULT_STEPS",
    "MIN_RECOMMENDED_STEPS",
    # config
    "SDEConfig",
    "DEFAULT_VESDE_CONFIG",
    "DEFAULT_VPSDE_CONFIG",
    # sdes
    "SDE",
    "VarianceExplodingSDE",
    "VariancePreservingSDE",
    "VESDE",
    "VPSDE",
    "get_sde",
    "sde_from_config",
    # helpers
    "time_grid",
    "euler_maruyama_step",
    "reverse_sde_step",
    "sample_reverse_sde",
    "probability_flow_drift",
    "probability_flow_ode",
    "log_likelihood_from_ode",
    "divergence",
    "exact_divergence",
    "hutchinson_divergence",
    "score_scaling_inverse_variance",
]

# ---------------------------------------------------------------------------
# Constants of the paper (Appendix A2.1)
# ---------------------------------------------------------------------------
DEFAULT_SIGMA_MAX = 15.0
DEFAULT_SIGMA_MIN = 1e-4
DEFAULT_BETA_MIN = 0.01
DEFAULT_BETA_MAX = 10.0
DEFAULT_T_MIN = 1e-5
DEFAULT_T_MAX = 1.0
DEFAULT_STEPS = 500
MIN_RECOMMENDED_STEPS = 50


# ---------------------------------------------------------------------------
# backend helpers
# ---------------------------------------------------------------------------
def _is_torch(x: Any) -> bool:
    """Cheap, import-free check whether ``x`` is a torch tensor."""
    return type(x).__module__.split(".")[0] == "torch"


def _xp_for(*args: Any):
    """Return the array library matching the first torch argument (else NumPy)."""
    for a in args:
        if _is_torch(a):
            import torch  # local import: torch is optional at import time

            return torch
    return np


def _zeros_like(x: Any) -> Any:
    xp = _xp_for(x)
    if xp is np:
        return np.zeros_like(np.asarray(x, dtype=np.float64))
    return xp.zeros_like(x)


def _ones_like(t: Any) -> Any:
    xp = _xp_for(t)
    if isinstance(t, (int, float)):
        return 1.0
    return xp.ones_like(t)


# ---------------------------------------------------------------------------
# SDE configuration
# ---------------------------------------------------------------------------
@dataclass
class SDEConfig:
    """Configuration of a diffusion SDE (paper Appendix A2.1)."""

    name: str = "vesde"
    sigma_max: float = DEFAULT_SIGMA_MAX
    sigma_min: float = DEFAULT_SIGMA_MIN
    beta_min: float = DEFAULT_BETA_MIN
    beta_max: float = DEFAULT_BETA_MAX
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    n_steps: int = DEFAULT_STEPS
    extra: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> "SDE":
        """Instantiate the corresponding :class:`SDE` object."""
        return sde_from_config(self)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(
            name=self.name,
            sigma_max=self.sigma_max,
            sigma_min=self.sigma_min,
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            t_min=self.t_min,
            t_max=self.t_max,
            n_steps=self.n_steps,
        )
        d.update(self.extra)
        return d


DEFAULT_VESDE_CONFIG = SDEConfig(
    name="vesde",
    sigma_max=DEFAULT_SIGMA_MAX,
    sigma_min=DEFAULT_SIGMA_MIN,
    t_min=DEFAULT_T_MIN,
    t_max=DEFAULT_T_MAX,
    n_steps=DEFAULT_STEPS,
)

DEFAULT_VPSDE_CONFIG = SDEConfig(
    name="vpsde",
    beta_min=DEFAULT_BETA_MIN,
    beta_max=DEFAULT_BETA_MAX,
    t_min=DEFAULT_T_MIN,
    t_max=DEFAULT_T_MAX,
    n_steps=DEFAULT_STEPS,
)


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class SDE:
    """Base class for the diffusion SDEs used by Simformer.

    Subclasses implement the drift ``f(x, t)``, the diffusion coefficient
    ``g(t)`` and the marginal transition kernel
    ``p_t(x_t | x_0) = N(x_t; mu(t) x_0, sigma(t)^2 I)``.
    """

    name: str = "sde"

    def __init__(
        self,
        name: Optional[str] = None,
        t_min: float = DEFAULT_T_MIN,
        t_max: float = DEFAULT_T_MAX,
        n_steps: int = DEFAULT_STEPS,
    ) -> None:
        if name is not None:
            self.name = name
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.n_steps = int(n_steps)
        if not (0.0 <= self.t_min < self.t_max):
            raise ValueError(f"expected 0 <= t_min < t_max, got {t_min}, {t_max}")

    # -- coefficients ------------------------------------------------------
    def drift(self, x: Any, t: Any) -> Any:
        """Drift coefficient ``f(x, t)`` (same shape as ``x``)."""
        raise NotImplementedError

    def diffusion(self, t: Any) -> Any:
        """Diffusion coefficient ``g(t)`` (scalar or array shaped like ``t``)."""
        raise NotImplementedError

    def drift_divergence(self, x: Any, t: Any) -> Any:
        """Trace of the Jacobian of the drift w.r.t. ``x`` (vector valued).

        Needed for the log-density ODE of the probability flow.  For this
        module's drift it is a constant vector along the data dimensions: ``0``
        for the VESDE and ``-0.5 * beta(t)`` per dimension for the VPSDE.
        """
        xp = _xp_for(x)
        if xp is np:
            x = np.asarray(x, dtype=np.float64)
        d = x.shape[-1]
        zeros = xp.zeros(()) if False else 0.0
        return zeros, d

    # -- marginal transition kernel ---------------------------------------
    def marginal_mean(self, t: Any) -> Any:
        """Mean multiplier ``mu(t)`` of ``p_t(x_t | x_0)``."""
        raise NotImplementedError

    def marginal_std(self, t: Any) -> Any:
        """Standard deviation ``sigma(t)`` of ``p_t(x_t | x_0)``."""
        raise NotImplementedError

    def marginal_mean_std(self, t: Any) -> Tuple[Any, Any]:
        return self.marginal_mean(t), self.marginal_std(t)

    def prior_mean_std(self) -> Tuple[float, float]:
        """Parameters of the terminal noise distribution ``p_T`` (t = t_max)."""
        mu = self.marginal_mean(self.t_max)
        sigma = self.marginal_std(self.t_max)
        return float(np.asarray(mu)), float(np.asarray(sigma))

    def sample_marginal(self, x0: Any, t: Any, noise: Any = None, rng=None) -> Any:
        """Sample ``x_t ~ N(mu(t) x_0, sigma(t)^2 I)``."""
        mu, sigma = self.marginal_mean_std(t)
        if noise is None:
            noise = _randn_like(x0, rng)
        return _scale(mu, x0) + _scale(sigma, noise)

    def sample_prior(self, shape: Sequence[int], rng=None, device: Any = None, dtype: Any = None) -> Any:
        """Sample from the terminal distribution used to initialise the reverse SDE."""
        mu, sigma = self.prior_mean_std()
        noise = _randn(shape, rng=rng, device=device, dtype=dtype)
        return mu + sigma * noise

    # -- score bookkeeping -------------------------------------------------
    def score_target(self, x_t: Any, x0: Any, t: Any) -> Any:
        """``grad_{x_t} log p_t(x_t | x_0) = -(x_t - mu(t) x_0) / sigma(t)^2``."""
        mu, sigma = self.marginal_mean_std(t)
        return -_scale(_inv_square(sigma), x_t - _scale(mu, x0))

    def epsilon_target(self, x_t: Any, x0: Any, t: Any) -> Any:
        """The noise ``eps`` that generated ``x_t`` from ``x_0``."""
        mu, sigma = self.marginal_mean_std(t)
        return _scale(_inv(sigma), x_t - _scale(mu, x0))

    def twedie_denoise(self, x_t: Any, t: Any, score: Any) -> Any:
        """One-step denoised estimate ``(x_t + sigma(t)^2 s) / mu(t)`` (Algorithm 1)."""
        mu, sigma = self.marginal_mean_std(t)
        return _scale(_inv(mu), x_t + _scale(_square(sigma), score))

    # alias matching the paper's notation
    denoise_from_score = twedie_denoise

    def score_scaling(self, t: Any) -> Any:
        """Scaling function used for guidance ``s(t) = 1 / sigma(t)^2`` (Appendix A3.3)."""
        _, sigma = self.marginal_mean_std(t)
        return _inv_square(sigma)

    # -- time grid ---------------------------------------------------------
    def time_grid(
        self,
        n_steps: Optional[int] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        descending: bool = True,
    ) -> np.ndarray:
        """Uniform discretization of ``[t_min, t_max]`` (Appendix A3.1)."""
        n = int(n_steps if n_steps is not None else self.n_steps)
        lo = self.t_min if t_min is None else float(t_min)
        hi = self.t_max if t_max is None else float(t_max)
        grid = np.linspace(hi, lo, n + 1) if descending else np.linspace(lo, hi, n + 1)
        return grid

    # -- repr --------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.__class__.__name__}(name={self.name!r}, t_min={self.t_min}, "
            f"t_max={self.t_max}, n_steps={self.n_steps})"
        )


# ---------------------------------------------------------------------------
# VESDE
# ---------------------------------------------------------------------------
class VarianceExplodingSDE(SDE):
    """Variance Exploding SDE of Song et al. (2021b) (Appendix A2.1)."""

    def __init__(
        self,
        sigma_max: float = DEFAULT_SIGMA_MAX,
        sigma_min: float = DEFAULT_SIGMA_MIN,
        t_min: float = DEFAULT_T_MIN,
        t_max: float = DEFAULT_T_MAX,
        n_steps: int = DEFAULT_STEPS,
        name: str = "vesde",
    ) -> None:
        super().__init__(name=name, t_min=t_min, t_max=t_max, n_steps=n_steps)
        if sigma_max <= sigma_min:
            raise ValueError("require sigma_max > sigma_min")
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)
        # g(t) = sigma(t) * sqrt(2 log(sigma_max / sigma_min))
        self.g_scale = math.sqrt(2.0 * self.log_ratio)

    # -- coefficients ------------------------------------------------------
    def drift(self, x: Any, t: Any) -> Any:
        """``f_VESDE(x, t) = 0``."""
        if isinstance(x, (int, float)):
            return 0.0
        return _zeros_like(x)

    def diffusion(self, t: Any) -> Any:
        """``g_VESDE(t) = sigma(t) sqrt(2 log(sigma_max / sigma_min))``."""
        return self.marginal_std(t) * self.g_scale

    def drift_divergence(self, x: Any, t: Any) -> Tuple[float, int]:
        xp = _xp_for(x)
        d = x.shape[-1] if hasattr(x, "shape") and x.ndim >= 1 else 1
        return 0.0, int(d)

    # -- marginals ---------------------------------------------------------
    def marginal_mean(self, t: Any) -> Any:
        """``mu_VESDE(t) = 1``."""
        return _ones_like(t)

    def marginal_std(self, t: Any) -> Any:
        """``sigma(t) = sigma_min (sigma_max / sigma_min)^t``."""
        xp = _xp_for(t)
        return self.sigma_min * xp.exp(self.log_ratio * t)


# ---------------------------------------------------------------------------
# VPSDE
# ---------------------------------------------------------------------------
class VariancePreservingSDE(SDE):
    """Variance Preserving SDE of Song et al. (2021b) (Appendix A2.1)."""

    def __init__(
        self,
        beta_min: float = DEFAULT_BETA_MIN,
        beta_max: float = DEFAULT_BETA_MAX,
        t_min: float = DEFAULT_T_MIN,
        t_max: float = DEFAULT_T_MAX,
        n_steps: int = DEFAULT_STEPS,
        name: str = "vpsde",
    ) -> None:
        super().__init__(name=name, t_min=t_min, t_max=t_max, n_steps=n_steps)
        if beta_max <= beta_min:
            raise ValueError("require beta_max > beta_min")
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.beta_delta = self.beta_max - self.beta_min

    # -- coefficients ------------------------------------------------------
    def beta(self, t: Any) -> Any:
        """``beta(t) = beta_min + t (beta_max - beta_min)``."""
        xp = _xp_for(t)
        return self.beta_min + xp.asarray(t) * self.beta_delta if not isinstance(t, (int, float)) else (
            self.beta_min + t * self.beta_delta
        )

    def drift(self, x: Any, t: Any) -> Any:
        """``f_VPSDE(x, t) = -0.5 beta(t) x``."""
        return _scale(-0.5 * _as_float_or_array(self.beta(t), x), x)

    def diffusion(self, t: Any) -> Any:
        """``g_VPSDE(t) = sqrt(beta(t))``."""
        xp = _xp_for(t)
        return xp.sqrt(self.beta(t))

    def drift_divergence(self, x: Any, t: Any) -> Tuple[Any, int]:
        d = x.shape[-1] if hasattr(x, "shape") and x.ndim >= 1 else 1
        return -0.5 * _as_scalar(self.beta(t)), int(d)

    # -- marginals ---------------------------------------------------------
    def marginal_mean(self, t: Any) -> Any:
        """``mu(t) = exp(-0.25 t^2 (beta_max-beta_min) - 0.5 t beta_min)``."""
        xp = _xp_for(t)
        return xp.exp(-0.25 * t * t * self.beta_delta - 0.5 * t * self.beta_min)

    def marginal_std(self, t: Any) -> Any:
        """``sigma(t) = sqrt(1 - mu(t)^2)``."""
        xp = _xp_for(t)
        mu = self.marginal_mean(t)
        return xp.sqrt(xp.clip(1.0 - mu * mu, a_min=0.0, a_max=None))


# aliases used across the code base
VESDE = VarianceExplodingSDE
VPSDE = VariancePreservingSDE


def sde_from_config(config: Union[SDEConfig, Dict[str, Any], str]) -> SDE:
    """Build an :class:`SDE` from a config object, dict or name."""
    if isinstance(config, str):
        return get_sde(config)
    if isinstance(config, dict):
        config = SDEConfig(**{k: v for k, v in config.items() if k in SDEConfig.__dataclass_fields__})
    if not isinstance(config, SDEConfig):
        raise TypeError(f"cannot build SDE from {type(config)}")
    kwargs = dict(t_min=config.t_min, t_max=config.t_max, n_steps=config.n_steps)
    name = str(config.name).lower().replace("_", "").replace("-", "")
    if name in ("vesde", "varianceexploding", "exploding", "ve"):
        return VarianceExplodingSDE(
            sigma_max=config.sigma_max, sigma_min=config.sigma_min, **kwargs
        )
    if name in ("vpsde", "variancepreserving", "preserving", "vp"):
        return VariancePreservingSDE(
            beta_min=config.beta_min, beta_max=config.beta_max, **kwargs
        )
    raise ValueError(f"unknown SDE '{config.name}'")


def get_sde(name: Union[str, SDEConfig] = "vesde", **kwargs: Any) -> SDE:
    """Return one of the paper's SDEs by name (``'vesde'`` / ``'vpsde'``)."""
    if isinstance(name, SDEConfig):
        return sde_from_config(name)
    key = str(name).lower().replace("_", "").replace("-", "")
    if key in ("vesde", "varianceexploding", "exploding", "ve"):
        return VarianceExplodingSDE(**kwargs)
    if key in ("vpsde", "variancepreserving", "preserving", "vp"):
        return VariancePreservingSDE(**kwargs)
    raise ValueError(f"unknown SDE '{name}'")


# ---------------------------------------------------------------------------
# small shape helpers (work for both torch and numpy)
# ---------------------------------------------------------------------------
def _randn(shape: Sequence[int], rng=None, device: Any = None, dtype: Any = None) -> Any:
    if device is not None or dtype is not None or _is_torch(device) or _is_torch(dtype):
        import torch

        return torch.randn(*tuple(int(s) for s in shape), generator=None, device=device, dtype=dtype)
    if rng is None:
        return np.random.randn(*tuple(int(s) for s in shape))
    return rng.standard_normal(tuple(int(s) for s in shape))


def _randn_like(x: Any, rng=None) -> Any:
    if _is_torch(x):
        import torch

        return torch.randn_like(x)
    if rng is None:
        return np.random.randn(*np.shape(x))
    return rng.standard_normal(np.shape(x))


def _as_scalar(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(np.asarray(v).reshape(-1)[0])


def _as_float_or_array(v: Any, like: Any) -> Any:
    """Return ``v`` shaped so that it broadcasts against the trailing axis of ``like``."""
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(like, "ndim") and like.ndim >= 1:
        if getattr(v, "ndim", 0) == 1 and getattr(like, "ndim", 1) > 1:
            xp = _xp_for(v)
            return v.reshape((-1,) + (1,) * (like.ndim - 1))
    return v


def _scale(coef: Any, x: Any) -> Any:
    """Multiply ``x`` by a coefficient that may be scalar, ``(B,)`` or ``(B,1,...)``."""
    return _as_float_or_array(coef, x) * x


def _inv(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return 1.0 / v
    xp = _xp_for(v)
    return 1.0 / v


def _inv_square(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return 1.0 / (v * v)
    return 1.0 / (v * v)


def _square(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return v * v
    return v * v


# ---------------------------------------------------------------------------
# time discretization / Euler-Maruyama
# ---------------------------------------------------------------------------
def time_grid(
    n_steps: int = DEFAULT_STEPS,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    descending: bool = True,
) -> np.ndarray:
    """Uniform discretization of the diffusion interval.

    ``descending=True`` returns ``[t_max, ..., t_min]`` (the order in which the
    reverse SDE is integrated, cf. Algorithm 1).
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    grid = np.linspace(t_max, t_min, int(n_steps) + 1)
    return grid if descending else grid[::-1]


def reverse_sde_step(
    sde: SDE,
    x: Any,
    t_cur: float,
    t_next: float,
    score: Any,
    noise: Any = None,
    rng=None,
) -> Any:
    """Single Euler-Maruyama step of the reverse SDE (Eq. 2 / Algorithm 1).

    ``x_next = x - (f(x, t) - g(t)^2 s(x, t)) dt - g(t) sqrt(dt) eps`` with
    ``dt = t_cur - t_next`` (the paper's Algorithm 1 sign convention for the
    noise term).
    """
    dt = float(t_cur) - float(t_next)
    g = sde.diffusion(t_cur)
    drift = sde.drift(x, t_cur) - _scale(_square(g), score)
    x_next = x - _scale(_as_float_or_array(dt, x), drift)
    if dt > 0.0:
        if noise is None:
            noise = _randn_like(x, rng)
        x_next = x_next - _scale(_as_float_or_array(math.sqrt(dt) * _as_float_or_array(g, x), x), noise)
    return x_next


def euler_maruyama_step(
    sde: SDE,
    x: Any,
    t_cur: float,
    t_next: float,
    score: Any,
    noise: Any = None,
    rng=None,
) -> Any:
    """Alias of :func:`reverse_sde_step` (reverse-time Euler-Maruyama)."""
    return reverse_sde_step(sde, x, t_cur, t_next, score, noise=noise, rng=rng)


def sample_reverse_sde(
    score_fn: Callable[[Any, float], Any],
    sde: SDE,
    shape: Sequence[int],
    n_steps: Optional[int] = None,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    condition_mask: Any = None,
    condition_values: Any = None,
    x_init: Any = None,
    rng=None,
    device: Any = None,
    dtype: Any = None,
    return_trajectory: bool = False,
    return_times: bool = False,
):
    """Sample from the reverse SDE (Anderson, 1982) with Euler-Maruyama.

    Parameters
    ----------
    score_fn:
        Callable ``(x, t) -> score`` with ``t`` a Python float; it must return an
        array of the same shape as ``x``.
    sde:
        The SDE object providing ``f``, ``g`` and the marginal coefficients.
    shape:
        Shape of the sample batch, e.g. ``(B, d)``.
    condition_mask / condition_values:
        Optional per-dimension conditioning.  Entries with ``condition_mask=1``
        are clamped to ``condition_values`` after every step; the reverse
        diffusion is only "run" on the unobserved variables -- this implements
        Sec. 3.3 ("run the reverse diffusion process on all unobserved
        variables, while keeping observed variables constant at their
        conditioning value").
    """
    n_steps = int(n_steps if n_steps is not None else sde.n_steps)
    lo = sde.t_min if t_min is None else float(t_min)
    hi = sde.t_max if t_max is None else float(t_max)
    grid = time_grid(n_steps, lo, hi, descending=True)

    if x_init is None:
        x = sde.sample_prior(shape, rng=rng, device=device, dtype=dtype)
    else:
        x = x_init

    mask = None
    if condition_mask is not None:
        mask = _mask_to_array(condition_mask, x)
        if condition_values is not None:
            vals = condition_values
        else:
            vals = None
        x = _apply_condition(x, vals, mask)

    trajectory: List[Any] = [] if return_trajectory else None
    if return_trajectory:
        trajectory.append(_copy(x))

    for i in range(n_steps):
        t_cur, t_next = float(grid[i]), float(grid[i + 1])
        score = score_fn(x, t_cur)
        x = reverse_sde_step(sde, x, t_cur, t_next, score, rng=rng)
        if mask is not None:
            x = _apply_condition(x, condition_values, mask)
        if return_trajectory:
            trajectory.append(_copy(x))

    if return_trajectory and return_times:
        return x, trajectory, grid
    if return_trajectory:
        return x, trajectory
    if return_times:
        return x, grid
    return x


def _copy(x: Any) -> Any:
    return x.clone() if _is_torch(x) else np.array(x, copy=True)


def _mask_to_array(mask: Any, like: Any) -> Any:
    """Broadcast a condition mask to ``like``'s trailing dimension (float 0/1)."""
    if _is_torch(mask):
        m = mask.to(dtype=like.dtype) if _is_torch(like) else mask.detach().cpu().numpy()
    else:
        m = np.asarray(mask, dtype=np.float64)
    if m.ndim == 1:
        m = m.reshape((1, -1))
    if hasattr(like, "ndim") and m.ndim < like.ndim:
        m = m.reshape(m.shape + (1,) * (like.ndim - m.ndim))
    if _is_torch(like) and not _is_torch(m):
        import torch

        m = torch.as_tensor(m, dtype=like.dtype, device=like.device)
    elif not _is_torch(like) and _is_torch(m):
        m = m.detach().cpu().numpy()
    return m


def _apply_condition(x: Any, values: Any, mask: Any) -> Any:
    """``(1 - M_C) x + M_C * values`` (Sec. 3.3)."""
    if values is None:
        # keep the current value where the mask is 1 (no-op)
        return x
    vals = values
    if _is_torch(mask) and not _is_torch(vals):
        import torch

        vals = torch.as_tensor(vals, dtype=x.dtype, device=x.device)
    if not _is_torch(mask) and _is_torch(vals):
        vals = vals.detach().cpu().numpy()
    return (1.0 - mask) * x + mask * vals


# ---------------------------------------------------------------------------
# probability-flow ODE (log-likelihood, Appendix A3.1)
# ---------------------------------------------------------------------------
def probability_flow_drift(sde: SDE, x: Any, t: Any, score: Any) -> Any:
    """Drift of the probability-flow ODE ``f(x, t) - 0.5 g(t)^2 s(x, t)``."""
    g = sde.diffusion(t)
    return sde.drift(x, t) - _scale(0.5 * _as_float_or_array(_square(g), x), score)


def probability_flow_ode(
    score_fn: Callable[[Any, float], Any],
    sde: SDE,
    x0: Any,
    n_steps: int = DEFAULT_STEPS,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    return_trajectory: bool = False,
):
    """Integrate the probability-flow ODE from ``t_min`` (data) to ``t_max`` (noise).

    Uses an Euler discretization of

        dx/dt = f(x, t) - 0.5 g(t)^2 s(x, t)

    which is the deterministic ODE with the same marginals as the SDE.
    """
    lo = sde.t_min if t_min is None else float(t_min)
    hi = sde.t_max if t_max is None else float(t_max)
    grid = time_grid(n_steps, lo, hi, descending=False)  # ascending: data -> noise
    x = x0
    traj = [_copy(x)] if return_trajectory else None
    for i in range(int(n_steps)):
        t_cur, t_next = float(grid[i]), float(grid[i + 1])
        dt = t_next - t_cur
        score = score_fn(x, t_cur)
        x = x + _scale(_as_float_or_array(dt, x), probability_flow_drift(sde, x, t_cur, score))
        if return_trajectory:
            traj.append(_copy(x))
    if return_trajectory:
        return x, traj
    return x


def _log_std_normal(x: Any) -> Any:
    """``log N(x; 0, I)`` summed over the last axis."""
    return -0.5 * (np.prod(np.shape(x)[1:]) if not _is_torch(x) else float(x[0].numel())) * math.log(
        2.0 * math.pi
    ) - 0.5 * _sum_last_axis(x * x)


def _sum_last_axis(x: Any) -> Any:
    if _is_torch(x):
        return x.reshape(x.shape[0], -1).sum(dim=-1)
    x = np.asarray(x)
    return x.reshape(x.shape[0], -1).sum(axis=-1)


def log_likelihood_from_ode(
    score_fn: Callable[[Any, float], Any],
    sde: SDE,
    x0: Any,
    n_steps: int = DEFAULT_STEPS,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
    divergence_fn: Optional[Callable[[Callable[[Any], Any], Any], Any]] = None,
    return_trajectory: bool = False,
):
    """Log-likelihood of ``x0`` via the instantaneous change-of-variables formula.

    Following Song et al. (2021b) / Appendix A3.1 we integrate

        d/dt log p_t(x_t) = -div(f(x_t, t)) + 0.5 g(t)^2 div(s(x_t, t))

    together with the probability-flow ODE, so that

        log p_0(x_0) = log p_T(x_T) - integral_{t_min}^{t_max} d/dt log p_t dt.

    ``divergence_fn(fn, x)`` defaults to :func:`divergence` (exact for small
    dimensions, Hutchinson estimator otherwise).
    """
    if divergence_fn is None:
        divergence_fn = divergence
    lo = sde.t_min if t_min is None else float(t_min)
    hi = sde.t_max if t_max is None else float(t_max)
    grid = time_grid(n_steps, lo, hi, descending=False)

    x = x0
    dlogp = _zeros_scalar_like(x)
    traj = [_copy(x)] if return_trajectory else None

    for i in range(int(n_steps)):
        t_cur, t_next = float(grid[i]), float(grid[i + 1])
        dt = t_next - t_cur

        def _flow(state, _t=t_cur):
            s = score_fn(state, _t)
            return probability_flow_drift(sde, state, _t, s)

        score = score_fn(x, t_cur)
        x = x + _scale(_as_float_or_array(dt, x), probability_flow_drift(sde, x, t_cur, score))

        div_f = sde.drift_divergence(x, t_cur)[0]
        div_s = divergence_fn(lambda z: score_fn(z, t_cur), x)
        g = sde.diffusion(t_cur)
        dlogp = dlogp + dt * (-div_f + 0.5 * _as_float_or_array(_square(g), x) * div_s)

        if return_trajectory:
            traj.append(_copy(x))

    # terminal log density at t_max (standard normal in the noise variables)
    mu_t, sigma_t = sde.prior_mean_std()
    if abs(mu_t) < 1e-12 and abs(sigma_t - 1.0) < 1e-12:
        log_pt = _log_std_normal(x)
    else:
        log_pt = _log_std_normal((x - mu_t) / sigma_t) - _sum_last_axis(
            _zeros_scalar_like(x) + math.log(sigma_t)
        )

    logp0 = log_pt - dlogp
    if return_trajectory:
        return logp0, traj
    return logp0


def _zeros_scalar_like(x: Any) -> Any:
    if _is_torch(x):
        import torch

        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    x = np.asarray(x)
    return np.zeros(x.shape[0], dtype=np.float64)


# ---------------------------------------------------------------------------
# divergence estimators (used by the ODE log-likelihood)
# ---------------------------------------------------------------------------
def exact_divergence(fn: Callable[[Any], Any], x: Any, chunk: int = 64) -> Any:
    """Exact divergence via the Jacobian diagonal (autograd for torch)."""
    if _is_torch(x):
        import torch

        with torch.enable_grad():
            xr = x.detach().clone().requires_grad_(True)
            out = fn(xr)
            out_sum = out.reshape(out.shape[0], -1)
            div = torch.zeros(out_sum.shape[0], dtype=x.dtype, device=x.device)
            for j in range(out_sum.shape[1]):
                grad = torch.autograd.grad(out_sum[:, j].sum(), xr, retain_graph=True)[0]
                div = div + grad.reshape(grad.shape[0], -1)[:, j]
        return div
    # numpy: central finite differences on the diagonal
    x = np.asarray(x, dtype=np.float64)
    f0 = np.asarray(fn(x))
    flat = x.reshape(x.shape[0], -1)
    fflat = f0.reshape(f0.shape[0], -1)
    div = np.zeros(x.shape[0], dtype=np.float64)
    eps = 1e-4
    for j in range(flat.shape[1]):
        xp = flat.copy()
        xp[:, j] += eps
        fp = np.asarray(fn(xp.reshape(x.shape))).reshape(f0.shape[0], -1)
        div += (fp[:, j] - fflat[:, j]) / eps
    return div


def hutchinson_divergence(
    fn: Callable[[Any], Any],
    x: Any,
    n_samples: int = 1,
    rng=None,
) -> Any:
    """Hutchinson trace estimator ``E_v[v^T J v]`` of ``div fn``."""
    n_samples = max(1, int(n_samples))
    total = None
    for _ in range(n_samples):
        if _is_torch(x):
            import torch

            with torch.enable_grad():
                xr = x.detach().clone().requires_grad_(True)
                out = fn(xr)
                v = torch.randn_like(xr)
                vjp = torch.autograd.grad((out * v).sum(), xr, retain_graph=False)[0]
                est = (vjp * v).reshape(vjp.shape[0], -1).sum(dim=-1)
        else:
            x = np.asarray(x, dtype=np.float64)
            if rng is None:
                v = np.random.randn(*x.shape)
            else:
                v = rng.standard_normal(x.shape)
            f0 = np.asarray(fn(x))
            eps = 1e-3
            fp = np.asarray(fn(x + eps * v))
            est = ((fp - f0) * v).reshape(x.shape[0], -1).sum(axis=-1) / eps
        total = est if total is None else total + est
    return total / n_samples


def divergence(
    fn: Callable[[Any], Any],
    x: Any,
    n_samples: int = 1,
    exact_max_dim: int = 32,
    rng=None,
) -> Any:
    """Divergence of ``fn`` at ``x``: exact for small dimensions, Hutchinson else."""
    dim = int(np.prod(np.shape(x)[1:]))
    if dim == 0:
        return _zeros_scalar_like(x)
    if dim <= exact_max_dim:
        return exact_divergence(fn, x)
    return hutchinson_divergence(fn, x, n_samples=n_samples, rng=rng)


def score_scaling_inverse_variance(sde: SDE, t: Any) -> Any:
    """The paper's default guidance scaling ``s(t) = 1 / sigma(t)^2`` (Appendix A3.3)."""
    return sde.score_scaling(t)
