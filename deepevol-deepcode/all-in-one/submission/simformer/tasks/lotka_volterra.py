"""Lotka-Volterra benchmark task (Simformer, Sec. 4.2 and Appendix A2.2).

The Lotka-Volterra model is the classic predator-prey model:

.. math::

    \\frac{dx}{dt} = \\alpha x - \\beta x y \\\\
    \\frac{dy}{dt} = \\delta x y - \\gamma y

with ``x`` the prey and ``y`` the predator population size and four positive
interaction/survival rates ``alpha, beta, gamma, delta``.

Paper specification (Appendix A2.2)
-----------------------------------
* Prior: *sigmoid-transformed Normal distribution, scaled to a range from one
  to three* -- i.e. ``z ~ N(0, 1)`` and ``theta = 1 + 2 * sigmoid(z)``, which is
  equivalent to ``theta = 2 + tanh(z / 2)``, so ``theta in (1, 3)``.
* Dynamics: the two coupled ODEs above, integrated from ``t = 0``.
* Observation noise: additive Gaussian with ``sigma = 0.1``.

Sec. 4.2 uses this task to demonstrate inference from *unstructured*
observations: measurements of the prey and predator population may be taken at
different (irregular) time points, and the number of observations can differ
between species.  Simformer handles this with the per-sample condition mask
``M_C``: every *potential* observation (one token per series and time point) is
represented in the joint vector and arbitrarily many of them can be conditioned
on / left latent at inference time.  Helper methods
:meth:`LotkaVolterraTask.observation_condition_mask` and
:meth:`LotkaVolterraTask.unstructured_observations` construct exactly the
scenarios of Fig. 5.

Design notes / defaults chosen where the paper is silent
--------------------------------------------------------
* Time grid: ``t in [0, 15]`` with ``n_times = 16`` uniformly spaced points
  (``times = np.linspace(0, 15, 16)``), matching the plan's ``t in [0, 15]``.
* Initial condition: ``(x(0), y(0)) = (1.0, 1.0)`` (configurable through
  ``config.initial_state``).
* The joint vector is laid out as ``[theta (4) | data (n_series * n_times)]``
  with **series-major** flattening (all prey observations, then all predator
  observations), matching ``attention_masks.ode_chain_mask(series_major=True)``
  / ``attention_masks.lotka_volterra_mask``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - package layout dependent
    from . import TaskBase
except Exception:  # pragma: no cover - standalone import fallback
    try:
        from simformer.tasks import TaskBase  # type: ignore
    except Exception:  # pragma: no cover
        class TaskBase:  # type: ignore
            """Minimal stand-in used when the task registry is unavailable."""

            name = "lotka_volterra"
            n_parameters = 4
            n_data = 32

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DEFAULT_PARAMETER_NAMES: Tuple[str, ...] = ("alpha", "beta", "gamma", "delta")
DEFAULT_SERIES_NAMES: Tuple[str, ...] = ("prey", "predator")
DEFAULT_N_PARAMETERS = 4
DEFAULT_N_SERIES = 2
DEFAULT_N_TIMES = 16
DEFAULT_T_MIN = 0.0
DEFAULT_T_MAX = 15.0
DEFAULT_OBSERVATION_SIGMA = 0.1
DEFAULT_INITIAL_STATE: Tuple[float, float] = (1.0, 1.0)
PRIOR_RANGE = (1.0, 3.0)
PRIOR_LOW, PRIOR_HIGH = PRIOR_RANGE
PRIOR_CENTER = 0.5 * (PRIOR_LOW + PRIOR_HIGH)  # == 2.0
PRIOR_HALF_WIDTH = 0.5 * (PRIOR_HIGH - PRIOR_LOW)  # == 1.0

_LOG_2PI = math.log(2.0 * math.pi)

# Fig. 5 observation scenarios: first four prey measurements placed irregularly
# in time, later complemented by nine additional predator measurements.
FIGURE5_PREY_TIMES: Tuple[float, ...] = (1.0, 4.0, 8.5, 12.0)
FIGURE5_PREDATOR_TIMES: Tuple[float, ...] = (
    0.5, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 13.5,
)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class LotkaVolterraConfig:
    """Configuration of the Lotka-Volterra simulator (Appendix A2.2)."""

    n_parameters: int = DEFAULT_N_PARAMETERS
    n_series: int = DEFAULT_N_SERIES
    n_times: int = DEFAULT_N_TIMES
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    times: Optional[Sequence[float]] = None
    observation_sigma: float = DEFAULT_OBSERVATION_SIGMA
    initial_state: Sequence[float] = DEFAULT_INITIAL_STATE
    parameter_names: Sequence[str] = DEFAULT_PARAMETER_NAMES
    series_names: Sequence[str] = DEFAULT_SERIES_NAMES
    name: str = "lotka_volterra"
    seed: int = 0
    # numerical settings of the ODE solver
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def n_data(self) -> int:
        return int(self.n_series) * int(self.n_times)

    @property
    def time_grid(self) -> np.ndarray:
        if self.times is not None:
            return np.asarray(self.times, dtype=float).reshape(-1)
        return np.linspace(float(self.t_min), float(self.t_max), int(self.n_times))

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["times"] = self.time_grid.tolist()
        d["initial_state"] = list(np.asarray(self.initial_state, dtype=float))
        d["n_data"] = self.n_data
        return d

    @classmethod
    def from_dict(
        cls, cfg: Optional[Union[Dict[str, Any], "LotkaVolterraConfig"]] = None, **kwargs: Any
    ) -> "LotkaVolterraConfig":
        if cfg is None:
            cfg = {}
        if isinstance(cfg, LotkaVolterraConfig):
            base = cfg.to_dict()
        elif isinstance(cfg, dict):
            base = dict(cfg)
        else:  # pragma: no cover - defensive
            base = vars(cfg)
        base.pop("n_data", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = dict(base.pop("extra", {}) or {})
        unknown = {k: base.pop(k) for k in list(base) if k not in known}
        extra.update(unknown)
        base.update({k: v for k, v in kwargs.items() if k in known})
        obj = cls(**base)  # type: ignore[arg-type]
        obj.extra = extra
        return obj


def _resolve_times(
    times: Optional[Sequence[float]] = None,
    n_times: Optional[int] = None,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
) -> np.ndarray:
    """Resolve an observation time grid."""

    if times is not None:
        arr = np.asarray(times, dtype=float).reshape(-1)
        if n_times is not None and int(n_times) != arr.size:
            # keep the requested number of points by interpolation-free slicing
            idx = np.linspace(0, arr.size - 1, int(n_times)).round().astype(int)
            arr = arr[idx]
        return arr
    n = int(n_times) if n_times else DEFAULT_N_TIMES
    return np.linspace(float(t_min), float(t_max), n)


# ---------------------------------------------------------------------------
# ODE right hand side / integration
# ---------------------------------------------------------------------------
def lotka_volterra_rhs(
    t: float, state: Sequence[float], theta: Sequence[float]
) -> Tuple[float, float]:
    """Right-hand side of the Lotka-Volterra ODEs (Eq. 7 of Appendix A2.2)."""

    x, y = float(state[0]), float(state[1])
    alpha, beta, gamma, delta = (float(tmp) for tmp in theta[:4])
    dx = alpha * x - beta * x * y
    dy = delta * x * y - gamma * y
    return dx, dy


def _integrate_rk4(
    theta: Sequence[float],
    times: np.ndarray,
    initial_state: Sequence[float],
    n_substeps: int = 40,
) -> np.ndarray:
    """Deterministic RK4 integrator used when :mod:`scipy` is unavailable."""

    times = np.asarray(times, dtype=float)
    order = np.argsort(times)
    traj = np.zeros((times.size, 2), dtype=float)
    state = np.asarray(initial_state, dtype=float).reshape(2).copy()
    t_cur = 0.0
    if times[order][0] > t_cur:
        # advance from t=0 to the first requested time
        pass
    for idx in range(times.size):
        i = int(order[idx])
        t_target = float(times[i])
        span = t_target - t_cur
        if span > 0:
            h = span / max(int(n_substeps), 1)
            for _ in range(max(int(n_substeps), 1)):
                k1 = np.asarray(lotka_volterra_rhs(t_cur, state, theta), dtype=float)
                k2 = np.asarray(lotka_volterra_rhs(t_cur + 0.5 * h, state + 0.5 * h * k1, theta), dtype=float)
                k3 = np.asarray(lotka_volterra_rhs(t_cur + 0.5 * h, state + 0.5 * h * k2, theta), dtype=float)
                k4 = np.asarray(lotka_volterra_rhs(t_cur + h, state + h * k3, theta), dtype=float)
                state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t_cur += h
        traj[i] = state
    return traj


def integrate_trajectory(
    theta: Sequence[float],
    times: Sequence[float],
    *,
    initial_state: Sequence[float] = DEFAULT_INITIAL_STATE,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: Optional[float] = None,
) -> np.ndarray:
    """Integrate the Lotka-Volterra ODEs, returning a ``(n_times, 2)`` array."""

    times = np.asarray(times, dtype=float).reshape(-1)
    y0 = np.asarray(initial_state, dtype=float).reshape(-1)[:2]
    if max_step is None:
        span = max(float(times.max()), 1e-9)
        max_step = max(span / 50.0, 1e-3)
    try:
        from scipy.integrate import solve_ivp  # local import (optional)

        sol = solve_ivp(
            lambda t, y: lotka_volterra_rhs(t, y, theta),
            (0.0, float(times.max())),
            y0,
            t_eval=times,
            method="RK45",
            rtol=float(rtol),
            atol=float(atol),
            max_step=float(max_step),
        )
        if sol.y.shape[1] == times.size and np.all(np.isfinite(sol.y)):
            return np.asarray(sol.y.T, dtype=float)
    except Exception:  # pragma: no cover - fall back to RK4
        pass
    return _integrate_rk4(theta, times, y0)


def flatten_trajectory(trajectory: np.ndarray) -> np.ndarray:
    """Flatten a ``(n_times, n_series)`` trajectory **series-major**."""

    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 1:
        return traj.reshape(-1)
    return np.concatenate([traj[:, s] for s in range(traj.shape[1])], axis=0)


def unflatten_trajectory(x: np.ndarray, n_series: int = 2, n_times: Optional[int] = None) -> np.ndarray:
    """Inverse of :func:`flatten_trajectory` -> ``(n_times, n_series)``."""

    x = np.asarray(x, dtype=float).reshape(-1)
    n_series = int(n_series)
    if n_times is None:
        n_times = x.size // n_series
    n_times = int(n_times)
    return np.stack([x[s * n_times : (s + 1) * n_times] for s in range(n_series)], axis=1)


# ---------------------------------------------------------------------------
# prior (sigmoid-transformed Normal scaled to [1, 3])
# ---------------------------------------------------------------------------
def prior_transform(z: np.ndarray) -> np.ndarray:
    """Map standard normal draws to ``theta in (1, 3)`` (Appendix A2.2)."""

    z = np.asarray(z, dtype=float)
    return PRIOR_LOW + (PRIOR_HIGH - PRIOR_LOW) / (1.0 + np.exp(-z))


def inverse_prior_transform(theta: np.ndarray) -> np.ndarray:
    """Map ``theta in (1, 3)`` back to the underlying standard normal scale."""

    theta = np.asarray(theta, dtype=float)
    u = np.clip((theta - PRIOR_LOW) / (PRIOR_HIGH - PRIOR_LOW), 1e-12, 1.0 - 1e-12)
    return np.log(u / (1.0 - u))


def log_prior(theta: Union[np.ndarray, float], **kwargs: Any) -> np.ndarray:
    """Log density of the sigmoid-transformed Normal prior (with Jacobian)."""

    theta = np.asarray(theta, dtype=float)
    squeeze = theta.ndim == 1
    theta2 = np.atleast_2d(theta)
    u = (theta2 - PRIOR_CENTER) / PRIOR_HALF_WIDTH  # in (-1, 1)
    inside = np.abs(u) < 1.0
    u_safe = np.clip(u, -1.0 + 1e-12, 1.0 - 1e-12)
    z = np.arctanh(u_safe) * 2.0
    log_norm = -0.5 * z ** 2 - 0.5 * _LOG_2PI
    # |d theta / d z| = 0.5 * (1 - u^2) (with u = theta - 2)
    log_jac = math.log(0.5) + np.log1p(-u_safe ** 2)
    out = (log_norm + log_jac).sum(axis=-1)
    out = np.where(np.all(inside, axis=-1), out, -np.inf)
    return out if not squeeze else out.reshape(())


def prior_sample(
    n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any
) -> np.ndarray:
    """Draw ``theta`` from the prior, shape ``(n_samples, 4)``."""

    rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(kwargs.get("seed", 0))
    n = int(n_samples)
    z = rng.standard_normal((n, DEFAULT_N_PARAMETERS))
    return prior_transform(z)


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------
class LotkaVolterraTask(TaskBase):
    """Lotka-Volterra simulator / prior / likelihood with Simformer plumbing."""

    name = "lotka_volterra"
    n_parameters = DEFAULT_N_PARAMETERS
    n_data = DEFAULT_N_SERIES * DEFAULT_N_TIMES
    parameter_names = DEFAULT_PARAMETER_NAMES
    data_names = tuple(
        f"{series}_{i}" for series in DEFAULT_SERIES_NAMES for i in range(DEFAULT_N_TIMES)
    )

    def __init__(
        self,
        config: Optional[Union[LotkaVolterraConfig, Dict[str, Any]]] = None,
        *,
        n_parameters: Optional[int] = None,
        n_series: Optional[int] = None,
        n_times: Optional[int] = None,
        times: Optional[Sequence[float]] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        observation_sigma: Optional[float] = None,
        initial_state: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        cfg = LotkaVolterraConfig.from_dict(config)
        if n_parameters is not None:
            cfg.n_parameters = int(n_parameters)
        if n_series is not None:
            cfg.n_series = int(n_series)
        if n_times is not None:
            cfg.n_times = int(n_times)
        if times is not None:
            cfg.times = times
        if t_min is not None:
            cfg.t_min = float(t_min)
        if t_max is not None:
            cfg.t_max = float(t_max)
        if observation_sigma is not None:
            cfg.observation_sigma = float(observation_sigma)
        if initial_state is not None:
            cfg.initial_state = initial_state
        if name is not None:
            cfg.name = str(name)
        if seed is not None:
            cfg.seed = int(seed)
        if kwargs:
            cfg.extra = {**dict(cfg.extra), **kwargs}
        self.config = cfg
        self.extra = dict(cfg.extra)
        self.n_series = int(cfg.n_series)
        self.n_times = int(cfg.n_times)
        self.times = _resolve_times(cfg.times, cfg.n_times, cfg.t_min, cfg.t_max)
        self.n_times = int(self.times.size)
        self.observation_sigma = float(cfg.observation_sigma)
        self.initial_state = np.asarray(cfg.initial_state, dtype=float).reshape(-1)[:2]
        self.n_parameters = int(cfg.n_parameters)
        cfg.n_times = self.n_times
        # class-level attributes are kept in sync for registry consumers
        type(self).n_parameters = self.n_parameters
        type(self).n_data = int(self.n_series * self.n_times)
        type(self).data_names = tuple(
            f"{s}_{i}" for s in self._series_names for i in range(self.n_times)
        )

    # -- metadata ----------------------------------------------------------
    @property
    def _series_names(self) -> Tuple[str, ...]:
        return tuple(self.config.series_names)[: self.n_series]

    @property
    def n_data(self) -> int:  # type: ignore[override]
        return int(self.n_series * self.n_times)

    @property
    def joint_dim(self) -> int:
        return int(self.n_parameters + self.n_data)

    # -- prior -------------------------------------------------------------
    def sample_prior(
        self, n_samples: int = 1, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        return self.prior_sample(n_samples, rng)

    def prior_sample(
        self, n_samples: int = 1, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None
    ) -> np.ndarray:
        if rng is None or not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(self.config.seed if seed is None else seed)
        n = int(n_samples)
        z = rng.standard_normal((n, self.n_parameters))
        return prior_transform(z)

    def prior_bounds(self, width: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
        """Broad symmetric box used to initialise MCMC chains."""

        w = float(width)
        return -w + PRIOR_CENTER * np.ones(self.n_parameters), w + PRIOR_CENTER * np.ones(self.n_parameters)

    def log_prior(self, theta: np.ndarray) -> np.ndarray:  # type: ignore[override]
        return log_prior(theta)

    # -- dynamics ----------------------------------------------------------
    def trajectory(
        self,
        theta: Sequence[float],
        times: Optional[Sequence[float]] = None,
        *,
        add_noise: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Deterministic (or noised) trajectory, shape ``(n_times, n_series)``."""

        t = self.times if times is None else np.asarray(times, dtype=float).reshape(-1)
        traj = integrate_trajectory(
            theta,
            t,
            initial_state=self.initial_state,
            rtol=self.config.rtol,
            atol=self.config.atol,
            max_step=self.config.max_step,
        )
        traj = np.asarray(traj, dtype=float)[:, : self.n_series]
        if add_noise:
            rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.config.seed)
            traj = traj + self.observation_sigma * rng.standard_normal(traj.shape)
        return traj

    def simulate(
        self,
        theta: Union[np.ndarray, Sequence[float]],
        rng: Optional[np.random.Generator] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        return_trajectory: bool = False,
        chunksize: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate observations ``x`` (flattened series-major).

        ``theta`` may be a single parameter vector (then ``n_samples`` copies are
        simulated) or a batch of parameter vectors.
        """

        if rng is None or not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(self.config.seed if seed is None else seed)
        theta_arr = np.asarray(theta, dtype=float)
        single = theta_arr.ndim == 1
        if single:
            theta_arr = theta_arr.reshape(1, -1)
        if n_samples is not None and single:
            theta_arr = np.repeat(theta_arr, int(n_samples), axis=0)
        trajs: List[np.ndarray] = []
        for row in theta_arr:
            traj = self.trajectory(row, add_noise=False, rng=rng)
            if add_noise:
                traj = traj + self.observation_sigma * rng.standard_normal(traj.shape)
            trajs.append(traj)
        stacked = np.stack(trajs, axis=0) if trajs else np.zeros((0, self.n_times, self.n_series))
        flat = np.concatenate([stacked[:, :, s] for s in range(self.n_series)], axis=1)
        if single:
            out = flat[0]
            return out
        return flat

    # sbi-style alias
    simulator = simulate

    def __call__(
        self, n_samples: int = 1, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        theta = self.prior_sample(int(n_samples), rng)
        x = self.simulate(theta, rng)
        return theta, x

    def simulate_from_joint(self, theta: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.simulate(theta, rng)

    # -- likelihood / joint -------------------------------------------------
    def data_mean(
        self,
        theta: Union[np.ndarray, Sequence[float]],
        times: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Noise-free observation means with the data layout of :meth:`simulate`."""

        theta_arr = np.asarray(theta, dtype=float)
        single = theta_arr.ndim == 1
        if single:
            theta_arr = theta_arr.reshape(1, -1)
        means = []
        for row in theta_arr:
            traj = self.trajectory(row, times=times, add_noise=False)
            means.append(flatten_trajectory(traj))
        out = np.stack(means, axis=0) if means else np.zeros((0, self.n_data))
        return out[0] if single else out

    def log_likelihood(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gaussian observation log-likelihood with ``sigma = 0.1``.

        ``mask`` optionally selects the observed entries of ``x`` (values ``1``).
        Both ``x`` and ``theta`` may be a single sample or a batch.
        """

        x_arr = np.asarray(x, dtype=float)
        theta_arr = np.asarray(theta, dtype=float)
        single = x_arr.ndim == 1 and theta_arr.ndim == 1
        x2 = np.atleast_2d(x_arr)
        th2 = np.atleast_2d(theta_arr)
        if th2.shape[0] != x2.shape[0]:
            if th2.shape[0] == 1:
                th2 = np.repeat(th2, x2.shape[0], axis=0)
            elif x2.shape[0] == 1:
                x2 = np.repeat(x2, th2.shape[0], axis=0)
            else:
                raise ValueError("theta and x batch sizes do not match")
        mean = self.data_mean(th2)
        resid = x2 - mean
        var = self.observation_sigma ** 2
        if mask is not None:
            m = np.asarray(mask, dtype=float).reshape(-1)[: x2.shape[1]]
            resid = resid * m[None, :]
            n_obs = float(m.sum())
        else:
            n_obs = float(x2.shape[1])
        if n_obs <= 0:
            ll = np.zeros(x2.shape[0], dtype=float)
        else:
            ll = -0.5 * np.sum(resid ** 2, axis=1) / var - 0.5 * n_obs * math.log(2.0 * math.pi * var)
        return ll[0] if single else ll

    def log_joint(self, theta: np.ndarray, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        lp = np.asarray(self.log_prior(theta), dtype=float)
        ll = np.asarray(self.log_likelihood(x, theta, mask=mask), dtype=float)
        out = lp + ll
        return out if out.ndim else out.reshape(())

    def posterior_log_prob(
        self,
        theta: np.ndarray,
        x_obs: np.ndarray,
        *,
        normalize: bool = True,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Unnormalised (or normalised) log posterior ``log p(theta | x_obs)``."""

        theta_arr = np.asarray(theta, dtype=float)
        squeeze = theta_arr.ndim == 1
        th2 = np.atleast_2d(theta_arr)
        x_arr = np.asarray(x_obs, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        lp = np.asarray(log_prior(th2), dtype=float)
        ll = np.asarray(self.log_likelihood(np.repeat(x_arr, th2.shape[0], axis=0), th2, mask=mask), dtype=float)
        out = lp + ll
        if normalize:
            m = float(np.max(out))
            if np.isfinite(m):
                out = out - (m + math.log(float(np.sum(np.exp(out - m))) + 1e-300))
        return out.reshape(()) if squeeze else out

    # -- reference posterior ------------------------------------------------
    def reference_posterior_sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        *,
        mask: Optional[np.ndarray] = None,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        use_mcmc_module: bool = True,
        protocol: Any = None,
    ) -> np.ndarray:
        """Ground-truth posterior samples via MCMC (Appendix A2.2 protocol)."""

        if rng is None or not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(self.config.seed if seed is None else seed)
        x_arr = np.asarray(x_obs, dtype=float).reshape(-1)
        if mask is None:
            m = np.ones(self.n_data, dtype=float)
        else:
            m = np.asarray(mask, dtype=float).reshape(-1)[: self.n_data]
        if use_mcmc_module:
            try:  # pragma: no cover - depends on optional package context
                try:
                    from simformer.reference.mcmc import MCProtocol, sample_reference  # type: ignore
                except Exception:
                    from ..reference.mcmc import MCProtocol, sample_reference  # type: ignore

                cond_mask = np.concatenate([np.zeros(self.n_parameters), m])
                cond_values = np.concatenate([np.zeros(self.n_parameters), x_arr * m])
                protocol = protocol or MCProtocol(
                    method="slice+mh",
                    n_slice=200,
                    slice_step=0.5,
                    n_mh=2000,
                    mh_step=0.05,
                    init_from_joint=True,
                    keep="last",
                )
                samples = sample_reference(
                    self,
                    cond_mask,
                    cond_values,
                    n_samples=int(n_samples),
                    protocol=protocol,
                    seed=int(seed) if seed is not None else None,
                    rng=rng,
                )
                samples = np.asarray(samples, dtype=float)
                if samples.ndim == 2 and samples.shape[1] > self.n_parameters:
                    samples = samples[:, : self.n_parameters]
                return samples.reshape(-1, self.n_parameters)
            except Exception:
                pass
        return self._mh_reference(x_arr, m, int(n_samples), rng, burn_in=burn_in, thinning=thinning)

    ground_truth_posterior = reference_posterior_sample

    def _mh_reference(
        self,
        x_obs: np.ndarray,
        mask: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        *,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        step_size: float = 0.05,
        n_steps: int = 2000,
    ) -> np.ndarray:
        """Self-contained random-walk Metropolis-Hastings reference sampler."""

        n_chains = int(n_samples)
        steps = int(burn_in if burn_in is not None else n_steps)
        chains = prior_sample(n_chains, rng)
        scales = step_size * np.array([0.5, 0.5, 0.5, 0.5])

        def log_post(theta: np.ndarray) -> float:
            lp = float(log_prior(theta))
            if not np.isfinite(lp):
                return -np.inf
            return lp + float(self.log_likelihood(x_obs, theta, mask=mask))

        logp = np.array([log_post(c) for c in chains], dtype=float)
        accept = np.zeros(n_chains, dtype=float)
        for step in range(steps):
            prop = chains + scales[None, :] * rng.standard_normal(chains.shape)
            logp_prop = np.array([log_post(p) for p in prop], dtype=float)
            u = np.log(rng.random(n_chains) + 1e-300)
            acc = (logp_prop - logp) > u
            chains = np.where(acc[:, None], prop, chains)
            logp = np.where(acc, logp_prop, logp)
            accept += acc
            if step == max(steps // 2, 1):
                rate = accept / (step + 1)
                scales = scales * np.where(rate < 0.2, 0.5, np.where(rate > 0.5, 2.0, 1.0))
        if thinning and int(thinning) > 1:
            # keep every thinning-th chain (chains are independent)
            chains = chains[:: int(thinning)]
        return chains.reshape(-1, self.n_parameters)

    def posterior_mean_std(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.reference_posterior_sample(x_obs, n_samples, rng, seed, **kwargs)
        return np.mean(samples, axis=0), np.std(samples, axis=0)

    # -- datasets -----------------------------------------------------------
    def make_dataset(
        self,
        n_simulations: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        *,
        chunk_size: int = 512,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate ``n_simulations`` ``(theta, x)`` pairs."""

        if rng is None or not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(self.config.seed if seed is None else seed)
        n = int(n_simulations)
        theta_out = np.zeros((n, self.n_parameters), dtype=float)
        x_out = np.zeros((n, self.n_data), dtype=float)
        chunk = max(int(chunk_size), 1)
        done = 0
        while done < n:
            m = min(chunk, n - done)
            th = self.prior_sample(m, rng)
            x = np.asarray(self.simulate(th, rng), dtype=float).reshape(m, -1)
            theta_out[done : done + m] = th
            x_out[done : done + m] = x
            done += m
            if verbose:
                print(f"[lotka_volterra] simulated {done}/{n}", flush=True)
        return theta_out, x_out

    def sample_joint(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        theta, x = self.__call__(n_samples, rng)
        return np.concatenate([np.atleast_2d(theta), np.atleast_2d(x)], axis=1)

    def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.concatenate([theta, x], axis=1)

    def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
        return joint[:, : self.n_parameters], joint[:, self.n_parameters : self.joint_dim]

    # -- unstructured-observation helpers (Fig. 5) --------------------------
    def data_index(self, series: int, time_index: int) -> int:
        """Index into the flattened ``x`` vector for ``(series, time index)``."""

        s = int(series)
        i = int(time_index)
        if not (0 <= s < self.n_series):
            raise IndexError(f"series index {s} out of range")
        if not (0 <= i < self.n_times):
            raise IndexError(f"time index {i} out of range")
        return s * self.n_times + i

    def time_index_of(self, t: float, tol: float = 1e-6) -> int:
        idx = int(np.argmin(np.abs(self.times - float(t))))
        if abs(self.times[idx] - float(t)) > tol:
            raise ValueError(f"time {t} is not on the observation grid")
        return idx

    def observation_condition_mask(
        self,
        pairs: Sequence[Tuple[int, float]],
        *,
        return_values: bool = False,
        x_obs: Optional[np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Build a *variable-level* condition mask for a set of observations.

        ``pairs`` is a sequence of ``(series_index, time)`` measurements, e.g.
        ``[(0, 1.0), (0, 4.0)]`` for four prey observations.  The returned mask
        covers the full joint vector ``[theta (4) | x (n_series * n_times)]``
        with ``1`` at the observed data entries (parameters remain latent).
        """

        mask = np.zeros(self.joint_dim, dtype=float)
        for series, t in pairs:
            try:
                i = self.time_index_of(float(t))
            except ValueError:
                i = int(np.argmin(np.abs(self.times - float(t))))
            mask[self.n_parameters + self.data_index(int(series), i)] = 1.0
        if not return_values:
            return mask
        values = np.zeros(self.joint_dim, dtype=float)
        if x_obs is not None:
            x_arr = np.asarray(x_obs, dtype=float).reshape(-1)
            values[self.n_parameters : self.n_parameters + x_arr.size] = x_arr[: self.n_data]
        return mask, values

    def unstructured_observations(
        self,
        rng: Optional[np.random.Generator] = None,
        *,
        n_prey: int = 4,
        n_predator: int = 0,
        seed: Optional[int] = None,
        sample: bool = True,
    ) -> Tuple[List[Tuple[int, float]], np.ndarray]:
        """Draw an unstructured observation set (irregular time points per series).

        Returns the list of ``(series, time)`` pairs and the corresponding
        condition mask over the joint vector.  ``sample=True`` also draws the
        ground-truth parameters and observations (see :meth:`simulate_scenario`).
        """

        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.config.seed if seed is None else seed)
        pairs: List[Tuple[int, float]] = []
        for series, k in ((0, int(n_prey)), (1, int(n_predator))):
            k = min(k, self.n_times)
            idx = np.sort(rng.choice(self.n_times, size=k, replace=False))
            pairs.extend((series, float(self.times[i])) for i in idx)
        mask = self.observation_condition_mask(pairs)
        return pairs, mask

    def simulate_scenario(
        self,
        pairs: Sequence[Tuple[int, float]],
        rng: Optional[np.random.Generator] = None,
        *,
        theta: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate a ground-truth scenario with observations at ``pairs``.

        Returns a dict with ``theta``, the full noised observation vector ``x``,
        the condition ``mask`` (variable level) and the ``condition_values``
        (full joint vector with the observed entries filled in).
        """

        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.config.seed if seed is None else seed)
        theta_true = self.prior_sample(1, rng)[0] if theta is None else np.asarray(theta, dtype=float).reshape(-1)
        x_full = np.asarray(self.simulate(theta_true, rng), dtype=float).reshape(-1)
        mask, values = self.observation_condition_mask(pairs, return_values=True, x_obs=x_full)
        return {
            "theta": theta_true,
            "x": x_full,
            "condition_mask": mask,
            "condition_values": values,
            "pairs": list(pairs),
        }

    def figure5_scenarios(self, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """The two observation scenarios of Fig. 5 (prey only; prey + predator)."""

        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.config.seed if seed is None else seed)
        theta_true = self.prior_sample(1, rng)[0]
        prey = [(0, t) for t in FIGURE5_PREY_TIMES]
        both = prey + [(1, t) for t in FIGURE5_PREDATOR_TIMES]
        return {
            "prey_only": self.simulate_scenario(prey, rng, theta=theta_true),
            "prey_and_predator": self.simulate_scenario(both, rng, theta=theta_true),
        }

    def posterior_predictive(
        self,
        theta_samples: np.ndarray,
        *,
        add_noise: bool = True,
        rng: Optional[np.random.Generator] = None,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate posterior-predictive trajectories for posterior draws."""

        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(self.config.seed)
        theta = np.atleast_2d(np.asarray(theta_samples, dtype=float))
        if n_samples is not None and theta.shape[0] > int(n_samples):
            idx = rng.choice(theta.shape[0], size=int(n_samples), replace=False)
            theta = theta[idx]
        out = np.asarray(self.simulate(theta, rng, add_noise=add_noise), dtype=float)
        return out.reshape(theta.shape[0], self.n_series, self.n_times).transpose(0, 2, 1)

    # -- Simformer plumbing -------------------------------------------------
    def spec(self, token_dim: int = 50, **kwargs: Any) -> Any:
        """Tokenizer spec for the joint vector ``[theta | data tokens]``."""

        layout = str(kwargs.get("layout", self.extra.get("layout", "grid"))).lower()
        n_parameters = int(kwargs.get("n_parameters", self.n_parameters))
        try:
            try:
                from ..tokenizer import FunctionValuedSpec, TokenSpec, build_benchmark_spec  # type: ignore
            except Exception:
                from simformer.tokenizer import FunctionValuedSpec, TokenSpec, build_benchmark_spec  # type: ignore
        except Exception:  # pragma: no cover - tokenizer unavailable
            return {
                "n_parameter_variables": n_parameters,
                "n_data_variables": self.n_data,
                "token_dim": int(token_dim),
            }
        if layout in ("function_valued", "function-valued", "functional", "times"):
            # One token per (series, time) observation: the observation times are
            # treated as the index set of a function-valued variable.  The second
            # series' index points are offset so that the random-Fourier metadata
            # also encodes the species identity.
            offset = 2.0 * float(np.max(self.times)) + 1.0
            fv = [
                FunctionValuedSpec(name=f"{self._series_names[s]}_trajectory", index_set=self.times.copy())
                for s in range(self.n_series)
            ]
            if self.n_series > 1:
                fv[1] = FunctionValuedSpec(
                    name=f"{self._series_names[1]}_trajectory", index_set=self.times.copy() + offset
                )
            return TokenSpec(
                parameter_names=tuple(self.parameter_names[:n_parameters]),
                data_names=(),
                function_valued=tuple(fv),
            )
        return build_benchmark_spec(n_parameters, self.n_data)

    token_spec = spec

    def metadata_times(self) -> np.ndarray:
        """Per-data-token metadata (observation time), series-major."""

        return np.concatenate([self.times for _ in range(self.n_series)])

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        """Directed (or undirected) base mask ``M_E`` for the LV dependency graph."""

        try:
            try:
                from ..attention_masks import build_attention_mask  # type: ignore
            except Exception:
                from simformer.attention_masks import build_attention_mask  # type: ignore
        except Exception:  # pragma: no cover - attention masks unavailable
            return self._fallback_mask(directed=directed, **kwargs)
        variants = [
            dict(
                n_theta=self.n_parameters,
                n_series=self.n_series,
                n_times=self.n_times,
                times=self.times,
                directed=directed,
            ),
            dict(n_theta=self.n_parameters, n_x=self.n_data, directed=directed),
            dict(directed=directed),
            dict(),
        ]
        for kw in variants:
            try:
                mask = build_attention_mask("lotka_volterra", **{**kw, **kwargs})
            except Exception:
                continue
            try:
                mask = np.asarray(mask, dtype=bool)
            except Exception:
                continue
            if mask.ndim == 2 and mask.shape[0] == mask.shape[1]:
                if mask.shape[0] == self.joint_dim:
                    return mask
        return self._fallback_mask(directed=directed, **kwargs)

    def _fallback_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        """Local fallback mask: dense theta->data + per-series temporal chain."""

        n = self.joint_dim
        mask = np.zeros((n, n), dtype=bool)
        np.fill_diagonal(mask, True)
        # parameters depend on each other (shared ODE state)
        mask[: self.n_parameters, : self.n_parameters] = True
        # data depends on all parameters
        mask[self.n_parameters :, : self.n_parameters] = True
        # temporal chain within each series (later observations depend on earlier)
        for s in range(self.n_series):
            base = self.n_parameters + s * self.n_times
            for i in range(1, self.n_times):
                mask[base + i, base + i - 1] = True
        if not directed:
            mask = mask | mask.T
        return mask

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any) -> Any:
        try:
            try:
                from ..tokenizer import Tokenizer  # type: ignore
            except Exception:
                from simformer.tokenizer import Tokenizer  # type: ignore

            return Tokenizer(self.spec(token_dim=token_dim, **kwargs), token_dim=int(token_dim))
        except Exception:
            return None

    def build_model(self, **kwargs: Any) -> Any:
        try:
            try:
                from ..transformer import build_score_network  # type: ignore
            except Exception:
                from simformer.transformer import build_score_network  # type: ignore

            kwargs.setdefault("task", "lotka_volterra")
            kwargs.setdefault("spec", self.spec(token_dim=int(kwargs.get("token_dim", 50))))
            mask = kwargs.pop("attention_mask", None)
            model = build_score_network(**kwargs)
            if mask is not None and model is not None:
                try:
                    model.attention_mask = mask
                except Exception:
                    pass
            return model
        except Exception:
            return None

    # -- misc ---------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d.update(
            {
                "n_data": self.n_data,
                "joint_dim": self.joint_dim,
                "times": self.times.tolist(),
                "initial_state": self.initial_state.tolist(),
            }
        )
        return d


# aliases used by the task registry / other modules
Task = LotkaVolterraTask
Simulator = LotkaVolterraTask
LotkaVolterra = LotkaVolterraTask


def build_task(
    config: Optional[Union[LotkaVolterraConfig, Dict[str, Any]]] = None, **kwargs: Any
) -> LotkaVolterraTask:
    """Factory returning a :class:`LotkaVolterraTask`."""

    return LotkaVolterraTask(config, **kwargs)


# ---------------------------------------------------------------------------
# module level convenience wrappers (cached default task)
# ---------------------------------------------------------------------------
_DEFAULT_TASK: Optional[LotkaVolterraTask] = None


def _default_task() -> LotkaVolterraTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = LotkaVolterraTask()
    return _DEFAULT_TASK


def simulate(theta: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().simulate(theta, rng, **kwargs)


def log_likelihood(x: np.ndarray, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_likelihood(x, theta, **kwargs)


def log_joint(theta: np.ndarray, x: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_joint(theta, x, **kwargs)


def posterior_log_prob(theta: np.ndarray, x_obs: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().posterior_log_prob(theta, x_obs, **kwargs)


def reference_posterior_sample(
    x_obs: np.ndarray, n_samples: int = 1000, rng: Optional[np.random.Generator] = None, **kwargs: Any
) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples, rng, **kwargs)


def make_dataset(
    n_simulations: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    return _default_task().make_dataset(n_simulations, rng, seed, **kwargs)


__all__ = [
    "LotkaVolterraConfig",
    "LotkaVolterraTask",
    "Task",
    "Simulator",
    "LotkaVolterra",
    "build_task",
    "prior_sample",
    "prior_transform",
    "inverse_prior_transform",
    "log_prior",
    "log_likelihood",
    "log_joint",
    "posterior_log_prob",
    "simulate",
    "make_dataset",
    "reference_posterior_sample",
    "lotka_volterra_rhs",
    "integrate_trajectory",
    "flatten_trajectory",
    "unflatten_trajectory",
    "DEFAULT_PARAMETER_NAMES",
    "DEFAULT_SERIES_NAMES",
    "DEFAULT_N_TIMES",
    "DEFAULT_N_SERIES",
    "DEFAULT_T_MAX",
    "DEFAULT_OBSERVATION_SIGMA",
    "DEFAULT_INITIAL_STATE",
    "PRIOR_RANGE",
    "FIGURE5_PREY_TIMES",
    "FIGURE5_PREDATOR_TIMES",
]
