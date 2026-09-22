"""SIRD epidemiological task with a time-dependent (function-valued) contact rate.

Implements the simulator of Appendix A2.2 ("SIRD Model with Time-Dependent Contact
Rate") of the Simformer paper:

    dS/dt = -beta(t) S I
    dI/dt =  beta(t) S I - gamma I - mu I
    dR/dt =  gamma I
    dD/dt =  mu I

with

* global parameters ``gamma, mu ~ Unif(0, 0.5)``  (the paper prints ``delta`` for the
  second global rate; the printed symbol is a typo for the mortality rate ``mu``),
* a time-dependent contact rate obtained by sampling ``beta_hat ~ GP(0, k)`` from a
  Gaussian-process prior with an RBF kernel
  ``k(t1, t2) = 2.5^2 * exp(-0.5 * ||t1 - t2||^2 / 7^2)`` and then applying a
  sigmoid transform so that ``beta(t) in [0, 1]``,
* observations of the infected (I), recovered (R) and deceased (D) population
  densities at irregularly spaced time points,
* log-normal observation noise with mean equal to the simulated value and standard
  deviation ``sigma = 0.05``.

The joint vector follows the tokenizer's canonical ordering used throughout this
repository::

    joint = [ theta scalars (gamma, mu) | data scalars (I,R,D x times) | function values (beta_hat) ]

so that ``joint_dim = 2 + n_series * n_obs + n_index_points``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional accelerated ODE integration
    from scipy.integrate import solve_ivp as _solve_ivp  # type: ignore
except Exception:  # pragma: no cover - scipy not available
    _solve_ivp = None


# --------------------------------------------------------------------------------------
# task base class (soft import, mirrors the other task modules)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - depends on package layout
    from . import TaskBase  # type: ignore
except Exception:  # pragma: no cover
    try:
        from simformer.tasks import TaskBase  # type: ignore
    except Exception:

        class TaskBase:  # type: ignore
            """Minimal stand-in so this module stays importable on its own."""

            name = "task"
            n_parameters = 0
            n_data = 0
            parameter_names: Tuple[str, ...] = ()
            data_names: Tuple[str, ...] = ()

            @property
            def joint_dim(self) -> int:  # pragma: no cover - trivial
                return int(self.n_parameters) + int(self.n_data)

            def to_joint(self, theta, x):  # pragma: no cover - trivial
                theta = np.atleast_2d(np.asarray(theta, dtype=float))
                x = np.atleast_2d(np.asarray(x, dtype=float))
                return np.concatenate([theta, x], axis=-1)

            def split_joint(self, joint):  # pragma: no cover - trivial
                joint = np.atleast_2d(np.asarray(joint, dtype=float))
                n = int(self.n_parameters)
                return joint[..., :n], joint[..., n:]


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
DEFAULT_T_MAX: float = 40.0
DEFAULT_N_INDEX_POINTS: int = 20
DEFAULT_GP_LENGTHSCALE: float = 7.0
DEFAULT_GP_AMPLITUDE: float = 2.5
DEFAULT_GP_MEAN: float = 0.0
DEFAULT_GP_JITTER: float = 1e-6
DEFAULT_OBSERVATION_SIGMA: float = 0.05
DEFAULT_PARAMETER_RANGE: Tuple[float, float] = (0.0, 0.5)
DEFAULT_INITIAL_STATE: Tuple[float, float, float, float] = (0.99, 0.01, 0.0, 0.0)
DEFAULT_OBSERVATION_TIMES: Tuple[float, ...] = (2.0, 7.5, 13.0, 19.5, 26.0)
DEFAULT_SERIES_NAMES: Tuple[str, ...] = ("I", "R", "D")
DEFAULT_PARAMETER_NAMES: Tuple[str, ...] = ("gamma", "mu")
DEFAULT_N_SERIES: int = len(DEFAULT_SERIES_NAMES)
DEFAULT_N_OBSERVATIONS: int = len(DEFAULT_OBSERVATION_TIMES)
ODE_SOLVER = "RK45"
ODE_RTOL = 1e-6
ODE_ATOL = 1e-8
RK4_MAX_STEP = 0.005
_MIN_POSITIVE = 1e-12


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class SIRDConfig:
    """Configuration of the SIRD simulator (Appendix A2.2)."""

    # --- parameterisation -------------------------------------------------------
    parameter_names: Tuple[str, ...] = DEFAULT_PARAMETER_NAMES
    series_names: Tuple[str, ...] = DEFAULT_SERIES_NAMES
    parameter_range: Tuple[float, float] = DEFAULT_PARAMETER_RANGE
    n_index_points: int = DEFAULT_N_INDEX_POINTS
    # --- time grid / observations ----------------------------------------------
    t_min: float = 0.0
    t_max: float = DEFAULT_T_MAX
    observation_times: Tuple[float, ...] = DEFAULT_OBSERVATION_TIMES
    beta_grid: Optional[Tuple[float, ...]] = None
    observation_sigma: float = DEFAULT_OBSERVATION_SIGMA
    initial_state: Tuple[float, float, float, float] = DEFAULT_INITIAL_STATE
    # --- Gaussian process prior on hat(beta) ------------------------------------
    gp_lengthscale: float = DEFAULT_GP_LENGTHSCALE
    gp_amplitude: float = DEFAULT_GP_AMPLITUDE
    gp_mean: float = DEFAULT_GP_MEAN
    gp_jitter: float = DEFAULT_GP_JITTER
    # --- misc -------------------------------------------------------------------
    name: str = "sird"
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ----- derived --------------------------------------------------------------
    @property
    def n_series(self) -> int:
        return int(len(self.series_names))

    @property
    def n_observations(self) -> int:
        return int(len(self.observation_times))

    @property
    def n_parameters(self) -> int:
        """Number of *variables*: the two global rates plus the beta function."""
        return int(len(self.parameter_names)) + 1

    @property
    def n_scalar_parameters(self) -> int:
        return int(len(self.parameter_names))

    @property
    def n_data(self) -> int:
        return int(self.n_series * self.n_observations)

    @property
    def index_grid(self) -> np.ndarray:
        """Latent GP / function-valued index set (the ``n_index_points`` nodes)."""
        if self.beta_grid is not None:
            return np.asarray(self.beta_grid, dtype=float)
        return np.linspace(float(self.t_min), float(self.t_max), int(self.n_index_points))

    @property
    def dense_index_grid(self) -> np.ndarray:
        """Fine grid used to evaluate ``beta(t)`` inside the ODE solver."""
        return np.linspace(float(self.t_min), float(self.t_max), 401)

    @property
    def theta_dim(self) -> int:
        """Full parameter dimension (global rates + function values)."""
        return int(self.n_scalar_parameters + self.index_grid.size)

    @property
    def joint_dim(self) -> int:
        return int(self.theta_dim + self.n_data)

    @property
    def n_index_points_effective(self) -> int:
        return int(self.index_grid.size)

    @property
    def parameter_names_all(self) -> Tuple[str, ...]:
        return tuple(list(self.parameter_names) + ["beta"])

    def observation_times_used(self, series: Optional[Sequence[str]] = None) -> np.ndarray:
        times = np.asarray(self.observation_times, dtype=float)
        if series is None:
            return times
        idx = self.series_index(series)
        return times[idx]

    def series_index(self, series: Optional[Sequence[str]] = None) -> np.ndarray:
        """Indices of the requested series inside ``series_names``."""
        if series is None:
            return np.arange(self.n_series)
        out: List[int] = []
        for name in series:
            key = str(name).upper()
            if key not in [s.upper() for s in self.series_names]:
                raise ValueError(f"unknown series {name!r}; known: {self.series_names}")
            out.append([s.upper() for s in self.series_names].index(key))
        return np.asarray(out, dtype=int)

    def masked_beta_grid(self, n_index_points: Optional[int] = None) -> np.ndarray:
        """Sub-sample the index set to ``n_index_points`` points (function-valued)."""
        grid = self.index_grid
        if n_index_points is None or int(n_index_points) <= 0 or int(n_index_points) >= grid.size:
            return grid
        return np.linspace(grid[0], grid[-1], int(n_index_points))

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "parameter_names": tuple(self.parameter_names),
            "series_names": tuple(self.series_names),
            "parameter_range": tuple(self.parameter_range),
            "n_index_points": int(self.n_index_points),
            "t_min": float(self.t_min),
            "t_max": float(self.t_max),
            "observation_times": tuple(self.observation_times),
            "observation_sigma": float(self.observation_sigma),
            "initial_state": tuple(self.initial_state),
            "gp_lengthscale": float(self.gp_lengthscale),
            "gp_amplitude": float(self.gp_amplitude),
            "gp_mean": float(self.gp_mean),
            "seed": int(self.seed),
        }
        out.update(dict(self.extra))
        return out

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "SIRDConfig"]] = None, **kwargs: Any) -> "SIRDConfig":
        if isinstance(cfg, SIRDConfig):
            data = cfg.to_dict()
        elif cfg is None:
            data = {}
        else:
            data = dict(cfg)
        data.update(kwargs)
        known = {
            "parameter_names",
            "series_names",
            "parameter_range",
            "n_index_points",
            "t_min",
            "t_max",
            "observation_times",
            "beta_grid",
            "observation_sigma",
            "initial_state",
            "gp_lengthscale",
            "gp_amplitude",
            "gp_mean",
            "gp_jitter",
            "name",
            "seed",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        init = {k: v for k, v in data.items() if k in known}
        if "observation_times" in init and init["observation_times"] is not None:
            init["observation_times"] = tuple(float(t) for t in init["observation_times"])
        if "parameter_range" in init and init["parameter_range"] is not None:
            init["parameter_range"] = tuple(float(v) for v in init["parameter_range"])
        if "initial_state" in init and init["initial_state"] is not None:
            init["initial_state"] = tuple(float(v) for v in init["initial_state"])
        if "beta_grid" in init and init["beta_grid"] is not None:
            init["beta_grid"] = tuple(float(t) for t in init["beta_grid"])
        obj = cls(**init)
        obj.extra = extra
        return obj


# --------------------------------------------------------------------------------------
# Gaussian process prior on the latent contact rate
# --------------------------------------------------------------------------------------
def rbf_kernel(
    t1: np.ndarray,
    t2: Optional[np.ndarray] = None,
    *,
    lengthscale: float = DEFAULT_GP_LENGTHSCALE,
    amplitude: float = DEFAULT_GP_AMPLITUDE,
) -> np.ndarray:
    """RBF kernel ``k(t1, t2) = amplitude^2 * exp(-0.5 ||t1 - t2||^2 / lengthscale^2)``."""
    t1 = np.asarray(t1, dtype=float).reshape(-1)
    t2 = t1 if t2 is None else np.asarray(t2, dtype=float).reshape(-1)
    diff = t1[:, None] - t2[None, :]
    return float(amplitude) ** 2 * np.exp(-0.5 * (diff ** 2) / float(lengthscale) ** 2)


def gp_covariance(
    times: Optional[np.ndarray] = None,
    *,
    lengthscale: float = DEFAULT_GP_LENGTHSCALE,
    amplitude: float = DEFAULT_GP_AMPLITUDE,
    jitter: float = DEFAULT_GP_JITTER,
) -> np.ndarray:
    """Covariance matrix of the ``hat(beta)`` GP prior on a finite index grid."""
    times = np.linspace(0.0, DEFAULT_T_MAX, DEFAULT_N_INDEX_POINTS) if times is None else np.asarray(times, float)
    K = rbf_kernel(times, times, lengthscale=lengthscale, amplitude=amplitude)
    return K + float(jitter) * np.eye(K.shape[0])


def sample_gp(
    n_samples: int = 1,
    times: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    *,
    lengthscale: float = DEFAULT_GP_LENGTHSCALE,
    amplitude: float = DEFAULT_GP_AMPLITUDE,
    mean: float = DEFAULT_GP_MEAN,
    jitter: float = DEFAULT_GP_JITTER,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw ``n_samples`` functions from the ``GP(0, k)`` prior at ``times``."""
    rng = _rng(rng, seed)
    K = gp_covariance(times, lengthscale=lengthscale, amplitude=amplitude, jitter=jitter)
    n = int(K.shape[0])
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:  # pragma: no cover - numerically degenerate
        vals, vecs = np.linalg.eigh(K)
        L = vecs @ np.diag(np.sqrt(np.clip(vals, 1e-12, None)))
    z = rng.standard_normal((int(n_samples), n))
    return float(mean) + z @ L.T


def log_gp_prior(
    beta_hat: np.ndarray,
    times: Optional[np.ndarray] = None,
    *,
    lengthscale: float = DEFAULT_GP_LENGTHSCALE,
    amplitude: float = DEFAULT_GP_AMPLITUDE,
    mean: float = DEFAULT_GP_MEAN,
    jitter: float = DEFAULT_GP_JITTER,
) -> np.ndarray:
    """Multivariate-normal log density of the ``hat(beta)`` GP prior."""
    K = gp_covariance(times, lengthscale=lengthscale, amplitude=amplitude, jitter=jitter)
    x = np.atleast_2d(np.asarray(beta_hat, dtype=float))
    if x.shape[-1] != K.shape[0]:
        # interpolate a differently sized index set onto the reference grid
        raise ValueError(
            f"beta_hat has {x.shape[-1]} entries but the GP prior is defined on {K.shape[0]} index points"
        )
    diff = x - float(mean)
    n = K.shape[0]
    try:
        L = np.linalg.cholesky(K)
        sol = np.linalg.solve(L, diff.T)
        quad = np.sum(sol ** 2, axis=0)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
    except np.linalg.LinAlgError:  # pragma: no cover
        sign, logdet = np.linalg.slogdet(K)
        quad = np.sum(diff * np.linalg.solve(K, diff.T).T, axis=-1)
    log_norm = -0.5 * (n * math.log(2.0 * math.pi) + logdet)
    return log_norm - 0.5 * quad


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def beta_from_hat(
    beta_hat: np.ndarray,
    grid: np.ndarray,
    times: Optional[np.ndarray] = None,
    *,
    apply_sigmoid: bool = True,
) -> np.ndarray:
    """Evaluate ``beta(t) = sigmoid(hat(beta)(t))`` by linear interpolation.

    Parameters
    ----------
    beta_hat:
        Latent GP values on ``grid``; last axis indexes the grid.
    grid:
        Index grid on which ``beta_hat`` is defined.
    times:
        Times at which to evaluate; defaults to ``grid`` itself.
    """
    beta_hat = np.atleast_2d(np.asarray(beta_hat, dtype=float))
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if grid.size == 1:
        vals = np.repeat(beta_hat, 1, axis=-1)
    else:
        t = grid if times is None else np.asarray(times, dtype=float).reshape(-1)
        vals = np.stack([np.interp(t, grid, row) for row in beta_hat], axis=0)
    return sigmoid(vals) if apply_sigmoid else vals


def sigmoid_jacobian(beta_hat: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid transform (for change-of-variables use)."""
    s = sigmoid(beta_hat)
    return s * (1.0 - s)


# --------------------------------------------------------------------------------------
# ODE simulator
# --------------------------------------------------------------------------------------
def sird_rhs(
    state: Sequence[float],
    beta: float,
    gamma: float,
    mu: float,
) -> List[float]:
    """Right-hand side of the SIRD ODE system (Appendix A2.2, Eq. 8)."""
    S, I, R, D = (float(v) for v in state)
    return [
        -float(beta) * S * I,
        float(beta) * S * I - float(gamma) * I - float(mu) * I,
        float(gamma) * I,
        float(mu) * I,
    ]


def _rk4_trajectory(
    rhs: Callable[[float, Sequence[float]], Sequence[float]],
    y0: np.ndarray,
    t_eval: np.ndarray,
    *,
    max_step: float = RK4_MAX_STEP,
) -> np.ndarray:
    """Fixed-step RK4 fallback (used when scipy is unavailable)."""
    t_end = float(np.max(t_eval))
    n_steps = max(1, int(math.ceil(t_end / max(float(max_step), 1e-6))))
    dt = t_end / n_steps
    y = np.asarray(y0, dtype=float)
    out = np.empty((t_eval.size, y.size), dtype=float)
    targets = np.sort(np.asarray(t_eval, dtype=float))
    order = np.argsort(np.asarray(t_eval, dtype=float))
    k = 0
    t = 0.0
    for step in range(n_steps + 1):
        while k < targets.size and targets[k] <= t + 1e-12:
            out[order[k]] = y
            k += 1
        if step == n_steps:
            break
        k1 = np.asarray(rhs(t, y), dtype=float)
        k2 = np.asarray(rhs(t + 0.5 * dt, y + 0.5 * dt * k1), dtype=float)
        k3 = np.asarray(rhs(t + 0.5 * dt, y + 0.5 * dt * k2), dtype=float)
        k4 = np.asarray(rhs(t + dt, y + dt * k3), dtype=float)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
    while k < targets.size:  # pragma: no cover - safety net
        out[order[k]] = y
        k += 1
    return out


def integrate_sird(
    beta_fn: Callable[[float], float],
    gamma: float,
    mu: float,
    times: Sequence[float],
    *,
    initial_state: Sequence[float] = DEFAULT_INITIAL_STATE,
    rtol: float = ODE_RTOL,
    atol: float = ODE_ATOL,
    max_step: Optional[float] = None,
) -> np.ndarray:
    """Integrate the SIRD ODEs; returns an ``(n_times, 4)`` array ``[S, I, R, D]``."""
    t_eval = np.sort(np.asarray(times, dtype=float))
    y0 = np.asarray(initial_state, dtype=float)
    t0 = 0.0
    t_end = float(max(t_eval.max(), t0))

    def rhs(t: float, y: Sequence[float]) -> List[float]:
        return sird_rhs(y, beta_fn(float(t)), gamma, mu)

    if _solve_ivp is not None and t_end > t0:
        try:
            sol = _solve_ivp(
                rhs,
                (t0, t_end),
                y0,
                t_eval=t_eval,
                method=ODE_SOLVER,
                rtol=rtol,
                atol=atol,
                max_step=np.inf if max_step is None else float(max_step),
            )
            if sol.success and sol.y.shape[1] == t_eval.size:
                return np.asarray(sol.y.T, dtype=float)
        except Exception:  # pragma: no cover - fall through to RK4
            pass
    return _rk4_trajectory(rhs, y0, t_eval, max_step=RK4_MAX_STEP if max_step is None else float(max_step))


def lognormal_mean_std_logpdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    """Log density of ``LogNormal`` parameterised by *mean* and *std* (Appendix A2.2)."""
    x = np.asarray(x, dtype=float)
    mean = np.maximum(np.asarray(mean, dtype=float), _MIN_POSITIVE)
    s = float(std)
    sigma2 = np.log1p((s ** 2) / (mean ** 2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    x_safe = np.maximum(x, _MIN_POSITIVE)
    return (
        -np.log(x_safe)
        - np.log(sigma)
        - 0.5 * math.log(2.0 * math.pi)
        - (np.log(x_safe) - mu) ** 2 / (2.0 * sigma2)
    )


def lognormal_mean_std_sample(
    mean: np.ndarray,
    std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw log-normal noise with the requested mean and standard deviation."""
    mean = np.maximum(np.asarray(mean, dtype=float), _MIN_POSITIVE)
    s = float(std)
    sigma2 = np.log1p((s ** 2) / (mean ** 2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - 0.5 * sigma2
    return np.exp(mu + sigma * rng.standard_normal(np.shape(mean)))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is not None:
        try:  # allow passing an int-like
            return np.random.default_rng(int(rng))
        except Exception:  # pragma: no cover
            pass
    return np.random.default_rng(0 if seed is None else int(seed))


def _as_theta(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a parameter vector into ``(gamma, mu, beta_hat)``."""
    theta = np.atleast_2d(np.asarray(theta, dtype=float))
    if theta.shape[-1] < 3:
        raise ValueError("theta must contain at least [gamma, mu, beta_hat...] entries")
    return theta[:, 0], theta[:, 1], theta[:, 2:]


# --------------------------------------------------------------------------------------
# task
# --------------------------------------------------------------------------------------
class SIRDTask(TaskBase):
    """SIRD simulator with a time-dependent, GP-sampled contact rate."""

    name = "sird"
    n_parameters = 3          # gamma, mu, beta(function-valued)
    n_data = DEFAULT_N_SERIES * DEFAULT_N_OBSERVATIONS
    parameter_names = ("gamma", "mu", "beta")
    data_names = tuple(
        f"{s}_{i}" for s in DEFAULT_SERIES_NAMES for i in range(DEFAULT_N_OBSERVATIONS)
    )

    def __init__(
        self,
        config: Optional[Union[SIRDConfig, Dict[str, Any]]] = None,
        *,
        n_index_points: Optional[int] = None,
        n_series: Optional[int] = None,
        n_observations: Optional[int] = None,
        observation_times: Optional[Sequence[float]] = None,
        t_max: Optional[float] = None,
        observation_sigma: Optional[float] = None,
        gp_lengthscale: Optional[float] = None,
        gp_amplitude: Optional[float] = None,
        initial_state: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        cfg = SIRDConfig.from_dict(config, **kwargs)
        if n_index_points is not None:
            cfg.n_index_points = int(n_index_points)
        if observation_times is not None:
            cfg.observation_times = tuple(float(t) for t in observation_times)
        if n_observations is not None:
            cfg.observation_times = cfg.observation_times[: int(n_observations)]
            if len(cfg.observation_times) < int(n_observations):
                cfg.observation_times = tuple(
                    np.linspace(2.0, float(cfg.t_max) * 0.65, int(n_observations))
                )
        if t_max is not None:
            cfg.t_max = float(t_max)
        if observation_sigma is not None:
            cfg.observation_sigma = float(observation_sigma)
        if gp_lengthscale is not None:
            cfg.gp_lengthscale = float(gp_lengthscale)
        if gp_amplitude is not None:
            cfg.gp_amplitude = float(gp_amplitude)
        if initial_state is not None:
            cfg.initial_state = tuple(float(v) for v in initial_state)
        if name is not None:
            cfg.name = str(name)
        if seed is not None:
            cfg.seed = int(seed)
        if n_series is not None:
            cfg.series_names = DEFAULT_SERIES_NAMES[: int(n_series)]

        self.config = cfg
        self.n_series = cfg.n_series
        self.n_observations = cfg.n_observations
        self.data_names = tuple(f"{s}_{i}" for s in cfg.series_names for i in range(self.n_observations))
        self.n_data = int(self.n_series * self.n_observations)
        self.n_parameters = int(cfg.n_scalar_parameters + 1)
        self.parameter_names = tuple(cfg.parameter_names_all)
        self._grid = cfg.index_grid
        self._dense_grid = cfg.dense_index_grid
        self._rng = np.random.default_rng(int(cfg.seed))

    # ------------------------------------------------------------------ properties
    @property
    def index_grid(self) -> np.ndarray:
        return self._grid

    @property
    def observation_times(self) -> np.ndarray:
        return np.asarray(self.config.observation_times, dtype=float)

    @property
    def observation_sigma(self) -> float:
        return float(self.config.observation_sigma)

    @property
    def n_index_points(self) -> int:
        return int(self._grid.size)

    @property
    def n_scalar_parameters(self) -> int:
        return int(self.config.n_scalar_parameters)

    @property
    def theta_dim(self) -> int:
        return int(self.config.theta_dim)

    @property
    def joint_dim(self) -> int:
        return int(self.config.joint_dim)

    @property
    def prior_low(self) -> float:
        return float(self.config.parameter_range[0])

    @property
    def prior_high(self) -> float:
        return float(self.config.parameter_range[1])

    # ---------------------------------------------------------------- prior
    def prior_sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None,
                     seed: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """Draw parameters ``[gamma, mu, beta_hat_1 ... beta_hat_K]``."""
        rng = _rng(rng if rng is not None else self._rng, seed)
        n = int(n_samples)
        lo, hi = self.config.parameter_range
        glob = rng.uniform(float(lo), float(hi), size=(n, self.n_scalar_parameters))
        beta_hat = sample_gp(
            n,
            self._grid,
            rng,
            lengthscale=self.config.gp_lengthscale,
            amplitude=self.config.gp_amplitude,
            mean=self.config.gp_mean,
            jitter=self.config.gp_jitter,
        )
        return np.concatenate([glob, beta_hat], axis=-1)

    sample_prior = prior_sample

    def log_prior(self, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Log prior: uniform global rates plus the GP prior on ``hat(beta)``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        lo, hi = self.config.parameter_range
        glob = theta[:, : self.n_scalar_parameters]
        with np.errstate(invalid="ignore"):
            inside = np.all((glob >= lo) & (glob <= hi), axis=-1)
        logp = np.where(inside, -self.n_scalar_parameters * math.log(hi - lo), -np.inf)
        beta_hat = theta[:, self.n_scalar_parameters :]
        gp = log_gp_prior(
            beta_hat,
            self._grid,
            lengthscale=self.config.gp_lengthscale,
            amplitude=self.config.gp_amplitude,
            mean=self.config.gp_mean,
            jitter=self.config.gp_jitter,
        )
        return logp + gp

    # ---------------------------------------------------------------- ODE / simulator
    def beta_function(self, beta_hat: np.ndarray) -> Callable[[float], float]:
        """Build ``t -> beta(t)`` by interpolating ``hat(beta)`` and applying sigmoid."""
        beta_hat = np.asarray(beta_hat, dtype=float).reshape(-1)
        grid = self._grid

        def _fn(t: float) -> float:
            val = float(np.interp(float(t), grid, beta_hat))
            return float(sigmoid(np.asarray(val)))

        return _fn

    def trajectory(
        self,
        theta: np.ndarray,
        times: Optional[Sequence[float]] = None,
        *,
        dense: bool = False,
    ) -> np.ndarray:
        """Deterministic trajectory ``(n_times, 4)`` with columns ``[S, I, R, D]``."""
        gamma, mu, beta_hat = _as_theta(theta)[0][0], _as_theta(theta)[1][0], _as_theta(theta)[2][0]
        t = self._dense_grid if dense else (self.observation_times if times is None else np.asarray(times, float))
        return integrate_sird(
            self.beta_function(beta_hat),
            float(gamma),
            float(mu),
            t,
            initial_state=self.config.initial_state,
        )

    def data_mean(self, theta: np.ndarray, times: Optional[Sequence[float]] = None) -> np.ndarray:
        """Noise-free observations ``(n, n_series * n_observations)`` (series-major)."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        t = self.observation_times if times is None else np.asarray(times, dtype=float)
        idx = self.config.series_index(self.config.series_names)
        rows = []
        for row in theta:
            traj = self.trajectory(row, t)  # (n_times, 4) = [S, I, R, D]
            obs = traj[:, 1:][:, [i - 1 for i in [1, 2, 3]]] if False else traj[:, 1:4]
            obs = obs[:, idx]
            rows.append(obs.T.reshape(-1))  # series-major
        return np.asarray(rows, dtype=float)

    def simulate(
        self,
        theta: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        times: Optional[Sequence[float]] = None,
        return_components: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Simulate noisy observations ``(n, n_data)`` of (I, R, D) densities."""
        rng = _rng(rng if rng is not None else self._rng, seed)
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if n_samples is not None and theta.shape[0] == 1:
            theta = np.repeat(theta, int(n_samples), axis=0)
        if theta.shape[-1] == self.joint_dim:  # tolerate full joint vectors
            theta = self.theta_from_joint(theta)
        mean = self.data_mean(theta, times=times)
        if not add_noise:
            return (mean, np.zeros_like(mean)) if return_components else mean
        x = lognormal_mean_std_sample(mean, self.observation_sigma, rng)
        if return_components:
            return x, mean
        return x

    simulator = simulate

    def __call__(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any):
        theta = self.prior_sample(n_samples, rng)
        return theta, self.simulate(theta, rng, add_noise=True)

    def sample_joint(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        theta, x = self.__call__(n_samples, rng)
        return self.to_joint(theta, x)

    # ---------------------------------------------------------------- density
    def log_likelihood(self, x: np.ndarray, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Log-normal log likelihood with mean ``S(t)`` and sd ``sigma = 0.05``."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if theta.shape[-1] == self.joint_dim:
            theta = self.theta_from_joint(theta)
        if x.shape[0] == 1 and theta.shape[0] > 1:
            x = np.repeat(x, theta.shape[0], axis=0)
        if theta.shape[0] == 1 and x.shape[0] > 1:
            theta = np.repeat(theta, x.shape[0], axis=0)
        mean = self.data_mean(theta)
        return np.sum(lognormal_mean_std_logpdf(x, mean, self.observation_sigma), axis=-1)

    def log_joint(self, theta: np.ndarray, x: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.log_prior(theta) + self.log_likelihood(x, theta)

    def posterior_log_prob(self, theta: np.ndarray, x_obs: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Unnormalised log posterior ``log p(theta) + log p(x_obs | theta)``."""
        lp = self.log_joint(theta, x_obs)
        if normalize:
            finite = np.isfinite(lp)
            if finite.any():
                m = np.max(lp[finite])
                lp = np.where(finite, lp - m, lp)
        return lp

    ground_truth_log_posterior = posterior_log_prob

    def log_posterior(self, theta: np.ndarray, x_obs: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.posterior_log_prob(theta, x_obs)

    # ---------------------------------------------------------------- joint layout
    def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Canonical joint vector ``[theta scalars | data | function values]``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        x = np.atleast_2d(np.asarray(x, dtype=float))
        n_scalar = self.n_scalar_parameters
        return np.concatenate([theta[:, :n_scalar], x, theta[:, n_scalar:]], axis=-1)

    def theta_from_joint(self, joint: np.ndarray) -> np.ndarray:
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
        n_scalar = self.n_scalar_parameters
        head = joint[:, :n_scalar]
        tail = joint[:, n_scalar + self.n_data :]
        return np.concatenate([head, tail], axis=-1)

    def data_from_joint(self, joint: np.ndarray) -> np.ndarray:
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
        n_scalar = self.n_scalar_parameters
        return joint[:, n_scalar : n_scalar + self.n_data]

    def function_values_from_joint(self, joint: np.ndarray) -> np.ndarray:
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
        return joint[:, self.n_scalar_parameters + self.n_data :]

    def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse of :meth:`to_joint`; returns ``(theta, x)``."""
        return self.theta_from_joint(joint), self.data_from_joint(joint)

    def beta_hat_from_joint(self, joint: np.ndarray) -> np.ndarray:
        return self.function_values_from_joint(joint)

    def model_inputs(self, joint: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract ``(theta_scalars, x, function_values)`` for the tokenizer."""
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
        n_scalar = self.n_scalar_parameters
        return {
            "theta": joint[:, :n_scalar],
            "x": joint[:, n_scalar : n_scalar + self.n_data],
            "function_values": joint[:, n_scalar + self.n_data :],
        }

    # ---------------------------------------------------------------- masks
    def index_of(self, series: str, observation_index: Optional[int] = None) -> np.ndarray:
        """Data-block index/indices of a series (or of one observation of it)."""
        s_idx = int(self.config.series_index([series])[0])
        if observation_index is None:
            offs = s_idx * self.n_observations + np.arange(self.n_observations)
            return offs
        return np.asarray([s_idx * self.n_observations + int(observation_index)], dtype=int)

    def observation_condition_mask(
        self,
        series: Optional[Sequence[str]] = None,
        times: Optional[Sequence[float]] = None,
        *,
        value_level: bool = True,
    ) -> np.ndarray:
        """Condition mask selecting observations of (a subset of) I, R, D."""
        series = list(self.config.series_names) if series is None else list(series)
        if times is None:
            t_sel: Sequence[float] = self.observation_times
        else:
            t_sel = list(times)
        all_times = list(self.observation_times)
        obs_idx = [all_times.index(float(t)) for t in t_sel if float(t) in all_times]

        mask = np.zeros(self.joint_dim, dtype=float)
        for s in series:
            block = self.index_of(s)
            mask[self.n_scalar_parameters + block[obs_idx]] = 1.0
        return mask if value_level else self.variable_mask(mask)

    def posterior_condition_mask(self, value_level: bool = True) -> np.ndarray:
        """All data observed, all parameters (including ``beta``) latent."""
        mask = np.zeros(self.joint_dim, dtype=float)
        mask[self.n_scalar_parameters : self.n_scalar_parameters + self.n_data] = 1.0
        return mask if value_level else self.variable_mask(mask)

    def likelihood_condition_mask(self, value_level: bool = True) -> np.ndarray:
        """All parameters observed, all data latent."""
        mask = np.ones(self.joint_dim, dtype=float)
        mask[self.n_scalar_parameters : self.n_scalar_parameters + self.n_data] = 0.0
        return mask if value_level else self.variable_mask(mask)

    def variable_mask(self, value_mask: np.ndarray) -> np.ndarray:
        """Collapse a value-level mask to ``[gamma, mu, data..., beta]`` variables."""
        value_mask = np.asarray(value_mask, dtype=float)
        n_scalar = self.n_scalar_parameters
        out = np.zeros(2 + self.n_data + 1, dtype=float)
        out[0] = value_mask[0]
        out[1] = value_mask[1]
        out[2 : 2 + self.n_data] = value_mask[n_scalar : n_scalar + self.n_data]
        out[-1] = float(np.max(value_mask[n_scalar + self.n_data :]))
        return out

    def value_mask_from_variable(self, variable_mask: np.ndarray) -> np.ndarray:
        """Expand ``[gamma, mu, data..., beta]`` variable mask to value level."""
        variable_mask = np.asarray(variable_mask, dtype=float)
        n_scalar = self.n_scalar_parameters
        out = np.zeros(self.joint_dim, dtype=float)
        out[0] = variable_mask[0]
        out[1] = variable_mask[1]
        out[n_scalar : n_scalar + self.n_data] = variable_mask[2 : 2 + self.n_data]
        out[n_scalar + self.n_data :] = variable_mask[-1]
        return out

    def parameter_measurement_condition_mask(
        self,
        *,
        beta_indices: Optional[Sequence[int]] = None,
        measure_gamma: bool = False,
        measure_mu: bool = False,
        infected_observations: Optional[Sequence[int]] = None,
        value_level: bool = True,
    ) -> np.ndarray:
        """Fig. 6b scenario: contact-rate measurements (+ optional global rates)
        together with one or more measurements of the infected population."""
        mask = np.zeros(self.joint_dim, dtype=float)
        if measure_gamma:
            mask[0] = 1.0
        if measure_mu:
            mask[1] = 1.0
        beta_offset = self.n_scalar_parameters + self.n_data
        if beta_indices is None:
            mask[beta_offset:] = 1.0
        else:
            mask[beta_offset + np.asarray(beta_indices, dtype=int)] = 1.0
        if infected_observations is not None:
            mask[self.n_scalar_parameters + self.index_of("I", None)[np.asarray(infected_observations, dtype=int)]] = 1.0
        return mask if value_level else self.variable_mask(mask)

    def measurement_condition_mask(self, **kwargs: Any) -> np.ndarray:
        """Alias of :meth:`parameter_measurement_condition_mask`."""
        return self.parameter_measurement_condition_mask(**kwargs)

    # ---------------------------------------------------------------- reference sampling
    def reference_posterior_sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        use_mcmc_module: bool = True,
    ) -> np.ndarray:
        """Ground-truth posterior samples ``(n_samples, theta_dim)``."""
        rng = _rng(rng if rng is not None else self._rng, seed)
        x_obs = np.atleast_2d(np.asarray(x_obs, dtype=float)).reshape(-1)
        mask = self.posterior_condition_mask(value_level=True)
        values = np.zeros(self.joint_dim, dtype=float)
        values[mask > 0] = x_obs

        if use_mcmc_module:
            try:  # pragma: no cover - depends on optional module
                from ..reference.mcmc import sample_reference  # type: ignore

                joint = sample_reference(
                    self,
                    mask,
                    values,
                    n_samples=int(n_samples),
                    seed=int(rng.integers(0, 2 ** 31 - 1)),
                    return_full=False,
                )
                joint = np.atleast_2d(np.asarray(joint, dtype=float))
                if joint.shape[-1] == self.joint_dim:
                    return self.theta_from_joint(joint)
                return joint[:, : self.theta_dim]
            except Exception:
                pass

        return self._mh_reference(x_obs, n_samples, rng, burn_in=burn_in, thinning=thinning)

    ground_truth_posterior = reference_posterior_sample

    def _mh_reference(
        self,
        x_obs: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        *,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        n_steps: int = 4000,
        step_size: float = 0.03,
    ) -> np.ndarray:
        """Self-contained random-walk Metropolis-Hastings fallback."""
        x_obs = np.asarray(x_obs, dtype=float).reshape(-1)
        init = self.prior_sample(int(n_samples), rng)
        scales = np.concatenate([
            np.full(2, float(step_size)),
            np.full(self.n_index_points, float(step_size) * 5.0),
        ])
        log_post = lambda th: self.posterior_log_prob(np.atleast_2d(th), x_obs)  # noqa: E731
        current = np.array(init, dtype=float)
        logp = log_post(current)
        accept = 0
        total = 0
        burn = int(burn_in) if burn_in is not None else max(200, n_steps // 5)
        kept: List[np.ndarray] = []
        for step in range(int(n_steps)):
            total += 1
            prop = current + rng.standard_normal(current.shape) * scales
            logp_prop = log_post(prop)
            u = np.log(np.maximum(rng.random(current.shape), 1e-300))
            accept_mask = u < (logp_prop - logp)
            current = np.where(accept_mask[:, None], prop, current)
            logp = np.where(accept_mask, logp_prop, logp)
            accept += int(np.sum(accept_mask))
            if step >= burn and (step - burn) % max(int(thinning), 1) == 0:
                kept.append(current.copy())
        if not kept:  # pragma: no cover - degenerate settings
            kept = [current.copy()]
        samples = np.concatenate(kept, axis=0)
        if samples.shape[0] >= int(n_samples):
            idx = rng.choice(samples.shape[0], size=int(n_samples), replace=False)
            return samples[idx]
        reps = int(math.ceil(int(n_samples) / samples.shape[0]))
        return np.concatenate([samples] * reps, axis=0)[: int(n_samples)]

    def posterior_mean_std(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.reference_posterior_sample(x_obs, n_samples, rng, **kwargs)
        return samples.mean(axis=0), samples.std(axis=0)

    # ---------------------------------------------------------------- datasets
    def make_dataset(
        self,
        n_simulations: int,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        *,
        chunk_size: int = 512,
        verbose: bool = False,
        return_joint: bool = False,
        **kwargs: Any,
    ):
        """Generate ``(theta, x)`` (or joint) training data."""
        rng = _rng(rng if rng is not None else self._rng, seed)
        n = int(n_simulations)
        thetas: List[np.ndarray] = []
        xs: List[np.ndarray] = []
        done = 0
        while done < n:
            m = min(int(chunk_size), n - done)
            theta = self.prior_sample(m, rng)
            x = self.simulate(theta, rng, add_noise=True)
            thetas.append(theta)
            xs.append(x)
            done += m
            if verbose:
                print(f"[sird] simulated {done}/{n}", flush=True)
        theta_all = np.concatenate(thetas, axis=0)
        x_all = np.concatenate(xs, axis=0)
        if return_joint:
            return self.to_joint(theta_all, x_all)
        return theta_all, x_all

    # ---------------------------------------------------------------- scenarios
    def unstructured_observations(
        self,
        theta: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        series_times: Optional[Dict[str, Sequence[float]]] = None,
        series: Sequence[str] = ("I", "R", "D"),
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Simulate observations at per-series irregular time points.

        Returns ``(x, observed)`` where ``observed`` maps series name to
        ``(times, values)`` arrays.
        """
        rng = _rng(rng if rng is not None else self._rng, None)
        theta = np.atleast_2d(np.asarray(theta, dtype=float))[0]
        gamma, mu, beta_hat = theta[0], theta[1], theta[2:]
        times_map: Dict[str, np.ndarray] = {}
        default = np.asarray(self.observation_times, dtype=float)
        for s in series:
            t = np.asarray(series_times.get(s, default), dtype=float) if series_times else default
            times_map[str(s)] = np.sort(t)
        t_all = np.unique(np.concatenate([times_map[s] for s in series]))
        traj = integrate_sird(
            self.beta_function(beta_hat),
            float(gamma),
            float(mu),
            t_all,
            initial_state=self.config.initial_state,
        )
        series_col = {"S": 0, "I": 1, "R": 2, "D": 3}
        values: Dict[str, np.ndarray] = {}
        chunks: List[np.ndarray] = []
        for s in series:
            t = times_map[s]
            vals = np.interp(t, t_all, traj[:, series_col[str(s).upper()]])
            if add_noise:
                vals = lognormal_mean_std_sample(vals, self.observation_sigma, rng)
            values[str(s)] = vals
            chunks.append(vals)
        x = np.concatenate(chunks)
        return x, {"times": times_map, "values": values}  # type: ignore[dict-item]

    def simulate_scenario(
        self,
        scenario: Dict[str, Any],
        rng: Optional[np.random.Generator] = None,
        *,
        theta: Optional[np.ndarray] = None,
        n_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate one of the scenarios produced by :meth:`figure6_scenarios`."""
        rng = _rng(rng if rng is not None else self._rng, scenario.get("seed"))
        if theta is None:
            theta = scenario.get("theta")
            if theta is None:
                theta = self.prior_sample(1, rng)[0]
        theta = np.asarray(theta, dtype=float).reshape(-1)
        kind = scenario.get("kind", "infections")
        out: Dict[str, Any] = {"kind": kind, "theta": theta}
        if kind == "parameter_measurements":
            x, obs = self.unstructured_observations(theta, rng, series=("I",), add_noise=True)
            beta_idx = np.asarray(scenario.get("beta_indices", []), dtype=int)
            beta_true = sigmoid(np.interp(
                self.index_grid[beta_idx] if beta_idx.size else self.index_grid,
                self.index_grid,
                theta[2:],
            ))
            out.update(
                {
                    "x": x,
                    "observed": obs,
                    "beta_true": beta_true,
                    "condition_mask": self.measurement_condition_mask(beta_indices=beta_idx),
                }
            )
        else:
            x, obs = self.unstructured_observations(theta, rng, series=("I", "R", "D"), add_noise=True)
            out.update({"x": x, "observed": obs, "condition_mask": self.posterior_condition_mask()})
        return out

    def figure6_scenarios(self, seed: int = 0, **kwargs: Any) -> Dict[str, Dict[str, Any]]:
        """Structured description of the two panels of Fig. 6.

        * ``parameter_measurements``: four measurements of ``beta(t)`` and a single
          measurement of the infected population (Fig. 6b).
        * ``infections``: five observations of I, R and D at irregular times (Fig. 6a).
        """
        grid = self.index_grid
        beta_indices = np.linspace(0, grid.size - 1, 4).astype(int)
        return {
            "infections": {
                "kind": "infections",
                "series": tuple(self.config.series_names),
                "times": tuple(self.observation_times),
                "n_observations": int(self.n_observations),
                "seed": int(seed),
            },
            "parameter_measurements": {
                "kind": "parameter_measurements",
                "series": ("I",),
                "beta_indices": beta_indices,
                "beta_times": grid[beta_indices],
                "infected_observations": (0,),
                "seed": int(seed),
            },
        }

    # ---------------------------------------------------------------- predictive
    def posterior_predictive(
        self,
        theta_samples: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        add_noise: bool = True,
        n_samples: Optional[int] = None,
        times: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Data-space predictive samples ``(n, n_data)`` (no simulator calls needed
        on the evaluation side: this simply evaluates the SIRD dynamics)."""
        rng = _rng(rng if rng is not None else self._rng, None)
        theta = np.atleast_2d(np.asarray(theta_samples, dtype=float))
        if theta.shape[-1] == self.joint_dim:
            theta = self.theta_from_joint(theta)
        if n_samples is not None and theta.shape[0] > int(n_samples):
            idx = rng.choice(theta.shape[0], size=int(n_samples), replace=False)
            theta = theta[idx]
        return self.simulate(theta, rng, add_noise=add_noise, times=times)

    def contact_rate_samples(
        self,
        theta_samples: np.ndarray,
        times: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """``beta(t)`` values for each posterior sample -> ``(n, n_times)`` in [0, 1]."""
        grid = self.index_grid
        t = grid if times is None else np.asarray(times, dtype=float)
        beta_hat = np.atleast_2d(np.asarray(theta_samples, dtype=float))
        if beta_hat.shape[-1] == self.joint_dim:
            beta_hat = self.function_values_from_joint(beta_hat)
        else:
            beta_hat = beta_hat[:, self.n_scalar_parameters :]
        return beta_from_hat(beta_hat, grid, times=t, apply_sigmoid=True)

    def contact_rate_quantiles(
        self,
        theta_samples: np.ndarray,
        times: Optional[Sequence[float]] = None,
        quantiles: Sequence[float] = (0.005, 0.5, 0.995),
    ) -> Dict[str, np.ndarray]:
        """Per-time quantiles of the inferred contact rate (Fig. 6a)."""
        vals = self.contact_rate_samples(theta_samples, times=times)
        q = np.asarray(quantiles, dtype=float)
        out = np.quantile(vals, q, axis=0)
        return {"quantiles": q, "values": out, "mean": vals.mean(axis=0)}

    def infected_uncertainty(
        self,
        theta_samples: np.ndarray,
        times: Optional[Sequence[float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Inferred infected density and contact-rate std at each time.

        Used to check that contact-rate uncertainty is larger where the infected
        count is close to zero (Sec. 4.3, Fig. 6a).
        """
        t = self.observation_times if times is None else np.asarray(times, dtype=float)
        beta_q = self.contact_rate_quantiles(theta_samples, times=self.index_grid)
        vals = np.asarray(theta_samples, dtype=float)
        if vals.shape[-1] == self.joint_dim:
            vals = self.theta_from_joint(vals)
        infected = np.stack([self.trajectory(row, t)[:, 1] for row in vals], axis=0)
        return {
            "times": np.asarray(t, dtype=float),
            "infected_mean": infected.mean(axis=0),
            "infected_std": infected.std(axis=0),
            "beta_times": self.index_grid,
            "beta_mean": beta_q["mean"],
            "beta_std": self.contact_rate_samples(vals).std(axis=0),
        }

    # ---------------------------------------------------------------- plotting helpers
    def posterior_intervals(
        self,
        theta_samples: np.ndarray,
        *,
        level: float = 0.99,
    ) -> Dict[str, Dict[str, float]]:
        """Marginal credible intervals for gamma, mu and the beta grid values."""
        vals = np.atleast_2d(np.asarray(theta_samples, dtype=float))
        if vals.shape[-1] == self.joint_dim:
            vals = self.theta_from_joint(vals)
        lo_q = 0.5 * (1.0 - float(level))
        hi_q = 1.0 - lo_q
        out: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.config.parameter_names):
            out[str(name)] = {
                "mean": float(np.mean(vals[:, i])),
                "std": float(np.std(vals[:, i])),
                "lower": float(np.quantile(vals[:, i], lo_q)),
                "upper": float(np.quantile(vals[:, i], hi_q)),
            }
        beta = self.contact_rate_samples(vals)
        out["beta"] = {
            "mean": float(np.mean(beta)),
            "std": float(np.std(beta)),
            "lower": float(np.quantile(beta, lo_q)),
            "upper": float(np.quantile(beta, hi_q)),
        }
        return out

    # ---------------------------------------------------------------- simformer plumbing
    def spec(self, token_dim: int = 50, *, n_index_points: Optional[int] = None, **kwargs: Any):
        """Tokenizer specification (function-valued ``beta`` with time indices)."""
        try:
            from ..tokenizer import FunctionValuedSpec, TokenSpec  # type: ignore
        except Exception:  # pragma: no cover
            from simformer.tokenizer import FunctionValuedSpec, TokenSpec  # type: ignore

        grid = self.config.masked_beta_grid(n_index_points)
        data_names = tuple(
            f"{s}_{i}" for s in self.config.series_names for i in range(self.n_observations)
        )
        return TokenSpec(
            parameter_names=tuple(self.config.parameter_names),
            data_names=data_names,
            function_valued=(FunctionValuedSpec(name="beta", index_set=grid),),
            metadata_dim=int(kwargs.get("metadata_dim", 0)),
        )

    def token_spec(self, token_dim: int = 50, **kwargs: Any):
        return self.spec(token_dim=token_dim, **kwargs)

    def metadata_times(self, n_index_points: Optional[int] = None) -> np.ndarray:
        return self.config.masked_beta_grid(n_index_points)

    def data_index(self, series: str, observation_index: int) -> int:
        """Flat data index of ``series`` at ``observation_index``."""
        return int(self.index_of(series, observation_index)[0])

    def time_index_of(self, t: float) -> int:
        times = list(self.observation_times)
        if float(t) not in times:
            raise ValueError(f"time {t} is not one of the observation times {times}")
        return int(times.index(float(t)))

    def observation_series(self, x: np.ndarray, series: str) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return x[:, self.index_of(series, None)]

    def attention_mask(self, directed: bool = True, n_index_points: Optional[int] = None, **kwargs: Any) -> np.ndarray:
        """Directed (or symmetrised) base mask ``M_E`` for the SIRD task."""
        n_theta = int(self.n_parameters)
        n_x = int(self.n_data)
        grid = self.config.masked_beta_grid(n_index_points)
        try:  # pragma: no cover - depends on mask module
            from ..attention_masks import build_attention_mask  # type: ignore

            mask = build_attention_mask(
                "sird",
                n_theta=n_theta,
                n_x=n_x,
                n_series=int(self.n_series),
                n_times=int(grid.size),
                n_index_points=int(grid.size),
                directed=bool(directed),
            )
            mask = np.asarray(mask)
            if mask.ndim == 2 and mask.shape[0] == self._n_tokens(grid.size):
                return mask.astype(bool)
        except Exception:
            pass
        try:  # pragma: no cover
            from simformer.attention_masks import build_attention_mask  # type: ignore

            mask = np.asarray(
                build_attention_mask(
                    "sird",
                    n_theta=n_theta,
                    n_x=n_x,
                    n_series=int(self.n_series),
                    n_times=int(grid.size),
                    n_index_points=int(grid.size),
                    directed=bool(directed),
                )
            )
            if mask.ndim == 2 and mask.shape[0] == self._n_tokens(grid.size):
                return mask.astype(bool)
        except Exception:
            pass
        return self._fallback_mask(directed=directed, n_index_points=grid.size)

    def _n_tokens(self, n_index_points: int) -> int:
        return int(self.n_scalar_parameters + self.n_data + int(n_index_points))

    def _fallback_mask(self, directed: bool = True, n_index_points: Optional[int] = None) -> np.ndarray:
        """Directed base ``M_E``: params/beta are roots, data depends on all of them.

        Token order follows the tokenizer: ``[gamma, mu | data... | beta]``.
        ``M[i, j] = 1`` means query ``i`` attends key ``j`` (edge ``j -> i``);
        the diagonal is always true.
        """
        k = int(self.n_index_points if n_index_points is None else n_index_points)
        n_scalar = self.n_scalar_parameters
        n_tokens = self._n_tokens(k)
        mask = np.zeros((n_tokens, n_tokens), dtype=bool)
        np.fill_diagonal(mask, True)
        beta_start = n_scalar + self.n_data
        # beta values are a smooth (fully correlated) GP -> dense within the beta block
        if k > 0:
            mask[beta_start:, beta_start:] = True
        # data tokens depend on the global rates and the full contact-rate trajectory
        mask[n_scalar:beta_start, :n_scalar] = True
        if k > 0:
            mask[n_scalar:beta_start, beta_start:] = True
        mask = mask | np.eye(n_tokens, dtype=bool)
        if directed:
            return mask
        return mask | mask.T

    def attention_mask_variants(self, n_index_points: Optional[int] = None, **kwargs: Any) -> Dict[str, np.ndarray]:
        """``{"dense", "undirected", "directed"}`` mask variants (Fig. 4 protocol)."""
        grid = self.config.masked_beta_grid(n_index_points)
        n_tokens = self._n_tokens(grid.size)
        try:  # pragma: no cover
            from ..attention_masks import mask_variants  # type: ignore

            variants = mask_variants("sird", n_theta=int(self.n_parameters), n_x=int(self.n_data), **kwargs)
            out = {k: np.asarray(v).astype(bool) for k, v in dict(variants).items()}
            if all(v.shape[0] == n_tokens for v in out.values()):
                return out
        except Exception:
            pass
        dense = np.ones((n_tokens, n_tokens), dtype=bool)
        undirected = self._fallback_mask(directed=False, n_index_points=grid.size)
        directed = self._fallback_mask(directed=True, n_index_points=grid.size)
        return {"dense": dense, "undirected": undirected, "directed": directed}

    def build_tokenizer(self, token_dim: int = 50, *, n_index_points: Optional[int] = None, **kwargs: Any):
        """Build a :class:`~simformer.tokenizer.Tokenizer` for this task."""
        try:
            from ..tokenizer import Tokenizer  # type: ignore
        except Exception:  # pragma: no cover
            from simformer.tokenizer import Tokenizer  # type: ignore

        return Tokenizer(self.spec(token_dim=token_dim, n_index_points=n_index_points), token_dim=token_dim)

    def build_model(self, **kwargs: Any):
        """Build a Simformer score network wired to this task's mask and spec."""
        try:
            from ..transformer import build_score_network  # type: ignore
        except Exception:  # pragma: no cover
            from simformer.transformer import build_score_network  # type: ignore

        n_index_points = kwargs.pop("n_index_points", None)
        token_dim = int(kwargs.pop("token_dim", 50))
        directed = bool(kwargs.pop("directed", True))
        kwargs.setdefault("task", "sird")
        mask = kwargs.pop("attention_mask", None)
        if mask is None:
            try:
                mask = self.attention_mask(directed=directed, n_index_points=n_index_points)
            except Exception:
                mask = None
        spec = self.spec(token_dim=token_dim, n_index_points=n_index_points)
        try:
            model = build_score_network(spec=spec, token_dim=token_dim, **kwargs)
        except TypeError:
            model = build_score_network(spec=spec, **kwargs)
        if mask is not None:
            for attr in ("attention_mask", "mask"):
                if hasattr(model, attr):
                    try:
                        setattr(model, attr, mask)
                        break
                    except Exception:  # pragma: no cover
                        continue
        return model

    # ---------------------------------------------------------------- misc
    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d.update(
            {
                "joint_dim": int(self.joint_dim),
                "theta_dim": int(self.theta_dim),
                "data_dim": int(self.n_data),
                "n_tokens": int(self._n_tokens(self.n_index_points)),
            }
        )
        return d

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SIRDTask(n_series={self.n_series}, n_obs={self.n_observations}, "
            f"n_index_points={self.n_index_points}, sigma={self.observation_sigma})"
        )


# --------------------------------------------------------------------------------------
# module-level convenience wrappers (cached default task)
# --------------------------------------------------------------------------------------
_DEFAULT_TASK: Optional[SIRDTask] = None


def _default_task() -> SIRDTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = SIRDTask()
    return _DEFAULT_TASK


def prior_sample(n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().prior_sample(n_samples, rng, **kwargs)


def log_prior(theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_prior(theta, **kwargs)


def simulate(theta: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().simulate(theta, rng, **kwargs)


def log_likelihood(x: np.ndarray, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_likelihood(x, theta, **kwargs)


def log_joint(theta: np.ndarray, x: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_joint(theta, x, **kwargs)


def posterior_log_prob(theta: np.ndarray, x_obs: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().posterior_log_prob(theta, x_obs, **kwargs)


def reference_posterior_sample(
    x_obs: np.ndarray,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples, rng, **kwargs)


def make_dataset(n_simulations: int, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None, **kwargs: Any):
    return _default_task().make_dataset(n_simulations, rng, seed, **kwargs)


def build_task(config: Optional[Union[SIRDConfig, Dict[str, Any]]] = None, **kwargs: Any) -> SIRDTask:
    """Factory used by the task registry."""
    return SIRDTask(config, **kwargs)


# aliases matching the conventions of the other task modules
Task = SIRDTask
Simulator = SIRDTask
SIRD = SIRDTask
SIRDModel = SIRDTask


__all__ = [
    "SIRDConfig",
    "SIRDTask",
    "Task",
    "Simulator",
    "SIRD",
    "SIRDModel",
    "build_task",
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "log_joint",
    "posterior_log_prob",
    "reference_posterior_sample",
    "make_dataset",
    "sird_rhs",
    "integrate_sird",
    "rbf_kernel",
    "gp_covariance",
    "sample_gp",
    "log_gp_prior",
    "sigmoid",
    "beta_from_hat",
    "lognormal_mean_std_logpdf",
    "lognormal_mean_std_sample",
    "DEFAULT_OBSERVATION_TIMES",
    "DEFAULT_OBSERVATION_SIGMA",
    "DEFAULT_T_MAX",
    "DEFAULT_N_INDEX_POINTS",
    "DEFAULT_GP_LENGTHSCALE",
    "DEFAULT_GP_AMPLITUDE",
]
