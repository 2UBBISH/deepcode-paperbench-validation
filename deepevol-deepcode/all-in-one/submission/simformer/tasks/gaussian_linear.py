"""Gaussian Linear benchmark task (Simformer, Appendix Sec. A2.2).

The task (Lueckmann et al., 2021) is a fully factorized linear-Gaussian model::

    theta ~ N(0, 0.1 * I)                (prior)
    x | theta ~ N(theta, 0.1 * I)       (likelihood)

with ``theta, x in R^10``.  Because the model is conjugate and diagonal, the
posterior is available in closed form::

    p(theta | x_obs) = N(theta ; x_obs / 2, 0.05 * I)

which we expose as :meth:`GaussianLinearTask.posterior_mean_std` and
:meth:`GaussianLinearTask.reference_posterior_sample` for evaluation.

The dependency structure is factorized across dimensions, which the directed
attention mask in :mod:`simformer.attention_masks` encodes with 2x2 blocks
``[[1, 1], [0, 1]]`` per dimension (Addendum, "Task Dependencies").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - package import guard
    from . import TaskBase
except Exception:  # pragma: no cover - script-like import
    try:
        from simformer.tasks import TaskBase  # type: ignore
    except Exception:  # pragma: no cover

        class TaskBase:  # type: ignore
            """Minimal fallback if the tasks package is unavailable."""

            name = "task"
            n_parameters = 1
            n_data = 1
            parameter_names: Tuple[str, ...] = ()
            data_names: Tuple[str, ...] = ()

            def to_joint(self, theta, x):
                theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
                x = np.atleast_2d(np.asarray(x, dtype=np.float64))
                return np.concatenate([theta, x], axis=-1)

            def make_dataset(self, n_simulations, *, rng=None, chunk_size=1024, **kw):
                rng = np.random.default_rng(rng)
                theta = self.prior_sample(n_simulations, rng)
                x = self.simulate(theta, rng)
                return theta, x


__all__ = [
    "GaussianLinearConfig",
    "GaussianLinearTask",
    "build_task",
    "Task",
    "Simulator",
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "posterior_mean_std",
    "reference_posterior_sample",
    "make_dataset",
    "DEFAULT_DIM",
    "DEFAULT_PRIOR_STD",
    "DEFAULT_NOISE_STD",
]

# ---------------------------------------------------------------------------
# Constants (Appendix A2.2)
# ---------------------------------------------------------------------------
DEFAULT_DIM: int = 10
DEFAULT_PRIOR_STD: float = 0.1
DEFAULT_NOISE_STD: float = 0.1
_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GaussianLinearConfig:
    """Configuration of the Gaussian Linear task."""

    n_dim: int = DEFAULT_DIM
    prior_mean: float = 0.0
    prior_std: float = DEFAULT_PRIOR_STD
    noise_std: float = DEFAULT_NOISE_STD
    name: str = "gaussian_linear"
    seed: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_dim": self.n_dim,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "noise_std": self.noise_std,
            "name": self.name,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "GaussianLinearConfig":
        if not cfg:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @property
    def posterior_variance(self) -> float:
        """Closed-form posterior variance ``1 / (1/prior_std^2 + 1/noise_std^2)``."""
        prec = 1.0 / self.prior_std**2 + 1.0 / self.noise_std**2
        return 1.0 / prec


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------
def _as_2d_theta(theta: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(theta, dtype=np.float64)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]
    return arr, squeeze


def _rng(rng: Union[np.random.Generator, int, None] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _diag_gaussian_logpdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    """Log density of ``N(x; mean, std^2 I)`` summed over the last axis."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    var = float(std) ** 2
    diff = x - mean
    return -0.5 * (np.sum(diff * diff, axis=-1) / var + x.shape[-1] * (_LOG_2PI + math.log(var)))


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class GaussianLinearTask(TaskBase):
    """Gaussian Linear benchmark simulator with closed-form posterior."""

    name = "gaussian_linear"
    n_parameters = DEFAULT_DIM
    n_data = DEFAULT_DIM
    parameter_names = tuple(f"theta_{i}" for i in range(DEFAULT_DIM))
    data_names = tuple(f"x_{i}" for i in range(DEFAULT_DIM))

    def __init__(
        self,
        config: Optional[GaussianLinearConfig] = None,
        *,
        n_dim: Optional[int] = None,
        prior_std: Optional[float] = None,
        noise_std: Optional[float] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        cfg = config or GaussianLinearConfig()
        if isinstance(config, dict):  # allow raw dicts
            cfg = GaussianLinearConfig.from_dict(config)
        if n_dim is not None:
            cfg.n_dim = int(n_dim)
        if prior_std is not None:
            cfg.prior_std = float(prior_std)
        if noise_std is not None:
            cfg.noise_std = float(noise_std)
        if name is not None:
            cfg.name = str(name)
        if seed is not None:
            cfg.seed = int(seed)
        # allow extra kwargs from a hydra config
        for key, value in kwargs.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        self.config = cfg
        self.n_parameters = int(cfg.n_dim)
        self.n_data = int(cfg.n_dim)
        self.parameter_names = tuple(f"theta_{i}" for i in range(self.n_parameters))
        self.data_names = tuple(f"x_{i}" for i in range(self.n_data))

    # -- properties --------------------------------------------------------
    @property
    def theta_dim(self) -> int:
        return self.n_parameters

    @property
    def data_dim(self) -> int:
        return self.n_data

    @property
    def joint_dim(self) -> int:
        return self.n_parameters + self.n_data

    @property
    def prior_std(self) -> float:
        return float(self.config.prior_std)

    @property
    def noise_std(self) -> float:
        return float(self.config.noise_std)

    # -- prior -------------------------------------------------------------
    def prior_sample(
        self,
        n_samples: int = 1,
        rng: Union[np.random.Generator, int, None] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Draw ``n_samples`` parameters from ``N(0, prior_std^2 I)``."""
        rng = _rng(rng if rng is not None else self.config.seed)
        n_samples = int(n_samples)
        return rng.normal(
            loc=self.config.prior_mean,
            scale=self.config.prior_std,
            size=(n_samples, self.n_parameters),
        )

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Log prior density (summed over dimensions)."""
        return _diag_gaussian_logpdf(theta, self.config.prior_mean, self.config.prior_std)

    def sample_prior(self, n_samples: int = 1, rng=None) -> np.ndarray:
        """Alias of :meth:`prior_sample`."""
        return self.prior_sample(n_samples, rng)

    # -- simulator ---------------------------------------------------------
    def simulate(
        self,
        theta: np.ndarray,
        rng: Union[np.random.Generator, int, None] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Sample ``x | theta ~ N(theta, noise_std^2 I)``.

        Parameters
        ----------
        theta:
            ``(n, d)`` or ``(d,)`` parameter array.
        rng:
            NumPy generator (or seed).
        n_samples:
            If ``theta`` is a single ``(d,)`` vector, number of data samples to
            draw from that parameter (result keeps the leading sample axis).

        Returns
        -------
        ``(n, d)`` data array (``(d,)`` when a single ``(d,)`` theta was given
        and ``n_samples`` is ``None``).
        """
        rng = _rng(rng if rng is not None else (seed if seed is not None else self.config.seed))
        theta_2d, squeeze = _as_2d_theta(theta)

        if n_samples is not None and squeeze:
            theta_2d = np.repeat(theta_2d, int(n_samples), axis=0)
            squeeze = False

        mean = theta_2d
        if not add_noise:
            x = mean
        else:
            x = mean + rng.normal(
                scale=self.config.noise_std, size=mean.shape
            )
        return x[0] if squeeze else x

    def simulator(self, theta: np.ndarray, rng=None, **kwargs: Any) -> np.ndarray:
        """Alias of :meth:`simulate` (sbi-style naming)."""
        return self.simulate(theta, rng, **kwargs)

    def __call__(self, n_samples: int = 1, rng=None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Draw a joint batch ``(theta, x)``."""
        rng = _rng(rng if rng is not None else self.config.seed)
        theta = self.prior_sample(int(n_samples), rng)
        x = self.simulate(theta, rng, **kwargs)
        return theta, x

    def log_likelihood(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Log likelihood ``log p(x | theta)`` (summed over dimensions)."""
        return _diag_gaussian_logpdf(x, theta, self.config.noise_std)

    # -- exact posterior ---------------------------------------------------
    def posterior_mean_std(self, x_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Closed-form posterior ``mean`` and ``std`` (per dimension)."""
        x_obs = np.asarray(x_obs, dtype=np.float64)
        prior_prec = 1.0 / self.config.prior_std**2
        noise_prec = 1.0 / self.config.noise_std**2
        prec = prior_prec + noise_prec
        mean = (prior_prec * self.config.prior_mean + noise_prec * x_obs) / prec
        std = np.full_like(mean, 1.0 / math.sqrt(prec))
        return mean, std

    def posterior_variance(self) -> float:
        prec = 1.0 / self.config.prior_std**2 + 1.0 / self.config.noise_std**2
        return 1.0 / prec

    def reference_posterior_sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Union[np.random.Generator, int, None] = None,
    ) -> np.ndarray:
        """Exact posterior samples ``(n_samples, d)`` for an observation."""
        rng = _rng(rng if rng is not None else self.config.seed)
        mean, std = self.posterior_mean_std(x_obs)
        return rng.normal(loc=mean, scale=std, size=(int(n_samples), self.n_parameters))

    # ground-truth alias used by evaluation code
    ground_truth_posterior = reference_posterior_sample

    def ground_truth_log_posterior(self, theta: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        """Unnormalized log posterior (used by MCMC references)."""
        mean, std = self.posterior_mean_std(x_obs)
        return _diag_gaussian_logpdf(theta, mean, float(np.asarray(std).reshape(-1)[0]))

    def posterior_log_prob(self, theta: np.ndarray, x_obs: np.ndarray) -> np.ndarray:
        """Alias of :meth:`ground_truth_log_posterior`."""
        return self.ground_truth_log_posterior(theta, x_obs)

    # -- dataset helpers ---------------------------------------------------
    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Union[np.random.Generator, int, None] = None,
        seed: Optional[int] = None,
        chunk_size: int = 1024,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate ``n_simulations`` joint ``(theta, x)`` pairs in chunks."""
        rng = _rng(rng if rng is not None else (seed if seed is not None else self.config.seed))
        n_simulations = int(n_simulations)
        chunk = max(1, int(chunk_size))
        thetas, xs = [], []
        remaining = n_simulations
        while remaining > 0:
            m = min(chunk, remaining)
            theta = self.prior_sample(m, rng)
            x = self.simulate(theta, rng)
            thetas.append(theta)
            xs.append(x)
            remaining -= m
        if verbose:  # pragma: no cover - informational
            print(f"[{self.name}] simulated {n_simulations} samples")
        return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)

    def sample_joint(self, n_samples: int = 1, rng=None) -> Tuple[np.ndarray, np.ndarray]:
        """Alias of :meth:`__call__`."""
        return self.__call__(n_samples, rng)

    # -- tokenizer / model plumbing ---------------------------------------
    def spec(self, **kwargs: Any):
        """Return a :class:`~simformer.tokenizer.TokenSpec` for this task."""
        try:
            from simformer.tokenizer import build_benchmark_spec
        except Exception:  # pragma: no cover
            from ..tokenizer import build_benchmark_spec  # type: ignore
        return build_benchmark_spec(self.n_parameters, self.n_data)

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        """Directed (or symmetrized) Gaussian Linear attention mask."""
        try:
            from simformer.attention_masks import build_attention_mask
        except Exception:  # pragma: no cover
            from ..attention_masks import build_attention_mask  # type: ignore
        return build_attention_mask(
            "gaussian_linear",
            n_theta=self.n_parameters,
            n_x=self.n_data,
            directed=directed,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d.update(
            {
                "n_parameters": self.n_parameters,
                "n_data": self.n_data,
                "parameter_names": list(self.parameter_names),
                "data_names": list(self.data_names),
            }
        )
        return d

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"GaussianLinearTask(n_dim={self.n_parameters}, "
            f"prior_std={self.config.prior_std}, noise_std={self.config.noise_std})"
        )


# ---------------------------------------------------------------------------
# Convenience aliases / module-level functional API
# ---------------------------------------------------------------------------
Task = GaussianLinearTask
Simulator = GaussianLinearTask
GaussianLinear = GaussianLinearTask

_DEFAULT_TASK = GaussianLinearTask()


def prior_sample(n_samples: int = 1, rng=None, **kwargs: Any) -> np.ndarray:
    """Module-level prior sampling (uses defaults from Appendix A2.2)."""
    return _DEFAULT_TASK.prior_sample(n_samples, rng)


def log_prior(theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _DEFAULT_TASK.log_prior(theta)


def simulate(theta: np.ndarray, rng=None, **kwargs: Any) -> np.ndarray:
    return _DEFAULT_TASK.simulate(theta, rng, **kwargs)


def log_likelihood(x: np.ndarray, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _DEFAULT_TASK.log_likelihood(x, theta)


def posterior_mean_std(x_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return _DEFAULT_TASK.posterior_mean_std(x_obs)


def reference_posterior_sample(x_obs: np.ndarray, n_samples: int = 1000, rng=None) -> np.ndarray:
    return _DEFAULT_TASK.reference_posterior_sample(x_obs, n_samples, rng)


def make_dataset(
    n_simulations: int, *, rng=None, seed=None, chunk_size: int = 1024, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    return _DEFAULT_TASK.make_dataset(
        n_simulations, rng=rng, seed=seed, chunk_size=chunk_size, **kwargs
    )


def build_task(
    config: Optional[Union[GaussianLinearConfig, Dict[str, Any]]] = None, **kwargs: Any
) -> GaussianLinearTask:
    """Factory used by :func:`simformer.tasks.build_task`."""
    if isinstance(config, dict):
        cfg = GaussianLinearConfig.from_dict(config)
        return GaussianLinearTask(cfg, **kwargs)
    return GaussianLinearTask(config, **kwargs)


if __name__ == "__main__":  # pragma: no cover - smoke test
    task = build_task()
    rng = np.random.default_rng(0)
    theta, x = task(4, rng)
    print("theta", theta.shape, "x", x.shape)
    print("posterior mean", task.posterior_mean_std(x[0])[0].shape)
