"""Gaussian Mixture benchmark task (Simformer, Appendix A2.2).

The task (Sisson et al., 2007; Beaumont et al., 2009) asks for the common mean of
a two-dimensional mixture of Gaussians with distinct covariances::

    theta ~ U(-10, 10)^2
    x | theta ~ 0.5 * N(x ; theta, I) + 0.5 * N(x ; theta, 0.01 * I)

with ``theta, x in R^2``.

Because both mixture components are Gaussians in ``x`` centred at ``theta``, the
likelihood seen as a function of ``theta`` is a mixture of two Gaussians centred at
the observation ``x`` with covariances ``I`` and ``0.01 * I``.  After multiplying by
the (flat) uniform prior and restricting to the box ``[-10, 10]^2``, the posterior is
therefore an *exact* two-component Gaussian mixture truncated to that box -- which
lets us provide closed-form ground truth and exact reference posterior samples.

This file mirrors the API of :mod:`simformer.tasks.gaussian_linear` so that all
training / sampling / evaluation plumbing treats every task identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - package import style
    from . import TaskBase  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    try:
        from simformer.tasks import TaskBase  # type: ignore
    except Exception:

        class TaskBase:  # type: ignore
            """Minimal fallback base class when the task package is unavailable."""

            name = "task"
            n_parameters = 0
            n_data = 0

            def to_dict(self) -> Dict[str, Any]:
                return {"name": getattr(self, "name", "task")}


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
DEFAULT_DIM = 2
DEFAULT_PRIOR_LOW = -10.0
DEFAULT_PRIOR_HIGH = 10.0
DEFAULT_COMPONENT_STD = 1.0
DEFAULT_COMPONENT_STD_SMALL = 0.1  # std of the narrow component (variance 0.01)
DEFAULT_COMPONENT_WEIGHT = 0.5

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_WEIGHT = math.log(DEFAULT_COMPONENT_WEIGHT)

__all__ = [
    "GaussianMixtureConfig",
    "GaussianMixtureTask",
    "Task",
    "Simulator",
    "GaussianMixture",
    "build_task",
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "reference_posterior_sample",
    "make_dataset",
    "DEFAULT_DIM",
    "DEFAULT_PRIOR_LOW",
    "DEFAULT_PRIOR_HIGH",
]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _rng(rng: Optional[Union[np.random.Generator, int]] = None, seed: Optional[int] = None) -> np.random.Generator:
    """Return a ``numpy`` Generator from ``rng``/``seed`` (default seed 0)."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        rng = seed
    if rng is None:
        rng = 0
    if isinstance(rng, np.random.RandomState):  # legacy support
        return np.random.default_rng(rng.randint(0, 2 ** 31 - 1))
    return np.random.default_rng(int(rng))


def _as_2d_theta(theta: Union[np.ndarray, Sequence[float]]) -> Tuple[np.ndarray, bool]:
    """Coerce ``theta`` to a 2-D array, reporting whether the input was 1-D."""
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def _diag_gaussian_logpdf(
    x: np.ndarray,
    mean: np.ndarray,
    std: Union[float, np.ndarray],
) -> np.ndarray:
    """Log density of a diagonal Gaussian, summed over the last axis."""
    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    var = std ** 2
    diff = x - mean
    return -0.5 * (np.sum(diff ** 2 / var, axis=-1) + np.sum(np.log(2.0 * np.pi * var), axis=-1))


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a, axis=axis, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    return np.squeeze(amax, axis=axis) + np.log(np.sum(np.exp(a - amax), axis=axis))


def _gaussian_box_mass(mean: np.ndarray, std: Union[float, np.ndarray], low: float, high: float) -> np.ndarray:
    """Probability mass of a diagonal Gaussian inside a hyper-rectangle.

    Computed with ``erf`` for numerical stability; returns an array matching the
    leading dimensions of ``mean`` (last axis is absorbed).
    """
    from math import erf  # noqa: F401  (kept for clarity of the intent)

    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    # Phi((b - mu)/sigma) - Phi((a - mu)/sigma)
    from scipy.special import ndtr  # type: ignore

    upper = ndtr((high - mean) / std)
    lower = ndtr((low - mean) / std)
    per_dim = np.clip(upper - lower, 0.0, 1.0)
    return np.prod(per_dim, axis=-1)


def _is_log_density(x: np.ndarray) -> bool:
    return np.asarray(x).ndim >= 1


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class GaussianMixtureConfig:
    """Configuration of the Gaussian Mixture benchmark task.

    Parameters
    ----------
    n_dim:
        Dimensionality of ``theta`` and ``x`` (paper uses 2).
    prior_low, prior_high:
        Bounds of the uniform prior ``U(-10, 10)``.
    component_std:
        Standard deviation of the broad mixture component (variance ``1``).
    component_std_small:
        Standard deviation of the narrow mixture component (variance ``0.01``).
    component_weight:
        Weight of the broad component (paper: 0.5).
    seed:
        Default RNG seed used when no generator is supplied.
    """

    n_dim: int = DEFAULT_DIM
    prior_low: float = DEFAULT_PRIOR_LOW
    prior_high: float = DEFAULT_PRIOR_HIGH
    component_std: float = DEFAULT_COMPONENT_STD
    component_std_small: float = DEFAULT_COMPONENT_STD_SMALL
    component_weight: float = DEFAULT_COMPONENT_WEIGHT
    name: str = "gaussian_mixture"
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived quantities ---------------------------------------------------------
    @property
    def component_variances(self) -> np.ndarray:
        return np.array([self.component_std ** 2, self.component_std_small ** 2], dtype=np.float64)

    @property
    def component_weights(self) -> np.ndarray:
        w = float(np.clip(self.component_weight, 0.0, 1.0))
        return np.array([w, 1.0 - w], dtype=np.float64)

    @property
    def prior_volume(self) -> float:
        return float((self.prior_high - self.prior_low) ** self.n_dim)

    # -- (de)serialisation ----------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_dim": self.n_dim,
            "prior_low": self.prior_low,
            "prior_high": self.prior_high,
            "component_std": self.component_std,
            "component_std_small": self.component_std_small,
            "component_weight": self.component_weight,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "GaussianMixtureConfig"]] = None, **kwargs: Any) -> "GaussianMixtureConfig":
        if cfg is None:
            cfg = {}
        if isinstance(cfg, GaussianMixtureConfig):
            data = cfg.to_dict()
            data.update(kwargs)
            cfg = data
        if not isinstance(cfg, dict):
            raise TypeError(f"Unsupported config type: {type(cfg)!r}")
        allowed = {
            "n_dim",
            "prior_low",
            "prior_high",
            "component_std",
            "component_std_small",
            "component_weight",
            "name",
            "seed",
            "extra",
        }
        data = {k: v for k, v in cfg.items() if k in allowed}
        extra = {k: v for k, v in cfg.items() if k not in allowed}
        if extra:
            data.setdefault("extra", {}).update(extra)
        data.update({k: v for k, v in kwargs.items() if k in allowed})
        return cls(**data)


# --------------------------------------------------------------------------------------
# task
# --------------------------------------------------------------------------------------
class GaussianMixtureTask(TaskBase):
    """Two-dimensional Gaussian-mixture SBI task with exact posterior utilities."""

    name = "gaussian_mixture"
    n_parameters = DEFAULT_DIM
    n_data = DEFAULT_DIM
    parameter_names = ("theta_1", "theta_2")
    data_names = ("x_1", "x_2")

    def __init__(
        self,
        config: Optional[Union[GaussianMixtureConfig, Dict[str, Any]]] = None,
        n_dim: Optional[int] = None,
        prior_low: Optional[float] = None,
        prior_high: Optional[float] = None,
        component_std: Optional[float] = None,
        component_std_small: Optional[float] = None,
        component_weight: Optional[float] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        cfg = GaussianMixtureConfig.from_dict(config) if config is not None else GaussianMixtureConfig()
        overrides: Dict[str, Any] = {}
        for key, value in (
            ("n_dim", n_dim),
            ("prior_low", prior_low),
            ("prior_high", prior_high),
            ("component_std", component_std),
            ("component_std_small", component_std_small),
            ("component_weight", component_weight),
            ("name", name),
            ("seed", seed),
        ):
            if value is not None:
                overrides[key] = value
        if kwargs:
            overrides.setdefault("extra", {}).update(kwargs)
        cfg = GaussianMixtureConfig.from_dict(cfg.to_dict(), **overrides)
        if "extra" in overrides:
            cfg.extra.update(overrides["extra"])

        self.config = cfg
        self.n_dim = int(cfg.n_dim)
        self.prior_low = float(cfg.prior_low)
        self.prior_high = float(cfg.prior_high)
        self.component_std = float(cfg.component_std)
        self.component_std_small = float(cfg.component_std_small)
        self.component_weight = float(cfg.component_weight)
        self.name = str(cfg.name)
        self.seed = int(cfg.seed)
        self.parameter_names = tuple(f"theta_{i + 1}" for i in range(self.n_dim))
        self.data_names = tuple(f"x_{i + 1}" for i in range(self.n_dim))
        self.n_parameters = self.n_dim
        self.n_data = self.n_dim

    # -- priors ---------------------------------------------------------------------
    @property
    def prior_volume(self) -> float:
        return float((self.prior_high - self.prior_low) ** self.n_dim)

    def prior_sample(self, n_samples: int = 1, rng: Optional[Union[np.random.Generator, int]] = None) -> np.ndarray:
        """Draw ``n_samples`` parameters uniformly from ``U(-10, 10)^{n_dim}``."""
        rng = _rng(rng, self.seed)
        n_samples = int(n_samples)
        return rng.uniform(self.prior_low, self.prior_high, size=(n_samples, self.n_dim))

    #: sbi-style alias
    sample_prior = prior_sample

    def log_prior(self, theta: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Log density of the uniform prior; ``-inf`` outside the box."""
        theta = np.asarray(theta, dtype=np.float64)
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high), axis=-1)
        log_norm = -math.log(self.prior_high - self.prior_low) * self.n_dim
        return np.where(inside, log_norm, -np.inf)

    # -- simulator ------------------------------------------------------------------
    def sample_component(self, theta: Union[np.ndarray, Sequence[float]], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Draw a data point per theta from the two-component mixture.

        Returns the samples and the (integer) component indicators.
        """
        theta2, _ = _as_2d_theta(theta)
        n = theta2.shape[0]
        w = np.clip(self.component_weight, 0.0, 1.0)
        component = (rng.random(n) >= w).astype(np.int64)  # 0 = broad, 1 = narrow
        std = np.where(component == 0, self.component_std, self.component_std_small)[:, None]
        x = theta2 + std * rng.standard_normal((n, self.n_dim))
        return x, component

    def simulate(
        self,
        theta: Union[np.ndarray, Sequence[float]],
        rng: Optional[Union[np.random.Generator, int]] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        return_components: bool = False,
    ) -> np.ndarray:
        """Simulate data ``x`` for (a batch of) parameters ``theta``."""
        rng = _rng(rng, seed if seed is not None else self.seed)
        theta2, was_1d = _as_2d_theta(theta)
        if not add_noise:
            out = np.repeat(theta2, int(n_samples or 1), axis=0) if n_samples else theta2
            return out[0] if was_1d else out
        if n_samples is not None and n_samples > 1:
            theta2 = np.repeat(theta2, int(n_samples), axis=0)
            was_1d = False
        x, components = self.sample_component(theta2, rng)
        if return_components:
            return (x, components)
        return x[0] if was_1d else x

    #: sbi-style alias
    def simulator(self, theta: Union[np.ndarray, Sequence[float]], rng: Optional[Union[np.random.Generator, int]] = None, **kwargs: Any) -> np.ndarray:
        return self.simulate(theta, rng, **kwargs)

    def __call__(self, n_samples: int = 1, rng: Optional[Union[np.random.Generator, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Draw a joint pair ``(theta, x)`` of shape ``(n_samples, n_dim)``."""
        rng = _rng(rng, self.seed)
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng)
        return theta, x

    # -- likelihood / posterior -----------------------------------------------------
    def log_likelihood(self, x: Union[np.ndarray, Sequence[float]], theta: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Log of the mixture likelihood ``0.5 N(x; theta, I) + 0.5 N(x; theta, 0.01 I)``.

        ``x`` and ``theta`` broadcast against each other on all axes.
        """
        x = np.asarray(x, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        w = self.config.component_weights
        comps = np.stack(
            [
                _diag_gaussian_logpdf(x, theta, self.component_std),
                _diag_gaussian_logpdf(x, theta, self.component_std_small),
            ],
            axis=0,
        )
        comps = comps + np.log(w)[:, None]
        return _logsumexp(comps, axis=0)

    def log_likelihood_components(
        self,
        x: Union[np.ndarray, Sequence[float]],
        theta: Union[np.ndarray, Sequence[float]],
    ) -> np.ndarray:
        """Per-component log densities, shape ``(2, ...)``."""
        x = np.asarray(x, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        return np.stack(
            [
                _diag_gaussian_logpdf(x, theta, self.component_std),
                _diag_gaussian_logpdf(x, theta, self.component_std_small),
            ],
            axis=0,
        )

    def posterior_component_weights(self, x_obs: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Exact posterior mixture weights given an observation ``x_obs``.

        The posterior over ``theta`` is proportional to
        ``w_k * N(theta ; x_obs, sigma_k^2 I)`` restricted to the prior box, so the
        weights are the prior mixture weights re-weighted by each component's
        truncated (in-box) probability mass.
        """
        x_obs = np.asarray(x_obs, dtype=np.float64)
        if x_obs.ndim > 1:
            x_obs = x_obs.reshape(-1, self.n_dim)
            multi = True
        else:
            multi = False
        w = self.config.component_weights
        masses = np.stack(
            [
                _gaussian_box_mass(x_obs, self.component_std, self.prior_low, self.prior_high),
                _gaussian_box_mass(x_obs, self.component_std_small, self.prior_low, self.prior_high),
            ],
            axis=0,
        )
        unnormalised = w[:, None] * masses
        denom = np.sum(unnormalised, axis=0)
        denom = np.where(denom > 0, denom, 1.0)
        weights = unnormalised / denom
        if not multi:
            weights = weights[:, 0]
        return weights

    def posterior_mean_std(self, x_obs: Union[np.ndarray, Sequence[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Exact posterior mean and per-dimension standard deviation.

        Accounts for truncation of both components to the prior box by Monte Carlo
        estimation (the weights themselves are exact).
        """
        x_obs = np.asarray(x_obs, dtype=np.float64)
        weights = np.atleast_2d(self.posterior_component_weights(x_obs))
        x2 = x_obs.reshape(-1, self.n_dim)
        rng = np.random.default_rng(0)
        samples = self.reference_posterior_sample(x2, n_samples=4096, rng=rng, component_weights=weights)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        return mean, std

    def reference_posterior_sample(
        self,
        x_obs: Union[np.ndarray, Sequence[float]],
        n_samples: int = 1000,
        rng: Optional[Union[np.random.Generator, int]] = None,
        *,
        component_weights: Optional[np.ndarray] = None,
        return_components: bool = False,
    ) -> np.ndarray:
        """Exact reference posterior samples via rejection within the prior box.

        For each observation the posterior is a two-component Gaussian mixture
        (covariances ``I`` and ``0.01 I``) truncated to ``[-10, 10]^2``; samples are
        obtained by drawing from the selected component and rejecting points outside
        the box (the acceptance probability is essentially 1 for the narrow component
        and high for the broad one when ``|x_obs| <= 10``).
        """
        rng = _rng(rng, self.seed)
        x_obs = np.asarray(x_obs, dtype=np.float64)
        single = x_obs.ndim == 1
        x2 = x_obs[None, :] if single else x_obs.reshape(-1, self.n_dim)
        n_obs = x2.shape[0]
        n_samples = int(n_samples)

        if component_weights is None:
            weights = np.atleast_2d(self.posterior_component_weights(x2))
        else:
            weights = np.atleast_2d(component_weights)
            if weights.shape[0] != n_obs or weights.shape[1] != 2:
                weights = weights.reshape(n_obs, 2)

        stds = np.array([self.component_std, self.component_std_small], dtype=np.float64)
        out = np.empty((n_obs, n_samples, self.n_dim), dtype=np.float64)
        comp_out = np.empty((n_obs, n_samples), dtype=np.int64)
        for j in range(n_obs):
            comp = rng.choice(2, size=n_samples, p=weights[j] / weights[j].sum())
            samples = x2[j] + stds[comp][:, None] * rng.standard_normal((n_samples, self.n_dim))
            # Rejection: resample any point that fell outside the prior box.
            for _ in range(1000):
                outside = np.any((samples < self.prior_low) | (samples > self.prior_high), axis=1)
                if not np.any(outside):
                    break
                idx = np.nonzero(outside)[0]
                comp[idx] = rng.choice(2, size=idx.size, p=weights[j] / weights[j].sum())
                samples[idx] = x2[j] + stds[comp[idx]][:, None] * rng.standard_normal((idx.size, self.n_dim))
            out[j] = samples
            comp_out[j] = comp

        if single:
            out = out[0]
            comp_out = comp_out[0]
        if return_components:
            return out, comp_out
        return out

    #: alias used by evaluation code
    ground_truth_posterior = reference_posterior_sample

    def posterior_log_prob(self, theta: Union[np.ndarray, Sequence[float]], x_obs: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """Unnormalised log posterior density ``log p(theta) + log p(x_obs | theta)``."""
        return self.log_prior(theta) + self.log_likelihood(x_obs, theta)

    #: alias
    ground_truth_log_posterior = posterior_log_prob

    # -- datasets -------------------------------------------------------------------
    def sample_joint(self, n_samples: int = 1, rng: Optional[Union[np.random.Generator, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``(theta, x)`` pairs from the joint distribution."""
        rng = _rng(rng, self.seed)
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng)
        return theta, x

    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Optional[Union[np.random.Generator, int]] = None,
        seed: Optional[int] = None,
        chunk_size: int = 1024,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ``(theta, x)`` training data of ``n_simulations`` samples."""
        rng = _rng(rng, seed if seed is not None else self.seed)
        n_simulations = int(n_simulations)
        chunk_size = max(1, int(chunk_size))
        thetas, xs = [], []
        done = 0
        while done < n_simulations:
            n = min(chunk_size, n_simulations - done)
            theta = self.prior_sample(n, rng)
            x = self.simulate(theta, rng)
            thetas.append(theta)
            xs.append(x)
            done += n
            if verbose and (done % (chunk_size * 10) == 0 or done == n_simulations):
                print(f"[gaussian_mixture] simulated {done}/{n_simulations}")
        return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)

    def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Concatenate parameters and data as ``[theta, x]``."""
        theta = np.asarray(theta, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        if x.ndim == 1:
            x = x[None, :]
        return np.concatenate([theta, x], axis=-1)

    def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split ``[theta, x]`` into its parameter and data parts."""
        joint = np.asarray(joint, dtype=np.float64)
        return joint[..., : self.n_parameters], joint[..., self.n_parameters:]

    # -- model plumbing -------------------------------------------------------------
    def spec(self, token_dim: int = 50, **kwargs: Any):
        """Tokenizer specification for this task."""
        try:
            from simformer.tokenizer import build_benchmark_spec  # type: ignore

            return build_benchmark_spec(self.n_parameters, self.n_data)
        except Exception:  # pragma: no cover - optional dependency
            return None

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        """Task-specific attention mask (dense theta->data, dense data block)."""
        from simformer.attention_masks import build_attention_mask  # type: ignore

        return build_attention_mask(
            "gaussian_mixture",
            n_theta=self.n_parameters,
            n_x=self.n_data,
            directed=directed,
            **kwargs,
        )

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any):
        """Build a :class:`simformer.tokenizer.Tokenizer` for this task."""
        from simformer.tokenizer import Tokenizer  # type: ignore

        return Tokenizer(self.spec(token_dim=token_dim), token_dim=token_dim, **kwargs)

    def build_model(self, **kwargs: Any):
        """Build a Simformer score network wired to this task's mask and tokenizer."""
        from simformer.transformer import build_score_network  # type: ignore

        kwargs.setdefault("task", self.name)
        kwargs.setdefault("spec", self.spec(token_dim=int(kwargs.get("token_dim", 50))))
        if kwargs.get("attention_mask") is None:
            kwargs["attention_mask"] = self.attention_mask()
        return build_score_network(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d.update(
            {
                "n_parameters": self.n_parameters,
                "n_data": self.n_data,
                "parameter_names": list(self.parameter_names),
                "data_names": list(self.data_names),
                "prior_volume": self.prior_volume,
            }
        )
        return d


#: aliases matching the naming conventions used across the code base
Task = GaussianMixtureTask
Simulator = GaussianMixtureTask
GaussianMixture = GaussianMixtureTask


def build_task(config: Optional[Union[GaussianMixtureConfig, Dict[str, Any]]] = None, **kwargs: Any) -> GaussianMixtureTask:
    """Factory accepting a config dataclass, a dict, or keyword overrides."""
    return GaussianMixtureTask(config, **kwargs)


# --------------------------------------------------------------------------------------
# module-level convenience wrappers (use a default task instance)
# --------------------------------------------------------------------------------------
_DEFAULT_TASK: Optional[GaussianMixtureTask] = None


def _default_task(**kwargs: Any) -> GaussianMixtureTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None or kwargs:
        _DEFAULT_TASK = GaussianMixtureTask(**kwargs)
    return _DEFAULT_TASK


def prior_sample(n_samples: int = 1, rng: Optional[Union[np.random.Generator, int]] = None, **kwargs: Any) -> np.ndarray:
    return _default_task(**kwargs).prior_sample(n_samples, rng)


def log_prior(theta: Union[np.ndarray, Sequence[float]], **kwargs: Any) -> np.ndarray:
    return _default_task(**kwargs).log_prior(theta)


def simulate(theta: Union[np.ndarray, Sequence[float]], rng: Optional[Union[np.random.Generator, int]] = None, **kwargs: Any) -> np.ndarray:
    return _default_task(**kwargs).simulate(theta, rng)


def log_likelihood(x: Union[np.ndarray, Sequence[float]], theta: Union[np.ndarray, Sequence[float]], **kwargs: Any) -> np.ndarray:
    return _default_task(**kwargs).log_likelihood(x, theta)


def reference_posterior_sample(
    x_obs: Union[np.ndarray, Sequence[float]],
    n_samples: int = 1000,
    rng: Optional[Union[np.random.Generator, int]] = None,
    **kwargs: Any,
) -> np.ndarray:
    return _default_task(**kwargs).reference_posterior_sample(x_obs, n_samples, rng)


def make_dataset(n_simulations: int, *, rng: Optional[Union[np.random.Generator, int]] = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _default_task(**kwargs).make_dataset(n_simulations, rng=rng, seed=seed)
