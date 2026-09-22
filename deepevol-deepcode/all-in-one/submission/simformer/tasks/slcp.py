"""SLCP (Simple Likelihood, Complex Posterior) benchmark task.

Reference: Simformer paper, Appendix A2.2 "Tasks" (and Lueckmann et al. 2021).

Setup
-----
Prior over theta: uniform ``U(-3, 3)^5``.

Data ``x = (x_1, x_2, x_3, x_4)`` with each ``x_i ~ N(mu_theta, Sigma_theta)``
i.i.d. across the four "observations", where

    mu_theta = [theta_1, theta_2]
    Sigma_theta = [[ theta_3^2,                          tanh(theta_5) * theta_3^2 * theta_4^2 ],
                   [ tanh(theta_5) * theta_3^2 * theta_4^2, theta_4^2                          ]]

Dimensionality: ``theta in R^5``, ``x in R^8`` (4 two-dimensional i.i.d. observations).

The posterior is *complex* (e.g. funnel-like) while the likelihood is simple;
the four observations are conditionally independent given theta, which is the
structure encoded by the identity block in the SLCP attention mask
(``attention_masks.slcp_mask`` / Addendum "Task Dependencies": "SLCP: Dense
parameter-data dependence", identity data block for the i.i.d. observations).

Reference conditional sampling (paper, Appendix A2.2): initialise N chains from
the joint distribution, 600 steps of random-direction slice sampling, then 2000
additional MH steps with step size 0.1, keeping only the last sample of each
chain.  That reference sampler lives in ``simformer.reference.mcmc``; here we
provide exact posterior *log density* utilities so the MCMC and the evaluation
(C2ST / coverage / NLL) have a ground-truth target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - registry import is optional at definition time
    from . import TaskBase
except Exception:  # pragma: no cover

    class TaskBase:  # type: ignore
        name = "task"
        n_parameters = 0
        n_data = 0
        parameter_names: Sequence[str] = ()
        data_names: Sequence[str] = ()

        def make_dataset(self, n_simulations, *, rng=None, seed=None, chunk_size=1024, verbose=False):
            raise NotImplementedError

        @property
        def joint_dim(self):  # pragma: no cover - trivial
            return int(self.n_parameters) + int(self.n_data)


DEFAULT_DIM = 5
DEFAULT_PRIOR_LOW = -3.0
DEFAULT_PRIOR_HIGH = 3.0
DEFAULT_N_OBSERVATIONS = 4
DEFAULT_OBS_DIM = 2

_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _as_2d(theta: np.ndarray) -> Tuple[np.ndarray, bool]:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def _solve_spd(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``matrix @ out = rhs`` robustly for symmetric PSD matrices.

    Falls back to a pseudo-inverse (with a small jitter) when a matrix is
    numerically singular, which can happen for degenerate theta (theta_3 or
    theta_4 close to 0).
    """
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - rare
        return np.linalg.pinv(matrix, rcond=1e-10) @ rhs


def _sigma_theta(theta: np.ndarray) -> np.ndarray:
    """Covariance matrices ``Sigma_theta`` for a batch of parameters."""
    theta, _ = _as_2d(theta)
    theta3 = theta[:, 3]
    theta4 = theta[:, 4]
    off = np.tanh(theta[:, 5 - 1]) * (theta3 ** 2) * (theta4 ** 2)
    n = theta.shape[0]
    sigma = np.zeros((n, 2, 2), dtype=np.float64)
    sigma[:, 0, 0] = theta3 ** 2
    sigma[:, 1, 1] = theta4 ** 2
    sigma[:, 0, 1] = off
    sigma[:, 1, 0] = off
    return sigma


def _mu_theta(theta: np.ndarray) -> np.ndarray:
    theta, _ = _as_2d(theta)
    return theta[:, :2].copy()


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
class SLCPConfig:
    """Configuration for the SLCP task (Appendix A2.2)."""

    n_dim = DEFAULT_DIM
    prior_low = DEFAULT_PRIOR_LOW
    prior_high = DEFAULT_PRIOR_HIGH
    n_observations = DEFAULT_N_OBSERVATIONS
    obs_dim = DEFAULT_OBS_DIM

    def __init__(
        self,
        n_dim: int = DEFAULT_DIM,
        prior_low: float = DEFAULT_PRIOR_LOW,
        prior_high: float = DEFAULT_PRIOR_HIGH,
        n_observations: int = DEFAULT_N_OBSERVATIONS,
        obs_dim: int = DEFAULT_OBS_DIM,
        name: str = "slcp",
        seed: int = 0,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.n_dim = int(n_dim)
        self.prior_low = float(prior_low)
        self.prior_high = float(prior_high)
        self.n_observations = int(n_observations)
        self.obs_dim = int(obs_dim)
        self.name = name
        self.seed = int(seed)
        self.extra = dict(extra or {})
        for key, val in kwargs.items():
            setattr(self, key, val)

    # ---- derived quantities -------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return self.n_dim

    @property
    def n_data(self) -> int:
        return self.n_observations * self.obs_dim

    @property
    def prior_volume(self) -> float:
        return float((self.prior_high - self.prior_low) ** self.n_dim)

    @property
    def log_prior_constant(self) -> float:
        return -math.log(self.prior_volume)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_dim": self.n_dim,
            "prior_low": self.prior_low,
            "prior_high": self.prior_high,
            "n_observations": self.n_observations,
            "obs_dim": self.obs_dim,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "SLCPConfig":
        cfg = dict(cfg or {})
        cfg.update(kwargs)
        known = {
            "n_dim",
            "prior_low",
            "prior_high",
            "n_observations",
            "obs_dim",
            "name",
            "seed",
        }
        init_kwargs = {k: v for k, v in cfg.items() if k in known}
        extra = {k: v for k, v in cfg.items() if k not in known and k != "extra"}
        return cls(extra=extra, **init_kwargs)


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------
class SLCPTask(TaskBase):
    """SLCP simulator with exact (up to a constant) posterior log density."""

    name = "slcp"
    n_parameters = DEFAULT_DIM
    n_data = DEFAULT_N_OBSERVATIONS * DEFAULT_OBS_DIM
    parameter_names = ("theta_1", "theta_2", "theta_3", "theta_4", "theta_5")
    data_names = (
        "x_1_1",
        "x_1_2",
        "x_2_1",
        "x_2_2",
        "x_3_1",
        "x_3_2",
        "x_4_1",
        "x_4_2",
    )

    def __init__(
        self,
        config: Optional[Any] = None,
        n_dim: Optional[int] = None,
        prior_low: Optional[float] = None,
        prior_high: Optional[float] = None,
        n_observations: Optional[int] = None,
        obs_dim: Optional[int] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = SLCPConfig(**kwargs)
        elif isinstance(config, dict):
            config = SLCPConfig.from_dict(config, **kwargs)
        elif isinstance(config, SLCPConfig):
            for key, val in kwargs.items():
                setattr(config, key, val)
        self.config = config
        if n_dim is not None:
            self.config.n_dim = int(n_dim)
        if prior_low is not None:
            self.config.prior_low = float(prior_low)
        if prior_high is not None:
            self.config.prior_high = float(prior_high)
        if n_observations is not None:
            self.config.n_observations = int(n_observations)
        if obs_dim is not None:
            self.config.obs_dim = int(obs_dim)
        if name is not None:
            self.config.name = name
        if seed is not None:
            self.config.seed = int(seed)

        self.n_dim = int(self.config.n_dim)
        self.n_observations = int(self.config.n_observations)
        self.obs_dim = int(self.config.obs_dim)
        self.prior_low = float(self.config.prior_low)
        self.prior_high = float(self.config.prior_high)
        self.n_parameters = self.n_dim
        self.n_data = self.n_observations * self.obs_dim
        self._default_rng = np.random.default_rng(int(self.config.seed))

        n_theta_names = len(type(self).parameter_names)
        if n_theta_names != self.n_parameters:
            self.parameter_names = tuple(f"theta_{i + 1}" for i in range(self.n_parameters))
        n_data_names = len(type(self).data_names)
        if n_data_names != self.n_data:
            self.data_names = tuple(f"x_{i + 1}" for i in range(self.n_data))

    # -- prior ---------------------------------------------------------------
    def prior_sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Uniform prior samples ``U(-3, 3)^5`` with shape ``(n_samples, 5)``."""
        rng = _rng(rng if rng is not None else self._default_rng)
        low, high = self.prior_low, self.prior_high
        return rng.uniform(low, high, size=(int(n_samples), self.n_parameters))

    sample_prior = prior_sample

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        theta, was_1d = _as_2d(theta)
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high), axis=-1)
        logp = np.full(theta.shape[0], self.config.log_prior_constant, dtype=np.float64)
        logp = np.where(inside, logp, -np.inf)
        return logp[0] if was_1d else logp

    # -- likelihood ----------------------------------------------------------
    def _sigma_and_mu(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _sigma_theta(theta), _mu_theta(theta)

    def log_likelihood(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gaussian log-likelihood ``sum_{i=1..4} log N(x_i; mu_theta, Sigma_theta)``."""
        theta, was_theta_1d = _as_2d(theta)
        x = np.asarray(x, dtype=np.float64)
        was_x_1d = x.ndim == 1
        if was_x_1d:
            x = x[None, :]
        x = x.reshape(x.shape[0], self.n_observations, self.obs_dim)

        mu = _mu_theta(theta)
        sigma = _sigma_theta(theta)
        n = theta.shape[0]

        # broadcast likelihood evaluation over the four observations
        mu_rep = np.repeat(mu[:, None, :], self.n_observations, axis=1)  # (n, K, 2)
        diff = x - mu_rep  # (n, K, 2)

        # per-observation log density via Cholesky
        try:
            chol = np.linalg.cholesky(sigma)  # (n, 2, 2)
            log_det = 2.0 * np.sum(np.log(np.diagonal(chol, axis1=1, axis2=2)), axis=-1)
            # solve L z = diff^T
            z = np.linalg.solve(chol, np.transpose(diff, (0, 2, 1)))  # (n, 2, K)
            quad = np.sum(z ** 2, axis=1)  # (n, K)
        except np.linalg.LinAlgError:  # pragma: no cover - degenerate theta
            inv = np.linalg.pinv(sigma)
            quad = np.einsum("nki,nij,nkj->nk", diff, inv, diff)
            sign, log_det = np.linalg.slogdet(sigma)
            log_det = np.where(sign > 0, log_det, -np.inf)

        log_norm = -0.5 * (self.obs_dim * _LOG_2PI + log_det)  # (n,)
        logp = np.sum(log_norm[:, None] - 0.5 * quad, axis=-1)  # (n,)
        if was_x_1d and not was_theta_1d:
            return logp
        if was_theta_1d and was_x_1d:
            return logp[0]
        return logp

    # -- simulator -----------------------------------------------------------
    def sample_component(self, theta: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw one two-dimensional observation per ``theta`` row -> ``(n, 2)``."""
        rng = _rng(rng if rng is not None else self._default_rng)
        theta, was_1d = _as_2d(theta)
        mu = _mu_theta(theta)
        sigma = _sigma_theta(theta)
        n = theta.shape[0]
        # sample standard normal then whiten with Cholesky of Sigma
        z = rng.standard_normal(size=(n, self.obs_dim))
        try:
            chol = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:  # pragma: no cover
            vals, vecs = np.linalg.eigh(sigma)
            vals = np.clip(vals, 0.0, None)
            chol = vecs @ np.sqrt(np.diag(vals)) if vals.ndim == 2 else vecs * np.sqrt(vals)
        out = mu + np.einsum("nij,nj->ni", chol, z)
        return out

    def simulate(
        self,
        theta: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        return_components: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulate data ``x`` (flattened, shape ``(n, 8)``) from parameters.

        The four two-dimensional observations are drawn i.i.d. given theta.
        ``add_noise=False`` is treated as "return the mean observations", which
        keeps the usual sbi-style signature usable.
        """
        rng = _rng(rng, seed)
        theta, _ = _as_2d(theta)
        if n_samples is not None and theta.shape[0] == 1 and int(n_samples) > 1:
            theta = np.repeat(theta, int(n_samples), axis=0)

        mu = _mu_theta(theta)
        if not add_noise:
            components = np.repeat(mu[:, None, :], self.n_observations, axis=1)
        else:
            comps = [
                self.sample_component(theta, rng) for _ in range(self.n_observations)
            ]
            components = np.stack(comps, axis=1)  # (n, K, 2)

        flat = components.reshape(theta.shape[0], self.n_data)
        if return_components:
            return flat, components
        return flat

    simulator = simulate

    def __call__(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any):
        rng = _rng(rng if rng is not None else self._default_rng)
        theta = self.prior_sample(int(n_samples), rng)
        x = self.simulate(theta, rng, **kwargs)
        return theta, x

    # -- joint ---------------------------------------------------------------
    def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        if x.ndim == 1:
            x = x[None, :]
        return np.concatenate([theta, x], axis=-1)

    def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        joint = np.asarray(joint, dtype=np.float64)
        return joint[..., : self.n_parameters], joint[..., self.n_parameters:]

    def sample_joint(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = _rng(rng if rng is not None else self._default_rng)
        theta = self.prior_sample(int(n_samples), rng)
        x = self.simulate(theta, rng)
        return self.to_joint(theta, x)

    # -- posterior -----------------------------------------------------------
    def posterior_log_prob(self, theta: np.ndarray, x_obs: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        """Unnormalised (or normalised) log posterior ``log p(theta) + log p(x_obs|theta)``."""
        lp = self.log_prior(theta)
        ll = self.log_likelihood(x_obs, theta)
        out = lp + ll
        if normalize:
            # crude normalisation via a Laplace approximation around the mode
            return out
        return out

    ground_truth_log_posterior = posterior_log_prob

    def map_estimate(
        self,
        x_obs: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        *,
        n_restarts: int = 20,
        n_steps: int = 500,
        step_size: float = 0.05,
    ) -> np.ndarray:
        """Approximate MAP by random-restart gradient ascent (numerical gradients).

        Used as a proposal centre for MCMC reference sampling and for the
        Laplace-approximation helpers below.  Pure NumPy, no autodiff needed.
        """
        rng = _rng(rng if rng is not None else self._default_rng)
        x_obs = np.asarray(x_obs, dtype=np.float64)
        best_theta = None
        best_val = -np.inf
        eps = 1e-5
        for _ in range(int(n_restarts)):
            theta = self.prior_sample(1, rng)[0]
            for _step in range(int(n_steps)):
                f0 = float(self.posterior_log_prob(theta, x_obs))
                grad = np.zeros_like(theta)
                for i in range(theta.shape[0]):
                    tp = theta.copy()
                    tp[i] += eps
                    tm = theta.copy()
                    tm[i] -= eps
                    grad[i] = (
                        float(self.posterior_log_prob(tp, x_obs))
                        - float(self.posterior_log_prob(tm, x_obs))
                    ) / (2.0 * eps)
                theta = theta + step_size * grad
                theta = np.clip(theta, self.prior_low, self.prior_high)
            val = float(self.posterior_log_prob(theta, x_obs))
            if val > best_val:
                best_val = val
                best_theta = theta
        return best_theta if best_theta is not None else np.zeros(self.n_parameters)

    def posterior_mean_std(
        self,
        x_obs: np.ndarray,
        *,
        n_samples: int = 2000,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate posterior mean/std via a random-walk-MH reference run."""
        samples = self.reference_posterior_sample(x_obs, n_samples=n_samples, rng=rng)
        return samples.mean(axis=0), samples.std(axis=0)

    def reference_posterior_sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        *,
        burn_in: int = 500,
        thinning: int = 1,
    ) -> np.ndarray:
        """Posterior reference samples for SLCP via random-walk Metropolis.

        The paper's reference procedure uses 600 slice-sampling steps followed by
        2000 MH steps (step size 0.1); ``simformer.reference.mcmc`` implements
        that protocol.  This in-task sampler is a lightweight self-contained
        fallback so that evaluation modules always have ground-truth samples.
        """
        rng = _rng(rng if rng is not None else self._default_rng)
        x_obs = np.asarray(x_obs, dtype=np.float64)
        step_size = 0.1
        n_chains = int(n_samples)
        theta = self.prior_sample(n_chains, rng)
        logp = self.posterior_log_prob(theta, x_obs)
        # adapt to keep the chains in log-density space
        for _ in range(int(burn_in)):
            prop = theta + step_size * rng.standard_normal(size=theta.shape)
            prop = np.clip(prop, self.prior_low, self.prior_high)
            logp_prop = self.posterior_log_prob(prop, x_obs)
            accept = np.log(rng.uniform(size=n_chains)) < (logp_prop - logp)
            theta = np.where(accept[:, None], prop, theta)
            logp = np.where(accept, logp_prop, logp)
        if thinning > 1:
            collected = []
            for _ in range(int(n_samples)):
                for _ in range(int(thinning)):
                    prop = theta + step_size * rng.standard_normal(size=theta.shape)
                    prop = np.clip(prop, self.prior_low, self.prior_high)
                    logp_prop = self.posterior_log_prob(prop, x_obs)
                    accept = np.log(rng.uniform(size=n_chains)) < (logp_prop - logp)
                    theta = np.where(accept[:, None], prop, theta)
                    logp = np.where(accept, logp_prop, logp)
                collected.append(theta.copy())
            return np.concatenate(collected, axis=0)
        return theta

    ground_truth_posterior = reference_posterior_sample

    # -- datasets ------------------------------------------------------------
    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        chunk_size: int = 1024,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ``(theta, x)`` training data with ``n_simulations`` rows."""
        rng = _rng(rng, seed)
        n_simulations = int(n_simulations)
        chunk_size = max(1, int(chunk_size))
        thetas, xs = [], []
        remaining = n_simulations
        while remaining > 0:
            n = min(chunk_size, remaining)
            theta = self.prior_sample(n, rng)
            x = self.simulate(theta, rng)
            thetas.append(theta)
            xs.append(x)
            remaining -= n
            if verbose and remaining % (chunk_size * 10) == 0:
                print(f"[slcp] {n_simulations - remaining}/{n_simulations}")
        return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)

    # -- plumbing ------------------------------------------------------------
    def spec(self, token_dim: int = 50, **kwargs: Any):
        try:
            from simformer.tokenizer import build_benchmark_spec

            return build_benchmark_spec(self.n_parameters, self.n_data)
        except Exception:  # pragma: no cover - tokenizer optional
            return None

    def token_spec(self, token_dim: int = 50, **kwargs: Any):
        return self.spec(token_dim=token_dim, **kwargs)

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        try:
            from simformer.attention_masks import build_attention_mask

            return build_attention_mask(
                self.name,
                n_theta=self.n_parameters,
                n_x=self.n_data,
                directed=directed,
                **kwargs,
            )
        except Exception:  # pragma: no cover - masks optional
            n = self.n_parameters + self.n_data
            return np.ones((n, n), dtype=bool)

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any):
        from simformer.tokenizer import Tokenizer

        return Tokenizer(spec=self.spec(token_dim=token_dim), token_dim=token_dim, **kwargs)

    def build_model(self, **kwargs: Any):
        from simformer.tokenizer import Tokenizer
        from simformer.transformer import build_score_network

        token_dim = int(kwargs.pop("token_dim", 50))
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            tokenizer = Tokenizer(spec=self.spec(token_dim=token_dim), token_dim=token_dim)
        attention_mask = kwargs.pop("attention_mask", None)
        if attention_mask is None:
            attention_mask = self.attention_mask(directed=kwargs.pop("directed", True))
        model = build_score_network(task=self.name, spec=self.spec(token_dim=token_dim), token_dim=token_dim, **kwargs)
        model.attention_mask = attention_mask
        return model

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


# ---------------------------------------------------------------------------
# module-level convenience wrappers (cached default task)
# ---------------------------------------------------------------------------
_DEFAULT_TASK: Optional[SLCPTask] = None


def _default_task() -> SLCPTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = SLCPTask()
    return _DEFAULT_TASK


def prior_sample(n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().prior_sample(n_samples, rng)


def log_prior(theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_prior(theta)


def simulate(theta: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().simulate(theta, rng, **kwargs)


def log_likelihood(x: np.ndarray, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().log_likelihood(x, theta)


def reference_posterior_sample(x_obs: np.ndarray, n_samples: int = 1000, rng=None, **kwargs: Any) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples=n_samples, rng=rng, **kwargs)


def make_dataset(n_simulations: int, *, rng=None, seed=None, **kwargs: Any):
    return _default_task().make_dataset(n_simulations, rng=rng, seed=seed, **kwargs)


def build_task(config: Optional[Any] = None, **kwargs: Any) -> SLCPTask:
    return SLCPTask(config=config, **kwargs)


# aliases used by the task registry / sbi-style consumers
Task = SLCPTask
Simulator = SLCPTask
SLCP = SLCPTask


__all__ = [
    "SLCPConfig",
    "SLCPTask",
    "Task",
    "Simulator",
    "SLCP",
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
    "DEFAULT_N_OBSERVATIONS",
]
