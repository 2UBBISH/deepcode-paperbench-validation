"""HMM (Hidden Markov Model) benchmark task for Simformer.

Implements the HMM task from Simformer Appendix A2.2:

    theta_0 ~ N(theta_0 ; 0., 0.5^2)
    theta_{i+1} ~ N(theta_{i+1} ; theta_i, 0.5^2)      for i = 0, ..., 8
    x_i ~ N(x_i ; theta_i^2, 0.5^2)                    for i = 0, ..., 9

leading to a nonlinear hidden Markov model with a bimodal correlated posterior and
dimensionality ``theta in R^10``, ``x in R^10``.

Reference samples for arbitrary conditionals are obtained (Appendix A2.2) by

    * initializing ``N`` Markov chains with samples from the joint distribution,
    * running 5000 steps of an HMC sampler,
    * keeping only the last sample of each chain.

The task exposes the usual ``TaskBase`` interface (prior / simulator / likelihood /
joint density / reference posterior), plus plumbing to the tokenizer, attention-mask
builder and transformer score network so it can be trained exactly like the other
tasks (see :mod:`simformer.tasks`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:  # pragma: no cover - registry base class, importable in both layouts
    from . import TaskBase  # type: ignore
except Exception:  # pragma: no cover - standalone fallback
    try:
        from simformer.tasks import TaskBase  # type: ignore
    except Exception:
        class TaskBase:  # type: ignore
            """Minimal stand-in when the task registry is unavailable."""

            name = "task"
            n_parameters = 0
            n_data = 0
            parameter_names: Tuple[str, ...] = ()
            data_names: Tuple[str, ...] = ()

            @property
            def joint_dim(self) -> int:
                return int(self.n_parameters + self.n_data)

            def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
                theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
                x = np.atleast_2d(np.asarray(x, dtype=np.float64))
                return np.concatenate([theta, x], axis=-1)

            def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                joint = np.atleast_2d(np.asarray(joint, dtype=np.float64))
                return joint[..., : self.n_parameters], joint[..., self.n_parameters :]


__all__ = [
    "HMMConfig",
    "HMMTask",
    "Task",
    "Simulator",
    "HMM",
    "build_task",
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "log_joint",
    "posterior_log_prob",
    "reference_posterior_sample",
    "make_dataset",
    "DEFAULT_N_DIM",
    "DEFAULT_THETA_STD",
    "DEFAULT_OBS_STD",
]

_LOG_2PI = math.log(2.0 * math.pi)

DEFAULT_N_DIM = 10
DEFAULT_THETA_STD = 0.5
DEFAULT_OBS_STD = 0.5

#: Paper's reference protocol for the HMM task: 5000 HMC steps, keep last sample.
HMM_REFERENCE_HMC_STEPS = 5000
HMM_REFERENCE_HMC_STEP_SIZE = 0.05


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _as_2d_theta(theta) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _diag_gaussian_logpdf(x: np.ndarray, mean: np.ndarray, std: float) -> np.ndarray:
    """Sum of independent 1-D Gaussian log-densities over the last axis."""
    z = (np.asarray(x, dtype=np.float64) - np.asarray(mean, dtype=np.float64)) / float(std)
    return -0.5 * z ** 2 - math.log(float(std)) - 0.5 * _LOG_2PI


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class HMMConfig:
    """Configuration of the HMM benchmark task."""

    n_dim: int = DEFAULT_N_DIM
    theta_std: float = DEFAULT_THETA_STD
    obs_std: float = DEFAULT_OBS_STD
    name: str = "hmm"
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ properties
    @property
    def n_parameters(self) -> int:
        return int(self.n_dim)

    @property
    def n_data(self) -> int:
        return int(self.n_dim)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_dim": int(self.n_dim),
            "n_parameters": self.n_parameters,
            "n_data": self.n_data,
            "theta_std": float(self.theta_std),
            "obs_std": float(self.obs_std),
            "seed": int(self.seed),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "HMMConfig"]] = None, **kwargs) -> "HMMConfig":
        if isinstance(cfg, HMMConfig):
            base = cfg.to_dict()
            base.update(kwargs)
            cfg = base
        cfg = dict(cfg or {})
        extra = dict(cfg.pop("extra", {}) or {})
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        known.setdefault("extra", extra)
        known.update(kwargs)
        return cls(**known)


# --------------------------------------------------------------------------------------
# the task
# --------------------------------------------------------------------------------------
class HMMTask(TaskBase):
    """Nonlinear hidden Markov model SBI task (Appendix A2.2)."""

    name = "hmm"
    n_parameters = DEFAULT_N_DIM
    n_data = DEFAULT_N_DIM
    parameter_names = tuple(f"theta_{i}" for i in range(DEFAULT_N_DIM))
    data_names = tuple(f"x_{i}" for i in range(DEFAULT_N_DIM))

    def __init__(
        self,
        config: Optional[Union[HMMConfig, Dict[str, Any]]] = None,
        n_dim: Optional[int] = None,
        theta_std: Optional[float] = None,
        obs_std: Optional[float] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = HMMConfig()
        elif isinstance(config, dict):
            config = HMMConfig.from_dict(config)
        self.config = config
        if n_dim is not None:
            self.config.n_dim = int(n_dim)
        if theta_std is not None:
            self.config.theta_std = float(theta_std)
        if obs_std is not None:
            self.config.obs_std = float(obs_std)
        if name is not None:
            self.config.name = str(name)
        if seed is not None:
            self.config.seed = int(seed)
        if kwargs:
            self.config.extra.update(kwargs)

        self.n_dim = int(self.config.n_dim)
        self.theta_std = float(self.config.theta_std)
        self.obs_std = float(self.config.obs_std)
        self.seed = int(self.config.seed)
        self.n_parameters = self.n_dim
        self.n_data = self.n_dim
        self.parameter_names = tuple(f"theta_{i}" for i in range(self.n_dim))
        self.data_names = tuple(f"x_{i}" for i in range(self.n_dim))

    # ------------------------------------------------------------------ properties
    @property
    def parameters(self) -> int:  # convenience alias
        return self.n_parameters

    # ------------------------------------------------------------------ prior
    def prior_sample(
        self,
        n_samples: int = 1,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Sample ``theta`` from the Markov-chain prior, shape ``(n_samples, n_dim)``."""
        n_samples = int(n_samples)
        rng = _rng(rng, seed if seed is not None else self.seed)
        std = self.theta_std
        theta = np.empty((n_samples, self.n_dim), dtype=np.float64)
        theta[:, 0] = rng.normal(0.0, std, size=n_samples)
        for i in range(self.n_dim - 1):
            theta[:, i + 1] = theta[:, i] + rng.normal(0.0, std, size=n_samples)
        return theta

    # sbi-style alias
    def sample_prior(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return self.prior_sample(n_samples, rng=rng)

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """Log density of the Markov-chain prior; ``(n,)`` for ``(n, n_dim)`` input."""
        theta = _as_2d_theta(theta)
        std = self.theta_std
        # initial state
        logp = _diag_gaussian_logpdf(theta[:, 0], 0.0, std)
        # transitions theta_{i+1} | theta_i ~ N(theta_i, std^2)
        if self.n_dim > 1:
            logp = logp + np.sum(_diag_gaussian_logpdf(theta[:, 1:], theta[:, :-1], std), axis=-1)
        return logp

    def prior_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Broad box used to initialize reference MCMC chains."""
        # random walk of n_dim steps of std 0.5 -> ~N(0, (sqrt(n_dim)*0.5)^2); use a
        # generous multiple of the standard deviation.
        width = 8.0 * self.theta_std * math.sqrt(max(self.n_dim, 1))
        low = np.full(self.n_dim, -width, dtype=np.float64)
        high = np.full(self.n_dim, width, dtype=np.float64)
        return low, high

    # ------------------------------------------------------------------ simulator
    def data_means(self, theta: np.ndarray) -> np.ndarray:
        """Deterministic observation means ``theta_i^2``."""
        theta = _as_2d_theta(theta)
        return theta ** 2

    def simulate(
        self,
        theta: np.ndarray,
        rng: Optional[np.random.Generator] = None,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        return_components: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Simulate observations ``x_i ~ N(theta_i^2, 0.5^2)``."""
        rng = _rng(rng, seed if seed is not None else self.seed)
        theta = _as_2d_theta(theta)
        if n_samples is not None and theta.shape[0] == 1 and int(n_samples) > 1:
            theta = np.repeat(theta, int(n_samples), axis=0)
        mean = self.data_means(theta)
        if not add_noise:
            return mean
        noise = rng.normal(0.0, self.obs_std, size=mean.shape)
        x = mean + noise
        if return_components:
            return x, mean
        return x

    # sbi-style alias
    def simulator(self, theta: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs) -> np.ndarray:
        return self.simulate(theta, rng=rng, **kwargs)

    def __call__(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None):
        rng = _rng(rng, self.seed)
        theta = self.prior_sample(n_samples, rng=rng)
        x = self.simulate(theta, rng=rng)
        return theta, x

    # ------------------------------------------------------------------ densities
    def log_likelihood(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """``log p(x | theta) = sum_i log N(x_i ; theta_i^2, obs_std^2)``."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        theta = _as_2d_theta(theta)
        if x.shape[0] != theta.shape[0]:
            if x.shape[0] == 1:
                x = np.repeat(x, theta.shape[0], axis=0)
            elif theta.shape[0] == 1:
                theta = np.repeat(theta, x.shape[0], axis=0)
        mean = theta ** 2
        return np.sum(_diag_gaussian_logpdf(x, mean, self.obs_std), axis=-1)

    def log_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """``log p(theta, x) = log p(theta) + log p(x | theta)``."""
        return self.log_prior(theta) + self.log_likelihood(x, theta)

    def posterior_log_prob(self, theta: np.ndarray, x_obs: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Unnormalized log posterior ``log p(theta) + log p(x_obs | theta)``."""
        theta = _as_2d_theta(theta)
        x_obs = np.atleast_2d(np.asarray(x_obs, dtype=np.float64))
        if x_obs.shape[0] != theta.shape[0]:
            if x_obs.shape[0] == 1:
                x_obs = np.repeat(x_obs, theta.shape[0], axis=0)
            elif theta.shape[0] == 1:
                theta = np.repeat(theta, x_obs.shape[0], axis=0)
        return self.log_joint(theta, x_obs)

    # alias used by evaluation code
    ground_truth_log_posterior = posterior_log_prob

    # ------------------------------------------------------------------ reference
    def reference_posterior_sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        use_mcmc_module: bool = True,
    ) -> np.ndarray:
        """Ground-truth posterior samples via the paper's 5000-step HMC reference."""
        rng = _rng(rng, seed if seed is not None else self.seed)
        n_samples = int(n_samples)
        x_obs = np.atleast_2d(np.asarray(x_obs, dtype=np.float64))

        if use_mcmc_module:
            try:  # pragma: no cover - optional dependency
                from simformer.reference.mcmc import TREE_PROTOCOL, HMM_PROTOCOL, sample_reference  # type: ignore

                protocol = HMM_PROTOCOL if "HMM_PROTOCOL" in locals() else TREE_PROTOCOL
                condition_mask = np.array(
                    [0.0] * self.n_parameters + [1.0] * self.n_data, dtype=np.float64
                )
                values = np.concatenate(
                    [np.zeros(self.n_parameters, dtype=np.float64), x_obs.reshape(-1)],
                    axis=-1,
                )
                samples = sample_reference(
                    self,
                    condition_mask,
                    values,
                    n_samples=n_samples,
                    protocol=protocol,
                    seed=seed if seed is not None else self.seed,
                )
                samples = np.asarray(samples, dtype=np.float64)
                if samples.ndim == 2 and samples.shape[-1] == self.n_parameters:
                    return samples
                if samples.ndim == 2 and samples.shape[-1] == self.n_parameters + self.n_data:
                    return samples[:, : self.n_parameters]
            except Exception:
                pass

        return self._hmc_reference(x_obs, n_samples=n_samples, rng=rng, thinning=thinning)

    # alias used by evaluation / baseline code
    ground_truth_posterior = reference_posterior_sample

    def _hmc_reference(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        thinning: int = 1,
        n_steps: int = HMM_REFERENCE_HMC_STEPS,
        step_size: float = HMM_REFERENCE_HMC_STEP_SIZE,
        n_leapfrog: int = 10,
    ) -> np.ndarray:
        """Self-contained HMC fallback over ``log p(theta | x_obs)`` (last sample kept)."""
        rng = _rng(rng, self.seed)
        x_obs = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        n_chains = int(n_samples)

        def log_prob(theta_flat: np.ndarray) -> float:
            theta = np.asarray(theta_flat, dtype=np.float64).reshape(1, -1)
            return float(self.posterior_log_prob(theta, x_obs.reshape(1, -1))[0])

        # initialize chains from the joint (theta drawn from the prior)
        theta = self.prior_sample(n_chains, rng=rng)
        out = np.empty((n_chains, self.n_dim), dtype=np.float64)
        for c in range(n_chains):
            q = theta[c].copy()
            logp = log_prob(q)
            eps = float(step_size)
            for step in range(int(n_steps)):
                p = rng.normal(0.0, 1.0, size=self.n_dim)
                q_new = q.copy()
                p_new = p.copy()
                # leapfrog
                grad = _numerical_gradient(log_prob, q_new)
                p_new = p_new + 0.5 * eps * grad
                for _l in range(int(n_leapfrog)):
                    q_new = q_new + eps * p_new
                    grad = _numerical_gradient(log_prob, q_new)
                    if _l < int(n_leapfrog) - 1:
                        p_new = p_new + eps * grad
                p_new = p_new + 0.5 * eps * grad
                p_new = -p_new  # reverse momentum

                logp_new = log_prob(q_new)
                h_old = logp - 0.5 * float(np.dot(p, p))
                h_new = logp_new - 0.5 * float(np.dot(p_new, p_new))
                if np.log(rng.uniform()) < (h_new - h_old):
                    q, logp = q_new, logp_new
                # simple adaptation over an initial warm-up window
                if step < 100:
                    eps *= math.exp((0.8 - 1.0) / math.sqrt(step + 1.0))
            out[c] = q
        return out


def _numerical_gradient(fn, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central finite-difference gradient of a scalar function."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (fn(xp) - fn(xm)) / (2.0 * eps)
    return grad


# --------------------------------------------------------------------------------------
# joint helpers
# --------------------------------------------------------------------------------------
def _to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    theta = _as_2d_theta(theta)
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.concatenate([theta, x], axis=-1)


def _split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    joint = np.atleast_2d(np.asarray(joint, dtype=np.float64))
    return joint[..., : self.n_parameters], joint[..., self.n_parameters :]


if not hasattr(HMMTask, "to_joint"):
    HMMTask.to_joint = _to_joint  # type: ignore[attr-defined]
if not hasattr(HMMTask, "split_joint"):
    HMMTask.split_joint = _split_joint  # type: ignore[attr-defined]


def _sample_joint(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    rng = _rng(rng, self.seed)
    theta = self.prior_sample(n_samples, rng=rng)
    x = self.simulate(theta, rng=rng)
    return np.concatenate([theta, x], axis=-1)


if not hasattr(HMMTask, "sample_joint"):
    HMMTask.sample_joint = _sample_joint  # type: ignore[attr-defined]


def _make_dataset(
    self,
    n_simulations: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    chunk_size: int = 1024,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = _rng(rng, seed if seed is not None else self.seed)
    n_simulations = int(n_simulations)
    thetas, xs = [], []
    done = 0
    while done < n_simulations:
        chunk = min(int(chunk_size), n_simulations - done)
        theta = self.prior_sample(chunk, rng=rng)
        x = self.simulate(theta, rng=rng)
        thetas.append(theta)
        xs.append(x)
        done += chunk
        if verbose:
            print(f"[hmm] simulated {done}/{n_simulations}", flush=True)
    return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)


if not hasattr(HMMTask, "make_dataset"):
    HMMTask.make_dataset = _make_dataset  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Simformer plumbing (tokenizer / attention mask / score network)
# --------------------------------------------------------------------------------------
def _spec(self, token_dim: int = 50, **kwargs):
    try:
        from simformer.tokenizer import build_benchmark_spec  # type: ignore
    except Exception:  # pragma: no cover
        from ..tokenizer import build_benchmark_spec  # type: ignore
    return build_benchmark_spec(self.n_parameters, self.n_data)


def _attention_mask(self, directed: bool = True, **kwargs) -> np.ndarray:
    """HMM attention mask: Markov chain over parameters, factorized data.

    The data variable ``x_i`` depends only on ``theta_i`` (factorized data), while the
    parameters form a Markov chain ``theta_{i-1} -> theta_i`` (Addendum, "Task
    Dependencies": *Markov chain for parameters and factorized data*).
    """
    try:
        from simformer.attention_masks import build_attention_mask  # type: ignore

        try:
            return build_attention_mask(
                "hmm",
                n_theta=self.n_parameters,
                n_x=self.n_data,
                n_states=self.n_parameters,
                directed=directed,
            )
        except TypeError:
            return build_attention_mask("hmm", directed=directed)
    except Exception:
        pass
    # fallback mask construction
    n = self.n_parameters + self.n_data
    mask = np.eye(n, dtype=np.float64)
    for i in range(self.n_parameters - 1):
        mask[i + 1, i] = 1.0  # theta chain
    for i in range(self.n_parameters):
        mask[self.n_parameters + i, i] = 1.0  # x_i <- theta_i
    mask[self.n_parameters + i if False else 0, 0] = 1.0
    if not directed:
        mask = np.maximum(mask, mask.T)
        np.fill_diagonal(mask, 1.0)
    return mask


if not hasattr(HMMTask, "spec"):
    HMMTask.spec = _spec  # type: ignore[attr-defined]
if not hasattr(HMMTask, "token_spec"):
    HMMTask.token_spec = _spec  # type: ignore[attr-defined]
if not hasattr(HMMTask, "attention_mask"):
    HMMTask.attention_mask = _attention_mask  # type: ignore[attr-defined]


def _build_tokenizer(self, token_dim: int = 50, **kwargs):
    try:
        from simformer.tokenizer import Tokenizer  # type: ignore
    except Exception:  # pragma: no cover
        from ..tokenizer import Tokenizer  # type: ignore
    return Tokenizer(self.spec(token_dim=token_dim), token_dim=token_dim)


def _build_model(self, **kwargs):
    try:
        from simformer.transformer import build_score_network  # type: ignore
    except Exception:  # pragma: no cover
        from ..transformer import build_score_network  # type: ignore
    tokenizer = kwargs.pop("tokenizer", None)
    if tokenizer is None:
        tokenizer = self.build_tokenizer(token_dim=kwargs.get("token_dim", 50))
    mask = kwargs.pop("attention_mask", None)
    if mask is None:
        mask = self.attention_mask()
    return build_score_network(task="hmm", tokenizer=tokenizer, attention_mask=mask, **kwargs)


if not hasattr(HMMTask, "build_tokenizer"):
    HMMTask.build_tokenizer = _build_tokenizer  # type: ignore[attr-defined]
if not hasattr(HMMTask, "build_model"):
    HMMTask.build_model = _build_model  # type: ignore[attr-defined]


def _to_dict(self) -> Dict[str, Any]:
    d = self.config.to_dict()
    d.update(
        {
            "name": self.name,
            "n_parameters": int(self.n_parameters),
            "n_data": int(self.n_data),
            "parameter_names": list(self.parameter_names),
            "data_names": list(self.data_names),
            "reference_protocol": {
                "method": "hmc",
                "n_steps": int(HMM_REFERENCE_HMC_STEPS),
                "step_size": float(HMM_REFERENCE_HMC_STEP_SIZE),
                "keep": "last_sample",
            },
        }
    )
    return d


if not hasattr(HMMTask, "to_dict"):
    HMMTask.to_dict = _to_dict  # type: ignore[attr-defined]


# aliases
Task = HMMTask
Simulator = HMMTask
HMM = HMMTask


# --------------------------------------------------------------------------------------
# factories
# --------------------------------------------------------------------------------------
def build_task(config: Optional[Union[HMMConfig, Dict[str, Any]]] = None, **kwargs) -> HMMTask:
    """Instantiate the HMM task (accepting a config dataclass/dict or overrides)."""
    return HMMTask(config=config, **kwargs)


# --------------------------------------------------------------------------------------
# module level wrappers (cached default task)
# --------------------------------------------------------------------------------------
_DEFAULT_TASK: Optional[HMMTask] = None


def _default_task() -> HMMTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = HMMTask()
    return _DEFAULT_TASK


def prior_sample(n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs) -> np.ndarray:
    return _default_task().prior_sample(n_samples, rng=rng, **kwargs)


def log_prior(theta: np.ndarray, **kwargs) -> np.ndarray:
    return _default_task().log_prior(theta)


def simulate(theta: np.ndarray, rng: Optional[np.random.Generator] = None, **kwargs) -> np.ndarray:
    return _default_task().simulate(theta, rng=rng, **kwargs)


def log_likelihood(x: np.ndarray, theta: np.ndarray, **kwargs) -> np.ndarray:
    return _default_task().log_likelihood(x, theta)


def log_joint(theta: np.ndarray, x: np.ndarray, **kwargs) -> np.ndarray:
    return _default_task().log_joint(theta, x)


def posterior_log_prob(theta: np.ndarray, x_obs: np.ndarray, **kwargs) -> np.ndarray:
    return _default_task().posterior_log_prob(theta, x_obs)


def reference_posterior_sample(
    x_obs: np.ndarray, n_samples: int = 1000, rng: Optional[np.random.Generator] = None, **kwargs
) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples=n_samples, rng=rng, **kwargs)


def make_dataset(
    n_simulations: int,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    return _default_task().make_dataset(n_simulations, rng=rng, seed=seed, **kwargs)
