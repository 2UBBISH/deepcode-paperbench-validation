"""Tree task (Simformer, Appendix A2.2).

This is a nonlinear tree-shaped task:

.. math::

    \\theta_0 \\sim \\mathcal{N}(0, 1),\\quad
    \\theta_1 \\sim \\mathcal{N}(\\theta_0, 1),\\quad
    \\theta_2 \\sim \\mathcal{N}(\\theta_0, 1)

Observable data is obtained through

.. math::

    x_0 \\sim \\mathcal{N}(\\sin(\\theta_1)^2, 0.2^2),\\quad
    x_1 \\sim \\mathcal{N}(0.1\\,\\theta_1^2, 0.2^2),\\quad
    x_2 \\sim \\mathcal{N}(0.1\\,\\theta_2^2, 0.6^2),\\quad
    x_3 \\sim \\mathcal{N}(\\cos(\\theta_2)^2, 0.1^2)

leading to a tree-like factorization with highly multimodal conditionals.
Reference samples for arbitrary conditionals are obtained with 5000 HMC steps
(only the last sample of each chain is kept), see ``reference/mcmc.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - package-relative import
    from . import TaskBase  # type: ignore
except Exception:  # pragma: no cover - standalone fallback

    class TaskBase:  # type: ignore
        """Minimal stand-in for :class:`simformer.tasks.TaskBase`."""

        name = "task"
        n_parameters = 0
        n_data = 0
        parameter_names: Tuple[str, ...] = ()
        data_names: Tuple[str, ...] = ()

        @property
        def joint_dim(self) -> int:
            return int(self.n_parameters) + int(self.n_data)

        def to_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
            theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
            x = np.atleast_2d(np.asarray(x, dtype=np.float64))
            return np.concatenate([theta, x], axis=-1)

        def split_joint(self, joint: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            joint = np.atleast_2d(np.asarray(joint, dtype=np.float64))
            return joint[..., : self.n_parameters], joint[..., self.n_parameters :]

        def to_dict(self) -> Dict[str, Any]:
            return {
                "name": self.name,
                "n_parameters": int(self.n_parameters),
                "n_data": int(self.n_data),
            }


__all__ = [
    "TreeConfig",
    "TreeTask",
    "Task",
    "Simulator",
    "Tree",
    "build_task",
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "log_joint",
    "posterior_log_prob",
    "reference_posterior_sample",
    "make_dataset",
    "DEFAULT_THETA_STD",
    "DEFAULT_OBS_STDS",
]

# ---------------------------------------------------------------------------
# Constants (Appendix A2.2)
# ---------------------------------------------------------------------------
DEFAULT_THETA_STD = 1.0
DEFAULT_OBS_STDS = (0.2, 0.2, 0.6, 0.1)
_LOG_2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class TreeConfig:
    """Configuration for the Tree task."""

    n_parameters: int = 3
    n_data: int = 4
    theta_std: float = DEFAULT_THETA_STD
    obs_stds: Tuple[float, float, float, float] = DEFAULT_OBS_STDS
    name: str = "tree"
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_parameters": int(self.n_parameters),
            "n_data": int(self.n_data),
            "theta_std": float(self.theta_std),
            "obs_stds": tuple(float(s) for s in self.obs_stds),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "TreeConfig":
        cfg = dict(cfg or {})
        cfg.update(kwargs)
        known = {
            "n_parameters",
            "n_data",
            "theta_std",
            "obs_stds",
            "name",
            "seed",
        }
        extra = {k: v for k, v in cfg.items() if k not in known}
        obs_stds = cfg.get("obs_stds", DEFAULT_OBS_STDS)
        if obs_stds is not None:
            obs_stds = tuple(float(s) for s in obs_stds)
            if len(obs_stds) != 4:
                obs_stds = DEFAULT_OBS_STDS
        return cls(
            n_parameters=int(cfg.get("n_parameters", 3)),
            n_data=int(cfg.get("n_data", 4)),
            theta_std=float(cfg.get("theta_std", DEFAULT_THETA_STD)),
            obs_stds=obs_stds if obs_stds is not None else DEFAULT_OBS_STDS,
            name=cfg.get("name", "tree"),
            seed=int(cfg.get("seed", 0)),
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rng(rng: Optional[Union[np.random.Generator, int]] = None, seed: Optional[int] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return np.random.default_rng(0 if seed is None else int(seed))


def _as_2d_theta(theta: np.ndarray) -> np.ndarray:
    arr = np.asarray(theta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _gaussian_logpdf(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Elementwise diagonal-Gaussian log density (no reduction)."""

    x = np.asarray(x, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    var = np.maximum(std ** 2, 1e-300)
    return -0.5 * (_LOG_2PI + np.log(var) + (x - mean) ** 2 / var)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
class TreeTask(TaskBase):
    """Nonlinear tree-shaped SBI task (Appendix A2.2)."""

    name = "tree"
    n_parameters = 3
    n_data = 4
    parameter_names: Tuple[str, ...] = ("theta_0", "theta_1", "theta_2")
    data_names: Tuple[str, ...] = ("x_0", "x_1", "x_2", "x_3")

    def __init__(
        self,
        config: Optional[Union[TreeConfig, Dict[str, Any]]] = None,
        *,
        n_parameters: Optional[int] = None,
        n_data: Optional[int] = None,
        theta_std: Optional[float] = None,
        obs_stds: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, TreeConfig):
            self.config = config
        else:
            merged: Dict[str, Any] = dict(config or {})
            merged.update(kwargs)
            if n_parameters is not None:
                merged["n_parameters"] = n_parameters
            if n_data is not None:
                merged["n_data"] = n_data
            if theta_std is not None:
                merged["theta_std"] = theta_std
            if obs_stds is not None:
                merged["obs_stds"] = obs_stds
            if name is not None:
                merged["name"] = name
            if seed is not None:
                merged["seed"] = seed
            self.config = TreeConfig.from_dict(merged)

        self.name = self.config.name
        self.n_parameters = int(self.config.n_parameters)
        self.n_data = int(self.config.n_data)
        self.theta_std = float(self.config.theta_std)
        self.obs_stds = tuple(float(s) for s in self.config.obs_stds)
        self.seed = int(self.config.seed)
        self._rng_default = np.random.default_rng(self.seed)

    # -- metadata ----------------------------------------------------------
    @property
    def data_dim(self) -> int:
        return int(self.n_data)

    @property
    def joint_dim(self) -> int:
        return int(self.n_parameters) + int(self.n_data)

    def prior_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Broad box used for MCMC initialization (prior is unbounded)."""

        low = -10.0 * np.ones(self.n_parameters, dtype=np.float64)
        high = 10.0 * np.ones(self.n_parameters, dtype=np.float64)
        return low, high

    # -- prior -------------------------------------------------------------
    def prior_sample(
        self,
        n_samples: int = 1,
        rng: Optional[Union[np.random.Generator, int]] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Draw ``n_samples`` parameter vectors from the tree prior."""

        gen = _rng(rng if rng is not None else self._rng_default, seed)
        n = int(n_samples)
        theta = np.empty((n, 3), dtype=np.float64)
        theta[:, 0] = gen.normal(0.0, self.theta_std, size=n)
        theta[:, 1] = theta[:, 0] + gen.normal(0.0, self.theta_std, size=n)
        theta[:, 2] = theta[:, 0] + gen.normal(0.0, self.theta_std, size=n)
        return theta

    # ``prior_sample`` accepts any subset of ``(n_samples, rng, seed)``;
    # provide a small normalizer so call sites with positional args work.
    def sample_prior(self, n_samples: int = 1, rng: Any = None) -> np.ndarray:
        return self.prior_sample(n_samples, rng)

    def log_prior(self, theta: np.ndarray) -> np.ndarray:
        """``log p(theta)`` for the tree prior (sum of three Gaussians)."""

        theta = _as_2d_theta(theta)
        std = self.theta_std
        logp = _gaussian_logpdf(theta[:, 0], 0.0, std)
        logp = logp + _gaussian_logpdf(theta[:, 1], theta[:, 0], std)
        logp = logp + _gaussian_logpdf(theta[:, 2], theta[:, 0], std)
        return logp

    # -- simulator ---------------------------------------------------------
    def data_means(self, theta: np.ndarray) -> np.ndarray:
        """Deterministic data means ``(n, 4)`` implied by ``theta``."""

        theta = _as_2d_theta(theta)
        means = np.empty((theta.shape[0], 4), dtype=np.float64)
        means[:, 0] = np.sin(theta[:, 1]) ** 2
        means[:, 1] = 0.1 * theta[:, 1] ** 2
        means[:, 2] = 0.1 * theta[:, 2] ** 2
        means[:, 3] = np.cos(theta[:, 2]) ** 2
        return means

    def simulate(
        self,
        theta: np.ndarray,
        rng: Optional[Union[np.random.Generator, int]] = None,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Simulate observables ``x`` from ``theta``."""

        gen = _rng(rng if rng is not None else self._rng_default, seed)
        theta2 = _as_2d_theta(theta)

        if n_samples is not None and theta2.shape[0] == 1 and int(n_samples) != 1:
            theta2 = np.repeat(theta2, int(n_samples), axis=0)

        means = self.data_means(theta2)
        if not add_noise:
            return means
        noise = gen.standard_normal(means.shape) * np.asarray(self.obs_stds, dtype=np.float64)[None, :]
        return means + noise

    def simulator(self, theta: np.ndarray, rng: Any = None, **kwargs: Any) -> np.ndarray:
        return self.simulate(theta, rng, **kwargs)

    def __call__(self, n_samples: int = 1, rng: Any = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng)
        return theta, x

    # -- likelihood --------------------------------------------------------
    def log_likelihood(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """``log p(x | theta)`` as a sum over the four conditionally
        independent observables."""

        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        theta = _as_2d_theta(theta)
        if x.shape[0] != theta.shape[0]:
            if x.shape[0] == 1:
                x = np.repeat(x, theta.shape[0], axis=0)
            elif theta.shape[0] == 1:
                theta = np.repeat(theta, x.shape[0], axis=0)

        means = self.data_means(theta)
        logp = np.zeros(x.shape[0], dtype=np.float64)
        for j, std in enumerate(self.obs_stds):
            logp = logp + _gaussian_logpdf(x[:, j], means[:, j], std)
        return logp

    def log_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self.log_prior(theta) + self.log_likelihood(x, theta)

    # -- posterior (ground truth via MCMC) ---------------------------------
    def posterior_log_prob(
        self,
        theta: np.ndarray,
        x_obs: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Unnormalized log posterior ``log p(theta) + log p(x_obs | theta)``."""

        x_obs = np.atleast_2d(np.asarray(x_obs, dtype=np.float64))
        if x_obs.shape[0] == 1:
            x_obs = np.repeat(x_obs, _as_2d_theta(theta).shape[0], axis=0)
        logp = self.log_joint(theta, x_obs)
        if normalize:
            logp = logp - np.max(logp)
        return logp

    ground_truth_log_posterior = posterior_log_prob

    def reference_posterior_sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Optional[Union[np.random.Generator, int]] = None,
        *,
        seed: Optional[int] = None,
        burn_in: Optional[int] = None,
        thinning: int = 1,
        use_mcmc_module: bool = True,
    ) -> np.ndarray:
        """Ground-truth posterior samples via HMC (5000 steps, Appendix A2.2)."""

        gen = _rng(rng if rng is not None else self._rng_default, seed)
        x_obs = np.atleast_2d(np.asarray(x_obs, dtype=np.float64))

        if use_mcmc_module:
            try:  # pragma: no cover - optional dependency path
                from ..reference.mcmc import HMM_PROTOCOL  # noqa: F401
                from ..reference.mcmc import TREE_PROTOCOL, sample_reference

                condition_mask = np.array(
                    [0] * self.n_parameters + [1] * self.n_data, dtype=np.float64
                )
                values = np.concatenate(
                    [np.zeros(self.n_parameters, dtype=np.float64), x_obs[0]]
                )
                protocol = TREE_PROTOCOL
                if burn_in is not None:
                    protocol = TREE_PROTOCOL
                samples = sample_reference(
                    self,
                    condition_mask,
                    values,
                    n_samples=int(n_samples),
                    protocol=protocol,
                    seed=int(seed if seed is not None else self.seed),
                    return_full=False,
                )
                return np.asarray(samples, dtype=np.float64)
            except Exception:
                pass

        return self._hmc_reference(x_obs[0], int(n_samples), gen, thinning=thinning)

    # Alias used by evaluation / baseline code.
    def ground_truth_posterior(self, x_obs: np.ndarray, n_samples: int = 1000, rng: Any = None) -> np.ndarray:
        return self.reference_posterior_sample(x_obs, n_samples, rng)

    def _hmc_reference(
        self,
        x_obs: np.ndarray,
        n_samples: int,
        rng: np.random.Generator,
        *,
        n_steps: int = 5000,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        thinning: int = 1,
    ) -> np.ndarray:
        """Self-contained HMC fallback (mirrors the reference protocol)."""

        def log_prob(theta: np.ndarray) -> float:
            val = self.log_prior(theta[None, :]) + self.log_likelihood(x_obs[None, :], theta[None, :])
            out = float(np.asarray(val).reshape(-1)[0])
            return out if np.isfinite(out) else -np.inf

        def grad(theta: np.ndarray) -> np.ndarray:
            eps = 1e-4
            g = np.zeros_like(theta)
            for i in range(theta.size):
                step = np.zeros_like(theta)
                step[i] = eps
                fp = log_prob(theta + step)
                fm = log_prob(theta - step)
                g[i] = (fp - fm) / (2.0 * eps)
            return g

        init = self.prior_sample(n_samples, rng)
        samples = np.empty_like(init)
        for c in range(n_samples):
            q = init[c].copy()
            cur = log_prob(q)
            # burn-in (half of the protocol length) then sampling
            for it in range(int(n_steps) + int(thinning)):
                p = rng.standard_normal(q.shape)
                cur_p = float(np.sum(p ** 2) / 2.0)
                q_new = q.copy()
                p_new = p.copy()
                p_new = p_new + 0.5 * step_size * grad(q_new)
                for _ in range(n_leapfrog):
                    q_new = q_new + step_size * p_new
                    if _ != n_leapfrog - 1:
                        p_new = p_new + step_size * grad(q_new)
                p_new = p_new + 0.5 * step_size * grad(q_new)
                new = log_prob(q_new)
                new_p = float(np.sum(p_new ** 2) / 2.0)
                log_alpha = (new - new_p) - (cur - cur_p)
                if np.log(rng.random()) < log_alpha:
                    q, cur = q_new, new
            samples[c] = q
        return samples

    def posterior_mean_std(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        rng: Any = None,
        *,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = self.reference_posterior_sample(x_obs, n_samples, rng, seed=seed)
        return samples.mean(axis=0), samples.std(axis=0)

    # -- datasets ----------------------------------------------------------
    def sample_joint(self, n_samples: int = 1, rng: Any = None) -> np.ndarray:
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng)
        return self.to_joint(theta, x)

    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Any = None,
        seed: Optional[int] = None,
        chunk_size: int = 1024,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(theta, x)`` arrays with ``n_simulations`` rows."""

        gen = _rng(rng if rng is not None else self._rng_default, seed)
        n = int(n_simulations)
        theta_parts = []
        x_parts = []
        done = 0
        while done < n:
            m = min(int(chunk_size), n - done)
            theta = self.prior_sample(m, gen)
            x = self.simulate(theta, gen)
            theta_parts.append(theta)
            x_parts.append(x)
            done += m
            if verbose:
                print(f"[tree] simulated {done}/{n}")
        return np.concatenate(theta_parts, axis=0), np.concatenate(x_parts, axis=0)

    # -- simformer plumbing ------------------------------------------------
    def spec(self, token_dim: int = 50, **kwargs: Any) -> Any:
        try:
            from ..tokenizer import build_benchmark_spec

            return build_benchmark_spec(self.n_parameters, self.n_data)
        except Exception:  # pragma: no cover
            return None

    token_spec = spec

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        try:
            from ..attention_masks import build_attention_mask

            return build_attention_mask(
                "tree", n_theta=self.n_parameters, n_x=self.n_data, directed=directed
            )
        except Exception:  # pragma: no cover - fallback tree structure
            n = self.joint_dim
            mask = np.eye(n, dtype=bool)
            edges = [
                (0, 1),
                (0, 2),  # theta0 -> theta1, theta2
                (1, 3),
                (1, 4),  # theta1 -> x0, x1
                (2, 5),
                (2, 6),  # theta2 -> x2, x3
            ]
            for i, j in edges:
                mask[j, i] = True
            if not directed:
                mask = mask | mask.T
            return mask

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any) -> Any:
        try:
            from ..tokenizer import Tokenizer

            return Tokenizer(self.spec(token_dim=token_dim), token_dim=token_dim)
        except Exception:  # pragma: no cover
            return None

    def build_model(self, **kwargs: Any) -> Any:
        try:
            from ..transformer import build_score_network

            kwargs.setdefault("task", "tree")
            kwargs.setdefault("n_theta", self.n_parameters)
            kwargs.setdefault("n_x", self.n_data)
            return build_score_network(**kwargs)
        except Exception:  # pragma: no cover
            return None

    # -- misc --------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        d = self.config.to_dict()
        d.update(
            {
                "n_parameters": int(self.n_parameters),
                "n_data": int(self.n_data),
                "theta_std": float(self.theta_std),
                "obs_stds": tuple(float(s) for s in self.obs_stds),
                "parameter_names": tuple(self.parameter_names),
                "data_names": tuple(self.data_names),
            }
        )
        return d


# Aliases used by the task registry / scripts.
Task = TreeTask
Simulator = TreeTask
Tree = TreeTask


def build_task(config: Optional[Union[TreeConfig, Dict[str, Any]]] = None, **kwargs: Any) -> TreeTask:
    """Factory for the Tree task."""

    if isinstance(config, dict):
        merged = dict(config)
        merged.update(kwargs)
        return TreeTask(TreeConfig.from_dict(merged))
    if isinstance(config, TreeConfig):
        if kwargs:
            merged = config.to_dict()
            merged.update(kwargs)
            return TreeTask(TreeConfig.from_dict(merged))
        return TreeTask(config)
    return TreeTask(TreeConfig.from_dict(kwargs) if kwargs else None)


# ---------------------------------------------------------------------------
# Module-level convenience wrappers around a cached default task
# ---------------------------------------------------------------------------
_DEFAULT_TASK: Optional[TreeTask] = None


def _default_task() -> TreeTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = TreeTask()
    return _DEFAULT_TASK


def prior_sample(n_samples: int = 1, rng: Any = None, **kwargs: Any) -> np.ndarray:
    task = _default_task() if not kwargs else build_task(**kwargs)
    return task.prior_sample(n_samples, rng)


def log_prior(theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    task = _default_task() if not kwargs else build_task(**kwargs)
    return task.log_prior(theta)


def simulate(theta: np.ndarray, rng: Any = None, **kwargs: Any) -> np.ndarray:
    return _default_task().simulate(theta, rng, **kwargs)


def log_likelihood(x: np.ndarray, theta: np.ndarray, **kwargs: Any) -> np.ndarray:
    task = _default_task() if not kwargs else build_task(**kwargs)
    return task.log_likelihood(x, theta)


def log_joint(theta: np.ndarray, x: np.ndarray, **kwargs: Any) -> np.ndarray:
    task = _default_task() if not kwargs else build_task(**kwargs)
    return task.log_joint(theta, x)


def posterior_log_prob(theta: np.ndarray, x_obs: np.ndarray, **kwargs: Any) -> np.ndarray:
    return _default_task().posterior_log_prob(theta, x_obs, **kwargs)


def reference_posterior_sample(x_obs: np.ndarray, n_samples: int = 1000, rng: Any = None, **kwargs: Any) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples, rng, **kwargs)


def make_dataset(n_simulations: int, rng: Any = None, seed: Optional[int] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _default_task().make_dataset(n_simulations, rng=rng, seed=seed, **kwargs)
