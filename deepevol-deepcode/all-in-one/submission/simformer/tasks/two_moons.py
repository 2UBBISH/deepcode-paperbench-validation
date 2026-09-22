"""Two Moons benchmark task.

Implements the *Two Moons* task used in Lueckmann et al. (2021) and described in
Appendix A2.2 of the Simformer paper:

    theta ~ U(-1, 1)^2

    x | theta = [ r cos(alpha) + 0.25 ]   +   [ -|theta_1 + theta_2| / sqrt(2) ]
               [ r sin(alpha)        ]       [ (-theta_1 + theta_2) / sqrt(2) ]

    alpha ~ U(-pi/2, pi/2),   r ~ N(0.1, 0.01)

so that ``theta in R^2`` and ``x in R^2``.

The simulator introduces randomness through the two latent variables ``alpha``
(crescent angle) and ``r`` (crescent thickness).  Because the mapping
``(r, alpha) -> x`` is invertible (with Jacobian ``r``), the *marginal*
likelihood ``p(x | theta)`` needed for ground-truth MCMC reference sampling can
be written in closed form::

    w    = x - t(theta) - [0.25, 0]                 # t(theta) = the theta term
    r    = ||w||,           alpha = atan2(w_y, w_x)
    p(x | theta) = p_r(r) * p_alpha(alpha) / r      if  w_x > 0 else 0

where ``p_r`` is the normal density of ``r`` and ``p_alpha = 1/pi`` on
``(-pi/2, pi/2)``.  This makes the task directly usable by
:mod:`simformer.reference.mcmc` for arbitrary conditional reference sampling.

Reference posterior samples (Sec. A2.2): random-direction slice sampling
(1000 steps) followed by 3000 steps of Metropolis-Hastings with step size 0.01,
keeping only the last sample of each chain.  That protocol lives in
:mod:`simformer.reference.mcmc`; a light-weight in-module MH sampler is provided
as a fallback so the task is self-contained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:  # prefer the package base class
    from simformer.tasks import TaskBase
except Exception:  # pragma: no cover - fallback for isolated imports
    try:
        from . import TaskBase  # type: ignore
    except Exception:

        class TaskBase:  # minimal stand-in
            name = "task"
            n_parameters = 0
            n_data = 0

            def to_joint(self, theta, x):  # pragma: no cover - trivial
                theta = np.atleast_2d(np.asarray(theta, dtype=np.float64))
                x = np.atleast_2d(np.asarray(x, dtype=np.float64))
                return np.concatenate([theta, x], axis=-1)

            def split_joint(self, joint):  # pragma: no cover - trivial
                joint = np.atleast_2d(np.asarray(joint, dtype=np.float64))
                return joint[:, : self.n_parameters], joint[:, self.n_parameters :]


__all__ = [
    "TwoMoonsConfig",
    "TwoMoonsTask",
    "Task",
    "Simulator",
    "TwoMoons",
    "build_task",
    "prior_sample",
    "log_prior",
    "simulate",
    "log_likelihood",
    "posterior_log_prob",
    "reference_posterior_sample",
    "make_dataset",
    "DEFAULT_PRIOR_LOW",
    "DEFAULT_PRIOR_HIGH",
    "DEFAULT_R_MEAN",
    "DEFAULT_R_STD",
    "DEFAULT_MOON_OFFSET",
]

# ---------------------------------------------------------------------------
# defaults (Appendix A2.2)
# ---------------------------------------------------------------------------
DEFAULT_DIM = 2
DEFAULT_PRIOR_LOW = -1.0
DEFAULT_PRIOR_HIGH = 1.0
DEFAULT_R_MEAN = 0.1
#: The paper prints ``r ~ N(0.1, 0.012)``; the canonical Two Moons task
#: (Lueckmann et al. 2021 / Greenberg et al. 2019) uses a standard deviation of
#: 0.01, which we adopt as default (the printed value is treated as a typo).
#: Set ``r_std=0.012`` to follow the printed number verbatim.
DEFAULT_R_STD = 0.01
PAPER_R_STD = 0.012
DEFAULT_MOON_OFFSET = 0.25
DEFAULT_TRUNCATION = 1.0

_LOG_2PI = math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy generator (accepting legacy RandomState)."""
    if rng is not None:
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(int(np.random.randint(0, 2**31 - 1)) if seed is None else seed)
    return np.random.default_rng(seed)


def _as_2d_theta(theta: Any, n_parameters: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta.reshape(1, -1)
    if theta.shape[-1] != n_parameters:
        raise ValueError(f"expected last dimension {n_parameters}, got shape {theta.shape}")
    return theta


def _theta_term(theta: np.ndarray) -> np.ndarray:
    """The theta-dependent translation of the Two Moons mapping (Eq. A2.2-2)."""
    theta = np.atleast_2d(theta)
    t0 = -np.abs(theta[:, 0] + theta[:, 1]) / _SQRT2
    t1 = (-theta[:, 0] + theta[:, 1]) / _SQRT2
    return np.stack([t0, t1], axis=-1)


def _moon_point(alpha: np.ndarray, r: np.ndarray, offset: float = DEFAULT_MOON_OFFSET) -> np.ndarray:
    """The rotated/scaled crescent point ``[r cos(a) + offset, r sin(a)]``."""
    return np.stack([r * np.cos(alpha) + offset, r * np.sin(alpha)], axis=-1)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class TwoMoonsConfig:
    """Configuration of the Two Moons task (Appendix A2.2)."""

    n_dim: int = DEFAULT_DIM
    prior_low: float = DEFAULT_PRIOR_LOW
    prior_high: float = DEFAULT_PRIOR_HIGH
    r_mean: float = DEFAULT_R_MEAN
    r_std: float = DEFAULT_R_STD
    moon_offset: float = DEFAULT_MOON_OFFSET
    name: str = "two_moons"
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def prior_volume(self) -> float:
        return float((self.prior_high - self.prior_low) ** self.n_dim)

    @property
    def log_prior_constant(self) -> float:
        return float(-math.log(self.prior_volume))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_dim": self.n_dim,
            "prior_low": self.prior_low,
            "prior_high": self.prior_high,
            "r_mean": self.r_mean,
            "r_std": self.r_std,
            "moon_offset": self.moon_offset,
            "seed": self.seed,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "TwoMoonsConfig":
        data = dict(cfg or {})
        data.update({k: v for k, v in kwargs.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = {k: data.pop(k) for k in list(data) if k not in known}
        cfg_obj = cls(**data)
        if unknown:
            cfg_obj.extra.update(unknown)
        return cfg_obj


# ---------------------------------------------------------------------------
# task
# ---------------------------------------------------------------------------
class TwoMoonsTask(TaskBase):
    """Two Moons simulator with exact semi-analytic likelihood."""

    name = "two_moons"
    n_parameters = 2
    n_data = 2
    parameter_names: Tuple[str, ...] = ("theta_1", "theta_2")
    data_names: Tuple[str, ...] = ("x_1", "x_2")

    def __init__(
        self,
        config: Optional[TwoMoonsConfig] = None,
        *,
        n_dim: Optional[int] = None,
        prior_low: Optional[float] = None,
        prior_high: Optional[float] = None,
        r_mean: Optional[float] = None,
        r_std: Optional[float] = None,
        moon_offset: Optional[float] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            cfg = TwoMoonsConfig()
        elif isinstance(config, dict):
            cfg = TwoMoonsConfig.from_dict(config)
        else:
            cfg = config
        overrides = dict(
            n_dim=n_dim,
            prior_low=prior_low,
            prior_high=prior_high,
            r_mean=r_mean,
            r_std=r_std,
            moon_offset=moon_offset,
            name=name,
            seed=seed,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        if kwargs:
            cfg.extra.update(kwargs)
        self.config = cfg
        self.n_dim = int(cfg.n_dim)
        self.n_parameters = self.n_dim
        self.n_data = self.n_dim
        self.prior_low = float(cfg.prior_low)
        self.prior_high = float(cfg.prior_high)
        self.r_mean = float(cfg.r_mean)
        self.r_std = float(cfg.r_std)
        self.moon_offset = float(cfg.moon_offset)
        self.seed = int(cfg.seed)

    # ------------------------------------------------------------------
    # prior
    # ------------------------------------------------------------------
    def prior_sample(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.ndarray:
        """Draw ``n_samples`` parameters from ``U(-1, 1)^2``."""
        rng = _rng(rng, self.seed if seed is None else seed)
        return rng.uniform(self.prior_low, self.prior_high, size=(int(n_samples), self.n_parameters))

    # sbi-style alias
    sample_prior = prior_sample

    def log_prior(self, theta: Any) -> np.ndarray:
        """Uniform prior log density on the box ``[prior_low, prior_high]^2``."""
        theta = _as_2d_theta(theta, self.n_parameters)
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high), axis=-1)
        out = np.full(theta.shape[0], -np.inf, dtype=np.float64)
        out[inside] = self.config.log_prior_constant
        return out

    # ------------------------------------------------------------------
    # simulator
    # ------------------------------------------------------------------
    def simulate(
        self,
        theta: Any,
        rng: Optional[np.random.Generator] = None,
        *,
        n_samples: Optional[int] = None,
        add_noise: bool = True,
        seed: Optional[int] = None,
        return_components: bool = False,
    ) -> Any:
        """Simulate observations ``x`` given parameters ``theta`` (Eq. A2.2-2).

        ``add_noise=False`` replaces the latent radius ``r`` by its mean, i.e. the
        radial noise is switched off (the task has no additive observation noise,
        the randomness is entirely carried by ``(r, alpha)``).
        """
        rng = _rng(rng, self.seed if seed is None else seed)
        theta_arr = _as_2d_theta(theta, self.n_parameters)
        if n_samples is not None and theta_arr.shape[0] == 1 and n_samples > 1:
            theta_arr = np.repeat(theta_arr, int(n_samples), axis=0)

        n = theta_arr.shape[0]
        alpha = rng.uniform(-np.pi / 2.0, np.pi / 2.0, size=n)
        if add_noise:
            r = rng.normal(self.r_mean, self.r_std, size=n)
        else:
            r = np.full(n, self.r_mean, dtype=np.float64)

        x = _moon_point(alpha, r, self.moon_offset) + _theta_term(theta_arr)
        if return_components:
            return x, {"alpha": alpha, "r": r}
        return x

    # sbi-style alias
    simulator = simulate

    def __call__(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng, **kwargs)
        return theta, x

    # ------------------------------------------------------------------
    # likelihood
    # ------------------------------------------------------------------
    def _log_likelihood_impl(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Closed-form ``log p(x | theta)`` after marginalising ``(r, alpha)``."""
        x = np.asarray(x, dtype=np.float64)
        t = _theta_term(theta) - _theta_term(theta) + _theta_term(theta)  # broadcast shape (m, 2)
        w = x - t - np.array([self.moon_offset, 0.0], dtype=np.float64)

        w_x = w[..., 0]
        w_y = w[..., 1]
        r = np.hypot(w_x, w_y)
        out = np.full(np.broadcast(w_x, w_y).shape, -np.inf, dtype=np.float64)

        valid = (w_x > 0.0) & (r > 1e-12)
        if np.any(valid):
            r_v = np.clip(r[valid], 1e-12, None)
            log_pr = -0.5 * ((r_v - self.r_mean) / self.r_std) ** 2 - math.log(self.r_std) - 0.5 * _LOG_2PI
            # alpha is uniform on (-pi/2, pi/2)  ->  log p = -log(pi)
            # + polar Jacobian 1/r
            out[valid] = log_pr - math.log(np.pi) - np.log(r_v)
        return out

    def log_likelihood(self, x: Any, theta: Any) -> np.ndarray:
        """``log p(x | theta)`` (exact up to the truncated support ``w_x > 0``).

        Shapes follow broadcasting over the last (variable) axis, so both
        ``log_likelihood(x_single, theta_batch)`` and
        ``log_likelihood(x_batch, theta_batch)`` work.
        """
        theta_arr = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        if theta_arr.shape[-1] != self.n_parameters:
            raise ValueError(f"theta must have last dimension {self.n_parameters}")
        x_arr = np.asarray(x, dtype=np.float64)

        if x_arr.ndim == 1:
            if theta_arr.shape[0] == 1:
                return np.asarray(self._log_likelihood_impl(x_arr.reshape(1, -1), theta_arr)).reshape(())
            return self._log_likelihood_impl(np.broadcast_to(x_arr, theta_arr.shape), theta_arr)

        x_arr = np.atleast_2d(x_arr)
        if x_arr.shape[0] == 1 and theta_arr.shape[0] > 1:
            x_arr = np.broadcast_to(x_arr, theta_arr.shape)
        elif theta_arr.shape[0] == 1 and x_arr.shape[0] > 1:
            theta_arr = np.broadcast_to(theta_arr, x_arr.shape)
        return self._log_likelihood_impl(x_arr, theta_arr)

    # ------------------------------------------------------------------
    # joint / posterior log densities
    # ------------------------------------------------------------------
    def log_joint(self, theta: Any, x: Any) -> np.ndarray:
        return self.log_prior(theta) + self.log_likelihood(x, theta)

    def posterior_log_prob(self, theta: Any, x_obs: Any, normalize: bool = False) -> np.ndarray:
        """Unnormalised posterior ``log p(theta | x_obs)`` (up to an additive constant)."""
        value = self.log_prior(theta) + self.log_likelihood(x_obs, theta)
        if normalize:
            value = value - np.max(value)
        return value

    # ground-truth alias used by reference/eval code
    ground_truth_log_posterior = posterior_log_prob

    # ------------------------------------------------------------------
    # reference posterior sampling
    # ------------------------------------------------------------------
    def _mh_reference(
        self,
        x_obs: Any,
        n_samples: int,
        rng: np.random.Generator,
        *,
        n_steps: int = 4000,
        step_size: float = 0.05,
        burn_in: int = 1000,
        thin: int = 1,
    ) -> np.ndarray:
        """Self-contained random-walk MH fallback (last sample per chain)."""
        n_chains = int(n_samples)
        x_obs = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        current = rng.uniform(self.prior_low, self.prior_high, size=(n_chains, self.n_parameters))
        logp = self.posterior_log_prob(current, x_obs)
        logp = np.where(np.isfinite(logp), logp, -np.inf)

        accept = 0
        n_accept_steps = max(n_steps - burn_in, 1)
        for step in range(n_steps):
            proposal = current + step_size * rng.normal(size=current.shape)
            prop_logp = self.posterior_log_prob(proposal, x_obs)
            log_alpha = prop_logp - logp
            u = np.log(rng.uniform(size=n_chains))
            move = (u < log_alpha) & np.isfinite(prop_logp)
            current[move] = proposal[move]
            logp[move] = prop_logp[move]
            if step >= burn_in:
                accept += int(np.count_nonzero(move))
            # crude adaptation over the first half of the burn-in
            if step < burn_in // 2 and (step + 1) % 200 == 0:
                rate = accept / max(200 * n_chains, 1)
                step_size *= math.exp(0.5 * (rate - 0.25))
                accept = 0
        return current if thin <= 1 else current

    def reference_posterior_sample(
        self,
        x_obs: Any,
        n_samples: int = 1000,
        rng: Optional[np.random.Generator] = None,
        *,
        seed: Optional[int] = None,
        burn_in: int = 1000,
        thinning: int = 1,
        use_mcmc_module: bool = True,
    ) -> np.ndarray:
        """Ground-truth posterior samples for ``x_obs``.

        Tries the project's canonical reference sampler
        (:mod:`simformer.reference.mcmc`, slice sampling 1000 + MH 3000 with step
        0.01 — Appendix A2.2) and falls back to the in-module MH sampler.
        """
        rng = _rng(rng, self.seed if seed is None else seed)
        x_obs = np.asarray(x_obs, dtype=np.float64).reshape(-1)
        if use_mcmc_module:
            try:  # pragma: no cover - depends on environment
                try:
                    from simformer.reference.mcmc import sample_reference
                except Exception:
                    from ..reference.mcmc import sample_reference  # type: ignore
                mask = np.zeros(self.n_parameters + self.n_data, dtype=np.float64)
                mask[: self.n_parameters] = 1.0
                values = np.concatenate([np.zeros(self.n_parameters), x_obs])
                samples = sample_reference(
                    self,
                    condition_mask=mask,
                    condition_values=values,
                    n_samples=int(n_samples),
                    seed=int(seed) if seed is not None else None,
                    rng=rng,
                    return_full=False,
                )
                samples = np.asarray(samples, dtype=np.float64)
                if samples.ndim == 1:
                    samples = samples.reshape(-1, self.n_parameters)
                if samples.shape[-1] != self.n_parameters:
                    samples = samples[:, : self.n_parameters]
                return samples
            except Exception:
                pass
        return self._mh_reference(x_obs, int(n_samples), rng, burn_in=burn_in, thin=thinning)

    ground_truth_posterior = reference_posterior_sample

    def posterior_mean_std(
        self,
        x_obs: Any,
        *,
        n_samples: int = 2000,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo posterior mean/std (Two Moons has no closed-form posterior)."""
        samples = self.reference_posterior_sample(x_obs, n_samples=n_samples, rng=rng, seed=seed)
        return samples.mean(axis=0), samples.std(axis=0)

    # ------------------------------------------------------------------
    # datasets
    # ------------------------------------------------------------------
    def sample_joint(self, n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
        rng = _rng(rng, self.seed)
        theta = self.prior_sample(n_samples, rng)
        x = self.simulate(theta, rng, **kwargs)
        return theta, x

    def make_dataset(
        self,
        n_simulations: int,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        chunk_size: int = 1024,
        verbose: bool = False,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ``(theta, x)`` pairs, chunked to limit peak memory."""
        rng = _rng(rng, self.seed if seed is None else seed)
        n_simulations = int(n_simulations)
        chunk_size = max(int(chunk_size), 1)
        thetas, xs = [], []
        done = 0
        while done < n_simulations:
            n = min(chunk_size, n_simulations - done)
            theta = self.prior_sample(n, rng)
            x = self.simulate(theta, rng, **kwargs)
            thetas.append(theta)
            xs.append(np.asarray(x, dtype=np.float64).reshape(n, -1))
            done += n
            if verbose:  # pragma: no cover - logging only
                print(f"[two_moons] simulated {done}/{n_simulations}")
        return np.concatenate(thetas, axis=0), np.concatenate(xs, axis=0)

    # ------------------------------------------------------------------
    # plumbing (tokenizer / masks / model)
    # ------------------------------------------------------------------
    def spec(self, token_dim: int = 50, **kwargs: Any) -> Any:
        try:
            from simformer.tokenizer import build_benchmark_spec
        except Exception:  # pragma: no cover
            from ..tokenizer import build_benchmark_spec  # type: ignore
        return build_benchmark_spec(self.n_parameters, self.n_data)

    token_spec = spec

    def attention_mask(self, directed: bool = True, **kwargs: Any) -> np.ndarray:
        try:
            from simformer.attention_masks import build_attention_mask
        except Exception:  # pragma: no cover
            from ..attention_masks import build_attention_mask  # type: ignore
        return build_attention_mask(
            "two_moons",
            n_theta=self.n_parameters,
            n_x=self.n_data,
            directed=directed,
            **kwargs,
        )

    def build_tokenizer(self, token_dim: int = 50, **kwargs: Any) -> Any:
        try:
            from simformer.tokenizer import Tokenizer
        except Exception:  # pragma: no cover
            from ..tokenizer import Tokenizer  # type: ignore
        return Tokenizer(spec=self.spec(), token_dim=token_dim, **kwargs)

    def build_model(self, **kwargs: Any) -> Any:
        """Build the Simformer score network wired to this task's spec and mask."""
        try:
            from simformer.transformer import build_score_network
        except Exception:  # pragma: no cover
            from ..transformer import build_score_network  # type: ignore
        spec = self.spec()
        tokenizer = kwargs.pop("tokenizer", None)
        mask = kwargs.pop("attention_mask", None)
        if mask is None:
            mask = self.attention_mask(directed=True)
        attempt_kwargs = dict(kwargs)
        attempt_kwargs.pop("task", None)
        model = None
        for candidate in (
            dict(task=self.name, spec=spec, tokenizer=tokenizer, attention_mask=mask, **attempt_kwargs),
            dict(task=self.name, spec=spec, **attempt_kwargs),
            dict(spec=spec, **attempt_kwargs),
        ):
            try:
                model = build_score_network(**candidate)
                break
            except TypeError:
                continue
        if model is None:  # last resort
            model = build_score_network(task=self.name, spec=spec)
        try:  # attach the mask for consumers that read it off the model
            model.attention_mask = mask
        except Exception:
            pass
        return model

    build_score_network = build_model

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = {
            "name": self.name,
            "n_parameters": self.n_parameters,
            "n_data": self.n_data,
            "parameter_names": list(self.parameter_names),
            "data_names": list(self.data_names),
        }
        data.update(self.config.to_dict())
        return data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TwoMoonsTask(n_parameters={self.n_parameters}, n_data={self.n_data}, "
            f"r_mean={self.r_mean}, r_std={self.r_std})"
        )


# ---------------------------------------------------------------------------
# aliases / factory
# ---------------------------------------------------------------------------
Task = TwoMoonsTask
Simulator = TwoMoonsTask
TwoMoons = TwoMoonsTask


def build_task(config: Optional[Any] = None, **kwargs: Any) -> TwoMoonsTask:
    """Factory accepting a :class:`TwoMoonsConfig`, a dict, or keyword overrides."""
    return TwoMoonsTask(config, **kwargs)


# ---------------------------------------------------------------------------
# module-level convenience wrappers (cached default task)
# ---------------------------------------------------------------------------
_DEFAULT_TASK: Optional[TwoMoonsTask] = None


def _default_task() -> TwoMoonsTask:
    global _DEFAULT_TASK
    if _DEFAULT_TASK is None:
        _DEFAULT_TASK = TwoMoonsTask()
    return _DEFAULT_TASK


def prior_sample(n_samples: int = 1, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().prior_sample(n_samples, rng, **kwargs)


def log_prior(theta: Any, **kwargs: Any) -> np.ndarray:
    return _default_task().log_prior(theta)


def simulate(theta: Any, rng: Optional[np.random.Generator] = None, **kwargs: Any) -> np.ndarray:
    return _default_task().simulate(theta, rng, **kwargs)


def log_likelihood(x: Any, theta: Any, **kwargs: Any) -> np.ndarray:
    return _default_task().log_likelihood(x, theta)


def posterior_log_prob(theta: Any, x_obs: Any, **kwargs: Any) -> np.ndarray:
    return _default_task().posterior_log_prob(theta, x_obs, **kwargs)


def reference_posterior_sample(
    x_obs: Any,
    n_samples: int = 1000,
    rng: Optional[np.random.Generator] = None,
    **kwargs: Any,
) -> np.ndarray:
    return _default_task().reference_posterior_sample(x_obs, n_samples, rng, **kwargs)


def make_dataset(
    n_simulations: int,
    *,
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    return _default_task().make_dataset(n_simulations, rng=rng, seed=seed, **kwargs)
