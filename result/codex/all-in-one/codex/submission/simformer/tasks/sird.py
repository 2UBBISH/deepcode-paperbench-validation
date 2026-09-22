"""SIRD model with a time dependent contact rate (Sec. 4.3, Appendix A2.2).

The simulator has three (mechanistic) parameters: the recovery rate ``gamma``,
the death rate ``mu`` and the contact rate ``beta(t)``.  The contact rate is
modelled as a *function of time* (an infinite dimensional parameter): its
(logit) values follow a Gaussian process prior with an RBF kernel
``k(t1, t2) = 2.5^2 exp(-0.5 (t1 - t2)^2 / 7^2)`` which is squashed through a
sigmoid so that ``beta(t) in [0, 1]``.  ``gamma`` and ``mu`` have a uniform
prior on ``[0, 0.5]``.  Observations are log-normal with a standard deviation of
``0.05``.

Because the identifiers of the variables of the contact rate (and of the
observations) are formed from a *shared* embedding plus a random Fourier
embedding of the time point, the trained model can be queried at an arbitrary
number of time points -- that is how the Simformer performs inference in an
infinite dimensional parameter space.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp

from ..masks import sird_mask
from ..problem import Problem
from .base import Task


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def rbf_kernel(t1: np.ndarray, t2: np.ndarray, length_scale: float = 7.0,
               amplitude: float = 2.5) -> np.ndarray:
    """``k(t1, t2) = amplitude^2 exp(-0.5 (t1 - t2)^2 / length_scale^2)``."""
    t1 = np.asarray(t1, dtype=float).reshape(-1, 1)
    t2 = np.asarray(t2, dtype=float).reshape(1, -1)
    return (amplitude ** 2) * np.exp(-0.5 * (t1 - t2) ** 2 / length_scale ** 2)


def simulate_sird(gamma: np.ndarray, mu: np.ndarray, beta: np.ndarray,
                  beta_times: np.ndarray, observation_times: np.ndarray,
                  initial_state=(0.99, 0.01, 0.0, 0.0),
                  noise_std: float = 0.05,
                  rng: Optional[np.random.Generator] = None):
    """Simulate the SIRD ODE and return noisy observations.

    Returns ``(I, R, D)`` evaluated at ``observation_times`` (log-normal noise
    with mean equal to the true value and standard deviation ``noise_std``).
    """
    gamma = np.atleast_1d(gamma)
    mu = np.atleast_1d(mu)
    beta = np.atleast_2d(beta)
    rng = rng if rng is not None else np.random.default_rng()
    observation_times = np.asarray(observation_times, dtype=float)
    beta_times = np.asarray(beta_times, dtype=float)
    n = beta.shape[0]
    out = np.empty((n, len(observation_times), 3))
    for i in range(n):
        bt = np.union1d([0.0], beta_times)
        values = np.interp(bt, beta_times, beta[i])

        def rhs(t, state):
            S, I, R, D = state
            b = np.interp(t, bt, values)
            return [-b * S * I, b * S * I - gamma[i % len(gamma)] * I
                    - mu[i % len(mu)] * I,
                    gamma[i % len(gamma)] * I, mu[i % len(mu)] * I]

        t_eval = np.union1d([0.0], observation_times)
        solution = solve_ivp(rhs, (0.0, max(t_eval[-1], 1e-6)),
                             initial_state, t_eval=t_eval, method="RK45",
                             rtol=1e-7, atol=1e-9, max_step=0.1)
        if solution.y.shape[1] != len(t_eval):
            out[i] = np.nan
            continue
        values_at_obs = solution.y[:, np.searchsorted(t_eval,
                                                      observation_times)]
        out[i] = values_at_obs[1:].T                     # I, R, D

    # log-normal observation noise with mean equal to the true value
    s = noise_std
    mean = np.clip(out, 1e-12, None)
    log_mean = np.log(mean) - 0.5 * s ** 2
    noisy = np.exp(log_mean + s * rng.normal(size=out.shape))
    return noisy


class SIRDTask(Task):
    """SIRD with a GP prior on the time dependent contact rate."""

    name = "sird"
    n_global_params = 2
    t_end = 60.0
    noise_std = 0.05
    gamma_range = (0.0, 0.5)
    mu_range = (0.0, 0.5)

    def __init__(self, n_beta: int = 10, n_obs: int = 6, t_end: Optional[float] = None,
                 beta_time_range: Optional[tuple] = None, seed: int = 0):
        self.n_beta = n_beta
        self.n_obs = n_obs
        self.n_params = self.n_global_params + n_beta
        self.n_data = 3 * n_obs
        if t_end is not None:
            self.t_end = t_end
        self.beta_time_range = beta_time_range or (0.0, self.t_end)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ prior
    def prior_sample(self, n, rng):
        """Return the global parameters only (see :meth:`joint_sample`)."""
        gamma = rng.uniform(*self.gamma_range, size=(n, 1))
        mu = rng.uniform(*self.mu_range, size=(n, 1))
        return np.concatenate([gamma, mu], axis=-1)

    def sample_beta(self, times: np.ndarray, n: int,
                    rng: np.random.Generator) -> np.ndarray:
        """Sample the (sigmoid transformed) contact rate at ``times``."""
        times = np.asarray(times, dtype=float)
        K = rbf_kernel(times, times) + 1e-8 * np.eye(len(times))
        L = np.linalg.cholesky(K)
        z = rng.normal(size=(n, len(times)))
        beta_hat = z @ L.T
        return sigmoid(beta_hat)

    def log_prior(self, theta, beta_times):
        """Log density of ``(gamma, mu, beta_hat(times))`` (with the sigmoid
        change of variables)."""
        theta = np.atleast_2d(theta)
        gamma, mu = theta[:, 0], theta[:, 1]
        beta = np.atleast_2d(theta[:, self.n_global_params:])
        log_p = np.zeros(theta.shape[0])
        in_range = ((gamma >= self.gamma_range[0]) & (gamma <= self.gamma_range[1])
                    & (mu >= self.mu_range[0]) & (mu <= self.mu_range[1]))
        log_p += np.where(in_range, -np.log(0.5 * 0.5), -np.inf)
        beta_hat = np.log(beta / (1.0 - beta))            # inverse sigmoid
        K = rbf_kernel(beta_times, beta_times) + 1e-8 * np.eye(len(beta_times))
        sign, log_det = np.linalg.slogdet(K)
        precision = np.linalg.inv(K)
        quad = np.einsum("ni,ij,nj->n", beta_hat, precision, beta_hat)
        log_p += -0.5 * quad - 0.5 * log_det - 0.5 * len(beta_times) * np.log(
            2 * np.pi)
        log_p += np.sum(np.log(beta * (1.0 - beta)), axis=-1)
        return log_p

    # -------------------------------------------------------------- simulation
    def joint_sample(self, n, rng):
        theta_global = self.prior_sample(n, rng)
        beta_times = np.sort(rng.uniform(self.beta_time_range[0],
                                         self.beta_time_range[1],
                                         size=(n, self.n_beta)), axis=-1)
        obs_times = np.sort(rng.uniform(0.0, self.t_end, size=(n, self.n_obs)),
                            axis=-1)
        beta = np.empty((n, self.n_beta))
        observations = np.empty((n, self.n_obs, 3))
        for i in range(n):
            beta[i] = self.sample_beta(beta_times[i], 1, rng)[0]
            observations[i] = simulate_sird(
                theta_global[i, 0:1], theta_global[i, 1:2], beta[i:i + 1],
                beta_times[i], obs_times[i], noise_std=self.noise_std,
                rng=rng)[0]
        x = np.concatenate([observations[:, :, 0], observations[:, :, 1],
                            observations[:, :, 2]], axis=-1)
        params = np.concatenate([theta_global, beta], axis=-1)
        index = np.zeros((n, self.n_variables), dtype=np.float32)
        index[:, self.n_global_params:self.n_global_params + self.n_beta] = \
            beta_times
        for k in range(3):
            start = self.n_global_params + self.n_beta + k * self.n_obs
            index[:, start:start + self.n_obs] = obs_times
        metadata = {"beta_times": beta_times, "observation_times": obs_times}
        return params, x, index, metadata

    # ---------------------------------------------------------------- density
    def _noise_free(self, params, beta_times, obs_times):
        params = np.atleast_2d(params)
        out = np.empty((params.shape[0], len(obs_times), 3))
        for i in range(params.shape[0]):
            out[i] = simulate_sird(
                params[i, 0:1], params[i, 1:2],
                params[i, self.n_global_params:], beta_times, obs_times,
                noise_std=0.0, rng=np.random.default_rng(0))[0]
        return out

    def log_joint(self, params, x, beta_times=None, obs_times=None):
        params = np.atleast_2d(params)
        x = np.atleast_2d(x)
        if beta_times is None:
            beta_times = self.default_beta_times()
        if obs_times is None:
            obs_times = self.default_observation_times()
        log_p = np.atleast_1d(self.log_prior(params, beta_times))
        mean = self._noise_free(params, beta_times, obs_times)
        s = self.noise_std
        for k in range(3):
            obs = x[:, k * len(obs_times):(k + 1) * len(obs_times)]
            m = np.clip(mean[:, :, k], 1e-12, None)
            log_mean = np.log(m) - 0.5 * s ** 2
            finite = np.isfinite(obs)
            log_like = np.where(
                finite,
                -np.log(np.clip(obs, 1e-12, None)) - np.log(s)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * ((np.log(np.clip(obs, 1e-12, None)) - log_mean) / s) ** 2,
                0.0)
            log_p = log_p + np.sum(log_like, axis=-1)
        return log_p

    def default_beta_times(self, n: int = 21) -> np.ndarray:
        return np.linspace(self.beta_time_range[0], self.beta_time_range[1], n)

    def default_observation_times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_end, self.n_obs)

    # -------------------------------------------------------------- structure
    def base_mask(self):
        return sird_mask(self.default_beta_times(),
                         np.concatenate([
                             self.default_observation_times()] * 3),
                         n_global_params=self.n_global_params)

    def variable_kind(self):
        kinds = np.zeros(self.n_variables, dtype=np.int64)
        kinds[1] = 1                                   # death rate
        kinds[self.n_global_params:self.n_global_params + self.n_beta] = 2
        for k in range(3):
            start = self.n_global_params + self.n_beta + k * self.n_obs
            kinds[start:start + self.n_obs] = 3 + k    # I, R, D
        return kinds

    def use_fourier(self):
        mask = np.zeros(self.n_variables, dtype=bool)
        mask[self.n_global_params:] = True
        return mask

    def mask_factory(self):
        """Mask builder; the contact rate / observations are indexed in time."""
        n_global, n_beta, n_obs = (self.n_global_params, self.n_beta,
                                   self.n_obs)

        def builder(condition_state, index=None, metadata=None):
            condition_state = np.asarray(condition_state)
            batch = condition_state.shape[0]
            if index is None:
                return np.broadcast_to(
                    self.base_mask()[None],
                    (batch, self.n_variables, self.n_variables)).copy()
            index = np.asarray(index, dtype=float)
            masks = np.empty((batch, self.n_variables, self.n_variables),
                             dtype=bool)
            for b in range(batch):
                beta_times = index[b, n_global:n_global + n_beta]
                data_times = index[b, n_global + n_beta:]
                masks[b] = sird_mask(beta_times, data_times,
                                     n_global_params=n_global)
            return masks

        return builder

    # ------------------------------------------------------ alternative layout
    def problem(self) -> Problem:
        return self._problem(self.n_beta, self.n_obs)

    def inference_problem(self, n_beta: int, n_obs: int) -> Problem:
        """Problem with a different number of contact rate / observation times.

        The transformer is a set function over tokens, so a trained Simformer can
        be evaluated with any number of (arbitrarily placed) time points.
        """
        return self._problem(n_beta, n_obs)

    def _problem(self, n_beta: int, n_obs: int) -> Problem:
        n_params = self.n_global_params + n_beta
        n_data = 3 * n_obs
        n_variables = n_params + n_data
        kinds = np.zeros(n_variables, dtype=np.int64)
        kinds[1] = 1
        kinds[self.n_global_params:n_params] = 2
        for k in range(3):
            start = n_params + k * n_obs
            kinds[start:start + n_obs] = 3 + k
        use_fourier = np.zeros(n_variables, dtype=bool)
        use_fourier[self.n_global_params:] = True
        beta_times = self.default_beta_times(max(n_beta, 2))
        obs_times = np.linspace(0.0, self.t_end, n_obs)
        base = sird_mask(beta_times[:n_beta],
                         np.concatenate([obs_times] * 3),
                         n_global_params=self.n_global_params)

        def mask_builder(condition_state, index=None, metadata=None):
            condition_state = np.asarray(condition_state)
            if index is None:
                return np.broadcast_to(base[None], (condition_state.shape[0],
                                                    n_variables,
                                                    n_variables)).copy()
            index = np.asarray(index, dtype=float)
            out = np.empty((condition_state.shape[0], n_variables, n_variables),
                           dtype=bool)
            for b in range(condition_state.shape[0]):
                out[b] = sird_mask(index[b, self.n_global_params:n_params],
                                   index[b, n_params:],
                                   n_global_params=self.n_global_params)
            return out

        return Problem(
            name=self.name,
            n_variables=n_variables,
            n_params=n_params,
            n_data=n_data,
            sample_batch=self.joint_sample,
            mask_builder=mask_builder,
            variable_kind=kinds,
            use_fourier=use_fourier,
            index_dim=1,
            n_kinds=6,
            log_joint=self.log_joint,
        )

    # ---------------------------------------------------------- synthetic data
    def synthetic_observation(self, theta_true_global, beta_true,
                              beta_times, observation_times,
                              rng: Optional[np.random.Generator] = None):
        """Create a synthetic observation for a given ground truth."""
        rng = rng if rng is not None else self._rng
        obs = simulate_sird(np.atleast_1d(theta_true_global[0]),
                            np.atleast_1d(theta_true_global[1]),
                            np.atleast_2d(beta_true), np.asarray(beta_times),
                            np.asarray(observation_times),
                            noise_std=self.noise_std, rng=rng)
        return obs[0]
