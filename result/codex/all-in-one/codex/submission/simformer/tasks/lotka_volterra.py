"""Lotka-Volterra task of Sec. 4.2 (Appendix A2.2).

The Lotka-Volterra equations describe the interaction of a prey (``x``) and a
predator (``y``) population,

    dx/dt = alpha x - beta x y
    dy/dt = delta x y - gamma y

with a sigmoid transformed Normal prior on the four parameters, scaled to the
range ``[1, 3]``, and Gaussian observation noise with ``sigma = 0.1``.  The
inference is performed for the *full time series* (no summary statistics) on a
uniform grid between ``t = 0`` and ``t = 15``; unstructured observations are
obtained by conditioning on a subset of (irregularly placed) time points only,
everything else is marginalised out by the Simformer.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..masks import lotka_volterra_mask
from ..problem import Problem
from .base import Task


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def lotka_volterra_rhs(t, state, params):
    x, y = state
    alpha, beta, delta, gamma = params
    return [alpha * x - beta * x * y, delta * x * y - gamma * y]


def simulate_lotka_volterra(theta: np.ndarray, times: np.ndarray,
                            initial_state: Sequence[float] = (1.0, 1.0),
                            noise_std: float = 0.1,
                            rng: Optional[np.random.Generator] = None,
                            max_step: float = 0.05) -> np.ndarray:
    """Simulate the Lotka-Volterra ODE at ``times`` for every row of ``theta``.

    Parameters
    ----------
    theta : ``(n, 4)`` parameters ``(alpha, beta, delta, gamma)``.
    times : ``(T,)`` ordered evaluation times (the initial state is applied at
        ``times[0]``).

    Returns
    -------
    ``(n, T, 2)`` array with the prey and predator population (including
    Gaussian observation noise with ``sigma = 0.1``).
    """
    theta = np.atleast_2d(theta)
    times = np.asarray(times, dtype=float)
    rng = rng if rng is not None else np.random.default_rng()
    out = np.empty((theta.shape[0], len(times), 2))
    for i, params in enumerate(theta):
        solution = solve_ivp(
            lambda t, y: lotka_volterra_rhs(t, y, params),
            (times[0], times[-1]), initial_state, t_eval=times,
            method="RK45", rtol=1e-8, atol=1e-8, max_step=max_step)
        if solution.y.shape[1] != len(times):
            # integration failed for (very unlikely) parameter values
            out[i] = np.nan
            continue
        out[i] = solution.y.T
    return out + rng.normal(0.0, noise_std, size=out.shape)


class LotkaVolterraTask(Task):
    """Lotka-Volterra with a fixed time grid and unstructured observations."""

    name = "lotka_volterra"
    n_params = 4
    n_data = 32                  # 16 prey + 16 predator observations
    noise_std = 0.1
    t_end = 15.0
    n_grid = 16

    def __init__(self, n_grid: Optional[int] = None, t_end: Optional[float] = None,
                 random_times: bool = False, seed: int = 0):
        if n_grid is not None:
            self.n_grid = n_grid
        if t_end is not None:
            self.t_end = t_end
        self.n_data = 2 * self.n_grid
        self.random_times = random_times
        self.time_grid = np.linspace(0.0, self.t_end, self.n_grid)
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ prior
    def prior_sample(self, n, rng):
        """``theta = 1 + 2 * sigmoid(raw)``, ``raw ~ N(0, 1)``."""
        raw = rng.normal(0.0, 1.0, size=(n, self.n_params))
        return 1.0 + 2.0 * sigmoid(raw)

    def log_prior(self, theta):
        theta = np.atleast_2d(theta)
        raw = np.log((theta - 1.0) / (3.0 - theta))       # inverse sigmoid
        log_det = np.log(2.0) - np.log(theta - 1.0) - np.log(3.0 - theta)
        return (np.sum(-0.5 * raw ** 2 - 0.5 * np.log(2 * np.pi), axis=-1)
                + np.sum(log_det, axis=-1))

    # -------------------------------------------------------------- simulation
    def simulate(self, theta, rng, times=None):
        times = self.time_grid if times is None else times
        traj = simulate_lotka_volterra(theta, times, noise_std=self.noise_std,
                                       rng=rng)
        return traj

    def joint_sample(self, n, rng):
        theta = self.prior_sample(n, rng)
        if self.random_times:
            prey_times = np.sort(rng.uniform(0.0, self.t_end,
                                             size=(n, self.n_grid)), axis=-1)
            predator_times = np.sort(rng.uniform(0.0, self.t_end,
                                                 size=(n, self.n_grid)), axis=-1)
            prey = np.empty((n, self.n_grid))
            predator = np.empty((n, self.n_grid))
            for i in range(n):
                times = np.union1d(prey_times[i], predator_times[i])
                if times[0] > 0.0:
                    times = np.concatenate([[0.0], times])
                trajectory = self.simulate(theta[i:i + 1], rng, times=times)
                prey[i] = np.interp(prey_times[i], times, trajectory[0, :, 0])
                predator[i] = np.interp(predator_times[i], times,
                                        trajectory[0, :, 1])
            times_prey, times_pred = prey_times, predator_times
        else:
            trajectory = self.simulate(theta, rng)
            prey = trajectory[:, :, 0]
            predator = trajectory[:, :, 1]
            times_prey = np.broadcast_to(self.time_grid, (n, self.n_grid))
            times_pred = np.broadcast_to(self.time_grid, (n, self.n_grid))

        x = np.concatenate([prey, predator], axis=-1)
        index = np.zeros((n, self.n_variables), dtype=np.float32)
        index[:, self.n_params:self.n_params + self.n_grid] = times_prey
        index[:, self.n_params + self.n_grid:] = times_pred
        return theta, x, index, {"prey_times": times_prey,
                                 "predator_times": times_pred}

    # ---------------------------------------------------------------- density
    def _noise_free(self, theta, times):
        traj = simulate_lotka_volterra(theta, times, noise_std=0.0,
                                       rng=np.random.default_rng(0))
        return traj

    def log_joint(self, theta, x, times=None, prey_slice=None,
                  predator_slice=None):
        """Exact log joint density (deterministic ODE + Gaussian noise)."""
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        times = self.time_grid if times is None else np.asarray(times)
        traj = self._noise_free(theta, times)
        n = theta.shape[0]
        if prey_slice is None:
            prey_slice = slice(0, len(times))
            predator_slice = slice(len(times), 2 * len(times))
        observed = []
        targets = []
        prey = x[:, prey_slice]
        predator = x[:, predator_slice]
        observed.append(prey)
        observed.append(predator)
        targets.append(np.nan_to_num(np.broadcast_to(
            traj[:, :, 0], prey.shape), nan=0.0))
        targets.append(np.nan_to_num(np.broadcast_to(
            traj[:, :, 1], predator.shape), nan=0.0))
        mask = [np.isfinite(prey), np.isfinite(predator)]
        log_like = np.zeros(n)
        for obs, mean, m in zip(observed, targets, mask):
            if m.size == 0:
                continue
            log_like += np.sum(
                np.where(m, -0.5 * ((obs - mean) / self.noise_std) ** 2
                         - np.log(self.noise_std * np.sqrt(2 * np.pi)), 0.0),
                axis=-1)
        return self.log_prior(theta) + log_like

    # -------------------------------------------------------------- structure
    def base_mask(self):
        return lotka_volterra_mask(self.time_grid, self.time_grid)

    def variable_kind(self):
        kinds = np.zeros(self.n_variables, dtype=np.int64)
        kinds[self.n_params:self.n_params + self.n_grid] = 1     # prey
        kinds[self.n_params + self.n_grid:] = 2                  # predator
        return kinds

    def use_fourier(self):
        mask = np.zeros(self.n_variables, dtype=bool)
        mask[self.n_params:] = True
        return mask

    def mask_factory(self):
        """Metadata dependent mask: depends on the observed time points."""
        n_params, n_grid = self.n_params, self.n_grid

        def builder(condition_state, index=None, metadata=None):
            condition_state = np.asarray(condition_state)
            if index is None:
                return np.broadcast_to(
                    self.base_mask()[None],
                    (condition_state.shape[0], self.n_variables,
                     self.n_variables)).copy()
            index = np.asarray(index, dtype=float)
            batch = condition_state.shape[0]
            masks = np.empty((batch, self.n_variables, self.n_variables),
                             dtype=bool)
            for b in range(batch):
                masks[b] = lotka_volterra_mask(
                    index[b, n_params:n_params + n_grid],
                    index[b, n_params + n_grid:])
            return masks

        return builder

    # ------------------------------------------------- unstructured conditions
    def unstructured_observation(self, theta_true: np.ndarray,
                                 n_prey: int = 4, n_predator: int = 0,
                                 rng: Optional[np.random.Generator] = None):
        """Create the unstructured observations of Fig. 5.

        Returns ``(x_obs, condition_value, condition_state)`` where only the
        selected prey / predator time points are conditioned on.
        """
        rng = rng if rng is not None else self._rng
        theta_true = np.atleast_2d(theta_true)
        trajectory = simulate_lotka_volterra(theta_true, self.time_grid,
                                             noise_std=self.noise_std, rng=rng)
        prey_idx = rng.choice(self.n_grid, size=n_prey, replace=False)
        pred_idx = (rng.choice(self.n_grid, size=n_predator, replace=False)
                    if n_predator > 0 else np.array([], dtype=int))
        x_obs = np.concatenate([trajectory[0, prey_idx, 0],
                                trajectory[0, pred_idx, 1]])
        value = np.zeros(self.n_variables)
        state = np.zeros(self.n_variables)
        value[self.n_params + prey_idx] = trajectory[0, prey_idx, 0]
        state[self.n_params + prey_idx] = 1.0
        value[self.n_params + self.n_grid + pred_idx] = \
            trajectory[0, pred_idx, 1]
        state[self.n_params + self.n_grid + pred_idx] = 1.0
        return x_obs, value, state

    def problem(self) -> Problem:
        return Problem(
            name=self.name,
            n_variables=self.n_variables,
            n_params=self.n_params,
            n_data=self.n_data,
            sample_batch=self.joint_sample,
            mask_builder=self.mask_factory(),
            variable_kind=self.variable_kind(),
            use_fourier=self.use_fourier(),
            index_dim=1,
            n_kinds=self.n_kinds(),
            log_joint=self.log_joint,
        )
