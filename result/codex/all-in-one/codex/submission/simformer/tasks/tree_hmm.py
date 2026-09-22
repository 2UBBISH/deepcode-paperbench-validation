"""The "Tree" and "HMM" tasks introduced in Sec. 4.1 (Appendix A2.2).

Both tasks have particularly interesting dependency structures (a tree shaped
generative model and a Markov chain in the parameters) which makes them a good
test bed for arbitrary conditionals of the joint distribution.
"""

from __future__ import annotations

import numpy as np

from ..masks import hmm_mask, tree_mask
from .base import Task


class TreeTask(Task):
    """Nonlinear tree shaped task.

    ``theta_0 ~ N(0, 1)``, ``theta_1 ~ N(theta_0, 1)``, ``theta_2 ~ N(theta_0, 1)``
    and

    ``x_0 ~ N(sin(theta_1)^2, 0.2^2)``, ``x_1 ~ N(0.1 theta_1^2, 0.2^2)``,
    ``x_2 ~ N(0.1 theta_2^2, 0.6^2)``, ``x_3 ~ N(cos(theta_2)^2, 0.1^2)``
    """

    name = "tree"
    n_params = 3
    n_data = 4
    # (parameter parent map, data parent map)
    param_parents = {0: [], 1: [0], 2: [0]}
    data_parents = {0: [1], 1: [1], 2: [2], 3: [2]}
    data_std = np.array([0.2, 0.2, 0.6, 0.1])

    def prior_sample(self, n, rng):
        theta = np.empty((n, 3))
        theta[:, 0] = rng.normal(0.0, 1.0, size=n)
        theta[:, 1] = theta[:, 0] + rng.normal(0.0, 1.0, size=n)
        theta[:, 2] = theta[:, 0] + rng.normal(0.0, 1.0, size=n)
        return theta

    @staticmethod
    def _data_mean(theta):
        t1 = theta[..., 1]
        t2 = theta[..., 2]
        return np.stack([
            np.sin(t1) ** 2,
            0.1 * t1 ** 2,
            0.1 * t2 ** 2,
            np.cos(t2) ** 2,
        ], axis=-1)

    def simulate(self, theta, rng):
        mean = self._data_mean(np.atleast_2d(theta))
        return mean + rng.normal(size=mean.shape) * self.data_std

    def base_mask(self):
        return tree_mask(self.n_params, self.n_data, self.param_parents,
                         self.data_parents)

    def log_joint(self, theta, x):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        log_p = (-0.5 * theta[:, 0] ** 2
                 - 0.5 * (theta[:, 1] - theta[:, 0]) ** 2
                 - 0.5 * (theta[:, 2] - theta[:, 0]) ** 2
                 - 1.5 * np.log(2 * np.pi))
        mean = self._data_mean(theta)
        std = self.data_std[None, :]
        log_p += np.sum(-0.5 * ((x - mean) / std) ** 2 - np.log(std)
                        - 0.5 * np.log(2 * np.pi), axis=-1)
        return log_p


class HMMTask(Task):
    """Nonlinear hidden Markov model.

    ``theta_0 ~ N(0, 0.5^2)``, ``theta_{i+1} ~ N(theta_i, 0.5^2)`` and
    ``x_i ~ N(theta_i^2, 0.5^2)`` for ``i = 0, ..., 9``.
    """

    name = "hmm"
    n_params = 10
    n_data = 10
    theta_std = 0.5
    data_std = 0.5

    def prior_sample(self, n, rng):
        theta = np.empty((n, self.n_params))
        theta[:, 0] = rng.normal(0.0, self.theta_std, size=n)
        for i in range(1, self.n_params):
            theta[:, i] = theta[:, i - 1] + rng.normal(
                0.0, self.theta_std, size=n)
        return theta

    def simulate(self, theta, rng):
        mean = np.atleast_2d(theta) ** 2
        return mean + rng.normal(0.0, self.data_std, size=mean.shape)

    def base_mask(self):
        return hmm_mask(self.n_params, self.n_data)

    def log_joint(self, theta, x):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        diff = np.diff(theta, axis=-1)
        log_p = (-0.5 * theta[:, 0] ** 2 / self.theta_std ** 2
                 - 0.5 * np.sum(diff ** 2, axis=-1) / self.theta_std ** 2
                 - self.n_params * np.log(self.theta_std * np.sqrt(2 * np.pi)))
        log_p += np.sum(
            -0.5 * (x - theta ** 2) ** 2 / self.data_std ** 2
            - np.log(self.data_std * np.sqrt(2 * np.pi)), axis=-1)
        return log_p
