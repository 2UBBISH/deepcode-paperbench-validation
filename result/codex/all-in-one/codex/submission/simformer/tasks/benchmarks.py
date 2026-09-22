"""The four benchmark tasks of Sec. 4.1 (Appendix A2.2).

Gaussian Linear, Gaussian Mixture, Two Moons and SLCP are the tasks of the
"Simulation-Based Inference Benchmark" of Lueckmann et al. (2021).
"""

from __future__ import annotations

import numpy as np

from ..masks import dense_data_mask, gaussian_linear_mask, slcp_mask
from .base import Task


class GaussianLinearTask(Task):
    """``theta ~ N(0, 0.1 I)``, ``x | theta ~ N(theta, 0.1 I)``, 10 + 10 dims."""

    name = "gaussian_linear"
    n_params = 10
    n_data = 10
    prior_std = np.sqrt(0.1)
    noise_std = np.sqrt(0.1)

    def prior_sample(self, n, rng):
        return rng.normal(0.0, self.prior_std, size=(n, self.n_params))

    def simulate(self, theta, rng):
        return theta + rng.normal(0.0, self.noise_std, size=theta.shape)

    def base_mask(self):
        return gaussian_linear_mask(self.n_params, self.n_data)

    def prior_distribution(self):
        from torch.distributions import Independent, Normal
        import torch
        return Independent(
            Normal(torch.zeros(self.n_params),
                   torch.full((self.n_params,), float(self.prior_std))), 1)

    def log_joint(self, theta, x):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        var_t = self.prior_std ** 2
        var_x = self.noise_std ** 2
        log_p_t = -0.5 * np.sum(theta ** 2, axis=-1) / var_t
        log_p_x = -0.5 * np.sum((x - theta) ** 2, axis=-1) / var_x
        const = (-0.5 * self.n_params * np.log(2 * np.pi * var_t)
                 - 0.5 * self.n_data * np.log(2 * np.pi * var_x))
        return log_p_t + log_p_x + const


class GaussianMixtureTask(Task):
    """``theta ~ U(-10, 10)``, mixture of two Gaussians (2 + 2 dims)."""

    name = "gaussian_mixture"
    n_params = 2
    n_data = 2
    prior_low, prior_high = -10.0, 10.0
    std_wide, std_narrow = 1.0, 0.1

    def prior_sample(self, n, rng):
        return rng.uniform(self.prior_low, self.prior_high,
                           size=(n, self.n_params))

    def simulate(self, theta, rng):
        n = theta.shape[0]
        wide = rng.random(n) < 0.5
        std = np.where(wide, self.std_wide, self.std_narrow)[:, None]
        return theta + std * rng.normal(size=theta.shape)

    def base_mask(self):
        return dense_data_mask(self.n_params, self.n_data)

    def prior_distribution(self):
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(low=torch.full((self.n_params,), self.prior_low),
                          high=torch.full((self.n_params,), self.prior_high))

    def log_joint(self, theta, x):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        inside = np.all((theta >= self.prior_low) & (theta <= self.prior_high),
                        axis=-1)
        log_prior = np.where(
            inside, -self.n_params * np.log(self.prior_high - self.prior_low),
            -np.inf)
        diff = x - theta
        log_like_wide = -0.5 * np.sum(diff ** 2, axis=-1) / self.std_wide ** 2 \
            - np.log(2 * np.pi * self.std_wide ** 2)
        log_like_narrow = -0.5 * np.sum(diff ** 2, axis=-1) / self.std_narrow ** 2 \
            - np.log(2 * np.pi * self.std_narrow ** 2)
        log_like = np.logaddexp(log_like_wide, log_like_narrow) - np.log(2.0)
        return log_prior + log_like


class TwoMoonsTask(Task):
    """The two moons task (2 + 2 dims, highly multimodal posterior)."""

    name = "two_moons"
    n_params = 2
    n_data = 2
    r_mean, r_std = 0.1, 0.012

    def prior_sample(self, n, rng):
        return rng.uniform(-1.0, 1.0, size=(n, self.n_params))

    def _mean(self, theta):
        theta1 = theta[..., 0]
        theta2 = theta[..., 1]
        m1 = 0.25 - np.abs(theta1 + theta2) / np.sqrt(2.0)
        m2 = (-theta1 + theta2) / np.sqrt(2.0)
        return np.stack([m1, m2], axis=-1)

    def simulate(self, theta, rng):
        n = theta.shape[0]
        r = rng.normal(self.r_mean, self.r_std, size=n)
        alpha = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
        x = np.stack([r * np.cos(alpha), r * np.sin(alpha)], axis=-1)
        return x + self._mean(theta)

    def base_mask(self):
        return dense_data_mask(self.n_params, self.n_data)

    def prior_distribution(self):
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(low=torch.full((self.n_params,), -1.0),
                          high=torch.full((self.n_params,), 1.0))

    def log_joint(self, theta, x):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        inside = np.all((theta >= -1.0) & (theta <= 1.0), axis=-1)
        log_prior = np.where(inside, -self.n_params * np.log(2.0), -np.inf)
        mean = self._mean(theta)
        diff = x - mean
        r = np.linalg.norm(diff, axis=-1)
        # density of (r, alpha) -> x has Jacobian 1 / r
        log_p_r = (-0.5 * ((r - self.r_mean) / self.r_std) ** 2
                   - np.log(self.r_std * np.sqrt(2 * np.pi)))
        log_p_alpha = -np.log(np.pi)                      # U(-pi/2, pi/2)
        log_jac = -np.log(np.clip(r, 1e-8, None))
        return log_prior + log_p_r + log_p_alpha + log_jac


class SLPCTask(Task):
    """SLCP (2 + 2 + 2 + 2 = 8 data dims, 5 parameters)."""

    name = "slcp"
    n_params = 5
    n_data = 8
    n_observations = 4

    def prior_sample(self, n, rng):
        return rng.uniform(-3.0, 3.0, size=(n, self.n_params))

    @staticmethod
    def _moments(theta):
        mu = theta[..., :2]
        s1 = theta[..., 2] ** 2
        s2 = theta[..., 3] ** 2
        rho = np.tanh(theta[..., 4])
        return mu, s1, s2, rho

    def simulate(self, theta, rng):
        theta = np.atleast_2d(theta)
        n = theta.shape[0]
        mu, s1, s2, rho = self._moments(theta)
        x = np.empty((n, self.n_data))
        for i in range(self.n_observations):
            z = rng.normal(size=(n, 2))
            obs = np.empty((n, 2))
            obs[:, 0] = mu[:, 0] + np.sqrt(s1) * z[:, 0]
            obs[:, 1] = (mu[:, 1] + np.sqrt(s2) * (rho * z[:, 0]
                                                   + np.sqrt(1 - rho ** 2) * z[:, 1]))
            x[:, 2 * i:2 * i + 2] = obs
        return x

    def base_mask(self):
        return slcp_mask(self.n_params, self.n_data, dim_per_obs=2)

    def prior_distribution(self):
        import torch
        from sbi.utils import BoxUniform
        return BoxUniform(low=torch.full((self.n_params,), -3.0),
                          high=torch.full((self.n_params,), 3.0))

    def log_joint(self, theta, x):
        theta = np.atleast_2d(theta)
        x = np.atleast_2d(x)
        inside = np.all((theta >= -3.0) & (theta <= 3.0), axis=-1)
        log_prior = np.where(inside, -self.n_params * np.log(6.0), -np.inf)
        mu, s1, s2, rho = self._moments(theta)
        det = s1 * s2 * (1.0 - rho ** 2)
        log_det = np.log(np.clip(det, 1e-30, None))
        log_like = np.zeros(theta.shape[0])
        for i in range(self.n_observations):
            d = x[:, 2 * i:2 * i + 2] - mu
            quad = (d[:, 0] ** 2 * s2 + d[:, 1] ** 2 * s1
                    - 2 * rho * np.sqrt(s1 * s2) * d[:, 0] * d[:, 1]) / det
            log_like += -0.5 * quad - 0.5 * log_det - np.log(2 * np.pi)
        return log_prior + log_like
