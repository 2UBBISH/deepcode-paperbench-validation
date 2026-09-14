"""Simple Likelihood Complex Posterior (SLCP) benchmark.

The SLCP task (Papamakarios et al., 2019; Lueckmann et al., 2021) has a
five-dimensional uniform prior and an eight-dimensional observation formed by
four independent draws from the same bivariate Gaussian.  The Gaussian mean and
covariance are nonlinear functions of the parameters, which produces a
posterior with four symmetric modes.

Parameters
----------
theta = (theta_1, ..., theta_5),  theta ~ Uniform(-3, 3)^5

Observation
-----------
For each of four independent copies::

    mean  = [theta_1, theta_2]
    cov   = [[theta_3^2,           theta_3 * theta_4 * tanh(theta_5)],
             [theta_3 * theta_4 * tanh(theta_5), theta_4^2          ]]

The eight-dimensional observation is the concatenation of the four
two-dimensional Gaussian draws.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["SLCP"]


class SLCP(Benchmark):
    """SLCP simulation-based inference benchmark.

    If ``sbibm`` is available, reference posterior samples for its stored test
    observations are exposed through :meth:`reference_posterior_samples` and
    :meth:`sample_observation` prefers the stored first observation for
    reproducible evaluation.
    """

    name = "slcp"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        low: float = -3.0,
        high: float = 3.0,
        n_draws: int = 4,
        use_sbibm: bool = True,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.low = float(low)
        self.high = float(high)
        self.n_draws = int(n_draws)
        self.theta_dim = 5
        self.x_dim = 2 * self.n_draws

        # Keep an optional handle on the sbibm task for reference posterior
        # samples and stored observations.
        self._sbibm_task = None
        self._sbibm_simulator = None
        if use_sbibm:
            try:
                import sbibm  # type: ignore

                self._sbibm_task = sbibm.get_task("slcp")
                self._sbibm_simulator = self._sbibm_task.get_simulator()
            except Exception:
                self._sbibm_task = None
                self._sbibm_simulator = None

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample ``theta ~ Uniform(low, high)^5``."""
        shape = (int(n), self.theta_dim)
        return (
            torch.rand(shape, device=self.device, dtype=self.dtype)
            * (self.high - self.low)
            + self.low
        )

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the (constant) log density inside the prior hypercube."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        flat = theta.reshape(-1, self.theta_dim)
        inside = ((flat >= self.low) & (flat <= self.high)).all(dim=-1)
        log_prob = torch.full(
            (flat.shape[0],),
            -float(self.theta_dim) * math.log(self.high - self.low),
            device=self.device,
            dtype=self.dtype,
        )
        log_prob[~inside] = -math.inf
        return log_prob.reshape(theta.shape[:-1])

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate four independent bivariate Gaussian observations."""
        theta = to_torch(theta, device=self.device, dtype=self.dtype)
        was_1d = theta.dim() == 1
        if was_1d:
            theta = theta.unsqueeze(0)

        batch = theta.shape[0]
        mean = theta[:, :2]  # (batch, 2)
        std1 = theta[:, 2].abs()  # |theta_3|
        std2 = theta[:, 3].abs()  # |theta_4|
        rho = torch.tanh(theta[:, 4])  # tanh(theta_5)

        # Lower-triangular Cholesky factor of the covariance matrix.  This is
        # numerically more robust than constructing the covariance matrix
        # directly because theta_3/theta_4 may be arbitrarily close to zero.
        sqrt_one_minus_rho2 = torch.sqrt(
            torch.clamp(1.0 - rho * rho, min=1e-12)
        )
        scale_tril = torch.zeros(batch, 2, 2, device=self.device, dtype=self.dtype)
        scale_tril[:, 0, 0] = std1
        scale_tril[:, 1, 0] = rho * std2
        scale_tril[:, 1, 1] = std2 * sqrt_one_minus_rho2

        dist = torch.distributions.MultivariateNormal(
            loc=mean, scale_tril=scale_tril
        )

        samples = [dist.sample() for _ in range(self.n_draws)]
        x = torch.cat(samples, dim=-1)  # (batch, 2 * n_draws)
        if was_1d:
            x = x.squeeze(0)
        return x

    # ------------------------------------------------------------------
    # Observations and references
    # ------------------------------------------------------------------
    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Return prior-predictive observation(s).

        When ``sbibm`` is available, the first stored test observation is used
        for ``n == 1`` so that the returned observation is paired with the
        reference posterior samples.
        """
        if self._sbibm_task is not None:
            try:
                observations = []
                for i in range(1, int(n) + 1):
                    obs = self._sbibm_task.get_observation(num_observation=i)
                    observations.append(
                        to_torch(obs, device=self.device, dtype=self.dtype)
                    )
                if observations:
                    return torch.stack(observations, dim=0)
            except Exception:
                pass

        theta = self.prior_sample(int(n))
        return self.simulator(theta)

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> Optional[torch.Tensor]:
        """Return sbibm reference posterior samples if available.

        The supplied observation is matched against sbibm's stored test
        observations.  If no match is found, reference samples for the first
        stored observation are returned, which is the observation returned by
        :meth:`sample_observation` for ``n == 1``.
        """
        if self._sbibm_task is None:
            return None

        try:
            num_observation = 1
            if x_obs is not None:
                x_obs = to_torch(x_obs, device=self.device, dtype=self.dtype)
                if x_obs.dim() > 1:
                    x_obs = x_obs[0]
                for i in range(1, 11):
                    candidate = self._sbibm_task.get_observation(
                        num_observation=i
                    )
                    candidate = to_torch(
                        candidate, device=self.device, dtype=self.dtype
                    )
                    if torch.allclose(x_obs, candidate, atol=1e-5, rtol=1e-5):
                        num_observation = i
                        break

            ref = self._sbibm_task.get_reference_posterior_samples(
                num_observation=num_observation
            )
            ref = to_torch(ref, device=self.device, dtype=self.dtype)
            if ref.dim() != 2 or ref.shape[0] < 1:
                return None
            if int(n) is None or int(n) >= ref.shape[0]:
                return ref
            idx = torch.randperm(ref.shape[0], device=self.device)[: int(n)]
            return ref[idx]
        except Exception:
            return None
