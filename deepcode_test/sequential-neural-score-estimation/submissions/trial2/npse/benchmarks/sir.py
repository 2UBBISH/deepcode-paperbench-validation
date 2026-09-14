"""SIR benchmark simulator.

The task is the stochastic Susceptible-Infected-Recovered (SIR) model used in
the SBI benchmark literature and in the NPSE paper.  Parameters are

    theta = (beta, gamma)
    beta  ~ LogNormal(log(0.4), 0.5)
    gamma ~ LogNormal(log(0.8), 0.2)

and the observation consists of 10 noisy Binomial recordings of the number of
infected individuals in a population of size ``N`` at equally spaced times.
The implementation prefers the official ``sbibm`` reference simulator when it
is installed, and otherwise falls back to a self-contained ``scipy`` ODE
simulator with the same generative structure.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.distributions import LogNormal

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["SIR"]


class SIR(Benchmark):
    """SIR simulation-based inference benchmark."""

    name = "sir"
    theta_dim = 2
    x_dim = 10

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        use_sbibm: bool = True,
        population_size: int = 1_000_000,
        n_timepoints: int = 10,
        time_horizon: float = 160.0,
    ) -> None:
        super().__init__(device=device, dtype=dtype)

        self.use_sbibm = use_sbibm
        self.population_size = int(population_size)
        self.n_timepoints = int(n_timepoints)
        self.time_horizon = float(time_horizon)

        self._sbibm_task = None
        self._sbibm_simulator = None
        self._sbibm_prior = None

        if self.use_sbibm:
            try:
                from sbibm.tasks import get_task  # type: ignore

                self._sbibm_task = get_task("sir")
                self._sbibm_simulator = self._sbibm_task.get_simulator()
                self._sbibm_prior = self._sbibm_task.get_prior()
            except Exception:
                self._sbibm_task = None
                self._sbibm_simulator = None
                self._sbibm_prior = None

        self._beta_dist = LogNormal(
            torch.tensor(math.log(0.4), dtype=torch.float64),
            torch.tensor(0.5, dtype=torch.float64),
        )
        self._gamma_dist = LogNormal(
            torch.tensor(math.log(0.8), dtype=torch.float64),
            torch.tensor(0.2, dtype=torch.float64),
        )

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample ``beta`` and ``gamma`` from the lognormal prior."""
        if self._sbibm_prior is not None:
            try:
                samples = self._sbibm_prior.sample((int(n),))
                samples = to_torch(samples, device=self.device, dtype=self.dtype)
                if samples.dim() == 1:
                    samples = samples.unsqueeze(-1)
                return samples
            except Exception:
                pass

        beta = self._beta_dist.sample((int(n),))
        gamma = self._gamma_dist.sample((int(n),))
        theta = torch.stack([beta, gamma], dim=-1)
        return to_torch(theta, device=self.device, dtype=self.dtype)

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the lognormal prior log density."""
        theta = to_torch(theta, device=torch.device("cpu"), dtype=torch.float64)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        beta = theta[..., 0]
        gamma = theta[..., 1]
        logp = self._beta_dist.log_prob(beta) + self._gamma_dist.log_prob(gamma)
        logp = to_torch(logp, device=self.device, dtype=self.dtype)
        if logp.numel() == 1:
            return logp.reshape(1)
        return logp

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate ``x`` given parameters ``theta``.

        The observation is a 10-dimensional vector of Binomial counts of
        infected individuals at equally spaced time points.
        """
        if self._sbibm_simulator is not None:
            try:
                theta_cpu = to_torch(
                    theta, device=torch.device("cpu"), dtype=torch.float32
                )
                x = self._sbibm_simulator(theta_cpu)
                return to_torch(x, device=self.device, dtype=self.dtype)
            except Exception:
                pass

        theta_cpu = to_torch(theta, device=torch.device("cpu"), dtype=torch.float64)
        if theta_cpu.dim() == 1:
            theta_cpu = theta_cpu.unsqueeze(0)

        xs = []
        for i in range(theta_cpu.shape[0]):
            beta = float(theta_cpu[i, 0].item())
            gamma = float(theta_cpu[i, 1].item())
            xs.append(self._simulate_single(beta, gamma))
        x = torch.tensor(
            [row for row in xs], device=self.device, dtype=self.dtype
        )
        if x.shape[0] == 1:
            return x.reshape(self.x_dim)
        return x

    def _simulate_single(self, beta: float, gamma: float) -> torch.Tensor:
        """Deterministic SIR ODE solution plus Binomial observation noise."""
        try:
            import numpy as np
            from scipy.integrate import solve_ivp  # type: ignore
        except Exception as exc:  # pragma: no cover - fallback below
            raise RuntimeError(
                "scipy is required for the fallback SIR simulator"
            ) from exc

        N = self.population_size
        I0 = 1.0
        R0 = 0.0
        S0 = float(N) - I0
        t_eval = np.linspace(
            0.0, self.time_horizon, self.n_timepoints + 1
        )[1:]

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            S, I, R = y
            dS = -beta * S * I / float(N)
            dI = beta * S * I / float(N) - gamma * I
            dR = gamma * I
            return np.asarray([dS, dI, dR], dtype=np.float64)

        sol = solve_ivp(
            rhs,
            (0.0, self.time_horizon),
            np.asarray([S0, I0, R0], dtype=np.float64),
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9,
            method="LSODA",
        )
        if not sol.success:
            raise RuntimeError(f"SIR ODE integration failed: {sol.message}")

        infected_fraction = np.clip(sol.y[1] / float(N), 0.0, 1.0)
        counts = np.random.binomial(N, infected_fraction).astype(np.float64)
        return torch.tensor(counts, dtype=torch.float64)

    # ------------------------------------------------------------------
    # Reference posterior and observations
    # ------------------------------------------------------------------
    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Return one or more stored or prior-predictive observations."""
        if self._sbibm_task is not None:
            try:
                obs = self._sbibm_task.get_observation(num_observation=int(n))
                return to_torch(obs, device=self.device, dtype=self.dtype)
            except Exception:
                pass

        theta = self.prior_sample(int(n))
        return self.simulator(theta)

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> Optional[torch.Tensor]:
        """Return sbibm reference posterior samples when available.

        If ``x_obs`` matches one of the first ten stored sbibm observations,
        the corresponding reference posterior is returned.  Otherwise the
        first stored reference posterior is used as a best-effort fallback.
        """
        if self._sbibm_task is None:
            return None

        try:
            obs_index = self._match_sbibm_observation(x_obs)
            ref = self._sbibm_task.get_reference_posterior_samples(
                num_observation=obs_index
            )
            ref = to_torch(ref, device=self.device, dtype=self.dtype)
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)
            if ref.shape[0] < int(n):
                return ref
            return ref[: int(n)]
        except Exception:
            return None

    def _match_sbibm_observation(self, x_obs: torch.Tensor) -> int:
        """Find the sbibm observation index matching ``x_obs``."""
        try:
            stored = self._sbibm_task.get_observation(num_observation=10)
            x = to_torch(x_obs, device=torch.device("cpu"), dtype=stored.dtype)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            for idx in range(stored.shape[0]):
                if torch.allclose(
                    stored[idx].reshape(1, -1), x.reshape(1, -1), atol=1e-4, rtol=1e-4
                ):
                    return idx + 1  # sbibm observation indices are 1-based
        except Exception:
            pass
        return 1
