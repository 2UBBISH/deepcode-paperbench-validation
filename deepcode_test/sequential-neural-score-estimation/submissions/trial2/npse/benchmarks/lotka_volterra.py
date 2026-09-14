"""Lotka-Volterra simulation-based inference benchmark.

The Lotka-Volterra predator-prey model is a classic four-parameter ODE
benchmark used in simulation-based inference.  Parameters are

    theta = (alpha, beta, gamma, delta)

with independent LogNormal priors

    log alpha ~ Normal(-0.125, 0.5)
    log beta  ~ Normal(-3.000, 0.5)
    log gamma ~ Normal(-0.125, 0.5)
    log delta ~ Normal(-3.000, 0.5)

The observation consists of 10 recordings of predator and prey populations
(20 summary statistics in total).  The implementation prefers the official
``sbibm`` task when available and falls back to a self-contained
``scipy.integrate.solve_ivp`` simulator with log-normal observation noise.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["LotkaVolterra"]


class LotkaVolterra(Benchmark):
    """Lotka-Volterra predator-prey simulation-based inference benchmark.

    Attributes
    ----------
    name : str
        Registry key ``"lotka_volterra"``.
    theta_dim : int
        Four parameters ``(alpha, beta, gamma, delta)``.
    x_dim : int
        Twenty summary statistics (10 log-prey and 10 log-predator recordings).
    """

    name = "lotka_volterra"
    theta_dim = 4
    x_dim = 20

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        use_sbibm: bool = True,
        n_timepoints: int = 10,
        time_horizon: float = 20.0,
        noise_std: float = 0.1,
        initial_prey: float = 30.0,
        initial_predator: float = 1.0,
    ) -> None:
        """Instantiate the benchmark.

        Parameters
        ----------
        device, dtype:
            Torch device and floating point type.
        use_sbibm:
            If ``True``, prefer the official ``sbibm`` task for the prior,
            simulator, stored observations, and reference posterior samples.
        n_timepoints:
            Number of equally spaced ODE evaluation times.
        time_horizon:
            Final ODE integration time.
        noise_std:
            Standard deviation of the log-normal observation noise used by the
            self-contained fallback simulator.
        initial_prey, initial_predator:
            Deterministic ODE initial conditions for the fallback simulator.
        """
        super().__init__(device=device, dtype=dtype)
        self.use_sbibm = use_sbibm
        self.n_timepoints = int(n_timepoints)
        self.time_horizon = float(time_horizon)
        self.noise_std = float(noise_std)
        self.initial_prey = float(initial_prey)
        self.initial_predator = float(initial_predator)

        # LogNormal prior parameters, as specified in the paper.
        lognormal_loc = torch.tensor(
            [-0.125, -3.0, -0.125, -3.0], dtype=self.dtype, device=self.device
        )
        self.register_buffer("_lognormal_loc", lognormal_loc)
        self._lognormal_scale = 0.5

        self._sbibm_task = None
        self._sbibm_prior = None
        self._sbibm_simulator = None
        if use_sbibm:
            try:
                import sbibm  # type: ignore

                self._sbibm_task = sbibm.get_task("lotka_volterra")
                self._sbibm_prior = self._sbibm_task.get_prior()
                self._sbibm_simulator = self._sbibm_task.get_simulator()
            except Exception:
                # Silently fall back to the self-contained implementation.
                self._sbibm_task = None
                self._sbibm_prior = None
                self._sbibm_simulator = None

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------
    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample ``theta ~ LogNormal(loc, 0.5)`` independently per dimension."""
        if self._sbibm_prior is not None:
            try:
                samples = self._sbibm_prior(num_samples=int(n))
                return to_torch(samples, device=self.device, dtype=self.dtype)
            except Exception:
                pass

        z = torch.randn(
            int(n), self.theta_dim, device=self.device, dtype=self.dtype
        )
        return torch.exp(self._lognormal_loc + self._lognormal_scale * z)

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the independent LogNormal prior log density."""
        theta_t = to_torch(theta, device=self.device, dtype=self.dtype)
        if self._sbibm_prior is not None:
            try:
                logp = self._sbibm_prior.log_prob(theta_t)
                if torch.is_tensor(logp):
                    return to_torch(logp, device=self.device, dtype=self.dtype)
            except Exception:
                pass

        if torch.any(theta_t <= 0.0):
            # Scalar -inf broadcast to the batch dimension, with valid entries
            # kept where possible.
            log_theta = torch.where(
                theta_t > 0.0, torch.log(theta_t), torch.zeros_like(theta_t)
            )
            z = (log_theta - self._lognormal_loc) / self._lognormal_scale
            logp = (
                -0.5 * z.square()
                - log_theta
                - math.log(self._lognormal_scale)
                - 0.5 * math.log(2.0 * math.pi)
            ).sum(dim=-1)
            logp = torch.where((theta_t > 0.0).all(dim=-1), logp, torch.full_like(logp, -float("inf")))
            return logp

        log_theta = torch.log(theta_t)
        z = (log_theta - self._lognormal_loc) / self._lognormal_scale
        logp = (
            -0.5 * z.square()
            - log_theta
            - math.log(self._lognormal_scale)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return logp.sum(dim=-1)

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate the 20-dimensional summary statistic for batched ``theta``."""
        theta_t = to_torch(theta, device=self.device, dtype=self.dtype)
        if theta_t.dim() == 1:
            theta_t = theta_t.unsqueeze(0)

        if self._sbibm_simulator is not None:
            try:
                x = self._sbibm_simulator(theta_t)
                return to_torch(x, device=self.device, dtype=self.dtype)
            except Exception:
                pass

        theta_np = theta_t.detach().cpu().numpy().astype(np.float64)
        batch = theta_np.shape[0]

        prey_logs: list[np.ndarray] = []
        predator_logs: list[np.ndarray] = []
        for i in range(batch):
            prey, predator = self._solve_lv(theta_np[i])
            prey_logs.append(np.log(np.maximum(prey, 1e-6)))
            predator_logs.append(np.log(np.maximum(predator, 1e-6)))

        # Shape (batch, 2, n_timepoints): [log prey, log predator].
        log_populations = np.stack(
            [np.stack(prey_logs, axis=0), np.stack(predator_logs, axis=0)],
            axis=1,
        )
        log_populations_t = to_torch(
            log_populations, device=self.device, dtype=self.dtype
        )
        noise = self.noise_std * torch.randn_like(log_populations_t)
        x = (log_populations_t + noise).reshape(batch, -1)
        return x

    # ------------------------------------------------------------------
    # Observations and reference posterior
    # ------------------------------------------------------------------
    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Return stored sbibm observations when possible, otherwise simulate."""
        if self._sbibm_task is not None:
            obs = self._sbibm_observation(1)
            if obs is not None:
                if int(n) == 1:
                    return obs
                return obs.repeat(int(n), 1)

        theta = self.prior_sample(int(n))
        return self.simulator(theta)

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> Optional[torch.Tensor]:
        """Return sbibm reference posterior samples if available."""
        if self._sbibm_task is None:
            return None

        idx = self._match_sbibm_observation(x_obs)
        try:
            samples = self._sbibm_task.get_reference_posterior_samples(
                num_observation=idx, num_samples=int(n)
            )
        except Exception:
            try:
                samples = self._sbibm_task.get_reference_posterior_samples(
                    idx, int(n)
                )
            except Exception:
                try:
                    samples = self._sbibm_task._sample_reference_posterior(
                        num_samples=int(n), observation=to_torch(
                            x_obs, device=self.device, dtype=self.dtype
                        )
                    )
                except Exception:
                    return None

        return to_torch(samples, device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sbibm_observation(self, idx: int) -> Optional[torch.Tensor]:
        """Retrieve a stored sbibm observation by one-based index."""
        if self._sbibm_task is None:
            return None
        try:
            obs = self._sbibm_task.get_observation(num_observation=int(idx))
        except Exception:
            try:
                obs = self._sbibm_task.get_observation(int(idx))
            except Exception:
                return None
        obs_t = to_torch(obs, device=self.device, dtype=self.dtype)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    def _match_sbibm_observation(self, x_obs: torch.Tensor) -> int:
        """Find the sbibm observation index matching ``x_obs`` (default 1)."""
        x_t = to_torch(x_obs, device=self.device, dtype=self.dtype)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)
        for idx in range(1, 11):
            obs = self._sbibm_observation(idx)
            if obs is not None and torch.allclose(
                x_t, obs, atol=1e-5, rtol=1e-5
            ):
                return idx
        return 1

    def _solve_lv(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Solve the deterministic Lotka-Volterra ODE for a single parameter.

        Returns arrays of prey and predator populations at ``n_timepoints``
        equally spaced times in ``[0, time_horizon]``.
        """
        try:
            from scipy.integrate import solve_ivp
        except Exception as exc:  # pragma: no cover - scipy is a core dependency
            raise ImportError(
                "scipy is required for the self-contained Lotka-Volterra "
                "simulator when sbibm is unavailable."
            ) from exc

        alpha, beta, gamma, delta = (float(v) for v in theta)

        def rhs(t: float, y: np.ndarray) -> list[float]:
            prey, predator = float(y[0]), float(y[1])
            dprey = alpha * prey - beta * prey * predator
            dpredator = -gamma * predator + delta * prey * predator
            return [dprey, dpredator]

        t_eval = np.linspace(0.0, self.time_horizon, self.n_timepoints)
        sol = solve_ivp(
            rhs,
            (0.0, self.time_horizon),
            [self.initial_prey, self.initial_predator],
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-5,
            atol=1e-5,
        )
        return np.asarray(sol.y[0]), np.asarray(sol.y[1])
