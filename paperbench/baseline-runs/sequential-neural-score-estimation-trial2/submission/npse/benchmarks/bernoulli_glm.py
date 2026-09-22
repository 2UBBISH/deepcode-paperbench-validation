"""Bernoulli GLM simulation-based inference benchmark.

This task follows the NPSE paper's benchmark suite and the standard
simulation-based-inference Bernoulli GLM task (Lueckmann et al., 2021 /
``sbibm``). The parameter vector is

    theta = (beta, f) in R^10,

with

    beta ~ N(0, 2)
    f    ~ N(0, (F^T F)^{-1})

where ``F`` is a discrete second-order-difference matrix on ``f``. The
observation ``x`` is a 10-dimensional binary vector of Bernoulli responses.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.distributions import Bernoulli, MultivariateNormal, Normal

from npse.benchmarks.base import Benchmark, to_torch

__all__ = ["BernoulliGLM"]


class BernoulliGLM(Benchmark):
    """Bernoulli GLM benchmark simulator.

    Parameters
    ----------
    device, dtype:
        Tensor device and dtype used for manual sampling.
    dim:
        Parameter/data dimensionality. Must be at least 2 (one intercept plus
        ``dim - 1`` coefficients).
    beta_scale:
        Standard deviation of the intercept prior, ``beta ~ N(0, beta_scale)``.
    design_seed:
        Seed used for the fixed covariate/design matrix in the manual simulator.
    use_sbibm:
        If ``True``, attempt to use the official ``sbibm`` task simulator and
        reference posterior when the ``sbibm`` package is installed. Otherwise
        use the self-contained implementation.
    """

    name = "bernoulli_glm"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        dim: int = 10,
        beta_scale: float = 2.0,
        design_seed: int = 0,
        use_sbibm: bool = True,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.theta_dim = int(dim)
        self.x_dim = int(dim)
        self.beta_scale = float(beta_scale)
        self.design_seed = int(design_seed)
        self.use_sbibm = bool(use_sbibm)

        self._device = self.device
        self._dtype = self.dtype

        # Optional sbibm integration. If available, the official task simulator
        # and reference posterior are preferred for exact benchmark alignment.
        self._sbibm_task = None
        if self.use_sbibm:
            try:
                from sbibm.tasks import get_task

                self._sbibm_task = get_task("bernoulli_glm")
            except Exception:
                self._sbibm_task = None

        self._setup_manual()

    # ------------------------------------------------------------------
    # Manual (self-contained) prior and simulator
    # ------------------------------------------------------------------
    def _setup_manual(self) -> None:
        """Create distributions and the fixed design matrix for the fallback."""
        m = self.theta_dim - 1
        if m < 1:
            raise ValueError("BernoulliGLM requires dim >= 2.")

        # Second-order-difference factor F, chosen so that F^T F is a
        # full-rank discrete-curvature precision matrix.
        F = torch.zeros((m, m), dtype=self._dtype, device=self._device)
        F[0, 0] = 1.0
        for i in range(1, m):
            F[i, i] = 2.0
            F[i, i - 1] = -1.0
        precision = F.t() @ F

        zero = torch.tensor(0.0, dtype=self._dtype, device=self._device)
        self._beta_dist = Normal(zero, torch.tensor(self.beta_scale, dtype=self._dtype, device=self._device))
        self._f_loc = torch.zeros(m, dtype=self._dtype, device=self._device)
        self._f_dist = MultivariateNormal(loc=self._f_loc, precision_matrix=precision)

        # Fixed covariate matrix for the manual simulator. A fixed design
        # matrix makes the ten binary observations sufficient for theta.
        gen = torch.Generator().manual_seed(self.design_seed)
        self._design_matrix = torch.randn(
            (self.x_dim, m), generator=gen, dtype=self._dtype
        ).to(self._device)

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------
    def prior_sample(self, n: int) -> torch.Tensor:
        """Sample ``theta = (beta, f)`` from the prior."""
        n = int(n)
        if n <= 0:
            return torch.empty((0, self.theta_dim), dtype=self._dtype, device=self._device)
        beta = self._beta_dist.sample((n, 1))
        f = self._f_dist.sample((n,))
        return torch.cat([beta, f], dim=1).to(dtype=self._dtype, device=self._device)

    def prior_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Evaluate the prior log density for ``theta``."""
        theta = to_torch(theta, device=self._device, dtype=self._dtype)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        beta_lp = self._beta_dist.log_prob(theta[:, 0])
        f_lp = self._f_dist.log_prob(theta[:, 1:])
        return (beta_lp + f_lp).to(dtype=self._dtype, device=self._device)

    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate ten Bernoulli observations given ``theta``."""
        theta = to_torch(theta, device=self._device, dtype=self._dtype)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)

        if self._sbibm_task is not None:
            try:
                x = self._sbibm_task.get_simulator()(theta)
                if isinstance(x, torch.Tensor):
                    return x.to(dtype=self._dtype, device=self._device)
                return to_torch(x, device=self._device, dtype=self._dtype)
            except Exception:
                # Fall through to the manual simulator.
                pass

        beta = theta[:, 0:1]
        f = theta[:, 1:]
        # (batch, m) @ (m, x_dim) -> (batch, x_dim)
        logits = beta + (f @ self._design_matrix.t())
        probs = torch.sigmoid(logits)
        return torch.bernoulli(probs).to(dtype=self._dtype, device=self._device)

    def sample_observation(self, n: int = 1) -> torch.Tensor:
        """Draw prior-predictive observations."""
        theta = self.prior_sample(n)
        return self.simulator(theta)

    def reference_posterior_samples(
        self, x_obs: torch.Tensor, n: int
    ) -> Optional[torch.Tensor]:
        """Return sbibm reference posterior samples when available.

        The manual implementation has no analytic reference posterior, so
        ``None`` is returned if ``sbibm`` is unavailable or the call fails.
        """
        if self._sbibm_task is None:
            return None

        try:
            posterior = self._sbibm_task.get_reference_posterior()
        except Exception:
            return None

        obs = to_torch(x_obs, device="cpu", dtype=torch.float32)
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)

        all_samples = []
        for i in range(obs.shape[0]):
            obs_np = obs[i].detach().cpu().numpy()
            samples = None
            # sbibm reference-posterior sampling APIs have varied slightly
            # between versions; try the common call signatures.
            for call in (
                lambda: posterior.sample(num_samples=int(n), observation=obs_np),
                lambda: posterior.sample((int(n),), observation=obs_np),
                lambda: posterior.sample(int(n), obs_np),
                lambda: posterior.sample(num_samples=int(n), x=obs_np),
                lambda: posterior.sample((int(n),), x=obs_np),
            ):
                try:
                    samples = call()
                    break
                except Exception:
                    continue
            if samples is None:
                return None
            if isinstance(samples, torch.Tensor):
                samples = samples.detach().cpu()
            else:
                samples = to_torch(samples, device="cpu", dtype=torch.float32)
            all_samples.append(samples)

        result = torch.stack(all_samples, dim=0)
        if single:
            result = result[0]
        return result.to(dtype=self._dtype, device=self._device)
