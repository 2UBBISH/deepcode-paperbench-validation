"""Highest Posterior Region (HPR) truncation and truncated proposal sampling.

This module implements the sequential-building blocks used by TSNPSE.

The HPR truncation procedure from the paper works as follows:

1. Draw ``n_density_samples`` approximate posterior samples from the current
   probability-flow posterior :math:`p_\\psi(\\theta \\mid x_\\mathrm{obs})`.
2. Evaluate their approximate log densities using the instantaneous
   change-of-variables formula.
3. Estimate the :math:`\\varepsilon`-quantile :math:`\\kappa` of those log
   densities.
4. Define the (approximate) highest posterior region
   :math:`\\Theta^s = \\{\\theta : \\log p_\\psi(\\theta \\mid x_\\mathrm{obs})
   > \\kappa\\}`.
5. Sample the truncated proposal
   :math:`\\tilde p(\\theta) \\propto 1\\{\\theta \\in \\Theta^s\\} p(\\theta)`
   by rejection sampling from the prior, using a cheap hypercube
   pre-rejection based on the min/max of the approximate posterior samples
   before evaluating the expensive approximate posterior log density.

The running TSNPSE proposal is the truncated mixture

.. math::
    \\tilde p^r(\\theta) \\propto c^r(\\theta) p(\\theta),
    \\qquad
    c^r(\\theta) = \\frac{1}{r}\\sum_{s=0}^{r-1}
                    1\\{\\theta \\in \\Theta^s\\},

where :math:`\\Theta^0` is the support of the prior.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch

from .density import ProbabilityFlowDensity, log_probability_flow
from .sampling import ProbabilityFlowSampler, sample_probability_flow
from .sde import SDE

__all__ = [
    "HPR_EPSILON_DEFAULT",
    "HPR_N_SAMPLES_DEFAULT",
    "estimate_hpr_threshold",
    "HPRRegion",
    "PriorSupportRegion",
    "HPRTruncation",
    "TruncatedProposalSampler",
    "sample_truncated_proposal",
]

# Paper values (Section 4 / Appendix C).
HPR_EPSILON_DEFAULT: float = 5e-4
HPR_N_SAMPLES_DEFAULT: int = 20000

ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
PriorSampler = Callable[[int], torch.Tensor]


def _sample_flow(
    sde: SDE,
    score_fn: ScoreFn,
    n_samples: int,
    x_obs: torch.Tensor,
    theta_dim: Optional[int],
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """Sample from the probability-flow posterior.

    The wrapper is intentionally defensive about the signature of
    ``sample_probability_flow`` so that HPR remains usable even if the
    convenience function accepts slightly different keyword arguments.
    """
    try:
        return sample_probability_flow(
            sde,
            score_fn,
            n_samples,
            x_obs,
            theta_dim=theta_dim,
            device=device,
            dtype=dtype,
            rtol=rtol,
            atol=atol,
        )
    except TypeError:
        # Fall back to the minimal signature.  If theta_dim is still needed by
        # the underlying implementation this will raise a clear error.
        kwargs = {"rtol": rtol, "atol": atol}
        if theta_dim is not None:
            kwargs["theta_dim"] = theta_dim
        return sample_probability_flow(
            sde, score_fn, n_samples, x_obs, **kwargs
        )


def _log_prob_flow(
    sde: SDE,
    score_fn: ScoreFn,
    theta0: torch.Tensor,
    x_obs: torch.Tensor,
    rtol: float,
    atol: float,
    hutchinson_samples: int,
    device: Optional[torch.device],
    dtype: Optional[torch.dtype],
) -> torch.Tensor:
    """Evaluate approximate posterior log densities.

    Defensively tries the convenience function first and then falls back to
    the :class:`ProbabilityFlowDensity` class.
    """
    try:
        return log_probability_flow(
            sde,
            score_fn,
            theta0,
            x_obs,
            rtol=rtol,
            atol=atol,
            hutchinson_samples=hutchinson_samples,
            device=device,
            dtype=dtype,
        )
    except TypeError:
        try:
            return log_probability_flow(
                sde,
                score_fn,
                theta0,
                x_obs,
                rtol=rtol,
                atol=atol,
                hutchinson_samples=hutchinson_samples,
            )
        except TypeError:
            density = ProbabilityFlowDensity(
                sde,
                score_fn,
                rtol=rtol,
                atol=atol,
                hutchinson_samples=hutchinson_samples,
            )
            return density.log_prob(theta0, x_obs)


def estimate_hpr_threshold(
    sde: SDE,
    score_fn: ScoreFn,
    x_obs: torch.Tensor,
    epsilon: float = HPR_EPSILON_DEFAULT,
    n_samples: int = HPR_N_SAMPLES_DEFAULT,
    theta_dim: Optional[int] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    hutchinson_samples: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate the HPR log-density threshold for one observation.

    Returns
    -------
    threshold : torch.Tensor
        Scalar :math:`\\kappa` equal to the ``epsilon``-quantile of the
        approximate posterior log densities.
    samples : torch.Tensor
        Approximate posterior samples used for the estimate, shape
        ``(n_samples, theta_dim)``.
    log_densities : torch.Tensor
        Approximate posterior log densities, shape ``(n_samples,)``.
    """
    samples = _sample_flow(
        sde,
        score_fn,
        n_samples,
        x_obs,
        theta_dim=theta_dim,
        device=device,
        dtype=dtype,
        rtol=rtol,
        atol=atol,
    )

    log_densities = _log_prob_flow(
        sde,
        score_fn,
        samples,
        x_obs,
        rtol=rtol,
        atol=atol,
        hutchinson_samples=hutchinson_samples,
        device=device,
        dtype=dtype,
    )

    # k-th smallest value (1-indexed) is the epsilon-quantile from below.
    n = log_densities.numel()
    k = max(1, min(n, int(math.ceil(epsilon * n))))
    threshold = torch.kthvalue(log_densities, k).values
    return threshold, samples, log_densities


class HPRRegion:
    """A single (approximate) highest posterior region :math:`\\Theta^s`.

    Membership is decided by ``log p_psi(theta | x_obs) > threshold``.  A
    hypercube pre-rejection using the min/max of the posterior samples avoids
    running the ODE density evaluation on points that are already known to be
    far from the posterior support.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        x_obs: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        threshold: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        hutchinson_samples: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.sde = sde
        self.score_fn = score_fn
        self.x_obs = x_obs
        self.lower = lower
        self.upper = upper
        self.threshold = threshold
        self.rtol = rtol
        self.atol = atol
        self.hutchinson_samples = hutchinson_samples
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def contains(self, theta: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask indicating membership in the HPR region.

        Parameters
        ----------
        theta : torch.Tensor
            Candidate parameters of shape ``(batch, theta_dim)``.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(batch,)``.
        """
        lower = self.lower.to(theta.device, theta.dtype)
        upper = self.upper.to(theta.device, theta.dtype)
        in_box = ((theta >= lower) & (theta <= upper)).all(dim=-1)

        out = torch.zeros(theta.shape[0], dtype=torch.bool, device=theta.device)
        idx = in_box.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return out

        candidate = theta[idx]
        log_dens = _log_prob_flow(
            self.sde,
            self.score_fn,
            candidate,
            self.x_obs,
            rtol=self.rtol,
            atol=self.atol,
            hutchinson_samples=self.hutchinson_samples,
            device=self.device,
            dtype=self.dtype,
        )
        out[idx] = log_dens > self.threshold.to(log_dens.device, log_dens.dtype)
        return out

    def sample(
        self,
        n: int,
        prior_sampler: PriorSampler,
        batch_size: int = 512,
        max_attempts_factor: int = 1000,
    ) -> torch.Tensor:
        """Rejection-sample the truncated proposal ``1{theta in Theta} p(theta)``."""
        accepted: List[torch.Tensor] = []
        attempts = 0
        max_total_attempts = max_attempts_factor * n
        while len(accepted) < n and attempts < max_total_attempts:
            m = min(batch_size, n - len(accepted))
            candidate = prior_sampler(m)
            attempts += m
            keep = self.contains(candidate)
            if keep.any():
                accepted.append(candidate[keep])

        if len(accepted) == 0:
            raise RuntimeError(
                "HPR truncated-proposal rejection sampling failed: "
                "no candidate was accepted. This usually means the posterior "
                "samples used to build the hypercube are degenerate or the "
                "prior sampler does not cover the HPR region."
            )
        return torch.cat(accepted, dim=0)[:n]


class PriorSupportRegion:
    """Round-zero region :math:`\\Theta^0`, equal to the support of the prior.

    Since the prior sampler only produces points inside the prior support by
    construction, this region trivially contains every candidate.
    """

    @torch.no_grad()
    def contains(self, theta: torch.Tensor) -> torch.Tensor:
        return torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)


class HPRTruncation:
    """Estimate an HPR region for an observation and sample its truncated prior.

    This is the building block used in each TSNPSE round.
    """

    def __init__(
        self,
        sde: SDE,
        score_fn: ScoreFn,
        epsilon: float = HPR_EPSILON_DEFAULT,
        n_density_samples: int = HPR_N_SAMPLES_DEFAULT,
        theta_dim: Optional[int] = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        hutchinson_samples: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.sde = sde
        self.score_fn = score_fn
        self.epsilon = epsilon
        self.n_density_samples = n_density_samples
        self.theta_dim = theta_dim
        self.rtol = rtol
        self.atol = atol
        self.hutchinson_samples = hutchinson_samples
        self.device = device
        self.dtype = dtype

        self._x_obs: Optional[torch.Tensor] = None
        self._threshold: Optional[torch.Tensor] = None
        self._samples: Optional[torch.Tensor] = None
        self._log_densities: Optional[torch.Tensor] = None

    def estimate_threshold(self, x_obs: torch.Tensor) -> torch.Tensor:
        """Compute (and cache) the HPR threshold for ``x_obs``."""
        threshold, samples, log_densities = estimate_hpr_threshold(
            self.sde,
            self.score_fn,
            x_obs,
            epsilon=self.epsilon,
            n_samples=self.n_density_samples,
            theta_dim=self.theta_dim,
            rtol=self.rtol,
            atol=self.atol,
            hutchinson_samples=self.hutchinson_samples,
            device=self.device,
            dtype=self.dtype,
        )
        self._x_obs = x_obs
        self._threshold = threshold
        self._samples = samples
        self._log_densities = log_densities
        return threshold

    def build_region(self, x_obs: torch.Tensor) -> HPRRegion:
        """Estimate the threshold and return the corresponding ``HPRRegion``."""
        if self._threshold is None or self._x_obs is None:
            self.estimate_threshold(x_obs)
        assert self._samples is not None and self._threshold is not None

        lower = self._samples.min(dim=0).values
        upper = self._samples.max(dim=0).values
        return HPRRegion(
            self.sde,
            self.score_fn,
            x_obs,
            lower=lower,
            upper=upper,
            threshold=self._threshold,
            rtol=self.rtol,
            atol=self.atol,
            hutchinson_samples=self.hutchinson_samples,
            device=self.device,
            dtype=self.dtype,
        )

    def sample(
        self,
        n: int,
        x_obs: torch.Tensor,
        prior_sampler: PriorSampler,
        batch_size: int = 512,
        max_attempts_factor: int = 1000,
    ) -> torch.Tensor:
        """Sample ``n`` points from the HPR-truncated prior."""
        region = self.build_region(x_obs)
        return region.sample(
            n,
            prior_sampler,
            batch_size=batch_size,
            max_attempts_factor=max_attempts_factor,
        )


class TruncatedProposalSampler:
    """Running truncated-mixture proposal for TSNPSE rounds.

    Implements

    .. math::
        \\tilde p^r(\\theta) \\propto c^r(\\theta) p(\\theta)

    by rejection sampling from the prior with acceptance probability
    :math:`c^r(\\theta) = (1/r)\\sum_s 1\\{\\theta \\in \\Theta^s\\}`.
    """

    def __init__(self) -> None:
        self.regions: List[Union[HPRRegion, PriorSupportRegion]] = [
            PriorSupportRegion()
        ]

    def add_round(self, region: Union[HPRRegion, PriorSupportRegion]) -> None:
        """Register the HPR region estimated after a TSNPSE round."""
        self.regions.append(region)

    @property
    def num_rounds(self) -> int:
        """Number of proposal components including the prior support."""
        return len(self.regions)

    @torch.no_grad()
    def _coverage(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute the running coverage function :math:`c^r(\\theta)`."""
        count = torch.zeros(theta.shape[0], dtype=torch.float32, device=theta.device)
        for region in self.regions:
            count = count + region.contains(theta).to(count.dtype)
        return count / float(len(self.regions))

    def sample(
        self,
        n: int,
        prior_sampler: PriorSampler,
        batch_size: int = 512,
        max_attempts_factor: int = 10000,
    ) -> torch.Tensor:
        """Rejection-sample the running truncated-mixture proposal."""
        r = len(self.regions)
        if r <= 1:
            return prior_sampler(n)

        accepted: List[torch.Tensor] = []
        attempts = 0
        max_total_attempts = max_attempts_factor * n
        while len(accepted) < n and attempts < max_total_attempts:
            m = min(batch_size, n - len(accepted))
            candidate = prior_sampler(m)
            attempts += m
            coverage = self._coverage(candidate)
            keep = torch.rand(m, device=candidate.device) < coverage
            if keep.any():
                accepted.append(candidate[keep])

        if len(accepted) == 0:
            raise RuntimeError(
                "Truncated-mixture proposal sampling failed: no candidate "
                "was accepted. Increase max_attempts_factor or check the "
                "registered HPR regions."
            )
        return torch.cat(accepted, dim=0)[:n]


def sample_truncated_proposal(
    sde: SDE,
    score_fn: ScoreFn,
    x_obs: torch.Tensor,
    prior_sampler: PriorSampler,
    n: int,
    epsilon: float = HPR_EPSILON_DEFAULT,
    n_density_samples: int = HPR_N_SAMPLES_DEFAULT,
    theta_dim: Optional[int] = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    hutchinson_samples: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 512,
    max_attempts_factor: int = 1000,
) -> torch.Tensor:
    """Convenience wrapper: estimate an HPR and sample its truncated prior.

    Parameters
    ----------
    n : int
        Number of truncated-proposal samples to return.
    prior_sampler : callable
        Function ``prior_sampler(k) -> Tensor(k, theta_dim)``.

    Returns
    -------
    torch.Tensor
        Truncated-proposal samples of shape ``(n, theta_dim)``.
    """
    truncation = HPRTruncation(
        sde,
        score_fn,
        epsilon=epsilon,
        n_density_samples=n_density_samples,
        theta_dim=theta_dim,
        rtol=rtol,
        atol=atol,
        hutchinson_samples=hutchinson_samples,
        device=device,
        dtype=dtype,
    )
    return truncation.sample(
        n,
        x_obs,
        prior_sampler,
        batch_size=batch_size,
        max_attempts_factor=max_attempts_factor,
    )
