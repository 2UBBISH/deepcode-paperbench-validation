"""Truncated proposals for TSNPSE (Section 3.1, Appendix E.3.3).

In round ``r`` of TSNPSE the truncation ``bar p^r`` of the prior is defined by
the highest-probability region (HPR) of the approximate posterior learned so
far (Eq. 9):

    bar p^r(theta) ∝ p(theta) 1{ theta in HPR_eps( p_psi^{r-1}(theta | x_obs) ) }

and the proposal used to draw simulations is a mixture over rounds,
``tilde p^r(theta) = 1/r sum_{s=0}^{r-1} bar p^s(theta)`` with ``bar p^0 = p``.
Sampling from ``bar p^s`` is done by rejection sampling, using the
log-probability threshold ``kappa`` at the ``eps``-quantile of the approximate
posterior's log density (Appendix E.3.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class TruncationBoundary:
    """The HPR truncation for a single round."""

    kappa: float
    low: torch.Tensor  # empirical hypercube of posterior samples (lower corner)
    high: torch.Tensor  # ... and upper corner
    num_samples: int = 0
    acceptance_rate: float = float("nan")
    round_index: int = 0
    log_probs: Optional[torch.Tensor] = None

    def within_hypercube(self, z: torch.Tensor) -> torch.Tensor:
        return ((z >= self.low.to(z.device)) & (z <= self.high.to(z.device))).all(-1)


class Proposal:
    """Base class for objects that can sample parameters and evaluate
    ``log p(theta)`` in the standardized parameter space ``z``."""

    def sample(self, num_samples: int) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class PriorProposal(Proposal):
    """``bar p^0 = p(theta)`` in standardized coordinates."""

    def __init__(self, prior, round_index: int = 0) -> None:
        self.prior = prior  # snpse.normalization.StandardizedDistribution
        self.round_index = round_index

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.prior.sample(num_samples)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        return self.prior.log_prob(z)


class TruncatedProposal(Proposal):
    """``bar p^r``: prior truncated to the HPR of the approximate posterior."""

    def __init__(
        self,
        prior,
        boundary: TruncationBoundary,
        log_prob_fn,
        sampling_method: str = "rejection",
    ) -> None:
        self.prior = prior
        self.boundary = boundary
        self.log_prob_fn = log_prob_fn  # z -> log p_psi(z | x_obs), up to a constant
        self.sampling_method = sampling_method

    def sample(self, num_samples: int, batch_size: Optional[int] = None, max_attempts: int = 2000) -> torch.Tensor:
        """Rejection sampling from the truncated prior.

        As described in Appendix E.3.3, a cheap first rejection step checks
        whether prior samples fall inside the empirical hypercube occupied by
        the approximate posterior samples; only surviving candidates require
        the (comparatively expensive) likelihood evaluation via Eq. 5.

        Setting ``sampling_method="sir"`` instead uses sampling-importance
        resampling with weights proportional to the approximate posterior
        density, which is the alternative suggested at the end of
        Appendix E.3.3 (and used by Deistler et al. (2022b, Section 3.2)).
        """
        if self.sampling_method == "sir":
            return self._sample_sir(num_samples, batch_size)
        batch_size = batch_size or max(1024, num_samples)
        accepted: List[torch.Tensor] = []
        num_accepted = 0
        attempts = 0
        num_proposed = 0
        while num_accepted < num_samples and attempts < max_attempts:
            attempts += 1
            cand = self.prior.sample(batch_size)
            num_proposed += cand.shape[0]
            inside = self.boundary.within_hypercube(cand)
            cand = cand[inside]
            if cand.shape[0] == 0:
                continue
            lp = self.log_prob_fn(cand)
            keep = lp > self.boundary.kappa
            if keep.any():
                accepted.append(cand[keep])
                num_accepted += int(keep.sum().item())
        if num_accepted == 0:
            raise RuntimeError("truncated proposal rejected every candidate")
        out = torch.cat(accepted, dim=0)[:num_samples]
        self.boundary.acceptance_rate = out.shape[0] / max(1, num_proposed)
        return out

    def _sample_sir(self, num_samples: int, batch_size: Optional[int] = None) -> torch.Tensor:
        """Sampling-importance resampling from the truncated proposal."""
        batch_size = batch_size or max(4096, 32 * num_samples)
        cand = self.prior.sample(batch_size)
        inside = self.boundary.within_hypercube(cand)
        cand = cand[inside]
        if cand.shape[0] == 0:
            raise RuntimeError("no prior samples fell inside the empirical hypercube")
        lp = self.log_prob_fn(cand)
        keep = lp > self.boundary.kappa
        cand, lp = cand[keep], lp[keep]
        if cand.shape[0] == 0:
            raise RuntimeError("no prior samples exceeded the truncation threshold")
        w = torch.softmax(lp - self.boundary.kappa, dim=0)
        idx = torch.multinomial(w, num_samples, replacement=True)
        self.boundary.acceptance_rate = cand.shape[0] / max(1, batch_size)
        return cand[idx]

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        lp = self.prior.log_prob(z)
        inside = self.boundary.within_hypercube(z)
        lp = torch.where(inside, lp, torch.full_like(lp, float("-inf")))
        above = self.log_prob_fn(z) > self.boundary.kappa
        return torch.where(above, lp, torch.full_like(lp, float("-inf")))


class MixtureProposal(Proposal):
    """``tilde p^r = 1/r sum_{s=0}^{r-1} bar p^s`` (uniform mixture)."""

    def __init__(self, components: List[Proposal]) -> None:
        if len(components) == 0:
            raise ValueError("mixture proposal needs at least one component")
        self.components = components

    def sample(self, num_samples: int) -> torch.Tensor:
        num_components = len(self.components)
        idx = torch.randint(0, num_components, (num_samples,))
        out = []
        for c, comp in enumerate(self.components):
            n = int((idx == c).sum().item())
            if n > 0:
                out.append(comp.sample(n))
        if not out:
            return self.components[0].sample(num_samples)
        return torch.cat(out, dim=0)

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        lps = torch.stack([c.log_prob(z) for c in self.components], dim=0)
        return torch.logsumexp(lps, dim=0) - torch.log(torch.tensor(float(len(self.components))))


@torch.no_grad()
def estimate_truncation_boundary(
    posterior,
    x_obs: torch.Tensor,
    num_samples: int = 20000,
    eps: float = 5e-4,
    round_index: int = 0,
    seed: Optional[int] = None,
    store_log_probs: bool = False,
) -> TruncationBoundary:
    """Estimate ``HPR_eps`` of the approximate posterior (Appendix E.3.3).

    Simulates ``num_samples`` samples from the approximate posterior via the
    probability flow ODE, evaluates their log density via the instantaneous
    change-of-variables formula, and returns the ``eps``-quantile of those log
    densities as the truncation boundary ``kappa`` (plus the empirical
    hypercube of the samples, used as a cheap pre-filter).
    """
    z, log_prob = posterior.sample_and_log_prob(x_obs, num_samples, seed=seed)
    kappa = float(torch.quantile(log_prob, torch.tensor(eps, dtype=log_prob.dtype)).item())
    return TruncationBoundary(
        kappa=kappa,
        low=z.min(dim=0).values.detach().clone(),
        high=z.max(dim=0).values.detach().clone(),
        num_samples=num_samples,
        round_index=round_index,
        log_probs=log_prob.detach().clone() if store_log_probs else None,
    )
