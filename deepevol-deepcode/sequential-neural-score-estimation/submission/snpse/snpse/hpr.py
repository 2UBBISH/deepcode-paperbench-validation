"""Highest-probability-region (HPR) truncation and truncated-prior proposals.

This module implements the machinery that makes TSNPSE work (paper Section 3.1,
Algorithm 1, Appendix E.3.3):

* ``HPR_eps`` / :func:`compute_hpr_region` -- compute the highest ``1 - eps``
  probability region of the current posterior approximation
  ``p_psi^{r-1}(theta | x_obs)``, summarised by the log-density threshold

      kappa = (eps)^th quantile of {log p_psi^{r-1}(theta | x_obs)}

  where the samples are obtained (as in the paper) by simulating 20000 draws
  through the (time-reversed) probability-flow ODE (eq. 4) and evaluating the
  log densities with the instantaneous change-of-variables formula (eq. 5).

* The truncated prior of round ``r`` (eq. 9)

      pbar^r(theta) ∝ p(theta) * I{ theta in HPR_eps(p_psi^{r-1}(cdot|x_obs)) }

* The proposal used in round ``r`` (eq. 10)

      ptilde^r(theta) = (1 / r) * sum_{s=0}^{r-1} pbar^s(theta),
      with pbar^0(theta) = p(theta).

* Rejection sampling from the truncated proposal, including the cheap
  empirical-hypercube pre-screen described in Appendix E.3.3 (which avoids most
  of the expensive change-of-variables evaluations).  A Sampling-Importance-
  Resampling alternative is also provided.

Everything is expressed in the *original* theta space (standardisation is
handled inside :mod:`snpse.sampler`), so the log densities returned here are the
same quantities that are compared against ``kappa``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:  # pragma: no cover - package import
    from .sampler import SamplerConfig, estimate_log_prob, sample_posterior
    from .utils import get_device, safe_log
    from .sdes import SDE
except ImportError:  # pragma: no cover - standalone import
    from sampler import SamplerConfig, estimate_log_prob, sample_posterior  # type: ignore
    from utils import get_device, safe_log  # type: ignore
    from sdes import SDE  # type: ignore


__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_N_HPR_SAMPLES",
    "HPRRegion",
    "TruncatedPrior",
    "HPR_eps",
    "highest_probability_region",
    "compute_hpr_region",
    "hpr_region_from_score",
    "build_truncated_prior",
    "sample_truncated_proposal",
    "hpr_threshold_quantile",
]


# --------------------------------------------------------------------------------------
# Defaults from the paper (Appendix E.3.3)
# --------------------------------------------------------------------------------------
DEFAULT_EPS: float = 5e-4
DEFAULT_N_HPR_SAMPLES: int = 20_000
DEFAULT_HYPERCUBE_MARGIN: float = 0.0
DEFAULT_MAX_TRIES: int = 200


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _as_2d(value: torch.Tensor) -> torch.Tensor:
    """Return a 2-D ``(n, d)`` view of ``value``."""
    if value.dim() == 0:
        return value.reshape(1, 1)
    if value.dim() == 1:
        return value.reshape(1, -1)
    if value.dim() > 2:
        return value.reshape(-1, value.shape[-1])
    return value


def _unpack_log_prob(result) -> torch.Tensor:
    """Accept a Tensor or a ``(theta, log_prob)`` tuple and return log probs."""
    if isinstance(result, (tuple, list)):
        for item in reversed(result):
            if isinstance(item, torch.Tensor):
                return item.reshape(-1)
        raise TypeError("Could not find a tensor in the result of the log-prob call.")
    return result.reshape(-1)


def _theta_dim_of(score_fn: torch.nn.Module, default: Optional[int] = None) -> Optional[int]:
    """Best-effort inference of the parameter dimension from a score network."""
    for attr in ("theta_dim", "dim", "d"):
        value = getattr(score_fn, attr, None)
        if isinstance(value, int):
            return value
    energy = getattr(score_fn, "energy_net", None)
    if energy is not None:
        for attr in ("theta_dim", "dim", "d"):
            value = getattr(energy, attr, None)
            if isinstance(value, int):
                return value
    return default


def _chunks(n_or_index: Union[int, torch.Tensor], size: int):
    """Yield index chunks covering ``range(n)`` or a given index tensor."""
    if isinstance(n_or_index, int):
        for start in range(0, n_or_index, size):
            yield torch.arange(start, min(start + size, n_or_index), dtype=torch.long)
    else:
        idx = n_or_index.reshape(-1)
        for start in range(0, idx.numel(), size):
            yield idx[start : start + size]


# --------------------------------------------------------------------------------------
# Prior adapters
# --------------------------------------------------------------------------------------
class _PriorAdapter:
    """Uniform interface over: an object with ``sample``/``log_prob``, or callables."""

    def __init__(
        self,
        prior: Optional[object] = None,
        sample_fn: Optional[Callable[..., torch.Tensor]] = None,
        log_prob_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if prior is not None:
            for name in ("sample", "rsample"):
                candidate = getattr(prior, name, None)
                if callable(candidate) and sample_fn is None:
                    sample_fn = candidate
                    break
            if callable(getattr(prior, "log_prob", None)) and log_prob_fn is None:
                log_prob_fn = prior.log_prob
        if sample_fn is None or log_prob_fn is None:
            raise ValueError(
                "A prior must be provided either as an object exposing sample/log_prob or "
                "as explicit prior_sample_fn / prior_log_prob_fn callables."
            )
        self.prior = prior
        self._sample_fn = sample_fn
        self._log_prob_fn = log_prob_fn
        self.device = device
        self.dtype = dtype

    def sample(self, num_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        num_samples = int(num_samples)
        samples = None
        for kwargs in (
            {"sample_shape": (num_samples,), "generator": generator},
            {"sample_shape": (num_samples,)},
            {"n": num_samples, "generator": generator},
            {"num_samples": num_samples, "generator": generator},
            {"size": (num_samples,), "generator": generator},
        ):
            try:
                samples = self._sample_fn(**kwargs)
                break
            except TypeError:
                continue
            except ValueError:
                continue
        if samples is None:
            # positional fallback
            try:
                samples = self._sample_fn(num_samples, generator)
            except TypeError:
                samples = self._sample_fn(num_samples)
        samples = samples if isinstance(samples, torch.Tensor) else torch.as_tensor(samples)
        samples = _as_2d(samples.detach())
        if self.dtype is not None and samples.dtype != self.dtype:
            samples = samples.to(self.dtype)
        if self.device is not None:
            samples = samples.to(self.device)
        return samples

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        theta = _as_2d(theta)
        value = self._log_prob_fn(theta)
        if isinstance(value, (tuple, list)):
            value = value[0]
        value = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        return value.reshape(-1)

    def __call__(self, num_samples: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.sample(num_samples, generator)


# --------------------------------------------------------------------------------------
# HPR threshold and regions
# --------------------------------------------------------------------------------------
def HPR_eps(values: torch.Tensor, eps: float = DEFAULT_EPS) -> float:
    """Return the ``eps``-th quantile ``kappa`` of the supplied (log-)densities.

    Following Appendix E.3.3 the truncation boundary is computed as the
    ``eps = 5e-4`` quantile of the log densities of posterior samples, i.e.
    ``kappa`` defines the highest ``1 - eps`` probability region: samples with
    ``log p(theta | x_obs) > kappa`` are kept.
    """
    flat = values.detach().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("HPR_eps received an empty tensor.")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0, 1); got {eps}.")
    work = flat.to(torch.float64)
    work = work[torch.isfinite(work)]
    if work.numel() == 0:
        raise ValueError("HPR_eps received only non-finite values.")
    return float(torch.quantile(work, float(eps)))


# Alias used in the reproduction plan.
highest_probability_region = HPR_eps
hpr_threshold_quantile = HPR_eps


@dataclass
class HPRRegion:
    """A truncated prior component ``pbar^r`` (eq. 9).

    Attributes
    ----------
    kappa:
        Log-density threshold: ``theta`` is inside the region iff
        ``log_prob_fn(theta) > kappa``.
    log_prob_fn:
        Callable evaluating ``log p_psi^{r-1}(theta | x_obs)`` (change-of-variables
        formula, eq. 5) in the original parameter space.
    hypercube_low / hypercube_high:
        Empirical hypercube occupied by the posterior samples, used as the cheap
        rejection pre-screen from Appendix E.3.3.
    log_normaliser:
        ``log Z_r`` where ``Z_r = int p(theta) I{theta in HPR} dtheta``, estimated
        by Monte Carlo over prior samples (needed by eq. 10 and SNPSE-B).
    """

    kappa: float
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor]
    hypercube_low: Optional[torch.Tensor] = None
    hypercube_high: Optional[torch.Tensor] = None
    eps: float = DEFAULT_EPS
    n_samples: int = 0
    round_index: int = 0
    log_normaliser: float = 0.0
    samples: Optional[torch.Tensor] = None
    log_probs: Optional[torch.Tensor] = None
    diagnostics: Dict[str, float] = field(default_factory=dict)

    # -- cheap pre-screen -------------------------------------------------------------
    def inside_hypercube(self, theta: torch.Tensor) -> torch.Tensor:
        """Boolean mask: which rows of ``theta`` lie in the empirical hypercube."""
        theta = _as_2d(theta)
        if self.hypercube_low is None or self.hypercube_high is None:
            return torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)
        low = self.hypercube_low.to(theta.device, theta.dtype)
        high = self.hypercube_high.to(theta.device, theta.dtype)
        return ((theta >= low) & (theta <= high)).all(dim=-1)

    # -- likelihood / membership ------------------------------------------------------
    def log_prob(self, theta: torch.Tensor, batch_size: int = 4096, screen: bool = True) -> torch.Tensor:
        """Evaluate ``log p_psi^{r-1}(theta | x_obs)`` (outside a region: ``-inf``)."""
        theta = _as_2d(theta)
        out = torch.full((theta.shape[0],), float("-inf"), device=theta.device, dtype=theta.dtype)
        mask = self.inside_hypercube(theta) if screen else torch.ones(
            theta.shape[0], dtype=torch.bool, device=theta.device
        )
        if bool(mask.any()):
            idx = mask.nonzero(as_tuple=False).reshape(-1)
            for chunk in _chunks(idx, batch_size):
                value = self.log_prob_fn(theta[chunk])
                out[chunk] = value.reshape(-1).to(out.dtype)
        return out

    def contains(self, theta: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """Boolean mask for ``theta in HPR_eps`` (with the cheap pre-screen)."""
        return self.log_prob(theta, batch_size=batch_size, screen=True) > self.kappa

    accept = contains

    # -- rejection sampling -----------------------------------------------------------
    def rejection_sample(
        self,
        prior_sample_fn: Callable[..., torch.Tensor],
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        max_hypercube_draws: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Sample ``theta ~ pbar^r`` by rejection sampling (Appendix E.3.3).

        We draw ``theta ~ p(theta)`` and accept iff
        ``log p_psi^{r-1}(theta | x_obs) > kappa``.  The expensive likelihood is
        only evaluated for candidates that already passed the cheap empirical
        hypercube pre-screen.
        """
        num_samples = int(num_samples)
        if num_samples <= 0:
            dim = int(self.hypercube_high.shape[-1]) if self.hypercube_high is not None else 1
            empty = torch.empty((0, dim), dtype=dtype or torch.float32, device=device)
            return empty, {"acceptance_rate": float("nan"), "num_proposed": 0.0}

        if batch_size is None:
            batch_size = max(2 * num_samples, 512)
        if max_batches is None:
            max_batches = DEFAULT_MAX_TRIES
        if max_hypercube_draws is None:
            # A safety valve so that a degenerate kappa cannot loop forever.
            max_hypercube_draws = 2000 * num_samples

        accepted: List[torch.Tensor] = []
        n_accepted = 0
        n_proposed = 0
        n_likelihood_evals = 0
        batches = 0

        while n_accepted < num_samples and batches < max_batches and n_proposed < max_hypercube_draws:
            batches += 1
            candidates = prior_sample_fn(batch_size, generator)
            candidates = _as_2d(candidates)
            if dtype is not None and candidates.dtype != dtype:
                candidates = candidates.to(dtype)
            if device is not None:
                candidates = candidates.to(device)
            n_proposed += int(candidates.shape[0])

            mask = self.inside_hypercube(candidates)
            candidates = candidates[mask]
            if candidates.shape[0] == 0:
                continue

            with torch.no_grad():
                log_post = self.log_prob(candidates, screen=False)
            n_likelihood_evals += int(candidates.shape[0])
            keep = log_post > self.kappa
            if bool(keep.any()):
                accepted.append(candidates[keep])
                n_accepted += int(keep.sum())

        if not accepted:
            raise RuntimeError(
                "Rejection sampling from the truncated prior accepted no samples "
                f"(kappa={self.kappa:.6g}, {n_proposed} proposals). Check that the HPR "
                "contains non-negligible prior mass."
            )

        pool = torch.cat(accepted, dim=0)
        if pool.shape[0] > num_samples:
            perm = torch.randperm(pool.shape[0], generator=generator, device=pool.device)
            pool = pool[perm[:num_samples]]
        stats = {
            "acceptance_rate": n_accepted / max(n_proposed, 1),
            "num_proposed": float(n_proposed),
            "num_accepted": float(n_accepted),
            "num_likelihood_evals": float(n_likelihood_evals),
            "num_batches": float(batches),
        }
        return pool, stats


# --------------------------------------------------------------------------------------
# Building a region from a score network
# --------------------------------------------------------------------------------------
def _call_sample_posterior(
    sde,
    score_fn,
    x_obs,
    num_samples: int,
    config,
    theta_dim,
    generator,
    device,
    dtype,
    theta_shift,
    theta_scale,
    x_shift,
    x_scale,
):
    """Call ``sample_posterior(..., with_log_prob=True)`` tolerating API variants."""
    kwargs = dict(
        config=config,
        with_log_prob=True,
        theta_dim=theta_dim,
        generator=generator,
        device=device,
        dtype=dtype,
        theta_shift=theta_shift,
        theta_scale=theta_scale,
        x_shift=x_shift,
        x_scale=x_scale,
    )
    try:
        return sample_posterior(sde, score_fn, x_obs, num_samples, **kwargs)
    except TypeError:
        return sample_posterior(
            sde, score_fn, x_obs, num_samples, config=config, with_log_prob=True, theta_dim=theta_dim
        )


def _make_log_prob_fn(
    sde,
    score_fn,
    x_obs,
    config,
    theta_dim,
    generator,
    device,
    dtype,
    theta_shift,
    theta_scale,
    x_shift,
    x_scale,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Closure evaluating ``log p_psi(theta | x_obs)`` via eq. (5)."""

    def log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
        theta = _as_2d(theta)
        kwargs = dict(
            config=config,
            theta_dim=theta_dim,
            device=device,
            dtype=dtype,
            theta_shift=theta_shift,
            theta_scale=theta_scale,
            x_shift=x_shift,
            x_scale=x_scale,
        )
        try:
            value = estimate_log_prob(sde, score_fn, x_obs, theta, **kwargs)
        except TypeError:
            value = estimate_log_prob(sde, score_fn, x_obs, theta, config=config, theta_dim=theta_dim)
        return _unpack_log_prob(value)

    return log_prob_fn


def compute_hpr_region(
    sde,
    score_fn,
    x_obs: torch.Tensor,
    num_samples: int = DEFAULT_N_HPR_SAMPLES,
    eps: float = DEFAULT_EPS,
    config=None,
    theta_dim: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    theta_shift: Optional[torch.Tensor] = None,
    theta_scale: Optional[torch.Tensor] = None,
    x_shift: Optional[torch.Tensor] = None,
    x_scale: Optional[torch.Tensor] = None,
    hypercube_margin: float = DEFAULT_HYPERCUBE_MARGIN,
    round_index: int = 0,
    keep_samples: bool = False,
    verbose: bool = False,
) -> HPRRegion:
    """Compute ``HPR_eps(p_psi(cdot | x_obs))`` for a trained score network.

    Steps (Appendix E.3.3):

    1. simulate ``num_samples`` (default 20000) posterior draws with the
       probability-flow ODE (eq. 4);
    2. compute their log densities with the instantaneous change-of-variables
       formula (eq. 5);
    3. set ``kappa`` to the ``eps = 5e-4`` quantile of those log densities;
    4. record the empirical hypercube for the cheap rejection pre-screen.
    """
    if config is None:
        config = SamplerConfig()
    if device is None:
        device = getattr(sde, "device", None)
    theta_dim = _theta_dim_of(score_fn, theta_dim)

    result = _call_sample_posterior(
        sde,
        score_fn,
        x_obs,
        int(num_samples),
        config,
        theta_dim,
        generator,
        device,
        dtype,
        theta_shift,
        theta_scale,
        x_shift,
        x_scale,
    )

    if isinstance(result, (tuple, list)) and len(result) >= 2:
        samples = result[0]
        log_probs = result[1]
    elif isinstance(result, dict):
        samples = result.get("theta", result.get("samples"))
        log_probs = result.get("log_prob")
    else:  # pragma: no cover - defensive
        samples = result
        log_probs = None

    samples = _as_2d(torch.as_tensor(samples))
    if log_probs is None:  # pragma: no cover - defensive
        log_prob_fn_fallback = _make_log_prob_fn(
            sde, score_fn, x_obs, config, theta_dim, generator, device, dtype,
            theta_shift, theta_scale, x_shift, x_scale,
        )
        with torch.no_grad():
            log_probs = log_prob_fn_fallback(samples)
    log_probs = torch.as_tensor(log_probs).reshape(-1).detach().to(samples.dtype)

    kappa = HPR_eps(log_probs, eps)

    low = samples.min(dim=0).values
    high = samples.max(dim=0).values
    if hypercube_margin and hypercube_margin > 0.0:
        half = 0.5 * (high - low)
        pad = hypercube_margin * torch.where(half > 0, half, torch.ones_like(half))
        low = low - pad
        high = high + pad

    log_prob_fn = _make_log_prob_fn(
        sde, score_fn, x_obs, config, theta_dim, generator, device, dtype,
        theta_shift, theta_scale, x_shift, x_scale,
    )

    region = HPRRegion(
        kappa=kappa,
        log_prob_fn=log_prob_fn,
        hypercube_low=low.detach(),
        hypercube_high=high.detach(),
        eps=float(eps),
        n_samples=int(num_samples),
        round_index=int(round_index),
        samples=samples.detach() if keep_samples else None,
        log_probs=log_probs.detach() if keep_samples else None,
    )
    region.diagnostics["mass_in_region"] = float((log_probs > kappa).to(torch.float32).mean())
    region.diagnostics["log_prob_mean"] = float(log_probs.mean())
    region.diagnostics["log_prob_std"] = float(log_probs.std(unbiased=False))

    if verbose:
        print(
            f"[hpr] round={round_index} n={num_samples} eps={eps:g} "
            f"kappa={kappa:.4f} retained={region.diagnostics['mass_in_region']:.4f}"
        )
    return region


# Plan-facing alias.
hpr_region_from_score = compute_hpr_region


# --------------------------------------------------------------------------------------
# Truncated proposal  ptilde^r = (1/r) sum_{s<r} pbar^s   (eq. 10)
# --------------------------------------------------------------------------------------
def build_truncated_prior(
    prior: Optional[object] = None,
    prior_sample_fn: Optional[Callable[..., torch.Tensor]] = None,
    prior_log_prob_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    regions: Optional[Sequence[HPRRegion]] = None,
    eps: float = DEFAULT_EPS,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    theta_dim: Optional[int] = None,
    estimate_normalisers: bool = True,
    n_mc: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> "TruncatedPrior":
    """Convenience constructor for :class:`TruncatedPrior`."""
    proposal = TruncatedPrior(
        prior=prior,
        prior_sample_fn=prior_sample_fn,
        prior_log_prob_fn=prior_log_prob_fn,
        eps=eps,
        device=device,
        dtype=dtype,
        theta_dim=theta_dim,
    )
    for region in regions or []:
        proposal.add_region(
            region, estimate_normaliser=estimate_normalisers, n_mc=n_mc, generator=generator
        )
    return proposal


class TruncatedPrior:
    """The TSNPSE proposal ``ptilde^r`` (eq. 10) built from HPR components.

    ``regions[s - 1]`` corresponds to ``pbar^s`` for ``s = 1, ..., r - 1``; the
    zeroth component is the untruncated prior ``pbar^0 = p(theta)``.  Therefore,
    after ``r - 1`` calls to :meth:`add_region`, :attr:`num_components` is ``r``
    and the mixture equals ``ptilde^r`` of eq. (10).
    """

    def __init__(
        self,
        prior: Optional[object] = None,
        prior_sample_fn: Optional[Callable[..., torch.Tensor]] = None,
        prior_log_prob_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        eps: float = DEFAULT_EPS,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        theta_dim: Optional[int] = None,
    ) -> None:
        self.prior = _PriorAdapter(
            prior=prior, sample_fn=prior_sample_fn, log_prob_fn=prior_log_prob_fn,
            device=device, dtype=dtype,
        )
        self.eps = float(eps)
        self.device = device
        self.dtype = dtype
        self.regions: List[HPRRegion] = []
        self._theta_dim = theta_dim
        self._sample_stats: List[Dict[str, float]] = []

    # -- introspection ----------------------------------------------------------------
    @property
    def num_components(self) -> int:
        """``r`` -- the number of mixture components in ``ptilde^r``."""
        return 1 + len(self.regions)

    @property
    def theta_dim(self) -> int:
        if self._theta_dim is not None:
            return int(self._theta_dim)
        probe = self.prior.sample(1)
        self._theta_dim = int(probe.shape[-1])
        return self._theta_dim

    @property
    def priors(self) -> List[HPRRegion]:
        """Alias matching the ``bar p^s`` notation of the paper."""
        return self.regions

    # -- construction -----------------------------------------------------------------
    def add_region(
        self,
        region: HPRRegion,
        estimate_normaliser: bool = True,
        n_mc: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        clip_log_normaliser: float = 1e-12,
    ) -> HPRRegion:
        """Append a truncated component ``pbar^s`` and estimate its normaliser ``Z_s``."""
        if region.hypercube_low is not None and self._theta_dim is None:
            self._theta_dim = int(region.hypercube_low.shape[-1])
        if estimate_normaliser:
            region.log_normaliser = self.estimate_log_normaliser(
                region, n_mc=n_mc, generator=generator, clip=clip_log_normaliser
            )
        self.regions.append(region)
        return region

    # Backwards/forwards-compatible aliases.
    add_component = add_region

    def estimate_log_normaliser(
        self,
        region: HPRRegion,
        n_mc: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        clip: float = 1e-12,
        batch_size: int = 4096,
    ) -> float:
        """Monte-Carlo estimate of ``log Z_s = log int p(theta) I{...} dtheta``.

        Because the prior integrates to one, ``Z_s`` is simply the prior
        probability mass of the HPR, estimated from prior samples (using the
        cheap hypercube pre-screen so that only a few likelihoods are required).
        """
        n_mc = int(n_mc or region.n_samples or DEFAULT_N_HPR_SAMPLES)
        n_mc = max(n_mc, 1000)
        draws = _as_2d(self.prior.sample(n_mc, generator))
        mask = region.inside_hypercube(draws)
        n_inside_hc = int(mask.sum())
        n_acc = 0
        if n_inside_hc > 0:
            candidates = draws[mask]
            with torch.no_grad():
                for chunk in _chunks(candidates, batch_size):
                    log_post = region.log_prob_fn(chunk).reshape(-1)
                    n_acc += int((log_post > region.kappa).sum())
        fraction = n_acc / n_mc
        region.diagnostics["prior_mass_hypercube"] = n_inside_hc / n_mc
        region.diagnostics["prior_mass_hpr"] = fraction
        if fraction <= 0.0:
            return math.log(clip)
        return math.log(max(fraction, clip))

    # -- densities --------------------------------------------------------------------
    def log_component(self, s: int, theta: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
        """``log pbar^s(theta)`` for the mixture index ``s`` (0 = prior)."""
        theta = _as_2d(theta)
        log_prior = self.prior.log_prob(theta)
        if s == 0:
            return log_prior
        region = self.regions[s - 1]
        out = torch.full_like(log_prior, float("-inf"))
        mask = region.inside_hypercube(theta)
        if bool(mask.any()):
            idx = mask.nonzero(as_tuple=False).reshape(-1)
            for chunk in _chunks(idx, batch_size):
                log_post = region.log_prob_fn(theta[chunk]).reshape(-1)
                keep = log_post > region.kappa
                if bool(keep.any()):
                    out[chunk[keep]] = log_prior[chunk[keep]] - region.log_normaliser
        return out

    def log_prob(
        self,
        theta: torch.Tensor,
        normalised: bool = True,
        batch_size: int = 4096,
        return_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """``log ptilde^r(theta)`` (eq. 10), optionally with per-component values."""
        theta = _as_2d(theta)
        components = [self.log_component(s, theta, batch_size=batch_size) for s in range(self.num_components)]
        stacked = torch.stack(components, dim=0)
        log_mix = torch.logsumexp(stacked, dim=0) - math.log(self.num_components)
        if return_components:
            return log_mix, components
        return log_mix

    def log_importance_ratio(
        self,
        theta: torch.Tensor,
        normalise: bool = False,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """``log [ p(theta) / ptilde^r(theta) ]`` used by SNPSE-A/B (eq. 12-15)."""
        theta = _as_2d(theta)
        log_prior = self.prior.log_prob(theta)
        log_proposal = self.log_prob(theta, batch_size=batch_size)
        ratio = log_prior - log_proposal
        if normalise:
            ratio = ratio - torch.logsumexp(ratio, dim=0) + math.log(ratio.shape[0])
        return ratio

    def density_ratio(self, theta: torch.Tensor, **kwargs) -> torch.Tensor:
        """``p(theta) / ptilde^r(theta)`` (exp of :meth:`log_importance_ratio`)."""
        return torch.exp(self.log_importance_ratio(theta, **kwargs))

    # -- sampling ---------------------------------------------------------------------
    def sample_pbar(
        self,
        s: int,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sample from a single component ``pbar^s``."""
        if s == 0:
            return self.prior.sample(num_samples, generator)
        region = self.regions[s - 1]
        samples, stats = region.rejection_sample(
            self.prior.sample, num_samples, generator=generator, device=self.device,
            dtype=self.dtype, **kwargs
        )
        self._sample_stats.append({"component": float(s), **stats})
        return samples

    def sample(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
        return_log_prob: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Draw ``theta ~ ptilde^r`` (the proposal of round ``r``, eq. 10)."""
        num_samples = int(num_samples)
        r = self.num_components
        if num_samples <= 0:
            empty = torch.empty((0, self.theta_dim), dtype=self.dtype or torch.float32)
            return (empty, empty.reshape(0)) if return_log_prob else empty

        component = torch.randint(0, r, (num_samples,), generator=generator)
        out = None
        for s in range(r):
            idx = (component == s).nonzero(as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            draw = self.sample_pbar(s, int(idx.numel()), generator=generator, **kwargs)
            draw = _as_2d(torch.as_tensor(draw))
            if out is None:
                out = torch.empty(
                    (num_samples, draw.shape[-1]), dtype=draw.dtype, device=draw.device
                )
            out[idx] = draw[: idx.numel()]
        assert out is not None
        if shuffle:
            perm = torch.randperm(num_samples, generator=generator, device=out.device)
            out = out[perm]
        if return_log_prob:
            with torch.no_grad():
                return out, self.log_prob(out)
        return out

    # Plan-facing aliases.
    sample_proposal = sample
    sample_truncated = sample

    def resample_from_prior(
        self,
        num_samples: int,
        n_candidates: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        replace: bool = True,
    ) -> torch.Tensor:
        """Sampling-Importance-Resampling alternative to rejection sampling.

        Draw ``n_candidates`` prior samples and resample them with weights
        proportional to ``ptilde^r`` (Deistler et al. 2022b, Section 3.2, as
        referenced in Appendix E.3.3).
        """
        n_candidates = int(n_candidates or max(10 * num_samples, 1000))
        candidates = self.prior.sample(n_candidates, generator)
        with torch.no_grad():
            log_w = self.log_prob(candidates)
        log_w = torch.where(torch.isfinite(log_w), log_w, torch.full_like(log_w, -1e30))
        probs = torch.softmax(log_w, dim=0)
        idx = torch.multinomial(
            probs, num_samples, replacement=replace, generator=generator
        )
        return candidates[idx]

    # -- bookkeeping ------------------------------------------------------------------
    def diagnostics(self) -> Dict[str, object]:
        """Summary of the mixture (thresholds, normalisers, prior masses)."""
        return {
            "num_components": self.num_components,
            "eps": self.eps,
            "kappa": [region.kappa for region in self.regions],
            "log_normaliser": [region.log_normaliser for region in self.regions],
            "prior_mass_hpr": [region.diagnostics.get("prior_mass_hpr") for region in self.regions],
            "hypercube_low": [None if r.hypercube_low is None else r.hypercube_low.tolist() for r in self.regions],
            "hypercube_high": [None if r.hypercube_high is None else r.hypercube_high.tolist() for r in self.regions],
            "sample_stats": list(self._sample_stats),
        }

    def summary(self) -> str:
        diag = self.diagnostics()
        lines = [f"TruncatedPrior(r={diag['num_components']}, eps={diag['eps']:g})"]
        for i, (kappa, log_z) in enumerate(zip(diag["kappa"], diag["log_normaliser"]), start=1):
            mass = diag["prior_mass_hpr"][i - 1]
            lines.append(
                f"  bar p^{i}: kappa={kappa:.4f} log Z={log_z:.4f} "
                f"prior-mass={mass if mass is None else f'{mass:.4g}'}"
            )
        return "\n".join(lines)

    # -- serialisation of the (tensor-only) state -------------------------------------
    def region_parameters(self) -> List[Dict[str, object]]:
        """Tensor/float-only description of the regions (the closures are not saved)."""
        return [
            {
                "kappa": region.kappa,
                "eps": region.eps,
                "log_normaliser": region.log_normaliser,
                "hypercube_low": None if region.hypercube_low is None else region.hypercube_low.clone(),
                "hypercube_high": None if region.hypercube_high is None else region.hypercube_high.clone(),
                "round_index": region.round_index,
            }
            for region in self.regions
        ]


def sample_truncated_proposal(
    proposal: TruncatedPrior,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    **kwargs,
) -> torch.Tensor:
    """Functional wrapper: draw ``theta ~ ptilde^r``."""
    return proposal.sample(num_samples, generator=generator, **kwargs)


# --------------------------------------------------------------------------------------
# Self-test: mixture density/weights and rejection sampling against an analytic target
# --------------------------------------------------------------------------------------
def _selftest() -> None:  # pragma: no cover - diagnostic helper
    torch.manual_seed(0)

    theta_dim = 2
    prior_scale = 2.0
    prior_var = prior_scale ** 2
    # Analytic "posterior": N(mu_post, sigma_post^2 I)
    mu_post = torch.tensor([1.2, -0.8])
    sigma_post = 0.45

    def prior_sample_fn(n, generator=None):
        return prior_scale * torch.randn(n, theta_dim, generator=generator)

    def prior_log_prob_fn(theta):
        theta = _as_2d(theta)
        return -0.5 * (theta / prior_scale).pow(2).sum(-1) - 0.5 * theta_dim * math.log(
            2 * math.pi * prior_var
        )

    def post_log_prob_fn(theta):
        theta = _as_2d(theta)
        return (
            -0.5 * ((theta - mu_post) / sigma_post).pow(2).sum(-1)
            - 0.5 * theta_dim * math.log(2 * math.pi * sigma_post ** 2)
        )

    # --- kappa from posterior samples ------------------------------------------------
    gen = torch.Generator().manual_seed(0)
    post_samples = mu_post + sigma_post * torch.randn(20000, theta_dim, generator=gen)
    logp = post_log_prob_fn(post_samples)
    kappa = HPR_eps(logp, DEFAULT_EPS)
    retained = (logp > kappa).to(torch.float32).mean().item()
    print(f"[hpr] kappa={kappa:.4f} retained={retained:.5f} (expect ~{1 - DEFAULT_EPS})")
    assert abs(retained - (1 - DEFAULT_EPS)) < 5e-3

    region = HPRRegion(
        kappa=kappa,
        log_prob_fn=post_log_prob_fn,
        hypercube_low=post_samples.min(0).values,
        hypercube_high=post_samples.max(0).values,
        eps=DEFAULT_EPS,
        n_samples=20000,
        round_index=1,
    )

    proposal = TruncatedPrior(
        prior_sample_fn=prior_sample_fn,
        prior_log_prob_fn=prior_log_prob_fn,
        eps=DEFAULT_EPS,
        theta_dim=theta_dim,
    )
    proposal.add_region(region, estimate_normaliser=True, n_mc=200000, generator=gen)
    print(proposal.summary())

    # --- acceptance rate --------------------------------------------------------------
    samples_1, stats = region.rejection_sample(prior_sample_fn, 4000, generator=gen)
    print(f"[hpr] rejection acceptance rate={stats['acceptance_rate']:.4f}")
    assert 0.0 < stats["acceptance_rate"] < 1.0
    # Truncation must concentrate samples near the posterior mode.
    assert (samples_1 - mu_post).norm(dim=-1).mean() < (post_samples - mu_post).norm(dim=-1).mean() + 1e-6

    # --- mixture sampling -------------------------------------------------------------
    mix = proposal.sample(4000, generator=gen)
    assert mix.shape == (4000, theta_dim)

    # --- normalisation check: E_{prior}[ptilde(theta) / p(theta)] should be 1 ---------
    probe = prior_sample_fn(200000, gen)
    with torch.no_grad():
        log_mix = proposal.log_prob(probe)
        log_prior = prior_log_prob_fn(probe)
    ratio = torch.exp(log_mix - log_prior)
    print(f"[hpr] E_prior[ptilde/p] = {ratio.mean().item():.5f} (expect ~1)")
    assert abs(ratio.mean().item() - 1.0) < 0.05

    # --- extra components accumulate ---------------------------------------------------
    region2 = HPRRegion(
        kappa=kappa + 0.5,
        log_prob_fn=post_log_prob_fn,
        hypercube_low=post_samples.min(0).values,
        hypercube_high=post_samples.max(0).values,
        round_index=2,
    )
    proposal.add_region(region2, n_mc=50000, generator=gen)
    assert proposal.num_components == 3
    log_mix2 = proposal.log_prob(probe)
    ratio2 = torch.exp(log_mix2 - log_prior)
    print(f"[hpr] after 2nd region E_prior[ptilde/p] = {ratio2.mean().item():.5f}")
    assert abs(ratio2.mean().item() - 1.0) < 0.08

    # --- importance ratio -------------------------------------------------------------
    lir = proposal.log_importance_ratio(mix)
    assert torch.isfinite(lir).all()
    print("[hpr] selftest passed.")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
