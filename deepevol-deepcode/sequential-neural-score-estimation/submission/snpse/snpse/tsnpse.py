"""Truncated Sequential Neural Posterior Score Estimation (TSNPSE).

Implements Algorithm 1 of *Sequential Neural Posterior Score Estimation* (SNPSE).

Round ``r`` of TSNPSE uses the highest-probability region (HPR) of the previous
posterior approximation to truncate the prior,

.. math::
    \\bar p^{r}(\\theta) \\propto p(\\theta)\\,\\mathbb{I}
        \\{\\theta \\in \\mathrm{HPR}_{\\varepsilon}(p_{\\psi}^{r-1}(\\theta \\mid x_{obs}))\\},

with :math:`\\bar p^{0} = p`, and the round-``r`` proposal is the mixture
:math:`\\tilde p^{r}(\\theta) = \\frac{1}{r}\\sum_{s=0}^{r-1}\\bar p^{s}(\\theta)` (eq. 9-10).

Algorithm 1 loop (simulation budget ``N``, ``R`` rounds, ``M = N / R`` per round):

1. draw ``M`` parameters from :math:`\\bar p^{r-1}` (the prior for ``r = 1``),
   simulate ``x ~ p(x|\\theta)`` and append to the accumulated dataset ``D``;
2. retrain the score network ``s_psi(theta_t, x, t)`` by minimising a Monte-Carlo
   estimate of the TSNPSE DSM objective (eq. 11) over the *accumulated* dataset ``D``
   -- because each round contributes the same number of simulations, the empirical
   distribution of ``D`` is exactly the uniform mixture :math:`\\tilde p^{r}`, so no
   importance weighting / correction is required (Proposition 3.1);
3. compute :math:`\\bar p^{r}` from ``s_psi(. , x_obs, .)`` via HPR_eps
   (20000 probability-flow ODE samples, log-densities by the instantaneous
   change-of-variables formula, threshold kappa = eps = 5e-4 quantile, rejection
   sampling from the prior with a cheap empirical-hypercube pre-screen).

After ``R`` rounds the posterior is sampled by substituting
``s_psi(theta_t, x_obs, t)`` into the probability-flow ODE (4).

Rounds are not required to be equally sized: ``initial_budget`` and
``simulations_per_round`` allow the neuroscience protocol (30000 initial +
20000 per round) used by the pyloric experiment.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# imports (robust to being executed both as a package and from a flat dir)
# ---------------------------------------------------------------------------
try:  # package-relative imports
    from .npse import (
        NPSE,
        NPSEConfig,
        build_sde,
        standardise_dataset,
        train_score_network,
    )
    from .losses import npse_loss, tsnpse_loss
    from .sampler import (
        SamplerConfig,
        estimate_log_prob,
        sample_posterior,
    )
    from .trainer import select_batch_size
    from .utils import (
        ProgressBar,
        ensure_2d,
        get_device,
        quantile,
        set_seed,
        to_tensor,
    )
except ImportError:  # pragma: no cover - flat import fallback
    from npse import (  # type: ignore
        NPSE,
        NPSEConfig,
        build_sde,
        standardise_dataset,
        train_score_network,
    )
    from losses import npse_loss, tsnpse_loss  # type: ignore
    from sampler import (  # type: ignore
        SamplerConfig,
        estimate_log_prob,
        sample_posterior,
    )
    from trainer import select_batch_size  # type: ignore
    from utils import (  # type: ignore
        ProgressBar,
        ensure_2d,
        get_device,
        quantile,
        set_seed,
        to_tensor,
    )

try:
    from .hpr import (
        DEFAULT_EPS,
        DEFAULT_HYPERCUBE_MARGIN,
        DEFAULT_N_HPR_SAMPLES,
        compute_hpr_region,
    )

    _HAS_HPR = True
except ImportError:  # pragma: no cover
    try:
        from hpr import (  # type: ignore
            DEFAULT_EPS,
            DEFAULT_HYPERCUBE_MARGIN,
            DEFAULT_N_HPR_SAMPLES,
            compute_hpr_region,
        )

        _HAS_HPR = True
    except ImportError:
        _HAS_HPR = False
        DEFAULT_EPS = 5e-4
        DEFAULT_N_HPR_SAMPLES = 20_000
        DEFAULT_HYPERCUBE_MARGIN = 0.0
        compute_hpr_region = None  # type: ignore


__all__ = [
    "TSNPSEConfig",
    "TSNPSE",
    "PriorAdapter",
    "run_tsnpse",
    "sample_from_region",
]


# ===========================================================================
# small helpers
# ===========================================================================
def _call_filtered(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` dropping keyword arguments it does not accept."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    allowed = set(params)
    kept = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = set(kwargs) - set(kept)
    if dropped:
        # positional-compatible names such as 'theta'/'t' are kept as-is
        pass
    return fn(*args, **kept)


def _as_float(x: Any, default: float = float("nan")) -> float:
    try:
        if isinstance(x, Tensor):
            return float(x.detach().reshape(-1)[0].item())
        return float(x)
    except Exception:
        return default


def _attr_first(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _to_2d(value: Any, dtype: torch.dtype, device: torch.device) -> Tensor:
    tensor = to_tensor(value, dtype=dtype, device=device)
    tensor = tensor.detach()
    if tensor.dim() == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.dim() == 1:
        tensor = tensor.reshape(1, -1)
    else:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    return tensor.to(dtype=dtype, device=device)


# ===========================================================================
# priors
# ===========================================================================
class PriorAdapter:
    """Uniform interface around torch / sbibm / numpy / callable priors.

    Provides ``sample(n) -> (n, d)`` and ``log_prob(theta) -> (n,)`` in the
    original parameter space.
    """

    def __init__(
        self,
        prior: Any = None,
        sample_fn: Optional[Callable] = None,
        log_prob_fn: Optional[Callable] = None,
        theta_dim: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = get_device(device)
        self.dtype = dtype
        self.theta_dim = theta_dim
        self._prior = prior
        self._explicit_sample_fn = sample_fn
        self._explicit_log_prob_fn = log_prob_fn

        # a sbibm task object exposes get_prior()
        if prior is not None and hasattr(prior, "get_prior") and not hasattr(prior, "sample"):
            prior = prior.get_prior()
            self._prior = prior

        if prior is not None:
            if sample_fn is None and hasattr(prior, "sample"):
                sample_fn = None  # handled below through the object
            if log_prob_fn is None and not hasattr(prior, "log_prob"):
                log_prob_fn = None

        self._sample_error: Optional[Exception] = None

    # -- sampling ----------------------------------------------------------
    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        n = int(n)
        if n <= 0:
            return torch.empty(0, self.theta_dim or 0, dtype=self.dtype, device=self.device)

        if self._explicit_sample_fn is not None:
            out = _call_filtered(self._explicit_sample_fn, n, generator=generator)
            return _to_2d(out, self.dtype, self.device)

        prior = self._prior
        if prior is None:
            raise ValueError("PriorAdapter requires either a prior object or sample_fn.")

        # sbibm-wrapped tasks / callables taking the batch size as an integer
        if callable(prior) and not hasattr(prior, "sample"):
            out = _call_filtered(prior, n)
            return _to_2d(out, self.dtype, self.device)

        errors: List[Exception] = []
        attempts = (
            lambda: prior.sample((n,)),
            lambda: prior.sample(torch.Size([n])),
            lambda: prior.rsample((n,)),
            lambda: prior.sample(n),
        )
        for attempt in attempts:
            try:
                out = attempt()
                return _to_2d(out, self.dtype, self.device)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        self._sample_error = errors[-1] if errors else None
        raise RuntimeError(
            f"Could not sample from the supplied prior object: {self._sample_error}"
        )

    # -- density -----------------------------------------------------------
    def log_prob(self, theta: Tensor) -> Tensor:
        theta = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))

        if self._explicit_log_prob_fn is not None:
            out = _call_filtered(self._explicit_log_prob_fn, theta)
            return _as_log_prob(out, theta.shape[0], self.dtype, self.device)

        prior = self._prior
        if prior is None:
            raise ValueError("PriorAdapter requires either a prior object or log_prob_fn.")

        if callable(prior) and not hasattr(prior, "log_prob"):
            out = _call_filtered(prior, theta)
            return _as_log_prob(out, theta.shape[0], self.dtype, self.device)

        if hasattr(prior, "log_prob"):
            try:
                out = prior.log_prob(theta)
            except Exception:  # pragma: no cover - numpy fallback
                out = prior.log_prob(theta.detach().cpu().numpy())
            return _as_log_prob(out, theta.shape[0], self.dtype, self.device)

        raise ValueError("Supplied prior object has no log_prob method.")

    # -- prior-sampling callback for HPR rejection sampling -----------------
    def sample_fn(self, n: int, generator: Optional[torch.Generator] = None) -> Tensor:
        return self.sample(n, generator=generator)


def _as_log_prob(value: Any, n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    tensor = to_tensor(value, dtype=dtype, device=device)
    if tensor.dim() > 1:
        tensor = tensor.reshape(tensor.shape[0]) if tensor.shape[0] == n else tensor.reshape(-1)
    else:
        tensor = tensor.reshape(-1)
    if tensor.numel() == 1 and n > 1:
        tensor = tensor.expand(n)
    return tensor.reshape(-1).to(dtype=dtype, device=device)


# ===========================================================================
# lightweight fallback HPR region (used only if hpr.py is unavailable)
# ===========================================================================
@dataclass
class FallbackHPRRegion:
    """Minimal stand-in for :class:`snpse.hpr.HPRRegion`."""

    kappa: float
    hypercube_low: Tensor
    hypercube_high: Tensor
    eps: float = DEFAULT_EPS
    n_samples: int = 0
    round_index: int = 0
    samples: Optional[Tensor] = None
    log_probs: Optional[Tensor] = None
    log_prob_fn: Optional[Callable[[Tensor], Tensor]] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def inside_hypercube(self, theta: Tensor) -> Tensor:
        theta = ensure_2d(theta)
        low = self.hypercube_low.to(theta.dtype).to(theta.device)
        high = self.hypercube_high.to(theta.dtype).to(theta.device)
        return ((theta >= low) & (theta <= high)).all(dim=-1)

    def log_prob(self, theta: Tensor, batch_size: int = 4096, **kwargs: Any) -> Tensor:
        theta = ensure_2d(theta)
        if self.log_prob_fn is None:
            return torch.full((theta.shape[0],), float("-inf"), dtype=theta.dtype, device=theta.device)
        out = self.log_prob_fn(theta)
        return _as_log_prob(out, theta.shape[0], theta.dtype, theta.device)

    def contains(self, theta: Tensor, batch_size: int = 4096, **kwargs: Any) -> Tensor:
        theta = ensure_2d(theta)
        inside = self.inside_hypercube(theta)
        if self.log_prob_fn is None:
            return inside
        lp = self.log_prob(theta, batch_size=batch_size)
        return inside & (lp > self.kappa)

    accept = contains

    def rejection_sample(
        self,
        prior_sample_fn: Callable,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        batch_size: Optional[int] = None,
        max_batches: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        return _local_rejection_sample(
            self,
            prior_sample_fn,
            num_samples,
            generator=generator,
            batch_size=batch_size,
            max_batches=max_batches,
        )


def _local_rejection_sample(
    region: Any,
    prior_sample_fn: Callable,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    batch_size: Optional[int] = None,
    max_batches: Optional[int] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Rejection sampling from ``pbar^r ∝ p(theta) I{theta in HPR}``.

    A cheap empirical-hypercube screen is applied before any (expensive)
    log-density evaluation, following Appendix E.3.3.
    """
    num_samples = int(num_samples)
    if num_samples <= 0:
        return torch.empty(0, 0), {"accepted": 0, "draws": 0}

    target_batch = int(batch_size) if batch_size else max(4 * num_samples, 1024)
    hard_cap = int(max_batches) if max_batches else 2000

    accepted: List[Tensor] = []
    n_acc = 0
    n_draw = 0
    batches = 0

    dim_hint = None
    if getattr(region, "hypercube_low", None) is not None:
        dim_hint = int(region.hypercube_low.reshape(-1).numel())

    while n_acc < num_samples and batches < hard_cap:
        batches += 1
        batch = min(target_batch, max(256, 4 * (num_samples - n_acc)))
        theta = _to_2d(
            prior_sample_fn(batch, generator=generator),
            torch.float32,
            region.hypercube_low.device if getattr(region, "hypercube_low", None) is not None else torch.device("cpu"),
        )
        n_draw += int(theta.shape[0])

        screen = region.inside_hypercube(theta)
        cand = theta[screen]
        if cand.numel() == 0:
            continue

        keep = region.contains(cand)
        if keep.numel() == 0:
            continue
        good = cand[keep]
        if good.numel() == 0:
            continue
        accepted.append(good)
        n_acc += int(good.shape[0])

    if not accepted:
        out = torch.empty(0, dim_hint or 0)
    else:
        out = torch.cat(accepted, dim=0)[:num_samples]
    diag = {
        "accepted": n_acc,
        "draws": n_draw,
        "acceptance_rate": (n_acc / n_draw) if n_draw else float("nan"),
        "batches": batches,
    }
    return out, diag


@dataclass
class _MixtureProposal:
    """Uniform mixture :math:`\\tilde p^{r} = (1/r)\\sum_{s<r}\\bar p^{s}` (eq. 10)."""

    prior: PriorAdapter
    regions: List[Any] = field(default_factory=list)

    def sample(self, num_samples: int, generator: Optional[torch.Generator] = None) -> Tensor:
        k = len(self.regions) + 1
        base, rest = divmod(int(num_samples), k)
        counts = [base + (1 if i < rest else 0) for i in range(k)]

        parts: List[Tensor] = []
        if counts[0] > 0:
            parts.append(self.prior.sample(counts[0], generator=generator))
        for region, count in zip(self.regions, counts[1:]):
            if count <= 0:
                continue
            part, _ = sample_from_region(region, self.prior, count, generator=generator)
            if part is not None and part.numel() > 0:
                parts.append(part)
        if not parts:
            return self.prior.sample(num_samples, generator=generator)
        return torch.cat(parts, dim=0)

    def log_prob(self, theta: Tensor) -> Tensor:
        theta = ensure_2d(theta)
        k = len(self.regions) + 1
        terms = [self.prior.log_prob(theta) - math.log(k)]
        for region in self.regions:
            inside = region.inside_hypercube(theta)
            lp = region.log_prob(theta)
            lp = torch.where(
                inside, lp, torch.full_like(lp, float("-inf"))
            ) - math.log(k)
            terms.append(lp)
        return torch.logsumexp(torch.stack(terms, dim=0), dim=0)


def sample_from_region(
    region: Any,
    prior: PriorAdapter,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    max_batches: int = 2000,
) -> Tuple[Tensor, Dict[str, Any]]:
    """Draw from a truncated prior component ``pbar^r`` (rejection sampling)."""
    num_samples = int(num_samples)
    if num_samples <= 0:
        return torch.empty(0, prior.theta_dim or 0), {}

    if hasattr(region, "rejection_sample"):
        try:
            out = _call_filtered(
                region.rejection_sample,
                prior.sample_fn,
                num_samples,
                generator=generator,
                dtype=prior.dtype,
                device=prior.device,
                max_batches=max_batches,
            )
            diag: Dict[str, Any] = {}
            if isinstance(out, tuple):
                out, diag = out[0], (out[1] if isinstance(out[1], dict) else {})
            if isinstance(out, Tensor) and out.numel() > 0 and out.shape[0] >= num_samples:
                return out[:num_samples].to(dtype=prior.dtype, device=prior.device), diag
        except Exception:  # pragma: no cover - fall through to local sampling
            pass

    return _local_rejection_sample(
        region, prior.sample_fn, num_samples, generator=generator, max_batches=max_batches
    )


# ===========================================================================
# configuration
# ===========================================================================
@dataclass
class TSNPSEConfig(NPSEConfig):
    """Hyper-parameters of the sequential TSNPSE procedure (Algorithm 1)."""

    num_rounds: int = 10
    eps: float = DEFAULT_EPS
    n_hpr_samples: int = DEFAULT_N_HPR_SAMPLES
    hypercube_margin: float = DEFAULT_HYPERCUBE_MARGIN
    proposal_mode: str = "latest"  # "latest": pbar^{r-1} (Algorithm 1) | "mixture"
    initial_budget: Optional[int] = None
    simulations_per_round: Optional[int] = None
    num_samples: int = 10_000
    compute_final_region: bool = False
    round_accept_batch_size: Optional[int] = None
    max_rejection_batches: int = 2000
    reuse_standardisers: bool = True
    verbose_rounds: bool = False
    track_round_diagnostics: bool = True

    # ------------------------------------------------------------------
    def resolved_round_budgets(self) -> List[int]:
        """Simulations used in each round (Algorithm 1 uses ``M = N/R``)."""
        rounds = max(1, int(self.num_rounds))
        if self.simulations_per_round is not None:
            first = int(self.initial_budget) if self.initial_budget else int(self.simulations_per_round)
            return [first] + [int(self.simulations_per_round)] * (rounds - 1)
        budget = int(self.budget) if self.budget else 1000
        base, rest = divmod(budget, rounds)
        return [base + (1 if i < rest else 0) for i in range(rounds)]

    def total_simulations(self) -> int:
        return int(sum(self.resolved_round_budgets()))

    def as_dict(self) -> Dict[str, Any]:  # noqa: D102
        out = super().as_dict() if hasattr(super(), "as_dict") else dict(self.__dict__)
        out["round_budgets"] = self.resolved_round_budgets()
        return out


# ===========================================================================
# TSNPSE
# ===========================================================================
class TSNPSE:
    """Sequential (truncated-proposal) posterior score estimator.

    Parameters
    ----------
    theta_dim, x_dim:
        Parameter / observation dimensionality.
    prior:
        Prior object (torch distribution, sbibm task, or callable).
    simulator:
        ``theta -> x`` simulator accepting torch tensors (numpy fallback).
    config:
        :class:`TSNPSEConfig`.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        prior: Any = None,
        simulator: Optional[Callable] = None,
        config: Optional[TSNPSEConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        network: Optional[torch.nn.Module] = None,
        prior_sample_fn: Optional[Callable] = None,
        prior_log_prob_fn: Optional[Callable] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.config = config if config is not None else TSNPSEConfig()
        self.device = get_device(device if device is not None else getattr(self.config, "device", None))
        self.dtype = dtype
        self.simulator = simulator
        self.prior = PriorAdapter(
            prior=prior,
            sample_fn=prior_sample_fn,
            log_prob_fn=prior_log_prob_fn,
            theta_dim=self.theta_dim,
            device=self.device,
            dtype=dtype,
        )
        self.network = network
        self.sde = None
        self.theta_standardiser = None
        self.x_standardiser = None
        self.regions: List[Any] = []
        self.history: List[Dict[str, Any]] = []
        self.dataset: Optional[Dict[str, Tensor]] = None

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------
    def _batch_size(self, total_budget: int) -> Optional[int]:
        if getattr(self.config, "batch_size", None):
            return int(self.config.batch_size)
        try:
            return int(select_batch_size(total_budget))
        except Exception:  # pragma: no cover
            return None

    def _sampler_config(self) -> Any:
        try:
            return self.config.sampler_config()
        except Exception:  # pragma: no cover
            return SamplerConfig(
                method=getattr(self.config, "sampler_method", "rk45"),
                atol=getattr(self.config, "sampler_atol", 1e-5),
                rtol=getattr(self.config, "sampler_rtol", 1e-5),
                n_steps=getattr(self.config, "sampler_n_steps", 1000),
            )

    def _simulate(self, theta: Tensor) -> Tensor:
        if self.simulator is None:
            raise ValueError("A simulator must be supplied to run TSNPSE.")
        theta = ensure_2d(theta)
        out: Any = None
        try:
            out = self.simulator(theta)
        except Exception as exc_torch:  # noqa: BLE001
            try:
                out = self.simulator(theta.detach().cpu().numpy())
            except Exception as exc_numpy:  # noqa: BLE001
                raise RuntimeError(
                    "Simulator call failed for both torch and numpy inputs: "
                    f"{exc_torch} / {exc_numpy}"
                )
        return _to_2d(out, self.dtype, self.device)

    def _std_shifts(self) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        theta_shift = _attr_first(self.theta_standardiser, ("shift", "mean", "loc")) if self.theta_standardiser is not None else None
        theta_scale = _attr_first(self.theta_standardiser, ("scale", "std")) if self.theta_standardiser is not None else None
        x_shift = _attr_first(self.x_standardiser, ("shift", "mean", "loc")) if self.x_standardiser is not None else None
        x_scale = _attr_first(self.x_standardiser, ("scale", "std")) if self.x_standardiser is not None else None
        return theta_shift, theta_scale, x_shift, x_scale

    # ------------------------------------------------------------------
    # HPR regions
    # ------------------------------------------------------------------
    def _compute_region(
        self,
        sde: Any,
        score_fn: Callable,
        x_obs: Tensor,
        round_index: int,
        generator: Optional[torch.Generator],
        num_samples: Optional[int] = None,
    ) -> Any:
        cfg = self.config
        n = int(num_samples if num_samples is not None else cfg.n_hpr_samples)
        theta_shift, theta_scale, x_shift, x_scale = self._std_shifts()

        if _HAS_HPR and compute_hpr_region is not None:
            try:
                region = _call_filtered(
                    compute_hpr_region,
                    sde,
                    score_fn,
                    x_obs,
                    num_samples=n,
                    eps=cfg.eps,
                    config=self._sampler_config(),
                    theta_dim=self.theta_dim,
                    generator=generator,
                    device=self.device,
                    dtype=self.dtype,
                    theta_shift=theta_shift,
                    theta_scale=theta_scale,
                    x_shift=x_shift,
                    x_scale=x_scale,
                    hypercube_margin=cfg.hypercube_margin,
                    round_index=round_index,
                    verbose=bool(cfg.verbose_rounds),
                )
                if region is not None:
                    return region
            except Exception as exc:  # noqa: BLE001
                if cfg.verbose_rounds:
                    print(f"[tsnpse] hpr.compute_hpr_region failed ({exc}); using fallback.")

        return self._fallback_region(sde, score_fn, x_obs, round_index, n, generator)

    def _fallback_region(
        self,
        sde: Any,
        score_fn: Callable,
        x_obs: Tensor,
        round_index: int,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> FallbackHPRRegion:
        """HPR_eps computed directly from the sampler (Appendix E.3.3)."""
        cfg = self.config
        theta_shift, theta_scale, x_shift, x_scale = self._std_shifts()

        samples, log_prob = sample_posterior(
            sde,
            score_fn,
            x_obs,
            num_samples,
            config=self._sampler_config(),
            with_log_prob=True,
            theta_dim=self.theta_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            theta_shift=theta_shift,
            theta_scale=theta_scale,
            x_shift=x_shift,
            x_scale=x_scale,
        )
        samples = ensure_2d(to_tensor(samples, dtype=self.dtype, device=self.device))
        log_prob = _as_log_prob(log_prob, samples.shape[0], self.dtype, self.device)

        kappa = _as_float(quantile(log_prob.detach(), cfg.eps))
        low = samples.min(dim=0).values
        high = samples.max(dim=0).values
        if cfg.hypercube_margin:
            width = (high - low).clamp_min(1e-12)
            low = low - cfg.hypercube_margin * width
            high = high + cfg.hypercube_margin * width

        def _log_prob_fn(theta: Tensor) -> Tensor:
            lp = estimate_log_prob(
                sde,
                score_fn,
                x_obs,
                ensure_2d(theta),
                config=self._sampler_config(),
                theta_dim=self.theta_dim,
                device=self.device,
                dtype=self.dtype,
                theta_shift=theta_shift,
                theta_scale=theta_scale,
                x_shift=x_shift,
                x_scale=x_scale,
            )
            return _as_log_prob(lp, ensure_2d(theta).shape[0], self.dtype, self.device)

        return FallbackHPRRegion(
            kappa=kappa,
            hypercube_low=low.detach(),
            hypercube_high=high.detach(),
            eps=cfg.eps,
            n_samples=int(samples.shape[0]),
            round_index=int(round_index),
            samples=samples.detach() if cfg.track_round_diagnostics else None,
            log_probs=log_prob.detach() if cfg.track_round_diagnostics else None,
            log_prob_fn=_log_prob_fn,
            diagnostics={
                "median_log_prob": _as_float(log_prob.median()),
                "retained_mass": float((log_prob > kappa).to(self.dtype).mean()),
            },
        )

    # ------------------------------------------------------------------
    # proposal sampling
    # ------------------------------------------------------------------
    def _draw_round_parameters(
        self,
        round_index: int,
        num_samples: int,
        generator: Optional[torch.Generator],
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Draw ``theta ~ pbar^{r-1}`` (the prior for ``r = 1``)."""
        cfg = self.config
        if round_index <= 1 or not self.regions:
            return self.prior.sample(num_samples, generator=generator), {"source": "prior"}

        if str(cfg.proposal_mode).lower() == "mixture":
            proposal = _MixtureProposal(self.prior, list(self.regions))
            return proposal.sample(num_samples, generator=generator), {"source": "mixture"}

        region = self.regions[-1]
        theta, diag = sample_from_region(
            region,
            self.prior,
            num_samples,
            generator=generator,
            max_batches=int(cfg.max_rejection_batches),
        )
        if theta is None or theta.numel() == 0:
            theta = self.prior.sample(num_samples, generator=generator)
            diag = dict(diag or {})
            diag["source"] = "prior_fallback"
        else:
            diag = dict(diag or {})
            diag["source"] = f"pbar^{round_index - 1}"
        return ensure_2d(theta), diag

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(
        self,
        x_obs: Tensor,
        num_samples: Optional[int] = None,
        with_log_prob: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        if seed is not None or getattr(cfg, "seed", None) is not None:
            generator = set_seed(int(seed if seed is not None else cfg.seed))
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(torch.initial_seed() % (2**31))

        x_obs = ensure_2d(to_tensor(x_obs, dtype=self.dtype, device=self.device))
        round_budgets = cfg.resolved_round_budgets()
        total_budget = int(sum(round_budgets))
        batch_size = self._batch_size(total_budget)
        if batch_size is not None:
            self.config.batch_size = batch_size

        self.regions = []
        self.history = []
        network = self.network
        sde = None
        theta_data: List[Tensor] = []
        x_data: List[Tensor] = []
        progress = ProgressBar(
            total=len(round_budgets),
            desc="TSNPSE rounds",
            enabled=bool(getattr(cfg, "verbose", False) or cfg.verbose_rounds),
        )

        for r, m in enumerate(round_budgets, start=1):
            theta_new, draw_diag = self._draw_round_parameters(r, int(m), generator)
            theta_new = theta_new.detach().to(self.dtype)
            x_new = self._simulate(theta_new).detach().to(self.dtype)
            theta_data.append(theta_new)
            x_data.append(x_new)

            theta_all = torch.cat(theta_data, dim=0)
            x_all = torch.cat(x_data, dim=0)

            # Standardisation fitted on round-1 data and reused (Appendix E.3.2).
            if r == 1 and bool(getattr(cfg, "standardise", True)) and cfg.reuse_standardisers:
                try:
                    _, _, theta_std, x_std = standardise_dataset(theta_all, x_all)
                    self.theta_standardiser, self.x_standardiser = theta_std, x_std
                except Exception:  # pragma: no cover
                    self.theta_standardiser = self.x_standardiser = None

            # Round 1 also fixes the VE sigma_max from round-1 data only (Addendum).
            train_kwargs: Dict[str, Any] = dict(
                config=cfg,
                sde=sde,
                network=network,
                objective=tsnpse_loss,
                theta_standardiser=self.theta_standardiser,
                x_standardiser=self.x_standardiser,
                device=self.device,
                generator=generator,
                verbose=cfg.verbose_rounds,
            )
            try:
                network, sde, info = _call_filtered(
                    train_score_network, theta_all, x_all, **train_kwargs
                )
            except TypeError:
                train_kwargs["objective"] = "tsnpse"
                network, sde, info = _call_filtered(
                    train_score_network, theta_all, x_all, **train_kwargs
                )

            diag = self._region_diagnostics(sde, network, x_obs, draw_diag)

            # Compute pbar^r from the freshly trained posterior approximation.
            if r < len(round_budgets) or cfg.compute_final_region:
                try:
                    region = self._compute_region(sde, network, x_obs, r, generator)
                    self.regions.append(region)
                    diag["kappa"] = _as_float(getattr(region, "kappa", float("nan")))
                    diag["hpr_region_index"] = r
                except Exception as exc:  # noqa: BLE001
                    diag["hpr_error"] = str(exc)

            record: Dict[str, Any] = {
                "round": r,
                "num_new": int(theta_new.shape[0]),
                "num_total": int(theta_all.shape[0]),
                "draw": dict(draw_diag),
                **diag,
            }
            self.history.append(record)
            if callback is not None:
                callback(r, record)
            if progress is not None:
                progress.update(1, postfix={"N": int(theta_all.shape[0])})

        if progress is not None:
            progress.close()

        self.network = network
        self.sde = sde
        self.dataset = {"theta": theta_all, "x": x_all}

        n_post = int(num_samples if num_samples is not None else cfg.num_samples)
        theta_shift, theta_scale, x_shift, x_scale = self._std_shifts()
        samples, log_prob = sample_posterior(
            sde,
            network,
            x_obs,
            n_post,
            config=self._sampler_config(),
            with_log_prob=True,
            theta_dim=self.theta_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            theta_shift=theta_shift,
            theta_scale=theta_scale,
            x_shift=x_shift,
            x_scale=x_scale,
        )
        samples = ensure_2d(to_tensor(samples, dtype=self.dtype, device=self.device))

        result: Dict[str, Any] = {
            "theta": samples,
            "model": self,
            "network": network,
            "sde": sde,
            "regions": list(self.regions),
            "history": list(self.history),
            "info": info if isinstance(info, dict) else {"info": info},
            "num_rounds": len(round_budgets),
            "num_simulations": total_budget,
            "dataset": self.dataset,
        }
        if with_log_prob:
            result["log_prob"] = _as_log_prob(log_prob, samples.shape[0], self.dtype, self.device)
        return result

    # ------------------------------------------------------------------
    def _region_diagnostics(
        self, sde: Any, network: torch.nn.Module, x_obs: Tensor, draw_diag: Dict[str, Any]
    ) -> Dict[str, Any]:
        diag: Dict[str, Any] = {}
        if not self.config.track_round_diagnostics:
            return diag
        diag["draw_source"] = draw_diag.get("source")
        if draw_diag:
            for key in ("acceptance_rate", "accepted", "draws"):
                if key in draw_diag:
                    diag[f"draw_{key}"] = draw_diag[key]
        return diag

    # ------------------------------------------------------------------
    # posterior utilities
    # ------------------------------------------------------------------
    def sample(
        self,
        x_obs: Tensor,
        num_samples: int = 10_000,
        with_log_prob: bool = False,
        method: str = "ode",
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Sample the final posterior approximation by reverse diffusion."""
        if self.network is None or self.sde is None:
            raise RuntimeError("TSNPSE must be run (or fitted) before sampling.")
        theta_shift, theta_scale, x_shift, x_scale = self._std_shifts()
        x_obs = ensure_2d(to_tensor(x_obs, dtype=self.dtype, device=self.device))
        if str(method).lower() in ("sde", "reverse-sde", "reverse_sde"):
            from .sampler import reverse_sde_sample  # local import to avoid cycles

            return reverse_sde_sample(
                self.sde,
                self.network,
                x_obs,
                num_samples,
                config=self._sampler_config(),
                theta_dim=self.theta_dim,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
                **kwargs,
            )
        samples, log_prob = sample_posterior(
            self.sde,
            self.network,
            x_obs,
            num_samples,
            config=self._sampler_config(),
            with_log_prob=True,
            theta_dim=self.theta_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            theta_shift=theta_shift,
            theta_scale=theta_scale,
            x_shift=x_shift,
            x_scale=x_scale,
            **kwargs,
        )
        samples = ensure_2d(to_tensor(samples, dtype=self.dtype, device=self.device))
        if with_log_prob:
            return samples, _as_log_prob(log_prob, samples.shape[0], self.dtype, self.device)
        return samples

    posterior_samples = sample

    def log_prob(self, x_obs: Tensor, theta: Tensor, **kwargs: Any) -> Tensor:
        if self.network is None or self.sde is None:
            raise RuntimeError("TSNPSE must be run (or fitted) before density evaluation.")
        theta_shift, theta_scale, x_shift, x_scale = self._std_shifts()
        lp = estimate_log_prob(
            self.sde,
            self.network,
            ensure_2d(to_tensor(x_obs, dtype=self.dtype, device=self.device)),
            ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device)),
            config=self._sampler_config(),
            theta_dim=self.theta_dim,
            device=self.device,
            dtype=self.dtype,
            theta_shift=theta_shift,
            theta_scale=theta_scale,
            x_shift=x_shift,
            x_scale=x_scale,
            **kwargs,
        )
        theta = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        return _as_log_prob(lp, theta.shape[0], self.dtype, self.device)

    posterior_log_prob = log_prob

    def score(self, theta: Tensor, x: Tensor, t: Optional[Tensor] = None) -> Tensor:
        if self.network is None:
            raise RuntimeError("TSNPSE holds no trained network yet.")
        theta = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        if t is None:
            t = torch.zeros(theta.shape[0], 1, dtype=self.dtype, device=self.device)
        out = self.network(theta, x, t)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def proposal_log_prob(self, theta: Tensor, round_index: Optional[int] = None) -> Tensor:
        """log density of ``ptilde^r = (1/r) sum_{s<r} pbar^s`` (eq. 10)."""
        theta = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        regions = self.regions if round_index is None else self.regions[:round_index]
        proposal = _MixtureProposal(self.prior, list(regions))
        return proposal.log_prob(theta)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict() if self.network is not None else None,
            "config": self.config.as_dict(),
            "history": self.history,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, theta_dim: int, x_dim: int, **kwargs: Any) -> "TSNPSE":
        payload = torch.load(path, map_location="cpu")
        raise NotImplementedError(
            "Rebuild TSNPSE with the same architecture, then load the state dict: "
            "torch.load(path)['network']."
        )


# ===========================================================================
# convenience functions
# ===========================================================================
def run_tsnpse(
    prior: Any,
    simulator: Callable,
    x_obs: Tensor,
    theta_dim: int,
    x_dim: int,
    config: Optional[TSNPSEConfig] = None,
    num_samples: Optional[int] = None,
    with_log_prob: bool = False,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the complete TSNPSE procedure (Algorithm 1) and return the posterior."""
    estimator = TSNPSE(
        theta_dim=theta_dim,
        x_dim=x_dim,
        prior=prior,
        simulator=simulator,
        config=config,
        device=device,
    )
    return estimator.run(
        x_obs,
        num_samples=num_samples,
        with_log_prob=with_log_prob,
        seed=seed,
        **kwargs,
    )


# ===========================================================================
# self test
# ===========================================================================
def _selftest(rounds: int = 3, budget: int = 3000, dtype: torch.dtype = torch.float32) -> None:
    """Sequential sanity check on a 2-D Gaussian-linear model.

    prior: theta ~ N(0, I); simulator: x = A theta + sigma * eps so that the
    posterior is N(m, S) in closed form.
    """
    torch.manual_seed(0)
    d, p = 2, 2
    A = torch.tensor([[1.0, 0.5], [-0.25, 1.25]], dtype=dtype)
    sigma = 0.1
    theta_obs = torch.tensor([0.6, -0.4], dtype=dtype)
    x_obs = A @ theta_obs

    cov_post = torch.linalg.inv(torch.eye(d) + (A.T @ A) / sigma**2)
    mean_post = cov_post @ (A.T @ x_obs) / sigma**2

    prior = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))

    def simulator(theta: Tensor) -> Tensor:
        return theta @ A.T + sigma * torch.randn(theta.shape[0], p, dtype=theta.dtype)

    cfg = TSNPSEConfig(
        num_rounds=rounds,
        budget=budget,
        sde="ve",
        max_iters=1500,
        batch_size=100,
        num_samples=4000,
        n_hpr_samples=2000,
        verbose=False,
    )
    est = TSNPSE(theta_dim=d, x_dim=p, prior=prior, simulator=simulator, config=cfg)
    out = est.run(x_obs, num_samples=20000, with_log_prob=True, seed=1)

    samples = out["theta"]
    emp_mean = samples.mean(dim=0)
    emp_cov = torch.cov(samples.T)
    mean_err = float((emp_mean - mean_post).abs().max())
    cov_err = float((emp_cov - cov_post).abs().max())

    log_prob_model = est.log_prob(x_obs, mean_post.reshape(1, -1))
    log_prob_true = float(
        torch.distributions.MultivariateNormal(mean_post, cov_post).log_prob(mean_post.reshape(1, -1))
    )
    print(f"[TSNPSE selftest] rounds={rounds} budget={budget}")
    print(f"  posterior mean error (max abs) : {mean_err:.4f}")
    print(f"  posterior cov  error (max abs) : {cov_err:.4f}")
    print(f"  log p(mode) model / true        : {float(log_prob_model):.3f} / {log_prob_true:.3f}")
    print(f"  final proposal components       : {len(out['regions'])}")
    for rec in out["history"]:
        print(
            f"  round {rec['round']}: N={rec['num_total']}, "
            f"kappa={rec.get('kappa', float('nan')):.3f}"
        )


if __name__ == "__main__":  # pragma: no cover
    _selftest()
