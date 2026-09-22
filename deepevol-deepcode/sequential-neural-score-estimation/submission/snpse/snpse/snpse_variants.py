"""Alternative sequential SNPSE variants: SNPSE-A, SNPSE-B and SNPSE-C.

Implements the additional sequential approaches discussed in Section 3.2 and
Appendices C.2-C.4 of the SNPSE paper:

* ``SNPSE-A`` (Algorithm 3, Appendix C.2).  In each round we draw parameters
  from the *most recent* posterior approximation, simulate data, and train a
  score network on the accumulated dataset with the plain denoising posterior
  score matching objective (79), i.e. the network learns
  :math:`\\tilde p_t^r(\\theta_t \\mid x)`, the score of the *proposal*
  posterior.  After training we draw samples from the proposal posterior with
  the probability flow ODE (4) and apply sampling-importance-resampling (SIR)
  with sample probabilities proportional to

      h_i = p(theta_i) / ptilde^r(theta_i)                          (eq. 80/13)

  to recover samples from the correct posterior.  Equality for all rounds is
  only available up to ``r = 2`` (the normalising constant of the corrected
  posterior is intractable afterwards, Appendix C.2.3), so the proposal prior
  is additionally approximated from samples (Section C.2.3, "Approximating the
  Proposal Prior").

* ``SNPSE-B`` (Algorithm 4, Appendix C.3).  Same data generation as SNPSE-A,
  but the importance weight is included *inside* the denoising score matching
  objective (eq. 99/15)

      J_post^{SNPSE-B}(psi) = 1/2 int_0^T lambda_t E[ p(theta_0)/ptilde^r(theta_0)
                                 * ||s_psi - grad log p_{t|0}||^2 ] dt

  which is minimised by the true posterior score (Appendix C.3.2), so no
  sampling-time correction is needed.  As in SNPE-B the weights may be high
  variance.

* ``SNPSE-C`` (Algorithm 5, Appendices C.4.1-C.4.3).  Also trains on proposal
  samples, but learns a *corrected* score network (eq. 103/116)

      tilde s_psi^r(theta_t, x, t) = s_psi(theta_t, x, t)
                                    + s_varphi^{r,prop}(theta_t, t)
                                    - grad_theta log p_t(theta_t),

  where ``s_varphi^{r,prop}`` estimates the score of the *perturbed proposal
  prior* (eq. 123) and ``grad log p_t(theta_t)`` is the perturbed prior score
  (analytic where available, otherwise estimated as in Algorithm 2 of
  Appendix B.2).  By the argument of Appendix C.4.2 this makes ``s_psi``
  converge to the score of the sequence
  ``p_t^{r,seq}`` which interpolates between the true posterior and the
  reference distribution --- not the standard forward-SDE marginals --- so
  (as the paper reports) SNPSE-C is expected to be markedly worse than TSNPSE.

The proposal prior used by all three variants is the equal mixture of the
posterior estimates from the previous rounds,

    ptilde^r(theta) = (1/r) sum_{s=0}^{r-1} p_psi^s(theta | x_obs),
    p_psi^0(theta | x_obs) := p(theta),                            (eq. 10/81)

which is exactly the marginal distribution of the concatenated dataset
``D = union_s {(theta_i^s, x_i^s)}``.  Its components are either known
analytically (the prior), evaluated exactly with the probability flow ODE and
the instantaneous change-of-variables formula (5) (available for the first
learned component), or approximated from round samples (Gaussian mixture,
kernel density estimate, or atomic/uniform-on-bounding-box components).
"""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Imports of the rest of the SNPSE package (relative first, flat fallback)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised by import style
    from .sdes import SDE, T_FINAL, get_sde
    from .sdes import sigma_max_technique1
except ImportError:  # pragma: no cover
    from sdes import SDE, T_FINAL, get_sde
    from sdes import sigma_max_technique1

try:  # pragma: no cover
    from .score_network import ScoreNetwork, get_score_network, count_parameters
except ImportError:  # pragma: no cover
    from score_network import ScoreNetwork, get_score_network, count_parameters

try:  # pragma: no cover
    from .losses import (
        dsm_loss,
        npse_loss,
        tsnpse_loss,
        snpse_a_loss,
        snpse_b_loss,
        snpse_c_loss,
        importance_weight,
    )
except ImportError:  # pragma: no cover
    from losses import (
        dsm_loss,
        npse_loss,
        tsnpse_loss,
        snpse_a_loss,
        snpse_b_loss,
        snpse_c_loss,
        importance_weight,
    )

try:  # pragma: no cover
    from .sampler import SamplerConfig, sample_posterior, estimate_log_prob
except ImportError:  # pragma: no cover
    from sampler import SamplerConfig, sample_posterior, estimate_log_prob

try:  # pragma: no cover
    from .trainer import (
        TrainConfig,
        TrainHistory,
        Trainer,
        train_network,
        select_batch_size,
    )
except ImportError:  # pragma: no cover
    from trainer import (
        TrainConfig,
        TrainHistory,
        Trainer,
        train_network,
        select_batch_size,
    )

try:  # pragma: no cover
    from .utils import (
        Standardiser,
        fit_standardiser,
        get_device,
        set_seed,
        ensure_2d,
        normalise_weights,
        safe_log,
        to_tensor,
        ProgressBar,
    )
except ImportError:  # pragma: no cover
    from utils import (
        Standardiser,
        fit_standardiser,
        get_device,
        set_seed,
        ensure_2d,
        normalise_weights,
        safe_log,
        to_tensor,
        ProgressBar,
    )

try:  # pragma: no cover - share the SDE/standardisation helpers of NPSE
    from .npse import build_sde as _npse_build_sde, standardise_dataset as _npse_standardise
except ImportError:  # pragma: no cover
    try:
        from npse import build_sde as _npse_build_sde, standardise_dataset as _npse_standardise
    except ImportError:
        _npse_build_sde = None
        _npse_standardise = None


__all__ = [
    "SNPSEConfig",
    "SampleDensity",
    "ProposalPrior",
    "CorrectedScore",
    "SNPSEA",
    "SNPSEB",
    "SNPSEC",
    "SNPSE_A",
    "SNPSE_B",
    "SNPSE_C",
    "run_snpse_a",
    "run_snpse_b",
    "run_snpse_c",
    "sir_resample",
    "PriorWrapper",
]

DEFAULT_EPS = 5e-4
_EPS = 1e-30
_LOG2PI = math.log(2.0 * math.pi)


# ===========================================================================
# small introspection / tensor helpers
# ===========================================================================
def _filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments that ``fn`` does not accept (unless it has **kwargs)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return dict(kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _call_with_fallbacks(
    fn: Callable, args: Sequence[Any], kwarg_candidates: Sequence[Dict[str, Any]]
) -> Any:
    """Call ``fn`` with the first set of keyword arguments that works."""
    last_error: Optional[BaseException] = None
    for kwargs in kwarg_candidates:
        try:
            return fn(*args, **_filter_kwargs(fn, kwargs))
        except TypeError as err:  # pragma: no cover - signature mismatch
            last_error = err
    if last_error is not None:
        raise last_error
    return fn(*args)  # pragma: no cover


def _dtype_from_name(name: Optional[str]) -> torch.dtype:
    if name is None:
        return torch.float32
    name = str(name).lower()
    return {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
    }.get(name, torch.float32)


def _net_call(net: nn.Module, theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate a score network accepting either ``(theta, x, t)`` or ``(theta, t)``."""
    try:
        out = net(theta, x, t)
    except TypeError:
        out = net(theta, t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _as_tensor(value: Any, device=None, dtype=None) -> torch.Tensor:
    try:
        return to_tensor(value, dtype=dtype, device=device)
    except Exception:  # pragma: no cover - very defensive
        if torch.is_tensor(value):
            return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device=device)
        import numpy as _np  # local import

        return torch.as_tensor(_np.asarray(value), dtype=dtype, device=device)


def _logsumexp(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim)


def _chunked(fn: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor, chunk: int = 4096):
    if theta.shape[0] <= chunk:
        return fn(theta)
    outs = [fn(theta[i : i + chunk]) for i in range(0, theta.shape[0], chunk)]
    return torch.cat(outs, dim=0)


# ===========================================================================
# prior adaptation
# ===========================================================================
class PriorWrapper:
    """Uniform access to ``sample``/``log_prob`` for torch / sbibm / callable priors.

    Supports
      * ``torch.distributions`` objects (``.sample`` / ``.log_prob``),
      * callables returning samples (e.g. the mackelab priors),
      * explicit ``sample_fn`` / ``log_prob_fn`` arguments.

    If no (analytic) log-density is available, one is estimated from prior
    samples using :class:`SampleDensity`, so that the importance weights
    ``p(theta)/ptilde^r(theta)`` remain computable for every task.
    """

    def __init__(
        self,
        prior: Any = None,
        sample_fn: Optional[Callable] = None,
        log_prob_fn: Optional[Callable] = None,
        theta_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        density_mode: str = "gmm",
        n_density_samples: int = 50_000,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.prior = prior
        self.theta_dim = theta_dim
        self.device = device
        self.dtype = dtype
        self._sample_fn = sample_fn
        self._log_prob_fn = log_prob_fn
        self._density: Optional[SampleDensity] = None

        if self._sample_fn is None:
            if callable(prior) and not hasattr(prior, "sample"):
                self._sample_fn = prior
            elif hasattr(prior, "sample"):
                self._sample_fn = prior.sample
            elif hasattr(prior, "get_prior") and callable(prior.get_prior):
                sub = prior.get_prior()
                if hasattr(sub, "sample"):
                    self._sample_fn = sub.sample
        if self._log_prob_fn is None:
            if hasattr(prior, "log_prob"):
                self._log_prob_fn = prior.log_prob
            elif hasattr(prior, "get_prior") and callable(prior.get_prior):
                sub = prior.get_prior()
                if hasattr(sub, "log_prob"):
                    self._log_prob_fn = sub.log_prob

        if self._sample_fn is None:
            raise ValueError(
                "PriorWrapper needs a prior with .sample, a callable prior, or an explicit sample_fn."
            )
        if self._log_prob_fn is None:
            samples = self.sample(n_density_samples, generator=generator, squeeze_dim=theta_dim)
            self._density = SampleDensity(
                samples, mode=density_mode, device=device, dtype=dtype
            )

    # -- sampling -----------------------------------------------------------
    def sample(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
        squeeze_dim: Optional[int] = None,
    ) -> torch.Tensor:
        sample_shape = (n,) if squeeze_dim is None else (n, squeeze_dim)
        out = None
        # try with an explicit on-device generator first (useful for reproducibility)
        try:
            out = self._sample_fn(sample_shape, generator=generator)  # type: ignore[call-arg]
        except TypeError:
            out = None
        if out is None:
            if generator is not None and not torch.is_tensor(self.prior):
                state = torch.get_rng_state()
                try:
                    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
                    out = self._sample_fn(sample_shape)
                finally:
                    torch.set_rng_state(state)
            else:
                out = self._sample_fn(sample_shape)
        out = _as_tensor(out, dtype=self.dtype)
        out = out.reshape(n, -1).to(device=self.device, dtype=self.dtype)
        return out

    # -- density ------------------------------------------------------------
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        theta = ensure_2d(_as_tensor(theta, device=self.device, dtype=self.dtype))
        if self._log_prob_fn is not None:
            try:
                return ensure_2d(self._log_prob_fn(theta), "log_prob").reshape(-1)
            except Exception:  # pragma: no cover - fall back to estimated density
                pass
        if self._density is None:
            samples = self.sample(20_000, squeeze_dim=self.theta_dim)
            self._density = SampleDensity(samples, device=self.device, dtype=self.dtype)
        return self._density.log_prob(theta)

    def analytic(self) -> bool:
        return self._log_prob_fn is not None


# ===========================================================================
# sample based densities
# ===========================================================================
class SampleDensity:
    """Density estimate built from samples.

    ``mode``:

    * ``"gmm"``    -- Gaussian mixture fitted with a few k-means iterations
      (fully covariance, shrunk towards the global covariance).
    * ``"kde"``    -- ``sklearn.neighbors.KernelDensity`` when available,
      otherwise falls back to ``"gmm"``.
    * ``"atomic"`` -- discrete/atomic (uniform-on-bounding-box) approximation,
      cf. the surrogate proposal priors of Appendix C.4.3.
    """

    def __init__(
        self,
        samples: torch.Tensor,
        mode: str = "gmm",
        n_components: int = 8,
        reg: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        max_fit_samples: int = 20_000,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        samples = ensure_2d(_as_tensor(samples, device=device, dtype=dtype))
        if samples.shape[0] < 2:
            raise ValueError("SampleDensity requires at least two samples.")
        self.device = samples.device
        self.dtype = samples.dtype
        self.n_samples, self.dim = samples.shape
        self.reg = float(reg)
        self.mode = str(mode).lower()

        fit = samples
        if samples.shape[0] > max_fit_samples:
            idx = torch.randperm(samples.shape[0], generator=generator)[:max_fit_samples]
            fit = samples[idx]
        self._fit_samples = fit

        # global statistics (also used as a fallback covariance)
        self.mean = fit.mean(dim=0)
        if fit.shape[0] > 1:
            centred = fit - self.mean
            cov = centred.t() @ centred / max(fit.shape[0] - 1, 1)
        else:  # pragma: no cover
            cov = torch.eye(self.dim, device=self.device, dtype=self.dtype)
        self.cov = cov + self.reg * torch.eye(self.dim, device=self.device, dtype=self.dtype)

        self._kde = None
        if self.mode == "kde":
            self._kde = self._fit_sklearn_kde(fit)
            if self._kde is None:
                self.mode = "gmm"
        if self.mode == "atomic":
            self.low = fit.min(dim=0).values
            self.high = fit.max(dim=0).values
            span = (self.high - self.low).clamp_min(1e-8)
            self.log_volume = float(torch.log(span).sum().item())
        elif self.mode == "gmm":
            self._fit_gmm(fit, n_components=n_components, generator=generator)
        elif self.mode != "kde":  # pragma: no cover - unknown mode
            raise ValueError(f"Unknown SampleDensity mode: {mode!r}")

    # -- fitting ------------------------------------------------------------
    def _fit_sklearn_kde(self, samples: torch.Tensor):
        try:  # pragma: no cover - optional dependency
            from sklearn.neighbors import KernelDensity  # type: ignore

            kde = KernelDensity(kernel="gaussian", bandwidth="scott")
            kde.fit(samples.detach().cpu().numpy())
            return kde
        except Exception:
            return None

    def _fit_gmm(self, samples: torch.Tensor, n_components: int, generator=None) -> None:
        n = samples.shape[0]
        k = int(max(1, min(n_components, n)))
        assign, centres = _kmeans(samples, k, generator=generator)
        means, covs, weights = [], [], []
        for j in range(k):
            mask = assign == j
            count = int(mask.sum().item())
            if count >= max(self.dim + 2, 3):
                pts = samples[mask]
                mu = pts.mean(dim=0)
                centred = pts - mu
                cov = centred.t() @ centred / max(count - 1, 1)
            else:  # too few points -> use the global statistics
                mu = centres[j] if count > 0 else self.mean
                cov = self.cov
            cov = cov + self.reg * torch.eye(self.dim, device=self.device, dtype=self.dtype)
            means.append(mu)
            covs.append(cov)
            weights.append(max(count, 1) / n)
        self.gmm_means = torch.stack(means, dim=0)
        self.gmm_covs = torch.stack(covs, dim=0)
        w = torch.tensor(weights, device=self.device, dtype=self.dtype)
        self.gmm_log_weights = torch.log(w / w.sum())

    # -- density ------------------------------------------------------------
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        theta = ensure_2d(_as_tensor(theta, device=self.device, dtype=self.dtype))
        if self.mode == "kde" and self._kde is not None:  # pragma: no cover
            import numpy as _np

            lp = self._kde.score_samples(theta.detach().cpu().numpy())
            return torch.as_tensor(_np.asarray(lp), dtype=self.dtype, device=self.device).reshape(-1)
        if self.mode == "atomic":
            inside = ((theta >= self.low) & (theta <= self.high)).all(dim=-1)
            base = torch.full(
                (theta.shape[0],), -self.log_volume, device=self.device, dtype=self.dtype
            )
            # small penalty outside the empirical hypercube instead of -inf
            return torch.where(inside, base, base - 10.0)
        return _chunked(self._gmm_log_prob, theta)

    def _gmm_log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        diff = theta[:, None, :] - self.gmm_means[None, :, :]  # (n, k, d)
        tril = torch.linalg.cholesky(self.gmm_covs)  # (k, d, d)
        sol = torch.linalg.solve_triangular(tril[None], diff.transpose(1, 2), upper=False)
        quad = (sol**2).sum(dim=1)  # (n, k)
        log_det = 2.0 * torch.log(torch.diagonal(tril, dim1=-2, dim2=-1)).sum(dim=-1)  # (k,)
        log_comp = (
            self.gmm_log_weights[None, :]
            - 0.5 * (quad + log_det[None, :] + self.dim * _LOG2PI)
        )
        return _logsumexp(log_comp, dim=-1)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        idx = torch.randint(0, self._fit_samples.shape[0], (n,), generator=generator)
        return self._fit_samples[idx].to(device=self.device, dtype=self.dtype)


def _kmeans(
    samples: torch.Tensor, k: int, n_iter: int = 25, generator: Optional[torch.Generator] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal k-means (k-means++ style init) used to fit the Gaussian mixture."""
    n = samples.shape[0]
    k = int(max(1, min(k, n)))
    perm = torch.randperm(n, generator=generator)[:k]
    centres = samples[perm].clone()
    assign = torch.zeros(n, dtype=torch.long, device=samples.device)
    for _ in range(int(n_iter)):
        d2 = torch.cdist(samples, centres) ** 2
        assign = d2.argmin(dim=1)
        for j in range(k):
            mask = assign == j
            if bool(mask.any()):
                centres[j] = samples[mask].mean(dim=0)
    return assign, centres


# ===========================================================================
# components of the proposal prior ptilde^r
# ===========================================================================
class _ExactComponent:
    """Analytic (known) density component, e.g. the prior ``p_psi^0 = p(theta)``."""

    def __init__(self, log_prob_fn: Callable, sample_fn: Optional[Callable] = None) -> None:
        self.log_prob_fn = log_prob_fn
        self.sample_fn = sample_fn

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return ensure_2d(self.log_prob_fn(theta), "log_prob").reshape(-1)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.sample_fn is None:
            raise RuntimeError("Exact component has no sampler; use sample-based component.")
        return ensure_2d(self.sample_fn(n, generator), "sample")


class _SampleComponent:
    """Approximate density component built from previous-round posterior samples."""

    def __init__(
        self,
        samples: torch.Tensor,
        mode: str = "gmm",
        n_components: int = 8,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.samples = ensure_2d(_as_tensor(samples, device=device, dtype=dtype))
        self.density = SampleDensity(
            self.samples,
            mode=mode,
            n_components=n_components,
            device=self.samples.device,
            dtype=self.samples.dtype,
            generator=generator,
        )

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return self.density.log_prob(theta)

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.density.sample(n, generator=generator)

    # a cheap bootstrap-free alternative: draw the stored samples directly
    def sample_empirical(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        idx = torch.randint(0, self.samples.shape[0], (n,), generator=generator)
        return self.samples[idx]


class _NetworkComponent:
    """Density component evaluated with the probability flow ODE + change of variables.

    This is the *exact* (up to numerical error) density of a learned proposal
    posterior, ``ptilde_psi^s(theta | x_obs)``, obtained by substituting the
    trained score network into the probability flow ODE (4) and evaluating the
    instantaneous change-of-variables formula (5) (Appendix C.2.3).
    """

    def __init__(
        self,
        network: nn.Module,
        sde: SDE,
        x_obs: torch.Tensor,
        theta_std: Optional[Standardiser] = None,
        x_std: Optional[Standardiser] = None,
        sampler_config: Optional[SamplerConfig] = None,
        theta_dim: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.network = network
        self.sde = sde
        self.x_obs = ensure_2d(_as_tensor(x_obs, device=device, dtype=dtype))
        self.theta_std = theta_std
        self.x_std = x_std
        self.sampler_config = sampler_config
        self.theta_dim = theta_dim
        self.device = device
        self.dtype = dtype

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        return _network_log_prob(
            self.sde,
            self.network,
            self.x_obs,
            theta,
            config=self.sampler_config,
            theta_dim=self.theta_dim,
            theta_std=self.theta_std,
            x_std=self.x_std,
            device=self.device,
            dtype=self.dtype,
        )

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        out = _network_sample(
            self.sde,
            self.network,
            self.x_obs,
            n,
            config=self.sampler_config,
            theta_dim=self.theta_dim,
            theta_std=self.theta_std,
            x_std=self.x_std,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out


class ProposalPrior:
    """Mixture of posterior estimates ``ptilde^r(theta)`` (eq. 10 / eq. 88).

    Components are added one per round: the prior (exact) plus the posterior
    estimates obtained in previous rounds (either evaluated exactly through the
    probability-flow ODE or approximated from their samples).
    """

    def __init__(
        self,
        mode: str = "gmm",
        n_components: int = 8,
        mixture: str = "average",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.components: List[Any] = []
        self.mixture = mixture
        self.mode = mode
        self.n_components = int(n_components)
        self.device = device
        self.dtype = dtype

    # -- construction -------------------------------------------------------
    def add_exact(self, log_prob_fn: Callable, sample_fn: Optional[Callable] = None) -> None:
        self.components.append(_ExactComponent(log_prob_fn, sample_fn))

    def add_samples(self, samples: torch.Tensor, generator: Optional[torch.Generator] = None) -> None:
        self.components.append(
            _SampleComponent(
                samples,
                mode=self.mode,
                n_components=self.n_components,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
        )

    def add_network(
        self,
        network: nn.Module,
        sde: SDE,
        x_obs: torch.Tensor,
        theta_std: Optional[Standardiser] = None,
        x_std: Optional[Standardiser] = None,
        sampler_config: Optional[SamplerConfig] = None,
        theta_dim: Optional[int] = None,
    ) -> None:
        self.components.append(
            _NetworkComponent(
                network,
                sde,
                x_obs,
                theta_std=theta_std,
                x_std=x_std,
                sampler_config=sampler_config,
                theta_dim=theta_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )

    def add_component(self, component: Any) -> None:
        self.components.append(component)

    # -- mixture ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.components)

    @property
    def num_components(self) -> int:
        return len(self.components)

    def component_weights(self) -> torch.Tensor:
        r = len(self.components)
        if r == 0:
            raise RuntimeError("ProposalPrior has no components.")
        if self.mixture == "latest":
            w = torch.zeros(r, device=self.device, dtype=self.dtype)
            w[-1] = 1.0
            return w
        return torch.full((r,), 1.0 / r, device=self.device, dtype=self.dtype)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """``log ptilde^r(theta)`` = log of the mean of the component densities."""
        theta = ensure_2d(_as_tensor(theta, device=self.device, dtype=self.dtype))
        if not self.components:
            raise RuntimeError("ProposalPrior has no components.")
        if len(self.components) == 1:
            return self.components[0].log_prob(theta)
        log_w = torch.log(self.component_weights())
        logps = torch.stack([c.log_prob(theta) for c in self.components], dim=-1)
        return _logsumexp(logps + log_w[None, :], dim=-1)

    def log_ratio(self, theta: torch.Tensor, log_prob_fn: Callable) -> torch.Tensor:
        """``log p(theta) - log ptilde^r(theta)`` (argument of eq. 80/eq. 99)."""
        theta = ensure_2d(_as_tensor(theta, device=self.device, dtype=self.dtype))
        numerator = ensure_2d(log_prob_fn(theta), "log_prob").reshape(-1)
        denominator = self.log_prob(theta)
        return numerator - denominator

    def importance_weights(
        self,
        theta: torch.Tensor,
        log_prob_fn: Callable,
        clip: Optional[float] = None,
        normalise: bool = True,
    ) -> torch.Tensor:
        """Multiplicative weights ``p(theta)/ptilde^r(theta)`` used in SNPSE-A/B."""
        log_ratio = self.log_ratio(theta, log_prob_fn)
        if clip is not None:
            log_ratio = torch.clamp(log_ratio, -float(clip), float(clip))
        weights = torch.exp(log_ratio - log_ratio.max())
        if normalise:
            weights = weights / weights.mean().clamp_min(_EPS)
        return weights

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample from the mixture ``ptilde^r`` (used e.g. to train prior-score nets)."""
        if not self.components:
            raise RuntimeError("ProposalPrior has no components.")
        probs = self.component_weights()
        draws = torch.multinomial(probs, n, replacement=True, generator=generator)
        out = torch.empty((n, 0), device=self.device, dtype=self.dtype)
        pieces: List[torch.Tensor] = []
        for j, comp in enumerate(self.components):
            count = int((draws == j).sum().item())
            if count == 0:
                continue
            piece = comp.sample(count, generator=generator)
            piece = ensure_2d(_as_tensor(piece, device=self.device, dtype=self.dtype))
            if hasattr(comp, "sample_empirical") and isinstance(comp, _SampleComponent):
                piece = comp.sample_empirical(count, generator=generator)
            pieces.append(piece)
        if not pieces:  # pragma: no cover
            return out
        return torch.cat(pieces, dim=0)

    def to(self, device) -> "ProposalPrior":  # pragma: no cover - convenience
        self.device = device
        return self

    def summary(self) -> Dict[str, Any]:
        counts = []
        for comp in self.components:
            if isinstance(comp, _SampleComponent):
                counts.append(int(comp.samples.shape[0]))
            else:
                counts.append(None)
        return {
            "num_components": len(self.components),
            "mixture": self.mixture,
            "density_mode": self.mode,
            "component_sample_counts": counts,
        }


# ===========================================================================
# sampling / density helpers with standardisation fallbacks
# ===========================================================================
def _std_kwargs(theta_std: Optional[Standardiser], x_std: Optional[Standardiser]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if theta_std is not None:
        kwargs["theta_shift"] = theta_std.shift
        kwargs["theta_scale"] = theta_std.scale
    if x_std is not None:
        kwargs["x_shift"] = x_std.shift
        kwargs["x_scale"] = x_std.scale
    return kwargs


def _log_abs_det(theta_std: Optional[Standardiser]) -> Optional[torch.Tensor]:
    if theta_std is None:
        return None
    scale = theta_std.scale
    if torch.is_tensor(scale):
        return torch.log(scale.abs().clamp_min(_EPS)).sum()
    return torch.tensor(float(math.log(abs(float(scale)))), dtype=torch.float32) * scale.numel()


def _network_sample(
    sde: SDE,
    network: nn.Module,
    x_obs: torch.Tensor,
    num_samples: int,
    config: Optional[SamplerConfig] = None,
    theta_dim: Optional[int] = None,
    theta_std: Optional[Standardiser] = None,
    x_std: Optional[Standardiser] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    with_log_prob: bool = False,
) -> Any:
    """Sample from a trained score network, handling standardisation transparently."""
    base = dict(
        config=config,
        with_log_prob=with_log_prob,
        theta_dim=theta_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    std = _std_kwargs(theta_std, x_std)
    try:
        return _call_with_fallbacks(
            sample_posterior,
            [sde, network, x_obs, num_samples],
            [{**base, **std}, base],
        )
    except TypeError:  # pragma: no cover - older sampler signature
        pass
    # manual standardisation
    x_in = x_obs if x_std is None else x_std.to_std(_as_tensor(x_obs, device=device, dtype=dtype))
    minimal = {k: v for k, v in base.items() if k in ("config", "theta_dim")}
    out = _call_with_fallbacks(sample_posterior, [sde, network, x_in, num_samples], [minimal])
    if with_log_prob:
        theta_out, log_prob = out
        theta_out = _as_tensor(theta_out)
        if theta_std is not None:
            log_prob = log_prob - _log_abs_det(theta_std)
            theta_out = theta_std.from_std(theta_out)
        return theta_out, log_prob
    theta_out = _as_tensor(out)
    if theta_std is not None:
        theta_out = theta_std.from_std(theta_out)
    return theta_out


def _network_log_prob(
    sde: SDE,
    network: nn.Module,
    x_obs: torch.Tensor,
    theta: torch.Tensor,
    config: Optional[SamplerConfig] = None,
    theta_dim: Optional[int] = None,
    theta_std: Optional[Standardiser] = None,
    x_std: Optional[Standardiser] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``log p_psi(theta | x_obs)`` via the probability flow ODE (eq. 4-5)."""
    theta = ensure_2d(_as_tensor(theta, device=device, dtype=dtype))
    base = dict(
        config=config,
        theta_dim=theta_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    std = _std_kwargs(theta_std, x_std)
    try:
        out = _call_with_fallbacks(
            estimate_log_prob, [sde, network, x_obs, theta], [{**base, **std}, base]
        )
        return _as_tensor(out).reshape(-1)
    except TypeError:  # pragma: no cover
        pass
    x_in = x_obs if x_std is None else x_std.to_std(_as_tensor(x_obs, device=device, dtype=dtype))
    theta_in = theta if theta_std is None else theta_std.to_std(theta)
    minimal = {k: v for k, v in base.items() if k in ("config", "theta_dim")}
    out = _call_with_fallbacks(estimate_log_prob, [sde, network, x_in, theta_in], [minimal])
    out = _as_tensor(out).reshape(-1)
    if theta_std is not None:
        out = out - _log_abs_det(theta_std)
    return out


def sir_resample(
    samples: torch.Tensor,
    log_weights: torch.Tensor,
    num_samples: int,
    generator: Optional[torch.Generator] = None,
    replace: bool = False,
    clip: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sampling-importance-resampling (eq. 80/81 of Appendix C.2.1).

    Returns the resampled parameters together with the log-weights of the
    selected draws (normalised so that ``logsumexp(log_weights) = 0``).
    """
    samples = ensure_2d(_as_tensor(samples))
    log_weights = _as_tensor(log_weights, dtype=samples.dtype, device=samples.device).reshape(-1)
    if clip is not None:
        log_weights = torch.clamp(log_weights, -float(clip), float(clip))
    log_weights = log_weights - log_weights.max()
    log_norm = torch.logsumexp(log_weights, dim=0)
    probs = torch.exp(log_weights - log_norm)
    num_samples = int(min(num_samples, samples.shape[0])) if not replace else int(num_samples)
    idx = torch.multinomial(probs, num_samples, replacement=bool(replace), generator=generator)
    return samples[idx], (log_weights[idx] - log_norm)


# ===========================================================================
# corrected score network used by SNPSE-C (eq. 103 / eq. 116)
# ===========================================================================
class _ThetaOnlyScore(nn.Module):
    """Wrap a ``(theta, t)``-style score network as a ``(theta, x, t)`` one.

    The perturbed prior / proposal-prior score networks of SNPSE-C (and the
    prior-score network of Appendix B.2, Algorithm 2) are unconditional in
    ``x``; ``x`` is ignored here so that the same DSM losses can be reused.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, theta_t: torch.Tensor, x: Optional[torch.Tensor], t: torch.Tensor) -> torch.Tensor:
        theta_t = ensure_2d(theta_t)
        dummy = torch.zeros(theta_t.shape[0], 1, dtype=theta_t.dtype, device=theta_t.device)
        return _net_call(self.net, theta_t, dummy, t)


class CorrectedScore(nn.Module):
    """``tilde s_psi^r(theta_t, x, t) = s_psi + s_prop - grad log p_t(theta_t)`` (eq. 103).

    ``proposal_score_fn``/``prior_score_fn`` accept ``(theta, t)`` (or
    ``(theta, x, t)``).  Passing ``None`` reproduces the round-1 case of
    Algorithm 5, where ``tilde s_psi^1 = s_psi``.
    """

    def __init__(
        self,
        base: nn.Module,
        proposal_score_fn: Optional[Callable] = None,
        prior_score_fn: Optional[Callable] = None,
        proposal_sign: float = 1.0,
        prior_sign: float = -1.0,
    ) -> None:
        super().__init__()
        self.base = base
        self.proposal_score_fn = proposal_score_fn
        self.prior_score_fn = prior_score_fn
        self.proposal_sign = float(proposal_sign)
        self.prior_sign = float(prior_sign)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = _net_call(self.base, theta_t, x, t)
        if self.proposal_score_fn is not None:
            out = out + self.proposal_sign * self.proposal_score_fn(theta_t, t)
        if self.prior_score_fn is not None:
            out = out + self.prior_sign * self.prior_score_fn(theta_t, t)
        return out


# ===========================================================================
# configuration
# ===========================================================================
@dataclass
class SNPSEConfig:
    """Hyper-parameters shared by SNPSE-A/B/C (defaults follow Section 5.1, E.3)."""

    # --- forward SDE (Appendix E.3.1) ---
    sde: str = "ve"
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    beta_min: float = 0.1
    beta_max: float = 11.0
    t_final: float = 1.0

    # --- score network (Appendix E.3.2) ---
    hidden_dim: int = 256
    n_layers: int = 3
    time_emb_dim: int = 64
    theta_emb_dim: Optional[int] = None
    x_emb_dim: Optional[int] = None
    parameterisation: str = "score"
    activation: str = "silu"

    # --- training (Section 5.1) ---
    lr: float = 1e-4
    max_iters: int = 3000
    batch_size: Optional[int] = None
    budget: int = 10_000
    val_fraction: float = 0.15
    patience: int = 1000
    min_delta: float = 0.0
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    log_every: int = 100

    # --- sequential scheme (Section 3.2) ---
    num_rounds: int = 10
    initial_budget: Optional[int] = None
    simulations_per_round: Optional[int] = None
    resample_size: Optional[int] = None
    proposal_density: str = "gmm"
    proposal_mixture: str = "average"
    proposal_n_components: int = 8
    reuse_standardisers: bool = True
    warm_start: bool = False
    normalise_weights: bool = True
    weight_clip: float = 10.0
    resample_with_replacement: bool = False
    use_analytic_prior_score: bool = True
    proposal_score_iters: Optional[int] = None
    n_proposal_score_samples: int = 20_000
    sample_with: str = "base"  # SNPSE-C: "base" | "corrected"

    # --- standardisation / sampler (Appendix E.3.2-E.3.3) ---
    standardise: bool = True
    sampler_method: str = "rk45"
    sampler_atol: float = 1e-5
    sampler_rtol: float = 1e-5
    sampler_n_steps: int = 1000
    trace_estimator: str = "auto"

    # --- miscellaneous ---
    seed: Optional[int] = None
    device: Optional[str] = None
    dtype_name: str = "float32"
    verbose: bool = True
    verbose_rounds: bool = True

    # -- derived quantities -------------------------------------------------
    def resolved_round_budgets(self) -> List[int]:
        """Simulations per round (paper: ``M = N / R``; pyloric: 30000 + 20000/round)."""
        rounds = max(1, int(self.num_rounds))
        if self.initial_budget is not None or self.simulations_per_round is not None:
            initial = int(self.initial_budget if self.initial_budget is not None else self.simulations_per_round)
            per_round = int(
                self.simulations_per_round
                if self.simulations_per_round is not None
                else max(1, int(self.budget) // rounds)
            )
            return [initial] + [per_round] * (rounds - 1)
        per_round = max(1, int(self.budget) // rounds)
        return [per_round] * rounds

    def total_simulations(self) -> int:
        return int(sum(self.resolved_round_budgets()))

    def resolved_resample_size(self, round_budget: int) -> int:
        """``M' >= M`` in step (iii) of Appendix C.2.1 (default ``M' = M``)."""
        if self.resample_size is None:
            return int(round_budget)
        return int(max(round_budget, self.resample_size))

    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return int(self.batch_size)
        try:
            return int(select_batch_size(self.total_simulations(), sequential=True))
        except Exception:  # pragma: no cover
            return 50

    def resolved_dtype(self) -> torch.dtype:
        return _dtype_from_name(self.dtype_name)

    def train_config(self, max_iters: Optional[int] = None) -> TrainConfig:
        kwargs = dict(
            lr=self.lr,
            max_iters=int(max_iters if max_iters is not None else self.max_iters),
            batch_size=self.resolved_batch_size(),
            val_fraction=self.val_fraction,
            patience=self.patience,
            min_delta=self.min_delta,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            log_every=self.log_every,
            verbose=False,
            seed=self.seed,
        )
        return TrainConfig(**_filter_kwargs(TrainConfig, kwargs))

    def sampler_config(self) -> SamplerConfig:
        kwargs = dict(
            method=self.sampler_method,
            atol=self.sampler_atol,
            rtol=self.sampler_rtol,
            n_steps=self.sampler_n_steps,
            t_max=self.t_final,
            trace_estimator=self.trace_estimator,
            seed=self.seed,
        )
        return SamplerConfig(**_filter_kwargs(SamplerConfig, kwargs))

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["round_budgets"] = self.resolved_round_budgets()
        return out

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None) -> "SNPSEConfig":
        if not values:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in values.items() if k in known})


# ===========================================================================
# shared sequential driver
# ===========================================================================
class _SequentialVariantBase:
    """Common machinery for the three alternative sequential approaches."""

    objective: str = "a"
    needs_correction: bool = False
    needs_importance_weights: bool = False
    variant_name: str = "SNPSE"

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        prior: Any = None,
        simulator: Optional[Callable] = None,
        config: Optional[SNPSEConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        network: Optional[nn.Module] = None,
        prior_sample_fn: Optional[Callable] = None,
        prior_log_prob_fn: Optional[Callable] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.config = config or SNPSEConfig()
        self.device = get_device(device if device is not None else self.config.device)
        self.dtype = dtype or self.config.resolved_dtype()
        self.simulator = simulator
        self.network = network
        self._prior_kwargs = dict(
            prior=prior,
            sample_fn=prior_sample_fn,
            log_prob_fn=prior_log_prob_fn,
        )
        self.prior: Optional[PriorWrapper] = None
        self.sde: Optional[SDE] = None

        # standardisers (Appendix E.3.2)
        identity = Standardiser.identity(self.theta_dim, dtype=self.dtype).to(self.device)
        self.theta_std: Standardiser = identity
        self.x_std: Standardiser = Standardiser.identity(self.x_dim, dtype=self.dtype).to(self.device)

        # sequential state
        self.dataset_theta: Optional[torch.Tensor] = None
        self.dataset_x: Optional[torch.Tensor] = None
        self.round_budgets: List[int] = self.config.resolved_round_budgets()
        self.history: List[Any] = []
        self.proposal_prior: Optional[ProposalPrior] = None
        self.posterior_samples_by_round: Dict[int, torch.Tensor] = {}
        self.info: Dict[str, Any] = {}

    # -- setup helpers ------------------------------------------------------
    def _get_prior(self, generator: Optional[torch.Generator] = None) -> PriorWrapper:
        if self.prior is None:
            self.prior = PriorWrapper(
                theta_dim=self.theta_dim,
                device=self.device,
                dtype=self.dtype,
                density_mode=self.config.proposal_density,
                generator=generator,
                **self._prior_kwargs,
            )
        return self.prior

    def _build_sde(self, data: Optional[torch.Tensor] = None) -> SDE:
        """Construct the forward SDE (Appendix E.3.1).

        For the VE SDE the ``sigma_max`` heuristic of Song & Ermon (Technique 1)
        is computed from the *first-round* training data only, as required by the
        Addendum for sequential methods.
        """
        cfg = self.config
        name = str(cfg.sde).lower()
        dim = self.theta_dim
        kwargs: Dict[str, Any] = dict(beta_min=cfg.beta_min, beta_max=cfg.beta_max, dim=dim)
        if name in ("ve", "variance_exploding", "variance-exploding"):
            kwargs["sigma_min"] = cfg.sigma_min
            kwargs["sigma_max"] = cfg.sigma_max
            if kwargs["sigma_max"] is None and data is not None:
                kwargs["data"] = data
            return get_sde("ve", **_filter_kwargs(get_sde, kwargs))
        return get_sde("vp", **_filter_kwargs(get_sde, kwargs))

    def _make_network(self, generator: Optional[torch.Generator] = None) -> nn.Module:
        if self.network is not None:
            return self.network
        kwargs = dict(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            time_emb_dim=self.config.time_emb_dim,
            theta_emb_dim=self.config.theta_emb_dim,
            x_emb_dim=self.config.x_emb_dim,
            parameterisation=self.config.parameterisation,
            activation=self.config.activation,
        )
        return get_score_network(**_filter_kwargs(get_score_network, kwargs)).to(
            device=self.device, dtype=self.dtype
        )

    def _make_theta_network(self, generator: Optional[torch.Generator] = None) -> nn.Module:
        """Unconditional ``(theta, t)`` score network (prior/proposal prior scores)."""
        kwargs = dict(
            theta_dim=self.theta_dim,
            x_dim=1,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            time_emb_dim=self.config.time_emb_dim,
            parameterisation="score",
            activation=self.config.activation,
        )
        return get_score_network(**_filter_kwargs(get_score_network, kwargs)).to(
            device=self.device, dtype=self.dtype
        )

    @staticmethod
    def _resolve_generator(
        generator: Optional[torch.Generator], seed: Optional[int]
    ) -> torch.Generator:
        if generator is not None:
            return generator
        set_seed(seed)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()
        return gen

    # -- simulator / prior --------------------------------------------------
    def _simulate(self, theta: torch.Tensor) -> torch.Tensor:
        if self.simulator is None:
            raise ValueError("A simulator p(x|theta) is required.")
        theta = theta.to(device="cpu")
        try:
            out = self.simulator(theta)
        except Exception:  # pragma: no cover - numpy-style simulators
            out = self.simulator(theta.detach().cpu().numpy())
        out = _as_tensor(out, dtype=self.dtype)
        return out.reshape(theta.shape[0], -1).to(device=self.device, dtype=self.dtype)

    # -- standardisation ----------------------------------------------------
    def _fit_standardisers(self, theta: torch.Tensor, x: torch.Tensor) -> None:
        if not self.config.standardise:
            self.theta_std = Standardiser.identity(self.theta_dim, dtype=self.dtype).to(self.device)
            self.x_std = Standardiser.identity(self.x_dim, dtype=self.dtype).to(self.device)
            return
        try:
            theta_std = fit_standardiser(theta)
            x_std = fit_standardiser(x)
        except Exception:  # pragma: no cover - fallback to direct construction
            theta_std = Standardiser.from_data(theta)
            x_std = Standardiser.from_data(x)
        self.theta_std = theta_std.to(self.device)
        self.x_std = x_std.to(self.device)

    def _standardise(self, theta: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.config.standardise:
            return theta, x
        return self.theta_std.to_std(theta), self.x_std.to_std(x)

    # -- training -----------------------------------------------------------
    def _objective_function(self, name: str) -> Optional[Callable]:
        name = str(name).lower()
        table = {
            "a": snpse_a_loss,
            "snpse-a": snpse_a_loss,
            "b": snpse_b_loss,
            "snpse-b": snpse_b_loss,
            "npse": npse_loss,
            "tsnpse": tsnpse_loss,
        }
        return table.get(name)

    def _dsm(
        self,
        score_fn: Callable,
        sde: SDE,
        theta0: torch.Tensor,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        objective: Optional[str] = None,
    ) -> torch.Tensor:
        """Call the loss of eq. (79)/(99)/(102), falling back to the generic DSM loss."""
        objective = objective or self.objective
        preferred = self._objective_function(objective)
        for fn in [preferred, dsm_loss]:
            if fn is None:
                continue
            kwargs = dict(weight=weight, generator=generator)
            try:
                return fn(score_fn, sde, theta0, x, **_filter_kwargs(fn, kwargs))
            except TypeError as err:  # pragma: no cover - signature mismatch
                last_error = err
        raise last_error  # pragma: no cover

    def _loss_closure(
        self,
        network: nn.Module,
        sde: SDE,
        generator: torch.Generator,
        objective: Optional[str] = None,
    ) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
        def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            theta0 = ensure_2d(_as_tensor(batch["theta"], device=self.device, dtype=self.dtype))
            x = ensure_2d(_as_tensor(batch["x"], device=self.device, dtype=self.dtype))
            weight = batch.get("weight", None)
            if weight is not None:
                weight = _as_tensor(weight, device=self.device, dtype=self.dtype).reshape(-1)
            return self._dsm(
                network,
                sde,
                theta0,
                x,
                weight=weight,
                generator=generator,
                objective=objective,
            )

        return loss_fn

    def _train(
        self,
        network: nn.Module,
        sde: SDE,
        theta: torch.Tensor,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        objective: Optional[str] = None,
        max_iters: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Any:
        theta_std, x_std = self._standardise(theta, x)
        cfg = self.config.train_config(max_iters=max_iters)
        if seed is not None:
            try:
                cfg = replace(cfg, seed=seed)
            except Exception:  # pragma: no cover
                pass
        loss_fn = self._loss_closure(network, sde, generator, objective=objective)
        kwargs = dict(config=cfg, device=self.device, generator=generator)
        try:
            return train_network(
                network,
                loss_fn,
                theta_std,
                x_std,
                weights=weights,
                **_filter_kwargs(train_network, kwargs),
            )
        except TypeError:  # pragma: no cover - older trainer signature
            minimal = {"config": cfg, "device": self.device}
            return train_network(
                network,
                loss_fn,
                theta_std,
                x_std,
                weights=weights,
                **_filter_kwargs(train_network, minimal),
            )

    def _train_prior_score_net(
        self,
        samples: torch.Tensor,
        sde: SDE,
        generator: torch.Generator,
        max_iters: Optional[int] = None,
        tag: str = "prior",
    ) -> nn.Module:
        """Learn ``s(theta_t, t) ~ grad log p_t(theta_t)`` by eq. (64)/(123) DSM.

        Samples must already be standardised, matching the space in which the
        correction is applied inside the SNPSE-C loss.
        """
        net = self._make_theta_network(generator)
        n = samples.shape[0]
        zeros = torch.zeros(n, 1, device=self.device, dtype=self.dtype)
        iters = max_iters if max_iters is not None else self.config.proposal_score_iters
        objective = "npse"  # plain DSM over the (standardised) samples

        def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            theta0 = ensure_2d(_as_tensor(batch["theta"], device=self.device, dtype=self.dtype))
            x = ensure_2d(_as_tensor(batch["x"], device=self.device, dtype=self.dtype))
            return self._dsm(net, sde, theta0, x, generator=generator, objective=objective)

        cfg = self.config.train_config(max_iters=iters)
        kwargs = dict(config=cfg, device=self.device, generator=generator)
        try:
            return train_network(
                net, loss_fn, samples, zeros, **_filter_kwargs(train_network, kwargs)
            )
        except TypeError:  # pragma: no cover
            minimal = {"config": cfg, "device": self.device}
            return train_network(net, loss_fn, samples, zeros, **_filter_kwargs(train_network, minimal))

    # -- analytic perturbed prior score (Appendix B.2.1) --------------------
    def _analytic_prior_score(self, theta_t: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        """``grad theta log p_t(theta_t)`` for Gaussian / uniform priors, if known."""
        if not self.config.use_analytic_prior_score:
            return None
        prior = self._prior_kwargs.get("prior")
        theta_t = ensure_2d(theta_t)
        t = t.reshape(-1)
        # t may be passed as a scalar tensor or per-sample tensor
        if t.numel() == 1:
            t = t.expand(theta_t.shape[0])
        t = t.to(device=theta_t.device, dtype=theta_t.dtype)

        # Gaussian -- exact for both VE and VP forward SDEs.
        mean = None
        scale = None
        if prior is not None:
            mean = getattr(prior, "loc", None) if hasattr(prior, "loc") else None
            scale = getattr(prior, "scale", None) if hasattr(prior, "scale") else None
            if mean is None and hasattr(prior, "mean"):
                try:
                    m = prior.mean
                    mean = m if torch.is_tensor(m) else None
                except Exception:  # pragma: no cover
                    mean = None
            if hasattr(prior, "covariance_matrix") and mean is not None:
                try:
                    cov = prior.covariance_matrix
                    var = torch.diagonal(cov) if cov.dim() == 2 else cov
                    return self._gaussian_perturbed_score(theta_t, t, mean, var)
                except Exception:  # pragma: no cover
                    pass
            if mean is not None and scale is not None:
                var = (scale * scale) if torch.is_tensor(scale) else torch.as_tensor(scale) ** 2
                var = var.to(device=theta_t.device, dtype=theta_t.dtype)
                return self._gaussian_perturbed_score(theta_t, t, mean.to(theta_t.device), var)

        # Uniform (continuous) prior -- eq. (58) of Appendix B.2.1.
        if prior is not None and (hasattr(prior, "low") and hasattr(prior, "high")):
            low = torch.as_tensor(prior.low, device=theta_t.device, dtype=theta_t.dtype).reshape(-1)
            high = torch.as_tensor(prior.high, device=theta_t.device, dtype=theta_t.dtype).reshape(-1)
            return self._uniform_perturbed_score(theta_t, t, low, high)
        return None

    def _gaussian_perturbed_score(
        self,
        theta_t: torch.Tensor,
        t: torch.Tensor,
        mean: torch.Tensor,
        var: torch.Tensor,
    ) -> torch.Tensor:
        """VE: theta_t ~ N(mu, (diag(var) + sigma_t^2 I));  VP: N(mu, diag(var))."""
        mean = torch.as_tensor(mean, device=theta_t.device, dtype=theta_t.dtype).reshape(-1)
        var = torch.as_tensor(var, device=theta_t.device, dtype=theta_t.dtype).reshape(-1)
        if getattr(self.sde, "name", "") == "ve":
            sigma = self.sde.marginal_std(t.reshape(-1, 1)).reshape(-1)
            total = var + sigma**2
        else:
            total = var.expand(theta_t.shape[0]).clone()
        return -(theta_t - mean) / total.clamp_min(_EPS).unsqueeze(-1)

    def _uniform_perturbed_score(
        self, theta_t: torch.Tensor, t: torch.Tensor, low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        """Score of a uniform prior convolved with the forward Gaussian kernel.

        For the VP SDE the marginal kernel variance is ``1 - exp(-int beta)`` and
        for VE it is ``sigma_t^2``; the convolution of a product of uniform
        densities with a Gaussian is a product of Gaussian-CDF differences,
        whose log-derivative is (up to the constant prefactor)
        ``(phi(a) - phi(b)) / (Phi(b) - Phi(a)) / sigma_t`` with
        ``a = (low - theta_t)/sigma_t``, ``b = (high - theta_t)/sigma_t``.
        """
        if getattr(self.sde, "name", "") == "ve":
            sigma = self.sde.marginal_std(t.reshape(-1, 1)).reshape(-1, 1)
        else:
            try:
                sigma = self.sde.marginal_std(t.reshape(-1, 1)).reshape(-1, 1)
            except Exception:  # pragma: no cover
                sigma = torch.ones(theta_t.shape[0], 1, device=theta_t.device, dtype=theta_t.dtype)
        sigma = sigma.clamp_min(1e-6)
        a = (low.reshape(1, -1) - theta_t) / sigma
        b = (high.reshape(1, -1) - theta_t) / sigma
        normal = torch.distributions.Normal(
            torch.zeros((), device=theta_t.device, dtype=theta_t.dtype),
            torch.ones((), device=theta_t.device, dtype=theta_t.dtype),
        )
        pdf_a, pdf_b = torch.exp(normal.log_prob(a)), torch.exp(normal.log_prob(b))
        cdf_a = torch.clamp(normal.cdf(a), min=1e-12)
        cdf_b = torch.clamp(normal.cdf(b), min=1e-12)
        return (pdf_b - pdf_a) / (cdf_b - cdf_a) / sigma

    def _prior_score_fn(self, sde: SDE, generator: torch.Generator) -> Callable:
        """Return ``f(theta_t, t) ~ grad log p_t(theta_t)``, analytic or learned."""
        analytic_ok = self._analytic_prior_score(
            torch.zeros(2, self.theta_dim, device=self.device, dtype=self.dtype),
            torch.zeros(2, device=self.device, dtype=self.dtype),
        )
        if analytic_ok is not None:
            def prior_score_fn(theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                out = self._analytic_prior_score(theta_t, t)
                if out is None:  # pragma: no cover - should not happen
                    raise RuntimeError("Analytic prior score unavailable mid-training.")
                return out

            self.info["prior_score"] = "analytic"
            return prior_score_fn

        # Algorithm 2 of Appendix B.2: learn the perturbed prior score on samples
        prior = self._get_prior(generator)
        n = int(self.config.n_proposal_score_samples)
        samples = prior.sample(n, generator=generator, squeeze_dim=self.theta_dim)
        samples_std = self.theta_std.to_std(samples) if self.config.standardise else samples
        net = self._train_prior_score_net(samples_std, sde, generator, tag="prior")
        wrapped = _ThetaOnlyScore(net).to(device=self.device, dtype=self.dtype)

        def prior_score_fn(theta_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return _net_call(wrapped, theta_t, None, t)

        self.info["prior_score"] = "network"
        return prior_score_fn

    # -- sampling -----------------------------------------------------------
    def _sample(
        self,
        network: nn.Module,
        sde: SDE,
        x_obs: torch.Tensor,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        with_log_prob: bool = False,
    ) -> Any:
        return _network_sample(
            sde,
            network,
            x_obs,
            num_samples,
            config=self.config.sampler_config(),
            theta_dim=self.theta_dim,
            theta_std=self.theta_std,
            x_std=self.x_std,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            with_log_prob=with_log_prob,
        )

    def _log_prob(
        self, network: nn.Module, sde: SDE, x_obs: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        return _network_log_prob(
            sde,
            network,
            x_obs,
            theta,
            config=self.config.sampler_config(),
            theta_dim=self.theta_dim,
            theta_std=self.theta_std,
            x_std=self.x_std,
            device=self.device,
            dtype=self.dtype,
        )

    # -- proposal prior bookkeeping ----------------------------------------
    def _new_proposal_prior(self, generator: torch.Generator) -> ProposalPrior:
        proposal = ProposalPrior(
            mode=self.config.proposal_density,
            n_components=self.config.proposal_n_components,
            mixture=self.config.proposal_mixture,
            device=self.device,
            dtype=self.dtype,
        )
        prior = self._get_prior(generator)
        proposal.add_exact(
            prior.log_prob, sample_fn=lambda n, g=None: prior.sample(n, generator=g, squeeze_dim=self.theta_dim)
        )
        return proposal

    def _append_round_samples(
        self, proposal: ProposalPrior, samples: torch.Tensor, generator: torch.Generator
    ) -> None:
        """Add the freshly obtained posterior samples as a mixture component."""
        proposal.add_samples(samples.detach(), generator=generator)

    def _importance_weights(
        self,
        theta: torch.Tensor,
        proposal: ProposalPrior,
        prior: PriorWrapper,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(weights, log_ratio)`` with ``weights ~ p(theta)/ptilde^r(theta)``."""
        log_ratio = proposal.log_ratio(theta, prior.log_prob)
        clip = self.config.weight_clip
        if clip is not None:
            log_ratio = torch.clamp(log_ratio, -float(clip), float(clip))
        weights = torch.exp(log_ratio - log_ratio.max())
        if self.config.normalise_weights:
            try:
                weights = normalise_weights(weights, clip=clip, eps=_EPS)
            except Exception:  # pragma: no cover
                weights = weights / weights.mean().clamp_min(_EPS)
        return weights, log_ratio

    # -- dataset ------------------------------------------------------------
    def _append_dataset(self, theta: torch.Tensor, x: torch.Tensor) -> None:
        theta = _as_tensor(theta, device=self.device, dtype=self.dtype).detach()
        x = _as_tensor(x, device=self.device, dtype=self.dtype).detach()
        if self.dataset_theta is None:
            self.dataset_theta, self.dataset_x = theta, x
        else:
            self.dataset_theta = torch.cat([self.dataset_theta, theta], dim=0)
            self.dataset_x = torch.cat([self.dataset_x, x], dim=0)

    # -- public API ---------------------------------------------------------
    def score(self, theta: torch.Tensor, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.network is None:
            raise RuntimeError("No trained network available.")
        theta = ensure_2d(_as_tensor(theta, device=self.device, dtype=self.dtype))
        x = ensure_2d(_as_tensor(x, device=self.device, dtype=self.dtype))
        if t is None:
            t = torch.zeros(theta.shape[0], device=self.device, dtype=self.dtype)
        t = _as_tensor(t, device=self.device, dtype=self.dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(theta.shape[0])
        return _net_call(self.network, theta, x, t)

    def sample(
        self,
        x_obs: torch.Tensor,
        num_samples: int = 10_000,
        with_log_prob: bool = False,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Any:
        """Re-sample the (final) posterior approximation for a new observation."""
        if self.network is None:
            raise RuntimeError("Call .run(...) (or .fit) before sampling.")
        generator = generator or self._resolve_generator(None, self.config.seed)
        return self._sample(self.network, self.sde, x_obs, num_samples, generator, with_log_prob)

    posterior_samples = sample

    def state_dict(self) -> Dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "config": self.config.as_dict(),
            "network": self.network.state_dict() if self.network is not None else None,
            "theta_std": self.theta_std.state_dict() if hasattr(self.theta_std, "state_dict") else None,
            "x_std": self.x_std.state_dict() if hasattr(self.x_std, "state_dict") else None,
        }

    def save(self, path: str) -> str:  # pragma: no cover - convenience
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)
        return path


# ===========================================================================
# SNPSE-A
# ===========================================================================
class SNPSEA(_SequentialVariantBase):
    """SNPSE-A: post-hoc SIR correction (Algorithm 3, Appendix C.2)."""

    objective = "a"
    variant_name = "SNPSE-A"

    def run(
        self,
        x_obs: torch.Tensor,
        num_samples: int = 10_000,
        with_log_prob: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        generator = self._resolve_generator(generator, seed if seed is not None else cfg.seed)
        x_obs = ensure_2d(_as_tensor(x_obs, device=self.device, dtype=self.dtype))
        prior = self._get_prior(generator)
        rounds = self.round_budgets
        n_rounds = len(rounds)

        progress = ProgressBar(total=n_rounds, desc=f"{self.variant_name} rounds", enabled=cfg.verbose_rounds)

        # p_psi^0 := p(theta); samples of the round-1 training distribution
        theta_draw = prior.sample(rounds[0], generator=generator, squeeze_dim=self.theta_dim)
        self.posterior_samples_by_round[0] = theta_draw
        proposal = self._new_proposal_prior(generator)

        net = self.network
        sde = None
        for r in range(1, n_rounds + 1):
            theta_new = self.posterior_samples_by_round[r - 1]
            if theta_new.shape[0] > rounds[r - 1]:
                idx = torch.randperm(theta_new.shape[0], generator=generator)[: rounds[r - 1]]
                theta_new = theta_new[idx]
            x_new = self._simulate(theta_new)
            self._append_dataset(theta_new, x_new)

            # standardisers and VE sigma_max fixed on round-1 data (Addendum)
            if r == 1 or not cfg.reuse_standardisers or sde is None:
                ref_theta = self.dataset_theta[: rounds[0]]
                ref_x = self.dataset_x[: rounds[0]]
                self._fit_standardisers(ref_theta, ref_x)
                sde = self._build_sde(data=self.theta_std.to_std(ref_theta))
                self.sde = sde

            if r == 1 or not cfg.warm_start:
                net = self._make_network(generator)
            hist = self._train(net, sde, self.dataset_theta, self.dataset_x, generator=generator, seed=seed)
            self.history.append(hist)
            self.network = net

            # (iii)-(iv): proposal posterior samples + SIR correction (eq. 80)
            m_prime = cfg.resolved_resample_size(rounds[r - 1])
            theta_prop = self._sample(net, sde, x_obs, m_prime, generator)
            if isinstance(theta_prop, tuple):
                theta_prop = theta_prop[0]
            weights, log_ratio = self._importance_weights(theta_prop, proposal, prior, generator)
            theta_post, log_w_sel = sir_resample(
                theta_prop,
                torch.log(weights.clamp_min(_EPS)),
                rounds[r - 1],
                generator=generator,
                replace=cfg.resample_with_replacement,
                clip=cfg.weight_clip,
            )
            # weighting from eq. (80): log h = log p(theta) - log ptilde^r(theta)
            theta_post, log_w_sel = sir_resample(
                theta_prop,
                log_ratio,
                rounds[r - 1],
                generator=generator,
                replace=cfg.resample_with_replacement,
                clip=cfg.weight_clip,
            )
            self.posterior_samples_by_round[r] = theta_post
            self._append_round_samples(proposal, theta_post, generator)

            hist_info = {
                "round": r,
                "budget": int(rounds[r - 1]),
                "dataset_size": int(self.dataset_theta.shape[0]),
                "weight_mean": float(weights.mean().item()),
                "weight_max": float(weights.max().item()),
                "ess": float(
                    (weights.sum() ** 2 / (weights**2).sum().clamp_min(_EPS)).item()
                ),
            }
            self.info.setdefault("rounds", []).append(hist_info)
            if cfg.verbose_rounds:
                progress.update(1, postfix=f"ESS={hist_info['ess']:.0f}")
            if callback is not None:
                callback(r, hist_info)
        progress.close()

        self.proposal_prior = proposal
        samples = self.posterior_samples_by_round[n_rounds]
        out: Dict[str, Any] = {
            "theta": samples,
            "network": net,
            "sde": sde,
            "history": self.history,
            "info": self.info,
            "dataset": (self.dataset_theta, self.dataset_x),
            "posterior_rounds": self.posterior_samples_by_round,
            "proposal_prior": proposal,
        }
        if with_log_prob:
            # log p_psi^R(theta) ~ log ptilde_psi^R(theta|x) + log h(theta) - log Z
            try:
                lp = self._log_prob(net, sde, x_obs, samples)
                log_w = proposal.log_ratio(samples, prior.log_prob)
                log_z = torch.logsumexp(log_w, dim=0) - math.log(max(log_w.numel(), 1))
                out["log_prob"] = lp + log_w - log_z
            except Exception:  # pragma: no cover
                out["log_prob"] = self._log_prob(net, sde, x_obs, samples)
        return out


# ===========================================================================
# SNPSE-B
# ===========================================================================
class SNPSEB(_SequentialVariantBase):
    """SNPSE-B: importance-weighted DSM loss, eq. 99/15 (Algorithm 4, Appendix C.3)."""

    objective = "b"
    variant_name = "SNPSE-B"

    def run(
        self,
        x_obs: torch.Tensor,
        num_samples: int = 10_000,
        with_log_prob: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        generator = self._resolve_generator(generator, seed if seed is not None else cfg.seed)
        x_obs = ensure_2d(_as_tensor(x_obs, device=self.device, dtype=self.dtype))
        prior = self._get_prior(generator)
        rounds = self.round_budgets
        n_rounds = len(rounds)

        progress = ProgressBar(total=n_rounds, desc=f"{self.variant_name} rounds", enabled=cfg.verbose_rounds)

        theta_draw = prior.sample(rounds[0], generator=generator, squeeze_dim=self.theta_dim)
        self.posterior_samples_by_round[0] = theta_draw
        proposal = self._new_proposal_prior(generator)

        net = self.network
        sde = None
        weights_all: Optional[torch.Tensor] = None
        for r in range(1, n_rounds + 1):
            theta_new = self.posterior_samples_by_round[r - 1]
            if theta_new.shape[0] > rounds[r - 1]:
                idx = torch.randperm(theta_new.shape[0], generator=generator)[: rounds[r - 1]]
                theta_new = theta_new[idx]
            x_new = self._simulate(theta_new)
            self._append_dataset(theta_new, x_new)

            if r == 1 or not cfg.reuse_standardisers or sde is None:
                ref_theta = self.dataset_theta[: rounds[0]]
                ref_x = self.dataset_x[: rounds[0]]
                self._fit_standardisers(ref_theta, ref_x)
                sde = self._build_sde(data=self.theta_std.to_std(ref_theta))
                self.sde = sde

            # per-datapoint importance weights p(theta_0)/ptilde^r(theta_0) (eq. 99)
            if r == 1:
                weights_all = torch.ones(self.dataset_theta.shape[0], device=self.device, dtype=self.dtype)
            else:
                weights_all, _ = self._importance_weights(self.dataset_theta, proposal, prior, generator)

            if r == 1 or not cfg.warm_start:
                net = self._make_network(generator)
            hist = self._train(
                net,
                sde,
                self.dataset_theta,
                self.dataset_x,
                weights=weights_all,
                generator=generator,
                seed=seed,
                objective="b",
            )
            self.history.append(hist)
            self.network = net

            # no sampling correction required (Appendix C.3.2)
            theta_post = self._sample(net, sde, x_obs, rounds[r - 1], generator)
            if isinstance(theta_post, tuple):
                theta_post = theta_post[0]
            self.posterior_samples_by_round[r] = theta_post
            self._append_round_samples(proposal, theta_post, generator)

            hist_info = {
                "round": r,
                "budget": int(rounds[r - 1]),
                "dataset_size": int(self.dataset_theta.shape[0]),
                "weight_mean": float(weights_all.mean().item()),
                "weight_max": float(weights_all.max().item()),
                "ess": float(
                    (weights_all.sum() ** 2 / (weights_all**2).sum().clamp_min(_EPS)).item()
                ),
            }
            self.info.setdefault("rounds", []).append(hist_info)
            if cfg.verbose_rounds:
                progress.update(1, postfix=f"ESS={hist_info['ess']:.0f}")
            if callback is not None:
                callback(r, hist_info)
        progress.close()

        self.proposal_prior = proposal
        samples = self.posterior_samples_by_round[n_rounds]
        out: Dict[str, Any] = {
            "theta": samples,
            "network": net,
            "sde": sde,
            "history": self.history,
            "info": self.info,
            "dataset": (self.dataset_theta, self.dataset_x),
            "posterior_rounds": self.posterior_samples_by_round,
            "proposal_prior": proposal,
        }
        if with_log_prob:
            out["log_prob"] = self._log_prob(net, sde, x_obs, samples)
        return out


# ===========================================================================
# SNPSE-C
# ===========================================================================
class SNPSEC(_SequentialVariantBase):
    """SNPSE-C: score-space correction, eq. 102/103 (Algorithm 5, Appendix C.4)."""

    objective = "a"
    variant_name = "SNPSE-C"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.corrected: Optional[nn.Module] = None
        self.proposal_score_nets: List[nn.Module] = []

    def run(
        self,
        x_obs: torch.Tensor,
        num_samples: int = 10_000,
        with_log_prob: bool = False,
        generator: Optional[torch.Generator] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        cfg = self.config
        generator = self._resolve_generator(generator, seed if seed is not None else cfg.seed)
        x_obs = ensure_2d(_as_tensor(x_obs, device=self.device, dtype=self.dtype))
        prior = self._get_prior(generator)
        rounds = self.round_budgets
        n_rounds = len(rounds)

        progress = ProgressBar(total=n_rounds, desc=f"{self.variant_name} rounds", enabled=cfg.verbose_rounds)

        theta_draw = prior.sample(rounds[0], generator=generator, squeeze_dim=self.theta_dim)
        self.posterior_samples_by_round[0] = theta_draw
        proposal = self._new_proposal_prior(generator)

        net = self.network
        sde = None
        prior_score_fn: Optional[Callable] = None
        for r in range(1, n_rounds + 1):
            theta_new = self.posterior_samples_by_round[r - 1]
            if theta_new.shape[0] > rounds[r - 1]:
                idx = torch.randperm(theta_new.shape[0], generator=generator)[: rounds[r - 1]]
                theta_new = theta_new[idx]
            x_new = self._simulate(theta_new)
            self._append_dataset(theta_new, x_new)

            if r == 1 or not cfg.reuse_standardisers or sde is None:
                ref_theta = self.dataset_theta[: rounds[0]]
                ref_x = self.dataset_x[: rounds[0]]
                self._fit_standardisers(ref_theta, ref_x)
                sde = self._build_sde(data=self.theta_std.to_std(ref_theta))
                self.sde = sde
                prior_score_fn = self._prior_score_fn(sde, generator)

            if r == 1 or not cfg.warm_start:
                net = self._make_network(generator)

            # Algorithm 5: learn the perturbed proposal prior score network (eq. 123)
            proposal_score_fn = None
            if r > 1:
                n_samp = int(max(cfg.n_proposal_score_samples, self.dataset_theta.shape[0]))
                samples = proposal.sample(n_samp, generator=generator)
                samples_std = self.theta_std.to_std(samples) if cfg.standardise else samples
                prop_net = self._train_prior_score_net(
                    samples_std, sde, generator, tag="proposal-prior"
                )
                self.proposal_score_nets.append(prop_net)
                wrapped = _ThetaOnlyScore(prop_net).to(device=self.device, dtype=self.dtype)

                def proposal_score_fn(th: torch.Tensor, t: torch.Tensor, _net=wrapped) -> torch.Tensor:  # type: ignore[misc]
                    return _net_call(_net, th, None, t)

            corrected = CorrectedScore(
                net,
                proposal_score_fn=proposal_score_fn,
                prior_score_fn=prior_score_fn,
            )
            hist = self._train(
                corrected,
                sde,
                self.dataset_theta,
                self.dataset_x,
                generator=generator,
                seed=seed,
                objective="a",
            )
            self.history.append(hist)
            self.network = net
            self.corrected = corrected

            sample_net: nn.Module = corrected if cfg.sample_with == "corrected" else net
            theta_post = self._sample(sample_net, sde, x_obs, rounds[r - 1], generator)
            if isinstance(theta_post, tuple):
                theta_post = theta_post[0]
            self.posterior_samples_by_round[r] = theta_post
            self._append_round_samples(proposal, theta_post, generator)

            hist_info = {
                "round": r,
                "budget": int(rounds[r - 1]),
                "dataset_size": int(self.dataset_theta.shape[0]),
                "proposal_score": r > 1,
                "prior_score": self.info.get("prior_score", "unknown"),
            }
            self.info.setdefault("rounds", []).append(hist_info)
            if cfg.verbose_rounds:
                progress.update(1)
            if callback is not None:
                callback(r, hist_info)
        progress.close()

        self.proposal_prior = proposal
        samples = self.posterior_samples_by_round[n_rounds]
        out: Dict[str, Any] = {
            "theta": samples,
            "network": net,
            "corrected_network": self.corrected,
            "sde": sde,
            "history": self.history,
            "info": self.info,
            "dataset": (self.dataset_theta, self.dataset_x),
            "posterior_rounds": self.posterior_samples_by_round,
            "proposal_prior": proposal,
        }
        if with_log_prob:
            out["log_prob"] = self._log_prob(net, sde, x_obs, samples)
        return out


# convenient aliases matching the paper's naming
SNPSE_A = SNPSEA
SNPSE_B = SNPSEB
SNPSE_C = SNPSEC


# ===========================================================================
# Functional drivers
# ===========================================================================
def _run_variant(
    cls: type,
    prior: Any,
    simulator: Callable,
    x_obs: torch.Tensor,
    theta_dim: int,
    x_dim: int,
    config: Optional[SNPSEConfig] = None,
    num_samples: int = 10_000,
    with_log_prob: bool = False,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    estimator = cls(
        theta_dim=theta_dim,
        x_dim=x_dim,
        prior=prior,
        simulator=simulator,
        config=config,
        device=device,
        **_filter_kwargs(cls.__init__, kwargs),
    )
    out = estimator.run(
        x_obs,
        num_samples=num_samples,
        with_log_prob=with_log_prob,
        seed=seed,
    )
    out["estimator"] = estimator
    return out


def run_snpse_a(prior, simulator, x_obs, theta_dim, x_dim, **kwargs) -> Dict[str, Any]:
    """Run SNPSE-A (Algorithm 3) end to end."""
    return _run_variant(SNPSEA, prior, simulator, x_obs, theta_dim, x_dim, **kwargs)


def run_snpse_b(prior, simulator, x_obs, theta_dim, x_dim, **kwargs) -> Dict[str, Any]:
    """Run SNPSE-B (Algorithm 4) end to end."""
    return _run_variant(SNPSEB, prior, simulator, x_obs, theta_dim, x_dim, **kwargs)


def run_snpse_c(prior, simulator, x_obs, theta_dim, x_dim, **kwargs) -> Dict[str, Any]:
    """Run SNPSE-C (Algorithm 5) end to end."""
    return _run_variant(SNPSEC, prior, simulator, x_obs, theta_dim, x_dim, **kwargs)


# ===========================================================================
# self test
# ===========================================================================
def _selftest(sde_name: str = "vp", rounds: int = 2, budget: int = 400) -> None:  # pragma: no cover
    """Small end-to-end check of the three variants on a 2D Gaussian task."""
    torch.manual_seed(0)
    d, p = 2, 2
    A = torch.tensor([[1.0, 0.5], [0.3, 0.8]])
    noise = 0.1

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        return theta @ A.t() + noise * torch.randn(theta.shape[0], p)

    prior = torch.distributions.MultivariateNormal(torch.zeros(d), torch.eye(d))
    x_obs = simulator(torch.tensor([[0.4, -0.3]])).reshape(1, p)

    cfg = SNPSEConfig(
        sde=sde_name,
        num_rounds=rounds,
        budget=budget,
        max_iters=200,
        batch_size=100,
        sampler_method="euler",
        sampler_n_steps=50,
        proposal_n_components=2,
        proposal_density="gmm",
        verbose_rounds=False,
        verbose=False,
        standardise=True,
    )

    for cls in (SNPSEA, SNPSEB, SNPSEC):
        estimator = cls(theta_dim=d, x_dim=p, prior=prior, simulator=simulator, config=cfg)
        out = estimator.run(x_obs, num_samples=64, seed=1)
        theta = out["theta"]
        assert theta.shape == (64, d), theta.shape
        assert torch.isfinite(theta).all()
        hist = out["history"]
        losses = [h.train_loss[-1] if h.train_loss else float("nan") for h in hist]
        print(f"[{cls.variant_name}] final round losses: {['%.4f' % l for l in losses]}")
        print(f"[{cls.variant_name}] posterior mean: {theta.mean(dim=0).tolist()}")
        print(f"[{cls.variant_name}] info: {estimator.info.get('rounds')}")
        if cls is SNPSEC:
            assert estimator.info.get("prior_score") in ("analytic", "network")
    print("snpse_variants self-test passed.")


if __name__ == "__main__":  # pragma: no cover
    import sys

    sde_arg = sys.argv[1] if len(sys.argv) > 1 else "vp"
    _selftest(sde_arg)
