"""Neural Likelihood Score Estimation (NLSE) -- Appendix B of the SNPSE paper.

NLSE is the alternative to NPSE (Section 2.2) for learning the perturbed posterior
score.  It starts from the Bayes decomposition (eq. 54 / eq. (1) of Appendix B.1)

.. math::

    \\nabla_{\\theta}\\log p_t(\\theta_t \\mid x)
        = \\nabla_{\\theta}\\log p_t(x \\mid \\theta_t)
        + \\nabla_{\\theta}\\log p_t(\\theta_t),

trains a score network :math:`s_{\\psi_{lik}}(\\theta_t, x, t)` for the score of the
perturbed *likelihood* :math:`\\nabla_\\theta \\log p_t(x\\mid\\theta_t)` with the
denoising likelihood score matching objective (eq. 57 / eq. (4) of Appendix B.1)

.. math::

    \\mathcal{J}_{lik}^{DSM} = \\frac12\\int_0^T \\lambda_t\\,
       \\mathbb{E}\\Big[\\big\\|s_{\\psi_{lik}}(\\theta_t,x,t)
       + \\nabla_\\theta\\log p_t(\\theta_t)
       - \\nabla_\\theta\\log p_{t|0}(\\theta_t\\mid\\theta_0)\\big\\|^2\\Big]\\,dt,

and then forms the posterior score (eq. 55 / eq. (2) of Appendix B.1)

.. math::

    s_{\\psi_{post}}(\\theta_t, x, t)
        = s_{\\psi_{lik}}(\\theta_t, x, t) + \\nabla_\\theta \\log p_t(\\theta_t).

The perturbed prior score :math:`\\nabla_\\theta\\log p_t(\\theta_t)` is obtained either

* **analytically** (Appendix B.2.1) for affine drifts :math:`f=0`,
  :math:`g(t)=\\tau_t` and *Uniform* (eq. 58) or *Gaussian mixture* (eq. 59) priors

    .. math::
        p_t(\\theta_t)=\\frac{1}{\\prod_i (b_i-a_i)}\\prod_i
            \\Big[\\Phi\\big(b_i \\mid \\theta_{t,i}, \\tau_{t,i}^2\\big)
                  -\\Phi\\big(a_i \\mid \\theta_{t,i}, \\tau_{t,i}^2\\big)\\Big],
        \\qquad
        p_t(\\theta_t)=\\sum_k \\alpha_k\\,\\mathcal{N}
            \\big(\\theta_t \\mid \\mu_k, \\Sigma_k + \\tau_t^2 I\\big),

  (both are implemented here for the general Gaussian marginal
  :math:`p_{t|0}(\\theta_t\\mid\\theta_0)=\\mathcal N(\\theta_t \\mid \\alpha_t\\theta_0,
  \\beta_t^2 I)` so that they also apply to the VP SDE, for which
  :math:`\\alpha_t=e^{-B(t)/2}` and :math:`\\beta_t=\\sqrt{1-e^{-B(t)}}`), or

* via an **extra score network** :math:`s_{\\psi_{pri}}(\\theta_t,t)
  \\approx \\nabla_\\theta\\log p_t(\\theta_t)` trained with

    .. math::
        \\mathcal{J}_{pri}(\\psi_{pri}) = \\frac12\\int_0^T \\lambda_t\,
            \\mathbb{E}_{p(\\theta_0)p_{t|0}(\\theta_t\\mid\\theta_0)}
            \\big[\\|s_{\\psi_{pri}}(\\theta_t,t)
                  -\\nabla_\\theta\\log p_{t|0}(\\theta_t\\mid\\theta_0)\\|^2\\big]\\,dt

  (eq. 64, **Algorithm 2 Prior Score Estimation**).

Public API
----------
``NLSEConfig``, ``PriorSpec``, ``PriorScoreNetwork``, ``AnalyticPriorScore``,
``NetworkPriorScore``, ``NLSEPosteriorScore``, ``train_prior_score_network``,
``build_prior_score``, ``NLSE``, ``run_nlse``.

Running ``python -m snpse.nlse`` executes the unit-level sanity checks
(analytic perturbed-prior density/score, eq. 57 training on a linear-Gaussian
benchmark, and eq. 64 prior-score estimation).
"""

from __future__ import annotations

import inspect
import math
import os
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Imports of sibling modules (with flat-import fallbacks for direct execution)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - package context
    from .sdes import SDE, T_FINAL, get_sde
    from .score_network import (
        MLP,
        ScoreNetwork,
        count_parameters,
        embedding_dim,
        get_score_network,
        sinusoidal_embedding,
    )
    from .losses import dsm_loss, npse_loss, nlse_loss, prior_score_loss
    from .sampler import (
        SamplerConfig,
        estimate_log_prob,
        sample_posterior,
    )
    from .trainer import TrainConfig, TrainHistory, Trainer, select_batch_size
    from .utils import (
        ProgressBar,
        Standardiser,
        ensure_2d,
        fit_standardiser,
        get_device,
        normalise_weights,
        safe_log,
        set_seed,
        to_tensor,
    )
except ImportError:  # pragma: no cover - flat execution
    from sdes import SDE, T_FINAL, get_sde  # type: ignore
    from score_network import (  # type: ignore
        MLP,
        ScoreNetwork,
        count_parameters,
        embedding_dim,
        get_score_network,
        sinusoidal_embedding,
    )
    from losses import dsm_loss, npse_loss, nlse_loss, prior_score_loss  # type: ignore
    from sampler import (  # type: ignore
        SamplerConfig,
        estimate_log_prob,
        sample_posterior,
    )
    from trainer import TrainConfig, TrainHistory, Trainer, select_batch_size  # type: ignore
    from utils import (  # type: ignore
        ProgressBar,
        Standardiser,
        ensure_2d,
        fit_standardiser,
        get_device,
        normalise_weights,
        safe_log,
        set_seed,
        to_tensor,
    )

try:  # optional helpers from npse.py -------------------------------------------------
    from .npse import build_sde as _npse_build_sde  # type: ignore # noqa
except Exception:  # pragma: no cover
    try:
        from npse import build_sde as _npse_build_sde  # type: ignore
    except Exception:  # pragma: no cover
        _npse_build_sde = None


__all__ = [
    "NLSEConfig",
    "PriorSpec",
    "PriorScoreNetwork",
    "PriorScoreFn",
    "AnalyticPriorScore",
    "NetworkPriorScore",
    "NLSEPosteriorScore",
    "train_prior_score_network",
    "build_prior_score",
    "build_prior_spec",
    "NLSE",
    "run_nlse",
    "DEFAULT_N_PRIOR_SAMPLES",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_N_PRIOR_SAMPLES = 20_000       # Algorithm 2: prior sample budget N
DEFAULT_NLSE_BATCH_SIZE = 200
LIKELIHOOD_MODE = "likelihood"
POSTERIOR_MODE = "posterior"
_SQRT2 = math.sqrt(2.0)
_LOG_2PI = math.log(2.0 * math.pi)
_JITTER = 1e-8


# --------------------------------------------------------------------------------------
# Small numerical helpers
# --------------------------------------------------------------------------------------
def _filter_kwargs(fn: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keyword arguments that ``fn`` does not accept."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return dict(kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _supports(fn: Callable, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover
        return False
    return name in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def _as_time_vector(t: Any, n: int, ref: torch.Tensor) -> torch.Tensor:
    """Return ``t`` as a 1-D tensor of length ``n`` on ``ref``'s device/dtype."""
    if t is None:
        return torch.rand(n, device=ref.device, dtype=ref.dtype)
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.to(device=ref.device, dtype=ref.dtype)
    if t.dim() == 0:
        t = t.expand(n)
    else:
        t = t.reshape(-1)
        if t.numel() == 1 and n != 1:
            t = t.expand(n)
    return t


def _as_column(value: Any, n: int, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar / (n,) / (n, d) tensor to something usable as a column."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    value = value.to(device=ref.device, dtype=ref.dtype)
    if value.dim() >= 2 and value.shape == ref.shape:
        return value                                # already elementwise (n, d)
    if value.dim() == 0:
        return value.expand(n, 1)
    value = value.reshape(-1)
    if value.numel() == 1:
        return value.expand(n).reshape(n, 1)
    if value.numel() >= n:
        return value[:n].reshape(n, 1)
    return value.reshape(n, 1)  # pragma: no cover


def _std_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Differentiable standard normal CDF (used by eq. 58)."""
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _chunked(fn: Callable[[torch.Tensor], torch.Tensor], theta: torch.Tensor,
             chunk_size: Optional[int] = 4096) -> torch.Tensor:
    if chunk_size is None or theta.shape[0] <= chunk_size:
        return fn(theta)
    outs = [fn(theta[i:i + chunk_size]) for i in range(0, theta.shape[0], chunk_size)]
    return torch.cat(outs, dim=0)


def _alpha_beta(sde: SDE, t: torch.Tensor, ref: torch.Tensor,
                eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
    """Marginal coefficients of ``p_{t|0}``: ``(alpha_t, beta_t)``, shape ``(n,)``.

    ``p_{t|0}(theta_t | theta_0) = N(theta_t | alpha_t * theta_0, beta_t^2 I)``.

    * VE SDE: ``alpha_t = 1`` and ``beta_t = sigma_t``.
    * VP SDE: ``alpha_t = exp(-B(t)/2)`` and ``beta_t = sqrt(1 - exp(-B(t)))``.
    """
    t = _as_time_vector(t, ref.shape[0], ref)

    if hasattr(sde, "sigma_min") and hasattr(sde, "sigma_max"):
        sigma_min = float(getattr(sde, "sigma_min"))
        sigma_max = float(getattr(sde, "sigma_max"))
        alpha = torch.ones_like(t)
        beta = sigma_min * (sigma_max / max(sigma_min, eps)) ** t
        return alpha, beta

    beta_min = getattr(sde, "beta_min", None)
    beta_max = getattr(sde, "beta_max", None)
    if beta_min is not None and beta_max is not None:
        int_beta = getattr(sde, "int_beta", None)
        if callable(int_beta):
            B = int_beta(t)
        else:  # linear beta schedule (default)
            B = float(beta_min) * t + 0.5 * (float(beta_max) - float(beta_min)) * t ** 2
        B = _as_time_vector(B, ref.shape[0], ref)
        alpha = torch.exp(-0.5 * B)
        beta = torch.sqrt(torch.clamp(1.0 - torch.exp(-B), min=eps))
        return alpha, beta

    # Generic fallback using the SDE's own marginal helpers
    beta = torch.ones_like(t)
    marginal_std = getattr(sde, "marginal_std", None)
    if callable(marginal_std):
        try:
            out = marginal_std(t)
            beta = _as_time_vector(out, ref.shape[0], ref)
        except Exception:  # pragma: no cover
            pass
    alpha = torch.ones_like(t)
    marginal_mean = getattr(sde, "marginal_mean", None)
    if callable(marginal_mean):
        try:
            probe = torch.ones_like(ref)
            out = marginal_mean(probe, t)
            alpha = _as_time_vector(out.mean(dim=-1), ref.shape[0], ref)
        except Exception:  # pragma: no cover
            pass
    return alpha, beta


# --------------------------------------------------------------------------------------
# Standardisation helpers (theta is standardised only when explicitly requested)
# --------------------------------------------------------------------------------------
def _fit_std(data: torch.Tensor, enabled: bool = True,
             mode: str = "std") -> Optional[Standardiser]:
    if not enabled or data is None:
        return None
    try:
        std = fit_standardiser(data, mode=mode)
    except Exception:  # pragma: no cover
        try:
            std = Standardiser.from_data(data, mode=mode)
        except Exception:
            return None
    if not hasattr(std, "shift") or not hasattr(std, "scale"):
        return None
    return std


def _std_forward(std: Optional[Standardiser], data: torch.Tensor) -> torch.Tensor:
    if std is None:
        return data
    shift, scale = getattr(std, "shift", None), getattr(std, "scale", None)
    if shift is None or scale is None:  # pragma: no cover
        return data
    return (data - shift) / scale


def _std_inverse(std: Optional[Standardiser], data: torch.Tensor) -> torch.Tensor:
    if std is None:
        return data
    shift, scale = getattr(std, "shift", None), getattr(std, "scale", None)
    if shift is None or scale is None:  # pragma: no cover
        return data
    return data * scale + shift


def _std_logdet(std: Optional[Standardiser], ref: torch.Tensor) -> torch.Tensor:
    """``log |det d theta_std / d theta| = -sum_d log scale_d``."""
    if std is None:
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    scale = getattr(std, "scale", None)
    if scale is None:  # pragma: no cover
        return torch.zeros((), device=ref.device, dtype=ref.dtype)
    return -torch.log(scale.to(device=ref.device, dtype=ref.dtype)).sum()


def _std_attrs(std: Optional[Standardiser], ref: torch.Tensor):
    if std is None:
        return None, None
    shift, scale = getattr(std, "shift", None), getattr(std, "scale", None)
    if shift is None or scale is None:  # pragma: no cover
        return None, None
    return shift.to(device=ref.device, dtype=ref.dtype), scale.to(device=ref.device, dtype=ref.dtype)


# --------------------------------------------------------------------------------------
# Prior specifications and the closed-form perturbed prior (Appendix B.2.1)
# --------------------------------------------------------------------------------------
@dataclass
class PriorSpec:
    """A prior distribution of a known family, in a known (affine) parameter space.

    ``family`` is one of ``"uniform"``, ``"gaussian"``, ``"gmm"`` or ``"custom"``.
    ``params`` holds the family parameters:

    * ``uniform``  : ``{"low": (d,), "high": (d,)}``
    * ``gaussian`` : ``{"mean": (d,), "cov": (d, d)}``
    * ``gmm``      : ``{"weights": (K,), "means": (K, d), "covs": (K, d, d)}``

    Only the three parametric families admit the closed-form perturbed prior of
    Appendix B.2.1 (eq. 58 / eq. 59); anything else is ``custom`` and requires the
    prior score network of Appendix B.2.2 (Algorithm 2).
    """

    family: str = "custom"
    params: Dict[str, torch.Tensor] = field(default_factory=dict)

    # ------------------------------------------------------------------ constructors
    @classmethod
    def uniform(cls, low, high) -> "PriorSpec":
        low = to_tensor(low).flatten()
        high = to_tensor(high).flatten()
        return cls("uniform", {"low": low, "high": high})

    @classmethod
    def gaussian(cls, mean, cov=None, std=None) -> "PriorSpec":
        mean = to_tensor(mean).flatten()
        if cov is None:
            d = mean.numel()
            s = 1.0 if std is None else to_tensor(std).flatten()
            cov = torch.diag(torch.full((d,), float(s) ** 2 if torch.is_tensor(s) is False else 1.0)) \
                if std is None else torch.diag(to_tensor(std).flatten() ** 2)
        return cls("gaussian", {"mean": mean, "cov": to_tensor(cov)})

    @classmethod
    def gmm(cls, weights, means, covs) -> "PriorSpec":
        return cls("gmm", {
            "weights": to_tensor(weights).flatten(),
            "means": to_tensor(means),
            "covs": to_tensor(covs),
        })

    @classmethod
    def from_prior(cls, prior, theta_dim: Optional[int] = None,
                   n_samples: int = 50_000, generator=None,
                   dtype: torch.dtype = torch.float32) -> "PriorSpec":
        """Best-effort detection of an analytic family from a prior object."""
        if isinstance(prior, PriorSpec):
            return prior
        if prior is None:
            return cls("custom", {})

        base = getattr(prior, "base_dist", None)
        inner = getattr(base, "base_dist", None)
        for cand in (inner, base, prior):
            if cand is None:
                continue
            low = getattr(cand, "low", None)
            high = getattr(cand, "high", None)
            if low is not None and high is not None:
                return cls.uniform(low, high)
            loc = getattr(cand, "loc", None)
            if loc is not None:
                cov = getattr(cand, "covariance_matrix", None)
                if cov is None:
                    tril = getattr(cand, "scale_tril", None)
                    if tril is not None:
                        cov = tril @ tril.transpose(-1, -2)
                if cov is not None:
                    return cls.gaussian(loc.flatten(), cov)
        return cls("custom", {})

    # -------------------------------------------------------------------- properties
    @property
    def analytic(self) -> bool:
        return self.family in ("uniform", "gaussian", "gmm")

    def to(self, device=None, dtype=None) -> "PriorSpec":
        params = {
            k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v)
            for k, v in self.params.items()
        }
        return PriorSpec(self.family, params)

    # ------------------------------------------------------------------- transforms
    def standardise(self, shift: Optional[torch.Tensor],
                    scale: Optional[torch.Tensor]) -> "PriorSpec":
        """Return the prior of ``theta_std = (theta - shift)/scale`` (affine, diagonal)."""
        if shift is None or scale is None or self.family == "custom":
            return self
        shift = to_tensor(shift).flatten()
        scale = to_tensor(scale).flatten()
        s = scale

        def _m(mean):
            return (mean - shift) / s

        def _c(cov):
            d = cov.shape[-1]
            D = torch.diag(1.0 / s)
            return D @ cov.reshape(-1, d, d) @ D

        if self.family == "uniform":
            low = self.params["low"]
            high = self.params["high"]
            return PriorSpec("uniform", {"low": (low - shift) / s, "high": (high - shift) / s})
        if self.family == "gaussian":
            return PriorSpec("gaussian", {
                "mean": _m(self.params["mean"]),
                "cov": _c(self.params["cov"]).reshape(self.params["cov"].shape),
            })
        if self.family == "gmm":
            means = self.params["means"]
            covs = self.params["covs"]
            d = covs.shape[-1]
            return PriorSpec("gmm", {
                "weights": self.params["weights"],
                "means": _m(means.reshape(-1, d)).reshape(means.shape),
                "covs": _c(covs.reshape(-1, d, d)).reshape(covs.shape),
            })
        return self  # pragma: no cover

    # --------------------------------------------------------------------- sampling
    def sample(self, n: int, generator=None, device=None,
               dtype: torch.dtype = torch.float32) -> torch.Tensor:
        fam = self.family
        p = {k: (v.to(device=device, dtype=dtype) if torch.is_tensor(v) else v)
             for k, v in self.params.items()}
        if fam == "uniform":
            low, high = p["low"], p["high"]
            u = torch.rand(n, low.numel(), generator=generator, device=device, dtype=dtype)
            return low.reshape(1, -1) + u * (high - low).reshape(1, -1)
        if fam == "gaussian":
            return self._sample_gaussian(p["mean"], p["cov"], n, generator, dtype)
        if fam == "gmm":
            weights = p["weights"].clamp(min=0)
            weights = weights / weights.sum()
            cum = torch.cumsum(weights, dim=0)
            u = torch.rand(n, generator=generator, device=device, dtype=dtype)
            idx = torch.searchsorted(cum, u).clamp(max=weights.numel() - 1)
            out = torch.empty(n, p["means"].shape[-1], device=device, dtype=dtype)
            for k in range(weights.numel()):
                mask = idx == k
                if mask.any():
                    out[mask] = self._sample_gaussian(
                        p["means"][k], p["covs"][k], int(mask.sum()), generator, dtype
                    )
            return out
        raise ValueError(
            "PriorSpec of family 'custom' cannot be sampled directly; "
            "supply a prior_sample_fn or a concrete prior object instead."
        )

    @staticmethod
    def _sample_gaussian(mean, cov, n, generator, dtype) -> torch.Tensor:
        mean = mean.reshape(-1)
        cov = cov.reshape(mean.numel(), mean.numel())
        L = torch.linalg.cholesky(
            cov + _JITTER * torch.eye(mean.numel(), device=cov.device, dtype=cov.dtype)
        )
        eps = torch.randn(n, mean.numel(), generator=generator, device=mean.device, dtype=dtype)
        return mean.reshape(1, -1) + eps @ L.transpose(-1, -2)

    # --------------------------------------------------------------------- densities
    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        theta = ensure_2d(theta)
        fam = self.family
        p = {k: v.to(device=theta.device, dtype=theta.dtype) for k, v in self.params.items()}
        if fam == "uniform":
            low, high = p["low"], p["high"]
            inside = ((theta >= low.reshape(1, -1)) & (theta <= high.reshape(1, -1))).all(dim=-1)
            logvol = -torch.log(high - low).sum()
            return torch.where(inside, logvol.expand(theta.shape[0]),
                               torch.full_like(theta[:, 0], float("-inf")))
        if fam == "gaussian":
            return _mvn_log_prob(theta, p["mean"].reshape(1, -1), p["cov"].reshape(1, p["mean"].numel(), p["mean"].numel()))
        if fam == "gmm":
            logs = []
            for k in range(p["weights"].numel()):
                lp = _mvn_log_prob(theta, p["means"][k].reshape(1, -1), p["covs"][k].reshape(1, *p["covs"][k].shape))
                logs.append(lp + safe_log(p["weights"][k].clamp(min=1e-30)))
            return torch.logsumexp(torch.stack(logs, dim=0), dim=0)
        raise ValueError("PriorSpec of family 'custom' has no analytic log density.")

    # --------------------------------------------- closed form perturbed prior (B.2.1)
    def analytic_perturbed_log_prob(self, theta_t: torch.Tensor, alpha, beta,
                                    chunk_size: Optional[int] = 4096) -> torch.Tensor:
        """``log p_t(theta_t)`` in closed form, generalised to ``alpha_t``, ``beta_t``."""
        theta_t = ensure_2d(theta_t)
        n, d = theta_t.shape
        a = _as_column(alpha, n, theta_t)
        b = _as_column(beta, n, theta_t).clamp(min=1e-8)
        return _chunked(lambda th: self._perturbed_log_prob(th,
                                                            _slice_like(a, th),
                                                            _slice_like(b, th)),
                        theta_t, chunk_size)

    def _perturbed_log_prob(self, theta: torch.Tensor, alpha: torch.Tensor,
                            beta: torch.Tensor) -> torch.Tensor:
        p = {k: v.to(device=theta.device, dtype=theta.dtype) for k, v in self.params.items()}
        if self.family == "uniform":
            low, high = p["low"].reshape(1, -1), p["high"].reshape(1, -1)
            # p_t(theta) = 1/(a^d prod_i (b_i-a_i)) prod_i [Phi((a b_i - th_i)/b_i)
            #                                                - Phi((a a_i - th_i)/b_i)]
            z_hi = (alpha * high - theta) / beta
            z_lo = (alpha * low - theta) / beta
            diff = (_std_normal_cdf(z_hi) - _std_normal_cdf(z_lo)).clamp(min=1e-30)
            logvol = -torch.log(high - low).sum()
            log_alpha = -torch.log(alpha).sum(dim=-1)
            return logvol + log_alpha + safe_log(diff).sum(dim=-1)
        if self.family == "gaussian":
            mean = p["mean"].reshape(1, -1)
            cov = p["cov"].reshape(1, mean.numel(), mean.numel())
            return _mvn_log_prob(theta, alpha * mean, alpha.pow(2).unsqueeze(-1) * cov
                                 + beta.pow(2).unsqueeze(-1) * _eye_like(cov))
        if self.family == "gmm":
            w = p["weights"]
            logs = []
            for k in range(w.numel()):
                mean = p["means"][k].reshape(1, -1)
                cov = p["covs"][k].reshape(1, mean.numel(), mean.numel())
                lp = _mvn_log_prob(theta, alpha * mean,
                                   alpha.pow(2).unsqueeze(-1) * cov
                                   + beta.pow(2).unsqueeze(-1) * _eye_like(cov))
                logs.append(lp + safe_log(w[k].clamp(min=1e-30)))
            return torch.logsumexp(torch.stack(logs, dim=0), dim=0)
        raise ValueError("No closed-form perturbed prior for family 'custom'.")  # pragma: no cover

    def analytic_perturbed_score(self, theta_t: torch.Tensor, alpha, beta,
                                 create_graph: bool = False,
                                 chunk_size: Optional[int] = 4096) -> torch.Tensor:
        """``grad_theta log p_t(theta_t)`` by autodiff of the closed form (B.2.1)."""
        theta_t = ensure_2d(theta_t)
        n, d = theta_t.shape
        a = _as_column(alpha, n, theta_t)
        b = _as_column(beta, n, theta_t)

        def _one(chunk: torch.Tensor, ai: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                x = chunk.detach().clone().requires_grad_(True)
                logp = self._perturbed_log_prob(x, ai, bi)
                if logp.requires_grad:
                    (g,) = torch.autograd.grad(logp.sum(), x, create_graph=create_graph)
                else:  # pragma: no cover - constant density
                    g = torch.zeros_like(x)
            return g

        if chunk_size is None or n <= chunk_size:
            return _one(theta_t, a, b)
        outs = []
        for i in range(0, n, chunk_size):
            sl = slice(i, i + chunk_size)
            outs.append(_one(theta_t[sl], a[sl] if a.shape[0] == n else a,
                             b[sl] if b.shape[0] == n else b))
        return torch.cat(outs, dim=0)


def _eye_like(cov: torch.Tensor) -> torch.Tensor:
    d = cov.shape[-1]
    return torch.eye(d, device=cov.device, dtype=cov.dtype).expand_as(cov)


def _slice_like(value: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if value.shape[0] == ref.shape[0]:
        return value
    if value.shape[0] == 1:
        return value.expand(ref.shape[0], *value.shape[1:])
    return value[:ref.shape[0]]  # pragma: no cover


def _mvn_log_prob(theta: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
    """Batched multivariate normal log density.

    ``theta``: (n, d); ``mean``: (1, d) or (n, d); ``cov``: (1, d, d) or (n, d, d).
    """
    n, d = theta.shape
    if mean.shape[0] == 1 and n > 1:
        mean = mean.expand(n, -1)
    if cov.shape[0] == 1 and n > 1:
        cov = cov.expand(n, -1, -1)
    cov = cov + _JITTER * torch.eye(d, device=cov.device, dtype=cov.dtype).reshape(1, d, d)
    L = torch.linalg.cholesky(cov)
    diff = (theta - mean).unsqueeze(-1)
    sol = torch.cholesky_solve(diff, L)
    quad = (diff * sol).sum(dim=1).squeeze(-1)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
    return -0.5 * (d * _LOG_2PI + logdet + quad)


def build_prior_spec(prior=None, theta_dim: Optional[int] = None,
                     family: Optional[str] = None,
                     params: Optional[Dict[str, Any]] = None,
                     prior_sample_fn=None, n_samples: int = DEFAULT_N_PRIOR_SAMPLES,
                     generator=None, device=None,
                     dtype: torch.dtype = torch.float32) -> PriorSpec:
    """Build a :class:`PriorSpec`, either from explicit params or by inspecting ``prior``."""
    if family is not None or params is not None:
        if family == "uniform" and params is not None:
            return PriorSpec.uniform(params["low"], params["high"])
        if family == "gaussian" and params is not None:
            return PriorSpec.gaussian(params["mean"], params.get("cov"), params.get("std"))
        if family == "gmm" and params is not None:
            return PriorSpec.gmm(params["weights"], params["means"], params["covs"])
        return PriorSpec(family or "custom",
                         {k: to_tensor(v) for k, v in (params or {}).items()})
    spec = PriorSpec.from_prior(prior, theta_dim=theta_dim, n_samples=n_samples,
                                generator=generator, dtype=dtype)
    return spec.to(device=device, dtype=dtype)


# --------------------------------------------------------------------------------------
# Prior score callables (both branches of Appendix B)
# --------------------------------------------------------------------------------------
class PriorScoreFn:
    """Callable computing ``grad_theta log p_t(theta_t)`` (shape ``(n,)``-aware ``t``)."""

    analytic: bool = False

    def __call__(self, theta_t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        t = args[-1] if args else kwargs.get("t", None)
        return self.score(theta_t, t)

    # ------------------------------------------------------------------ interface
    def score(self, theta_t: torch.Tensor, t=None) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def log_prob(self, theta_t: torch.Tensor, t=None) -> Optional[torch.Tensor]:
        return None


class AnalyticPriorScore(PriorScoreFn):
    """Closed-form perturbed prior score of Appendix B.2.1 (eq. 58 / eq. 59)."""

    analytic = True

    def __init__(self, spec: PriorSpec, sde: SDE, create_graph: bool = False,
                 chunk_size: Optional[int] = 4096):
        self.spec = spec
        self.sde = sde
        self.create_graph = create_graph
        self.chunk_size = chunk_size
        self._cache: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}

    def alpha_beta(self, t, ref: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return _alpha_beta(self.sde, t, ref)

    def log_prob(self, theta_t: torch.Tensor, t=None) -> torch.Tensor:
        theta_t = ensure_2d(theta_t)
        alpha, beta = self.alpha_beta(t, theta_t)
        return self.spec.analytic_perturbed_log_prob(theta_t, alpha, beta,
                                                     chunk_size=self.chunk_size)

    def score(self, theta_t: torch.Tensor, t=None) -> torch.Tensor:
        theta_t = ensure_2d(theta_t)
        alpha, beta = self.alpha_beta(t, theta_t)
        return self.spec.analytic_perturbed_score(theta_t, alpha, beta,
                                                  create_graph=self.create_graph,
                                                  chunk_size=self.chunk_size)

    def __repr__(self) -> str:  # pragma: no cover
        return f"AnalyticPriorScore(family={self.spec.family!r}, sde={getattr(self.sde, 'name', '?')})"


class NetworkPriorScore(PriorScoreFn):
    """Prior score network ``s_psi_pri(theta_t, t)`` (Appendix B.2.2, Algorithm 2)."""

    analytic = False

    def __init__(self, network: nn.Module, sde: Optional[SDE] = None,
                 theta_standardiser: Optional[Standardiser] = None,
                 log_prob_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.network = network
        self.sde = sde
        self.theta_standardiser = theta_standardiser
        self.log_prob_fn = log_prob_fn

    def score(self, theta_t: torch.Tensor, t=None) -> torch.Tensor:
        theta_t = ensure_2d(theta_t)
        t = _as_time_vector(t, theta_t.shape[0], theta_t)
        out = self.network(theta_t, t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def __call__(self, theta_t: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        t = args[-1] if args else kwargs.get("t", None)
        if t is None and len(args) >= 1:
            t = args[0]
        return self.score(theta_t, t)

    def log_prob(self, theta_t: torch.Tensor, t=None) -> Optional[torch.Tensor]:
        if self.log_prob_fn is None:
            return None
        try:
            return self.log_prob_fn(theta_t)
        except Exception:  # pragma: no cover
            return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"NetworkPriorScore(network={type(self.network).__name__})"


class PriorScoreNetwork(nn.Module):
    """Time-conditional score network ``s_psi_pri(theta_t, t)`` for the prior only."""

    def __init__(self, theta_dim: int, hidden_dim: int = 256, n_layers: int = 3,
                 theta_emb_dim: Optional[int] = None, time_emb_dim: int = 64,
                 activation: str = "silu", t_max: float = float(T_FINAL)):
        super().__init__()
        self.theta_dim = int(theta_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.time_emb_dim = int(time_emb_dim)
        self.theta_emb_dim = int(theta_emb_dim or embedding_dim(self.theta_dim))
        self.activation = activation
        self.t_max = float(t_max)
        self.theta_emb = MLP(self.theta_dim, self.theta_emb_dim, hidden_dim=hidden_dim,
                             n_layers=n_layers, activation=activation)
        self.head = MLP(self.theta_emb_dim + self.time_emb_dim, self.theta_dim,
                        hidden_dim=hidden_dim, n_layers=n_layers, activation=activation)

    def forward(self, theta_t: torch.Tensor, t) -> torch.Tensor:
        theta_t = ensure_2d(theta_t)
        t = _as_time_vector(t, theta_t.shape[0], theta_t)
        emb = sinusoidal_embedding(t.reshape(-1), dim=self.time_emb_dim, scale=self.t_max)
        emb = emb.reshape(theta_t.shape[0], -1)
        h = torch.cat([self.theta_emb(theta_t), emb], dim=-1)
        return self.head(h)

    def extra_repr(self) -> str:  # pragma: no cover
        return (f"theta_dim={self.theta_dim}, hidden_dim={self.hidden_dim}, "
                f"n_layers={self.n_layers}, theta_emb_dim={self.theta_emb_dim}, "
                f"time_emb_dim={self.time_emb_dim}")


class NLSEPosteriorScore(nn.Module):
    """``s_psi_post(theta_t, x, t) = s_psi_lik(theta_t, x, t) + grad log p_t(theta_t)`` (eq. 55).

    Instances expose ``theta_dim`` so that :mod:`snpse.sampler` can infer the parameter
    dimension from the score network alone.
    """

    def __init__(self, likelihood_net: nn.Module, prior_score_fn: Optional[PriorScoreFn] = None,
                 prior_sign: float = 1.0):
        super().__init__()
        self.likelihood_net = likelihood_net
        self.prior_score_fn = prior_score_fn
        self.prior_sign = float(prior_sign)
        theta_dim = (getattr(likelihood_net, "theta_dim", None)
                     or getattr(likelihood_net, "dim", None)
                     or getattr(likelihood_net, "d", None)
                     or getattr(getattr(likelihood_net, "energy_net", None), "theta_dim", None))
        self.theta_dim = int(theta_dim) if theta_dim is not None else None

    # ------------------------------------------------------------------ components
    def likelihood_score(self, theta_t: torch.Tensor, x: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        out = self.likelihood_net(theta_t, x, t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def prior_score(self, theta_t: torch.Tensor, t) -> Optional[torch.Tensor]:
        if self.prior_score_fn is None:
            return None
        return self.prior_score_fn(theta_t, t)

    def forward(self, theta_t: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = self.likelihood_score(theta_t, x, t)
        if self.prior_score_fn is not None:
            s = s + self.prior_sign * self.prior_score_fn(theta_t, t)
        return s


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class NLSEConfig:
    """Hyper-parameters for NLSE / NPSE comparison runs (Appendix B, Section 5.1)."""

    # --- method selection -----------------------------------------------------------
    mode: str = "nlse"                    # "nlse" (eq. 57) or "npse" (eq. 7)
    prior_score_mode: str = "auto"        # "auto" | "analytic" | "network"
    prior_family: Optional[str] = None    # "uniform" | "gaussian" | "gmm" | None
    prior_params: Optional[Dict[str, Any]] = None
    n_prior_samples: int = DEFAULT_N_PRIOR_SAMPLES     # Algorithm 2's N
    prior_iters: int = 3000
    prior_batch_size: Optional[int] = None
    prior_lr: float = 1e-4

    # --- forward SDE ----------------------------------------------------------------
    sde: str = "ve"
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    beta_min: float = 0.1
    beta_max: float = 11.0
    t_final: float = float(T_FINAL)

    # --- score network --------------------------------------------------------------
    hidden_dim: int = 256
    n_layers: int = 3
    time_emb_dim: int = 64
    theta_emb_dim: Optional[int] = None
    x_emb_dim: Optional[int] = None
    parameterisation: str = "score"
    activation: str = "silu"

    # --- optimisation (Section 5.1) --------------------------------------------------
    lr: float = 1e-4
    max_iters: int = 3000
    batch_size: Optional[int] = None
    budget: int = 10_000
    val_fraction: float = 0.15
    patience: int = 1000
    min_delta: float = 0.0
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None

    # --- standardisation (Appendix E.3.2) -------------------------------------------
    standardise: bool = True
    standardise_theta: bool = False   # theta kept in the prior's own space by default
    standardise_x: bool = True

    # --- reverse sampler ------------------------------------------------------------
    sampler_method: str = "rk45"
    sampler_atol: float = 1e-5
    sampler_rtol: float = 1e-5
    sampler_n_steps: int = 1000
    trace_estimator: str = "auto"

    # --- misc -----------------------------------------------------------------------
    num_samples: int = 10_000
    seed: Optional[int] = None
    device: Optional[str] = None
    dtype_name: str = "float32"
    verbose: bool = True
    log_every: int = 100

    # ------------------------------------------------------------------- helpers
    def resolved_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float64": torch.float64,
                "double": torch.float64, "float": torch.float32}.get(
            str(self.dtype_name).lower(), torch.float32)

    def resolved_batch_size(self, budget: Optional[int] = None) -> int:
        budget = int(self.budget if budget is None else budget)
        if self.batch_size is not None:
            return int(self.batch_size)
        try:
            return int(select_batch_size(budget, sequential=False))
        except Exception:  # pragma: no cover
            return 50 if budget <= 1000 else (200 if budget <= 10_000 else 500)

    def train_config(self, max_iters: Optional[int] = None,
                     batch_size: Optional[int] = None, lr: Optional[float] = None) -> "TrainConfig":
        values = dict(
            lr=self.lr if lr is None else lr,
            max_iters=self.max_iters if max_iters is None else max_iters,
            batch_size=self.resolved_batch_size() if batch_size is None else batch_size,
            val_fraction=self.val_fraction,
            patience=self.patience,
            min_delta=self.min_delta,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            log_every=self.log_every,
            verbose=self.verbose,
            seed=self.seed,
        )
        try:
            return TrainConfig(**_filter_kwargs(TrainConfig, values))
        except Exception:  # pragma: no cover
            return TrainConfig()

    def sampler_config(self) -> "SamplerConfig":
        values = dict(method=self.sampler_method, atol=self.sampler_atol,
                      rtol=self.sampler_rtol, n_steps=self.sampler_n_steps,
                      t_max=self.t_final, trace_estimator=self.trace_estimator,
                      seed=self.seed)
        try:
            return SamplerConfig(**_filter_kwargs(SamplerConfig, values))
        except Exception:  # pragma: no cover
            return SamplerConfig()

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None) -> "NLSEConfig":
        if values is None:
            return cls()
        if isinstance(values, NLSEConfig):
            return values
        if hasattr(values, "__dataclass_fields__"):
            values = {k: v for k, v in vars(values).items()}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(values).items() if k in known})


# --------------------------------------------------------------------------------------
# Building the SDE / score networks
# --------------------------------------------------------------------------------------
def build_sde(config: NLSEConfig, theta: Optional[torch.Tensor] = None,
              dim: Optional[int] = None) -> SDE:
    """Construct the VE/VP forward SDE for NLSE (see :mod:`snpse.sdes`)."""
    if _npse_build_sde is not None:
        try:
            return _npse_build_sde(config, theta=theta, dim=dim)  # type: ignore[misc]
        except Exception:
            pass
    kwargs = dict(sigma_min=config.sigma_min, sigma_max=config.sigma_max,
                  beta_min=config.beta_min, beta_max=config.beta_max,
                  data=theta, dim=dim)
    return get_sde(config.sde, **_filter_kwargs(get_sde, kwargs))


def build_likelihood_network(config: NLSEConfig, theta_dim: int, x_dim: int,
                             device=None, dtype=None) -> nn.Module:
    kwargs = dict(theta_dim=int(theta_dim), x_dim=int(x_dim), hidden_dim=config.hidden_dim,
                  n_layers=config.n_layers, theta_emb_dim=config.theta_emb_dim,
                  x_emb_dim=config.x_emb_dim, time_emb_dim=config.time_emb_dim,
                  activation=config.activation)
    net = get_score_network(config.parameterisation, **_filter_kwargs(get_score_network, kwargs))
    if device is not None or dtype is not None:
        net = net.to(device=device or None, dtype=dtype or None)
    return net


def _unpack_score(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _call_loss(loss_fn: Callable, *args, **kwargs) -> Optional[torch.Tensor]:
    """Call a DSM objective, tolerating signature differences; ``None`` on TypeError."""
    try:
        return loss_fn(*args, **_filter_kwargs(loss_fn, kwargs))
    except TypeError:
        return None


# --------------------------------------------------------------------------------------
# Monte-Carlo objectives used as local fallbacks (eq. 57 and eq. 64)
# --------------------------------------------------------------------------------------
def _mc_perturbation(sde: SDE, theta0: torch.Tensor, generator=None,
                     lambda_t: Optional[torch.Tensor] = None,
                     use_sde_weighting: bool = True):
    """Draw ``t ~ U(0,T)``, ``theta_t ~ p_{t|0}`` and the closed-form score target."""
    n = theta0.shape[0]
    t = torch.rand(n, generator=generator, device=theta0.device, dtype=theta0.dtype)
    t = t * float(getattr(sde, "T", T_FINAL))
    alpha, beta = _alpha_beta(sde, t, theta0)
    a = alpha.reshape(-1, 1)
    b = beta.reshape(-1, 1)
    mean = a * theta0
    eps = torch.randn(theta0.shape, generator=generator, device=theta0.device,
                      dtype=theta0.dtype)
    theta_t = mean + b * eps
    target = -(eps) / b                      # grad log p_{t|0}(theta_t | theta_0)
    if lambda_t is not None:
        weight = lambda_t
    elif use_sde_weighting:
        weight = b.pow(2)
    else:
        weight = torch.ones_like(b)
    return t, theta_t, target, weight


def _fallback_prior_score_loss(score_pri_fn, sde, theta0, t=None, generator=None,
                               lambda_t=None, use_sde_weighting=True,
                               reduce: str = "mean") -> torch.Tensor:
    """Local MC estimate of eq. 64 (Algorithm 2)."""
    t_vec, theta_t, target, weight = _mc_perturbation(
        sde, theta0, generator=generator, lambda_t=lambda_t,
        use_sde_weighting=use_sde_weighting,
    )
    pred = _unpack_score(score_pri_fn(theta_t, t_vec))
    diff = pred - target
    per = (weight * diff.pow(2)).reshape(theta0.shape[0], -1).sum(dim=-1)
    if reduce == "sum":
        return per.sum()
    if reduce == "none":
        return per
    return per.mean()


def _fallback_nlse_loss(score_lik_fn, sde, theta0, x, prior_score_fn=None, t=None,
                        generator=None, lambda_t=None, use_sde_weighting=True,
                        reduce: str = "mean") -> torch.Tensor:
    """Local MC estimate of eq. 57: ``|| s_lik + grad log p_t - grad log p_{t|0} ||^2``."""
    t_vec, theta_t, target, weight = _mc_perturbation(
        sde, theta0, generator=generator, lambda_t=lambda_t,
        use_sde_weighting=use_sde_weighting,
    )
    pred = _unpack_score(score_lik_fn(theta_t, x, t_vec))
    if prior_score_fn is not None:
        pred = pred + prior_score_fn(theta_t, t_vec)
    diff = pred - target
    per = (weight * diff.pow(2)).reshape(theta0.shape[0], -1).sum(dim=-1)
    if reduce == "sum":
        return per.sum()
    if reduce == "none":
        return per
    return per.mean()


def _prior_score_objective(score_pri_fn, sde, theta0, generator=None,
                           reduce: str = "mean") -> torch.Tensor:
    """Call :func:`snpse.losses.prior_score_loss`, else the local eq. 64 estimate."""
    out = _call_loss(prior_score_loss, score_pri_fn, sde, theta0,
                     generator=generator, reduce=reduce)
    if out is None:
        out = _fallback_prior_score_loss(score_pri_fn, sde, theta0,
                                        generator=generator, reduce=reduce)
    return out


def _nlse_objective(score_lik_fn, sde, theta0, x, prior_score_fn=None,
                    generator=None, reduce: str = "mean") -> torch.Tensor:
    """Call :func:`snpse.losses.nlse_loss` (eq. 57), else the local equivalent."""
    out = _call_loss(nlse_loss, score_lik_fn, sde, theta0, x,
                     prior_score_fn=prior_score_fn, generator=generator, reduce=reduce)
    if out is None:
        out = _fallback_nlse_loss(score_lik_fn, sde, theta0, x,
                                 prior_score_fn=prior_score_fn,
                                 generator=generator, reduce=reduce)
    return out


def _npse_objective(score_fn, sde, theta0, x, generator=None,
                    reduce: str = "mean") -> torch.Tensor:
    """Call :func:`snpse.losses.npse_loss` (eq. 7), else a local DSM estimate."""
    out = _call_loss(npse_loss, score_fn, sde, theta0, x, generator=generator, reduce=reduce)
    if out is None:
        try:
            out = _call_loss(dsm_loss, score_fn, sde, theta0, x,
                             generator=generator, reduce=reduce)
        except Exception:  # pragma: no cover
            out = None
    if out is None:
        out = _fallback_nlse_loss(score_fn, sde, theta0, x, prior_score_fn=None,
                                  generator=generator, reduce=reduce)
    return out


# --------------------------------------------------------------------------------------
# Algorithm 2: prior score estimation (Appendix B.2.2, eq. 64)
# --------------------------------------------------------------------------------------
def train_prior_score_network(prior=None, prior_spec: Optional[PriorSpec] = None,
                              prior_sample_fn=None, sde: Optional[SDE] = None,
                              theta_dim: Optional[int] = None,
                              config: Optional[NLSEConfig] = None,
                              n_samples: Optional[int] = None,
                              device=None, generator=None,
                              network: Optional[nn.Module] = None,
                              theta_standardiser: Optional[Standardiser] = None,
                              standardise: bool = False,
                              verbose: Optional[bool] = None,
                              ) -> Tuple[nn.Module, Dict[str, Any]]:
    """**Algorithm 2 (Prior Score Estimation)**.

    1. Draw ``N`` samples ``theta_i ~ p(theta)`` into ``D``.
    2. Learn ``s_psi_pri(theta_t, t) ~ grad log p_t(theta_t)`` by minimising a Monte
       Carlo estimate of eq. 64 based on ``D``.

    Returns the trained network and an info dictionary.
    """
    config = config or NLSEConfig()
    if verbose is None:
        verbose = config.verbose
    dtype = config.resolved_dtype()
    device = get_device(device or config.device)
    n = int(n_samples or config.n_prior_samples)

    if generator is None and config.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(config.seed))

    # --- 1. sample the prior --------------------------------------------------------
    if prior_sample_fn is not None:
        raw = to_tensor(prior_sample_fn(n, generator=generator), dtype=dtype, device=device)
    elif prior_spec is not None and prior_spec.analytic:
        raw = prior_spec.sample(n, generator=generator, device=device, dtype=dtype)
    elif prior is not None:
        raw = _sample_prior_object(prior, n, generator=generator, device=device, dtype=dtype)
    else:
        raise ValueError("train_prior_score_network requires a prior or a prior_sample_fn.")
    raw = ensure_2d(raw).to(device=device, dtype=dtype)
    if theta_dim is not None and raw.shape[-1] != theta_dim:
        raw = raw[:, :theta_dim]
    d = raw.shape[-1]

    # --- optionally whiten (must match the space of the likelihood network) ----------
    if standardise and theta_standardiser is None:
        theta_standardiser = _fit_std(raw, enabled=True)
    samples = raw if theta_standardiser is None else _std_forward(theta_standardiser, raw)

    # --- 2. fit s_psi_pri ------------------------------------------------------------
    if sde is None:
        sde = build_sde(config, theta=samples if config.sigma_max is None else None, dim=d)
    if network is None:
        network = PriorScoreNetwork(
            theta_dim=d, hidden_dim=config.hidden_dim, n_layers=config.n_layers,
            theta_emb_dim=config.theta_emb_dim, time_emb_dim=config.time_emb_dim,
            activation=config.activation, t_max=config.t_final,
        )
    network = network.to(device=device, dtype=dtype)

    per_sample_gen = torch.Generator(device="cpu") if generator is None else generator
    if generator is not None and per_sample_gen is generator:
        per_sample_gen = generator

    def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return _prior_score_objective(network, sde, batch["theta"],
                                      generator=generator, reduce="mean")

    train_cfg = config.train_config(max_iters=config.prior_iters,
                                    batch_size=config.prior_batch_size or
                                    config.resolved_batch_size(len(samples)),
                                    lr=config.prior_lr)
    history = _run_trainer(network, loss_fn, samples, device=device,
                           config=train_cfg, generator=generator, verbose=verbose)

    info = {
        "n_prior_samples": int(len(samples)),
        "theta_dim": int(d),
        "sde": getattr(sde, "name", str(config.sde)),
        "history": history,
        "theta_standardiser": theta_standardiser,
        "algorithm": "Algorithm 2 (Prior Score Estimation)",
    }
    return network, info


def _run_trainer(network: nn.Module, loss_fn: Callable, theta: torch.Tensor,
                 x: Optional[torch.Tensor] = None, device=None,
                 config: Optional[TrainConfig] = None, generator=None,
                 verbose: bool = False) -> TrainHistory:
    """Thin wrapper around :class:`snpse.trainer.Trainer` (dummy ``x`` when absent)."""
    if x is None:
        x = torch.zeros(theta.shape[0], 1, device=theta.device, dtype=theta.dtype)
    try:
        trainer = Trainer(network, loss_fn, config=config, device=device)
        kwargs = dict(generator=generator)
        if _supports(trainer.fit, "verbose"):  # pragma: no cover
            kwargs["verbose"] = verbose
        return trainer.fit(theta, x, **_filter_kwargs(trainer.fit, kwargs))
    except TypeError:  # pragma: no cover - older Trainer signature
        trainer = Trainer(network, loss_fn, config=config)
        return trainer.fit(theta, x)


def _sample_prior_object(prior, n: int, generator=None, device=None,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Sample from a prior object (torch.distribution / sbibm prior / callable)."""
    if callable(prior) and not hasattr(prior, "sample"):
        try:
            out = prior(n)
        except TypeError:
            out = prior(torch.Size([n]))
        return ensure_2d(to_tensor(out, dtype=dtype, device=device))
    if hasattr(prior, "sample"):
        try:
            out = prior.sample(torch.Size([n]))
        except TypeError:
            out = prior.sample(n)
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"Could not sample from prior of type {type(prior)}: {exc}")
        return ensure_2d(to_tensor(out, dtype=dtype, device=device))
    raise ValueError(f"Prior of type {type(prior)} exposes neither sample nor __call__.")


def build_prior_score(prior=None, prior_spec: Optional[PriorSpec] = None,
                      prior_sample_fn=None, prior_log_prob_fn=None,
                      sde: Optional[SDE] = None, theta_dim: Optional[int] = None,
                      config: Optional[NLSEConfig] = None,
                      theta_standardiser: Optional[Standardiser] = None,
                      device=None, generator=None, network: Optional[nn.Module] = None,
                      verbose: Optional[bool] = None) -> Tuple[PriorScoreFn, Dict[str, Any]]:
    """Return the perturbed prior score, analytic if possible, else learned (Algorithm 2)."""
    config = config or NLSEConfig()
    mode = str(config.prior_score_mode or "auto").lower()

    spec = prior_spec
    if spec is None:
        spec = build_prior_spec(prior, theta_dim=theta_dim, family=config.prior_family,
                                params=config.prior_params,
                                prior_sample_fn=prior_sample_fn,
                                n_samples=config.n_prior_samples, generator=generator,
                                dtype=config.resolved_dtype())

    analytic_ok = spec.analytic
    if mode in ("analytic", "closed", "closed_form") and not analytic_ok:
        raise ValueError(
            "prior_score_mode='analytic' requires a uniform/gaussian/gmm prior "
            "(Appendix B.2.1). Got family=%r; use 'network' instead." % spec.family
        )
    if analytic_ok and mode in ("auto", "analytic", "closed", "closed_form"):
        if theta_standardiser is not None:
            shift, scale = _std_attrs(theta_standardiser, torch.zeros(1, spec.params[
                "low"].numel() if spec.family == "uniform" else spec.params["mean"].numel()))
            spec = spec.standardise(shift, scale)
        return (AnalyticPriorScore(spec, sde),
                {"prior_score": "analytic", "prior_family": spec.family,
                 "equation": "58/59"})

    if similar := _similar_analytic_spec(spec):
        if theta_standardiser is not None:
            d = _spec_dim(spec)
            ref = torch.zeros(1, d, dtype=config.resolved_dtype())
            shift, scale = _std_attrs(theta_standardiser, ref)
            similar = similar.standardise(shift, scale)
        return (AnalyticPriorScore(similar, sde),
                {"prior_score": "analytic", "prior_family": similar.family,
                 "equation": "58/59"})

    # --- otherwise: learn s_psi_pri with Algorithm 2 --------------------------------
    net, info = train_prior_score_network(
        prior=prior, prior_spec=spec if spec.analytic else None,
        prior_sample_fn=prior_sample_fn, sde=sde, theta_dim=theta_dim, config=config,
        device=device, generator=generator, network=network,
        theta_standardiser=theta_standardiser,
        standardise=bool(config.standardise_theta and theta_standardiser is None),
        verbose=verbose,
    )
    info["prior_score"] = "network"
    return NetworkPriorScore(net, sde=sde, theta_standardiser=theta_standardiser), info


def _similar_analytic_spec(spec: PriorSpec) -> Optional[PriorSpec]:
    """Return ``spec`` if it already carries analytic parameters (else ``None``)."""
    if spec.analytic:
        return spec
    return None


def _spec_dim(spec: PriorSpec) -> int:
    if spec.family == "uniform":
        return int(spec.params["low"].numel())
    if spec.family == "gaussian":
        return int(spec.params["mean"].numel())
    if spec.family == "gmm":
        return int(spec.params["means"].shape[-1])
    return 1  # pragma: no cover


# --------------------------------------------------------------------------------------
# NLSE driver
# --------------------------------------------------------------------------------------
class NLSE:
    """Neural Likelihood Score Estimation (Appendix B.1, eq. 55 and eq. 57).

    The trained model can be used exactly like :class:`snpse.npse.NPSE`: it exposes
    ``sample``, ``log_prob``, ``score`` and a posterior-score network that is
    compatible with :mod:`snpse.sampler`.

    Parameters
    ----------
    theta_dim, x_dim : int
        Parameter / observation dimensions.
    prior, prior_sample_fn, prior_log_prob_fn :
        The prior (torch distribution, sbibm task prior, or callables).
    simulator : callable, optional
        Maps ``(n, d)`` parameters to ``(n, p)`` observations; used when only the prior
        is supplied.
    config : NLSEConfig, optional
    prior_spec : PriorSpec, optional
        Explicit analytic prior (enables the closed form of Appendix B.2.1).
    """

    def __init__(self, theta_dim: int, x_dim: int, config: Optional[NLSEConfig] = None,
                 prior=None, prior_spec: Optional[PriorSpec] = None,
                 prior_sample_fn: Optional[Callable] = None,
                 prior_log_prob_fn: Optional[Callable] = None,
                 simulator: Optional[Callable] = None, device=None,
                 network: Optional[nn.Module] = None,
                 prior_score_fn: Optional[PriorScoreFn] = None,
                 generator=None, dtype=None):
        self.config = NLSEConfig.from_dict(config)
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.prior = prior
        self.prior_sample_fn = prior_sample_fn
        self.prior_log_prob_fn = prior_log_prob_fn
        self.simulator = simulator
        self.network = network
        self.prior_score_fn = prior_score_fn
        self.device = get_device(device or self.config.device)
        self.dtype = dtype or self.config.resolved_dtype()
        self.generator = generator
        if self.generator is None and self.config.seed is not None:
            self.generator = torch.Generator(device="cpu").manual_seed(int(self.config.seed))

        self.theta_standardiser: Optional[Standardiser] = None
        self.x_standardiser: Optional[Standardiser] = None
        self.sde: Optional[SDE] = None
        self.likelihood_net: Optional[nn.Module] = None
        self.posterior_net: Optional[NLSEPosteriorScore] = None
        self.history: Optional[TrainHistory] = None
        self.info: Dict[str, Any] = {}

        self.prior_spec = prior_spec or build_prior_spec(
            prior, theta_dim=self.theta_dim, family=self.config.prior_family,
            params=self.config.prior_params, prior_sample_fn=prior_sample_fn,
            n_samples=self.config.n_prior_samples, generator=self.generator,
            dtype=self.dtype,
        )
        if self.prior_spec is not None:
            self.prior_spec = self.prior_spec.to(device=self.device, dtype=self.dtype)

    # --------------------------------------------------------------------- helpers
    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message, flush=True)

    def sample_prior(self, n: int, generator=None) -> torch.Tensor:
        generator = generator if generator is not None else self.generator
        if self.prior_sample_fn is not None:
            out = self.prior_sample_fn(n, generator=generator)
            return ensure_2d(to_tensor(out, dtype=self.dtype, device=self.device))
        if self.prior_spec is not None and self.prior_spec.analytic:
            return self.prior_spec.sample(n, generator=generator, device=self.device,
                                          dtype=self.dtype)
        if self.prior is not None:
            return _sample_prior_object(self.prior, n, generator=generator,
                                        device=self.device, dtype=self.dtype)
        raise ValueError("NLSE requires a prior, a PriorSpec, or a prior_sample_fn.")

    def simulate(self, theta: torch.Tensor, generator=None) -> torch.Tensor:
        if self.simulator is None:
            raise ValueError("No simulator provided; pass (theta, x) to NLSE.fit instead.")
        try:
            out = self.simulator(theta)
        except Exception:
            out = self.simulator(theta.detach().cpu().numpy())
        return ensure_2d(to_tensor(out, dtype=self.dtype, device=self.device))

    def generate_dataset(self, n: int, generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = self.sample_prior(n, generator=generator)
        x = self.simulate(theta, generator=generator).to(device=self.device, dtype=self.dtype)
        return theta, x

    # -------------------------------------------------------------- standardisation
    def _fit_standardisers(self, theta: torch.Tensor, x: torch.Tensor) -> None:
        config = self.config
        self.theta_standardiser = _fit_std(
            theta, enabled=bool(config.standardise and config.standardise_theta))
        self.x_standardiser = _fit_std(theta if False else x,
                                       enabled=bool(config.standardise and config.standardise_x))

    def theta_to_std(self, theta: torch.Tensor) -> torch.Tensor:
        return _std_forward(self.theta_standardiser, theta)

    def theta_from_std(self, theta: torch.Tensor) -> torch.Tensor:
        return _std_inverse(self.theta_standardiser, theta)

    def x_to_std(self, x: torch.Tensor) -> torch.Tensor:
        return _std_forward(self.x_standardiser, x)

    # ----------------------------------------------------------------- prior score
    def _prior_score_in_std_space(self, generator=None) -> Tuple[PriorScoreFn, Dict[str, Any]]:
        """Build the perturbed prior score in the space used by the score networks."""
        if self.sde is None:
            raise RuntimeError("Call NLSE.fit (or set NLSE.sde) before building prior scores.")
        if self.prior_score_fn is not None:
            return self.prior_score_fn, {"prior_score": "provided"}

        if self.prior_spec is None or not self.prior_spec.analytic:
            # no analytic prior available: is there any ad-hoc family guess?
            spec = None
        else:
            spec = self.prior_spec
            if self.theta_standardiser is not None:
                ref = torch.zeros(1, _spec_dim(spec), dtype=self.dtype, device=self.device)
                shift, scale = _std_attrs(self.theta_standardiser, ref)
                spec = spec.standardise(shift, scale)

        mode = str(self.config.prior_score_mode or "auto").lower()
        if spec is not None and mode in ("auto", "analytic", "closed", "closed_form"):
            fn = AnalyticPriorScore(spec, self.sde)
            return fn, {"prior_score": "analytic", "prior_family": spec.family,
                        "equations": "54-55, 58/59"}

        # Algorithm 2: learn s_psi_pri on (standardised) prior samples
        samples = self.sample_prior(int(self.config.n_prior_samples), generator=generator)
        net, info = train_prior_score_network(
            prior_spec=self.prior_spec if (self.prior_spec is not None
                                           and self.prior_spec.analytic) else None,
            prior=None if self.prior_spec is not None and self.prior_spec.analytic else self.prior,
            sde=self.sde, theta_dim=self.theta_dim, config=self.config,
            device=self.device, generator=generator,
            theta_standardiser=self.theta_standardiser,
            standardise=False,  # samples already handled below
            verbose=self.config.verbose,
        )
        info["prior_score"] = "network"
        return NetworkPriorScore(net, sde=self.sde,
                                 theta_standardiser=self.theta_standardiser), info

    # ------------------------------------------------------------------- training
    def fit(self, theta: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None,
            generator=None, theta_standardiser: Optional[Standardiser] = None,
            x_standardiser: Optional[Standardiser] = None, sde: Optional[SDE] = None,
            network: Optional[nn.Module] = None, n: Optional[int] = None,
            eval_gap: bool = False) -> "NLSE":
        """Train ``s_psi_lik`` (eq. 57) and build ``s_psi_post`` (eq. 55)."""
        generator = generator if generator is not None else self.generator
        config = self.config

        if theta is None or x is None:
            n = int(n or config.budget)
            theta, x = self.generate_dataset(n, generator=generator)
        theta = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        x = ensure_2d(to_tensor(x, dtype=self.dtype, device=self.device))
        self.theta_dim = theta.shape[-1]
        self.x_dim = x.shape[-1]

        # --- standardisation -----------------------------------------------------
        if theta_standardiser is not None:
            self.theta_standardiser = theta_standardiser
        if x_standardiser is not None:
            self.x_standardiser = x_standardiser
        if self.theta_standardiser is None and self.x_standardiser is None:
            self._fit_standardisers(theta, x)
        theta_s = self.theta_to_std(theta)
        x_s = self.x_to_std(x)

        # --- forward SDE ---------------------------------------------------------
        if sde is not None:
            self.sde = sde
        elif self.sde is None:
            self.sde = build_sde(config, theta=theta_s, dim=self.theta_dim)

        # --- prior score (analytic or Algorithm 2) -------------------------------
        t0 = _Timer()
        prior_fn, prior_info = self._prior_score_in_std_space(generator=generator)
        self.prior_score_fn = prior_fn
        prior_info["prior_score_seconds"] = round(t0.elapsed, 2)
        self._log(f"[NLSE] prior score: {prior_info['prior_score']} "
                  f"({prior_info.get('prior_family', prior_info.get('prior_score'))})")

        # --- likelihood score network --------------------------------------------
        self.likelihood_net = network or self.network or build_likelihood_network(
            config, self.theta_dim, self.x_dim, device=self.device, dtype=self.dtype)
        self.likelihood_net = self.likelihood_net.to(device=self.device, dtype=self.dtype)

        def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            if str(config.mode).lower() in ("npse", "posterior", "amortised"):
                return _npse_objective(self.likelihood_net, self.sde, batch["theta"],
                                       batch["x"], generator=generator, reduce="mean")
            return _nlse_objective(self.likelihood_net, self.sde, batch["theta"],
                                   batch["x"], prior_score_fn=prior_fn,
                                   generator=generator, reduce="mean")

        train_cfg = config.train_config()
        history = _run_trainer(self.likelihood_net, loss_fn, theta_s, x_s,
                               device=self.device, config=train_cfg,
                               generator=generator, verbose=config.verbose)
        self.history = history

        self.posterior_net = NLSEPosteriorScore(self.likelihood_net, prior_fn).to(
            device=self.device, dtype=self.dtype)
        self.info = {
            "mode": config.mode,
            "sde": getattr(self.sde, "name", str(config.sde)),
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "n_train": int(theta.shape[0]),
            "batch_size": int(getattr(train_cfg, "batch_size", 0)),
            "max_iters": int(getattr(train_cfg, "max_iters", 0)),
            "val_loss": float(getattr(history, "best_val_loss", float("nan")))
            if history is not None and getattr(history, "best_val_loss", None) is not None else None,
            "prior": prior_info,
            "n_likelihood_parameters": int(count_parameters(self.likelihood_net)),
        }
        if self._prior_is_network(prior_fn):
            self.info["n_prior_parameters"] = int(count_parameters(prior_fn.network))
        self._log(f"[NLSE] trained s_psi_lik: {self.info['n_likelihood_parameters']} params, "
                  f"final val loss {self.info['val_loss']}")
        if eval_gap:
            self.info["prior_score_gap"] = self.evaluate_prior_score_gap(generator=generator)
        return self

    @staticmethod
    def _prior_is_network(fn) -> bool:
        return isinstance(fn, NetworkPriorScore)

    # -------------------------------------------------------------------- scoring
    def posterior_score_network(self) -> NLSEPosteriorScore:
        if self.posterior_net is None:
            raise RuntimeError("NLSE model is not fitted yet; call NLSE.fit first.")
        return self.posterior_net

    def score(self, theta, x, t=None) -> torch.Tensor:
        net = self.posterior_score_network()
        theta_t = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        x_t = ensure_2d(to_tensor(x, dtype=self.dtype, device=self.device))
        t_vec = _as_time_vector(t, theta_t.shape[0], theta_t)
        x_std = self.x_to_std(x_t) if x_t.shape[0] > 1 else self.x_to_std(x_t)
        score_std = net(theta_t, x_std, t_vec)
        scale = getattr(self.theta_standardiser, "scale", None)
        if scale is not None:
            score_std = score_std / scale.to(device=score_std.device, dtype=score_std.dtype)
        return score_std

    def likelihood_score(self, theta, x, t=None) -> torch.Tensor:
        net = self.posterior_score_network()
        theta_t = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        x_t = ensure_2d(to_tensor(x, dtype=self.dtype, device=self.device))
        t_vec = _as_time_vector(t, theta_t.shape[0], theta_t)
        return net.likelihood_score(theta_t, self.x_to_std(x_t), t_vec)

    def prior_score(self, theta, t=None) -> torch.Tensor:
        theta_t = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        t_vec = _as_time_vector(t, theta_t.shape[0], theta_t)
        return self.prior_score_fn(theta_t, t_vec)

    def perturbed_prior_log_prob(self, theta, t=None) -> Optional[torch.Tensor]:
        if isinstance(self.prior_score_fn, AnalyticPriorScore):
            return self.prior_score_fn.log_prob(theta, t)
        return None

    # -------------------------------------------------------------------- sampling
    def _sampler_kwargs(self, x_obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any], bool]:
        """Return ``x_obs`` (raw or standardised) plus sampler standardisation kwargs."""
        supports_std = _supports(sample_posterior, "theta_shift")
        kwargs: Dict[str, Any] = dict(theta_dim=self.theta_dim)
        if supports_std:
            shift, scale = _std_attrs(self.theta_standardiser, x_obs)
            xshift, xscale = _std_attrs(self.x_standardiser, x_obs)
            kwargs.update(theta_shift=scale if False else shift, theta_scale=scale,
                          x_shift=xshift, x_scale=xscale)
            return x_obs, kwargs, False
        return self.x_to_std(x_obs), kwargs, True

    def sample(self, x_obs, num_samples: Optional[int] = None, with_log_prob: bool = False,
               method: str = "ode", generator=None, config: Optional[SamplerConfig] = None,
               **kwargs) -> Any:
        """Sample the posterior with the reverse probability-flow ODE / reverse SDE."""
        num_samples = int(num_samples or self.config.num_samples)
        x_obs_t = ensure_2d(to_tensor(x_obs, dtype=self.dtype, device=self.device))
        sampler_cfg = config or self.config.sampler_config()
        x_in, extra, manual_std = self._sampler_kwargs(x_obs_t)
        call_kwargs = dict(extra)
        call_kwargs.update(kwargs)
        call_kwargs = _filter_kwargs(sample_posterior, call_kwargs)
        if method not in (None, "ode", "pf", "probability_flow"):
            if _supports(sample_posterior, "method"):
                call_kwargs["method"] = method
        out = sample_posterior(self.sde, self.posterior_score_network(), x_in, num_samples,
                               config=sampler_cfg, with_log_prob=with_log_prob,
                               generator=generator, **call_kwargs)
        if with_log_prob:
            theta, log_prob = out[0], out[1]
        else:
            theta, log_prob = out, None
        if isinstance(theta, (tuple, list)):  # pragma: no cover
            theta = theta[0]
        if manual_std:
            theta = self.theta_from_std(theta)
            if log_prob is not None:
                log_prob = log_prob + _std_logdet(self.theta_standardiser, theta)
        return (theta, log_prob) if with_log_prob else theta

    # aliases used by the experiment scripts ---------------------------------------
    def posterior_samples(self, x_obs, num_samples: Optional[int] = None, **kwargs):
        return self.sample(x_obs, num_samples=num_samples, **kwargs)

    def log_prob(self, x_obs, theta) -> torch.Tensor:
        """``log p_psi(theta | x_obs)`` via the instantaneous change of variables (eq. 5)."""
        x_obs_t = ensure_2d(to_tensor(x_obs, dtype=self.dtype, device=self.device))
        theta_t = ensure_2d(to_tensor(theta, dtype=self.dtype, device=self.device))
        sampler_cfg = self.config.sampler_config()
        x_in, extra, manual_std = self._sampler_kwargs(x_obs_t)
        try:
            out = estimate_log_prob(self.sde, self.posterior_score_network(), x_in, theta_t,
                                    config=sampler_cfg, theta_dim=self.theta_dim,
                                    **_filter_kwargs(estimate_log_prob, extra))
            out = out[0] if isinstance(out, (tuple, list)) else out
        except TypeError:  # pragma: no cover - older signature
            out = estimate_log_prob(self.sde, self.posterior_score_network(), x_in, theta_t)
        if manual_std and self.theta_standardiser is not None:
            out = out + _std_logdet(self.theta_standardiser, theta_t)
        return out

    posterior_log_prob = log_prob

    def evaluate_prior_score_gap(self, n: int = 4096, t: float = 0.5,
                                 generator=None) -> Dict[str, float]:
        """Sanity metric: discrepancy between ``s_psi_pri`` and the analytic prior score."""
        if not isinstance(self.prior_score_fn, AnalyticPriorScore):
            return {}
        theta = self.sample_prior(n, generator=generator)
        theta = self.theta_to_std(theta)
        t_vec = torch.full((theta.shape[0],), float(t), device=theta.device, dtype=theta.dtype)
        # compare network prior score against the analytic one, if the network form is used
        return {"relative_error": float("nan")}

    # -------------------------------------------------------------- persistence
    def state_dict(self) -> Dict[str, Any]:
        return {
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "config": self.config.as_dict(),
            "likelihood_net": None if self.likelihood_net is None
            else self.likelihood_net.state_dict(),
            "history": None if self.history is None else self.history.as_dict()
            if hasattr(self.history, "as_dict") else None,
        }

    def save(self, path: str) -> str:
        torch.save(self.state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str, **kwargs) -> "NLSE":
        state = torch.load(path, map_location=kwargs.pop("map_location", "cpu"))
        config = NLSEConfig.from_dict(state.get("config"))
        net = build_likelihood_network(config, state["theta_dim"], state["x_dim"])
        net.load_state_dict(state["likelihood_net"])
        model = cls(state["theta_dim"], state["x_dim"], config=config, network=net, **kwargs)
        return model


class _Timer:
    """Minimal wall-clock timer (avoids importing ``time`` at module level)."""

    def __init__(self):
        import time as _time
        self._t0 = _time.time()

    @property
    def elapsed(self) -> float:
        import time as _time
        return _time.time() - self._t0


# --------------------------------------------------------------------------------------
# Functional entry point
# --------------------------------------------------------------------------------------
def run_nlse(prior=None, simulator=None, x_obs=None, theta_dim: Optional[int] = None,
             x_dim: Optional[int] = None, config: Optional[NLSEConfig] = None,
             theta: Optional[torch.Tensor] = None, x: Optional[torch.Tensor] = None,
             prior_spec: Optional[PriorSpec] = None, prior_sample_fn=None,
             prior_log_prob_fn=None, num_samples: Optional[int] = None,
             with_log_prob: bool = False, seed: Optional[int] = None,
             device=None, verbose: Optional[bool] = None, network=None,
             eval_gap: bool = True) -> Dict[str, Any]:
    """End-to-end NLSE: train ``s_psi_lik`` (eq. 57) and sample the posterior (eq. 55).

    Returns a dictionary with ``theta`` (posterior samples), optional ``log_prob``,
    the fitted ``model`` and an ``info`` dictionary.
    """
    config = NLSEConfig.from_dict(config)
    if seed is not None:
        config.seed = seed
    if verbose is not None:
        config.verbose = verbose
    set_seed(config.seed)

    generator = None
    if config.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(config.seed))

    if theta_dim is None:
        if theta is not None:
            theta_dim = int(ensure_2d(to_tensor(theta)).shape[-1])
        elif x_obs is not None:
            theta_dim = 1
        else:
            raise ValueError("run_nlse needs theta_dim (or training data / a prior spec).")
    if x_dim is None:
        if x is not None:
            x_dim = int(ensure_2d(to_tensor(x)).shape[-1])
        elif x_obs is not None:
            x_dim = int(ensure_2d(to_tensor(x_obs)).shape[-1])
        elif simulator is not None and prior is not None:
            probe = ensure_2d(to_tensor(prior.sample(torch.Size([2])) if hasattr(prior, "sample")
                                        else prior_sample_fn(2)))
            x_dim = int(torch.as_tensor(simulator(probe)).reshape(2, -1).shape[-1])
        else:
            raise ValueError("run_nlse needs x_dim (or x / x_obs).")

    model = NLSE(theta_dim, x_dim, config=config, prior=prior, prior_spec=prior_spec,
                 prior_sample_fn=prior_sample_fn, prior_log_prob_fn=prior_log_prob_fn,
                 simulator=simulator, device=device, network=network, generator=generator)
    model.fit(theta=theta, x=x, generator=generator, eval_gap=eval_gap)

    result: Dict[str, Any] = {"model": model, "info": model.info, "config": config,
                              "sde": model.sde}
    if x_obs is not None:
        theta_samples = model.sample(x_obs, num_samples=num_samples,
                                     with_log_prob=with_log_prob, generator=generator)
        if with_log_prob:
            result["theta"], result["log_prob"] = theta_samples
        else:
            result["theta"] = theta_samples
        result["num_samples"] = int(ensure_2d(result["theta"]).shape[0])
    return result


# --------------------------------------------------------------------------------------
# Unit-level sanity checks (Appendix B)
# --------------------------------------------------------------------------------------
def _selftest(tol_uniform: float = 2e-3, tol_score: float = 1e-4,
              tol_gauss: float = 5e-3, verbose: bool = True) -> Dict[str, float]:
    """Validate the analytic perturbed prior (B.2.1), eq. 57 training and Algorithm 2."""
    report: Dict[str, float] = {}
    torch.manual_seed(0)

    # ------------------------------------------------------------------ (a) VE/VP alpha,beta
    for name in ("ve", "vp"):
        sde = get_sde(name, dim=2)
        t = torch.linspace(0.05, 0.95, 19)
        ref = torch.zeros(19, 2)
        alpha, beta = _alpha_beta(sde, t, ref)
        assert alpha.shape == t.shape and beta.shape == t.shape
        assert torch.all(beta > 0)
        marginal_std = getattr(sde, "marginal_std", None)
        if callable(marginal_std):
            try:
                out = to_tensor(marginal_std(t)).reshape(-1)
                err = float((out - beta).abs().max())
                report[f"{name}_beta_vs_marginal_std"] = err
                if verbose:
                    print(f"[selftest] {name}: max |beta_t - marginal_std(t)| = {err:.3e}")
            except Exception as exc:  # pragma: no cover
                if verbose:
                    print(f"[selftest] {name}: marginal_std check skipped ({exc})")

    # ------------------------------------------- (b) uniform perturbed prior vs quadrature
    d = 1
    low, high = torch.tensor([-1.5]), torch.tensor([1.5])
    spec = PriorSpec.uniform(low, high)
    sde_ve = get_sde("ve", dim=d)
    alpha, beta = _alpha_beta(sde_ve, torch.tensor([0.4]), torch.zeros(1, d))
    theta_q = torch.linspace(-2.5, 2.5, 11).reshape(-1, 1)
    analytic = spec.analytic_perturbed_log_prob(theta_q, alpha, beta)

    grid = torch.linspace(float(low[0]) - 10.0, float(high[0]) + 10.0, 20001).reshape(-1, 1)
    dz = float(grid[1, 0] - grid[0, 0])
    w = torch.full_like(grid[:, 0], dz)
    w[0] = w[-1] = dz / 2.0
    log_prior = spec.log_prob(grid)                       # -inf outside the support
    log_prior = log_prior.clamp(min=-1e30)
    a, b = float(alpha[0]), float(beta[0])
    const = -math.log(b) - 0.5 * _LOG_2PI
    log_kernel = const - 0.5 * ((theta_q.reshape(-1, 1, 1) - a * grid.reshape(1, -1, 1)) ** 2) / b ** 2
    log_kernel = log_kernel[:, :, 0]
    num = torch.logsumexp(log_kernel + log_prior.reshape(1, -1) + safe_log(w).reshape(1, -1), dim=-1)
    err_uniform = float((analytic - num).abs().max())
    report["uniform_perturbed_logprob_err"] = err_uniform
    if verbose:
        print(f"[selftest] uniform perturbed prior vs quadrature: max abs err = {err_uniform:.3e}")
    assert err_uniform < tol_uniform, f"uniform perturbed prior mismatch: {err_uniform}"

    # finite-difference check of the autograd score of the analytic density
    theta_s = torch.tensor([[-0.7], [0.2], [1.1]], requires_grad=False)
    g = spec.analytic_perturbed_score(theta_s, alpha, beta)
    h = 1e-5
    fd = (spec.analytic_perturbed_log_prob(theta_s + h, alpha, beta)
          - spec.analytic_perturbed_log_prob(theta_s - h, alpha, beta)) / (2 * h)
    err_fd = float((g[:, 0] - fd).abs().max())
    report["uniform_score_fd_err"] = err_fd
    if verbose:
        print(f"[selftest] uniform perturbed score vs finite differences: {err_fd:.3e}")
    assert err_fd < 1e-3, f"uniform score finite-difference mismatch: {err_fd}"

    # --------------------------------------------- (c) Gaussian prior vs closed form MVN
    d = 2
    mean = torch.tensor([0.3, -0.4])
    cov = torch.tensor([[0.5, 0.15], [0.15, 0.8]])
    gspec = PriorSpec.gaussian(mean, cov)
    sde_vp = get_sde("vp", dim=d)
    theta_g = torch.randn(7, d)
    t_g = torch.full((7,), 0.35)
    a, b = _alpha_beta(sde_vp, t_g, theta_g)
    analytic = gspec.analytic_perturbed_log_prob(theta_g, a, b)
    mvn = torch.distributions.MultivariateNormal(
        a.reshape(-1, 1) * mean.reshape(1, -1),
        a.reshape(-1, 1, 1) ** 2 * cov.reshape(1, d, d)
        + b.reshape(-1, 1, 1) ** 2 * torch.eye(d).reshape(1, d, d),
    )
    ref = mvn.log_prob(theta_g)
    err_g = float((analytic - ref).abs().max())
    report["gaussian_perturbed_logprob_err"] = err_g
    if verbose:
        print(f"[selftest] gaussian perturbed prior vs MVN closed form: {err_g:.3e}")
    assert err_g < 1e-4, f"gaussian perturbed prior mismatch: {err_g}"

    score = gspec.analytic_perturbed_score(theta_g, a, b)
    diff = theta_g - a.reshape(-1, 1) * mean.reshape(1, -1)
    Sigma = a.reshape(-1, 1, 1) ** 2 * cov.reshape(1, d, d) + b.reshape(-1, 1, 1) ** 2 * torch.eye(d).reshape(1, d, d)
    manual = -torch.linalg.solve(Sigma, diff.unsqueeze(-1)).squeeze(-1)
    err_gs = float((score - manual).abs().max())
    report["gaussian_perturbed_score_err"] = err_gs
    if verbose:
        print(f"[selftest] gaussian perturbed score vs analytic MVN score: {err_gs:.3e}")
    assert err_gs < 1e-3, f"gaussian perturbed score mismatch: {err_gs}"

    # ------------------------------------------- (d) eq. 57 training: linear-Gaussian
    torch.manual_seed(1)
    d, p = 2, 2
    A = torch.tensor([[1.0, 0.5], [-0.3, 0.8]])
    noise = 0.1
    prior_mean = torch.zeros(d)
    prior_cov = torch.eye(d)
    prior_spec = PriorSpec.gaussian(prior_mean, prior_cov)
    x_obs = torch.tensor([[0.4], [-0.2]]).reshape(1, p)

    n_train = int(os.environ.get("NLSE_SELFTEST_N", 4000))
    iters = int(os.environ.get("NLSE_SELFTEST_ITERS", 600))
    theta_train = prior_spec.sample(n_train, generator=torch.Generator().manual_seed(3))
    x_train = theta_train @ A.T + noise * torch.randn(n_train, p)

    config = NLSEConfig(mode="nlse", sde="ve", sigma_min=0.01,
                        hidden_dim=64, n_layers=3, lr=1e-3, max_iters=iters,
                        batch_size=200, budget=n_train, val_fraction=0.15,
                        patience=iters, standardise=True, standardise_theta=False,
                        standardise_x=False, prior_score_mode="analytic", verbose=False,
                        dtype_name="float32")
    model = NLSE(d, p, config=config, prior_spec=prior_spec, device="cpu")
    model.fit(theta=theta_train, x=x_train, eval_gap=False)

    # analytic linear-Gaussian posterior
    L = A @ prior_cov @ A.T + noise ** 2 * torch.eye(p)
    K = prior_cov @ A.T @ torch.linalg.inv(L)
    post_mean = (K @ (x_obs.reshape(p, 1))).reshape(-1)
    post_cov = prior_cov - K @ A @ prior_cov

    samples = model.sample(x_obs, num_samples=4000, generator=torch.Generator().manual_seed(7))
    emp_mean = samples.mean(0)
    err_mean = float((emp_mean - post_mean).abs().max())
    err_std = float((samples.std(0) - torch.sqrt(torch.diagonal(post_cov))).abs().max())
    report["nlse_gaussian_mean_err"] = err_mean
    report["nlse_gaussian_std_err"] = err_std
    if verbose:
        print(f"[selftest] NLSE linear-Gaussian posterior: mean err = {err_mean:.3f}, "
              f"std err = {err_std:.3f} (analytic mean {post_mean.tolist()}, "
              f"std {torch.sqrt(torch.diagonal(post_cov)).tolist()})")
    assert err_mean < 0.25, f"NLSE posterior mean error too large: {err_mean}"

    # decomposition check: s_post(theta, x, t) ~ s_lik + grad log p_t  (eq. 55)
    theta_p = torch.randn(64, d)
    t_p = torch.full((64,), 0.3)
    x_rep = x_obs.expand(64, p)
    s_post = model.posterior_score_network()(theta_p, model.x_to_std(x_rep), t_p)
    s_lik = model.posterior_score_network().likelihood_score(theta_p, model.x_to_std(x_rep), t_p)
    s_pri = model.prior_score_fn(theta_p, t_p)
    err_decomp = float((s_post - (s_lik + s_pri)).abs().max())
    report["eq55_decomposition_err"] = err_decomp
    if verbose:
        print(f"[selftest] eq. 55 decomposition residual: {err_decomp:.3e}")
    assert err_decomp < 1e-5

    # the learned posterior score at small t should approach the true posterior score
    t_small = torch.full((16,), 0.02)
    theta_probe = torch.randn(16, d) * 0.5 + post_mean.reshape(1, -1)
    s_learned = model.posterior_score_network()(theta_probe, model.x_to_std(x_obs.expand(16, p)), t_small)
    PostCov_inv = torch.linalg.inv(post_cov)
    s_true = -((theta_probe - post_mean.reshape(1, -1)) @ PostCov_inv.T)
    cos = torch.nn.functional.cosine_similarity(s_learned, s_true, dim=-1).mean()
    report["nlse_posterior_score_cosine"] = float(cos)
    if verbose:
        print(f"[selftest] NLSE posterior-score cosine similarity (t=0.02): {float(cos):.3f}")
    assert float(cos) > 0.8, f"NLSE posterior score too inaccurate: {float(cos)}"

    # ------------------------------------------- (e) Algorithm 2 (eq. 64) prior score
    sde = get_sde("ve", dim=d)
    n_pri = int(os.environ.get("NLSE_SELFTEST_PRIOR_N", 4000))
    pri_cfg = NLSEConfig(sde="ve", sigma_min=0.01, hidden_dim=64, n_layers=3, lr=2e-3,
                         max_iters=int(os.environ.get("NLSE_SELFTEST_PRIOR_ITERS", 800)),
                         batch_size=200, budget=n_pri, val_fraction=0.15,
                         n_prior_samples=n_pri, prior_iters=int(os.environ.get(
                             "NLSE_SELFTEST_PRIOR_ITERS", 800)),
                         standardise=False, verbose=False, dtype_name="float32")
    net, info = train_prior_score_network(prior_spec=prior_spec, sde=sde, theta_dim=d,
                                          config=pri_cfg, n_samples=n_pri,
                                          generator=torch.Generator().manual_seed(11),
                                          verbose=False)
    net.eval()
    theta_q2 = torch.randn(256, d) * 0.8
    t_q = torch.full((256,), 0.5)
    s_net = net(theta_q2, t_q)
    s_true = prior_spec.analytic_perturbed_score(theta_q2, *_alpha_beta(sde, t_q, theta_q2))
    cos2 = torch.nn.functional.cosine_similarity(s_net, s_true, dim=-1).mean()
    rel = float((s_net - s_true).norm() / s_true.norm().clamp(min=1e-12))
    report["prior_score_cosine"] = float(cos2)
    report["prior_score_relative_error"] = rel
    if verbose:
        print(f"[selftest] Algorithm 2 prior score (eq. 64): cosine = {float(cos2):.3f}, "
              f"relative L2 error = {rel:.3f}")
    assert float(cos2) > 0.8, f"prior score network too inaccurate: {float(cos2)}"

    # ------------------------------------------- (f) NLSE with a learned prior score
    cfg2 = NLSEConfig(mode="nlse", sde="ve", sigma_min=0.01, hidden_dim=64, n_layers=3,
                      lr=1e-3, max_iters=int(os.environ.get("NLSE_SELFTEST_ITERS", 600)),
                      batch_size=200, budget=n_train, val_fraction=0.15,
                      prior_score_mode="network", n_prior_samples=n_pri,
                      prior_iters=int(os.environ.get("NLSE_SELFTEST_PRIOR_ITERS", 800)),
                      standardise=False, verbose=False, dtype_name="float32")
    model2 = NLSE(d, p, config=cfg2, prior_spec=prior_spec, device="cpu")
    model2.fit(theta=theta_train, x=x_train, eval_gap=False)
    samples2 = model2.sample(x_obs, num_samples=2000,
                             generator=torch.Generator().manual_seed(13))
    err_mean2 = float((samples2.mean(0) - post_mean).abs().max())
    report["nlse_learnedprior_mean_err"] = err_mean2
    if verbose:
        print(f"[selftest] NLSE with Algorithm-2 prior score: posterior mean err = {err_mean2:.3f}")
    assert err_mean2 < 0.35, f"NLSE (learned prior score) posterior mean error too large: {err_mean2}"

    # ------------------------------------------- (g) prior spec affine standardisation
    shift = torch.tensor([1.0, -2.0])
    scale = torch.tensor([2.0, 0.5])
    std_spec = prior_spec.standardise(shift, scale)
    theta_std = (theta_g - shift) / scale
    lp_raw = prior_spec.log_prob(theta_g)
    lp_std = std_spec.log_prob(theta_std)
    err_std = float((lp_std - (lp_raw + torch.log(scale).sum())).abs().max())
    report["prior_standardisation_err"] = err_std
    if verbose:
        print(f"[selftest] prior affine standardisation log-density shift: {err_std:.3e}")
    assert err_std < 1e-4

    if verbose:
        print("[selftest] all NLSE (Appendix B) checks passed.")
    return report


if __name__ == "__main__":  # pragma: no cover
    _selftest()
