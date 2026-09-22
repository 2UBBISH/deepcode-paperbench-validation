#!/usr/bin/env python
"""Baseline inference algorithms used in the SNPSE paper (Section 5.2 / 5.3, Appendix F).

The paper compares the proposed NPSE / TSNPSE (and the SNPSE-A/B/C, NLSE ablations)
against three families of baselines:

* **NPE** (amortised neural posterior estimation) -- run through ``sbibm`` with its
  default hyper-parameters (Section 5.2 / Appendix F of the paper).
* **SNPE-C** (sequential NPE with the automatic-posterior-transformation / APT loss,
  Greenberg et al. 2019) -- also run through ``sbibm`` with default hyper-parameters.
* **TSNPE** (Deistler et al. 2022a) -- the mackelab reference implementation
  (``github.com/mackelab/tsnpe_neurips``) is used *as is* (no re-implementation), as
  stated in the reproduction plan / Addendum.

In addition, **FMPE** numbers (Dax et al. 2023, Appendix F) are reported from the
literature (see :data:`FMPE_PUBLISHED_C2ST`) rather than re-run.

Because ``sbibm`` and the mackelab ``tsnpe`` package are optional external
dependencies, this module also ships *self-contained* fallbacks that implement the
very same baselines with plain PyTorch:

* ``npe``    : a conditional mixture-density network (MDN) trained by maximum
               likelihood on prior-predictive pairs ``(theta, x) ~ p(theta) p(x|theta)``.
* ``snpe_c`` : the same density network trained sequentially with the APT loss
               :math:`\\mathcal L = -\\sum_n \\log \\frac{q_\\phi(\\theta_n|x) w_n}{\\sum_k q_\\phi(\\theta_k|x) w_k}`,
               :math:`w_n = p(\\theta_n) / \\tilde p(\\theta_n)`, with the proposal
               :math:`\\tilde p` given by the previous posterior approximation
               (or the prior), i.e. the standard SNPE-C / APT algorithm.

The fallbacks make the benchmark experiments runnable in environments without
``sbibm``; whenever ``sbibm`` is importable the paper's own implementations are used
by default (``backend="auto"``).

Public entry points
-------------------
``run_baseline(method, task, ...)``   -- dispatcher for ``npe`` / ``snpe_c`` / ``tsnpe``
``run_npe / run_snpe_c / run_tsnpe``  -- individual methods
``MixtureDensityNetwork``             -- the fallback conditional density estimator
``MixturePosterior``                  -- mixture of previous posterior approximations
``evaluate_c2st(task, samples)``      -- C2ST score via :mod:`snpse.tasks.c2st`

CLI::

    python -m snpse.experiments.baselines --method snpe_c --task slcp --budget 10000
"""

# --------------------------------------------------------------------------------------
# imports (with robust resolution of the surrounding package layout)
# --------------------------------------------------------------------------------------
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "BaselineConfig",
    "MixtureDensityNetwork",
    "MixturePosterior",
    "PosteriorDensity",
    "run_npe",
    "run_snpe_c",
    "run_tsnpe",
    "run_baseline",
    "run_npe_sbibm",
    "run_snpe_c_sbibm",
    "sbibm_available",
    "tsnpe_available",
    "evaluate_c2st",
    "FMPE_PUBLISHED_C2ST",
    "BASELINE_METHODS",
]


# --------------------------------------------------------------------------------------
# module resolution: works both as ``snpse.experiments.baselines`` (inner package
# ``snpse.snpse``) and as a flat checkout where the modules sit on ``sys.path``.
# --------------------------------------------------------------------------------------
def _import_first(candidates: Sequence[str], required: bool = False) -> Optional[Any]:
    """Import the first importable module out of ``candidates``.

    Relative candidates (starting with ``.``) are resolved against this module's
    package; absolute ones are imported as-is.  Returns ``None`` when nothing could
    be imported and ``required`` is ``False``.
    """
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            if candidate.startswith("."):
                return importlib.import_module(candidate, __package__)
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - depends on layout
            last_error = exc
    if required:
        raise ImportError(
            f"could not import any of {list(candidates)!r}: {last_error!r}"
        )
    return None


_CORE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "trainer": ("..snpse.trainer", "snpse.trainer", "trainer"),
    "utils": ("..snpse.utils", "snpse.utils", "utils"),
    "score_network": ("..snpse.score_network", "snpse.score_network", "score_network"),
    "sampler": ("..snpse.sampler", "snpse.sampler", "sampler"),
    "sdes": ("..snpse.sdes", "snpse.sdes", "sdes"),
    "benchmarks": ("..tasks.benchmarks", "snpse.tasks.benchmarks", "tasks.benchmarks"),
    "c2st": ("..tasks.c2st", "snpse.tasks.c2st", "tasks.c2st"),
}

_MODULES: Dict[str, Optional[Any]] = {
    name: _import_first(cands) for name, cands in _CORE_CANDIDATES.items()
}


def _module(name: str) -> Any:
    """Return an already-resolved core module, raising a clear error otherwise."""
    mod = _MODULES.get(name)
    if mod is None:
        raise ImportError(
            f"the snpse module {name!r} could not be imported; make sure the repository "
            f"root (containing the 'snpse' package) is on PYTHONPATH."
        )
    return mod


# --------------------------------------------------------------------------------------
# optional external dependencies
# --------------------------------------------------------------------------------------
def _try_import(name: str) -> Optional[Any]:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


_SBIBM = _try_import("sbibm")
_SBI = _try_import("sbi")
_TSNPE = None


def sbibm_available() -> bool:
    """Whether the ``sbibm`` package (used for the NPE / SNPE-C baselines) is importable."""
    return _SBIBM is not None


def sbi_available() -> bool:
    """Whether the ``sbi`` package is importable."""
    return _SBI is not None


def tsnpe_available() -> bool:
    """Whether the mackelab ``tsnpe`` package (TSNPE baseline) is importable."""
    return _load_tsnpe_class(default=None) is not None


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
#: Methods that :func:`run_baseline` understands.
BASELINE_METHODS: Tuple[str, ...] = ("npe", "snpe_c", "snpe-c", "tsnpe", "fmpe")


@dataclass
class BaselineConfig:
    """Hyper-parameters of the fallback (pure-PyTorch) baseline implementations.

    Whenever the paper (or the reproduction plan) prescribes a value it is used as the
    default; otherwise standard ``sbibm`` / ``sbi`` defaults are adopted, as the paper
    runs the baselines "via sbibm with default hyper-parameters".
    """

    method: str = "npe"
    # data / rounds
    budget: int = 10_000
    num_rounds: int = 10
    num_samples: int = 10_000
    num_reference_samples: Optional[int] = None
    seed: int = 0
    # density-network architecture (fallback MDN)
    n_components: int = 10
    hidden_dim: int = 256
    n_layers: int = 3
    activation: str = "silu"
    min_scale: float = 1e-3
    max_scale: float = 30.0
    # optimisation (mirrors snpse.trainer defaults)
    lr: float = 1e-4
    max_iters: int = 3000
    batch_size: Optional[int] = None
    val_fraction: float = 0.15
    patience: int = 1000
    weight_decay: float = 0.0
    # sequential SNPE-C specifics
    proposal: str = "posterior"  # "posterior" (l/r mixture) or "prior"
    clip_weights: float = 10.0  # APT importance-weight clipping (sbi default style)
    proposal_mixture: str = "average"  # "average" (l/r mixture) or "latest"
    # misc
    standardise: bool = True
    device: Optional[str] = None
    dtype_name: str = "float32"
    backend: str = "auto"  # "auto" | "sbibm" | "fallback"
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved_dtype(self) -> torch.dtype:
        name = (self.dtype_name or "float32").lower()
        if name in ("float64", "double"):
            return torch.float64
        if name in ("float16", "half"):
            return torch.float16
        return torch.float32

    def resolved_batch_size(self) -> int:
        if self.batch_size:
            return int(self.batch_size)
        return _default_batch_size(self.budget)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Optional[Dict[str, Any]] = None, **overrides: Any) -> "BaselineConfig":
        cfg = cls(**(values or {}))
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg


def _default_batch_size(budget: int) -> int:
    """Paper batch sizes (50 / 200 / 500 for 1e3 / 1e4 / 1e5 simulations)."""
    trainer = _MODULES.get("trainer")
    if trainer is not None and hasattr(trainer, "select_batch_size"):
        try:
            return int(trainer.select_batch_size(int(budget)))
        except Exception:
            pass
    if budget <= 1_000:
        return 50
    if budget <= 10_000:
        return 200
    return 500


# --------------------------------------------------------------------------------------
# the fallback conditional density estimator: mixture density network q(theta | x)
# --------------------------------------------------------------------------------------
class MixtureDensityNetwork(nn.Module):
    """Conditional Gaussian mixture ``q_phi(theta | x)`` -- the fallback NPE density.

    The network maps ``x`` through an embedding MLP (``max(30, 4 * p)`` units,
    matching the SNPSE score-network embedding convention) followed by ``n_layers``
    residual-free SiLU layers, and outputs the logits, means and (log) scales of a
    diagonal Gaussian mixture over ``theta``.

    Densities are evaluated in the *standardised* space; standardisation is handled by
    :class:`PosteriorDensity`, which wraps this module.
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        n_components: int = 10,
        hidden_dim: int = 256,
        n_layers: int = 3,
        activation: str = "silu",
        min_scale: float = 1e-3,
        max_scale: float = 30.0,
    ) -> None:
        super().__init__()
        if theta_dim <= 0 or x_dim <= 0:
            raise ValueError("theta_dim and x_dim must be positive")
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.n_components = int(n_components)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.activation_name = activation

        x_emb_dim = max(30, 4 * int(x_dim))
        layers: List[nn.Module] = [
            nn.Linear(int(x_dim), x_emb_dim),
            _make_activation(activation),
        ]
        in_dim = x_emb_dim
        for _ in range(max(1, int(n_layers)) - 1):
            layers += [nn.Linear(in_dim, int(hidden_dim)), _make_activation(activation)]
            in_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.n_components * (1 + 2 * self.theta_dim))
        self.reset_parameters()

    # -- helpers -----------------------------------------------------------------
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _flatten_x(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = torch.as_tensor(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.reshape(-1, x.shape[-1])
        if x.shape[0] == 1 and batch_size > 1:
            x = x.expand(batch_size, -1)
        return x

    def parameters_tuple(
        self, x: torch.Tensor, batch_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(logits, means, scales)`` of shape ``(n, K)``, ``(n, K, d)``."""
        if batch_size is None:
            batch_size = int(x.shape[0])
        x = self._flatten_x(x, batch_size)
        features = self.trunk(x)
        out = self.head(features)
        K, d = self.n_components, self.theta_dim
        logits = out[..., :K]
        means = out[..., K : K + K * d].reshape(-1, K, d)
        raw_scale = out[..., K + K * d :].reshape(-1, K, d)
        scales = F.softplus(raw_scale) + self.min_scale
        scales = scales.clamp(max=self.max_scale)
        return logits, means, scales

    # -- density / sampling ------------------------------------------------------
    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``log q_phi(theta | x)`` -> ``(n,)`` tensor."""
        theta = torch.as_tensor(theta)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        n = int(theta.shape[0])
        logits, means, scales = self.parameters_tuple(x, batch_size=n)
        if means.shape[0] == 1 and n > 1:
            means = means.expand(n, -1, -1)
            scales = scales.expand(n, -1, -1)
            logits = logits.expand(n, -1)
        z = (theta.unsqueeze(1) - means) / scales
        log_comp = -0.5 * (z ** 2).sum(-1) - scales.log().sum(-1) - 0.5 * self.theta_dim * math.log(2 * math.pi)
        return torch.logsumexp(logits + log_comp, dim=1)

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Draw ``num_samples`` values of ``theta`` from ``q_phi(.|x)`` -> ``(n, d)``."""
        logits, means, scales = self.parameters_tuple(x, batch_size=1)
        logits = logits[0]
        means = means[0]
        scales = scales[0]
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
        mean = means[idx]
        scale = scales[idx]
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + scale * eps

    def component(
        self, x: torch.Tensor, theta_shift=None, theta_scale=None
    ) -> "GMMComponent":
        """Deterministic mixture component description of ``q_phi(.|x)``."""
        logits, means, scales = self.parameters_tuple(x, batch_size=1)
        logits, means, scales = logits[0], means[0], scales[0]
        if theta_shift is not None and theta_scale is not None:
            shift = torch.as_tensor(theta_shift, dtype=means.dtype, device=means.device).reshape(1, -1)
            scale = torch.as_tensor(theta_scale, dtype=means.dtype, device=means.device).reshape(1, -1)
            means = means * scale + shift
            scales = scales * scale
        return GMMComponent(logits=logits, means=means, scales=scales)


def _make_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("elu",):
        return nn.ELU()
    return nn.SiLU()


# --------------------------------------------------------------------------------------
# posterior density wrapper (handles standardisation) + mixture of posteriors
# --------------------------------------------------------------------------------------
class PosteriorDensity:
    """``q(theta | x)`` built on a :class:`MixtureDensityNetwork` plus standardisers.

    All evaluation happens in standardised coordinates, so the Jacobian of the affine
    transform is subtracted when reporting densities in the original theta space.
    """

    def __init__(
        self,
        network: MixtureDensityNetwork,
        theta_standardiser: Optional[Any] = None,
        x_standardiser: Optional[Any] = None,
    ) -> None:
        self.network = network
        self.theta_standardiser = theta_standardiser
        self.x_standardiser = x_standardiser

    # -- transforms --------------------------------------------------------------
    def _std_theta(self, theta: torch.Tensor) -> torch.Tensor:
        if self.theta_standardiser is None:
            return theta
        return self.theta_standardiser.to_std(theta)

    def _std_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.x_standardiser is None:
            return x
        return self.x_standardiser.to_std(x)

    def _log_det_theta(self) -> float:
        if self.theta_standardiser is None:
            return 0.0
        scale = self.theta_standardiser.scale
        return float(-torch.log(scale).sum().item())

    # -- interface ---------------------------------------------------------------
    def log_prob(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        theta_std = self._std_theta(torch.as_tensor(theta))
        x_std = self._std_x(torch.as_tensor(x))
        return self.network.log_prob(theta_std, x_std) + self._log_det_theta()

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        x_std = self._std_x(torch.as_tensor(x))
        theta_std = self.network.sample(x_std, num_samples=num_samples, generator=generator)
        if self.theta_standardiser is None:
            return theta_std
        return self.theta_standardiser.from_std(theta_std)

    def component(self, x: torch.Tensor) -> "GMMComponent":
        """Gaussian-mixture description of the posterior in *original* theta space."""
        x_std = self._std_x(torch.as_tensor(x))
        shift = None
        scale = None
        if self.theta_standardiser is not None:
            shift = self.theta_standardiser.shift
            scale = self.theta_standardiser.scale
        return self.network.component(x_std, theta_shift=shift, theta_scale=scale)


@dataclass
class GMMComponent:
    """A diagonal Gaussian mixture in theta space (used for sequential proposals)."""

    logits: torch.Tensor  # (K,)
    means: torch.Tensor  # (K, d)
    scales: torch.Tensor  # (K, d)

    @property
    def n_components(self) -> int:
        return int(self.logits.shape[0])


class MixturePosterior:
    """Equal (or weighted) mixture of previous posterior approximations ``ptilde``.

    This is the proposal used by the fallback SNPE-C implementation (and by SNPSE's
    eq. (10) in the main paper); a *mixture* of diagonal Gaussian mixtures is again a
    diagonal Gaussian mixture, so densities and sampling stay exact and cheap.
    """

    def __init__(self, prior_log_prob_fn: Optional[Callable] = None) -> None:
        self._components: List[GMMComponent] = []
        self._mixture_weights: List[float] = []
        self.prior_log_prob_fn = prior_log_prob_fn
        self._cached_logits: Optional[torch.Tensor] = None
        self._cached_means: Optional[torch.Tensor] = None
        self._cached_scales: Optional[torch.Tensor] = None

    # -- construction ------------------------------------------------------------
    def add_component(self, component: GMMComponent, weight: float = 1.0) -> None:
        self._components.append(component)
        self._mixture_weights.append(float(weight))
        self._invalidate()

    def add_posterior(self, posterior: PosteriorDensity, x: torch.Tensor, weight: float = 1.0) -> None:
        self.add_component(posterior.component(x), weight=weight)

    @property
    def is_prior(self) -> bool:
        return len(self._components) == 0

    def mixture_probs(self) -> torch.Tensor:
        weights = torch.tensor(self._mixture_weights, dtype=torch.float32)
        if weights.numel() == 0:
            raise RuntimeError("mixture has no components")
        if weights.sum() <= 0:
            weights = torch.ones_like(weights)
        return weights / weights.sum()

    # -- flattened mixture -------------------------------------------------------
    def _invalidate(self) -> None:
        self._cached_logits = None
        self._cached_means = None
        self._cached_scales = None

    def _flatten(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._cached_logits is None:
            probs = self.mixture_probs()
            logits, means, scales = [], [], []
            for weight, component in zip(probs, self._components):
                logits.append(component.logits + torch.log(weight.to(component.logits)))
                means.append(component.means)
                scales.append(component.scales)
            self._cached_logits = torch.cat(logits, dim=0)
            self._cached_means = torch.cat(means, dim=0)
            self._cached_scales = torch.cat(scales, dim=0)
        return self._cached_logits, self._cached_means, self._cached_scales

    # -- interface ---------------------------------------------------------------
    def log_prob(self, theta: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
        if self.is_prior:
            if self.prior_log_prob_fn is None:
                raise RuntimeError("prior log-probability function not provided")
            return torch.as_tensor(self.prior_log_prob_fn(torch.as_tensor(theta)))
        logits, means, scales = self._flatten()
        theta = torch.as_tensor(theta)
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        log_scales = scales.log()
        const = -0.5 * float(means.shape[1]) * math.log(2 * math.pi)
        outs: List[torch.Tensor] = []
        for start in range(0, theta.shape[0], chunk_size):
            chunk = theta[start : start + chunk_size]
            z = (chunk.unsqueeze(1) - means.unsqueeze(0)) / scales.unsqueeze(0)
            log_comp = -0.5 * (z ** 2).sum(-1) - log_scales.sum(-1).unsqueeze(0) + const
            outs.append(torch.logsumexp(logits.unsqueeze(0) + log_comp, dim=1))
        return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]

    def sample(
        self,
        num_samples: int,
        generator: Optional[torch.Generator] = None,
        prior_sample_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        if self.is_prior:
            if prior_sample_fn is None:
                raise RuntimeError("prior sampler not provided")
            return torch.as_tensor(prior_sample_fn(num_samples))
        logits, means, scales = self._flatten()
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples, replacement=True, generator=generator)
        mean = means[idx]
        scale = scales[idx]
        eps = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        return mean + scale * eps

    def summary(self) -> Dict[str, Any]:
        return {
            "n_components": sum(c.n_components for c in self._components),
            "n_posteriors": len(self._components),
            "is_prior": self.is_prior,
        }


# --------------------------------------------------------------------------------------
# losses for the fallback baselines
# --------------------------------------------------------------------------------------
def _nll_loss_fn(network: PosteriorDensity, x_observation: Optional[torch.Tensor] = None) -> Callable:
    """NPE loss: negative log-likelihood of ``theta`` under ``q_phi(.|x)``."""

    def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        theta = batch["theta"]
        x = batch["x"]
        return -network.log_prob(theta, x).mean()

    return loss_fn


def _grouped_logsumexp(values: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``logsumexp`` of ``values`` within blocks of identical rows of ``x``."""
    n = int(values.shape[0])
    if n == 0:
        return values
    x = torch.as_tensor(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.shape[0] == 1 or n == 1:
        return torch.logsumexp(values, dim=0).expand(n)
    unique = torch.unique(x.detach(), dim=0, return_inverse=True)
    inverse = unique[1]
    if unique[0].shape[0] == 1:
        return torch.logsumexp(values, dim=0).expand(n)
    out = torch.empty_like(values)
    for group in range(int(unique[0].shape[0])):
        mask = inverse == group
        if not bool(mask.any()):
            continue
        out[mask] = torch.logsumexp(values[mask], dim=0)
    return out


def _apt_loss_fn(
    network: PosteriorDensity, clip_weights: float = 10.0
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """SNPE-C / APT loss (Greenberg et al. 2019; Lueckmann et al. 2017).

    ``L = -sum_n log [ q(theta_n|x_n) w_n / sum_k q(theta_k|x_n) w_k ]`` with importance
    weights ``w_n = p(theta_n) / ptilde(theta_n)`` supplied per datapoint by the
    simulation loop (clipped to ``[0, clip_weights]``, ``sbi``-style).
    """

    def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        theta = batch["theta"]
        x = batch["x"]
        log_q = network.log_prob(theta, x)
        weight = batch.get("weight")
        if weight is None:
            log_w = torch.zeros_like(log_q)
        else:
            weight = torch.as_tensor(weight).reshape(-1).clamp(0.0, float(clip_weights))
            log_w = torch.log(weight.clamp_min(1e-30))
        log_num = log_q + log_w
        log_denom = _grouped_logsumexp(log_num, x)
        log_ratio = log_num - log_denom
        return -(log_ratio).mean()

    return loss_fn


# --------------------------------------------------------------------------------------
# training helpers
# --------------------------------------------------------------------------------------
def _fit_density_network(
    network: nn.Module,
    loss_fn: Callable,
    dataset: Dict[str, torch.Tensor],
    config: BaselineConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Any:
    """Train ``network`` with the shared SNPSE trainer (Adam, 15% early stopping)."""
    trainer_mod = _MODULES.get("trainer")
    theta = dataset["theta"]
    x = dataset["x"]
    weights = dataset.get("weight")

    if trainer_mod is not None and hasattr(trainer_mod, "TrainConfig"):
        train_config = trainer_mod.TrainConfig(
            lr=float(config.lr),
            max_iters=int(config.max_iters),
            batch_size=int(config.resolved_batch_size()),
            val_fraction=float(config.val_fraction),
            patience=int(config.patience),
            weight_decay=float(config.weight_decay),
            verbose=False,
        )
        if hasattr(trainer_mod, "train_network"):
            return trainer_mod.train_network(
                network,
                loss_fn,
                theta,
                x,
                weights=weights,
                config=train_config,
                device=device,
                generator=generator,
                return_history=True,
            )
    # minimal self-contained training loop (used only if snpse.trainer is missing)
    return _train_fallback(network, loss_fn, dataset, config, device, generator)


def _train_fallback(
    network: nn.Module,
    loss_fn: Callable,
    dataset: Dict[str, torch.Tensor],
    config: BaselineConfig,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    theta = dataset["theta"].to(device)
    x = dataset["x"].to(device)
    weight = dataset.get("weight")
    if weight is not None:
        weight = weight.to(device)
    n = theta.shape[0]
    optimiser = torch.optim.Adam(network.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    network.to(device).train()
    batch_size = min(int(config.resolved_batch_size()), n)
    history: List[float] = []
    for _ in range(int(config.max_iters)):
        idx = torch.randint(0, n, (batch_size,), generator=generator, device=device)
        batch = {"theta": theta[idx], "x": x[idx]}
        if weight is not None:
            batch["weight"] = weight[idx]
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(batch)
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach().cpu()))
    network.eval()
    return network, {"train_loss": history}


def _standardisers(
    theta: torch.Tensor, x: torch.Tensor, enabled: bool
) -> Tuple[Optional[Any], Optional[Any]]:
    if not enabled:
        return None, None
    utils = _MODULES.get("utils")
    if utils is None or not hasattr(utils, "fit_standardiser"):
        return None, None
    return utils.fit_standardiser(theta), utils.fit_standardiser(x)


def _check_prior_support(theta: torch.Tensor, prior_log_prob_fn: Callable) -> None:
    """Warn when a proposal sample falls outside the prior support (weight 0)."""
    with torch.no_grad():
        log_p = prior_log_prob_fn(theta)
    n_bad = int((~torch.isfinite(log_p)).sum().item())
    if n_bad:
        warnings.warn(
            f"{n_bad} proposal sample(s) have zero / undefined prior density; "
            "their APT importance weights are set to zero.",
            RuntimeWarning,
        )


# --------------------------------------------------------------------------------------
# fallback NPE (amortised)
# --------------------------------------------------------------------------------------
def run_npe(
    task: Any,
    config: Optional[BaselineConfig] = None,
    dataset: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Amortised NPE baseline (maximum-likelihood MDN on prior-predictive pairs)."""
    config = config or BaselineConfig(method="npe")
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    device = _resolve_device(device or config.device)
    generator = generator or _make_generator(config.seed)

    if dataset is None:
        dataset = _simulate_prior_predictive(task, int(config.budget), generator, config)
    theta, x = dataset["theta"], dataset["x"]

    theta_std_obj, x_std_obj = _standardisers(theta, x, config.standardise)
    network = MixtureDensityNetwork(
        theta_dim=int(theta.shape[1]),
        x_dim=int(x.shape[1]),
        n_components=int(config.n_components),
        hidden_dim=int(config.hidden_dim),
        n_layers=int(config.n_layers),
        activation=config.activation,
        min_scale=config.min_scale,
        max_scale=config.max_scale,
    ).to(device)
    posterior = PosteriorDensity(network, theta_std_obj, x_std_obj)
    network, history = _fit_density_network(
        network, _nll_loss_fn(posterior), dataset, config, device, generator
    )
    network.eval()

    x_obs = _task_observation(task).to(device)
    samples = posterior.sample(x_obs, int(config.num_samples), generator=generator)
    return {
        "method": "npe",
        "backend": "fallback",
        "theta": samples.detach().cpu(),
        "network": network,
        "posterior": posterior,
        "history": history,
        "info": {"num_simulations": int(theta.shape[0]), "config": config.as_dict()},
        "dataset": {"theta": theta, "x": x},
    }


# --------------------------------------------------------------------------------------
# fallback SNPE-C (sequential, APT loss)
# --------------------------------------------------------------------------------------
def run_snpe_c(
    task: Any,
    config: Optional[BaselineConfig] = None,
    device: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    callback: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """SNPE-C / APT baseline with sequential rounds (fallback pure-PyTorch version).

    Data from all rounds is accumulated (as in ``sbi`` / ``sbibm``); each round draws
    parameters from the current proposal (mixture of previous posteriors by default,
    the prior in round 1) and stores the importance weight ``p(theta)/ptilde(theta)``
    used by the APT loss.
    """
    config = config or BaselineConfig(method="snpe_c")
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    device = _resolve_device(device or config.device)
    generator = generator or _make_generator(config.seed)

    prior_sample_fn, prior_log_prob_fn, simulator_fn, x_obs, theta_dim, x_dim = _task_adapters(task, device)
    num_rounds = max(1, int(config.num_rounds))
    round_budget = max(1, int(config.budget) // num_rounds)

    proposal = MixturePosterior(prior_log_prob_fn=prior_log_prob_fn)
    theta_all: List[torch.Tensor] = []
    x_all: List[torch.Tensor] = []
    weight_all: List[torch.Tensor] = []
    history: List[Dict[str, Any]] = []
    posterior = None

    for round_index in range(num_rounds):
        # ---- 1. proposal sampling ------------------------------------------------
        if proposal.is_prior or config.proposal.lower() in ("prior", "none"):
            theta = _as_tensor(prior_sample_fn(round_budget, generator))
        else:
            theta = proposal.sample(round_budget, generator=generator, prior_sample_fn=prior_sample_fn)
        theta = theta.reshape(-1, theta_dim).to(device)
        # importance weights p(theta) / ptilde(theta) (clip to [0, clip_weights])
        with torch.no_grad():
            log_prior = _as_tensor(prior_log_prob_fn(theta)).reshape(-1)
            log_prop = proposal.log_prob(theta).reshape(-1)
            ratio = torch.exp(log_prior - log_prop).clamp(0.0, float(config.clip_weights))
            ratio = torch.where(torch.isfinite(ratio), ratio, torch.zeros_like(ratio))
        # ---- 2. simulate ---------------------------------------------------------
        x = _as_tensor(simulator_fn(theta)).reshape(-1, x_dim).to(device)
        theta_all.append(theta)
        x_all.append(x)
        weight_all.append(ratio)
        # ---- 3. retrain on accumulated data --------------------------------------
        dataset = {
            "theta": torch.cat(theta_all, dim=0),
            "x": torch.cat(x_all, dim=0),
            "weight": torch.cat(weight_all, dim=0),
        }
        theta_std_obj, x_std_obj = _standardisers(dataset["theta"], dataset["x"], config.standardise)
        if posterior is not None and posterior.theta_standardiser is not None and config.standardise:
            # keep the transforms stable across rounds (as sbi does for theta)
            theta_std_obj = posterior.theta_standardiser
        network = MixtureDensityNetwork(
            theta_dim=theta_dim,
            x_dim=x_dim,
            n_components=int(config.n_components),
            hidden_dim=int(config.hidden_dim),
            n_layers=int(config.n_layers),
            activation=config.activation,
            min_scale=config.min_scale,
            max_scale=config.max_scale,
        ).to(device)
        posterior = PosteriorDensity(network, theta_std_obj, x_std_obj)
        network, net_history = _fit_density_network(
            network, _apt_loss_fn(posterior, clip_weights=config.clip_weights), dataset, config, device, generator
        )
        network.eval()
        # ---- 4. rebuild the proposal --------------------------------------------
        weight = 1.0
        if config.proposal_mixture.lower() in ("latest", "last"):
            proposal = MixturePosterior(prior_log_prob_fn=prior_log_prob_fn)
        proposal.add_posterior(posterior, x_obs, weight=weight)
        info = {
            "round": round_index,
            "num_simulations": int(dataset["theta"].shape[0]),
            "round_simulations": int(theta.shape[0]),
            "mean_weight": float(weight_all[-1].mean().item()),
            "network": net_history,
        }
        history.append(info)
        if callback is not None:
            callback(round_index, info)
        if config.verbose:
            print(
                f"[snpe_c] round {round_index + 1}/{num_rounds} | "
                f"sims={info['num_simulations']} | mean weight={info['mean_weight']:.3f}"
            )

    samples = posterior.sample(x_obs, int(config.num_samples), generator=generator)
    return {
        "method": "snpe_c",
        "backend": "fallback",
        "theta": samples.detach().cpu(),
        "network": posterior.network,
        "posterior": posterior,
        "history": history,
        "info": {
            "num_simulations": int(sum(t.shape[0] for t in theta_all)),
            "num_rounds": num_rounds,
            "config": config.as_dict(),
        },
        "dataset": {"theta": torch.cat(theta_all, 0), "x": torch.cat(x_all, 0)},
    }


# --------------------------------------------------------------------------------------
# task adaptation helpers
# --------------------------------------------------------------------------------------
def _resolve_device(device: Optional[Any]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_generator(seed: Optional[int]) -> torch.Generator:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator


def _as_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value if dtype is None else value.to(dtype)
    try:
        import numpy as np  # local import keeps the module importable without numpy

        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, dtype=dtype or torch.float32)
        if isinstance(value, (np.floating, np.integer)):
            return torch.tensor(float(value), dtype=dtype or torch.float32)
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        try:
            return torch.as_tensor(list(value), dtype=dtype or torch.float32)
        except Exception:
            pass
    return torch.as_tensor(value, dtype=dtype or torch.float32)


def _task_observation(task: Any) -> torch.Tensor:
    """Reference observation ``x_obs`` of a task (repo task or sbibm task)."""
    x_obs = getattr(task, "x_obs", None)
    if x_obs is None:
        x_obs = getattr(task, "observation", None)
    if callable(x_obs):
        x_obs = x_obs()
    if x_obs is not None:
        return _as_tensor(x_obs).reshape(-1)
    if hasattr(task, "get_observation"):
        return _as_tensor(task.get_observation(num_observation=1)).reshape(-1)
    raise AttributeError("could not obtain the reference observation from the task object")


def _task_adapters(
    task: Any, device: torch.device
) -> Tuple[Callable, Callable, Callable, torch.Tensor, int, int]:
    """Return ``(prior_sample_fn, prior_log_prob_fn, simulator_fn, x_obs, d, p)``.

    Handles both this repository's :class:`~snpse.tasks.benchmarks.BenchmarkTask`
    objects and raw ``sbibm`` task objects.
    """
    # prior sampling ---------------------------------------------------------------
    if hasattr(task, "sample_prior"):
        def prior_sample_fn(n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
            return _try_calls(
                [
                    lambda: task.sample_prior(n, generator),
                    lambda: task.sample_prior(n),
                    lambda: task.sample_prior(num_samples=n),
                ]
            )
    elif hasattr(task, "get_prior"):
        prior = task.get_prior()

        def prior_sample_fn(n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
            out = _try_calls(
                [
                    lambda: prior.sample((n,)),
                    lambda: prior.sample(sample_shape=(n,)),
                    lambda: prior.sample(num_samples=n),
                ]
            )
            return _as_tensor(out)
    else:
        raise AttributeError("task provides neither sample_prior nor get_prior")

    # prior density ----------------------------------------------------------------
    if hasattr(task, "prior_log_prob"):
        def prior_log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
            return _as_tensor(task.prior_log_prob(theta))
    elif hasattr(task, "get_prior"):
        prior = task.get_prior()

        def prior_log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
            out = _try_calls(
                [
                    lambda: prior.log_prob(theta),
                    lambda: prior.log_prob(theta.detach().cpu()),
                ]
            )
            return _as_tensor(out).reshape(-1)
    else:
        raise AttributeError("task provides neither prior_log_prob nor get_prior")

    # simulator ---------------------------------------------------------------------
    if hasattr(task, "simulate"):
        def simulator_fn(theta: torch.Tensor) -> torch.Tensor:
            return _as_tensor(_try_calls([lambda: task.simulate(theta), lambda: task.simulate(theta.detach().cpu())]))
    elif hasattr(task, "get_simulator"):
        simulator = task.get_simulator()
        simulator = getattr(simulator, "simulate", simulator)

        def simulator_fn(theta: torch.Tensor) -> torch.Tensor:
            return _as_tensor(simulator(theta))
    else:
        raise AttributeError("task provides neither simulate nor get_simulator")

    x_obs = _task_observation(task)

    # dimensions ---------------------------------------------------------------------
    theta_dim = int(getattr(task, "dim_theta", 0) or getattr(task, "dim_parameters", 0) or 0)
    x_dim = int(getattr(task, "dim_x", 0) or getattr(task, "dim_data", 0) or 0)
    if theta_dim <= 0:
        theta_dim = int(_as_tensor(prior_sample_fn(2)).shape[-1])
    if x_dim <= 0:
        probe = _as_tensor(simulator_fn(_as_tensor(prior_sample_fn(1))))
        x_dim = int(probe.reshape(probe.shape[0], -1).shape[-1]) if probe.dim() > 1 else int(probe.numel())
    return prior_sample_fn, prior_log_prob_fn, simulator_fn, x_obs, theta_dim, x_dim


def _try_calls(callables: Sequence[Callable]) -> Any:
    """Return the result of the first callable that does not raise."""
    last: Optional[BaseException] = None
    for fn in callables:
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - API differences
            last = exc
    raise RuntimeError(f"none of the candidate calls succeeded: {last!r}")


def _simulate_prior_predictive(
    task: Any,
    num_simulations: int,
    generator: torch.Generator,
    config: BaselineConfig,
) -> Dict[str, torch.Tensor]:
    """Draw ``(theta, x) ~ p(theta) p(x|theta)`` using the task's own API."""
    loader = None
    benchmarks = _MODULES.get("benchmarks")
    if benchmarks is not None and hasattr(benchmarks, "load_dataset"):
        loader = benchmarks.load_dataset
    if loader is not None:
        try:
            data = loader(task, int(num_simulations), seed=int(config.seed))
            return {"theta": _as_tensor(data["theta"]), "x": _as_tensor(data["x"])}
        except Exception:
            pass
    prior_sample_fn, _, simulator_fn, _, theta_dim, _ = _task_adapters(task, _resolve_device(config.device))
    theta = _as_tensor(prior_sample_fn(num_simulations, generator)).reshape(num_simulations, theta_dim)
    x = _as_tensor(simulator_fn(theta))
    x = x.reshape(x.shape[0], -1)
    return {"theta": theta, "x": x}


# --------------------------------------------------------------------------------------
# sbibm backends (the paper's own baseline implementations)
# --------------------------------------------------------------------------------------
_SBIBM_ALGORITHM_CANDIDATES: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "npe": (
        ("sbibm.algorithms.pytorch", "npe"),
        ("sbibm.algorithms.sbi", "npe"),
        ("sbibm.algorithms", "npe"),
        ("sbibm.algorithms", "npe_pytorch"),
        ("sbibm.algorithms", "sbi_npe"),
    ),
    "snpe_c": (
        ("sbibm.algorithms.pytorch", "snpe_c"),
        ("sbibm.algorithms.sbi", "snpe_c"),
        ("sbibm.algorithms", "snpe_c"),
        ("sbibm.algorithms", "snpe_c_pytorch"),
        ("sbibm.algorithms", "sbi_snpe_c"),
        ("sbibm.algorithms", "snpe-c"),
    ),
}


def _resolve_sbibm_algorithm(key: str) -> Callable:
    if _SBIBM is None:
        raise ImportError(
            "sbibm is not installed; install it with `pip install sbibm` or use "
            "backend='fallback' for the self-contained NPE / SNPE-C implementations."
        )
    for module_name, attr in _SBIBM_ALGORITHM_CANDIDATES[key]:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    # last resort: sbibm's unified dispatcher
    try:
        algorithms = importlib.import_module("sbibm.algorithms")
        dispatcher = getattr(algorithms, "configure_algorithm", None)
        if callable(dispatcher):
            for name in (key, key.replace("_", "-"), f"sbi_{key}", f"{key}_pytorch"):
                try:
                    return _partial_algorithm(dispatcher, name)
                except Exception:
                    continue
    except Exception:
        pass
    raise ImportError(f"could not locate an sbibm implementation for {key!r}")


def _partial_algorithm(dispatcher: Callable, name: str) -> Callable:
    def algorithm(**kwargs: Any) -> Any:
        return dispatcher(algorithm=name, **kwargs)

    return algorithm


def _call_with_signature(fn: Callable, kwargs: Dict[str, Any]) -> Any:
    """Call ``fn`` with the subset of ``kwargs`` accepted by its signature."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    params = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(**accepted)


def _sbibm_task_for(task: Any, observation_index: int = 1) -> Any:
    """Return an sbibm task object matching ``task``."""
    sbibm_task = getattr(task, "sbibm_task", None)
    if sbibm_task is not None:
        return sbibm_task
    name = getattr(task, "name", None)
    if name is None and isinstance(task, str):
        name = task
    if name is None:
        raise ValueError("cannot determine the sbibm task name")
    return _SBIBM.get_task(name)


def _sbibm_run(
    key: str,
    task: Any,
    config: BaselineConfig,
    num_samples: Optional[int] = None,
    num_rounds: Optional[int] = None,
    observation_index: int = 1,
) -> Dict[str, Any]:
    """Run NPE / SNPE-C through sbibm with (as far as possible) default settings."""
    fn = _resolve_sbibm_algorithm(key)
    sbibm_task = _sbibm_task_for(task, observation_index=observation_index)
    num_samples = int(num_samples or config.num_samples)
    kwargs: Dict[str, Any] = {
        "task": sbibm_task,
        "num_simulations": int(config.budget),
        "num_observation": int(observation_index),
        "observation": _task_observation(task),
        "num_samples": num_samples,
        "seed": int(config.seed),
        "num_rounds": int(num_rounds or config.num_rounds),
        "prior": sbibm_task.get_prior() if hasattr(sbibm_task, "get_prior") else None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    samples = _call_with_signature(fn, kwargs)
    samples = _as_tensor(samples)
    return {
        "method": key,
        "backend": "sbibm",
        "theta": samples.detach().cpu() if isinstance(samples, torch.Tensor) else samples,
        "info": {"num_simulations": int(config.budget), "num_samples": num_samples},
    }


def run_npe_sbibm(task: Any, config: Optional[BaselineConfig] = None, **kwargs: Any) -> Dict[str, Any]:
    """NPE baseline via ``sbibm`` (default hyper-parameters)."""
    config = config or BaselineConfig(method="npe", backend="sbibm")
    return _sbibm_run("npe", task, config, **kwargs)


def run_snpe_c_sbibm(task: Any, config: Optional[BaselineConfig] = None, **kwargs: Any) -> Dict[str, Any]:
    """SNPE-C baseline via ``sbibm`` (default hyper-parameters)."""
    config = config or BaselineConfig(method="snpe_c", backend="sbibm")
    return _sbibm_run("snpe_c", task, config, **kwargs)


# --------------------------------------------------------------------------------------
# TSNPE (mackelab reference implementation -- intentionally NOT re-implemented)
# --------------------------------------------------------------------------------------
def _load_tsnpe_class(default: Any = "raise") -> Any:
    """Import the mackelab ``TSNPE`` class from any of its known locations."""
    global _TSNPE
    if _TSNPE is not None:
        return _TSNPE
    candidates = (
        ("tsnpe", "TSNPE"),
        ("tsnpe.tsnpe", "TSNPE"),
        ("tsnpe.inference", "TSNPE"),
        ("tsnpe.run", "TSNPE"),
        ("tsnpe.snpe", "TSNPE"),
        ("tsnpe.sequential", "TSNPE"),
    )
    for module_name, attr in candidates:
        module = _try_import(module_name)
        if module is None:
            continue
        cls = getattr(module, attr, None)
        if cls is not None:
            _TSNPE = cls
            return _TSNPE
    if default == "raise":
        raise ImportError(
            "the mackelab TSNPE baseline could not be imported.\n"
            "The paper (Section 5.2 / Appendix F) runs TSNPE via the reference "
            "implementation github.com/mackelab/tsnpe_neurips, which is intentionally "
            "not re-implemented here. To use it:\n"
            "  git clone https://github.com/mackelab/tsnpe_neurips\n"
            "  export PYTHONPATH=$PYTHONPATH:$PWD/tsnpe_neurips\n"
            "(the package expects the `sbi` dependency to be installed)."
        )
    return default


def run_tsnpe(
    task: Any,
    config: Optional[BaselineConfig] = None,
    observation_index: int = 1,
    **kwargs: Any,
) -> Dict[str, Any]:
    """TSNPE baseline (Deistler et al. 2022a) via the mackelab reference code."""
    config = config or BaselineConfig(method="tsnpe")
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)
    tsnpe_cls = _load_tsnpe_class()
    sbibm_task = _sbibm_task_for(task, observation_index=observation_index)
    call_kwargs: Dict[str, Any] = {
        "task": sbibm_task,
        "num_simulations": int(config.budget),
        "num_rounds": int(config.num_rounds),
        "num_samples": int(config.num_samples),
        "num_observation": int(observation_index),
        "observation": _task_observation(task),
        "seed": int(config.seed),
    }
    instance = _call_with_signature(tsnpe_cls, call_kwargs)
    samples = None
    if instance is not None:
        for attr in ("sample", "run", "infer", "__call__"):
            fn = getattr(instance, attr, None)
            if callable(fn):
                try:
                    samples = _call_with_signature(
                        fn,
                        {
                            "num_samples": int(config.num_samples),
                            "observation": _task_observation(task),
                            "num_observation": int(observation_index),
                        },
                    )
                    break
                except Exception:
                    continue
    if samples is None:
        raise RuntimeError("the mackelab TSNPE implementation returned no samples")
    samples = _as_tensor(samples)
    return {
        "method": "tsnpe",
        "backend": "mackelab",
        "theta": samples.detach().cpu(),
        "info": {
            "num_simulations": int(config.budget),
            "num_rounds": int(config.num_rounds),
            "reference": "github.com/mackelab/tsnpe_neurips",
        },
    }


# --------------------------------------------------------------------------------------
# FMPE (Dax et al. 2023) -- published numbers only
# --------------------------------------------------------------------------------------
#: C2ST of FMPE (Dax et al. 2023, Appendix F) on the eight ``sbibm`` benchmark tasks.
#: The values are *transcribed from the literature* -- FMPE is not re-run here (the
#: reproduction plan states that these numbers are taken from Dax et al. 2023, App. F).
#: Fill in / verify against Appendix F of the paper before reporting them.
FMPE_PUBLISHED_C2ST: Dict[str, Dict[str, Optional[float]]] = {
    "gaussian_linear": {"1e3": None, "1e4": None, "1e5": None},
    "gaussian_mixture": {"1e3": None, "1e4": None, "1e5": None},
    "two_moons": {"1e3": None, "1e4": None, "1e5": None},
    "gaussian_linear_uniform": {"1e3": None, "1e4": None, "1e5": None},
    "bernoulli_glm": {"1e3": None, "1e4": None, "1e5": None},
    "slcp": {"1e3": None, "1e4": None, "1e5": None},
    "sir": {"1e3": None, "1e4": None, "1e5": None},
    "lotka_volterra": {"1e3": None, "1e4": None, "1e5": None},
}


# --------------------------------------------------------------------------------------
# evaluation + dispatcher
# --------------------------------------------------------------------------------------
def evaluate_c2st(
    task: Any,
    samples: torch.Tensor,
    num_reference_samples: Optional[int] = None,
    seed: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """C2ST of posterior ``samples`` against the task's reference posterior."""
    c2st_mod = _MODULES.get("c2st")
    if c2st_mod is None:
        raise ImportError("snpse.tasks.c2st is required to evaluate the baselines")
    if hasattr(c2st_mod, "task_c2st"):
        result = c2st_mod.task_c2st(
            task,
            samples,
            num_reference_samples=num_reference_samples,
            seed=seed,
            **kwargs,
        )
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return {"c2st": float(result)}
    raise ImportError("snpse.tasks.c2st does not expose task_c2st")


def run_baseline(
    method: str,
    task: Any,
    config: Optional[BaselineConfig] = None,
    backend: Optional[str] = None,
    dataset: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[Any] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run one baseline method on ``task`` and return samples (+ bookkeeping).

    ``method`` is one of ``"npe"``, ``"snpe_c"``, ``"tsnpe"``; ``backend`` is
    ``"auto"`` (prefer sbibm/mackelab when installed, fall back to the built-in
    implementations), ``"sbibm"`` / ``"mackelab"`` (external, no fallback) or
    ``"fallback"`` (always the built-in implementation).
    """
    key = method.lower().replace("-", "_")
    if key not in ("npe", "snpe_c", "tsnpe"):
        raise ValueError(f"unknown baseline {method!r}; expected one of {BASELINE_METHODS}")
    config = config or BaselineConfig(method=key)
    config.method = key
    backend = (backend or config.backend or "auto").lower()
    if backend != "auto":
        config.backend = backend

    if key == "npe":
        if backend in ("sbibm",) or (backend == "auto" and sbibm_available()):
            try:
                return run_npe_sbibm(task, config, num_samples=config.num_samples)
            except Exception as exc:
                if backend == "sbibm":
                    raise
                warnings.warn(f"sbibm NPE failed ({exc!r}); using the built-in fallback.", RuntimeWarning)
        return run_npe(task, config, dataset=dataset, device=device, generator=generator, **kwargs)

    if key == "snpe_c":
        if backend in ("sbibm",) or (backend == "auto" and sbibm_available()):
            try:
                return run_snpe_c_sbibm(
                    task, config, num_samples=config.num_samples, num_rounds=config.num_rounds
                )
            except Exception as exc:
                if backend == "sbibm":
                    raise
                warnings.warn(f"sbibm SNPE-C failed ({exc!r}); using the built-in fallback.", RuntimeWarning)
        return run_snpe_c(task, config, device=device, generator=generator, **kwargs)

    # TSNPE: mackelab only (never re-implemented)
    return run_tsnpe(task, config)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SNPSE baselines (NPE / SNPE-C / TSNPE)")
    parser.add_argument("--method", default="snpe_c", choices=list(BASELINE_METHODS))
    parser.add_argument("--task", default="slcp")
    parser.add_argument("--budget", type=int, default=10_000)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="auto", choices=("auto", "sbibm", "mackelab", "fallback"))
    parser.add_argument("--observation-index", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-standardise", action="store_true")
    parser.add_argument("--output", default=None, help="optional .pt path to store the samples")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    benchmarks = _module("benchmarks")
    task = benchmarks.get_task(args.task)
    config = BaselineConfig(
        method=args.method,
        budget=int(args.budget),
        num_rounds=int(args.num_rounds),
        num_samples=int(args.num_samples),
        seed=int(args.seed),
        device=args.device,
        standardise=not args.no_standardise,
        backend=args.backend,
        verbose=not args.quiet,
    )
    result = run_baseline(
        args.method, task, config=config, backend=args.backend
    )
    theta = result["theta"]
    c2st_values = evaluate_c2st(task, theta, seed=args.seed)
    summary = {
        "method": args.method,
        "task": args.task,
        "budget": args.budget,
        "backend": result.get("backend", args.backend),
        "c2st": c2st_values.get("c2st"),
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        torch.save({"samples": theta, "config": config.as_dict(), "summary": summary}, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
