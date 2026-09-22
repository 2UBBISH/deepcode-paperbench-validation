"""NPSE baseline: posterior score estimation with a conditional MLP.

This module implements the *Neural Posterior Score Estimation* (NPSE) baseline
(Geffner et al., 2023) used as a comparison to Simformer in the paper
(Sec. 4.1, Appendix A2.1 / A3.1).

Two flavours are provided:

1. :class:`NPSEBaseline` -- a plain conditional score network
   ``s_phi(theta_t, t | x)`` (an MLP, *not* a transformer) trained with
   denoising score matching on the parameters only, exactly as the Simformer
   objective (Eqs. 1-2 of Sec. 3.3) restricted to the posterior conditional
   ``p(theta | x)`` (i.e. ``M_C = posterior``, so the data block is always
   clean/conditioned and there is no masked-conditioning signal).

2. :class:`SimformerPosteriorOnlyBaseline` -- the *posterior-only Simformer
   variant*: the paper's own architecture (tokenizer + transformer score
   network) but trained with the condition mask restricted to the posterior
   mode (``M_C = posterior``) instead of the full mixture of joint / posterior
   / likelihood / random masks (Sec. 3.1, A2.1).

Both share the same interface as the other baselines in
``simformer/baselines/npe_nle_nre.py``: ``fit()``, ``posterior_samples()`` and
``to_dict()``, so they can be dropped into the experiment sweeps.

Diffusion conventions follow the paper (Appendix A2.1):
``sigma_max = 15``, ``sigma_min = 1e-4`` (VESDE) or ``beta_min = 0.01``,
``beta_max = 10`` (VPSDE), ``t in [1e-5, 1]``, reverse SDE solved by
Euler-Maruyama with 500 steps by default.  The network predicts epsilon and the
score is recovered as ``grad log p_t = -eps / sigma(t)``.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # torch is the neural-network backend (soft dependency)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is expected in practice
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# diffusion / sibling-baseline imports (tolerant, supports both layouts)
# ---------------------------------------------------------------------------
def _import_first(candidates: Sequence[str], what: str = "module") -> Any:
    import importlib

    last_exc: Optional[Exception] = None
    for path in candidates:
        try:
            return importlib.import_module(path)
        except Exception as exc:  # pragma: no cover
            last_exc = exc
    raise ImportError(f"could not import {what}; tried {list(candidates)} ({last_exc})")


try:
    _diffusion = _import_first(
        ["simformer.simformer.diffusion", "simformer.diffusion", "..diffusion"],
        "simformer diffusion module",
    )
except Exception:  # pragma: no cover
    try:
        from .. import diffusion as _diffusion  # type: ignore
    except Exception:
        _diffusion = None


def _sde_factory(name: str = "vesde", **kwargs):
    """Build an SDE using the library factory when available."""
    if _diffusion is not None and hasattr(_diffusion, "get_sde"):
        return _diffusion.get_sde(name, **kwargs)
    raise ImportError("simformer.diffusion.get_sde unavailable; cannot build an SDE")


def _reverse_step(sde, x, t_cur, t_next, score, noise=None):
    if _diffusion is not None and hasattr(_diffusion, "reverse_sde_step"):
        return _diffusion.reverse_sde_step(sde, x, t_cur, t_next, score, noise=noise)
    raise ImportError("simformer.diffusion.reverse_sde_step unavailable")


def _time_grid(n_steps: int, t_min: float, t_max: float, descending: bool = True):
    if _diffusion is not None and hasattr(_diffusion, "time_grid"):
        return _diffusion.time_grid(n_steps, t_min=t_min, t_max=t_max, descending=descending)
    if descending:
        return np.linspace(t_max, t_min, n_steps + 1)
    return np.linspace(t_min, t_max, n_steps + 1)


def _sde_marginals(sde, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mu(t), sigma(t))`` for the SDE, tolerating API differences."""
    t = np.asarray(t, dtype=np.float64)
    if hasattr(sde, "marginal_mean_std"):
        mu, sigma = sde.marginal_mean_std(t)
        return np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(sde.marginal_mean(t), dtype=np.float64)
    sigma = np.asarray(sde.marginal_std(t), dtype=np.float64)
    return mu, sigma


def _sde_prior(sde, shape, rng: np.random.Generator):
    if hasattr(sde, "sample_prior"):
        try:
            return np.asarray(sde.sample_prior(shape, rng=rng), dtype=np.float64)
        except TypeError:
            return np.asarray(sde.sample_prior(shape, rng), dtype=np.float64)
    mu, sigma = _sde_marginals(sde, np.array([getattr(sde, "t_max", 1.0)]))
    return np.asarray(rng.normal(0.0, 1.0, size=shape), dtype=np.float64) * float(
        np.ravel(sigma)[0]
    )


try:  # shared baseline utilities (config, standardizer, reference posteriors)
    from .npe_nle_nre import (  # type: ignore
        BaselineConfig,
        BaselinePosterior,
        Standardizer,
        _mcmc_posterior_samples,
        normalize_method,
        reference_posterior_samples,
        simulate_dataset,
    )

    _HAS_SIBLING_BASELINES = True
except Exception:  # pragma: no cover - standalone execution
    _HAS_SIBLING_BASELINES = False
    BaselineConfig = None  # type: ignore
    BaselinePosterior = object  # type: ignore
    Standardizer = None  # type: ignore
    normalize_method = None  # type: ignore
    reference_posterior_samples = None  # type: ignore
    simulate_dataset = None  # type: ignore
    _mcmc_posterior_samples = None  # type: ignore


# ---------------------------------------------------------------------------
# defaults (Appendix A2.1)
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 1000
DEFAULT_LR = 5e-4
DEFAULT_VALIDATION_FRACTION = 0.1
DEFAULT_PATIENCE = 20
DEFAULT_MAX_EPOCHS = 300
DEFAULT_HIDDEN_DIMS: Tuple[int, ...] = (256, 256, 256)
DEFAULT_TIME_EMBED_DIM = 128
DEFAULT_TIME_EMBED_SCALE = 16.0
DEFAULT_SDE = "vesde"
DEFAULT_N_SAMPLING_STEPS = 500
DEFAULT_MIN_SAMPLING_STEPS = 50
DEFAULT_N_EVAL_SAMPLES = 1000
DEFAULT_N_REFERENCE = 1000
DEFAULT_N_TARGETS = 10
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_SIGMA_MAX = 15.0
DEFAULT_SIGMA_MIN = 1e-4
DEFAULT_BETA_MIN = 0.01
DEFAULT_BETA_MAX = 10.0
DEFAULT_T_MIN = 1e-5
DEFAULT_T_MAX = 1.0

IMPLEMENTED_METHODS: Tuple[str, ...] = ("npse", "npse_simformer")
METHOD_ALIASES: Dict[str, str] = {
    "npse": "npse",
    "posterior_score": "npse",
    "posterior_score_estimation": "npse",
    "nps": "npse",
    "npse_simformer": "npse_simformer",
    "simformer_posterior": "npse_simformer",
    "posterior_only_simformer": "npse_simformer",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a.reshape(1, -1) if a.ndim == 1 else a


def _as_rng(rng: Optional[Union[np.random.Generator, int]] = None) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(int(rng))


def _normalize_method(method: str) -> str:
    key = str(method).lower().strip().replace("-", "_")
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    if callable(normalize_method):
        try:
            return normalize_method(method)  # type: ignore[misc]
        except Exception:
            pass
    return key


def _tensor(x: np.ndarray, device=None, dtype=None):
    if not _HAS_TORCH:
        raise ImportError("torch is required for the NPSE baseline")
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=dtype or torch.float32).to(
        device or "cpu"
    )


def _to_numpy(x) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class NPSEConfig:
    """Configuration of the NPSE baseline (Appendix A2.1 defaults).

    Attributes
    ----------
    hidden_dims:
        Widths of the conditional MLP body.
    sde / sigma_max / sigma_min / beta_min / beta_max / t_min / t_max:
        Diffusion configuration (paper values: 15, 1e-4, 0.01, 10, 1e-5, 1).
    n_sampling_steps:
        Number of Euler-Maruyama steps for the reverse SDE (500 by default,
        >= 50 recommended, cf. Fig. A7).
    condition_mask_modes / p_random_low / p_random_high:
        Only relevant for :class:`SimformerPosteriorOnlyBaseline`; defaults to
        the posterior-only mode.
    """

    method: str = "npse"
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    max_epochs: int = DEFAULT_MAX_EPOCHS
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION
    patience: int = DEFAULT_PATIENCE
    hidden_dims: Tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    time_embed_dim: int = DEFAULT_TIME_EMBED_DIM
    time_embed_scale: float = DEFAULT_TIME_EMBED_SCALE
    standardize: bool = True
    loss_space: str = "epsilon"  # "epsilon" or "score"
    gradient_clip: float = DEFAULT_GRAD_CLIP
    sde: str = DEFAULT_SDE
    sigma_max: float = DEFAULT_SIGMA_MAX
    sigma_min: float = DEFAULT_SIGMA_MIN
    beta_min: float = DEFAULT_BETA_MIN
    beta_max: float = DEFAULT_BETA_MAX
    t_min: float = DEFAULT_T_MIN
    t_max: float = DEFAULT_T_MAX
    n_sampling_steps: int = DEFAULT_N_SAMPLING_STEPS
    n_simulations: int = 10000
    condition_mask_modes: Tuple[str, ...] = ("posterior",)
    p_random_low: float = 0.3
    p_random_high: float = 0.7
    device: str = "cpu"
    seed: int = 0
    verbose: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "NPSEConfig"]] = None, **kwargs):
        if cfg is None:
            return cls(**kwargs)
        if isinstance(cfg, NPSEConfig):
            return cfg
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        payload = {k: v for k, v in dict(cfg).items() if k in known}
        payload.update(kwargs)
        if "hidden_dims" in payload and payload["hidden_dims"] is not None:
            payload["hidden_dims"] = tuple(payload["hidden_dims"])
        if "condition_mask_modes" in payload and payload["condition_mask_modes"] is not None:
            payload["condition_mask_modes"] = tuple(payload["condition_mask_modes"])
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["hidden_dims"] = list(self.hidden_dims)
        out["condition_mask_modes"] = list(self.condition_mask_modes)
        return out

    def build_sde(self):
        """Instantiate the diffusion SDE with the paper's coefficients."""
        kwargs: Dict[str, Any] = {"t_min": self.t_min, "t_max": self.t_max}
        if str(self.sde).lower().startswith(("vesde", "ve")):
            kwargs.update(sigma_max=self.sigma_max, sigma_min=self.sigma_min)
        else:
            kwargs.update(beta_min=self.beta_min, beta_max=self.beta_max)
        try:
            return _sde_factory(self.sde, n_steps=self.n_sampling_steps, **kwargs)
        except TypeError:
            return _sde_factory(self.sde, **kwargs)


# ---------------------------------------------------------------------------
# conditional score MLP
# ---------------------------------------------------------------------------
class TimeFourierEmbedding(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    """128-dimensional random Gaussian Fourier embedding of the diffusion time."""

    def __init__(self, out_dim: int = DEFAULT_TIME_EMBED_DIM, scale: float = 16.0, seed: int = 0):
        if not _HAS_TORCH:
            raise ImportError("torch is required for TimeFourierEmbedding")
        super().__init__()
        g = np.random.default_rng(seed)
        # fixed random frequencies (as in the Simformer transformer)
        freqs = g.normal(0.0, scale, size=(out_dim // 2,)).astype(np.float32)
        self.register_buffer("freqs", torch.as_tensor(freqs, dtype=torch.float32))
        self.out_dim = int(out_dim)
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.float32, device=self.freqs.device)
        t = t.reshape(-1)
        ang = t[:, None] * self.freqs[None, :] * (2.0 * math.pi)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.out_dim:
            pad = self.out_dim - emb.shape[-1]
            emb = torch.cat([emb, torch.zeros(emb.shape[0], pad, device=emb.device)], dim=-1)
        return self.proj(emb)


class ConditionalScoreMLP(nn.Module if _HAS_TORCH else object):  # type: ignore[misc]
    """Conditional score network ``s_phi(theta_t, t | x)`` (epsilon prediction).

    The network is an MLP with residual blocks operating on the concatenation
    of the noisy parameters, a 128-dim Fourier embedding of the diffusion time
    and an encoding of the conditioning data ``x``.
    """

    def __init__(
        self,
        target_dim: int,
        context_dim: int,
        hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
        time_embed_dim: int = DEFAULT_TIME_EMBED_DIM,
        time_embed_scale: float = DEFAULT_TIME_EMBED_SCALE,
        activation: str = "silu",
        seed: int = 0,
    ):
        if not _HAS_TORCH:
            raise ImportError("torch is required for ConditionalScoreMLP")
        super().__init__()
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        self.time_embed = TimeFourierEmbedding(time_embed_dim, time_embed_scale, seed=seed)

        self.context_net = nn.Sequential(
            nn.Linear(self.context_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0]),
        )

        in_dim = self.target_dim + time_embed_dim + hidden_dims[0]
        self.input_proj = nn.Linear(in_dim, hidden_dims[0])
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dims[i]),
                    nn.Linear(hidden_dims[i], 2 * hidden_dims[i]),
                    nn.SiLU(),
                    nn.Linear(2 * hidden_dims[i], hidden_dims[i]),
                )
            )
        self.out_norm = nn.LayerNorm(hidden_dims[-1])
        self.out = nn.Linear(hidden_dims[-1], self.target_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero-init output -> network starts as a no-op score
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, theta_t, t, context):
        ctx = self.context_net(context)
        temb = self.time_embed(t)
        h = torch.cat([theta_t, temb, ctx], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = h + block(h)
        return self.out(self.out_norm(h))


# ---------------------------------------------------------------------------
# posterior sampler for the NPSE model
# ---------------------------------------------------------------------------
class NPSESampler:
    """Reverse-SDE posterior sampler for a trained conditional score network."""

    def __init__(
        self,
        model,
        sde,
        *,
        theta_mean: Optional[np.ndarray] = None,
        theta_std: Optional[np.ndarray] = None,
        loss_space: str = "epsilon",
        device: str = "cpu",
        dtype=None,
    ):
        self.model = model
        self.sde = sde
        self.theta_mean = None if theta_mean is None else np.asarray(theta_mean, dtype=np.float64)
        self.theta_std = None if theta_std is None else np.asarray(theta_std, dtype=np.float64)
        self.loss_space = loss_space
        self.device = device
        self.dtype = dtype

    # -- score evaluation ---------------------------------------------------
    def score(self, theta_t: np.ndarray, t: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Return ``grad_theta log p_t(theta_t | x)`` for a batch."""
        theta_t = _as_2d(theta_t)
        context = _as_2d(context)
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
        if t_arr.size == 1:
            t_arr = np.repeat(t_arr, theta_t.shape[0])
        _, sigma = _sde_marginals(self.sde, t_arr)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
        if sigma.size == 1:
            sigma = np.repeat(sigma, theta_t.shape[0])

        # normalize inputs
        z = theta_t
        if self.theta_mean is not None:
            z = (theta_t - self.theta_mean[None, :]) / self.theta_std[None, :]
        with torch.no_grad():
            out = self.model(
                _tensor(z, self.device, self.dtype),
                _tensor(t_arr, self.device, self.dtype),
                _tensor(context, self.device, self.dtype),
            )
        out = _to_numpy(out).astype(np.float64).reshape(theta_t.shape)
        if self.loss_space == "epsilon":
            s_z = -out / sigma[:, None]
        else:
            s_z = out
        # chain rule for standardisation: s_theta = s_z / std
        if self.theta_std is not None:
            s_z = s_z / self.theta_std[None, :]
        return s_z

    # -- sampling -----------------------------------------------------------
    def sample(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        *,
        n_steps: Optional[int] = None,
        seed: int = 0,
        rng: Optional[np.random.Generator] = None,
        return_trajectory: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Draw ``n_samples`` from the approximate posterior ``p(theta|x_obs)``."""
        rng = _as_rng(rng if rng is not None else seed)
        context = np.repeat(_as_2d(x_obs), n_samples, axis=0)
        dim = int(self.model.target_dim)
        n_steps = int(n_steps or getattr(self.sde, "n_steps", DEFAULT_N_SAMPLING_STEPS))
        t_min = float(getattr(self.sde, "t_min", DEFAULT_T_MIN))
        t_max = float(getattr(self.sde, "t_max", DEFAULT_T_MAX))
        times = _time_grid(n_steps, t_min=t_min, t_max=t_max, descending=True)

        x = _sde_prior(self.sde, (n_samples, dim), rng).astype(np.float64)
        traj = [x.copy()] if return_trajectory else None
        for i in range(len(times) - 1):
            t_cur, t_next = float(times[i]), float(times[i + 1])
            s = self.score(x, np.full(n_samples, t_cur), context)
            noise = rng.normal(0.0, 1.0, size=x.shape) if (t_cur - t_next) > 0 else None
            x = _reverse_step(self.sde, x, t_cur, t_next, s, noise=noise)
            if return_trajectory:
                traj.append(np.asarray(x).copy())
        x = np.asarray(x, dtype=np.float64)
        if return_trajectory:
            return x, np.asarray(traj)  # type: ignore[return-value]
        return x

    # -- log density (probability-flow ODE) ---------------------------------
    def log_prob(
        self,
        theta: np.ndarray,
        x_obs: np.ndarray,
        *,
        n_steps: int = 200,
        return_shape: str = "samples",
    ) -> np.ndarray:
        """Approximate ``log p(theta | x_obs)`` via the probability-flow ODE.

        Uses :func:`simformer.diffusion.log_likelihood_from_ode` when available;
        otherwise returns ``nan`` (the metric is only used for the NLL table,
        which the paper declares optional, cf. Addendum "Experiments").
        """
        theta = _as_2d(theta)
        context = np.repeat(_as_2d(x_obs), theta.shape[0], axis=0)

        def score_fn(x, t):
            t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
            if t_arr.size == 1:
                t_arr = np.repeat(t_arr, np.asarray(x).shape[0])
            return self.score(np.asarray(x), t_arr, context)

        if _diffusion is not None and hasattr(_diffusion, "log_likelihood_from_ode"):
            try:
                lp = _diffusion.log_likelihood_from_ode(
                    score_fn, self.sde, theta, n_steps=n_steps, t_min=self.sde.t_min, t_max=self.sde.t_max
                )
                return np.asarray(lp, dtype=np.float64).reshape(-1)
            except Exception:
                pass
        return np.full(theta.shape[0], np.nan)


# ---------------------------------------------------------------------------
# NPSE baseline
# ---------------------------------------------------------------------------
class NPSEBaseline(BaselinePosterior if _HAS_SIBLING_BASELINES else object):  # type: ignore[misc]
    """Neural Posterior Score Estimation baseline.

    Trains a conditional score network on ``(theta, x)`` pairs from the task
    simulator with denoising score matching (the Simformer objective of
    Sec. 3.3 restricted to the posterior conditional), then samples the
    posterior by integrating the reverse SDE in theta space while keeping the
    data block clamped.
    """

    method = "npse"

    def __init__(self, task: Any, config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None, **kwargs):
        if not _HAS_TORCH:
            raise ImportError("torch is required for NPSEBaseline")
        self.task = task
        self.config = NPSEConfig.from_dict(config, **kwargs)
        self.device = self.config.device
        self.dtype = torch.float32
        self.sde = self.config.build_sde()
        self.model: Optional[ConditionalScoreMLP] = None
        self.standardizer = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self.fitted = False
        self._n_parameters: Optional[int] = None
        self._n_data: Optional[int] = None

    # -- dimensions ---------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        if self._n_parameters is not None:
            return int(self._n_parameters)
        for attr in ("n_parameters", "theta_dim"):
            if hasattr(self.task, attr):
                try:
                    val = getattr(self.task, attr)
                    return int(val() if callable(val) else val)
                except Exception:
                    continue
        raise AttributeError("cannot determine the number of parameters from the task")

    @property
    def n_data(self) -> int:
        if self._n_data is not None:
            return int(self._n_data)
        for attr in ("n_data", "data_dim"):
            if hasattr(self.task, attr):
                try:
                    val = getattr(self.task, attr)
                    return int(val() if callable(val) else val)
                except Exception:
                    continue
        raise AttributeError("cannot determine the data dimension from the task")

    # -- data ---------------------------------------------------------------
    def make_dataset(
        self, n_simulations: Optional[int] = None, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = int(n_simulations or self.config.n_simulations)
        seed = self.config.seed if seed is None else int(seed)
        if callable(simulate_dataset):
            try:
                return simulate_dataset(self.task, n, seed=seed, verbose=self.config.verbose)  # type: ignore[misc]
            except TypeError:
                pass
        rng = np.random.default_rng(seed)
        if hasattr(self.task, "make_dataset"):
            theta, x = self.task.make_dataset(n, rng=rng)
        else:
            theta = np.asarray(self.task.prior_sample(n, rng=rng), dtype=np.float64)
            x = np.asarray(self.task.simulate(theta, rng=rng), dtype=np.float64)
        return _as_2d(theta), _as_2d(x)

    # -- training -----------------------------------------------------------
    def _build_model(self, theta_dim: int, data_dim: int) -> ConditionalScoreMLP:
        return ConditionalScoreMLP(
            target_dim=theta_dim,
            context_dim=data_dim,
            hidden_dims=tuple(self.config.hidden_dims),
            time_embed_dim=self.config.time_embed_dim,
            time_embed_scale=self.config.time_embed_scale,
            seed=self.config.seed,
        ).to(self.device)

    def _batch_loss(self, model, theta0, x_cond, rng: np.random.Generator):
        """Masked denoising score matching restricted to the posterior conditional.

        The data block is always conditioned (``M_C`` posterior mode), so the
        loss of Sec. 3.3 reduces to the standard DSM loss on theta.
        """
        n = theta0.shape[0]
        t = rng.uniform(self.sde.t_min, self.sde.t_max, size=n)
        mu, sigma = _sde_marginals(self.sde, t)
        mu = np.asarray(mu, dtype=np.float64).reshape(n, 1)
        sigma = np.asarray(sigma, dtype=np.float64).reshape(n, 1)
        eps = rng.normal(0.0, 1.0, size=theta0.shape)
        theta_t = mu * theta0 + sigma * eps

        # network operates in standardised space
        z0, zt = theta0, theta_t
        if self.standardizer is not None:
            z0 = self.standardizer.transform(theta0)
            zt = self.standardizer.transform(theta_t)

        pred = model(
            _tensor(zt, self.device, self.dtype),
            _tensor(t, self.device, self.dtype),
            _tensor(x_cond, self.device, self.dtype),
        )
        eps_t = _tensor(eps, self.device, self.dtype)
        if self.config.loss_space == "score":
            target = -(eps_t) / _tensor(sigma, self.device, self.dtype)
            loss = ((pred - target) ** 2).mean()
        else:
            loss = ((pred - eps_t) ** 2).mean()
        return loss

    def fit(
        self,
        n_simulations: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        max_epochs: Optional[int] = None,
        verbose: Optional[bool] = None,
    ) -> "NPSEBaseline":
        seed = self.config.seed if seed is None else int(seed)
        verbose = self.config.verbose if verbose is None else bool(verbose)
        n_epochs = int(max_epochs or self.config.max_epochs)
        rng = np.random.default_rng(seed)
        if _HAS_TORCH:
            torch.manual_seed(seed)

        theta, x = self.make_dataset(n_simulations, seed=seed)
        self._n_parameters = int(theta.shape[1])
        self._n_data = int(x.shape[1])

        if self.config.standardize:
            if callable(Standardizer):
                self.standardizer = Standardizer()
                self.standardizer.fit(theta)
            else:  # pragma: no cover - minimal fallback
                self.standardizer = _SimpleStandardizer(theta)

        # validation split
        n = theta.shape[0]
        n_val = max(1, int(self.config.validation_fraction * n)) if n > 10 else 0
        theta_tr, x_tr = theta[:-n_val] if n_val else theta, x[:-n_val] if n_val else x
        theta_va, x_va = theta[-n_val:] if n_val else theta, x[-n_val:] if n_val else x

        model = self._build_model(theta.shape[1], x.shape[1])
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        batch = min(int(self.config.batch_size), max(1, theta_tr.shape[0]))

        best_val = float("inf")
        best_state = None
        patience = 0
        t0 = time.time()
        for epoch in range(n_epochs):
            model.train()
            perm = rng.permutation(theta_tr.shape[0])
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, theta_tr.shape[0], batch):
                idx = perm[start : start + batch]
                if idx.size < 2:
                    continue
                opt.zero_grad(set_to_none=True)
                loss = self._batch_loss(model, theta_tr[idx], x_tr[idx], rng)
                loss.backward()
                if self.config.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                opt.step()
                epoch_loss += float(loss.detach().cpu())
                n_batches += 1
            train_loss = epoch_loss / max(1, n_batches)
            self.history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if n_val:
                model.eval()
                with torch.no_grad():
                    vals = []
                    for start in range(0, theta_va.shape[0], batch):
                        idx = slice(start, min(start + batch, theta_va.shape[0]))
                        vals.append(
                            float(self._batch_loss(model, theta_va[idx], x_va[idx], rng).cpu())
                        )
                    val_loss = float(np.mean(vals)) if vals else float("nan")
            self.history["val_loss"].append(val_loss)

            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(
                    f"[npse] epoch {epoch + 1}/{n_epochs} train={train_loss:.4f} val={val_loss:.4f}"
                )

            if n_val and np.isfinite(val_loss):
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= int(self.config.patience):
                        if verbose:
                            print(f"[npse] early stopping at epoch {epoch + 1}")
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.fitted = True
        self.history["fit_seconds"] = [time.time() - t0]  # type: ignore[list-item]
        return self

    # -- inference ----------------------------------------------------------
    def sampler(self) -> NPSESampler:
        if self.model is None:
            raise RuntimeError("NPSEBaseline.fit() must be called before sampling")
        mean = getattr(self.standardizer, "mean", None) if self.standardizer is not None else None
        std = getattr(self.standardizer, "std", None) if self.standardizer is not None else None
        return NPSESampler(
            self.model,
            self.sde,
            theta_mean=mean,
            theta_std=std,
            loss_space=self.config.loss_space,
            device=self.device,
            dtype=self.dtype,
        )

    def posterior_samples(
        self,
        x_obs: np.ndarray,
        n_samples: int = 1000,
        *,
        seed: int = 0,
        n_steps: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        return self.sampler().sample(
            x_obs, n_samples, n_steps=n_steps, seed=seed, **kwargs
        )

    def sample(self, n_samples: int = 1000, x_obs: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if x_obs is None:
            raise ValueError("NPSE sampling requires an observation x_obs")
        return self.posterior_samples(x_obs, n_samples, **kwargs)

    __call__ = sample

    def posterior_log_prob(self, theta: np.ndarray, x_obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.sampler().log_prob(theta, x_obs, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config.to_dict(),
            "n_parameters": self._n_parameters,
            "n_data": self._n_data,
            "fitted": self.fitted,
            "history": {k: v for k, v in self.history.items()},
        }

    def save(self, path: str) -> str:
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if self.model is not None:
            torch.save(
                {
                    "state_dict": self.model.state_dict(),
                    "config": self.config.to_dict(),
                    "n_parameters": self.n_parameters,
                    "n_data": self.n_data,
                },
                path,
            )
        return path

    @classmethod
    def load(cls, path: str, task: Any, **kwargs) -> "NPSEBaseline":
        payload = torch.load(path, map_location="cpu")
        cfg = NPSEConfig.from_dict(payload.get("config", {}), **kwargs)
        obj = cls(task, cfg)
        obj._n_parameters = payload.get("n_parameters")
        obj._n_data = payload.get("n_data")
        obj.model = obj._build_model(int(payload["n_parameters"]), int(payload["n_data"]))
        obj.model.load_state_dict(payload["state_dict"])
        obj.model.eval()
        obj.fitted = True
        return obj


class _SimpleStandardizer:
    """Minimal per-dimension z-scoring fallback."""

    def __init__(self, x: np.ndarray, eps: float = 1e-6):
        x = _as_2d(x)
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), eps)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (_as_2d(x) - self.mean[None, :]) / self.std[None, :]

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return _as_2d(z) * self.std[None, :] + self.mean[None, :]


# ---------------------------------------------------------------------------
# posterior-only Simformer variant
# ---------------------------------------------------------------------------
class SimformerPosteriorOnlyBaseline:
    """Simformer trained only on the posterior conditional (``M_C`` posterior).

    This isolates the benefit of the full mixture of condition masks
    (joint / posterior / likelihood / random, Sec. 3.1 & A2.1) from the
    amortised transformer architecture: everything else (tokenizer, 6-layer
    4-head transformer, VESDE, 500-step reverse SDE) is identical.
    """

    method = "npse_simformer"

    def __init__(self, task: Any, config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None, **kwargs):
        if not _HAS_TORCH:
            raise ImportError("torch is required for SimformerPosteriorOnlyBaseline")
        self.task = task
        self.config = NPSEConfig.from_dict(config, **kwargs)
        if not self.config.condition_mask_modes:
            self.config.condition_mask_modes = ("posterior",)
        self.device = self.config.device
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.attention_mask = None
        self.history: Dict[str, Any] = {}
        self.fitted = False

    def _attention_mask(self, n_parameters: int, n_data: int):
        try:
            mod = _import_first(
                ["simformer.simformer.attention_masks", "simformer.attention_masks"], "attention masks"
            )
            return mod.build_attention_mask(self._task_name(), n_theta=n_parameters, n_x=n_data)
        except Exception:
            return None

    def _task_name(self) -> str:
        return str(getattr(self.task, "name", self.task.__class__.__name__.lower()))

    def fit(self, n_simulations: Optional[int] = None, *, seed: Optional[int] = None, **kwargs) -> "SimformerPosteriorOnlyBaseline":
        training = _import_first(
            ["simformer.simformer.training", "simformer.training"], "simformer training module"
        )
        transformer = _import_first(
            ["simformer.simformer.transformer", "simformer.transformer"], "simformer transformer"
        )
        n = int(n_simulations or self.config.n_simulations)
        seed = self.config.seed if seed is None else int(seed)

        theta, x = NPSEBaseline(self.task, self.config).make_dataset(n, seed=seed)
        n_parameters, n_data = int(theta.shape[1]), int(x.shape[1])

        if hasattr(self.task, "build_tokenizer"):
            self.tokenizer = self.task.build_tokenizer()
        else:  # pragma: no cover
            tokenizer_mod = _import_first(
                ["simformer.simformer.tokenizer", "simformer.tokenizer"], "tokenizer module"
            )
            self.tokenizer = tokenizer_mod.Tokenizer(
                tokenizer_mod.build_benchmark_spec(n_parameters, n_data)
            )
        self.model = transformer.build_score_network(
            task=self._task_name(),
            spec=getattr(self.tokenizer, "spec", None),
            token_dim=getattr(self.tokenizer, "token_dim", 50),
        )
        self.attention_mask = self._attention_mask(n_parameters, n_data)

        train_cfg = training.TrainingConfig(
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            max_steps=int(kwargs.get("max_steps", 50000)),
            condition_mask_modes=tuple(self.config.condition_mask_modes),
            seed=seed,
            device=self.device,
        )
        self.trainer = training.SimformerTrainer(
            model=self.model,
            sde=self.config.build_sde(),
            attention_mask=self.attention_mask,
            tokenizer=self.tokenizer,
            config=train_cfg,
        )
        joint = np.concatenate([theta, x], axis=1)
        self.history = self.trainer.fit(joint, verbose=self.config.verbose)
        self.fitted = True
        return self

    def posterior_samples(
        self, x_obs: np.ndarray, n_samples: int = 1000, *, seed: int = 0, n_steps: Optional[int] = None, **kwargs
    ) -> np.ndarray:
        sampling_mod = _import_first(
            ["simformer.simformer.sampling", "simformer.sampling"], "sampling module"
        )
        sampler = sampling_mod.ConditionalSampler(
            self.model,
            sde=self.trainer.sde if self.trainer is not None else self.config.build_sde(),
            tokenizer=self.tokenizer,
            attention_mask=self.attention_mask,
        )
        n_parameters = int(np.asarray(x_obs).reshape(-1).size * 0 + self.tokenizer.n_parameter_variables)
        n_data = self.tokenizer.n_data_variables
        out = sampler.posterior(
            x_obs,
            n_samples=n_samples,
            n_steps=int(n_steps or self.config.n_sampling_steps),
            seed=seed,
        )
        return np.asarray(getattr(out, "samples", out), dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config.to_dict(),
            "fitted": self.fitted,
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# builders / training entry points
# ---------------------------------------------------------------------------
def build_npse(task: Any, config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None, **kwargs) -> NPSEBaseline:
    """Construct (but do not train) an NPSE baseline."""
    return NPSEBaseline(task, config, **kwargs)


def build_npse_simformer(
    task: Any, config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None, **kwargs
) -> SimformerPosteriorOnlyBaseline:
    """Construct (but do not train) the posterior-only Simformer variant."""
    return SimformerPosteriorOnlyBaseline(task, config, **kwargs)


def train_npse(
    task: Any,
    n_simulations: int = DEFAULT_N_EVAL_SAMPLES * 10,
    config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None,
    seed: int = 0,
    verbose: bool = False,
    **kwargs,
) -> NPSEBaseline:
    """Train an NPSE baseline on ``task`` with ``n_simulations`` simulations."""
    cfg = NPSEConfig.from_dict(config, **kwargs)
    cfg.n_simulations = int(n_simulations)
    cfg.seed = int(seed)
    cfg.verbose = bool(verbose)
    model = NPSEBaseline(task, cfg)
    return model.fit(n_simulations, seed=seed, verbose=verbose)


def train_npse_simformer(
    task: Any,
    n_simulations: int = 10000,
    config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None,
    seed: int = 0,
    verbose: bool = False,
    **kwargs,
) -> SimformerPosteriorOnlyBaseline:
    cfg = NPSEConfig.from_dict(config, **kwargs)
    cfg.n_simulations = int(n_simulations)
    cfg.seed = int(seed)
    cfg.verbose = bool(verbose)
    model = SimformerPosteriorOnlyBaseline(task, cfg)
    return model.fit(n_simulations, seed=seed, verbose=verbose)


def build_baseline(method: str, task: Any, config: Optional[Union[NPSEConfig, Dict[str, Any]]] = None, **kwargs):
    """Dispatch on the method name (``npse`` / ``npse_simformer``)."""
    name = _normalize_method(method)
    if name == "npse":
        return build_npse(task, config, **kwargs)
    if name == "npse_simformer":
        return build_npse_simformer(task, config, **kwargs)
    raise ValueError(f"unknown NPSE baseline method {method!r}")


def available_methods() -> Tuple[str, ...]:
    """Return the canonical method names implemented in this module."""
    return IMPLEMENTED_METHODS


BASELINE_BUILDERS: Dict[str, Callable[..., Any]] = {
    "npse": build_npse,
    "npse_simformer": build_npse_simformer,
}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def reference_samples(
    task: Any, x_obs: np.ndarray, n_samples: int = DEFAULT_N_REFERENCE, seed: int = 0
) -> np.ndarray:
    """Ground-truth posterior samples via the task's exact sampler or MCMC."""
    if callable(reference_posterior_samples):
        try:
            return np.asarray(
                reference_posterior_samples(task, x_obs, n_samples=n_samples, seed=seed),  # type: ignore[misc]
                dtype=np.float64,
            )
        except Exception:
            pass
    for name in ("reference_posterior_sample", "ground_truth_posterior"):
        if hasattr(task, name):
            rng = np.random.default_rng(seed)
            return _as_2d(getattr(task, name)(x_obs, n_samples=n_samples, rng=rng))
    raise RuntimeError("no reference posterior sampler available for this task")


def evaluate_npse_c2st(
    baseline: Any,
    task: Any,
    *,
    n_targets: int = DEFAULT_N_TARGETS,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    n_steps: Optional[int] = None,
    seed: int = 0,
    verbose: bool = False,
    use_mcmc: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """C2ST of the NPSE posterior against ground-truth reference samples.

    Mirrors the benchmark protocol of Sec. 4.1: for each of ``n_targets``
    observations, draw ``n_samples`` approximate and ``n_reference`` reference
    posterior samples and classify them with a 100-tree random forest
    (0.5 = indistinguishable = perfect).
    """
    from ..eval.c2st import c2st_accuracy  # local import: optional sklearn dep

    rng = np.random.default_rng(seed)
    n_targets = max(1, int(n_targets))
    accuracies: List[float] = []
    for i in range(n_targets):
        r = np.random.default_rng(seed + 1000 * i)
        if hasattr(task, "sample_joint"):
            theta_gt, x_gt = task.sample_joint(1, rng=r)
        else:
            theta_gt = _as_2d(task.prior_sample(1, rng=r))
            x_gt = _as_2d(task.simulate(theta_gt, rng=r))
        x_obs = x_gt[0]
        approx = baseline.posterior_samples(x_obs, n_samples, seed=seed + i, n_steps=n_steps)
        approx = _as_2d(approx)
        try:
            ref = reference_samples(task, x_obs, n_samples=n_reference, seed=seed + i)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[npse] reference sampling failed: {exc!r}")
            continue
        acc = c2st_accuracy(np.asarray(approx), np.asarray(ref), n_trees=100, seed=seed + i)
        accuracies.append(float(np.asarray(acc).reshape(-1)[0]))
        if verbose:
            print(f"[npse] target {i + 1}/{n_targets}: C2ST = {accuracies[-1]:.3f}")

    values = np.asarray(accuracies, dtype=np.float64)
    return {
        "method": getattr(baseline, "method", "npse"),
        "c2st_mean": float(values.mean()) if values.size else float("nan"),
        "c2st_std": float(values.std()) if values.size else float("nan"),
        "c2st_min": float(values.min()) if values.size else float("nan"),
        "c2st_max": float(values.max()) if values.size else float("nan"),
        "n_targets": int(values.size),
        "per_target": [float(v) for v in values],
    }


def summarize(results: Dict[str, Any]) -> Dict[str, float]:
    """Reduce an :func:`evaluate_npse_c2st` output to a flat summary dict."""
    return {
        "c2st_mean": float(results.get("c2st_mean", float("nan"))),
        "c2st_std": float(results.get("c2st_std", float("nan"))),
        "n_targets": int(results.get("n_targets", 0)),
    }


__all__ = [
    "NPSEConfig",
    "NPSEBaseline",
    "SimformerPosteriorOnlyBaseline",
    "ConditionalScoreMLP",
    "TimeFourierEmbedding",
    "NPSESampler",
    "build_npse",
    "build_npse_simformer",
    "train_npse",
    "train_npse_simformer",
    "build_baseline",
    "available_methods",
    "BASELINE_BUILDERS",
    "evaluate_npse_c2st",
    "reference_samples",
    "summarize",
    "IMPLEMENTED_METHODS",
    "METHOD_ALIASES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_HIDDEN_DIMS",
    "DEFAULT_TIME_EMBED_DIM",
    "DEFAULT_SDE",
    "DEFAULT_N_SAMPLING_STEPS",
    "DEFAULT_MIN_SAMPLING_STEPS",
]
