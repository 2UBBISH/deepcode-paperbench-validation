"""Neural Posterior Score Estimation (NPSE).

This module implements the amortised (round-1) version of the SNPSE family, i.e.
Neural Posterior Score Estimation (NPSE) as described in Section 2.2 of the paper.

The recipe (see Section 2.2, "we now have all of the necessary ingredients ...") is

    (i)   draw ``theta_0 ~ p(theta)`` from the prior, ``x ~ p(x | theta_0)`` from the
          simulator and ``theta_t ~ p_{t|0}(theta_t | theta_0)`` from the forward process (2);
    (ii)  train a time-varying score network ``s_psi(theta_t, x, t)`` to approximate the
          score ``grad_theta log p_t(theta_t | x)`` of the perturbed posterior by
          minimising the Monte Carlo estimate of the conditional denoising posterior score
          matching objective (7);
    (iii) draw ``theta_T ~ pi`` and simulate an approximation of the reverse-time process (3)
          or of the time-reversal of the probability-flow ODE (4) with ``x = x_obs``,
          replacing ``grad_theta log p_t(theta_t | x_obs)`` by ``s_psi(theta_t, x_obs, t)``.

The probability-flow ODE additionally allows evaluating the density of the resulting
samples via the instantaneous change-of-variables formula (5).

In the sequential algorithm (Algorithm 1) the first round is exactly this amortised
objective, with the prior used as the proposal ``pbar^0(theta) = p(theta)``:

    for r = 1, ..., R:
        draw theta_i ~ pbar^{r-1}(theta), x_i ~ p(x | theta_i), add (theta_i, x_i) to D
        learn s_psi by minimising a Monte Carlo estimate of (11) based on dataset D
        compute pbar^r(theta) using s_psi

Hence the training utilities below are shared with :mod:`snpse.tsnpse`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:  # relative imports when used as ``snpse.snpse.npse``
    from .sdes import SDE, get_sde, sigma_max_technique1, T_FINAL
    from .score_network import ScoreNetwork, get_score_network, count_parameters
    from .losses import npse_loss, tsnpse_loss, DSMScoreFn
    from .sampler import (
        SamplerConfig,
        sample_posterior,
        estimate_log_prob,
        reverse_sde_sample,
    )
    from .trainer import TrainConfig, TrainHistory, Trainer, train_network, select_batch_size
    from .utils import (
        Standardiser,
        fit_standardiser,
        set_seed,
        get_device,
        ensure_2d,
    )
except ImportError:  # pragma: no cover - standalone execution
    from sdes import SDE, get_sde, sigma_max_technique1, T_FINAL  # type: ignore
    from score_network import ScoreNetwork, get_score_network, count_parameters  # type: ignore
    from losses import npse_loss, tsnpse_loss  # type: ignore

    DSMScoreFn = Callable[..., torch.Tensor]  # type: ignore
    from sampler import (  # type: ignore
        SamplerConfig,
        sample_posterior,
        estimate_log_prob,
        reverse_sde_sample,
    )
    from trainer import (  # type: ignore
        TrainConfig,
        TrainHistory,
        Trainer,
        train_network,
        select_batch_size,
    )
    from utils import (  # type: ignore
        Standardiser,
        fit_standardiser,
        set_seed,
        get_device,
        ensure_2d,
    )

__all__ = [
    "NPSEConfig",
    "NPSE",
    "train_score_network",
    "standardise_dataset",
    "run_npse",
]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class NPSEConfig:
    """Hyper-parameters for NPSE round-1 / amortised training.

    Defaults follow Section 5.1 and Appendix E.3.2 of the paper:

    * Adam with learning rate ``1e-4`` and at most ``3000`` iterations;
    * batch size ``50``/``200`` for budgets ``1e3``/``1e4`` and ``500`` for ``1e5``
      (see :func:`snpse.trainer.select_batch_size`);
    * ``15%`` of the data is held back for validation and training stops if the
      validation loss has not improved for ``1000`` steps, restoring the best weights;
    * the score network consists of three embedding MLPs (theta, x, t) followed by a
      3-layer 256-unit head (:mod:`snpse.score_network`);
    * the probability-flow ODE is solved with RK45 at ``atol = rtol = 1e-5``.

    Values left unspecified in the paper use the documented defaults (see
    ``configs/default.yaml``).
    """

    # --- SDE (Section E.3.1 / Section 2.2 eq. 2) ---
    sde: str = "ve"
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    beta_min: float = 0.1
    beta_max: float = 11.0
    t_final: float = T_FINAL

    # --- score network (Section 5.1 / Appendix E.3.2) ---
    hidden_dim: int = 256
    n_layers: int = 3
    time_emb_dim: int = 64
    theta_emb_dim: Optional[int] = None
    x_emb_dim: Optional[int] = None
    parameterisation: str = "score"  # "score" | "energy"
    activation: str = "silu"

    # --- training (Section 5.1 / Appendix E.3.2) ---
    lr: float = 1e-4
    max_iters: int = 3000
    batch_size: Optional[int] = None
    budget: Optional[int] = None  # used to pick the paper batch size when batch_size is None
    val_fraction: float = 0.15
    patience: int = 1000
    min_delta: float = 0.0
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    log_every: int = 100

    # --- standardisation (Section E.3.2) ---
    standardise: bool = True

    # --- sampling (Section E.3.3) ---
    sampler_method: str = "rk45"
    sampler_atol: float = 1e-5
    sampler_rtol: float = 1e-5
    sampler_n_steps: int = 1000
    trace_estimator: str = "auto"

    # --- misc ---
    seed: Optional[int] = 0
    device: Optional[str] = None
    dtype_name: str = "float32"
    verbose: bool = False

    def resolved_batch_size(self) -> int:
        if self.batch_size is not None:
            return int(self.batch_size)
        if self.budget is not None:
            return int(select_batch_size(int(self.budget)))
        return 200

    def train_config(self) -> TrainConfig:
        kwargs: Dict[str, Any] = dict(
            lr=self.lr,
            max_iters=self.max_iters,
            batch_size=self.resolved_batch_size(),
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
            return TrainConfig(**kwargs)
        except TypeError:  # tolerate a differently named field set
            kwargs.pop("grad_clip", None)
            return TrainConfig(**kwargs)

    def sampler_config(self) -> SamplerConfig:
        kwargs: Dict[str, Any] = dict(
            method=self.sampler_method,
            atol=self.sampler_atol,
            rtol=self.sampler_rtol,
            n_steps=self.sampler_n_steps,
            t_max=self.t_final,
            trace_estimator=self.trace_estimator,
            seed=self.seed,
        )
        try:
            return SamplerConfig(**kwargs)
        except TypeError:  # pragma: no cover
            kwargs.pop("trace_estimator", None)
            return SamplerConfig(**kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
    }.get(str(name).lower(), torch.float32)


def _dtype_of(value: Optional[torch.Tensor], default: torch.dtype) -> torch.dtype:
    return value.dtype if isinstance(value, torch.Tensor) else default


def standardise_dataset(
    theta: torch.Tensor,
    x: torch.Tensor,
    standardise: bool = True,
    theta_standardiser: Optional[Standardiser] = None,
    x_standardiser: Optional[Standardiser] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Standardiser, Standardiser]:
    """Fit (or reuse) affine standardisers and whiten ``theta`` and ``x``.

    Appendix E.3.2 standardises both the parameters and the observations before
    training the score network. The inverse transform is required by the sampler so
    that posterior samples are returned in the original parameter space.

    Returns
    -------
    theta_std, x_std, theta_standardiser, x_standardiser
    """
    theta = ensure_2d(theta, "theta")
    x = ensure_2d(x, "x")

    if theta_standardiser is None:
        theta_standardiser = (
            fit_standardiser(theta) if standardise else Standardiser.identity(theta.shape[-1], dtype=theta.dtype)
        )
    if x_standardiser is None:
        x_standardiser = (
            fit_standardiser(x) if standardise else Standardiser.identity(x.shape[-1], dtype=x.dtype)
        )

    theta_std = theta_standardiser.to_std(theta)
    x_std = x_standardiser.to_std(x)
    return theta_std, x_std, theta_standardiser, x_standardiser


def build_sde(
    config: NPSEConfig,
    theta: Optional[torch.Tensor] = None,
    dim: Optional[int] = None,
) -> SDE:
    """Construct the forward noising SDE (eq. 2) for the given configuration.

    Notes
    -----
    * ``sigma_min`` defaults to ``0.01`` for 2-dimensional problems (SIR, Two Moons)
      and ``0.05`` otherwise (Appendix E.3.1).
    * For the VE SDE, ``sigma_max`` defaults to the Song & Ermon (2020) Technique 1
      heuristic computed from ``theta``.
    * Addendum requirement for the sequential methods: when the VE SDE is used within
      TSNPSE/SNPSE-*, ``sigma_max`` must be estimated from the *first-round* training
      data only. Pass ``theta`` = round-1 standardised parameters to honour this.
    """
    if dim is None:
        if theta is not None:
            dim = int(theta.shape[-1])
        else:
            raise ValueError("Either `theta` or `dim` must be provided to build the SDE.")

    kwargs: Dict[str, Any] = dict(dim=dim, eps=1e-8)
    if config.sde.lower() in ("ve", "vpsde", "variance_exploding"):
        pass

    sigma_max = config.sigma_max
    if config.sde.lower() == "ve" and sigma_max is None and theta is not None:
        sigma_max = float(sigma_max_technique1(theta))

    sde = get_sde(
        name=config.sde,
        sigma_min=config.sigma_min,
        sigma_max=sigma_max,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        data=theta,
        dim=dim,
    )
    # allow a non-default terminal time if a custom SDE class exposes it
    if getattr(sde, "T", T_FINAL) != config.t_final:
        try:
            sde = get_sde(
                name=config.sde,
                sigma_min=getattr(sde, "sigma_min", config.sigma_min),
                sigma_max=getattr(sde, "sigma_max", config.sigma_max),
                beta_min=config.beta_min,
                beta_max=config.beta_max,
                data=theta,
                dim=dim,
            )
        except TypeError:  # pragma: no cover
            pass
    return sde


def _make_loss_fn(
    network: nn.Module,
    sde: SDE,
    objective: Callable[..., torch.Tensor],
    generator: Optional[torch.Generator] = None,
) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    """Wrap a DSM objective from :mod:`snpse.losses` into the trainer's closure API.

    The trainer hands the closure a batch dict ``{"theta", "x", optional "weight"}`` of
    already-standardised tensors and expects a scalar loss.
    """

    def loss_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        weight = batch.get("weight", None)
        kwargs: Dict[str, Any] = {}
        if weight is not None:
            kwargs["weight"] = weight
        if generator is not None and "generator" not in kwargs:
            kwargs["generator"] = generator
        return objective(score_fn=network, sde=sde, theta0=batch["theta"], x=batch["x"], **kwargs)

    return loss_fn


def train_score_network(
    theta: torch.Tensor,
    x: torch.Tensor,
    config: Optional[NPSEConfig] = None,
    sde: Optional[SDE] = None,
    network: Optional[nn.Module] = None,
    weights: Optional[torch.Tensor] = None,
    theta_standardiser: Optional[Standardiser] = None,
    x_standardiser: Optional[Standardiser] = None,
    objective: Optional[Callable[..., torch.Tensor]] = None,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    split: Optional[Dict[str, torch.Tensor]] = None,
    verbose: Optional[bool] = None,
) -> Tuple[nn.Module, SDE, Dict[str, Any]]:
    """Train a conditional score network on an accumulated dataset.

    This is the shared training routine used both by the amortised NPSE objective (7)
    and by the sequential TSNPSE objective (11); the two differ only through the
    distribution of the ``theta`` samples that are passed in (prior vs. truncated
    proposal ``tilde p^r``) and, for SNPSE-B, through the importance ``weights``.

    Parameters
    ----------
    theta, x
        Parameter/simulation pairs ``D`` collected so far (original space).
    config
        See :class:`NPSEConfig`.
    sde
        Optional pre-built SDE (e.g. built from round-1 data for the sequential methods).
    network
        Optional existing network to fine-tune (defaults to a freshly initialised one).
    weights
        Optional per-datapoint importance weights ``p(theta_0) / tilde p^r(theta_0)``
        used by the SNPSE-B objective (15).
    objective
        DSM objective from :mod:`snpse.losses`; defaults to ``npse_loss`` (eq. 7).

    Returns
    -------
    network, sde, info
        ``info`` contains the standardisers, the history and the raw training tensors so
        that sequential drivers can reuse the exact same transform across rounds.
    """
    config = config or NPSEConfig()
    if verbose is None:
        verbose = config.verbose
    device = device if device is not None else get_device(config.device)
    dtype = _torch_dtype(config.dtype_name)

    if config.seed is not None and generator is None:
        generator = set_seed(config.seed)

    theta = ensure_2d(theta, "theta")
    x = ensure_2d(x, "x")
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=dtype)
        if weights.ndim == 0:
            weights = weights.expand(theta.shape[0])
        weights = weights.reshape(-1)

    theta_std, x_std, theta_std_obj, x_std_obj = standardise_dataset(
        theta.to(dtype),
        x.to(dtype),
        standardise=config.standardise,
        theta_standardiser=theta_standardiser,
        x_standardiser=x_standardiser,
    )

    n_data = int(theta.shape[0])
    budget = config.budget if config.budget is not None else n_data
    if config.batch_size is None:
        try:
            config = copy.copy(config)
            config.batch_size = select_batch_size(int(budget))
        except Exception:  # pragma: no cover
            pass

    if sde is None:
        sde = build_sde(config, theta=theta_std, dim=int(theta.shape[-1]))

    if network is None:
        network = get_score_network(
            theta_dim=int(theta.shape[-1]),
            x_dim=int(x.shape[-1]),
            parameterisation=config.parameterisation,
            hidden_dim=config.hidden_dim,
            n_layers=config.n_layers,
            time_emb_dim=config.time_emb_dim,
            theta_emb_dim=config.theta_emb_dim,
            x_emb_dim=config.x_emb_dim,
            activation=config.activation,
        )
    network = network.to(device=device, dtype=dtype)

    objective = objective or npse_loss
    loss_fn = _make_loss_fn(network, sde, objective, generator=generator)

    trainer = Trainer(network, loss_fn, config=config.train_config(), device=device)
    history = trainer.fit(
        theta_std.to(device),
        x_std.to(device),
        weights=None if weights is None else weights.to(device),
        split=split,
        generator=generator,
    )

    info: Dict[str, Any] = {
        "theta_standardiser": theta_std_obj,
        "x_standardiser": x_std_obj,
        "theta_std": theta_std,
        "x_std": x_std,
        "history": history,
        "sde": sde,
        "n_data": n_data,
        "batch_size": config.resolved_batch_size(),
        "n_parameters": count_parameters(network),
        "split": getattr(history, "val_indices", None),
    }
    if verbose:
        print(
            f"[NPSE] trained on {n_data} simulations | batch={info['batch_size']} "
            f"| params={info['n_parameters']} | best_val={getattr(history, 'best_val_loss', float('nan')):.4e} "
            f"| iters={getattr(history, 'n_iters', -1)}"
        )
    return network, sde, info


# --------------------------------------------------------------------------------------
# NPSE
# --------------------------------------------------------------------------------------
class NPSE:
    """Amortised Neural Posterior Score Estimation (Section 2.2).

    Example
    -------
    >>> npse = NPSE(theta_dim=2, x_dim=2, config=NPSEConfig(sde="ve"))
    >>> npse.fit(theta, x)                     # theta ~ p(theta), x ~ p(x | theta)
    >>> samples = npse.sample(x_obs, 10_000)   # posterior samples
    >>> logp = npse.log_prob(x_obs, samples)   # log density via eq. (5)
    """

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        config: Optional[NPSEConfig] = None,
        device: Optional[torch.device] = None,
        network: Optional[nn.Module] = None,
    ) -> None:
        self.config = config or NPSEConfig()
        self.theta_dim = int(theta_dim)
        self.x_dim = int(x_dim)
        self.device = device if device is not None else get_device(self.config.device)
        self.dtype = _torch_dtype(self.config.dtype_name)
        self.network: Optional[nn.Module] = network
        self.sde: Optional[SDE] = None
        self.theta_standardiser: Optional[Standardiser] = None
        self.x_standardiser: Optional[Standardiser] = None
        self.history: Optional[TrainHistory] = None
        self.info: Dict[str, Any] = {}
        self._fitted = False

    # -- training ----------------------------------------------------------------------
    def fit(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        sde: Optional[SDE] = None,
        objective: Optional[Callable[..., torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
        split: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "NPSE":
        """Minimise a Monte Carlo estimate of the DSM objective (7).

        ``theta`` must be drawn from the prior and ``x`` simulated from the
        corresponding likelihood, which is exactly step (i) of the Section 2.2 recipe.
        """
        network, sde, info = train_score_network(
            theta=theta,
            x=x,
            config=self.config,
            sde=sde,
            network=self.network,
            weights=weights,
            theta_standardiser=self.theta_standardiser,
            x_standardiser=self.x_standardiser,
            objective=objective,
            device=self.device,
            generator=generator,
            split=split,
        )
        self.network = network
        self.sde = sde
        self.theta_standardiser = info["theta_standardiser"]
        self.x_standardiser = info["x_standardiser"]
        self.history = info["history"]
        self.info = info
        self._fitted = True
        return self

    def loss(self, theta: torch.Tensor, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Evaluate the DSM objective (7) on a dataset without updating parameters."""
        self._check_fitted(require=False)
        if self.sde is None:
            self.sde = build_sde(self.config, theta=theta, dim=int(theta.shape[-1]))
        network = self.network
        if network is None:
            network = get_score_network(
                theta_dim=int(theta.shape[-1]),
                x_dim=int(x.shape[-1]),
                parameterisation=self.config.parameterisation,
                hidden_dim=self.config.hidden_dim,
                n_layers=self.config.n_layers,
                time_emb_dim=self.config.time_emb_dim,
            ).to(self.device)
        value = npse_loss(score_fn=network, sde=self.sde, theta0=theta, x=x, **kwargs)
        return value

    # -- inference ---------------------------------------------------------------------
    def _check_fitted(self, require: bool = True) -> None:
        if require and not self._fitted:
            raise RuntimeError("NPSE must be fitted before inference; call `.fit(theta, x)` first.")

    def _sampler_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.theta_standardiser is not None:
            kwargs["theta_shift"] = self.theta_standardiser.shift
            kwargs["theta_scale"] = self.theta_standardiser.scale
        if self.x_standardiser is not None:
            kwargs["x_shift"] = self.x_standardiser.shift
            kwargs["x_scale"] = self.x_standardiser.scale
        return kwargs

    def sample(
        self,
        x_obs: torch.Tensor,
        num_samples: int,
        with_log_prob: bool = False,
        method: str = "ode",
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> Any:
        """Draw posterior samples for ``x_obs`` (step (iii) of the Section 2.2 recipe).

        ``method="ode"`` integrates the time-reversal of the probability-flow ODE (4)
        (deterministic, allows density evaluation); ``method="sde"`` simulates the
        reverse-time SDE (3) stochastically.
        """
        self._check_fitted()
        cfg = self.config.sampler_config()
        common = dict(
            sde=self.sde,
            score_fn=self.network,
            x_obs=x_obs,
            num_samples=int(num_samples),
            config=cfg,
            theta_dim=self.theta_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            **self._sampler_kwargs(),
        )
        common.update(kwargs)
        if method in ("sde", "reverse_sde", "em"):
            return reverse_sde_sample(**common)
        return sample_posterior(with_log_prob=with_log_prob, **common)

    def log_prob(self, x_obs: torch.Tensor, theta: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Evaluate ``log p_psi(theta | x_obs)`` via the change-of-variables formula (5)."""
        self._check_fitted()
        common = dict(
            sde=self.sde,
            score_fn=self.network,
            x_obs=x_obs,
            theta=theta,
            config=self.config.sampler_config(),
            theta_dim=self.theta_dim,
            device=self.device,
            dtype=self.dtype,
            **self._sampler_kwargs(),
        )
        common.update(kwargs)
        return estimate_log_prob(**common)

    # convenience aliases used by the experiment drivers ------------------------------
    posterior_samples = sample
    posterior_log_prob = log_prob

    # -- misc --------------------------------------------------------------------------
    def score(self, theta: torch.Tensor, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the trained score ``s_psi(theta_t, x, t)``.

        ``theta`` is expected in the original parameter space when the model was fitted
        with standardisation; it is whitened here so that the returned value corresponds
        to ``grad_theta log p_t(theta | x)`` in the original space.
        """
        self._check_fitted()
        theta = ensure_2d(theta, "theta")
        x = ensure_2d(x, "x")
        theta_t = self.theta_standardiser.to_std(theta) if self.theta_standardiser else theta
        x_t = self.x_standardiser.to_std(x) if self.x_standardiser else x
        if t is None:
            t = torch.zeros(theta_t.shape[0], dtype=theta_t.dtype, device=theta_t.device)
        elif not torch.is_tensor(t):
            t = torch.full((theta_t.shape[0],), float(t), dtype=theta_t.dtype, device=theta_t.device)
        out = self.network(theta_t.to(self.device), x_t.to(self.device), t.to(self.device))
        if isinstance(out, tuple):
            out = out[0]
        if self.theta_standardiser is not None:
            out = out / self.theta_standardiser.scale.to(out.device)
        return out

    def state_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network.state_dict() if self.network is not None else None,
            "config": self.config.as_dict(),
            "theta_dim": self.theta_dim,
            "x_dim": self.x_dim,
            "theta_standardiser": self.theta_standardiser.state_dict() if self.theta_standardiser else None,
            "x_standardiser": self.x_standardiser.state_dict() if self.x_standardiser else None,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> "NPSE":
        payload = torch.load(path, map_location=self.device)
        if payload.get("network") is not None and self.network is None:
            self.network = get_score_network(
                theta_dim=self.theta_dim,
                x_dim=self.x_dim,
                parameterisation=self.config.parameterisation,
                hidden_dim=self.config.hidden_dim,
                n_layers=self.config.n_layers,
                time_emb_dim=self.config.time_emb_dim,
            ).to(self.device)
        if payload.get("network") is not None:
            self.network.load_state_dict(payload["network"])
        if payload.get("theta_standardiser") is not None:
            self.theta_standardiser = self.theta_standardiser or Standardiser(
                payload["theta_standardiser"]["shift"], payload["theta_standardiser"]["scale"]
            )
            self.theta_standardiser = Standardiser(
                payload["theta_standardiser"]["shift"], payload["theta_standardiser"]["scale"]
            )
        if payload.get("x_standardiser") is not None:
            self.x_standardiser = Standardiser(
                payload["x_standardiser"]["shift"], payload["x_standardiser"]["scale"]
            )
        self.sde = build_sde(self.config, dim=self.theta_dim)
        self._fitted = True
        return self


# --------------------------------------------------------------------------------------
# functional entry point
# --------------------------------------------------------------------------------------
def run_npse(
    theta: torch.Tensor,
    x: torch.Tensor,
    x_obs: torch.Tensor,
    num_samples: int = 10_000,
    config: Optional[NPSEConfig] = None,
    with_log_prob: bool = False,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Train NPSE on ``(theta, x)`` and return posterior samples and log densities.

    Implements one pass of the Section 2.2 recipe (steps (i)-(iii)) with the DSM
    objective (7) and the time-reversed probability-flow ODE (4).
    """
    config = config or NPSEConfig()
    if seed is not None:
        config = copy.copy(config)
        config.seed = seed
    model = NPSE(
        theta_dim=int(theta.shape[-1]),
        x_dim=int(x.shape[-1]),
        config=config,
        device=device,
    )
    model.fit(theta, x)
    theta_samples, log_prob = model.sample(x_obs, num_samples, with_log_prob=True)
    result: Dict[str, Any] = {
        "theta": theta_samples,
        "log_prob": log_prob,
        "model": model,
        "info": model.info,
    }
    if not with_log_prob:
        result.pop("log_prob")
    return result


# --------------------------------------------------------------------------------------
# validation utility (Section 5.1 sanity checks)
# --------------------------------------------------------------------------------------
def _selftest_sde_scores(dtype: torch.dtype = torch.float32) -> None:
    """Finite-difference check of the analytic transition score targets (eq. 6)."""
    d = 3
    theta0 = torch.randn(5, d, dtype=dtype)
    for name in ("ve", "vp"):
        sde = get_sde(name, dim=d)
        theta_t, t, target = torch.zeros(5, d), torch.zeros(5), torch.zeros(5, d)
        torch.manual_seed(0)
        theta_t, t, target = sde.sample_perturbed(theta0, t=None)
        t_col = t.reshape(-1, 1) if t.ndim == 1 else t
        mean = sde.marginal_mean(theta0, t)
        std = sde.marginal_std(t)
        std = std.reshape(-1, 1) if std.ndim == 1 else std
        num = -(theta_t - mean) / std.pow(2)
        err = (num - target).abs().max().item()
        assert err < 1e-4, f"{name}: analytic score target mismatch ({err:.2e})"
        # finite differences of log p_{t|0}
        eps = 1e-5
        def log_trans(z):
            diff = z - mean
            return -0.5 * (diff.pow(2) / std.pow(2)).sum(-1) - std.log().sum(-1) * 0 - 0.5 * d * (2 * torch.log(std.squeeze(-1)))
        base = log_trans(theta_t)
        fd = torch.zeros_like(theta_t)
        for j in range(d):
            e = torch.zeros_like(theta_t)
            e[:, j] = eps
            fd[:, j] = (log_trans(theta_t + e) - log_trans(theta_t - e)) / (2 * eps)
        assert (fd - target).abs().max().item() < 1e-3, f"{name}: finite-difference score mismatch"
        print(f"[selftest] {name.upper()} SDE analytic score target verified")


def _selftest_gaussian(dtype: torch.dtype = torch.float32, n_train: int = 3000) -> float:
    """Train NPSE on a 2D Gaussian linear model and compare with the analytic score.

    Model: ``theta ~ N(0, I_2)``, ``x | theta ~ N(theta, sigma^2 I_2)`` with
    ``sigma = 0.5``. The posterior is Gaussian with
    ``S = (I + I / sigma^2)^{-1}`` and mean ``S x / sigma^2``, so its score is
    ``-(theta - mean) / diag(S)`` for this isotropic case.
    """
    torch.manual_seed(0)
    sigma = 0.5
    d = p = 2
    theta = torch.randn(n_train, d, dtype=dtype)
    x = theta + sigma * torch.randn(n_train, p, dtype=dtype)

    config = NPSEConfig(
        sde="ve",
        sigma_min=0.01,
        hidden_dim=128,
        n_layers=3,
        max_iters=4000,
        batch_size=200,
        lr=1e-4,
        patience=1000,
        budget=n_train,
        seed=0,
    )
    model = NPSE(d, p, config=config).fit(theta, x)

    x_obs = torch.zeros(1, p, dtype=dtype)
    precision = torch.eye(d) + torch.eye(d) / sigma**2
    cov = torch.linalg.inv(precision)
    mean = (cov @ (x_obs[0] / sigma**2)).reshape(1, d)

    theta_grid = mean + 0.5 * torch.randn(200, d, dtype=dtype)
    analytic = -(theta_grid - mean) @ torch.linalg.inv(cov).T
    learned = model.score(theta_grid, x_obs.expand(theta_grid.shape[0], p), t=torch.full((theta_grid.shape[0],), 1e-3))
    rel_err = ((learned - analytic).norm(dim=-1) / (analytic.norm(dim=-1) + 1e-8)).mean().item()
    print(f"[selftest] NPSE Gaussian-linear score relative error: {rel_err:.4f}")

    samples = model.sample(x_obs, 4000)
    emp_mean = samples.mean(0)
    emp_std = samples.std(0)
    print(f"[selftest] posterior mean error: {(emp_mean - mean[0]).abs().max().item():.4f} "
          f"| true std: {cov.diagonal().sqrt().mean().item():.4f} | emp std: {emp_std.mean().item():.4f}")
    assert (emp_mean - mean[0]).abs().max().item() < 0.1, "posterior mean not recovered"
    assert ((emp_std - cov.diagonal().sqrt()).abs().max().item() < 0.1), "posterior std not recovered"
    assert rel_err < 0.5, "score approximation too inaccurate"

    logp = model.log_prob(x_obs, samples[:50])
    assert torch.isfinite(logp).all(), "log density produced non-finite values"
    return rel_err


if __name__ == "__main__":  # pragma: no cover
    print("Running NPSE self-tests (Section 5.1 validation approach) ...")
    _selftest_sde_scores()
    _selftest_gaussian()
    print("All NPSE self-tests passed.")
