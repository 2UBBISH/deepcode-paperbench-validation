"""Neural Posterior Score Estimation (NPSE), Section 2.2."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .diffusion import DiffusionConfig, DiffusionPosterior
from .networks import EnergyNetwork, ScoreNetwork
from .normalization import Standardizer
from .sde import SDE, compute_sigma_max, get_sde
from .training import train_score_network


@dataclass
class TrainingConfig:
    """Hyperparameters from Section 5.1 / Appendix E.3.2."""

    lr: float = 1e-4
    batch_size: int = 50  # 50 (non-seq, N<=10^4), 200 (seq, N<=10^4), 500 (N=10^5)
    max_iters: int = 3000
    val_fraction: float = 0.15
    patience: int = 1000
    loss_weight: str = "sigma^2"  # lambda_t = sigma_t^2
    hidden_dim: int = 256
    num_layers: int = 3
    time_embed_dim: int = 64
    t_scale: float = 1000.0
    seed: int = 0


class NPSE:
    """Non-sequential Neural Posterior Score Estimation.

    Trains a conditional score-based diffusion model
    ``s_psi(theta_t, x, t) ~ grad_theta log p_t(theta_t | x)`` on samples
    ``(theta, x) ~ p(theta) p(x | theta)`` and generates posterior samples by
    integrating the probability flow ODE (Eq. 4) with ``x = x_obs``.
    """

    def __init__(
        self,
        dim_parameters: int,
        dim_data: int,
        prior,
        sde: str = "ve",
        sde_kwargs: Optional[dict] = None,
        sigma_min: float = 0.05,
        diffusion_config: Optional[DiffusionConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        device: str = "cpu",
        parameterisation: str = "score",
    ) -> None:
        self.dim_parameters = dim_parameters
        self.dim_data = dim_data
        self.prior = prior  # distribution over theta in the *original* space
        self.sde_name = sde
        self.sde_kwargs = dict(sde_kwargs or {})
        self.sigma_min = sigma_min
        self.diffusion_config = diffusion_config or DiffusionConfig()
        self.training_config = training_config or TrainingConfig()
        self.device = device
        self.parameterisation = parameterisation

        self.theta_scaler: Optional[Standardizer] = None
        self.x_scaler: Optional[Standardizer] = None
        self.sde: Optional[SDE] = None
        self.net: Optional[torch.nn.Module] = None
        self.posterior: Optional[DiffusionPosterior] = None
        self.train_info: dict = {}

    # ------------------------------------------------------------------ utils
    def _build_network(self) -> torch.nn.Module:
        cfg = self.training_config
        cls = EnergyNetwork if self.parameterisation == "energy" else ScoreNetwork
        return cls(
            dim_parameters=self.dim_parameters,
            dim_data=self.dim_data,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            time_embed_dim=cfg.time_embed_dim,
            t_scale=cfg.t_scale,
        ).to(self.device)

    def _make_sde(self, z_reference_data: Optional[torch.Tensor] = None) -> SDE:
        kwargs = dict(self.sde_kwargs)
        if self.sde_name.lower() in ("ve", "vesde"):
            kwargs.setdefault("sigma_min", self.sigma_min)
            if "sigma_max" not in kwargs:
                if z_reference_data is None:
                    raise ValueError("VE SDE requires either sigma_max or data to compute it")
                kwargs["sigma_max"] = compute_sigma_max(z_reference_data, technique=1)
        return get_sde(self.sde_name, **kwargs)

    def fit_standardizers(self, theta: torch.Tensor, x: torch.Tensor) -> None:
        self.theta_scaler = Standardizer.fit(theta)
        self.x_scaler = Standardizer.fit(x)

    def set_standardizers(self, theta_scaler: Standardizer, x_scaler: Standardizer) -> None:
        self.theta_scaler = theta_scaler
        self.x_scaler = x_scaler

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        theta_scaler: Optional[Standardizer] = None,
        x_scaler: Optional[Standardizer] = None,
        sigma_max: Optional[float] = None,
        init_network: Optional[torch.nn.Module] = None,
        sample_weights: Optional[torch.Tensor] = None,
        loss_weight: Optional[str] = None,
        verbose: bool = True,
    ) -> "NPSE":
        """Train the score network on ``(theta, x)`` (in the original space)."""
        theta = theta.reshape(-1, self.dim_parameters).to(self.device)
        x = x.reshape(-1, self.dim_data).to(self.device)
        if theta_scaler is None or x_scaler is None:
            self.fit_standardizers(theta, x)
        else:
            self.set_standardizers(theta_scaler, x_scaler)
        assert self.theta_scaler is not None and self.x_scaler is not None

        z = self.theta_scaler.transform(theta)
        x_std = self.x_scaler.transform(x)

        if sigma_max is not None:
            self.sde_kwargs["sigma_max"] = sigma_max
        self.sde = self._make_sde(z)

        self.net = init_network if init_network is not None else self._build_network()
        self.net = self.net.to(self.device)
        net, info = train_score_network(
            self.net,
            self.sde,
            z,
            x_std,
            lr=self.training_config.lr,
            batch_size=self.training_config.batch_size,
            max_iters=self.training_config.max_iters,
            val_fraction=self.training_config.val_fraction,
            patience=self.training_config.patience,
            loss_weight=loss_weight or self.training_config.loss_weight,
            sample_weights=sample_weights,
            seed=self.training_config.seed,
            verbose=verbose,
        )
        self.net = net.eval()
        self.train_info = info
        self.posterior = DiffusionPosterior(
            self.net, self.sde, self.dim_parameters, copy.deepcopy(self.diffusion_config), self.device
        )
        return self

    # ---------------------------------------------------------------- helpers
    def _standardize_x(self, x_obs: torch.Tensor) -> torch.Tensor:
        assert self.x_scaler is not None and self.sde is not None and self.posterior is not None
        return self.x_scaler.transform(x_obs.reshape(1, -1).to(self.device))

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample_z(self, x_obs: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Posterior samples in the standardized parameter space."""
        assert self.posterior is not None
        return self.posterior.sample(self._standardize_x(x_obs), num_samples, **kwargs)

    @torch.no_grad()
    def sample(self, x_obs: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Posterior samples in the original parameter space."""
        z = self.sample_z(x_obs, num_samples, **kwargs)
        assert self.theta_scaler is not None
        return self.theta_scaler.inverse(z)

    @torch.no_grad()
    def sample_and_log_prob_z(
        self, x_obs: torch.Tensor, num_samples: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.posterior is not None
        return self.posterior.sample_and_log_prob(self._standardize_x(x_obs), num_samples, **kwargs)

    def log_prob_z(self, x_obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Log density of the approximate posterior in ``z`` space (up to a
        constant that is independent of ``z``)."""
        assert self.posterior is not None
        return self.posterior.log_prob(self._standardize_x(x_obs), z)

    def log_prob(self, x_obs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Log density of the approximate posterior in the original space."""
        assert self.theta_scaler is not None
        z = self.theta_scaler.transform(theta.reshape(-1, self.dim_parameters))
        return self.log_prob_z(x_obs, z)

    # ------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict() if self.net is not None else None,
            "theta_scaler": self.theta_scaler.state_dict() if self.theta_scaler else None,
            "x_scaler": self.x_scaler.state_dict() if self.x_scaler else None,
            "sde_name": self.sde_name,
            "sde_kwargs": self.sde_kwargs,
            "dim_parameters": self.dim_parameters,
            "dim_data": self.dim_data,
            "parameterisation": self.parameterisation,
        }

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)
