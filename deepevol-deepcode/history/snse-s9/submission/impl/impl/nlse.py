"""Neural Likelihood Score Estimation with the VE SDE.

NLSE decomposes the perturbed posterior score into a likelihood score and a
perturbed prior score. The likelihood score network is trained with the
denoising likelihood score matching objective, while the prior score is
estimated with a separate denoising score matching network on prior samples.
Sampling then uses the summed posterior score in the probability flow ODE.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from impl.config import SDEConfig, TrialConfig
from impl.losses import (
    denoising_likelihood_score_matching_loss,
    denoising_posterior_score_matching_loss,
)
from impl.npse import (
    _build_sde_for_kind,
    _dataset_statistics,
    _get_param,
    _sample_from_prior,
    _simulate,
    _standardize_dataset,
)
from impl.sampler import ProbabilityFlowODESampler
from impl.score_network import ScoreNetworkMLP
from impl.training import train_score_network


class _AdditiveStandardizedScoreModel(nn.Module):
    """Sum a likelihood score model and a prior score model, both defined in
    standardised parameter space, and return the raw-space posterior score."""

    def __init__(
        self,
        raw_likelihood_model: nn.Module,
        prior_model: nn.Module,
        theta_mean: Tensor,
        theta_std: Tensor,
        x_mean: Tensor,
        x_std: Tensor,
    ):
        super().__init__()
        self.raw_likelihood_model = raw_likelihood_model
        self.prior_model = prior_model
        self.register_buffer("theta_mean", theta_mean.detach())
        self.register_buffer("theta_std", theta_std.detach())
        self.register_buffer("x_mean", x_mean.detach())
        self.register_buffer("x_std", x_std.detach())
        self.theta_dim = int(raw_likelihood_model.theta_dim)
        self.x_dim = int(raw_likelihood_model.x_dim)

    def forward(self, theta_t: Tensor, x: Tensor, t) -> Tensor:
        theta_z = (theta_t - self.theta_mean) / self.theta_std
        x_z = (x - self.x_mean) / self.x_std
        likelihood_score_z = self.raw_likelihood_model(theta_z, x_z, t)

        if theta_z.ndim == 2:
            prior_x = torch.zeros(
                theta_z.size(0), 1, dtype=theta_z.dtype, device=theta_z.device
            )
        else:
            prior_x = torch.zeros(1, dtype=theta_z.dtype, device=theta_z.device)
        prior_score_z = self.prior_model(theta_z, prior_x, t)
        return (likelihood_score_z + prior_score_z) / self.theta_std


class NLSEVE:
    """NLSE method with the variance-exploding forward SDE."""

    def __init__(
        self,
        prior,
        simulator,
        theta_dim: int | None = None,
        x_dim: int | None = None,
        setting: str = "",
        device: str = "cpu",
        sde_config: SDEConfig | None = None,
        task=None,
    ):
        if task is not None:
            prior = task.get_prior()
            simulator = task.get_simulator()
            theta_dim = getattr(task, "dim_parameters", theta_dim)
            x_dim = getattr(task, "dim_data", x_dim)
            setting = setting or getattr(task, "name", "")
        if prior is None or simulator is None:
            raise ValueError("prior and simulator are required")

        self.prior = prior
        self.simulator = simulator
        self.theta_dim = None if theta_dim is None else int(theta_dim)
        self.x_dim = None if x_dim is None else int(x_dim)
        self.setting = str(setting or "")
        self.device = str(device)
        self.sde_config = sde_config
        self.model: _AdditiveStandardizedScoreModel | None = None
        self.sde = None

    @classmethod
    def from_task(cls, task, setting: str = "", device: str = "cpu", params=None):
        del params
        return cls(
            task.get_prior(),
            task.get_simulator(),
            theta_dim=getattr(task, "dim_parameters", None),
            x_dim=getattr(task, "dim_data", None),
            setting=setting or getattr(task, "name", ""),
            device=device,
            task=task,
        )

    def _ensure_dims(self, theta: Tensor, x: Tensor) -> None:
        if self.theta_dim is None:
            self.theta_dim = theta.shape[1] if theta.ndim > 1 else theta.numel()
        if self.x_dim is None:
            self.x_dim = x.shape[1] if x.ndim > 1 else x.numel()

    def _fit_prior_score(
        self,
        theta_raw: Tensor,
        theta_mean: Tensor,
        theta_std: Tensor,
        sde,
        params,
    ) -> nn.Module:
        budget = max(8, min(512, int(_get_param(params, "simulation_budget", 1000))))
        prior_theta = _sample_from_prior(self.prior, budget, self.device)
        prior_theta_z = (prior_theta - theta_mean) / theta_std
        prior_x = torch.zeros(prior_theta_z.size(0), 1, device=prior_theta_z.device)
        prior_model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=1, hidden=128, layers=3
        ).to(self.device)
        train_score_network(
            prior_model,
            (prior_theta_z, prior_x),
            params,
            sde=sde,
            loss_fn=denoising_posterior_score_matching_loss,
        )
        prior_model.eval()
        return prior_model

    def _evaluate_likelihood(
        self, loader, prior_model, sde, device
    ):
        total = 0.0
        count = 0
        with torch.no_grad():
            for theta_0, x in loader:
                theta_0 = theta_0.to(device)
                x = x.to(device)
                n = theta_0.size(0)
                t = torch.rand(n, 1, device=device) * (sde.t_max - sde.t_min) + sde.t_min
                theta_t = sde.sample_transition(theta_0, t)
                prior_x = torch.zeros(n, 1, device=device)
                prior_score = prior_model(theta_t, prior_x, t).detach()
                score = self.likelihood_model(theta_t, x, t)
                loss = denoising_likelihood_score_matching_loss(
                    score, theta_t, theta_0, prior_score, sde, t=t
                )
                total += float(loss.detach().item()) * n
                count += n
        return float("inf") if count == 0 else total / count

    def _train_likelihood_model(
        self,
        raw_model: nn.Module,
        theta_z: Tensor,
        x_z: Tensor,
        prior_model: nn.Module,
        sde,
        params,
    ):
        batch_size = max(1, int(_get_param(params, "batch_size", 50)))
        validation_fraction = min(
            max(float(_get_param(params, "validation_fraction", 0.15)), 0.0), 0.5
        )
        patience = int(_get_param(params, "early_stopping_patience", 1000))
        max_iters = int(_get_param(params, "training_steps", 3000))
        lr = float(_get_param(params, "learning_rate", 0.0001))
        seed = int(_get_param(params, "seed", 0))
        device = self.device

        dataset = TensorDataset(theta_z, x_z)
        n_total = len(dataset)
        if n_total >= 2:
            valid_size = max(1, int(n_total * validation_fraction))
            train_size = n_total - valid_size
            generator = torch.Generator().manual_seed(seed)
            train_ds, valid_ds = random_split(
                dataset, [train_size, valid_size], generator=generator
            )
        else:
            train_ds = dataset
            valid_ds = None

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        valid_loader = None
        if valid_ds is not None:
            valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
        eval_loader = valid_loader if valid_loader is not None else train_loader

        optimizer = torch.optim.Adam(raw_model.parameters(), lr=lr)
        prior_model.eval()
        self.likelihood_model = raw_model.to(device)
        best_loss = float("inf")
        best_state = copy.deepcopy(raw_model.state_dict())
        steps_no_improve = 0

        for _ in range(max_iters):
            raw_model.train()
            for theta_0, x in train_loader:
                theta_0 = theta_0.to(device)
                x = x.to(device)
                n = theta_0.size(0)
                t = torch.rand(n, 1, device=device) * (sde.t_max - sde.t_min) + sde.t_min
                theta_t = sde.sample_transition(theta_0, t)
                prior_x = torch.zeros(n, 1, device=device)
                with torch.no_grad():
                    prior_score = prior_model(theta_t, prior_x, t).detach()
                score = raw_model(theta_t, x, t)
                loss = denoising_likelihood_score_matching_loss(
                    score, theta_t, theta_0, prior_score, sde, t=t
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            valid_loss = self._evaluate_likelihood(
                eval_loader, prior_model, sde, device
            )
            if valid_loss < best_loss - 1e-12:
                best_loss = valid_loss
                best_state = copy.deepcopy(raw_model.state_dict())
                steps_no_improve = 0
            else:
                steps_no_improve += 1
            if steps_no_improve >= patience:
                break

        raw_model.load_state_dict(best_state)
        raw_model.eval()
        return raw_model

    def train(self, dataset, params=None):
        """Train the likelihood score network and prior score network."""
        theta, x = dataset
        theta = torch.as_tensor(theta, dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)
        self._ensure_dims(theta, x)

        sde = _build_sde_for_kind("ve", self.setting, theta, self.sde_config)
        theta_mean, theta_std, x_mean, x_std = _dataset_statistics(theta, x)
        theta_z, x_z = _standardize_dataset(
            theta, x, theta_mean, theta_std, x_mean, x_std
        )

        prior_model = self._fit_prior_score(
            theta, theta_mean, theta_std, sde, params
        )
        likelihood_model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=self.x_dim
        ).to(self.device)
        raw_model = self._train_likelihood_model(
            likelihood_model, theta_z, x_z, prior_model, sde, params
        )

        self.model = _AdditiveStandardizedScoreModel(
            raw_model, prior_model, theta_mean, theta_std, x_mean, x_std
        ).to(self.device)
        self.sde = sde
        return self

    def sample(self, observation, n_samples: int | None = None, params=None) -> Tensor:
        if self.model is None or self.sde is None:
            raise RuntimeError("train must be called before sample")
        if n_samples is None:
            n_samples = int(_get_param(params, "posterior_samples", 128))
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        sampler = ProbabilityFlowODESampler(self.model, self.sde, observation)
        return sampler.sample(int(n_samples))

    def posterior_samples(self, observation, n_samples=None, params=None) -> Tensor:
        return self.sample(observation, n_samples, params)

    def run(self, observation, params=None) -> Tensor:
        budget = max(1, int(_get_param(params, "simulation_budget", 1000)))
        theta = _sample_from_prior(self.prior, budget, self.device)
        x = _simulate(self.simulator, theta)
        self.train((theta, x), params)
        return self.sample(observation, None, params)
