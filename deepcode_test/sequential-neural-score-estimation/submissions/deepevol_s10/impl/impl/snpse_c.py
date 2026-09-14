'''SNPSE-C score-space correction variant.

SNPSE-C trains a proposal-posterior score network under the current proposal
prior and then recovers the original posterior score by adding the learned
proposal-prior score and subtracting the original prior score. The proposal-prior
score is estimated with denoising score matching on samples from the current
mixture of truncated priors.
'''

from __future__ import annotations

import torch
from torch import Tensor, nn

from impl.losses import denoising_posterior_score_matching_loss
from impl.npse import (
    _StandardizedScoreModel,
    _dataset_statistics,
    _get_param,
    _sample_from_prior,
    _standardize_dataset,
)
from impl.score_network import ScoreNetworkMLP
from impl.training import train_score_network
from impl.tsnpse import Tsnpse


class _CorrectedStandardizedScoreModel(nn.Module):
    '''Combine proposal-posterior, proposal-prior, and original prior scores
    into a raw-space posterior score.
    '''

    def __init__(
        self,
        proposal_posterior_model: nn.Module,
        proposal_prior_model: nn.Module,
        prior_model: nn.Module,
        theta_mean: Tensor,
        theta_std: Tensor,
        x_mean: Tensor,
        x_std: Tensor,
    ):
        super().__init__()
        self.proposal_posterior_model = proposal_posterior_model
        self.proposal_prior_model = proposal_prior_model
        self.prior_model = prior_model
        self.theta_dim = int(proposal_posterior_model.theta_dim)
        self.x_dim = int(proposal_posterior_model.x_dim)
        self.register_buffer('theta_mean', theta_mean.detach())
        self.register_buffer('theta_std', theta_std.detach())
        self.register_buffer('x_mean', x_mean.detach())
        self.register_buffer('x_std', x_std.detach())

    def _dummy_x(self, theta_z: Tensor) -> Tensor:
        if theta_z.ndim == 2:
            return torch.zeros(
                theta_z.size(0), 1, dtype=theta_z.dtype, device=theta_z.device
            )
        return torch.zeros(1, dtype=theta_z.dtype, device=theta_z.device)

    def forward(self, theta_t: Tensor, x: Tensor, t) -> Tensor:
        theta_z = (theta_t - self.theta_mean) / self.theta_std
        x_z = (x - self.x_mean) / self.x_std

        proposal_posterior_score_z = self.proposal_posterior_model(theta_z, x_z, t)
        proposal_prior_score_z = self.proposal_prior_model(
            theta_z, self._dummy_x(theta_z), t
        )
        prior_score_z = self.prior_model(theta_z, self._dummy_x(theta_z), t)

        return (
            proposal_posterior_score_z - proposal_prior_score_z + prior_score_z
        ) / self.theta_std


class SNPSEC(Tsnpse):
    '''Sequential NPSE variant with score-space correction instead of
    importance weights.
    '''

    sde_kind = None

    def _fit_theta_score(
        self,
        theta: Tensor,
        theta_mean: Tensor,
        theta_std: Tensor,
        sde,
        params,
        device: str,
    ) -> nn.Module:
        theta = torch.as_tensor(theta, dtype=torch.float32, device=device)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        if theta.shape[0] < 2:
            theta = theta.repeat(max(1, 2 // theta.shape[0]), 1)

        theta_z = (theta - theta_mean) / theta_std
        dummy_x = torch.zeros(theta_z.size(0), 1, device=device)
        model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=1, hidden=128, layers=3
        ).to(device)
        train_score_network(
            model,
            (theta_z, dummy_x),
            params,
            sde=sde,
            loss_fn=denoising_posterior_score_matching_loss,
        )
        model.eval()
        return model

    def _train_on_data(self, theta: Tensor, x: Tensor, params):
        theta = torch.as_tensor(theta, dtype=torch.float32)
        x = torch.as_tensor(x, dtype=torch.float32)
        self._ensure_dims(theta, x)
        sde = self._ensure_sde(theta)

        theta_mean, theta_std, x_mean, x_std = _dataset_statistics(theta, x)
        theta_z, x_z = _standardize_dataset(
            theta, x, theta_mean, theta_std, x_mean, x_std
        )

        prior_budget = max(
            8, min(256, max(1, int(_get_param(params, 'simulation_budget', 1000))))
        )
        prior_theta = _sample_from_prior(self.prior, prior_budget, self.device)
        prior_model = self._fit_theta_score(
            prior_theta, theta_mean, theta_std, sde, params, self.device
        )

        proposal_budget = max(8, min(256, prior_budget // 2))
        proposal_theta = self._sample_proposal(proposal_budget, params)
        proposal_prior_model = self._fit_theta_score(
            proposal_theta, theta_mean, theta_std, sde, params, self.device
        )

        proposal_posterior_model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=self.x_dim
        ).to(self.device)
        train_score_network(
            proposal_posterior_model,
            (theta_z, x_z),
            params,
            sde=sde,
            loss_fn=denoising_posterior_score_matching_loss,
        )
        proposal_posterior_model.eval()

        self.model = _CorrectedStandardizedScoreModel(
            proposal_posterior_model,
            proposal_prior_model,
            prior_model,
            theta_mean,
            theta_std,
            x_mean,
            x_std,
        ).to(self.device)
        self.sde = sde
        return self.model
