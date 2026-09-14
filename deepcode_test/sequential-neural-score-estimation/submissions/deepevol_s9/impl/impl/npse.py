"""Non-sequential Neural Posterior Score Estimation methods.

This module implements NPSE-VE and NPSE-VP. Each method draws
prior-simulator pairs, diffuses the parameters with the selected forward SDE,
trains the conditional score network with the denoising posterior score
matching objective, and samples from the approximate posterior by solving the
time-reversed probability flow ODE.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from impl.config import SDEConfig, TrialConfig, ve_sigma_min_for_setting
from impl.losses import denoising_posterior_score_matching_loss
from impl.sampler import ProbabilityFlowODESampler
from impl.score_network import ScoreNetworkMLP
from impl.sde import build_sde, choose_sigma_max
from impl.training import train_score_network


def _get_param(params, key, default):
    """Return *key* from a TrialConfig, a dict, or fall back to *default*."""
    if isinstance(params, TrialConfig):
        return getattr(params, key, default)
    if isinstance(params, dict):
        return params.get(key, default)
    return default


def _sample_from_prior(prior, n: int, device: str = "cpu") -> Tensor:
    """Sample *n* parameters from a callable or torch.distributions prior."""
    n = int(n)
    if n <= 0:
        return torch.empty((0, 0), dtype=torch.float32, device=device)

    samples = None
    if callable(prior):
        try:
            samples = prior(n)
        except TypeError:
            try:
                samples = prior(num_samples=n)
            except TypeError as exc:
                raise TypeError(
                    "prior callable must accept n or num_samples=n"
                ) from exc
    elif hasattr(prior, "sample"):
        try:
            samples = prior.sample((n,))
        except TypeError:
            try:
                samples = prior.sample(n)
            except TypeError as exc:
                raise TypeError(
                    "prior.sample must accept sample_shape or n"
                ) from exc
    else:
        raise TypeError("prior must be callable or expose a sample method")

    samples = torch.as_tensor(samples, dtype=torch.float32, device=device)
    if samples.ndim == 0:
        samples = samples.reshape(1, 1)
    elif samples.ndim == 1:
        samples = samples.reshape(n, -1)
    return samples


def _simulate(simulator, theta: Tensor) -> Tensor:
    """Call a simulator with a batch of parameters and coerce its output."""
    theta = torch.as_tensor(theta, dtype=torch.float32)
    if theta.ndim == 1:
        theta = theta.unsqueeze(0)

    out = simulator(theta)
    if isinstance(out, (tuple, list)):
        out = out[0]
    out = torch.as_tensor(out, dtype=torch.float32)

    if out.ndim == 0:
        out = out.reshape(theta.shape[0], 1)
    elif out.ndim == 1:
        if theta.shape[0] == 1:
            out = out.unsqueeze(0)
        else:
            out = out.unsqueeze(-1)
    elif out.ndim == 2 and out.shape[0] != theta.shape[0] and out.shape[1] == theta.shape[0]:
        out = out.transpose(0, 1)
    return out


def _as_pair(dataset):
    """Return ``(theta, x)`` tensors from a tuple, list of tensors, or pair list."""
    if isinstance(dataset, (tuple, list)) and len(dataset) == 2:
        theta, x = dataset
        if torch.is_tensor(theta) or torch.is_tensor(x):
            return torch.as_tensor(theta, dtype=torch.float32), torch.as_tensor(
                x, dtype=torch.float32
            )

    thetas = []
    xs = []
    for a, b in dataset:
        thetas.append(torch.as_tensor(a, dtype=torch.float32))
        xs.append(torch.as_tensor(b, dtype=torch.float32))
    if not thetas:
        raise ValueError("cannot process an empty dataset")
    return torch.stack(thetas), torch.stack(xs)


def _dataset_statistics(theta: Tensor, x: Tensor):
    """Return per-dimension means and standard deviations for scoring inputs."""
    theta = torch.as_tensor(theta, dtype=torch.float32)
    x = torch.as_tensor(x, dtype=torch.float32)
    theta_mean = theta.mean(dim=0)
    theta_std = theta.std(dim=0).clamp_min(1e-6)
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0).clamp_min(1e-6)
    return theta_mean, theta_std, x_mean, x_std


def _standardize_dataset(theta, x, theta_mean, theta_std, x_mean, x_std):
    theta_z = (theta - theta_mean) / theta_std
    x_z = (x - x_mean) / x_std
    return theta_z, x_z


def _robust_sigma_max(theta: Tensor) -> float:
    """Return a robust VE sigma_max from the supplied parameter batch."""
    theta = torch.as_tensor(theta, dtype=torch.float32)
    if theta.numel() == 0:
        return 1.0
    sample = theta.reshape(theta.shape[0], -1)
    if sample.shape[0] < 2:
        sigma = float(sample.std())
    else:
        sigma = float(choose_sigma_max(sample[:1024]))
    if not math.isfinite(sigma) or sigma <= 1e-6:
        sigma = 1.0
    return sigma


def _build_sde_for_kind(
    kind: str | None,
    setting: str,
    theta: Tensor,
    sde_config: SDEConfig | None = None,
):
    """Build the selected VE or VP SDE, defaulting VA/VP by dimension when
    *kind* is unspecified."""
    if sde_config is not None:
        return build_sde(sde_config)

    if kind == "ve":
        sigma_min = ve_sigma_min_for_setting(setting)
        sigma_max = _robust_sigma_max(theta)
        if sigma_max <= sigma_min:
            sigma_max = max(2.0 * sigma_min, 1.0)
        return build_sde(SDEConfig.ve(sigma_min=sigma_min, sigma_max=sigma_max))

    if kind == "vp":
        return build_sde(SDEConfig.vp())

    default_kind = "ve" if theta.shape[1] <= 2 else "vp"
    return _build_sde_for_kind(default_kind, setting, theta, sde_config=None)


class _StandardizedScoreModel(nn.Module):
    """Wrap a score model trained on standardised inputs so that sampling can
    consume raw parameter and observation values."""

    def __init__(
        self,
        raw_model: nn.Module,
        theta_mean: Tensor,
        theta_std: Tensor,
        x_mean: Tensor,
        x_std: Tensor,
    ):
        super().__init__()
        self.raw_model = raw_model
        self.register_buffer("theta_mean", theta_mean.detach())
        self.register_buffer("theta_std", theta_std.detach())
        self.register_buffer("x_mean", x_mean.detach())
        self.register_buffer("x_std", x_std.detach())
        self.theta_dim = int(raw_model.theta_dim)
        self.x_dim = int(raw_model.x_dim)

    def forward(self, theta_t: Tensor, x: Tensor, t) -> Tensor:
        theta_z = (theta_t - self.theta_mean) / self.theta_std
        x_z = (x - self.x_mean) / self.x_std
        score_z = self.raw_model(theta_z, x_z, t)
        return score_z / self.theta_std


class NpseBase:
    """Shared NPSE implementation used by both VE and VP variants."""

    sde_kind: str | None = None

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
        self.model: _StandardizedScoreModel | None = None
        self.sde = None

    @classmethod
    def from_task(cls, task, setting: str = "", device: str = "cpu", params=None):
        """Construct a method from a sbibm-style task object."""
        del params  # accepted for a uniform API in dispatchers
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
            if theta.ndim == 1:
                self.theta_dim = theta.numel()
            else:
                self.theta_dim = theta.shape[1]
        if self.x_dim is None:
            if x.ndim == 1:
                self.x_dim = x.numel()
            else:
                self.x_dim = x.shape[1]

    def _simulate_dataset(self, n: int, params=None):
        n = max(1, int(n))
        theta = _sample_from_prior(self.prior, n, self.device)
        x = _simulate(self.simulator, theta)
        return theta, x

    def train(self, dataset, params=None):
        """Train the conditional score network on a dataset of (theta, x) pairs."""
        theta, x = _as_pair(dataset)
        self._ensure_dims(theta, x)

        sde = _build_sde_for_kind(
            self.sde_kind, self.setting, theta, self.sde_config
        )
        theta_mean, theta_std, x_mean, x_std = _dataset_statistics(theta, x)
        theta_z, x_z = _standardize_dataset(
            theta, x, theta_mean, theta_std, x_mean, x_std
        )

        raw_model = ScoreNetworkMLP(
            theta_dim=self.theta_dim, x_dim=self.x_dim
        ).to(self.device)
        train_score_network(
            raw_model,
            (theta_z, x_z),
            params,
            sde=sde,
            loss_fn=denoising_posterior_score_matching_loss,
        )
        raw_model.eval()

        self.model = _StandardizedScoreModel(
            raw_model, theta_mean, theta_std, x_mean, x_std
        ).to(self.device)
        self.sde = sde
        return self

    def sample(self, observation, n_samples: int | None = None, params=None) -> Tensor:
        """Return approximate posterior samples for *observation*."""
        if self.model is None or self.sde is None:
            raise RuntimeError("train must be called before sample")
        if n_samples is None:
            n_samples = int(_get_param(params, "posterior_samples", 128))
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        sampler = ProbabilityFlowODESampler(self.model, self.sde, observation)
        return sampler.sample(int(n_samples))

    def posterior_samples(
        self, observation, n_samples: int | None = None, params=None
    ) -> Tensor:
        """Alias for :meth:`sample` used by older callers."""
        return self.sample(observation, n_samples, params)

    def run(self, observation, params=None) -> Tensor:
        """Run a complete NPSE trial: simulate, train, and sample."""
        budget = int(_get_param(params, "simulation_budget", 1000))
        theta, x = self._simulate_dataset(budget, params)
        self.train((theta, x), params)
        return self.sample(observation, None, params)


class NpseVE(NpseBase):
    """NPSE with the variance-exploding forward SDE."""

    sde_kind = "ve"


class NpseVP(NpseBase):
    """NPSE with the variance-preserving forward SDE."""

    sde_kind = "vp"
