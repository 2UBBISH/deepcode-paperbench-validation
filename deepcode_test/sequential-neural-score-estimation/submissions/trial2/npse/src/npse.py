"""Non-sequential Neural Posterior Score Estimation (NPSE) trainer.

This module implements the non-sequential NPSE training procedure from

    "Neural Posterior Score Estimation for Simulation-Based Inference"
    (Geffner et al.)

NPSE learns a score network :math:`s_\\psi(\\theta_t, x, t)` that approximates
the *posterior score* :math:`\\nabla_\\theta \\log p_t(\\theta_t \\mid x)` by
minimising a denoising posterior score-matching objective:

.. math::
    J_{NPSE}(\\psi) = \\frac{1}{2} \\int_0^T \\lambda(t)
    \\mathbb{E}_{p(\\theta_0) p(x|\\theta_0) p_{t|0}(\\theta_t|\\theta_0)}
    \\left[ \\| s_\\psi(\\theta_t, x, t)
    - \\nabla_\\theta \\log p_{t|0}(\\theta_t | \\theta_0) \\|^2 \\right] dt .

After training, approximate posterior samples are obtained by solving the
probability-flow ODE backwards from :math:`t=T` to :math:`t=0`.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .losses import npse_dsm_loss, sample_times
from .networks import ScoreNetwork, compute_standardization_stats
from .sampling import sample_probability_flow
from .sde import SDE, VESDE, VPSDE, estimate_sigma_max, get_sde

__all__ = [
    "NPSETrainer",
    "build_sde",
    "train_npse",
    "default_npse_config",
]


def _resolve_device(benchmark: Any) -> torch.device:
    """Return the device used by a benchmark, falling back to CPU/CUDA."""
    device = getattr(benchmark, "device", None)
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(benchmark: Any) -> torch.dtype:
    dtype = getattr(benchmark, "dtype", torch.float32)
    return dtype if dtype is not None else torch.float32


def default_npse_config(budget: int, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the default NPSE training configuration for a simulation budget.

    Batch sizes follow the paper's specification:
        * budgets 1_000 / 10_000  -> batch size 50
        * budget  100_000         -> batch size 500
    """
    if budget >= 100_000:
        batch_size = 500
    else:
        batch_size = 50

    cfg: Dict[str, Any] = {
        "lr": 1e-4,
        "validation_split": 0.15,
        "patience": 1000,          # steps without validation improvement
        "max_steps": 3000,
        "batch_size": batch_size,
        "eval_every": 25,
        "num_val_batches": 8,
        "weighting": "g2",         # lambda(t) = g(t)^2
        "min_delta": 0.0,
        "seed": None,
        "n_stats_samples": 4096,   # samples used for input standardisation
    }
    if overrides:
        cfg.update(overrides)
    return cfg


def build_sde(
    sde_type: str,
    theta_samples: torch.Tensor,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    T: float = 1.0,
    **kwargs: Any,
) -> SDE:
    """Instantiate an SDE, auto-calibrating ``sigma_max`` for the VE SDE.

    For the VE SDE, ``sigma_max`` is selected with Technique 1 of
    Song & Ermon (2020) (maximum pairwise Euclidean distance between prior
    samples) unless it is supplied explicitly.
    """
    sde_type = sde_type.lower()
    if sde_type in ("ve", "vesde"):
        if sigma_max is None:
            sigma_max = estimate_sigma_max(theta_samples)
        if sigma_min is None:
            sigma_min = kwargs.pop("default_sigma_min", 0.05)
        return VESDE(sigma_min=sigma_min, sigma_max=sigma_max, T=T)
    if sde_type in ("vp", "vpsde"):
        beta_min = kwargs.get("beta_min", 0.1)
        beta_max = kwargs.get("beta_max", 11.0)
        return VPSDE(beta_min=beta_min, beta_max=beta_max, T=T)
    return get_sde(sde_type, **kwargs)


class NPSETrainer:
    """Non-sequential NPSE trainer.

    Parameters
    ----------
    benchmark:
        A benchmark object exposing ``sample_joint``, ``theta_dim``, ``x_dim``,
        ``device`` and ``dtype`` (see ``npse.benchmarks.base.Benchmark``).
    sde:
        Forward SDE (e.g. ``VESDE`` or ``VPSDE``).
    score_net:
        Optional ``ScoreNetwork``. A new one is created if ``None``.
    config:
        Training configuration. Missing entries are filled from
        :func:`default_npse_config`.
    """

    def __init__(
        self,
        benchmark: Any,
        sde: SDE,
        score_net: Optional[ScoreNetwork] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.benchmark = benchmark
        self.sde = sde
        self.device = _resolve_device(benchmark)
        self.dtype = _resolve_dtype(benchmark)

        self.theta_dim = int(getattr(benchmark, "theta_dim", 0))
        self.x_dim = int(getattr(benchmark, "x_dim", 0))

        if score_net is None:
            score_net = ScoreNetwork(theta_dim=self.theta_dim, x_dim=self.x_dim)
        self.score_net = score_net
        self.score_net.to(device=self.device, dtype=self.dtype)
        if isinstance(self.sde, torch.nn.Module):
            self.sde.to(device=self.device, dtype=self.dtype)

        self.config = default_npse_config(
            int(config.pop("budget", 10_000)) if config else 10_000, config
        )
        if config:
            self.config.update(config)

        # Populated by :meth:`generate_dataset`.
        self.theta0: Optional[torch.Tensor] = None
        self.x: Optional[torch.Tensor] = None
        self.n_samples: int = 0

        # Training book-keeping.
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self.best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Dataset generation and standardisation
    # ------------------------------------------------------------------
    def generate_dataset(self, n_samples: int, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate ``n_samples`` joint prior/likelihood pairs from the benchmark."""
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            if hasattr(torch, "set_rng_state") and seed is not None:
                torch.manual_seed(seed)
        theta0, x = self.benchmark.sample_joint(int(n_samples))
        theta0 = torch.as_tensor(theta0, device=self.device, dtype=self.dtype)
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        self.theta0 = theta0
        self.x = x
        self.n_samples = int(n_samples)
        return theta0, x

    def fit_standardization(self, n_stats: Optional[int] = None) -> None:
        """Fit input standardisation statistics to the network.

        Both ``theta_t`` (perturbed parameters) and ``x`` are standardised
        per dimension before being fed into the score network.
        """
        if self.theta0 is None or self.x is None:
            raise RuntimeError("Call generate_dataset() before fit_standardization().")

        n_stats = n_stats or int(self.config.get("n_stats_samples", 4096))
        n_stats = min(n_stats, self.n_samples)
        if n_stats == 0:
            return

        idx = torch.randperm(self.n_samples, device=self.device)[:n_stats]
        theta0_sub = self.theta0[idx]
        x_sub = self.x[idx]

        # Perturb parameters at random times to get representative theta_t values.
        t = sample_times(
            n_stats, T=float(self.sde.T), device=self.device, dtype=self.dtype
        )
        theta_t = self.sde.transition_sample(theta0_sub, t)

        theta_mean, theta_std = compute_standardization_stats(theta_t)
        x_mean, x_std = compute_standardization_stats(x_sub)
        self.score_net.set_standardization(theta_mean, theta_std, x_mean, x_std)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> "NPSETrainer":
        """Train the posterior score network.

        Parameters
        ----------
        n_samples:
            Simulation budget. Defaults to the dataset size if a dataset was
            already generated, otherwise 10_000.
        seed:
            Random seed used for dataset generation and batch sampling.
        """
        if n_samples is not None:
            self.generate_dataset(n_samples, seed=seed)
        if self.theta0 is None or self.x is None:
            self.generate_dataset(int(self.config.get("budget", 10_000)), seed=seed)

        if seed is not None:
            torch.manual_seed(seed)

        self.fit_standardization()

        n = self.n_samples
        val_size = max(1, int(n * float(self.config["validation_split"])))
        train_size = n - val_size
        perm = torch.randperm(n, device=self.device)
        train_theta = self.theta0[perm[:train_size]]
        train_x = self.x[perm[:train_size]]
        val_theta = self.theta0[perm[train_size:]]
        val_x = self.x[perm[train_size:]]

        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=float(self.config["lr"]))
        weighting = self.config.get("weighting", "g2")
        batch_size = int(self.config["batch_size"])
        max_steps = int(self.config["max_steps"])
        patience = int(self.config["patience"])
        eval_every = int(self.config["eval_every"])
        num_val_batches = int(self.config["num_val_batches"])
        min_delta = float(self.config.get("min_delta", 0.0))
        T = float(self.sde.T)

        best_state = copy.deepcopy(self.score_net.state_dict())
        best_val_loss = float("inf")
        steps_since_best = 0

        self.history = {"train_loss": [], "val_loss": []}

        for step in range(max_steps):
            # Sample a minibatch (with replacement; dataset may be small).
            batch_idx = torch.randint(0, train_size, (batch_size,), device=self.device)
            theta_b = train_theta[batch_idx]
            x_b = train_x[batch_idx]
            t_b = sample_times(batch_size, T=T, device=self.device, dtype=self.dtype)

            optimizer.zero_grad(set_to_none=True)
            loss = npse_dsm_loss(self.sde, self.score_net, theta_b, x_b, t_b, weighting=weighting)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}: {loss.item()}")
            loss.backward()
            optimizer.step()

            self.history["train_loss"].append(float(loss.detach().cpu()))

            if (step + 1) % eval_every == 0:
                self.score_net.eval()
                val_loss = self._evaluate_validation(
                    val_theta, val_x, num_val_batches, batch_size, T, weighting
                )
                self.score_net.train()
                self.history["val_loss"].append(val_loss)

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.score_net.state_dict())
                    steps_since_best = 0
                else:
                    steps_since_best += eval_every

                if steps_since_best >= patience:
                    if verbose:
                        print(
                            f"[NPSE] early stopping at step {step + 1}, "
                            f"best val loss {best_val_loss:.6e}"
                        )
                    break

        self.score_net.load_state_dict(best_state)
        self.score_net.eval()
        self.best_val_loss = best_val_loss
        return self

    def _evaluate_validation(
        self,
        val_theta: torch.Tensor,
        val_x: torch.Tensor,
        num_batches: int,
        batch_size: int,
        T: float,
        weighting: Any,
    ) -> float:
        n_val = val_theta.shape[0]
        total = 0.0
        count = 0
        with torch.no_grad():
            for _ in range(num_batches):
                idx = torch.randint(0, n_val, (batch_size,), device=self.device)
                theta_b = val_theta[idx]
                x_b = val_x[idx]
                t_b = sample_times(batch_size, T=T, device=self.device, dtype=self.dtype)
                loss = npse_dsm_loss(
                    self.sde, self.score_net, theta_b, x_b, t_b, weighting=weighting
                )
                total += float(loss.detach().cpu())
                count += 1
        return total / max(1, count)

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------
    def _make_score_fn(self, x_obs: torch.Tensor) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Wrap the score network so a single observation is broadcast over the batch."""
        x_obs = torch.as_tensor(x_obs, device=self.device, dtype=self.dtype)
        net = self.score_net

        def score_fn(theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if theta.dim() == 1:
                theta = theta.unsqueeze(0)
            if x.dim() == 1:
                x = x.unsqueeze(0).expand(theta.shape[0], -1)
            elif x.dim() == 2 and x.shape[0] == 1 and theta.shape[0] != 1:
                x = x.expand(theta.shape[0], -1)
            return net(theta, x, t)

        return score_fn

    def sample_posterior(
        self,
        x_obs: torch.Tensor,
        n_samples: int,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "rk45",
    ) -> torch.Tensor:
        """Draw approximate posterior samples via the probability-flow ODE."""
        x_obs = torch.as_tensor(x_obs, device=self.device, dtype=self.dtype)
        score_fn = self._make_score_fn(x_obs)
        return sample_probability_flow(
            self.sde,
            score_fn,
            n_samples,
            x_obs,
            theta_dim=self.theta_dim,
            device=self.device,
            dtype=self.dtype,
            rtol=rtol,
            atol=atol,
            method=method,
        )


def train_npse(
    benchmark: Any,
    budget: int,
    sde_type: str = "ve",
    sde: Optional[SDE] = None,
    score_net: Optional[ScoreNetwork] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    verbose: bool = True,
) -> NPSETrainer:
    """Convenience entry point: build an SDE/score network, train, and return the trainer.

    The returned trainer exposes :meth:`NPSETrainer.sample_posterior` for
    generating approximate posterior samples at a test observation.
    """
    cfg = dict(config or {})
    cfg["budget"] = budget

    if sde is None:
        if seed is not None:
            torch.manual_seed(seed)
        prior_samples = benchmark.prior_sample(min(2000, max(256, budget)))
        prior_samples = torch.as_tensor(
            prior_samples,
            device=_resolve_device(benchmark),
            dtype=_resolve_dtype(benchmark),
        )
        sde = build_sde(
            sde_type,
            prior_samples,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            T=cfg.pop("T", 1.0),
        )

    trainer = NPSETrainer(benchmark, sde, score_net=score_net, config=cfg)
    trainer.train(n_samples=budget, seed=seed, verbose=verbose)
    return trainer
