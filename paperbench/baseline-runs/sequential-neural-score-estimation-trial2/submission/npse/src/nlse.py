"""Neural Likelihood Score Estimation (NLSE).

This module implements the NLSE baseline from the paper.  Instead of learning
the posterior score directly (as NPSE does), NLSE learns the *likelihood score*

    s_psi_lik(theta_t, x, t) ~= grad_theta log p_t(x | theta_t)

using the decomposition

    grad_theta log p_t(theta_t | x)
        = grad_theta log p_t(x | theta_t) + grad_theta log p_t(theta_t).

The perturbed prior score ``grad_theta log p_t(theta_t)`` is supplied by the
caller (either an analytic expression or a learned prior-score network, see
``npse.src.prior``).  During posterior sampling the two terms are added.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional

import torch

from .losses import nlse_dsm_loss, sample_times
from .networks import ScoreNetwork, compute_standardization_stats
from .sampling import sample_probability_flow
from .sde import SDE

__all__ = ["NLSETrainer", "train_nlse", "default_nlse_config"]


def default_nlse_config(budget: int, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Default NLSE training configuration.

    The hyper-parameters mirror those of NPSE: Adam with learning rate 1e-4,
    a 15% validation split, early stopping with patience 1000 steps, a maximum
    of 3000 training steps, and batch sizes 50/500 depending on the simulation
    budget.
    """
    batch_size = 50 if budget < 100_000 else 500
    config = {
        "lr": 1e-4,
        "batch_size": batch_size,
        "val_fraction": 0.15,
        "patience": 1000,
        "max_steps": 3000,
        "eval_every": 25,
        "weighting": "g2",
        "grad_clip_norm": 1.0,
        "num_workers": 0,
        "validation_samples": 500,
    }
    if overrides:
        config.update(overrides)
    return config


class NLSETrainer:
    """Trainer for the NLSE likelihood-score network.

    Parameters
    ----------
    benchmark:
        Benchmark object exposing ``sample_joint``, ``prior_sample``,
        ``theta_dim``, ``x_dim``, ``device`` and ``dtype``.
    sde:
        Diffusion SDE used to perturb parameters.
    prior_score_fn:
        Callable ``(theta_t, t) -> prior_score`` representing
        ``grad_theta log p_t(theta_t)``.
    likelihood_net:
        Optional ``ScoreNetwork``; created automatically if not supplied.
    config:
        Optional training configuration (see ``default_nlse_config``).
    """

    def __init__(
        self,
        benchmark: Any,
        sde: SDE,
        prior_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        likelihood_net: Optional[ScoreNetwork] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.benchmark = benchmark
        self.sde = sde
        self.prior_score_fn = prior_score_fn
        self.config = config or default_nlse_config(1000)
        self.device = getattr(benchmark, "device", torch.device("cpu"))
        self.dtype = getattr(benchmark, "dtype", torch.float32)

        theta_dim = getattr(benchmark, "theta_dim", None)
        x_dim = getattr(benchmark, "x_dim", None)
        if theta_dim is None or x_dim is None:
            theta_dim = theta_dim or 1
            x_dim = x_dim or 1

        self.likelihood_net = likelihood_net or ScoreNetwork(theta_dim, x_dim)
        self.likelihood_net = self.likelihood_net.to(device=self.device, dtype=self.dtype)

        self.theta_mean: Optional[torch.Tensor] = None
        self.theta_std: Optional[torch.Tensor] = None
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None

        self._theta0_train: Optional[torch.Tensor] = None
        self._x_train: Optional[torch.Tensor] = None
        self._theta0_val: Optional[torch.Tensor] = None
        self._x_val: Optional[torch.Tensor] = None

        self.best_state: Optional[Dict[str, Any]] = None
        self.history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Data generation and standardization
    # ------------------------------------------------------------------
    def generate_dataset(self, budget: int, seed: Optional[int] = None) -> None:
        """Sample parameter-observation pairs from the benchmark."""
        if seed is not None:
            torch.manual_seed(seed)
        theta, x = self.benchmark.sample_joint(budget)
        theta = torch.as_tensor(theta, device=self.device, dtype=self.dtype)
        x = torch.as_tensor(x, device=self.device, dtype=self.dtype)
        if theta.dim() == 1:
            theta = theta.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        val_fraction = float(self.config.get("val_fraction", 0.15))
        n_val = max(1, int(budget * val_fraction))
        n_train = budget - n_val

        idx = torch.randperm(budget, device=self.device)
        self._theta0_train = theta[idx[:n_train]]
        self._x_train = x[idx[:n_train]]
        self._theta0_val = theta[idx[n_train:]]
        self._x_val = x[idx[n_train:]]

    def fit_standardization(self, n_samples: int = 2000) -> None:
        """Fit per-dimension standardization for theta_t and x.

        Perturbed parameters are sampled at random diffusion times from the
        training (or prior) parameters.
        """
        if self._theta0_train is not None and len(self._theta0_train) > 0:
            idx = torch.randperm(len(self._theta0_train), device=self.device)[:n_samples]
            theta0 = self._theta0_train[idx]
            x = self._x_train[idx] if self._x_train is not None else torch.zeros(
                (len(theta0), self.likelihood_net.x_dim), device=self.device, dtype=self.dtype
            )
        else:
            theta0 = self.benchmark.prior_sample(n_samples)
            theta0 = torch.as_tensor(theta0, device=self.device, dtype=self.dtype)
            if theta0.dim() == 1:
                theta0 = theta0.unsqueeze(-1)
            x = self.benchmark.simulator(theta0)
            x = torch.as_tensor(x, device=self.device, dtype=self.dtype)

        t = sample_times(len(theta0), T=float(self.sde.T), device=self.device, dtype=self.dtype)
        theta_t = self.sde.transition_sample(theta0, t)

        self.theta_mean, self.theta_std = compute_standardization_stats(theta_t)
        self.x_mean, self.x_std = compute_standardization_stats(x)
        self.likelihood_net.set_standardization(
            theta_mean=self.theta_mean,
            theta_std=self.theta_std,
            x_mean=self.x_mean,
            x_std=self.x_std,
        )

    # ------------------------------------------------------------------
    # Loss and score construction
    # ------------------------------------------------------------------
    def likelihood_loss(
        self, theta0: torch.Tensor, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return nlse_dsm_loss(
            self.sde,
            self.likelihood_net,
            self.prior_score_fn,
            theta0,
            x,
            t,
            weighting=self.config.get("weighting", "g2"),
        )

    def posterior_score_fn(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the combined posterior score ``s_lik + grad log p_t(theta_t)``."""
        net = self.likelihood_net

        def score(theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            prior = self.prior_score_fn(theta, t)
            # Broadcast prior score to the batch dimension of theta.
            return net(theta, x, t) + prior

        return score

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, seed: Optional[int] = None, verbose: bool = True) -> None:
        """Train the likelihood-score network with denoising score matching."""
        if self._theta0_train is None:
            raise RuntimeError("generate_dataset() must be called before train().")

        if seed is not None:
            torch.manual_seed(seed)

        optimizer = torch.optim.Adam(self.likelihood_net.parameters(), lr=float(self.config["lr"]))
        batch_size = int(self.config["batch_size"])
        max_steps = int(self.config["max_steps"])
        patience = int(self.config["patience"])
        eval_every = int(self.config["eval_every"])
        grad_clip = float(self.config.get("grad_clip_norm", 1.0))

        best_val = float("inf")
        best_state: Dict[str, Any] = copy.deepcopy(self.likelihood_net.state_dict())
        steps_without_improvement = 0

        n_train = len(self._theta0_train)
        n_val = len(self._theta0_val) if self._theta0_val is not None else 0

        for step in range(1, max_steps + 1):
            self.likelihood_net.train()
            idx = torch.randint(0, n_train, (batch_size,), device=self.device)
            theta0 = self._theta0_train[idx]
            x = self._x_train[idx]
            t = sample_times(batch_size, T=float(self.sde.T), device=self.device, dtype=self.dtype)
            theta_t = self.sde.transition_sample(theta0, t)

            optimizer.zero_grad(set_to_none=True)
            loss = self.likelihood_loss(theta0, x, t)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.likelihood_net.parameters(), grad_clip)
            optimizer.step()

            self.history["train_loss"].append(float(loss.detach().cpu()))

            if step % eval_every == 0 or step == max_steps:
                val_loss = self._evaluate_validation(batch_size)
                self.history["val_loss"].append(val_loss)
                if verbose:
                    print(f"[NLSE] step {step}/{max_steps} train={loss.item():.6f} val={val_loss:.6f}")

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(self.likelihood_net.state_dict())
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if steps_without_improvement >= patience:
                    if verbose:
                        print(f"[NLSE] early stopping at step {step} (patience {patience}).")
                    break

        self.likelihood_net.load_state_dict(best_state)
        self.best_state = best_state

    def _evaluate_validation(self, batch_size: int) -> float:
        if self._theta0_val is None or len(self._theta0_val) == 0:
            return float("inf")
        self.likelihood_net.eval()
        with torch.no_grad():
            total = 0.0
            count = 0
            n_val = len(self._theta0_val)
            n_batches = max(1, min(int(self.config.get("validation_samples", 500)) // batch_size + 1, n_val // batch_size + 1))
            for _ in range(n_batches):
                idx = torch.randint(0, n_val, (batch_size,), device=self.device)
                theta0 = self._theta0_val[idx]
                x = self._x_val[idx]
                t = sample_times(batch_size, T=float(self.sde.T), device=self.device, dtype=self.dtype)
                loss = self.likelihood_loss(theta0, x, t)
                total += float(loss.detach().cpu()) * batch_size
                count += batch_size
        return total / max(count, 1)

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------
    def sample_posterior(
        self,
        n_samples: int,
        x_obs: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        method: str = "rk45",
    ) -> torch.Tensor:
        """Sample approximate posteriors using the probability-flow ODE."""
        x_obs = torch.as_tensor(x_obs, device=self.device, dtype=self.dtype)
        return sample_probability_flow(
            self.sde,
            self.posterior_score_fn(),
            n_samples=n_samples,
            x_obs=x_obs,
            theta_dim=getattr(self.benchmark, "theta_dim", None),
            device=self.device,
            dtype=self.dtype,
            rtol=rtol,
            atol=atol,
            method=method,
        )


def train_nlse(
    benchmark: Any,
    prior_score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    budget: int,
    sde: SDE,
    likelihood_net: Optional[ScoreNetwork] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> NLSETrainer:
    """Convenience entry point for non-sequential NLSE training.

    Parameters
    ----------
    benchmark:
        Benchmark simulator.
    prior_score_fn:
        Perturbed prior-score function ``(theta_t, t) -> score``.
    budget:
        Number of joint simulations.
    sde:
        Diffusion SDE (must already be calibrated).
    likelihood_net:
        Optional likelihood-score network.
    config:
        Optional training configuration.
    seed:
        Optional random seed for data generation.
    verbose:
        Whether to print training progress.
    """
    cfg = config or default_nlse_config(budget)
    trainer = NLSETrainer(benchmark, sde, prior_score_fn, likelihood_net=likelihood_net, config=cfg)
    trainer.generate_dataset(budget, seed=seed)
    trainer.fit_standardization()
    trainer.train(seed=seed, verbose=verbose)
    return trainer
