"""Sequential Neural Posterior Score Estimation variants.

This module implements the three sequential NPSE variants described in the
paper:

* **SNPSE-A** learns a *proposal*-posterior score
  :math:`\\tilde s^r_\\psi(\\theta_t, x, t)` from proposal-prior samples.  After
  probability-flow ODE sampling from the proposal posterior it applies
  sampling-importance-resampling (SIR) with importance weights
  :math:`h_i = p(\\theta_i)/\\tilde p^r(\\theta_i)` to recover approximate
  samples from the target posterior.
* **SNPSE-B** directly learns the target posterior score by reweighting each
  proposal-prior sample in the denoising-score-matching loss by
  :math:`w_i = p(\\theta_i)/\\tilde p^r(\\theta_i)`.
* **SNPSE-C** learns a proposal-posterior score using the decomposed
  parameterisation
  :math:`\\tilde s^r_\\psi = s_\\psi + \\nabla_\\theta \\log \\tilde p_t^r -
  \\nabla_\\theta \\log p_t`, where :math:`\\tilde p_t^r` is the perturbed
  proposal prior and :math:`p_t` is the perturbed target prior.

For the truncated-mixture proposal construction
:math:`\\tilde p^r(\\theta) \\propto c^r(\\theta) p(\\theta)` with
:math:`c^r(\\theta) = \\frac{1}{r}\\sum_{s=0}^{r-1}
\\mathbb{1}\\{\\theta\\in\\Theta^s\\}`, the importance weight
:math:`p(\\theta)/\\tilde p^r(\\theta)` is proportional to
:math:`1/c^r(\\theta)`.  This module therefore computes weights directly from
the running HPR regions, avoiding an explicit proposal-density normaliser.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from .hpr import (
    HPR_EPSILON_DEFAULT,
    HPR_N_SAMPLES_DEFAULT,
    HPRTruncation,
    PriorSupportRegion,
    TruncatedProposalSampler,
)
from .losses import npse_dsm_loss, sample_times, weighted_npse_dsm_loss
from .networks import ScoreNetwork, compute_standardization_stats
from .npse import build_sde, default_npse_config
from .prior import train_prior_score_network
from .sampling import sample_probability_flow
from .sde import SDE

__all__ = [
    "SNPSETrainer",
    "default_snpse_config",
    "train_snpse",
    "train_snpse_a",
    "train_snpse_b",
    "train_snpse_c",
]


def default_snpse_config(
    total_budget: int,
    rounds: int = 10,
    variant: str = "b",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return default configuration for the SNPSE sequential trainers.

    The defaults mirror the NPSE/TSNPSE training settings: Adam with learning
    rate ``1e-4``, 15% validation split, early stopping with patience ``1000``
    steps, and a maximum of ``3000`` training steps.
    """

    cfg = default_npse_config(total_budget)
    cfg.update(
        {
            "rounds": int(rounds),
            "total_budget": int(total_budget),
            "budget_per_round": max(1, int(total_budget) // int(rounds)),
            "variant": variant,
            "hpr_epsilon": HPR_EPSILON_DEFAULT,
            "hpr_n_samples": HPR_N_SAMPLES_DEFAULT,
            "rtol": 1e-5,
            "atol": 1e-5,
            "hutchinson_samples": 1,
            "proposal_batch_size": 512,
            "max_attempts_factor": 10000,
            "sir_oversampling": 4,
            "grad_clip": 5.0,
            "prior_score_samples": 5000,
            "prior_score_steps": 2000,
            "prior_score_batch_size": 256,
            "prior_score_patience": 200,
        }
    )
    if overrides:
        cfg.update(overrides)
    return cfg


class SNPSETrainer:
    """Sequential NPSE trainer with selectable correction strategy.

    Parameters
    ----------
    benchmark:
        Benchmark object exposing ``prior_sample``, ``simulator``,
        ``theta_dim``, ``x_dim``, ``device`` and ``dtype``.
    sde:
        Diffusion SDE used to perturb parameters.
    x_obs:
        Observed dataset conditioning the posterior.
    total_budget:
        Total number of simulated parameter/observation pairs.
    rounds:
        Number of sequential rounds.
    variant:
        One of ``"a"`` (SIR correction), ``"b"`` (loss reweighting) or
        ``"c"`` (decomposed proposal-posterior score).
    score_net:
        Optional initial score network (a fresh network is built each round).
    config:
        Optional configuration dictionary overriding defaults.
    seed:
        Optional random seed.
    prior_score_fn:
        Optional perturbed prior score ``(theta, t) -> score`` used by
        SNPSE-C.  If ``None`` an implicit prior score network is trained.
    """

    def __init__(
        self,
        benchmark: Any,
        sde: SDE,
        x_obs: Any,
        total_budget: int,
        rounds: int = 10,
        variant: str = "b",
        score_net: Optional[ScoreNetwork] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        prior_score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if variant not in ("a", "b", "c"):
            raise ValueError(f"Unknown SNPSE variant '{variant}'; expected 'a', 'b' or 'c'.")

        self.benchmark = benchmark
        self.x_obs = x_obs
        self.total_budget = int(total_budget)
        self.rounds = int(rounds)
        self.variant = variant
        self.seed = seed
        self.prior_score_fn = prior_score_fn
        self.config = (
            config
            if config is not None
            else default_snpse_config(total_budget, rounds=rounds, variant=variant)
        )

        self.theta_dim = int(getattr(benchmark, "theta_dim", 1))
        self.x_dim = int(getattr(benchmark, "x_dim", 1))
        self.device = getattr(benchmark, "device", None) or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = getattr(benchmark, "dtype", torch.float32)

        self.sde = sde.to(self.device)
        self.T = float(getattr(self.sde, "T", 1.0))

        if not isinstance(self.x_obs, torch.Tensor):
            self.x_obs = torch.as_tensor(self.x_obs, dtype=self.dtype, device=self.device)
        else:
            self.x_obs = self.x_obs.to(dtype=self.dtype, device=self.device)

        self.use_weights = variant in ("b", "c")

        # Round-zero region is the prior support.  Subsequent HPR regions are
        # appended after each training round.
        self.regions: List[Any] = [PriorSupportRegion()]
        self.proposal_sampler = TruncatedProposalSampler()
        self.proposal_sampler.add_round(self.regions[0])

        self.all_theta: List[torch.Tensor] = []
        self.all_x: List[torch.Tensor] = []
        self.all_weights: List[torch.Tensor] = []

        self.score_net: Optional[ScoreNetwork] = None
        self.proposal_prior_net = None
        self.prior_score_net = None
        self._initial_score_net = score_net

    # ------------------------------------------------------------------
    # Proposal and weighting helpers
    # ------------------------------------------------------------------
    def _coverage(self, theta: torch.Tensor) -> torch.Tensor:
        """Return the running coverage :math:`c^r(\\theta)` for each sample."""

        cov = torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype)
        for region in self.regions:
            cov = cov + region.contains(theta).to(theta.dtype)
        n_regions = max(1, len(self.regions))
        cov = cov / n_regions
        return cov.clamp_min(1.0 / n_regions)

    def _importance_weights(self, theta: torch.Tensor) -> torch.Tensor:
        """Return unnormalised importance weights ``p(theta)/p_tilde(theta)``.

        Under the truncated-mixture proposal these are proportional to
        ``1 / coverage(theta)``; the unknown normaliser cancels in SIR and is a
        constant rescaling for the SNPSE-B loss.
        """

        return 1.0 / self._coverage(theta)

    def _sir_resample(
        self, samples: torch.Tensor, weights: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        """Sampling-importance-resampling (SIR)."""

        if weights.numel() == 0 or weights.sum().item() <= 0.0:
            return samples[:n_samples]
        probs = weights / weights.sum()
        idx = torch.multinomial(probs, n_samples, replacement=True)
        return samples[idx]

    # ------------------------------------------------------------------
    # Network construction / training
    # ------------------------------------------------------------------
    def _create_score_net(self) -> ScoreNetwork:
        net = ScoreNetwork(
            self.theta_dim,
            self.x_dim,
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            time_embedding_dim=int(self.config.get("time_embedding_dim", 64)),
        )
        return net.to(device=self.device, dtype=self.dtype)

    def _fit_standardization(
        self, net: ScoreNetwork, theta: torch.Tensor, x: torch.Tensor, n: int = 2000
    ) -> None:
        n = min(int(n), theta.shape[0])
        if n <= 0:
            return
        idx = torch.randperm(theta.shape[0], device=theta.device)[:n]
        theta_sub = theta[idx]
        t = sample_times(n, T=self.T, device=theta.device, dtype=theta.dtype)
        theta_t = self.sde.transition_sample(theta_sub, t)
        theta_mean, theta_std = compute_standardization_stats(theta_t)
        x_mean, x_std = compute_standardization_stats(x[idx])
        net.set_standardization(theta_mean, theta_std, x_mean, x_std)

    def _compute_loss(
        self,
        net: ScoreNetwork,
        theta: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weighting = self.config.get("weighting", "g2")
        if weights is not None:
            return weighted_npse_dsm_loss(
                self.sde, net, theta, x, t, weights, weighting=weighting
            )
        return npse_dsm_loss(self.sde, net, theta, x, t, weighting=weighting)

    def _validation_loss(
        self,
        net: ScoreNetwork,
        theta_val: torch.Tensor,
        x_val: torch.Tensor,
        w_val: Optional[torch.Tensor],
        n_batches: int = 4,
    ) -> float:
        if theta_val.shape[0] == 0:
            return float("inf")
        net.eval()
        total = 0.0
        count = 0
        batch_size = min(int(self.config.get("batch_size", 50)), theta_val.shape[0])
        with torch.no_grad():
            for _ in range(n_batches):
                idx = torch.randint(
                    0, theta_val.shape[0], (batch_size,), device=theta_val.device
                )
                t = sample_times(
                    batch_size, T=self.T, device=theta_val.device, dtype=theta_val.dtype
                )
                w = w_val[idx] if w_val is not None else None
                loss = self._compute_loss(net, theta_val[idx], x_val[idx], t, w)
                total += float(loss.item())
                count += 1
        return total / max(1, count)

    def _train(
        self,
        net: ScoreNetwork,
        theta: torch.Tensor,
        x: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> ScoreNetwork:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        cfg = self.config
        self._fit_standardization(net, theta, x, n=int(cfg.get("std_samples", 2000)))

        n = theta.shape[0]
        n_val = max(1, int(n * float(cfg.get("validation_frac", 0.15))))
        perm = torch.randperm(n, device=theta.device)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        theta_train, x_train = theta[train_idx], x[train_idx]
        theta_val, x_val = theta[val_idx], x[val_idx]
        w_train = weights[train_idx] if weights is not None else None
        w_val = weights[val_idx] if weights is not None else None

        optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.get("lr", 1e-4)))
        best_loss = float("inf")
        best_state = copy.deepcopy(net.state_dict())
        patience = int(cfg.get("patience", 1000))
        max_steps = int(cfg.get("max_steps", 3000))
        eval_every = int(cfg.get("eval_every", 25))
        grad_clip = float(cfg.get("grad_clip", 5.0))
        batch_size = min(int(cfg.get("batch_size", 50)), max(1, theta_train.shape[0]))
        steps_since_improvement = 0

        for step in range(max_steps):
            net.train()
            idx = torch.randint(
                0, theta_train.shape[0], (batch_size,), device=theta_train.device
            )
            t = sample_times(
                batch_size, T=self.T, device=theta_train.device, dtype=theta_train.dtype
            )
            w = w_train[idx] if w_train is not None else None
            loss = self._compute_loss(net, theta_train[idx], x_train[idx], t, w)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            if step % eval_every == 0:
                val_loss = self._validation_loss(net, theta_val, x_val, w_val)
                if val_loss < best_loss - 1e-8:
                    best_loss = val_loss
                    best_state = copy.deepcopy(net.state_dict())
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += eval_every
                if steps_since_improvement >= patience:
                    break

        net.load_state_dict(best_state)
        return net

    # ------------------------------------------------------------------
    # HPR regions and sequential rounds
    # ------------------------------------------------------------------
    def _build_region(self, score_fn: Callable) -> Any:
        hpr = HPRTruncation(
            self.sde,
            score_fn,
            epsilon=float(self.config.get("hpr_epsilon", HPR_EPSILON_DEFAULT)),
            n_density_samples=int(self.config.get("hpr_n_samples", HPR_N_SAMPLES_DEFAULT)),
            theta_dim=self.theta_dim,
            rtol=float(self.config.get("rtol", 1e-5)),
            atol=float(self.config.get("atol", 1e-5)),
            hutchinson_samples=int(self.config.get("hutchinson_samples", 1)),
            device=self.device,
            dtype=self.dtype,
        )
        return hpr.build_region(self.x_obs)

    def _prepare_round(self, r: int) -> tuple:
        budget = int(self.config.get("budget_per_round", max(1, self.total_budget // self.rounds)))
        if r == 0:
            theta = self.benchmark.prior_sample(budget)
        else:
            theta = self.proposal_sampler.sample(
                budget,
                prior_sampler=self.benchmark.prior_sample,
                batch_size=int(self.config.get("proposal_batch_size", 512)),
                max_attempts_factor=int(self.config.get("max_attempts_factor", 10000)),
            )
        theta = theta.to(device=self.device, dtype=self.dtype)
        if self.use_weights:
            weights = self._importance_weights(theta)
        else:
            weights = torch.ones(theta.shape[0], device=theta.device, dtype=theta.dtype)
        return theta, weights

    def _ensure_prior_score_fn(self) -> None:
        """Train an implicit perturbed prior score for SNPSE-C if not supplied."""

        if self.prior_score_fn is not None:
            return
        self.prior_score_net = train_prior_score_network(
            self.benchmark.prior_sample,
            self.theta_dim,
            self.sde,
            num_samples=int(self.config.get("prior_score_samples", 5000)),
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            time_embedding_dim=int(self.config.get("time_embedding_dim", 64)),
            lr=float(self.config.get("lr", 1e-4)),
            steps=int(self.config.get("prior_score_steps", 2000)),
            batch_size=int(self.config.get("prior_score_batch_size", 256)),
            patience=int(self.config.get("prior_score_patience", 200)),
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            verbose=False,
        )
        self.prior_score_fn = self.prior_score_net

    def _train_proposal_prior_score(self) -> None:
        """Train the perturbed proposal-prior score for SNPSE-C."""

        def sampler(n: int) -> torch.Tensor:
            return self.proposal_sampler.sample(
                n,
                prior_sampler=self.benchmark.prior_sample,
                batch_size=int(self.config.get("proposal_batch_size", 512)),
                max_attempts_factor=int(self.config.get("max_attempts_factor", 10000)),
            )

        self.proposal_prior_net = train_prior_score_network(
            sampler,
            self.theta_dim,
            self.sde,
            num_samples=int(self.config.get("prior_score_samples", 5000)),
            hidden_dim=int(self.config.get("hidden_dim", 256)),
            time_embedding_dim=int(self.config.get("time_embedding_dim", 64)),
            lr=float(self.config.get("lr", 1e-4)),
            steps=int(self.config.get("prior_score_steps", 2000)),
            batch_size=int(self.config.get("prior_score_batch_size", 256)),
            patience=int(self.config.get("prior_score_patience", 200)),
            device=self.device,
            dtype=self.dtype,
            seed=self.seed,
            verbose=False,
        )

    def run(self, verbose: bool = True) -> "SNPSETrainer":
        """Run all sequential rounds and return the trained trainer."""

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        rounds = int(self.config.get("rounds", self.rounds))
        for r in range(rounds):
            theta, weights = self._prepare_round(r)
            x = self.benchmark.simulator(theta)
            x = x.to(device=self.device, dtype=self.dtype)

            self.all_theta.append(theta)
            self.all_x.append(x)
            self.all_weights.append(weights)

            theta_all = torch.cat(self.all_theta, dim=0)
            x_all = torch.cat(self.all_x, dim=0)
            w_all = (
                torch.cat(self.all_weights, dim=0) if self.use_weights else None
            )

            # Retrain from scratch on the accumulated dataset.
            self.score_net = self._create_score_net()
            self._train(self.score_net, theta_all, x_all, w_all, seed=self.seed)

            if verbose:
                print(
                    f"[SNPSE-{self.variant.upper()}] round {r + 1}/{rounds}: "
                    f"proposal samples={theta.shape[0]}, "
                    f"accumulated={theta_all.shape[0]}"
                )

            if r < rounds - 1:
                region = self._build_region(self.score_net)
                self.regions.append(region)
                self.proposal_sampler.add_round(region)

        if self.variant == "c":
            self._ensure_prior_score_fn()
            self._train_proposal_prior_score()

        return self

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------
    def final_score_fn(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return the score function used by the probability-flow ODE.

        For SNPSE-A the network is the proposal-posterior score (SIR is applied
        after sampling).  For SNPSE-B the network directly approximates the
        target posterior score.  For SNPSE-C the decomposed proposal-posterior
        score is returned.
        """

        if self.variant == "c":
            score_net = self.score_net
            proposal_prior = self.proposal_prior_net
            prior_score = self.prior_score_fn

            def combined(theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return (
                    score_net(theta, x, t)
                    + proposal_prior(theta, t)
                    - prior_score(theta, t)
                )

            return combined

        return self.score_net

    def sample_proposal_posterior(
        self, n_samples: int, x_obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample the proposal posterior without SIR correction."""

        x = self.x_obs if x_obs is None else x_obs
        return sample_probability_flow(
            self.sde,
            self.score_net,
            n_samples,
            x,
            theta_dim=self.theta_dim,
            device=self.device,
            dtype=self.dtype,
            rtol=float(self.config.get("rtol", 1e-5)),
            atol=float(self.config.get("atol", 1e-5)),
            method="rk45",
        )

    def sample_posterior(
        self, n_samples: int, x_obs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample the approximate target posterior."""

        x = self.x_obs if x_obs is None else x_obs
        needs_sir = self.variant in ("a", "c")
        n_draw = n_samples
        if needs_sir:
            n_draw = max(n_samples, int(n_samples * float(self.config.get("sir_oversampling", 4))))

        samples = sample_probability_flow(
            self.sde,
            self.final_score_fn(),
            n_draw,
            x,
            theta_dim=self.theta_dim,
            device=self.device,
            dtype=self.dtype,
            rtol=float(self.config.get("rtol", 1e-5)),
            atol=float(self.config.get("atol", 1e-5)),
            method="rk45",
        )

        if needs_sir:
            weights = self._importance_weights(samples)
            samples = self._sir_resample(samples, weights, n_samples)
        return samples


# ----------------------------------------------------------------------
# Convenience entry points
# ----------------------------------------------------------------------
def train_snpse(
    benchmark: Any,
    x_obs: Any,
    total_budget: int,
    rounds: int = 10,
    variant: str = "b",
    sde: Optional[SDE] = None,
    sde_type: str = "ve",
    score_net: Optional[ScoreNetwork] = None,
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    prior_score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    verbose: bool = True,
) -> SNPSETrainer:
    """Build an SDE if necessary, run SNPSE training, and return the trainer."""

    if sde is None:
        n_cal = min(2000, max(256, int(total_budget)))
        theta_cal = benchmark.prior_sample(n_cal)
        sde = build_sde(
            sde_type,
            theta_cal,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

    trainer = SNPSETrainer(
        benchmark,
        sde,
        x_obs,
        total_budget,
        rounds=rounds,
        variant=variant,
        score_net=score_net,
        config=config,
        seed=seed,
        prior_score_fn=prior_score_fn,
    )
    trainer.run(verbose=verbose)
    return trainer


def train_snpse_a(
    benchmark: Any,
    x_obs: Any,
    total_budget: int,
    rounds: int = 10,
    sde: Optional[SDE] = None,
    sde_type: str = "ve",
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    verbose: bool = True,
) -> SNPSETrainer:
    return train_snpse(
        benchmark,
        x_obs,
        total_budget,
        rounds=rounds,
        variant="a",
        sde=sde,
        sde_type=sde_type,
        config=config,
        seed=seed,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        verbose=verbose,
    )


def train_snpse_b(
    benchmark: Any,
    x_obs: Any,
    total_budget: int,
    rounds: int = 10,
    sde: Optional[SDE] = None,
    sde_type: str = "ve",
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    verbose: bool = True,
) -> SNPSETrainer:
    return train_snpse(
        benchmark,
        x_obs,
        total_budget,
        rounds=rounds,
        variant="b",
        sde=sde,
        sde_type=sde_type,
        config=config,
        seed=seed,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        verbose=verbose,
    )


def train_snpse_c(
    benchmark: Any,
    x_obs: Any,
    total_budget: int,
    rounds: int = 10,
    sde: Optional[SDE] = None,
    sde_type: str = "ve",
    config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    prior_score_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    sigma_min: Optional[float] = None,
    sigma_max: Optional[float] = None,
    verbose: bool = True,
) -> SNPSETrainer:
    return train_snpse(
        benchmark,
        x_obs,
        total_budget,
        rounds=rounds,
        variant="c",
        sde=sde,
        sde_type=sde_type,
        config=config,
        seed=seed,
        prior_score_fn=prior_score_fn,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        verbose=verbose,
    )
