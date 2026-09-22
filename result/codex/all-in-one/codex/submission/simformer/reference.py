"""Reference (ground truth) samples of arbitrary conditionals with MCMC.

Appendix A2.2 of the paper describes how the reference samples for every task
are obtained:

* Two Moons: initialise ``N`` chains from the joint distribution, run 1000 steps
  of a *random direction slice sampler*, then 3000 steps of Metropolis-Hastings
  with step size ``0.01``.
* SLCP: 600 slice sampling steps, then 2000 MH steps with step size ``0.1``.
* Tree / HMM: 5000 steps of Hamiltonian Monte Carlo.
* Lotka-Volterra: MCMC on the (tractable) posterior of the ODE parameters.

All samplers here operate on the *joint* distribution of the task and keep the
conditioned variables fixed, so that they can be used for **all** conditionals
(posterior, likelihood, arbitrary parameter conditionals).
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def _log_prob_fn(task, condition_state: np.ndarray,
                 condition_values: np.ndarray,
                 index: Optional[np.ndarray] = None,
                 metadata: Optional[dict] = None) -> Callable:
    """Return a **batched** function of the latent variables.

    The returned callable maps ``(n_chains, n_latent)`` to ``(n_chains,)`` and
    evaluates the joint log density of the task with the conditioned variables
    held fixed.  All samplers of this module operate on whole batches of chains
    at once, which makes the MCMC reference sampling of Appendix A2.2 feasible on
    a CPU.
    """
    condition_state = np.asarray(condition_state, dtype=float).reshape(-1)
    condition_values = np.asarray(condition_values, dtype=float).reshape(-1)
    latent_idx = np.flatnonzero(condition_state < 0.5)
    kwargs = {}
    if index is not None:
        if task.name == "lotka_volterra":
            kwargs["times"] = np.asarray(index)[task.n_params:
                                                task.n_params + task.n_grid]
        if task.name == "sird":
            kwargs["beta_times"] = np.asarray(index)[
                task.n_global_params:task.n_params]
            kwargs["obs_times"] = np.asarray(index)[task.n_params:
                                                    task.n_params + task.n_obs]

    def log_prob(z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, dtype=float))
        full = np.tile(condition_values, (z.shape[0], 1))
        full[:, latent_idx] = z
        theta = full[:, :task.n_params]
        x = full[:, task.n_params:]
        value = np.asarray(task.log_joint(theta, x, **kwargs), dtype=float).reshape(-1)
        return np.where(np.isfinite(value), value, -np.inf)

    return log_prob


def _numerical_gradient(log_prob: Callable, z: np.ndarray,
                        eps: float = 1e-5) -> np.ndarray:
    """Batched central finite difference gradient of ``log_prob``."""
    grad = np.empty_like(z)
    for i in range(z.shape[1]):
        step = np.zeros_like(z)
        step[:, i] = eps
        grad[:, i] = (log_prob(z + step) - log_prob(z - step)) / (2 * eps)
    return np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
#  Samplers
# --------------------------------------------------------------------------- #
def random_direction_slice_sampling(log_prob: Callable, x0: np.ndarray,
                                    n_steps: int, rng: np.random.Generator,
                                    width: float = 1.0) -> np.ndarray:
    """Random direction slice sampling (as used in the SBI benchmark).

    ``x0`` may be a single state ``(d,)`` or a batch of states ``(n_chains, d)``.
    """
    single = np.asarray(x0).ndim == 1
    x = np.atleast_2d(np.array(x0, dtype=float))
    n_chains, d = x.shape
    logp = log_prob(x)
    for _ in range(n_steps):
        direction = rng.normal(size=(n_chains, d))
        direction /= np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-300
        log_y = logp + np.log(rng.random() + 1e-300)
        # step out
        left = x - (rng.random() * width * direction)
        right = left + width * direction
        for _ in range(50):     # bounded step-out
            expand_left = log_prob(left) > log_y
            expand_right = log_prob(right) > log_y
            if not (expand_left.any() or expand_right.any()):
                break
            left = np.where(expand_left[:, None], left - width * direction, left)
            right = np.where(expand_right[:, None], right + width * direction,
                             right)
        # shrink
        for _ in range(100):
            proposal = left + (right - left) * rng.random(size=(n_chains, 1))
            logp_prop = log_prob(proposal)
            accept = logp_prop > log_y
            x = np.where(accept[:, None], proposal, x)
            logp = np.where(accept, logp_prop, logp)
            if accept.all():
                break
            below = (np.sum((proposal - left) * direction, axis=-1) < 0)
            left = np.where((below & ~accept)[:, None], proposal, left)
            right = np.where((~below & ~accept)[:, None], proposal, right)
    return x[0] if single else x


def metropolis_hastings(log_prob: Callable, x0: np.ndarray, n_steps: int,
                        rng: np.random.Generator,
                        step_size: float = 0.01) -> np.ndarray:
    """Random walk Metropolis-Hastings with a Gaussian proposal (batched)."""
    single = np.asarray(x0).ndim == 1
    x = np.atleast_2d(np.array(x0, dtype=float))
    logp = log_prob(x)
    for _ in range(n_steps):
        proposal = x + step_size * rng.normal(size=x.shape)
        logp_prop = log_prob(proposal)
        accept = np.log(rng.random(x.shape[0]) + 1e-300) < logp_prop - logp
        x = np.where(accept[:, None], proposal, x)
        logp = np.where(accept, logp_prop, logp)
    return x[0] if single else x


def hmc(log_prob: Callable, x0: np.ndarray, n_steps: int,
        rng: np.random.Generator, step_size: float = 0.1,
        n_leapfrog: int = 10, grad_fn: Optional[Callable] = None) -> np.ndarray:
    """Hamiltonian Monte Carlo with leapfrog integration (batched over chains)."""
    single = np.asarray(x0).ndim == 1
    x = np.atleast_2d(np.array(x0, dtype=float))
    grad = grad_fn if grad_fn is not None else (
        lambda z: _numerical_gradient(log_prob, z))
    logp = log_prob(x)
    n_accept = 0
    for _ in range(n_steps):
        p = rng.normal(size=x.shape)
        x_new, p_new = x.copy(), p.copy()
        p_new = p_new + 0.5 * step_size * grad(x_new)
        for _ in range(n_leapfrog - 1):
            x_new = x_new + step_size * p_new
            p_new = p_new + step_size * grad(x_new)
        x_new = x_new + step_size * p_new
        p_new = p_new + 0.5 * step_size * grad(x_new)
        logp_new = log_prob(x_new)
        energy_old = -logp + 0.5 * np.sum(p ** 2, axis=-1)
        energy_new = -logp_new + 0.5 * np.sum(p_new ** 2, axis=-1)
        accept = np.log(rng.random(x.shape[0]) + 1e-300) < energy_old - energy_new
        x = np.where(accept[:, None], x_new, x)
        logp = np.where(accept, logp_new, logp)
        n_accept += int(accept.sum())
    hmc.last_acceptance_rate = n_accept / (n_steps * x.shape[0])
    return x[0] if single else x


# --------------------------------------------------------------------------- #
#  Reference sampler for arbitrary conditionals
# --------------------------------------------------------------------------- #
REFERENCE_SETTINGS = {
    # task name: (n_slice_steps, n_mh_steps, mh_step_size) or HMC settings
    "gaussian_linear": dict(mode="mh", n_steps=5000, step_size=0.05),
    "gaussian_mixture": dict(mode="slice+mh", n_slice=1000, n_mh=3000,
                             mh_step_size=0.05),
    "two_moons": dict(mode="slice+mh", n_slice=1000, n_mh=3000,
                      mh_step_size=0.01),
    "slcp": dict(mode="slice+mh", n_slice=600, n_mh=2000, mh_step_size=0.1),
    "tree": dict(mode="hmc", n_steps=5000, step_size=0.15, n_leapfrog=10),
    "hmm": dict(mode="hmc", n_steps=5000, step_size=0.15, n_leapfrog=10),
    "lotka_volterra": dict(mode="hmc", n_steps=2000, step_size=0.02,
                           n_leapfrog=10),
}


def sample_conditional(task, condition_state: np.ndarray,
                       condition_values: np.ndarray, n_samples: int = 1000,
                       rng: Optional[np.random.Generator] = None,
                       n_chains: Optional[int] = None,
                       index: Optional[np.ndarray] = None,
                       mode: Optional[str] = None,
                       burn_in: Optional[int] = None,
                       thin: int = 1,
                       verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from ``p(latent | conditioned)`` with MCMC.

    Returns ``(theta_samples, x_samples)`` where the latent variables are drawn
    from the conditional of the joint distribution and the conditioned
    variables are constant.
    """
    settings = REFERENCE_SETTINGS.get(task.name, dict(mode="mh", n_steps=5000,
                                                      step_size=0.05))
    mode = mode or settings["mode"]
    rng = rng if rng is not None else np.random.default_rng(0)
    n_chains = n_chains or n_samples
    condition_state = np.asarray(condition_state, dtype=float).reshape(-1)
    latent_idx = np.flatnonzero(condition_state < 0.5)
    log_prob = _log_prob_fn(task, condition_state, condition_values, index)

    # initialise the chains from the prior / joint distribution
    theta0, x0, _, _ = task.joint_sample(n_chains, rng)
    init = np.concatenate([theta0, x0], axis=-1)
    init[:, np.flatnonzero(condition_state >= 0.5)] = \
        np.asarray(condition_values, dtype=float)[condition_state >= 0.5]

    # all chains are advanced in parallel (the samplers are batched)
    z = init[:, latent_idx]
    if mode in ("slice+mh", "slice"):
        n_slice = settings.get("n_slice", 1000)
        if verbose:
            print(f"  [reference] {n_chains} chains, {n_slice} slice sampling "
                  f"steps")
        z = random_direction_slice_sampling(log_prob, z, n_slice, rng, width=0.5)
        if mode == "slice+mh":
            n_mh = settings.get("n_mh", 3000)
            if verbose:
                print(f"  [reference] {n_mh} Metropolis-Hastings steps")
            z = metropolis_hastings(log_prob, z, n_mh, rng,
                                    settings.get("mh_step_size", 0.01))
        z = metropolis_hastings(log_prob, z, 500, rng, 0.01)
    elif mode == "hmc":
        n_steps = settings.get("n_steps", 5000)
        if verbose:
            print(f"  [reference] {n_chains} chains, {n_steps} HMC steps")
        z = hmc(log_prob, z, n_steps, rng,
                step_size=settings.get("step_size", 0.05),
                n_leapfrog=settings.get("n_leapfrog", 10))
        if verbose:
            print(f"  [reference] HMC acceptance rate "
                  f"{getattr(hmc, 'last_acceptance_rate', float('nan')):.2f}")
    else:
        z = metropolis_hastings(log_prob, z, settings.get("n_steps", 5000),
                                rng, settings.get("step_size", 0.05))

    samples = np.tile(np.asarray(condition_values, dtype=float),
                      (n_chains, 1))
    samples[:, latent_idx] = z
    theta_samples = samples[:, :task.n_params]
    x_samples = samples[:, task.n_params:]
    return theta_samples, x_samples
