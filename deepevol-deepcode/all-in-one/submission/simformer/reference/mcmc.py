"""Ground-truth conditional sampling via MCMC.

This module produces the reference samples used to evaluate Simformer's
arbitrary conditionals (Sec. 4.1 and Appendix A2.2 of the paper).

The paper's protocol for the benchmark tasks is:

* **Two Moons** -- initialize N Markov chains from the joint distribution, run
  1000 steps of a random-direction slice sampler, then 3000 additional steps of
  Metropolis-Hastings with step size 0.01 and keep only the last sample of each
  chain (yielding N reference samples).
* **SLCP** -- the same procedure with 600 slice-sampling steps and 2000 MH
  steps with step size 0.1.
* **Tree** and **HMM** -- initialize N chains from the joint distribution, run
  5000 steps of a HMC sampler and keep only the last sample of each chain.

Because the references have to cover *all possible conditionals* ``p(z | x_obs)``,
the samplers operate on an arbitrary subset of the joint vector ``(theta, x)``:
the observed coordinates are clamped to their observed values and the chain
explores the remaining (latent) coordinates.  The target density is the joint
``log p(theta, x) = log p(theta) + log p(x | theta)`` evaluated at the clamped
point, which is proportional to the conditional density in the latent block.

Everything is implemented in NumPy so that it works with the task simulators
(``simformer.tasks``) and is independent of the torch-based score network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # protocols / config
    "MCProtocol",
    "MCMCConfig",
    "TWO_MOONS_PROTOCOL",
    "SLCP_PROTOCOL",
    "TREE_PROTOCOL",
    "HMM_PROTOCOL",
    "TASK_PROTOCOLS",
    "task_protocol",
    # densities
    "make_joint_log_prob",
    "make_conditional_log_prob",
    "log_joint_from_task",
    "numerical_gradient",
    "make_grad_fn",
    # samplers
    "SliceSampler",
    "MetropolisHastings",
    "HamiltonianMonteCarlo",
    "slice_sample",
    "metropolis_hastings",
    "hamiltonian_monte_carlo",
    "random_direction_slice_sample",
    "sphere_direction",
    # chain drivers
    "run_chains",
    "run_slice_then_mh",
    "ReferenceSampler",
    "sample_reference",
    "sample_reference_conditionals",
    "reference_samples_for_task",
]

ArrayLike = Union[np.ndarray, Sequence[float]]
LogProb = Callable[[np.ndarray], float]
GradFn = Callable[[np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED = 0
DEFAULT_N_CHAINS = 1000
DEFAULT_SLICE_STEP = 1.0
DEFAULT_MAX_DOUBLINGS = 16
DEFAULT_HMC_LEAPFROG = 10
DEFAULT_TARGET_ACCEPT = 0.8


def _as_rng(rng: Optional[np.random.Generator] = None, seed: Optional[int] = None) -> np.random.Generator:
    """Coerce ``rng``/``seed`` into a ``numpy.random.Generator``."""
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):  # legacy support
        return np.random.default_rng(rng.randint(0, 2**31 - 1))
    if rng is None:
        return np.random.default_rng(DEFAULT_SEED if seed is None else seed)
    return np.random.default_rng(int(rng))


def _as_vector(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.atleast_1d(arr).ravel().copy()


# ---------------------------------------------------------------------------
# protocols
# ---------------------------------------------------------------------------


@dataclass
class MCProtocol:
    """Reference-sampling protocol for one task (Appendix A2.2).

    ``method`` is one of ``"slice+mh"`` (random-direction slice sampling
    followed by Metropolis-Hastings) or ``"hmc"``.
    """

    method: str = "slice+mh"
    n_slice: int = 1000
    slice_step: float = DEFAULT_SLICE_STEP
    n_mh: int = 3000
    mh_step: float = 0.01
    n_hmc: int = 5000
    hmc_step: float = 0.1
    n_leapfrog: int = DEFAULT_HMC_LEAPFROG
    adapt_step_size: bool = True
    init_from_joint: bool = True
    keep: str = "last"
    burn_in: int = 0
    thin: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# Protocols exactly as described in Appendix A2.2.
TWO_MOONS_PROTOCOL = MCProtocol(
    method="slice+mh", n_slice=1000, slice_step=1.0, n_mh=3000, mh_step=0.01
)
SLCP_PROTOCOL = MCProtocol(
    method="slice+mh", n_slice=600, slice_step=1.0, n_mh=2000, mh_step=0.1
)
TREE_PROTOCOL = MCProtocol(
    method="hmc", n_hmc=5000, hmc_step=0.1, n_leapfrog=DEFAULT_HMC_LEAPFROG, adapt_step_size=True
)
HMM_PROTOCOL = MCProtocol(
    method="hmc", n_hmc=5000, hmc_step=0.05, n_leapfrog=DEFAULT_HMC_LEAPFROG, adapt_step_size=True
)

TASK_PROTOCOLS: Dict[str, MCProtocol] = {
    "two_moons": TWO_MOONS_PROTOCOL,
    "two-moons": TWO_MOONS_PROTOCOL,
    "twomoons": TWO_MOONS_PROTOCOL,
    "slcp": SLCP_PROTOCOL,
    "tree": TREE_PROTOCOL,
    "hmm": HMM_PROTOCOL,
}


def task_protocol(task: Any) -> MCProtocol:
    """Return the reference protocol for a task name (or task instance).

    Falls back to the Two Moons style slice+MH protocol for unknown names.
    """
    name = task if isinstance(task, str) else getattr(task, "name", None)
    if name is None:
        return TWO_MOONS_PROTOCOL
    key = str(name).lower()
    if key in TASK_PROTOCOLS:
        return TASK_PROTOCOLS[key]
    # aliases such as "gaussian_linear"
    for alias, proto in TASK_PROTOCOLS.items():
        if alias in key:
            return proto
    return TWO_MOONS_PROTOCOL


# ---------------------------------------------------------------------------
# densities
# ---------------------------------------------------------------------------


def log_joint_from_task(task: Any) -> Callable[[ArrayLike, ArrayLike], float]:
    """Build ``log p(theta, x) = log p(theta) + log p(x | theta)`` from a task.

    Works with any object exposing ``log_prior(theta)`` and
    ``log_likelihood(x, theta)`` (the entire ``simformer.tasks`` family).
    """
    log_prior = getattr(task, "log_prior", None)
    log_likelihood = getattr(task, "log_likelihood", None)
    if log_prior is None or log_likelihood is None:
        raise AttributeError(
            f"task {task!r} must expose log_prior(theta) and log_likelihood(x, theta)"
        )

    def log_joint(theta: ArrayLike, x: ArrayLike) -> float:
        theta_v = _as_vector(theta)
        x_v = _as_vector(x)
        lp = float(np.sum(np.asarray(log_prior(theta_v), dtype=float)))
        ll = float(np.sum(np.asarray(log_likelihood(x_v, theta_v), dtype=float)))
        if not np.isfinite(lp):
            return -np.inf
        return lp + ll

    return log_joint


def make_joint_log_prob(
    task: Any = None,
    *,
    log_prior: Optional[Callable[[np.ndarray], float]] = None,
    log_likelihood: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    n_parameters: Optional[int] = None,
    n_data: Optional[int] = None,
) -> Callable[[np.ndarray], float]:
    """Vectorised joint log-density ``log p(joint)`` over the full joint vector.

    Either pass ``task`` or explicit ``log_prior``/``log_likelihood`` callables.
    """
    if log_prior is None or log_likelihood is None:
        if task is None:
            raise ValueError("provide either `task` or `log_prior`/`log_likelihood`")
        log_prior = task.log_prior
        log_likelihood = task.log_likelihood
        n_parameters = n_parameters if n_parameters is not None else getattr(task, "n_parameters", None)
        n_data = n_data if n_data is not None else getattr(task, "n_data", None)

    def joint_log_prob(joint: ArrayLike) -> float:
        joint_v = _as_vector(joint)
        if n_parameters is None:
            theta = joint_v[: joint_v.size // 2]
            x = joint_v[joint_v.size // 2 :]
        else:
            theta = joint_v[: int(n_parameters)]
            x = joint_v[int(n_parameters) :] if n_data is None else joint_v[int(n_parameters) : int(n_parameters) + int(n_data)]
        lp = float(np.sum(np.asarray(log_prior(theta), dtype=float)))
        if not np.isfinite(lp):
            return -np.inf
        ll = float(np.sum(np.asarray(log_likelihood(x, theta), dtype=float)))
        return lp + ll

    return joint_log_prob


def make_conditional_log_prob(
    joint_log_prob: Callable[[np.ndarray], float],
    condition_mask: ArrayLike,
    condition_values: ArrayLike,
) -> Tuple[LogProb, np.ndarray, np.ndarray]:
    """Conditional log-density over the latent block of the joint vector.

    ``condition_mask[i] = 1`` marks an observed (clamped) coordinate whose value
    is taken from ``condition_values[i]``; ``0`` marks a latent coordinate that
    the sampler explores.

    Returns
    -------
    (log_prob, latent_indices, condition_mask)
        ``log_prob`` accepts a vector of the latent coordinates only.
    """
    mask = np.asarray(condition_mask).astype(bool).ravel()
    values = _as_vector(condition_values)
    if values.size != mask.size:
        raise ValueError(
            f"condition_values has size {values.size} but condition_mask has size {mask.size}"
        )
    latent_idx = np.nonzero(~mask)[0]
    joint_dim = mask.size

    def log_prob(z_latent: ArrayLike) -> float:
        z = _as_vector(z_latent)
        if z.size != latent_idx.size:
            raise ValueError(
                f"latent vector has size {z.size}, expected {latent_idx.size}"
            )
        joint = values.copy()
        joint[latent_idx] = z
        if joint.size != joint_dim:  # pragma: no cover - defensive
            raise ValueError("joint dimension mismatch")
        return float(joint_log_prob(joint))

    return log_prob, latent_idx, mask


def numerical_gradient(fn: LogProb, x: ArrayLike, eps: float = 1e-4) -> np.ndarray:
    """Central finite-difference gradient of ``fn`` at ``x``."""
    x_v = _as_vector(x)
    grad = np.empty_like(x_v)
    for i in range(x_v.size):
        step = eps * max(1.0, abs(float(x_v[i])))
        xp = x_v.copy()
        xm = x_v.copy()
        xp[i] += step
        xm[i] -= step
        fp = fn(xp)
        fm = fn(xm)
        if not np.isfinite(fp) and not np.isfinite(fm):
            grad[i] = 0.0
        elif not np.isfinite(fp):
            grad[i] = -1e6
        elif not np.isfinite(fm):
            grad[i] = 1e6
        else:
            grad[i] = (fp - fm) / (2.0 * step)
    return grad


def make_grad_fn(
    log_prob: LogProb,
    *,
    use_torch: bool = False,
    eps: float = 1e-4,
    vectorised: bool = True,
) -> GradFn:
    """Build the gradient of ``log_prob``.

    Uses torch autograd when requested (and available), otherwise central
    finite differences.  ``vectorised`` currently only affects the torch branch.
    """
    if use_torch:
        try:  # pragma: no cover - optional dependency dispatch
            import torch

            def torch_grad(x: ArrayLike) -> np.ndarray:
                x_t = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
                x_t.requires_grad_(True)
                value = log_prob(x_t.detach().numpy())
                x_t2 = x_t.clone().requires_grad_(True)

                def numpy_fn(v: np.ndarray) -> float:
                    return log_prob(v)

                # log_prob is numpy-based; differentiate a torch wrapper instead
                with torch.enable_grad():
                    out = torch.autograd.functional.jacobian(
                        lambda z: torch.tensor(numpy_fn(z.detach().numpy()), dtype=torch.float64),
                        x_t2,
                    )
                return np.asarray(out.detach().numpy(), dtype=float).ravel()

            _ = value  # noqa: F841 - kept for clarity
            return torch_grad
        except Exception:  # pragma: no cover - fall back silently
            pass
    return lambda z: numerical_gradient(log_prob, z, eps=eps)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def sphere_direction(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed direction on the unit sphere in R^dim."""
    vec = rng.normal(size=dim)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0 or not np.isfinite(norm):  # pragma: no cover - astronomically unlikely
        vec = np.ones(dim)
        norm = float(np.linalg.norm(vec))
    return vec / norm


def slice_sample(
    log_prob: LogProb,
    x0: ArrayLike,
    n_steps: int = 1000,
    step_size: float = DEFAULT_SLICE_STEP,
    rng: Optional[np.random.Generator] = None,
    *,
    max_doublings: int = DEFAULT_MAX_DOUBLINGS,
    max_shrinks: int = 100,
    warmup: int = 0,
    record: bool = True,
) -> np.ndarray:
    """Random-direction slice sampling (Neal, 2003) for one chain.

    At each step a random direction ``d`` is drawn, a slice level is sampled
    below the current log-density, the interval ``[x - w d, x + w d]`` is
    stepped out until both ends are below the slice level, and a point is drawn
    uniformly in the interval, shrinking whenever the point is rejected.

    Returns an array of shape ``(n_steps, dim)`` (or ``(0, dim)`` when
    ``record=False``).
    """
    rng = _as_rng(rng)
    x = _as_vector(x0)
    dim = x.size
    logp = float(log_prob(x))
    if not np.isfinite(logp):
        raise ValueError("initial point has non-finite log density")

    chain = np.empty((n_steps, dim)) if record else np.empty((0, dim))
    for total in range(warmup + n_steps):
        direction = sphere_direction(dim, rng)
        logy = logp + math.log(max(rng.random(), 1e-300))

        # stepping out
        w = float(step_size) * max(1.0, float(rng.random()))
        left = x - w * direction
        right = left + w * direction
        logp_left = float(log_prob(left))
        logp_right = float(log_prob(right))
        k = int(max_doublings)
        while k > 0 and (logp_left > logy or logp_right > logy):
            if rng.random() < 0.5:
                left = left - (right - left)
                logp_left = float(log_prob(left))
            else:
                right = right + (right - left)
                logp_right = float(log_prob(right))
            k -= 1

        # shrinkage
        accepted = False
        for _ in range(int(max_shrinks)):
            u = float(rng.random())
            x_new = left + u * (right - left)
            logp_new = float(log_prob(x_new))
            if np.isfinite(logp_new) and logp_new > logy:
                x = x_new
                logp = logp_new
                accepted = True
                break
            if u < 0.5:
                left = x_new
            else:
                right = x_new
        if not accepted:
            # Keep the current point (the slice level is already below `logp`).
            pass

        if total >= warmup and record:
            chain[total - warmup] = x
    return chain


# Alias with the explicit name used in the paper.
random_direction_slice_sample = slice_sample


def metropolis_hastings(
    log_prob: LogProb,
    x0: ArrayLike,
    n_steps: int = 3000,
    step_size: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    *,
    adapt_step_size: bool = False,
    target_accept: float = 0.234,
    warmup: int = 0,
    record: bool = True,
) -> np.ndarray:
    """Random-walk Metropolis-Hastings with Gaussian proposals.

    ``step_size`` is the standard deviation of the isotropic proposal, exactly
    as specified in Appendix A2.2 (0.01 for Two Moons, 0.1 for SLCP).
    """
    rng = _as_rng(rng)
    x = _as_vector(x0)
    dim = x.size
    logp = float(log_prob(x))
    if not np.isfinite(logp):
        raise ValueError("initial point has non-finite log density")

    chain = np.empty((n_steps, dim)) if record else np.empty((0, dim))
    scale = float(step_size)
    log_scale = math.log(max(scale, 1e-12))
    accepts_window: List[float] = []
    for total in range(warmup + n_steps):
        scale = math.exp(log_scale)
        proposal = x + scale * rng.normal(size=dim)
        logp_new = float(log_prob(proposal))
        accept_prob = 1.0 if logp_new >= logp else math.exp(logp_new - logp)
        accepted = rng.random() < accept_prob
        if accepted:
            x = proposal
            logp = logp_new

        if adapt_step_size:
            accepts_window.append(1.0 if accepted else 0.0)
            if len(accepts_window) >= 50:
                rate = float(np.mean(accepts_window))
                accepts_window = []
                log_scale += 0.5 * (rate - float(target_accept))

        if total >= warmup and record:
            chain[total - warmup] = x
    return chain


def hamiltonian_monte_carlo(
    log_prob: LogProb,
    x0: ArrayLike,
    n_steps: int = 5000,
    step_size: float = 0.1,
    n_leapfrog: int = DEFAULT_HMC_LEAPFROG,
    rng: Optional[np.random.Generator] = None,
    *,
    grad_fn: Optional[GradFn] = None,
    mass: Union[float, np.ndarray] = 1.0,
    adapt_step_size: bool = True,
    target_accept: float = DEFAULT_TARGET_ACCEPT,
    warmup: int = 0,
    record: bool = True,
) -> np.ndarray:
    """Hamiltonian Monte Carlo with leapfrog integration and optional dual averaging.

    Used for the Tree and HMM tasks (5000 steps, keep the last sample).
    Gradients default to central finite differences of ``log_prob``.
    """
    rng = _as_rng(rng)
    x = _as_vector(x0)
    dim = x.size
    logp = float(log_prob(x))
    if not np.isfinite(logp):
        raise ValueError("initial point has non-finite log density")
    grad = grad_fn if grad_fn is not None else make_grad_fn(log_prob)

    mass_arr = np.asarray(mass, dtype=float)
    if mass_arr.ndim == 0:
        mass_arr = np.full(dim, float(mass_arr))
    inv_mass = 1.0 / mass_arr

    chain = np.empty((n_steps, dim)) if record else np.empty((0, dim))

    log_eps = math.log(max(float(step_size), 1e-8))
    mu = math.log(10.0 * max(float(step_size), 1e-8))
    hbar = 0.0
    log_eps_bar = 0.0
    kappa = 0.75
    t0 = 10.0
    gamma = 0.05

    for total in range(warmup + n_steps):
        eps = math.exp(log_eps)
        momentum = rng.normal(size=dim) * np.sqrt(mass_arr)
        kinetic = 0.5 * float(np.sum(momentum**2 * inv_mass))
        h0 = -logp + kinetic

        x_new = x.copy()
        p_new = momentum.copy()
        p_new = p_new - 0.5 * eps * grad(x_new)
        for j in range(int(n_leapfrog)):
            x_new = x_new + eps * p_new * inv_mass
            g = grad(x_new)
            if j < int(n_leapfrog) - 1:
                p_new = p_new - eps * g
            else:
                p_new = p_new - 0.5 * eps * g

        logp_new = float(log_prob(x_new))
        kinetic_new = 0.5 * float(np.sum(p_new**2 * inv_mass))
        h1 = -logp_new + kinetic_new
        accept_prob = 1.0 if h1 <= h0 else math.exp(h0 - h1)
        if not np.isfinite(accept_prob):
            accept_prob = 0.0
        accepted = rng.random() < accept_prob
        if accepted:
            x = x_new
            logp = logp_new

        if adapt_step_size and total < max(warmup, 1) + 1000:
            # dual averaging on the log step size
            eta = 1.0 / (total + 1.0 + t0)
            hbar = (1.0 - eta) * hbar + eta * (float(target_accept) - accept_prob)
            log_eps = mu - math.sqrt(total + 1.0) / gamma * hbar
            log_eps_bar = (1.0 - eta) ** kappa * log_eps_bar + (1.0 - (1.0 - eta) ** kappa) * log_eps
        elif total == max(warmup, 1) + 1000 and adapt_step_size:
            log_eps = log_eps_bar

        if total >= warmup and record:
            chain[total - warmup] = x
    return chain


# ---------------------------------------------------------------------------
# chain drivers
# ---------------------------------------------------------------------------


def run_chains(
    log_prob: LogProb,
    init: ArrayLike,
    protocol: MCProtocol,
    rng: Optional[np.random.Generator] = None,
    *,
    grad_fn: Optional[GradFn] = None,
) -> np.ndarray:
    """Run one MCMC chain from ``init`` under ``protocol`` and keep the last sample.

    Returns the reference sample of shape ``(latent_dim,)``.
    """
    rng = _as_rng(rng)
    init_v = _as_vector(init)
    if protocol.method == "hmc":
        chain = hamiltonian_monte_carlo(
            log_prob,
            init_v,
            n_steps=int(protocol.n_hmc),
            step_size=float(protocol.hmc_step),
            n_leapfrog=int(protocol.n_leapfrog),
            rng=rng,
            grad_fn=grad_fn,
            adapt_step_size=bool(protocol.adapt_step_size),
        )
    elif protocol.method in ("slice+mh", "slice", "mh", "slice_mh"):
        chain = run_slice_then_mh(log_prob, init_v, protocol, rng)
    else:
        raise ValueError(f"unknown MCMC method {protocol.method!r}")

    if chain.shape[0] == 0:
        return init_v
    burn = int(protocol.burn_in)
    thin = max(int(protocol.thin), 1)
    samples = chain[burn::thin]
    if samples.shape[0] == 0:
        samples = chain
    if protocol.keep == "last":
        return samples[-1]
    return samples


def run_slice_then_mh(
    log_prob: LogProb,
    init: ArrayLike,
    protocol: MCProtocol,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Slice sampling followed by Metropolis-Hastings (Two Moons / SLCP).

    Returns the concatenated chain of shape ``(n_slice + n_mh, dim)``.
    """
    rng = _as_rng(rng)
    x = _as_vector(init)
    pieces: List[np.ndarray] = []
    if protocol.n_slice > 0:
        slice_chain = slice_sample(
            log_prob,
            x,
            n_steps=int(protocol.n_slice),
            step_size=float(protocol.slice_step),
            rng=rng,
        )
        pieces.append(slice_chain)
        x = slice_chain[-1]
    if protocol.n_mh > 0:
        mh_chain = metropolis_hastings(
            log_prob,
            x,
            n_steps=int(protocol.n_mh),
            step_size=float(protocol.mh_step),
            rng=rng,
        )
        pieces.append(mh_chain)
    if not pieces:  # pragma: no cover - degenerate protocol
        return np.zeros((0, x.size))
    return np.concatenate(pieces, axis=0)


def _draw_joint_init(task: Any, condition_mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Initial latent points drawn from the joint distribution (paper protocol)."""
    mask = np.asarray(condition_mask).astype(bool).ravel()
    latent_idx = np.nonzero(~mask)[0]
    n_parameters = getattr(task, "n_parameters", None)
    n_data = getattr(task, "n_data", None)
    if n_parameters is None:
        # Without explicit dims assume the first half holds the parameters.
        n_parameters = mask.size // 2

    theta = np.asarray(task.prior_sample(n, rng), dtype=float)
    theta = theta.reshape(n, -1)
    x = np.asarray(task.simulate(theta, rng), dtype=float)
    x = x.reshape(n, -1)
    joint = np.concatenate([theta, x], axis=1)

    if joint.shape[1] < mask.size:  # pad function-valued/extra coordinates
        joint = np.concatenate(
            [joint, np.zeros((n, mask.size - joint.shape[1]))], axis=1
        )
    return joint[:, latent_idx]


class ReferenceSampler:
    """Reference conditional sampler for one task.

    Parameters
    ----------
    task:
        Task object (or its name) exposing ``prior_sample``, ``simulate``,
        ``log_prior`` and ``log_likelihood``.
    protocol:
        MCMC protocol; defaults to the task-specific protocol of Appendix A2.2.
    """

    def __init__(
        self,
        task: Any,
        protocol: Optional[MCProtocol] = None,
        *,
        n_parameters: Optional[int] = None,
        n_data: Optional[int] = None,
        use_finite_differences: bool = True,
        joint_log_prob: Optional[Callable[[np.ndarray], float]] = None,
    ) -> None:
        self.task = task
        self.task_name = task if isinstance(task, str) else getattr(task, "name", "unknown")
        self.protocol = protocol or task_protocol(task)
        self.n_parameters = n_parameters if n_parameters is not None else getattr(task, "n_parameters", None)
        self.n_data = n_data if n_data is not None else getattr(task, "n_data", None)
        self.use_finite_differences = bool(use_finite_differences)
        if joint_log_prob is not None:
            self.joint_log_prob = joint_log_prob
        else:
            self.joint_log_prob = make_joint_log_prob(
                task, n_parameters=self.n_parameters, n_data=self.n_data
            )

    # -- helpers -----------------------------------------------------------
    @property
    def joint_dim(self) -> int:
        if self.n_parameters is None or self.n_data is None:
            return 2 * (self.n_parameters or self.n_data or 0)
        return int(self.n_parameters) + int(self.n_data)

    def conditional_log_prob(
        self, condition_mask: ArrayLike, condition_values: ArrayLike
    ) -> Tuple[LogProb, np.ndarray]:
        return make_conditional_log_prob(
            self.joint_log_prob, condition_mask, condition_values
        )[:2]

    def init_from_joint(
        self, condition_mask: ArrayLike, n: int, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        rng = _as_rng(rng)
        if self.protocol.init_from_joint:
            return _draw_joint_init(self.task, np.asarray(condition_mask), n, rng)
        mask = np.asarray(condition_mask).astype(bool).ravel()
        return rng.normal(size=int((~mask).sum())) * 0.5

    # -- sampling ----------------------------------------------------------
    def sample(
        self,
        condition_mask: ArrayLike,
        condition_values: ArrayLike,
        n_samples: int = DEFAULT_N_CHAINS,
        *,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        init: Optional[ArrayLike] = None,
        protocol: Optional[MCProtocol] = None,
        return_full: bool = False,
    ) -> np.ndarray:
        """Draw ``n_samples`` reference samples of ``p(latent | observed)``.

        Following the paper, ``n_samples`` chains are initialised from the joint
        distribution and only the last sample of each chain is kept.  If more
        samples than the batch size are requested the procedure is repeated in
        groups.
        """
        rng = _as_rng(rng, seed)
        proto = protocol or self.protocol
        mask = np.asarray(condition_mask).astype(bool).ravel()
        values = _as_vector(condition_values)
        log_prob, latent_idx = self.conditional_log_prob(mask, values)
        dim = latent_idx.size
        if dim == 0:  # nothing latent -> deterministic answer
            joint = values.copy()
            return joint[None, :] if return_full else joint[latent_idx][None, :]

        if init is not None:
            init_arr = np.atleast_2d(_as_vector(init))
            if init_arr.shape[0] == 1:
                init_arr = np.repeat(init_arr, n_samples, axis=0)
        else:
            init_arr = self.init_from_joint(mask, n_samples, rng)

        out = np.empty((n_samples, dim))
        for i in range(n_samples):
            out[i] = run_chains(log_prob, init_arr[i], proto, rng)
        if return_full:
            full = np.tile(values[None, :], (n_samples, 1))
            full[:, latent_idx] = out
            return full
        return out

    # aliases
    sample_conditional = sample

    def __call__(
        self,
        condition_mask: ArrayLike,
        condition_values: ArrayLike,
        n_samples: int = DEFAULT_N_CHAINS,
        **kwargs: Any,
    ) -> np.ndarray:
        return self.sample(condition_mask, condition_values, n_samples, **kwargs)


def sample_reference(
    task: Any,
    condition_mask: ArrayLike,
    condition_values: ArrayLike,
    n_samples: int = DEFAULT_N_CHAINS,
    *,
    protocol: Optional[MCProtocol] = None,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    return_full: bool = False,
    init: Optional[ArrayLike] = None,
) -> np.ndarray:
    """One-shot reference sampling for a conditional of ``task``."""
    sampler = ReferenceSampler(task, protocol)
    return sampler.sample(
        condition_mask,
        condition_values,
        n_samples,
        seed=seed,
        rng=rng,
        init=init,
        return_full=return_full,
    )


def sample_reference_conditionals(
    task: Any,
    condition_masks: ArrayLike,
    condition_values: ArrayLike,
    n_samples: int = DEFAULT_N_CHAINS,
    *,
    protocol: Optional[MCProtocol] = None,
    seed: Optional[int] = None,
    return_full: bool = False,
) -> List[np.ndarray]:
    """Reference samples for a list of conditionals (Sec. 4.1 protocol)."""
    masks = np.atleast_2d(np.asarray(condition_masks).astype(bool))
    values = np.atleast_2d(_as_vector(condition_values)) if np.asarray(condition_values).ndim <= 1 else np.asarray(condition_values, dtype=float)
    if values.shape[0] != masks.shape[0]:
        values = np.tile(values.reshape(1, -1), (masks.shape[0], 1))
    sampler = ReferenceSampler(task, protocol)
    out: List[np.ndarray] = []
    for i in range(masks.shape[0]):
        out.append(
            sampler.sample(
                masks[i],
                values[i],
                n_samples,
                seed=None if seed is None else int(seed) + i,
                return_full=return_full,
            )
        )
    return out


def reference_samples_for_task(
    task_name: str = "two_moons",
    n_targets: int = 100,
    n_samples: int = DEFAULT_N_CHAINS,
    *,
    seed: int = DEFAULT_SEED,
    protocol: Optional[MCProtocol] = None,
    task: Any = None,
    return_masks: bool = True,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], np.ndarray, np.ndarray]]:
    """Reference samples for ``n_targets`` random conditionals of a benchmark task.

    Implements the evaluation protocol of Sec. 4.1 (100 random joint targets) for
    the arbitrary-conditional experiments.
    """
    if task is None:  # lazy import to avoid a circular dependency
        from simformer.tasks import build_task as _build_task  # type: ignore

        task = _build_task(task_name)
    rng = np.random.default_rng(seed)
    n_parameters = int(getattr(task, "n_parameters", 0))
    n_data = int(getattr(task, "n_data", 0))
    n_variables = n_parameters + n_data

    # Draw a ground-truth joint sample and random condition masks (protocol used
    # in sampling.random_conditional_targets).
    joint = None
    try:
        from simformer.sampling import random_conditional_targets  # type: ignore

        masks = random_conditional_targets(n_variables, n_targets, seed=seed)
    except Exception:  # pragma: no cover - fallback mask generator
        masks = rng.random((n_targets, n_variables)) < 0.5
    theta = np.asarray(task.prior_sample(1, rng), dtype=float).reshape(-1)
    x = np.asarray(task.simulate(theta[None, :], rng), dtype=float).reshape(-1)
    joint = np.concatenate([theta, x])

    # Expand variable-level masks to the joint vector (function-valued tasks use
    # one joint entry per variable in the plain benchmark setting).
    value_masks = np.tile(masks, (1, 1))
    if joint.size != n_variables:
        value_masks = np.zeros((n_targets, joint.size), dtype=bool)
        widths = joint.size // max(n_variables, 1)
        for i in range(n_targets):
            for v in range(n_variables):
                value_masks[i, v * widths : (v + 1) * widths] = masks[i, v]
        joint = joint[: value_masks.shape[1]]

    values = np.tile(joint[None, :], (n_targets, 1)) * value_masks
    refs = sample_reference_conditionals(
        task, value_masks, values, n_samples, protocol=protocol, seed=seed
    )
    if return_masks:
        return refs, value_masks, values
    return refs
