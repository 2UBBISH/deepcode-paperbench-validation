"""Benchmark simulators for NPSE experiments.

Each benchmark subclasses :class:`npse.benchmarks.base.Benchmark` and exposes a
uniform interface used by training/evaluation scripts:

- ``prior_sample(n)``        -> parameter samples of shape ``(n, theta_dim)``
- ``simulator(theta)``       -> observation samples of shape ``(batch, x_dim)``
- ``prior_log_prob(theta)``  -> exact prior log density (when available)
- ``reference_posterior_samples(n, x)`` -> reference posterior draws (when available)
"""

from .base import Benchmark, to_torch, ensure_tensor
from .gaussian_linear import GaussianLinear
from .gaussian_mixture import GaussianMixture
from .two_moons import TwoMoons
from .gaussian_linear_uniform import GaussianLinearUniform
from .bernoulli_glm import BernoulliGLM
from .slcp import SLCP
from .sir import SIR
from .lotka_volterra import LotkaVolterra

__all__ = [
    "Benchmark",
    "to_torch",
    "ensure_tensor",
    "GaussianLinear",
    "GaussianMixture",
    "TwoMoons",
    "GaussianLinearUniform",
    "BernoulliGLM",
    "SLCP",
    "SIR",
    "LotkaVolterra",
]

# Registry mapping benchmark names to classes for configuration-driven scripts.
BENCHMARK_REGISTRY = {
    "gaussian_linear": GaussianLinear,
    "gaussian_mixture": GaussianMixture,
    "two_moons": TwoMoons,
    "gaussian_linear_uniform": GaussianLinearUniform,
    "bernoulli_glm": BernoulliGLM,
    "slcp": SLCP,
    "sir": SIR,
    "lotka_volterra": LotkaVolterra,
}


def get_benchmark(name: str, **kwargs) -> Benchmark:
    """Instantiate a benchmark by its registry name.

    Args:
        name: Registry key (e.g. ``"gaussian_linear"``).
        **kwargs: Forwarded to the benchmark constructor.

    Returns:
        A :class:`Benchmark` instance.

    Raises:
        KeyError: If ``name`` is not a known benchmark.
    """
    try:
        cls = BENCHMARK_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(BENCHMARK_REGISTRY))
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available}") from exc
    return cls(**kwargs)
