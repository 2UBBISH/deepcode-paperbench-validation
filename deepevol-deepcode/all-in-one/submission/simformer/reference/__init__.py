"""Reference (ground-truth) conditional samplers for Simformer.

This package provides NumPy-based MCMC machinery used to obtain reference
samples of arbitrary conditionals ``p(z | x_obs)`` so that amortized
conditional samplers (Simformer and baselines) can be scored with C2ST,
coverage and likelihood metrics.

The sampling protocols follow Appendix A2.2 of the Simformer paper:

* Two Moons / SLCP: random-direction slice sampling (Neal 2003) followed by
  Metropolis-Hastings, keeping the last sample of each chain.
* Tree / HMM: 5000-step Hamiltonian Monte Carlo, keeping the last sample of
  each chain.

The heavy lifting lives in :mod:`simformer.reference.mcmc`; this module only
re-exports the public surface and adds a small ``__getattr__`` hook so the
module can be imported lazily (mirroring ``simformer.tasks``).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # protocol configuration
    "MCProtocol",
    "TASK_PROTOCOLS",
    "TWO_MOONS_PROTOCOL",
    "SLCP_PROTOCOL",
    "TREE_PROTOCOL",
    "HMM_PROTOCOL",
    "task_protocol",
    # samplers
    "ReferenceSampler",
    "sample_reference",
    "sample_reference_conditionals",
    "reference_samples_for_task",
    # building blocks
    "log_joint_from_task",
    "make_joint_log_prob",
    "make_conditional_log_prob",
    "make_grad_fn",
    "numerical_gradient",
    "slice_sample",
    "random_direction_slice_sample",
    "metropolis_hastings",
    "hamiltonian_monte_carlo",
    "run_chains",
    "run_slice_then_mh",
    "sphere_direction",
]

#: Attributes re-exported from :mod:`simformer.reference.mcmc`.
_EXPORTED: Tuple[str, ...] = tuple(__all__)

_MODULE_CACHE: Dict[str, Any] = {}

DEFAULT_SEED = 0
DEFAULT_N_CHAINS = 1000


def _mcmc_module() -> Any:
    """Import (and cache) :mod:`simformer.reference.mcmc`."""
    module = _MODULE_CACHE.get("mcmc")
    if module is None:
        try:  # absolute import (installed / package on sys.path)
            module = importlib.import_module("simformer.reference.mcmc")
        except ImportError:  # pragma: no cover - relative fallback
            module = importlib.import_module(__name__ + ".mcmc")
        _MODULE_CACHE["mcmc"] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation
    if name in _EXPORTED:
        module = _mcmc_module()
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + list(_EXPORTED))


def reference_protocol(task: Any) -> Any:
    """Return the reference-sampling protocol for ``task``.

    Thin wrapper around :func:`simformer.reference.mcmc.task_protocol` that
    accepts either a task name or an instantiated task object.
    """
    module = _mcmc_module()
    if isinstance(task, str):
        return module.task_protocol(task)
    return module.task_protocol(task)


def get_sampler(
    task: Any,
    protocol: Optional[Any] = None,
    *,
    n_parameters: Optional[int] = None,
    n_data: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build a :class:`ReferenceSampler` for ``task`` with an optional protocol."""
    module = _mcmc_module()
    if protocol is None:
        protocol = module.task_protocol(task)
    return module.ReferenceSampler(
        task,
        protocol,
        n_parameters=n_parameters,
        n_data=n_data,
        **kwargs,
    )
