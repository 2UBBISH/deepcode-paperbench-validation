"""Base classes and utilities for simulation-based inference benchmarks.

All benchmark simulators implement the small :class:`Benchmark` interface so
that training and evaluation scripts can operate on them uniformly.  The
interface intentionally keeps the simulator API minimal:

* ``prior_sample(n)`` draws parameters from the prior.
* ``simulator(theta)`` draws observations conditionally on parameters.
* ``prior_log_prob(theta)`` optionally evaluates the prior density.
* ``reference_posterior_samples(x_obs, n)`` optionally provides reference
  posterior draws (usually through ``sbibm``) for evaluation.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple

import numpy as np
import torch

__all__ = [
    "Benchmark",
    "to_torch",
    "ensure_tensor",
]


def to_torch(
    value,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Convert a numpy array, list, or scalar to a torch tensor.

    Parameters
    ----------
    value:
        Input object. Torch tensors are returned with the requested device and
        dtype; numpy arrays and lists are converted first.
    device:
        Destination device. If ``None``, the input tensor's device is kept for
        tensors and CPU is used for newly created tensors.
    dtype:
        Destination dtype. Defaults to ``torch.float32`` when ``value`` is not
        already a floating point tensor.

    Returns
    -------
    torch.Tensor
    """
    if isinstance(value, torch.Tensor):
        out = value
    else:
        out = torch.as_tensor(np.asarray(value))

    if dtype is not None:
        out = out.to(dtype=dtype)
    elif out.dtype not in (torch.float16, torch.float32, torch.float64):
        out = out.to(dtype=torch.float32)

    if device is not None:
        out = out.to(device=device)
    return out


def ensure_tensor(
    value,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Alias for :func:`to_torch` kept for API clarity."""
    return to_torch(value, device=device, dtype=dtype)


class Benchmark(abc.ABC):
    """Abstract benchmark simulator.

    Subclasses must set ``theta_dim`` and ``x_dim`` and implement
    :meth:`prior_sample` and :meth:`simulator`.  A simulator is expected to be
    vectorized over the leading batch dimension: if ``theta`` has shape
    ``(batch, theta_dim)`` then the returned observations have shape
    ``(batch, x_dim)``.
    """

    name: str = "abstract_benchmark"
    theta_dim: int = 0
    x_dim: int = 0

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def prior_sample(self, n_samples: int) -> torch.Tensor:
        """Draw ``n_samples`` parameter vectors from the prior.

        Returns a tensor of shape ``(n_samples, theta_dim)``.
        """

    @abc.abstractmethod
    def simulator(self, theta: torch.Tensor) -> torch.Tensor:
        """Simulate observations for a batch of parameters.

        ``theta`` has shape ``(batch, theta_dim)``. Returns observations of
        shape ``(batch, x_dim)``.
        """

    # ------------------------------------------------------------------
    # Optional, overridable interface
    # ------------------------------------------------------------------
    def prior_log_prob(self, theta: torch.Tensor) -> Optional[torch.Tensor]:
        """Evaluate the (possibly unnormalised) prior log density.

        The default implementation returns ``None``, indicating that the prior
        density is implicit/unavailable. Benchmarks with a tractable prior may
        override this to return log densities of shape ``(batch,)``.
        """
        return None

    def reference_posterior_samples(
        self,
        x_obs: torch.Tensor,
        n_samples: int = 10_000,
    ) -> Optional[torch.Tensor]:
        """Return reference posterior samples for an observation if available.

        The default implementation returns ``None``.  SBIBM-backed benchmarks
        should override this method using their task's reference sampler.
        """
        return None

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def sample_joint(
        self,
        n_samples: int,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw ``n_samples`` parameter-observation pairs.

        Parameters
        ----------
        n_samples:
            Total number of joint samples to generate.
        batch_size:
            If provided, the simulator is called on chunks of at most this
            size. This is useful for GPU simulators with memory constraints.

        Returns
        -------
        (theta, x)
            Tensors of shape ``(n_samples, theta_dim)`` and
            ``(n_samples, x_dim)``.
        """
        theta = self.prior_sample(n_samples)
        if batch_size is None or n_samples <= batch_size:
            x = self.simulator(theta)
            return theta, x

        chunks = []
        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            chunks.append(self.simulator(theta[start:stop]))
        x = torch.cat(chunks, dim=0)
        return theta, x

    def sample_observation(self, n_observations: int = 1) -> torch.Tensor:
        """Sample a test observation from the prior predictive distribution.

        Returns a tensor of shape ``(n_observations, x_dim)``.
        """
        theta = self.prior_sample(n_observations)
        return self.simulator(theta)

    def to(self, device: Optional[torch.device] = None) -> "Benchmark":
        """Move any registered tensors to ``device`` and return ``self``."""
        if device is not None:
            self.device = torch.device(device)
        return self
