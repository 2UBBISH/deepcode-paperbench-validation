"""Definition of the inference problem that the Simformer is trained on."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch


@dataclass
class Problem:
    """Everything the Simformer needs to know about a (theta, x) simulator.

    Attributes
    ----------
    name:
        Name of the task.
    n_variables, n_params, n_data:
        Size of the joint vector ``x_hat = (theta, x)``.
    sample_batch:
        ``(batch, rng) -> (theta, x, index, metadata)``.  ``theta`` and ``x`` are
        ``(batch, n_params)`` / ``(batch, n_data)`` numpy arrays; ``index`` is
        ``(batch, n_variables)`` (or ``None``) and holds the element of the index
        set of every variable (e.g. the time point of a time series entry);
        ``metadata`` is an optional dict with additional information used to
        build the attention mask (e.g. the observation times).
    variable_kind:
        ``(n_variables,)`` integer id of the "kind" of every variable.  Used for
        the shared part of the identifier embedding.
    use_fourier, index_dim:
        Which variables receive a random Fourier embedding of their index.
    mask_builder:
        ``(condition_state, index, metadata) -> (batch, n, n)`` boolean attention
        masks.  See :mod:`simformer.masks`.
    log_joint:
        Optional callable ``(theta, x) -> float`` with the (unnormalised) log
        joint density, used for reference MCMC sampling.
    """

    name: str
    n_variables: int
    n_params: int
    n_data: int
    sample_batch: Callable
    mask_builder: Callable
    variable_kind: np.ndarray
    use_fourier: np.ndarray
    index_dim: int = 1
    n_kinds: int = 1
    log_joint: Optional[Callable] = None
    param_names: Optional[list] = None
    data_names: Optional[list] = None
    info: Dict = field(default_factory=dict)

    @property
    def param_indices(self) -> np.ndarray:
        return np.arange(self.n_params)

    @property
    def data_indices(self) -> np.ndarray:
        return np.arange(self.n_params, self.n_variables)

    # ------------------------------------------------------------------ helpers
    def torch_variable_kind(self) -> torch.Tensor:
        return torch.as_tensor(self.variable_kind, dtype=torch.long)

    def torch_use_fourier(self) -> torch.Tensor:
        return torch.as_tensor(self.use_fourier, dtype=torch.bool)

    def base_attention_mask(self, condition_state=None, index=None,
                            metadata=None) -> np.ndarray:
        """Convenience wrapper around the mask builder."""
        if condition_state is None:
            condition_state = np.zeros((1, self.n_variables))
        masks = np.asarray(self.mask_builder(np.asarray(condition_state),
                                             index, metadata))
        return masks

    def sample_numpy(self, batch: int, rng: np.random.Generator):
        return self.sample_batch(batch, rng)

    def sample_torch(self, batch: int, rng: np.random.Generator,
                     device: str = "cpu"):
        theta, x, index, metadata = self.sample_batch(batch, rng)
        theta = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        index_t = None if index is None else torch.as_tensor(
            np.asarray(index), dtype=torch.float32)
        meta_t = None
        if metadata:
            meta_t = {k: (torch.as_tensor(np.asarray(v), dtype=torch.float32)
                          if np.asarray(v).dtype != object else v)
                      for k, v in metadata.items()}
        return theta.to(device), x.to(device), index_t, meta_t


def static_mask_builder(base_mask: np.ndarray) -> Callable:
    """Mask builder for problems whose attention mask does not depend on
    the condition state or the metadata."""
    base_mask = np.asarray(base_mask, dtype=bool)

    def builder(condition_state, index=None, metadata=None):
        batch = np.asarray(condition_state).shape[0]
        return np.broadcast_to(base_mask[None], (batch,) + base_mask.shape).copy()

    return builder
