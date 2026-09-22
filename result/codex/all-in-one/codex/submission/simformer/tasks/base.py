"""Base class of the simulator tasks of the paper."""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..problem import Problem, static_mask_builder


class Task:
    """A simulator with parameters ``theta`` and data ``x``.

    Subclasses have to implement :meth:`prior_sample`, :meth:`simulate` and
    :meth:`base_mask`.  Implementing :meth:`log_joint` is required to generate
    reference samples of arbitrary conditionals with MCMC (Appendix A2.2).
    """

    name: str = "task"
    n_params: int = 0
    n_data: int = 0
    index_dim: int = 1

    # ------------------------------------------------------------- simulation
    @property
    def n_variables(self) -> int:
        return self.n_params + self.n_data

    def prior_sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def simulate(self, theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def joint_sample(self, n: int, rng: np.random.Generator):
        """Return ``(theta, x, index, metadata)`` for a batch of ``n`` samples."""
        theta = self.prior_sample(n, rng)
        x = self.simulate(theta, rng)
        index = np.broadcast_to(np.arange(self.n_variables, dtype=np.float32),
                                (n, self.n_variables)).copy()
        return theta, x, index, None

    def parameters_to_vector(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return np.concatenate([theta, x], axis=-1)

    # -------------------------------------------------------------- structure
    def base_mask(self) -> np.ndarray:
        """Directed attention mask ``M_E`` of the generative model."""
        raise NotImplementedError

    def variable_kind(self) -> np.ndarray:
        return np.zeros(self.n_variables, dtype=np.int64)

    def n_kinds(self) -> int:
        return int(self.variable_kind().max()) + 1

    def use_fourier(self) -> np.ndarray:
        return np.zeros(self.n_variables, dtype=bool)

    def mask_factory(self) -> Callable:
        """Return the mask builder passed to :class:`~simformer.problem.Problem`."""
        return static_mask_builder(self.base_mask())

    # ---------------------------------------------------------------- density
    def log_joint(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Unnormalised log joint density ``log p(theta, x)`` (per sample)."""
        raise NotImplementedError(
            f"{self.name} does not provide an explicit log joint density.")

    # ----------------------------------------------------------------- problem
    def problem(self) -> Problem:
        return Problem(
            name=self.name,
            n_variables=self.n_variables,
            n_params=self.n_params,
            n_data=self.n_data,
            sample_batch=self.joint_sample,
            mask_builder=self.mask_factory(),
            variable_kind=self.variable_kind(),
            use_fourier=self.use_fourier(),
            index_dim=self.index_dim,
            n_kinds=self.n_kinds(),
            log_joint=self.log_joint,
        )

    # ------------------------------------------------------------ conditioning
    def posterior_condition(self, theta_true: np.ndarray,
                            x_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Condition vector / mask for the posterior ``p(theta | x = x_obs)``."""
        value = np.concatenate([np.zeros_like(theta_true), x_obs])
        state = np.concatenate([np.zeros_like(theta_true), np.ones_like(x_obs)])
        return value, state
