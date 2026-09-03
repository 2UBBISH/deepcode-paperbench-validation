"""Random explanation baseline for RICE.

This module provides a random importance scorer that can be swapped into the
refining pipeline as a negative control.  It assigns a uniform random
criticality score to each state (or state-action pair), independent of the
policy or value function.
"""

from typing import Any, Callable, Optional, Union

import numpy as np
import torch


class RandomExplanation:
    """Random baseline explanation scorer.

    The scorer returns a random criticality score in ``[0, 1]`` for every
    input state.  It is stateless except for an optional random seed, so it
    can be used anywhere a mask network or other explanation method is
    expected.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        action: Optional[Union[np.ndarray, torch.Tensor]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return random criticality scores.

        Parameters
        ----------
        observation : np.ndarray or torch.Tensor
            Observations of shape ``(batch_size, obs_dim)`` or ``(obs_dim,)``.
        action : np.ndarray or torch.Tensor, optional
            Ignored; present only for API compatibility.

        Returns
        -------
        np.ndarray
            Random scores with shape ``(batch_size, 1)`` or ``(1,)``.
        """
        del action, kwargs  # unused
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().cpu().numpy()
        observation = np.asarray(observation)
        if observation.ndim == 1:
            return self.rng.random(size=(1,))
        batch_size = observation.shape[0]
        return self.rng.random(size=(batch_size, 1))

    def predict(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        **kwargs: Any,
    ) -> np.ndarray:
        """Alias for ``__call__`` returning only the score.

        This matches the ``MaskNetwork.predict`` interface so that the random
        baseline can be dropped into code that expects a mask network.
        """
        return self(observation, **kwargs)

    def state_dict(self) -> dict:
        """Return a serializable state dict (only the RNG state)."""
        return {"rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the RNG state."""
        self.rng.bit_generator.state = state_dict["rng_state"]


def random_explanation(
    observation: Union[np.ndarray, torch.Tensor],
    action: Optional[Union[np.ndarray, torch.Tensor]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Functional random explanation scorer.

    Parameters
    ----------
    observation : np.ndarray or torch.Tensor
        Observations of shape ``(batch_size, obs_dim)`` or ``(obs_dim,)``.
    action : np.ndarray or torch.Tensor, optional
        Ignored; present only for API compatibility.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Random criticality scores in ``[0, 1]``.
    """
    scorer = RandomExplanation(seed=seed)
    return scorer(observation, action=action)
