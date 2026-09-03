"""AIRS explanation stub for the RICE refining pipeline.

AIRS (Attribution-based Interpretability for Reinforcement Learning) is an
alternative explanation method discussed in the paper.  Because the official
AIRS implementation is not part of this reproduction, this module provides a
minimal *stub* scorer that exposes the same interface as :class:`MaskNetwork`
and the other explanation baselines.  It can be swapped into the refining
pipeline so that the ablation "Explanation source" runs end-to-end.

The stub returns a constant criticality score by default.  Setting
``randomize=True`` makes it behave like a random baseline, which is useful for
sanity-checking the pipeline when AIRS itself is not available.
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch


class AIRSStub:
    """Placeholder AIRS explanation scorer.

    Parameters
    ----------
    score : float, optional
        Constant criticality score returned for every observation.  Must be in
        ``[0, 1]``.  Default is ``0.5``.
    randomize : bool, optional
        If ``True``, ignore ``score`` and return uniform random scores in
        ``[0, 1]``.  Default is ``False``.
    seed : int, optional
        Seed for the internal RNG when ``randomize=True``.
    """

    def __init__(
        self,
        score: float = 0.5,
        randomize: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be in [0, 1]")
        self.score = float(score)
        self.randomize = bool(randomize)
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        action: Optional[Union[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """Return criticality scores for ``observation``.

        Parameters
        ----------
        observation : np.ndarray or torch.Tensor
            Single observation of shape ``(obs_dim,)`` or a batch of shape
            ``(batch_size, obs_dim)``.
        action : int or np.ndarray, optional
            Ignored.  Present only for API compatibility.

        Returns
        -------
        np.ndarray
            Criticality scores of shape ``(1,)`` for a single input or
            ``(batch_size, 1)`` for a batch.
        """
        return self.predict(observation)

    def predict(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> np.ndarray:
        """Predict AIRS criticality scores.

        Parameters
        ----------
        observation : np.ndarray or torch.Tensor
            Observation(s) to score.
        deterministic : bool, optional
            Ignored.  Present for API compatibility with ``MaskNetwork``.

        Returns
        -------
        np.ndarray
            Criticality scores with a trailing dimension of size 1.
        """
        if isinstance(observation, torch.Tensor):
            obs = observation.detach().cpu().numpy()
        else:
            obs = np.asarray(observation)

        if obs.ndim == 1:
            batch_size = 1
        else:
            batch_size = obs.shape[0]

        if self.randomize:
            scores = self._rng.random(batch_size).astype(np.float32)
        else:
            scores = np.full(batch_size, self.score, dtype=np.float32)

        return scores.reshape(-1, 1) if batch_size > 1 else scores.reshape(1)

    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable state dictionary."""
        return {
            "score": self.score,
            "randomize": self.randomize,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state from ``state_dict``."""
        self.score = float(state_dict.get("score", 0.5))
        self.randomize = bool(state_dict.get("randomize", False))
        rng_state = state_dict.get("rng_state")
        if rng_state is not None:
            self._rng.bit_generator.state = rng_state


def airs_explanation(
    observation: Union[np.ndarray, torch.Tensor],
    action: Optional[Union[int, np.ndarray]] = None,
    score: float = 0.5,
    randomize: bool = False,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Functional convenience wrapper for :class:`AIRSStub`.

    Parameters
    ----------
    observation : np.ndarray or torch.Tensor
        Observation(s) to score.
    action : int or np.ndarray, optional
        Ignored.  Present for API compatibility.
    score : float, optional
        Constant score returned when ``randomize=False``.  Default ``0.5``.
    randomize : bool, optional
        Return uniform random scores instead.  Default ``False``.
    seed : int, optional
        RNG seed for the random case.

    Returns
    -------
    np.ndarray
        Criticality scores.
    """
    scorer = AIRSStub(score=score, randomize=randomize, seed=seed)
    return scorer.predict(observation)
