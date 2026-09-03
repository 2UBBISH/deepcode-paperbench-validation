"""Intrinsic blinding bonus for the RICE mask network.

The mask network is trained to maximize the return of a perturbed policy
:math:`\\bar{\\pi}` while receiving an auxiliary reward that encourages
sparsity of the mask:

.. math::
    r^{\\text{mask}}_t = r^{\\text{env}}_t + \\alpha \\big(1 - \\xi(s_t)\\big),

where :math:`\\xi(s_t) \\in (0, 1)` is the mask probability and
:math:`\\alpha` controls the trade-off between explanation fidelity and
sparsity.
"""

from typing import Union

import numpy as np
import torch


class MaskIntrinsicReward:
    """Computes the mask-network training reward.

    Parameters
    ----------
    alpha : float
        Blinding coefficient. Larger values encourage a sparser mask by
        penalizing high :math:`\\xi(s)` values.
    """

    def __init__(self, alpha: float = 1e-4):
        self.alpha = float(alpha)

    def __call__(
        self,
        env_reward: Union[float, np.ndarray, torch.Tensor],
        xi: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Return the mask reward for a single step or a batch.

        Parameters
        ----------
        env_reward : float or array-like
            Original environment reward.
        xi : float or array-like
            Mask probability :math:`\\xi(s)`. Must be in ``[0, 1]``.

        Returns
        -------
        float or array-like
            ``env_reward + alpha * (1 - xi)`` with the same type/shape as
            ``env_reward``.
        """
        bonus = self.alpha * (1.0 - xi)
        return env_reward + bonus

    def bonus(
        self,
        xi: Union[float, np.ndarray, torch.Tensor],
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """Return only the blinding bonus term ``alpha * (1 - xi)``.

        Parameters
        ----------
        xi : float or array-like
            Mask probability.

        Returns
        -------
        float or array-like
            The blinding bonus.
        """
        return self.alpha * (1.0 - xi)


def mask_reward(
    env_reward: Union[float, np.ndarray, torch.Tensor],
    xi: Union[float, np.ndarray, torch.Tensor],
    alpha: float = 1e-4,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Functional form of the mask-network intrinsic reward.

    Parameters
    ----------
    env_reward : float or array-like
        Original environment reward.
    xi : float or array-like
        Mask probability :math:`\\xi(s)`.
    alpha : float, optional
        Blinding coefficient (default: 1e-4).

    Returns
    -------
    float or array-like
        ``env_reward + alpha * (1 - xi)``.
    """
    return env_reward + alpha * (1.0 - xi)
