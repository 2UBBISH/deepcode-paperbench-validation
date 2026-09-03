"""Masked environment wrapper for training the RICE MaskNet.

The wrapper turns the original task into a binary-decision problem for the mask
network. At each step the mask selects :math:`a^e \in \{0, 1\}`:

* ``0`` – execute the frozen target policy's action (the step is treated as
  critical).
* ``1`` – execute a random action from the original action space (the step is
  treated as non-critical).

The reward returned to the mask is the original environment reward plus the
intrinsic blinding bonus:

.. math::
    r_{mask}(s_t) = r_{env}(s_t, a_t) + \alpha (1 - \xi(s_t)),

where :math:`\xi(s_t)` is the mask-network output (probability of marking the
step as critical).
"""

from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

try:
    from ..agents.target_policy import BaseTargetPolicy
    from .intrinsic_reward import mask_reward
    from .mask_network import MaskNetwork
except ImportError:  # pragma: no cover
    from rice.agents.target_policy import BaseTargetPolicy
    from rice.masknet.intrinsic_reward import mask_reward
    from rice.masknet.mask_network import MaskNetwork


class MaskedEnv(gym.Wrapper):
    """Wrap an environment so the agent is the mask network.

    The wrapper's action space is ``Discrete(2)`` (mask decision). The original
    action is chosen either from the frozen target policy or uniformly at random,
    and the returned reward is the mask-training reward.

    Parameters
    ----------
    env : gym.Env
        Original task environment.
    target_policy : BaseTargetPolicy
        Frozen target policy :math:`\\pi` whose actions are executed when the
        mask decides the step is critical.
    mask_network : MaskNetwork
        Mask network :math:`\\xi` that outputs the critical-step probability.
    alpha : float, optional
        Blinding bonus coefficient :math:`\\alpha` (default 1e-4).
    device : str or torch.device, optional
        Device on which to run the mask network (default "cpu").
    """

    def __init__(
        self,
        env: gym.Env,
        target_policy: BaseTargetPolicy,
        mask_network: MaskNetwork,
        alpha: float = 1e-4,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__(env)
        self.target_policy = target_policy
        self.mask_network = mask_network
        self.alpha = float(alpha)
        self.device = torch.device(device) if device != "auto" else torch.device("cpu")

        # The mask network makes a binary decision.
        self.action_space = spaces.Discrete(2)
        self.observation_space = env.observation_space

        self._last_obs: np.ndarray = None

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the underlying environment and store the observation."""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return self._last_obs, info

    def step(
        self, mask_action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one masked step.

        Parameters
        ----------
        mask_action : int or ndarray
            ``0`` uses the target policy action; ``1`` uses a random action.

        Returns
        -------
        next_obs : ndarray
        mask_reward : float
        terminated : bool
        truncated : bool
        info : dict
            Contains ``env_reward``, ``xi``, ``mask_action``, ``env_action``.
        """
        mask_action = int(np.asarray(mask_action).reshape(-1)[0])
        obs = self._last_obs

        # Critical-step probability xi(s).
        xi = self._get_xi(obs)

        # Choose the action that is actually sent to the original environment.
        if mask_action == 0:
            env_action, _ = self.target_policy.predict(obs, deterministic=False)
        else:
            env_action = self.env.action_space.sample()
        env_action = self._to_numpy(env_action)

        # Step the underlying environment, normalising the return signature.
        step_result = self.env.step(env_action)
        if len(step_result) == 5:
            next_obs, env_reward, terminated, truncated, info = step_result
        else:
            next_obs, env_reward, done, info = step_result
            terminated = bool(done)
            truncated = False

        next_obs = np.asarray(next_obs, dtype=np.float32)
        self._last_obs = next_obs

        # Mask-training reward: r_env + alpha * (1 - xi).
        reward = mask_reward(env_reward, xi, alpha=self.alpha)

        info = dict(info)
        info["env_reward"] = float(env_reward)
        info["mask_reward"] = float(reward)
        info["xi"] = float(xi)
        info["mask_action"] = mask_action
        info["env_action"] = env_action

        return next_obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_xi(self, obs: np.ndarray) -> float:
        """Compute :math:`\\xi(s)` as a Python float."""
        obs_t = self._prepare_obs(obs)
        with torch.no_grad():
            xi = self.mask_network.predict(obs_t)
        # Robustly convert any tensor/array shape to a scalar.
        return float(np.asarray(xi).reshape(-1)[0])

    def _prepare_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Convert observation to a batched tensor on the correct device."""
        obs = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs).to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    @staticmethod
    def _to_numpy(action: Any) -> np.ndarray:
        """Convert an action returned by a policy to a NumPy array."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.asarray(action, dtype=np.float32)
        # Remove a leading batch dimension of size 1 if present.
        if action.ndim >= 1 and action.shape[0] == 1:
            action = action.squeeze(0)
        return action
