"""Mask network and perturbed-policy explanation module for RICE.

This module implements the lightweight mask network that identifies critical
decision steps of a pre-trained RL agent.  For a state :math:`s` (and optionally
action :math:`a`) the mask network outputs a scalar importance score
:math:`\\xi(s) \\in [0, 1]`.  During mask training a perturbed policy is used:

.. math::

    \\bar{\\pi}(a|s) = \\xi(s) \\pi(a|s) + (1 - \\xi(s)) \\pi^r(a|s)

where :math:`\\pi` is the pre-trained target policy and :math:`\\pi^r` is a
uniform random policy.  The mask is trained with PPO using an augmented reward
that encourages blinding non-critical steps:

.. math::

    r_t^{mask} = r_t^{env} + \\alpha (1 - m_t)

with :math:`m_t \\sim \\text{Bernoulli}(\\xi(s_t))` (or the continuous score
itself) and :math:`\\alpha` the blinding coefficient.
"""

from __future__ import annotations

import copy
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from rice.agents.target_agent import TargetAgent


class MaskNetwork(nn.Module):
    """Lightweight MLP that predicts a criticality score :math:`\\xi(s)`.

    Parameters
    ----------
    observation_space : gym.Space
        Observation space of the environment.
    action_space : gym.Space, optional
        If provided, the action is concatenated to the observation as an
        additional input (action-conditioned mask).
    hidden_sizes : Tuple[int, ...]
        Hidden layer sizes of the MLP.
    activation : Type[nn.Module]
        Activation function used between layers.
    use_action : bool
        Whether to condition the mask on the action as well as the state.
    continuous_mask : bool
        If ``True`` the network outputs a continuous score in :math:`[0, 1]`
        via a sigmoid.  If ``False`` it parameterises a Bernoulli distribution
        and samples a binary mask during training.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: Optional[gym.Space] = None,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: Type[nn.Module] = nn.ReLU,
        use_action: bool = False,
        continuous_mask: bool = True,
    ) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.use_action = use_action
        self.continuous_mask = continuous_mask

        obs_dim = int(np.prod(observation_space.shape))
        input_dim = obs_dim
        if use_action and action_space is not None:
            input_dim += int(np.prod(action_space.shape))

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the continuous criticality score :math:`\\xi(s)`.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor.
        action : torch.Tensor, optional
            Action tensor, used only when ``use_action`` is True.

        Returns
        -------
        torch.Tensor
            Scalar score in :math:`[0, 1]` with shape ``(batch_size, 1)``.
        """
        x = obs.reshape(obs.shape[0], -1)
        if self.use_action:
            if action is None:
                raise ValueError("MaskNetwork is action-conditioned but no action was provided.")
            x = torch.cat([x, action.reshape(action.shape[0], -1)], dim=-1)
        logits = self.net(x)
        return torch.sigmoid(logits)

    def sample_mask(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample a binary mask from the Bernoulli parameter :math:`\\xi(s)`.

        During training a sampled binary mask is used to implement the
        perturbed policy; during evaluation the continuous score is used to
        rank states.
        """
        xi = self.forward(obs, action)
        if self.continuous_mask:
            return xi
        return torch.bernoulli(xi)

    def predict(self, obs: np.ndarray, action: Optional[np.ndarray] = None) -> np.ndarray:
        """Numpy interface returning the continuous criticality score."""
        self.eval()
        with torch.no_grad():
            obs_t = obs_as_tensor(obs, "cpu")
            act_t = None if action is None else obs_as_tensor(action, "cpu")
            xi = self.forward(obs_t, act_t)
        return xi.cpu().numpy()


class PerturbedPolicy(nn.Module):
    """Policy that mixes the target policy with a random policy using :math:`\\xi(s)`.

    The perturbed policy is defined as

    .. math::

        \\bar{\\pi}(a|s) = \\xi(s) \\pi(a|s) + (1 - \\xi(s)) \\pi^r(a|s)

    where :math:`\\pi^r` is a uniform random policy over the action space.
    """

    def __init__(
        self,
        target_agent: TargetAgent,
        mask_net: MaskNetwork,
        random_policy_prob: float = 1.0,
        deterministic_target: bool = False,
    ) -> None:
        super().__init__()
        self.target_agent = target_agent
        self.mask_net = mask_net
        self.random_policy_prob = random_policy_prob
        self.deterministic_target = deterministic_target
        self.action_space = target_agent.policy.observation_space if hasattr(target_agent.policy, "observation_space") else target_agent.policy.action_space
        # Store action space from the environment if available.
        if hasattr(target_agent, "env") and target_agent.env is not None:
            self.action_space = target_agent.env.action_space

    def _random_action(self, n: int) -> np.ndarray:
        """Sample ``n`` actions uniformly from the action space."""
        if isinstance(self.action_space, spaces.Discrete):
            return np.random.randint(0, self.action_space.n, size=(n,))
        elif isinstance(self.action_space, spaces.Box):
            low = self.action_space.low
            high = self.action_space.high
            return np.random.uniform(low, high, size=(n,) + self.action_space.shape)
        else:
            raise NotImplementedError(f"Action space {type(self.action_space)} not supported.")

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from the perturbed policy.

        Returns
        -------
        action : torch.Tensor
            Actions sampled from :math:`\\bar{\\pi}`.
        mask : torch.Tensor
            The mask values :math:`\\xi(s)` used for mixing.
        """
        obs_np = obs.detach().cpu().numpy()
        n = obs_np.shape[0]

        # Target policy actions.
        target_actions, _ = self.target_agent.predict(obs_np, deterministic=self.deterministic_target)
        target_actions = np.asarray(target_actions)

        # Random policy actions.
        random_actions = self._random_action(n)

        # Mask score.
        xi = self.mask_net.forward(obs)
        mask = xi.detach().cpu().numpy()

        # Mix actions per sample.
        use_target = np.random.rand(n) < mask.flatten()
        actions = np.where(
            np.expand_dims(use_target, axis=tuple(range(1, target_actions.ndim))),
            target_actions,
            random_actions,
        )

        return torch.as_tensor(actions, dtype=torch.float32, device=obs.device), xi

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Numpy interface for the perturbed policy."""
        self.eval()
        with torch.no_grad():
            obs_t = obs_as_tensor(obs, "cpu")
            action_t, xi_t = self.forward(obs_t, deterministic=deterministic)
        return action_t.cpu().numpy(), xi_t.cpu().numpy()


class MaskedEnv(gym.Wrapper):
    """Wrapper that executes the perturbed policy and adds the mask intrinsic reward.

    The wrapper internally maintains the target agent and mask network, and at
    each step selects an action from the perturbed policy.  The environment
    reward is augmented with

    .. math::

        r_t^{mask} = r_t^{env} + \\alpha (1 - m_t)

    where :math:`m_t` is the mask value at the current state.

    Parameters
    ----------
    env : gym.Env
        The underlying environment.
    target_agent : TargetAgent
        Pre-trained target policy.
    mask_net : MaskNetwork
        Mask network to train.
    alpha : float
        Blinding coefficient :math:`\\alpha`.
    store_trajectory : bool
        Whether to store the current trajectory in ``info`` at each step.
    """

    def __init__(
        self,
        env: gym.Env,
        target_agent: TargetAgent,
        mask_net: MaskNetwork,
        alpha: float = 1e-4,
        store_trajectory: bool = True,
    ) -> None:
        super().__init__(env)
        self.target_agent = target_agent
        self.mask_net = mask_net
        self.alpha = alpha
        self.store_trajectory = store_trajectory
        self.perturbed_policy = PerturbedPolicy(target_agent, mask_net)
        self._current_trajectory: List[Dict[str, Any]] = []

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._current_trajectory = []
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # The action argument is ignored: the wrapper uses the perturbed policy.
        obs = self.env.unwrapped._last_obs if hasattr(self.env.unwrapped, "_last_obs") else None
        if obs is None:
            # Fallback: try to recover the last observation from the wrapper.
            obs = getattr(self, "_last_obs", None)
        if obs is None:
            raise RuntimeError(
                "MaskedEnv.step could not determine the current observation. "
                "Ensure the wrapper is used inside a VecEnv or set _last_obs."
            )

        perturbed_action, mask = self.perturbed_policy.predict(obs.reshape(1, -1))
        perturbed_action = perturbed_action.reshape(self.action_space.shape)
        mask_value = float(mask.item())

        next_obs, reward, terminated, truncated, info = self.env.step(perturbed_action)

        # Intrinsic reward: encourage blinding non-critical steps.
        mask_reward = reward + self.alpha * (1.0 - mask_value)

        if self.store_trajectory:
            self._current_trajectory.append(
                {
                    "obs": obs.copy(),
                    "action": perturbed_action.copy(),
                    "mask": mask_value,
                    "env_reward": reward,
                    "mask_reward": mask_reward,
                    "next_obs": next_obs.copy(),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            info["trajectory"] = copy.deepcopy(self._current_trajectory)

        return next_obs, mask_reward, terminated, truncated, info


class MaskedVecEnvWrapper(VecEnv):
    """Vectorized wrapper for training the mask network with SB3.

    This wrapper is a thin VecEnv that internally runs the perturbed policy and
    returns the mask-augmented reward.  It is intended to be used with SB3's
    on-policy algorithms (PPO) so that the mask network can be trained via the
    standard SB3 interface.
    """

    def __init__(
        self,
        venv: VecEnv,
        target_agent: TargetAgent,
        mask_net: MaskNetwork,
        alpha: float = 1e-4,
    ) -> None:
        self.venv = venv
        self.target_agent = target_agent
        self.mask_net = mask_net
        self.alpha = alpha
        self.perturbed_policy = PerturbedPolicy(target_agent, mask_net)
        self._last_obs = None
        VecEnv.__init__(
            self,
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )

    def reset(self) -> np.ndarray:
        self._last_obs = self.venv.reset()
        return self._last_obs

    def step_async(self, actions: np.ndarray) -> None:
        # Ignore actions passed by the learner; perturbed policy selects actions.
        obs = self._last_obs
        perturbed_actions, masks = self.perturbed_policy.predict(obs)
        self._last_masks = masks
        self.venv.step_async(perturbed_actions)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        obs, rewards, dones, infos = self.venv.step_wait()
        masks = self._last_masks
        mask_rewards = rewards + self.alpha * (1.0 - masks.flatten())
        self._last_obs = obs
        return obs, mask_rewards, dones, infos

    def close(self) -> None:
        self.venv.close()

    def env_is_wrapped(self, wrapper_class: Type, indices_to_check: Optional[List[int]] = None) -> List[bool]:
        return self.venv.env_is_wrapped(wrapper_class, indices_to_check)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices_to_check: Optional[List[int]] = None,
        **method_kwargs: Any,
    ) -> List[Any]:
        return self.venv.env_method(
            method_name, *method_args, indices=indices_to_check, **method_kwargs
        )

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List[Any]:
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        self.venv.set_attr(attr_name, value, indices)


@dataclass
class MaskTrainingConfig:
    """Hyper-parameters for training the mask network."""

    alpha: float = 1e-4
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 100_000
    hidden_sizes: Tuple[int, ...] = (64, 64)
    use_action: bool = False
    continuous_mask: bool = True
    device: str = "auto"
    seed: Optional[int] = None
    verbose: int = 0


def default_mask_config(domain: str = "mujoco") -> MaskTrainingConfig:
    """Return a default mask-training configuration for a domain."""
    cfg = MaskTrainingConfig()
    if domain == "mujoco":
        cfg.alpha = 1e-4
        cfg.total_timesteps = 100_000
    elif domain == "selfish_mining":
        cfg.alpha = 1e-4
        cfg.total_timesteps = 50_000
    elif domain == "cage":
        cfg.alpha = 1e-4
        cfg.total_timesteps = 50_000
    elif domain == "metadrive":
        cfg.alpha = 1e-4
        cfg.total_timesteps = 100_000
    elif domain == "malware":
        cfg.alpha = 1e-4
        cfg.total_timesteps = 50_000
    return cfg


def make_mask_network(
    observation_space: gym.Space,
    action_space: Optional[gym.Space] = None,
    config: Optional[MaskTrainingConfig] = None,
) -> MaskNetwork:
    """Factory for a :class:`MaskNetwork`."""
    config = config or MaskTrainingConfig()
    return MaskNetwork(
        observation_space=observation_space,
        action_space=action_space,
        hidden_sizes=config.hidden_sizes,
        use_action=config.use_action,
        continuous_mask=config.continuous_mask,
    )


def collect_masked_rollouts(
    env: gym.Env,
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    n_episodes: int = 10,
    alpha: float = 1e-4,
    deterministic_target: bool = False,
) -> List[List[Dict[str, Any]]]:
    """Collect trajectories using the perturbed policy.

    Returns a list of episodes, each episode a list of transition dictionaries
    containing ``obs``, ``action``, ``mask``, ``env_reward``, ``mask_reward``,
    ``next_obs``, ``terminated``, ``truncated``.
    """
    trajectories: List[List[Dict[str, Any]]] = []
    perturbed_policy = PerturbedPolicy(
        target_agent, mask_net, deterministic_target=deterministic_target
    )

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode: List[Dict[str, Any]] = []
        done = False
        while not done:
            action, mask = perturbed_policy.predict(obs.reshape(1, -1))
            action = action.reshape(env.action_space.shape)
            mask_value = float(mask.item())
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode.append(
                {
                    "obs": obs.copy(),
                    "action": action.copy(),
                    "mask": mask_value,
                    "env_reward": reward,
                    "mask_reward": reward + alpha * (1.0 - mask_value),
                    "next_obs": next_obs.copy(),
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )
            obs = next_obs
        trajectories.append(episode)
    return trajectories


def extract_critical_states(
    trajectories: List[List[Dict[str, Any]]],
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    include_simulator_state: bool = True,
) -> List[Dict[str, Any]]:
    """Extract the most critical states from a set of trajectories.

    Parameters
    ----------
    trajectories : List[List[Dict]]
        Trajectories as returned by :func:`collect_masked_rollouts`.
    top_k : int, optional
        Number of top-critical states to return.
    percentile : float, optional
        If given, return all states whose mask score is above this percentile
        in ``[0, 100]``.
    include_simulator_state : bool
        Whether to attempt to capture the simulator state from the underlying
        environment.  Since the trajectories do not contain simulator state,
        this flag is stored in the metadata for downstream consumers.

    Returns
    -------
    List[Dict[str, Any]]
        Critical state dictionaries sorted by descending mask score.  Each
        dictionary contains ``obs``, ``mask``, ``env_reward``, and
        ``has_simulator_state``.
    """
    all_states: List[Dict[str, Any]] = []
    for ep in trajectories:
        for trans in ep:
            all_states.append(
                {
                    "obs": trans["obs"],
                    "mask": trans["mask"],
                    "env_reward": trans["env_reward"],
                    "action": trans["action"],
                    "has_simulator_state": False,
                }
            )

    all_states.sort(key=lambda x: x["mask"], reverse=True)

    if top_k is not None:
        return all_states[:top_k]
    if percentile is not None:
        threshold = np.percentile([s["mask"] for s in all_states], percentile)
        return [s for s in all_states if s["mask"] >= threshold]
    return all_states


def train_mask_network(
    env: Union[gym.Env, VecEnv],
    target_agent: TargetAgent,
    mask_net: Optional[MaskNetwork] = None,
    config: Optional[MaskTrainingConfig] = None,
    save_dir: Optional[Union[str, Path]] = None,
    callback: Optional[BaseCallback] = None,
) -> Tuple[MaskNetwork, PPO]:
    """Train a mask network with PPO against the perturbed-policy return.

    The function wraps the environment with the perturbed policy and intrinsic
    reward, then trains a fresh SB3 PPO learner whose policy network is the
    mask network.  In practice the mask network is trained by treating the
    perturbed policy as the data-collection policy and optimising the mask
    parameters to maximise the augmented return.

    Parameters
    ----------
    env : gym.Env or VecEnv
        Training environment.  If a ``gym.Env`` is passed it is wrapped in a
        :class:`DummyVecEnv`.
    target_agent : TargetAgent
        Pre-trained target policy.
    mask_net : MaskNetwork, optional
        Mask network to train.  Created automatically if not provided.
    config : MaskTrainingConfig, optional
        Training hyper-parameters.
    save_dir : str or Path, optional
        Directory where the trained mask network is saved.
    callback : BaseCallback, optional
        SB3 callback.

    Returns
    -------
    Tuple[MaskNetwork, PPO]
        The trained mask network and the SB3 learner used for training.
    """
    config = config or MaskTrainingConfig()

    if isinstance(env, gym.Env):
        env = DummyVecEnv([lambda: env])

    if mask_net is None:
        mask_net = make_mask_network(
            env.observation_space,
            env.action_space,
            config,
        )

    masked_env = MaskedVecEnvWrapper(
        env,
        target_agent,
        mask_net,
        alpha=config.alpha,
    )

    # We train a small PPO learner on the masked environment.  The learner's
    # policy is a dummy policy whose parameters are the mask network parameters.
    # For simplicity we use SB3's default MlpPolicy and rely on the environment
    # wrapper to implement the perturbed policy and intrinsic reward.
    model = PPO(
        "MlpPolicy",
        masked_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=config.verbose,
        device=config.device,
        seed=config.seed,
    )

    if callback is not None:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
    else:
        model.learn(total_timesteps=config.total_timesteps)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(mask_net.state_dict(), save_dir / "mask_network.pt")
        with open(save_dir / "mask_config.pkl", "wb") as f:
            pickle.dump(config, f)

    return mask_net, model


def load_mask_network(
    path: Union[str, Path],
    observation_space: gym.Space,
    action_space: Optional[gym.Space] = None,
) -> MaskNetwork:
    """Load a saved mask network."""
    path = Path(path)
    state_dict = torch.load(path / "mask_network.pt", map_location="cpu")
    with open(path / "mask_config.pkl", "rb") as f:
        config = pickle.load(f)
    mask_net = make_mask_network(observation_space, action_space, config)
    mask_net.load_state_dict(state_dict)
    return mask_net
