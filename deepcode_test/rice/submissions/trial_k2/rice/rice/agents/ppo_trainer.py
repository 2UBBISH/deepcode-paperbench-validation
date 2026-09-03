"""Generic PPO training loop for RICE.

This module implements a reusable PPO trainer that works with the
``BaseTargetPolicy`` interface defined in ``target_policy.py``.  It is used to
train (1) the initial target policies for tasks that do not rely on
Stable-Baselines3, (2) the MaskNet explanation module, and (3) refined policies
during the RICE refinement stage.

The implementation follows the canonical PPO-clip algorithm with Generalized
Advantage Estimation (GAE) and supports both discrete and continuous action
spaces.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal

from .target_policy import BaseTargetPolicy, TorchTargetPolicy


@dataclass
class PPOConfig:
    """Hyper-parameters for the generic PPO trainer.

    Defaults are chosen to match Stable-Baselines3 PPO defaults where possible,
    so that the custom loop can be used as a drop-in replacement for SB3 when
    needed (e.g. for the malware environment or for fine-grained control during
    refinement).
    """

    # Rollout / update schedule
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64

    # PPO objective coefficients
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5

    # Logging / misc
    normalize_advantage: bool = True
    device: Union[str, torch.device] = "auto"
    seed: Optional[int] = None


class RolloutBuffer:
    """Simple rollout buffer for on-policy PPO updates.

    Stores observations, actions, log-probabilities, rewards, values, dones and
    the resulting advantages / returns.  The buffer is filled during a rollout
    and then consumed in mini-batches for ``n_epochs`` update epochs.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        action_is_discrete: bool,
        device: Union[str, torch.device] = "cpu",
    ):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_is_discrete = action_is_discrete
        self.device = device

        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        if action_is_discrete:
            self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.int64)
        else:
            self.actions = np.zeros((buffer_size,) + action_shape, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        value: float,
        log_prob: float,
    ) -> None:
        """Add a single transition to the buffer."""
        if self.full:
            raise RuntimeError("RolloutBuffer is full. Call compute_returns_and_advantage then clear.")
        self.observations[self.pos] = np.asarray(obs, dtype=np.float32)
        self.actions[self.pos] = np.asarray(action)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.values[self.pos] = float(value)
        self.log_probs[self.pos] = float(log_prob)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(
        self, last_value: float, last_done: bool
    ) -> None:
        """Compute GAE advantages and discounted returns in-place."""
        last_gae_lam = 0.0
        last_done = float(last_done)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_done
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self.config_gamma * next_value * next_non_terminal
                - self.values[step]
            )
            last_gae_lam = (
                delta
                + self.config_gamma
                * self.config_gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
            self.advantages[step] = last_gae_lam
            self.returns[step] = self.advantages[step] + self.values[step]

    def get(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Yield mini-batches of transitions for PPO updates."""
        indices = np.random.permutation(self.buffer_size)
        start_idx = 0
        batches = []
        while start_idx < self.buffer_size:
            end_idx = min(start_idx + batch_size, self.buffer_size)
            batch_indices = indices[start_idx:end_idx]
            batch = {
                "obs": torch.as_tensor(
                    self.observations[batch_indices], device=self.device
                ),
                "actions": torch.as_tensor(
                    self.actions[batch_indices], device=self.device
                ),
                "old_log_prob": torch.as_tensor(
                    self.log_probs[batch_indices], device=self.device
                ),
                "advantages": torch.as_tensor(
                    self.advantages[batch_indices], device=self.device
                ),
                "returns": torch.as_tensor(
                    self.returns[batch_indices], device=self.device
                ),
                "values": torch.as_tensor(
                    self.values[batch_indices], device=self.device
                ),
            }
            batches.append(batch)
            start_idx = end_idx
        return batches

    def clear(self) -> None:
        """Reset the buffer after an update."""
        self.pos = 0
        self.full = False


class PPOTrainer:
    """Generic PPO trainer built on top of ``BaseTargetPolicy``.

    The trainer handles environment interaction, advantage computation and the
    clipped surrogate PPO update loop.  It is intentionally lightweight so that
    it can be reused for target-policy training, MaskNet training and the
    refinement stage.

    Parameters
    ----------
    policy:
        The policy to train.  Must expose ``predict``, ``evaluate_actions`` and
        ``get_value`` as defined by ``BaseTargetPolicy``.
    env:
        A vectorized or single Gym/Gymnasium environment.  Must follow the
        standard ``reset`` / ``step`` API and expose ``observation_space`` and
        ``action_space``.
    config:
        ``PPOConfig`` instance with hyper-parameters.
    """

    def __init__(
        self,
        policy: BaseTargetPolicy,
        env: Any,
        config: Optional[PPOConfig] = None,
    ):
        self.policy = policy
        self.env = env
        self.config = config or PPOConfig()

        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Move policy to the requested device if it is a torch-backed policy.
        if isinstance(self.policy, TorchTargetPolicy):
            self.policy.model.to(self.device)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Determine action space type and shapes.
        self.discrete_actions = not hasattr(self.action_space, "sample")
        # The line above is a heuristic; use gymnasium API when available.
        try:
            from gymnasium import spaces as gym_spaces
        except ImportError:  # pragma: no cover
            import gym.spaces as gym_spaces  # type: ignore

        self.discrete_actions = isinstance(self.action_space, gym_spaces.Discrete)

        obs_shape = tuple(self.observation_space.shape)
        if self.discrete_actions:
            action_shape = ()
        else:
            action_shape = tuple(self.action_space.shape)

        self.buffer = RolloutBuffer(
            buffer_size=self.config.n_steps,
            obs_shape=obs_shape,
            action_shape=action_shape,
            action_is_discrete=self.discrete_actions,
            device=self.device,
        )
        # Attach config values needed inside the buffer for GAE computation.
        self.buffer.config_gamma = self.config.gamma
        self.buffer.config_gae_lambda = self.config.gae_lambda

        # Optimizer operates on the underlying torch model.
        if isinstance(self.policy, TorchTargetPolicy):
            self.optimizer = optim.Adam(
                self.policy.model.parameters(), lr=self.config.learning_rate
            )
        else:
            raise ValueError(
                "PPOTrainer currently only supports TorchTargetPolicy. "
                "For SB3 models use Stable-Baselines3's built-in PPO."
            )

        self.num_timesteps = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def _to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def collect_rollouts(self) -> Tuple[float, int, float]:
        """Collect a single rollout of ``n_steps`` transitions.

        Returns
        -------
        mean_reward:
            Average reward per step in the collected rollout.
        num_episodes:
            Number of completed episodes during the rollout.
        explained_var:
            Explained variance of the value function on the collected data.
        """
        if not hasattr(self, "_last_obs"):
            self._last_obs = self.env.reset()
            if isinstance(self._last_obs, tuple):
                self._last_obs = self._last_obs[0]

        self.buffer.clear()
        ep_rewards: List[float] = []
        ep_lengths: List[int] = []
        current_ep_reward = 0.0
        current_ep_length = 0
        num_episodes = 0

        for step in range(self.config.n_steps):
            obs_tensor = self._to_tensor(self._last_obs)
            with torch.no_grad():
                action, log_prob, value = self.policy.predict(
                    obs_tensor, deterministic=False
                )

            # Convert action to numpy for the environment.
            np_action = action.cpu().numpy()
            if self.discrete_actions:
                np_action = int(np_action.item())

            step_result = self.env.step(np_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            self.buffer.add(
                obs=self._last_obs,
                action=np_action if not self.discrete_actions else np.array([np_action]),
                reward=float(reward),
                done=bool(done),
                value=float(value.cpu().numpy()),
                log_prob=float(log_prob.cpu().numpy()),
            )

            self._last_obs = obs
            self.num_timesteps += 1
            current_ep_reward += float(reward)
            current_ep_length += 1

            if done:
                ep_rewards.append(current_ep_reward)
                ep_lengths.append(current_ep_length)
                current_ep_reward = 0.0
                current_ep_length = 0
                num_episodes += 1
                self._last_obs = self.env.reset()
                if isinstance(self._last_obs, tuple):
                    self._last_obs = self._last_obs[0]

        # Bootstrap value for the last observation.
        with torch.no_grad():
            last_value = self.policy.get_value(self._to_tensor(self._last_obs))
            last_value = float(last_value.cpu().numpy())

        self.buffer.compute_returns_and_advantage(
            last_value=last_value, last_done=False
        )

        # Logging helpers.
        mean_reward = float(np.mean(self.buffer.rewards))
        explained_var = (
            1.0
            - np.var(self.buffer.returns - self.buffer.values)
            / (np.var(self.buffer.returns) + 1e-8)
        )

        if ep_rewards:
            self.episode_rewards.extend(ep_rewards)
            self.episode_lengths.extend(ep_lengths)

        return mean_reward, num_episodes, explained_var

    def update_policy(self) -> Dict[str, float]:
        """Perform ``n_epochs`` PPO-clip updates on the collected rollout.

        Returns
        -------
        A dictionary with average policy loss, value loss, entropy loss and
        approximate KL divergence over the update.
        """
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kls = []

        clip_range = self.config.clip_range

        for epoch in range(self.config.n_epochs):
            batches = self.buffer.get(self.config.batch_size)
            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_prob = batch["old_log_prob"]
                advantages = batch["advantages"]
                returns = batch["returns"]
                old_values = batch["values"]

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Evaluate actions with the current policy.
                log_prob, entropy, value = self.policy.evaluate_actions(obs, actions)

                # Policy loss (PPO-clip).
                ratio = torch.exp(log_prob - old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss (clipped).
                value_pred_clipped = old_values + torch.clamp(
                    value - old_values, -clip_range, clip_range
                )
                value_loss_1 = nn.functional.mse_loss(value, returns)
                value_loss_2 = nn.functional.mse_loss(value_pred_clipped, returns)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                # Entropy bonus.
                if entropy is None:
                    entropy_loss = torch.tensor(0.0, device=self.device)
                else:
                    entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    + self.config.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                with torch.no_grad():
                    kl = (old_log_prob - log_prob).mean().item()
                    kls.append(kl)

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
            "approx_kl": float(np.mean(kls)),
        }

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        save_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the policy for ``total_timesteps`` environment steps.

        Parameters
        ----------
        total_timesteps:
            Total number of environment steps to train for.
        log_interval:
            Print training statistics every ``log_interval`` iterations.
        save_path:
            Optional path to save intermediate model checkpoints.
        save_interval:
            Save a checkpoint every ``save_interval`` iterations (only used if
            ``save_path`` is provided).

        Returns
        -------
        A dictionary containing the final training statistics.
        """
        iterations = 0
        while self.num_timesteps < total_timesteps:
            mean_reward, num_episodes, explained_var = self.collect_rollouts()
            update_info = self.update_policy()
            iterations += 1

            recent_rewards = self.episode_rewards[-100:]
            recent_lengths = self.episode_lengths[-100:]
            mean_ep_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            mean_ep_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0

            if log_interval > 0 and iterations % log_interval == 0:
                print(
                    f"Iter {iterations:4d} | steps {self.num_timesteps:7d} | "
                    f"ep_reward {mean_ep_reward:8.2f} | ep_length {mean_ep_length:6.1f} | "
                    f"policy_loss {update_info['policy_loss']:8.4f} | "
                    f"value_loss {update_info['value_loss']:8.4f} | "
                    f"explained_var {explained_var:6.3f}"
                )

            if save_path is not None and save_interval is not None:
                if iterations % save_interval == 0:
                    self.save(save_path)

        return {
            "iterations": iterations,
            "num_timesteps": self.num_timesteps,
            "mean_episode_reward": float(np.mean(self.episode_rewards[-100:])),
            "mean_episode_length": float(np.mean(self.episode_lengths[-100:])),
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save the policy and optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "num_timesteps": self.num_timesteps,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load the policy and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.model.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_timesteps = checkpoint.get("num_timesteps", 0)
