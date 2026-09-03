"""Training loop and critical-state extraction for the RICE MaskNet.

The MaskNet is trained with PPO on the :class:`MaskedEnv` wrapper, where the
action space is binary (execute target action vs. randomize action). After
convergence, the trainer rolls out the frozen target policy, scores each visited
state with the learned :math:`\\xi(s)`, and returns the highest-scoring states
as the critical-state buffer used by the refinement module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rice.agents import PPOConfig, PPOTrainer, TorchTargetPolicy
from rice.agents.target_policy import BaseTargetPolicy

from .intrinsic_reward import MaskIntrinsicReward
from .mask_network import MaskNetwork, build_mask_network, match_target_mask_network
from .masked_env import MaskedEnv


class MaskActorCritic(nn.Module):
    """Actor-critic network for MaskNet training.

    The *actor* is a :class:`MaskNetwork` that outputs :math:`\\xi(s)`. The
    binary mask action distribution is defined as
    :math:`P(a=0|s)=\\xi(s)` (execute target action) and
    :math:`P(a=1|s)=1-\\xi(s)` (randomize action). A separate MLP *critic*
    estimates the state-value function required by PPO.
    """

    def __init__(
        self,
        mask_net: MaskNetwork,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: type = nn.Tanh,
    ) -> None:
        super().__init__()
        self.mask_net = mask_net
        obs_dim = mask_net.obs_dim
        layers: List[nn.Module] = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xi = self.mask_net(obs)
        value = self.critic(obs)
        return xi, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample/evaluate a mask action and return the value estimate."""
        xi = self.mask_net(obs)
        # xi shape: (batch, 1). Build categorical probs [P(a=0), P(a=1)].
        probs = torch.cat([xi, 1.0 - xi], dim=-1)
        # Clamp for numerical stability.
        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        dist = Categorical(probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs)
        return action, log_prob, entropy, value


class MaskTorchPolicy(TorchTargetPolicy):
    """Frozen target-policy interface for a MaskNet actor-critic.

    This wrapper exposes ``predict``, ``evaluate_actions`` and ``get_value`` so
    that the generic :class:`PPOTrainer` can optimize the mask network.
    """

    def __init__(
        self,
        model: MaskActorCritic,
        observation_space,
        action_space,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        # TorchTargetPolicy expects a model with get_action_and_value, get_value.
        super().__init__(model, observation_space, action_space, device=device)

    def predict(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        obs_t = self._to_tensor(observation)
        with torch.no_grad():
            action, _, _, _ = self.model.get_action_and_value(
                obs_t, deterministic=deterministic
            )
        return action.cpu().numpy(), {}

    def evaluate_actions(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_t = self._to_tensor(observation)
        act_t = self._to_tensor(actions)
        if act_t.dim() > 1:
            act_t = act_t.squeeze(-1)
        _, log_prob, entropy, value = self.model.get_action_and_value(
            obs_t, action=act_t
        )
        return log_prob, value.squeeze(-1), entropy

    def get_value(
        self, observation: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        obs_t = self._to_tensor(observation)
        return self.model.get_value(obs_t).squeeze(-1)


class MaskTrainer:
    """Train a MaskNet and extract critical states for RICE refinement.

    Parameters
    ----------
    env :
        The original task environment (will be wrapped with :class:`MaskedEnv`).
    target_policy :
        The frozen target policy :math:`\\pi` whose critical steps are to be
        explained.
    mask_network :
        The MaskNet :math:`\\xi(s)`. If ``None``, one matching the target
        policy architecture is created automatically.
    alpha :
        Blinding bonus coefficient for the mask reward
        :math:`r_{mask}=r_{env}+\\alpha(1-\\xi(s))`.
    ppo_config :
        PPO hyperparameters. If ``None``, SB3-style defaults are used.
    device :
        Device for the mask network and PPO updates.
    """

    def __init__(
        self,
        env,
        target_policy: BaseTargetPolicy,
        mask_network: Optional[MaskNetwork] = None,
        alpha: float = 1e-4,
        ppo_config: Optional[PPOConfig] = None,
        device: Union[str, torch.device] = "auto",
    ) -> None:
        self.env = env
        self.target_policy = target_policy
        self.alpha = alpha
        self.device = device

        if mask_network is None:
            mask_network = match_target_mask_network(
                target_policy, activation=nn.Tanh
            )
        self.mask_network = mask_network.to(self._device)

        self.actor_critic = MaskActorCritic(
            self.mask_network,
            hidden_sizes=self.mask_network.hidden_sizes,
            activation=self.mask_network.activation,
        ).to(self._device)

        self.mask_policy = MaskTorchPolicy(
            self.actor_critic,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=self._device,
        )

        self.masked_env = MaskedEnv(
            env=env,
            target_policy=target_policy,
            mask_network=self.mask_network,
            alpha=alpha,
            device=self._device,
        )

        self.config = ppo_config if ppo_config is not None else PPOConfig()
        self.trainer = PPOTrainer(
            policy=self.mask_policy,
            env=self.masked_env,
            config=self.config,
        )

    @property
    def _device(self) -> torch.device:
        if self.device == "auto" or self.device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        save_path: Optional[Union[str, Path]] = None,
        save_interval: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train the MaskNet with PPO on the masked environment.

        Returns
        -------
        stats :
            Training statistics returned by :class:`PPOTrainer.learn`.
        """
        stats = self.trainer.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            save_path=save_path,
            save_interval=save_interval,
        )
        return stats

    @torch.no_grad()
    def collect_critical_states(
        self,
        n_trajectories: int = 100,
        top_p: Optional[float] = None,
        threshold: Optional[float] = None,
        max_steps_per_episode: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Collect critical states by rolling out the target policy.

        For each visited state the trained MaskNet outputs :math:`\\xi(s)`. The
        states with the highest scores are returned as the critical-state
        buffer. Selection follows ``top_p`` if provided (top percentile), else
        ``threshold`` (default ``0.5``), else all states with :math:`\\xi>0.5`.

        Parameters
        ----------
        n_trajectories :
            Number of target-policy episodes to collect.
        top_p :
            Fraction of states to keep, e.g. ``0.1`` keeps the top 10%.
        threshold :
            Minimum :math:`\\xi(s)` for a state to be considered critical.
        max_steps_per_episode :
            Optional per-episode step limit.

        Returns
        -------
        critical_states :
            List of dictionaries with keys ``state``, ``xi``, ``action``,
            ``reward``, ``info``.
        """
        self.mask_network.eval()
        self.target_policy.eval()

        records: List[Dict[str, Any]] = []
        for _ in range(n_trajectories):
            obs, info = self.env.reset(), {}
            if isinstance(obs, tuple):
                obs, info = obs
            done = False
            steps = 0
            while not done:
                action, _ = self.target_policy.predict(obs, deterministic=True)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, info = step_result

                obs_t = self._to_tensor(obs)
                xi = float(self.mask_network(obs_t).squeeze().cpu().numpy())
                records.append(
                    {
                        "state": obs,
                        "xi": xi,
                        "action": action,
                        "reward": reward,
                        "info": info,
                    }
                )

                obs = next_obs
                steps += 1
                if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                    break

        if top_p is not None:
            scores = np.array([r["xi"] for r in records])
            cutoff = np.percentile(scores, 100 * (1.0 - top_p))
            critical = [r for r in records if r["xi"] >= cutoff]
        elif threshold is not None:
            critical = [r for r in records if r["xi"] >= threshold]
        else:
            critical = [r for r in records if r["xi"] >= 0.5]

        return critical

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained MaskNet network."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mask_network_state_dict": self.mask_network.state_dict(),
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "hidden_sizes": self.mask_network.hidden_sizes,
                "activation": self.mask_network.activation,
                "obs_dim": self.mask_network.obs_dim,
                "alpha": self.alpha,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load a trained MaskNet network."""
        checkpoint = torch.load(path, map_location=self._device)
        self.mask_network.load_state_dict(checkpoint["mask_network_state_dict"])
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self._device)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self._device)


def train_mask_network(
    env,
    target_policy: BaseTargetPolicy,
    total_timesteps: int = 1_000_000,
    alpha: float = 1e-4,
    ppo_config: Optional[PPOConfig] = None,
    device: Union[str, torch.device] = "auto",
    save_path: Optional[Union[str, Path]] = None,
) -> MaskTrainer:
    """Convenience factory that creates and trains a :class:`MaskTrainer`.

    Returns the trainer object, which exposes ``mask_network`` and
    ``collect_critical_states``.
    """
    trainer = MaskTrainer(
        env=env,
        target_policy=target_policy,
        alpha=alpha,
        ppo_config=ppo_config,
        device=device,
    )
    trainer.train(total_timesteps=total_timesteps, save_path=save_path)
    if save_path is not None:
        trainer.save(save_path)
    return trainer
