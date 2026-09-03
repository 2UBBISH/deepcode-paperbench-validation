"""Mask network implementation for RICE.

The mask network is an alternative design to StateMask. It is trained to identify
critical states by learning to "blind" the target agent at non-critical steps.
"""
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical

from rice.env_utils import sample_random_action
from rice.utils import compute_returns, get_device


class MaskNetwork(nn.Module):
    """Policy network for the mask action a^m in {0, 1}.

    Outputs a Bernoulli probability P(a^m=1 | s). A value head is included for
    PPO training.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_size = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev_size, h), nn.ReLU()])
            prev_size = h
        self.shared = nn.Sequential(*layers)
        self.actor = nn.Linear(prev_size, 1)
        self.critic = nn.Linear(prev_size, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(obs)
        logits = self.actor(x).squeeze(-1)
        value = self.critic(x).squeeze(-1)
        return logits, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        probs = torch.sigmoid(logits)
        dist = Bernoulli(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


def collect_mask_trajectory(
    env: gym.Env,
    target_policy: Any,
    mask_net: MaskNetwork,
    alpha: float,
    max_steps: int,
    device: torch.device,
    deterministic_target: bool = False,
) -> Dict[str, np.ndarray]:
    """Collect a single trajectory using the perturbed policy.

    At each step, the mask network decides whether to keep the target agent's
    action (a^m=0) or replace it with a random action (a^m=1). The mask network
    receives reward R' = R + alpha * a^m.

    Returns the trajectory plus a ``next_value`` entry used to bootstrap
    advantages when the trajectory was truncated before a terminal state.
    """
    obs_list: List[np.ndarray] = []
    next_obs_list: List[np.ndarray] = []
    mask_actions: List[int] = []
    rewards: List[float] = []
    dones: List[bool] = []
    values: List[float] = []
    log_probs: List[float] = []

    obs, _ = env.reset()
    next_value = 0.0
    for step in range(max_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_mask, log_prob, _, value = mask_net.get_action_and_value(obs_t)
        action_mask = int(action_mask.cpu().item())

        # Sample target action.
        if hasattr(target_policy, "predict"):
            target_action, _ = target_policy.predict(obs, deterministic=deterministic_target)
            target_action = np.asarray(target_action).reshape(env.action_space.shape)
        elif hasattr(target_policy, "act"):
            target_action = target_policy.act(obs, deterministic=deterministic_target)
        else:
            target_action = target_policy(obs)

        # Apply mask.
        if action_mask == 0:
            actual_action = target_action
        else:
            actual_action = sample_random_action(env)

        next_obs, env_reward, terminated, truncated, _ = env.step(actual_action)
        done = terminated or truncated
        mask_reward = env_reward + alpha * float(action_mask)

        obs_list.append(obs)
        next_obs_list.append(next_obs)
        mask_actions.append(action_mask)
        rewards.append(mask_reward)
        dones.append(done)
        values.append(float(value.cpu().item()))
        log_probs.append(float(log_prob.cpu().item()))

        obs = next_obs
        if done:
            break

    # Bootstrap value for truncated trajectories.
    if not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, next_value_tensor = mask_net(obs_t)
            next_value = float(next_value_tensor.cpu().item())

    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "next_obs": np.array(next_obs_list, dtype=np.float32),
        "mask_actions": np.array(mask_actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "dones": np.array(dones, dtype=np.float32),
        "values": np.array(values, dtype=np.float32),
        "log_probs": np.array(log_probs, dtype=np.float32),
        "next_value": np.array(next_value, dtype=np.float32),
    }


class MaskNetworkTrainer:
    """Train a mask network via PPO using the reformulated objective.

    Under Assumption 3.1, maximizing eta(pi_bar) is equivalent to the original
    StateMask objective and can be optimized with vanilla PPO. An intrinsic
    reward alpha * a^m is added to avoid the trivial "never blind" solution.
    """

    def __init__(
        self,
        env: gym.Env,
        target_policy: Any,
        obs_dim: int,
        alpha: float = 1e-4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        device: Optional[torch.device] = None,
        deterministic_target: bool = False,
    ) -> None:
        self.env = env
        self.target_policy = target_policy
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or get_device()
        self.deterministic_target = deterministic_target

        self.mask_net = MaskNetwork(obs_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.mask_net.parameters(), lr=lr)

    def collect_trajectories(
        self, n_steps: int, max_steps_per_episode: int = 1000
    ) -> Dict[str, np.ndarray]:
        """Collect n_steps of experience."""
        all_obs: List[np.ndarray] = []
        all_next_obs: List[np.ndarray] = []
        all_masks: List[np.ndarray] = []
        all_rewards: List[np.ndarray] = []
        all_dones: List[np.ndarray] = []
        all_values: List[np.ndarray] = []
        all_log_probs: List[np.ndarray] = []
        total_steps = 0
        last_next_value = 0.0
        while total_steps < n_steps:
            traj = collect_mask_trajectory(
                self.env,
                self.target_policy,
                self.mask_net,
                self.alpha,
                max_steps=max_steps_per_episode,
                device=self.device,
                deterministic_target=self.deterministic_target,
            )
            all_obs.append(traj["obs"])
            all_next_obs.append(traj["next_obs"])
            all_masks.append(traj["mask_actions"])
            all_rewards.append(traj["rewards"])
            all_dones.append(traj["dones"])
            all_values.append(traj["values"])
            all_log_probs.append(traj["log_probs"])
            total_steps += len(traj["obs"])
            last_next_value = float(traj["next_value"])
        return {
            "obs": np.concatenate(all_obs, axis=0),
            "next_obs": np.concatenate(all_next_obs, axis=0),
            "mask_actions": np.concatenate(all_masks, axis=0),
            "rewards": np.concatenate(all_rewards, axis=0),
            "dones": np.concatenate(all_dones, axis=0),
            "values": np.concatenate(all_values, axis=0),
            "log_probs": np.concatenate(all_log_probs, axis=0),
            "next_value": np.array(last_next_value, dtype=np.float32),
        }

    def compute_advantages(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns for collected trajectories."""
        # Split at episode boundaries for correct advantage computation.
        advantages_list: List[np.ndarray] = []
        returns_list: List[np.ndarray] = []
        start = 0
        n = len(rewards)
        dones_int = dones.astype(bool)
        # If the last step was non-terminal, bootstrap from next_value.
        if not dones_int[-1]:
            dones_int[-1] = True
            bootstrap_value = next_value
        else:
            bootstrap_value = 0.0
        for end in np.where(dones_int)[0]:
            ep_rewards = rewards[start : end + 1]
            ep_values = values[start : end + 1]
            ep_advantages = np.zeros(len(ep_rewards), dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(len(ep_rewards))):
                if t == len(ep_rewards) - 1:
                    next_v = bootstrap_value if end == n - 1 else 0.0
                else:
                    next_v = ep_values[t + 1]
                delta = ep_rewards[t] + self.gamma * next_v - ep_values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
                ep_advantages[t] = last_gae
            ep_returns = ep_advantages + ep_values
            advantages_list.append(ep_advantages)
            returns_list.append(ep_returns)
            start = end + 1
            bootstrap_value = 0.0
        advantages = np.concatenate(advantages_list)
        returns = np.concatenate(returns_list)
        return advantages, returns

    def update(
        self,
        batch: Dict[str, np.ndarray],
        n_epochs: int = 4,
        mini_batch_size: int = 64,
    ) -> Dict[str, float]:
        """Perform PPO update on the collected mask network data."""
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["mask_actions"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)
        rewards = batch["rewards"]
        dones = batch["dones"]
        values = batch["values"]
        next_value = float(batch.get("next_value", 0.0))

        advantages, returns = self.compute_advantages(rewards, dones, values, next_value)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize advantages.
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = obs.shape[0]
        dataset = torch.utils.data.TensorDataset(
            obs, actions, old_log_probs, advantages_t, returns_t
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=mini_batch_size, shuffle=True, drop_last=False
        )

        policy_losses = []
        value_losses = []
        entropy_losses = []
        for _ in range(n_epochs):
            for mb_obs, mb_actions, mb_old_log_probs, mb_adv, mb_ret in loader:
                _, log_probs, entropy, value = self.mask_net.get_action_and_value(mb_obs, mb_actions)
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(value, mb_ret)
                entropy_loss = -entropy.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.mask_net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy_loss": float(np.mean(entropy_losses)),
        }

    def train(
        self,
        total_timesteps: int,
        steps_per_iter: int = 2048,
        n_iters: Optional[int] = None,
        max_steps_per_episode: int = 1000,
    ) -> List[Dict[str, float]]:
        """Train the mask network for the specified number of timesteps."""
        if n_iters is None:
            n_iters = max(1, total_timesteps // steps_per_iter)
        logs: List[Dict[str, float]] = []
        for iteration in range(n_iters):
            batch = self.collect_trajectories(
                steps_per_iter, max_steps_per_episode=max_steps_per_episode
            )
            update_logs = self.update(batch)
            update_logs["iteration"] = iteration
            update_logs["collected_steps"] = len(batch["obs"])
            logs.append(update_logs)
        return logs

    def save(self, path: str) -> None:
        """Save the mask network."""
        torch.save(self.mask_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load the mask network."""
        self.mask_net.load_state_dict(torch.load(path, map_location=self.device))

    def importance_scores(self, obs: np.ndarray) -> np.ndarray:
        """Return P(a^m=0 | s) for each observation as the state importance."""
        self.mask_net.eval()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            logits, _ = self.mask_net(obs_t)
            probs_blind = torch.sigmoid(logits).cpu().numpy()
        # Importance is the probability of NOT blinding (mask=0).
        return 1.0 - probs_blind
