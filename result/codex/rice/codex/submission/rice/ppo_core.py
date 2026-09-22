"""A small, self-contained PPO implementation.

The same update rule is shared by every learning component of the reproduction:

* Algorithm 1 -- training the mask network,
* Algorithm 2 -- refining the agent (RICE),
* the three refining baselines (PPO fine-tuning, StateMask-R, JSRL),
* the generator of the GAIL imitation step (Experiment IV).

Keeping one implementation guarantees that the comparison between RICE and the
baselines is not confounded by different optimisers.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclasses.dataclass
class PPOConfig:
    """Hyper-parameters of the PPO optimiser."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    target_kl: Optional[float] = None
    #: RICE / baselines fine-tune with a lowered learning rate (Section 4.1)
    finetune_learning_rate: float = 1e-4


class RolloutBuffer:
    """On-policy rollout storage with Generalised Advantage Estimation."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.infos: List[dict] = []
        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.obs)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        info: Optional[dict] = None,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32).copy())
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.dones.append(bool(done))
        self.infos.append(info or {})

    def extend(self, other: "RolloutBuffer") -> None:
        self.obs.extend(other.obs)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.log_probs.extend(other.log_probs)
        self.dones.extend(other.dones)
        self.infos.extend(other.infos)

    def compute_returns_and_advantage(self, last_value: float) -> None:
        rewards = np.asarray(self.rewards, dtype=np.float64)
        values = np.asarray(self.values + [float(last_value)], dtype=np.float64)
        dones = np.asarray(self.dones, dtype=np.float64)
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float64)
        last_gae = 0.0
        for t in reversed(range(n)):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        self.advantages = torch.as_tensor(advantages, dtype=torch.float32)
        self.returns = self.advantages + torch.as_tensor(values[:n], dtype=torch.float32)

    # ------------------------------------------------------------- tensors
    def as_tensors(self, device: str = "cpu"):
        obs = torch.as_tensor(np.asarray(self.obs, dtype=np.float32), device=device)
        actions = np.asarray(self.actions)
        if actions.dtype.kind in "iub":
            actions_t = torch.as_tensor(actions.astype(np.int64), device=device)
        else:
            actions_t = torch.as_tensor(actions.astype(np.float32), device=device)
        return obs, actions_t

    def to_tensors(self, device: str = "cpu") -> Dict[str, torch.Tensor]:
        obs, actions = self.as_tensors(device)
        return {
            "obs": obs,
            "actions": actions,
            "old_log_probs": torch.as_tensor(
                np.asarray(self.log_probs, dtype=np.float32), device=device
            ),
            "old_values": torch.as_tensor(
                np.asarray(self.values, dtype=np.float32), device=device
            ),
            "advantages": self.advantages.to(device),
            "returns": self.returns.to(device),
        }


class PPOUpdater:
    """Runs the clipped PPO objective on a :class:`RolloutBuffer`."""

    def __init__(
        self,
        policy: nn.Module,
        config: PPOConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
    ):
        self.policy = policy
        self.config = config
        self.device = device
        self.optimizer = optimizer or torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate, eps=1e-5
        )

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        cfg = self.config
        data = buffer.to_tensors(self.device)
        n = len(buffer)
        batch_size = min(cfg.batch_size, n)
        indices = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        clip_fraction = 0.0

        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, batch_size):
                batch_idx = indices[start : start + batch_size]
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                obs = data["obs"][idx]
                actions = data["actions"][idx]
                old_log_probs = data["old_log_probs"][idx]
                old_values = data["old_values"][idx]
                advantages = data["advantages"][idx]
                returns = data["returns"][idx]
                if cfg.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                new_log_probs, entropy, values = self.policy.evaluate_actions(
                    obs, actions
                )
                ratio = torch.exp(new_log_probs - old_log_probs)
                clip_fraction += float(
                    (torch.abs(ratio - 1.0) > cfg.clip_range).float().mean().item()
                )
                policy_loss = -torch.min(
                    ratio * advantages,
                    torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                    * advantages,
                ).mean()
                if cfg.clip_range_vf is None:
                    value_loss = ((values - returns) ** 2).mean()
                else:
                    value_clipped = old_values + torch.clamp(
                        values - old_values, -cfg.clip_range_vf, cfg.clip_range_vf
                    )
                    value_loss = torch.max(
                        (values - returns) ** 2, (value_clipped - returns) ** 2
                    ).mean()
                entropy_loss = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.vf_coef * value_loss
                    - cfg.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), cfg.max_grad_norm
                    )
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_log_probs - new_log_probs).mean()
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_loss.item())
                stats["approx_kl"] += float(approx_kl.item())
                n_updates += 1

            if cfg.target_kl is not None:
                mean_kl = stats["approx_kl"] / max(1, n_updates)
                if mean_kl > 1.5 * cfg.target_kl:
                    break

        for key in stats:
            stats[key] /= max(1, n_updates)
        stats["clip_fraction"] = clip_fraction / max(1, n_updates)
        stats["n_updates"] = float(n_updates)
        return stats


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true))
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)
