"""GAIL imitation of an off-policy agent (Experiment IV).

Experiment IV refines an agent that was *not* trained by PPO.  The paper first
obtains a pre-trained SAC agent and then uses Generative Adversarial Imitation
Learning (Ho & Ermon, 2016) to learn an approximated policy network, which is
then refined with RICE (and with the baselines).

Implementation: a discriminator ``D(s, a)`` is trained to separate expert
(SAC) transitions from the transitions of the generator policy, and the
generator is optimised with PPO on the GAIL reward ``-log(1 - D(s, a))``.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from rice.networks import ActorCritic, init_orthogonal, mlp
from rice.ppo_core import PPOConfig, PPOUpdater, RolloutBuffer
from rice.utils import set_global_seeds


@dataclasses.dataclass
class GAILConfig:
    total_steps: int = 200_000
    n_steps: int = 2048
    batch_size: int = 256
    disc_epochs: int = 3
    disc_learning_rate: float = 3e-4
    ppo_learning_rate: float = 3e-4
    gae_lambda: float = 0.95
    gamma: float = 0.99
    entropy_coef: float = 0.0
    #: weight of the environment reward on top of the GAIL reward
    env_reward_coef: float = 0.0
    expert_episodes: int = 50
    log_interval: int = 5
    seed: int = 0


class Discriminator(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(128, 128)):
        super().__init__()
        self.net = mlp((obs_dim + act_dim,) + tuple(hidden) + (1,), "tanh")
        init_orthogonal(self.net, gain=np.sqrt(2))

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


def collect_expert_transitions(env, expert_policy, n_episodes: int, seed: int = 0):
    obs_list, act_list = [], []
    np.random.seed(seed)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = np.asarray(expert_policy.act(obs, deterministic=False), dtype=np.float32)
            obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            act_list.append(np.atleast_1d(action).copy())
            obs, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
    return np.asarray(obs_list, dtype=np.float32), np.asarray(act_list, dtype=np.float32)


def train_gail(
    env,
    expert_policy,
    policy: ActorCritic,
    config: Optional[GAILConfig] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, object]:
    """Imitate ``expert_policy`` with ``policy`` (PPO generator + GAIL reward)."""
    config = config or GAILConfig()
    set_global_seeds(config.seed, env=env)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(policy.act_dim)
    discriminator = Discriminator(obs_dim, act_dim).to(device)
    disc_opt = torch.optim.Adam(
        discriminator.parameters(), lr=config.disc_learning_rate
    )

    expert_obs, expert_act = collect_expert_transitions(
        env, expert_policy, config.expert_episodes, seed=config.seed
    )
    expert_obs_t = torch.as_tensor(expert_obs, device=device)
    expert_act_t = torch.as_tensor(expert_act, device=device)
    bce = nn.BCEWithLogitsLoss()

    ppo = PPOConfig(
        learning_rate=config.ppo_learning_rate,
        n_steps=config.n_steps,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.entropy_coef,
    )
    ppo.finetune_learning_rate = config.ppo_learning_rate
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.ppo_learning_rate, eps=1e-5)
    updater = PPOUpdater(policy, ppo, optimizer=optimizer, device=device)

    history: List[Dict[str, float]] = []
    steps = 0
    iteration = 0
    obs, _ = env.reset()
    while steps < config.total_steps:
        buffer = RolloutBuffer(gamma=config.gamma, gae_lambda=config.gae_lambda)
        policy_obs_list, policy_act_list = [], []
        while len(buffer) < config.n_steps and steps < config.total_steps:
            action, log_prob, value = policy.act(obs)
            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device)
                act_t = torch.as_tensor(np.atleast_1d(action).astype(np.float32), device=device)
                d_val = discriminator(obs_t, act_t)
                gail_reward = -torch.nn.functional.logsigmoid(-d_val).item()
            reward = gail_reward + config.env_reward_coef * float(env_reward)
            buffer.add(obs, action, reward, value, log_prob, done, {})
            policy_obs_list.append(np.asarray(obs, dtype=np.float32).copy())
            policy_act_list.append(np.atleast_1d(action).astype(np.float32).copy())
            steps += 1
            obs = next_obs if not done else env.reset()[0]

        # ------------------------------------------------- discriminator step
        p_obs = torch.as_tensor(np.asarray(policy_obs_list), device=device)
        p_act = torch.as_tensor(np.asarray(policy_act_list), device=device)
        disc_loss_value = 0.0
        for _ in range(config.disc_epochs):
            idx_e = torch.randint(0, expert_obs_t.shape[0], (config.batch_size,))
            idx_p = torch.randint(0, p_obs.shape[0], (config.batch_size,))
            logits_e = discriminator(expert_obs_t[idx_e], expert_act_t[idx_e])
            logits_p = discriminator(p_obs[idx_p], p_act[idx_p])
            loss = bce(logits_e, torch.ones_like(logits_e)) + bce(
                logits_p, torch.zeros_like(logits_p)
            )
            disc_opt.zero_grad()
            loss.backward()
            disc_opt.step()
            disc_loss_value = float(loss.item())

        # --------------------------------------------------------- PPO step
        buffer.compute_returns_and_advantage(0.0)
        stats = updater.update(buffer)
        with torch.no_grad():
            acc = float(
                (
                    (torch.sigmoid(discriminator(expert_obs_t[:256], expert_act_t[:256])) > 0.5)
                    .float()
                    .mean()
                    .item()
                )
            )
        history.append(
            {
                "iteration": float(iteration),
                "steps": float(steps),
                "disc_loss": disc_loss_value,
                "expert_accuracy": acc,
                **stats,
            }
        )
        if verbose and iteration % config.log_interval == 0:
            print(
                "[gail] iter {:4d} steps {:8d} disc_loss {:.4f} expert_acc {:.3f}".format(
                    iteration, steps, disc_loss_value, acc
                ),
                flush=True,
            )
        iteration += 1

    return {"policy": policy, "discriminator": discriminator, "history": history}
