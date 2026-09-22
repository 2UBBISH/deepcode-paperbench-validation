"""StateMask baseline (Cheng et al., 2023) re-implemented for the comparison.

StateMask learns the same kind of binary mask network as RICE, but optimises it
with the primal-dual objective

    J(theta) = min |eta(pi) - eta(pi_bar)|

where ``pi`` is the target agent and ``pi_bar`` the perturbed (masked) agent.
t the optimum the mask concentrates on the *non critical* steps: blinding them
does not change the final reward.  The reproduction solves the problem with a
Lagrangian multiplier ``lambda`` updated by dual ascent on the reward gap,

    r_t^m = a_t^m - lambda * |G_pert(t) - G_target(t)|
    lambda <- max(0, lambda + eta_dual * (mean|G_pert - G_target| - eps))

where ``G.(t)`` are the discounted accumulated rewards (Monte-Carlo returns) of
the perturbed and of the target agent.  Estimating ``G_target`` requires an
additional roll-out of the target policy in every iteration; this is exactly
the extra computation reported in Table 4 of the paper ("the training algorithm
of the mask network in StateMask involves an estimation of the discounted
accumulated reward with respect to the current policy of the perturbed agent
and the policy of the target agent").

The official implementation (https://github.com/nuwuxian/RL-state_mask) can be
plugged in instead by training a mask network elsewhere and passing its
checkpoint through ``rice.explanation.mask_io.load_mask_net``.
"""

from __future__ import annotations

import dataclasses
import time
from typing import List, Optional

import numpy as np
import torch

from rice.explanation.mask_trainer import (
    MaskActorCritic,
    MaskTrainingConfig,
    MaskTrainingResult,
)
from rice.ppo_core import PPOUpdater, RolloutBuffer
from rice.utils import set_global_seeds


@dataclasses.dataclass
class StateMaskTrainingConfig(MaskTrainingConfig):
    """MaskTrainingConfig + the dual variables of StateMask."""

    dual_learning_rate: float = 0.1
    epsilon: float = 0.0
    init_lambda: float = 1.0
    max_lambda: float = 100.0


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def train_state_mask(
    env,
    target_policy,
    config: Optional[StateMaskTrainingConfig] = None,
    mask_net: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> MaskTrainingResult:
    """Train a StateMask style mask network (primal-dual, reward preserving)."""
    config = config or StateMaskTrainingConfig()
    set_global_seeds(config.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    mask_net = mask_net or MaskActorCritic(obs_dim)
    mask_net = mask_net.to(device)
    optimizer = torch.optim.Adam(
        mask_net.parameters(), lr=config.ppo.learning_rate, eps=1e-5
    )
    updater = PPOUpdater(mask_net, config.ppo, optimizer=optimizer, device=device)

    dual = float(config.init_lambda)
    history: List[dict] = []
    samples = 0
    start_time = time.time()

    for iteration in range(config.iterations):
        if config.max_samples is not None and samples >= config.max_samples:
            break

        # ------------------------------------------------------------------
        # extra computation of StateMask: roll out the *target* agent to
        # estimate its discounted accumulated reward
        # ------------------------------------------------------------------
        target_rewards, target_obs = [], []
        obs, _ = env.reset()
        steps = 0
        while steps < config.rollout_length:
            action = np.asarray(target_policy.act(obs, deterministic=False))
            target_obs.append(np.asarray(obs, dtype=np.float32).copy())
            obs, reward, terminated, truncated, _ = env.step(action)
            target_rewards.append(float(reward))
            steps += 1
            if terminated or truncated:
                obs, _ = env.reset()
        target_returns = _discounted_returns(
            np.asarray(target_rewards), config.ppo.gamma
        )

        # ------------------------------------------------------------------
        # roll out the perturbed agent (mask applied) and collect PPO data
        # ------------------------------------------------------------------
        buffer = RolloutBuffer(
            gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda
        )
        obs, _ = env.reset()
        perturbed_rewards: List[float] = []
        perturbed_obs: List[np.ndarray] = []
        mask_actions: List[int] = []
        log_probs: List[float] = []
        values: List[float] = []
        dones: List[bool] = []
        infos: List[dict] = []
        steps = 0
        while steps < config.rollout_length:
            action_t = np.asarray(target_policy.act(obs, deterministic=False))
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                dist = mask_net.dist(obs_t)
                mask_action = dist.sample()
                log_prob = float(dist.log_prob(mask_action).item())
                value = float(mask_net.value(obs_t).item())
            mask = int(mask_action.item())
            executed = env.random_action() if mask == 1 else action_t
            perturbed_obs.append(np.asarray(obs, dtype=np.float32).copy())
            next_obs, reward, terminated, truncated, info = env.step(executed)
            perturbed_rewards.append(float(reward))
            mask_actions.append(mask)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(bool(terminated or truncated))
            infos.append(info if isinstance(info, dict) else {})
            obs = next_obs if not (terminated or truncated) else env.reset()[0]
            steps += 1
            samples += 1

        perturbed_returns = _discounted_returns(
            np.asarray(perturbed_rewards), config.ppo.gamma
        )
        n = min(len(perturbed_returns), len(target_returns))
        gap = np.abs(perturbed_returns[:n] - target_returns[:n])
        # the mask is rewarded for blinding steps whose perturbation does not
        # change the accumulated reward, penalised otherwise
        mask_rewards = np.asarray(mask_actions[:n], dtype=np.float64) - dual * gap

        for t in range(n):
            buffer.add(
                obs=perturbed_obs[t],
                action=np.asarray([mask_actions[t]], dtype=np.int64),
                reward=float(mask_rewards[t]),
                value=values[t],
                log_prob=log_probs[t],
                done=dones[t],
                info=infos[t],
            )
        buffer.compute_returns_and_advantage(0.0)
        stats = updater.update(buffer)

        # -------------------------------------------------- dual ascent on lambda
        mean_gap = float(gap.mean()) if n else 0.0
        dual = float(
            np.clip(
                dual + config.dual_learning_rate * (mean_gap - config.epsilon),
                0.0,
                config.max_lambda,
            )
        )
        history.append(
            {
                "iteration": float(iteration),
                "samples": float(samples),
                "mask_rate": float(np.mean(mask_actions)),
                "reward_gap": mean_gap,
                "dual": dual,
                **stats,
            }
        )
        if verbose and iteration % config.log_interval == 0:
            print(
                "[state-mask] iter {:4d} samples {:8d} mask_rate {:.3f} "
                "gap {:8.4f} lambda {:.4f}".format(
                    iteration, samples, history[-1]["mask_rate"], mean_gap, dual
                ),
                flush=True,
            )

    wall_time = time.time() - start_time
    return MaskTrainingResult(
        mask_net=mask_net,
        samples=samples,
        wall_time=wall_time,
        history=history,
        config=config,
    )
