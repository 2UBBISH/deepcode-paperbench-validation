"""Algorithm 1 -- training the mask network (our explanation method).

The mask network of RICE is a PPO policy with a binary action space.  At every
step the target agent proposes an action ``a_t ~ pi(.|s_t)`` and the mask
network proposes ``a_t^m``; the action that is actually executed is

    a_t            if a_t^m = 0   (the agent is not blinded)
    a_random       if a_t^m = 1   (the agent is blinded)

Instead of the primal-dual formulation of StateMask, RICE optimises

    R'(s_t, a_t) = R(s_t, a_t) + alpha * a_t^m

with vanilla PPO.  Theorem 3.3 of the paper guarantees ``eta(pi_bar) <=
eta(pi)`` under Assumption 3.1, therefore maximising ``eta(pi_bar)`` is
equivalent to minimising ``|eta(pi) - eta(pi_bar)|`` -- the objective of
StateMask -- while the extra ``alpha`` bonus prevents the trivial solution
"never blind the agent".  The importance of a state is the probability that the
mask network does *not* blind the agent, i.e. ``P(a^m = 0 | s)``.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from rice.explanation.critical_states import importance_scores
from rice.networks import MaskNet
from rice.ppo_core import PPOConfig, PPOUpdater, RolloutBuffer
from rice.running_stats import RunningMeanStd
from rice.utils import set_global_seeds


@dataclasses.dataclass
class MaskTrainingConfig:
    """Hyper-parameters of Algorithm 1 (Appendix C.3 default: ``alpha=0.01``)."""

    iterations: int = 300
    #: number of environment steps collected per iteration
    rollout_length: int = 1000
    #: coefficient of the "blind the agent" bonus
    alpha: float = 0.01
    #: multiplier of the environment reward before adding the blinding bonus.
    #: ``1.0`` reproduces ``R'(s,a) = R(s,a) + alpha * a^m`` of the paper with
    #: raw rewards.  ``None`` (default) normalises the reward by the running
    #: standard deviation of the episode returns, which is what the paper's
    #: Stable-Baselines3 / VecNormalize pipeline does; without it an ``alpha``
    #: of 0.01 is negligible next to e.g. the ~3.6 reward per step of Hopper and
    #: the mask network degenerates to "never blind the agent".
    reward_scale: Optional[float] = None
    #: clipping of the normalised reward (analogue of SB3's ``clip_reward``)
    reward_norm_clip: float = 10.0
    #: stop early once this many samples were consumed (Table 4 budgets)
    max_samples: Optional[int] = None
    seed: int = 0
    log_interval: int = 25
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)


@dataclasses.dataclass
class MaskTrainingResult:
    mask_net: MaskNet
    samples: int
    wall_time: float
    history: List[Dict[str, float]]
    config: MaskTrainingConfig

    def importance(self, states) -> np.ndarray:
        return importance_scores(self.mask_net, states)


# Backwards friendly alias
MaskLearningResult = MaskTrainingResult


class MaskActorCritic(MaskNet):
    """Mask network with a value head so that PPO has a critic to train."""

    def __init__(self, obs_dim: int, hidden=(64, 64), activation: str = "tanh"):
        super().__init__(obs_dim, hidden=hidden, activation=activation)
        from rice.networks import init_orthogonal, mlp

        hidden = tuple(int(h) for h in hidden)
        self.value_head = mlp((obs_dim,) + hidden + (1,), activation)
        init_orthogonal(self.value_head, gain=1.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_head(obs).squeeze(-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.dist(obs)
        log_prob = dist.log_prob(actions.reshape(-1))
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        obs_t = torch.as_tensor(np.atleast_2d(np.asarray(obs, dtype=np.float32)))
        dist = self.dist(obs_t)
        action = torch.argmax(dist.probs, dim=-1) if deterministic else dist.sample()
        log_prob = float(dist.log_prob(action).sum().item())
        return action.squeeze(-1).numpy().astype(np.int64), log_prob, float(
            self.value(obs_t).squeeze(-1).item()
        )


def train_mask_network(
    env,
    target_policy,
    config: Optional[MaskTrainingConfig] = None,
    mask_net: Optional[torch.nn.Module] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> MaskTrainingResult:
    """Run Algorithm 1 on ``env`` for the given target policy.

    Parameters
    ----------
    env:
        stateful environment (see :mod:`rice.envs`)
    target_policy:
        pre-trained policy ``pi``; must expose ``act(obs, deterministic)``
    config:
        :class:`MaskTrainingConfig`
    """
    config = config or MaskTrainingConfig()
    set_global_seeds(config.seed, env=env)

    obs_dim = int(np.prod(env.observation_space.shape))
    mask_net = mask_net or MaskActorCritic(obs_dim)
    mask_net = mask_net.to(device)

    optimizer = torch.optim.Adam(
        mask_net.parameters(), lr=config.ppo.learning_rate, eps=1e-5
    )
    updater = PPOUpdater(mask_net, config.ppo, optimizer=optimizer, device=device)

    history: List[Dict[str, float]] = []
    samples = 0
    start_time = time.time()
    reward_rms = RunningMeanStd(shape=()) if config.reward_scale is None else None
    running_episode_return = 0.0

    for iteration in range(config.iterations):
        if config.max_samples is not None and samples >= config.max_samples:
            break
        buffer = RolloutBuffer(gamma=config.ppo.gamma, gae_lambda=config.ppo.gae_lambda)
        obs, _ = env.reset()
        steps_this_iteration = 0
        episode_reward = 0.0
        mask_rate_sum = 0.0

        # ------------------------------------------------------------------
        # Algorithm 1, inner loop: one trajectory of length T
        # ------------------------------------------------------------------
        while steps_this_iteration < config.rollout_length:
            action_t = np.asarray(target_policy.act(obs, deterministic=False))
            with torch.no_grad():
                obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                dist = mask_net.dist(obs_t)
                mask_action = dist.sample()
                log_prob = float(dist.log_prob(mask_action).item())
                value = float(mask_net.value(obs_t).item())
            mask = int(mask_action.item())
            executed = env.random_action() if mask == 1 else action_t
            next_obs, env_reward, terminated, truncated, info = env.step(executed)
            done = bool(terminated or truncated)
            running_episode_return += float(env_reward)
            if reward_rms is not None:
                if done:
                    reward_rms.update(
                        np.asarray([running_episode_return], dtype=np.float64)
                    )
                    running_episode_return = 0.0
                scale = 1.0 / (float(reward_rms.std) + 1e-6)
                scaled_reward = float(
                    np.clip(
                        float(env_reward) * scale,
                        -config.reward_norm_clip,
                        config.reward_norm_clip,
                    )
                )
            else:
                scaled_reward = float(config.reward_scale) * float(env_reward)
            reward = scaled_reward + config.alpha * mask
            buffer.add(
                obs=obs,
                action=np.asarray([mask], dtype=np.int64),
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                info=info,
            )
            steps_this_iteration += 1
            samples += 1
            episode_reward += float(env_reward)
            mask_rate_sum += mask
            obs = next_obs
            if done:
                obs, _ = env.reset()

        # ------------------------------------------------------------------
        # Algorithm 1, update theta_old <- theta using D with PPO
        # ------------------------------------------------------------------
        with torch.no_grad():
            last_value = 0.0
        buffer.compute_returns_and_advantage(last_value)
        stats = updater.update(buffer)
        history.append(
            {
                "iteration": float(iteration),
                "samples": float(samples),
                "mean_env_reward": episode_reward / max(1, config.rollout_length),
                "mask_rate": mask_rate_sum / max(1, config.rollout_length),
                **stats,
            }
        )
        if verbose and (iteration % config.log_interval == 0):
            print(
                "[mask] iter {:4d} samples {:8d} env_reward/step {:8.3f} "
                "mask_rate {:.3f} pi_loss {:8.4f}".format(
                    iteration,
                    samples,
                    history[-1]["mean_env_reward"],
                    history[-1]["mask_rate"],
                    history[-1]["policy_loss"],
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
