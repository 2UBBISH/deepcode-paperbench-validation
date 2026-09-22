"""The refining loop shared by RICE and every baseline.

The loop implements the structure of Algorithm 2:

    for iteration:
        choose an initial state (default initial distribution rho, or a
            critical state identified by the mask network)
        collect a rollout with the current policy
        optionally add an exploration bonus (RND) to the reward
        PPO update of the policy (+ RND predictor update)

Swapping the ``initial_state_provider``, the ``reward_shaper`` and the
``rollin`` hooks turns this single implementation into

* RICE                : mixed initial distribution + RND exploration,
* PPO fine-tuning     : default initial distribution, no exploration,
* StateMask-R         : critical states only, no exploration,
* JSRL                : guided roll-in with the pre-trained policy.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from rice.networks import ActorCritic
from rice.ppo_core import PPOConfig, PPOUpdater, RolloutBuffer, explained_variance
from rice.running_stats import RunningMeanStd
from rice.utils import set_global_seeds


@dataclasses.dataclass
class RefineConfig:
    """Configuration of the refining phase."""

    total_steps: int = 300_000
    n_steps: int = 2048
    seed: int = 0
    log_interval: int = 5
    eval_interval: int = 25_000
    eval_episodes: int = 10
    max_episode_steps: Optional[int] = None
    normalize_obs: bool = False
    #: PPO hyper-parameters (fine-tuning uses ``finetune_learning_rate``)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    #: RICE: probability of starting from a critical state (``p`` in the paper)
    p: float = 0.25
    #: RICE: coefficient of the RND exploration bonus (``lambda``)
    rnd_lambda: float = 0.01
    #: length of the trajectory used to identify a critical state
    rollin_length: int = 128
    #: JSRL: maximum number of guided steps (curriculum horizon)
    jsrl_max_rollin: int = 100

    def __post_init__(self):
        self.ppo.learning_rate = self.ppo.finetune_learning_rate


@dataclasses.dataclass
class RefineResult:
    policy: ActorCritic
    history: List[Dict[str, float]]
    eval_history: List[Dict[str, float]]
    wall_time: float
    total_steps: int
    method: str
    config: RefineConfig

    @property
    def final_eval(self) -> float:
        if not self.eval_history:
            return float("nan")
        return float(self.eval_history[-1]["mean_return"])


class RefiningTrainer:
    def __init__(
        self,
        env,
        policy: ActorCritic,
        config: Optional[RefineConfig] = None,
        method: str = "rice",
        device: str = "cpu",
        initial_state_provider: Optional[Callable[[int], Optional[dict]]] = None,
        reward_shaper=None,
        rollin_policy=None,
        rollin_steps_fn: Optional[Callable[[float], int]] = None,
        eval_env=None,
        verbose: bool = True,
    ):
        self.env = env
        self.policy = policy.to(device)
        self.config = config or RefineConfig()
        self.method = method
        self.device = device
        self.initial_state_provider = initial_state_provider
        self.reward_shaper = reward_shaper
        self.rollin_policy = rollin_policy
        self.rollin_steps_fn = rollin_steps_fn
        self.eval_env = eval_env
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.config.ppo.learning_rate, eps=1e-5
        )
        self.updater = PPOUpdater(
            self.policy, self.config.ppo, optimizer=self.optimizer, device=device
        )
        obs_dim = int(np.prod(env.observation_space.shape))
        self.obs_normalizer = (
            RunningMeanStd(shape=(obs_dim,)) if self.config.normalize_obs else None
        )

    # ------------------------------------------------------------- utilities
    def _policy_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if self.obs_normalizer is None:
            return obs
        return self.obs_normalizer.normalize(obs)

    def _start_episode(self, iteration: int):
        snapshot = None
        if self.initial_state_provider is not None:
            snapshot = self.initial_state_provider(iteration)
        if snapshot is not None:
            self.env.set_state(snapshot)
            obs = self.env.current_obs()
        else:
            obs, _ = self.env.reset()
        rollin = 0
        if self.rollin_steps_fn is not None:
            progress = min(1.0, iteration * self.config.n_steps / max(1, self.config.total_steps))
            rollin = int(self.rollin_steps_fn(progress))
        return obs, rollin

    @torch.no_grad()
    def evaluate(self, episodes: Optional[int] = None) -> Dict[str, float]:
        env = self.eval_env or self.env
        episodes = episodes or self.config.eval_episodes
        returns = []
        lengths = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            total = 0.0
            length = 0
            while not done:
                action = self.policy.act(self._policy_obs(obs), deterministic=True)[0]
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                done = bool(terminated or truncated)
                if self.config.max_episode_steps and length >= self.config.max_episode_steps:
                    break
            returns.append(total)
            lengths.append(length)
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
        }

    # ------------------------------------------------------------- training
    def train(self, total_steps: Optional[int] = None) -> RefineResult:
        cfg = self.config
        total_steps = int(total_steps or cfg.total_steps)
        set_global_seeds(cfg.seed)
        self.policy.train()
        history: List[Dict[str, float]] = []
        eval_history: List[Dict[str, float]] = []
        steps = 0
        iteration = 0
        episode_return = 0.0
        episode_length = 0
        recent_returns: List[float] = []
        start_time = time.time()
        obs, rollin_left = self._start_episode(iteration)
        next_eval = cfg.eval_interval

        while steps < total_steps:
            buffer = RolloutBuffer(gamma=cfg.ppo.gamma, gae_lambda=cfg.ppo.gae_lambda)
            while len(buffer) < cfg.n_steps and steps < total_steps:
                # ---- JSRL guided roll-in (data is not used for the update)
                if rollin_left > 0:
                    action = self.rollin_policy.act(obs, deterministic=False)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    rollin_left -= 1
                    if terminated or truncated:
                        obs, rollin_left = self._start_episode(iteration)
                        episode_return, episode_length = 0.0, 0
                    continue

                if self.obs_normalizer is not None:
                    self.obs_normalizer.update(np.asarray(obs, dtype=np.float64))
                policy_obs = self._policy_obs(obs)
                action, log_prob, value = self.policy.act(policy_obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = bool(terminated or truncated)
                env_reward = float(reward)
                shaped = env_reward
                if self.reward_shaper is not None:
                    shaped = self.reward_shaper.shape(
                        obs, next_obs, env_reward, done, info
                    )
                buffer.add(
                    obs=policy_obs,
                    action=action,
                    reward=shaped,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    info=info,
                )
                episode_return += env_reward
                episode_length += 1
                steps += 1
                if done:
                    recent_returns.append(episode_return)
                    obs, rollin_left = self._start_episode(iteration)
                    episode_return, episode_length = 0.0, 0
                else:
                    obs = next_obs

            with torch.no_grad():
                last_obs = self._policy_obs(obs)
                last_value = float(
                    self.policy.value(
                        torch.as_tensor(np.atleast_2d(last_obs))
                    ).item()
                )
            buffer.compute_returns_and_advantage(last_value)
            stats = self.updater.update(buffer)
            shaper_stats = {}
            if self.reward_shaper is not None and hasattr(
                self.reward_shaper, "after_iteration"
            ):
                shaper_stats = self.reward_shaper.after_iteration(buffer)

            values = np.asarray(buffer.values, dtype=np.float64)
            returns = buffer.returns.numpy().astype(np.float64)
            history.append(
                {
                    "iteration": float(iteration),
                    "steps": float(steps),
                    "mean_episode_return": float(np.mean(recent_returns[-20:]))
                    if recent_returns
                    else float("nan"),
                    "explained_variance": explained_variance(values, returns),
                    **{k: float(v) for k, v in stats.items()},
                    **{k: float(v) for k, v in shaper_stats.items()},
                }
            )
            if self.verbose and iteration % cfg.log_interval == 0:
                print(
                    "[{}] iter {:4d} steps {:8d} ep_return {:9.2f} pi_loss {:8.4f}".format(
                        self.method,
                        iteration,
                        steps,
                        history[-1]["mean_episode_return"],
                        history[-1]["policy_loss"],
                    ),
                    flush=True,
                )
            iteration += 1

            if self.eval_env is not None and steps >= next_eval:
                self.policy.eval()
                metrics = self.evaluate()
                self.policy.train()
                metrics["steps"] = float(steps)
                eval_history.append(metrics)
                next_eval += cfg.eval_interval

        if self.eval_env is not None:
            self.policy.eval()
            metrics = self.evaluate()
            metrics["steps"] = float(steps)
            eval_history.append(metrics)
        wall_time = time.time() - start_time
        return RefineResult(
            policy=self.policy,
            history=history,
            eval_history=eval_history,
            wall_time=wall_time,
            total_steps=steps,
            method=self.method,
            config=cfg,
        )
