"""Pre-training the target agents.

The paper obtains the pre-trained (bottlenecked) agents with the default
Stable-Baselines3 algorithms: PPO for the simulated games and the security
applications, SAC for the multi-policy experiment (Experiment IV).  Both are
supported here:

* :func:`train_ppo_policy` uses the PPO implementation of this repository
  (identical optimiser to the refining phase, only the learning rate and the
  number of steps differ),
* :func:`train_sb3_agent` uses Stable-Baselines3 when it is installed
  (``PPO``, ``SAC``, ``A2C``, ...) exactly like the paper does.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, Optional

import numpy as np

from rice.envs.registry import ENV_SPECS, make_env
from rice.networks import ActorCritic
from rice.ppo_core import PPOConfig
from rice.refining.methods import make_eval_env
from rice.refining.trainer import RefineConfig, RefiningTrainer
from rice.utils import set_global_seeds


@dataclasses.dataclass
class PretrainConfig:
    total_steps: int = 300_000
    seed: int = 0
    learning_rate: float = 3e-4
    n_steps: int = 2048
    gamma: float = 0.99
    normalize_obs: bool = False
    eval_interval: int = 50_000
    eval_episodes: int = 10


def train_ppo_policy(
    env_name: str,
    config: Optional[PretrainConfig] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, object]:
    """Pre-train a PPO agent with our own implementation."""
    config = config or PretrainConfig()
    spec = ENV_SPECS[env_name]
    set_global_seeds(config.seed)
    env = make_env(env_name, seed=config.seed)
    eval_env = make_eval_env(env_name, seed=config.seed + 1000)
    obs_dim = int(np.prod(env.observation_space.shape))
    discrete = hasattr(env.action_space, "n")
    act_dim = int(env.action_space.n) if discrete else int(np.prod(env.action_space.shape))
    policy = ActorCritic(obs_dim, act_dim, hidden=spec.policy_hidden, discrete=discrete)

    ppo = PPOConfig(learning_rate=config.learning_rate, n_steps=config.n_steps, gamma=config.gamma)
    ppo.finetune_learning_rate = config.learning_rate
    refine_config = RefineConfig(
        total_steps=config.total_steps,
        n_steps=config.n_steps,
        seed=config.seed,
        ppo=ppo,
        eval_interval=config.eval_interval,
        eval_episodes=config.eval_episodes,
        normalize_obs=config.normalize_obs,
    )
    trainer = RefiningTrainer(
        env=env,
        policy=policy,
        config=refine_config,
        method="pretrain_ppo",
        device=device,
        eval_env=eval_env,
        verbose=verbose,
    )
    result = trainer.train(config.total_steps)
    return {
        "policy": result.policy,
        "history": result.history,
        "eval_history": result.eval_history,
        "final_return": result.final_eval,
        "obs_normalizer": trainer.obs_normalizer,
    }


def train_sb3_agent(
    env_name: str,
    algo: str = "sac",
    total_steps: int = 200_000,
    seed: int = 0,
    env=None,
    **algo_kwargs,
):
    """Pre-train an agent with Stable-Baselines3 (paper: SAC for Experiment IV)."""
    from stable_baselines3 import SAC  # noqa: F401  (imported for the error message)

    import stable_baselines3 as sb3

    if env is None:
        env = make_env(env_name, seed=seed)
    cls = {
        "ppo": sb3.PPO,
        "sac": sb3.SAC,
        "a2c": sb3.A2C,
        "td3": sb3.TD3,
    }[algo.lower()]
    model = cls("MlpPolicy", env.env if hasattr(env, "env") else env, seed=seed, **algo_kwargs)
    model.learn(total_timesteps=total_steps)
    return model


def evaluate_policy(env, policy, n_episodes: int = 10, deterministic: bool = True):
    """Mean / std episode reward of ``policy`` from the default initial states."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = np.asarray(policy.act(obs, deterministic=deterministic))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        returns.append(total)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "returns": returns,
    }


def save_policy(policy: ActorCritic, path: str, meta: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    policy.save(path)
    if meta is not None:
        import json

        with open(path + ".json", "w") as handle:
            json.dump(meta, handle, indent=2)
