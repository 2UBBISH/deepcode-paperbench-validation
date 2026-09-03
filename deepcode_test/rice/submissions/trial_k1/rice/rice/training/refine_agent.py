"""RICE agent-refining pipeline.

This module implements the full refining stage of RICE:

1. Load a pre-trained target agent and a trained mask network.
2. Collect rollouts under the target policy and extract the top-critical states
   using the mask network scores.
3. Build a mixed-initial-state distribution by wrapping the environment with
   :class:`ResettableEnv`, which starts episodes from the critical-state buffer
   with probability ``p`` and from the default initial distribution otherwise.
4. Augment environment rewards with a normalized RND exploration bonus:
   ``r'_t = r_t + λ b_RND(s_t)``.
5. Continue training the target policy (default SB3 PPO) from the mixed
   distribution with the augmented reward.
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

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from rice.agents.target_agent import (
    TargetAgent,
    TargetAgentConfig,
    default_mujoco_config,
    train_target_agent_sb3,
)
from rice.agents.mask_network import (
    MaskNetwork,
    MaskTrainingConfig,
    collect_masked_rollouts,
    default_mask_config,
    extract_critical_states,
    load_mask_network,
    make_mask_network,
)
from rice.agents.rnd_network import (
    RNDModule,
    RNDRewardWrapper,
    default_rnd_config,
    make_rnd_module,
)
from rice.envs.resettable_env import (
    CriticalStateBuffer,
    ResettableEnv,
    make_resettable,
)
from rice.envs.mujoco_wrappers import NormalizeObservationWrapper


@dataclass
class RefineConfig:
    """Hyper-parameters for the RICE refining loop."""

    # Mixed-initial-state parameters
    p: float = 0.25
    """Probability of starting an episode from the critical-state buffer."""
    top_k_critical: Optional[int] = None
    """Number of top-critical states to store. If None, store all collected."""
    critical_percentile: Optional[float] = None
    """Alternative to top_k: keep states above this percentile of mask scores."""

    # RND parameters
    use_rnd: bool = True
    lambda_coef: float = 0.01
    rnd_output_dim: int = 64
    rnd_hidden_sizes: Tuple[int, ...] = (64, 64)
    rnd_normalize_inputs: bool = True

    # Training parameters (SB3 PPO defaults unless overridden)
    algorithm: str = "PPO"
    total_timesteps: int = 1_000_000
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

    # Logging / checkpointing
    seed: Optional[int] = None
    device: Union[str, torch.device] = "auto"
    verbose: int = 1
    save_freq: int = 0
    eval_episodes: int = 10

    # Domain-specific overrides
    policy_type: str = "MlpPolicy"
    policy_kwargs: Optional[Dict[str, Any]] = None
    normalize_obs: bool = False
    normalize_reward: bool = False

    # Trajectory collection for critical-state extraction
    n_collection_episodes: int = 50
    collection_deterministic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
        }


def default_refine_config(domain: str = "mujoco", env_id: Optional[str] = None) -> RefineConfig:
    """Return a domain-specific default refining configuration.

    The values are taken from the RICE paper (Table 3) where available.
    """
    cfg = RefineConfig()
    if domain == "mujoco":
        cfg.p = 0.25
        cfg.lambda_coef = 0.01
        cfg.total_timesteps = 1_000_000
        cfg.learning_rate = 3e-5  # lower LR for fine-tuning
        cfg.n_steps = 2048
        cfg.batch_size = 64
        cfg.n_epochs = 10
        if env_id in {"Walker2d-v3", "HalfCheetah-v3", "Walker2d-v4", "HalfCheetah-v4"}:
            cfg.normalize_obs = True
    elif domain == "selfish_mining":
        cfg.p = 0.5
        cfg.lambda_coef = 0.1
        cfg.total_timesteps = 500_000
        cfg.learning_rate = 3e-5
    elif domain == "cage":
        cfg.p = 0.25
        cfg.lambda_coef = 0.01
        cfg.total_timesteps = 300_000
        cfg.learning_rate = 3e-5
    elif domain == "metadrive":
        cfg.p = 0.5
        cfg.lambda_coef = 0.1
        cfg.total_timesteps = 1_000_000
        cfg.learning_rate = 3e-5
    elif domain == "malware":
        cfg.p = 0.5
        cfg.lambda_coef = 0.1
        cfg.total_timesteps = 200_000
        cfg.learning_rate = 3e-5
    else:
        warnings.warn(f"No default refine config for domain '{domain}'; using generic defaults.")
    return cfg


class RNDAugmentedEnv(gym.Wrapper):
    """Single-env wrapper that augments each step reward with an RND bonus.

    The wrapper is intentionally lightweight: it does not train the RND
    predictor; it only computes ``r + λ b_RND(s)`` using the current predictor.
    The predictor is updated by the refining loop between PPO updates.
    """

    def __init__(
        self,
        env: gym.Env,
        rnd_module: RNDModule,
        lambda_coef: float = 0.01,
        device: Union[str, torch.device] = "auto",
    ):
        super().__init__(env)
        self.rnd_module = rnd_module
        self.lambda_coef = lambda_coef
        self.device = device
        self._rnd_wrapper = RNDRewardWrapper(rnd_module, lambda_coef=lambda_coef, device=device)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.rnd_module.update_obs_stats(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rnd_module.update_obs_stats(obs)
        aug_reward = self._rnd_wrapper.augment_reward(obs, reward)
        info["rnd_bonus"] = float(aug_reward - reward)
        info["original_reward"] = float(reward)
        return obs, aug_reward, terminated, truncated, info


class RNDVecEnvWrapper(VecEnv):
    """Vectorized wrapper that augments rewards with an RND bonus.

    This mirrors :class:`RNDAugmentedEnv` but operates on an SB3 ``VecEnv``.
    """

    def __init__(
        self,
        venv: VecEnv,
        rnd_module: RNDModule,
        lambda_coef: float = 0.01,
        device: Union[str, torch.device] = "auto",
    ):
        self.venv = venv
        self.rnd_module = rnd_module
        self.lambda_coef = lambda_coef
        self.device = device
        self._rnd_wrapper = RNDRewardWrapper(rnd_module, lambda_coef=lambda_coef, device=device)
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )

    def reset(self):
        obs = self.venv.reset()
        self.rnd_module.update_obs_stats(obs)
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.rnd_module.update_obs_stats(obs)
        aug_rewards = self._rnd_wrapper.augment_reward(obs, rewards)
        for i, info in enumerate(infos):
            info["rnd_bonus"] = float(aug_rewards[i] - rewards[i])
            info["original_reward"] = float(rewards[i])
        return obs, aug_rewards, dones, infos

    def close(self) -> None:
        self.venv.close()

    def env_is_wrapped(self, wrapper_class, indices_to_check=None):
        return self.venv.env_is_wrapped(wrapper_class, indices_to_check)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)

    def get_attr(self, attr_name: str, indices=None):
        return self.venv.get_attr(attr_name, indices)

    def set_attr(self, attr_name: str, value, indices=None):
        return self.venv.set_attr(attr_name, value, indices)


def _build_refine_env(
    base_env: gym.Env,
    critical_buffer: CriticalStateBuffer,
    config: RefineConfig,
    rnd_module: Optional[RNDModule] = None,
) -> gym.Env:
    """Wrap ``base_env`` with mixed-initial-state and optional RND reward."""
    env = make_resettable(base_env, p=config.p, critical_buffer=critical_buffer)
    if config.use_rnd and rnd_module is not None:
        env = RNDAugmentedEnv(env, rnd_module, lambda_coef=config.lambda_coef, device=config.device)
    return env


def _build_vec_refine_env(
    make_env_fn: Callable[[], gym.Env],
    critical_buffer: CriticalStateBuffer,
    config: RefineConfig,
    rnd_module: Optional[RNDModule] = None,
    n_envs: int = 1,
) -> VecEnv:
    """Build a vectorized refining environment."""
    def _make() -> gym.Env:
        env = make_env_fn()
        env = make_resettable(env, p=config.p, critical_buffer=critical_buffer)
        if config.use_rnd and rnd_module is not None:
            env = RNDAugmentedEnv(env, rnd_module, lambda_coef=config.lambda_coef, device=config.device)
        return env

    venv = DummyVecEnv([_make for _ in range(n_envs)])
    if config.use_rnd and rnd_module is not None:
        venv = RNDVecEnvWrapper(venv, rnd_module, lambda_coef=config.lambda_coef, device=config.device)
    return venv


def extract_critical_states_for_refining(
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    env: gym.Env,
    config: Optional[RefineConfig] = None,
) -> CriticalStateBuffer:
    """Collect target-policy rollouts and build a critical-state buffer.

    Parameters
    ----------
    target_agent : TargetAgent
        Pre-trained policy used to generate trajectories.
    mask_net : MaskNetwork
        Trained mask network that scores state criticality.
    env : gym.Env
        Environment instance for rollout collection.
    config : RefineConfig, optional
        Refining configuration. If None, defaults are used.

    Returns
    -------
    CriticalStateBuffer
        Buffer populated with the selected critical states.
    """
    config = config or RefineConfig()
    trajectories = collect_masked_rollouts(
        env=env,
        target_agent=target_agent,
        mask_net=mask_net,
        n_episodes=config.n_collection_episodes,
        alpha=0.0,  # no perturbation during collection
        deterministic_target=config.collection_deterministic,
    )
    critical_states = extract_critical_states(
        trajectories,
        top_k=config.top_k_critical,
        percentile=config.critical_percentile,
        include_simulator_state=True,
    )
    buffer = CriticalStateBuffer(capacity=config.top_k_critical)
    for state in critical_states:
        buffer.add(state)
    return buffer


def refine_agent(
    env: gym.Env,
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    config: Optional[RefineConfig] = None,
    save_dir: Optional[Union[str, Path]] = None,
    callback: Optional[BaseCallback] = None,
) -> TargetAgent:
    """Run the full RICE refining pipeline.

    Parameters
    ----------
    env : gym.Env
        Base environment (will be wrapped with mixed-initial-state and RND).
    target_agent : TargetAgent
        Pre-trained target policy to refine.
    mask_net : MaskNetwork
        Trained mask network for critical-state extraction.
    config : RefineConfig, optional
        Refining hyper-parameters.
    save_dir : str or Path, optional
        Directory where refined checkpoints and buffers are saved.
    callback : BaseCallback, optional
        SB3 callback passed to the trainer.

    Returns
    -------
    TargetAgent
        The refined target agent.
    """
    config = config or RefineConfig()
    if config.seed is not None:
        set_random_seed(config.seed)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build critical-state buffer from target-policy rollouts.
    critical_buffer = extract_critical_states_for_refining(
        target_agent=target_agent,
        mask_net=mask_net,
        env=env,
        config=config,
    )
    if save_dir is not None:
        critical_buffer.save(save_dir / "critical_state_buffer.pkl")

    # 2. Build RND module if requested.
    rnd_module = None
    if config.use_rnd:
        rnd_module = make_rnd_module(
            observation_space=env.observation_space,
            output_dim=config.rnd_output_dim,
            hidden_sizes=config.rnd_hidden_sizes,
            normalize_inputs=config.rnd_normalize_inputs,
            device=config.device,
        )

    # 3. Wrap environment with mixed-initial-state and RND.
    train_env = _build_refine_env(env, critical_buffer, config, rnd_module)

    # 4. Continue training with SB3 PPO from the target agent's policy.
    #    We reuse the backend model if it is an SB3 algorithm; otherwise we
    #    initialize a fresh PPO model with the same policy architecture.
    backend = target_agent.backend_model
    if isinstance(backend, BaseAlgorithm) and config.algorithm.upper() == backend.__class__.__name__.upper():
        model = backend
        model.set_env(train_env)
        # Optionally reset learning rate and other train hyper-parameters.
        model.learning_rate = config.learning_rate
        model.n_steps = config.n_steps
        model.batch_size = config.batch_size
        model.n_epochs = config.n_epochs
        model.gamma = config.gamma
        model.gae_lambda = config.gae_lambda
        model.clip_range = config.clip_range
        model.ent_coef = config.ent_coef
        model.vf_coef = config.vf_coef
        model.max_grad_norm = config.max_grad_norm
        model.verbose = config.verbose
        model.seed = config.seed
        model.device = config.device
    else:
        algo_class = PPO if config.algorithm.upper() == "PPO" else SAC
        model = algo_class(
            config.policy_type,
            train_env,
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
            policy_kwargs=config.policy_kwargs,
            verbose=config.verbose,
            seed=config.seed,
            device=config.device,
        )
        # Try to load policy weights from the target agent.
        if hasattr(target_agent.policy, "state_dict"):
            try:
                model.policy.load_state_dict(target_agent.policy.state_dict(), strict=False)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Could not load target policy weights into refined model: {exc}")

    # 5. Train. We interleave PPO updates with RND predictor updates by using a
    #    custom callback if RND is enabled.
    callbacks = []
    if rnd_module is not None:
        callbacks.append(_RNDUpdateCallback(rnd_module, train_env))
    if config.save_freq > 0 and save_dir is not None:
        callbacks.append(CheckpointCallback(save_freq=config.save_freq, save_path=str(save_dir / "checkpoints")))
    if callback is not None:
        callbacks.append(callback)
    callback_obj = callbacks[0] if len(callbacks) == 1 else (callbacks if callbacks else None)

    model.learn(total_timesteps=config.total_timesteps, callback=callback_obj, reset_num_timesteps=False)

    if save_dir is not None:
        model.save(save_dir / "refined_model.zip")
        if rnd_module is not None:
            rnd_module.save(save_dir / "rnd_module.pt")

    refined_agent = TargetAgent(
        policy=model.policy,
        env=train_env,
        backend_model=model,
        deterministic=True,
    )
    return refined_agent


class _RNDUpdateCallback(BaseCallback):
    """SB3 callback that updates the RND predictor after each rollout."""

    def __init__(
        self,
        rnd_module: RNDModule,
        train_env: Union[gym.Env, VecEnv],
        n_gradient_steps: int = 4,
        learning_rate: float = 1e-4,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.rnd_module = rnd_module
        self.train_env = train_env
        self.n_gradient_steps = n_gradient_steps
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.rnd_module.predictor.parameters(), lr=learning_rate)

    def _on_rollout_end(self) -> None:
        # Gather recent observations from the rollout buffer if available.
        observations = None
        if (
            self.model is not None
            and hasattr(self.model, "rollout_buffer")
            and self.model.rollout_buffer is not None
        ):
            observations = self.model.rollout_buffer.observations
            if observations is not None:
                # SB3 stores observations as (n_steps, n_envs, *obs_shape) or flattened.
                obs_tensor = torch.as_tensor(observations, device=self.rnd_module.device)
                if obs_tensor.dim() > 2:
                    obs_tensor = obs_tensor.reshape(-1, obs_tensor.shape[-1])
                for _ in range(self.n_gradient_steps):
                    self.optimizer.zero_grad()
                    loss = self.rnd_module.predictor_loss(obs_tensor)
                    loss.backward()
                    self.optimizer.step()

    def _on_step(self) -> bool:
        return True


def refine_agent_vec(
    make_env_fn: Callable[[], gym.Env],
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    config: Optional[RefineConfig] = None,
    save_dir: Optional[Union[str, Path]] = None,
    n_envs: int = 1,
    callback: Optional[BaseCallback] = None,
) -> TargetAgent:
    """Vectorized version of :func:`refine_agent`.

    Useful for MuJoCo experiments where SB3 PPO benefits from multiple
    parallel workers.
    """
    config = config or RefineConfig()
    if config.seed is not None:
        set_random_seed(config.seed)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Collect critical states using a single env.
    collection_env = make_env_fn()
    critical_buffer = extract_critical_states_for_refining(
        target_agent=target_agent,
        mask_net=mask_net,
        env=collection_env,
        config=config,
    )
    if save_dir is not None:
        critical_buffer.save(save_dir / "critical_state_buffer.pkl")

    rnd_module = None
    if config.use_rnd:
        rnd_module = make_rnd_module(
            observation_space=collection_env.observation_space,
            output_dim=config.rnd_output_dim,
            hidden_sizes=config.rnd_hidden_sizes,
            normalize_inputs=config.rnd_normalize_inputs,
            device=config.device,
        )

    train_env = _build_vec_refine_env(make_env_fn, critical_buffer, config, rnd_module, n_envs=n_envs)

    backend = target_agent.backend_model
    if isinstance(backend, BaseAlgorithm) and config.algorithm.upper() == backend.__class__.__name__.upper():
        model = backend
        model.set_env(train_env)
        model.learning_rate = config.learning_rate
        model.n_steps = config.n_steps
        model.batch_size = config.batch_size
        model.n_epochs = config.n_epochs
        model.gamma = config.gamma
        model.gae_lambda = config.gae_lambda
        model.clip_range = config.clip_range
        model.ent_coef = config.ent_coef
        model.vf_coef = config.vf_coef
        model.max_grad_norm = config.max_grad_norm
        model.verbose = config.verbose
        model.seed = config.seed
        model.device = config.device
    else:
        algo_class = PPO if config.algorithm.upper() == "PPO" else SAC
        model = algo_class(
            config.policy_type,
            train_env,
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
            policy_kwargs=config.policy_kwargs,
            verbose=config.verbose,
            seed=config.seed,
            device=config.device,
        )
        if hasattr(target_agent.policy, "state_dict"):
            try:
                model.policy.load_state_dict(target_agent.policy.state_dict(), strict=False)
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"Could not load target policy weights into refined model: {exc}")

    callbacks = []
    if rnd_module is not None:
        callbacks.append(_RNDUpdateCallback(rnd_module, train_env))
    if config.save_freq > 0 and save_dir is not None:
        callbacks.append(CheckpointCallback(save_freq=config.save_freq, save_path=str(save_dir / "checkpoints")))
    if callback is not None:
        callbacks.append(callback)
    callback_obj = callbacks[0] if len(callbacks) == 1 else (callbacks if callbacks else None)

    model.learn(total_timesteps=config.total_timesteps, callback=callback_obj, reset_num_timesteps=False)

    if save_dir is not None:
        model.save(save_dir / "refined_model.zip")
        if rnd_module is not None:
            rnd_module.save(save_dir / "rnd_module.pt")

    refined_agent = TargetAgent(
        policy=model.policy,
        env=train_env,
        backend_model=model,
        deterministic=True,
    )
    return refined_agent


def load_refined_agent(
    model_path: Union[str, Path],
    env: gym.Env,
    rnd_path: Optional[Union[str, Path]] = None,
    algorithm: str = "PPO",
    device: Union[str, torch.device] = "auto",
) -> TargetAgent:
    """Load a refined agent checkpoint produced by :func:`refine_agent`."""
    algo_class = PPO if algorithm.upper() == "PPO" else SAC
    model = algo_class.load(model_path, env=env, device=device)
    rnd_module = None
    if rnd_path is not None:
        rnd_module = RNDModule(
            observation_space=env.observation_space,
            output_dim=64,
            hidden_sizes=(64, 64),
            normalize_inputs=True,
        )
        rnd_module.load(rnd_path)
    agent = TargetAgent(policy=model.policy, env=env, backend_model=model, deterministic=True)
    return agent
