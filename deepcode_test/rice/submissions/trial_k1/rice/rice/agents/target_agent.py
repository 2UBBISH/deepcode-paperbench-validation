"""Target-agent training wrappers for RICE.

This module trains the pre-trained target policies :math:`\\pi` used by RICE.
It is intentionally backend-agnostic at the interface level: MuJoCo agents use
Stable-Baselines3 PPO, while other domains can supply custom trainers.
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
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize


@dataclass
class TargetAgentConfig:
    """Hyper-parameters for training a target agent."""

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
    normalize_obs: bool = False
    normalize_reward: bool = False
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    device: Union[str, torch.device] = "auto"
    eval_episodes: int = 10
    save_freq: int = 0
    verbose: int = 1


class TargetAgent:
    """Wrapper around a trained RL policy used as the RICE target agent.

    The wrapper exposes a Gymnasium-compatible ``predict`` interface, methods to
    save/load checkpoints, and helpers to collect trajectory rollouts for the
    explanation and refining stages.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Optional[gym.Env] = None,
        backend_model: Optional[Any] = None,
        deterministic: bool = True,
    ):
        self.policy = policy
        self.env = env
        self.backend_model = backend_model
        self.deterministic = deterministic
        self._device = getattr(policy, "device", torch.device("cpu"))

    # ------------------------------------------------------------------
    # Prediction interface
    # ------------------------------------------------------------------
    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: Optional[bool] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return action and next recurrent state for *observation*."""
        det = self.deterministic if deterministic is None else deterministic
        if hasattr(self.policy, "predict"):
            return self.policy.predict(observation, state=state, deterministic=det)
        # Fallback for raw torch modules
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self._device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs_t)
        return action.cpu().numpy(), state

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Convenience callable returning the action only."""
        action, _ = self.predict(observation)
        return action

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path], save_env: bool = True) -> None:
        """Persist the policy and optional environment normalizer."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        policy_path = path / "policy.zip"
        if self.backend_model is not None and hasattr(self.backend_model, "save"):
            self.backend_model.save(str(policy_path))
        elif hasattr(self.policy, "save"):
            self.policy.save(str(policy_path))
        else:
            torch.save(self.policy.state_dict(), path / "policy_state.pth")

        meta = {"deterministic": self.deterministic}
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        if save_env and self.env is not None:
            try:
                env_path = path / "env.pkl"
                with open(env_path, "wb") as f:
                    pickle.dump(self.env, f)
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"Could not pickle environment: {exc}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        algorithm: str = "PPO",
        device: Union[str, torch.device] = "auto",
    ) -> "TargetAgent":
        """Load a persisted target agent."""
        path = Path(path)
        policy_path = path / "policy.zip"
        meta_path = path / "meta.pkl"

        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            deterministic = meta.get("deterministic", True)
        else:
            deterministic = True

        algo_cls = {"PPO": PPO, "SAC": SAC}.get(algorithm.upper(), PPO)
        if policy_path.exists():
            model = algo_cls.load(str(policy_path), env=env, device=device)
            return cls(model.policy, env=env, backend_model=model, deterministic=deterministic)

        state_path = path / "policy_state.pth"
        if state_path.exists():
            raise NotImplementedError(
                "Loading from raw state dict requires explicit policy constructor."
            )
        raise FileNotFoundError(f"No policy found at {path}")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------
    def collect_rollouts(
        self,
        env: Optional[gym.Env] = None,
        n_episodes: int = 10,
        max_steps: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Collect *n_episodes* trajectories using the target policy.

        Each trajectory is a dictionary with keys:
        ``observations``, ``actions``, ``rewards``, ``dones``, ``infos``,
        ``states`` (optional simulator state), ``total_reward``.
        """
        env = env if env is not None else self.env
        if env is None:
            raise ValueError("TargetAgent.collect_rollouts requires an environment.")

        det = self.deterministic if deterministic is None else deterministic
        trajectories: List[Dict[str, Any]] = []
        for _ in range(n_episodes):
            obs, info = env.reset()
            obs_seq, act_seq, rew_seq, done_seq, info_seq, state_seq = (
                [obs],
                [],
                [],
                [],
                [info],
                [],
            )
            total_reward = 0.0
            steps = 0
            while True:
                action, _ = self.predict(obs, deterministic=det)
                action = np.asarray(action).reshape(env.action_space.shape)
                state_seq.append(self._capture_simulator_state(env))
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                obs_seq.append(obs)
                act_seq.append(action)
                rew_seq.append(reward)
                done_seq.append(terminated or truncated)
                info_seq.append(info)
                total_reward += reward
                if terminated or truncated or (max_steps is not None and steps >= max_steps):
                    break
            trajectories.append(
                {
                    "observations": np.array(obs_seq),
                    "actions": np.array(act_seq),
                    "rewards": np.array(rew_seq),
                    "dones": np.array(done_seq),
                    "infos": info_seq,
                    "states": state_seq,
                    "total_reward": total_reward,
                }
            )
        return trajectories

    @staticmethod
    def _capture_simulator_state(env: gym.Env) -> Any:
        """Best-effort capture of simulator state for resetting later."""
        unwrapped = env.unwrapped
        # MuJoCo-style
        if hasattr(unwrapped, "get_state"):
            try:
                return unwrapped.get_state()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "get_state"):
            try:
                return unwrapped.sim.get_state()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(unwrapped, "data") and hasattr(unwrapped.data, "qpos"):
            try:
                return (copy.deepcopy(unwrapped.data.qpos), copy.deepcopy(unwrapped.data.qvel))
            except Exception:  # noqa: BLE001
                pass
        return None


# ----------------------------------------------------------------------
# Training helpers
# ----------------------------------------------------------------------
def _make_vec_env(env: gym.Env, seed: Optional[int] = None) -> VecEnv:
    """Wrap a single Gymnasium env in a DummyVecEnv."""

    def _init() -> gym.Env:
        if seed is not None:
            env.reset(seed=seed)
        return env

    return DummyVecEnv([_init])


def train_target_agent_sb3(
    env: gym.Env,
    config: Optional[TargetAgentConfig] = None,
    save_dir: Optional[Union[str, Path]] = None,
    algorithm: str = "PPO",
    policy_type: str = "MlpPolicy",
    callback: Optional[BaseCallback] = None,
) -> TargetAgent:
    """Train a target agent with Stable-Baselines3.

    Parameters
    ----------
    env:
        A Gymnasium environment.
    config:
        Training hyper-parameters. Defaults are chosen for MuJoCo dense tasks.
    save_dir:
        If provided, the final model is saved under ``save_dir/target_agent/``.
    algorithm:
        ``PPO`` or ``SAC``.
    policy_type:
        SB3 policy string (e.g. ``MlpPolicy``).
    callback:
        Optional SB3 callback.

    Returns
    -------
    TargetAgent
        The trained target agent wrapper.
    """
    config = config or TargetAgentConfig()
    if config.seed is not None:
        set_random_seed(config.seed)
        env.reset(seed=config.seed)

    algo_cls = {"PPO": PPO, "SAC": SAC}.get(algorithm.upper(), PPO)
    vec_env = _make_vec_env(env, seed=config.seed)

    # Optional SB3 VecNormalize for observation/reward normalization
    if config.normalize_obs or config.normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=config.normalize_obs,
            norm_reward=config.normalize_reward,
        )

    policy_kwargs = copy.deepcopy(config.policy_kwargs)
    if "activation_fn" not in policy_kwargs:
        policy_kwargs["activation_fn"] = torch.nn.Tanh

    model = algo_cls(
        policy_type,
        vec_env,
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
        policy_kwargs=policy_kwargs,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    callbacks = []
    if config.save_freq > 0 and save_dir is not None:
        callbacks.append(
            CheckpointCallback(
                save_freq=config.save_freq,
                save_path=str(Path(save_dir) / "checkpoints"),
                name_prefix="target_agent",
            )
        )
    if callback is not None:
        callbacks.append(callback)

    model.learn(total_timesteps=config.total_timesteps, callback=callbacks or None)

    agent = TargetAgent(model.policy, env=env, backend_model=model, deterministic=True)

    if save_dir is not None:
        agent.save(Path(save_dir) / "target_agent", save_env=True)
        if config.normalize_obs or config.normalize_reward:
            vec_env.save(str(Path(save_dir) / "target_agent" / "vecnormalize.pkl"))

    return agent


def train_target_agent_custom(
    env: gym.Env,
    trainer_fn: Callable[[gym.Env, Dict[str, Any]], Any],
    trainer_kwargs: Optional[Dict[str, Any]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    deterministic: bool = True,
) -> TargetAgent:
    """Train a target agent with a domain-specific trainer.

    ``trainer_fn(env, **trainer_kwargs)`` must return an object whose ``policy``
    attribute is a callable / SB3-compatible policy, or return the policy
    directly.
    """
    trainer_kwargs = trainer_kwargs or {}
    result = trainer_fn(env, **trainer_kwargs)
    if hasattr(result, "policy"):
        policy = result.policy
        backend = result
    else:
        policy = result
        backend = None
    agent = TargetAgent(policy, env=env, backend_model=backend, deterministic=deterministic)
    if save_dir is not None:
        agent.save(Path(save_dir) / "target_agent", save_env=True)
    return agent


def evaluate_target_agent(
    agent: TargetAgent,
    env: Optional[gym.Env] = None,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
) -> Tuple[float, float]:
    """Evaluate a target agent and return (mean_return, std_return)."""
    env = env if env is not None else agent.env
    if env is None:
        raise ValueError("evaluate_target_agent requires an environment.")

    # Use SB3 helper when possible
    if agent.backend_model is not None and hasattr(agent.backend_model, "get_env"):
        try:
            mean_reward, std_reward = evaluate_policy(
                agent.backend_model,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                render=render,
            )
            return float(mean_reward), float(std_reward)
        except Exception:  # noqa: BLE001
            pass

    returns = []
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def default_mujoco_config(env_id: str, sparse: bool = False) -> TargetAgentConfig:
    """Return a sensible default SB3 PPO config for MuJoCo tasks."""
    cfg = TargetAgentConfig()
    cfg.total_timesteps = 1_000_000
    cfg.n_steps = 2048
    cfg.batch_size = 64
    cfg.n_epochs = 10
    cfg.learning_rate = 3e-4
    cfg.normalize_obs = env_id in {"Walker2d-v3", "HalfCheetah-v3"}
    cfg.normalize_reward = False
    cfg.policy_kwargs = {"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}
    if sparse:
        # Sparse tasks may need longer horizons and a smaller learning rate.
        cfg.total_timesteps = 2_000_000
        cfg.learning_rate = 2.5e-4
    return cfg


def default_selfish_mining_config() -> TargetAgentConfig:
    """Return default config for the selfish-mining target agent."""
    cfg = TargetAgentConfig()
    cfg.total_timesteps = 500_000
    cfg.n_steps = 2048
    cfg.batch_size = 64
    cfg.n_epochs = 10
    cfg.learning_rate = 3e-4
    cfg.policy_kwargs = {"net_arch": [128, 128, 128, 128]}
    return cfg


def default_cage_config(trial_length: int = 50) -> TargetAgentConfig:
    """Return default config for the CAGE Challenge 2 blue agent."""
    cfg = TargetAgentConfig()
    cfg.total_timesteps = trial_length * 2000
    cfg.n_steps = trial_length * 10
    cfg.batch_size = trial_length
    cfg.n_epochs = 5
    cfg.learning_rate = 3e-4
    cfg.policy_kwargs = {"net_arch": [128, 128]}
    return cfg


def default_metadrive_config() -> TargetAgentConfig:
    """Return default config for the MetaDrive autonomous-driving agent."""
    cfg = TargetAgentConfig()
    cfg.total_timesteps = 1_000_000
    cfg.n_steps = 2048
    cfg.batch_size = 128
    cfg.n_epochs = 10
    cfg.learning_rate = 3e-4
    cfg.policy_kwargs = {"net_arch": [256, 256]}
    return cfg


def default_malware_config() -> TargetAgentConfig:
    """Return default config for the malware-mutation target agent."""
    cfg = TargetAgentConfig()
    cfg.total_timesteps = 200_000
    cfg.n_steps = 256
    cfg.batch_size = 64
    cfg.n_epochs = 10
    cfg.learning_rate = 3e-4
    cfg.policy_kwargs = {"net_arch": [128, 128]}
    return cfg
