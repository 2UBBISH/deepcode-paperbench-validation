"""RICE refining algorithm (Algorithm 2)."""
import copy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rice.env_utils import StateResetWrapper, sample_random_action
from rice.mask_network import MaskNetwork
from rice.rnd import RNDBonus
from rice.utils import get_device


class RICERefiningEnv(gym.Wrapper):
    """Gym environment implementing RICE mixed resets and RND exploration.

    This wrapper can be used directly with Stable-Baselines3 PPO so that the
    standard .learn() loop automatically uses:
      - a mixed initial state distribution (default initial states with
        probability 1-p, critical states identified by the mask network with
        probability p); and
      - an exploration bonus based on Random Network Distillation.
    """

    def __init__(
        self,
        env: gym.Env,
        policy: Any,
        mask_net: MaskNetwork,
        p: float = 0.25,
        lambda_rnd: float = 0.01,
        trajectory_length: int = 1000,
        rnd_hidden_sizes: Tuple[int, ...] = (64, 64),
        rnd_lr: float = 1e-4,
        rnd_update_interval: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(env)
        self.state_reset_env = StateResetWrapper(env)
        self.policy = policy
        self.mask_net = mask_net
        self.p = p
        self.lambda_rnd = lambda_rnd
        self.trajectory_length = trajectory_length
        self.rnd_update_interval = rnd_update_interval
        self.device = device or get_device()

        obs_dim = int(np.prod(env.observation_space.shape))
        self.rnd = RNDBonus(
            obs_dim=obs_dim,
            lr=rnd_lr,
            hidden_sizes=rnd_hidden_sizes,
            device=self.device,
        )
        self._step_count = 0

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_count = 0
        obs, info = self._mixed_reset()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        update_rnd = (self._step_count % self.rnd_update_interval) == 0
        rnd_bonus, _ = self.rnd.compute_and_update(
            next_obs[np.newaxis, ...], update=update_rnd
        )
        total_reward = reward + self.lambda_rnd * float(rnd_bonus[0])

        info["env_reward"] = reward
        info["rnd_bonus"] = float(rnd_bonus[0])
        return next_obs, total_reward, terminated, truncated, info

    def _mixed_reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        if np.random.rand() < self.p:
            critical_state, _ = self._identify_critical_state()
            try:
                return self.state_reset_env.reset_to_state(critical_state)
            except Exception:
                # Fallback to default reset if state reset is unavailable.
                return self.env.reset()
        else:
            return self.env.reset()

    def _identify_critical_state(self) -> Tuple[np.ndarray, int]:
        """Run the current policy and return the most critical visited state.

        For simulators that expose an internal state vector (e.g. MuJoCo's
        qpos/qvel), the returned state is the simulator state so that it can be
        restored exactly via set_state. Otherwise the observation is returned and
        the environment will fall back to replay or default reset.
        """
        obs, _ = self.env.reset()
        observations: List[np.ndarray] = [obs.copy()]
        states: List[Any] = []
        if hasattr(self.env.unwrapped, "state"):
            states.append(self.env.unwrapped.state().copy())

        for step in range(self.trajectory_length):
            if hasattr(self.policy, "predict"):
                action, _ = self.policy.predict(obs, deterministic=True)
            elif hasattr(self.policy, "act"):
                action = self.policy.act(obs, deterministic=True)
            else:
                action = self.policy(obs)
            action = np.asarray(action).reshape(self.env.action_space.shape)
            next_obs, _, terminated, truncated, _ = self.env.step(action)
            obs = next_obs
            observations.append(obs.copy())
            if hasattr(self.env.unwrapped, "state"):
                states.append(self.env.unwrapped.state().copy())
            if terminated or truncated:
                break

        observations_arr = np.array(observations, dtype=np.float32)
        importances = self.mask_net.importance_scores(observations_arr)
        critical_step = int(np.argmax(importances))
        if states:
            return np.array(states[critical_step], dtype=np.float32), critical_step
        return observations_arr[critical_step], critical_step


def refine_rice(
    env: gym.Env,
    policy: PPO,
    mask_net: MaskNetwork,
    total_timesteps: int,
    p: float = 0.25,
    lambda_rnd: float = 0.01,
    alpha: float = 1e-4,
    trajectory_length: int = 1000,
    rnd_hidden_sizes: Tuple[int, ...] = (64, 64),
    rnd_lr: float = 1e-4,
    **learn_kwargs: Any,
) -> PPO:
    """Refine a pre-trained PPO policy with RICE.

    Args:
        env: Base gym environment.
        policy: Pre-trained PPO policy.
        mask_net: Trained mask network.
        total_timesteps: Total number of refining timesteps.
        p: Probability of resetting to a critical state.
        lambda_rnd: Weight of the RND exploration bonus.
        alpha: Mask-network blinding bonus (kept for interface compatibility).
        trajectory_length: Length of the trajectory used to identify critical states.
        rnd_hidden_sizes: Hidden layer sizes of the RND networks.
        rnd_lr: Learning rate of the RND predictor.
        **learn_kwargs: Additional arguments passed to PPO.learn().

    Returns:
        The refined PPO policy.
    """
    # Build the RICE-wrapped environment.
    rice_env = RICERefiningEnv(
        env,
        policy=policy,
        mask_net=mask_net,
        p=p,
        lambda_rnd=lambda_rnd,
        trajectory_length=trajectory_length,
        rnd_hidden_sizes=rnd_hidden_sizes,
        rnd_lr=rnd_lr,
    )
    vec_env = DummyVecEnv([lambda: rice_env])

    # Create a fresh PPO learner on the RICE environment with the same hyperparameters.
    refined_policy = PPO(
        policy=policy.policy.__class__,
        env=vec_env,
        learning_rate=policy.learning_rate,
        n_steps=policy.n_steps,
        batch_size=policy.batch_size,
        n_epochs=policy.n_epochs,
        gamma=policy.gamma,
        gae_lambda=policy.gae_lambda,
        clip_range=policy.clip_range,
        ent_coef=policy.ent_coef,
        vf_coef=policy.vf_coef,
        max_grad_norm=policy.max_grad_norm,
        verbose=policy.verbose,
        seed=policy.seed,
        device=policy.device,
    )
    # Initialize from the pre-trained policy.
    refined_policy.set_parameters(policy.get_parameters(), exact_match=True)
    refined_policy.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, **learn_kwargs)
    return refined_policy
