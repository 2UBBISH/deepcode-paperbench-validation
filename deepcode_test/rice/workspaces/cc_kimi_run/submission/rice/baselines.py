"""Baseline refining methods for comparison with RICE."""
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from rice.env_utils import StateResetWrapper
from rice.explanations import ExplanationMethod
from rice.mask_network import MaskNetwork


def ppo_finetune(
    env: gym.Env,
    policy: PPO,
    total_timesteps: int,
    learning_rate: Optional[float] = None,
    **learn_kwargs: Any,
) -> PPO:
    """Baseline: continue PPO training with a lower learning rate.

    This corresponds to the "PPO fine-tuning" baseline in the paper.
    """
    if learning_rate is not None:
        policy.learning_rate = learning_rate
    policy.set_env(env)
    policy.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, **learn_kwargs)
    return policy


def statemask_r_finetune(
    env: gym.Env,
    policy: PPO,
    explanation: ExplanationMethod,
    total_timesteps: int,
    trajectory_length: int = 1000,
    **learn_kwargs: Any,
) -> PPO:
    """Baseline: StateMask refining by resetting to critical states.

    This method always resets the environment to a critical state identified by
    the explanation method and continues fine-tuning from there.
    """
    wrapped_env = StateResetWrapper(env)

    class CriticalStateEnv(gym.Wrapper):
        """Wrapper that resets to a critical state each episode."""

        def __init__(self, env: gym.Env) -> None:
            super().__init__(env)
            self.explanation = explanation
            self.trajectory_length = trajectory_length

        def reset(self, **kwargs: Any) -> Any:
            # Sample a trajectory with the current policy.
            obs, _ = self.env.reset(**kwargs)
            observations = [obs.copy()]
            states = []
            if hasattr(self.env.unwrapped, "state"):
                states.append(self.env.unwrapped.state().copy())
            actions = []
            for _ in range(self.trajectory_length):
                action, _ = policy.predict(obs, deterministic=True)
                action = np.asarray(action).reshape(self.env.action_space.shape)
                next_obs, _, terminated, truncated, _ = self.env.step(action)
                observations.append(next_obs.copy())
                if hasattr(self.env.unwrapped, "state"):
                    states.append(self.env.unwrapped.state().copy())
                actions.append(action.copy())
                obs = next_obs
                if terminated or truncated:
                    break
            observations_arr = np.array(observations, dtype=np.float32)
            scores = self.explanation.explain(observations_arr)
            critical_idx = int(np.argmax(scores))
            critical_state = (
                np.array(states[critical_idx], dtype=np.float32)
                if states
                else observations_arr[critical_idx]
            )
            try:
                return self.env.reset_to_state(critical_state)
            except Exception:
                # Fallback: replay actions.
                obs, info = self.env.reset()
                for i in range(min(critical_idx, len(actions))):
                    obs, _, term, trunc, _ = self.env.step(actions[i])
                    if term or trunc:
                        break
                return obs, info

    crit_env = CriticalStateEnv(wrapped_env)
    policy.set_env(crit_env)
    policy.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, **learn_kwargs)
    return policy


def jsrl_finetune(
    env: gym.Env,
    guide_policy: PPO,
    total_timesteps: int,
    horizon_schedule: Optional[Any] = None,
    **learn_kwargs: Any,
) -> PPO:
    """Baseline: Jump-Start Reinforcement Learning (JSRL).

    JSRL uses a guide policy to roll in and then lets an exploration policy
    take over. For refining, the exploration policy is initialized to the guide
    policy and gradually extends the exploration horizon.

    Args:
        env: Training environment.
        guide_policy: Pre-trained guide policy.
        total_timesteps: Total training timesteps.
        horizon_schedule: Optional schedule mapping timestep -> horizon length.
            If None, a linear schedule from full guide rollout to full
            exploration rollout is used.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Start exploration policy as a copy of the guide policy.
    exploration_policy = PPO(
        policy="MlpPolicy",
        env=DummyVecEnv([lambda: env]),
        verbose=guide_policy.verbose,
        seed=guide_policy.seed,
        device=guide_policy.device,
        learning_rate=guide_policy.learning_rate,
    )
    exploration_policy.set_parameters(guide_policy.get_parameters(), exact_match=True)

    max_horizon = 1000  # default episode length cap

    class JSRLEnv(gym.Wrapper):
        """Environment wrapper for JSRL curriculum."""

        def __init__(self, env: gym.Env) -> None:
            super().__init__(env)
            self.guide_policy = guide_policy
            self.exploration_policy = exploration_policy
            self.horizon = 0
            self.step_count = 0
            self.in_guide_phase = False

        def reset(self, **kwargs: Any) -> Any:
            self.step_count = 0
            self.in_guide_phase = True
            obs, info = self.env.reset(**kwargs)
            self._current_obs = obs
            return obs, info

        def step(self, action: np.ndarray) -> Any:
            if self.in_guide_phase:
                # During the guide phase we override the action with the guide's action.
                guide_action, _ = self.guide_policy.predict(self._current_obs, deterministic=True)
                action = np.asarray(guide_action).reshape(self.env.action_space.shape)
                self.step_count += 1
                if self.step_count >= self.horizon:
                    self.in_guide_phase = False
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._current_obs = obs
            return obs, reward, terminated, truncated, info

    jsrl_env = JSRLEnv(env)
    exploration_policy.set_env(jsrl_env)

    # Simple curriculum: linearly increase exploration horizon.
    n_iters = max(1, total_timesteps // 2048)
    for i in range(n_iters):
        if horizon_schedule is not None:
            jsrl_env.horizon = horizon_schedule(i)
        else:
            # Linearly decrease guide horizon to 0.
            jsrl_env.horizon = int(max_horizon * (1 - i / max(1, n_iters - 1)))
        exploration_policy.learn(
            total_timesteps=2048,
            reset_num_timesteps=False,
            **learn_kwargs,
        )
    return exploration_policy
