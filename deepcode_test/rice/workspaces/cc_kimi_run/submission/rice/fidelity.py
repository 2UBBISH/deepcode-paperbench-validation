"""Fidelity score computation for explanation methods."""
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from rice.env_utils import sample_random_action
from rice.explanations import ExplanationMethod


def sample_trajectory(
    env: gym.Env,
    policy: Any,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> Dict[str, np.ndarray]:
    """Sample a single trajectory from the target policy.

    Also records simulator states (e.g. MuJoCo qpos/qvel) when available so that
    fidelity evaluation can restore exact states.
    """
    obs, _ = env.reset()
    observations: list[np.ndarray] = [obs.copy()]
    states: list[np.ndarray] = []
    if hasattr(env.unwrapped, "state"):
        states.append(env.unwrapped.state().copy())
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in range(max_steps):
        if hasattr(policy, "predict"):
            action, _ = policy.predict(obs, deterministic=deterministic)
        elif hasattr(policy, "act"):
            action = policy.act(obs, deterministic=deterministic)
        else:
            action = policy(obs)
        action = np.asarray(action).reshape(env.action_space.shape)
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs.copy())
        if hasattr(env.unwrapped, "state"):
            states.append(env.unwrapped.state().copy())
        actions.append(action.copy())
        rewards.append(reward)
        if terminated or truncated:
            break
    result: Dict[str, np.ndarray] = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "total_reward": float(sum(rewards)),
    }
    if states:
        result["states"] = np.array(states, dtype=np.float32)
    return result


def compute_fidelity_score(
    env: gym.Env,
    policy: Any,
    explanation: ExplanationMethod,
    d_max: float,
    k: float = 0.1,
    n_steps: Optional[int] = None,
    n_trajectories: int = 500,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Compute the fidelity score of an explanation method.

    For each trajectory, identify the most critical segment of consecutive steps,
    replace the target agent's actions with random actions in that segment, and
    measure the change in the final episode reward.

    Args:
        env: The environment.
        policy: The target policy.
        explanation: The explanation method to evaluate.
        d_max: Maximum possible reward change in a single episode.
        k: Fraction of trajectory length used as critical window size.
        n_steps: Exact window size (overrides k).
        n_trajectories: Number of trajectories to evaluate.
        max_steps: Maximum trajectory length.
        deterministic: Whether to sample actions deterministically.

    Returns:
        Dictionary containing mean fidelity score, std, and raw deltas.
    """
    fidelity_scores: list[float] = []
    deltas: list[float] = []

    for _ in range(n_trajectories):
        traj = sample_trajectory(env, policy, max_steps=max_steps, deterministic=deterministic)
        observations = traj["observations"]
        actions = traj["actions"]
        original_reward = traj["total_reward"]
        states = traj.get("states")
        T = len(actions)

        if n_steps is None:
            l_steps = max(1, int(T * k))
        else:
            l_steps = n_steps
        l_steps = min(l_steps, T)

        # Identify critical segment.
        selected = explanation.identify_critical_steps(observations[:-1], n_steps=l_steps)
        critical_indices = np.where(selected)[0]
        if len(critical_indices) == 0:
            continue

        # Fast-forward to start of segment and take random actions.
        start_idx = critical_indices[0]
        env.reset()
        obs = observations[start_idx]
        # For MuJoCo, directly set simulator state if available.
        if states is not None and hasattr(env.unwrapped, "set_state"):
            env.unwrapped.set_state(states[start_idx])
            obs = env.unwrapped._get_obs()
        elif states is not None and hasattr(env.unwrapped, "state"):
            env.unwrapped.state()[:] = states[start_idx]
            obs = env.unwrapped._get_obs()
        else:
            # Otherwise replay actions up to the critical point.
            obs, _ = env.reset()
            for i in range(start_idx):
                obs, _, terminated, truncated, _ = env.step(actions[i])
                if terminated or truncated:
                    break

        randomized_reward = 0.0
        alive = True
        for i in range(start_idx, min(start_idx + l_steps, T)):
            if not alive:
                break
            random_action = sample_random_action(env)
            obs, reward, terminated, truncated, _ = env.step(random_action)
            randomized_reward += reward
            alive = not (terminated or truncated)

        # Continue with target policy until the episode ends.
        step = start_idx + l_steps
        while alive and step < T:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=deterministic)
            elif hasattr(policy, "act"):
                action = policy.act(obs, deterministic=deterministic)
            else:
                action = policy(obs)
            action = np.asarray(action).reshape(env.action_space.shape)
            obs, reward, terminated, truncated, _ = env.step(action)
            randomized_reward += reward
            alive = not (terminated or truncated)
            step += 1

        delta = abs(randomized_reward - original_reward)
        deltas.append(delta)
        # Avoid log(0).
        if delta <= 0 or d_max <= 0:
            continue
        score = np.log(delta / d_max) - np.log(l_steps / T)
        fidelity_scores.append(score)

    return {
        "mean": float(np.mean(fidelity_scores)) if fidelity_scores else float("nan"),
        "std": float(np.std(fidelity_scores)) if fidelity_scores else float("nan"),
        "deltas": np.array(deltas),
        "scores": np.array(fidelity_scores),
    }
