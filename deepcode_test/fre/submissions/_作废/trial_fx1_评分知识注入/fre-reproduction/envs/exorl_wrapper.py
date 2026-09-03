"""Environment wrapper and task-reward definitions for ExORL walker/cheetah.

The ExORL datasets used by FRE are exploratory DeepMind Control Suite replay
buffers for the ``walker`` and ``cheetah`` domains.  This module provides:

* vectorized downstream task-reward functions for the four ExORL evaluation
  tasks from the paper: ``walker-goals``, ``cheetah-goals``,
  ``walker-velocity`` and ``cheetah-velocity``;
* a thin dm_control wrapper that exposes a gym-like ``reset/step`` API and
  flattens DMC observations into numpy vectors;
* zero-shot evaluation helpers that mirror the AntMaze/Kitchen protocol.

The module is intentionally defensive about imports: if ``dm_control`` is not
installed (e.g. CPU-only unit-test environments), a small dummy environment is
used so evaluation code can still be imported and exercised.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional, used only for tensor -> numpy conversion
    import torch
except Exception:  # pragma: no cover - torch is expected in normal runs
    torch = None  # type: ignore


EXORL_TASKS: Tuple[str, ...] = (
    "walker-goals",
    "cheetah-goals",
    "walker-velocity",
    "cheetah-velocity",
)

# Sensible fallbacks used when a dm_control environment cannot be created.
# Real values are inferred from the environment when available.
_DOMAIN_FALLBACKS = {
    "walker": {"state_dim": 24, "action_dim": 6},
    "cheetah": {"state_dim": 18, "action_dim": 6},
}


def _to_numpy(x: Any) -> np.ndarray:
    """Convert torch tensors / lists to a float64 numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    elif not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return np.asarray(x, dtype=np.float64)


def _flatten_observation(observation: Any) -> np.ndarray:
    """Flatten a dm_control observation (array or OrderedDict) to 1D."""
    if isinstance(observation, dict):
        parts = []
        # OrderedDict order is stable, which matters for encoder inputs.
        for value in observation.values():
            arr = np.asarray(value, dtype=np.float64)
            parts.append(arr.reshape(-1))
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    return np.asarray(observation, dtype=np.float64).reshape(-1)


class ExORLReward:
    """Callable vectorized reward with optional success predicate."""

    def __init__(
        self,
        name: str,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        success_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.reward_fn = reward_fn
        self.success_fn = success_fn
        self.metadata = metadata or {}

    def __call__(self, states: Any) -> np.ndarray:
        states_np = _to_numpy(states)
        single = states_np.ndim == 1
        if single:
            states_np = states_np[None, :]
        rewards = np.asarray(self.reward_fn(states_np), dtype=np.float64)
        rewards = np.clip(rewards, -1.0, 1.0)
        if single:
            rewards = rewards[0]
        return rewards

    def success(self, states: Any) -> np.ndarray:
        if self.success_fn is None:
            return np.zeros_like(self(states), dtype=bool)
        states_np = _to_numpy(states)
        single = states_np.ndim == 1
        if single:
            states_np = states_np[None, :]
        out = np.asarray(self.success_fn(states_np), dtype=bool)
        return out[0] if single else out


class DummyExORLEnv:
    """Minimal gym-like fallback used when dm_control is unavailable."""

    def __init__(self, state_dim: int, action_dim: int, seed: int = 0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self._rng = np.random.RandomState(seed)
        self._state = np.zeros(state_dim, dtype=np.float64)

    def reset(self) -> np.ndarray:
        self._state = self._rng.uniform(-0.5, 0.5, size=self.state_dim)
        return self._state.copy()

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        self._state = 0.99 * self._state + 0.05 * action + self._rng.normal(
            0.0, 0.01, size=self.state_dim
        )
        self._state = np.clip(self._state, -5.0, 5.0)
        return self._state.copy(), 0.0, False, {}


class ExORLWrapper:
    """Gym-like wrapper for a DeepMind Control Suite walker/cheetah env."""

    def __init__(
        self,
        domain: str = "walker",
        task: Optional[str] = None,
        env: Any = None,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.domain = domain.lower()
        if self.domain not in _DOMAIN_FALLBACKS:
            raise ValueError(f"Unsupported ExORL domain: {domain!r}")

        if task is None:
            task = "walk" if self.domain == "walker" else "run"

        self._env: Any = None
        self._owns_env = env is None
        if env is not None:
            self._env = env
        else:
            try:
                from dm_control import suite  # type: ignore

                self._env = suite.load(domain=self.domain, task=task)
                if seed is not None:
                    self._env = self._env  # dm_control env has no global seed API
            except Exception:
                fb = _DOMAIN_FALLBACKS[self.domain]
                self._env = DummyExORLEnv(fb["state_dim"], fb["action_dim"])

        self.max_episode_steps = max_episode_steps or 1000

        # Infer dimensions; DMC observation is flattened below, so inspect it.
        try:
            if hasattr(self._env, "action_spec"):
                self._action_dim = int(self._env.action_spec().shape[0])
            else:
                self._action_dim = int(self._env.action_dim)
        except Exception:
            self._action_dim = _DOMAIN_FALLBACKS[self.domain]["action_dim"]

        try:
            if hasattr(self._env, "reset"):
                ts = self._env.reset()
                obs = getattr(ts, "observation", ts)
                self._state_dim = int(_flatten_observation(obs).shape[0])
            else:
                self._state_dim = _DOMAIN_FALLBACKS[self.domain]["state_dim"]
        except Exception:
            self._state_dim = _DOMAIN_FALLBACKS[self.domain]["state_dim"]

        self._step_count = 0

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def env(self) -> Any:
        return self._env

    def reset(self) -> np.ndarray:
        self._step_count = 0
        try:
            time_step = self._env.reset()
            obs = getattr(time_step, "observation", time_step)
        except TypeError:
            # Dummy env reset accepts no args
            obs = self._env.reset()
        return _flatten_observation(obs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        action_np = np.asarray(action, dtype=np.float64)
        self._step_count += 1
        try:
            time_step = self._env.step(action_np)
            obs = getattr(time_step, "observation", time_step)
            reward = float(getattr(time_step, "reward", 0.0) or 0.0)
            done = bool(getattr(time_step, "last", False))
            info: Dict[str, Any] = {"time_step": time_step}
        except TypeError:
            obs, reward, done, info = self._env.step(action_np)
        if self.max_episode_steps is not None and self._step_count >= self.max_episode_steps:
            done = True
        return _flatten_observation(obs), float(reward), done, info

    def get_state(self) -> np.ndarray:
        return self.reset()

    def close(self) -> None:
        if hasattr(self._env, "close"):
            try:
                self._env.close()
            except Exception:
                pass

    def evaluate_policy(
        self,
        policy_fn: Callable[[np.ndarray], Any],
        task_name: str = "walker-goals",
        num_episodes: int = 20,
        seed: Optional[int] = None,
        goal: Optional[np.ndarray] = None,
        velocity_dim: Optional[int] = None,
        target_velocity: float = 1.0,
    ) -> Dict[str, float]:
        """Evaluate a policy under an ExORL downstream reward.

        The environment's native reward is ignored; rewards are computed by
        ``make_exorl_task_reward`` from the observed state, matching the FRE
        zero-shot evaluation protocol.
        """
        rng = np.random.RandomState(seed)
        reward_obj = make_exorl_task_reward(
            task_name,
            goal=goal,
            velocity_dim=velocity_dim,
            target_velocity=target_velocity,
        )

        episode_returns: list[float] = []
        successes: list[bool] = []
        for _ in range(num_episodes):
            obs = self.reset()
            done = False
            ep_return = 0.0
            ep_success = False
            ep_steps = 0
            while not done:
                action = policy_fn(obs)
                if isinstance(action, tuple):
                    action = action[0]
                obs, _, done, _ = self.step(action)
                reward = float(np.asarray(reward_obj(obs), dtype=np.float64).reshape(-1)[0])
                ep_return += reward
                if reward_obj.success_fn is not None:
                    ep_success = ep_success or bool(
                        np.asarray(reward_obj.success(obs)).reshape(-1)[0]
                    )
                ep_steps += 1
            episode_returns.append(ep_return)
            successes.append(ep_success)

        mean_return = float(np.mean(episode_returns))
        if reward_obj.success_fn is not None:
            score = float(np.mean(successes)) * 100.0
        else:
            # Velocity rewards are normalized to a target speed, clipped to [0, 1].
            score = float(np.clip(mean_return / max(target_velocity, 1e-6), 0.0, 1.0)) * 100.0

        return {
            "score": score,
            "mean_return": mean_return,
            "success_rate": float(np.mean(successes)),
            "num_episodes": num_episodes,
            "task": task_name,
        }


def make_exorl_task_reward(
    task_name: str,
    goal: Optional[Any] = None,
    epsilon: float = 0.5,
    velocity_dim: Optional[int] = None,
    target_velocity: float = 1.0,
) -> ExORLReward:
    """Build a vectorized downstream reward for an ExORL task.

    Goal-reaching tasks use a sparse reward: ``0`` within ``epsilon`` of the
    goal and ``-1`` otherwise.  Velocity tasks use a clipped signed velocity
    reward normalized by ``target_velocity`` (default 1.0 m/s).
    """
    task_name = task_name.lower()
    domain = "walker" if task_name.startswith("walker") else "cheetah"
    kind = "goals" if task_name.endswith("goals") else "velocity"

    if kind == "goals":
        goal_np = _to_numpy(goal) if goal is not None else None

        def reward_fn(states: np.ndarray) -> np.ndarray:
            if goal_np is None:
                g = np.zeros(states.shape[1], dtype=np.float64)
            else:
                g = goal_np
            dist = np.linalg.norm(states - g.reshape(1, -1), axis=1)
            return np.where(dist <= epsilon, 0.0, -1.0)

        def success_fn(states: np.ndarray) -> np.ndarray:
            if goal_np is None:
                g = np.zeros(states.shape[1], dtype=np.float64)
            else:
                g = goal_np
            dist = np.linalg.norm(states - g.reshape(1, -1), axis=1)
            return dist <= epsilon

        return ExORLReward(
            task_name,
            reward_fn,
            success_fn,
            metadata={"domain": domain, "kind": "goals", "epsilon": epsilon},
        )

    def velocity_reward_fn(states: np.ndarray) -> np.ndarray:
        if states.shape[1] == 0:
            return np.zeros(states.shape[0], dtype=np.float64)
        if velocity_dim is None:
            # DMC stores qpos followed by qvel; the first qvel coordinate is
            # usually the root horizontal velocity for both walker and cheetah.
            vdim = states.shape[1] // 2
        else:
            vdim = int(velocity_dim)
        vdim = min(max(vdim, 0), states.shape[1] - 1)
        velocity = states[:, vdim]
        return np.clip(velocity / max(target_velocity, 1e-6), -1.0, 1.0)

    return ExORLReward(
        task_name,
        velocity_reward_fn,
        success_fn=None,
        metadata={
            "domain": domain,
            "kind": "velocity",
            "velocity_dim": velocity_dim,
            "target_velocity": target_velocity,
        },
    )


def evaluate_exorl_policy(
    policy_fn: Callable[[np.ndarray], Any],
    task_name: str = "walker-goals",
    domain: Optional[str] = None,
    num_episodes: int = 20,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    goal: Optional[Any] = None,
    velocity_dim: Optional[int] = None,
    target_velocity: float = 1.0,
) -> Dict[str, float]:
    """Create an ExORL wrapper and evaluate ``policy_fn`` on ``task_name``."""
    if domain is None:
        domain = "walker" if task_name.startswith("walker") else "cheetah"
    wrapper = ExORLWrapper(
        domain=domain,
        max_episode_steps=max_episode_steps,
        seed=seed,
    )
    try:
        return wrapper.evaluate_policy(
            policy_fn,
            task_name=task_name,
            num_episodes=num_episodes,
            seed=seed,
            goal=goal,
            velocity_dim=velocity_dim,
            target_velocity=target_velocity,
        )
    finally:
        wrapper.close()


def sample_task_reward_states(
    task_name: str,
    state_pool: Any,
    num_examples: int = 32,
    seed: Optional[int] = None,
    goal: Optional[Any] = None,
    velocity_dim: Optional[int] = None,
    target_velocity: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample exactly ``num_examples`` state-reward pairs for FRE encoding."""
    rng = np.random.RandomState(seed)
    pool = _to_numpy(state_pool)
    if pool.ndim == 1:
        pool = pool[None, :]
    if len(pool) == 0:
        raise ValueError("Cannot sample task reward states from an empty state pool")
    idx = rng.randint(0, len(pool), size=num_examples)
    states = pool[idx]
    reward_obj = make_exorl_task_reward(
        task_name,
        goal=goal,
        velocity_dim=velocity_dim,
        target_velocity=target_velocity,
    )
    rewards = np.asarray(reward_obj(states), dtype=np.float64)
    return states, rewards
