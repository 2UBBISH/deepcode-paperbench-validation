"""State save/restore adapters.

RICE (Algorithm 2) and the fidelity metric (Experiment I) both need to put the
environment back into an *arbitrary previously visited state*: the refining
procedure resets the agent to a critical state, and the fidelity metric
fast-forwards the agent to the beginning of the most critical window.  The
paper follows (Ecoffet et al., 2019) and restores the simulator state directly;
this module implements that mechanism for every simulator used in the paper.

Every adapter exposes

    get_state()            -> hashable/copyable snapshot
    set_state(snapshot)    -> restore
    random_action()        -> uniform sample from the action space
    reset()/step()         -> thin passthrough to the wrapped env
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np


class StatefulEnv:
    """Generic wrapper adding state snapshotting on top of a gymnasium env."""

    #: whether :meth:`set_state` is implemented for this environment
    supports_state_restore = True

    def __init__(self, env, name: str = "env"):
        self.env = env
        self.name = name
        self._last_obs = None

    # ------------------------------------------------------------------ gym
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple):
            obs, info = out
        else:
            obs, info = out, {}
        self._last_obs = obs
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, rew, terminated, truncated, info = out
        else:
            obs, rew, terminated, info = out
            truncated = False
        self._last_obs = obs
        return obs, rew, terminated, truncated, info

    def current_obs(self) -> np.ndarray:
        """Observation of the current (possibly restored) simulator state."""
        env = self.env
        unwrapped = env.unwrapped if env is not None else self
        for name in ("_get_obs", "_obs"):
            getter = getattr(unwrapped, name, None)
            if callable(getter):
                try:
                    return np.asarray(getter(), dtype=np.float32)
                except TypeError:  # gymnasium MuJoCo takes an optional data argument
                    try:
                        return np.asarray(getter(None), dtype=np.float32)
                    except Exception:
                        pass
                except Exception:
                    pass
        if self._last_obs is not None:
            return np.asarray(self._last_obs, dtype=np.float32)
        raise RuntimeError("no observation available for {}".format(self.name))

    def close(self):
        self.env.close()

    def random_action(self):
        return self.action_space.sample()

    # ------------------------------------------------------------ snapshots
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError(
            "{} does not implement state snapshots".format(type(self).__name__)
        )

    def set_state(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError(
            "{} does not implement state snapshots".format(type(self).__name__)
        )


class MujocoStatefulEnv(StatefulEnv):
    """Snapshot/restore for MuJoCo environments (gymnasium ``* -v4`` ids).

    The full simulator state (``qpos``/``qvel``/``act`` plus the episode timer)
    is copied, which makes the environment behave exactly as it did when the
    snapshot was taken.  Environment specific extras (e.g. the random
    ``goal`` of ``Reacher``) are copied as well.
    """

    _EXTRA_ATTRS = ("goal", "target", "target_pos")

    def get_state(self) -> Dict[str, Any]:
        env = self.unwrapped
        data = env.data
        state: Dict[str, Any] = {
            "qpos": np.array(data.qpos, dtype=np.float64, copy=True),
            "qvel": np.array(data.qvel, dtype=np.float64, copy=True),
            "time": float(data.time),
        }
        act = getattr(data, "act", None)
        if act is not None and np.size(act) > 0:
            state["act"] = np.array(act, dtype=np.float64, copy=True)
        qacc_warmstart = getattr(data, "qacc_warmstart", None)
        if qacc_warmstart is not None and np.size(qacc_warmstart) > 0:
            state["qacc_warmstart"] = np.array(
                qacc_warmstart, dtype=np.float64, copy=True
            )
        mo = getattr(data, "mocap_pos", None)
        if mo is not None and np.size(mo) > 0:
            state["mocap_pos"] = np.array(mo, dtype=np.float64, copy=True)
            state["mocap_quat"] = np.array(
                data.mocap_quat, dtype=np.float64, copy=True
            )
        for attr in self._EXTRA_ATTRS:
            value = getattr(env, attr, None)
            if isinstance(value, np.ndarray):
                state[attr] = np.array(value, copy=True)
        steps = getattr(env, "_elapsed_steps", None)
        if steps is not None:
            state["steps"] = int(np.asarray(steps).reshape(-1)[0])
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        env = self.unwrapped
        env.set_state(np.array(state["qpos"]), np.array(state["qvel"]))
        data = env.data
        if "act" in state:
            data.act[:] = state["act"]
        data.time = state.get("time", 0.0)
        if "qacc_warmstart" in state:
            data.qacc_warmstart[:] = state["qacc_warmstart"]
        if "mocap_pos" in state:
            data.mocap_pos[:] = state["mocap_pos"]
            data.mocap_quat[:] = state["mocap_quat"]
        for attr in self._EXTRA_ATTRS:
            if attr in state:
                setattr(env, attr, np.array(state[attr], copy=True))
        if "steps" in state:
            env._elapsed_steps = state["steps"]
        if hasattr(env, "mujoco"):
            env.mujoco.mj_forward(env.model, env.data)


class CartPoleStatefulEnv(StatefulEnv):
    """Snapshot/restore for ``CartPole`` (used by the unit tests)."""

    def get_state(self) -> Dict[str, Any]:
        env = self.unwrapped
        return {
            "state": np.array(env.state, dtype=np.float64, copy=True),
            "steps": int(getattr(env, "_elapsed_steps", 0)),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        env = self.unwrapped
        env.state = np.array(state["state"], dtype=np.float64, copy=True)
        env._elapsed_steps = state.get("steps", 0)
        if hasattr(env, "steps_beyond_done"):
            env.steps_beyond_done = None


class DictStatefulEnv(StatefulEnv):
    """Generic deep-copy adapter.

    Works for pure-python environments that keep their whole state in
    ``env.unwrapped.__dict__`` (e.g. the selfish mining environment shipped in
    this repository).  Simulator backed environments whose state lives outside
    of python (CAGE / MetaDrive) need a dedicated adapter.
    """

    _SKIP = {"spec", "np_random", "_np_random"}

    def get_state(self) -> Dict[str, Any]:
        env = self.unwrapped
        state = {
            key: copy.deepcopy(value)
            for key, value in vars(env).items()
            if key not in self._SKIP
        }
        rng = getattr(env, "np_random", None)
        if rng is not None:
            state["__np_random__"] = copy.deepcopy(rng.bit_generator.state)
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        env = self.unwrapped
        rng_state = state.get("__np_random__")
        for key, value in state.items():
            if key == "__np_random__":
                continue
            setattr(env, key, copy.deepcopy(value))
        if rng_state is not None and getattr(env, "np_random", None) is not None:
            env.np_random.bit_generator.state = copy.deepcopy(rng_state)


class ActionReplayStatefulEnv(DictStatefulEnv):
    """Fallback for environments without a direct state setter.

    The state is represented by the *sequence of actions* that was executed
    since the last ``reset``.  Restoring replays that sequence deterministically.
    Only works when the environment is deterministic, and is used for the
    third-party simulators (CAGE Challenge 2, MetaDrive) where MuJoCo-style
    snapshotting is unavailable.
    """

    supports_state_restore = True

    def __init__(self, env, name: str = "env", seed: Optional[int] = None):
        super().__init__(env, name=name)
        self._seed = seed
        self._action_log: list = []
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._action_log = []
        self._last_obs = obs
        return obs, info

    def step(self, action):
        out = super().step(action)
        self._action_log.append(copy.deepcopy(action))
        self._last_obs = out[0]
        return out

    def get_state(self) -> Dict[str, Any]:
        return {"actions": copy.deepcopy(self._action_log)}

    def set_state(self, state: Dict[str, Any]) -> None:
        actions = state["actions"]
        obs, _ = self.env.reset(seed=self._seed)
        for action in actions:
            obs, *_ = self.env.step(action)
        self._action_log = copy.deepcopy(actions)
        self._last_obs = obs
