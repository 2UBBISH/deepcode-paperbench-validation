"""MetaDrive autonomous-driving environment adapter for RICE.

This module provides a Gymnasium-compatible wrapper around the MetaDrive
simulator (``metadrive``) for the Macro-v1 scenario used in the paper.
The action space is normalized to :math:`[-1, 1]^2` and mapped to
steering / acceleration / brake as required by MetaDrive.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Soft import so that missing metadrive does not break import-time.
try:
    from metadrive import MacroMap
    from metadrive.envs import MetaDriveMacroEnv

    _METADRIVE_AVAILABLE = True
except Exception as _import_exc:  # pragma: no cover
    _METADRIVE_AVAILABLE = False
    MacroMap = None  # type: ignore
    MetaDriveMacroEnv = None  # type: ignore


class MetaDriveMacroAdapter(gym.Env):
    """Gymnasium adapter for MetaDrive Macro-v1 used in RICE.

    The underlying MetaDrive environment expects actions in a vehicle-native
    format.  RICE normalizes the action to a 2-D Box in :math:`[-1, 1]` and
    maps the two dimensions to steering and acceleration/brake respectively.

    Parameters
    ----------
    use_render : bool, optional
        Whether to enable MetaDrive rendering.  Default ``False``.
    map_config : Any, optional
        Macro map configuration.  If ``None`` the default ``MacroMap`` is used.
    traffic_density : float, optional
        Density of traffic vehicles.  Default ``0.1``.
    num_scenarios : int, optional
        Number of scenario seeds to cycle through.  Default ``1``.
    start_seed : int, optional
        Random seed for scenario generation.  Default ``0``.
    accident_prob : float, optional
        Probability of traffic accidents.  Default ``0.0``.
    decision_repeat : int, optional
        Number of simulation steps per agent decision.  Default ``5``.
    action_check : bool, optional
        Whether MetaDrive should check action validity.  Default ``False``.
    random_traffic : bool, optional
        Whether traffic is randomized.  Default ``True``.
    debug : bool, optional
        MetaDrive debug flag.  Default ``False``.
    cull_scene : bool, optional
        Whether to cull distant objects.  Default ``True``.
    manual_control : bool, optional
        Whether to accept keyboard control.  Default ``False``.
    use_chase_camera : bool, optional
        Use chase camera in rendering.  Default ``False``.
    max_steps : int, optional
        Episode horizon.  Default ``1000``.
    **kwargs
        Additional keyword arguments forwarded to ``MetaDriveMacroEnv``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        use_render: bool = False,
        map_config: Any = None,
        traffic_density: float = 0.1,
        num_scenarios: int = 1,
        start_seed: int = 0,
        accident_prob: float = 0.0,
        decision_repeat: int = 5,
        action_check: bool = False,
        random_traffic: bool = True,
        debug: bool = False,
        cull_scene: bool = True,
        manual_control: bool = False,
        use_chase_camera: bool = False,
        max_steps: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not _METADRIVE_AVAILABLE:
            raise ImportError(
                "MetaDrive is required for the autonomous-driving domain. "
                "Install it via: pip install metadrive-simulator"
            ) from _import_exc

        self._use_render = use_render
        self._max_steps = max_steps
        self._elapsed_steps = 0

        if map_config is None:
            map_config = MacroMap if MacroMap is not None else None

        config: Dict[str, Any] = {
            "use_render": use_render,
            "traffic_density": traffic_density,
            "num_scenarios": num_scenarios,
            "start_seed": start_seed,
            "accident_prob": accident_prob,
            "decision_repeat": decision_repeat,
            "action_check": action_check,
            "random_traffic": random_traffic,
            "debug": debug,
            "cull_scene": cull_scene,
            "manual_control": manual_control,
            "use_chase_camera": use_chase_camera,
            "map": map_config,
        }
        # Forward any additional user-provided config keys.
        config.update(kwargs)

        self._env = MetaDriveMacroEnv(config)

        # RICE uses a normalized 2-D continuous action space.
        # Dimension 0 -> steering, dimension 1 -> acceleration/brake.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Observation space: MetaDrive returns a vector observation by default
        # for Macro-v1.  We mirror the underlying space but force float32.
        obs_space = self._env.observation_space
        if isinstance(obs_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=np.asarray(obs_space.low, dtype=np.float32),
                high=np.asarray(obs_space.high, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = obs_space

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Clip action to [-1, 1]^2 and return as float32 array."""
        action = np.asarray(action, dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized action to MetaDrive vehicle action format.

        MetaDrive vehicle action is typically ``[steering, throttle, brake]``
        or ``{"steering": ..., "throttle": ..., "brake": ...}``.  We use the
        array form: ``[steering, throttle_or_brake]`` where the second
        dimension is positive for throttle and negative for brake.
        """
        action = self._normalize_action(action)
        steering = action[0]
        throttle_brake = action[1]
        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        return np.array([steering, throttle, brake], dtype=np.float32)

    def _process_obs(self, obs: Any) -> np.ndarray:
        """Ensure observation is a float32 numpy array."""
        if isinstance(obs, dict):
            # If MetaDrive ever returns a dict, flatten the vector part.
            obs = obs.get("state", obs)
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._env.seed(seed)
            self._env.reset_config(seed)

        self._elapsed_steps = 0
        obs = self._env.reset()
        obs = self._process_obs(obs)
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        meta_action = self._convert_action(action)
        obs, reward, terminated, info = self._env.step(meta_action)
        obs = self._process_obs(obs)

        self._elapsed_steps += 1
        truncated = bool(self._elapsed_steps >= self._max_steps)

        # Ensure info is a dict and record the normalized action for debugging.
        if not isinstance(info, dict):
            info = {"raw_info": info}
        info["normalized_action"] = self._normalize_action(action).copy()

        return obs, float(reward), bool(terminated), truncated, info

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        if mode is None:
            mode = "human" if self._use_render else "rgb_array"
        try:
            return self._env.render(mode=mode)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"MetaDrive render failed: {exc}")
            return None

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Simulator state capture / restore for critical-state resetting
    # ------------------------------------------------------------------
    def get_simulator_state(self) -> Dict[str, Any]:
        """Return a best-effort snapshot of the MetaDrive simulator state."""
        state: Dict[str, Any] = {
            "elapsed_steps": self._elapsed_steps,
            "seed": getattr(self._env, "current_seed", None),
        }
        # MetaDrive exposes engine state via the underlying engine manager.
        engine = getattr(self._env, "engine", None)
        if engine is not None:
            state["engine"] = getattr(engine, "get_state", lambda: None)()
        # Fallback: capture vehicle state if available.
        vehicle = getattr(self._env, "vehicle", None)
        if vehicle is not None:
            state["vehicle"] = {
                "position": getattr(vehicle, "position", None),
                "heading": getattr(vehicle, "heading_theta", None),
                "velocity": getattr(vehicle, "velocity", None),
            }
        return state

    def set_simulator_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously captured simulator state.

        MetaDrive does not expose a public full-state setter, so this is a
        best-effort implementation that resets the scenario to the stored seed
        and restores the elapsed-step counter.
        """
        self._elapsed_steps = state.get("elapsed_steps", 0)
        seed = state.get("seed")
        if seed is not None:
            self._env.reset_config(seed)
            self._env.reset(force_seed=seed)


def make_metadrive_env(**kwargs: Any) -> MetaDriveMacroAdapter:
    """Factory that creates a :class:`MetaDriveMacroAdapter`."""
    return MetaDriveMacroAdapter(**kwargs)
