"""Autonomous-driving environment wrapper for RICE (MetaDrive ``Macro-v1``).

Paper reference
---------------
§C.2 "Autonomous Driving" (verbatim, relevant parts):

    "One representative driving simulator is MetaDrive (Li et al., 2022).  A DRL
    agent is trained to guide a vehicle safely and efficiently to travel to its
    destination.  MetaDrive converts the Birds Eye View (BEV) of the road
    conditions and the sensor information such as the vehicle's steering,
    direction, velocity, and relative distance to traffic lanes into a vector
    representation of the current state.  The policy network takes this state
    vector as input and yields driving actions, including accelerating, braking,
    and steering commands.  MetaDrive employs a set of reward functions to shape
    the learning process.  For instance, penalties are assigned when the agent
    collides with other vehicles or drives out of the road boundary.  To promote
    smooth and efficient driving, MetaDrive also incorporates rewards to encourage
    forward motion and the maintenance of an appropriate speed."

    "We select the "Macro-v1" environment powered by the MetaDrive simulator
    (Li et al., 2022).  The goal of the agent is to learn a deep policy to
    successfully cross the car flow and reach the destination.  We train the
    target agent and our mask network by the PPO algorithm following the
    implementation of DI-drive (drive Contributors, 2021).  The environment
    receives normalized action to control the target agent
    a = [a1, a2] \in [-1, 1]^2.  The action vector a will then be converted to
    the steering (degree), acceleration (hp), and brake signal (hp)."

Design notes
------------
Two back-ends are supported, mirroring the pattern already used by
``selfish_mining.py`` / ``cage_challenge2.py``:

* ``backend="di_drive"`` / ``"metadrive"`` -- thin adapter over MetaDrive
  (``metadrive.envs.MetaDriveEnv``) with the DI-drive "Macro-v1" configuration
  (dense traffic, 2-D normalised action ``[a1, a2]``).  Selected automatically
  when MetaDrive is importable *and* ``RICE_ALLOW_METADRIVE=1`` (or
  ``RICE_USE_REAL_METADRIVE=1``); otherwise the simulator is used.
* ``backend="sim"`` (default) -- a self-contained, dependency-free
  pure-Python/Numpy driving simulator with the same interface contract:
  egocentric BEV vector observation, 2-D normalised continuous action mapped to
  (steering, acceleration, brake), collision / off-road penalties, forward and
  speed-maintenance shaping rewards, and full simulator state save/restore for
  the Go-Explore style reset used by RICE's refining loop (§C.1).

The paper does *not* specify the observation encoding, the exact reward
coefficients, the traffic-density parameters or the horizon; these are
documented defaults below (see ``AutoDrivingConfig``) and flagged in the README.

Only ``numpy`` is required at import time -- MetaDrive/DI-drive are imported
lazily inside :func:`_build_metadrive_env`.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# shared plumbing (tolerant gym/gymnasium detection + fallbacks)
# --------------------------------------------------------------------------
try:  # pragma: no cover - import shim
    from ._common import (  # type: ignore
        EnvBase,
        GYM_AVAILABLE,
        IS_GYMNASIUM,
        WrapperBase,
        env_max_episode_steps,
        import_running_mean_std,
        make_box,
        normalize_reset,
        normalize_step,
    )
except ImportError:  # pragma: no cover
    from rice.environments._common import (  # type: ignore
        EnvBase,
        GYM_AVAILABLE,
        IS_GYMNASIUM,
        WrapperBase,
        env_max_episode_steps,
        import_running_mean_std,
        make_box,
        normalize_reset,
        normalize_step,
    )


__all__ = [
    "AutoDrivingConfig",
    "AutoDrivingEnv",
    "AutoDrivingObsNormalizeWrapper",
    "AUTODRIVING_SPECS",
    "MACRO_V1_OBS_DIM",
    "MACRO_V1_ACTION_DIM",
    "normalized_to_control",
    "control_to_normalized",
    "make_env",
    "make_autodriving",
    "make_autodriving_env",
    "make_metadrive_env",
    "obstacle_avoids_collision",
    "resolve_name",
    "get_spec",
    "resolve_backend",
    "metadrive_available",
]


# --------------------------------------------------------------------------
# Constants (§C.2)
# --------------------------------------------------------------------------
MACRO_V1_ACTION_DIM = 2          # a = [a1, a2] \in [-1, 1]^2
MACRO_V1_NET_ARCH: Tuple[int, ...] = (256, 256)

#: Steering (degree) limit used when converting the normalised action component
#: ``a1`` -> steering command.  MetaDrive's default vehicle steering range is
#: about +-40 degrees; the exact DI-drive constant is not stated in the paper.
MAX_STEERING_DEG = 40.0
#: Acceleration / brake gains for converting ``a2`` (hp).  Both are documented
#: defaults: ``a2 > 0`` maps to acceleration (hp) and ``a2 < 0`` to brake (hp).
MAX_ACCELERATION_HP = 300.0
MAX_BRAKE_HP = 150.0

#: Observation layout of the internal simulator (egocentric BEV vector).
#:   0-2   : normalised forward / lateral offset to the destination
#:   3-4   : current velocity (forward, lateral), normalised
#:   5     : steering angle (normalised by MAX_STEERING_DEG)
#:   6     : heading error to the destination
#:   7..   : per-obstacle [forward gap, lateral offset, relative speed]
OBSTACLE_FEATURES = 3
MACRO_V1_OBS_DIM = 7 + OBSTACLE_FEATURES * 5  # 7 + 15 = 22 (5 traffic cars)


# --------------------------------------------------------------------------
# Normalised <-> physical control conversion (§C.2)
# --------------------------------------------------------------------------
def normalized_to_control(
    action: Sequence[float],
    max_steering_deg: float = MAX_STEERING_DEG,
    max_acceleration_hp: float = MAX_ACCELERATION_HP,
    max_brake_hp: float = MAX_BRAKE_HP,
) -> Tuple[float, float, float]:
    """Convert a normalised action ``a=[a1, a2] in [-1, 1]^2`` to physical commands.

    Returns ``(steering_degree, acceleration_hp, brake_hp)`` exactly as described
    in §C.2: the action vector "will then be converted to the steering (degree),
    acceleration (hp), and brake signal (hp)".

    ``a1`` (steering) is symmetric in ``[-1, 1]``; ``a2`` (longitudinal) is split
    into a positive acceleration branch and a negative brake branch so that a
    single scalar drives both pedals (MetaDrive/DI-drive convention).
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.size < 2:
        raise ValueError(
            "Autonomous driving action must have two components [a1, a2] in [-1, 1]^2, "
            f"got action with {a.size} element(s)."
        )
    a1 = float(np.clip(a[0], -1.0, 1.0))
    a2 = float(np.clip(a[1], -1.0, 1.0))

    steering_deg = a1 * float(max_steering_deg)
    if a2 >= 0.0:
        acceleration_hp = a2 * float(max_acceleration_hp)
        brake_hp = 0.0
    else:
        acceleration_hp = 0.0
        brake_hp = -a2 * float(max_brake_hp)
    return steering_deg, acceleration_hp, brake_hp


def control_to_normalized(
    steering_deg: float,
    acceleration_hp: float = 0.0,
    brake_hp: float = 0.0,
    max_steering_deg: float = MAX_STEERING_DEG,
    max_acceleration_hp: float = MAX_ACCELERATION_HP,
    max_brake_hp: float = MAX_BRAKE_HP,
) -> np.ndarray:
    """Inverse of :func:`normalized_to_control` (used by tests / round trips)."""
    a1 = np.clip(steering_deg / max(max_steering_deg, 1e-8), -1.0, 1.0)
    if brake_hp > 0.0:
        a2 = -np.clip(brake_hp / max(max_brake_hp, 1e-8), 0.0, 1.0)
    else:
        a2 = np.clip(acceleration_hp / max(max_acceleration_hp, 1e-8), -1.0, 1.0)
    return np.asarray([a1, a2], dtype=np.float32)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class AutoDrivingConfig:
    """Configuration of the ``Macro-v1`` autonomous-driving task.

    Fields marked *unspecified* do not appear in the paper (Source: §C.2 only
    fixes "Macro-v1", the normalised 2-D action and the general reward shaping).
    They carry documented defaults and are listed as unspecified in the README.
    """

    # --- task / episode -------------------------------------------------
    env_name: str = "Macro-v1"          # §C.2
    max_episode_steps: int = 1000       # unspecified (MetaDrive Macro-v1 default)
    n_obstacles: int = 5                # unspecified (traffic density)
    road_length: float = 300.0          # unspecified (destination distance, metres)
    lane_width: float = 3.5             # unspecified (standard highway lane)
    n_lanes: int = 3                    # unspecified
    init_speed: float = 10.0            # unspecified (m/s)
    dt: float = 0.1                     # unspecified (10 Hz control)
    max_speed: float = 30.0             # m/s
    vehicle_length: float = 4.5         # unspecified (metres)
    vehicle_width: float = 2.0          # unspecified
    safe_gap: float = 8.0               # unspecified (metres)

    # --- traffic ---------------------------------------------------------
    traffic_speed_range: Tuple[float, float] = (6.0, 14.0)  # unspecified
    traffic_lateral_jitter: float = 0.3                      # unspecified
    spawn_radius: float = 260.0                              # unspecified

    # --- reward shaping (all coefficients unspecified) -------------------
    #: "penalties are assigned when the agent collides with other vehicles or
    #: drives out of the road boundary" (§C.2)
    collision_penalty: float = -10.0
    off_road_penalty: float = -5.0
    #: "rewards to encourage forward motion and the maintenance of an
    #: appropriate speed" (§C.2)
    forward_reward_weight: float = 0.5
    speed_reward_weight: float = 0.2
    target_speed: float = 15.0
    #: small per-step cost so that idling is not free
    step_cost: float = 0.01
    #: bonus for reaching the destination
    success_reward: float = 20.0
    out_of_route_penalty: float = -2.0

    # --- observation ------------------------------------------------------
    normalize_obs: bool = True          # per rice.environments registry
    obs_clip: float = 10.0
    obs_bias: float = 0.0               # additive bias keeps Box bounds finite

    # --- infrastructure ---------------------------------------------------
    backend: str = "auto"               # "auto" | "sim" | "metadrive" | "di_drive"
    action_mode: str = "continuous"     # §C.2: normalised 2-D continuous action
    render_mode: Optional[str] = None
    seed: Optional[int] = None
    net_arch: Tuple[int, ...] = MACRO_V1_NET_ARCH

    # --- DI-drive / MetaDrive passthrough (used only by the real backend) --
    metadrive_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["net_arch"] = tuple(self.net_arch)
        d["traffic_speed_range"] = tuple(self.traffic_speed_range)
        return d

    def clone(self, **overrides: Any) -> "AutoDrivingConfig":
        return replace(self, **overrides)


# --------------------------------------------------------------------------
# Observation normalization wrapper
# --------------------------------------------------------------------------
class AutoDrivingObsNormalizeWrapper(WrapperBase):
    """Running mean/std observation normalisation with serialisable statistics.

    The paper only says the target agent is trained "following the implementation
    of DI-drive" for this task; DI-drive normalises its vector observation, so we
    provide the same serialisable Welford normaliser used by the other RICE
    environment wrappers (important because RICE restores simulator states and
    replays observations during refining).
    """

    def __init__(
        self,
        env: Any,
        normalize: bool = True,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        update: bool = True,
    ) -> None:
        super().__init__(env)
        self.normalize = bool(normalize)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.update = bool(update)
        _RunningMeanStd = import_running_mean_std()
        obs_dim = int(np.prod(env.observation_space.shape))
        self.obs_rms = _RunningMeanStd(shape=(obs_dim,), epsilon=epsilon, clip=clip)
        self._obs_dim = obs_dim

    # -- plumbing --------------------------------------------------------
    def _flatten(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        if arr.size < self._obs_dim:
            arr = np.concatenate(
                [arr, np.zeros(self._obs_dim - arr.size, dtype=np.float32)]
            )
        return arr[: self._obs_dim]

    def normalize_obs(self, obs: Any, update: Optional[bool] = None) -> np.ndarray:
        arr = self._flatten(obs)
        if not self.normalize:
            return arr
        do_update = self.update if update is None else bool(update)
        return np.asarray(
            self.obs_rms.normalize(arr, clip=self.clip, update=do_update),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any):
        result = self.env.reset(**kwargs)
        obs, info = normalize_reset(result)
        return self.normalize_obs(obs), info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        return self.normalize_obs(obs), reward, terminated, truncated, info

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "normalize": self.normalize,
            "obs_rms": self.obs_rms.state_dict(),
            "update": self.update,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.normalize = bool(state.get("normalize", self.normalize))
        self.update = bool(state.get("update", self.update))
        if "obs_rms" in state and state["obs_rms"] is not None:
            try:
                self.obs_rms.load_state_dict(state["obs_rms"])
            except Exception:  # pragma: no cover - defensive
                pass


# --------------------------------------------------------------------------
# Internal simulator: egocentric BEV driving
# --------------------------------------------------------------------------
class _Obstacle:
    """A single traffic vehicle in the internal simulator."""

    __slots__ = ("x", "y", "speed", "length", "width", "alive")

    def __init__(
        self,
        x: float,
        y: float,
        speed: float,
        length: float = 4.5,
        width: float = 2.0,
        alive: bool = True,
    ) -> None:
        self.x = float(x)
        self.y = float(y)
        self.speed = float(speed)
        self.length = float(length)
        self.width = float(width)
        self.alive = bool(alive)

    # -- serialisation ---------------------------------------------------
    def to_state(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "length": self.length,
            "width": self.width,
            "alive": self.alive,
        }

    def from_state(self, state: Dict[str, Any]) -> None:
        self.x = float(state.get("x", self.x))
        self.y = float(state.get("y", self.y))
        self.speed = float(state.get("speed", self.speed))
        self.length = float(state.get("length", self.length))
        self.width = float(state.get("width", self.width))
        self.alive = bool(state.get("alive", self.alive))

    def clone(self) -> "_Obstacle":
        return _Obstacle(
            self.x, self.y, self.speed, self.length, self.width, self.alive
        )


class AutoDrivingEnv(EnvBase):
    """``Macro-v1`` MetaDrive task ("cross the car flow and reach the destination").

    The class hosts two implementations under one interface:

    * the built-in egocentric BEV simulator (default, always available), and
    * an optional thin adapter over MetaDrive's ``MetaDriveEnv`` chosen when
      MetaDrive is installed and explicitly enabled (``backend="metadrive"`` or
      ``RICE_ALLOW_METADRIVE=1``), which keeps the DI-drive "Macro-v1" flavour
      through ``AutoDrivingConfig.metadrive_config``.

    In both cases ``step`` takes the normalised action ``a = [a1, a2] in
    [-1, 1]^2`` and internally converts it to steering (degree), acceleration
    (hp) and brake (hp) per §C.2.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Optional[AutoDrivingConfig] = None,
        *,
        env_name: Optional[str] = None,
        max_episode_steps: Optional[int] = None,
        n_obstacles: Optional[int] = None,
        normalize_obs: Optional[bool] = None,
        backend: Optional[str] = None,
        action_mode: Optional[str] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = AutoDrivingConfig()
        overrides: Dict[str, Any] = {}
        for key, value in (
            ("env_name", env_name),
            ("max_episode_steps", max_episode_steps),
            ("n_obstacles", n_obstacles),
            ("normalize_obs", normalize_obs),
            ("backend", backend),
            ("action_mode", action_mode),
            ("render_mode", render_mode),
            ("seed", seed),
        ):
            if value is not None:
                overrides[key] = value
        # accept any other AutoDrivingConfig field as a keyword argument
        for key, value in kwargs.items():
            if hasattr(config, key):
                overrides[key] = value
        if overrides:
            config = config.clone(**overrides)

        self.config: AutoDrivingConfig = config
        self.env_name = config.env_name
        self.rise_canonical_name = "Macro-v1"
        self._max_steps = max(1, int(config.max_episode_steps))
        self.render_mode = config.render_mode

        self._backend, self._metadrive = _make_backend(config)

        # spaces ---------------------------------------------------------
        self.action_space = make_box(
            low=-np.ones(MACRO_V1_ACTION_DIM, dtype=np.float32),
            high=np.ones(MACRO_V1_ACTION_DIM, dtype=np.float32),
            dtype=np.float32,
        )
        if self._backend == "metadrive":
            self.observation_space = self._metadrive.observation_space
        else:
            high = np.full((MACRO_V1_OBS_DIM,), np.inf, dtype=np.float32)
            low = -high
            self.observation_space = make_box(low=low, high=high, dtype=np.float32)

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = MACRO_V1_ACTION_DIM

        # episode state ---------------------------------------------------
        self._rng = np.random.default_rng(config.seed)
        self.ego = {
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,          # radians, 0 = +x (forward)
            "speed": float(config.init_speed),
            "steering": 0.0,         # radians
            "lateral_speed": 0.0,
        }
        self.obstacles: List[_Obstacle] = []
        self.episode_step = 0
        self.episode_return = 0.0
        self._done = False
        self._last_action = np.zeros(MACRO_V1_ACTION_DIM, dtype=np.float32)
        self._last_control: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._start_lane = 1
        self._success = False

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def max_episode_steps(self) -> int:
        if self._backend == "metadrive":
            try:
                return int(env_max_episode_steps(self._metadrive, self._max_steps))
            except Exception:  # pragma: no cover
                pass
        return self._max_steps

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # core API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs: Any):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self.config.seed = int(seed)

        if self._backend == "metadrive":
            result = self._metadrive.reset()
            obs, info = normalize_reset(result)
            self.episode_step = 0
            self.episode_return = 0.0
            self._done = False
            self._success = False
            info = dict(info or {})
            info["rise_canonical_name"] = "Macro-v1"
            return obs, info

        cfg = self.config
        lane = int(self._rng.integers(0, max(1, cfg.n_lanes)))
        self._start_lane = lane
        self.ego = {
            "x": 0.0,
            "y": (lane - (cfg.n_lanes - 1) / 2.0) * cfg.lane_width,
            "heading": 0.0,
            "speed": float(cfg.init_speed),
            "steering": 0.0,
            "lateral_speed": 0.0,
        }
        self.obstacles = []
        for _ in range(int(cfg.n_obstacles)):
            self.obstacles.append(self._spawn_obstacle())
        self.episode_step = 0
        self.episode_return = 0.0
        self._done = False
        self._success = False
        self._last_action = np.zeros(MACRO_V1_ACTION_DIM, dtype=np.float32)
        self._last_control = (0.0, 0.0, 0.0)

        obs = self.observation()
        info = {
            "rise_canonical_name": "Macro-v1",
            "success": False,
            "episode_step": 0,
            "destination_x": float(cfg.road_length),
            "lane": lane,
        }
        return obs, info

    def step(self, action: Any):
        if self._backend == "metadrive":
            obs, reward, terminated, truncated, info = normalize_step(
                self._metadrive.step(self._convert_action(action))
            )
            self.episode_step += 1
            self.episode_return += float(reward)
            self._done = bool(terminated or truncated)
            info = dict(info or {})
            info["rise_canonical_name"] = "Macro-v1"
            return obs, float(reward), bool(terminated), bool(truncated), info

        cfg = self.config

        if self._done:
            # Gym semantics: stepping a finished env keeps returning done.
            obs = self.observation()
            return obs, 0.0, True, False, {"rise_canonical_name": "Macro-v1"}

        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size < MACRO_V1_ACTION_DIM:
            action = np.concatenate(
                [action, np.zeros(MACRO_V1_ACTION_DIM - action.size)]
            )
        action = action[:MACRO_V1_ACTION_DIM]
        action = np.where(np.isfinite(action), action, 0.0)
        action = np.clip(action, -1.0, 1.0)

        steering_deg, accel_hp, brake_hp = normalized_to_control(action)
        self._last_action = action.astype(np.float32)
        self._last_control = (steering_deg, accel_hp, brake_hp)

        # --- lateral dynamics (bicycle model, simplified) ---------------
        self.ego["steering"] = math.radians(steering_deg)
        yaw_rate = (
            self.ego["speed"] * math.tan(self.ego["steering"]) / max(cfg.vehicle_length, 1e-6)
        )
        self.ego["heading"] += yaw_rate * cfg.dt
        self.ego["lateral_speed"] = yaw_rate * cfg.vehicle_length * 0.5

        # --- longitudinal dynamics --------------------------------------
        # acceleration (hp) -> m/s^2 and brake (hp) -> m/s^2, both scaled.
        a_long = accel_hp / 30.0 - brake_hp / 20.0
        self.ego["speed"] = float(
            np.clip(self.ego["speed"] + a_long * cfg.dt, 0.0, cfg.max_speed)
        )

        # --- integrate position ------------------------------------------
        self.ego["x"] += self.ego["speed"] * math.cos(self.ego["heading"]) * cfg.dt
        self.ego["y"] += self.ego["speed"] * math.sin(self.ego["heading"]) * cfg.dt

        # --- traffic ------------------------------------------------------
        for ob in self.obstacles:
            ob.x += ob.speed * cfg.dt
            if ob.alive and ob.x - self.ego["x"] > cfg.spawn_radius:
                # recycle far-ahead traffic back behind the ego
                recycled = self._spawn_obstacle(behind=True)
                ob.from_state(recycled.to_state())

        # --- reward -------------------------------------------------------
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {"rise_canonical_name": "Macro-v1"}

        forward_progress = self.ego["speed"] * math.cos(self.ego["heading"]) * cfg.dt

        # (a) collision with another vehicle -> penalty + termination
        collided = self._collision()
        # (b) off-road / out-of-route -> penalty
        off_road = self._off_road()

        reward += cfg.forward_reward_weight * forward_progress
        reward -= cfg.speed_reward_weight * abs(self.ego["speed"] - cfg.target_speed) * cfg.dt
        reward -= cfg.step_cost

        if collided:
            reward += cfg.collision_penalty
            terminated = True
            info["collision"] = True
        if off_road:
            reward += cfg.off_road_penalty
            terminated = True
            info["off_road"] = True

        # (c) destination reached
        if self.ego["x"] >= cfg.road_length:
            reward += cfg.success_reward
            terminated = True
            self._success = True
            info["success"] = True

        self.episode_step += 1
        if self.episode_step >= self._max_steps:
            truncated = True
            info["time_limit"] = True

        self._done = bool(terminated or truncated)
        self.episode_return += float(reward)

        info.update(
            {
                "episode_step": self.episode_step,
                "speed": float(self.ego["speed"]),
                "steering_degree": float(steering_deg),
                "acceleration_hp": float(accel_hp),
                "brake_hp": float(brake_hp),
                "x": float(self.ego["x"]),
                "y": float(self.ego["y"]),
                "success": bool(self._success),
                "episode_return": float(self.episode_return),
            }
        )
        if terminated and not truncated:
            info["terminal_observation"] = self.observation()
        return self.observation(), float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------
    # observation construction
    # ------------------------------------------------------------------
    def observation(self) -> np.ndarray:
        if self._backend == "metadrive":
            try:
                return np.asarray(self._metadrive.observation(), dtype=np.float32)
            except Exception:  # pragma: no cover - defensive
                return np.zeros((MACRO_V1_OBS_DIM,), dtype=np.float32)

        cfg = self.config
        obs = np.zeros(MACRO_V1_OBS_DIM, dtype=np.float32)
        scale_pos = max(cfg.road_length, 1.0)

        obs[0] = (cfg.road_length - self.ego["x"]) / scale_pos      # forward distance
        obs[1] = self.ego["y"] / max(cfg.lane_width * cfg.n_lanes, 1.0)
        obs[2] = self.ego["x"] / scale_pos
        obs[3] = self.ego["speed"] / max(cfg.max_speed, 1e-6)
        obs[4] = self.ego["lateral_speed"] / max(cfg.max_speed, 1e-6)
        obs[5] = self.ego["steering"] / math.radians(MAX_STEERING_DEG)
        heading_err = math.atan2(-self.ego["y"], max(cfg.road_length - self.ego["x"], 1e-3))
        obs[6] = heading_err / math.pi

        idx = 7
        for ob in self._ordered_obstacles():
            if idx + OBSTACLE_FEATURES > MACRO_V1_OBS_DIM:
                break
            dx = ob.x - self.ego["x"]
            dy = ob.y - self.ego["y"]
            obs[idx] = np.clip(dx / max(cfg.spawn_radius, 1.0), -2.0, 2.0)
            obs[idx + 1] = np.clip(dy / max(cfg.lane_width * cfg.n_lanes, 1.0), -2.0, 2.0)
            obs[idx + 2] = (ob.speed - self.ego["speed"]) / max(cfg.max_speed, 1e-6)
            idx += OBSTACLE_FEATURES

        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)

    def _ordered_obstacles(self) -> List[_Obstacle]:
        """Traffic vehicles within the forward sensor cone, nearest first."""
        forward = [
            ob
            for ob in self.obstacles
            if ob.alive and (ob.x - self.ego["x"]) > -self.config.vehicle_length
        ]
        forward.sort(key=lambda ob: abs(ob.x - self.ego["x"]))
        return forward

    # ------------------------------------------------------------------
    # collision / road model
    # ------------------------------------------------------------------
    def _collision(self) -> bool:
        cfg = self.config
        for ob in self.obstacles:
            if not ob.alive:
                continue
            if abs(ob.x - self.ego["x"]) < (ob.length + cfg.vehicle_length) / 2.0 * 0.9:
                if abs(ob.y - self.ego["y"]) < (ob.width + cfg.vehicle_width) / 2.0 * 0.9:
                    return True
        return False

    def _off_road(self) -> bool:
        cfg = self.config
        half = cfg.lane_width * cfg.n_lanes / 2.0
        return bool(abs(self.ego["y"]) > half + cfg.lane_width / 2.0)

    def _spawn_obstacle(self, behind: bool = False) -> _Obstacle:
        cfg = self.config
        lane = int(self._rng.integers(0, max(1, cfg.n_lanes)))
        y = (lane - (cfg.n_lanes - 1) / 2.0) * cfg.lane_width
        if cfg.traffic_lateral_jitter > 0:
            y += float(
                self._rng.uniform(-cfg.traffic_lateral_jitter, cfg.traffic_lateral_jitter)
            )
        lo, hi = cfg.traffic_speed_range
        speed = float(self._rng.uniform(lo, hi))
        if behind:
            x = self.ego["x"] - float(self._rng.uniform(20.0, cfg.spawn_radius))
        else:
            x = self.ego["x"] + float(self._rng.uniform(15.0, cfg.spawn_radius))
        return _Obstacle(
            x=x,
            y=y,
            speed=speed,
            length=cfg.vehicle_length,
            width=cfg.vehicle_width,
        )

    # ------------------------------------------------------------------
    # backend plumbing
    # ------------------------------------------------------------------
    def _convert_action(self, action: Any) -> np.ndarray:
        """Real backend expects the normalised ``[a1, a2]`` already (§C.2)."""
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size < MACRO_V1_ACTION_DIM:
            arr = np.concatenate([arr, np.zeros(MACRO_V1_ACTION_DIM - arr.size)])
        return np.clip(arr[:MACRO_V1_ACTION_DIM], -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # simulator state save / restore (Go-Explore style, §C.1)
    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        if self._backend == "metadrive":
            inner = None
            for name in ("get_state", "snapshot", "save_state"):
                fn = getattr(self._metadrive, name, None)
                if callable(fn):
                    try:
                        inner = fn()
                        break
                    except Exception:  # pragma: no cover - defensive
                        inner = None
            return {
                "backend": "metadrive",
                "episode_step": self.episode_step,
                "episode_return": self.episode_return,
                "done": self._done,
                "success": self._success,
                "inner": copy.deepcopy(inner),
            }
        return {
            "backend": "sim",
            "ego": copy.deepcopy(self.ego),
            "obstacles": [ob.to_state() for ob in self.obstacles],
            "episode_step": self.episode_step,
            "episode_return": self.episode_return,
            "done": self._done,
            "success": self._success,
            "last_action": self._last_action.copy(),
            "last_control": tuple(self._last_control),
            "start_lane": self._start_lane,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def set_state(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        if not isinstance(state, dict):
            raise TypeError("AutoDrivingEnv.set_state expects a state dict")
        if state.get("backend") == "metadrive":
            inner = state.get("inner")
            if inner is not None:
                for name in ("set_state", "restore", "load_state"):
                    fn = getattr(self._metadrive, name, None)
                    if callable(fn):
                        try:
                            fn(inner)
                            break
                        except Exception:  # pragma: no cover - defensive
                            pass
            self.episode_step = int(state.get("episode_step", 0))
            self.episode_return = float(state.get("episode_return", 0.0))
            self._done = bool(state.get("done", False))
            self._success = bool(state.get("success", False))
            return self.observation()

        self.ego = copy.deepcopy(state.get("ego", self.ego))
        stored = state.get("obstacles") or []
        obstacles: List[_Obstacle] = []
        for spec in stored:
            ob = _Obstacle(0.0, 0.0, 0.0)
            ob.from_state(spec)
            obstacles.append(ob)
        self.obstacles = obstacles
        self.episode_step = int(state.get("episode_step", 0))
        self.episode_return = float(state.get("episode_return", 0.0))
        self._done = bool(state.get("done", False))
        self._success = bool(state.get("success", False))
        if "last_action" in state and state["last_action"] is not None:
            self._last_action = np.asarray(state["last_action"], dtype=np.float32)
        if "last_control" in state and state["last_control"] is not None:
            self._last_control = tuple(state["last_control"])
        self._start_lane = int(state.get("start_lane", self._start_lane))
        if state.get("rng_state") is not None:
            try:
                self._rng.bit_generator.state = copy.deepcopy(state["rng_state"])
            except Exception:  # pragma: no cover - defensive
                pass
        return self.observation()

    # duck-typed aliases (see env_reset.py)
    def snapshot(self) -> Dict[str, Any]:
        return self.get_state()

    def restore(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def save_state(self) -> Dict[str, Any]:
        return self.get_state()

    def load_state(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def state_dict(self) -> Dict[str, Any]:
        return self.get_state()

    def load_state_dict(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def restore_state(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def current_observation(self) -> np.ndarray:
        return self.observation()

    def get_observation(self) -> np.ndarray:
        return self.observation()

    def set_state_from_observation(self, obs: Any) -> np.ndarray:
        """Best-effort restore when only an observation is available.

        The driving observation is egocentric and therefore lossy with respect
        to the absolute simulator state; this helper simply seeds the ego speed
        from the observation so that degraded resets remain usable.
        """
        arr = np.asarray(obs, dtype=np.float64).reshape(-1)
        if arr.size >= 4:
            self.ego["speed"] = float(np.clip(arr[3], 0.0, 1.0) * self.config.max_speed)
        return self.observation()

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def heuristic_action(self, obs: Optional[Any] = None) -> np.ndarray:
        """Simple lane-keeping / cruise-control baseline (not from the paper)."""
        obs_arr = self.observation() if obs is None else np.asarray(obs, dtype=np.float64).reshape(-1)
        a1 = 0.0
        a2 = 0.0
        if obs_arr.size >= 7:
            a1 = float(np.clip(-1.5 * obs_arr[6] - 0.5 * obs_arr[1], -1.0, 1.0))
            a2 = float(np.clip((self.config.target_speed / self.config.max_speed) - obs_arr[3] + 0.5, -1.0, 1.0))
        return np.asarray([a1, a2], dtype=np.float32)

    def clone(self, **overrides: Any) -> "AutoDrivingEnv":
        return AutoDrivingEnv(config=self.config.clone(**overrides))

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self.config.seed = int(seed)
            self._rng = np.random.default_rng(int(seed))
        return [int(seed) if seed is not None else int(self.config.seed or 0)]

    def render(self, *args: Any, **kwargs: Any):  # pragma: no cover - optional
        if self._backend == "metadrive":
            try:
                return self._metadrive.render(*args, **kwargs)
            except Exception:
                return None
        return None

    def close(self) -> None:
        if self._backend == "metadrive":
            try:
                self._metadrive.close()
            except Exception:  # pragma: no cover - defensive
                pass


# --------------------------------------------------------------------------
# backend resolution
# --------------------------------------------------------------------------
def metadrive_available() -> bool:
    """True when MetaDrive (and hence DI-drive's Macro-v1) can be imported."""
    try:
        import importlib

        return importlib.util.find_spec("metadrive") is not None
    except Exception:  # pragma: no cover - defensive
        return False


def resolve_backend(backend: str = "auto") -> str:
    """Resolve ``"auto"`` to a concrete backend name."""
    name = (backend or "auto").strip().lower().replace("-", "_")
    if name in ("di_drive", "didrive", "di"):
        name = "metadrive"
    if name == "metadrive":
        return "metadrive" if metadrive_available() else "sim"
    if name == "sim":
        return "sim"
    # "auto"
    requested = os.environ.get("RICE_USE_REAL_METADRIVE") or os.environ.get(
        "RICE_ALLOW_METADRIVE"
    )
    if str(requested).strip() in ("1", "true", "True", "yes"):
        return "metadrive" if metadrive_available() else "sim"
    return "sim"


def _build_metadrive_env(config: AutoDrivingConfig) -> Any:
    """Thin adapter over MetaDrive's ``MetaDriveEnv`` with a DI-drive flavour.

    MetaDrive's ``MetaDriveEnv`` takes a config dict; DI-drive's "Macro-v1"
    settings are approximated with a dense-traffic, 2-D normalised-action
    configuration.  Any user-supplied keys in ``config.metadrive_config`` take
    precedence.  Raises on failure so the caller can fall back to the simulator.
    """
    from metadrive.envs import MetaDriveEnv  # type: ignore  # noqa: WPS433

    md_config: Dict[str, Any] = {
        "environment_num": 200,
        "start_seed": int(config.seed or 0),
        "traffic_density": 0.3,
        "map": "SSS",                 # straight road segment series
        "random_traffic": True,
        "use_render": False,
        "manual_control": False,
        "decision_repeat": 5,
        "action_config": {"steering": {"max": MAX_STEERING_DEG}, "throttle": {"max": 1.0}},
        # dense shaping: forward motion + speed maintenance + crash/out penalties
        "out_of_route_done": True,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "horizon": int(config.max_episode_steps),
    }
    md_config.update(config.metadrive_config or {})
    return MetaDriveEnv(md_config)


def _make_backend(config: AutoDrivingConfig) -> Tuple[str, Any]:
    backend = resolve_backend(config.backend)
    if backend == "metadrive":
        try:
            return "metadrive", _build_metadrive_env(config)
        except Exception:  # pragma: no cover - fall back to the simulator
            return "sim", None
    return "sim", None


# --------------------------------------------------------------------------
# Factory contract used by rice.environments registry
# --------------------------------------------------------------------------
def make_env(
    name: str = "Macro-v1",
    *,
    seed: Optional[int] = None,
    normalize: Optional[bool] = None,
    normalize_obs: Optional[bool] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    config: Optional[AutoDrivingConfig] = None,
    backend: Optional[str] = None,
    action_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Build a single (non-vectorized) autonomous-driving env.

    Mirrors the factory signature used by the other RICE environments so that
    ``rice.environments.make_env("Macro-v1", ...)`` works uniformly.
    """
    canonical = resolve_name(name)
    if config is None:
        config = AutoDrivingConfig()
    overrides: Dict[str, Any] = {"env_name": canonical}
    if seed is not None:
        overrides["seed"] = int(seed)
    if max_episode_steps is not None:
        overrides["max_episode_steps"] = int(max_episode_steps)
    if render_mode is not None:
        overrides["render_mode"] = render_mode
    if backend is not None:
        overrides["backend"] = backend
    if action_mode is not None:
        overrides["action_mode"] = action_mode
    do_normalize = normalize_obs if normalize_obs is not None else normalize
    if do_normalize is not None:
        overrides["normalize_obs"] = bool(do_normalize)
    # remaining kwargs that match config fields are forwarded
    for key, value in kwargs.items():
        if hasattr(config, key):
            overrides[key] = value
    config = config.clone(**overrides)

    env: Any = AutoDrivingEnv(config=config)
    if config.normalize_obs:
        env = AutoDrivingObsNormalizeWrapper(
            env, normalize=True, clip=config.obs_clip
        )
    env.rise_canonical_name = canonical
    env.rise_net_arch = tuple(config.net_arch)
    try:
        env.rise_backend = config.backend
    except Exception:  # pragma: no cover - defensive
        pass
    return env


def make_autodriving(**kwargs: Any) -> Any:
    """Registry alias of :func:`make_env`."""
    return make_env(**kwargs)


def make_autodriving_env(**kwargs: Any) -> Any:
    """Registry alias of :func:`make_env`."""
    return make_env(**kwargs)


def make_metadrive_env(
    name: str = "Macro-v1", *, backend: str = "metadrive", **kwargs: Any
) -> Any:
    """Explicitly request the MetaDrive / DI-drive back-end."""
    return make_env(name, backend=backend, **kwargs)


# --------------------------------------------------------------------------
# registry metadata
# --------------------------------------------------------------------------
_AUTODRIVING_ALIASES = {
    "macro_v1": "Macro-v1",
    "macrov1": "Macro-v1",
    "macro1": "Macro-v1",
    "macro": "Macro-v1",
    "autodriving": "Macro-v1",
    "auto_driving": "Macro-v1",
    "autonomousdriving": "Macro-v1",
    "autonomous_driving": "Macro-v1",
    "metadrive": "Macro-v1",
    "di_drive": "Macro-v1",
    "didrive": "Macro-v1",
    "driving": "Macro-v1",
}

AUTODRIVING_SPECS: Dict[str, Dict[str, Any]] = {
    "Macro-v1": {
        "name": "Macro-v1",
        "simulator": "MetaDrive",
        "algorithm_source": "DI-drive (drive Contributors, 2021)",
        "obs_dim": MACRO_V1_OBS_DIM,
        "act_dim": MACRO_V1_ACTION_DIM,
        "net_arch": MACRO_V1_NET_ARCH,
        "normalize_obs": True,
        "action_range": (-1.0, 1.0),
        "max_episode_steps": 1000,
        "mask_samples_budget": 2443260,   # Table 4
        "hyperparams": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},  # Table 3
    }
}


def resolve_name(name: str) -> str:
    """Canonicalise an alias to ``"Macro-v1"``."""
    if not name:
        return "Macro-v1"
    key = str(name).strip().lower().replace(" ", "").replace("-", "_")
    if key in _AUTODRIVING_ALIASES:
        return _AUTODRIVING_ALIASES[key]
    # already canonical
    for canonical in AUTODRIVING_SPECS:
        if canonical.lower().replace("-", "_") == key:
            return canonical
    raise KeyError(
        f"Unknown autonomous-driving environment {name!r}; "
        f"known names: {sorted(AUTODRIVING_SPECS)}"
    )


def get_spec(name: str = "Macro-v1") -> Dict[str, Any]:
    """Return registry metadata for the autonomous-driving task."""
    canonical = resolve_name(name)
    spec = dict(AUTODRIVING_SPECS[canonical])
    spec["backend"] = resolve_backend("auto")
    return spec


def obstacle_avoids_collision(cfg: Optional[AutoDrivingConfig] = None) -> bool:
    """Sanity helper: the traffic cone is inside the configured spawn radius."""
    cfg = cfg or AutoDrivingConfig()
    return bool(cfg.spawn_radius > 15.0 and cfg.safe_gap < cfg.spawn_radius)
