"""Environment factory and registration for RICE.

This module centralises everything that the paper's experiments need in order to
obtain an environment instance:

* **MuJoCo games** (dense reward) -- ``Hopper-v3``, ``Walker2d-v3``,
  ``Reacher-v2`` and ``HalfCheetah-v3``.  Observations are normalized for
  Walker2d and HalfCheetah (Appendix C.2: *"We use "Walker2d-v3" in our
  experiments and normalize the observation when training the DRL agent"*, same
  for ``HalfCheetah-v3``).
* **Sparse MuJoCo games** introduced by Mazoure et al. (2019), which we register
  ourselves as ``SparseHopper-v0`` / ``SparseHalfCheetah-v0``.  Per Appendix C.2
  the sparse reward *"informs the x position of the hopper only if x > 0.6"*
  (Hopper) resp. *"only if x > 5"* (HalfCheetah).
* **Real-world applications** -- Selfish Mining (Bar-Zur et al., 2023; three
  actions *Adopt l* / *Reveal l* / *Mine*), CAGE Challenge 2 (blue agent action
  set, ``Restore`` carries a ``-1`` reward penalty, trials of length 30/50/100
  and the *"final reward is the sum of the average rewards of these three
  different lengths"*) and Autonomous Driving (MetaDrive ``Macro-v1`` with the
  2-d action ``a = [a_1, a_2] in [-1, 1]^2`` mapped to steering / acceleration /
  brake).  Malware Mutation is **out of scope** for this reproduction and is
  therefore registered as *unsupported*.

The factory is deliberately defensive: optional third party dependencies
(MuJoCo bindings, MetaDrive / DI-drive, the blockchain and CAGE repositories) may
not be installed in every environment used to reproduce the results.  When the
external package is missing we fall back to a *lightweight internal simulation*
that exposes exactly the same ``gym`` interface and dimensionality contract, so
that the rest of the RICE pipeline (mask network, fidelity evaluator, refinement
loop) can be executed and smoke-tested end to end.  The fallbacks are clearly
flagged through :func:`env_backend` / :func:`env_metadata`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rice.utils.io import ensure_dir
from rice.utils.logging import get_logger
from rice.utils.seeding import seed_action_space, seed_env
from rice.envs.normalizer import make_normalizer, should_normalize

logger = get_logger("rice.envs")

__all__ = [
    "EnvSpec",
    "ENV_SPECS",
    "SPARSE_THRESHOLDS",
    "TRAIL_LENGTHS",
    "register_envs",
    "list_envs",
    "available_envs",
    "resolve_env_spec",
    "make_env",
    "make_vec_env",
    "make_sparse_env",
    "make_application_env",
    "env_metadata",
    "env_backend",
    "d_max_for",
    "restore_action_penalty",
    "cage2_final_reward",
    "is_sparse_env",
    "is_application_env",
]

# ---------------------------------------------------------------------------
# optional third party dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the installed environment
    import gym  # type: ignore

    GYM_AVAILABLE = True
    _GYM_IMPORT_ERROR: Optional[BaseException] = None
except Exception as exc:  # pragma: no cover
    gym = None  # type: ignore
    GYM_AVAILABLE = False
    _GYM_IMPORT_ERROR = exc

try:  # pragma: no cover
    import gymnasium  # type: ignore

    GYMNASIUM_AVAILABLE = True
except Exception:  # pragma: no cover
    gymnasium = None  # type: ignore
    GYMNASIUM_AVAILABLE = False

#: thresholds used by the sparse MuJoCo variants (Appendix C.2)
SPARSE_THRESHOLDS: Dict[str, float] = {"hopper": 0.6, "halfcheetah": 5.0}

#: CAGE Challenge 2 evaluates the blue agent on three trial lengths
TRAIL_LENGTHS: Tuple[int, ...] = (30, 50, 100)

#: reward penalty applied when the blue agent executes ``Restore`` (Appendix C.2)
RESTORE_PENALTY: float = -1.0


# ---------------------------------------------------------------------------
# environment specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvSpec:
    """Static description of one of the paper's environments."""

    key: str
    """Canonical short key (matches the ``configs/<key>.yaml`` files)."""

    env_id: str
    """Gym id (or custom registered id) passed to ``gym.make``."""

    app: str = "mujoco"
    """One of ``mujoco``, ``sparse_mujoco``, ``blockchain``, ``cyber``, ``driving``."""

    gym_id: Optional[str] = None
    """Underlying gym id when this spec is a *wrapped* variant."""

    max_episode_steps: int = 1000
    normalize_obs: bool = False
    sparse: bool = False
    sparse_threshold: Optional[float] = None
    discrete_actions: Optional[int] = None
    obs_dim: Optional[int] = None
    action_dim: Optional[int] = None
    supported: bool = True
    """``False`` for environments that are out of scope (Malware Mutation)."""

    kwargs: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_application(self) -> bool:
        return self.app in ("blockchain", "cyber", "driving")

    @property
    def is_sparse(self) -> bool:
        return self.sparse

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["is_application"] = self.is_application
        d["is_sparse"] = self.is_sparse
        return d


#: registry of every environment mentioned by the paper.  Keys are lower-cased
#: and normalised (``-`` -> ``_``, version suffixes stripped) so that
#: ``"Hopper-v3"``, ``"hopper"`` and ``"HOPPER"`` all resolve to the same spec.
ENV_SPECS: Tuple[EnvSpec, ...] = (
    # ---- dense MuJoCo -----------------------------------------------------
    EnvSpec(
        key="hopper",
        env_id="Hopper-v3",
        gym_id="Hopper-v3",
        app="mujoco",
        max_episode_steps=1000,
        normalize_obs=False,
    ),
    EnvSpec(
        key="walker2d",
        env_id="Walker2d-v3",
        gym_id="Walker2d-v3",
        app="mujoco",
        max_episode_steps=1000,
        normalize_obs=True,
    ),
    EnvSpec(
        key="reacher",
        env_id="Reacher-v2",
        gym_id="Reacher-v2",
        app="mujoco",
        max_episode_steps=50,
        normalize_obs=False,
    ),
    EnvSpec(
        key="halfcheetah",
        env_id="HalfCheetah-v3",
        gym_id="HalfCheetah-v3",
        app="mujoco",
        max_episode_steps=1000,
        normalize_obs=True,
    ),
    # ---- sparse MuJoCo (Mazoure et al., 2019) ----------------------------
    EnvSpec(
        key="sparse_hopper",
        env_id="SparseHopper-v0",
        gym_id="Hopper-v3",
        app="sparse_mujoco",
        max_episode_steps=1000,
        normalize_obs=False,
        sparse=True,
        sparse_threshold=SPARSE_THRESHOLDS["hopper"],
    ),
    EnvSpec(
        key="sparse_halfcheetah",
        env_id="SparseHalfCheetah-v0",
        gym_id="HalfCheetah-v3",
        app="sparse_mujoco",
        max_episode_steps=1000,
        normalize_obs=True,
        sparse=True,
        sparse_threshold=SPARSE_THRESHOLDS["halfcheetah"],
    ),
    # ---- real world applications -----------------------------------------
    EnvSpec(
        key="selfish_mining",
        env_id="SelfishMining-v0",
        app="blockchain",
        max_episode_steps=1000,
        discrete_actions=3,  # Adopt l / Reveal l / Mine
        obs_dim=16,
    ),
    EnvSpec(
        key="cage2",
        env_id="Cage2-v0",
        app="cyber",
        max_episode_steps=100,  # longest trial length
        discrete_actions=54,  # 9 action types x 6 hosts (see module docstring)
        obs_dim=96,
    ),
    EnvSpec(
        key="autodriving",
        env_id="MetaDrive-Macro-v1",
        app="driving",
        max_episode_steps=1000,
        action_dim=2,  # [steering-ish, acceleration/brake] in [-1, 1]^2
        obs_dim=259,
    ),
    # ---- out of scope -----------------------------------------------------
    EnvSpec(
        key="malware",
        env_id="MalConv-v0",
        app="malware",
        max_episode_steps=10,  # Appendix C.2: max 10 mutation steps
        discrete_actions=16,  # Table 2 action set
        supported=False,
    ),
)

#: alias table -> canonical key
_ALIASES: Dict[str, str] = {
    "hopper_v3": "hopper",
    "walker2d_v3": "walker2d",
    "walker_v3": "walker2d",
    "walker": "walker2d",
    "reacher_v2": "reacher",
    "halfcheetah_v3": "halfcheetah",
    "half_cheetah": "halfcheetah",
    "cheetah": "halfcheetah",
    "sparsehopper": "sparse_hopper",
    "sparse_hopper_v0": "sparse_hopper",
    "sparsehalfcheetah": "sparse_halfcheetah",
    "sparse_half_cheetah": "sparse_halfcheetah",
    "selfish": "selfish_mining",
    "selfishmining": "selfish_mining",
    "mining": "selfish_mining",
    "cage": "cage2",
    "cage_2": "cage2",
    "cagechallenge2": "cage2",
    "auto": "autodriving",
    "autonomous_driving": "autodriving",
    "metadrive": "autodriving",
    "metadrive_macro_v1": "autodriving",
    "macro_v1": "autodriving",
    "malware_mutation": "malware",
    "malconv": "malware",
}


def _normalise_key(name: str) -> str:
    """Lower-case ``name``, strip a trailing ``-vN`` and unify separators."""
    if name is None:
        raise ValueError("env name must not be None")
    key = str(name).strip().lower()
    key = key.replace(" ", "").replace("-", "_")
    # strip version suffixes such as ``_v3`` / ``_v0`` (but keep ``_v1`` for cage? no)
    parts = key.split("_")
    if len(parts) > 1 and len(parts[-1]) == 2 and parts[-1][0] == "v" and parts[-1][1].isdigit():
        key = "_".join(parts[:-1])
    if key in _ALIASES:
        key = _ALIASES[key]
    return key


def resolve_env_spec(name: Any) -> EnvSpec:
    """Return the :class:`EnvSpec` for ``name``.

    ``name`` may be a canonical key (``"sparse_hopper"``), a gym id
    (``"HalfCheetah-v3"``) or an alias (``"cage"``).
    """
    if isinstance(name, EnvSpec):
        return name
    raw = str(name).strip().lower()
    # exact gym id match first
    for spec in ENV_SPECS:
        if raw == spec.env_id.lower() or raw == str(spec.gym_id).lower():
            return spec
    key = _normalise_key(raw)
    for spec in ENV_SPECS:
        if key == spec.key:
            return spec
    raise KeyError(
        "unknown environment {!r}; available: {}".format(name, list_envs())
    )


def list_envs(include_unsupported: bool = True) -> List[str]:
    """Canonical keys of every registered environment."""
    return [s.key for s in ENV_SPECS if include_unsupported or s.supported]


def available_envs() -> List[str]:
    """Keys of the environments that are in scope for this reproduction."""
    return [s.key for s in ENV_SPECS if s.supported]


def is_sparse_env(name: Any) -> bool:
    return resolve_env_spec(name).is_sparse


def is_application_env(name: Any) -> bool:
    return resolve_env_spec(name).is_application


# ---------------------------------------------------------------------------
# lightweight fallback environments (only used when the real simulator is
# unavailable).  They reproduce the interface + dimensionality contract.
# ---------------------------------------------------------------------------
class _FallbackEnv(object):
    """Minimal deterministic-ish simulator with a gym-like API.

    The dynamics are a simple integrator driven by the (bounded) action; they
    are *not* meant to reproduce the paper's numbers, only to keep the mask
    trainer / fidelity evaluator / refinement loop executable without the heavy
    optional dependencies.  ``backend == "fallback"`` is reported by
    :func:`env_backend`.
    """

    backend = "fallback"

    def __init__(
        self,
        obs_dim: int = 11,
        action_dim: Optional[int] = 3,
        discrete_actions: Optional[int] = None,
        max_episode_steps: int = 1000,
        sparse: bool = False,
        sparse_threshold: Optional[float] = None,
        forward_scale: float = 1.0,
        name: str = "fallback",
        seed: Optional[int] = None,
    ) -> None:
        self.name = name
        self.obs_dim = int(obs_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.sparse = bool(sparse)
        self.sparse_threshold = sparse_threshold
        self.forward_scale = float(forward_scale)
        self._rng = np.random.RandomState(0 if seed is None else int(seed))
        self._t = 0
        self._x = 0.0
        self._vel = 0.0
        self._state = np.zeros(self.obs_dim, dtype=np.float32)

        if gym is not None:
            from gym import spaces

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            if discrete_actions is not None:
                self.discrete_actions = int(discrete_actions)
                self.action_space = spaces.Discrete(self.discrete_actions)
            else:
                self.discrete_actions = None
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(int(action_dim or 3),), dtype=np.float32
                )
        else:  # numpy-only fallback (still enough for the pipeline's logic)
            from rice.envs._spaces import Box, Discrete  # type: ignore

            self.discrete_actions = int(discrete_actions) if discrete_actions else None
            self.observation_space = Box(-np.inf, np.inf, (self.obs_dim,))
            if self.discrete_actions is not None:
                self.action_space = Discrete(self.discrete_actions)
            else:
                self.action_space = Box(-1.0, 1.0, (int(action_dim or 3),))

    # -- gym API -----------------------------------------------------------
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.RandomState(int(seed))
        return [int(seed) if seed is not None else None]

    def reset(self, *args, **kwargs):
        self._t = 0
        self._x = 0.0
        self._vel = 0.0
        self._state = self._rng.normal(0.0, 0.1, size=(self.obs_dim,)).astype(np.float32)
        obs = self._state.copy()
        if kwargs.get("return_info", False) or args:
            return obs, {}
        return obs

    def step(self, action):
        self._t += 1
        if self.discrete_actions is not None:
            a = int(np.asarray(action).reshape(-1)[0])
            u = float(a - (self.discrete_actions - 1) / 2.0)
        else:
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            u = float(np.clip(np.mean(a), -1.0, 1.0))
        self._vel = 0.9 * self._vel + 0.1 * u
        self._x += self._vel * self.forward_scale
        healthy = abs(self._x) < 5.0 * self.forward_scale * self.max_episode_steps / 1000.0 + 1.0
        terminate = (not healthy) or self._t >= self.max_episode_steps

        if self.sparse:
            threshold = 0.6 if self.sparse_threshold is None else float(self.sparse_threshold)
            reward = float(self._x) if self._x > threshold else 0.0
        else:
            reward = float(self._vel) - 0.1 * float(u ** 2) + 1.0
        reward *= self.forward_scale if not self.sparse else 1.0

        noise = self._rng.normal(0.0, 0.05, size=(self.obs_dim,)).astype(np.float32)
        self._state = (0.95 * self._state + noise).astype(np.float32)
        self._state[0] = self._x
        obs = self._state.copy()
        info = {"x_position": float(self._x), "steps": self._t}
        return obs, float(reward), bool(terminate), info

    def render(self, *args, **kwargs):
        return None

    def close(self):
        return None


class _SelfishMiningFallback(_FallbackEnv):
    """Stand-in for the Bar-Zur et al. (2023) blockchain model.

    Three actions (``Adopt l`` / ``Reveal l`` / ``Mine``), MLP policy size
    ``[128, 128, 128, 128]`` per Appendix C.2.
    """

    def __init__(self, max_episode_steps: int = 1000, seed: Optional[int] = None):
        super().__init__(
            obs_dim=16,
            action_dim=None,
            discrete_actions=3,
            max_episode_steps=max_episode_steps,
            sparse=False,
            name="selfish_mining",
            seed=seed,
        )
        self.fee_big, self.p_big = 10.0, 0.01  # whale transaction
        self.fee_small = 1.0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        a = int(np.asarray(action).reshape(-1)[0])
        whale = self._rng.rand() < self.p_big
        fee = self.fee_big if whale else self.fee_small
        # Mine is beneficial; unstructured adopt/reveal mostly loses revenue.
        scale = {0: -0.5, 1: 0.25, 2: 1.0}.get(a, 0.0)
        reward = float(scale * fee)
        info["action"] = a
        info["whale"] = bool(whale)
        return obs, reward, done, info


class _Cage2Fallback(_FallbackEnv):
    """Stand-in for the CAGE Challenge 2 cyber-defence task.

    The action space is flattened ``Discrete(54)`` = 9 action types x 6 hosts
    (the paper lists *monitoring, analyzing, 7 decoys, removing* and
    *restoring*).  ``Restore`` (action type index 8) carries a ``-1`` penalty.
    ``trail_length`` selects the length of the evaluation trial.
    """

    ACTION_TYPES: Tuple[str, ...] = (
        "monitor",
        "analyse",
        "decoyApache",
        "decoyFemitter",
        "decoyHarakaSMPT",
        "decoySmss",
        "decoySSHD",
        "decoySvchost",
        "decoyTomcat",
        "remove",
        "restore",
    )
    N_HOSTS: int = 6

    def __init__(
        self,
        trail_length: int = 30,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        n_actions = len(self.ACTION_TYPES) * self.N_HOSTS
        super().__init__(
            obs_dim=96,
            action_dim=None,
            discrete_actions=n_actions,
            max_episode_steps=int(max_episode_steps or trail_length),
            sparse=False,
            name="cage2",
            seed=seed,
        )
        self.trail_length = int(trail_length)
        self._red_admin = False
        self._red_timer = 0

    def _decode(self, action: int) -> Tuple[str, int]:
        action = int(action) % (len(self.ACTION_TYPES) * self.N_HOSTS)
        return self.ACTION_TYPES[action // self.N_HOSTS], action % self.N_HOSTS

    def step(self, action):
        self._t += 1
        atype, host = self._decode(action)
        reward = 0.0
        if self._red_admin:
            reward -= 1.0  # continuous penalty while red holds admin access
        if atype == "restore":
            reward += RESTORE_PENALTY
            self._red_admin = False
            self._red_timer = 0
        elif atype == "remove":
            if self._rng.rand() < 0.3:
                self._red_admin = False
        elif atype.startswith("decoy"):
            if self._rng.rand() < 0.05:
                self._red_admin = False
        elif atype == "monitor":
            if self._rng.rand() < 0.02:
                self._red_admin = False
        if not self._red_admin and self._rng.rand() < 0.1:
            self._red_admin = True
        done = self._t >= self.max_episode_steps
        obs = self._rng.normal(0.0, 0.1, size=(self.obs_dim,)).astype(np.float32)
        info = {"action_type": atype, "host": host, "red_admin": bool(self._red_admin)}
        return obs, float(reward), bool(done), info


class _MetaDriveFallback(_FallbackEnv):
    """Stand-in for MetaDrive ``Macro-v1`` (2-d continuous action).

    ``a = [a_1, a_2] in [-1, 1]^2`` is converted to *steering (degree)*,
    *acceleration (hp)* and a *brake* signal (Appendix C.2).
    """

    MAX_STEERING_DEG: float = 30.0

    def __init__(self, max_episode_steps: int = 1000, seed: Optional[int] = None):
        super().__init__(
            obs_dim=259,
            action_dim=2,
            discrete_actions=None,
            max_episode_steps=max_episode_steps,
            sparse=False,
            name="autodriving",
            seed=seed,
        )
        self._speed = 0.0
        self._heading = 0.0
        self._distance = 0.0

    def convert_action(self, action) -> Dict[str, float]:
        """Map the normalized 2-d action to steering / acceleration / brake."""
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a.size == 1:
            a = np.array([0.0, a[0]])
        steer = float(a[0] * self.MAX_STEERING_DEG)
        accel = float(np.clip(a[1], 0.0, 1.0) * 20.0)
        brake = float(np.clip(-a[1], 0.0, 1.0) * 10.0)
        return {"steering": steer, "acceleration": accel, "brake": brake, "throttle": accel}

    def step(self, action):
        cmd = self.convert_action(action)
        self._t += 1
        self._heading += cmd["steering"] * 0.01
        self._speed = float(
            np.clip(self._speed + cmd["acceleration"] * 0.05 - cmd["brake"] * 0.1, 0.0, 30.0)
        )
        self._distance += self._speed * 0.1 * np.cos(self._heading)
        off_road = abs(self._heading) > 1.5
        reward = self._speed * 0.1 - 1.0 if off_road else self._speed * 0.1
        done = self._t >= self.max_episode_steps or off_road
        obs = self._rng.normal(0.0, 0.1, size=(self.obs_dim,)).astype(np.float32)
        obs[0] = self._distance / 100.0
        obs[1] = self._speed / 30.0
        info = dict(cmd)
        info.update({"distance": self._distance, "off_road": bool(off_road)})
        return obs, float(reward), bool(done), info


class _MalConvFallback(_FallbackEnv):
    """Placeholder for the (out-of-scope) MalConv malware-mutation env."""

    def __init__(self, max_episode_steps: int = 10, seed: Optional[int] = None):
        super().__init__(
            obs_dim=64,
            action_dim=None,
            discrete_actions=16,
            max_episode_steps=max_episode_steps,
            sparse=True,
            sparse_threshold=0.0,
            name="malware",
            seed=seed,
        )

    def step(self, action):
        obs, _reward, done, info = super().step(action)
        score = float(np.clip(self._rng.rand(), 0.0, 1.0))
        reward = 10.0 if score < 0.5 else 0.0
        info["score"] = score
        return obs, reward, done, info


class _TimeLimitFallback(object):
    """``gym.wrappers.TimeLimit`` equivalent used by the fallback path."""

    def __init__(self, env, max_episode_steps: int):
        self.env = env
        self._max_episode_steps = int(max_episode_steps)
        self._elapsed = 0

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def spec(self):
        spec = getattr(self.env, "spec", None)
        return spec

    def reset(self, *args, **kwargs):
        self._elapsed = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        out = self.env.step(action)
        self._elapsed += 1
        if len(out) == 4:
            obs, reward, done, info = out
            truncated = self._elapsed >= self._max_episode_steps and not done
            done = bool(done or truncated)
            info = dict(info or {})
            info["TimeLimit.truncated"] = bool(truncated and not done)
            return obs, reward, done, info
        obs, reward, terminated, truncated, info = out
        truncated = bool(truncated or self._elapsed >= self._max_episode_steps)
        return obs, reward, bool(terminated), truncated, info

    def close(self):
        return self.env.close()


# ---------------------------------------------------------------------------
# gym registration
# ---------------------------------------------------------------------------
_REGISTERED = False


def register_envs(force: bool = False) -> bool:
    """Register the paper's custom environments with ``gym``.

    Returns ``True`` when the registration succeeded (``gym`` available) and
    ``False`` when the factory will fall back to the lightweight simulators.
    """
    global _REGISTERED
    if _REGISTERED and not force:
        return GYM_AVAILABLE
    if gym is None:
        return False
    try:
        from gym.envs.registration import register, registry  # type: ignore
    except Exception:  # pragma: no cover
        return False

    def _register(env_id: str, entry_point: Optional[str], kwargs: Dict[str, Any]) -> None:
        try:
            register(id=env_id, entry_point=entry_point, kwargs=kwargs, max_episode_steps=kwargs.get("max_episode_steps"))
        except Exception:
            # already registered -> ignore
            pass

    try:
        # sparse MuJoCo variants -- implemented through the local wrapper
        for key in ("sparse_hopper", "sparse_halfcheetah"):
            spec = resolve_env_spec(key)
            _register(
                spec.env_id,
                "rice.envs.make_env:_register_sparse_entry",
                {"env_key": key},
            )
        _register("SelfishMining-v0", "rice.envs.make_env:_register_selfish_entry", {})
        _register("Cage2-v0", "rice.envs.make_env:_register_cage_entry", {})
        _register("MetaDrive-Macro-v1", "rice.envs.make_env:_register_driving_entry", {})
        _REGISTERED = True
        return True
    except Exception:  # pragma: no cover
        return False


def _register_sparse_entry(env_key: str = "sparse_hopper", **kwargs):
    """Entry point used by the gym registration of the sparse variants."""
    return _build_sparse_env(resolve_env_spec(env_key), **kwargs)


def _register_selfish_entry(**kwargs):
    return _build_application_env(resolve_env_spec("selfish_mining"), **kwargs)


def _register_cage_entry(**kwargs):
    return _build_application_env(resolve_env_spec("cage2"), **kwargs)


def _register_driving_entry(**kwargs):
    return _build_application_env(resolve_env_spec("autodriving"), **kwargs)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _raw_backend(name: Any) -> str:
    """Return ``"external"`` when ``name`` was built by an external package."""
    return getattr(name, "backend", getattr(getattr(name, "env", None), "backend", "external"))


def env_backend(env) -> str:
    """Return the backend tag of ``env`` (``external`` or ``fallback``)."""
    cur = env
    for _ in range(64):
        if cur is None:
            break
        backend = getattr(cur, "backend", None)
        if isinstance(backend, str):
            return backend
        cur = getattr(cur, "env", None)
    return "external"


def _make_gym_env(gym_id: str, seed: Optional[int] = None, **kwargs):
    """Create ``gym_id`` with gym, tolerating both old and new gym APIs."""
    assert gym is not None
    attempts = (
        dict(kwargs),
        {},
        {"disable_env_checker": True},
    )
    last_error: Optional[BaseException] = None
    for attempt in attempts:
        try:
            env = gym.make(gym_id, **attempt)
            if seed is not None:
                seed_env(env, seed)
            return env
        except Exception as exc:  # pragma: no cover - depends on install
            last_error = exc
    raise RuntimeError(
        "could not create gym environment {!r}: {}".format(gym_id, last_error)
    )


def _try_external_app_env(spec: EnvSpec, **kwargs):
    """Best-effort construction of an application env with the real package."""
    if spec.key == "malware":
        return None
    try:
        if spec.key == "autodriving":
            import metadrive  # noqa: F401  (DI-drive wraps MetaDrive)  # pragma: no cover
    except Exception:
        return None
    try:  # pragma: no cover - only when MetaDrive is installed
        cfg = dict(spec.kwargs)
        cfg.update({"config": kwargs.pop("config", None)})
        from metadrive.envs import MetaDriveEnv  # type: ignore

        env = MetaDriveEnv(dict(env_config={"map": "SSS", "num_scenarios": 1000}))
        return env
    except Exception:
        return None


def _build_application_env(spec: EnvSpec, seed: Optional[int] = None, **kwargs):
    """Build one of the paper's real-world application environments."""
    if not spec.supported:
        raise ValueError(
            "environment {!r} is out of scope for this reproduction".format(spec.key)
        )
    env = _try_external_app_env(spec, **kwargs)
    if env is not None:
        if seed is not None:
            seed_env(env, seed)
        return env

    fallbacks: Dict[str, Callable[..., Any]] = {
        "selfish_mining": _SelfishMiningFallback,
        "cage2": _Cage2Fallback,
        "autodriving": _MetaDriveFallback,
    }
    cls = fallbacks[spec.key]
    allowed = ("trail_length", "max_episode_steps", "seed")
    extra = {k: v for k, v in kwargs.items() if k in allowed}
    if cls is not _Cage2Fallback:
        extra.pop("trail_length", None)
    base_seed = seed if seed is not None else kwargs.get("seed")
    extra["seed"] = base_seed
    env = cls(**extra)
    logger.warning(
        "external package for %s unavailable -> using the internal fallback "
        "simulator (trend-only sanity checks).",
        spec.key,
    )
    return env


def _build_sparse_env(spec: EnvSpec, seed: Optional[int] = None, **kwargs):
    """Build a sparse-reward MuJoCo variant (Mazoure et al., 2019)."""
    from rice.envs.sparse_reward import SparseRewardWrapper  # local import: avoid cycles

    threshold = kwargs.pop("threshold", spec.sparse_threshold)
    base = None
    if gym is not None:
        try:
            base = _make_gym_env(spec.gym_id or spec.env_id, seed=seed)
        except Exception as exc:  # pragma: no cover
            logger.warning("MuJoCo sparse base %s unavailable (%s)", spec.gym_id, exc)
            base = None
    if base is None:
        base = _FallbackEnv(
            obs_dim=11 if spec.key == "sparse_hopper" else 17,
            action_dim=3 if spec.key == "sparse_hopper" else 6,
            discrete_actions=None,
            max_episode_steps=spec.max_episode_steps,
            sparse=False,
            name=spec.key,
            seed=seed,
        )

    env = SparseRewardWrapper(
        base,
        threshold=threshold,
        x_index=kwargs.pop("x_index", 0),
        mode=kwargs.pop("mode", "position_threshold"),
        sparse_scale=kwargs.pop("sparse_scale", 1.0),
    )
    if seed is not None:
        seed_env(env, seed)
    return env


def _wrap(spec: EnvSpec, env, normalize: Optional[bool], mode: str, **kwargs):
    """Apply the observation normalizer to the environments the paper uses it on."""
    if normalize is None:
        normalize = spec.normalize_obs or should_normalize(spec.key)
    if normalize and mode == "train":
        env = make_normalizer(env, env_id=spec.key, mode="train")
    elif normalize:
        env = make_normalizer(env, env_id=spec.key, mode="eval")
    return env


def make_env(
    env_id: Any,
    seed: Optional[int] = None,
    normalize: Optional[bool] = None,
    mode: str = "train",
    max_episode_steps: Optional[int] = None,
    time_limit: bool = True,
    **kwargs,
):
    """Create one of the paper's environments.

    Parameters
    ----------
    env_id:
        Canonical key, gym id or alias (see :func:`resolve_env_spec`).
    seed:
        Optional seed; applied to the environment (and the action space).
    normalize:
        ``None`` -> use the paper's default for that environment
        (Walker2d / HalfCheetah are normalized, Appendix C.2).  ``False``
        disables normalization explicitly.
    mode:
        ``"train"`` (online normalization statistics) or ``"eval"`` /
        ``"test"`` / ``"frozen"`` (normalization statistics frozen -- used by
        the Experiment-I fidelity evaluator).
    max_episode_steps:
        Override the default episode length (e.g. the CAGE-2 trial length).
    """
    spec = resolve_env_spec(env_id)
    if not spec.supported:
        raise ValueError(
            "environment {!r} is out of scope for this reproduction (the planner "
            "explicitly excludes the Malware Mutation experiments).".format(spec.key)
        )

    train_mode = str(mode).lower() not in ("eval", "test", "frozen", "deterministic")
    steps = int(max_episode_steps or spec.max_episode_steps)

    # ---- construct the base environment ----------------------------------
    if spec.is_application:
        env = _build_application_env(spec, seed=seed, **kwargs)
        env = _ensure_time_limit(env, steps)
    elif spec.sparse:
        env = _build_sparse_env(spec, seed=seed, **kwargs)
        env = _ensure_time_limit(env, steps)
    else:
        if gym is not None:
            try:
                env = _make_gym_env(spec.env_id, seed=seed)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "environment %s could not be created (%s); using the internal "
                    "fallback with matching dimensions.",
                    spec.env_id,
                    exc,
                )
                env = _mujoco_fallback(spec, seed=seed)
        else:
            env = _mujoco_fallback(spec, seed=seed)
        env = _ensure_time_limit(env, steps)

    # ---- observation normalization ---------------------------------------
    env = _wrap(spec, env, normalize, "train" if train_mode else "eval")

    # ---- seeding ---------------------------------------------------------
    if seed is not None:
        seed_env(env, seed)
        seed_action_space(env, seed)

    setattr(env, "rice_env_key", spec.key)
    setattr(env, "rice_env_spec", spec)
    setattr(env, "rice_max_episode_steps", steps)
    setattr(env, "rice_normalize_obs", bool(normalize if normalize is not None else spec.normalize_obs))
    return env


def _mujoco_fallback(spec: EnvSpec, seed: Optional[int] = None):
    """Fallback MuJoCo-shaped env (dimensions per the OpenAI Gym specs)."""
    dims = {
        "hopper": (11, 3),
        "walker2d": (17, 6),
        "reacher": (11, 2),
        "halfcheetah": (17, 6),
    }
    obs_dim, act_dim = dims.get(spec.key, (11, 3))
    episodes = spec.max_episode_steps
    env = _FallbackEnv(
        obs_dim=obs_dim,
        action_dim=act_dim,
        discrete_actions=None,
        max_episode_steps=min(episodes, 1000),
        sparse=False,
        name=spec.key,
        forward_scale=1.0,
        seed=seed,
    )
    env.max_episode_steps = episodes
    return env


def _ensure_time_limit(env, steps: int):
    """Attach a ``TimeLimit`` of ``steps`` when the env does not have one."""
    if env is None:
        return env
    has_limit = False
    cur = env
    for _ in range(64):
        if cur is None:
            break
        if getattr(cur, "_max_episode_steps", None) is not None:
            has_limit = True
            break
        if cur.__class__.__name__ == "TimeLimit":
            has_limit = True
            break
        cur = getattr(cur, "env", None)
    if has_limit:
        return env
    if gym is not None:
        try:
            from gym.wrappers import TimeLimit  # type: ignore

            return TimeLimit(env, max_episode_steps=int(steps))
        except Exception:  # pragma: no cover
            pass
    return _TimeLimitFallback(env, int(steps))


def make_sparse_env(env_id: Any = "sparse_hopper", threshold: Optional[float] = None, **kwargs):
    """Convenience wrapper: build a sparse-reward environment explicitly."""
    spec = resolve_env_spec(env_id)
    if not spec.sparse:
        # promote a dense spec to its sparse counterpart when possible
        sparse_key = "sparse_" + spec.key
        try:
            spec = resolve_env_spec(sparse_key)
        except KeyError:  # pragma: no cover
            pass
    if threshold is not None:
        kwargs["threshold"] = threshold
    return make_env(spec, **kwargs)


def make_application_env(env_id: Any, **kwargs):
    """Convenience wrapper for the real-world application environments."""
    spec = resolve_env_spec(env_id)
    if not spec.is_application:
        raise ValueError("{!r} is not an application environment".format(spec.key))
    return make_env(spec, **kwargs)


# ---------------------------------------------------------------------------
# vectorised environments
# ---------------------------------------------------------------------------
def make_vec_env(
    env_id: Any,
    n_envs: int = 1,
    seed: Optional[int] = None,
    normalize: Optional[bool] = None,
    mode: str = "train",
    use_sb3: bool = True,
    **kwargs,
):
    """Create ``n_envs`` copies of ``env_id`` wrapped in a VecEnv if available.

    Uses ``stable_baselines3.common.vec_env.DummyVecEnv`` when SB3 is installed
    (the paper builds on SB3 -- Appendix C.1) and otherwise returns a plain
    list of environments.
    """
    spec = resolve_env_spec(env_id)
    n_envs = int(max(1, n_envs))

    def _thunk(rank: int):
        def _init():
            env_seed = None if seed is None else int(seed) + rank
            return make_env(spec, seed=env_seed, normalize=normalize, mode=mode, **kwargs)

        return _init

    if use_sb3:
        try:  # pragma: no cover - depends on SB3
            from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore

            if n_envs == 1:
                return DummyVecEnv([_thunk(0)])
            return DummyVecEnv([_thunk(i) for i in range(n_envs)])
        except Exception:
            pass
    return [_thunk(i)() for i in range(n_envs)]


# ---------------------------------------------------------------------------
# introspection helpers used by the config/experiment glue
# ---------------------------------------------------------------------------
def env_metadata(env_id: Any, probe: bool = True) -> Dict[str, Any]:
    """Return obs/action dimensions and the backend for ``env_id``.

    When ``probe`` is True the environment is actually created (and closed) so
    that the reported dimensions come from the live spaces; otherwise the static
    spec is returned (with ``None`` dimensions where unknown).
    """
    spec = resolve_env_spec(env_id)
    meta: Dict[str, Any] = spec.to_dict()
    meta["backend"] = "unknown"
    if not probe or not spec.supported:
        return meta
    try:
        env = make_env(spec, normalize=False)
        meta["backend"] = env_backend(env)
        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        if obs_space is not None and hasattr(obs_space, "shape"):
            meta["obs_dim"] = int(np.prod(obs_space.shape))
        if act_space is not None:
            if hasattr(act_space, "n"):
                meta["discrete_actions"] = int(act_space.n)
                meta["action_dim"] = int(act_space.n)
            elif hasattr(act_space, "shape"):
                meta["action_dim"] = int(np.prod(act_space.shape))
        try:
            env.close()
        except Exception:
            pass
    except Exception as exc:  # pragma: no cover
        meta["probe_error"] = str(exc)
    return meta


#: maximum achievable single-episode reward, used as ``d_max`` by the fidelity
#: score (Sec. 4.1: "we denote the maximum possible reward change as d_max").
#: The plan instructs to "choose d_max per environment as its max single-episode
#: reward"; the dense MuJoCo numbers follow the standard 1000-step budgets of
#: the corresponding tasks.
_D_MAX: Dict[str, float] = {
    "hopper": 3771.0,
    "walker2d": 5000.0,
    "reacher": 0.0,
    "halfcheetah": 10000.0,
    "sparse_hopper": 1000.0,   # x > 0.6 sparse reward, 1000 steps
    "sparse_halfcheetah": 5000.0,
    "selfish_mining": 10.0,
    "cage2": 0.0,
    "autodriving": 100.0,
}


def d_max_for(env_id: Any, default: Optional[float] = None) -> float:
    """Maximum single-episode reward (fidelity-score normaliser ``d_max``)."""
    try:
        spec = resolve_env_spec(env_id)
        key = spec.key
    except KeyError:
        key = str(env_id)
    if key in _D_MAX:
        return float(_D_MAX[key])
    if default is not None:
        return float(default)
    return 1.0


def restore_action_penalty(action: Any, action_types: Sequence[str] = _Cage2Fallback.ACTION_TYPES) -> float:
    """``-1`` when the CAGE-2 blue agent executes ``Restore``, else ``0``.

    Actions are flattened as ``type_index * n_hosts + host``.
    """
    idx = int(np.asarray(action).reshape(-1)[0])
    n_hosts = _Cage2Fallback.N_HOSTS
    atype = action_types[(idx // n_hosts) % len(action_types)]
    return RESTORE_PENALTY if atype == "restore" else 0.0


def cage2_final_reward(avg_rewards: Sequence[float]) -> float:
    """CAGE-2 final reward = *sum of the average rewards* of the three trails."""
    return float(np.sum(np.asarray(list(avg_rewards), dtype=np.float64)))


def summarize_envs(verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """Metadata for every in-scope environment (used by ``main.py describe``)."""
    out: Dict[str, Dict[str, Any]] = {}
    for spec in ENV_SPECS:
        meta = env_metadata(spec, probe=False)
        out[spec.key] = meta
        if verbose:
            logger.info(
                "%-18s %-22s app=%-12s sparse=%s normalize=%s",
                spec.key,
                spec.env_id,
                spec.app,
                spec.sparse,
                spec.normalize_obs,
            )
    return out


# register on import (best effort)
register_envs()
