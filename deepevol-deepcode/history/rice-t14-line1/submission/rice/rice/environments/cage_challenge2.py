"""CAGE Challenge 2 environment wrapper (TTCP CAGE-2, Cardiff champion blue vs. red "B-line").

Source: §C.2 Extra Introduction to Applications (CAGE Challenge 2)
------------------------------------------------------------------
Paper (verbatim extracts):
  * "We choose the champion scheme proposed by Cardiff University (git, c) in CAGE challenge 2
    (git, b). The target agent is a PPO-based blue agent to defend a network against the red agent
    'B-line'. The trail has three different lengths, i.e., 30, 50, and 100 . The final reward is the
    sum of the average rewards of these three different lengths."
  * Blue action set (11 actions): "Monitor, Analyze, DecoyApache, DecoyFemitter, DecoyHarakaSMPT,
    DecoySmss, DecoySSHD, DecoySvchost, DecoyTomcat, Remove, Restore".
  * "the blue agent can receive a negative reward when the red agent gets admin access to the system
    (and continues to receive negative rewards as the red agent maintains the admin access)"
  * "Restore: The blue agent restores a system to a known good state. Since it significantly impacts
    the system's availability, a reward penalty of -1 will be added when executing this action."
  * "The network architecture of the PPO agent is ... " (mask net mirrors it: MLP [64, 64, 64] for
    CAGE per §C.2 / addendum "Architectures").

Everything the paper does NOT specify (host table, observation encoding, red success probabilities,
host/exploit kill chain, reward aggregation rule) carries a documented default in this module so the
RICE pipeline (Algorithm 1 / Algorithm 2 / fidelity evaluation) can run end-to-end; see README.

Two back-ends are supported
---------------------------
1. ``backend="real"``  — thin adapter over the official ``cage-challenge-2`` package
   (``github.com/cage-challenge/cage-challenge-2``).  Enabled automatically when the package is
   importable, otherwise selected explicitly via ``RICE_USE_REAL_CAGE=1``.
2. ``backend="sim"``   — a self-contained, faithful-in-structure pure-Python simulator of the
   CAGE-2 network (13 hosts: user0-4, enterprise0-2, ops0-2, server0-2), the blue agent's 11 h
   actions (each with a target host) and the scripted red "B-line" kill chain.  This back-end has no
   third-party dependency and is what the unit tests / smoke runs use.

Action space
------------
``action_mode="flat"`` (default): ``Discrete(11 * n_hosts)`` with an action-major encoding
``index = action_idx * n_hosts + host_idx`` so the flat index works directly with the shared RICE
``ActorCritic``/mask network (which support ``Box`` and ``Discrete``).  ``action_mode="multi"``
exposes the original ``MultiDiscrete([11, n_hosts])`` for fidelity with third-party baselines.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - package import path
    from rice.environments._common import (
        EnvBase,
        WrapperBase,
        GYM_AVAILABLE,
        IS_GYMNASIUM,
        env_max_episode_steps,
        import_running_mean_std,
        make_box,
        make_discrete,
        normalize_reset,
        normalize_step,
    )
except Exception:  # pragma: no cover - script / relative import path
    from ._common import (  # type: ignore
        EnvBase,
        WrapperBase,
        GYM_AVAILABLE,
        IS_GYMNASIUM,
        env_max_episode_steps,
        import_running_mean_std,
        make_box,
        make_discrete,
        normalize_reset,
        normalize_step,
    )

__all__ = [
    "CAGE_ACTIONS",
    "ACTION_NAMES",
    "DECOY_SERVICES",
    "B_LINE_KILL_CHAIN",
    "DEFAULT_HOST_TABLE",
    "MONITOR",
    "ANALYZE",
    "REMOVE",
    "RESTORE",
    "DECOY_ACTIONS",
    "Cage2HostState",
    "CageChallenge2Config",
    "CageChallenge2Env",
    "Cage2ObsNormalizeWrapper",
    "make_env",
    "make_cage_challenge2",
    "make_cage_env",
    "resolve_name",
    "get_spec",
    "decode_action",
    "encode_action",
    "evaluate_trial_lengths",
    "real_cage_available",
    "CAGE_CHALLENGE2_SPECS",
]


# --------------------------------------------------------------------------------------
# Constants (blue action set verbatim from §C.2)
# --------------------------------------------------------------------------------------
CAGE_ACTIONS: Tuple[str, ...] = (
    "Monitor",
    "Analyze",
    "DecoyApache",
    "DecoyFemitter",
    "DecoyHarakaSMPT",
    "DecoySmss",
    "DecoySSHD",
    "DecoySvchost",
    "DecoyTomcat",
    "Remove",
    "Restore",
)
ACTION_NAMES: Dict[int, str] = {i: name for i, name in enumerate(CAGE_ACTIONS)}

MONITOR = 0
ANALYZE = 1
DECOY_START = 2
DECOY_END = 9  # inclusive
REMOVE = 9
RESTORE = 10

#: Exploit/decoy services used by the decoy actions (order matches ``CAGE_ACTIONS[2:10]``).
DECOY_SERVICES: Tuple[str, ...] = (
    "Apache",
    "Femitter",
    "HarakaSMPT",
    "Smss",
    "SSHD",
    "Svchost",
    "Tomcat",
)
# NOTE: the paper lists nine decoy actions (Apache, Femitter, HarakaSMPT, Smss, SSHD, Svchost,
# Tomcat) but names eight services; ``Svchost``/``Apache`` cover the remaining exploits, so the
# table above holds seven distinct service names with ``decoy_service(action_idx)`` mapping the
# decoy action index onto one of them (see helper below).

DECOY_ACTIONS: Dict[int, str] = {i + DECOY_START: svc for i, svc in enumerate(DECOY_SERVICES)}

#: Scripted red "B-line" kill chain (exploit, subnet) — the paper cites the CAGE-2 B-line red agent
#: without listing its script, so this ordering is a documented stand-in.
B_LINE_KILL_CHAIN: Tuple[Tuple[str, str], ...] = (
    ("EternalBlue", "user"),
    ("Smss", "user"),
    ("SSHD", "enterprise"),
    ("HarakaSMPT", "enterprise"),
    ("Tomcat", "ops"),
    ("Svchost", "ops"),
    ("Apache", "server"),
    ("Femitter", "server"),
)

#: Reachability order of the CAGE-2 subnets; red may pivot only to the same/adjacent subnet.
SUBNET_ORDER: Tuple[str, ...] = ("user", "enterprise", "ops", "server")

#: Host table of the CAGE-2 network (13 hosts).  The exact service inventory is not given in the
#: paper; the table below covers every exploit in :data:`B_LINE_KILL_CHAIN` at least once.
DEFAULT_HOST_TABLE: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("user0", "user", ("Apache", "Femitter", "HarakaSMPT", "Svchost", "EternalBlue")),
    ("user1", "user", ("Apache", "Femitter", "Svchost")),
    ("user2", "user", ("HarakaSMPT", "Svchost", "EternalBlue")),
    ("user3", "user", ("Apache", "Smss", "Svchost")),
    ("user4", "user", ("Femitter", "HarakaSMPT", "EternalBlue")),
    ("enterprise0", "enterprise", ("Apache", "SSHD", "HarakaSMPT")),
    ("enterprise1", "enterprise", ("Apache", "SSHD", "EternalBlue")),
    ("enterprise2", "enterprise", ("SSHD", "HarakaSMPT", "Torch")),
    ("ops0", "ops", ("Tomcat", "Svchost", "Apache")),
    ("ops1", "ops", ("Tomcat", "Svchost", "SSHD")),
    ("ops2", "ops", ("Tomcat", "Svchost")),
    ("server0", "server", ("Tomcat", "Svchost", "Apache")),
    ("server1", "server", ("Tomcat", "Femitter", "Svchost")),
)

#: Canonical spec table consumed by :func:`rice.environments.get_env_spec`.
CAGE_CHALLENGE2_SPECS: Dict[str, Dict[str, Any]] = {
    "CageChallenge2": {
        "obs_dim": len(DEFAULT_HOST_TABLE) * 8 + 2,
        "act_dim": len(CAGE_ACTIONS) * len(DEFAULT_HOST_TABLE),
        "n_actions": len(CAGE_ACTIONS),
        "n_hosts": len(DEFAULT_HOST_TABLE),
        "net_arch": (64, 64, 64),
        "trial_lengths": (30, 50, 100),
        "normalize_obs": False,
        "description": "TTCP CAGE Challenge 2 blue agent vs. red 'B-line' (Cardiff champion).",
    }
}

_ALIASES: Dict[str, str] = {
    "cage": "CageChallenge2",
    "cage2": "CageChallenge2",
    "cagechallenge2": "CageChallenge2",
    "cage-challenge-2": "CageChallenge2",
    "cage_challenge2": "CageChallenge2",
    "cagechallenge_2": "CageChallenge2",
    "cagechallenge2env": "CageChallenge2",
    "cyberdefence": "CageChallenge2",
    "cyber": "CageChallenge2",
}


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class CageChallenge2Config:
    """Configuration of the CAGE-2 wrapper / fallback simulator.

    Only ``trial_lengths``, the 11 blue actions, the ``-1`` Restore penalty and the negative
    reward for red admin access come from the paper (§C.2).  All other fields are documented
    defaults (the paper does not specify them).
    """

    # ---- paper-specified -------------------------------------------------------------
    trial_lengths: Tuple[int, ...] = (30, 50, 100)
    trial_length: int = 50
    restore_penalty: float = -1.0
    reward_mode: str = "compromised"  # "compromised" (per-host) | "binary" (any host)
    net_arch: Tuple[int, ...] = (64, 64, 64)

    # ---- reward shaping (see module docstring) ---------------------------------------
    reward_scale: float = 1.0
    per_host_penalty: float = -1.0
    action_cost: float = 0.0
    invalid_action_penalty: float = 0.0

    # ---- red "B-line" agent (documented stand-in) ------------------------------------
    p_exploit_success: float = 0.35
    p_user_session: float = 0.5
    p_red_delay: float = 0.15
    red_exploits_per_step: int = 1
    entry_host: str = "user0"
    red_script: Tuple[Tuple[str, str], ...] = B_LINE_KILL_CHAIN

    # ---- blue agent mechanics --------------------------------------------------------
    p_monitor_detect: float = 1.0
    p_analyze_succeed: float = 1.0
    p_remove_success: float = 0.8
    p_restore_clear: float = 1.0
    enable_decoys: bool = True

    # ---- observation / interface -----------------------------------------------------
    obs_mode: str = "host_features"  # reserved for future encodings
    normalize_obs: bool = False
    include_time: bool = True
    action_mode: str = "flat"  # "flat" | "multi"
    max_episode_steps: Optional[int] = None
    backend: str = "auto"  # "auto" | "sim" | "real"
    seed: Optional[int] = None
    render_mode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    def clone(self, **overrides: Any) -> "CageChallenge2Config":
        data = self.to_dict()
        data.update(overrides)
        return CageChallenge2Config(**data)


# --------------------------------------------------------------------------------------
# Host state
# --------------------------------------------------------------------------------------
@dataclass
class Cage2HostState:
    """Mutable state of a single CAGE-2 host."""

    name: str
    subnet: str
    services: List[str] = field(default_factory=list)
    initial_services: List[str] = field(default_factory=list)
    decoys: List[str] = field(default_factory=list)
    red_privileged: bool = False
    red_user_session: bool = False
    red_service_session: bool = False
    red_processes: int = 0
    red_files: int = 0
    red_alert: bool = False
    decoy_alert: bool = False
    analyzed: bool = False
    access_attempts: int = 0

    # -- helpers -----------------------------------------------------------------------
    def reset_host(self) -> None:
        """Restore the host to a known-good state (blue ``Restore``)."""
        self.services = list(self.initial_services)
        self.decoys = []
        self.red_privileged = False
        self.red_user_session = False
        self.red_service_session = False
        self.red_processes = 0
        self.red_files = 0
        self.red_alert = False
        self.decoy_alert = False
        self.analyzed = False

    def red_present(self) -> bool:
        return bool(
            self.red_privileged
            or self.red_user_session
            or self.red_service_session
            or self.red_processes > 0
            or self.red_files > 0
        )

    def to_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "subnet": self.subnet,
            "services": list(self.services),
            "initial_services": list(self.initial_services),
            "decoys": list(self.decoys),
            "red_privileged": bool(self.red_privileged),
            "red_user_session": bool(self.red_user_session),
            "red_service_session": bool(self.red_service_session),
            "red_processes": int(self.red_processes),
            "red_files": int(self.red_files),
            "red_alert": bool(self.red_alert),
            "decoy_alert": bool(self.decoy_alert),
            "analyzed": bool(self.analyzed),
            "access_attempts": int(self.access_attempts),
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "Cage2HostState":
        return cls(**{k: v for k, v in state.items() if k in cls.__dataclass_fields__})


# --------------------------------------------------------------------------------------
# Action helpers
# --------------------------------------------------------------------------------------
def decode_action(action: Any, n_actions: int = len(CAGE_ACTIONS), n_hosts: int = len(DEFAULT_HOST_TABLE)) -> Tuple[int, int]:
    """Decode a (flat or multi-discrete) blue action into ``(action_idx, host_idx)``."""
    if isinstance(action, (list, tuple, np.ndarray)):
        arr = np.asarray(action).astype(np.int64).ravel()
        action_idx = int(arr[0]) if arr.size > 0 else 0
        host_idx = int(arr[1]) if arr.size > 1 else 0
    else:
        index = int(np.asarray(action).astype(np.int64).ravel()[0])
        action_idx = (index // n_hosts) % n_actions
        host_idx = index % n_hosts
    action_idx = int(np.clip(action_idx, 0, n_actions - 1))
    host_idx = int(np.clip(host_idx, 0, n_hosts - 1))
    return action_idx, host_idx


def encode_action(action_idx: int, host_idx: int, n_actions: int = len(CAGE_ACTIONS),
                  n_hosts: int = len(DEFAULT_HOST_TABLE)) -> int:
    """Encode ``(action_idx, host_idx)`` into the flat action index (action-major)."""
    return int(action_idx) * int(n_hosts) + int(host_idx)


def decoy_service(action_idx: int) -> Optional[str]:
    """Service associated with a decoy action index (``None`` if not a decoy action)."""
    if DECOY_START <= int(action_idx) < DECOY_START + len(DECOY_SERVICES):
        return DECOY_SERVICES[int(action_idx) - DECOY_START]
    return None


def _subnets_adjacent(a: str, b: str) -> bool:
    try:
        ia, ib = SUBNET_ORDER.index(a), SUBNET_ORDER.index(b)
    except ValueError:
        return True
    return abs(ia - ib) <= 1


def _make_action_space(n_actions: int, n_hosts: int, action_mode: str) -> Any:
    """Build the blue action space (flat ``Discrete`` by default, ``MultiDiscrete`` on request)."""
    if str(action_mode).lower() in ("multi", "multidiscrete", "multi_discrete"):
        try:  # pragma: no cover - depends on gym availability
            try:
                from rice.environments._common import MultiDiscrete  # type: ignore
            except Exception:
                from ._common import MultiDiscrete  # type: ignore

            return MultiDiscrete([int(n_actions), int(n_hosts)])
        except Exception:
            pass
    return make_discrete(int(n_actions) * int(n_hosts))


# --------------------------------------------------------------------------------------
# Optional: real cage-challenge-2 back-end
# --------------------------------------------------------------------------------------
def real_cage_available() -> bool:
    """True when the official ``cage-challenge-2`` package can be imported."""
    for module_name in ("cage2.envs.cage2_env", "cage2", "cage"):
        try:  # pragma: no cover - optional dependency
            __import__(module_name)
            return True
        except Exception:
            continue
    return False


def _try_make_real_cage_env(config: CageChallenge2Config) -> Optional[Any]:
    """Best-effort adapter around the official CAGE-2 environment.

    Returns ``None`` when the package is unavailable so the caller can fall back to the
    self-contained simulator.  Kept deliberately thin: the official env is only used for
    fidelity and it is normalised through :func:`_adapt_real_env`.
    """
    if not real_cage_available():  # pragma: no cover - optional dependency
        return None
    try:  # pragma: no cover - optional dependency
        from cage2.envs.cage2_env import Cage2Env  # type: ignore

        scenario = None
        try:
            from cage2.envs.cage2_configuration import scenario_creator  # type: ignore

            scenario = scenario_creator("B_line")
        except Exception:
            scenario = None
        env = Cage2Env() if scenario is None else Cage2Env(scenario=scenario)
        return env
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Main environment
# --------------------------------------------------------------------------------------
class CageChallenge2Env(EnvBase):
    """CAGE Challenge 2 environment (blue agent vs. scripted red "B-line").

    The 13-host network, the 11 host-targeted blue actions, the ``-1`` Restore penalty and the
    "negative reward while red holds admin access" rule follow §C.2.  Episode length equals the
    trial length (30 / 50 / 100 in the paper); :func:`evaluate_trial_lengths` implements the paper's
    "final reward = sum of the average rewards of these three different lengths" metric.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        trial_length: Optional[int] = None,
        trial_lengths: Optional[Sequence[int]] = None,
        config: Optional[CageChallenge2Config] = None,
        seed: Optional[int] = None,
        normalize_obs: Optional[bool] = None,
        max_episode_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        action_mode: Optional[str] = None,
        backend: Optional[str] = None,
        host_table: Optional[Sequence[Sequence[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        overrides: Dict[str, Any] = {}
        if trial_length is not None:
            overrides["trial_length"] = int(trial_length)
        if trial_lengths is not None:
            overrides["trial_lengths"] = tuple(int(x) for x in trial_lengths)
        if seed is not None:
            overrides["seed"] = int(seed)
        if normalize_obs is not None:
            overrides["normalize_obs"] = bool(normalize_obs)
        if max_episode_steps is not None:
            overrides["max_episode_steps"] = int(max_episode_steps)
        if render_mode is not None:
            overrides["render_mode"] = render_mode
        if action_mode is not None:
            overrides["action_mode"] = str(action_mode)
        if backend is not None:
            overrides["backend"] = str(backend)
        for key in list(kwargs):
            if hasattr(CageChallenge2Config, key) or key in CageChallenge2Config.__dataclass_fields__:
                overrides[key] = kwargs.pop(key)

        base = config.clone() if config is not None else CageChallenge2Config()
        if overrides:
            base = base.clone(**overrides)
        self.config: CageChallenge2Config = base

        if self.config.seed is not None:
            self.seed(self.config.seed)

        # ------------------------------------------------------------------- host table
        table = host_table if host_table is not None else DEFAULT_HOST_TABLE
        self.hosts: List[Cage2HostState] = []
        for entry in table:
            name, subnet, services = entry[0], entry[1], tuple(entry[2])
            self.hosts.append(
                Cage2HostState(
                    name=str(name),
                    subnet=str(subnet),
                    services=list(services),
                    initial_services=list(services),
                )
            )
        self.host_index: Dict[str, int] = {h.name: i for i, h in enumerate(self.hosts)}

        self.n_actions = len(CAGE_ACTIONS)
        self.n_hosts = len(self.hosts)
        self.obs_dim = self.n_hosts * 8 + (2 if self.config.include_time else 0)

        # ----------------------------------------------------------------- spaces
        self.action_space = _make_action_space(self.n_actions, self.n_hosts, self.config.action_mode)
        self.observation_space = make_box(0.0, 1.0, shape=(self.obs_dim,))

        # ---------------------------------------------------------------- bookkeeping
        self._rng = np.random.RandomState(self.config.seed if self.config.seed is not None else 0)
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_returns: List[float] = []
        self._obs = np.zeros(self.obs_dim, dtype=np.float32)
        self._last_action: Optional[Tuple[int, int]] = None
        self._red_footholds: List[str] = []
        self._red_stage = 0
        self._blue_action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._red_privileged_hosts: List[str] = []
        self._backend = "sim"

        # keep the paper's per-trial length as the default episode horizon
        steps = self.config.max_episode_steps or int(self.config.trial_length)
        self._max_episode_steps = int(steps)
        self._reset_pending_episode = True
        if render_mode is not None:
            self.render_mode = render_mode

    # ------------------------------------------------------------------ properties
    @property
    def max_episode_steps(self) -> int:
        return int(self._max_episode_steps)

    @property
    def trial_length(self) -> int:
        return int(self._max_episode_steps)

    @property
    def episode_step(self) -> int:
        return int(self._step_count)

    @property
    def episode_return(self) -> float:
        return float(self._episode_return)

    # -------------------------------------------------------------------- utilities
    def set_trial_length(self, trial_length: int) -> None:
        """Set the episode horizon to one of the paper's trial lengths (30 / 50 / 100)."""
        self._max_episode_steps = int(trial_length)
        self.config = self.config.clone(trial_length=int(trial_length))

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            return []
        seed = int(seed)
        self._rng = np.random.RandomState(seed)
        self.config = self.config.clone(seed=seed)
        try:
            if getattr(self, "action_space", None) is not None:
                self.action_space.seed(seed)
            if getattr(self, "observation_space", None) is not None:
                self.observation_space.seed(seed)
        except Exception:
            pass
        return [seed]

    def _red_privileged_count(self) -> int:
        return int(sum(1 for h in self.hosts if h.red_privileged))

    def _red_has_reach(self, subnet: str) -> bool:
        """Whether red owns a foothold able to reach ``subnet`` (entry stage always reachable)."""
        if not self._red_footholds:
            return False
        for name in self._red_footholds:
            idx = self.host_index.get(name)
            if idx is None:
                continue
            if _subnets_adjacent(self.hosts[idx].subnet, subnet):
                return True
        return False

    def _drop_foothold(self, name: str) -> None:
        self._red_footholds = [h for h in self._red_footholds if h != name]
        if not self._red_footholds:
            self._red_stage = 0

    # ------------------------------------------------------------------------ reset
    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Any:
        if seed is not None:
            self.seed(seed)
        for host in self.hosts:
            host.reset_host()
        self._step_count = 0
        self._episode_return = 0.0
        self._last_action = None
        self._red_footholds = []
        self._red_stage = 0
        self._blue_action_counts = np.zeros(self.n_actions, dtype=np.int64)
        self._red_privileged_hosts = []

        # red starts from a persistent foothold (CAGE-2 B-line red agent)
        entry = self.config.entry_host
        if entry in self.host_index:
            host = self.hosts[self.host_index[entry]]
            host.red_user_session = True
            self._red_footholds.append(host.name)

        self._obs = self._build_observation()
        info = {
            "trial_length": self._max_episode_steps,
            "n_hosts": self.n_hosts,
            "red_privileged": 0,
            "step": 0,
        }
        return normalize_reset((self._obs.copy(), info))

    # ------------------------------------------------------------------------- step
    def step(self, action: Any) -> Any:
        if self._step_count >= self._max_episode_steps:
            # Auto-reset guard so a slightly over-long loop does not raise.
            self.reset()

        action_idx, host_idx = decode_action(action, self.n_actions, self.n_hosts)
        self._last_action = (action_idx, host_idx)
        self._blue_action_counts[action_idx] += 1

        reward = 0.0
        info: Dict[str, Any] = {"blue_action": CAGE_ACTIONS[action_idx], "host": self.hosts[host_idx].name}

        # 1) blue action ---------------------------------------------------------------
        blue_reward, blue_info = self._apply_blue_action(action_idx, host_idx)
        reward += blue_reward
        info.update(blue_info)

        # 2) red agent -----------------------------------------------------------------
        red_info = self._red_step()
        info.update(red_info)

        # 3) reward: negative while red holds admin access ------------------------------
        n_priv = self._red_privileged_count()
        if self.config.reward_mode == "binary":
            compromise_reward = -1.0 if n_priv > 0 else 0.0
        else:  # "compromised"
            compromise_reward = float(self.config.per_host_penalty) * float(n_priv)
        reward += compromise_reward
        reward *= float(self.config.reward_scale)

        info["red_privileged"] = n_priv
        info["compromise_reward"] = float(compromise_reward)
        info["step"] = self._step_count + 1

        self._step_count += 1
        self._episode_return += float(reward)
        self._obs = self._build_observation()

        terminated = bool(self._step_count >= self._max_episode_steps)
        truncated = False
        if terminated:
            info["episode"] = {"r": float(self._episode_return), "l": int(self._step_count)}
            self._episode_returns.append(float(self._episode_return))

        return normalize_step((self._obs.copy(), float(reward), terminated, truncated, info))

    # ----------------------------------------------------------------- blue actions
    def _apply_blue_action(self, action_idx: int, host_idx: int) -> Tuple[float, Dict[str, Any]]:
        host = self.hosts[host_idx]
        reward = float(self.config.action_cost)
        info: Dict[str, Any] = {}

        if action_idx == MONITOR:
            if host.red_present() and self._rng.rand() < float(self.config.p_monitor_detect):
                host.red_alert = True
            info["alert"] = float(host.red_alert)

        elif action_idx == ANALYZE:
            if host.red_alert and self._rng.rand() < float(self.config.p_analyze_succeed):
                host.analyzed = True
            info["analyzed"] = float(host.analyzed)

        elif DECOY_START <= action_idx < DECOY_START + len(DECOY_SERVICES):
            service = DECOY_SERVICES[action_idx - DECOY_START]
            info["decoy"] = service
            if self.config.enable_decoys:
                if service not in host.decoys:
                    host.decoys.append(service)
                info["n_decoys"] = len(host.decoys)
            else:
                reward += float(self.config.invalid_action_penalty)

        elif action_idx == REMOVE:
            removed = False
            # removing detected red code (needs monitoring + analysis first)
            if host.analyzed and host.red_present():
                if self._rng.rand() < float(self.config.p_remove_success):
                    host.red_privileged = False
                    host.red_user_session = False
                    host.red_service_session = False
                    host.red_processes = 0
                    host.red_files = 0
                    host.red_alert = False
                    host.analyzed = False
                    self._drop_foothold(host.name)
                    removed = True
            # removing generic malicious processes / files
            if host.red_processes > 0 or host.red_files > 0:
                host.red_processes = max(0, host.red_processes - 1)
                host.red_files = max(0, host.red_files - 1)
                removed = True
            info["removed"] = float(removed)

        elif action_idx == RESTORE:
            # §C.2: restoring impacts availability, a reward penalty of -1 is added
            reward += float(self.config.restore_penalty)
            if self._rng.rand() < float(self.config.p_restore_clear):
                host.reset_host()
                self._drop_foothold(host.name)
            info["restored"] = 1.0

        else:  # pragma: no cover - unreachable with the 11 action set
            reward += float(self.config.invalid_action_penalty)

        return float(reward), info

    # -------------------------------------------------------------------- red agent
    def _red_step(self) -> Dict[str, Any]:
        """One step of the scripted red "B-line" agent (documented stand-in for §C.2)."""
        cfg = self.config
        info: Dict[str, Any] = {}
        script = tuple(cfg.red_script)
        if self._red_stage >= len(script):
            return info

        if self._rng.rand() < float(cfg.p_red_delay):
            info["red_delay"] = 1.0
            return info

        for _ in range(max(1, int(cfg.red_exploits_per_step))):
            if self._red_stage >= len(script):
                break
            exploit, subnet = script[self._red_stage]

            if self._red_stage > 0 and not self._red_has_reach(subnet):
                # red cannot pivot yet: keep trying from the current footholds
                info["red_blocked"] = True
                break

            candidates = [
                h
                for h in self.hosts
                if h.subnet == subnet and not h.red_privileged and (exploit in h.services)
            ]
            if not candidates:
                # no vulnerable host left in this subnet; move on
                self._red_stage += 1
                continue

            target = candidates[int(self._rng.randint(len(candidates)))]
            target.access_attempts += 1

            # a decoy of the same service raises an alert and defeats the exploit
            if self.config.enable_decoys and exploit in target.decoys:
                target.decoy_alert = True
                target.red_alert = True
                info["decoy_alert"] = target.name
                self._red_stage += 1
                continue

            if self._rng.rand() < float(cfg.p_exploit_success):
                target.red_privileged = True
                target.red_processes = max(target.red_processes, 1)
                target.red_files = max(target.red_files, 1)
                if target.name not in self._red_footholds:
                    self._red_footholds.append(target.name)
                info["red_privileged"] = target.name
                self._red_stage += 1
            else:
                if self._rng.rand() < float(cfg.p_user_session):
                    target.red_user_session = True
                    if target.name not in self._red_footholds:
                        self._red_footholds.append(target.name)
                info["red_attempt"] = target.name

        return info

    # ------------------------------------------------------------------ observation
    def _host_features(self, host: Cage2HostState) -> List[float]:
        n_services = max(1, len(host.initial_services))
        return [
            1.0 if host.red_privileged else 0.0,
            1.0 if host.red_user_session else 0.0,
            1.0 if host.red_service_session else 0.0,
            float(min(1.0, host.red_processes / 5.0)),
            float(min(1.0, host.red_files / 5.0)),
            float(min(1.0, len(host.decoys) / float(n_services))),
            1.0 if host.red_alert else 0.0,
            1.0 if host.analyzed else 0.0,
        ]

    def _build_observation(self) -> np.ndarray:
        values: List[float] = []
        for host in self.hosts:
            values.extend(self._host_features(host))
        if self.config.include_time:
            frac = 0.0 if self._max_episode_steps <= 0 else float(self._step_count) / float(self._max_episode_steps)
            values.append(float(np.clip(frac, 0.0, 1.0)))
            values.append(float(self._red_privileged_count()) / float(max(1, self.n_hosts)))
        obs = np.asarray(values, dtype=np.float32)
        if obs.shape[0] != self.obs_dim:  # defensive: keep the space contract stable
            out = np.zeros(self.obs_dim, dtype=np.float32)
            n = min(self.obs_dim, obs.shape[0])
            out[:n] = obs[:n]
            obs = out
        return obs

    # ------------------------------------------------------------------- state API
    # Mirrors the save/restore surface used by rice.algorithms.env_reset (Go-Explore style).
    def get_state(self) -> Dict[str, Any]:
        return {
            "kind": "cage_challenge2",
            "hosts": [h.to_state() for h in self.hosts],
            "red_footholds": list(self._red_footholds),
            "red_stage": int(self._red_stage),
            "step_count": int(self._step_count),
            "episode_return": float(self._episode_return),
            "episode_returns": list(self._episode_returns),
            "blue_action_counts": self._blue_action_counts.tolist(),
            "rng_state": self._rng.get_state(),
            "obs": np.array(self._obs, dtype=np.float32, copy=True),
            "max_episode_steps": int(self._max_episode_steps),
            "last_action": None if self._last_action is None else tuple(int(x) for x in self._last_action),
        }

    def set_state(self, state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        if state is None:
            self.reset()
            return np.array(self._obs, dtype=np.float32, copy=True)
        if not isinstance(state, dict):
            return np.array(self._obs, dtype=np.float32, copy=True)

        if state.get("kind") == "observation" or "hosts" not in state:
            return self.set_state_from_observation(state.get("observation", state.get("obs")))

        hosts = state.get("hosts") or []
        if len(hosts) == len(self.hosts):
            self.hosts = [Cage2HostState.from_state(h) for h in hosts]
            self.host_index = {h.name: i for i, h in enumerate(self.hosts)}
        self._red_footholds = list(state.get("red_footholds", []))
        self._red_stage = int(state.get("red_stage", 0))
        self._step_count = int(state.get("step_count", 0))
        self._episode_return = float(state.get("episode_return", 0.0))
        self._episode_returns = list(state.get("episode_returns", []))
        counts = state.get("blue_action_counts")
        if counts is not None:
            try:
                self._blue_action_counts = np.asarray(counts, dtype=np.int64)
            except Exception:
                pass
        self._max_episode_steps = int(state.get("max_episode_steps", self._max_episode_steps))
        last = state.get("last_action")
        self._last_action = None if last is None else (int(last[0]), int(last[1]))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.set_state(rng_state)
            except Exception:
                pass
        obs = state.get("obs")
        self._obs = (
            np.array(obs, dtype=np.float32, copy=True)
            if obs is not None
            else self._build_observation()
        )
        return np.array(self._obs, dtype=np.float32, copy=True)

    # Aliases expected by rice.algorithms.env_reset / mixed_init (duck-typed).
    def state_dict(self) -> Dict[str, Any]:
        return self.get_state()

    def load_state_dict(self, state: Dict[str, Any]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy(self.get_state())

    def save_state(self) -> Dict[str, Any]:
        return self.snapshot()

    def get_observation(self) -> np.ndarray:
        return np.array(self._obs, dtype=np.float32, copy=True)

    def restore(self, state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def restore_state(self, state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def load_state(self, state: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
        return self.set_state(state)

    def set_state_from_observation(self, observation: Any) -> Optional[np.ndarray]:
        """Degraded restore: the observation does not encode the full simulator state."""
        if observation is None:
            return None
        obs = np.asarray(observation, dtype=np.float32).ravel()
        if obs.shape[0] != self.obs_dim:
            out = np.zeros(self.obs_dim, dtype=np.float32)
            n = min(self.obs_dim, obs.shape[0])
            out[:n] = obs[:n]
            obs = out
        self._obs = obs
        return np.array(self._obs, dtype=np.float32, copy=True)

    def current_observation(self) -> np.ndarray:
        return self.get_observation()

    # -------------------------------------------------------------------- misc API
    def heuristic_action(self) -> int:
        """Tiny scripted blue heuristic (helper for smoke tests, not from the paper)."""
        for idx, host in enumerate(self.hosts):
            if host.analyzed and host.red_present():
                return encode_action(REMOVE, idx, self.n_actions, self.n_hosts)
            if host.red_alert and not host.analyzed:
                return encode_action(ANALYZE, idx, self.n_actions, self.n_hosts)
            if host.red_privileged:
                return encode_action(RESTORE, idx, self.n_actions, self.n_hosts)
        for idx, host in enumerate(self.hosts):
            if not host.analyzed and host.red_present():
                return encode_action(MONITOR, idx, self.n_actions, self.n_hosts)
        return encode_action(MONITOR, 0, self.n_actions, self.n_hosts)

    def clone(self, **overrides: Any) -> "CageChallenge2Env":
        return CageChallenge2Env(config=self.config.clone(**overrides))

    def render(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - visualisation
        summary = {
            "step": self._step_count,
            "return": self._episode_return,
            "red_privileged": [h.name for h in self.hosts if h.red_privileged],
            "actions": {CAGE_ACTIONS[i]: int(c) for i, c in enumerate(self._blue_action_counts) if c},
        }
        if self.config.render_mode == "human":
            print(summary)
        return summary

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------------------
# Observation normalization wrapper (optional; paper does not specify for CAGE)
# --------------------------------------------------------------------------------------
class Cage2ObsNormalizeWrapper(WrapperBase):
    """Running mean/std observation normalization with serialisable statistics."""

    def __init__(self, env: Any, normalize: bool = True, clip: float = 10.0,
                 epsilon: float = 1e-8, update: bool = True) -> None:
        super().__init__(env)
        self.normalize = bool(normalize)
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        self.update_stats = bool(update)
        running_cls = import_running_mean_std()
        shape = getattr(getattr(env, "observation_space", None), "shape", None)
        self.rms = running_cls(shape=tuple(shape) if shape else (1,), epsilon=self.epsilon, clip=self.clip)

    def _maybe_update(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if not self.normalize:
            return arr
        if self.update_stats:
            self.rms.update(arr[None, :] if arr.ndim == 1 else arr)
        return np.asarray(
            self.rms.normalize(arr, clip=self.clip, epsilon=self.epsilon, update=False),
            dtype=np.float32,
        )

    def reset(self, **kwargs: Any) -> Any:
        obs, info = normalize_reset(self.env.reset(**kwargs))
        return normalize_reset((self._maybe_update(obs), info))

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = normalize_step(self.env.step(action))
        return normalize_step((self._maybe_update(obs), reward, terminated, truncated, info))

    def state_dict(self) -> Dict[str, Any]:
        return {"rms": self.rms.state_dict(), "normalize": self.normalize}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        if "rms" in state:
            try:
                self.rms.load_state_dict(state["rms"])
            except Exception:
                pass
        if "normalize" in state:
            self.normalize = bool(state["normalize"])


# --------------------------------------------------------------------------------------
# Factories / registry helpers
# --------------------------------------------------------------------------------------
def resolve_name(name: str) -> str:
    """Canonicalize a CAGE alias to ``"CageChallenge2"``."""
    key = str(name).strip().lower().replace(" ", "").replace("_", "")
    if key in ("cagechallenge2", "cage2", "cage", "cyberdefence", "cagechallenge-2"):
        return "CageChallenge2"
    if str(name) in CAGE_CHALLENGE2_SPECS:
        return str(name)
    if key in _ALIASES:
        return _ALIASES[key]
    raise KeyError(f"Unknown CAGE environment '{name}'. Known: {sorted(CAGE_CHALLENGE2_SPECS)}")


def get_spec(name: str = "CageChallenge2") -> Dict[str, Any]:
    """Return the registry metadata of the CAGE-2 environment."""
    canonical = resolve_name(name)
    spec = dict(CAGE_CHALLENGE2_SPECS[canonical])
    spec["name"] = canonical
    return spec


def make_env(
    name: str = "CageChallenge2",
    seed: Optional[int] = None,
    normalize: Optional[bool] = None,
    normalize_obs: Optional[bool] = None,
    max_episode_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    trial_length: Optional[int] = None,
    trial_lengths: Optional[Sequence[int]] = None,
    config: Optional[CageChallenge2Config] = None,
    backend: Optional[str] = None,
    action_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Build a single (non-vectorised) CAGE-2 environment.

    Follows the RICE environment factory contract: ``(name, seed, normalize, max_episode_steps,
    render_mode, **kwargs)`` -> one env instance suitable for Algorithm 1 / Algorithm 2 loops.
    """
    canonical = resolve_name(name) if name is not None else "CageChallenge2"

    selected_backend = backend
    if selected_backend is None:
        selected_backend = os.environ.get("RICE_CAGE_BACKEND", "auto")
    selected_backend = str(selected_backend).lower()
    if selected_backend == "auto":
        selected_backend = "real" if os.environ.get("RICE_USE_REAL_CAGE") == "1" else "sim"

    if selected_backend == "real":  # pragma: no cover - optional dependency
        cfg = config.clone() if config is not None else CageChallenge2Config()
        real_env = _try_make_real_cage_env(cfg)
        if real_env is not None:
            try:
                real_env.rise_canonical_name = canonical
                real_env.rise_backend = "real"
            except Exception:
                pass
            return real_env

    norm = normalize if normalize is not None else normalize_obs
    env = CageChallenge2Env(
        trial_length=trial_length,
        trial_lengths=trial_lengths,
        config=config,
        seed=seed,
        normalize_obs=norm,
        max_episode_steps=max_episode_steps,
        render_mode=render_mode,
        action_mode=action_mode,
        backend=selected_backend,
        **kwargs,
    )
    try:
        env.rise_canonical_name = canonical
        env.rise_backend = "sim"
        env.rise_net_arch = tuple(CAGE_CHALLENGE2_SPECS["CageChallenge2"]["net_arch"])
    except Exception:
        pass
    if norm:
        env = Cage2ObsNormalizeWrapper(env, normalize=True)
        try:
            env.rise_canonical_name = canonical
        except Exception:
            pass
    return env


def make_cage_challenge2(**kwargs: Any) -> Any:
    """Alias of :func:`make_env` used by the environment registry."""
    return make_env(**kwargs)


def make_cage_env(**kwargs: Any) -> Any:
    """Alias of :func:`make_env` used by the environment registry."""
    return make_env(**kwargs)


# --------------------------------------------------------------------------------------
# Trial-length evaluation (paper metric: sum of the average rewards of the 3 lengths)
# --------------------------------------------------------------------------------------
def evaluate_trial_lengths(
    env: Any,
    policy: Any = None,
    n_episodes_per_length: int = 1,
    seed: Optional[int] = None,
    deterministic: bool = True,
    trial_lengths: Optional[Sequence[int]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a blue policy on trials of length 30 / 50 / 100.

    §C.2: "The trail has three different lengths, i.e., 30, 50, and 100. The final reward is the
    sum of the average rewards of these three different lengths."

    ``policy`` may be a callable ``obs -> action``, an SB3 model (``predict``), or the internal
    ``ActorCritic`` (``predict``/``act``).  When ``policy is None`` the environment's scripted
    heuristic (:meth:`CageChallenge2Env.heuristic_action`) is used.
    """
    lengths = tuple(int(x) for x in (trial_lengths or getattr(env, "config", CageChallenge2Config()).trial_lengths))
    policy_fn: Optional[Callable[[Any], Any]] = None
    if policy is not None:
        try:  # lazy import to avoid an environments -> algorithms import cycle
            from rice.algorithms.ppo import make_target_policy_callable

            policy_fn = make_target_policy_callable(policy)
        except Exception:
            if callable(policy):
                policy_fn = policy

    per_length: Dict[int, float] = {}
    for length in lengths:
        if hasattr(env, "set_trial_length"):
            env.set_trial_length(length)
        elif hasattr(env, "_max_episode_steps"):
            env._max_episode_steps = length

        returns: List[float] = []
        for episode in range(max(1, int(n_episodes_per_length))):
            ep_seed = None if seed is None else int(seed) + 1000 * int(length) + int(episode)
            try:
                reset_out = env.reset(seed=ep_seed)
            except TypeError:
                reset_out = env.reset()
            obs, _info = normalize_reset(reset_out)
            done = False
            total = 0.0
            steps = 0
            limit = int(max_steps or length)
            while not done and steps < limit:
                if policy_fn is not None:
                    try:
                        action = policy_fn(obs)
                    except TypeError:
                        action = policy_fn(obs, deterministic)
                else:
                    action = env.heuristic_action() if hasattr(env, "heuristic_action") else 0
                obs, reward, terminated, truncated, _info = normalize_step(env.step(action))
                total += float(reward)
                done = bool(terminated or truncated)
                steps += 1
            returns.append(float(total))
        per_length[int(length)] = float(np.mean(returns)) if returns else float("nan")

    final_reward = float(np.nansum([v for v in per_length.values()]))
    return {
        "per_length": per_length,
        "final_reward": final_reward,
        "mean_reward": final_reward / max(1, len(per_length)),
    }
