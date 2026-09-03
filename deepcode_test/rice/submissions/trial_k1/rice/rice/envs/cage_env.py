"""Gymnasium-compatible adapter for the CAGE Challenge 2 / CybORG cyber-defense environment.

The adapter exposes the blue-agent action list used in the RICE paper:
monitor, analyze, decoy services, remove, and restore.  The red agent is fixed
to the "B-line" attacker.  A reward penalty of ``-1`` is applied for the
``Restore`` action, matching the paper's reward design.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Soft import of CybORG so that ``import rice.envs`` does not fail when the
# external repository is not installed.
try:
    from CybORG import CybORG
    from CybORG.Agents import B_lineAgent
    from CybORG.Agents.Wrappers import BlueTableWrapper, FixedFlatWrapper
    from CybORG.Shared import Results
    _CYBORG_AVAILABLE = True
except Exception as _err:  # pragma: no cover
    _CYBORG_AVAILABLE = False
    _CYBORG_IMPORT_ERROR = _err


# Blue-agent action names used in the RICE paper.
BLUE_ACTION_NAMES = [
    "Monitor",
    "Analyze",
    "DecoyServices",
    "Remove",
    "Restore",
]

# Default path to the CAGE Challenge 2 scenario file.  The CybORG package ships
# scenario definitions under ``CybORG/Shared/Scenarios``; ``Scenario1b.yaml`` is
# the CAGE Challenge 2 scenario used in the paper.
DEFAULT_SCENARIO = "Scenario1b.yaml"


class CageChallenge2Adapter(gym.Env):
    """Gymnasium adapter for CAGE Challenge 2 (CybORG Blue vs. B-line Red).

    Parameters
    ----------
    scenario_file:
        Path to the CybORG scenario YAML file.  If ``None``, the adapter tries
        to locate ``Scenario1b.yaml`` inside the installed CybORG package.
    trial_length:
        Maximum number of steps per episode (paper uses 30, 50, and 100).
    red_agent:
        Name of the red-agent policy to use.  Defaults to ``"B-lineAgent"``.
    blue_action_names:
        Ordered list of blue-agent action names exposed as the discrete action
        space.  Defaults to the paper's list.
    restore_penalty:
        Additional reward penalty applied when the blue agent selects the
        ``Restore`` action.  Defaults to ``-1``.
    seed:
        Random seed passed to the underlying CybORG environment.
    **kwargs:
        Extra arguments forwarded to ``CybORG``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        scenario_file: Optional[str] = None,
        trial_length: int = 50,
        red_agent: str = "B-lineAgent",
        blue_action_names: Optional[List[str]] = None,
        restore_penalty: float = -1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not _CYBORG_AVAILABLE:
            raise ImportError(
                "CybORG is required for CAGE Challenge 2 experiments. "
                "Install it from https://github.com/cage-challenge/CybORG "
                f"(original import error: {_CYBORG_IMPORT_ERROR})"
            )

        self.trial_length = int(trial_length)
        self.red_agent_name = red_agent
        self.blue_action_names = blue_action_names or list(BLUE_ACTION_NAMES)
        self.restore_penalty = float(restore_penalty)
        self._seed = seed
        self._kwargs = kwargs

        # Resolve scenario file path.
        if scenario_file is None:
            try:
                import CybORG as _cyborg_pkg
                scenario_file = str(
                    Path(_cyborg_pkg.__file__).parent
                    / "Shared"
                    / "Scenarios"
                    / DEFAULT_SCENARIO
                )
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    f"Could not locate default CybORG scenario {DEFAULT_SCENARIO}: {exc}"
                )
        self.scenario_file = str(scenario_file)

        # Build the underlying CybORG environment.
        self._cyborg: Optional[Any] = None
        self._env: Optional[Any] = None
        self._red_agent: Optional[Any] = None
        self._step_count = 0
        self._last_obs: Optional[np.ndarray] = None
        self._last_info: Dict[str, Any] = {}
        self._last_action_name: Optional[str] = None

        self._build_env()

        # Discrete action space over the ordered blue-agent action list.
        self.action_space = spaces.Discrete(len(self.blue_action_names))

        # Observation space: CybORG's FixedFlatWrapper produces a fixed-length
        # vector.  We expose it as a Box of floats; if the wrapper is not used
        # we fall back to a large Box.
        obs_shape = getattr(self._env, "observation_space_shape", None)
        if obs_shape is None:
            obs_shape = getattr(self._env, "observation_space", None)
            if obs_shape is not None and hasattr(obs_shape, "shape"):
                obs_shape = obs_shape.shape
        if obs_shape is None:
            warnings.warn(
                "Could not infer CybORG observation shape; using fallback Box(256)."
            )
            obs_shape = (256,)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(obs_shape),
            dtype=np.float32,
        )

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _build_env(self) -> None:
        """Instantiate CybORG and the chosen red agent."""
        self._cyborg = CybORG(
            self.scenario_file,
            "sim",
            agents={"Red": B_lineAgent},
            seed=self._seed,
            **self._kwargs,
        )
        # The paper's blue-agent view is the table-based wrapper, which we then
        # flatten to a fixed vector for the RL policy.
        self._env = FixedFlatWrapper(BlueTableWrapper(self._cyborg, agent="Blue"))
        self._red_agent = B_lineAgent()

    def _action_name_to_cyborg(self, action_name: str) -> Any:
        """Map a blue-agent action name to a CybORG action object.

        The mapping is best-effort: we try the action name directly, then
        common aliases used in different CybORG versions.
        """
        env = self._env
        if env is None:  # pragma: no cover
            raise RuntimeError("Environment not built.")

        # Try exact name and a few aliases.
        candidates = [action_name]
        if action_name == "DecoyServices":
            candidates.extend(["Decoy", "DecoyApache", "DecoyFemitter", "DecoyHaraka"])
        if action_name == "Analyze":
            candidates.extend(["Analyse", "Analyze"])

        # CybORG exposes available actions via the unwrapped environment.
        unwrapped = getattr(env, "environment", env)
        action_space = getattr(unwrapped, "action_space", None)
        if action_space is not None:
            for name in candidates:
                for action in action_space["Blue"]:
                    if getattr(action, "__name__", str(action)) == name:
                        return action
                    if hasattr(action, "name") and action.name == name:
                        return action

        # Fallback: construct by string lookup in the module namespace.
        import CybORG.Shared.Actions as actions_module
        for name in candidates:
            cls = getattr(actions_module, name, None)
            if cls is not None:
                return cls

        raise ValueError(
            f"Could not map blue action '{action_name}' to a CybORG action."
        )

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._seed = seed
            # CybORG does not expose a clean re-seeding API; rebuild.
            self._build_env()

        self._step_count = 0
        self._last_action_name = None

        env = self._env
        if env is None:  # pragma: no cover
            raise RuntimeError("Environment not built.")

        result = env.reset(agent="Blue")
        obs = self._extract_obs(result)
        info = self._extract_info(result)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_name = self.blue_action_names[int(action)]
        self._last_action_name = action_name

        env = self._env
        if env is None:  # pragma: no cover
            raise RuntimeError("Environment not built.")

        cyborg_action = self._action_name_to_cyborg(action_name)
        result = env.step(action=cyborg_action, agent="Blue")

        obs = self._extract_obs(result)
        reward = float(self._extract_reward(result))
        terminated = bool(self._extract_terminated(result))
        truncated = False

        # Apply restore penalty.
        if action_name.lower() == "restore":
            reward += self.restore_penalty

        self._step_count += 1
        if self._step_count >= self.trial_length:
            truncated = True

        info = self._extract_info(result)
        info["blue_action"] = action_name
        info["step_count"] = self._step_count
        info["restore_penalty"] = self.restore_penalty if action_name.lower() == "restore" else 0.0

        self._last_obs = obs
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        if self._cyborg is not None and hasattr(self._cyborg, "shutdown"):
            self._cyborg.shutdown()
        self._cyborg = None
        self._env = None

    # --------------------------------------------------------------------- #
    # Result extraction helpers (tolerant to different CybORG APIs)
    # --------------------------------------------------------------------- #
    def _extract_obs(self, result: Any) -> np.ndarray:
        if isinstance(result, (tuple, list)) and len(result) >= 1:
            obs = result[0]
        elif isinstance(result, dict):
            obs = result.get("observation", result.get("obs", None))
        elif hasattr(result, "observation"):
            obs = result.observation
        elif hasattr(result, "obs"):
            obs = result.obs
        else:
            obs = result

        if obs is None:
            obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return np.asarray(obs, dtype=self.observation_space.dtype)

    def _extract_reward(self, result: Any) -> float:
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return float(result[1])
        if isinstance(result, dict):
            return float(result.get("reward", 0.0))
        if hasattr(result, "reward"):
            return float(result.reward)
        return 0.0

    def _extract_terminated(self, result: Any) -> bool:
        if isinstance(result, (tuple, list)) and len(result) >= 3:
            return bool(result[2])
        if isinstance(result, dict):
            return bool(result.get("done", result.get("terminated", False)))
        if hasattr(result, "done"):
            return bool(result.done)
        if hasattr(result, "terminated"):
            return bool(result.terminated)
        return False

    def _extract_info(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, (tuple, list)) and len(result) >= 4:
            info = result[3]
        elif isinstance(result, dict):
            info = result.get("info", {})
        elif hasattr(result, "info"):
            info = result.info
        else:
            info = {}
        return dict(info) if info is not None else {}

    # --------------------------------------------------------------------- #
    # Simulator-state capture for mixed-initial-state refining
    # --------------------------------------------------------------------- #
    def get_simulator_state(self) -> Dict[str, Any]:
        """Return a dictionary that can be used to restore the CybORG state.

        .. note::
            Full state restoration in CybORG is non-trivial because the
            simulator keeps internal randomness and agent state.  We therefore
            store the episode step count, last observation, and the raw CybORG
            ``get_agent_state`` snapshot when available.  The reset wrapper
            falls back to re-initialising the episode if exact restoration is
            not possible.
        """
        state: Dict[str, Any] = {
            "step_count": self._step_count,
            "last_obs": self._last_obs.copy() if self._last_obs is not None else None,
            "last_info": dict(self._last_info),
            "last_action_name": self._last_action_name,
            "trial_length": self.trial_length,
            "scenario_file": self.scenario_file,
        }
        if self._cyborg is not None and hasattr(self._cyborg, "get_agent_state"):
            try:
                state["cyborg_state"] = self._cyborg.get_agent_state("Blue")
            except Exception:  # pragma: no cover
                pass
        return state

    def set_simulator_state(self, state: Dict[str, Any]) -> None:
        """Best-effort restoration of a previously captured CybORG state."""
        self._step_count = state.get("step_count", 0)
        self._last_info = dict(state.get("last_info", {}))
        self._last_action_name = state.get("last_action_name", None)
        # CybORG does not expose a public ``set_state``; the reset wrapper will
        # fall back to a fresh reset if exact restoration fails.


def make_cage_env(
    trial_length: int = 50,
    scenario_file: Optional[str] = None,
    red_agent: str = "B-lineAgent",
    restore_penalty: float = -1.0,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> CageChallenge2Adapter:
    """Factory for the CAGE Challenge 2 adapter."""
    return CageChallenge2Adapter(
        scenario_file=scenario_file,
        trial_length=trial_length,
        red_agent=red_agent,
        restore_penalty=restore_penalty,
        seed=seed,
        **kwargs,
    )
