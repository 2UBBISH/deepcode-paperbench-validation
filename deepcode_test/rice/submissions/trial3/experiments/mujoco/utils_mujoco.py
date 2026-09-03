"""
MuJoCo-specific utilities for the RICE pipeline.

Provides helper functions for:
- Saving/loading MuJoCo simulation states to/from disk
- Extracting and setting state vectors
- Collecting environment states during trajectory rollouts
- Creating sparse reward variants of MuJoCo environments
- Environment creation with consistent configuration
- Environment metadata (state_dim, action_dim, max_steps)

These utilities complement the generic StateSaveWrapper in rice.env_wrappers
with MuJoCo-specific optimizations and convenience functions.
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable

import gym

# Try importing mujoco for version-adaptive state handling
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

try:
    from mujoco_py import MjSimState
    HAS_MUJOCO_PY = True
except ImportError:
    HAS_MUJOCO_PY = False


# ==============================================================================
# Environment Metadata
# ==============================================================================

# Known MuJoCo environment specifications
MUJOCO_ENV_INFO = {
    "Hopper-v4": {"state_dim": 11, "action_dim": 3, "max_episode_steps": 1000},
    "Hopper-v3": {"state_dim": 11, "action_dim": 3, "max_episode_steps": 1000},
    "Hopper-v2": {"state_dim": 11, "action_dim": 3, "max_episode_steps": 1000},
    "Walker2d-v4": {"state_dim": 17, "action_dim": 6, "max_episode_steps": 1000},
    "Walker2d-v3": {"state_dim": 17, "action_dim": 6, "max_episode_steps": 1000},
    "Walker2d-v2": {"state_dim": 17, "action_dim": 6, "max_episode_steps": 1000},
    "HalfCheetah-v4": {"state_dim": 17, "action_dim": 6, "max_episode_steps": 1000},
    "HalfCheetah-v3": {"state_dim": 17, "action_dim": 6, "max_episode_steps": 1000},
    "HalfCheetah-v2": {"state_dim": 17, "action_dim": 6, "max_episode_steps": 1000},
    "Reacher-v4": {"state_dim": 11, "action_dim": 2, "max_episode_steps": 50},
    "Reacher-v2": {"state_dim": 11, "action_dim": 2, "max_episode_steps": 50},
    "Ant-v4": {"state_dim": 27, "action_dim": 8, "max_episode_steps": 1000},
    "Ant-v3": {"state_dim": 27, "action_dim": 8, "max_episode_steps": 1000},
    "Humanoid-v4": {"state_dim": 376, "action_dim": 17, "max_episode_steps": 1000},
    "Humanoid-v3": {"state_dim": 376, "action_dim": 17, "max_episode_steps": 1000},
    "Swimmer-v4": {"state_dim": 8, "action_dim": 2, "max_episode_steps": 1000},
    "Swimmer-v3": {"state_dim": 8, "action_dim": 2, "max_episode_steps": 1000},
    "InvertedPendulum-v4": {"state_dim": 4, "action_dim": 1, "max_episode_steps": 1000},
    "InvertedDoublePendulum-v4": {"state_dim": 11, "action_dim": 1, "max_episode_steps": 1000},
    "MountainCarContinuous-v0": {"state_dim": 2, "action_dim": 1, "max_episode_steps": 999},
}


def get_env_info(env_name: str) -> Dict[str, Any]:
    """
    Get metadata for a known MuJoCo environment.

    Args:
        env_name: Gym environment name (e.g., "Hopper-v4")

    Returns:
        Dict with keys: state_dim, action_dim, max_episode_steps.
        If env_name not in known list, attempts to infer from environment.
    """
    if env_name in MUJOCO_ENV_INFO:
        return MUJOCO_ENV_INFO[env_name].copy()

    # Try to infer from environment
    try:
        env = gym.make(env_name)
        info = {
            "state_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "max_episode_steps": env.spec.max_episode_steps if env.spec is not None else 1000,
        }
        env.close()
        return info
    except Exception:
        # Fallback defaults
        return {"state_dim": 11, "action_dim": 3, "max_episode_steps": 1000}


# ==============================================================================
# State Save/Load Utilities
# ==============================================================================

def save_mujoco_state(env: gym.Env, path: str) -> None:
    """
    Save the current MuJoCo simulation state to disk.

    Uses sim.get_state() for mujoco-py or the newer mujoco bindings.
    Falls back to pickle of the full state dict.

    Args:
        env: MuJoCo Gym environment (must have sim attribute or get_state method)
        path: File path to save the state (will be pickled)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    if hasattr(env, "sim"):
        sim = env.sim
        if hasattr(sim, "get_state"):
            # mujoco-py style
            state = sim.get_state()
        elif hasattr(sim, "get_state"):
            # Newer mujoco bindings
            state = sim.get_state()
        else:
            # Fallback: try to capture via MuJoCoStateWrapper
            state = _capture_state_fallback(env)
    elif hasattr(env, "save_state"):
        state = env.save_state()
    elif hasattr(env, "get_state"):
        state = env.get_state()
    else:
        state = _capture_state_fallback(env)

    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_mujoco_state(env: gym.Env, path: str) -> None:
    """
    Load a MuJoCo simulation state from disk and restore it.

    Args:
        env: MuJoCo Gym environment
        path: File path to the saved state (pickle file)
    """
    with open(path, "rb") as f:
        state = pickle.load(f)

    restore_mujoco_state(env, state)


def restore_mujoco_state(env: gym.Env, state: Any) -> None:
    """
    Restore a MuJoCo simulation state.

    Handles both mujoco-py MjSimState objects and newer mujoco data objects.

    Args:
        env: MuJoCo Gym environment
        state: State object (MjSimState, mujoco.MjData, or dict)
    """
    if hasattr(env, "sim"):
        sim = env.sim
        if hasattr(sim, "set_state"):
            sim.set_state(state)
        elif hasattr(sim, "set_state"):
            sim.set_state(state)
        else:
            _restore_state_fallback(env, state)
    elif hasattr(env, "restore_state"):
        env.restore_state(state)
    elif hasattr(env, "set_state"):
        env.set_state(state)
    else:
        _restore_state_fallback(env, state)

    # Force a re-computation of the observation
    if hasattr(env, "sim"):
        env.sim.forward()


def _capture_state_fallback(env: gym.Env) -> Dict[str, Any]:
    """
    Fallback state capture: deep copy of sim data arrays.
    """
    state = {}
    if hasattr(env, "sim"):
        sim = env.sim
        if hasattr(sim.data, "qpos"):
            state["qpos"] = sim.data.qpos.copy()
        if hasattr(sim.data, "qvel"):
            state["qvel"] = sim.data.qvel.copy()
        if hasattr(sim.data, "ctrl"):
            state["ctrl"] = sim.data.ctrl.copy()
        if hasattr(sim.data, "act"):
            state["act"] = sim.data.act.copy()
        if hasattr(sim.data, "time"):
            state["time"] = sim.data.time
        # Capture mocap if present
        if hasattr(sim.data, "mocap_pos"):
            state["mocap_pos"] = sim.data.mocap_pos.copy()
        if hasattr(sim.data, "mocap_quat"):
            state["mocap_quat"] = sim.data.mocap_quat.copy()
    elif hasattr(env, "get_state"):
        state = env.get_state()
    else:
        # Last resort: just save observation
        state = {"observation": _get_observation(env)}
    return state


def _restore_state_fallback(env: gym.Env, state: Any) -> None:
    """
    Fallback state restoration.
    """
    if isinstance(state, dict):
        if hasattr(env, "sim"):
            sim = env.sim
            if "qpos" in state:
                sim.data.qpos[:] = state["qpos"]
            if "qvel" in state:
                sim.data.qvel[:] = state["qvel"]
            if "ctrl" in state:
                sim.data.ctrl[:] = state["ctrl"]
            if "act" in state:
                sim.data.act[:] = state["act"]
            if "time" in state:
                sim.data.time = state["time"]
            if "mocap_pos" in state:
                sim.data.mocap_pos[:] = state["mocap_pos"]
            if "mocap_quat" in state:
                sim.data.mocap_quat[:] = state["mocap_quat"]
            sim.forward()
    elif hasattr(env, "set_state"):
        env.set_state(state)
    elif hasattr(env, "restore_state"):
        env.restore_state(state)


def _get_observation(env: gym.Env) -> np.ndarray:
    """Get current observation from environment."""
    if hasattr(env, "sim"):
        # Try to compute observation from sim state
        if hasattr(env, "_get_obs"):
            return env._get_obs()
        elif hasattr(env, "get_obs"):
            return env.get_obs()
    # Fallback: return last observation if stored
    if hasattr(env, "_last_obs"):
        return env._last_obs
    # Last resort
    obs, _ = env.reset()
    return obs


# ==============================================================================
# State Vector Utilities
# ==============================================================================

def get_mujoco_state_vector(env: gym.Env) -> np.ndarray:
    """
    Extract the full MuJoCo state as a flat numpy vector.

    Concatenates qpos and qvel from the simulation data.

    Args:
        env: MuJoCo Gym environment

    Returns:
        Flat numpy array of [qpos, qvel]
    """
    if hasattr(env, "sim"):
        sim = env.sim
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()
        return np.concatenate([qpos, qvel])
    elif hasattr(env, "get_state_vector"):
        return env.get_state_vector()
    else:
        # Fallback: return observation
        return _get_observation(env)


def set_mujoco_state_from_vector(env: gym.Env, state_vector: np.ndarray) -> None:
    """
    Set the MuJoCo simulation state from a flat vector [qpos, qvel].

    Args:
        env: MuJoCo Gym environment
        state_vector: Flat numpy array with qpos followed by qvel
    """
    if hasattr(env, "sim"):
        sim = env.sim
        nq = sim.model.nq
        nv = sim.model.nv
        if len(state_vector) == nq + nv:
            sim.data.qpos[:] = state_vector[:nq]
            sim.data.qvel[:] = state_vector[nq:nq + nv]
            sim.forward()
        else:
            raise ValueError(
                f"State vector length {len(state_vector)} does not match "
                f"expected nq+nv={nq + nv}"
            )
    elif hasattr(env, "set_state_from_vector"):
        env.set_state_from_vector(state_vector)
    else:
        raise RuntimeError("Environment does not support state vector operations")


def get_qpos_qvel(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get qpos and qvel separately from a MuJoCo environment.

    Args:
        env: MuJoCo Gym environment

    Returns:
        Tuple of (qpos, qvel) as numpy arrays
    """
    if hasattr(env, "sim"):
        sim = env.sim
        return sim.data.qpos.copy(), sim.data.qvel.copy()
    else:
        raise RuntimeError("Environment does not have sim attribute")


# ==============================================================================
# State Collection
# ==============================================================================

def collect_mujoco_states(
    env: gym.Env,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    num_episodes: int = 100,
    max_episode_steps: int = 1000,
    save_states: bool = True,
    deterministic: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Collect full environment states during trajectory rollouts.

    For each episode, records the state at every timestep along with
    the observation, action, reward, and importance metadata.

    Args:
        env: MuJoCo Gym environment (should be wrapped with StateSaveWrapper)
        policy_fn: Function mapping observation -> action
        num_episodes: Number of episodes to collect
        max_episode_steps: Maximum steps per episode
        save_states: Whether to save full environment states (via save_state)
        deterministic: Whether to use deterministic policy actions
        verbose: Print progress

    Returns:
        List of dicts, each containing:
            - episode: episode index
            - states: list of per-step dicts with keys:
                - step: step index
                - observation: numpy array
                - action: numpy array
                - reward: float
                - done: bool
                - env_state: saved environment state (if save_states=True)
                - state_vector: qpos+qvel vector (if available)
            - total_reward: float
            - length: int
    """
    from rice.env_wrappers import save_env_state

    all_episodes = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_data = {
            "episode": ep,
            "states": [],
            "total_reward": 0.0,
            "length": 0,
        }

        for step in range(max_episode_steps):
            action = policy_fn(obs)
            if isinstance(action, tuple):
                action = action[0]  # Handle (action, ...) tuples

            next_obs, reward, terminated, truncated, info = _step_env(env, action)
            done = terminated or truncated

            step_data = {
                "step": step,
                "observation": obs.copy(),
                "action": np.asarray(action).copy(),
                "reward": float(reward),
                "done": bool(done),
                "info": info,
            }

            if save_states:
                try:
                    step_data["env_state"] = save_env_state(env)
                except Exception:
                    step_data["env_state"] = None

            # Try to get state vector
            try:
                step_data["state_vector"] = get_mujoco_state_vector(env)
            except Exception:
                step_data["state_vector"] = None

            episode_data["states"].append(step_data)
            episode_data["total_reward"] += reward
            episode_data["length"] += 1

            obs = next_obs

            if done:
                break

        all_episodes.append(episode_data)

        if verbose and (ep + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Collected {ep + 1}/{num_episodes} episodes, "
                  f"last reward: {episode_data['total_reward']:.2f}")

    return all_episodes


def _step_env(env: gym.Env, action: np.ndarray) -> Tuple:
    """
    Step the environment, handling both old and new Gym APIs.

    Returns: (obs, reward, terminated, truncated, info)
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        terminated = done
        truncated = False
    else:
        raise ValueError(f"Unexpected step result length: {len(result)}")
    return obs, reward, terminated, truncated, info


# ==============================================================================
# Sparse Reward Wrapper
# ==============================================================================

class SparseRewardWrapper(gym.Wrapper):
    """
    Converts dense MuJoCo rewards to sparse rewards based on x-position threshold.

    For locomotion tasks (Hopper, Walker2d, HalfCheetah, Ant), the agent
    receives a reward of +1 only when its x-coordinate exceeds a threshold,
    and 0 otherwise. This matches the sparse reward experiments in the paper
    (Figure 10, Appendix C.4).

    For Reacher, the sparse reward is based on distance to target.
    """

    def __init__(
        self,
        env: gym.Env,
        threshold: float = 1.0,
        reward_value: float = 1.0,
        use_distance: bool = False,
    ):
        """
        Args:
            env: MuJoCo Gym environment
            threshold: X-position threshold for sparse reward
            reward_value: Reward value when condition is met
            use_distance: If True, use distance-based sparse reward (for Reacher)
        """
        super().__init__(env)
        self.threshold = threshold
        self.reward_value = reward_value
        self.use_distance = use_distance
        self._max_x = -float("inf")

    def step(self, action):
        obs, reward, terminated, truncated, info = self.step_wrapped(action)

        # Track max x-position
        if hasattr(self.env, "sim"):
            x_pos = self.env.sim.data.qpos[0]
            self._max_x = max(self._max_x, x_pos)

        if self.use_distance:
            # For Reacher: sparse reward based on distance to target
            # The observation contains the distance-related info
            sparse_reward = self.reward_value if reward > -1.0 else 0.0
        else:
            # For locomotion: sparse reward based on x-position threshold
            sparse_reward = self.reward_value if self._max_x >= self.threshold else 0.0

        info["dense_reward"] = reward
        info["sparse_reward"] = sparse_reward
        info["max_x"] = self._max_x

        return obs, sparse_reward, terminated, truncated, info

    def step_wrapped(self, action):
        """Call the underlying env step, handling API differences."""
        result = self.env.step(action)
        if len(result) == 5:
            return result
        elif len(result) == 4:
            obs, reward, done, info = result
            return obs, reward, done, False, info
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._max_x = -float("inf")
        return obs, info


def make_sparse_env(
    env_name: str,
    threshold: float = 1.0,
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """
    Create a sparse-reward MuJoCo environment.

    Args:
        env_name: Gym environment name
        threshold: X-position threshold for sparse reward
        seed: Random seed
        max_episode_steps: Maximum episode steps (overrides default)

    Returns:
        SparseRewardWrapper-wrapped environment
    """
    from rice.env_wrappers import make_state_saveable

    env = gym.make(env_name)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    # Determine if we should use distance-based sparse reward (for Reacher)
    use_distance = "reacher" in env_name.lower()

    env = SparseRewardWrapper(env, threshold=threshold, use_distance=use_distance)
    env = make_state_saveable(env)
    env.reset(seed=seed)

    return env


# ==============================================================================
# Environment Creation
# ==============================================================================

def make_mujoco_env(
    env_name: str,
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    sparse: bool = False,
    sparse_threshold: float = 1.0,
) -> gym.Env:
    """
    Create a MuJoCo environment with consistent configuration.

    Wraps the environment with StateSaveWrapper for state save/restore.

    Args:
        env_name: Gym environment name (e.g., "Hopper-v4")
        seed: Random seed
        max_episode_steps: Maximum episode steps (overrides default)
        sparse: If True, wrap with SparseRewardWrapper
        sparse_threshold: Threshold for sparse reward

    Returns:
        Configured Gym environment with state save/restore capability
    """
    from rice.env_wrappers import make_state_saveable

    env = gym.make(env_name)

    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

    if sparse:
        use_distance = "reacher" in env_name.lower()
        env = SparseRewardWrapper(
            env, threshold=sparse_threshold, use_distance=use_distance
        )

    env = make_state_saveable(env)
    env.reset(seed=seed)

    return env


# ==============================================================================
# State Comparison
# ==============================================================================

def compare_mujoco_states(
    state1: Any, state2: Any, tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Compare two MuJoCo states and return differences.

    Args:
        state1: First state (MjSimState, dict, or numpy array)
        state2: Second state
        tolerance: Tolerance for equality check

    Returns:
        Dict with keys:
            - equal: bool
            - max_diff: float
            - qpos_diff: numpy array (if applicable)
            - qvel_diff: numpy array (if applicable)
    """
    result = {"equal": True, "max_diff": 0.0}

    # Handle MjSimState objects
    if hasattr(state1, "qpos") and hasattr(state2, "qpos"):
        qpos_diff = np.abs(state1.qpos - state2.qpos)
        qvel_diff = np.abs(state1.qvel - state2.qvel)
        result["qpos_diff"] = qpos_diff
        result["qvel_diff"] = qvel_diff
        result["max_diff"] = max(qpos_diff.max(), qvel_diff.max())
        result["equal"] = result["max_diff"] < tolerance

    # Handle dict states
    elif isinstance(state1, dict) and isinstance(state2, dict):
        max_diff = 0.0
        for key in set(state1.keys()) | set(state2.keys()):
            if key in state1 and key in state2:
                v1, v2 = state1[key], state2[key]
                if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
                    diff = np.abs(v1 - v2).max()
                    max_diff = max(max_diff, diff)
                    result[f"{key}_diff"] = diff
        result["max_diff"] = max_diff
        result["equal"] = max_diff < tolerance

    # Handle numpy arrays
    elif isinstance(state1, np.ndarray) and isinstance(state2, np.ndarray):
        diff = np.abs(state1 - state2).max()
        result["max_diff"] = diff
        result["equal"] = diff < tolerance

    return result


# ==============================================================================
# Batch State Operations
# ==============================================================================

def save_critical_states_batch(
    states: List[Dict[str, Any]],
    output_dir: str,
    env_name: str,
) -> List[str]:
    """
    Save a batch of critical states to disk.

    Args:
        states: List of state dicts (from collect_mujoco_states or RICERefine)
        output_dir: Directory to save states
        env_name: Environment name for file naming

    Returns:
        List of file paths to saved states
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, state in enumerate(states):
        path = os.path.join(output_dir, f"{env_name}_critical_state_{i:05d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
        paths.append(path)

    return paths


def load_critical_states_batch(
    paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Load a batch of critical states from disk.

    Args:
        paths: List of file paths to saved states

    Returns:
        List of state dicts
    """
    states = []
    for path in paths:
        with open(path, "rb") as f:
            state = pickle.load(f)
        states.append(state)
    return states


# ==============================================================================
# Utility: Check if environment is MuJoCo-based
# ==============================================================================

def is_mujoco_env(env: gym.Env) -> bool:
    """
    Check if an environment is MuJoCo-based.

    Args:
        env: Gym environment

    Returns:
        True if the environment has a sim attribute (MuJoCo)
    """
    return hasattr(env, "sim") or hasattr(env.unwrapped, "sim")


# ==============================================================================
# Test
# ==============================================================================

def test_utils():
    """Quick test of MuJoCo utilities."""
    print("Testing MuJoCo utilities...")

    # Test get_env_info
    info = get_env_info("Hopper-v4")
    print(f"  Hopper-v4 info: {info}")

    # Test environment creation
    try:
        env = make_mujoco_env("Hopper-v4", seed=42)
        print(f"  Created env: {env}")

        # Test state vector
        obs, _ = env.reset()
        state_vec = get_mujoco_state_vector(env)
        print(f"  State vector shape: {state_vec.shape}")

        # Test state save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        save_mujoco_state(env, tmp_path)
        print(f"  Saved state to {tmp_path}")

        # Modify state slightly
        env.step(env.action_space.sample())
        new_state_vec = get_mujoco_state_vector(env)

        # Restore
        load_mujoco_state(env, tmp_path)
        restored_vec = get_mujoco_state_vector(env)
        print(f"  State restored, diff: {np.abs(state_vec - restored_vec).max():.6f}")

        os.unlink(tmp_path)
        env.close()
        print("  All tests passed!")
    except Exception as e:
        print(f"  Test failed: {e}")


if __name__ == "__main__":
    test_utils()