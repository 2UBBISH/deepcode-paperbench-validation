"""
CAGE Challenge 2 Environment Wrapper for RICE

Implements a gym-compatible environment wrapper for the CAGE Challenge 2
cybersecurity domain, with state save/restore capabilities enabling the RICE
pipeline to collect critical states and reset to them during refinement.

The CAGE Challenge 2 (https://github.com/cage-challenge/cage-challenge-2) is a
cyber defense competition where a blue agent must defend a network against a red
agent. The environment provides:
- Network topology with multiple hosts and services
- Discrete action space (defensive actions like patch, isolate, scan, etc.)
- Observation space encoding network state, host compromises, service status
- Reward based on successful defense and service availability

If the real CAGE2 environment is not installed, a simulated environment is
provided as fallback for development and testing.

Reference: "CAGE Challenge 2" - TTCP CAGE Working Group
Champion scheme: https://github.com/john-cardiff/-cyborg-cage-2
"""

import copy
import pickle
import os
import sys
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import gym
from gym import spaces

# ---------------------------------------------------------------------------
# Try to import the real CAGE2 environment
# ---------------------------------------------------------------------------
HAS_CAGE2 = False
try:
    # Attempt to import CybORG (the underlying simulator for CAGE2)
    import CybORG
    from CybORG import CybORG as CybORGEnv
    HAS_CAGE2 = True
except ImportError:
    pass

if not HAS_CAGE2:
    try:
        # Alternative import path
        import cyborg
        HAS_CAGE2 = True
    except ImportError:
        pass


# ===========================================================================
# Simulated CAGE2 Environment (fallback when real env not available)
# ===========================================================================
class SimulatedCage2Env(gym.Env):
    """
    A simplified simulation of the CAGE Challenge 2 environment.
    
    Models a network with multiple hosts where the blue agent must defend
    against red agent attacks. The state encodes host compromise levels,
    service status, and network topology information.
    
    This is used as a fallback when the real CybORG/CAGE2 environment is
    not installed, allowing development and testing of the RICE pipeline.
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Action definitions (matching CAGE2 blue agent actions)
    ACTION_NAMES = [
        "Analyze_0",      # Analyze host 0
        "Analyze_1",      # Analyze host 1
        "Analyze_2",      # Analyze host 2
        "Remove_0",       # Remove malware from host 0
        "Remove_1",       # Remove malware from host 1
        "Remove_2",       # Remove malware from host 2
        "Restore_0",      # Restore host 0
        "Restore_1",      # Restore host 1
        "Restore_2",      # Restore host 2
        "DeployDecoy_0",  # Deploy decoy on host 0
        "DeployDecoy_1",  # Deploy decoy on host 1
        "DeployDecoy_2",  # Deploy decoy on host 2
        "Monitor_0",      # Monitor subnet 0
        "Monitor_1",      # Monitor subnet 1
        "Sleep",          # No action
    ]
    
    NUM_ACTIONS = len(ACTION_NAMES)
    
    def __init__(
        self,
        num_hosts: int = 3,
        num_subnets: int = 2,
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.num_hosts = num_hosts
        self.num_subnets = num_subnets
        self.max_steps = max_steps
        self._seed = seed
        
        # State components per host:
        # - compromised (0/1)
        # - service_available (0/1) 
        # - analyzed (0/1)
        # - decoy_deployed (0/1)
        # - malware_type (0=none, 1-3=types)
        # Plus global: step_count, red_activity_level, alerts
        self.state_dim = num_hosts * 5 + 3
        self.action_dim = self.NUM_ACTIONS
        
        # Observation space: continuous vector encoding network state
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Action space: discrete actions
        self.action_space = spaces.Discrete(self.NUM_ACTIONS)
        
        # Internal state
        self.rng = np.random.RandomState(seed)
        self.current_step = 0
        self._state = None
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self._seed = seed
            
        self.current_step = 0
        
        # Initialize host states: [compromised, service_available, analyzed, decoy_deployed, malware_type]
        self.host_states = np.zeros((self.num_hosts, 5), dtype=np.float32)
        self.host_states[:, 1] = 1.0  # All services initially available
        
        # Randomly compromise 0-1 hosts initially
        if self.rng.random() < 0.3:
            host_idx = self.rng.randint(0, self.num_hosts)
            self.host_states[host_idx, 0] = 1.0  # compromised
            self.host_states[host_idx, 1] = 0.0  # service unavailable
            self.host_states[host_idx, 4] = self.rng.randint(1, 4)  # malware type
        
        self.red_activity = 0.0
        self.alerts = 0.0
        
        self._state = self._build_state()
        return self._state.copy(), {}
    
    def step(self, action: int):
        """Execute an action and return (obs, reward, terminated, truncated, info)."""
        self.current_step += 1
        
        reward = 0.0
        info = {"action_name": self.ACTION_NAMES[action]}
        
        # Parse action
        action_name = self.ACTION_NAMES[action]
        parts = action_name.split("_")
        action_type = parts[0]
        target = int(parts[1]) if len(parts) > 1 else -1
        
        # Execute blue agent action
        if action_type == "Analyze" and 0 <= target < self.num_hosts:
            # Analyze reveals compromise status
            self.host_states[target, 2] = 1.0  # analyzed
            if self.host_states[target, 0] > 0:
                reward += 0.5  # Found compromise
                
        elif action_type == "Remove" and 0 <= target < self.num_hosts:
            # Remove malware from host
            if self.host_states[target, 0] > 0:
                success_prob = 0.7
                if self.rng.random() < success_prob:
                    self.host_states[target, 0] = 0.0
                    self.host_states[target, 4] = 0.0
                    self.host_states[target, 1] = 1.0  # restore service
                    reward += 2.0
                    
        elif action_type == "Restore" and 0 <= target < self.num_hosts:
            # Restore host to clean state
            self.host_states[target, 0] = 0.0
            self.host_states[target, 1] = 1.0
            self.host_states[target, 4] = 0.0
            reward += 1.0
            
        elif action_type == "DeployDecoy" and 0 <= target < self.num_hosts:
            # Deploy decoy (attracts red agent, reduces compromise probability)
            self.host_states[target, 3] = 1.0
            reward += 0.2
            
        elif action_type == "Monitor":
            # Monitor subnet - detect red activity
            self.alerts = min(1.0, self.alerts + 0.3)
            reward += 0.1
            
        elif action_type == "Sleep":
            # No action
            pass
        
        # Red agent activity (simplified)
        self.red_activity = self.rng.random() * 0.5
        
        # Red agent may compromise hosts
        for i in range(self.num_hosts):
            if self.host_states[i, 0] == 0:  # not compromised
                # Decoy reduces compromise probability
                compromise_prob = 0.1 * (1.0 - 0.7 * self.host_states[i, 3])
                if self.rng.random() < compromise_prob:
                    self.host_states[i, 0] = 1.0
                    self.host_states[i, 1] = 0.0
                    self.host_states[i, 4] = self.rng.randint(1, 4)
                    reward -= 1.0
        
        # Service availability reward
        for i in range(self.num_hosts):
            if self.host_states[i, 1] > 0:
                reward += 0.05  # Small ongoing reward for available services
        
        # Build state
        self._state = self._build_state()
        
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return self._state.copy(), reward, terminated, truncated, info
    
    def _build_state(self) -> np.ndarray:
        """Build the observation vector from internal state."""
        state_parts = []
        
        # Host states flattened
        state_parts.append(self.host_states.flatten())
        
        # Global features
        global_features = np.array([
            self.current_step / self.max_steps,
            self.red_activity,
            self.alerts,
        ], dtype=np.float32)
        state_parts.append(global_features)
        
        return np.concatenate(state_parts).astype(np.float32)
    
    def get_state(self) -> Dict[str, Any]:
        """Return a serializable state dictionary."""
        return {
            "host_states": self.host_states.copy(),
            "red_activity": self.red_activity,
            "alerts": self.alerts,
            "current_step": self.current_step,
            "rng_state": self.rng.get_state(),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore environment from a state dictionary."""
        self.host_states = state["host_states"].copy()
        self.red_activity = state["red_activity"]
        self.alerts = state["alerts"]
        self.current_step = state["current_step"]
        self.rng.set_state(state["rng_state"])
        self._state = self._build_state()
    
    def render(self, mode="human"):
        """Render the current state."""
        if mode == "human":
            print(f"Step: {self.current_step}/{self.max_steps}")
            for i in range(self.num_hosts):
                status = "COMPROMISED" if self.host_states[i, 0] > 0 else "CLEAN"
                svc = "UP" if self.host_states[i, 1] > 0 else "DOWN"
                analyzed = "YES" if self.host_states[i, 2] > 0 else "NO"
                decoy = "YES" if self.host_states[i, 3] > 0 else "NO"
                print(f"  Host {i}: {status}, Service: {svc}, Analyzed: {analyzed}, Decoy: {decoy}")
            print(f"  Red Activity: {self.red_activity:.2f}, Alerts: {self.alerts:.2f}")
    
    def seed(self, seed=None):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)
        self._seed = seed
        return [seed]


# ===========================================================================
# Real CAGE2 Environment Wrapper
# ===========================================================================
class RealCage2Wrapper(gym.Wrapper):
    """
    Wrapper for the real CybORG/CAGE2 environment.
    
    Adapts the CybORG interface to the standard Gym API and adds
    state save/restore capabilities.
    """
    
    def __init__(self, env, max_steps: int = 100):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        self._saved_state = None
        
        # Determine observation and action spaces from the wrapped env
        if hasattr(env, 'observation_space'):
            self.observation_space = env.observation_space
        else:
            # Default: assume flat vector observation
            obs = self._get_obs()
            if isinstance(obs, np.ndarray):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.shape, dtype=np.float32
                )
            elif isinstance(obs, dict):
                # Dict observation space - flatten for RICE
                flat_dim = self._flatten_obs(obs).shape[0]
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(flat_dim,), dtype=np.float32
                )
        
        if hasattr(env, 'action_space'):
            self.action_space = env.action_space
        else:
            # Default: assume discrete actions
            self.action_space = spaces.Discrete(15)  # Typical CAGE2 blue actions
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        self.current_step = 0
        if hasattr(self.env, 'reset'):
            obs = self.env.reset()
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return self._process_obs(obs), {}
    
    def step(self, action):
        """Execute an action."""
        self.current_step += 1
        
        if hasattr(self.env, 'step'):
            result = self.env.step(action)
            # Handle different return formats
            if len(result) == 4:
                obs, reward, done, info = result
                terminated, truncated = done, False
            elif len(result) == 5:
                obs, reward, terminated, truncated, info = result
            else:
                obs, reward = result[0], result[1]
                terminated = False
                truncated = self.current_step >= self.max_steps
                info = {}
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            reward = 0.0
            terminated = False
            truncated = self.current_step >= self.max_steps
            info = {}
        
        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Get current observation from the environment."""
        if hasattr(self.env, 'get_observation'):
            return self.env.get_observation()
        elif hasattr(self.env, 'observation'):
            return self.env.observation
        else:
            return np.zeros(1)
    
    def _process_obs(self, obs):
        """Process observation to flat vector."""
        if isinstance(obs, dict):
            return self._flatten_obs(obs)
        elif isinstance(obs, np.ndarray):
            return obs.astype(np.float32).flatten()
        elif isinstance(obs, (list, tuple)):
            return np.array(obs, dtype=np.float32).flatten()
        return np.array([obs], dtype=np.float32)
    
    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        """Flatten a dictionary observation into a vector."""
        parts = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, np.ndarray):
                parts.append(val.flatten())
            elif isinstance(val, (list, tuple)):
                parts.append(np.array(val).flatten())
            else:
                parts.append(np.array([val]))
        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.zeros(1, dtype=np.float32)
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current environment state."""
        state = {
            "current_step": self.current_step,
        }
        
        # Try to save underlying environment state
        if hasattr(self.env, 'get_state'):
            state["env_state"] = self.env.get_state()
        elif hasattr(self.env, 'state'):
            state["env_state"] = copy.deepcopy(self.env.state)
        elif hasattr(self.env, '__getstate__'):
            state["env_state"] = self.env.__getstate__()
        else:
            # Fallback: save observation
            state["observation"] = self._get_obs()
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore the environment to a saved state."""
        self.current_step = state.get("current_step", 0)
        
        if "env_state" in state:
            if hasattr(self.env, 'set_state'):
                self.env.set_state(state["env_state"])
            elif hasattr(self.env, '__setstate__'):
                self.env.__setstate__(state["env_state"])
        # If only observation saved, we cannot fully restore
    
    def render(self, mode="human"):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)


# ===========================================================================
# State Save/Restore Wrapper (unified interface for RICE pipeline)
# ===========================================================================
class Cage2StateWrapper(gym.Wrapper):
    """
    Unified state save/restore wrapper for CAGE2 environments.
    
    Provides the standard interface expected by the RICE pipeline:
    - save_state() -> Dict
    - restore_state(state: Dict) -> None
    - reset_to_state(state: Dict) -> np.ndarray
    
    Works with both the simulated and real CAGE2 environments.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def save_state(self) -> Dict[str, Any]:
        """Save the full environment state for later restoration."""
        state = {}
        
        # Save from underlying env if available
        if hasattr(self.env, 'save_state'):
            state = self.env.save_state()
        elif hasattr(self.env, 'get_state'):
            state = {
                "env_state": self.env.get_state(),
                "unwrapped_state": copy.deepcopy(
                    self.env.unwrapped.get_state()
                ) if hasattr(self.env.unwrapped, 'get_state') else None,
            }
        elif hasattr(self.env.unwrapped, 'get_state'):
            state = {
                "env_state": self.env.unwrapped.get_state(),
            }
        else:
            # Deep copy fallback
            try:
                state = {
                    "env_copy": copy.deepcopy(self.env),
                }
            except Exception:
                state = {"observation": None}
        
        # Always include current step count
        if hasattr(self.env, 'current_step'):
            state["current_step"] = self.env.current_step
        
        return state
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore environment to a previously saved state."""
        if "env_state" in state:
            if hasattr(self.env, 'set_state'):
                self.env.set_state(state["env_state"])
            elif hasattr(self.env.unwrapped, 'set_state'):
                self.env.unwrapped.set_state(state["env_state"])
        
        if "current_step" in state and hasattr(self.env, 'current_step'):
            self.env.current_step = state["current_step"]
    
    def reset_to_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Reset the environment and restore to a saved state.
        
        This is the primary interface used by the RICE refining pipeline
        for the mixed initial distribution.
        """
        # Perform a normal reset first to initialize internals
        obs, _ = self.env.reset()
        
        # Then restore the saved state
        self.restore_state(state)
        
        # Get the current observation after restoration
        if hasattr(self.env, '_get_obs'):
            obs = self.env._get_obs()
        elif hasattr(self.env, 'get_observation'):
            obs = self.env.get_observation()
        
        # Process observation
        if isinstance(obs, dict):
            obs = self._flatten_obs(obs)
        elif isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32).flatten()
        
        return obs
    
    def _flatten_obs(self, obs_dict: Dict) -> np.ndarray:
        """Flatten dictionary observation to vector."""
        parts = []
        for key in sorted(obs_dict.keys()):
            val = obs_dict[key]
            if isinstance(val, np.ndarray):
                parts.append(val.flatten())
            elif isinstance(val, (list, tuple)):
                parts.append(np.array(val).flatten())
            else:
                parts.append(np.array([val]))
        if parts:
            return np.concatenate(parts).astype(np.float32)
        return np.zeros(1, dtype=np.float32)


# ===========================================================================
# Factory Functions
# ===========================================================================
def make_env(
    env_name: str = "Cage2-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    use_real_env: bool = False,
    **kwargs
) -> gym.Env:
    """
    Create a CAGE2 environment with state save/restore capability.
    
    Args:
        env_name: Environment identifier (default "Cage2-v0")
        seed: Random seed
        max_episode_steps: Maximum steps per episode (default 100)
        use_real_env: If True, attempt to use real CybORG environment
        **kwargs: Additional arguments passed to the environment constructor
    
    Returns:
        Wrapped gym.Env with state save/restore
    """
    if max_episode_steps is None:
        max_episode_steps = kwargs.pop("max_steps", 100)
    
    # Try to create real environment if requested and available
    if use_real_env and HAS_CAGE2:
        try:
            # Create CybORG environment with CAGE2 scenario
            from CybORG import CybORG
            from CybORG.Shared import Scenario
            
            # Use CAGE2 scenario (Scenario 2)
            scenario = Scenario.Scenario2 if hasattr(Scenario, 'Scenario2') else "Scenario2"
            cyborg_env = CybORG(scenario, 'sim')
            
            # Wrap in Gym interface
            env = RealCage2Wrapper(cyborg_env, max_steps=max_episode_steps)
        except Exception as e:
            print(f"Warning: Failed to create real CAGE2 env: {e}")
            print("Falling back to simulated environment.")
            env = SimulatedCage2Env(
                max_steps=max_episode_steps,
                seed=seed,
                **kwargs
            )
    else:
        # Use simulated environment
        env = SimulatedCage2Env(
            max_steps=max_episode_steps,
            seed=seed,
            **kwargs
        )
    
    # Wrap with state save/restore
    env = Cage2StateWrapper(env)
    
    # Set seed
    if hasattr(env, 'seed'):
        env.seed(seed)
    
    return env


def make_state_saveable(env: gym.Env) -> Cage2StateWrapper:
    """
    Wrap an existing CAGE2 environment with state save/restore capability.
    
    Args:
        env: An existing gym environment (simulated or real CAGE2)
    
    Returns:
        Cage2StateWrapper with save_state/restore_state/reset_to_state
    """
    if isinstance(env, Cage2StateWrapper):
        return env
    return Cage2StateWrapper(env)


def get_state_dim(env: gym.Env) -> int:
    """
    Get the observation dimension for a CAGE2 environment.
    
    Args:
        env: CAGE2 environment (wrapped or unwrapped)
    
    Returns:
        Dimension of the flattened observation vector
    """
    if hasattr(env, 'observation_space'):
        space = env.observation_space
        if isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
        elif isinstance(space, spaces.Discrete):
            return space.n
    
    # Try to infer from unwrapped env
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(unwrapped, 'state_dim'):
        return unwrapped.state_dim
    if hasattr(unwrapped, 'num_hosts'):
        return unwrapped.num_hosts * 5 + 3
    
    # Default for CAGE2
    return 18  # 3 hosts * 5 features + 3 global


def get_action_dim(env: gym.Env) -> int:
    """
    Get the action dimension for a CAGE2 environment.
    
    Args:
        env: CAGE2 environment
    
    Returns:
        Number of discrete actions
    """
    if hasattr(env, 'action_space'):
        space = env.action_space
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.Box):
            return int(np.prod(space.shape))
    
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(unwrapped, 'NUM_ACTIONS'):
        return unwrapped.NUM_ACTIONS
    if hasattr(unwrapped, 'action_dim'):
        return unwrapped.action_dim
    
    return 15  # Default CAGE2 blue actions


def is_discrete_action(env: gym.Env) -> bool:
    """
    Check if the environment has a discrete action space.
    
    Args:
        env: CAGE2 environment
    
    Returns:
        True if action space is discrete
    """
    if hasattr(env, 'action_space'):
        return isinstance(env.action_space, spaces.Discrete)
    
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(unwrapped, 'action_space'):
        return isinstance(unwrapped.action_space, spaces.Discrete)
    
    return True  # CAGE2 actions are discrete


# ===========================================================================
# Sparse Reward Wrapper (for ablation studies)
# ===========================================================================
class SparseRewardWrapper(gym.Wrapper):
    """
    Converts dense CAGE2 rewards to sparse rewards.
    
    In sparse mode, reward is only given for:
    - Successful malware removal (+1)
    - Host compromise (-1)
    - All hosts clean at episode end (+5 bonus)
    """
    
    def __init__(
        self,
        env: gym.Env,
        success_reward: float = 5.0,
        removal_reward: float = 1.0,
        compromise_penalty: float = -1.0,
    ):
        super().__init__(env)
        self.success_reward = success_reward
        self.removal_reward = removal_reward
        self.compromise_penalty = compromise_penalty
        self._prev_host_states = None
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_host_states = self._get_host_states()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Compute sparse reward
        sparse_reward = 0.0
        current_host_states = self._get_host_states()
        
        if self._prev_host_states is not None and current_host_states is not None:
            # Check for changes in compromise status
            for i in range(len(current_host_states)):
                prev_comp = self._prev_host_states[i][0] if len(self._prev_host_states[i]) > 0 else 0
                curr_comp = current_host_states[i][0] if len(current_host_states[i]) > 0 else 0
                
                if prev_comp > 0 and curr_comp == 0:
                    sparse_reward += self.removal_reward  # Removed malware
                elif prev_comp == 0 and curr_comp > 0:
                    sparse_reward += self.compromise_penalty  # New compromise
        
        # Episode end bonus for all clean
        if terminated or truncated:
            all_clean = all(
                (s[0] == 0 if len(s) > 0 else True)
                for s in (current_host_states or [])
            )
            if all_clean:
                sparse_reward += self.success_reward
        
        self._prev_host_states = current_host_states
        
        return obs, sparse_reward, terminated, truncated, info
    
    def _get_host_states(self):
        """Extract host states from the environment."""
        unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        if hasattr(unwrapped, 'host_states'):
            return unwrapped.host_states
        return None


# ===========================================================================
# Testing
# ===========================================================================
def test_env():
    """Quick integration test for the CAGE2 environment."""
    print("Testing CAGE2 environment...")
    
    env = make_env("Cage2-v0", seed=42, max_episode_steps=50)
    
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  State dim: {get_state_dim(env)}")
    print(f"  Action dim: {get_action_dim(env)}")
    print(f"  Discrete actions: {is_discrete_action(env)}")
    
    # Test reset
    obs, _ = env.reset()
    print(f"  Initial obs shape: {obs.shape}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Step result: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
    
    # Test state save/restore
    state = env.save_state()
    print(f"  Saved state keys: {list(state.keys())}")
    
    obs2, _ = env.reset()
    obs3 = env.reset_to_state(state)
    print(f"  Reset-to-state obs shape: {obs3.shape}")
    
    # Test multiple episodes
    total_reward = 0.0
    obs, _ = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  Episode total reward: {total_reward:.3f}")
    
    # Test sparse reward wrapper
    sparse_env = SparseRewardWrapper(make_env("Cage2-v0", seed=42, max_episode_steps=30))
    obs, _ = sparse_env.reset()
    sparse_total = 0.0
    for _ in range(20):
        action = sparse_env.action_space.sample()
        obs, reward, terminated, truncated, info = sparse_env.step(action)
        sparse_total += reward
        if terminated or truncated:
            break
    print(f"  Sparse episode total reward: {sparse_total:.3f}")
    
    print("CAGE2 environment test PASSED!")
    return True


if __name__ == "__main__":
    test_env()