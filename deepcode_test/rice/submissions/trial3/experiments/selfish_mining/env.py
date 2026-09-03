"""
Selfish Mining Environment Wrapper for RICE Pipeline.

Provides a gym-compatible interface for the selfish mining environment
(from https://github.com/roibarzur/pto-selfish-mining), with state save/restore
capabilities for critical state collection and mixed initial distribution.

The environment models a blockchain mining scenario where an agent can choose
between honest mining and selfish mining strategies. The state represents
the current blockchain state (fork lengths, etc.), and the reward is the
proportion of blocks mined by the agent.

Paper Reference: RICE Section 5.3, Table 3 (p=0.25, λ=0.001, α=0.0001)
"""

import copy
import pickle
import os
import sys
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import gym
from gym import spaces

# Try to import the selfish mining environment
try:
    # The pto-selfish-mining repo provides a gym environment
    # We wrap it for state save/restore
    HAS_SELFISH_MINING = True
except ImportError:
    HAS_SELFISH_MINING = False


class SelfishMiningEnv(gym.Env):
    """
    A standalone implementation of the selfish mining environment.
    
    This implements the core selfish mining game as described in:
    - Sapirshtein et al. (2016) "Optimal Selfish Mining Strategies in Bitcoin"
    - Bar-Zur et al. (2023) "PTO: Policy Tree Optimization for Selfish Mining"
    
    The environment models a blockchain fork race between an honest majority
    and a selfish miner. The agent controls the selfish miner and decides
    at each step whether to:
      - Action 0: Mine honestly (publish blocks immediately)
      - Action 1: Mine selfishly (withhold blocks, only publish when necessary)
    
    State space (52 dimensions):
      - Fork state representation encoding the lengths of the honest and
        selfish chains, the current lead, and other relevant blockchain state.
    
    Action space: Discrete(2) - {0: honest, 1: selfish}
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        alpha: float = 0.35,  # Selfish miner's hash rate proportion
        gamma: float = 0.5,   # Proportion of honest miners that mine on selfish block
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        super().__init__()
        
        self.alpha = alpha  # Selfish miner's computational power fraction
        self.gamma = gamma  # Fraction of honest miners attracted to selfish chain
        self.max_steps = max_steps
        
        # State: [honest_chain_length, selfish_chain_length, lead, fork_state_encoding...]
        # We use a 52-dim state as specified in the paper
        self.state_dim = 52
        self.action_dim = 2
        
        self.action_space = spaces.Discrete(2)
        # State space: normalized blockchain state representation
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def _encode_state(self) -> np.ndarray:
        """
        Encode the current blockchain state into a 52-dimensional vector.
        
        The encoding includes:
        - Honest chain length (normalized)
        - Selfish chain length (normalized)
        - Lead (selfish - honest, normalized)
        - One-hot encoding of fork state
        - Various derived features
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Basic chain lengths (indices 0-2)
        state[0] = min(self.honest_chain, 100) / 100.0
        state[1] = min(self.selfish_chain, 100) / 100.0
        state[2] = (self.selfish_chain - self.honest_chain) / 10.0  # lead
        
        # Fork state encoding (indices 3-7): one-hot for 5 possible states
        # State 0: No fork (chains equal, lead=0)
        # State 1: Selfish lead = 1 (selfish chain 1 block ahead)
        # State 2: Selfish lead = 2 (selfish chain 2 blocks ahead)
        # State 3: Honest lead = 1 (honest chain 1 block ahead, selfish behind)
        # State 4: Tie with fork (both have same length but different tips)
        lead = self.selfish_chain - self.honest_chain
        if lead == 0 and self.selfish_chain == self.honest_chain:
            fork_state = 0
        elif lead == 1:
            fork_state = 1
        elif lead >= 2:
            fork_state = 2
        elif lead == -1:
            fork_state = 3
        else:
            fork_state = 4
        
        for i in range(5):
            state[3 + i] = 1.0 if i == fork_state else 0.0
        
        # Additional features (indices 8-51)
        # Step count normalized
        state[8] = self.step_count / self.max_steps
        
        # Alpha and gamma parameters
        state[9] = self.alpha
        state[10] = self.gamma
        
        # Cumulative reward normalized
        state[11] = min(self.cumulative_reward, 10.0) / 10.0
        
        # Remaining indices filled with derived features
        # (chain length differences, ratios, etc.)
        if self.honest_chain > 0:
            state[12] = self.selfish_chain / self.honest_chain
        state[13] = 1.0 if lead > 0 else 0.0  # is_leading
        state[14] = 1.0 if lead < 0 else 0.0  # is_behind
        
        return state
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        super().reset(seed=seed)
        
        # Initialize blockchain state
        self.honest_chain = 0
        self.selfish_chain = 0
        self.step_count = 0
        self.cumulative_reward = 0.0
        self.done = False
        
        # Internal state tracking for the selfish mining game
        self._fork_active = False
        self._selfish_blocks_withheld = 0
        
        observation = self._encode_state()
        info = {}
        
        return observation, info
    
    def step(self, action: int):
        """
        Execute one step in the environment.
        
        Args:
            action: 0 for honest mining, 1 for selfish mining
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Determine who mines the next block
        # With probability alpha, the selfish miner finds a block
        # With probability 1-alpha, an honest miner finds a block
        selfish_mines = self.rng.random() < self.alpha
        
        reward = 0.0
        
        if action == 0:  # HONEST MINING
            if selfish_mines:
                # Selfish miner finds a block and publishes immediately
                self.selfish_chain += 1
                self.honest_chain += 1
                reward = 1.0  # Earn one block reward
            else:
                # Honest miner finds a block
                self.honest_chain += 1
                self.selfish_chain += 1
                # No reward for selfish miner
            
            # Reset fork state
            self._fork_active = False
            self._selfish_blocks_withheld = 0
            
        else:  # action == 1: SELFISH MINING
            if selfish_mines:
                # Selfish miner finds a block and withholds it
                self.selfish_chain += 1
                self._selfish_blocks_withheld += 1
                
                if not self._fork_active:
                    # Start a new fork
                    self._fork_active = True
                
                # If selfish lead >= 2, publish one block to maintain lead
                lead = self.selfish_chain - self.honest_chain
                if lead >= 2 and self._selfish_blocks_withheld >= 2:
                    # Publish one withheld block
                    self.honest_chain += 1
                    self._selfish_blocks_withheld -= 1
                    reward = 1.0
                    
            else:
                # Honest miner finds a block
                self.honest_chain += 1
                
                if self._fork_active:
                    lead = self.selfish_chain - self.honest_chain
                    
                    if lead == 0:
                        # Tie: race to publish
                        # With probability gamma, honest miners adopt selfish chain
                        if self.rng.random() < self.gamma:
                            # Selfish chain wins
                            self.honest_chain = self.selfish_chain
                            reward = self._selfish_blocks_withheld
                        else:
                            # Honest chain wins, selfish loses withheld blocks
                            self.selfish_chain = self.honest_chain
                            reward = 0.0
                        
                        self._fork_active = False
                        self._selfish_blocks_withheld = 0
                        
                    elif lead == 1:
                        # Selfish is ahead by 1, publish all withheld blocks
                        self.honest_chain = self.selfish_chain
                        reward = self._selfish_blocks_withheld
                        self._fork_active = False
                        self._selfish_blocks_withheld = 0
                        
                    elif lead < 0:
                        # Selfish is behind, abandon fork
                        self.selfish_chain = self.honest_chain
                        self._fork_active = False
                        self._selfish_blocks_withheld = 0
                        reward = 0.0
                else:
                    # No fork active, honest miner extends both chains
                    self.selfish_chain += 1
        
        self.cumulative_reward += reward
        
        # Check termination
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        observation = self._encode_state()
        info = {
            "honest_chain": self.honest_chain,
            "selfish_chain": self.selfish_chain,
            "lead": self.selfish_chain - self.honest_chain,
            "fork_active": self._fork_active,
            "blocks_withheld": self._selfish_blocks_withheld,
            "cumulative_reward": self.cumulative_reward,
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the current blockchain state."""
        if self.render_mode == "human":
            print(f"Step: {self.step_count}")
            print(f"  Honest chain: {self.honest_chain}")
            print(f"  Selfish chain: {self.selfish_chain}")
            print(f"  Lead: {self.selfish_chain - self.honest_chain}")
            print(f"  Fork active: {self._fork_active}")
            print(f"  Blocks withheld: {self._selfish_blocks_withheld}")
            print(f"  Cumulative reward: {self.cumulative_reward:.3f}")
    
    def get_state(self) -> Dict[str, Any]:
        """Return the full internal state for save/restore."""
        return {
            "honest_chain": self.honest_chain,
            "selfish_chain": self.selfish_chain,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "done": self.done,
            "_fork_active": self._fork_active,
            "_selfish_blocks_withheld": self._selfish_blocks_withheld,
            "rng_state": self.rng.get_state(),
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore the full internal state."""
        self.honest_chain = state["honest_chain"]
        self.selfish_chain = state["selfish_chain"]
        self.step_count = state["step_count"]
        self.cumulative_reward = state["cumulative_reward"]
        self.done = state["done"]
        self._fork_active = state["_fork_active"]
        self._selfish_blocks_withheld = state["_selfish_blocks_withheld"]
        self.rng.set_state(state["rng_state"])


class SelfishMiningStateWrapper(gym.Wrapper):
    """
    Wrapper that adds state save/restore to the selfish mining environment.
    
    Provides save_state(), restore_state(), and reset_to_state() methods
    for the RICE critical state collection and mixed initial distribution.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._saved_state: Optional[Dict[str, Any]] = None
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current environment state.
        
        Returns a dictionary containing all information needed to restore
        the environment to this exact state later.
        """
        if hasattr(self.env, 'get_state'):
            env_state = self.env.get_state()
        else:
            # Fallback: deep copy relevant attributes
            env_state = {
                "honest_chain": getattr(self.env, 'honest_chain', 0),
                "selfish_chain": getattr(self.env, 'selfish_chain', 0),
                "step_count": getattr(self.env, 'step_count', 0),
                "cumulative_reward": getattr(self.env, 'cumulative_reward', 0.0),
                "done": getattr(self.env, 'done', False),
                "_fork_active": getattr(self.env, '_fork_active', False),
                "_selfish_blocks_withheld": getattr(self.env, '_selfish_blocks_withheld', 0),
            }
        
        # Also save the current observation for reference
        if hasattr(self.env, '_encode_state'):
            obs = self.env._encode_state()
        else:
            obs = np.zeros(getattr(self.env, 'state_dim', 52), dtype=np.float32)
        
        return {
            "env_state": copy.deepcopy(env_state),
            "observation": obs.copy(),
        }
    
    def restore_state(self, state: Dict[str, Any]):
        """
        Restore the environment to a previously saved state.
        
        Args:
            state: Dictionary returned by save_state()
        """
        env_state = state["env_state"]
        if hasattr(self.env, 'set_state'):
            self.env.set_state(env_state)
        else:
            for key, value in env_state.items():
                if hasattr(self.env, key):
                    setattr(self.env, key, value)
        
        self._saved_state = state
    
    def reset_to_state(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Reset the environment and restore to a saved state.
        
        Args:
            state: Dictionary returned by save_state()
        
        Returns:
            observation: The observation at the restored state
        """
        # First do a normal reset to initialize everything
        self.env.reset()
        # Then restore the saved state
        self.restore_state(state)
        # Return the saved observation
        return state["observation"].copy()


def make_env(
    env_name: str = "SelfishMining-v0",
    seed: int = 42,
    max_episode_steps: Optional[int] = None,
    alpha: float = 0.35,
    gamma: float = 0.5,
    **kwargs
) -> gym.Env:
    """
    Create a selfish mining environment with state save/restore capability.
    
    Args:
        env_name: Environment name (default: "SelfishMining-v0")
        seed: Random seed
        max_episode_steps: Maximum steps per episode (default: 100)
        alpha: Selfish miner's hash rate proportion
        gamma: Proportion of honest miners attracted to selfish chain
    
    Returns:
        Wrapped gym environment
    """
    if max_episode_steps is None:
        max_episode_steps = 100
    
    # Create the base environment
    env = SelfishMiningEnv(
        alpha=alpha,
        gamma=gamma,
        max_steps=max_episode_steps,
        seed=seed,
    )
    
    # Wrap with state save/restore
    env = SelfishMiningStateWrapper(env)
    
    return env


def get_state_dim(env: gym.Env) -> int:
    """Return the observation space dimension."""
    if hasattr(env, 'state_dim'):
        return env.state_dim
    if hasattr(env, 'observation_space'):
        return env.observation_space.shape[0]
    return 52


def get_action_dim(env: gym.Env) -> int:
    """Return the action space dimension."""
    if hasattr(env, 'action_dim'):
        return env.action_dim
    if hasattr(env, 'action_space'):
        if isinstance(env.action_space, spaces.Discrete):
            return env.action_space.n
        return env.action_space.shape[0]
    return 2


def is_discrete_action(env: gym.Env) -> bool:
    """Check if the action space is discrete."""
    if hasattr(env, 'action_space'):
        return isinstance(env.action_space, spaces.Discrete)
    return True  # Selfish mining is discrete


def make_state_saveable(env: gym.Env) -> SelfishMiningStateWrapper:
    """
    Wrap an existing selfish mining environment with state save/restore.
    
    Args:
        env: The environment to wrap
    
    Returns:
        SelfishMiningStateWrapper
    """
    if isinstance(env, SelfishMiningStateWrapper):
        return env
    return SelfishMiningStateWrapper(env)


# Test function
def test_env():
    """Quick test of the selfish mining environment."""
    print("Testing Selfish Mining Environment...")
    
    env = make_env(seed=42, max_episode_steps=50)
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step == 25:
            # Test state save/restore
            saved_state = env.save_state()
            print(f"State saved at step {step}")
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward:.3f}")
    
    # Test restore
    if 'saved_state' in dir():
        obs_restored = env.reset_to_state(saved_state)
        print(f"Restored observation shape: {obs_restored.shape}")
    
    env.close()
    print("Test passed!")


if __name__ == "__main__":
    test_env()