"""
Selfish Mining Environment for RICE Framework

Implements the blockchain selfish mining MDP from Bar-Zur et al. (2023):
"WeRLman: To Tackle Whale (Transactions), Go Deep (RL)"

The environment models a selfish mining attack where a miner strategically
withholds blocks to gain a disproportionate share of mining rewards.

State: chain state (honest chain length, private chain length, fork state)
Actions: Adopt (0), Reveal (1), Mine (2)
Reward: positive if block accepted by network, negative if unsuccessful

Reference: https://github.com/roibarzur/pto-selfish-mining
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import random


class SelfishMiningEnv(gym.Env):
    """
    Selfish Mining Environment.
    
    The environment simulates a blockchain where a selfish miner (the agent)
    competes against honest miners. The agent can choose to:
    - Adopt: Accept the honest chain and discard private work
    - Reveal: Publish private blocks to the network
    - Mine: Continue mining on the private chain
    
    State space: [honest_chain_len, private_chain_len, fork_state, gamma_ratio]
    Action space: Discrete(3) - {0: Adopt, 1: Reveal, 2: Mine}
    
    Based on the model from Bar-Zur et al. (2023) and Eyal & Sirer (2014).
    """
    
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    def __init__(
        self,
        alpha: float = 0.35,          # Selfish miner's hash power fraction
        gamma: float = 0.5,           # Fraction of honest miners that mine on selfish block
        max_chain_length: int = 100,  # Maximum chain length before episode ends
        max_steps: int = 500,         # Maximum steps per episode
        reward_scale: float = 1.0,    # Scaling factor for rewards
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the selfish mining environment.
        
        Args:
            alpha: Fraction of total hash power controlled by selfish miner (0 < alpha < 0.5)
            gamma: Fraction of honest miners that adopt the selfish chain when tied
            max_chain_length: Maximum length of either chain before episode truncation
            max_steps: Maximum number of steps per episode
            reward_scale: Multiplier for rewards
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()
        
        # Validate parameters
        assert 0 < alpha < 0.5, f"alpha must be in (0, 0.5), got {alpha}"
        assert 0 <= gamma <= 1, f"gamma must be in [0, 1], got {gamma}"
        
        self.alpha = alpha
        self.gamma = gamma
        self.max_chain_length = max_chain_length
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.render_mode = render_mode
        
        # Action space: 0=Adopt, 1=Reveal, 2=Mine
        self.action_space = spaces.Discrete(3)
        
        # Observation space:
        # [honest_chain_len, private_chain_len, fork_state, gamma_ratio]
        # fork_state: 0=no fork (private <= honest), 1=lead by 1, 2=lead by 2, 3=lead by >=3
        # gamma_ratio: gamma parameter (constant, included for completeness)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([max_chain_length, max_chain_length, 3, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Internal state
        self.honest_chain = 0   # Length of honest (public) chain
        self.private_chain = 0  # Length of private chain (mined by selfish miner)
        self.steps = 0
        self.total_reward = 0.0
        self.blocks_accepted = 0
        self.blocks_orphaned = 0
        
        # For rendering
        self.render_buffer = []
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation from internal state."""
        # Determine fork state
        lead = self.private_chain - self.honest_chain
        if lead <= 0:
            fork_state = 0  # No lead or behind
        elif lead == 1:
            fork_state = 1  # Lead by 1
        elif lead == 2:
            fork_state = 2  # Lead by 2
        else:
            fork_state = 3  # Lead by 3+
        
        return np.array(
            [self.honest_chain, self.private_chain, fork_state, self.gamma],
            dtype=np.float32
        )
    
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary information."""
        return {
            "honest_chain": self.honest_chain,
            "private_chain": self.private_chain,
            "lead": self.private_chain - self.honest_chain,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "blocks_accepted": self.blocks_accepted,
            "blocks_orphaned": self.blocks_orphaned,
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # Initialize chains at genesis
        self.honest_chain = 0
        self.private_chain = 0
        self.steps = 0
        self.total_reward = 0.0
        self.blocks_accepted = 0
        self.blocks_orphaned = 0
        self.render_buffer = []
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=Adopt, 1=Reveal, 2=Mine
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        reward = 0.0
        self.steps += 1
        
        # --- Process action ---
        if action == 0:  # Adopt
            # Accept the honest chain, discard private work
            self.private_chain = self.honest_chain
            # No reward for adopting
            
        elif action == 1:  # Reveal
            # Publish private blocks
            if self.private_chain > self.honest_chain:
                # We have a lead; reveal private blocks
                lead = self.private_chain - self.honest_chain
                
                if lead == 1:
                    # Lead by 1: race condition
                    # With probability gamma, honest miners adopt our block
                    if random.random() < self.gamma:
                        # We win the race: both chains get our block
                        reward += 2 * self.reward_scale  # 2 blocks accepted
                        self.blocks_accepted += 2
                        self.honest_chain = self.private_chain
                    else:
                        # Honest miners win: each gets 1 block
                        reward += 1 * self.reward_scale
                        self.blocks_accepted += 1
                        self.honest_chain = self.private_chain  # Tie resolved
                elif lead == 2:
                    # Lead by 2: we get all blocks
                    reward += (lead + 1) * self.reward_scale
                    self.blocks_accepted += (lead + 1)
                    self.honest_chain = self.private_chain
                else:
                    # Lead by 3+: we get all blocks
                    reward += (lead + 1) * self.reward_scale
                    self.blocks_accepted += (lead + 1)
                    self.honest_chain = self.private_chain
                
                self.private_chain = self.honest_chain  # Reset after reveal
            else:
                # No lead; revealing does nothing (or minor penalty)
                reward -= 0.1 * self.reward_scale
                
        elif action == 2:  # Mine
            # Attempt to mine a new block
            if random.random() < self.alpha:
                # Selfish miner finds a block
                self.private_chain += 1
            else:
                # Honest miners find a block
                self.honest_chain += 1
                
                # Check if honest chain overtakes
                if self.honest_chain > self.private_chain:
                    # We lost our lead; private blocks orphaned
                    orphaned = self.private_chain - max(0, self.honest_chain - 1)
                    if orphaned > 0:
                        self.blocks_orphaned += orphaned
                        reward -= orphaned * self.reward_scale
                    self.private_chain = self.honest_chain
        
        # --- Check termination ---
        terminated = False
        truncated = False
        
        if self.honest_chain >= self.max_chain_length or self.private_chain >= self.max_chain_length:
            truncated = True
        
        if self.steps >= self.max_steps:
            truncated = True
        
        self.total_reward += reward
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            output = f"Step: {self.steps}\n"
            output += f"Honest chain: {'█' * self.honest_chain} ({self.honest_chain})\n"
            output += f"Private chain: {'▓' * self.private_chain} ({self.private_chain})\n"
            output += f"Lead: {self.private_chain - self.honest_chain}\n"
            output += f"Total reward: {self.total_reward:.2f}\n"
            output += f"Blocks accepted: {self.blocks_accepted}, Orphaned: {self.blocks_orphaned}\n"
            return output
        elif self.render_mode == "human":
            print(self.render())
            return None
        return None
    
    def close(self):
        """Clean up resources."""
        pass
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Set the environment to a specific state.
        Used by RICE for resetting to critical states.
        
        Args:
            state: Array [honest_chain, private_chain, fork_state, gamma]
        """
        self.honest_chain = int(state[0])
        self.private_chain = int(state[1])
        # fork_state and gamma are derived/computed, not directly set
        self.steps = 0  # Reset step counter for new episode
        self.total_reward = 0.0
        self.blocks_accepted = 0
        self.blocks_orphaned = 0
    
    def get_state(self) -> np.ndarray:
        """Get the current environment state."""
        return self._get_obs()


class SelfishMiningEnvWrapper(gym.Wrapper):
    """
    Wrapper for the selfish mining environment that provides additional
    functionality for RICE integration, including state saving/restoration
    and compatibility with Stable-Baselines3.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._saved_state = None
    
    def step(self, action):
        """Step with additional logging."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset with state saving."""
        obs, info = self.env.reset(**kwargs)
        self._saved_state = self.env.get_state() if hasattr(self.env, 'get_state') else obs
        return obs, info
    
    def save_state(self) -> np.ndarray:
        """Save current environment state."""
        if hasattr(self.env, 'get_state'):
            self._saved_state = self.env.get_state()
        return self._saved_state
    
    def restore_state(self, state: np.ndarray) -> np.ndarray:
        """Restore environment to a saved state."""
        if hasattr(self.env, 'set_state'):
            self.env.set_state(state)
        return self.env._get_obs() if hasattr(self.env, '_get_obs') else state


def make_selfish_mining_env(
    alpha: float = 0.35,
    gamma: float = 0.5,
    max_chain_length: int = 100,
    max_steps: int = 500,
    reward_scale: float = 1.0,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> gym.Env:
    """
    Create a selfish mining environment.
    
    Args:
        alpha: Selfish miner's hash power fraction
        gamma: Fraction of honest miners adopting selfish chain
        max_chain_length: Maximum chain length
        max_steps: Maximum steps per episode
        reward_scale: Reward multiplier
        render_mode: Rendering mode
        seed: Random seed
        
    Returns:
        Gym environment
    """
    env = SelfishMiningEnv(
        alpha=alpha,
        gamma=gamma,
        max_chain_length=max_chain_length,
        max_steps=max_steps,
        reward_scale=reward_scale,
        render_mode=render_mode,
    )
    
    if seed is not None:
        env.reset(seed=seed)
    
    return env


# Register the environment with Gymnasium
try:
    gym.register(
        id="SelfishMining-v0",
        entry_point="envs.selfish_mining_env:SelfishMiningEnv",
        max_episode_steps=500,
    )
except gym.error.Error:
    # Already registered
    pass


if __name__ == "__main__":
    # Quick test
    env = make_selfish_mining_env(alpha=0.35, gamma=0.5, seed=42)
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    print(f"Info: {info}")
    
    total_reward = 0.0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    
    print(f"Total reward after {i+1} steps: {total_reward:.2f}")
    print(f"Blocks accepted: {info['blocks_accepted']}, Orphaned: {info['blocks_orphaned']}")
    env.close()