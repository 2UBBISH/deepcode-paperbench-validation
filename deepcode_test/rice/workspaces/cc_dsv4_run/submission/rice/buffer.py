"""
Replay buffer for storing and sampling RL trajectories.

Provides:
- RolloutBuffer: Standard buffer for on-policy PPO rollouts
- TrajectoryBuffer: Buffer for storing complete trajectories (for SIL priority)
- StateBuffer: Simple state storage for RND normalization
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing on-policy rollout data for PPO training.
    Stores one complete rollout at a time (on-policy: data is discarded after update).
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []  # for RND training

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        next_state: Optional[np.ndarray] = None,
    ) -> None:
        """Add a single transition to the buffer."""
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.values.pop(0)
            self.log_probs.pop(0)
            self.dones.pop(0)
            if self.next_states:
                self.next_states.pop(0)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(float(done))
        if next_state is not None:
            self.next_states.append(next_state)

    def get_all(self) -> Dict[str, np.ndarray]:
        """Return all stored data as numpy arrays."""
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "values": np.array(self.values, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "dones": np.array(self.dones, dtype=np.float32),
        }

    def get_states(self) -> np.ndarray:
        """Return all stored states."""
        return np.array(self.states, dtype=np.float32)

    def get_next_states(self) -> np.ndarray:
        """Return all stored next states (for RND training)."""
        if self.next_states:
            return np.array(self.next_states, dtype=np.float32)
        return np.array(self.states[1:] + [self.states[-1]], dtype=np.float32)

    def clear(self) -> None:
        """Clear buffer after PPO update."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.next_states.clear()

    def __len__(self) -> int:
        return len(self.states)

    def is_full(self) -> bool:
        return len(self.states) >= self.capacity


class TrajectoryBuffer:
    """
    Buffer for storing complete episodes (trajectories).
    Each trajectory is a list of (state, action, reward, done) tuples.
    Used for fidelity evaluation and replay-based methods.
    """

    def __init__(self, max_trajectories: int = 1000):
        self.max_trajectories = max_trajectories
        self.trajectories: List[Dict[str, np.ndarray]] = []

    def add_trajectory(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        rewards: List[float],
        dones: List[bool] = None,
    ) -> None:
        """Add a complete trajectory to the buffer."""
        if len(self.trajectories) >= self.max_trajectories:
            self.trajectories.pop(0)

        self.trajectories.append({
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones or [False] * len(states), dtype=np.float32),
        })

    def sample_trajectory(self) -> Optional[Dict[str, np.ndarray]]:
        """Sample a random trajectory from the buffer."""
        if not self.trajectories:
            return None
        idx = np.random.randint(0, len(self.trajectories))
        return self.trajectories[idx]

    def get_all_trajectories(self) -> List[Dict[str, np.ndarray]]:
        """Return all stored trajectories."""
        return self.trajectories

    def clear(self) -> None:
        self.trajectories.clear()

    def __len__(self) -> int:
        return len(self.trajectories)


class PriorityBuffer:
    """
    Priority-based replay buffer.
    Stores transitions with priority weights.
    Used for Self-Imitation Learning (SIL) to prioritize high-return experiences.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.returns = []

    def add(self, state: np.ndarray, action: np.ndarray, ret: float) -> None:
        """Add a transition with its cumulative return."""
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.returns.append(ret)

        # If over capacity, remove lowest-return transition
        if len(self.states) > self.capacity:
            min_idx = int(np.argmin(self.returns))
            del self.states[min_idx]
            del self.actions[min_idx]
            del self.returns[min_idx]

    def sample(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Sample a batch with probability proportional to returns.
        Only samples transitions with positive returns (good experiences).
        """
        if len(self.states) < batch_size:
            return None

        returns_arr = np.array(self.returns)
        # Only sample from positive returns
        pos_mask = returns_arr > 0
        if pos_mask.sum() == 0:
            # Fallback: uniform sampling
            indices = np.random.choice(len(self.states), batch_size, replace=False)
        else:
            pos_indices = np.where(pos_mask)[0]
            probs = returns_arr[pos_indices]
            probs = probs / probs.sum()
            indices = np.random.choice(pos_indices, batch_size, replace=True, p=probs)

        return (
            np.array([self.states[i] for i in indices], dtype=np.float32),
            np.array([self.actions[i] for i in indices], dtype=np.float32),
            np.array([self.returns[i] for i in indices], dtype=np.float32),
        )

    def sample_uniform(self, batch_size: int) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Uniform random sampling (fallback)."""
        if len(self.states) < batch_size:
            return None
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        return (
            np.array([self.states[i] for i in indices], dtype=np.float32),
            np.array([self.actions[i] for i in indices], dtype=np.float32),
            np.array([self.returns[i] for i in indices], dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.states)