"""Random reward-function prior used to train the Functional Reward Encoding.

The prior is a uniform mixture over three reward families:

1. Singleton goal-reaching rewards:
       eta(s) = -1 if ||s - g|| > epsilon else 0
   with g sampled uniformly from the offline state pool.

2. Random linear rewards:
       eta(s) = <w, s>
   with w ~ Uniform(-1, 1)^d and an independent Bernoulli mask applied
   to encourage simple/sparse reward functions.

3. Random two-layer MLP rewards:
       eta(s) = MLP_{2-layer}(s)
   with ReLU activations and randomly initialized weights.

All rewards are clipped to [-1, 1] during training. The returned reward
functions operate on torch tensors and are intentionally lightweight.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


class RewardFunction:
    """Callable wrapper that also stores metadata for logging/debugging."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor], kind: str):
        self.fn = fn
        self.kind = kind

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        return self.fn(states)


class SingletonGoalReward(RewardFunction):
    def __init__(self, goal: torch.Tensor, epsilon: float):
        self.goal = goal
        self.epsilon = epsilon
        super().__init__(self._call, "singleton_goal")

    def _call(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(self.goal.device, dtype=self.goal.dtype)
        dist = torch.norm(states - self.goal.unsqueeze(0), dim=-1)
        reward = torch.where(dist > self.epsilon, -torch.ones_like(dist), torch.zeros_like(dist))
        return torch.clamp(reward, -1.0, 1.0)


class LinearReward(RewardFunction):
    def __init__(self, weights: torch.Tensor, mask: torch.Tensor):
        self.weights = weights
        self.mask = mask
        super().__init__(self._call, "linear")

    def _call(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(self.weights.device, dtype=self.weights.dtype)
        effective_w = self.weights * self.mask
        reward = states @ effective_w
        return torch.clamp(reward, -1.0, 1.0)


class MLPReward(RewardFunction):
    def __init__(self, net: nn.Module):
        self.net = net
        super().__init__(self._call, "mlp")

    def _call(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(next(self.net.parameters()).device, dtype=next(self.net.parameters()).dtype)
        with torch.no_grad():
            reward = self.net(states).squeeze(-1)
        return torch.clamp(reward, -1.0, 1.0)


class RewardPrior:
    """Uniform mixture over singleton-goal, linear, and MLP reward families.

    Args:
        state_dim: Dimensionality of the state space.
        state_pool: Array-like pool of states from which goals are sampled.
            Shape ``[N, state_dim]``. If ``None``, goals are sampled uniformly
            from ``[-1, 1]^state_dim``.
        goal_epsilon: Distance threshold for singleton goal rewards.
        p_mask: Bernoulli mask probability for random linear rewards.
        mlp_hidden: Hidden width of the random MLP reward network.
        device: Torch device used for all sampled reward functions.
        seed: Optional RNG seed for reproducible reward sampling.
    """

    def __init__(
        self,
        state_dim: int,
        state_pool: Optional[Union[np.ndarray, torch.Tensor]] = None,
        goal_epsilon: float = 1.0,
        p_mask: float = 0.75,
        mlp_hidden: int = 256,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        self.state_dim = int(state_dim)
        self.goal_epsilon = float(goal_epsilon)
        self.p_mask = float(p_mask)
        self.mlp_hidden = int(mlp_hidden)
        self.device = torch.device(device)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        if state_pool is None:
            self.state_pool = None
        else:
            if isinstance(state_pool, np.ndarray):
                state_pool = torch.from_numpy(state_pool.astype(np.float32))
            elif isinstance(state_pool, torch.Tensor):
                state_pool = state_pool.float()
            else:
                state_pool = torch.tensor(np.asarray(state_pool, dtype=np.float32))
            self.state_pool = state_pool.to(self.device)

    def _sample_singleton(self) -> SingletonGoalReward:
        if self.state_pool is not None and len(self.state_pool) > 0:
            idx = np.random.randint(0, len(self.state_pool))
            goal = self.state_pool[idx].clone()
        else:
            goal = torch.empty(self.state_dim, device=self.device).uniform_(-1.0, 1.0)
        return SingletonGoalReward(goal, self.goal_epsilon)

    def _sample_linear(self) -> LinearReward:
        weights = torch.empty(self.state_dim, device=self.device).uniform_(-1.0, 1.0)
        mask = torch.bernoulli(torch.full((self.state_dim,), self.p_mask, device=self.device))
        return LinearReward(weights, mask)

    def _sample_mlp(self) -> MLPReward:
        net = nn.Sequential(
            nn.Linear(self.state_dim, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, self.mlp_hidden),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden, 1),
        ).to(self.device)
        # Random initialization by PyTorch defaults is sufficient; no training is performed.
        net.eval()
        return MLPReward(net)

    def sample_reward_fn(self) -> RewardFunction:
        """Sample one reward function from the uniform mixture prior."""
        rng = np.random.rand()
        if rng < 1.0 / 3.0:
            return self._sample_singleton()
        elif rng < 2.0 / 3.0:
            return self._sample_linear()
        return self._sample_mlp()

    def sample_reward_fns(self, batch_size: int) -> Sequence[RewardFunction]:
        """Sample a batch of independent reward functions."""
        return [self.sample_reward_fn() for _ in range(batch_size)]

    def evaluate(self, reward_fn: RewardFunction, states: torch.Tensor) -> torch.Tensor:
        """Evaluate a sampled reward function on a batch of states."""
        return reward_fn(states)


def make_default_reward_prior(
    state_dim: int,
    state_pool: Optional[Union[np.ndarray, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
) -> RewardPrior:
    """Convenience constructor using paper-informed default hyperparameters."""
    return RewardPrior(
        state_dim=state_dim,
        state_pool=state_pool,
        goal_epsilon=1.0,
        p_mask=0.75,
        mlp_hidden=256,
        device=device,
        seed=seed,
    )
