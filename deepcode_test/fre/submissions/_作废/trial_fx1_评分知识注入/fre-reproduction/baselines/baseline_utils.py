"""Shared utilities for baseline implementations.

This module contains the small building blocks used by the Forward-Backward,
Successor-Features, goal-conditioned IQL/BC, and OPAL baselines.  It avoids
re-implementing tensor conversion, soft-updates, logging, and reward-predictor
regression inside every baseline file.

All functions are deliberately small and self-contained so that the baselines
remain importable even if a particular MuJoCo/D4RL environment is missing.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.utils import (
    Timer,
    freeze_module,
    get_logger,
    resolve_device,
    set_seed,
    to_numpy,
    to_torch,
)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    activation: Union[str, nn.Module] = "relu",
    output_activation: Optional[Union[str, nn.Module]] = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build a small MLP.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input vector.
    hidden_dims:
        Widths of the hidden layers.
    output_dim:
        Dimensionality of the output.
    activation:
        Activation used between hidden layers.
    output_activation:
        Optional activation applied after the final linear layer.
    dropout:
        Dropout probability inserted after each hidden activation.
    """

    act: nn.Module
    if isinstance(activation, str):
        act = _get_activation(activation)
    else:
        act = activation

    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(act)
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden
    layers.append(nn.Linear(prev_dim, output_dim))

    if output_activation is not None:
        if isinstance(output_activation, str):
            layers.append(_get_activation(output_activation))
        else:
            layers.append(output_activation)

    return nn.Sequential(*layers)


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "identity" or name == "none":
        return nn.Identity()
    raise ValueError(f"Unknown activation '{name}'")


class DeterministicPolicy(nn.Module):
    """Deterministic policy used by DDPG-style baselines (FB and SF).

    The policy conditions on both the environment state and a task/feature
    representation vector ``context`` and outputs a squashed action in
    ``[-max_action, max_action]``.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        max_action: float = 1.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.net = build_mlp(
            state_dim + context_dim,
            hidden_dims,
            action_dim,
            activation=activation,
            output_activation=nn.Tanh(),
        )

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(state.shape[0], -1)
        x = torch.cat([state, context], dim=-1)
        return self.max_action * self.net(x)


class TwinQNetwork(nn.Module):
    """Twin Q critics for DDPG-style baselines.

    Inputs are ``[state, action, context]`` and outputs are two scalar
    Q-value estimates.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_dim = context_dim
        input_dim = state_dim + action_dim + context_dim
        self.q1 = build_mlp(input_dim, hidden_dims, 1, activation=activation)
        self.q2 = build_mlp(input_dim, hidden_dims, 1, activation=activation)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(state.shape[0], -1)
        x = torch.cat([state, action, context], dim=-1)
        return self.q1(x), self.q2(x)

    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        q1, q2 = self.forward(state, action, context)
        return torch.min(q1, q2)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian policy with reparameterized sampling.

    Used by OPAL and can also be used for goal-conditioned baselines that
    need stochastic actions.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        max_action: float = 1.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.max_action = max_action

        self.net = build_mlp(
            state_dim + context_dim,
            hidden_dims,
            2 * action_dim,
            activation=activation,
        )

    def _mean_std(self, state: torch.Tensor, context: torch.Tensor):
        if context.dim() == 1:
            context = context.unsqueeze(0).expand(state.shape[0], -1)
        x = torch.cat([state, context], dim=-1)
        out = self.net(x)
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def forward(self, state: torch.Tensor, context: torch.Tensor):
        mean, log_std = self._mean_std(state, context)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh) * self.max_action
        log_prob = dist.log_prob(pre_tanh) - torch.log(
            1.0 - action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, mean, log_std, log_prob

    def get_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        context: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
        device: Union[str, torch.device] = "cpu",
    ) -> np.ndarray:
        state_t = to_torch(state, device=device)
        ctx_t = to_torch(context, device=device)
        if state_t.dim() == 1:
            state_t = state_t.unsqueeze(0)
        if ctx_t.dim() == 1:
            ctx_t = ctx_t.unsqueeze(0)
        with torch.no_grad():
            mean, log_std = self._mean_std(state_t, ctx_t)
            if deterministic:
                action = torch.tanh(mean) * self.max_action
            else:
                std = log_std.exp()
                action = torch.tanh(mean + std * torch.randn_like(mean)) * self.max_action
        return to_numpy(action.squeeze(0))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.005) -> None:
    """Polyak-averaging update for target networks."""
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy source network parameters into target network."""
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(source_param.data)


def expectile_loss(diff: torch.Tensor, expectile: float = 0.9) -> torch.Tensor:
    """Implicit expectile regression loss.

    For ``expectile > 0.5``, positive residuals (underestimation) are weighted
    more heavily than negative residuals.
    """
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * (diff.pow(2))


def td_target(
    rewards: torch.Tensor,
    next_v: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute ``reward + gamma * (1 - done) * next_v``."""
    if dones.dim() == 0:
        dones = dones.unsqueeze(0)
    if dones.dim() == 1:
        dones = dones.unsqueeze(-1)
    return rewards + gamma * (1.0 - dones) * next_v


# ---------------------------------------------------------------------------
# Reward-predictor / evaluation regression utilities
# ---------------------------------------------------------------------------

def sample_reward_pairs(
    reward_fn: Callable[[np.ndarray], np.ndarray],
    state_pool: np.ndarray,
    num_samples: int = 5120,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample state-reward pairs from a pool using a callable reward.

    This is the evaluation-time reward regression data generator used by
    FB and SF baselines (they use 5120 samples; FRE uses only 32).
    """
    rng = np.random.RandomState(seed)
    n_pool = len(state_pool)
    if num_samples >= n_pool:
        idx = np.arange(n_pool)
        rng.shuffle(idx)
    else:
        idx = rng.randint(0, n_pool, size=num_samples)
    states = state_pool[idx]
    rewards = np.asarray(reward_fn(states), dtype=np.float32)
    if rewards.ndim > 1:
        rewards = rewards.reshape(-1)
    return states, rewards


def ridge_regression(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    ridge: float = 1e-3,
) -> np.ndarray:
    """Solve ``min_w ||X w - y||^2 + ridge * ||w||^2``.

    Returns
    -------
    np.ndarray
        Weight vector of shape ``(X.shape[1],)``.
    """
    X_np = to_numpy(X).astype(np.float64)
    y_np = to_numpy(y).astype(np.float64).reshape(-1)
    n, d = X_np.shape
    A = X_np.T @ X_np + ridge * np.eye(d)
    b = X_np.T @ y_np
    w = np.linalg.solve(A, b)
    return w.astype(np.float32)


def least_squares(
    X: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """Plain least-squares regression without ridge penalty."""
    X_np = to_numpy(X).astype(np.float64)
    y_np = to_numpy(y).astype(np.float64).reshape(-1)
    w, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
    return w.astype(np.float32)


def infer_linear_reward_vector(
    feature_fn: Callable[[np.ndarray], np.ndarray],
    reward_fn: Callable[[np.ndarray], np.ndarray],
    state_pool: np.ndarray,
    num_samples: int = 5120,
    ridge: float = 1e-3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Convenience wrapper: sample pairs and solve ridge regression.

    ``feature_fn`` maps states to a representation vector.  The returned
    vector can be used to predict rewards as ``features @ w``.
    """
    states, rewards = sample_reward_pairs(
        reward_fn, state_pool, num_samples=num_samples, seed=seed
    )
    features = np.asarray(feature_fn(states), dtype=np.float32)
    if features.ndim == 1:
        features = features[:, None]
    return ridge_regression(features, rewards, ridge=ridge)


# ---------------------------------------------------------------------------
# Generic policy-evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_deterministic_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env: object,
    num_episodes: int = 20,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a policy in a gym-like environment.

    The environment only needs ``reset`` and ``step`` returning
    ``(next_obs, reward, done, info)``.  ``policy_fn`` receives an observation
    and returns an action.
    """
    if seed is not None:
        if hasattr(env, "seed"):
            env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    returns: List[float] = []
    lengths: List[int] = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        while not done:
            action = np.asarray(policy_fn(obs), dtype=np.float32)
            obs, reward, done, info = env.step(action)
            episode_return += float(reward)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        returns.append(episode_return)
        lengths.append(steps)

    returns = np.asarray(returns, dtype=np.float32)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "num_episodes": int(num_episodes),
    }


def make_policy_fn_from_net(
    net: nn.Module,
    context: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device] = "cpu",
    deterministic: bool = True,
    state_is_tensor: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a state -> action closure from a PyTorch policy.

    The context (task embedding) is fixed for the evaluation episode.
    """
    net.eval()
    ctx_t = to_torch(context, device=device)
    if ctx_t.dim() == 1:
        ctx_t = ctx_t.unsqueeze(0)

    def _policy(obs: np.ndarray) -> np.ndarray:
        state = to_torch(obs, device=device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            if hasattr(net, "get_action"):
                action = net.get_action(state, ctx_t, deterministic=deterministic)
            else:
                action = net(state, ctx_t)
        return to_numpy(action.squeeze(0))

    return _policy


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def freeze(module: nn.Module) -> nn.Module:
    return freeze_module(module)


def average_dicts(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    keys = None
    totals: Dict[str, float] = {}
    count = 0
    for d in dicts:
        if d is None:
            continue
        if keys is None:
            keys = list(d.keys())
            totals = {k: 0.0 for k in keys}
        for k in keys:
            totals[k] += float(d.get(k, 0.0))
        count += 1
    if count == 0:
        return {}
    return {k: v / count for k, v in totals.items()}


def std_dicts(dicts: Iterable[Dict[str, float]]) -> Dict[str, float]:
    keys = None
    all_values: Dict[str, List[float]] = {}
    for d in dicts:
        if d is None:
            continue
        if keys is None:
            keys = list(d.keys())
            all_values = {k: [] for k in keys}
        for k in keys:
            all_values[k].append(float(d.get(k, 0.0)))
    if not all_values:
        return {}
    return {
        k: float(np.std(vals)) if len(vals) > 1 else 0.0
        for k, vals in all_values.items()
    }


__all__ = [
    "DeterministicPolicy",
    "GaussianPolicy",
    "TwinQNetwork",
    "average_dicts",
    "build_mlp",
    "evaluate_deterministic_policy",
    "expectile_loss",
    "freeze",
    "hard_update",
    "infer_linear_reward_vector",
    "least_squares",
    "make_policy_fn_from_net",
    "ridge_regression",
    "sample_reward_pairs",
    "soft_update",
    "std_dicts",
    "td_target",
]
