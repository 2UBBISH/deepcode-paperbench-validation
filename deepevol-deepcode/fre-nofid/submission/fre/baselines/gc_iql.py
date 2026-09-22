"""Goal-Conditioned IQL (GC-IQL) baseline for FRE.

This implements the Goal-Conditioned Implicit Q-Learning baseline used in the
FRE paper (Section 5, Table 1).  It is intentionally *not* z-conditioned: the
policy, value and action-value networks receive the ground-truth goal ``g``
concatenated to the observation, instead of a reward-encoded latent ``z``.

Design (mirrors :mod:`fre.baselines.gc_bc` for a consistent baseline API):

* Networks: ``Q(s, a, g)``, ``V(s, g)``, ``pi(a | s, g)`` with ``[512, 512, 512]``
  hidden layers; ``g`` is concatenated to the observation (and to ``[s; a]`` for
  ``Q``).
* Goal sampling for relabelling follows the paper's mixture
  ``0.3 random / 0.5 geometric-future / 0.2 current``.
* Reward: ``r(s, g) = 0`` if ``s == g`` else ``-1`` (sparse goal-reaching).
* IQL hyper-parameters: expectile ``0.8``, AWR temperature ``3.0``, discount
  ``0.88``, target update rate ``0.001``, Adam lr ``1e-4``, batch size ``512``.
* Evaluation conditions the policy on the *ground-truth* goal supplied by the
  environment/task (no goal sampling at test time).

The module exposes the same convenience surface as the other baselines:
``build_gc_iql``, ``train_gc_iql``, ``GCIAgent``/``GC-IQL`` aliases and a small
CLI (``python -m fre.baselines.gc_iql --domain antmaze``).
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Torch (soft import so that pure-numpy tooling can still import this module)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "GC-IQL requires PyTorch. Install torch>=1.13 to use "
            "fre.baselines.gc_iql."
        )


# ---------------------------------------------------------------------------
# Hyper-parameters (paper Section 4.3 / Appendix A)
# ---------------------------------------------------------------------------
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (512, 512, 512)
DEFAULT_LR: float = 1e-4
DEFAULT_BATCH_SIZE: int = 512
DEFAULT_DISCOUNT: float = 0.88
DEFAULT_EXPECTILE: float = 0.8
DEFAULT_AWR_TEMPERATURE: float = 3.0
DEFAULT_AWR_MAX_WEIGHT: float = 100.0
DEFAULT_TARGET_UPDATE_RATE: float = 0.001
DEFAULT_MAX_GRAD_NORM: float = 10.0
DEFAULT_LOG_STD_MIN: float = -10.0
DEFAULT_LOG_STD_MAX: float = 2.0
DEFAULT_NUM_CANDIDATE_ACTIONS: int = 10

# Goal relabelling mixture: 0.3 random / 0.5 geometric-future / 0.2 current.
DEFAULT_P_CURRENT: float = 0.2
DEFAULT_P_FUTURE: float = 0.5
DEFAULT_P_RANDOM: float = 0.3
DEFAULT_GEOM_P: float = 0.5

# Success tolerance for the sparse ``s == goal`` check (goal observations are
# continuous, so an exact equality test is replaced by a tolerance).
DEFAULT_GOAL_TOLERANCE: float = 1e-3
DEFAULT_SUCCESS_REWARD: float = 0.0
DEFAULT_FAILURE_REWARD: float = -1.0

_ACTIVATIONS = {
    "relu": nn.ReLU if _TORCH_AVAILABLE else None,
    "gelu": nn.GELU if _TORCH_AVAILABLE else None,
    "tanh": nn.Tanh if _TORCH_AVAILABLE else None,
    "silu": nn.SiLU if _TORCH_AVAILABLE else None,
    "elu": nn.ELU if _TORCH_AVAILABLE else None,
}


# ---------------------------------------------------------------------------
# Goal sampling helpers (shared semantics with the FRE reward prior)
# ---------------------------------------------------------------------------
def episode_boundaries(terminals: Optional[np.ndarray], num_states: int) -> List[Tuple[int, int]]:
    """Convert terminal flags into inclusive ``(start, end)`` episode bounds."""
    if terminals is None or len(terminals) == 0:
        return [(0, max(num_states - 1, 0))]

    terminals = np.asarray(terminals).reshape(-1)
    ends: List[int] = []
    for i, t in enumerate(terminals):
        if bool(t):
            ends.append(i)
    # Ensure the final transition is an episode end even if the data was
    # truncated rather than terminated.
    if not ends or ends[-1] != num_states - 1:
        ends.append(num_states - 1)

    bounds: List[Tuple[int, int]] = []
    start = 0
    for end in ends:
        end = min(end, num_states - 1)
        if end >= start:
            bounds.append((start, end))
        start = end + 1
    if not bounds:
        bounds = [(0, max(num_states - 1, 0))]
    return bounds


def _episode_lookup(bounds: Sequence[Tuple[int, int]], indices: np.ndarray) -> np.ndarray:
    """Map flat transition indices to episode indices."""
    starts = np.array([b[0] for b in bounds], dtype=np.int64)
    return np.searchsorted(starts, indices, side="right") - 1


def sample_goal_indices(
    indices: np.ndarray,
    terminals: Optional[np.ndarray] = None,
    episode_ends: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    num_states: Optional[int] = None,
    p_current: float = DEFAULT_P_CURRENT,
    p_future: float = DEFAULT_P_FUTURE,
    p_random: float = DEFAULT_P_RANDOM,
    geom_p: float = DEFAULT_GEOM_P,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample goal indices with the 0.2 current / 0.5 future / 0.3 random mixture.

    Returns ``(goal_indices, strategy_ids)`` where ``0 = current``,
    ``1 = future`` (geometric within the same episode) and ``2 = random``.
    """
    rng = np.random.default_rng() if rng is None else rng
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    if num_states is None:
        if episode_ends is not None and len(episode_ends):
            num_states = int(np.max(episode_ends)) + 1
        elif terminals is not None and len(terminals):
            num_states = int(len(terminals))
        else:
            num_states = int(indices.max()) + 1 if indices.size else 1

    bounds = episode_boundaries(terminals, num_states)
    ep_of = _episode_lookup(bounds, indices)

    probs = np.array([p_current, p_future, p_random], dtype=np.float64)
    probs = probs / probs.sum()
    strategies = rng.choice(3, size=indices.shape[0], p=probs)

    goals = indices.copy()

    # --- future (geometric offset within the same episode) -----------------
    future_mask = strategies == 1
    if np.any(future_mask):
        n_future = int(future_mask.sum())
        geom = rng.geometric(geom_p, size=n_future).astype(np.int64)
        cand = indices[future_mask] + geom
        ends = np.array([bounds[e][1] for e in ep_of[future_mask]], dtype=np.int64)
        goals[future_mask] = np.minimum(cand, ends)

    # --- random (uniform over the dataset) ---------------------------------
    random_mask = strategies == 2
    if np.any(random_mask):
        goals[random_mask] = rng.integers(0, num_states, size=int(random_mask.sum()))

    # --- current -----------------------------------------------------------
    # goals already equal indices for strategy 0
    return goals, strategies


def relabel_goals(
    observations: np.ndarray,
    indices: np.ndarray,
    terminals: Optional[np.ndarray] = None,
    episode_ends: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    p_current: float = DEFAULT_P_CURRENT,
    p_future: float = DEFAULT_P_FUTURE,
    p_random: float = DEFAULT_P_RANDOM,
    geom_p: float = DEFAULT_GEOM_P,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(goal_observations, strategy_ids)`` for the given indices."""
    observations = np.asarray(observations)
    goals_idx, strategies = sample_goal_indices(
        indices,
        terminals=terminals,
        episode_ends=episode_ends,
        rng=rng,
        num_states=observations.shape[0],
        p_current=p_current,
        p_future=p_future,
        p_random=p_random,
        geom_p=geom_p,
    )
    return observations[goals_idx], strategies


def sparse_goal_reward(
    next_observations: np.ndarray,
    goals: np.ndarray,
    tolerance: float = DEFAULT_GOAL_TOLERANCE,
    success_reward: float = DEFAULT_SUCCESS_REWARD,
    failure_reward: float = DEFAULT_FAILURE_REWARD,
) -> np.ndarray:
    """``r = 0`` when ``s' == g`` (within tolerance) else ``-1``."""
    nxt = np.asarray(next_observations, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if nxt.ndim == 1:
        nxt = nxt[None, :]
    if goals.ndim > nxt.ndim:
        goals = goals.reshape(goals.shape[0], -1)
    if nxt.shape != goals.shape:
        # Broadcast / truncate to the shared trailing dimension.
        dim = min(nxt.shape[-1], goals.shape[-1])
        nxt = nxt.reshape(nxt.shape[0], -1)[:, :dim]
        goals = goals.reshape(goals.shape[0], -1)[:, :dim]
    close = np.linalg.norm(nxt - goals, axis=-1) <= float(tolerance)
    return np.where(close, float(success_reward), float(failure_reward)).astype(np.float32)


def sample_goal_batch(
    batch: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
    observations: Optional[np.ndarray] = None,
    terminals: Optional[np.ndarray] = None,
    p_current: float = DEFAULT_P_CURRENT,
    p_future: float = DEFAULT_P_FUTURE,
    p_random: float = DEFAULT_P_RANDOM,
    geom_p: float = DEFAULT_GEOM_P,
    goal_tolerance: float = DEFAULT_GOAL_TOLERANCE,
) -> Dict[str, np.ndarray]:
    """Attach goal observations and sparse goal rewards to a transition batch.

    Expects the batch to contain ``indices`` or to be sampleable directly from a
    full dataset (passed as ``observations``/``terminals``).
    """
    nxt = np.asarray(batch["next_observations"])
    n = nxt.shape[0]
    idx = batch.get("indices")
    if idx is None:
        idx = np.arange(n, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)

    if observations is None:
        observations = batch.get("all_observations")
    if observations is None:
        # Fall back to using the batch itself as the goal pool (weaker, but keeps
        # the interface total when a full dataset is unavailable).
        observations = nxt

    goals, strategies = relabel_goals(
        np.asarray(observations),
        idx,
        terminals=terminals,
        rng=rng,
        p_current=p_current,
        p_future=p_future,
        p_random=p_random,
        geom_p=geom_p,
    )
    rewards = sparse_goal_reward(nxt, goals, tolerance=goal_tolerance)

    out = dict(batch)
    out["goals"] = goals
    out["goal_observations"] = goals
    out["rewards"] = rewards
    out["goal_strategies"] = strategies
    return out


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def _build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    out_dim: Optional[int] = None,
    activation: str = "relu",
) -> "nn.Sequential":
    act_cls = _ACTIVATIONS.get(activation, nn.ReLU)
    layers: List[nn.Module] = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, int(h)))
        layers.append(act_cls())
        last = int(h)
    if out_dim is not None:
        layers.append(nn.Linear(last, int(out_dim)))
    return nn.Sequential(*layers)


class GoalQNetwork(nn.Module):
    """Action-value network ``Q(s, a, g)`` with the goal concatenated to ``[s; a]``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        goal_dim = obs_dim if goal_dim is None else goal_dim
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.net = _build_mlp(
            self.obs_dim + self.action_dim + self.goal_dim,
            hidden_sizes,
            out_dim=1,
            activation=activation,
        )

    def forward(self, obs, action, goal):
        x = torch.cat(
            [
                obs.reshape(*obs.shape[:-1], self.obs_dim),
                action.reshape(*action.shape[:-1], self.action_dim),
                goal.reshape(*goal.shape[:-1], self.goal_dim),
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


class GoalVNetwork(nn.Module):
    """State-value network ``V(s, g)``."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        goal_dim = obs_dim if goal_dim is None else goal_dim
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.net = _build_mlp(
            self.obs_dim + self.goal_dim, hidden_sizes, out_dim=1, activation=activation
        )

    def forward(self, obs, goal):
        x = torch.cat(
            [
                obs.reshape(*obs.shape[:-1], self.obs_dim),
                goal.reshape(*goal.shape[:-1], self.goal_dim),
            ],
            dim=-1,
        )
        return self.net(x).squeeze(-1)


class GoalGaussianPolicy(nn.Module):
    """Diagonal Gaussian policy ``pi(a | s, g)`` with tanh-squashed actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
        squashed: bool = True,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
    ) -> None:
        super().__init__()
        goal_dim = obs_dim if goal_dim is None else goal_dim
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim)
        self.squashed = bool(squashed)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.body = _build_mlp(
            self.obs_dim + self.goal_dim, hidden_sizes, out_dim=None, activation=activation
        )
        last = int(hidden_sizes[-1]) if len(hidden_sizes) else self.obs_dim + self.goal_dim
        self.mean_head = nn.Linear(last, self.action_dim)
        self.log_std_head = nn.Linear(last, self.action_dim)

    def _features(self, obs, goal):
        x = torch.cat(
            [
                obs.reshape(*obs.shape[:-1], self.obs_dim),
                goal.reshape(*goal.shape[:-1], self.goal_dim),
            ],
            dim=-1,
        )
        return self.body(x)

    def forward(self, obs, goal):
        h = self._features(obs, goal)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs, goal, deterministic: bool = False):
        mean, log_std = self.forward(obs, goal)
        if deterministic:
            action = torch.tanh(mean) if self.squashed else mean
            return action, torch.zeros_like(mean[..., 0])
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        log_prob = normal.log_prob(x).sum(dim=-1)
        if self.squashed:
            action = torch.tanh(x)
            log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        else:
            action = x
        return action, log_prob

    def log_prob(self, obs, goal, action, squash: bool = None):
        if squash is None:
            squash = self.squashed
        mean, log_std = self.forward(obs, goal)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        if squash:
            action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            x = torch.atanh(action)
            lp = normal.log_prob(x).sum(dim=-1)
            lp = lp - torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1)
        else:
            lp = normal.log_prob(action).sum(dim=-1)
        return lp

    @torch.no_grad()
    def act(self, obs, goal, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self._device())
        goal_t = torch.as_tensor(np.asarray(goal), dtype=torch.float32, device=self._device())
        action, _ = self.sample(obs_t, goal_t, deterministic=deterministic)
        return action.cpu().numpy()

    def _device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# IQL losses
# ---------------------------------------------------------------------------
def expectile_loss(diff, expectile: float = DEFAULT_EXPECTILE):
    """Asymmetric squared loss used for the IQL value function."""
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * (diff ** 2)


def awr_weights(advantage, temperature: float = DEFAULT_AWR_TEMPERATURE, max_weight: float = DEFAULT_AWR_MAX_WEIGHT):
    """Advantage-weighted regression weights ``exp(temp * A)`` (clamped)."""
    w = torch.exp(temperature * advantage)
    return torch.clamp(w, max=max_weight).detach()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GCIConfig:
    """Hyper-parameters for the GC-IQL baseline."""

    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    discount: float = DEFAULT_DISCOUNT
    expectile: float = DEFAULT_EXPECTILE
    awr_temperature: float = DEFAULT_AWR_TEMPERATURE
    awr_max_weight: float = DEFAULT_AWR_MAX_WEIGHT
    target_update_rate: float = DEFAULT_TARGET_UPDATE_RATE
    activation: str = "relu"
    squashed_policy: bool = True
    log_std_min: float = DEFAULT_LOG_STD_MIN
    log_std_max: float = DEFAULT_LOG_STD_MAX
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    num_candidate_actions: int = DEFAULT_NUM_CANDIDATE_ACTIONS
    bellman_target: str = "v"  # "v" (canonical IQL) or "max_q"

    # goal relabelling mixture
    p_current: float = DEFAULT_P_CURRENT
    p_future: float = DEFAULT_P_FUTURE
    p_random: float = DEFAULT_P_RANDOM
    geom_p: float = DEFAULT_GEOM_P
    goal_tolerance: float = DEFAULT_GOAL_TOLERANCE
    success_reward: float = DEFAULT_SUCCESS_REWARD
    failure_reward: float = DEFAULT_FAILURE_REWARD

    seed: int = 0
    device: str = "cpu"
    steps: Optional[int] = None
    log_interval: int = 1000
    eval_interval: int = 20_000
    checkpoint_interval: int = 50_000
    output_dir: str = "./runs/gc_iql"

    def as_dict(self) -> Dict[str, Any]:
        keys = (
            "hidden_sizes",
            "lr",
            "batch_size",
            "discount",
            "expectile",
            "awr_temperature",
            "awr_max_weight",
            "target_update_rate",
            "activation",
            "squashed_policy",
            "log_std_min",
            "log_std_max",
            "max_grad_norm",
            "num_candidate_actions",
            "bellman_target",
            "p_current",
            "p_future",
            "p_random",
            "geom_p",
            "goal_tolerance",
            "success_reward",
            "failure_reward",
            "seed",
            "device",
            "steps",
            "log_interval",
            "eval_interval",
            "checkpoint_interval",
            "output_dir",
        )
        return {k: getattr(self, k) for k in keys}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class GCIAgent:
    """Goal-Conditioned IQL trainer / agent.

    The agent owns ``Q``, ``V`` and ``pi`` (all goal-conditioned), their target
    copies, three Adam optimizers and the dataset goal-relabelling helpers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[GCIConfig] = None,
        goal_dim: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        _require_torch()
        self.config = config or GCIConfig()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim if goal_dim is not None else obs_dim)
        self.device = torch.device(device or self.config.device or "cpu")

        hs = tuple(self.config.hidden_sizes)
        activ = self.config.activation

        self.q_network = GoalQNetwork(
            self.obs_dim, self.action_dim, self.goal_dim, hs, activ
        ).to(self.device)
        self.v_network = GoalVNetwork(self.obs_dim, self.goal_dim, hs, activ).to(self.device)
        self.policy_network = GoalGaussianPolicy(
            self.obs_dim,
            self.action_dim,
            self.goal_dim,
            hs,
            activ,
            squashed=self.config.squashed_policy,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)

        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_v_network = copy.deepcopy(self.v_network).to(self.device)
        for p in self.target_q_network.parameters():
            p.requires_grad_(False)
        for p in self.target_v_network.parameters():
            p.requires_grad_(False)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.lr)
        self.v_optimizer = torch.optim.Adam(self.v_network.parameters(), lr=self.config.lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.config.lr
        )

        self._rng = np.random.default_rng(self.config.seed)
        self._observations: Optional[np.ndarray] = None
        self._terminals: Optional[np.ndarray] = None
        self.step = 0

    # -- dataset wiring -----------------------------------------------------
    def set_dataset(
        self,
        observations: np.ndarray,
        terminals: Optional[np.ndarray] = None,
        episode_ends: Optional[np.ndarray] = None,
    ) -> None:
        self._observations = np.asarray(observations)
        self._terminals = None if terminals is None else np.asarray(terminals)
        self._episode_ends = None if episode_ends is None else np.asarray(episode_ends)

    # -- relabelling --------------------------------------------------------
    def relabel(self, next_observations: np.ndarray, indices: np.ndarray):
        """Sample goal observations and sparse goal rewards for a batch."""
        if self._observations is not None:
            goals, strategies = relabel_goals(
                self._observations,
                np.asarray(indices, dtype=np.int64),
                terminals=self._terminals,
                rng=self._rng,
                p_current=self.config.p_current,
                p_future=self.config.p_future,
                p_random=self.config.p_random,
                geom_p=self.config.geom_p,
            )
        else:
            goals = np.asarray(next_observations)
            strategies = np.zeros(len(goals), dtype=np.int64)
        rewards = sparse_goal_reward(
            next_observations,
            goals,
            tolerance=self.config.goal_tolerance,
            success_reward=self.config.success_reward,
            failure_reward=self.config.failure_reward,
        )
        return goals, rewards, strategies

    # -- update -------------------------------------------------------------
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One IQL update given a transition batch (dict of numpy arrays/tensors)."""

        def _t(x, dtype=torch.float32):
            if isinstance(x, torch.Tensor):
                return x.to(self.device, dtype=dtype)
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=self.device)

        obs = _t(batch["observations"])
        actions = _t(batch["actions"])
        next_obs = _t(batch["next_observations"])
        terminals = _t(batch.get("terminals", np.zeros(len(obs), dtype=np.float32)))

        n = obs.shape[0]
        indices = batch.get("indices")
        if indices is None:
            indices = np.arange(n, dtype=np.int64)
        indices = np.asarray(indices, dtype=np.int64)

        if "goals" in batch or "goal_observations" in batch:
            goals_np = np.asarray(batch.get("goals", batch.get("goal_observations")))
            rewards_np = np.asarray(
                batch["rewards"]
                if "rewards" in batch
                else sparse_goal_reward(
                    batch["next_observations"], goals_np, tolerance=self.config.goal_tolerance
                )
            )
        else:
            goals_np, rewards_np, _ = self.relabel(batch["next_observations"], indices)

        goals = _t(goals_np)
        rewards = _t(rewards_np)

        # -- V update (expectile regression of Q onto V) --------------------
        with torch.no_grad():
            q_target = self.q_network(obs, actions, goals)
        v = self.v_network(obs, goals)
        v_loss = expectile_loss(q_target - v, self.config.expectile).mean()

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.v_network.parameters(), self.config.max_grad_norm
        )
        self.v_optimizer.step()

        # -- Q update (Bellman) ---------------------------------------------
        with torch.no_grad():
            if self.config.bellman_target == "max_q":
                cand = next_obs.unsqueeze(1).repeat(
                    1, self.config.num_candidate_actions, 1
                )  # (B, N, D)
                cand_goals = goals.unsqueeze(1).repeat(
                    1, self.config.num_candidate_actions, 1
                )
                cand_actions, _ = self.policy_network.sample(cand, cand_goals, deterministic=False)
                q_next = self.target_q_network(
                    cand.reshape(-1, next_obs.shape[-1]),
                    cand_actions.reshape(-1, self.action_dim),
                    cand_goals.reshape(-1, self.goal_dim),
                ).reshape(n, self.config.num_candidate_actions)
                q_next = q_next.max(dim=-1).values
            else:
                q_next = self.target_v_network(next_obs, goals)
            backup = rewards + self.config.discount * (1.0 - terminals) * q_next

        q = self.q_network(obs, actions, goals)
        q_loss = F.mse_loss(q, backup)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.max_grad_norm)
        self.q_optimizer.step()

        # -- Policy update (AWR) -------------------------------------------
        with torch.no_grad():
            advantage = self.q_network(obs, actions, goals) - self.v_network(obs, goals)
            weights = awr_weights(
                advantage,
                temperature=self.config.awr_temperature,
                max_weight=self.config.awr_max_weight,
            )
        log_prob = self.policy_network.log_prob(obs, goals, actions)
        policy_loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(), self.config.max_grad_norm
        )
        self.policy_optimizer.step()

        self.soft_update(self.config.target_update_rate)
        self.step += 1

        with torch.no_grad():
            return {
                "loss": float(v_loss.item() + q_loss.item() + policy_loss.item()),
                "q_loss": float(q_loss.item()),
                "v_loss": float(v_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "q_mean": float(q.mean().item()),
                "v_mean": float(v.mean().item()),
                "reward_mean": float(rewards.mean().item()),
                "weight_mean": float(weights.mean().item()),
                "step": float(self.step),
            }

    # -- target networks ----------------------------------------------------
    def soft_update(self, rate: Optional[float] = None) -> None:
        rate = self.config.target_update_rate if rate is None else float(rate)
        with torch.no_grad():
            for tp, p in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                tp.data.mul_(1.0 - rate).add_(rate * p.data)
            for tp, p in zip(self.target_v_network.parameters(), self.v_network.parameters()):
                tp.data.mul_(1.0 - rate).add_(rate * p.data)

    def hard_update_targets(self) -> None:
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_v_network.load_state_dict(self.v_network.state_dict())

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def select_action(self, obs, goal, deterministic: bool = True) -> np.ndarray:
        """Act conditioned on the *ground-truth* goal (no sampling at eval)."""
        return self.policy_network.act(obs, goal, deterministic=deterministic)

    act = select_action

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "q_network": self.q_network.state_dict(),
            "v_network": self.v_network.state_dict(),
            "policy_network": self.policy_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "target_v_network": self.target_v_network.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "config": self.config.as_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
            "step": self.step,
        }

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        self.q_network.load_state_dict(state["q_network"])
        self.v_network.load_state_dict(state["v_network"])
        self.policy_network.load_state_dict(state["policy_network"])
        if "target_q_network" in state:
            self.target_q_network.load_state_dict(state["target_q_network"])
        else:
            self.hard_update_targets()
        if "target_v_network" in state:
            self.target_v_network.load_state_dict(state["target_v_network"])
        if load_optimizer:
            for key, opt in (
                ("q_optimizer", self.q_optimizer),
                ("v_optimizer", self.v_optimizer),
                ("policy_optimizer", self.policy_optimizer),
            ):
                if key in state:
                    try:
                        opt.load_state_dict(state[key])
                    except Exception:
                        pass
        self.step = int(state.get("step", 0))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> "GCIAgent":
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, load_optimizer=load_optimizer)
        return self

    def train(self) -> "GCIAgent":
        for m in (self.q_network, self.v_network, self.policy_network):
            m.train()
        return self

    def eval(self) -> "GCIAgent":
        for m in (self.q_network, self.v_network, self.policy_network):
            m.eval()
        return self


# Plan-facing aliases.
GCIQL = GCIAgent
GCIQLAgent = GCIAgent


# ---------------------------------------------------------------------------
# Factories / training loop
# ---------------------------------------------------------------------------
def build_gc_iql(
    obs_dim: int,
    action_dim: int,
    config: Optional[GCIConfig] = None,
    goal_dim: Optional[int] = None,
    device: Optional[str] = None,
) -> GCIAgent:
    return GCIAgent(obs_dim, action_dim, config=config, goal_dim=goal_dim, device=device)


def _to_numpy_batch(batch: Any) -> Dict[str, Any]:
    if hasattr(batch, "as_dict"):
        batch = batch.as_dict()
    elif hasattr(batch, "data") and isinstance(batch.data, dict):  # pragma: no cover
        batch = batch.data
    out: Dict[str, Any] = {}
    for k, v in dict(batch).items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = v
    return out


def train_gc_iql(
    buffer: Any,
    obs_dim: int,
    action_dim: int,
    config: Optional[GCIConfig] = None,
    device: Optional[str] = None,
    steps: Optional[int] = None,
    logger: Any = None,
    agent: Optional[GCIAgent] = None,
) -> GCIAgent:
    """Train GC-IQL for ``steps`` updates over a FRE replay buffer."""
    config = config or GCIConfig()
    agent = agent or build_gc_iql(obs_dim, action_dim, config=config, device=device)

    # Wire the dataset for relabelling if the buffer exposes it.
    observations = getattr(buffer, "observations", None)
    terminals = getattr(buffer, "terminals", None)
    if observations is not None:
        agent.set_dataset(observations, terminals)

    total = int(steps if steps is not None else (config.steps or 100_000))
    t0 = time.time()
    last_log: Dict[str, float] = {}

    for i in range(1, total + 1):
        batch = buffer.sample(config.batch_size)
        batch = _to_numpy_batch(batch)
        if observations is not None:
            n = np.asarray(batch["observations"]).shape[0]
            batch.setdefault("indices", np.arange(n, dtype=np.int64))
        metrics = agent.update(batch)
        last_log = metrics
        if logger is not None and hasattr(logger, "metric") and i % config.log_interval == 0:
            logger.metric(step=i, **metrics)
        elif i % config.log_interval == 0:
            print(f"[gc_iql] step {i}/{total} " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    if logger is not None and hasattr(logger, "info"):
        logger.info(f"GC-IQL finished {total} steps in {time.time() - t0:.1f}s")
    return agent


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Goal-Conditioned IQL baseline (FRE paper Table 1).")
    p.add_argument("--domain", type=str, default="antmaze", choices=["antmaze", "kitchen"])
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output-dir", type=str, default="./runs/gc_iql")
    return p.parse_args(argv)


def main(argv: Sequence[str] = None) -> GCIAgent:
    args = parse_args(argv)
    cfg = GCIConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        output_dir=args.output_dir,
    )

    # Lazy data imports so the module is importable without D4RL installed.
    if args.domain == "antmaze":
        from ..data.d4rl_loader import load_antmaze_buffer

        buffer = load_antmaze_buffer(args.dataset) if args.dataset else load_antmaze_buffer()
        obs_dim, action_dim = 29, 8
    else:
        from ..data.d4rl_loader import load_kitchen_buffer

        buffer = load_kitchen_buffer(args.dataset) if args.dataset else load_kitchen_buffer()
        obs_dim, action_dim = 60, 9

    agent = train_gc_iql(buffer, obs_dim, action_dim, config=cfg, steps=cfg.steps)
    os.makedirs(cfg.output_dir, exist_ok=True)
    agent.save(os.path.join(cfg.output_dir, "gc_iql.pt"))
    print(f"Saved GC-IQL checkpoint to {cfg.output_dir}")
    return agent


if __name__ == "__main__":  # pragma: no cover
    main()
