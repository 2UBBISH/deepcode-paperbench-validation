"""Evaluation utilities for SAPG and vanilla PPO.

The paper reports two different metrics depending on the task family:

* **Hard tasks** (AllegroKuka: Regrasping / Throw / Reorientation) are evaluated
  with *successes per episode* (the number of times the object reaches the goal
  within the current tolerance ``delta`` during an episode).
* **Easy tasks** (ShadowHand / AllegroHand reorientation) are evaluated with the
  *net episode reward*.

This module provides a simulator-agnostic evaluation loop that works with any of
the vectorized environments in :mod:`sapg.envs` and either the :class:`SAPG` or
:class:`PPO` algorithm objects (both expose ``act`` / ``value``).

The evaluation is intentionally dependency-light: it only requires ``torch`` and
the environment/algorithm objects passed in by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

__all__ = [
    "EvalConfig",
    "EvalResult",
    "evaluate_policy",
    "evaluate_sapg",
    "evaluate_ppo",
    "compute_successes_per_episode",
]


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Configuration for an evaluation rollout.

    Attributes:
        num_episodes: Number of full episodes to run per environment.
        deterministic: Whether to use the mean action (True) or to sample.
        max_steps: Optional hard cap on the number of control steps.  When
            ``None`` the environment's own ``episode_length`` is used.
        record_trajectories: If True, keep per-step rewards for debugging.
        device: Torch device string used for the evaluation tensors.
    """

    num_episodes: int = 10
    deterministic: bool = True
    max_steps: Optional[int] = None
    record_trajectories: bool = False
    device: str = "cpu"


@dataclass
class EvalResult:
    """Aggregated evaluation metrics.

    Attributes:
        mean_return: Mean *net* episode reward across all evaluated episodes.
        std_return: Standard deviation of the episode returns.
        mean_successes_per_episode: Mean number of successes per episode
            (hard-task metric).
        success_rate: Fraction of episodes that achieved at least one success.
        mean_episode_length: Mean number of control steps per episode.
        num_episodes: Number of episodes aggregated.
        per_episode_returns: List of individual episode returns.
        per_episode_successes: List of individual episode success counts.
    """

    mean_return: float = 0.0
    std_return: float = 0.0
    mean_successes_per_episode: float = 0.0
    success_rate: float = 0.0
    mean_episode_length: float = 0.0
    num_episodes: int = 0
    per_episode_returns: List[float] = field(default_factory=list)
    per_episode_successes: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        """Return the scalar metrics as a flat dictionary (for logging)."""
        return {
            "eval/mean_return": self.mean_return,
            "eval/std_return": self.std_return,
            "eval/mean_successes_per_episode": self.mean_successes_per_episode,
            "eval/success_rate": self.success_rate,
            "eval/mean_episode_length": self.mean_episode_length,
            "eval/num_episodes": float(self.num_episodes),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _reset_hidden(hidden: Any, done: torch.Tensor) -> Any:
    """Zero the recurrent hidden state for environments that just finished.

    ``hidden`` is either ``None`` (feed-forward policy) or a tuple
    ``(h, c)`` of tensors with shape ``(num_layers, num_envs, hidden_size)``.
    """
    if hidden is None:
        return None
    if not isinstance(hidden, (tuple, list)):
        return hidden
    new_hidden = []
    for h in hidden:
        if h is None:
            new_hidden.append(None)
            continue
        # done: (num_envs,) -> (1, num_envs, 1)
        mask = (~done).to(h.dtype).view(1, -1, 1)
        new_hidden.append(h * mask)
    return tuple(new_hidden)


def _extract_success(info: Dict[str, Any], num_envs: int, device: torch.device) -> torch.Tensor:
    """Pull a per-env success flag out of an env ``info`` dict.

    Different environments expose success under slightly different keys; we
    accept ``success`` (preferred), ``is_success`` or ``successes``.  If none
    are present we fall back to zeros so evaluation still runs.
    """
    for key in ("success", "is_success", "successes"):
        if key in info and info[key] is not None:
            val = info[key]
            if isinstance(val, torch.Tensor):
                return val.to(device=device, dtype=torch.float32).reshape(-1)
            return torch.as_tensor(val, device=device, dtype=torch.float32).reshape(-1)
    return torch.zeros(num_envs, device=device, dtype=torch.float32)


def compute_successes_per_episode(
    success_flags: torch.Tensor,
    episode_lengths: torch.Tensor,
) -> float:
    """Compute mean successes per episode from per-step success flags.

    Args:
        success_flags: Tensor of shape ``(num_envs, horizon)`` containing 0/1
            success indicators for each control step.
        episode_lengths: Tensor of shape ``(num_envs,)`` with the number of
            steps each environment actually ran.

    Returns:
        The mean number of successes per episode across environments.
    """
    if success_flags.numel() == 0:
        return 0.0
    flags = success_flags.float()
    lengths = episode_lengths.float().clamp(min=1.0)
    totals = flags.sum(dim=-1)
    return float((totals / lengths).mean().item())


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_policy(
    env: Any,
    algorithm: Any,
    config: Optional[EvalConfig] = None,
    worker_ids: Optional[torch.Tensor] = None,
    num_workers: int = 1,
) -> EvalResult:
    """Run a deterministic (or stochastic) evaluation of ``algorithm`` on ``env``.

    Args:
        env: A vectorized environment exposing ``reset()``, ``step(actions)``,
            ``num_envs``, ``observation_dim`` and ``num_actions``.
        algorithm: Either a :class:`~sapg.sapg_algorithm.SAPG` or
            :class:`~sapg.ppo_baseline.PPO` instance.  Must expose
            ``act(obs, worker_ids, hidden, deterministic)`` (SAPG) or
            ``act(obs, hidden, deterministic)`` (PPO).
        config: Optional :class:`EvalConfig`.
        worker_ids: Optional per-env worker/block ids used to condition the
            shared network.  Defaults to zeros (single policy).
        num_workers: Number of distinct workers (used when ``worker_ids`` is
            ``None`` and we want to spread envs across blocks).

    Returns:
        An :class:`EvalResult` with the aggregated metrics.
    """
    config = config or EvalConfig()
    device = torch.device(config.device)

    num_envs = int(getattr(env, "num_envs", 1))
    if worker_ids is None:
        if num_workers > 1:
            # Spread environments evenly across workers/blocks.
            worker_ids = torch.arange(num_envs, device=device) % num_workers
        else:
            worker_ids = torch.zeros(num_envs, device=device, dtype=torch.long)
    else:
        worker_ids = worker_ids.to(device)

    max_steps = config.max_steps
    if max_steps is None:
        max_steps = int(getattr(env, "episode_length", 200))

    # Per-env accumulators.
    ep_returns = torch.zeros(num_envs, device=device)
    ep_successes = torch.zeros(num_envs, device=device)
    ep_lengths = torch.zeros(num_envs, device=device)

    finished_returns: List[float] = []
    finished_successes: List[float] = []
    finished_lengths: List[float] = []

    obs = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
    obs = obs.to(device)

    hidden = None
    if getattr(algorithm, "is_recurrent", False):
        hidden = algorithm.init_hidden(num_envs) if hasattr(algorithm, "init_hidden") else None

    episodes_done = 0
    target_episodes = max(1, config.num_episodes) * num_envs

    step = 0
    while episodes_done < target_episodes and step < max_steps * config.num_episodes:
        actions, _, hidden = _act(algorithm, obs, worker_ids, hidden, config.deterministic)
        obs, reward, done, info = env.step(actions)

        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
        obs = obs.to(device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, device=device, dtype=torch.float32)
        reward = reward.to(device).reshape(-1)
        if not isinstance(done, torch.Tensor):
            done = torch.as_tensor(done, device=device, dtype=torch.bool)
        done = done.to(device).reshape(-1).bool()

        success = _extract_success(info, num_envs, device)

        ep_returns += reward
        ep_successes += success
        ep_lengths += 1.0

        if done.any():
            done_idx = done.nonzero(as_tuple=False).reshape(-1)
            for idx in done_idx.tolist():
                finished_returns.append(float(ep_returns[idx].item()))
                finished_successes.append(float(ep_successes[idx].item()))
                finished_lengths.append(float(ep_lengths[idx].item()))
                episodes_done += 1
            # Reset accumulators for finished envs.
            ep_returns[done_idx] = 0.0
            ep_successes[done_idx] = 0.0
            ep_lengths[done_idx] = 0.0
            hidden = _reset_hidden(hidden, done)

        step += 1

    # Aggregate.
    result = EvalResult(num_episodes=len(finished_returns))
    if finished_returns:
        returns_t = torch.tensor(finished_returns, dtype=torch.float32)
        successes_t = torch.tensor(finished_successes, dtype=torch.float32)
        lengths_t = torch.tensor(finished_lengths, dtype=torch.float32)
        result.mean_return = float(returns_t.mean().item())
        result.std_return = float(returns_t.std(unbiased=False).item()) if returns_t.numel() > 1 else 0.0
        result.mean_successes_per_episode = float(successes_t.mean().item())
        result.success_rate = float((successes_t > 0).float().mean().item())
        result.mean_episode_length = float(lengths_t.mean().item())
        result.per_episode_returns = finished_returns
        result.per_episode_successes = finished_successes
    return result


def _act(
    algorithm: Any,
    obs: torch.Tensor,
    worker_ids: torch.Tensor,
    hidden: Any,
    deterministic: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Any]:
    """Dispatch to the correct ``act`` signature for SAPG vs PPO."""
    try:
        # SAPG signature: act(obs, worker_ids, hidden, deterministic)
        out = algorithm.act(obs, worker_ids, hidden, deterministic)
    except TypeError:
        # PPO signature: act(obs, hidden, deterministic)
        out = algorithm.act(obs, hidden, deterministic)

    if isinstance(out, tuple):
        if len(out) == 3:
            actions, logprobs, hidden = out
        elif len(out) == 2:
            actions, hidden = out
            logprobs = None
        else:  # pragma: no cover - defensive
            actions = out[0]
            logprobs = None
            hidden = None
    else:  # pragma: no cover - defensive
        actions = out
        logprobs = None
    return actions, logprobs, hidden


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------
def evaluate_sapg(
    env: Any,
    sapg: Any,
    config: Optional[EvalConfig] = None,
    num_workers: Optional[int] = None,
) -> EvalResult:
    """Evaluate a :class:`SAPG` algorithm, spreading envs across its workers."""
    if num_workers is None:
        num_workers = int(getattr(getattr(sapg, "config", None), "num_blocks", 1))
    return evaluate_policy(env, sapg, config=config, num_workers=num_workers)


def evaluate_ppo(
    env: Any,
    ppo: Any,
    config: Optional[EvalConfig] = None,
) -> EvalResult:
    """Evaluate a vanilla :class:`PPO` algorithm (single policy)."""
    return evaluate_policy(env, ppo, config=config, num_workers=1)
