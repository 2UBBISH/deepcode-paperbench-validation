"""Goal-Conditioned Behaviour Cloning (GC-BC) baseline for FRE.

Implements the GC-BC baseline described in the FRE reproduction plan:

* MLP with **3 hidden layers of 512** units, ReLU activations and LayerNorm.
* Diagonal Gaussian output head: a linear layer for the mean and a learned
  ``log_std`` clamped above at ``-5.0`` (a very small fixed variance, as in the
  original goal-conditioned BC baselines).
* Maximum-likelihood objective ``-E[ log pi(a | s, g) ]`` on the offline data.
* Goal sampling: **geometric future-state sampling only** (as specified for
  GC-BC in the paper's baseline appendix).

The module is intentionally import-light: :mod:`torch` is required, while the
replay-buffer / goal-relabelling helpers are imported lazily so the file can be
executed standalone (``python -m fre.baselines.gc_bc``) as well as imported.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is a hard requirement in practice
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = exc


__all__ = [
    "GCBConfig",
    "GaussianPolicy",
    "GCBCAgent",
    "GCBC",
    "geometric_future_goal_indices",
    "relabel_geometric_goals",
    "sample_goal_batch",
    "build_gc_bc",
    "main",
]


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (512, 512, 512)
DEFAULT_LR = 1e-4
DEFAULT_BATCH_SIZE = 512
DEFAULT_LOG_STD_CLAMP = -5.0
DEFAULT_GEOM_P = 0.5
DEFAULT_MAX_GRAD_NORM = 10.0

_ACTIVATIONS = {
    "relu": nn.ReLU if _TORCH_AVAILABLE else None,
    "gelu": nn.GELU if _TORCH_AVAILABLE else None,
    "tanh": nn.Tanh if _TORCH_AVAILABLE else None,
    "silu": nn.SiLU if _TORCH_AVAILABLE else None,
}


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "GC-BC requires PyTorch. Install it with `pip install torch`."
        ) from globals().get("_TORCH_IMPORT_ERROR")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class GCBConfig:
    """Hyper-parameters for the GC-BC baseline."""

    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    log_std_min: float = DEFAULT_LOG_STD_CLAMP
    log_std_max: float = 2.0
    layer_norm: bool = True
    activation: str = "relu"
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    geom_p: float = DEFAULT_GEOM_P
    goal_mode: str = "geometric"
    seed: int = 0
    device: str = "cpu"
    steps: Optional[int] = None
    log_interval: int = 1000
    output_dir: str = "./runs/gc_bc"

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["hidden_sizes"] = list(self.hidden_sizes)
        return out


# ---------------------------------------------------------------------------
# networks
# ---------------------------------------------------------------------------


def _build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    activation: str = "relu",
    layer_norm: bool = True,
) -> Tuple[nn.Module, int]:
    """Return a sequential MLP body and its output feature dimension."""

    layers: List[nn.Module] = []
    last = in_dim
    act_cls = _ACTIVATIONS.get(str(activation).lower(), nn.ReLU)
    for h in hidden_sizes:
        layers.append(nn.Linear(last, int(h)))
        if layer_norm:
            layers.append(nn.LayerNorm(int(h)))
        layers.append(act_cls())
        last = int(h)
    return nn.Sequential(*layers), last


class GaussianPolicy(nn.Module):
    """Goal-conditioned diagonal Gaussian policy ``pi(a | s, g)``.

    Parameters
    ----------
    obs_dim, action_dim, goal_dim:
        Feature dimensions.  ``goal_dim`` defaults to ``obs_dim`` (the goal is a
        state).
    hidden_sizes:
        Hidden layer widths (default ``(512, 512, 512)``).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
        layer_norm: bool = True,
        log_std_min: float = DEFAULT_LOG_STD_CLAMP,
        log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        _require_torch()
        goal_dim = int(goal_dim if goal_dim is not None else obs_dim)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = goal_dim
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.body, feat = _build_mlp(
            self.obs_dim + self.goal_dim,
            hidden_sizes,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.mean = nn.Linear(feat, self.action_dim)
        # Learned but clamped; initialised at the (very negative) paper default.
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(log_std_min))
        )

    # -- helpers ---------------------------------------------------------
    def _prepare(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        goal = torch.as_tensor(goal, dtype=torch.float32, device=self._device)
        if goal.shape[-1] != self.goal_dim:
            # allow single-goal broadcast
            if goal.numel() == self.goal_dim:
                goal = goal.reshape(*([1] * (obs.dim() - 1)), self.goal_dim)
            goal = goal.expand(*obs.shape[:-1], self.goal_dim)
        if obs.shape[:-1] != goal.shape[:-1]:
            goal = goal.expand_as(obs[..., : self.goal_dim])
        return torch.cat([obs, goal], dim=-1)

    @property
    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self, obs: torch.Tensor, goal: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._prepare(obs, goal)
        h = self.body(x)
        mean = self.mean(h)
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max)
        log_std = log_std.expand_as(mean)
        return mean, log_std

    # -- distribution ----------------------------------------------------
    def sample(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs, goal)
        if deterministic:
            return torch.tanh(mean), torch.zeros(mean.shape[:-1], device=mean.device)
        std = log_std.exp()
        eps = torch.randn_like(mean)
        x = mean + std * eps
        action = torch.tanh(x)
        # tanh change-of-variables
        log_prob = -0.5 * (
            ((x - mean) / (std + 1e-8)) ** 2
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(-1)
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob

    def log_prob(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        action: torch.Tensor,
        squash: bool = True,
    ) -> torch.Tensor:
        mean, log_std = self.forward(obs, goal)
        action = torch.as_tensor(action, dtype=torch.float32, device=mean.device)
        if squash:
            action = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            action = torch.atanh(action)
        std = log_std.exp()
        var = std.pow(2)
        log_prob = -0.5 * (
            ((action - mean).pow(2) / (var + 1e-8))
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        log_prob = log_prob.sum(-1)
        if squash:
            log_prob = log_prob - torch.log(1.0 - torch.tanh(action).pow(2) + 1e-6).sum(-1)
        return log_prob

    def act(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        with torch.no_grad():
            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self._device)
            if not isinstance(goal, torch.Tensor):
                goal = torch.as_tensor(np.asarray(goal), dtype=torch.float32, device=self._device)
            action, _ = self.sample(obs, goal, deterministic=deterministic)
            return action

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"goal_dim={self.goal_dim}, log_std_min={self.log_std_min}"
        )


# ---------------------------------------------------------------------------
# goal relabelling (geometric future-state sampling only)
# ---------------------------------------------------------------------------


def geometric_future_goal_indices(
    indices: np.ndarray,
    episode_ends: np.ndarray,
    episode_starts: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    geom_p: float = DEFAULT_GEOM_P,
) -> np.ndarray:
    """Sample a future state within the *same trajectory* for each index.

    Uses a geometric offset (success probability ``geom_p``) clamped to the end
    of the containing episode, matching the "geometric future-state sampling
    only" specification of GC-BC.
    """

    indices = np.asarray(indices, dtype=np.int64)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)
    if rng is None:
        rng = np.random.default_rng(0)
    if episode_starts is None:
        episode_starts = np.concatenate([[0], episode_ends[:-1]])
    episode_starts = np.asarray(episode_starts, dtype=np.int64)

    goals = np.empty_like(indices)
    for i, idx in enumerate(indices):
        ep = int(np.searchsorted(episode_ends, idx, side="right"))
        ep = min(ep, len(episode_ends) - 1)
        start = int(episode_starts[ep]) if ep < len(episode_starts) else 0
        end = int(episode_ends[ep])  # exclusive
        span = max(end - idx - 1, 1)
        offset = int(rng.geometric(geom_p)) if geom_p > 0 else 1
        goal_idx = min(idx + offset, end - 1)
        goal_idx = max(goal_idx, idx)
        goals[i] = goal_idx
    return goals


def relabel_geometric_goals(
    observations: np.ndarray,
    terminals: Optional[np.ndarray] = None,
    episode_ends: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    geom_p: float = DEFAULT_GEOM_P,
    indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return goal observations for the given ``indices`` via geometric sampling."""

    observations = np.asarray(observations, dtype=np.float32)
    if indices is None:
        indices = np.arange(len(observations))
    indices = np.asarray(indices, dtype=np.int64)
    if episode_ends is None:
        episode_ends = _episode_ends_from_terminals(terminals, len(observations))
    goal_idx = geometric_future_goal_indices(
        indices, episode_ends, rng=rng, geom_p=geom_p
    )
    return observations[goal_idx]


def _episode_ends_from_terminals(
    terminals: Optional[np.ndarray], num_states: int
) -> np.ndarray:
    if terminals is None or len(terminals) == 0:
        return np.array([num_states], dtype=np.int64)
    terminals = np.asarray(terminals).reshape(-1)
    ends = np.nonzero(terminals > 0.5)[0]
    if len(ends) == 0:
        return np.array([num_states], dtype=np.int64)
    ends = (ends + 1).astype(np.int64)
    if ends[-1] < num_states:
        ends = np.concatenate([ends, [num_states]])
    return ends


def sample_goal_batch(
    batch: Dict[str, np.ndarray],
    rng: Optional[np.random.Generator] = None,
    geom_p: float = DEFAULT_GEOM_P,
    episode_ends: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Attach geometrically sampled future goals to a transition batch."""

    rng = rng if rng is not None else np.random.default_rng()
    obs = np.asarray(batch["observations"])
    goals = batch.get("goals")
    if goals is None:
        goals = relabel_geometric_goals(
            obs,
            terminals=batch.get("terminals"),
            episode_ends=episode_ends,
            rng=rng,
            geom_p=geom_p,
        )
    out = dict(batch)
    out["goals"] = np.asarray(goals, dtype=np.float32)
    out["goal_observations"] = out["goals"]
    return out


# ---------------------------------------------------------------------------
# agent / trainer
# ---------------------------------------------------------------------------


class GCBCAgent:
    """Goal-conditioned behaviour-cloning trainer."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[GCBConfig] = None,
        goal_dim: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        _require_torch()
        self.config = config or GCBConfig()
        self.device = torch.device(device or self.config.device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.goal_dim = int(goal_dim if goal_dim is not None else obs_dim)

        self.policy = GaussianPolicy(
            self.obs_dim,
            self.action_dim,
            goal_dim=self.goal_dim,
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            layer_norm=self.config.layer_norm,
            log_std_min=self.config.log_std_min,
            log_std_max=self.config.log_std_max,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)
        self.rng = np.random.default_rng(self.config.seed)
        self._step = 0
        self._episode_ends: Optional[np.ndarray] = None

    # -- data helpers ----------------------------------------------------
    def set_dataset(self, observations: np.ndarray, terminals: Optional[np.ndarray] = None) -> None:
        """Cache episode boundaries for geometric goal sampling."""

        if terminals is not None:
            self._episode_ends = _episode_ends_from_terminals(terminals, len(observations))
        else:
            self._episode_ends = None
        self._observations = np.asarray(observations, dtype=np.float32)

    def _to_tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    # -- core update -----------------------------------------------------
    def loss(
        self,
        obs: torch.Tensor,
        goal: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """MLE loss ``-E[log pi(a|s,g)]``."""

        log_prob = self.policy.log_prob(obs, goal, actions)
        return -log_prob.mean()

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One gradient step on a transition batch (goals sampled geometrically)."""

        obs = np.asarray(batch["observations"], dtype=np.float32)
        actions = np.asarray(batch["actions"], dtype=np.float32)
        goals = batch.get("goals")
        if goals is None:
            goals = relabel_geometric_goals(
                obs,
                terminals=batch.get("terminals"),
                episode_ends=batch.get("episode_ends", self._episode_ends),
                rng=self.rng,
                geom_p=self.config.geom_p,
            )
        goals = np.asarray(goals, dtype=np.float32)

        obs_t = self._to_tensor(obs)
        goal_t = self._to_tensor(goals)
        act_t = self._to_tensor(actions)

        loss = self.loss(obs_t, goal_t, act_t)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.max_grad_norm
            )
        self.optimizer.step()
        self._step += 1

        with torch.no_grad():
            mean, log_std = self.policy(obs_t, goal_t)
            mse = F.mse_loss(torch.tanh(mean), act_t).item()
        return {
            "loss": float(loss.item()),
            "bc_loss": float(loss.item()),
            "action_mse": float(mse),
            "log_std_mean": float(log_std.mean().item()),
            "step": self._step,
        }

    # -- inference -------------------------------------------------------
    def select_action(
        self, obs, goal, deterministic: bool = True
    ) -> np.ndarray:
        obs_t = self._to_tensor(obs)
        goal_t = self._to_tensor(goal)
        with torch.no_grad():
            action, _ = self.policy.sample(obs_t, goal_t, deterministic=deterministic)
        return action.detach().cpu().numpy()

    act = select_action

    # -- checkpointing ---------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.as_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "goal_dim": self.goal_dim,
            "step": self._step,
        }

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        self.policy.load_state_dict(state["policy"])
        if load_optimizer and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        self._step = int(state.get("step", 0))

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> "GCBCAgent":
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state, load_optimizer=load_optimizer)
        return self

    def train(self) -> "GCBCAgent":
        self.policy.train()
        return self

    def eval(self) -> "GCBCAgent":
        self.policy.eval()
        return self


# Plan-facing alias.
GCBC = GCBCAgent


# ---------------------------------------------------------------------------
# training loop
# ---------------------------------------------------------------------------


def _sample_buffer_batch(buffer, batch_size: int) -> Dict[str, np.ndarray]:
    """Sample a batch from any FRE replay buffer instance."""

    try:
        batch = buffer.sample(batch_size)
    except TypeError:
        batch = buffer.sample(batch_size=batch_size)
    if hasattr(batch, "as_dict"):
        batch = batch.as_dict()
    out = {}
    for k, v in dict(batch).items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        out[k] = np.asarray(v)
    return out


def build_gc_bc(
    obs_dim: int,
    action_dim: int,
    config: Optional[GCBConfig] = None,
    goal_dim: Optional[int] = None,
    device: Optional[str] = None,
) -> GCBCAgent:
    """Factory mirroring the other ``build_*`` helpers in the codebase."""

    return GCBCAgent(obs_dim, action_dim, config=config, goal_dim=goal_dim, device=device)


def train_gc_bc(
    buffer,
    obs_dim: int,
    action_dim: int,
    config: Optional[GCBConfig] = None,
    device: Optional[str] = None,
    steps: Optional[int] = None,
    logger: Any = None,
) -> GCBCAgent:
    """Train GC-BC on a FRE replay buffer for ``steps`` gradient steps."""

    cfg = config or GCBConfig()
    agent = GCBCAgent(obs_dim, action_dim, config=cfg, device=device)
    observations = None
    terminals = None
    if hasattr(buffer, "observations"):
        observations = np.asarray(buffer.observations)
    if hasattr(buffer, "terminals"):
        terminals = np.asarray(buffer.terminals)
    if observations is not None:
        agent.set_dataset(observations, terminals)

    total = int(steps or cfg.steps or 100_000)
    t0 = time.time()
    for step in range(total):
        batch = _sample_buffer_batch(buffer, cfg.batch_size)
        metrics = agent.update(batch)
        if logger is not None and (step % max(cfg.log_interval, 1) == 0):
            try:
                logger.metric(step=step, **metrics)
            except Exception:
                pass
        elif step % max(cfg.log_interval, 1) == 0:
            print(
                f"[gc_bc] step {step}/{total} loss={metrics['loss']:.4f} "
                f"({time.time() - t0:.1f}s)"
            )
    return agent


def main(argv: Optional[Sequence[str]] = None) -> GCBCAgent:
    parser = argparse.ArgumentParser(description="Goal-conditioned BC baseline")
    parser.add_argument("--domain", default="antmaze")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="./runs/gc_bc")
    args = parser.parse_args(argv)

    from ..data.d4rl_loader import (  # noqa: WPS433
        load_antmaze_buffer,
        load_kitchen_buffer,
    )

    cfg = GCBConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        steps=args.steps,
        output_dir=args.output_dir,
    )
    if args.domain == "kitchen":
        buffer = load_kitchen_buffer()
        action_dim, obs_dim = 9, 60
    else:
        buffer = load_antmaze_buffer()
        action_dim, obs_dim = 8, 29
    agent = train_gc_bc(buffer, obs_dim, action_dim, config=cfg, steps=args.steps)
    agent.save(os.path.join(args.output_dir, "gc_bc.pt"))
    return agent


if __name__ == "__main__":  # pragma: no cover
    main()
