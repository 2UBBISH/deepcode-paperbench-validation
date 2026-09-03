"""Zero-shot downstream FRE evaluation.

This module evaluates a pretrained FRE VAE plus an FRE-conditioned IQL agent on
downstream reward functions. For each task we:

1. Sample ``K`` states from the offline dataset.
2. Label them with the task reward function ``eta_task(s)``.
3. Encode the resulting context pairs into a latent reward vector ``z_task``.
4. Roll out the policy ``pi(a | s, z_task)`` for ``num_episodes`` episodes in the
   live environment.
5. Compute mean and standard deviation of the task reward and report normalized
   scores in ``[0, 100]`` where applicable.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from fre.config import (
    ALL_TASKS,
    ANTMAZE_TASKS,
    EXORL_TASKS,
    KITCHEN_TASKS,
    Config,
    get_config,
    resolve_device,
)
from fre.data.dataset import OfflineDataset
from fre.modeling.fre_vae import FREVAE
from fre.rl.iql import IQL, ImplicitQLearning

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defensive config access helpers (mirror the helpers used by pretrain_encoder
# and train_agent so evaluate.py can be called with either full Config objects
# or smaller standalone dataclass configs).
# ---------------------------------------------------------------------------
def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Return ``obj.name`` if it exists, otherwise ``default``."""
    if obj is None:
        return default
    if hasattr(obj, name):
        value = getattr(obj, name)
        if value is not None:
            return value
    return default


def _cfg_value(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted config path defensively (``a.b.c``)."""
    if cfg is None:
        return default
    obj: Any = cfg
    for part in path.split("."):
        if not hasattr(obj, part):
            return default
        obj = getattr(obj, part)
        if obj is None:
            return default
    return obj


# ---------------------------------------------------------------------------
# Task reward abstraction
# ---------------------------------------------------------------------------
class TaskReward:
    """Callable reward wrapper with a flag indicating sparse success tasks.

    ``sparse=True`` means the reward is a binary success indicator and the
    evaluation should report success rate rather than average per-step reward.
    """

    def __init__(
        self,
        fn: Callable[..., torch.Tensor],
        sparse: bool = False,
        task_name: str = "",
        domain: str = "",
    ) -> None:
        self.fn = fn
        self.sparse = sparse
        self.task_name = task_name
        self.domain = domain

    def __call__(self, states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fn(states, next_states)

    def to(self, device: torch.device | str) -> "TaskReward":
        # Reward functions in this module are stateless with respect to device;
        # we simply remember the requested device for callers that need it.
        self._device = torch.device(device)
        return self


def _to_tensor(x: Any, device: torch.device | str = "cpu") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def _positions(states: torch.Tensor) -> torch.Tensor:
    """Return the 2D position component of observations when available.

    D4RL AntMaze observations start with ``(x, y)``; DeepMind Control
    observations usually contain horizontal positions in the first few
    dimensions. We conservatively use the first two dimensions for all domains
    and document that domain-specific position extractors can be added later.
    """
    if states.dim() == 1:
        return states[:2]
    return states[..., :2]


def _sample_goal_from_dataset(dataset: OfflineDataset, device: torch.device, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    n = max(1, len(dataset.states))
    idx = int(rng.integers(0, n))
    return dataset.states[idx].to(device)


def _gaussian_path_reward(states: torch.Tensor, waypoints: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return ``max_k exp(-||pos - w_k||^2 / (2 sigma^2))`` for each state."""
    pos = _positions(states)  # (..., 2)
    # shape (..., 1, 2) - (num_waypoints, 2) -> (..., num_waypoints)
    diff = pos.unsqueeze(-2) - waypoints.to(pos.device).unsqueeze(0)
    sq = (diff ** 2).sum(dim=-1)
    g = torch.exp(-sq / (2.0 * sigma * sigma))
    return g.max(dim=-1).values


def _linear_reward(states: torch.Tensor, weights: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dot a linear weight vector with either state or velocity.

    If ``next_states`` is provided, the reward is the velocity along
    ``weights`` (i.e. ``(next_state - state) dot weights``). Otherwise it is
    ``state dot weights``.
    """
    if next_states is not None:
        vel = next_states - states
        return (vel * weights.to(states.device)).sum(dim=-1)
    return (states * weights.to(states.device)).sum(dim=-1)


def _make_antmaze_reward(task_name: str, dataset: OfflineDataset, device: torch.device, seed: int) -> TaskReward:
    pos = _positions(dataset.states.to(device))

    if task_name == "ant-goal-reaching":
        goal = _sample_goal_from_dataset(dataset, device, seed)
        threshold = 1.0

        def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
            d = torch.norm(_positions(states) - goal.to(states.device), dim=-1)
            return (d < threshold).float()

        return TaskReward(fn, sparse=True, task_name=task_name, domain="antmaze")

    if task_name == "ant-directional":
        rng = np.random.default_rng(seed)
        direction = torch.tensor(rng.normal(size=(2,)), dtype=torch.float32)
        direction = direction / (direction.norm() + 1e-8)

        def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
            if next_states is None:
                r = (states[..., :2] * direction.to(states.device)).sum(dim=-1)
            else:
                vel = next_states[..., :2] - states[..., :2]
                r = (vel * direction.to(states.device)).sum(dim=-1)
            # Sigmoid-style squash to [0, 1] so normalized scores are meaningful.
            return torch.sigmoid(r)

        return TaskReward(fn, sparse=False, task_name=task_name, domain="antmaze")

    if task_name == "ant-random-simplex":
        rng = np.random.default_rng(seed)
        weights = torch.tensor(rng.dirichlet(np.ones(2)), dtype=torch.float32)

        def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
            r = (states[..., :2] * weights.to(states.device)).sum(dim=-1)
            return torch.sigmoid(r)

        return TaskReward(fn, sparse=False, task_name=task_name, domain="antmaze")

    # Path-family rewards: place waypoints in the AntMaze's normalized 2D
    # coordinate range and reward proximity to the nearest waypoint.
    n_waypoints = 24
    t = torch.linspace(0.0, 2.0 * math.pi, n_waypoints + 1)[:-1]
    loop = torch.stack([0.5 + 0.35 * torch.cos(t), 0.5 + 0.35 * torch.sin(t)], dim=-1)
    edges = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    center = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    if task_name == "ant-path-loop":
        waypoints = loop
    elif task_name == "ant-path-edges":
        waypoints = edges
    elif task_name == "ant-path-center":
        waypoints = center
    else:
        waypoints = center

    def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _gaussian_path_reward(states, waypoints, sigma=0.15)

    return TaskReward(fn, sparse=False, task_name=task_name, domain="antmaze")


def _make_exorl_reward(task_name: str, dataset: OfflineDataset, device: torch.device, seed: int) -> TaskReward:
    # ExORL task names are expected to be e.g. "walker-goal-reaching",
    # "cheetah-goal-reaching", "walker-velocity-forward", etc. Accept both the
    # compact config names and more explicit names.
    name = task_name.lower()

    if "goal" in name:
        goal = _sample_goal_from_dataset(dataset, device, seed)
        threshold = 1.0

        def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
            d = torch.norm(states - goal.to(states.device), dim=-1)
            return (d < threshold).float()

        return TaskReward(fn, sparse=True, task_name=task_name, domain="exorl")

    # Velocity tasks.
    rng = np.random.default_rng(seed)
    direction = torch.tensor(rng.normal(size=(states_dim(dataset),)), dtype=torch.float32)
    direction = direction / (direction.norm() + 1e-8)

    def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        if next_states is None:
            r = (states * direction.to(states.device)).sum(dim=-1)
        else:
            r = ((next_states - states) * direction.to(states.device)).sum(dim=-1)
        return torch.sigmoid(r)

    return TaskReward(fn, sparse=False, task_name=task_name, domain="exorl")


def _make_kitchen_reward(task_name: str, dataset: OfflineDataset, device: torch.device, seed: int) -> TaskReward:
    # The D4RL Kitchen environment exposes seven standard subtask names. Exact
    # object-level subtask detection requires the environment's internal
    # ``env.tasks_to_complete`` machinery; this lightweight implementation uses
    # a deterministic observation-index/threshold heuristic so evaluation can
    # run even when the environment is not importable. A later env module can
    # replace this with semantically accurate subtask checks.
    rng = np.random.default_rng(seed)
    dim = int(rng.integers(0, max(1, dataset.states.shape[1])))
    values = dataset.states[:, dim].float().numpy()
    threshold = float(np.percentile(values, 70)) if len(values) else 0.0

    def fn(states: torch.Tensor, next_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        return (states[..., dim] > threshold).float()

    return TaskReward(fn, sparse=True, task_name=task_name, domain="kitchen")


def states_dim(dataset: OfflineDataset) -> int:
    return int(dataset.states.shape[-1])


def make_task_reward(
    task_name: str,
    dataset: OfflineDataset,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> TaskReward:
    """Build a downstream task reward function.

    Parameters
    ----------
    task_name: one of the names in ``ALL_TASKS``, or a domain-prefixed task
        name (``antmaze:ant-goal-reaching``, ``exorl:walker-goal-reaching``,
        ``kitchen:microwave``).
    dataset: offline dataset used to sample goals/thresholds.
    device: torch device for reward tensors.
    seed: deterministic seed for goal/direction selection.
    """
    device = torch.device(device)
    domain = ""
    name = task_name

    if ":" in task_name:
        domain, name = task_name.split(":", 1)
    else:
        if task_name in ANTMAZE_TASKS:
            domain = "antmaze"
        elif task_name in EXORL_TASKS:
            domain = "exorl"
        elif task_name in KITCHEN_TASKS:
            domain = "kitchen"
        else:
            # Infer from prefix.
            if task_name.startswith("ant"):
                domain = "antmaze"
            elif "walker" in task_name or "cheetah" in task_name:
                domain = "exorl"
            else:
                domain = "kitchen"

    if domain == "antmaze":
        return _make_antmaze_reward(name, dataset, device, seed)
    if domain == "exorl":
        return _make_exorl_reward(name, dataset, device, seed)
    if domain == "kitchen":
        return _make_kitchen_reward(name, dataset, device, seed)
    raise ValueError(f"Unsupported task/domain: {task_name!r}")


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------
def make_eval_env(cfg: Config, dataset: Optional[OfflineDataset] = None) -> Any:
    """Create a live environment for rollouts.

    AntMaze and Kitchen use D4RL. ExORL environments are created through the
    ExORL loader if possible, and otherwise return ``None``.
    """
    domain = (_get(cfg, "domain") or "").lower()
    data_cfg = _get(cfg, "data", cfg)

    if domain in ("antmaze", "kitchen"):
        from fre.data.d4rl_loader import load_d4rl_env

        if domain == "antmaze":
            env_name = (
                _get(data_cfg, "env_name")
                or _get(data_cfg, "antmaze_env_name")
                or "antmaze-large-diverse-v2"
            )
        else:
            env_name = (
                _get(data_cfg, "env_name")
                or _get(data_cfg, "kitchen_env_name")
                or "kitchen-complete-v0"
            )
        return load_d4rl_env(env_name)

    if domain in ("walker", "cheetah", "exorl"):
        try:
            from fre.data.exorl_loader import load_exorl_dataset_and_env

            env_name = _get(data_cfg, "env_name") or _get(data_cfg, "exorl_env_name")
            if env_name is None:
                env_name = "walker" if domain == "walker" else "cheetah"
            _, env = load_exorl_dataset_and_env(cfg, env_name=env_name, device="cpu")
            return env
        except Exception as exc:  # pragma: no cover - depends on optional DMC
            logger.warning("Could not create ExORL environment: %s", exc)
            return None

    # Generic fallback for D4RL-style environments.
    try:
        from fre.data.d4rl_loader import load_d4rl_env

        env_name = _get(data_cfg, "env_name") or _get(cfg, "env_name")
        if env_name:
            return load_d4rl_env(env_name)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not create generic D4RL environment: %s", exc)
    return None


def _seed_env(env: Any, seed: int) -> None:
    if env is None:
        return
    if hasattr(env, "seed"):
        try:
            env.seed(seed)
        except Exception:
            pass
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        try:
            env.observation_space.seed(seed)
        except Exception:
            pass


def _reset_env(env: Any, seed: Optional[int] = None) -> Any:
    if seed is not None:
        _seed_env(env, seed)
    reset_fn = getattr(env, "reset", None)
    if reset_fn is None:
        raise RuntimeError("Evaluation environment does not expose reset().")
    out = reset_fn()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env: Any, action: np.ndarray) -> Tuple[Any, float, bool, bool]:
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) >= 4:
            state, reward, terminated, truncated = out[0], float(out[1]), bool(out[2]), bool(out[3])
        elif len(out) == 3:
            state, reward, done = out[0], float(out[1]), bool(out[2])
            terminated, truncated = done, False
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(out)}")
    else:
        state, reward, terminated, truncated = out, 0.0, False, False
    if isinstance(state, tuple):
        state = state[0]
    return state, reward, terminated, truncated


# ---------------------------------------------------------------------------
# Rollout and evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_task_latent(
    dataset: OfflineDataset,
    model: FREVAE,
    task_reward: TaskReward,
    num_reward_samples: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Encode a downstream task into a stationary latent ``z_task``."""
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(dataset.states)
    idx = rng.integers(0, n, size=num_reward_samples) if n > 0 else np.zeros(num_reward_samples, dtype=np.int64)
    states = dataset.states[idx].to(device)
    rewards = task_reward(states).to(device)
    # Clip rewards to the VAE's configured reward range for stable embedding.
    r_min = float(getattr(model, "reward_min", -1.0))
    r_max = float(getattr(model, "reward_max", 1.0))
    rewards = rewards.clamp(r_min, r_max)
    if rewards.dim() == 0:
        rewards = rewards.reshape(1)
    if states.dim() == 1:
        states = states.unsqueeze(0)
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(0)
    mu, log_sigma, z = model.encode(states, rewards, return_z=True)
    # Use the posterior mean for deterministic zero-shot evaluation.
    return mu.squeeze(0)


def rollout_task(
    env: Any,
    dataset: OfflineDataset,
    agent: ImplicitQLearning,
    task_reward: TaskReward,
    z_task: torch.Tensor,
    device: torch.device,
    num_episodes: int = 20,
    max_episode_steps: int = 1000,
    seed: int = 0,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Roll out a policy conditioned on ``z_task`` and score it with ``task_reward``."""
    if env is None:
        raise RuntimeError("Cannot evaluate without a live environment.")

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    episode_successes: List[float] = []

    for ep in range(num_episodes):
        state = _reset_env(env, seed=seed + ep)
        done = False
        step = 0
        total_reward = 0.0
        any_success = 0.0

        while not done and step < max_episode_steps:
            raw_state = np.asarray(state, dtype=np.float32)
            norm_state = dataset.normalize_states(raw_state)
            state_tensor = torch.as_tensor(norm_state, dtype=torch.float32, device=device).unsqueeze(0)
            action_tensor = agent.get_action(state_tensor, condition=z_task, deterministic=deterministic)
            action = action_tensor.detach().cpu().numpy().reshape(-1)

            next_state, env_reward, terminated, truncated = _step_env(env, action)
            next_raw = np.asarray(next_state, dtype=np.float32)
            next_tensor = torch.as_tensor(next_raw, dtype=torch.float32, device=device).unsqueeze(0)
            reward_tensor = task_reward(state_tensor.squeeze(0), next_tensor.squeeze(0))
            r = float(reward_tensor.detach().cpu().item())

            total_reward += r
            any_success = max(any_success, float((reward_tensor > 0.5).any().cpu().item()))
            state = next_state
            done = bool(terminated) or bool(truncated)
            step += 1

        episode_returns.append(total_reward)
        episode_lengths.append(step)
        episode_successes.append(any_success)

    arr = np.asarray(episode_returns, dtype=np.float32)
    succ_arr = np.asarray(episode_successes, dtype=np.float32)
    lengths = np.asarray(episode_lengths, dtype=np.float32)
    normalized = succ_arr * 100.0 if task_reward.sparse else (arr / np.maximum(lengths, 1.0)) * 100.0

    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std()),
        "mean_normalized": float(normalized.mean()),
        "std_normalized": float(normalized.std()),
        "mean_episode_length": float(lengths.mean()),
        "success_rate": float(succ_arr.mean()),
        "num_episodes": int(len(arr)),
    }


@torch.no_grad()
def evaluate_task(
    cfg: Config,
    dataset: OfflineDataset,
    model: FREVAE,
    agent: ImplicitQLearning,
    task_name: str,
    env: Optional[Any] = None,
    device: Optional[torch.device | str] = None,
    num_episodes: Optional[int] = None,
    num_reward_samples: Optional[int] = None,
    seed: int = 0,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Evaluate FRE on a single downstream task."""
    device = torch.device(resolve_device(device or _get(cfg, "device", "auto")))
    model.eval()
    agent.eval() if hasattr(agent, "eval") else None

    eval_cfg = _get(cfg, "eval", cfg)
    num_episodes = num_episodes or int(_get(eval_cfg, "num_episodes", 20))
    num_reward_samples = num_reward_samples or int(_get(eval_cfg, "num_reward_samples", 32))
    max_episode_steps = int(_get(eval_cfg, "max_episode_steps", 1000))

    task_reward = make_task_reward(task_name, dataset, device=device, seed=seed)
    z_task = encode_task_latent(
        dataset,
        model,
        task_reward,
        num_reward_samples=num_reward_samples,
        device=device,
        seed=seed,
    )
    z_task = z_task.unsqueeze(0) if z_task.dim() == 1 else z_task

    if env is None:
        env = make_eval_env(cfg, dataset)

    result = rollout_task(
        env=env,
        dataset=dataset,
        agent=agent,
        task_reward=task_reward,
        z_task=z_task,
        device=device,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        deterministic=deterministic,
    )
    result["task"] = task_name
    result["domain"] = _get(cfg, "domain", "")
    result["seed"] = seed
    return result


def evaluate_all_tasks(
    cfg: Config,
    dataset: OfflineDataset,
    model: FREVAE,
    agent: ImplicitQLearning,
    env: Optional[Any] = None,
    device: Optional[torch.device | str] = None,
    task_names: Optional[Sequence[str]] = None,
    seeds: Optional[Sequence[int]] = None,
    num_episodes: Optional[int] = None,
    num_reward_samples: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Evaluate all tasks (or a subset) and return per-task/aggregate metrics."""
    domain = (_get(cfg, "domain") or "").lower()
    if task_names is None:
        if domain == "antmaze":
            task_names = ANTMAZE_TASKS
        elif domain in ("walker", "cheetah", "exorl"):
            task_names = EXORL_TASKS
        elif domain == "kitchen":
            task_names = KITCHEN_TASKS
        else:
            task_names = ALL_TASKS

    seeds = seeds or [int(_get(cfg, "seed", 0))]
    results: List[Dict[str, float]] = []
    for seed in seeds:
        for task in task_names:
            result = evaluate_task(
                cfg=cfg,
                dataset=dataset,
                model=model,
                agent=agent,
                task_name=task,
                env=env,
                device=device,
                num_episodes=num_episodes,
                num_reward_samples=num_reward_samples,
                seed=seed,
                deterministic=deterministic,
            )
            results.append(result)
            logger.info("Task %-28s seed %d: return %.2f +/- %.2f | normalized %.2f", task, seed, result["mean_return"], result["std_return"], result["mean_normalized"])

    # Aggregate by task across seeds.
    per_task: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        per_task.setdefault(r["task"], {"mean": [], "std": [], "norm": []})
        per_task[r["task"]]["mean"].append(r["mean_return"])
        per_task[r["task"]]["std"].append(r["std_return"])
        per_task[r["task"]]["norm"].append(r["mean_normalized"])

    task_summary: Dict[str, Dict[str, float]] = {}
    all_norms: List[float] = []
    for task, values in per_task.items():
        task_summary[task] = {
            "mean_return": float(np.mean(values["mean"])),
            "std_return": float(np.std(values["mean"])),
            "mean_normalized": float(np.mean(values["norm"])),
            "std_normalized": float(np.std(values["norm"])),
        }
        all_norms.extend(values["norm"])

    return {
        "per_task": task_summary,
        "results": results,
        "aggregate_mean_normalized": float(np.mean(all_norms)) if all_norms else 0.0,
        "aggregate_std_normalized": float(np.std(all_norms)) if all_norms else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot FRE evaluation")
    parser.add_argument("--config", type=str, default=None, help="YAML config name or path")
    parser.add_argument("--task", type=str, default=None, help="Single task name (default: all tasks for config domain)")
    parser.add_argument("--model-path", type=str, default=None, help="FRE VAE checkpoint path")
    parser.add_argument("--agent-path", type=str, default=None, help="IQL agent checkpoint path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--num-reward-samples", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy rollouts")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    cfg = get_config(args.config or "default")
    if args.num_episodes is not None:
        if not hasattr(cfg, "eval"):
            cfg.eval = type("EvalCfg", (), {})()
        cfg.eval.num_episodes = args.num_episodes
    if args.num_reward_samples is not None:
        if not hasattr(cfg, "eval"):
            cfg.eval = type("EvalCfg", (), {})()
        cfg.eval.num_reward_samples = args.num_reward_samples

    device = resolve_device(args.device)

    # Load dataset.
    domain = _get(cfg, "domain", "antmaze").lower()
    if domain in ("walker", "cheetah", "exorl"):
        from fre.data.exorl_loader import load_exorl_dataset
        dataset = load_exorl_dataset(cfg, device=str(device))
    else:
        from fre.data.d4rl_loader import load_d4rl_dataset
        dataset = load_d4rl_dataset(cfg, device=str(device))

    # Load FRE model.
    model = FREVAE.from_config(cfg.fre, state_dim=dataset.states.shape[-1]) if hasattr(FREVAE, "from_config") else None
    if model is None:
        from fre.modeling.fre_vae import FREVAE as F
        model = F(state_dim=dataset.states.shape[-1])
    model = model.to(device)
    if args.model_path and os.path.exists(args.model_path):
        state_dict = torch.load(args.model_path, map_location=device)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    model.eval()

    # Build/load IQL agent.
    state_dim = dataset.states.shape[-1]
    action_dim = int(np.prod(dataset.actions.shape[-1]))
    agent = IQL(state_dim=state_dim, action_dim=action_dim, condition_dim=cfg.fre.z_dim if hasattr(cfg.fre, "z_dim") else 64, device=str(device))
    agent = agent.to(device)
    if args.agent_path and os.path.exists(args.agent_path):
        agent.load(args.agent_path)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    env = make_eval_env(cfg, dataset)

    if args.task:
        result = evaluate_task(cfg, dataset, model, agent, args.task, env=env, device=device, num_episodes=args.num_episodes, num_reward_samples=args.num_reward_samples, seed=seeds[0], deterministic=not args.stochastic)
        print(result)
    else:
        summary = evaluate_all_tasks(cfg, dataset, model, agent, env=env, device=device, seeds=seeds, num_episodes=args.num_episodes, num_reward_samples=args.num_reward_samples, deterministic=not args.stochastic)
        print("Aggregate normalized:", summary["aggregate_mean_normalized"], "+/-", summary["aggregate_std_normalized"])
        for task, vals in summary["per_task"].items():
            print(f"{task}: {vals}")


__all__ = [
    "TaskReward",
    "make_task_reward",
    "make_eval_env",
    "encode_task_latent",
    "rollout_task",
    "evaluate_task",
    "evaluate_all_tasks",
    "build_parser",
    "main",
]
