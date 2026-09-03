"""Qualitative visualization for FRE on AntMaze (Figure 3 reproduction).

This script loads a trained FRE agent checkpoint, selects a downstream
AntMaze task, encodes the task from a small number of state-reward examples,
and produces a single figure containing:

    1. true task reward over an XY slice of the state space
    2. the 32 (or ``--num_examples``) encoding states colored by reward
    3. the FRE decoder's reconstruction of the reward over the same slice
    4. policy rollouts in the D4RL AntMaze environment
    5. the FRE-conditioned IQL value function over the same slice

The script is intentionally defensive: if MuJoCo/D4RL is unavailable, the
policy-rollout panel is skipped instead of crashing the whole visualization.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

# Make repository root importable when the script is executed directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from fre.agent import FREAgent  # noqa: E402
from fre.dataset import build_state_pool, load_offline_dataset  # noqa: E402
from fre.utils import get_logger, resolve_device, set_seed  # noqa: E402
from envs.antmaze_wrapper import (  # noqa: E402
    ANTMAZE_TASKS,
    DEFAULT_ANTMAZE_GOAL,
    make_antmaze_task_reward,
    sample_task_reward_states,
)

LOGGER = get_logger("visualize_antmaze")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _to_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy arrays or tensors to a torch tensor on ``device``."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device, dtype=dtype)


def _unnormalize(states: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    """Map normalized states back to raw observation space when stats exist."""
    if mean is None or std is None:
        return states
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return states * (std + 1e-6) + mean


def _normalize(states: np.ndarray, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> np.ndarray:
    """Map raw states to normalized observation space when stats exist."""
    if mean is None or std is None:
        return states
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (states - mean) / (std + 1e-6)


def _get_iql_networks(agent: FREAgent) -> Any:
    """Return the agent's IQL network container using robust attribute lookup."""
    for name in ("iql", "iql_networks", "networks"):
        if hasattr(agent, name):
            return getattr(agent, name)
    raise AttributeError("Could not locate IQL networks on FREAgent")


def _get_value_network(agent: FREAgent) -> torch.nn.Module:
    iql = _get_iql_networks(agent)
    for name in ("v_network", "v", "value_network"):
        if hasattr(iql, name):
            return getattr(iql, name)
    raise AttributeError("Could not locate V network on IQL container")


def _get_dataset_stats(dataset: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read state normalization statistics from an offline dataset if present."""
    mean = getattr(dataset, "state_mean", None)
    std = getattr(dataset, "state_std", None)
    if mean is not None:
        mean = np.asarray(mean, dtype=np.float32)
    if std is not None:
        std = np.asarray(std, dtype=np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_agent(
    checkpoint: str,
    state_dim: int,
    action_dim: int,
    state_pool: np.ndarray,
    dataset: Any,
    device: torch.device,
    latent_dim: int = 128,
) -> FREAgent:
    """Construct an :class:`FREAgent` and load a checkpoint flexibly."""
    agent = FREAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        dataset=dataset,
        state_pool=state_pool,
        device=device,
        freeze_vae=True,
    )
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device)
    if isinstance(state, dict):
        state_dict = None
        for key in ("state_dict", "agent", "model", "agent_state_dict"):
            if key in state and isinstance(state[key], dict):
                state_dict = state[key]
                break
        if state_dict is None:
            state_dict = state
        agent.load_state_dict(state_dict)
    else:
        agent.load_state_dict(state)
    agent.to(device)
    agent.eval() if hasattr(agent, "eval") else None
    return agent


def build_policy_fn(
    agent: FREAgent,
    z: torch.Tensor,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a policy closure that normalizes observations and is conditioned on ``z``."""

    def policy(obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        if mean is not None and std is not None:
            obs = (obs - mean) / (std + 1e-6)
        return agent.get_action(obs, z=z, deterministic=True)

    return policy


# ---------------------------------------------------------------------------
# Task encoding
# ---------------------------------------------------------------------------
def encode_task_for_visualization(
    agent: FREAgent,
    reward_fn: Callable[[np.ndarray], np.ndarray],
    state_pool: np.ndarray,
    num_examples: int,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Encode a task into a latent code and return the raw encoding states.

    The encoder receives *normalized* states (matching RL training), while
    rewards are computed in raw observation space where task semantics live.
    """
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(state_pool), size=min(num_examples, len(state_pool)), replace=False)
    normalized_states = np.asarray(state_pool[idx], dtype=np.float32)
    raw_states = _unnormalize(normalized_states, mean, std)
    rewards = np.asarray(reward_fn(raw_states), dtype=np.float32)

    states_t = _to_tensor(normalized_states, device)
    rewards_t = _to_tensor(rewards, device)
    agent.eval() if hasattr(agent, "eval") else None
    with torch.no_grad():
        mu, logvar, z = agent.vae.encode(states_t, rewards_t)
    return z, raw_states, rewards


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------
def make_xy_grid(
    raw_state_pool: np.ndarray,
    resolution: int,
    mean: Optional[np.ndarray],
    std: Optional[np.ndarray],
    state_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an XY grid in raw coordinates and matching normalized states."""
    xs = np.asarray(raw_state_pool[:, 0], dtype=np.float32)
    ys = np.asarray(raw_state_pool[:, 1], dtype=np.float32)
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    pad_x = max(0.05 * (x_max - x_min), 0.5)
    pad_y = max(0.05 * (y_max - y_min), 0.5)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    xy = np.stack([X.ravel(), Y.ravel()], axis=1).astype(np.float32)

    raw_grid = np.zeros((xy.shape[0], state_dim), dtype=np.float32)
    raw_grid[:, 0] = xy[:, 0]
    raw_grid[:, 1] = xy[:, 1]
    normalized_grid = _normalize(raw_grid, mean, std)
    return raw_grid, normalized_grid, (x_min, x_max, y_min, y_max)


# ---------------------------------------------------------------------------
# Policy rollout
# ---------------------------------------------------------------------------
def collect_policy_positions(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env_name: str,
    num_rollouts: int,
    max_episode_steps: int,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """Collect XY positions from a few policy rollouts.

    Returns ``None`` if the environment cannot be created, keeping the figure
    usable on CPU-only/no-MuJoCo machines.
    """
    try:
        from envs.antmaze_wrapper import AntMazeWrapper

        env = AntMazeWrapper(env_name=env_name, max_episode_steps=max_episode_steps)
    except Exception as exc:  # noqa: BLE001 - environment optional
        LOGGER.warning("Could not create AntMaze environment for rollout panel: %s", exc)
        return None

    positions: list[Tuple[float, float]] = []
    try:
        for ep in range(num_rollouts):
            if hasattr(env, "seed"):
                try:
                    env.seed(seed + ep)
                except Exception:  # noqa: BLE001
                    pass
            obs = env.reset()
            done = False
            steps = 0
            while not done and steps < max_episode_steps:
                action = policy_fn(np.asarray(obs, dtype=np.float32))
                obs, _, done, _ = env.step(action)
                try:
                    xy = env.get_xy(obs)
                except Exception:  # noqa: BLE001
                    xy = np.asarray(obs)[:2]
                positions.append((float(xy[0]), float(xy[1])))
                steps += 1
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Policy rollout failed: %s", exc)
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    if not positions:
        return None
    return np.asarray(positions, dtype=np.float32)


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------
def build_figure(
    task_name: str,
    true_rewards: np.ndarray,
    decoded_rewards: np.ndarray,
    values: np.ndarray,
    grid_shape: Tuple[int, int],
    encoding_raw_states: np.ndarray,
    encoding_rewards: np.ndarray,
    policy_positions: Optional[np.ndarray],
    extent: Tuple[float, float, float, float],
    output_path: str,
) -> None:
    """Render and save the five-panel qualitative figure."""
    resolution_x, resolution_y = grid_shape
    true_map = true_rewards.reshape(resolution_y, resolution_x)
    decoded_map = decoded_rewards.reshape(resolution_y, resolution_x)
    value_map = values.reshape(resolution_y, resolution_x)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.2))
    x_min, x_max, y_min, y_max = extent
    imshow_kwargs = dict(
        extent=(x_min, x_max, y_min, y_max),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
    )

    ax = axes[0]
    im0 = ax.imshow(true_map, cmap="viridis", **imshow_kwargs)
    ax.set_title("True reward")
    fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    sc = ax.scatter(
        encoding_raw_states[:, 0],
        encoding_raw_states[:, 1],
        c=encoding_rewards,
        cmap="viridis",
        s=18,
        edgecolors="black",
        linewidths=0.4,
    )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Encoding states (n={len(encoding_raw_states)})")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    im2 = ax.imshow(decoded_map, cmap="viridis", **imshow_kwargs)
    ax.set_title("Decoded reward")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[3]
    if policy_positions is not None and len(policy_positions) > 0:
        ax.plot(
            policy_positions[:, 0],
            policy_positions[:, 1],
            "o-",
            markersize=1.5,
            linewidth=0.8,
            color="tab:red",
            alpha=0.8,
        )
        ax.set_title("Policy behavior")
    else:
        ax.text(
            0.5,
            0.5,
            "policy rollout\nunavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title("Policy behavior")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax = axes[4]
    im4 = ax.imshow(value_map, cmap="plasma", **imshow_kwargs)
    ax.set_title("Predicted V(s, z)")
    fig.colorbar(im4, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(f"FRE AntMaze visualization: {task_name}", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved visualization to %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize FRE on AntMaze (Figure 3)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to FRE agent checkpoint (.pt)")
    parser.add_argument("--domain", type=str, default="antmaze", help="Domain identifier")
    parser.add_argument("--dataset_name", type=str, default=None, help="D4RL dataset name")
    parser.add_argument("--env_name", type=str, default="antmaze-large-diverse-v2")
    parser.add_argument("--task", type=str, default="ant-goal-reaching", choices=ANTMAZE_TASKS)
    parser.add_argument("--num_examples", type=int, default=32)
    parser.add_argument("--state_pool_size", type=int, default=None)
    parser.add_argument("--grid_resolution", type=int, default=96)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--no_policy_rollout", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)

    dataset_name = args.dataset_name or ("antmaze-large-diverse-v2" if args.domain == "antmaze" else args.domain)
    dataset = load_offline_dataset(args.domain, dataset_name=dataset_name)
    state_pool = build_state_pool(dataset, max_pool_size=args.state_pool_size)
    if state_pool is None or len(state_pool) == 0:
        raise ValueError("Empty state pool; cannot produce AntMaze visualization")

    state_dim = int(getattr(dataset, "state_dim", state_pool.shape[1]))
    action_dim = int(getattr(dataset, "action_dim", 8))
    mean, std = _get_dataset_stats(dataset)

    agent = load_agent(
        args.checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        state_pool=state_pool,
        dataset=dataset,
        device=device,
        latent_dim=args.latent_dim,
    )

    # Downstream task reward.
    reward_fn = make_antmaze_task_reward(args.task)

    # Encode task from a small number of state-reward examples.
    z, raw_encoding_states, encoding_rewards = encode_task_for_visualization(
        agent=agent,
        reward_fn=reward_fn,
        state_pool=state_pool,
        num_examples=args.num_examples,
        mean=mean,
        std=std,
        device=device,
        seed=args.seed,
    )

    # Build an XY slice for reward/value/decoder heatmaps.
    raw_pool = _unnormalize(np.asarray(state_pool, dtype=np.float32), mean, std)
    raw_grid, normalized_grid, extent = make_xy_grid(
        raw_pool, args.grid_resolution, mean, std, state_dim=state_dim
    )

    true_rewards = np.asarray(reward_fn(raw_grid), dtype=np.float32)
    z_batch = z.unsqueeze(0).expand(normalized_grid.shape[0], -1)

    grid_t = _to_tensor(normalized_grid, device)
    with torch.no_grad():
        decoded_t = agent.vae.decode_reward(grid_t, z_batch)
        decoded_rewards = decoded_t.detach().cpu().numpy().reshape(-1)

        value_net = _get_value_network(agent)
        value_t = value_net(grid_t, z_batch)
        values = value_t.detach().cpu().numpy().reshape(-1)

    # Policy rollouts (optional, requires MuJoCo/D4RL).
    policy_positions = None
    if not args.no_policy_rollout:
        policy_fn = build_policy_fn(agent, z, mean=mean, std=std)
        policy_positions = collect_policy_positions(
            policy_fn,
            env_name=args.env_name,
            num_rollouts=args.num_rollouts,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
        )

    output_name = args.output_name or f"visualize_antmaze_{args.task}.png"
    output_path = os.path.join(args.output_dir, output_name)
    build_figure(
        task_name=args.task,
        true_rewards=true_rewards,
        decoded_rewards=decoded_rewards,
        values=values,
        grid_shape=(args.grid_resolution, args.grid_resolution),
        encoding_raw_states=raw_encoding_states,
        encoding_rewards=encoding_rewards,
        policy_positions=policy_positions,
        extent=extent,
        output_path=output_path,
    )

    # Save lightweight metadata alongside the figure for reproducibility.
    meta_path = Path(output_path).with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "domain": args.domain,
                "dataset_name": dataset_name,
                "task": args.task,
                "num_examples": args.num_examples,
                "grid_resolution": args.grid_resolution,
                "latent_dim": args.latent_dim,
                "policy_rollout_available": policy_positions is not None,
                "output_path": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
