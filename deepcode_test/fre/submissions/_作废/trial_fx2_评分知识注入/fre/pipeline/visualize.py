"""Visualization utilities for FRE.

This module reproduces the qualitative visualizations described in the paper,
in particular the AntMaze reward/value/trajectory panels (Figure 3) and the
aggregated bar plots used for the reward-prior scaling and domain-prior
augmentation ablations (Figures 5 and 6).

Heavy dependencies (matplotlib, gym/d4rl/dm_control) are imported lazily so the
module can always be imported as part of ``fre.pipeline``.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from fre.config import Config, get_config, resolve_device
from fre.data.dataset import OfflineDataset
from fre.modeling.fre_vae import FREVAE
from fre.rl.iql import IQL, ImplicitQLearning

logger = logging.getLogger(__name__)

__all__ = [
    "visualize_antmaze_task",
    "plot_bar_comparison",
    "plot_grouped_bar_comparison",
    "build_parser",
    "main",
]


# ---------------------------------------------------------------------------
# Lazy imports and helpers
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)  # noqa
    import matplotlib.pyplot as plt
    return plt


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Dict or attribute access with a default value."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# AntMaze grid construction
# ---------------------------------------------------------------------------
def _get_antmaze_bounds(env: Any = None, default: Tuple[float, float, float, float] = (0.0, 24.0, 0.0, 24.0)) -> Tuple[float, float, float, float]:
    """Return ``(xmin, xmax, ymin, ymax)`` for an AntMaze figure."""
    try:
        from fre.envs.antmaze import get_antmaze_bounds
        return get_antmaze_bounds(env=env)
    except Exception:
        return default


def _build_antmaze_grid(
    dataset: OfflineDataset,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    n: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an ``n x n`` grid over AntMaze XY coordinates.

    Returns
    -------
    raw_states : (n*n, state_dim) raw observation grid with non-XY dims set to
        the dataset raw mean.
    norm_states : (n*n, state_dim) normalized states expected by the model.
    xs, ys : 1-D coordinate arrays used for contour plotting.
    """
    xmin, xmax, ymin, ymax = bounds if bounds is not None else _get_antmaze_bounds()
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)

    state_dim = dataset.states.shape[1]
    raw_states = np.tile(dataset.state_mean.reshape(1, -1), (grid_xy.shape[0], 1))
    if raw_states.shape[1] >= 2:
        raw_states[:, 0] = grid_xy[:, 0]
        raw_states[:, 1] = grid_xy[:, 1]
    else:
        # Degenerate state space: use XY directly.
        raw_states = grid_xy.copy()

    if hasattr(dataset, "normalize_states"):
        norm_states = dataset.normalize_states(raw_states)
    else:
        norm_states = (raw_states - dataset.state_mean) / (dataset.state_std + 1e-6)
    return raw_states, norm_states.astype(np.float32), xs, ys


def _rollout_trajectory(
    env: Any,
    dataset: OfflineDataset,
    agent: ImplicitQLearning,
    z: torch.Tensor,
    device: torch.device,
    max_steps: int = 1000,
    deterministic: bool = True,
    seed: int = 0,
) -> List[np.ndarray]:
    """Collect raw states from a single evaluation trajectory."""
    try:
        env.seed(seed)
    except Exception:
        pass
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.asarray(obs, dtype=np.float32)

    states: List[np.ndarray] = [obs.copy()]
    done = False
    step = 0
    while not done and step < max_steps:
        norm_obs = dataset.normalize_states(obs) if hasattr(dataset, "normalize_states") else obs
        state_t = torch.as_tensor(norm_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(state_t, condition=z, deterministic=deterministic)
        action = _as_numpy(action).reshape(-1)
        result = env.step(action)
        # Gym 0.23 returns (obs, r, done, info); Gym 0.26 returns 5-tuple.
        if len(result) == 5:
            obs, _, terminated, truncated, _ = result
            done = bool(terminated or truncated)
        else:
            obs, _, done, _ = result
            done = bool(done)
        obs = np.asarray(obs, dtype=np.float32)
        states.append(obs.copy())
        step += 1
    return states


def _plot_antmaze_background(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")


# ---------------------------------------------------------------------------
# Figure 3 style AntMaze visualization
# ---------------------------------------------------------------------------
def visualize_antmaze_task(
    cfg: Optional[Config] = None,
    dataset: Optional[OfflineDataset] = None,
    model: Optional[FREVAE] = None,
    agent: Optional[ImplicitQLearning] = None,
    task_name: str = "ant-goal-reaching",
    env: Optional[Any] = None,
    device: Optional[torch.device] = None,
    num_reward_samples: int = 32,
    grid_n: int = 64,
    seed: int = 0,
    max_episode_steps: int = 1000,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (20, 10),
) -> Any:
    """Generate the AntMaze qualitative panel analogous to Figure 3.

    Panels show the true task reward, sampled reward-context states, the reward
    decoded by the FRE decoder, the FRE-conditioned value function, and a policy
    trajectory. All panels share the same XY axes.
    """
    if dataset is None:
        raise ValueError("dataset is required for AntMaze visualization")

    if cfg is None:
        cfg = Config.default()
    if device is None:
        device = resolve_device(cfg.device if hasattr(cfg, "device") else "auto")

    # Lazy imports from evaluate for task reward and latent encoding.
    from fre.pipeline.evaluate import encode_task_latent, make_eval_env, make_task_reward

    if model is None or agent is None:
        raise ValueError("model and agent must be provided")

    model = model.to(device).eval()
    agent = agent.to(device)

    task_reward = make_task_reward(task_name, dataset, device, seed=seed)
    z = encode_task_latent(
        dataset,
        model,
        task_reward,
        num_reward_samples=num_reward_samples,
        device=device,
        seed=seed,
    )

    bounds = _get_antmaze_bounds(env=env)
    raw_states, norm_states, xs, ys = _build_antmaze_grid(dataset, bounds=bounds, n=grid_n)
    norm_t = torch.as_tensor(norm_states, dtype=torch.float32, device=device)

    with torch.no_grad():
        true_r = _as_numpy(task_reward(norm_t)).reshape(grid_n, grid_n)
        decoded_r = _as_numpy(model.decode(norm_t, z)).reshape(grid_n, grid_n)
        try:
            values = _as_numpy(agent.value(norm_t, condition=z)).reshape(grid_n, grid_n)
        except Exception:
            values = None

    # Sample the same context states used by encode_task_latent for plotting.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset.states), size=min(num_reward_samples, len(dataset.states)), replace=False)
    context_raw = dataset.unnormalize_states(dataset.states[idx]) if hasattr(dataset, "unnormalize_states") else dataset.states[idx].cpu().numpy()
    context_xy = context_raw[:, :2]

    # Collect a policy trajectory if a live environment is available.
    if env is None:
        try:
            env = make_eval_env(cfg, dataset)
        except Exception as e:
            logger.warning("Could not create eval environment for trajectory: %s", e)
    trajectory_states: List[np.ndarray] = []
    if env is not None:
        try:
            trajectory_states = _rollout_trajectory(
                env,
                dataset,
                agent,
                z,
                device,
                max_steps=max_episode_steps,
                deterministic=True,
                seed=seed,
            )
        except Exception as e:
            logger.warning("Policy trajectory collection failed: %s", e)

    plt = _plt()
    n_panels = 5 if values is not None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: true reward.
    ax = axes[0]
    im = ax.contourf(xs, ys, true_r, levels=32, cmap="viridis")
    fig.colorbar(im, ax=ax)
    _plot_antmaze_background(ax, f"True reward: {task_name}")

    # Panel 2: sampled context states.
    ax = axes[1]
    ax.scatter(context_xy[:, 0], context_xy[:, 1], c="red", s=30, marker="o", label="context states")
    _plot_antmaze_background(ax, "Sampled context states")
    ax.legend(loc="best")

    # Panel 3: decoded reward.
    ax = axes[2]
    im = ax.contourf(xs, ys, decoded_r, levels=32, cmap="plasma")
    fig.colorbar(im, ax=ax)
    _plot_antmaze_background(ax, "Decoded reward")

    # Panel 4: value function.
    panel_idx = 3
    if values is not None:
        ax = axes[panel_idx]
        im = ax.contourf(xs, ys, values, levels=32, cmap="cividis")
        fig.colorbar(im, ax=ax)
        _plot_antmaze_background(ax, "Predicted value")
        panel_idx += 1

    # Panel 5 (or 4): policy trajectory.
    ax = axes[panel_idx]
    # Background as decoded reward or maze outline.
    ax.contourf(xs, ys, decoded_r, levels=32, cmap="plasma", alpha=0.2)
    if trajectory_states:
        traj = np.stack([s[:2] for s in trajectory_states], axis=0)
        ax.plot(traj[:, 0], traj[:, 1], "-o", color="black", linewidth=2, markersize=3)
        ax.scatter(traj[0, 0], traj[0, 1], color="green", s=80, marker="*", label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], color="blue", s=80, marker="*", label="end")
    _plot_antmaze_background(ax, "Policy trajectory")
    ax.legend(loc="best")

    fig.suptitle(f"FRE AntMaze visualization: {task_name}")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved AntMaze visualization to %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Bar plots
# ---------------------------------------------------------------------------
def plot_bar_comparison(
    results: Mapping[str, Tuple[float, float]],
    title: str = "FRE evaluation",
    xlabel: str = "Task",
    ylabel: str = "Normalized score",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (12, 6),
) -> Any:
    """Plot a single-method bar chart with standard-deviation error bars.

    ``results`` maps task names to ``(mean, std)`` tuples.
    """
    plt = _plt()
    names = list(results.keys())
    means = np.array([results[n][0] for n in names], dtype=float)
    stds = np.array([results[n][1] if len(results[n]) > 1 else 0.0 for n in names], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved bar plot to %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_grouped_bar_comparison(
    results_by_method: Mapping[str, Mapping[str, Tuple[float, float]]],
    title: str = "Baseline comparison",
    xlabel: str = "Task",
    ylabel: str = "Normalized score",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: Tuple[float, float] = (14, 7),
) -> Any:
    """Plot a grouped bar chart comparing multiple methods across tasks."""
    plt = _plt()
    methods = list(results_by_method.keys())
    # Use union of task names in method order, falling back to first method's order.
    task_names: List[str] = []
    for m in methods:
        for t in results_by_method[m]:
            if t not in task_names:
                task_names.append(t)

    x = np.arange(len(task_names))
    width = 0.8 / max(1, len(methods))
    fig, ax = plt.subplots(figsize=figsize)

    for i, method in enumerate(methods):
        means = []
        stds = []
        for t in task_names:
            val = results_by_method[method].get(t)
            if val is None:
                means.append(0.0)
                stds.append(0.0)
            else:
                means.append(float(val[0]))
                stds.append(float(val[1]) if len(val) > 1 else 0.0)
        offset = (i - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=3, label=method, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved grouped bar plot to %s", save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FRE visualization utilities")
    subparsers = parser.add_subparsers(dest="command")

    ant_parser = subparsers.add_parser("antmaze", help="Visualize an AntMaze task")
    ant_parser.add_argument("--config", default="antmaze", help="Config name or YAML path")
    ant_parser.add_argument("--task", default="ant-goal-reaching")
    ant_parser.add_argument("--dataset", default=None, help="Dataset path (optional)")
    ant_parser.add_argument("--model-path", required=True, help="Path to pretrained FRE VAE checkpoint")
    ant_parser.add_argument("--agent-path", required=True, help="Path to trained IQL agent checkpoint")
    ant_parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory")
    ant_parser.add_argument("--num-reward-samples", type=int, default=32)
    ant_parser.add_argument("--grid-n", type=int, default=64)
    ant_parser.add_argument("--seed", type=int, default=0)
    ant_parser.add_argument("--max-episode-steps", type=int, default=1000)
    ant_parser.add_argument("--save-path", default=None)
    ant_parser.add_argument("--show", action="store_true")
    ant_parser.add_argument("--device", default="auto")
    ant_parser.add_argument("--override", action="append", default=[], help="key=value config overrides")

    bar_parser = subparsers.add_parser("bar", help="Plot bar comparison")
    bar_parser.add_argument("--results", required=True, help="JSON file mapping method->task->[mean, std]")
    bar_parser.add_argument("--title", default="FRE evaluation")
    bar_parser.add_argument("--xlabel", default="Task")
    bar_parser.add_argument("--ylabel", default="Normalized score")
    bar_parser.add_argument("--save-path", default=None)
    bar_parser.add_argument("--show", action="store_true")
    bar_parser.add_argument("--grouped", action="store_true", help="Grouped multiple methods")
    return parser


def _parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        # Primitive type conversion.
        if v.lower() == "true":
            v = True
        elif v.lower() == "false":
            v = False
        elif v.replace(".", "", 1).isdigit() and "." in v:
            v = float(v)
        elif v.isdigit():
            v = int(v)
        result[k] = v
    return result


def _load_dataset(cfg: Config, device: torch.device) -> OfflineDataset:
    domain = getattr(cfg, "domain", "antmaze")
    if domain == "antmaze":
        from fre.data.d4rl_loader import load_antmaze_dataset
        return load_antmaze_dataset(cfg.data, device=str(device))
    if domain == "kitchen":
        from fre.data.d4rl_loader import load_kitchen_dataset
        return load_kitchen_dataset(cfg.data, device=str(device))
    if domain == "exorl":
        from fre.data.exorl_loader import load_exorl_dataset
        return load_exorl_dataset(cfg.data, device=str(device))
    from fre.data.d4rl_loader import load_d4rl_dataset
    return load_d4rl_dataset(cfg.data, device=str(device))


def _load_model(path: str, cfg: Config, dataset: OfflineDataset, device: torch.device) -> FREVAE:
    state_dim = dataset.states.shape[1]
    model = FREVAE.from_config(cfg.fre, state_dim=state_dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _load_agent(path: str, cfg: Config, dataset: OfflineDataset, device: torch.device) -> ImplicitQLearning:
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    z_dim = getattr(cfg.fre, "z_dim", 64)
    agent = IQL(
        state_dim=state_dim,
        action_dim=action_dim,
        condition_dim=z_dim,
        cfg=cfg.iql,
        device=str(device),
    )
    agent.load(path)
    return agent


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "bar":
        import json

        with open(args.results, "r") as f:
            raw = json.load(f)
        # Accept two JSON layouts: {task: [mean, std]} or {method: {task: [mean, std]}}.
        if args.grouped or any(isinstance(v, dict) for v in raw.values()):
            plot_grouped_bar_comparison(
                raw,
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                save_path=args.save_path,
                show=args.show,
            )
        else:
            plot_bar_comparison(
                raw,
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                save_path=args.save_path,
                show=args.show,
            )
        return

    if args.command != "antmaze":
        parser.print_help()
        return

    overrides = _parse_overrides(args.override)
    cfg = get_config(args.config, **overrides)
    device = resolve_device(args.device)

    dataset = _load_dataset(cfg, device)
    model = _load_model(args.model_path, cfg, dataset, device)
    agent = _load_agent(args.agent_path, cfg, dataset, device)

    visualize_antmaze_task(
        cfg=cfg,
        dataset=dataset,
        model=model,
        agent=agent,
        task_name=args.task,
        device=device,
        num_reward_samples=args.num_reward_samples,
        grid_n=args.grid_n,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
        save_path=args.save_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
