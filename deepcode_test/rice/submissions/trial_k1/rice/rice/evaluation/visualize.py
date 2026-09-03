"""Visualization utilities for RICE experiments.

This module produces trajectory/importance heat-maps and other figures used in the
paper's case studies (MetaDrive lane-switch, malware mutation, MuJoCo critical steps).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from rice.agents.mask_network import MaskNetwork
from rice.agents.target_agent import TargetAgent
from rice.evaluation.evaluate_policy import evaluate_policy


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    _HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    _HAS_MATPLOTLIB = False


try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _ensure_matplotlib() -> None:
    if not _HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib and seaborn are required for visualization. "
            "Install them with: pip install matplotlib seaborn"
        )


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def plot_trajectory_importance_heatmap(
    trajectory: Sequence[Dict[str, Any]],
    mask_net: Optional[MaskNetwork] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Trajectory Importance Heatmap",
    figsize: Tuple[int, int] = (12, 4),
    cmap: str = "YlOrRd",
    show: bool = False,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot a heat-map of per-step criticality scores along a trajectory.

    Lower scores are rendered in yellow, higher scores in red, matching the paper's
    color scheme for critical-state visualization.

    Args:
        trajectory: List of transition dicts with keys ``obs`` and optionally ``mask_score``.
        mask_net: Optional trained mask network used to compute scores when they are absent.
        save_path: If provided, save the figure to this path.
        title: Figure title.
        figsize: Matplotlib figure size.
        cmap: Colormap name (default ``YlOrRd`` for yellow-to-red).
        show: Whether to call ``plt.show()``.

    Returns:
        The matplotlib figure, or ``None`` if matplotlib is unavailable.
    """
    _ensure_matplotlib()

    scores: List[float] = []
    for step in trajectory:
        if "mask_score" in step:
            scores.append(float(step["mask_score"]))
        elif mask_net is not None:
            obs = _to_numpy(step["obs"])
            score = mask_net.predict(obs, deterministic=True)
            scores.append(float(score))
        else:
            scores.append(0.0)

    scores_arr = np.asarray(scores).reshape(1, -1)
    n_steps = len(scores)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        scores_arr,
        cmap=cmap,
        cbar_kws={"label": "Criticality ξ(s)"},
        xticklabels=max(1, n_steps // 10),
        yticklabels=False,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Time step")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_multiple_trajectory_heatmaps(
    trajectories: Sequence[Sequence[Dict[str, Any]]],
    labels: Optional[Sequence[str]] = None,
    mask_net: Optional[MaskNetwork] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Trajectory Importance Comparison",
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "YlOrRd",
    show: bool = False,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot criticality heat-maps for multiple trajectories side-by-side."""
    _ensure_matplotlib()

    n = len(trajectories)
    if n == 0:
        raise ValueError("At least one trajectory is required.")
    labels = labels or [f"Trajectory {i + 1}" for i in range(n)]

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, traj, label in zip(axes, trajectories, labels):
        scores: List[float] = []
        for step in traj:
            if "mask_score" in step:
                scores.append(float(step["mask_score"]))
            elif mask_net is not None:
                obs = _to_numpy(step["obs"])
                scores.append(float(mask_net.predict(obs, deterministic=True)))
            else:
                scores.append(0.0)
        scores_arr = np.asarray(scores).reshape(1, -1)
        sns.heatmap(
            scores_arr,
            cmap=cmap,
            cbar=False,
            xticklabels=max(1, len(scores) // 10),
            yticklabels=False,
            ax=ax,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel("Time step")
        ax.set_title(label)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Add a single colorbar on the right.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.02)
    cbar.set_label("Criticality ξ(s)")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_observation_importance_overlay(
    trajectory: Sequence[Dict[str, Any]],
    feature_names: Optional[Sequence[str]] = None,
    mask_net: Optional[MaskNetwork] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Observation Feature Importance",
    figsize: Tuple[int, int] = (12, 6),
    show: bool = False,
) -> Optional["matplotlib.figure.Figure"]:
    """Visualize observation magnitudes colored by per-step criticality.

    Useful for low-dimensional MuJoCo / malware vector observations.
    """
    _ensure_matplotlib()

    obs_list: List[np.ndarray] = []
    scores: List[float] = []
    for step in trajectory:
        obs_list.append(_to_numpy(step["obs"]).flatten())
        if "mask_score" in step:
            scores.append(float(step["mask_score"]))
        elif mask_net is not None:
            scores.append(float(mask_net.predict(_to_numpy(step["obs"]), deterministic=True)))
        else:
            scores.append(0.0)

    obs_mat = np.stack(obs_list, axis=0)  # (T, D)
    n_features = obs_mat.shape[1]
    feature_names = feature_names or [f"f{i}" for i in range(n_features)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(obs_mat.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel("Time step")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Observation value")

    # Overlay criticality as a red line on top.
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(scores)), scores, color="red", linewidth=2, label="Criticality")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Criticality ξ(s)", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_critical_state_distribution(
    trajectories: Sequence[Sequence[Dict[str, Any]]],
    mask_net: Optional[MaskNetwork] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Distribution of Criticality Scores",
    figsize: Tuple[int, int] = (8, 5),
    show: bool = False,
) -> Optional["matplotlib.figure.Figure"]:
    """Plot a histogram of criticality scores across trajectories."""
    _ensure_matplotlib()

    scores: List[float] = []
    for traj in trajectories:
        for step in traj:
            if "mask_score" in step:
                scores.append(float(step["mask_score"]))
            elif mask_net is not None:
                scores.append(float(mask_net.predict(_to_numpy(step["obs"]), deterministic=True)))
            else:
                scores.append(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(scores, bins=50, range=(0.0, 1.0), color="coral", edgecolor="black")
    ax.set_xlabel("Criticality ξ(s)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def visualize_metadrive_case_study(
    target_agent: TargetAgent,
    env: Any,
    mask_net: MaskNetwork,
    save_dir: Union[str, Path],
    n_episodes: int = 10,
    seed: Optional[int] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """Generate MetaDrive case-study visualizations.

    Collects trajectories, identifies the most critical lane-switch step, and saves
    trajectory heat-maps.

    Args:
        target_agent: Pre-trained driving policy.
        env: MetaDrive-compatible environment.
        mask_net: Trained RICE mask network.
        save_dir: Directory where figures are saved.
        n_episodes: Number of episodes to visualize.
        seed: Optional random seed.
        show: Whether to display figures interactively.

    Returns:
        Dictionary with paths to saved figures and the most critical step info.
    """
    _ensure_matplotlib()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    result = evaluate_policy(
        target_agent,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        collect_trajectories=True,
        seed=seed,
    )
    trajectories = result.get("trajectories", [])

    # Score each trajectory and find the single most critical step across episodes.
    best_step: Optional[Dict[str, Any]] = None
    best_score = -1.0
    scored_trajectories: List[List[Dict[str, Any]]] = []
    for traj in trajectories:
        scored = []
        for step in traj:
            obs = _to_numpy(step["obs"])
            score = float(mask_net.predict(obs, deterministic=True))
            step = dict(step)
            step["mask_score"] = score
            scored.append(step)
            if score > best_score:
                best_score = score
                best_step = step
        scored_trajectories.append(scored)

    heatmap_path = save_dir / "metadrive_trajectory_heatmaps.png"
    plot_multiple_trajectory_heatmaps(
        scored_trajectories[: min(4, len(scored_trajectories))],
        mask_net=None,
        save_path=heatmap_path,
        title="MetaDrive Critical-State Heatmaps",
        show=show,
    )

    dist_path = save_dir / "metadrive_criticality_distribution.png"
    plot_critical_state_distribution(
        scored_trajectories,
        mask_net=None,
        save_path=dist_path,
        title="MetaDrive Criticality Score Distribution",
        show=show,
    )

    out = {
        "heatmap_path": str(heatmap_path),
        "distribution_path": str(dist_path),
        "n_episodes": n_episodes,
        "best_critical_score": float(best_score),
        "best_critical_step": best_step,
    }
    return out


def visualize_malware_case_study(
    trajectories: Sequence[Sequence[Dict[str, Any]]],
    mask_net: MaskNetwork,
    save_dir: Union[str, Path],
    feature_names: Optional[Sequence[str]] = None,
    show: bool = False,
) -> Dict[str, Any]:
    """Generate malware-mutation case-study visualizations.

    Args:
        trajectories: Mutation trajectories from the MalConv environment.
        mask_net: Trained RICE mask network.
        save_dir: Directory where figures are saved.
        feature_names: Optional names for mutation features.
        show: Whether to display figures interactively.

    Returns:
        Dictionary with paths to saved figures.
    """
    _ensure_matplotlib()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scored_trajectories: List[List[Dict[str, Any]]] = []
    for traj in trajectories:
        scored = []
        for step in traj:
            obs = _to_numpy(step["obs"])
            score = float(mask_net.predict(obs, deterministic=True))
            step = dict(step)
            step["mask_score"] = score
            scored.append(step)
        scored_trajectories.append(scored)

    heatmap_path = save_dir / "malware_trajectory_heatmaps.png"
    plot_multiple_trajectory_heatmaps(
        scored_trajectories[: min(6, len(scored_trajectories))],
        labels=[f"Sample {i + 1}" for i in range(min(6, len(scored_trajectories)))],
        mask_net=None,
        save_path=heatmap_path,
        title="Malware Mutation Critical-Step Heatmaps",
        figsize=(14, 4),
        show=show,
    )

    overlay_path = save_dir / "malware_observation_importance.png"
    if scored_trajectories:
        plot_observation_importance_overlay(
            scored_trajectories[0],
            feature_names=feature_names,
            mask_net=None,
            save_path=overlay_path,
            title="Malware Observation Importance Overlay",
            show=show,
        )

    dist_path = save_dir / "malware_criticality_distribution.png"
    plot_critical_state_distribution(
        scored_trajectories,
        mask_net=None,
        save_path=dist_path,
        title="Malware Criticality Score Distribution",
        show=show,
    )

    return {
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "distribution_path": str(dist_path),
    }


def visualize_mujoco_critical_steps(
    trajectories: Sequence[Sequence[Dict[str, Any]]],
    mask_net: MaskNetwork,
    save_dir: Union[str, Path],
    env_id: str = "Hopper-v3",
    show: bool = False,
) -> Dict[str, Any]:
    """Generate MuJoCo trajectory heat-maps highlighting critical steps."""
    _ensure_matplotlib()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    scored_trajectories: List[List[Dict[str, Any]]] = []
    for traj in trajectories:
        scored = []
        for step in traj:
            obs = _to_numpy(step["obs"])
            score = float(mask_net.predict(obs, deterministic=True))
            step = dict(step)
            step["mask_score"] = score
            scored.append(step)
        scored_trajectories.append(scored)

    heatmap_path = save_dir / f"{env_id.replace('/', '_')}_critical_steps.png"
    plot_multiple_trajectory_heatmaps(
        scored_trajectories[: min(4, len(scored_trajectories))],
        mask_net=None,
        save_path=heatmap_path,
        title=f"{env_id} Critical-Step Heatmaps",
        show=show,
    )

    dist_path = save_dir / f"{env_id.replace('/', '_')}_criticality_distribution.png"
    plot_critical_state_distribution(
        scored_trajectories,
        mask_net=None,
        save_path=dist_path,
        title=f"{env_id} Criticality Distribution",
        show=show,
    )

    return {
        "heatmap_path": str(heatmap_path),
        "distribution_path": str(dist_path),
    }


def main() -> None:
    """CLI entry point for quick visualization of a saved trajectory/mask pair."""
    import argparse

    parser = argparse.ArgumentParser(description="RICE visualization tool")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to pickled trajectory list")
    parser.add_argument("--mask", type=str, default=None, help="Path to mask network checkpoint")
    parser.add_argument("--save-dir", type=str, default="results/visualizations", help="Output directory")
    parser.add_argument("--title", type=str, default="Trajectory Importance Heatmap")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _ensure_matplotlib()
    import pickle

    with open(args.trajectory, "rb") as f:
        trajectories = pickle.load(f)

    mask_net = None
    if args.mask is not None:
        from rice.agents.mask_network import load_mask_network
        import gymnasium as gym

        # Infer observation space from the first transition.
        first_obs = _to_numpy(trajectories[0][0]["obs"])
        obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=first_obs.shape,
            dtype=np.float32,
        )
        mask_net = load_mask_network(args.mask, obs_space)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(trajectories[0], dict):
        trajectories = [trajectories]

    plot_multiple_trajectory_heatmaps(
        trajectories[:4],
        mask_net=mask_net,
        save_path=save_dir / "heatmap.png",
        title=args.title,
    )
    plot_critical_state_distribution(
        trajectories,
        mask_net=mask_net,
        save_path=save_dir / "distribution.png",
        title="Criticality Distribution",
    )
    print(f"Saved visualizations to {save_dir}")


if __name__ == "__main__":
    main()
