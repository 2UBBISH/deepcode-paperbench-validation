"""Domain-specific MaskNet training orchestrator for RICE.

This module provides high-level trainers that:
  1. Load a pre-trained target agent for a given domain.
  2. Construct the domain environment and a MaskNetwork.
  3. Train the mask with the perturbed-policy formulation and blinding reward.
  4. Extract top-critical states and save them to a replay buffer for refining.

The low-level mask training loop lives in :mod:`rice.agents.mask_network`;
this file adds per-domain defaults and a unified CLI entry point.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym

from rice.agents.mask_network import (
    MaskNetwork,
    MaskTrainingConfig,
    collect_masked_rollouts,
    default_mask_config,
    extract_critical_states,
    make_mask_network,
    train_mask_network,
)
from rice.agents.target_agent import TargetAgent, default_mujoco_config
from rice.envs import (
    make_cage_env,
    make_malware_env,
    make_metadrive_env,
    make_mujoco_env,
    make_selfish_mining_env,
)
from rice.envs.resettable_env import CriticalStateBuffer


def _resolve_target_agent(
    target_agent: Optional[TargetAgent],
    target_path: Optional[str],
    env: gym.Env,
) -> TargetAgent:
    """Return a usable TargetAgent, loading from disk if necessary."""
    if target_agent is not None:
        return target_agent
    if target_path is None:
        raise ValueError("Either target_agent or target_path must be provided.")
    return TargetAgent.load(target_path, env=env)


def _save_mask_artifacts(
    mask_net: MaskNetwork,
    critical_buffer: CriticalStateBuffer,
    save_dir: Path,
    trajectories: Optional[Any] = None,
) -> None:
    """Persist mask network weights and critical-state buffer."""
    save_dir.mkdir(parents=True, exist_ok=True)
    mask_path = save_dir / "mask_net.pt"
    mask_net.save(mask_path)
    buffer_path = save_dir / "critical_buffer.pkl"
    critical_buffer.save(buffer_path)
    if trajectories is not None:
        traj_path = save_dir / "trajectories.pkl"
        try:
            import pickle

            with open(traj_path, "wb") as f:
                pickle.dump(trajectories, f)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Could not save trajectories: {exc}")


def train_mujoco_mask(
    env_id: str,
    target_agent: Optional[TargetAgent] = None,
    target_path: Optional[str] = None,
    sparse: bool = False,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train MaskNet for a MuJoCo continuous-control task.

    Parameters
    ----------
    env_id:
        Base Gymnasium MuJoCo env id, e.g. ``Hopper-v3``.
    target_agent:
        Pre-trained target agent (optional if ``target_path`` is given).
    target_path:
        Path to a saved :class:`~rice.agents.target_agent.TargetAgent`.
    sparse:
        Whether to use the sparse-reward variant of the task.
    save_dir:
        Directory where the mask checkpoint and critical buffer are saved.
    seed:
        Random seed.
    device:
        PyTorch device for the mask network.
    total_timesteps:
        Mask-training budget. Defaults to the domain config.
    top_k:
        Number of top-critical states to retain. If ``None``, a percentile
        threshold is used.
    percentile:
        Percentile (0-100) of critical states to retain when ``top_k`` is
        ``None``.
    kwargs:
        Overrides for :class:`~rice.agents.mask_network.MaskTrainingConfig`.

    Returns
    -------
    Dictionary with keys ``mask_net``, ``model`` (SB3 PPO), ``buffer``,
    ``trajectories``, ``save_dir``.
    """
    env = make_mujoco_env(env_id, sparse=sparse, seed=seed)
    target = _resolve_target_agent(target_agent, target_path, env)

    config = default_mask_config(domain="mujoco")
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown MaskTrainingConfig field: {key}")
    config.seed = seed
    config.device = device

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )

    mask_net, model = train_mask_network(
        env,
        target,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir) if save_dir else None,
    )

    trajectories = collect_masked_rollouts(
        env,
        target,
        mask_net,
        n_episodes=kwargs.get("n_eval_episodes", 50),
        alpha=config.alpha,
        deterministic_target=True,
    )
    critical_states = extract_critical_states(
        trajectories,
        top_k=top_k,
        percentile=percentile,
        include_simulator_state=True,
    )

    buffer = CriticalStateBuffer(capacity=kwargs.get("buffer_capacity", len(critical_states) + 1))
    for state in critical_states:
        buffer.add(state)

    if save_dir is not None:
        _save_mask_artifacts(mask_net, buffer, Path(save_dir), trajectories)

    return {
        "mask_net": mask_net,
        "model": model,
        "buffer": buffer,
        "trajectories": trajectories,
        "save_dir": save_dir,
    }


def train_selfish_mining_mask(
    target_agent: Optional[TargetAgent] = None,
    target_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train MaskNet for the selfish-mining domain."""
    env = make_selfish_mining_env(seed=seed)
    target = _resolve_target_agent(target_agent, target_path, env)

    config = default_mask_config(domain="selfish_mining")
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed
    config.device = device

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )
    mask_net, model = train_mask_network(
        env,
        target,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir) if save_dir else None,
    )

    trajectories = collect_masked_rollouts(
        env,
        target,
        mask_net,
        n_episodes=kwargs.get("n_eval_episodes", 50),
        alpha=config.alpha,
        deterministic_target=True,
    )
    critical_states = extract_critical_states(
        trajectories,
        top_k=top_k,
        percentile=percentile,
        include_simulator_state=True,
    )
    buffer = CriticalStateBuffer(capacity=kwargs.get("buffer_capacity", len(critical_states) + 1))
    for state in critical_states:
        buffer.add(state)

    if save_dir is not None:
        _save_mask_artifacts(mask_net, buffer, Path(save_dir), trajectories)

    return {
        "mask_net": mask_net,
        "model": model,
        "buffer": buffer,
        "trajectories": trajectories,
        "save_dir": save_dir,
    }


def train_cage_mask(
    trial_length: int = 50,
    target_agent: Optional[TargetAgent] = None,
    target_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train MaskNet for the CAGE Challenge 2 cyber-defense domain."""
    env = make_cage_env(trial_length=trial_length, seed=seed)
    target = _resolve_target_agent(target_agent, target_path, env)

    config = default_mask_config(domain="cage")
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed
    config.device = device

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )
    mask_net, model = train_mask_network(
        env,
        target,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir) if save_dir else None,
    )

    trajectories = collect_masked_rollouts(
        env,
        target,
        mask_net,
        n_episodes=kwargs.get("n_eval_episodes", 50),
        alpha=config.alpha,
        deterministic_target=True,
    )
    critical_states = extract_critical_states(
        trajectories,
        top_k=top_k,
        percentile=percentile,
        include_simulator_state=True,
    )
    buffer = CriticalStateBuffer(capacity=kwargs.get("buffer_capacity", len(critical_states) + 1))
    for state in critical_states:
        buffer.add(state)

    if save_dir is not None:
        _save_mask_artifacts(mask_net, buffer, Path(save_dir), trajectories)

    return {
        "mask_net": mask_net,
        "model": model,
        "buffer": buffer,
        "trajectories": trajectories,
        "save_dir": save_dir,
    }


def train_metadrive_mask(
    target_agent: Optional[TargetAgent] = None,
    target_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train MaskNet for the MetaDrive autonomous-driving domain."""
    env = make_metadrive_env(seed=seed)
    target = _resolve_target_agent(target_agent, target_path, env)

    config = default_mask_config(domain="metadrive")
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed
    config.device = device

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )
    mask_net, model = train_mask_network(
        env,
        target,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir) if save_dir else None,
    )

    trajectories = collect_masked_rollouts(
        env,
        target,
        mask_net,
        n_episodes=kwargs.get("n_eval_episodes", 50),
        alpha=config.alpha,
        deterministic_target=True,
    )
    critical_states = extract_critical_states(
        trajectories,
        top_k=top_k,
        percentile=percentile,
        include_simulator_state=True,
    )
    buffer = CriticalStateBuffer(capacity=kwargs.get("buffer_capacity", len(critical_states) + 1))
    for state in critical_states:
        buffer.add(state)

    if save_dir is not None:
        _save_mask_artifacts(mask_net, buffer, Path(save_dir), trajectories)

    return {
        "mask_net": mask_net,
        "model": model,
        "buffer": buffer,
        "trajectories": trajectories,
        "save_dir": save_dir,
    }


def train_malware_mask(
    target_agent: Optional[TargetAgent] = None,
    target_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
    reward_scale: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train MaskNet for the malware-mutation domain."""
    env = make_malware_env(seed=seed, reward_scale=reward_scale)
    target = _resolve_target_agent(target_agent, target_path, env)

    config = default_mask_config(domain="malware")
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    config.seed = seed
    config.device = device

    mask_net = make_mask_network(
        env.observation_space,
        env.action_space,
        config=config,
    )
    mask_net, model = train_mask_network(
        env,
        target,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir) if save_dir else None,
    )

    trajectories = collect_masked_rollouts(
        env,
        target,
        mask_net,
        n_episodes=kwargs.get("n_eval_episodes", 50),
        alpha=config.alpha,
        deterministic_target=True,
    )
    critical_states = extract_critical_states(
        trajectories,
        top_k=top_k,
        percentile=percentile,
        include_simulator_state=True,
    )
    buffer = CriticalStateBuffer(capacity=kwargs.get("buffer_capacity", len(critical_states) + 1))
    for state in critical_states:
        buffer.add(state)

    if save_dir is not None:
        _save_mask_artifacts(mask_net, buffer, Path(save_dir), trajectories)

    return {
        "mask_net": mask_net,
        "model": model,
        "buffer": buffer,
        "trajectories": trajectories,
        "save_dir": save_dir,
    }


DOMAIN_MASK_TRAINERS = {
    "mujoco": train_mujoco_mask,
    "selfish_mining": train_selfish_mining_mask,
    "cage": train_cage_mask,
    "metadrive": train_metadrive_mask,
    "malware": train_malware_mask,
}


def train_mask(
    domain: str,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Dispatch mask training to the appropriate domain trainer.

    Parameters
    ----------
    domain:
        One of ``mujoco``, ``selfish_mining``, ``cage``, ``metadrive``,
        ``malware``.
    save_dir:
        Directory for mask checkpoint and critical-state buffer.
    seed:
        Random seed.
    device:
        PyTorch device.
    kwargs:
        Domain-specific keyword arguments forwarded to the trainer.

    Returns
    -------
    The dictionary returned by the domain trainer.
    """
    if domain not in DOMAIN_MASK_TRAINERS:
        raise ValueError(
            f"Unknown domain '{domain}'. Supported: {list(DOMAIN_MASK_TRAINERS.keys())}"
        )

    if save_dir is None:
        save_dir = f"results/masks/{domain}/seed_{seed}"

    return DOMAIN_MASK_TRAINERS[domain](
        save_dir=save_dir,
        seed=seed,
        device=device,
        **kwargs,
    )


def main() -> None:
    """Command-line entry point for training a MaskNet."""
    parser = argparse.ArgumentParser(description="Train a RICE MaskNet.")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=list(DOMAIN_MASK_TRAINERS.keys()),
        help="Domain to train the mask on.",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Path to a saved target agent checkpoint.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Hopper-v3",
        help="(MuJoCo only) base Gymnasium env id.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="(MuJoCo only) use sparse-reward variant.",
    )
    parser.add_argument(
        "--trial-length",
        type=int,
        default=50,
        help="(CAGE only) trial length.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save mask checkpoint and critical buffer.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Mask-training sample budget.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top-critical states to store.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile of critical states to store (alternative to top-k).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Blinding coefficient for mask training.",
    )
    args = parser.parse_args()

    kwargs: Dict[str, Any] = {
        "target_path": args.target_path,
        "seed": args.seed,
        "device": args.device,
        "total_timesteps": args.total_timesteps,
        "top_k": args.top_k,
        "percentile": args.percentile,
    }
    if args.alpha is not None:
        kwargs["alpha"] = args.alpha

    if args.domain == "mujoco":
        kwargs["env_id"] = args.env_id
        kwargs["sparse"] = args.sparse
    elif args.domain == "cage":
        kwargs["trial_length"] = args.trial_length

    result = train_mask(args.domain, save_dir=args.save_dir, **kwargs)
    print(f"Mask training complete. Artifacts saved to: {result['save_dir']}")
    print(f"Critical states stored: {len(result['buffer'])}")


if __name__ == "__main__":
    main()
