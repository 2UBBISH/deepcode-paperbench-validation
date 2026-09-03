"""Train pre-trained target agents for every RICE domain.

This module is the main entry point for producing the policies :math:`\\pi`
that RICE later explains and refines.  It wraps the backend-agnostic
``TargetAgent`` training utilities in ``rice.agents.target_agent`` with
per-domain factories, default hyper-parameters, and persistence conventions.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from rice.agents.target_agent import (
    TargetAgent,
    TargetAgentConfig,
    default_cage_config,
    default_malware_config,
    default_metadrive_config,
    default_mujoco_config,
    default_selfish_mining_config,
    evaluate_target_agent,
    train_target_agent_sb3,
)
from rice.envs import (
    make_cage_env,
    make_malware_env,
    make_metadrive_env,
    make_mujoco_env,
    make_selfish_mining_env,
)


# ---------------------------------------------------------------------------
# Domain-specific factories
# ---------------------------------------------------------------------------

def train_mujoco_target(
    env_id: str,
    sparse: bool = False,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    **kwargs: Any,
) -> TargetAgent:
    """Train a target PPO agent on a MuJoCo continuous-control task.

    Parameters
    ----------
    env_id:
        Gymnasium-registered MuJoCo id, e.g. ``Hopper-v3``.
    sparse:
        Whether to use the sparse-reward wrapper variant.
    save_dir:
        Directory where the model, config, and rollout buffer are saved.
    seed:
        Random seed.
    device:
        PyTorch device passed to SB3.
    total_timesteps:
        Optional override of the default training budget.
    **kwargs:
        Overrides for ``TargetAgentConfig`` fields.

    Returns
    -------
    TargetAgent
        The trained target agent wrapper.
    """
    config = default_mujoco_config(env_id=env_id, sparse=sparse)
    config.seed = seed
    config.device = device
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown TargetAgentConfig field: {key}")

    env = make_mujoco_env(env_id, sparse=sparse, seed=seed)
    if save_dir is None:
        suffix = "sparse" if sparse else "dense"
        save_dir = f"results/targets/mujoco/{env_id}/{suffix}"

    agent = train_target_agent_sb3(
        env=env,
        config=config,
        save_dir=save_dir,
        algorithm=config.algorithm,
        policy_type="MlpPolicy",
    )
    return agent


def train_selfish_mining_target(
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    **kwargs: Any,
) -> TargetAgent:
    """Train a target PPO agent on the selfish-mining MDP."""
    config = default_selfish_mining_config()
    config.seed = seed
    config.device = device
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown TargetAgentConfig field: {key}")

    env = make_selfish_mining_env(seed=seed)
    if save_dir is None:
        save_dir = "results/targets/selfish_mining"

    agent = train_target_agent_sb3(
        env=env,
        config=config,
        save_dir=save_dir,
        algorithm=config.algorithm,
        policy_type="MlpPolicy",
    )
    return agent


def train_cage_target(
    trial_length: int = 50,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    **kwargs: Any,
) -> TargetAgent:
    """Train a target PPO blue agent on CAGE Challenge 2.

    The paper averages results over trial lengths 30, 50, and 100.  This
    helper trains a single trial length; callers can invoke it three times.
    """
    config = default_cage_config(trial_length=trial_length)
    config.seed = seed
    config.device = device
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown TargetAgentConfig field: {key}")

    env = make_cage_env(trial_length=trial_length, seed=seed)
    if save_dir is None:
        save_dir = f"results/targets/cage/trial_{trial_length}"

    agent = train_target_agent_sb3(
        env=env,
        config=config,
        save_dir=save_dir,
        algorithm=config.algorithm,
        policy_type="MlpPolicy",
    )
    return agent


def train_metadrive_target(
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    **kwargs: Any,
) -> TargetAgent:
    """Train a target PPO agent on MetaDrive Macro-v1."""
    config = default_metadrive_config()
    config.seed = seed
    config.device = device
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown TargetAgentConfig field: {key}")

    env = make_metadrive_env(seed=seed)
    if save_dir is None:
        save_dir = "results/targets/metadrive"

    agent = train_target_agent_sb3(
        env=env,
        config=config,
        save_dir=save_dir,
        algorithm=config.algorithm,
        policy_type="MlpPolicy",
    )
    return agent


def train_malware_target(
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    total_timesteps: Optional[int] = None,
    reward_scale: float = 1.0,
    **kwargs: Any,
) -> TargetAgent:
    """Train a target PPO agent on the MalConv malware-mutation environment."""
    config = default_malware_config()
    config.seed = seed
    config.device = device
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown TargetAgentConfig field: {key}")

    env = make_malware_env(seed=seed, reward_scale=reward_scale)
    if save_dir is None:
        save_dir = "results/targets/malware"

    agent = train_target_agent_sb3(
        env=env,
        config=config,
        save_dir=save_dir,
        algorithm=config.algorithm,
        policy_type="MlpPolicy",
    )
    return agent


# ---------------------------------------------------------------------------
# Generic dispatcher
# ---------------------------------------------------------------------------

DOMAIN_TRAINERS: Dict[str, Any] = {
    "mujoco": train_mujoco_target,
    "selfish_mining": train_selfish_mining_target,
    "cage": train_cage_target,
    "metadrive": train_metadrive_target,
    "malware": train_malware_target,
}


def train_target_agent(
    domain: str,
    save_dir: Optional[str] = None,
    seed: int = 0,
    device: str = "auto",
    **kwargs: Any,
) -> TargetAgent:
    """Dispatch target-agent training for a given domain.

    Parameters
    ----------
    domain:
        One of ``mujoco``, ``selfish_mining``, ``cage``, ``metadrive``,
        ``malware``.
    save_dir:
        Root directory for checkpoints and rollouts.
    seed:
        Random seed.
    device:
        PyTorch device.
    **kwargs:
        Domain-specific keyword arguments (e.g. ``env_id``, ``sparse``,
        ``trial_length``).

    Returns
    -------
    TargetAgent
        The trained target agent.
    """
    if domain not in DOMAIN_TRAINERS:
        raise ValueError(
            f"Unknown domain {domain!r}. Choose from {list(DOMAIN_TRAINERS)}."
        )
    trainer = DOMAIN_TRAINERS[domain]
    return trainer(save_dir=save_dir, seed=seed, device=device, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a pre-trained target RL agent for RICE."
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=list(DOMAIN_TRAINERS),
        help="Domain to train the target agent on.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Gymnasium env id (required for MuJoCo).",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Use sparse-reward variant (MuJoCo only).",
    )
    parser.add_argument(
        "--trial-length",
        type=int,
        default=50,
        help="CAGE trial length (CAGE only).",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Malware reward scale (malware only).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the trained agent.",
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
        help="Override default training budget.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    kwargs: Dict[str, Any] = {}
    if args.domain == "mujoco":
        if args.env_id is None:
            raise ValueError("--env-id is required for MuJoCo domain.")
        kwargs["env_id"] = args.env_id
        kwargs["sparse"] = args.sparse
    elif args.domain == "cage":
        kwargs["trial_length"] = args.trial_length
    elif args.domain == "malware":
        kwargs["reward_scale"] = args.reward_scale

    agent = train_target_agent(
        domain=args.domain,
        save_dir=args.save_dir,
        seed=args.seed,
        device=args.device,
        total_timesteps=args.total_timesteps,
        **kwargs,
    )

    env = agent.env
    mean_return, std_return = evaluate_target_agent(
        agent, env=env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    print(
        f"Target agent trained on {args.domain}: "
        f"mean_return={mean_return:.3f} +/- {std_return:.3f}"
    )


if __name__ == "__main__":
    main()
