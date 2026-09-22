"""Shared helpers for the experiment drivers (checkpoints, JSON I/O)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from rice.envs.registry import ENV_SPECS, make_env
from rice.networks import ActorCritic
from rice.policies import TorchPolicy
from rice.training import evaluate_policy


DEFAULT_ROOT = os.environ.get("RICE_ROOT", ".")


def ckpt_path(name: str, kind: str = "agents", root: Optional[str] = None) -> str:
    root = root or DEFAULT_ROOT
    folder = {
        "agents": "checkpoints/agents",
        "masks": "checkpoints/masks",
        "results": "results",
    }[kind]
    return os.path.join(root, folder, name)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: str) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return str(obj)


def load_agent(env_name: str, path: str, device: str = "cpu") -> ActorCritic:
    """Load a pre-trained target agent (``checkpoints/agents/<env>.pt``)."""
    ckpt = torch.load(path, map_location=device)
    spec = ENV_SPECS[env_name]
    if "state_dict" in ckpt:
        policy = ActorCritic(
            ckpt["obs_dim"],
            ckpt["act_dim"],
            hidden=ckpt.get("hidden", spec.policy_hidden),
            discrete=ckpt.get("discrete", True),
        )
        policy.load_state_dict(ckpt["state_dict"])
    else:
        raise ValueError("unexpected checkpoint format in {}".format(path))
    policy.eval()
    return policy


def clone_policy(policy: ActorCritic) -> ActorCritic:
    import copy

    with torch.no_grad():
        clone = copy.deepcopy(policy)
    return clone


def save_agent_tmp(policy: ActorCritic, env_name: str, seed: int = 0) -> str:
    """Persist a policy that was produced on the fly (e.g. GAIL imitation)."""
    import tempfile

    folder = os.path.join(tempfile.gettempdir(), "rice_agents")
    ensure_dir(folder)
    path = os.path.join(folder, "{}_{}.pt".format(env_name, seed))
    policy.save(path)
    return path


def evaluate_agent(
    env_name: str, policy, n_episodes: int = 10, seed: int = 1234
) -> Dict[str, float]:
    env = make_env(env_name, seed=seed)
    try:
        return evaluate_policy(env, TorchPolicy(policy), n_episodes=n_episodes)
    finally:
        close_env(env)


def close_env(env) -> None:
    try:
        env.close()
    except Exception:  # pragma: no cover - optional simulators
        pass


def mean_std(values: List[float]) -> Dict[str, float]:
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}
