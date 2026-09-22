"""Experiment I -- fidelity and efficiency of the explanation methods."""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, Optional, Sequence

import numpy as np

from rice.envs.registry import ENV_SPECS, make_env
from rice.explanation.mask_io import save_mask_net
from rice.explanation.mask_trainer import (
    MaskActorCritic,
    MaskTrainingConfig,
    MaskTrainingResult,
    train_mask_network,
)
from rice.explanation.state_mask import (
    StateMaskTrainingConfig,
    train_state_mask,
)
from rice.fidelity import (
    FidelityConfig,
    compute_fidelity,
    mask_importance_fn,
    random_importance_fn,
)
from rice.experiments.common import ckpt_path, ensure_dir, save_json
from rice.policies import TorchPolicy
from rice.ppo_core import PPOConfig


def mask_config_for(env_name: str, method: str = "ours", **overrides) -> MaskTrainingConfig:
    """Default mask training configuration of one application (Table 3/4)."""
    spec = ENV_SPECS[env_name]
    ppo = PPOConfig(learning_rate=3e-4, n_steps=spec.horizon, batch_size=64, n_epochs=10)
    ppo.finetune_learning_rate = 3e-4
    kwargs = dict(
        iterations=spec.mask_iterations,
        rollout_length=spec.horizon,
        alpha=spec.alpha,
        reward_scale=spec.mask_reward_scale,
        reward_norm_clip=spec.mask_reward_clip,
        max_samples=spec.mask_samples,
        ppo=ppo,
    )
    kwargs.update(overrides)
    if method == "statemask":
        return StateMaskTrainingConfig(**kwargs)
    return MaskTrainingConfig(**kwargs)


def train_explanation(
    env_name: str,
    agent_path: str,
    method: str = "ours",
    seed: int = 0,
    device: str = "cpu",
    save: bool = True,
    alpha: Optional[float] = None,
    max_samples: Optional[int] = None,
    iterations: Optional[int] = None,
    verbose: bool = True,
) -> MaskTrainingResult:
    """Train one mask network (our method or StateMask) and time it."""
    from rice.experiments.common import load_agent

    spec = ENV_SPECS[env_name]
    overrides = {"seed": seed}
    if alpha is not None:
        overrides["alpha"] = alpha
    if max_samples is not None:
        overrides["max_samples"] = max_samples
    if iterations is not None:
        overrides["iterations"] = iterations
    config = mask_config_for(env_name, method=method, **overrides)

    env = make_env(env_name, seed=seed)
    policy = TorchPolicy(load_agent(env_name, agent_path, device=device), device=device)
    obs_dim = int(np.prod(env.observation_space.shape))
    mask_net = MaskActorCritic(obs_dim, hidden=spec.mask_hidden).to(device)
    trainer = train_mask_network if method == "ours" else train_state_mask
    result = trainer(
        env,
        policy,
        config=config,
        mask_net=mask_net,
        device=device,
        verbose=verbose,
    )
    if save:
        path = ckpt_path(
            "{}_{}_seed{}.pt".format(env_name, method, seed), kind="masks"
        )
        save_mask_net(
            result.mask_net,
            path,
            meta={
                "env": env_name,
                "method": method,
                "alpha": config.alpha,
                "samples": result.samples,
                "wall_time": result.wall_time,
            },
        )
        result.checkpoint = path
    return result


@dataclasses.dataclass
class FidelityExperimentResult:
    env: str
    results: Dict[str, Dict[str, object]]
    efficiency: Dict[str, float]
    config: Dict[str, object]


def run_experiment_i(
    env_name: str,
    agent_path: str,
    explanation_paths: Optional[Dict[str, str]] = None,
    n_trajectories: int = 500,
    k_values: Sequence[float] = (0.1, 0.2, 0.3, 0.4),
    n_seeds: int = 3,
    seed: int = 0,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    train_if_missing: bool = True,
    measure_efficiency: bool = True,
    verbose: bool = True,
) -> FidelityExperimentResult:
    """Fidelity of {Random, StateMask, Ours} + timing of the mask training.

    ``explanation_paths`` maps a method name to a mask checkpoint; when a
    checkpoint is missing and ``train_if_missing`` is set, the mask network is
    trained on the spot (which also yields the efficiency numbers of Table 4).
    """
    from rice.explanation.mask_io import load_mask_meta, load_mask_net
    from rice.experiments.common import load_agent

    spec = ENV_SPECS[env_name]
    explanation_paths = dict(explanation_paths or {})
    efficiency: Dict[str, object] = {}
    mask_results: Dict[str, MaskTrainingResult] = {}
    masks: Dict[str, object] = {}

    for method in ("ours", "statemask"):
        path = explanation_paths.get(method)
        trained_now = False
        if path is None or not os.path.exists(path):
            if not train_if_missing:
                continue
            result = train_explanation(
                env_name, agent_path, method=method, seed=seed, device=device,
                verbose=verbose,
            )
            explanation_paths[method] = getattr(result, "checkpoint", None)
            mask_results[method] = result
            masks[method] = result.mask_net
            trained_now = True
        else:
            masks[method] = load_mask_net(path, hidden=spec.mask_hidden)
            if measure_efficiency:
                meta = load_mask_meta(path)
                if meta:
                    efficiency[method] = {
                        "wall_time": float(meta.get("wall_time", float("nan"))),
                        "samples": int(meta.get("samples", 0)),
                    }
        if measure_efficiency and trained_now:
            efficiency[method] = {
                "wall_time": float(mask_results[method].wall_time),
                "samples": int(mask_results[method].samples),
            }
    if "ours" in efficiency and "statemask" in efficiency:
        efficiency["time_reduction_percent"] = float(
            100.0
            * (
                1.0
                - efficiency["ours"]["wall_time"]
                / max(1e-9, efficiency["statemask"]["wall_time"])
            )
        )

    env = make_env(env_name, seed=seed)
    policy = TorchPolicy(load_agent(env_name, agent_path, device=device), device=device)
    results: Dict[str, Dict[str, object]] = {}
    fidelity_config = FidelityConfig(
        n_trajectories=n_trajectories,
        k_values=tuple(k_values),
        d_max=spec.d_max,
        seed=seed,
    )
    for method in ("random", "statemask", "ours"):
        if method == "random":
            importance_fn = random_importance_fn(seed)
        else:
            if method not in masks:
                continue
            importance_fn = mask_importance_fn(masks[method])
        if verbose:
            print("[exp I] fidelity of {} on {}".format(method, env_name), flush=True)
        results[method] = compute_fidelity(
            env, policy, importance_fn, fidelity_config, n_seeds=n_seeds
        )

    payload = FidelityExperimentResult(
        env=env_name,
        results=results,
        efficiency=efficiency,
        config={
            "n_trajectories": n_trajectories,
            "k_values": list(k_values),
            "n_seeds": n_seeds,
            "d_max": spec.d_max,
        },
    )
    out_dir = ensure_dir(out_dir or ckpt_path("experiment_i", kind="results"))
    save_json(
        {
            "env": payload.env,
            "fidelity": payload.results,
            "efficiency": payload.efficiency,
            "config": payload.config,
        },
        os.path.join(out_dir, "{}_fidelity.json".format(env_name)),
    )
    return payload
