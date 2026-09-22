"""Experiment IV -- refining a pre-trained agent of another algorithm.

The paper trains a SAC agent in Hopper, imitates it with GAIL so that a PPO
compatible policy network is available, and then compares the refining methods
on that imitated policy.  Fine-tuning the SAC agent with SAC is included as an
additional baseline (Figure 3).
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from rice.envs.registry import ENV_SPECS, make_env
from rice.experiments.common import (
    ckpt_path,
    close_env,
    ensure_dir,
    save_json,
)
from rice.experiments.refining import default_refine_config, run_single_refine
from rice.imitation import GAILConfig, train_gail
from rice.networks import ActorCritic
from rice.policies import SB3Policy, TorchPolicy
from rice.refining.methods import make_eval_env


@dataclasses.dataclass
class SACExperimentConfig:
    env_name: str = "Hopper"
    sac_steps: int = 200_000
    gail_steps: int = 200_000
    refine_steps: int = 200_000
    sac_finetune_steps: int = 100_000
    #: budget used to train a mask network for the imitated policy when no
    #: checkpoint is supplied (the paper trains it with the normal budget)
    mask_samples: int = 50_000
    mask_iterations: int = 150
    seeds: Sequence[int] = (0, 1, 2)
    device: str = "cpu"
    eval_episodes: int = 10


def train_sac_agent(config: SACExperimentConfig, seed: int):
    """Pre-train the SAC agent of Experiment IV."""
    from rice.training import train_sb3_agent

    return train_sb3_agent(
        config.env_name, algo="sac", total_steps=config.sac_steps, seed=seed
    )


def imitate_sac_with_gail(
    config: SACExperimentConfig,
    sac_model,
    seed: int,
    save_path: Optional[str] = None,
):
    """GAIL imitation of the SAC agent (Section 4.1, Experiment IV)."""
    spec = ENV_SPECS[config.env_name]
    env = make_env(config.env_name, seed=seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    policy = ActorCritic(
        obs_dim, act_dim, hidden=spec.policy_hidden, discrete=False
    )
    expert = SB3Policy(sac_model)
    gail_config = GAILConfig(
        total_steps=config.gail_steps,
        seed=seed,
        env_reward_coef=0.0,
    )
    out = train_gail(env, expert, policy, gail_config, device=config.device)
    close_env(env)
    if save_path:
        ensure_dir(os.path.dirname(os.path.abspath(save_path)))
        out["policy"].save(save_path)
    return out


def sac_finetune(model, env_name: str, steps: int, learning_rate: float = 1e-4):
    """Baseline: keep training the SAC agent with the SAC algorithm."""
    model.learning_rate = learning_rate
    model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
    return model


def run_experiment_iv(
    config: Optional[SACExperimentConfig] = None,
    mask_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    config = config or SACExperimentConfig()
    spec = ENV_SPECS[config.env_name]
    results: Dict[str, Dict[str, List[float]]] = {}
    curves: Dict[str, List[Dict[str, float]]] = {}
    sac_curves: List[Dict[str, float]] = []

    for seed in config.seeds:
        if verbose:
            print("[exp IV] seed {}: pre-training SAC".format(seed), flush=True)
        sac_model = train_sac_agent(config, seed)
        eval_env = make_env(config.env_name, seed=seed + 777)
        sac_before = evaluate_agent_any(eval_env, sac_model, config.eval_episodes)
        results.setdefault("sac_no_refine", []).append(sac_before)

        # ------------------------------------------------ GAIL imitation step
        gail_out = imitate_sac_with_gail(config, sac_model, seed)
        imitated = gail_out["policy"]
        imitated_before = evaluate_agent_any(
            eval_env, TorchPolicy(imitated, device=config.device), config.eval_episodes
        )
        results.setdefault("imitated_no_refine", []).append(imitated_before)

        # --------------------------------------------------- refine the policy
        refine_env = make_env(config.env_name, seed=seed)
        for method in ("ppo", "jsrl", "statemask_r", "ours"):
            mask_net = None
            if mask_path is not None and os.path.exists(mask_path):
                from rice.explanation.mask_io import load_mask_net

                mask_net = load_mask_net(mask_path, hidden=spec.mask_hidden)
            if mask_net is None and method in ("ours", "statemask_r"):
                # a mask network is required; train one quickly on the fly
                from rice.experiments.explanation import train_explanation
                from rice.experiments.common import save_agent_tmp

                agent_tmp = save_agent_tmp(imitated, config.env_name, seed)
                mask_net = train_explanation(
                    config.env_name,
                    agent_tmp,
                    method="ours",
                    seed=seed,
                    device=config.device,
                    max_samples=config.mask_samples,
                    iterations=config.mask_iterations,
                    verbose=False,
                ).mask_net
            refine_config = default_refine_config(
                config.env_name,
                seed=seed,
                total_steps=config.refine_steps,
                normalize_obs=False,
            )
            result = run_single_refine(
                config.env_name,
                imitated,
                mask_net,
                method=method,
                config=refine_config,
                device=config.device,
                eval_env=make_eval_env(config.env_name, seed=seed + 500),
                verbose=verbose,
            )
            results.setdefault(method, []).append(float(result.final_eval))
            curves.setdefault(method, []).append(
                {
                    "seed": float(seed),
                    "steps": [float(h["steps"]) for h in result.eval_history],
                    "returns": [float(h["mean_return"]) for h in result.eval_history],
                }
            )

        # ------------------------------------------------ SAC fine-tuning base
        sac_trained = sac_finetune(
            sac_model, config.env_name, config.sac_finetune_steps
        )
        after = evaluate_agent_any(eval_env, sac_trained, config.eval_episodes)
        results.setdefault("sac_finetune", []).append(after)
        sac_curves.append({"seed": float(seed), "return": float(after)})
        close_env(eval_env)
        close_env(refine_env)

    summary = {
        method: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "per_seed": [float(v) for v in values],
        }
        for method, values in results.items()
    }
    payload = {
        "env": config.env_name,
        "results": summary,
        "curves": curves,
        "config": dataclasses.asdict(config),
    }
    out_dir = ensure_dir(out_dir or ckpt_path("experiment_iv", kind="results"))
    save_json(payload, os.path.join(out_dir, "{}_sac_refining.json".format(config.env_name)))
    return payload


def evaluate_agent_any(env, policy, n_episodes: int = 10) -> float:
    """Evaluate either an SB3 model or a torch policy."""
    if hasattr(policy, "predict"):  # Stable-Baselines3 model
        adapter = SB3Policy(policy)
    elif isinstance(policy, torch.nn.Module):  # our ActorCritic
        adapter = TorchPolicy(policy)
    elif hasattr(policy, "act"):  # already an adapter (TorchPolicy/SB3Policy)
        adapter = policy
    else:
        raise TypeError("unsupported policy type {}".format(type(policy)))
    from rice.training import evaluate_policy

    return float(evaluate_policy(env, adapter, n_episodes=n_episodes)["mean_return"])
