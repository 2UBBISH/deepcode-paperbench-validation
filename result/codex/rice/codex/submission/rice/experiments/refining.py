"""Experiments II and III -- effectiveness of the refining methods.

* Experiment II ("Fix Explanation; Vary Refine Methods"): the explanation is
  always the one produced by our mask network, and the refining method varies
  between PPO fine-tuning, JSRL, StateMask-R and RICE.
* Experiment III ("Fix Refine; Vary Explanation Methods"): the refining method
  is always RICE, and the explanation varies between Random, StateMask and our
  mask network.

Both groups make up Table 1 of the paper; on the sparse MuJoCo games the same
driver produces Figure 2, which reports the performance *during* refining.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from rice.envs.registry import ENV_SPECS, make_env
from rice.experiments.common import (
    ckpt_path,
    clone_policy,
    close_env,
    ensure_dir,
    evaluate_agent,
    load_agent,
    save_json,
)
from rice.explanation.mask_io import load_mask_net
from rice.policies import TorchPolicy
from rice.refining.critical_state_provider import RandomStateProvider
from rice.refining.methods import (
    frozen_copy,
    make_eval_env,
    refine_jsrl,
    refine_ppo_finetune,
    refine_rice,
    refine_statemask_r,
)
from rice.refining.trainer import RefineConfig


REFINE_METHODS = ("ppo", "jsrl", "statemask_r", "ours")
EXPLANATION_METHODS = ("random", "statemask", "ours")


def default_refine_config(
    env_name: str,
    seed: int = 0,
    total_steps: Optional[int] = None,
    p: Optional[float] = None,
    rnd_lambda: Optional[float] = None,
    normalize_obs: bool = False,
) -> RefineConfig:
    spec = ENV_SPECS[env_name]
    steps = int(total_steps or spec.refine_steps)
    config = RefineConfig(
        total_steps=steps,
        n_steps=2048,
        seed=seed,
        p=spec.p if p is None else p,
        rnd_lambda=spec.rnd_lambda if rnd_lambda is None else rnd_lambda,
        rollin_length=spec.rollin_length,
        eval_interval=max(1, steps // 20) if spec.sparse_rewards else max(1, steps // 5),
        eval_episodes=10,
        normalize_obs=normalize_obs,
    )
    return config


def run_single_refine(
    env_name: str,
    policy,
    mask_net,
    method: str = "ours",
    explanation: str = "ours",
    config: Optional[RefineConfig] = None,
    device: str = "cpu",
    eval_env=None,
    verbose: bool = False,
):
    """Refine one copy of ``policy`` and return the :class:`RefineResult`."""
    config = config or default_refine_config(env_name)
    env = make_env(env_name, seed=config.seed)
    policy = clone_policy(policy)
    owns_eval_env = eval_env is None
    if owns_eval_env:
        eval_env = make_eval_env(env_name, seed=config.seed + 500)

    try:
        if method == "ours":
            if explanation != "ours":
                close_env(env)
                return run_rice_with_explanation(
                    env_name,
                    policy,
                    mask_net,
                    explanation=explanation,
                    config=config,
                    device=device,
                    eval_env=eval_env,
                    verbose=verbose,
                )
            return refine_rice(
                env, policy, mask_net, config=config, env_name=env_name,
                device=device, eval_env=eval_env, verbose=verbose,
            )
        if method == "statemask_r":
            return refine_statemask_r(
                env, policy, mask_net, config=config, env_name=env_name,
                device=device, eval_env=eval_env, verbose=verbose,
            )
        if method == "ppo":
            return refine_ppo_finetune(
                env, policy, config=config, env_name=env_name,
                device=device, eval_env=eval_env, verbose=verbose,
            )
        if method == "jsrl":
            return refine_jsrl(
                env, policy, config=config, env_name=env_name,
                device=device, eval_env=eval_env, verbose=verbose,
            )
        raise ValueError("unknown refining method {!r}".format(method))
    finally:
        close_env(env)
        if owns_eval_env:
            close_env(eval_env)


def config_seed(config: Optional[RefineConfig]) -> int:
    return int(config.seed) if config is not None else 0


def run_rice_with_explanation(
    env_name: str,
    policy,
    mask_net,
    explanation: str,
    config: Optional[RefineConfig] = None,
    device: str = "cpu",
    eval_env=None,
    verbose: bool = False,
):
    """RICE refining where the critical states come from ``explanation``."""
    config = config or default_refine_config(env_name)
    env = make_env(env_name, seed=config.seed)
    policy = clone_policy(policy)
    owns_eval_env = eval_env is None
    if owns_eval_env:
        eval_env = make_eval_env(env_name, seed=config.seed + 500)
    rollin_policy = TorchPolicy(frozen_copy(policy), device=device)
    try:
        if explanation == "random":
            provider = RandomStateProvider(
                env=env,
                mask_net=mask_net,
                p=config.p,
                rollin_length=config.rollin_length,
                rollin_policy=rollin_policy,
                seed=config.seed,
            )
            from rice.refining.trainer import RefiningTrainer
            from rice.refining.rnd import RND, RNDRewardShaper

            shaper = None
            if config.rnd_lambda > 0:
                obs_dim = int(np.prod(env.observation_space.shape))
                rnd = RND(obs_dim, hidden=(64, 64), device=device)
                shaper = RNDRewardShaper(rnd, coef=config.rnd_lambda)
            trainer = RefiningTrainer(
                env=env,
                policy=policy,
                config=config,
                method="rice_random_explanation",
                device=device,
                initial_state_provider=provider,
                reward_shaper=shaper,
                eval_env=eval_env,
                verbose=verbose,
            )
            result = trainer.train(config.total_steps)
            result.provider_stats = {
                "critical_resets": provider.n_resets,
                "default_resets": provider.n_default,
            }
            return result
        return refine_rice(
            env,
            policy,
            mask_net,
            config=config,
            env_name=env_name,
            device=device,
            eval_env=eval_env,
            verbose=verbose,
        )
    finally:
        close_env(env)
        if owns_eval_env:
            close_env(eval_env)


@dataclasses.dataclass
class RefiningExperimentResult:
    env: str
    no_refine: Dict[str, float]
    vary_refine: Dict[str, Dict[str, float]]
    vary_explanation: Dict[str, Dict[str, float]]
    curves: Dict[str, List[Dict[str, float]]]
    config: Dict[str, object]


def run_experiment_ii_and_iii(
    env_name: str,
    agent_path: str,
    mask_paths: Dict[str, str],
    total_steps: Optional[int] = None,
    seeds: Sequence[int] = (0, 1, 2),
    methods: Sequence[str] = REFINE_METHODS,
    explanations: Sequence[str] = EXPLANATION_METHODS,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> RefiningExperimentResult:
    """Produce both halves of Table 1 (and Figure 2 for the sparse games)."""
    spec = ENV_SPECS[env_name]
    policy = load_agent(env_name, agent_path, device=device)
    eval_env = make_eval_env(env_name, seed=987)

    no_refine: Dict[str, float] = {"mean": float("nan"), "std": float("nan")}
    if eval_env is not None:
        metrics = evaluate_agent(env_name, policy, n_episodes=10, seed=987)
        no_refine = {
            "mean": float(metrics["mean_return"]),
            "std": float(metrics["std_return"]),
        }

    masks = {
        name: load_mask_net(path, hidden=spec.mask_hidden)
        for name, path in mask_paths.items()
        if path and os.path.exists(path)
    }
    reference_mask = masks.get("ours")

    vary_refine: Dict[str, Dict[str, float]] = {}
    vary_explanation: Dict[str, Dict[str, float]] = {}
    curves: Dict[str, List[Dict[str, float]]] = {}

    # ---------------------------------------------------------- Experiment II
    for method in methods:
        values: List[float] = []
        method_curves: List[Dict[str, float]] = []
        for seed in seeds:
            config = default_refine_config(
                env_name, seed=seed, total_steps=total_steps,
                normalize_obs=env_name in ("Walker2d", "HalfCheetah"),
            )
            eval_env_seed = make_eval_env(env_name, seed=seed + 500)
            try:
                result = run_single_refine(
                    env_name,
                    policy,
                    reference_mask,
                    method=method,
                    config=config,
                    device=device,
                    eval_env=eval_env_seed,
                    verbose=verbose,
                )
            finally:
                close_env(eval_env_seed)
            final = result.final_eval
            if not np.isfinite(final) and result.history:
                final = result.history[-1]["mean_episode_return"]
            values.append(float(final))
            method_curves.append(
                {
                    "seed": float(seed),
                    "steps": [float(h["steps"]) for h in result.eval_history]
                    or [float(h["steps"]) for h in result.history],
                    "returns": [float(h["mean_return"]) for h in result.eval_history]
                    or [float(h["mean_episode_return"]) for h in result.history],
                }
            )
        vary_refine[method] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "per_seed": [float(v) for v in values],
        }
        curves[method] = method_curves

    # --------------------------------------------------------- Experiment III
    for explanation in explanations:
        values = []
        for seed in seeds:
            config = default_refine_config(
                env_name, seed=seed, total_steps=total_steps,
                normalize_obs=env_name in ("Walker2d", "HalfCheetah"),
            )
            mask_for_explanation = masks.get(explanation, reference_mask)
            eval_env_seed = make_eval_env(env_name, seed=seed + 500)
            try:
                result = run_rice_with_explanation(
                    env_name,
                    policy,
                    mask_for_explanation,
                    explanation=explanation,
                    config=config,
                    device=device,
                    eval_env=eval_env_seed,
                    verbose=verbose,
                )
            finally:
                close_env(eval_env_seed)
            final = result.final_eval
            if not np.isfinite(final) and result.history:
                final = result.history[-1]["mean_episode_return"]
            values.append(float(final))
        vary_explanation[explanation] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
            "per_seed": [float(v) for v in values],
        }

    payload = RefiningExperimentResult(
        env=env_name,
        no_refine=no_refine,
        vary_refine=vary_refine,
        vary_explanation=vary_explanation,
        curves=curves,
        config={
            "total_steps": int(total_steps or spec.refine_steps),
            "seeds": [int(s) for s in seeds],
            "p": spec.p,
            "rnd_lambda": spec.rnd_lambda,
            "sparse_rewards": spec.sparse_rewards,
        },
    )
    out_dir = ensure_dir(out_dir or ckpt_path("experiment_ii_iii", kind="results"))
    save_json(
        {
            "env": payload.env,
            "no_refine": payload.no_refine,
            "vary_refine": payload.vary_refine,
            "vary_explanation": payload.vary_explanation,
            "curves": payload.curves,
            "config": payload.config,
        },
        os.path.join(out_dir, "{}_refining.json".format(env_name)),
    )
    close_env(eval_env)
    return payload
