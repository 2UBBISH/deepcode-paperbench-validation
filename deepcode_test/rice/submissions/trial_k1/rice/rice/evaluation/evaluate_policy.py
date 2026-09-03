"""Standardized policy evaluation rollouts for RICE.

This module implements the evaluation protocol used across all experiments:
run ``n_eval_episodes`` full episodes, optionally with deterministic actions,
and report mean return, standard deviation, standard error, and optionally the
full trajectory list.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from rice.agents.target_agent import TargetAgent
from rice.envs import make_mujoco_env
from rice.utils.config import DomainConfig, get_domain_config
from rice.utils.logger import Logger, make_logger


def evaluate_policy(
    agent: TargetAgent,
    env: gym.Env,
    n_eval_episodes: int = 50,
    deterministic: bool = True,
    render: bool = False,
    collect_trajectories: bool = False,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a :class:`TargetAgent` on a Gymnasium environment.

    Parameters
    ----------
    agent:
        The policy to evaluate.
    env:
        The environment. It is *not* reset before evaluation; a fresh seed is
        passed to ``reset`` for each episode.
    n_eval_episodes:
        Number of episodes to run.
    deterministic:
        Whether to use deterministic actions.
    render:
        Whether to call ``env.render()`` each step.
    collect_trajectories:
        If ``True``, include the per-episode trajectory list in the returned
        dictionary.
    seed:
        Base seed for episode resets. Each episode uses ``seed + i``.
    max_episode_steps:
        Hard cap on episode length. If ``None``, the environment's own
        truncation signal is used.

    Returns
    -------
    dict
        Dictionary with keys ``mean_return``, ``std_return``,
        ``stderr_return``, ``returns``, ``lengths``, and optionally
        ``trajectories``.
    """
    returns: List[float] = []
    lengths: List[int] = []
    trajectories: List[List[Dict[str, Any]]] = [] if collect_trajectories else None  # type: ignore[assignment]

    for ep in range(n_eval_episodes):
        ep_seed = None if seed is None else seed + ep
        reset_result = env.reset(seed=ep_seed)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode_return = 0.0
        step = 0
        episode_traj: List[Dict[str, Any]] = [] if collect_trajectories else None  # type: ignore[assignment]

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result
                terminated, truncated = done, False

            episode_return += float(reward)
            step += 1

            if collect_trajectories:
                episode_traj.append({  # type: ignore[union-attr]
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                })

            obs = next_obs

            if render:
                env.render()

            if max_episode_steps is not None and step >= max_episode_steps:
                done = True

        returns.append(episode_return)
        lengths.append(step)
        if collect_trajectories:
            trajectories.append(episode_traj)  # type: ignore[union-attr]

    returns_arr = np.array(returns, dtype=np.float64)
    lengths_arr = np.array(lengths, dtype=np.float64)
    mean_return = float(np.mean(returns_arr))
    std_return = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
    stderr_return = std_return / np.sqrt(len(returns_arr))

    result: Dict[str, Any] = {
        "mean_return": mean_return,
        "std_return": std_return,
        "stderr_return": float(stderr_return),
        "returns": returns,
        "mean_length": float(np.mean(lengths_arr)),
        "std_length": float(np.std(lengths_arr, ddof=1)) if len(lengths_arr) > 1 else 0.0,
        "lengths": lengths,
        "n_eval_episodes": n_eval_episodes,
        "deterministic": deterministic,
    }
    if collect_trajectories:
        result["trajectories"] = trajectories
    return result


def evaluate_policy_from_domain(
    agent: TargetAgent,
    domain: str,
    n_eval_episodes: int = 50,
    deterministic: bool = True,
    seed: int = 0,
    collect_trajectories: bool = False,
    config: Optional[DomainConfig] = None,
    **env_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a target agent by constructing the domain environment.

    This is a convenience wrapper around :func:`evaluate_policy` that builds
    the environment from a :class:`DomainConfig`.
    """
    if config is None:
        config = get_domain_config(domain, **env_kwargs)

    env = _build_env_from_config(config)
    try:
        result = evaluate_policy(
            agent,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            seed=seed,
            collect_trajectories=collect_trajectories,
        )
    finally:
        env.close()
    return result


def _build_env_from_config(config: DomainConfig) -> gym.Env:
    """Build a Gymnasium environment from a domain config."""
    name = config.name
    env_kwargs = dict(config.env_kwargs)

    if name in {"hopper", "walker2d", "reacher", "halfcheetah"} or "mujoco" in name:
        env_id = config.env_id or env_kwargs.pop("env_id", None)
        if env_id is None:
            raise ValueError(f"MuJoCo domain config missing env_id: {config}")
        return make_mujoco_env(
            env_id=env_id,
            sparse=env_kwargs.pop("sparse", False),
            normalize_obs=env_kwargs.pop("normalize_obs", None),
            **env_kwargs,
        )

    # Domain-specific factories are imported lazily to keep hard dependencies
    # minimal.
    if name == "selfish_mining":
        from rice.envs import make_selfish_mining_env

        return make_selfish_mining_env(**env_kwargs)
    if name == "cage":
        from rice.envs import make_cage_env

        return make_cage_env(**env_kwargs)
    if name == "metadrive":
        from rice.envs import make_metadrive_env

        return make_metadrive_env(**env_kwargs)
    if name == "malware":
        from rice.envs import make_malware_env

        return make_malware_env(**env_kwargs)

    raise ValueError(f"Unsupported domain for evaluation: {name}")


def log_evaluation(
    result: Dict[str, Any],
    logger: Optional[Logger] = None,
    prefix: str = "eval",
    step: Optional[int] = None,
) -> None:
    """Log evaluation metrics to a :class:`Logger`."""
    metrics = {
        f"{prefix}/mean_return": result["mean_return"],
        f"{prefix}/std_return": result["std_return"],
        f"{prefix}/stderr_return": result["stderr_return"],
        f"{prefix}/mean_length": result["mean_length"],
    }
    if logger is not None:
        logger.log(metrics, step=step)


def compare_policies(
    agents: Dict[str, TargetAgent],
    env: gym.Env,
    n_eval_episodes: int = 50,
    deterministic: bool = True,
    seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate multiple agents on the same environment and return a table."""
    table: Dict[str, Dict[str, Any]] = {}
    for name, agent in agents.items():
        table[name] = evaluate_policy(
            agent,
            env,
            n_eval_episodes=n_eval_episodes,
            deterministic=deterministic,
            seed=seed,
        )
    return table


def main() -> None:
    """CLI entry point for evaluating a saved target agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a saved RICE target agent.")
    parser.add_argument("--domain", type=str, required=True, help="Domain name.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model.")
    parser.add_argument("--env-id", type=str, default=None, help="Gymnasium env id (MuJoCo only).")
    parser.add_argument("--sparse", action="store_true", help="Use sparse reward variant.")
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    deterministic = not args.stochastic

    config_kwargs: Dict[str, Any] = {}
    if args.env_id is not None:
        config_kwargs["env_id"] = args.env_id
    if args.sparse:
        config_kwargs["sparse"] = True

    config = get_domain_config(args.domain, **config_kwargs)
    env = _build_env_from_config(config)

    agent = TargetAgent.load(args.model_path, env=env, device=args.device)
    result = evaluate_policy(
        agent,
        env,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=deterministic,
        seed=args.seed,
    )

    logger: Optional[Logger] = None
    if args.log_dir is not None:
        logger = make_logger(args.log_dir, f"eval_{args.domain}", use_tensorboard=False, use_csv=True)
        log_evaluation(result, logger=logger)
        logger.close()

    print(f"Evaluation over {result['n_eval_episodes']} episodes:")
    print(f"  Mean return:  {result['mean_return']:.4f} ± {result['stderr_return']:.4f}")
    print(f"  Std return:   {result['std_return']:.4f}")
    print(f"  Mean length:  {result['mean_length']:.2f}")


if __name__ == "__main__":
    main()
