"""CLI driver for RICE Experiment IV: sparse-reward MuJoCo refining.

This script reproduces Figures 10-13 and the associated ablation tables from the
paper.  It evaluates RICE and several baselines on sparse-reward variants of
Hopper-v3, Walker2d-v3 and HalfCheetah-v3, and performs sensitivity sweeps over
the mixed-initial-state probability ``p`` and the RND bonus coefficient ``λ``.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rice.agents.mask_network import MaskNetwork, load_mask_network
from rice.agents.target_agent import (
    TargetAgent,
    TargetAgentConfig,
    default_mujoco_config,
    train_target_agent_sb3,
)
from rice.envs import make_mujoco_env
from rice.envs.resettable_env import CriticalStateBuffer, ResettableEnv, make_resettable
from rice.evaluation.evaluate_policy import evaluate_policy
from rice.evaluation.fidelity import rank_statemask
from rice.evaluation.visualize import visualize_mujoco_critical_steps
from rice.training.refine_agent import RefineConfig, default_refine_config, refine_agent
from rice.training.train_mask import train_mujoco_mask
from rice.utils.config import get_domain_config, table_3
from rice.utils.logger import make_logger
from rice.utils.replay_buffer import TrajectoryBuffer

warnings.filterwarnings("ignore", category=UserWarning)

SPARSE_ENVS = ["Hopper-v3", "Walker2d-v3", "HalfCheetah-v3"]
P_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
LAMBDA_GRID = [0.0, 0.1, 0.01, 0.001]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RICE Experiment IV: sparse-reward MuJoCo refining"
    )
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=SPARSE_ENVS,
        help="Sparse MuJoCo environment IDs to benchmark",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default="results/targets/sparse_mujoco",
        help="Directory containing pre-trained sparse MuJoCo target agents",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="results/masks/sparse_mujoco",
        help="Directory containing trained mask networks",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/exp_iv_sparse",
        help="Directory to save experiment outputs",
    )
    parser.add_argument(
        "--train-mask",
        action="store_true",
        help="Train mask networks if they are not found in --mask-dir",
    )
    parser.add_argument(
        "--train-target",
        action="store_true",
        help="Train target agents if they are not found in --target-dir",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["RICE", "Vanilla", "JSRL", "SIL", "StateMask-R", "Random"],
        help="Refining methods to compare",
    )
    parser.add_argument(
        "--p-grid",
        nargs="+",
        type=float,
        default=P_GRID,
        help="Mixed-initial-state probability grid",
    )
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=LAMBDA_GRID,
        help="RND bonus coefficient grid",
    )
    parser.add_argument(
        "--refine-timesteps",
        type=int,
        default=None,
        help="Total timesteps for refining (default: domain config)",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=50,
        help="Number of evaluation episodes per final policy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the p/lambda sensitivity sweep and only run default configs",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip trajectory visualization",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )
    return parser.parse_args(argv)


def _build_env(env_id: str, seed: int = 0) -> Any:
    """Build a sparse-reward MuJoCo environment with correct normalization."""
    normalize_obs = env_id in ("Walker2d-v3", "HalfCheetah-v3")
    return make_mujoco_env(
        env_id,
        sparse=True,
        normalize_obs=normalize_obs,
        terminate_on_unhealthy=True,
    )


def _target_path(target_dir: Path, env_id: str) -> Path:
    return target_dir / env_id.replace("-", "_") / "target_agent.zip"


def _mask_path(mask_dir: Path, env_id: str) -> Path:
    return mask_dir / env_id.replace("-", "_") / "mask_net.pt"


def _load_target_agent(target_path: Path, env: Any, device: str = "auto") -> TargetAgent:
    if not target_path.exists():
        raise FileNotFoundError(f"Target agent not found at {target_path}")
    return TargetAgent.load(str(target_path), env=env, device=device)


def _load_or_train_target(
    env_id: str,
    target_dir: Path,
    seed: int,
    device: str,
    train: bool,
    verbose: int,
) -> Tuple[TargetAgent, Any]:
    env = _build_env(env_id, seed=seed)
    target_path = _target_path(target_dir, env_id)
    if target_path.exists():
        if verbose:
            print(f"[ExpIV] Loading target agent for {env_id} from {target_path}")
        return _load_target_agent(target_path, env, device=device), env
    if not train:
        raise FileNotFoundError(
            f"Target agent missing at {target_path}; use --train-target to train one."
        )
    if verbose:
        print(f"[ExpIV] Training target agent for {env_id}")
    config = default_mujoco_config(env_id, sparse=True)
    config.seed = seed
    config.device = device
    agent = train_target_agent_sb3(
        env,
        config=config,
        save_dir=str(target_path.parent),
        algorithm=config.algorithm,
        policy_type=config.policy_type,
    )
    return agent, env


def _load_or_train_mask(
    env_id: str,
    target_agent: TargetAgent,
    env: Any,
    mask_dir: Path,
    seed: int,
    device: str,
    train: bool,
    verbose: int,
) -> MaskNetwork:
    mask_path = _mask_path(mask_dir, env_id)
    if mask_path.exists():
        if verbose:
            print(f"[ExpIV] Loading mask network for {env_id} from {mask_path}")
        return load_mask_network(
            str(mask_path),
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
        )
    if not train:
        raise FileNotFoundError(
            f"Mask network missing at {mask_path}; use --train-mask to train one."
        )
    if verbose:
        print(f"[ExpIV] Training mask network for {env_id}")
    result = train_mujoco_mask(
        env_id=env_id,
        target_agent=target_agent,
        sparse=True,
        save_dir=str(mask_path.parent),
        seed=seed,
        device=device,
        verbose=verbose,
    )
    return result["mask_net"]


def _collect_target_trajectories(
    target_agent: TargetAgent,
    env: Any,
    n_episodes: int = 200,
    seed: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    eval_out = evaluate_policy(
        target_agent,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        collect_trajectories=True,
        seed=seed,
    )
    return eval_out.get("trajectories", [])


def _trajectories_to_critical_buffer(
    trajectories: List[List[Dict[str, Any]]],
    target_agent: TargetAgent,
    env: Any,
    top_k: Optional[int] = None,
) -> CriticalStateBuffer:
    """Build a critical-state buffer using StateMask-style ranking."""
    buffer = CriticalStateBuffer(capacity=None)
    for traj in trajectories:
        ranked = rank_statemask(target_agent, traj, env)
        k = top_k if top_k is not None else max(1, len(traj) // 5)
        for idx in ranked[:k]:
            step = traj[idx]
            buffer.add(
                obs=step["obs"],
                simulator_state=step.get("simulator_state"),
                mask_score=step.get("mask_score", 1.0),
            )
    return buffer


def _evaluate_and_log(
    agent: TargetAgent,
    env: Any,
    n_eval: int,
    seed: int,
    logger: Any,
    method: str,
    env_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result = evaluate_policy(agent, env, n_eval_episodes=n_eval, deterministic=True, seed=seed)
    metrics = {
        "mean_return": float(result["mean_return"]),
        "std_return": float(result["std_return"]),
        "stderr_return": float(result["stderr_return"]),
        "mean_length": float(result["mean_length"]),
        "method": method,
        "env_id": env_id,
    }
    if extra:
        metrics.update(extra)
    if logger is not None:
        logger.log(metrics, step=seed)
    if getattr(logger, "verbose", 0):
        print(
            f"[ExpIV] {method:16s} {env_id:18s} "
            f"return={metrics['mean_return']:8.2f} ± {metrics['stderr_return']:6.2f}"
        )
    return metrics


def _run_rice(
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    env: Any,
    config: RefineConfig,
    save_dir: Path,
    seed: int,
    device: str,
) -> TargetAgent:
    return refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir),
        seed=seed,
        device=device,
    )


def _run_vanilla(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int,
    device: str,
) -> TargetAgent:
    config = default_mujoco_config(env.unwrapped.spec.id if hasattr(env.unwrapped, "spec") else "Hopper-v3", sparse=True)
    config.total_timesteps = total_timesteps
    config.learning_rate = 1e-4
    config.seed = seed
    config.device = device
    return train_target_agent_sb3(
        env,
        config=config,
        save_dir=str(save_dir),
        algorithm=config.algorithm,
        policy_type=config.policy_type,
    )


def _run_jsrl(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int,
    device: str,
) -> TargetAgent:
    """Approximate Jump-Start RL by warm-starting a fresh PPO model with target weights."""
    from stable_baselines3 import PPO

    config = default_mujoco_config(
        env.unwrapped.spec.id if hasattr(env.unwrapped, "spec") else "Hopper-v3", sparse=True
    )
    config.total_timesteps = total_timesteps
    config.learning_rate = 1e-4
    config.seed = seed
    config.device = device
    new_agent = train_target_agent_sb3(
        env,
        config=config,
        save_dir=str(save_dir),
        algorithm=config.algorithm,
        policy_type=config.policy_type,
    )
    if isinstance(getattr(new_agent, "backend_model", None), PPO) and isinstance(
        getattr(target_agent, "backend_model", None), PPO
    ):
        new_agent.backend_model.policy.load_state_dict(
            target_agent.backend_model.policy.state_dict(), strict=False
        )
    return new_agent


def _run_sil(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int,
    device: str,
) -> TargetAgent:
    """Approximate Self-Imitation Learning via behaviour cloning on good trajectories."""
    import torch as th
    import torch.nn.functional as F
    from stable_baselines3 import PPO

    trajectories = _collect_target_trajectories(target_agent, env, n_episodes=100, seed=seed)
    returns = [sum(step["reward"] for step in traj) for traj in trajectories]
    median_return = float(np.median(returns)) if returns else -np.inf
    good = [traj for traj, ret in zip(trajectories, returns) if ret >= median_return]

    config = default_mujoco_config(
        env.unwrapped.spec.id if hasattr(env.unwrapped, "spec") else "Hopper-v3", sparse=True
    )
    config.total_timesteps = total_timesteps
    config.learning_rate = 1e-4
    config.seed = seed
    config.device = device
    new_agent = train_target_agent_sb3(
        env,
        config=config,
        save_dir=str(save_dir),
        algorithm=config.algorithm,
        policy_type=config.policy_type,
    )

    if isinstance(getattr(new_agent, "backend_model", None), PPO) and good:
        policy = new_agent.backend_model.policy
        optimizer = th.optim.Adam(policy.parameters(), lr=1e-4)
        obs_list, act_list = [], []
        for traj in good:
            for step in traj:
                obs_list.append(step["obs"])
                act_list.append(step["action"])
        obs_t = th.as_tensor(np.array(obs_list), dtype=th.float32, device=policy.device)
        act_t = th.as_tensor(np.array(act_list), dtype=th.float32, device=policy.device)
        for _ in range(100):
            dist = policy.get_distribution(obs_t)
            log_prob = dist.log_prob(act_t)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return new_agent


def _run_statemask_r(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int,
    device: str,
    use_rnd: bool = True,
) -> TargetAgent:
    trajectories = _collect_target_trajectories(target_agent, env, n_episodes=200, seed=seed)
    critical_buffer = _trajectories_to_critical_buffer(trajectories, target_agent, env)
    resettable = make_resettable(env, p=0.25, critical_buffer=critical_buffer)
    config = default_refine_config(domain="mujoco_sparse")
    config.total_timesteps = total_timesteps
    config.p = 0.25
    config.lambda_coef = 0.01 if use_rnd else 0.0
    config.seed = seed
    config.device = device
    return refine_agent(
        env=resettable,
        target_agent=target_agent,
        mask_net=None,
        config=config,
        save_dir=str(save_dir),
        seed=seed,
        device=device,
    )


def _run_random(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int,
    device: str,
) -> TargetAgent:
    """Baseline that refines from randomly sampled states (no mask)."""
    trajectories = _collect_target_trajectories(target_agent, env, n_episodes=200, seed=seed)
    buffer = CriticalStateBuffer(capacity=None)
    for traj in trajectories:
        for step in traj:
            buffer.add(
                obs=step["obs"],
                simulator_state=step.get("simulator_state"),
                mask_score=1.0,
            )
    resettable = make_resettable(env, p=0.25, critical_buffer=buffer)
    config = default_refine_config(domain="mujoco_sparse")
    config.total_timesteps = total_timesteps
    config.p = 0.25
    config.lambda_coef = 0.0
    config.seed = seed
    config.device = device
    return refine_agent(
        env=resettable,
        target_agent=target_agent,
        mask_net=None,
        config=config,
        save_dir=str(save_dir),
        seed=seed,
        device=device,
    )


def _run_method(
    method: str,
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int,
    device: str,
    p: Optional[float] = None,
    lambda_coef: Optional[float] = None,
) -> TargetAgent:
    if method == "RICE":
        config = default_refine_config(domain="mujoco_sparse")
        config.total_timesteps = total_timesteps
        if p is not None:
            config.p = p
        if lambda_coef is not None:
            config.lambda_coef = lambda_coef
        config.seed = seed
        config.device = device
        return _run_rice(
            target_agent, mask_net, env, config, save_dir, seed, device
        )
    if method == "Vanilla":
        return _run_vanilla(target_agent, env, total_timesteps, save_dir, seed, device)
    if method == "JSRL":
        return _run_jsrl(target_agent, env, total_timesteps, save_dir, seed, device)
    if method == "SIL":
        return _run_sil(target_agent, env, total_timesteps, save_dir, seed, device)
    if method == "StateMask-R":
        return _run_statemask_r(
            target_agent, env, total_timesteps, save_dir, seed, device, use_rnd=True
        )
    if method == "Random":
        return _run_random(target_agent, env, total_timesteps, save_dir, seed, device)
    raise ValueError(f"Unknown method: {method}")


def _sensitivity_sweep(
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    env: Any,
    env_id: str,
    save_dir: Path,
    p_grid: List[float],
    lambda_grid: List[float],
    total_timesteps: int,
    n_eval: int,
    seed: int,
    device: str,
    logger: Any,
) -> Dict[str, Any]:
    """Run a grid search over p and lambda for the RICE method."""
    results: Dict[str, Any] = {"env_id": env_id, "grid": []}
    for p in p_grid:
        for lam in lambda_grid:
            method_dir = save_dir / f"rice_p_{p}_lambda_{lam}"
            method_dir.mkdir(parents=True, exist_ok=True)
            refined = _run_method(
                "RICE",
                target_agent,
                mask_net,
                env,
                total_timesteps,
                method_dir,
                seed,
                device,
                p=p,
                lambda_coef=lam,
            )
            metrics = _evaluate_and_log(
                refined,
                env,
                n_eval,
                seed,
                logger,
                f"RICE_p{p}_l{lam}",
                env_id,
                extra={"p": p, "lambda": lam},
            )
            results["grid"].append(metrics)
    return results


def _run_single_env(
    env_id: str,
    args: argparse.Namespace,
    save_dir: Path,
) -> Dict[str, Any]:
    env_save = save_dir / env_id.replace("-", "_")
    env_save.mkdir(parents=True, exist_ok=True)
    logger = make_logger(
        str(env_save / "logs"),
        experiment_name=f"exp_iv_sparse_{env_id.replace('-', '_')}",
        use_tensorboard=False,
        use_csv=True,
        verbose=args.verbose,
    )
    logger.log_hyperparams(vars(args))

    target_dir = Path(args.target_dir)
    mask_dir = Path(args.mask_dir)
    target_agent, env = _load_or_train_target(
        env_id,
        target_dir,
        args.seed,
        args.device,
        args.train_target,
        args.verbose,
    )
    mask_net = _load_or_train_mask(
        env_id,
        target_agent,
        env,
        mask_dir,
        args.seed,
        args.device,
        args.train_mask,
        args.verbose,
    )

    # Baseline target evaluation
    baseline = _evaluate_and_log(
        target_agent,
        env,
        args.n_eval,
        args.seed,
        logger,
        "Target",
        env_id,
    )

    domain_cfg = get_domain_config("mujoco", env_id=env_id, sparse=True)
    total_timesteps = args.refine_timesteps or domain_cfg.refine.total_timesteps

    method_results: Dict[str, Any] = {"target": baseline, "methods": {}}

    for method in args.methods:
        method_dir = env_save / method.lower().replace("-", "_")
        method_dir.mkdir(parents=True, exist_ok=True)
        refined = _run_method(
            method,
            target_agent,
            mask_net,
            env,
            total_timesteps,
            method_dir,
            args.seed,
            args.device,
        )
        metrics = _evaluate_and_log(
            refined,
            env,
            args.n_eval,
            args.seed,
            logger,
            method,
            env_id,
        )
        method_results["methods"][method] = metrics

    sweep_results = None
    if not args.skip_sweep and "RICE" in args.methods:
        sweep_dir = env_save / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_results = _sensitivity_sweep(
            target_agent,
            mask_net,
            env,
            env_id,
            sweep_dir,
            args.p_grid,
            args.lambda_grid,
            total_timesteps,
            args.n_eval,
            args.seed,
            args.device,
            logger,
        )
        method_results["sweep"] = sweep_results

    if not args.skip_visualize:
        try:
            trajectories = _collect_target_trajectories(
                target_agent, env, n_episodes=20, seed=args.seed
            )
            visualize_mujoco_critical_steps(
                trajectories,
                mask_net,
                save_dir=env_save / "visualize",
                env_id=env_id,
                show=False,
            )
        except Exception as exc:  # pragma: no cover
            if args.verbose:
                print(f"[ExpIV] Visualization skipped for {env_id}: {exc}")

    logger.close()
    out_path = env_save / "results.json"
    with out_path.open("w") as f:
        json.dump(method_results, f, indent=2)
    return method_results


def _format_table(results_by_env: Dict[str, Dict[str, Any]]) -> str:
    lines = ["# RICE Experiment IV — Sparse MuJoCo Refining Results", ""]
    lines.append("| Environment | Method | Mean Return | Std Error |")
    lines.append("|-------------|--------|------------:|----------:|")
    for env_id, data in results_by_env.items():
        target = data.get("target", {})
        lines.append(
            f"| {env_id:11s} | Target | "
            f"{target.get('mean_return', 0.0):10.2f} | "
            f"{target.get('stderr_return', 0.0):9.2f} |"
        )
        for method, metrics in data.get("methods", {}).items():
            lines.append(
                f"| {env_id:11s} | {method:6s} | "
                f"{metrics.get('mean_return', 0.0):10.2f} | "
                f"{metrics.get('stderr_return', 0.0):9.2f} |"
            )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}
    for env_id in args.env_ids:
        if args.verbose:
            print(f"\n[ExpIV] ===== Running sparse MuJoCo environment: {env_id} =====")
        try:
            all_results[env_id] = _run_single_env(env_id, args, save_dir)
        except Exception as exc:
            if args.verbose:
                print(f"[ExpIV] Failed on {env_id}: {exc}")
            all_results[env_id] = {"error": str(exc)}

    summary_path = save_dir / "results.json"
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)

    table = _format_table(all_results)
    md_path = save_dir / "results.md"
    with md_path.open("w") as f:
        f.write(table)
    if args.verbose:
        print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
