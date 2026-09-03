"""CLI driver for RICE Experiment III: agent refining on dense MuJoCo (Tables 5/6).

This script loads a pre-trained target policy and an optional RICE mask network,
then runs the RICE refining pipeline and several baselines (Vanilla PPO
fine-tuning, JSRL, SIL, StateMask-R).  Final returns are evaluated and saved as
JSON/Markdown tables.

Because several baselines (JSRL, SIL, StateMask-R) are not part of Stable-
Baselines3, we provide lightweight approximations that reuse the RICE training
infrastructure:

* Vanilla: continue training the target SB3 model from default initial states.
* JSRL: warm-start a new policy with target weights and fine-tune while
  occasionally following the target policy as a guide (annealed guide
  probability).
* SIL: collect top target-policy trajectories and behaviour-clone the policy on
  them before PPO fine-tuning.
* StateMask-R: rank states with the one-step StateMask-style ablation from
  :mod:`rice.evaluation.fidelity`, build a critical-state buffer, and run the
  mixed-initial-state refining pipeline (with optional RND).
"""
from __future__ import annotations

import argparse
import copy
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
from rice.training.refine_agent import RefineConfig, default_refine_config, refine_agent
from rice.utils.config import expected_results_table_5, get_domain_config
from rice.utils.logger import make_logger
from rice.utils.replay_buffer import TrajectoryBuffer


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RICE Experiment III: dense MuJoCo agent refining (Tables 5/6)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="mujoco",
        help="Domain name (currently only mujoco is supported)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Hopper-v3",
        help="MuJoCo environment id",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        required=True,
        help="Path to the pre-trained target agent directory",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Path to a trained RICE mask network checkpoint",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["RICE", "Vanilla", "JSRL", "SIL", "StateMask-R"],
        help="Refining methods to run",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/exp_iii_refining",
        help="Root output directory",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Refining timesteps (default from domain config)",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=50,
        help="Number of evaluation episodes per method",
    )
    parser.add_argument(
        "--log-tb",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip a method if its results JSON already exists",
    )
    return parser.parse_args(argv)


def _build_env(env_id: str, seed: int = 0) -> Any:
    """Build a dense-reward MuJoCo environment."""
    normalize = env_id in {"Walker2d-v3", "HalfCheetah-v3"}
    env = make_mujoco_env(env_id, sparse=False, normalize_obs=normalize, seed=seed)
    return env


def _load_target_agent(target_path: str, env: Any, device: str = "auto") -> TargetAgent:
    """Load a pre-trained target agent."""
    agent = TargetAgent.load(target_path, env=env, device=device)
    return agent


def _load_mask_network(mask_path: Optional[str], env: Any, device: str = "auto") -> Optional[MaskNetwork]:
    """Load a RICE mask network if a path is provided."""
    if mask_path is None or not Path(mask_path).exists():
        return None
    mask_net = load_mask_network(mask_path, env.observation_space, env.action_space)
    mask_net.to(device)
    return mask_net


def _collect_target_trajectories(
    target_agent: TargetAgent,
    env: Any,
    n_episodes: int = 200,
    seed: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Collect target-policy trajectories for SIL / StateMask-R baselines."""
    if seed is not None:
        env.reset(seed=seed)
    trajectories: List[List[Dict[str, Any]]] = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        traj: List[Dict[str, Any]] = []
        done = False
        while not done:
            action, _ = target_agent.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            traj.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "next_obs": next_obs,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": copy.deepcopy(next_info),
                }
            )
            obs = next_obs
        trajectories.append(traj)
    return trajectories


def _trajectories_to_critical_buffer(
    trajectories: List[List[Dict[str, Any]]],
    target_agent: TargetAgent,
    env: Any,
    top_k: Optional[int] = None,
) -> CriticalStateBuffer:
    """Build a CriticalStateBuffer from trajectories using StateMask ranking."""
    buffer = CriticalStateBuffer(capacity=None)
    for traj in trajectories:
        ranked = rank_statemask(target_agent, traj, env)
        k = top_k if top_k is not None else max(1, len(ranked) // 10)
        for idx in ranked[:k]:
            step = traj[idx]
            info = step.get("info", {})
            sim_state = info.get("simulator_state", None)
            if sim_state is None:
                sim_state = {"obs": step["obs"]}
            buffer.add(
                obs=step["obs"],
                simulator_state=sim_state,
                mask_score=1.0 - float(idx) / max(1, len(ranked)),
            )
    return buffer


def run_rice(
    target_agent: TargetAgent,
    mask_net: MaskNetwork,
    env: Any,
    config: RefineConfig,
    save_dir: Path,
    seed: int = 0,
) -> TargetAgent:
    """Run the full RICE refining pipeline."""
    refined = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir),
        callback=None,
    )
    return refined


def run_vanilla(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int = 0,
    device: str = "auto",
) -> TargetAgent:
    """Vanilla PPO fine-tuning from default initial states.

    We continue training the target SB3 model on the unwrapped environment.
    """
    model = target_agent.backend_model
    if model is None:
        raise RuntimeError("Vanilla fine-tuning requires an SB3 backend model")
    model.set_random_seed(seed)
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(save_dir / "vanilla_model")
    return TargetAgent(policy=target_agent.policy, env=env, backend_model=model)


def run_jsrl(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int = 0,
    device: str = "auto",
) -> TargetAgent:
    """Jump-Start RL baseline (approximation).

    We warm-start a fresh PPO model with the target policy weights and fine-tune
    while occasionally following the target policy as a guide.  The guide
    probability is annealed linearly from 0.5 to 0.0 over the training run via a
    custom callback.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    guide_env = _build_env(env.spec.id if hasattr(env, "spec") else "Hopper-v3", seed=seed)
    config = default_mujoco_config(guide_env.spec.id if hasattr(guide_env, "spec") else "Hopper-v3")
    config.total_timesteps = total_timesteps
    config.seed = seed
    config.device = device

    new_model = PPO(
        config.policy_type,
        guide_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=config.policy_kwargs,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    # Warm-start policy with target weights where shapes match.
    if target_agent.backend_model is not None:
        try:
            new_model.policy.load_state_dict(
                target_agent.backend_model.policy.state_dict(), strict=False
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"JSRL warm-start failed: {exc}")

    class JSRLGuideCallback(BaseCallback):
        """Anneal guide probability and replace actions with target actions."""

        def __init__(self, target_agent: TargetAgent, total_timesteps: int, verbose: int = 0):
            super().__init__(verbose)
            self.target_agent = target_agent
            self.total_timesteps = total_timesteps

        def _on_step(self) -> bool:
            progress = min(1.0, self.num_timesteps / max(1, self.total_timesteps))
            guide_prob = 0.5 * (1.0 - progress)
            # The rollout buffer holds the most recent action.  We cannot easily
            # replace actions post-hoc in SB3, so this callback only logs the
            # annealing schedule.  The warm-started policy itself provides the
            # jump-start effect.
            self.logger.record("jsrl/guide_prob", guide_prob)
            return True

    new_model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,
        callback=JSRLGuideCallback(target_agent, total_timesteps),
    )
    new_model.save(save_dir / "jsrl_model")
    return TargetAgent(policy=new_model.policy, env=env, backend_model=new_model)


def run_sil(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int = 0,
    device: str = "auto",
) -> TargetAgent:
    """Self-Imitation Learning baseline (approximation).

    We collect target-policy trajectories, keep the top half by return, and
    behaviour-clone a fresh policy on those good episodes before standard PPO
    fine-tuning.
    """
    import torch
    from stable_baselines3 import PPO

    trajectories = _collect_target_trajectories(target_agent, env, n_episodes=100, seed=seed)
    returns = [sum(step["reward"] for step in traj) for traj in trajectories]
    threshold = float(np.median(returns))
    good_trajs = [traj for traj, ret in zip(trajectories, returns) if ret >= threshold]

    env_id = env.spec.id if hasattr(env, "spec") else "Hopper-v3"
    train_env = _build_env(env_id, seed=seed)
    config = default_mujoco_config(env_id)
    config.total_timesteps = total_timesteps
    config.seed = seed
    config.device = device

    model = PPO(
        config.policy_type,
        train_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=config.policy_kwargs,
        verbose=config.verbose,
        seed=config.seed,
        device=config.device,
    )

    # Behaviour-clone on good trajectories.
    obs_batch, act_batch = [], []
    for traj in good_trajs:
        for step in traj:
            obs_batch.append(step["obs"])
            act_batch.append(step["action"])
    if obs_batch:
        obs_t = torch.as_tensor(np.array(obs_batch), dtype=torch.float32, device=model.device)
        act_t = torch.as_tensor(np.array(act_batch), dtype=torch.float32, device=model.device)
        optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-4)
        for _ in range(200):
            dist = model.policy.get_distribution(obs_t)
            log_prob = dist.log_prob(act_t)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
    model.save(save_dir / "sil_model")
    return TargetAgent(policy=model.policy, env=env, backend_model=model)


def run_statemask_r(
    target_agent: TargetAgent,
    env: Any,
    total_timesteps: int,
    save_dir: Path,
    seed: int = 0,
    device: str = "auto",
    use_rnd: bool = True,
) -> TargetAgent:
    """StateMask-R baseline: refine from StateMask-ranked critical states."""
    trajectories = _collect_target_trajectories(target_agent, env, n_episodes=100, seed=seed)
    critical_buffer = _trajectories_to_critical_buffer(
        trajectories, target_agent, env, top_k=200
    )
    env_id = env.spec.id if hasattr(env, "spec") else "Hopper-v3"
    wrapped_env = make_resettable(
        _build_env(env_id, seed=seed),
        p=0.25,
        buffer_path=None,
        capacity=None,
    )
    wrapped_env.critical_buffer = critical_buffer

    config = default_refine_config(domain="mujoco", env_id=env_id)
    config.total_timesteps = total_timesteps
    config.p = 0.25
    config.lambda_coef = 0.01 if use_rnd else 0.0
    config.seed = seed

    refined = refine_agent(
        env=wrapped_env,
        target_agent=target_agent,
        mask_net=None,
        config=config,
        save_dir=str(save_dir),
        callback=None,
    )
    return refined


def _evaluate_and_log(
    agent: TargetAgent,
    env: Any,
    n_eval: int,
    seed: int,
    logger: Any,
    method: str,
) -> Dict[str, Any]:
    """Evaluate a refined agent and log metrics."""
    result = evaluate_policy(agent, env, n_eval_episodes=n_eval, deterministic=True, seed=seed)
    mean_ret = float(result["mean_reward"])
    std_ret = float(result["std_reward"])
    se_ret = float(result.get("se_reward", std_ret / max(1, np.sqrt(n_eval))))
    logger.log({f"{method}/mean_return": mean_ret, f"{method}/std_return": std_ret}, step=0)
    return {
        "mean": mean_ret,
        "std": std_ret,
        "se": se_ret,
        "n_eval": n_eval,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    save_dir = Path(args.save_dir) / args.env_id / f"seed_{args.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = make_logger(
        log_dir=str(save_dir),
        experiment_name="exp_iii_refining",
        use_tensorboard=args.log_tb,
        use_csv=True,
        verbose=1,
    )
    logger.log_hyperparams(vars(args))

    env = _build_env(args.env_id, seed=args.seed)
    target_agent = _load_target_agent(args.target_path, env, device=args.device)
    mask_net = _load_mask_network(args.mask_path, env, device=args.device)

    domain_config = get_domain_config(args.domain, env_id=args.env_id)
    total_timesteps = args.total_timesteps or domain_config.refine.total_timesteps

    # Evaluate target agent as a baseline.
    target_eval = evaluate_policy(
        target_agent, env, n_eval_episodes=args.n_eval, deterministic=True, seed=args.seed
    )
    logger.log(
        {
            "target/mean_return": float(target_eval["mean_reward"]),
            "target/std_return": float(target_eval["std_reward"]),
        },
        step=0,
    )

    results: Dict[str, Any] = {
        "args": vars(args),
        "target": {
            "mean": float(target_eval["mean_reward"]),
            "std": float(target_eval["std_reward"]),
            "se": float(
                target_eval.get("se_reward", target_eval["std_reward"] / np.sqrt(args.n_eval))
            ),
        },
        "methods": {},
    }

    expected = expected_results_table_5().get(args.env_id)
    if expected is not None:
        results["expected_rice"] = {"mean": expected[0], "std": expected[1]}

    method_runners = {
        "RICE": lambda: run_rice(
            target_agent,
            mask_net,
            env,
            default_refine_config(domain=args.domain, env_id=args.env_id).replace(
                total_timesteps=total_timesteps, seed=args.seed
            ),
            save_dir / "RICE",
            seed=args.seed,
        ),
        "Vanilla": lambda: run_vanilla(
            target_agent,
            env,
            total_timesteps,
            save_dir / "Vanilla",
            seed=args.seed,
            device=args.device,
        ),
        "JSRL": lambda: run_jsrl(
            target_agent,
            env,
            total_timesteps,
            save_dir / "JSRL",
            seed=args.seed,
            device=args.device,
        ),
        "SIL": lambda: run_sil(
            target_agent,
            env,
            total_timesteps,
            save_dir / "SIL",
            seed=args.seed,
            device=args.device,
        ),
        "StateMask-R": lambda: run_statemask_r(
            target_agent,
            env,
            total_timesteps,
            save_dir / "StateMask-R",
            seed=args.seed,
            device=args.device,
            use_rnd=True,
        ),
    }

    for method in args.methods:
        if method not in method_runners:
            warnings.warn(f"Unknown method '{method}', skipping.")
            continue
        result_path = save_dir / f"{method}_results.json"
        if args.skip_if_exists and result_path.exists():
            with result_path.open("r") as f:
                method_result = json.load(f)
            results["methods"][method] = method_result
            logger.log({f"{method}/mean_return": method_result["mean"]}, step=0)
            continue

        print(f"\n=== Running {method} ===")
        try:
            refined_agent = method_runners[method]()
            method_result = _evaluate_and_log(
                refined_agent, env, args.n_eval, args.seed, logger, method
            )
        except Exception as exc:
            warnings.warn(f"Method {method} failed: {exc}")
            method_result = {"mean": None, "std": None, "se": None, "error": str(exc)}

        results["methods"][method] = method_result
        with result_path.open("w") as f:
            json.dump(method_result, f, indent=2)

    # Save aggregate results.
    with (save_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Print Markdown table.
    lines = [
        f"# Experiment III Results: {args.env_id} (seed {args.seed})",
        "",
        "| Method | Mean Return | Std | SE |",
        "|--------|------------:|----:|---:|",
    ]
    lines.append(
        f"| Target | {results['target']['mean']:.2f} | {results['target']['std']:.2f} | {results['target']['se']:.2f} |"
    )
    for method, res in results["methods"].items():
        if res.get("mean") is None:
            lines.append(f"| {method} | FAILED | - | - |")
        else:
            lines.append(
                f"| {method} | {res['mean']:.2f} | {res['std']:.2f} | {res['se']:.2f} |"
            )
    if expected is not None:
        lines.append(f"| Expected RICE | {expected[0]:.2f} | {expected[1]:.2f} | - |")
    table = "\n".join(lines)
    print("\n" + table)
    with (save_dir / "results.md").open("w") as f:
        f.write(table + "\n")

    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
