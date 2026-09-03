"""CLI driver for RICE Experiment V -- case studies.

This script reproduces the paper's case-study experiments:

1. Malware mutation (Table 7)
   - Baseline pre-trained evasion rate.
   - Continue training (vanilla fine-tuning).
   - Refine from critical states only (overfitting demonstration).
   - Mixed initial distribution (RICE without RND).
   - Full RICE (mixed initial distribution + RND exploration bonus).
   - Reward-design fix: scale intermediate confidence-drop reward by 3.

2. Autonomous driving (Figure 14)
   - Train/load a MetaDrive Macro-v1 target policy.
   - Train/load a RICE mask network and extract critical lane-switch states.
   - Refine the policy and visualize trajectory importance heat-maps.

3. Negative control -- MountainCarContinuous-v0 (Figure 15)
   - Show that RICE does not help when the pre-trained policy has almost no
     state coverage; performance stays similar to RND-only fine-tuning.

Usage
-----
    python -m scripts.run_exp_v_case_study --domain malware --target-path ...
    python -m scripts.run_exp_v_case_study --domain metadrive --target-path ...
    python -m scripts.run_exp_v_case_study --domain mountaincar --target-path ...
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from rice.agents.mask_network import MaskNetwork, load_mask_network, make_mask_network
from rice.agents.target_agent import TargetAgent
from rice.envs import make_malware_env, make_metadrive_env, make_mujoco_env
from rice.envs.resettable_env import CriticalStateBuffer, ResettableEnv, make_resettable
from rice.evaluation.evaluate_policy import evaluate_policy
from rice.training.refine_agent import RefineConfig, default_refine_config, refine_agent
from rice.training.train_mask import train_mask
from rice.utils.config import get_domain_config
from rice.utils.logger import make_logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RICE Experiment V -- case studies (malware, autonomous driving, MountainCar)."
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["malware", "metadrive", "mountaincar"],
        help="Case-study domain to run.",
    )
    parser.add_argument(
        "--target-path",
        type=str,
        default=None,
        help="Path to a saved pre-trained target agent.",
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        default=None,
        help="Path to a saved RICE mask network (optional; will train if missing).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory where results and checkpoints are written.",
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
        help="PyTorch device string.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=100,
        help="Number of evaluation episodes for malware evasion rates.",
    )
    parser.add_argument(
        "--train-mask",
        action="store_true",
        help="Train a new mask network if --mask-path is not provided.",
    )
    parser.add_argument(
        "--skip-visualize",
        action="store_true",
        help="Skip trajectory visualization for MetaDrive.",
    )
    parser.add_argument(
        "--reward-fix-scale",
        type=float,
        default=3.0,
        help="Scale factor for the malware reward-design fix ablation.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level.",
    )
    return parser.parse_args(argv)


def _default_save_dir(domain: str, seed: int) -> Path:
    return Path("results") / "exp_v_case_study" / domain / f"seed_{seed}"


def _load_target_agent(target_path: Optional[str], env: Any, device: str = "auto") -> TargetAgent:
    if target_path is None:
        raise ValueError("--target-path is required for case-study experiments.")
    agent = TargetAgent.load(target_path, env=env, device=device)
    return agent


def _load_or_train_mask(
    mask_path: Optional[str],
    env: Any,
    target_agent: TargetAgent,
    domain: str,
    seed: int,
    device: str,
    train: bool,
    save_dir: Path,
) -> Optional[MaskNetwork]:
    if mask_path is not None:
        return load_mask_network(mask_path, env.observation_space, env.action_space, device=device)
    if not train:
        return None
    mask_result = train_mask(
        domain=domain,
        target_agent=target_agent,
        save_dir=str(save_dir / "mask"),
        seed=seed,
        device=device,
    )
    return mask_result.get("mask_net")


# ---------------------------------------------------------------------------
# Malware case study
# ---------------------------------------------------------------------------


def _malware_evasion_rate(agent: TargetAgent, env: Any, n_eval: int, seed: int) -> Dict[str, Any]:
    """Evaluate an agent on the malware domain and report evasion rate."""
    result = evaluate_policy(
        agent,
        env,
        n_eval_episodes=n_eval,
        deterministic=True,
        seed=seed,
        collect_trajectories=True,
    )
    trajectories = result.get("trajectories", [])
    successes = 0
    for ep in trajectories:
        if not ep:
            continue
        last_info = ep[-1].get("info", {})
        if last_info.get("evaded", False) or last_info.get("success", False):
            successes += 1
    evasion_rate = successes / max(len(trajectories), 1)
    return {
        "mean_return": float(result["mean_return"]),
        "std_return": float(result["std_return"]),
        "evasion_rate": float(evasion_rate),
        "n_success": int(successes),
        "n_eval": int(len(trajectories)),
    }


def _run_malware_experiment(
    env: Any,
    target_agent: TargetAgent,
    mask_net: Optional[MaskNetwork],
    config: RefineConfig,
    save_dir: Path,
    seed: int,
    device: str,
    method_name: str,
    n_eval: int,
) -> Dict[str, Any]:
    """Run one malware refining configuration and evaluate evasion rate."""
    refined = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir / method_name),
    )
    eval_result = _malware_evasion_rate(refined, env, n_eval=n_eval, seed=seed)
    eval_result["method"] = method_name
    return eval_result


def run_malware_study(
    target_path: Optional[str],
    mask_path: Optional[str],
    save_dir: Path,
    seed: int,
    device: str,
    n_eval: int,
    train_mask: bool,
    reward_fix_scale: float,
    verbose: int,
) -> Dict[str, Any]:
    """Run the full malware-mutation case study (Table 7)."""
    logger = make_logger(str(save_dir), "malware_case_study", use_tensorboard=False, use_csv=True, verbose=verbose)
    logger.log_hyperparams({
        "domain": "malware",
        "seed": seed,
        "n_eval": n_eval,
        "reward_fix_scale": reward_fix_scale,
    })

    # Default malware environment (original reward design).
    env = make_malware_env(max_steps=10, mutation_actions=16, reward_scale=1.0, random_seed=seed)
    target_agent = _load_target_agent(target_path, env, device)

    # Baseline pre-trained policy.
    baseline = _malware_evasion_rate(target_agent, env, n_eval=n_eval, seed=seed)
    baseline["method"] = "Pre-trained"
    logger.log({"malware/pretrained_evasion_rate": baseline["evasion_rate"]}, step=0)

    # Continue training (vanilla fine-tuning) with the original reward.
    vanilla_config = default_refine_config(domain="malware")
    vanilla_config.p = 0.0
    vanilla_config.lambda_coef = 0.0
    vanilla = _run_malware_experiment(
        env, target_agent, None, vanilla_config, save_dir, seed, device,
        "vanilla", n_eval,
    )
    logger.log({"malware/vanilla_evasion_rate": vanilla["evasion_rate"]}, step=1)

    # Load or train mask network for RICE variants.
    mask_net = _load_or_train_mask(
        mask_path, env, target_agent, "malware", seed, device, train_mask, save_dir,
    )

    # Refine from critical states only (p=1.0, no RND). Demonstrates overfitting.
    critical_only_config = default_refine_config(domain="malware")
    critical_only_config.p = 1.0
    critical_only_config.lambda_coef = 0.0
    critical_only = _run_malware_experiment(
        env, target_agent, mask_net, critical_only_config, save_dir, seed, device,
        "critical_only", n_eval,
    )
    logger.log({"malware/critical_only_evasion_rate": critical_only["evasion_rate"]}, step=2)

    # Mixed initial distribution without RND (p=0.25, lambda=0).
    mixed_no_rnd_config = default_refine_config(domain="malware")
    mixed_no_rnd_config.p = 0.25
    mixed_no_rnd_config.lambda_coef = 0.0
    mixed_no_rnd = _run_malware_experiment(
        env, target_agent, mask_net, mixed_no_rnd_config, save_dir, seed, device,
        "mixed_no_rnd", n_eval,
    )
    logger.log({"malware/mixed_no_rnd_evasion_rate": mixed_no_rnd["evasion_rate"]}, step=3)

    # Full RICE (mixed initial distribution + RND).
    full_rice_config = default_refine_config(domain="malware")
    full_rice_config.p = 0.25
    full_rice = _run_malware_experiment(
        env, target_agent, mask_net, full_rice_config, save_dir, seed, device,
        "full_rice", n_eval,
    )
    logger.log({"malware/full_rice_evasion_rate": full_rice["evasion_rate"]}, step=4)

    # Reward-design fix: scale intermediate confidence-drop reward by reward_fix_scale.
    fixed_env = make_malware_env(
        max_steps=10,
        mutation_actions=16,
        reward_scale=reward_fix_scale,
        random_seed=seed,
    )
    fixed_target = _load_target_agent(target_path, fixed_env, device)
    fixed_rice_config = default_refine_config(domain="malware")
    fixed_rice_config.p = 0.25
    fixed_mask_net = _load_or_train_mask(
        mask_path, fixed_env, fixed_target, "malware", seed, device, train_mask, save_dir,
    )
    reward_fix = _run_malware_experiment(
        fixed_env, fixed_target, fixed_mask_net, fixed_rice_config, save_dir, seed, device,
        "reward_fix", n_eval,
    )
    logger.log({"malware/reward_fix_evasion_rate": reward_fix["evasion_rate"]}, step=5)

    results = {
        "baseline": baseline,
        "vanilla": vanilla,
        "critical_only": critical_only,
        "mixed_no_rnd": mixed_no_rnd,
        "full_rice": full_rice,
        "reward_fix": reward_fix,
    }

    # Persist results.
    with open(save_dir / "malware_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print Markdown table.
    table = "| Method | Evasion Rate | Mean Return |\n"
    table += "|--------|-------------:|------------:|\n"
    for key, res in results.items():
        table += f"| {res['method']} | {res['evasion_rate']*100:.1f}% | {res['mean_return']:.2f} ± {res['std_return']:.2f} |\n"
    print(table)
    logger.log_text("malware/results_table", table, step=0)
    logger.close()
    return results


# ---------------------------------------------------------------------------
# MetaDrive case study
# ---------------------------------------------------------------------------


def _collect_metadrive_trajectory(agent: TargetAgent, env: Any, seed: int) -> List[Dict[str, Any]]:
    result = evaluate_policy(
        agent,
        env,
        n_eval_episodes=1,
        deterministic=True,
        seed=seed,
        collect_trajectories=True,
    )
    trajectories = result.get("trajectories", [[]])
    return trajectories[0] if trajectories else []


def run_metadrive_study(
    target_path: Optional[str],
    mask_path: Optional[str],
    save_dir: Path,
    seed: int,
    device: str,
    train_mask: bool,
    skip_visualize: bool,
    verbose: int,
) -> Dict[str, Any]:
    """Run the MetaDrive autonomous-driving case study (Figure 14)."""
    logger = make_logger(str(save_dir), "metadrive_case_study", use_tensorboard=False, use_csv=True, verbose=verbose)
    logger.log_hyperparams({"domain": "metadrive", "seed": seed})

    env = make_metadrive_env(use_render=False, random_traffic=False, start_seed=seed)
    target_agent = _load_target_agent(target_path, env, device)

    # Baseline evaluation.
    baseline_eval = evaluate_policy(target_agent, env, n_eval_episodes=10, deterministic=True, seed=seed)
    logger.log({"metadrive/baseline_mean_return": baseline_eval["mean_return"]}, step=0)

    # Load or train mask network.
    mask_net = _load_or_train_mask(
        mask_path, env, target_agent, "metadrive", seed, device, train_mask, save_dir,
    )

    # Collect a trajectory and identify the most critical lane-switch step.
    trajectory = _collect_metadrive_trajectory(target_agent, env, seed)
    critical_step = None
    if mask_net is not None and trajectory:
        scores = []
        for step in trajectory:
            obs = step["obs"]
            score = float(mask_net.predict(obs, deterministic=True))
            scores.append(score)
        critical_idx = int(np.argmax(scores))
        critical_step = {
            "index": critical_idx,
            "mask_score": float(scores[critical_idx]),
            "action": trajectory[critical_idx]["action"].tolist() if hasattr(trajectory[critical_idx]["action"], "tolist") else trajectory[critical_idx]["action"],
        }
        logger.log({"metadrive/critical_step_index": critical_idx}, step=0)

    # Refine with full RICE.
    refine_config = default_refine_config(domain="metadrive")
    refined_agent = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=mask_net,
        config=refine_config,
        save_dir=str(save_dir / "refined"),
    )
    refined_eval = evaluate_policy(refined_agent, env, n_eval_episodes=10, deterministic=True, seed=seed)
    logger.log({"metadrive/refined_mean_return": refined_eval["mean_return"]}, step=1)

    # Visualization (best-effort).
    if not skip_visualize:
        try:
            from rice.evaluation.visualize import plot_trajectory_importance
            if trajectory and mask_net is not None:
                fig_path = save_dir / "metadrive_critical_heatmap.png"
                plot_trajectory_importance(
                    trajectory=trajectory,
                    mask_net=mask_net,
                    save_path=str(fig_path),
                    title="MetaDrive Critical Lane-Switch Steps",
                )
                logger.log_text("metadrive/figure_path", str(fig_path), step=0)
        except Exception as exc:
            warnings.warn(f"MetaDrive visualization skipped: {exc}")

    results = {
        "baseline": {
            "mean_return": float(baseline_eval["mean_return"]),
            "std_return": float(baseline_eval["std_return"]),
        },
        "refined": {
            "mean_return": float(refined_eval["mean_return"]),
            "std_return": float(refined_eval["std_return"]),
        },
        "critical_step": critical_step,
    }
    with open(save_dir / "metadrive_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    logger.close()
    return results


# ---------------------------------------------------------------------------
# MountainCar negative control
# ---------------------------------------------------------------------------


def run_mountaincar_study(
    target_path: Optional[str],
    save_dir: Path,
    seed: int,
    device: str,
    verbose: int,
) -> Dict[str, Any]:
    """Run the MountainCarContinuous-v0 negative control (Figure 15)."""
    logger = make_logger(str(save_dir), "mountaincar_case_study", use_tensorboard=False, use_csv=True, verbose=verbose)
    logger.log_hyperparams({"domain": "mountaincar", "seed": seed})

    env_id = "MountainCarContinuous-v0"
    env = make_mujoco_env(env_id, seed=seed)
    target_agent = _load_target_agent(target_path, env, device)

    # Baseline evaluation.
    baseline_eval = evaluate_policy(target_agent, env, n_eval_episodes=50, deterministic=True, seed=seed)
    logger.log({"mountaincar/baseline_mean_return": baseline_eval["mean_return"]}, step=0)

    # Vanilla fine-tuning.
    vanilla_config = default_refine_config(domain="mujoco", env_id=env_id)
    vanilla_config.p = 0.0
    vanilla_config.lambda_coef = 0.0
    vanilla_agent = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=None,
        config=vanilla_config,
        save_dir=str(save_dir / "vanilla"),
    )
    vanilla_eval = evaluate_policy(vanilla_agent, env, n_eval_episodes=50, deterministic=True, seed=seed)
    logger.log({"mountaincar/vanilla_mean_return": vanilla_eval["mean_return"]}, step=1)

    # RND-only fine-tuning (no critical states).
    rnd_only_config = default_refine_config(domain="mujoco", env_id=env_id)
    rnd_only_config.p = 0.0
    rnd_agent = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=None,
        config=rnd_only_config,
        save_dir=str(save_dir / "rnd_only"),
    )
    rnd_eval = evaluate_policy(rnd_agent, env, n_eval_episodes=50, deterministic=True, seed=seed)
    logger.log({"mountaincar/rnd_only_mean_return": rnd_eval["mean_return"]}, step=2)

    # Full RICE (mixed starts + RND). Expected to be similar to RND-only because
    # the pre-trained policy has poor state coverage.
    rice_config = default_refine_config(domain="mujoco", env_id=env_id)
    rice_agent = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=None,
        config=rice_config,
        save_dir=str(save_dir / "rice"),
    )
    rice_eval = evaluate_policy(rice_agent, env, n_eval_episodes=50, deterministic=True, seed=seed)
    logger.log({"mountaincar/rice_mean_return": rice_eval["mean_return"]}, step=3)

    results = {
        "baseline": {
            "mean_return": float(baseline_eval["mean_return"]),
            "std_return": float(baseline_eval["std_return"]),
        },
        "vanilla": {
            "mean_return": float(vanilla_eval["mean_return"]),
            "std_return": float(vanilla_eval["std_return"]),
        },
        "rnd_only": {
            "mean_return": float(rnd_eval["mean_return"]),
            "std_return": float(rnd_eval["std_return"]),
        },
        "rice": {
            "mean_return": float(rice_eval["mean_return"]),
            "std_return": float(rice_eval["std_return"]),
        },
    }
    with open(save_dir / "mountaincar_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    logger.close()
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    save_dir = Path(args.save_dir) if args.save_dir else _default_save_dir(args.domain, args.seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.domain == "malware":
        run_malware_study(
            target_path=args.target_path,
            mask_path=args.mask_path,
            save_dir=save_dir,
            seed=args.seed,
            device=args.device,
            n_eval=args.n_eval,
            train_mask=args.train_mask,
            reward_fix_scale=args.reward_fix_scale,
            verbose=args.verbose,
        )
    elif args.domain == "metadrive":
        run_metadrive_study(
            target_path=args.target_path,
            mask_path=args.mask_path,
            save_dir=save_dir,
            seed=args.seed,
            device=args.device,
            train_mask=args.train_mask,
            skip_visualize=args.skip_visualize,
            verbose=args.verbose,
        )
    elif args.domain == "mountaincar":
        run_mountaincar_study(
            target_path=args.target_path,
            save_dir=save_dir,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unknown domain: {args.domain}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
