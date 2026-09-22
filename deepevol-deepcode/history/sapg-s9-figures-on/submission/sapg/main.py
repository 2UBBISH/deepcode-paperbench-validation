"""SAPG entry point: train / eval dispatch and argument parsing.

This module is the top-level CLI for the SAPG (Split and Aggregate Policy
Gradients) codebase.  It dispatches to:

  * ``train``  -- SAPG training (``sapg.experiments.train_sapg``)
  * ``baseline`` -- PPO / PBT / PQL baselines (``sapg.experiments.train_baselines``)
  * ``ablate`` -- ablation sweeps (``sapg.experiments.ablations``)
  * ``diversity`` -- diversity metrics (``sapg.experiments.diversity_metrics``)
  * ``plot`` -- result plotting (``sapg.experiments.plot_results``)
  * ``eval`` -- evaluate a saved checkpoint

Usage examples
--------------
    python main.py train --task regrasping --num-envs 24576 --num-policies 6
    python main.py baseline --algo ppo --task throw
    python main.py ablate --ablation symmetric --task reorientation
    python main.py diversity --task shadowhand
    python main.py plot --results-dir results
    python main.py eval --checkpoint results/sapg_regrasping_seed0.pt --task regrasping
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Make the repository root importable when running ``python main.py`` directly.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sapg.envs import TASK_NAMES  # noqa: E402
from sapg.sapg.utils import get_logger, set_seed  # noqa: E402

LOGGER = get_logger("sapg.main")


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------
def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every sub-command."""
    parser.add_argument(
        "--task",
        type=str,
        default="regrasping",
        choices=list(TASK_NAMES),
        help="Task to run (one of the five paper tasks).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device string (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for logs / JSON results / checkpoints.",
    )
    parser.add_argument(
        "--force-mock",
        action="store_true",
        help="Use the dependency-free mock env (no IsaacGym / GPU required).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run IsaacGym headless (default: True).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )


def _add_scale_args(parser: argparse.ArgumentParser) -> None:
    """Environment / rollout scale arguments."""
    parser.add_argument(
        "--num-envs",
        type=int,
        default=24576,
        help="Total number of parallel environments N (paper: 24576).",
    )
    parser.add_argument(
        "--num-policies",
        type=int,
        default=6,
        help="Number of policies M = 1 leader + (M-1) followers (paper: 6).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Rollout horizon H per env (paper: 16 AllegroKuka, 8 hands).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help="Stop after this many environment transitions (paper: 2e10).",
    )


def _add_optim_args(parser: argparse.ArgumentParser) -> None:
    """Optimization hyperparameters (Appendix B, Tables 2-4)."""
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--critic-coef", type=float, default=4.0)
    parser.add_argument("--lambda-off", type=float, default=1.0)
    parser.add_argument("--bounds-coef", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mini-epochs", type=int, default=None)
    parser.add_argument("--num-mini-batches", type=int, default=4)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument(
        "--learnable-entropy-coef",
        action="store_true",
        help="Use per-block learnable entropy coefficients.",
    )
    parser.add_argument(
        "--no-kl-adaptive-lr",
        action="store_true",
        help="Disable KL-adaptive learning-rate scheduling.",
    )
    parser.add_argument("--kl-threshold", type=float, default=0.016)
    parser.add_argument("--kl-adaptive-factor", type=float, default=1.5)
    parser.add_argument(
        "--no-normalize-advantage",
        action="store_true",
        help="Disable advantage normalization.",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Model architecture arguments."""
    parser.add_argument("--phi-dim", type=int, default=None)
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        default=None,
        help="Use LSTM backbone (AllegroKuka tasks).",
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        default="concat",
        choices=["concat", "film"],
        help="How phi_j conditions the shared backbone.",
    )


def _add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=50)


# ---------------------------------------------------------------------------
# Sub-command builders
# ---------------------------------------------------------------------------
def build_train_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("train", help="Train SAPG (Algorithm 1).")
    _add_common_args(p)
    _add_scale_args(p)
    _add_optim_args(p)
    _add_model_args(p)
    _add_eval_args(p)
    # Ablation switches (also exposed via the dedicated `ablate` command).
    p.add_argument("--symmetric", action="store_true",
                   help="Ablation: every policy uses all others' data.")
    p.add_argument("--no-off-policy", action="store_true",
                   help="Ablation: disable leader off-policy aggregation.")
    p.add_argument("--no-subsample", action="store_true",
                   help="Ablation: high off-policy ratio (no subsampling).")
    return p


def build_baseline_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("baseline", help="Train PPO / PBT / PQL baselines.")
    _add_common_args(p)
    _add_scale_args(p)
    _add_optim_args(p)
    _add_model_args(p)
    _add_eval_args(p)
    p.add_argument("--algo", type=str, default="ppo",
                   choices=["ppo", "pbt", "pql"], help="Baseline algorithm.")
    p.add_argument("--population-size", type=int, default=6)
    p.add_argument("--pbt-interval", type=int, default=10)
    p.add_argument("--pbt-exploit-frac", type=float, default=0.2)
    p.add_argument("--pbt-perturb", type=float, default=0.1)
    p.add_argument("--pql-buffer-size", type=int, default=1_000_000)
    p.add_argument("--pql-batch-size", type=int, default=4096)
    p.add_argument("--pql-updates-per-iter", type=int, default=1)
    p.add_argument("--pql-prior-frac", type=float, default=0.25)
    p.add_argument("--pql-alpha", type=float, default=0.2)
    p.add_argument("--pql-tau", type=float, default=0.005)
    return p


def build_ablate_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("ablate", help="Run SAPG ablations (Figure 6).")
    _add_common_args(p)
    _add_scale_args(p)
    _add_optim_args(p)
    _add_model_args(p)
    _add_eval_args(p)
    p.add_argument(
        "--ablation",
        type=str,
        default="all",
        choices=[
            "all",
            "entropy",
            "symmetric",
            "no_off_policy",
            "high_off_policy_ratio",
        ],
        help="Which ablation to run.",
    )
    p.add_argument("--entropy-coefs", type=float, nargs="+",
                   default=[0.0, 0.003, 0.005],
                   help="Entropy coefficients to sweep.")
    return p


def build_diversity_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("diversity", help="Diversity metrics (Figures 7-8).")
    _add_common_args(p)
    _add_scale_args(p)
    _add_model_args(p)
    p.add_argument("--num-transitions", type=int, default=400_000,
                   help="State-transitions used for the MLP reconstruction metric.")
    p.add_argument("--pca-components", type=int, nargs="+",
                   default=[1, 2, 4, 8, 16, 32, 64, 128],
                   help="Top-k PCA components for the reconstruction metric.")
    p.add_argument("--mlp-hidden-sizes", type=int, nargs="+",
                   default=[8, 16, 32, 64, 128, 256],
                   help="Hidden sizes for the MLP reconstruction metric.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Optional SAPG checkpoint to analyse.")
    return p


def build_plot_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("plot", help="Plot results (Figures 5-6, Table 1).")
    _add_common_args(p)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--figures-dir", type=str, default="figures")
    p.add_argument("--smooth", type=int, default=1,
                   help="Moving-average window for curves.")
    return p


def build_eval_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = sub.add_parser("eval", help="Evaluate a saved checkpoint.")
    _add_common_args(p)
    _add_scale_args(p)
    _add_model_args(p)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a saved actor-critic checkpoint (.pt).")
    p.add_argument("--policy-index", type=int, default=0,
                   help="Which policy to evaluate (0 = leader).")
    p.add_argument("--episodes", type=int, default=32)
    return p


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def _dispatch_train(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.train_sapg import SAPGTrainConfig, train_sapg

    cfg = SAPGTrainConfig(
        task=args.task,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        n_step=args.n_step,
        clip_eps=args.clip_eps,
        critic_coef=args.critic_coef,
        lambda_off=args.lambda_off,
        bounds_coef=args.bounds_coef,
        max_grad_norm=args.max_grad_norm,
        mini_epochs=args.mini_epochs,
        num_mini_batches=args.num_mini_batches,
        use_kl_adaptive_lr=not args.no_kl_adaptive_lr,
        kl_threshold=args.kl_threshold,
        kl_adaptive_factor=args.kl_adaptive_factor,
        normalize_advantage=not args.no_normalize_advantage,
        entropy_coef=args.entropy_coef,
        learnable_entropy_coef=args.learnable_entropy_coef,
        phi_dim=args.phi_dim,
        use_lstm=args.use_lstm,
        conditioning=args.conditioning,
        seed=args.seed,
        device=args.device,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        symmetric=args.symmetric,
        use_off_policy=not args.no_off_policy,
        subsample_off_policy=not args.no_subsample,
        force_mock=args.force_mock,
        headless=args.headless,
    )
    return train_sapg(cfg)


def _dispatch_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.train_baselines import BaselineConfig, main as baseline_main

    cfg = BaselineConfig(
        algo=args.algo,
        task=args.task,
        num_envs=args.num_envs,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        n_step=args.n_step,
        clip_eps=args.clip_eps,
        critic_coef=args.critic_coef,
        entropy_coef=args.entropy_coef,
        bounds_coef=args.bounds_coef,
        max_grad_norm=args.max_grad_norm,
        mini_epochs=args.mini_epochs,
        num_mini_batches=args.num_mini_batches,
        use_kl_adaptive_lr=not args.no_kl_adaptive_lr,
        kl_threshold=args.kl_threshold,
        kl_adaptive_factor=args.kl_adaptive_factor,
        normalize_advantage=not args.no_normalize_advantage,
        seed=args.seed,
        device=args.device,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        population_size=args.population_size,
        pbt_interval=args.pbt_interval,
        pbt_exploit_frac=args.pbt_exploit_frac,
        pbt_perturb=args.pbt_perturb,
        pql_buffer_size=args.pql_buffer_size,
        pql_batch_size=args.pql_batch_size,
        pql_updates_per_iter=args.pql_updates_per_iter,
        pql_prior_frac=args.pql_prior_frac,
        pql_alpha=args.pql_alpha,
        pql_tau=args.pql_tau,
    )
    return baseline_main(cfg)


def _dispatch_ablate(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.ablations import AblationConfig, run_ablation

    cfg = AblationConfig(
        ablation=args.ablation,
        task=args.task,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        n_step=args.n_step,
        clip_eps=args.clip_eps,
        critic_coef=args.critic_coef,
        lambda_off=args.lambda_off,
        bounds_coef=args.bounds_coef,
        max_grad_norm=args.max_grad_norm,
        mini_epochs=args.mini_epochs,
        num_mini_batches=args.num_mini_batches,
        use_kl_adaptive_lr=not args.no_kl_adaptive_lr,
        kl_threshold=args.kl_threshold,
        kl_adaptive_factor=args.kl_adaptive_factor,
        normalize_advantage=not args.no_normalize_advantage,
        entropy_coefs=list(args.entropy_coefs),
        phi_dim=args.phi_dim,
        use_lstm=args.use_lstm,
        conditioning=args.conditioning,
        seed=args.seed,
        device=args.device,
        max_iterations=args.max_iterations,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        force_mock=args.force_mock,
        headless=args.headless,
    )
    return run_ablation(cfg)


def _dispatch_diversity(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.diversity_metrics import (
        DiversityConfig,
        run_diversity_analysis,
    )

    cfg = DiversityConfig(
        task=args.task,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon,
        phi_dim=args.phi_dim,
        use_lstm=args.use_lstm,
        conditioning=args.conditioning,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        num_transitions=args.num_transitions,
        pca_components=list(args.pca_components),
        mlp_hidden_sizes=list(args.mlp_hidden_sizes),
        checkpoint=args.checkpoint,
    )
    return run_diversity_analysis(cfg)


def _dispatch_plot(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.plot_results import PlotConfig, plot_all

    cfg = PlotConfig(
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        smooth=args.smooth,
    )
    return plot_all(cfg)


def _dispatch_eval(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from sapg.envs import make_task_env
    from sapg.experiments.train_sapg import evaluate, task_defaults
    from sapg.sapg.models import build_actor_critic
    from sapg.sapg.utils import get_device

    device = get_device(args.device)
    env = make_task_env(
        args.task,
        num_envs=args.num_envs,
        device=str(device),
        headless=args.headless,
        seed=args.seed,
        force_mock=args.force_mock,
    )
    defaults = task_defaults(args.task)
    phi_dim = args.phi_dim if args.phi_dim is not None else defaults.get("phi_dim")
    use_lstm = args.use_lstm if args.use_lstm is not None else defaults.get("use_lstm", False)

    actor_critic = build_actor_critic(
        args.task,
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        num_policies=args.num_policies,
        conditioning=args.conditioning,
        phi_dim=phi_dim,
        use_lstm=use_lstm,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    actor_critic.load_state_dict(state)
    actor_critic.eval()

    metrics = evaluate(
        env,
        actor_critic,
        policy_index=args.policy_index,
        episodes=args.episodes,
        deterministic=True,
        device=device,
    )
    env.close()
    LOGGER.info("Evaluation on %s: %s", args.task, json.dumps(metrics, indent=2))
    return metrics


_DISPATCH = {
    "train": _dispatch_train,
    "baseline": _dispatch_baseline,
    "ablate": _dispatch_ablate,
    "diversity": _dispatch_diversity,
    "plot": _dispatch_plot,
    "eval": _dispatch_eval,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sapg",
        description="SAPG: Split and Aggregate Policy Gradients.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build_train_parser(sub)
    build_baseline_parser(sub)
    build_ablate_parser(sub)
    build_diversity_parser(sub)
    build_plot_parser(sub)
    build_eval_parser(sub)
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        LOGGER.setLevel("DEBUG")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    LOGGER.info("Running command '%s' on task '%s' (seed=%d).",
                args.command, args.task, args.seed)
    handler = _DISPATCH[args.command]
    result = handler(args)
    LOGGER.info("Command '%s' finished.", args.command)
    return result


if __name__ == "__main__":
    main()
