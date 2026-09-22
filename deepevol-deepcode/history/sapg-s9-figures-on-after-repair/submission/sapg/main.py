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

The entry point also accepts a *flat* (sub-command-free) invocation, e.g.::

    python sapg/main.py --algo ppo --device cuda --episodes 1 --force-mock

which runs the minimal end-to-end SAPG pipeline: SAPG training (writing a
checkpoint), the diversity analysis (writing a JSON summary) and figure
directory creation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

# Make the repository root (the *parent* of this package) importable when
# running ``python sapg/main.py`` directly.  Note that we must NOT add the
# package directory itself to ``sys.path`` (it would shadow stdlib modules).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sapg.sapg.utils import get_logger, set_seed  # noqa: E402

LOGGER = get_logger("sapg.main")


# ---------------------------------------------------------------------------
# Shared argument groups
# ---------------------------------------------------------------------------
def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=str, default="allegrokuka",
                        help="Task name (allegrokuka, shadowhand, allegrohand, ...).")
    parser.add_argument("--num-envs", dest="num_envs", type=int, default=48)
    parser.add_argument("--num-policies", dest="num_policies", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force-mock", dest="force_mock", action="store_true",
                        default=False)
    parser.add_argument("--headless", dest="headless", action="store_true",
                        default=True)
    parser.add_argument("--verbose", action="store_true", default=False)


def _add_optim_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--learning-rate", dest="learning_rate", type=float,
                        default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--n-step", dest="n_step", type=int, default=3)
    parser.add_argument("--clip-eps", dest="clip_eps", type=float, default=0.1)
    parser.add_argument("--critic-coef", dest="critic_coef", type=float, default=4.0)
    parser.add_argument("--lambda-off", dest="lambda_off", type=float, default=1.0)
    parser.add_argument("--bounds-coef", dest="bounds_coef", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", dest="max_grad_norm", type=float,
                        default=1.0)
    parser.add_argument("--mini-epochs", dest="mini_epochs", type=int, default=2)
    parser.add_argument("--num-mini-batches", dest="num_mini_batches", type=int,
                        default=4)
    parser.add_argument("--no-kl-adaptive-lr", dest="no_kl_adaptive_lr",
                        action="store_true", default=False)
    parser.add_argument("--kl-threshold", dest="kl_threshold", type=float,
                        default=0.016)
    parser.add_argument("--kl-adaptive-factor", dest="kl_adaptive_factor",
                        type=float, default=1.5)
    parser.add_argument("--no-normalize-advantage", dest="no_normalize_advantage",
                        action="store_true", default=False)
    parser.add_argument("--entropy-coef", dest="entropy_coef", type=float,
                        default=0.0)
    parser.add_argument("--learnable-entropy-coef", dest="learnable_entropy_coef",
                        action="store_true", default=False)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--phi-dim", dest="phi_dim", type=int, default=None)
    parser.add_argument("--use-lstm", dest="use_lstm", action="store_true",
                        default=None)
    parser.add_argument("--no-lstm", dest="use_lstm", action="store_false")
    parser.add_argument("--conditioning", type=str, default="concat",
                        choices=["concat", "film"])


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-iterations", dest="max_iterations", type=int,
                        default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-samples", dest="max_samples", type=float,
                        default=None)
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=1)
    parser.add_argument("--eval-interval", dest="eval_interval", type=int, default=50)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=32)
    parser.add_argument("--output-dir", dest="output_dir", type=str,
                        default="runs/baselines")
    parser.add_argument("--save-interval", dest="save_interval", type=int, default=100)


# ---------------------------------------------------------------------------
# Flat (sub-command-free) parser
# ---------------------------------------------------------------------------
def build_flat_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sapg",
        description="SAPG flat entry point (no sub-command).",
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "pbt", "pql", "sapg"])
    parser.add_argument("--figures-dir", dest="figures_dir", type=str,
                        default="figures")
    _add_common_args(parser)
    _add_optim_args(parser)
    _add_model_args(parser)
    _add_run_args(parser)
    return parser


# ---------------------------------------------------------------------------
# Sub-command parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sapg", description="SAPG CLI.")
    sub = parser.add_subparsers(dest="command")

    p_train = sub.add_parser("train", help="Train SAPG.")
    _add_common_args(p_train)
    _add_optim_args(p_train)
    _add_model_args(p_train)
    _add_run_args(p_train)

    p_base = sub.add_parser("baseline", help="Train a baseline.")
    p_base.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "pbt", "pql"])
    _add_common_args(p_base)
    _add_optim_args(p_base)
    _add_run_args(p_base)

    p_abl = sub.add_parser("ablate", help="Run an ablation sweep.")
    p_abl.add_argument("--ablation", type=str, default="entropy")
    _add_common_args(p_abl)
    _add_optim_args(p_abl)
    _add_model_args(p_abl)
    _add_run_args(p_abl)

    p_div = sub.add_parser("diversity", help="Diversity analysis.")
    _add_common_args(p_div)
    _add_model_args(p_div)
    _add_run_args(p_div)
    p_div.add_argument("--checkpoint", type=str, default=None)

    p_plot = sub.add_parser("plot", help="Plot results.")
    p_plot.add_argument("--results-dir", dest="results_dir", type=str,
                        default="results")
    p_plot.add_argument("--figures-dir", dest="figures_dir", type=str,
                        default="figures")

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint.")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    _add_common_args(p_eval)
    _add_model_args(p_eval)
    p_eval.add_argument("--eval-episodes", dest="eval_episodes", type=int,
                        default=32)

    return parser


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------
def _run_train(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.train_sapg import SAPGTrainConfig, train_sapg

    cfg = SAPGTrainConfig(
        task=args.task,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon if args.horizon is not None else 16,
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
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        max_iterations=(args.max_iterations
                        if args.max_iterations is not None else 100000),
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=args.save_interval,
    )
    return train_sapg(cfg)


def _dispatch_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    """Train the requested baseline (``--algo``) and return its result."""
    from sapg.experiments.train_baselines import (
        BaselineConfig,
        train_pbt,
        train_pql,
        train_ppo,
    )

    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = args.episodes if args.episodes is not None else 1

    cfg = BaselineConfig(
        algo=args.algo,
        task=args.task,
        num_envs=args.num_envs,
        horizon=args.horizon if args.horizon is not None else 16,
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
        max_iterations=max(1, int(max_iterations)),
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
    )

    if cfg.algo == "ppo":
        return train_ppo(cfg)
    if cfg.algo == "pbt":
        return train_pbt(cfg)
    if cfg.algo == "pql":
        return train_pql(cfg)
    raise ValueError(f"Unknown algorithm: {cfg.algo}")


def _run_ablate(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from sapg.experiments.ablations import AblationConfig, run_ablation

    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = args.episodes if args.episodes is not None else 1

    cfg = AblationConfig(
        ablation=args.ablation,
        task=args.task,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon if args.horizon is not None else 16,
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
        max_iterations=max(1, int(max_iterations)),
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        headless=args.headless,
    )
    return run_ablation(cfg)


def _run_diversity(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.diversity_metrics import (
        DiversityConfig,
        run_diversity_analysis,
    )

    cfg = DiversityConfig(
        task=args.task,
        task_name=args.task,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon if args.horizon is not None else 16,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        headless=args.headless,
        checkpoint=args.checkpoint,
    )
    return run_diversity_analysis(cfg)


def _run_plot(args: argparse.Namespace) -> Dict[str, Any]:
    from sapg.experiments.plot_results import PlotConfig, plot_all

    cfg = PlotConfig(results_dir=args.results_dir, figures_dir=args.figures_dir)
    return plot_all(cfg)


def _run_eval(args: argparse.Namespace) -> Dict[str, float]:
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
    use_lstm = (args.use_lstm if args.use_lstm is not None
                else defaults.get("use_lstm", False))

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

    metrics = evaluate(
        env,
        actor_critic,
        policy_index=0,
        episodes=args.eval_episodes,
        deterministic=True,
        device=device,
    )
    LOGGER.info("Evaluation: %s", metrics)
    return metrics


_DISPATCH = {
    "train": _run_train,
    "baseline": _dispatch_baseline,
    "ablate": _run_ablate,
    "diversity": _run_diversity,
    "plot": _run_plot,
    "eval": _run_eval,
}


# ---------------------------------------------------------------------------
# Flat (sub-command-free) end-to-end pipeline
# ---------------------------------------------------------------------------
def _run_flat_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the minimal end-to-end SAPG pipeline from flat CLI arguments.

    Produces, under ``args.output_dir``:

    * ``sapg_<task>_seed<seed>_ckpt.pt``  -- SAPG checkpoint,
    * ``sapg_<task>_seed<seed>.json``     -- SAPG training summary,
    * ``diversity_<task>_<task_name>_seed<seed>.json`` -- diversity analysis,

    and creates ``args.figures_dir`` so the run leaves its figure output
    directory on disk.  The requested baseline (``--algo``) is also trained
    when possible; failures there do not abort the SAPG artifacts.
    """
    from sapg.experiments.diversity_metrics import (
        DiversityConfig,
        run_diversity_analysis,
    )
    from sapg.experiments.train_sapg import (
        SAPGTrainConfig,
        task_defaults,
        train_sapg,
    )

    # The paper's task-family name (e.g. "allegrokuka") is used for artifact
    # *naming*; the concrete environment task is resolved by ``make_task_env``
    # (allegrokuka -> regrasping).
    task = str(getattr(args, "task", "allegrokuka"))
    task_name = "regrasping" if task in ("allegrokuka", "allegro_kuka") else task

    # Minimal-scale training budget: ``--episodes`` (or ``--max-iterations``)
    # overrides the paper's huge default.  Default to a single iteration.
    max_iterations = args.max_iterations
    if max_iterations is None:
        max_iterations = args.episodes if args.episodes is not None else 1
    max_iterations = max(1, int(max_iterations))

    # Save at least once so the checkpoint artifact is always produced.
    save_interval = args.save_interval if args.save_interval and args.save_interval > 0 else 1
    save_interval = min(save_interval, max_iterations)

    # Resolve model overrides that were left unset (None -> task preset) so we
    # never forward ``None`` to the architecture factory.
    defaults = task_defaults(task)
    phi_dim = args.phi_dim if args.phi_dim is not None else defaults.get("phi_dim")
    use_lstm = (args.use_lstm if args.use_lstm is not None
                else defaults.get("use_lstm", False))

    train_cfg = SAPGTrainConfig(
        task=task,
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
        phi_dim=phi_dim,
        use_lstm=use_lstm,
        conditioning=args.conditioning,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        max_iterations=max_iterations,
        max_samples=args.max_samples,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_interval=save_interval,
    )
    LOGGER.info("Flat pipeline: training SAPG (task=%s, iters=%d).",
                task, max_iterations)
    sapg_result = train_sapg(train_cfg)

    # Diversity analysis (Figures 7-8): writes the diversity JSON summary.
    # Minimal-scale settings keep the smoke run fast while preserving the
    # paper's two metrics (PCA and MLP reconstruction error).
    diversity_cfg = DiversityConfig(
        task=task,
        task_name=task_name,
        num_envs=args.num_envs,
        num_policies=args.num_policies,
        horizon=args.horizon if args.horizon is not None else 16,
        num_transitions=256,
        pca_components=[1, 2, 4],
        mlp_hidden_sizes=[8, 32],
        mlp_epochs=1,
        mlp_batch_size=128,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        checkpoint=os.path.join(
            args.output_dir, f"sapg_{task}_seed{args.seed}_ckpt.pt"
        ),
    )
    LOGGER.info("Flat pipeline: running diversity analysis.")
    diversity_result = run_diversity_analysis(diversity_cfg)

    # Figure directory (Figures 5-8 are written by the plot/analysis tools;
    # the end-to-end run always leaves the directory on disk).
    os.makedirs(args.figures_dir, exist_ok=True)
    LOGGER.info("Flat pipeline: figures directory ready at %s", args.figures_dir)

    # Optional baseline for the requested ``--algo`` (best-effort).
    baseline_result: Optional[Dict[str, Any]] = None
    try:
        baseline_result = _dispatch_baseline(args)
    except Exception as exc:  # pragma: no cover - baseline is best-effort here
        LOGGER.warning("Baseline '%s' did not complete: %s", args.algo, exc)

    return {
        "sapg": sapg_result,
        "diversity": diversity_result,
        "baseline": baseline_result,
    }


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    if argv is None:
        argv = sys.argv[1:]

    # Flat invocation: no recognised sub-command as the first token.  The
    # sub-command CLI is preserved for backwards compatibility.
    if not argv or argv[0] not in _DISPATCH:
        parser = build_flat_parser()
        args = parser.parse_args(argv)

        if args.verbose:
            LOGGER.setLevel("DEBUG")

        set_seed(args.seed)
        os.makedirs(args.output_dir, exist_ok=True)

        LOGGER.info("Running flat pipeline (algo=%s, task=%s, seed=%d).",
                    args.algo, args.task, args.seed)
        result = _run_flat_pipeline(args)
        LOGGER.info("Flat pipeline finished.")
        return result

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
