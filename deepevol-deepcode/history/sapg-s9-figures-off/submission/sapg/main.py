"""SAPG entry point.

Parses command-line arguments, selects the task/algorithm configuration, builds
the environment + trainer, and launches training (Algorithm 1 of the paper).

Usage examples
--------------
    # Train SAPG on the Regrasping task with default hyperparameters
    python main.py --task regrasping --algorithm sapg

    # Train vanilla PPO on ShadowHand
    python main.py --task shadowhand --algorithm ppo

    # Override hyperparameters
    python main.py --task throw --algorithm sapg --num-envs 24576 --num-blocks 6 \
        --entropy-coef 0.005 --seed 3

    # Smoke test with a tiny run (no IsaacGym required)
    python main.py --task regrasping --algorithm sapg --num-envs 64 --num-blocks 2 \
        --horizon 4 --num-iterations 2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Make the repository importable both as a package and as a script.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sapg.config import (  # noqa: E402
    BASELINE_CONFIGS,
    EXPECTED_RESULTS,
    TASK_CONFIGS,
    SAPGConfig,
    get_config,
)


# ---------------------------------------------------------------------------
# Task metadata (obs/action dims per task family, Sec 5.1 / Appendix A)
# ---------------------------------------------------------------------------
TASK_METADATA: Dict[str, Dict[str, Any]] = {
    # AllegroKuka (23 DoF): obs o_t = [q, q_dot, x_t, v_t, omega_t, g_t, z_t]
    "regrasping": {"family": "allegrokuka", "obs_dim": 68, "action_dim": 23},
    "throw": {"family": "allegrokuka", "obs_dim": 68, "action_dim": 23},
    "reorientation": {"family": "allegrokuka", "obs_dim": 68, "action_dim": 23},
    # ShadowHand (24 DoF), goal quaternion g_t in R^4
    "shadowhand": {"family": "shadowhand", "obs_dim": 100, "action_dim": 24},
    # AllegroHand (16 DoF)
    "allegrohand": {"family": "allegrohand", "obs_dim": 72, "action_dim": 16},
}

ALGORITHMS = ("sapg", "ppo", "pql", "dexpbt")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sapg",
        description="SAPG: Split and Aggregate Policy Gradients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- core selection -----------------------------------------------------
    parser.add_argument(
        "--task",
        type=str,
        default="regrasping",
        choices=sorted(TASK_CONFIGS.keys()),
        help="Task to train on.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="sapg",
        choices=ALGORITHMS,
        help="Algorithm to run (sapg or a baseline).",
    )

    # --- environment / splitting -------------------------------------------
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Total number of parallel environments N.")
    parser.add_argument("--num-blocks", type=int, default=None,
                        help="Number of env blocks M (one policy per block).")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Rollout horizon T per env instance per update.")

    # --- optimization -------------------------------------------------------
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-mini-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size-factor", type=int, default=None)
    parser.add_argument("--clip-eps", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--grad-norm-clip", type=float, default=None)
    parser.add_argument("--kl-threshold", type=float, default=None)
    parser.add_argument("--no-adaptive-lr", action="store_true",
                        help="Disable KL-based adaptive learning rate.")

    # --- SAPG specifics -----------------------------------------------------
    parser.add_argument("--aggregation", type=str, default=None,
                        choices=["leader_follower", "symmetric",
                                 "high_off_policy_ratio", "no_off_policy"],
                        help="Data aggregation strategy (Sec 4.2-4.3).")
    parser.add_argument("--off-policy-coef", type=float, default=None,
                        help="lambda in Eq. 4 (default 1.0).")
    parser.add_argument("--critic-coef", type=float, default=None,
                        help="lambda' applied to the total critic loss (default 4.0).")
    parser.add_argument("--entropy-coef", type=float, default=None,
                        help="sigma in the follower entropy term sigma*(i-1)*H.")
    parser.add_argument("--no-subsample-off-policy", action="store_true",
                        help="Use ALL off-policy data (high_off_policy_ratio ablation).")
    parser.add_argument("--latent-dim", type=int, default=None,
                        help="Dimension of the per-policy latent phi_j.")

    # --- run control --------------------------------------------------------
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Number of outer training iterations.")
    parser.add_argument("--total-transitions", type=float, default=None,
                        help="Target number of environment transitions (~2e10).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs",
                        help="Directory for logs and checkpoints.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name of the run (defaults to task_algorithm_seed).")

    # --- misc ---------------------------------------------------------------
    parser.add_argument("--dry-run", action="store_true",
                        help="Use a lightweight dummy env (no IsaacGym required).")
    parser.add_argument("--print-config", action="store_true",
                        help="Print the resolved config and exit.")
    parser.add_argument("--expected-results", action="store_true",
                        help="Print the paper's expected Table 1 results and exit.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List available tasks and exit.")

    return parser


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
# Fields that may be overridden from the command line.
_OVERRIDE_FIELDS = (
    "num_envs",
    "num_blocks",
    "horizon",
    "learning_rate",
    "num_mini_epochs",
    "minibatch_size_factor",
    "clip_eps",
    "gamma",
    "gae_lambda",
    "grad_norm_clip",
    "kl_threshold",
    "aggregation",
    "off_policy_coef",
    "critic_coef",
    "entropy_coef",
    "latent_dim",
    "num_iterations",
    "total_transitions",
    "seed",
    "log_interval",
    "save_interval",
)


def resolve_config(args: argparse.Namespace) -> SAPGConfig:
    """Build a config from CLI args, applying only explicitly-set overrides."""
    overrides: Dict[str, Any] = {}

    for field in _OVERRIDE_FIELDS:
        cli_name = field.replace("_", "-")
        value = getattr(args, field, None)
        if value is not None:
            overrides[field] = value

    # Boolean flags -> config fields
    if args.no_adaptive_lr:
        overrides["adaptive_lr"] = False
    if args.no_subsample_off_policy:
        overrides["subsample_off_policy"] = False
        # The high-off-policy-ratio ablation is exactly "no subsampling".
        overrides.setdefault("aggregation", "high_off_policy_ratio")

    if args.device is not None:
        overrides["device"] = args.device

    config = get_config(args.task, algorithm=args.algorithm, **overrides)
    return config


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------
def build_env(config: SAPGConfig, dry_run: bool = False):
    """Construct the vectorized environment for the given config.

    Falls back to a lightweight dummy environment when IsaacGym is unavailable
    or when ``--dry-run`` is requested, so the training loop can be smoke-tested
    on any machine.
    """
    if not dry_run:
        try:
            from envs.isaacgym_wrapper import make_env

            return make_env(config)
        except Exception as exc:  # pragma: no cover - depends on IsaacGym
            print(f"[main] IsaacGym env unavailable ({exc}); "
                  f"falling back to dummy env.", file=sys.stderr)

    from envs.isaacgym_wrapper import DummyVectorEnv

    meta = TASK_METADATA[config.task]
    return DummyVectorEnv(
        num_envs=config.num_envs,
        obs_dim=meta["obs_dim"],
        action_dim=meta["action_dim"],
        horizon=config.horizon,
        seed=config.seed,
    )


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------
def run_sapg(config: SAPGConfig, env, obs_dim: int, action_dim: int,
             output_dir: str, run_name: str) -> Dict[str, Any]:
    from sapg.algorithm import build_trainer

    trainer = build_trainer(config, env, obs_dim, action_dim)
    return _run_trainer(trainer, config, output_dir, run_name)


def run_baseline(config: SAPGConfig, env, obs_dim: int, action_dim: int,
                 output_dir: str, run_name: str) -> Dict[str, Any]:
    algorithm = config.algorithm if hasattr(config, "algorithm") else "ppo"

    if algorithm == "ppo":
        from baselines.ppo import PPOTrainer, build_ppo_trainer

        trainer = build_ppo_trainer(config, env, obs_dim, action_dim)
    elif algorithm == "pql":
        from baselines.pql import PQLTrainer, build_pql_trainer

        trainer = build_pql_trainer(config, env, obs_dim, action_dim)
    elif algorithm == "dexpbt":
        from baselines.dexpbt import DexPBTTrainer, build_dexpbt_trainer

        trainer = build_dexpbt_trainer(config, env, obs_dim, action_dim)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown baseline algorithm: {algorithm}")

    return _run_trainer(trainer, config, output_dir, run_name)


def _run_trainer(trainer, config: SAPGConfig, output_dir: str,
                 run_name: str) -> Dict[str, Any]:
    """Run a trainer and persist its history."""
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    history = trainer.train(config.num_iterations)
    elapsed = time.time() - start

    # Persist history + final checkpoint.
    history_path = os.path.join(output_dir, f"{run_name}_history.json")
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)

    try:
        ckpt_path = os.path.join(output_dir, f"{run_name}_final.pt")
        import torch

        torch.save(trainer.state_dict(), ckpt_path)
    except Exception as exc:  # pragma: no cover
        print(f"[main] Could not save checkpoint: {exc}", file=sys.stderr)

    print(f"[main] Finished {run_name} in {elapsed:.1f}s "
          f"({len(history)} iterations). History -> {history_path}")
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        print("Available tasks:")
        for name, meta in TASK_METADATA.items():
            print(f"  {name:15s} family={meta['family']:12s} "
                  f"obs_dim={meta['obs_dim']:3d} action_dim={meta['action_dim']}")
        return 0

    if args.expected_results:
        print(json.dumps(EXPECTED_RESULTS, indent=2))
        return 0

    config = resolve_config(args)

    if args.print_config:
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    meta = TASK_METADATA[config.task]
    obs_dim = meta["obs_dim"]
    action_dim = meta["action_dim"]

    run_name = args.run_name or f"{config.task}_{args.algorithm}_seed{config.seed}"
    output_dir = os.path.join(args.output_dir, run_name)

    print("=" * 72)
    print(f"SAPG :: task={config.task} algorithm={args.algorithm} "
          f"seed={config.seed}")
    print(f"  num_envs={config.num_envs} num_blocks={config.num_blocks} "
          f"horizon={config.horizon} latent_dim={config.latent_dim}")
    print(f"  aggregation={config.aggregation} "
          f"off_policy_coef={config.off_policy_coef} "
          f"critic_coef={config.critic_coef} entropy_coef={config.entropy_coef}")
    print(f"  lr={config.learning_rate} mini_epochs={config.num_mini_epochs} "
          f"clip_eps={config.clip_eps} device={config.device}")
    print(f"  output_dir={output_dir}")
    print("=" * 72)

    env = build_env(config, dry_run=args.dry_run)

    if args.algorithm == "sapg":
        history = run_sapg(config, env, obs_dim, action_dim, output_dir, run_name)
    else:
        history = run_baseline(config, env, obs_dim, action_dim, output_dir, run_name)

    if history:
        last = history[-1]
        print(f"[main] Final iteration {last.get('iteration')}: "
              f"transitions={last.get('total_transitions')} "
              f"policy_loss={last.get('policy_loss'):.4f} "
              f"value_loss={last.get('value_loss'):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
