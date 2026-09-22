#!/usr/bin/env python
"""Main training entry point for SAPG and its baselines.

Usage examples
--------------
    python scripts/train.py --task regrasping --method sapg
    python scripts/train.py --config configs/allegro_kuka.yaml --method sapg
    python scripts/train.py --task shadow_hand --method ppo --num-envs 24576
    python scripts/train.py --task reorientation --method sapg \\
        --entropy-coefficient 0.005 --aggregation leader_follower

The script is intentionally thin: it resolves a configuration (YAML file +
command-line overrides), builds the vectorised environment and the policy
network(s), optionally attaches the success-tolerance curriculum, and hands
everything to the appropriate trainer:

* ``sapg``   -> :func:`sapg.algorithms.sapg.train_sapg`   (Algorithm 1)
* ``ppo``    -> :func:`sapg.baselines.ppo_baseline.train_ppo_baseline`  (§5.2)
* ``pql``    -> :func:`sapg.baselines.pql.train_pql`      (§5.2)
* ``dexpbt`` -> :func:`sapg.baselines.dexpbt.train_dexpbt` (§5.2)

All heavy dependencies (torch, IsaacGym) are imported lazily so that
``--help`` and configuration validation work on any machine.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Make the repository importable when the script is executed directly
# (``python scripts/train.py``) as well as as a module.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # .../sapg  (package root containing sapg/)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sapg import N, M  # noqa: E402  (paper constants: 24576 envs, 6 policies)
from sapg.utils.config import SAPGConfig, build_config, ensure_dir  # noqa: E402

__all__ = [
    "METHODS",
    "TASKS",
    "build_parser",
    "load_config_from_args",
    "make_env_for_config",
    "make_curriculum_for",
    "make_policy_for_config",
    "make_logger_for_config",
    "train",
    "main",
]

TASKS: Tuple[str, ...] = (
    "regrasping",
    "throw",
    "reorientation",
    "shadow_hand",
    "allegro_hand",
)

METHODS: Tuple[str, ...] = ("sapg", "ppo", "pql", "dexpbt")

#: Per-task default entropy coefficients from §5.2 (sigma sweep {0, .003, .005}).
TASK_ENTROPY: Dict[str, float] = {
    "regrasping": 0.0,
    "throw": 0.0,
    "shadow_hand": 0.0,
    "allegro_hand": 0.0,
    "reorientation": 0.005,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="train.py",
        description="Train SAPG (or a baseline) on the paper's IsaacGym tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        default="regrasping",
        choices=list(TASKS),
        help="Task to train on (Appendix A).",
    )
    parser.add_argument(
        "--method",
        default="sapg",
        choices=list(METHODS),
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file (see configs/).",
    )
    parser.add_argument("--num-envs", type=int, default=None, help="N (default 24576).")
    parser.add_argument(
        "--num-policies", type=int, default=None, help="M (default 6, §4.3)."
    )
    parser.add_argument(
        "--aggregation",
        default=None,
        choices=["leader_follower", "symmetric", "none"],
        help="Aggregation topology (§4.3 / §4.2 ablation).",
    )
    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=None,
        help="sigma in Eq. (10). Defaults per task from §5.2.",
    )
    parser.add_argument(
        "--off-policy-weight", type=float, default=None, help="lambda in Eq. (4)."
    )
    parser.add_argument(
        "--no-subsample",
        action="store_true",
        help="Disable the |D'_1| = |D_1| subsampling (high off-policy ratio ablation).",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--horizon-length", type=int, default=None)
    parser.add_argument("--mini-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu.")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of outer iterations (samples = N * horizon * iterations).",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help="Training budget in transitions (paper uses ~2e10).",
    )
    parser.add_argument(
        "--phi-dim", type=int, default=None, help="Latent dimension (32 or 16, §5.2)."
    )
    parser.add_argument(
        "--random-phi",
        action="store_true",
        help="Freeze phi at its random initialisation (diversity baseline).",
    )
    parser.add_argument(
        "--no-lstm",
        action="store_true",
        help="Force a feed-forward policy (default: LSTM on AllegroKuka tasks).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print training logs.")
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging."
    )
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="Force the pure-torch environment surrogate (no IsaacGym).",
    )
    parser.add_argument(
        "--resume", default=None, help="Checkpoint path to resume from."
    )
    return parser


# ---------------------------------------------------------------------------
# Configuration resolution
# ---------------------------------------------------------------------------
def load_config_from_args(args: argparse.Namespace) -> SAPGConfig:
    """Build an :class:`SAPGConfig` from a YAML file plus CLI overrides."""
    base: Optional[SAPGConfig] = None
    if getattr(args, "config", None):
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"config file not found: {args.config}")
        base = SAPGConfig.from_yaml(args.config)

    overrides: Dict[str, Any] = {}
    if getattr(args, "num_envs", None) is not None:
        overrides["num_envs"] = int(args.num_envs)
    if getattr(args, "num_policies", None) is not None:
        overrides["num_policies"] = int(args.num_policies)
    if getattr(args, "aggregation", None):
        overrides["aggregation"] = args.aggregation
    if getattr(args, "entropy_coefficient", None) is not None:
        overrides["entropy_coefficient"] = float(args.entropy_coefficient)
    if getattr(args, "off_policy_weight", None) is not None:
        overrides["off_policy_weight"] = float(args.off_policy_weight)
    if getattr(args, "no_subsample", False):
        overrides["subsample_off_policy"] = False
    if getattr(args, "learning_rate", None) is not None:
        overrides["learning_rate"] = float(args.learning_rate)
    if getattr(args, "clip_epsilon", None) is not None:
        overrides["clip_epsilon"] = float(args.clip_epsilon)
    if getattr(args, "horizon_length", None) is not None:
        overrides["horizon_length"] = int(args.horizon_length)
    if getattr(args, "mini_epochs", None) is not None:
        overrides["mini_epochs"] = int(args.mini_epochs)
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = int(args.seed)
    if getattr(args, "device", None):
        overrides["device"] = args.device
    if getattr(args, "log_dir", None):
        overrides["log_dir"] = args.log_dir
    if getattr(args, "phi_dim", None) is not None:
        overrides["phi_dim"] = int(args.phi_dim)
    if getattr(args, "random_phi", False):
        overrides["random_phi"] = True
    if getattr(args, "no_lstm", False):
        overrides["use_lstm"] = False

    if base is not None:
        merged = base.to_dict()
        merged.update(overrides)
        merged.setdefault("task", args.task)
        # The CLI task wins unless the YAML explicitly narrowed it.
        if not getattr(args, "config", None):
            merged["task"] = args.task
        else:
            merged["task"] = args.task if args.task else merged.get("task", args.task)
        config = SAPGConfig.from_dict(merged)
    else:
        config = build_config(args.task, **overrides)

    # Task-appropriate entropy default (§5.2) when the user said nothing.
    if getattr(args, "entropy_coefficient", None) is None and base is None:
        config.entropy_coefficient = TASK_ENTROPY.get(config.task, 0.0)

    # Method tag drives downstream dispatch.
    config.method = _normalise_method(args.method)
    if config.method != "sapg":
        # Baselines use a single policy.
        config.num_policies = 1
    if config.method == "sapg" and not getattr(args, "no_phi", False):
        pass  # phi_dim resolved per task by build_config / YAML
    return config


def _normalise_method(method: Optional[str]) -> str:
    name = str(method or "sapg").strip().lower().replace("-", "_")
    aliases = {
        "vanilla_ppo": "ppo",
        "ppo_baseline": "ppo",
        "pbt": "dexpbt",
        "expbt": "dexpbt",
        "parallel_q_learning": "pql",
        "apql": "pql",
    }
    return aliases.get(name, name)


# ---------------------------------------------------------------------------
# Environment / policy / curriculum construction
# ---------------------------------------------------------------------------
def make_env_for_config(config: SAPGConfig, force_surrogate: bool = False) -> Any:
    """Instantiate the vectorised environment for ``config``."""
    if force_surrogate:
        os.environ["SAPG_FORCE_SURROGATE"] = "1"
    from sapg.envs import make_env as _make_env

    try:
        env = _make_env(
            task=config.task,
            config=config,
            num_envs=config.num_envs,
            device=config.device,
        )
    except TypeError:
        # Older factories may not accept ``device``.
        env = _make_env(task=config.task, config=config, num_envs=config.num_envs)
    return env


def make_curriculum_for(config: SAPGConfig, env: Any) -> Any:
    """Attach the 7.5cm -> 1cm success-tolerance curriculum (Appendix A).

    Only tasks that use a success tolerance (the AllegroKuka family) get a
    curriculum; other tasks simply return ``None``.
    """
    if getattr(config, "task", "") not in ("regrasping", "throw", "reorientation"):
        return None
    try:
        from sapg.envs.curriculum import make_curriculum
    except Exception:  # pragma: no cover - curriculum is optional
        return None
    try:
        return make_curriculum(task=config.task, env=env, config=config)
    except TypeError:
        return make_curriculum(config.task, env=env)


def make_policy_for_config(config: SAPGConfig) -> Any:
    """Build the (shared-backbone, phi-conditioned) actor-critic policy."""
    from sapg.models.actor import ActorCritic, Policy  # noqa: F401

    kwargs: Dict[str, Any] = {
        "obs_dim": config.obs_dim,
        "action_dim": config.action_dim,
        "phi_dim": config.phi_dim,
        "num_policies": max(int(config.num_policies), 1),
        "mlp_units": tuple(config.actor_mlp_units),
        "activation": config.actor_activation,
        "use_lstm": bool(config.use_lstm),
        "lstm_hidden_size": config.lstm_hidden_size,
        "lstm_num_layers": config.lstm_num_layers,
        "per_block_sigma": bool(config.per_block_sigma),
        "action_scale": float(getattr(config, "action_scale", 1.0)),
        "random_phi": bool(getattr(config, "random_phi", False)),
        "config": config,
    }
    try:
        policy = ActorCritic(**kwargs)
    except TypeError:
        kwargs.pop("config", None)
        policy = ActorCritic(**kwargs)
    return policy


def make_logger_for_config(config: SAPGConfig, verbose: bool = False) -> Any:
    """Create an experiment logger (console + optional TensorBoard)."""
    try:
        from sapg.utils.logging import make_logger
    except Exception:  # pragma: no cover
        return None
    log_dir = ensure_dir(os.path.join(config.log_dir, f"{config.task}_{config.method}"))
    try:
        return make_logger(
            config=config, log_dir=log_dir, name=f"{config.task}-{config.method}",
            verbose=verbose,
        )
    except TypeError:
        return make_logger(config, log_dir=log_dir)


def set_seed(seed: int, env: Any = None) -> int:
    """Seed python/numpy/torch (best effort) and return the seed."""
    import random

    random.seed(seed)
    try:  # pragma: no cover - numpy optional
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    for name in ("seed", "set_seed"):
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                fn(seed)
            except Exception:
                pass
            break
    return seed


# ---------------------------------------------------------------------------
# Training dispatch
# ---------------------------------------------------------------------------
def train(
    config: SAPGConfig,
    env: Any = None,
    policy: Any = None,
    logger: Any = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[Any, List[Dict[str, float]]]:
    """Run the selected algorithm and return ``(trainer, history)``."""
    method = _normalise_method(getattr(config, "method", "sapg"))

    if method == "sapg":
        from sapg.algorithms.sapg import train_sapg

        return train_sapg(
            config,
            env=env,
            policy=policy,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            logger=logger,
            **kwargs,
        )

    if method == "ppo":
        from sapg.baselines.ppo_baseline import train_ppo_baseline

        result = train_ppo_baseline(
            config=config,
            env=env,
            policy=policy,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            logger=logger,
            device=config.device,
            **kwargs,
        )
        return _as_pair(result)

    if method == "pql":
        from sapg.baselines.pql import train_pql

        result = train_pql(
            config=config,
            env=env,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            logger=logger,
            device=config.device,
            **kwargs,
        )
        return _as_pair(result)

    if method == "dexpbt":
        from sapg.baselines.dexpbt import train_dexpbt

        result = train_dexpbt(
            config=config,
            env=env,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            logger=logger,
            device=config.device,
            **kwargs,
        )
        return _as_pair(result)

    raise ValueError(
        f"unknown method {method!r}; expected one of {sorted(METHODS)}"
    )


def _as_pair(result: Any) -> Tuple[Any, List[Dict[str, float]]]:
    """Normalise a trainer return value to ``(trainer, history)``."""
    if isinstance(result, tuple) and len(result) == 2:
        return result
    trainer = result
    history = getattr(trainer, "history", None) or []
    return trainer, history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = load_config_from_args(args)
    if args.seed is not None:
        config.seed = int(args.seed)

    if args.verbose:
        print("=" * 72)
        print(f"SAPG training :: task={config.task} method={config.method}")
        print(
            f"  N={config.num_envs} envs | M={config.num_policies} policies | "
            f"horizon={config.horizon_length} | mini_epochs={config.mini_epochs}"
        )
        print(
            f"  lr={config.learning_rate} clip={config.clip_epsilon} "
            f"gamma={config.gamma} tau(GAE)={config.tau} "
            f"lambda={config.off_policy_weight} lambda'={config.critic_coefficient}"
        )
        print(
            f"  sigma={config.entropy_coefficient} phi_dim={config.phi_dim} "
            f"aggregation={config.aggregation} subsample={config.subsample_off_policy}"
        )
        print("=" * 72)

    env = make_env_for_config(config, force_surrogate=bool(args.surrogate))
    set_seed(int(config.seed), env)

    curriculum = make_curriculum_for(config, env)
    policy = make_policy_for_config(config)
    logger = make_logger_for_config(config, verbose=bool(args.verbose))

    if logger is not None and hasattr(logger, "log_hyperparameters"):
        try:
            logger.log_hyperparameters(config.to_dict())
        except Exception:
            pass

    start = time.time()
    trainer, history = train(
        config,
        env=env,
        policy=policy,
        logger=logger,
        num_iterations=args.iterations,
        max_samples=args.max_samples,
        verbose=bool(args.verbose),
        curriculum=curriculum,
    )
    elapsed = time.time() - start

    # Persist a checkpoint and the training history next to the logs.
    try:
        if trainer is not None and hasattr(trainer, "save"):
            ckpt = os.path.join(
                ensure_dir(os.path.join(config.log_dir, f"{config.task}_{config.method}")),
                "checkpoint.pt",
            )
            trainer.save(ckpt)
            if args.verbose:
                print(f"saved checkpoint -> {ckpt}")
    except Exception as exc:  # pragma: no cover - checkpointing is best effort
        if args.verbose:
            print(f"[warn] could not save checkpoint: {exc}")

    if args.verbose:
        final: Dict[str, float] = {}
        if history:
            for key, value in history[-1].items():
                if isinstance(value, (int, float)):
                    final[key] = float(value)
        samples = float(getattr(trainer, "total_samples", 0.0) or 0.0)
        print("-" * 72)
        print(f"finished in {elapsed:.1f}s | samples={samples:.4g}")
        for key in sorted(final):
            print(f"  {key} = {final[key]:.6g}")
        print("-" * 72)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
