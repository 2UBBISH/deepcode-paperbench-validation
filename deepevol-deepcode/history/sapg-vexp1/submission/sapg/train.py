"""Training loop driver for SAPG and its baselines.

This module provides a unified ``train`` entry point that dispatches to the
appropriate trainer (SAPG, PPO, DexPBT, PQL) based on the requested method.
It is intentionally thin: all algorithm logic lives in the corresponding
trainer modules, while this file is responsible for

  * loading / merging YAML configs,
  * resolving task aliases and building the vectorized environment factory,
  * constructing the per-method ``*Config`` dataclass,
  * wiring up logging and checkpointing,
  * and running the training loop.

The module can be used either programmatically::

    from train import train
    train(method="sapg", task="allegrokuka_regrasping", config="configs/sapg.yaml")

or from the command line::

    python train.py --method sapg --task allegrokuka_regrasping \
        --config configs/sapg.yaml --num-envs 24576 --seed 0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Path to a YAML file, or ``None``.

    Returns:
        Parsed config dict (empty dict when ``path`` is ``None``).

    Raises:
        FileNotFoundError: If ``path`` is given but does not exist.
        ValueError: If the file does not parse to a mapping.
    """
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return data


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow left-to-right merge of config dicts (later entries win)."""
    merged: Dict[str, Any] = {}
    for cfg in configs:
        if not cfg:
            continue
        merged.update(cfg)
    return merged


def resolve_task(task: str) -> str:
    """Normalize a task alias to its canonical name."""
    try:
        from envs import resolve_task_name

        return resolve_task_name(task)
    except Exception:
        # Fall back to a local alias table so this module stays importable
        # even when the envs package cannot be imported.
        aliases = {
            "regrasping": "allegrokuka_regrasping",
            "throw": "allegrokuka_throw",
            "reorientation": "allegrokuka_reorientation",
            "allegrokuka": "allegrokuka_regrasping",
            "shadow": "shadowhand",
            "shadow_hand": "shadowhand",
            "allegro": "allegrohand",
            "allegro_hand": "allegrohand",
        }
        key = task.strip().lower()
        return aliases.get(key, key)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------


def make_env_factory(
    task: str,
    num_envs: int,
    cfg: Dict[str, Any],
) -> Callable[[int, int], Any]:
    """Build a callable ``factory(n_envs, seed) -> VectorEnv``.

    The factory prefers the IsaacGym-backed vectorized environment and falls
    back to the NumPy ``DummyVectorEnv`` when IsaacGym is unavailable (or when
    ``SAPG_FORCE_DUMMY_ENV`` is set).
    """
    task = resolve_task(task)
    device = cfg.get("device", "cuda:0")
    force_dummy = bool(cfg.get("force_dummy_env", False)) or bool(
        os.environ.get("SAPG_FORCE_DUMMY_ENV")
    )

    def factory(n_envs: int = num_envs, seed: int = 0) -> Any:
        try:
            from envs.isaacgym_wrapper import make_vector_env

            return make_vector_env(
                task=task,
                num_envs=n_envs,
                seed=seed,
                device=device,
                force_dummy=force_dummy,
            )
        except Exception as exc:  # pragma: no cover - depends on environment
            from envs.isaacgym_wrapper import DummyVectorEnv

            print(
                f"[train] IsaacGym env unavailable ({exc}); "
                f"falling back to DummyVectorEnv for task={task}",
                file=sys.stderr,
            )
            return DummyVectorEnv(task=task, num_envs=n_envs, seed=seed)

    return factory


# ---------------------------------------------------------------------------
# Per-method config builders
# ---------------------------------------------------------------------------


def _common_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides on top of a config dict."""
    out = dict(cfg)
    if getattr(args, "num_envs", None) is not None:
        out["num_envs"] = args.num_envs
    if getattr(args, "seed", None) is not None:
        out["seed"] = args.seed
    if getattr(args, "output_dir", None) is not None:
        out["output_dir"] = args.output_dir
    return out


def _filter_kwargs(cls: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keys that are valid fields of the dataclass ``cls``."""
    try:
        import dataclasses

        valid = {f.name for f in dataclasses.fields(cls)}
    except Exception:
        return dict(cfg)
    return {k: v for k, v in cfg.items() if k in valid}


def build_sapg_config(cfg: Dict[str, Any], task: str) -> Any:
    from sapg.sapg_trainer import SAPGConfig

    cfg = dict(cfg)
    cfg.setdefault("task", task)
    cfg.setdefault("recurrent", task.startswith("allegrokuka"))
    return SAPGConfig(**_filter_kwargs(SAPGConfig, cfg))


def build_ppo_config(cfg: Dict[str, Any], task: str) -> Any:
    from baselines.ppo import PPOConfig

    cfg = dict(cfg)
    cfg.setdefault("task", task)
    cfg.setdefault("recurrent", task.startswith("allegrokuka"))
    return PPOConfig(**_filter_kwargs(PPOConfig, cfg))


def build_dexpbt_config(cfg: Dict[str, Any], task: str) -> Any:
    from baselines.dexpbt import DexPBTConfig

    cfg = dict(cfg)
    cfg.setdefault("task", task)
    cfg.setdefault("recurrent", task.startswith("allegrokuka"))
    return DexPBTConfig(**_filter_kwargs(DexPBTConfig, cfg))


def build_pql_config(cfg: Dict[str, Any], task: str) -> Any:
    from baselines.pql import PQLConfig

    cfg = dict(cfg)
    cfg.setdefault("task", task)
    cfg.setdefault("recurrent", task.startswith("allegrokuka"))
    return PQLConfig(**_filter_kwargs(PQLConfig, cfg))


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------


def train_sapg(cfg: Dict[str, Any], task: str, logger: Any = None) -> Any:
    from sapg.sapg_trainer import train_sapg as _train

    config = build_sapg_config(cfg, task)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    return _train(config, env_factory, logger=logger)


def train_ppo(cfg: Dict[str, Any], task: str, logger: Any = None) -> Any:
    from baselines.ppo import train_ppo as _train

    config = build_ppo_config(cfg, task)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    return _train(config, env_factory, logger=logger)


def train_dexpbt(cfg: Dict[str, Any], task: str, logger: Any = None) -> Any:
    from baselines.dexpbt import train_dexpbt as _train

    config = build_dexpbt_config(cfg, task)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    return _train(config, env_factory, logger=logger)


def train_pql(cfg: Dict[str, Any], task: str, logger: Any = None) -> Any:
    from baselines.pql import train_pql as _train

    config = build_pql_config(cfg, task)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    return _train(config, env_factory, logger=logger)


TRAINERS: Dict[str, Callable[[Dict[str, Any], str, Any], Any]] = {
    "sapg": train_sapg,
    "ppo": train_ppo,
    "dexpbt": train_dexpbt,
    "pql": train_pql,
}


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def train(
    method: str = "sapg",
    task: str = "allegrokuka_regrasping",
    config: Optional[str] = None,
    extra_configs: Optional[list] = None,
    num_envs: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    use_tensorboard: bool = True,
    verbose: bool = True,
    **overrides: Any,
) -> Any:
    """Run a training job.

    Args:
        method: One of ``sapg``, ``ppo``, ``dexpbt``, ``pql``.
        task: Task name or alias.
        config: Path to a YAML config file (optional).
        extra_configs: Additional YAML config paths merged after ``config``.
        num_envs: Override the number of parallel environments.
        seed: Override the random seed.
        output_dir: Override the output directory.
        use_tensorboard: Whether to enable TensorBoard logging.
        verbose: Whether to print progress.
        **overrides: Arbitrary config key overrides.

    Returns:
        The trainer instance (already trained).
    """
    method = method.strip().lower()
    if method not in TRAINERS:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of {sorted(TRAINERS)}"
        )

    task = resolve_task(task)

    configs = [load_config(config)]
    for extra in extra_configs or []:
        configs.append(load_config(extra))
    cfg = merge_configs(*configs)

    # CLI / keyword overrides take precedence over config files.
    if num_envs is not None:
        cfg["num_envs"] = num_envs
    if seed is not None:
        cfg["seed"] = seed
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    cfg.setdefault("task", task)
    cfg.setdefault("seed", 0)
    cfg.setdefault("output_dir", os.path.join("runs", f"{method}_{task}"))

    # Logger
    logger = None
    try:
        from utils.logger import Logger

        logger = Logger(
            log_dir=cfg["output_dir"],
            use_tensorboard=use_tensorboard,
            verbose=verbose,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[train] Logger unavailable ({exc}); continuing without logging",
              file=sys.stderr)

    if verbose:
        print(f"[train] method={method} task={task} num_envs={cfg.get('num_envs')} "
              f"seed={cfg.get('seed')} output_dir={cfg['output_dir']}")

    try:
        trainer = TRAINERS[method](cfg, task, logger)
    finally:
        if logger is not None:
            try:
                logger.close()
            except Exception:
                pass
    return trainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train SAPG or a baseline on a manipulation task."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sapg",
        choices=sorted(TRAINERS),
        help="Algorithm to run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="allegrokuka_regrasping",
        help="Task name or alias.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--extra-config",
        type=str,
        action="append",
        default=None,
        help="Additional YAML config file(s), merged after --config.",
    )
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for logs and checkpoints.")
    parser.add_argument("--no-tensorboard", action="store_true",
                        help="Disable TensorBoard logging.")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return parser


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train(
        method=args.method,
        task=args.task,
        config=args.config,
        extra_configs=args.extra_config,
        num_envs=args.num_envs,
        seed=args.seed,
        output_dir=args.output_dir,
        use_tensorboard=not args.no_tensorboard,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
