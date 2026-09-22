"""SAPG entry point: dispatch to training / evaluation for SAPG and baselines.

Usage examples
--------------
    python main.py --method sapg --task allegrokuka_regrasping --config configs/sapg.yaml
    python main.py --method ppo  --task shadowhand --config configs/shadowhand.yaml
    python main.py --method dexpbt --task allegrohand --config configs/allegrohand.yaml
    python main.py --method pql --task allegrokuka_throw --config configs/allegrokuka.yaml
    python main.py --eval --checkpoint runs/sapg/ckpt.pt --task shadowhand

The script is intentionally tolerant of a missing IsaacGym installation: if the
simulator is unavailable it falls back to a lightweight dummy vectorized
environment so that the algorithm code can still be exercised end-to-end.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

import yaml

# Make the package importable when running `python main.py` from inside sapg/.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------- #
# Config helpers
# --------------------------------------------------------------------------- #
def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file, returning an empty dict when path is None."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level")
    return cfg


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-merge config dicts left-to-right (later wins)."""
    merged: Dict[str, Any] = {}
    for cfg in configs:
        if cfg:
            merged.update(cfg)
    return merged


def resolve_task(task: str) -> str:
    """Normalize task aliases to canonical names used by the env factory."""
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
    key = task.lower().strip()
    return aliases.get(key, key)


# --------------------------------------------------------------------------- #
# Environment factory
# --------------------------------------------------------------------------- #
def make_env_factory(task: str, num_envs: int, cfg: Dict[str, Any]):
    """Return a callable ``(num_envs, seed) -> vectorized env``.

    Tries IsaacGym first; falls back to a dummy env when unavailable.
    """
    task = resolve_task(task)

    def factory(n_envs: int = num_envs, seed: int = 0):
        try:
            from envs.isaacgym_wrapper import make_vector_env

            return make_vector_env(task, n_envs, seed=seed, **cfg.get("env", {}))
        except Exception as exc:  # pragma: no cover - depends on IsaacGym
            print(f"[main] IsaacGym env unavailable ({exc}); using DummyVectorEnv.")
            from envs.isaacgym_wrapper import DummyVectorEnv

            return DummyVectorEnv(task, n_envs, seed=seed)

    return factory


# --------------------------------------------------------------------------- #
# Method dispatch
# --------------------------------------------------------------------------- #
def run_sapg(args, cfg: Dict[str, Any]) -> None:
    from sapg.sapg_trainer import SAPGConfig, train_sapg

    task = resolve_task(args.task)
    defaults = dict(
        task=task,
        num_policies=int(cfg.get("num_policies", 6)),
        num_envs=int(cfg.get("num_envs", args.num_envs)),
        horizon=int(cfg.get("horizon", 16)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.95)),
        n_step=int(cfg.get("n_step", 3)),
        clip_eps=float(cfg.get("clip_eps", 0.1)),
        lam=float(cfg.get("lam", 1.0)),
        entropy_coef=float(cfg.get("entropy_coef", 0.0)),
        critic_coef=float(cfg.get("critic_coef", 4.0)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        kl_threshold=float(cfg.get("kl_threshold", 0.016)),
        mini_epochs=int(cfg.get("mini_epochs", 2)),
        num_mini_batches=int(cfg.get("num_mini_batches", 4)),
        aggregation=str(cfg.get("aggregation", "leader_follower")),
        subsample=bool(cfg.get("subsample", True)),
        phi_dim=int(cfg.get("phi_dim", 32)),
        recurrent=bool(cfg.get("recurrent", task.startswith("allegrokuka"))),
        lstm_hidden=int(cfg.get("lstm_hidden", 768)),
        device=str(cfg.get("device", "cuda")),
        target_transitions=float(cfg.get("target_transitions", 2e10)),
        output_dir=args.output_dir or cfg.get("output_dir", f"runs/sapg_{task}"),
        seed=int(args.seed),
    )
    config = SAPGConfig(**defaults)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    train_sapg(config, env_factory)


def run_ppo(args, cfg: Dict[str, Any]) -> None:
    from baselines.ppo import PPOConfig, train_ppo

    task = resolve_task(args.task)
    defaults = dict(
        task=task,
        num_envs=int(cfg.get("num_envs", args.num_envs)),
        horizon=int(cfg.get("horizon", 16)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.95)),
        clip_eps=float(cfg.get("clip_eps", 0.1)),
        critic_coef=float(cfg.get("critic_coef", 4.0)),
        entropy_coef=float(cfg.get("entropy_coef", 0.0)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        kl_threshold=float(cfg.get("kl_threshold", 0.016)),
        mini_epochs=int(cfg.get("mini_epochs", 2)),
        num_mini_batches=int(cfg.get("num_mini_batches", 4)),
        recurrent=bool(cfg.get("recurrent", task.startswith("allegrokuka"))),
        lstm_hidden=int(cfg.get("lstm_hidden", 768)),
        device=str(cfg.get("device", "cuda")),
        target_transitions=float(cfg.get("target_transitions", 2e10)),
        output_dir=args.output_dir or cfg.get("output_dir", f"runs/ppo_{task}"),
        seed=int(args.seed),
    )
    config = PPOConfig(**defaults)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    train_ppo(config, env_factory)


def run_dexpbt(args, cfg: Dict[str, Any]) -> None:
    from baselines.dexpbt import DexPBTConfig, train_dexpbt

    task = resolve_task(args.task)
    defaults = dict(
        task=task,
        num_policies=int(cfg.get("num_policies", 6)),
        num_envs=int(cfg.get("num_envs", args.num_envs)),
        horizon=int(cfg.get("horizon", 16)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.95)),
        clip_eps=float(cfg.get("clip_eps", 0.1)),
        critic_coef=float(cfg.get("critic_coef", 4.0)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        kl_threshold=float(cfg.get("kl_threshold", 0.016)),
        mini_epochs=int(cfg.get("mini_epochs", 2)),
        num_mini_batches=int(cfg.get("num_mini_batches", 4)),
        recurrent=bool(cfg.get("recurrent", task.startswith("allegrokuka"))),
        lstm_hidden=int(cfg.get("lstm_hidden", 768)),
        device=str(cfg.get("device", "cuda")),
        target_transitions=float(cfg.get("target_transitions", 2e10)),
        output_dir=args.output_dir or cfg.get("output_dir", f"runs/dexpbt_{task}"),
        seed=int(args.seed),
    )
    config = DexPBTConfig(**defaults)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    train_dexpbt(config, env_factory)


def run_pql(args, cfg: Dict[str, Any]) -> None:
    from baselines.pql import PQLConfig, train_pql

    task = resolve_task(args.task)
    defaults = dict(
        task=task,
        num_envs=int(cfg.get("num_envs", args.num_envs)),
        horizon=int(cfg.get("horizon", 16)),
        gamma=float(cfg.get("gamma", 0.99)),
        tau=float(cfg.get("tau", 0.95)),
        learning_rate=float(cfg.get("learning_rate", 3e-4)),
        critic_coef=float(cfg.get("critic_coef", 4.0)),
        recurrent=bool(cfg.get("recurrent", task.startswith("allegrokuka"))),
        lstm_hidden=int(cfg.get("lstm_hidden", 768)),
        device=str(cfg.get("device", "cuda")),
        target_transitions=float(cfg.get("target_transitions", 2e10)),
        output_dir=args.output_dir or cfg.get("output_dir", f"runs/pql_{task}"),
        seed=int(args.seed),
    )
    config = PQLConfig(**defaults)
    env_factory = make_env_factory(task, config.num_envs, cfg)
    train_pql(config, env_factory)


METHODS = {
    "sapg": run_sapg,
    "ppo": run_ppo,
    "dexpbt": run_dexpbt,
    "pql": run_pql,
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAPG: Split and Aggregate Policy Gradients"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="sapg",
        choices=sorted(METHODS.keys()),
        help="Which algorithm to run.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="allegrokuka_regrasping",
        help="Task name (allegrokuka_regrasping|allegrokuka_throw|"
        "allegrokuka_reorientation|shadowhand|allegrohand).",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--num-envs", type=int, default=24576, help="Parallel envs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default=None, help="Run directory.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
    parser.add_argument(
        "--eval-episodes", type=int, default=1024, help="Episodes for evaluation."
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    # CLI overrides take precedence over config file values.
    if args.num_envs:
        cfg.setdefault("num_envs", args.num_envs)

    if args.eval:
        from eval import evaluate_checkpoint

        evaluate_checkpoint(
            checkpoint=args.checkpoint,
            task=resolve_task(args.task),
            config=cfg,
            num_episodes=args.eval_episodes,
            env_factory=make_env_factory(resolve_task(args.task), args.num_envs, cfg),
        )
        return

    runner = METHODS[args.method]
    runner(args, cfg)


if __name__ == "__main__":
    main()
