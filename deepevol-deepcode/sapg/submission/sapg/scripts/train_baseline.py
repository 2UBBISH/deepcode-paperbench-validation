#!/usr/bin/env python
"""Baseline training entry point for the SAPG reproduction (SAPG paper, Sec. 5.2).

This script trains the three comparison methods used in Table 1 / Figure 5:

* ``ppo``    -- vanilla massively-parallel PPO on all ``N = 24576`` envs
                (``sapg.baselines.ppo_baseline``),
* ``pql``    -- Parallel Q-Learning / massively parallel DDPG with mixed
                exploration noise (``sapg.baselines.pql``),
* ``dexpbt`` -- PPO + population based training with ``M = 6`` groups
                (``sapg.baselines.dexpbt``).

It mirrors ``scripts/train.py`` (same flag names, same config plumbing) but
always forces a *single* population member for PPO/PQL and ``M`` members for
DexPBT, and it can run the paper's 5-seed protocol with the paper's shaded-band
aggregation ``(2/sqrt(n)) * sum_i (mean - y_i)^2``.

Examples
--------
::

    # Regrasping PPO baseline, 5 seeds, 2e10 transitions each
    python scripts/train_baseline.py --method ppo --task regrasping \\
        --max-samples 2e10 --num-seeds 5

    # PQL on AllegroHand, short smoke run
    python scripts/train_baseline.py --method pql --task allegro_hand \\
        --iterations 10 --verbose

    # DexPBT on Reorientation with the entropy-regularised hyperparameters
    python scripts/train_baseline.py --method dexpbt --task reorientation \\
        --entropy-coefficient 0.005 --num-seeds 5

The module is importable and CPU-smoke-testable: torch / IsaacGym are imported
lazily inside the functions that need them.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Make the repository importable when executed directly (python scripts/...)
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, "sapg"), _HERE):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

TASKS: Tuple[str, ...] = (
    "regrasping",
    "throw",
    "reorientation",
    "shadow_hand",
    "allegro_hand",
)

#: Methods handled by this script (SAPG itself lives in ``scripts/train.py``).
BASELINE_METHODS: Tuple[str, ...] = ("ppo", "pql", "dexpbt")

#: Per-task best entropy coefficient sigma reported in Sec. 6.2 / Fig. 5.
TASK_ENTROPY: Dict[str, float] = {
    "regrasping": 0.0,
    "throw": 0.0,
    "reorientation": 0.005,
    "shadow_hand": 0.0,
    "allegro_hand": 0.0,
}

#: Reference Table 1 finals (mean over 5 seeds) for the SAPG entries; used only
#: to print an informational comparison, never to alter training.
PAPER_REFERENCE: Dict[str, Dict[str, float]] = {
    "sapg_sigma_0": {
        "allegro_hand": 1.23e4,
        "shadow_hand": 1.17e4,
        "regrasping": 35.7,
        "throw": 23.7,
        "reorientation": 33.2,
    },
    "sapg_sigma_0.005": {
        "allegro_hand": 9.14e3,
        "shadow_hand": 1.28e4,
        "regrasping": 33.4,
        "throw": 18.7,
        "reorientation": 38.6,
    },
}

#: Metric key preference order when summarising a baseline run.
METRIC_KEYS: Tuple[str, ...] = (
    "episode_return",
    "return",
    "rewards",
    "mean_episode_return",
    "successes",
    "success",
    "episode_successes",
    "episode_length",
)

#: Paper training budget: ~2e10 environment transitions per experiment.
PAPER_MAX_SAMPLES: float = 2.0e10

#: Paper protocol: 5 seeds, mean + shaded band.
PAPER_NUM_SEEDS: int = 5

_METHOD_ALIASES: Dict[str, str] = {
    "ppo": "ppo",
    "vanilla_ppo": "ppo",
    "ppo_baseline": "ppo",
    "ppo-baseline": "ppo",
    "pql": "pql",
    "apql": "pql",
    "parallel_q_learning": "pql",
    "parallel-q-learning": "pql",
    "dexpbt": "dexpbt",
    "dex_pbt": "dexpbt",
    "pbt": "dexpbt",
    "expbt": "dexpbt",
    "sapg": "sapg",
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for baseline training."""
    parser = argparse.ArgumentParser(
        description="Train SAPG baselines (PPO / PQL / DexPBT) from Sec. 5.2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ppo",
        choices=sorted(set(_METHOD_ALIASES)),
        help="Baseline method to train.",
    )
    parser.add_argument("--task", type=str, default="regrasping", choices=TASKS, help="Task name.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")

    # Environment / population size (§5.2: N = 24576, M = 6).
    parser.add_argument("--num-envs", type=int, default=None, help="Number of parallel envs N.")
    parser.add_argument(
        "--num-policies",
        type=int,
        default=None,
        help="Population size M for DexPBT (ignored by PPO/PQL).",
    )

    # Optimization overrides.
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=None, help="GAE lambda (paper's tau).")
    parser.add_argument("--horizon-length", type=int, default=None)
    parser.add_argument("--mini-epochs", type=int, default=None)
    parser.add_argument("--entropy-coefficient", type=float, default=None)
    parser.add_argument("--off-policy-weight", type=float, default=None)

    # Bookkeeping / protocol.
    parser.add_argument("--seed", type=int, default=0, help="First seed.")
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds (paper: 5).")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (outer loops).",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=None,
        help="Transition budget per seed (paper: 2e10).",
    )
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0.")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for logs/checkpoints.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSON file to write the aggregated summary to.",
    )
    parser.add_argument("--print-every", type=int, default=None, help="Metrics print interval.")

    # Switch / shorthands.
    parser.add_argument("--no-lstm", action="store_true", help="Force a feed-forward backbone.")
    parser.add_argument(
        "--surrogate",
        action="store_true",
        help="Force the dependency-free torch surrogate environment.",
    )
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging.")
    parser.add_argument("--verbose", action="store_true", help="Print per-iteration metrics.")
    parser.add_argument(
        "--compare-paper",
        action="store_true",
        help="Print the Table 1 reference numbers next to the results.",
    )
    parser.add_argument("--save-checkpoints", action="store_true", help="Save per-seed checkpoints.")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return parser


def _normalise_method(method: Optional[str]) -> str:
    """Resolve method aliases to a canonical baseline name."""
    key = str(method or "ppo").strip().lower().replace(" ", "_")
    return _METHOD_ALIASES.get(key, key)


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


def load_config_from_args(args: argparse.Namespace) -> Any:
    """Build the per-method config from defaults, an optional YAML and the CLI.

    Per Section 5.2 the baselines all run at ``N = 24576``; PPO and PQL use a
    single policy while DexPBT keeps ``M = 6`` independent population members.
    """
    from sapg.utils.config import SAPGConfig, build_config  # lazy

    method = _normalise_method(args.method)
    task = args.task
    if method == "sapg":  # pragma: no cover - SAPG is trained by scripts/train.py
        raise ValueError("scripts/train_baseline.py handles ppo/pql/dexpbt; use scripts/train.py for sapg")

    config = build_config(task)

    # Optional YAML base (per task group: AllegroKuka / ShadowHand / AllegroHand).
    if args.config:
        try:
            loaded = SAPGConfig.from_yaml(args.config)
        except Exception:  # pragma: no cover - defensive: plain dict YAML
            import yaml  # type: ignore

            with open(args.config, "r") as fh:
                loaded = SAPGConfig.from_dict(yaml.safe_load(fh) or {})
        # Task specific values (task name / dims) come from the CLI-driven build.
        for field in (
            "num_envs",
            "num_policies",
            "learning_rate",
            "clip_epsilon",
            "gamma",
            "tau",
            "horizon_length",
            "mini_epochs",
            "entropy_coefficient",
            "off_policy_weight",
            "actor_mlp_units",
            "critic_mlp_units",
            "use_lstm",
            "lstm_hidden_size",
            "lstm_num_layers",
            "device",
            "obs_dim",
            "action_dim",
            "phi_dim",
        ):
            value = getattr(loaded, field, None)
            if value is not None and hasattr(config, field):
                setattr(config, field, value)

    # --- method specific overrides -----------------------------------------
    config.method = method
    if method in ("ppo", "pql"):
        # A single policy over all N environments.
        config.num_policies = 1
        config.leader_index = 1
        config.aggregation = "none"
        config.off_policy_weight = 0.0
        config.subsample_off_policy = False
        config.phi_dim = 0
        config.per_block_sigma = False
    else:  # dexpbt: M independent PPO groups, no shared backbone / phi.
        config.num_policies = int(args.num_policies or getattr(config, "num_policies", 6) or 6)
        config.aggregation = "none"
        config.off_policy_weight = 0.0
        config.phi_dim = 0
        config.per_block_sigma = False

    # --- CLI overrides ------------------------------------------------------
    if args.num_envs is not None:
        config.num_envs = int(args.num_envs)
    if method == "dexpbt" and args.num_policies is not None:
        config.num_policies = int(args.num_policies)
    if args.learning_rate is not None:
        config.learning_rate = float(args.learning_rate)
        config.critic_learning_rate = float(args.learning_rate)
    if args.clip_epsilon is not None:
        config.clip_epsilon = float(args.clip_epsilon)
    if args.gamma is not None:
        config.gamma = float(args.gamma)
    if args.tau is not None:
        config.tau = float(args.tau)
    if args.horizon_length is not None:
        config.horizon_length = int(args.horizon_length)
    if args.mini_epochs is not None:
        config.mini_epochs = int(args.mini_epochs)
    if args.entropy_coefficient is not None:
        config.entropy_coefficient = float(args.entropy_coefficient)
    elif method == "dexpbt":
        # DexPBT inherits the SAPG per-task tuned entropy coefficient (Sec. 6.2).
        config.entropy_coefficient = float(TASK_ENTROPY.get(task, 0.0))
    else:
        # PPO / PQL baselines use no entropy regularisation by default.
        config.entropy_coefficient = 0.0
    if args.off_policy_weight is not None:
        config.off_policy_weight = float(args.off_policy_weight)
    if args.device is not None:
        config.device = args.device
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.no_lstm:
        config.use_lstm = False
    if args.tensorboard:
        config.use_tensorboard = True
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.print_every is not None:
        config.print_every = int(args.print_every)

    try:
        config._apply_derived()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - config may be a plain object
        pass
    return config


# --------------------------------------------------------------------------- #
# Environment / policy / logger factories (duck-typed against the trainers)
# --------------------------------------------------------------------------- #


def make_env_for_config(config: Any, force_surrogate: bool = False) -> Any:
    """Instantiate the vectorised environment for a config."""
    if force_surrogate:
        os.environ["SAPG_FORCE_SURROGATE"] = "1"
    from sapg.envs import make_env  # lazy

    try:
        return make_env(config.task, config=config, num_envs=config.num_envs)
    except TypeError:  # pragma: no cover - signature drift
        return make_env(config.task, config=config)


def make_policy_for_config(config: Any, num_policies: Optional[int] = None) -> Any:
    """Build the policy object expected by the trainers.

    PPO/PQL use ``phi_dim = 0`` and ``num_policies = 1``; DexPBT constructs one
    such policy per population member inside its own module.
    """
    from sapg.models.actor import ActorCritic  # lazy

    kwargs: Dict[str, Any] = {}
    if num_policies is not None:
        kwargs["num_policies"] = int(num_policies)
    try:
        return ActorCritic(config, **kwargs)
    except TypeError:  # pragma: no cover - signature drift
        return ActorCritic(
            obs_dim=config.obs_dim,
            action_dim=config.action_dim,
            phi_dim=getattr(config, "phi_dim", 0),
            num_policies=int(num_policies or getattr(config, "num_policies", 1)),
            config=config,
        )


def make_logger_for_config(config: Any, verbose: bool = False) -> Any:
    """Create the experiment logger (console + optional TensorBoard)."""
    from sapg.utils.logging import make_logger  # lazy

    try:
        return make_logger(
            config=config,
            log_dir=getattr(config, "log_dir", "runs"),
            name=str(getattr(config, "method", "baseline")),
            verbose=bool(verbose),
            use_tensorboard=bool(getattr(config, "use_tensorboard", False)),
        )
    except TypeError:  # pragma: no cover - signature drift
        return make_logger(config)


def set_seed(seed: int, env: Any = None) -> int:
    """Seed python/numpy/torch (and the env when it supports it)."""
    seed = int(seed)
    random.seed(seed)
    try:  # pragma: no cover - numpy optional
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:
        pass
    try:  # pragma: no cover - torch optional
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    if env is not None:
        for name in ("seed", "set_seed"):
            fn = getattr(env, name, None)
            if callable(fn):
                try:
                    fn(seed)
                    break
                except Exception:
                    continue
    return seed


# --------------------------------------------------------------------------- #
# Training dispatch
# --------------------------------------------------------------------------- #


def train_baseline(
    config: Any,
    method: Optional[str] = None,
    env: Any = None,
    policy: Any = None,
    logger: Any = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> Tuple[Any, List[Dict[str, float]]]:
    """Dispatch to the requested baseline trainer.

    Returns ``(trainer, history)`` where ``history`` is a list of per-iteration
    metric dictionaries.  Missing optional modules degrade into an informative
    :class:`ImportError` mentioning which baseline is unavailable.
    """
    method = _normalise_method(method or getattr(config, "method", "ppo"))

    if method == "ppo":
        try:
            from sapg.baselines.ppo_baseline import train_ppo_baseline as _train
        except Exception as exc:  # pragma: no cover
            raise ImportError(f"PPO baseline unavailable: {exc}") from exc
    elif method == "pql":
        try:
            from sapg.baselines.pql import train_pql as _train
        except Exception as exc:  # pragma: no cover
            raise ImportError(f"PQL baseline unavailable: {exc}") from exc
    elif method == "dexpbt":
        try:
            from sapg.baselines.dexpbt import train_dexpbt as _train
        except Exception as exc:  # pragma: no cover
            raise ImportError(f"DexPBT baseline unavailable: {exc}") from exc
    else:  # pragma: no cover - guarded by the parser
        raise ValueError(f"Unknown baseline method {method!r}; expected one of {BASELINE_METHODS}")

    call_kwargs: Dict[str, Any] = {
        "config": config,
        "num_iterations": num_iterations,
        "max_samples": max_samples,
        "verbose": verbose,
        "logger": logger,
    }
    if env is not None:
        call_kwargs["env"] = env
    if policy is not None:
        call_kwargs["policy"] = policy
    call_kwargs.update(kwargs)

    result = _train(**call_kwargs)
    return _as_pair(result)


def _as_pair(result: Any) -> Tuple[Any, List[Dict[str, float]]]:
    """Normalise a trainer return value to ``(trainer, history)``."""
    if isinstance(result, tuple):
        if len(result) == 2:
            return result[0], list(result[1] or [])
        if len(result) == 1:
            return result[0], []
    trainer = result
    history: List[Dict[str, float]] = []
    for attr in ("history", "metrics_history", "log_history"):
        value = getattr(trainer, attr, None)
        if isinstance(value, list):
            history = list(value)
            break
    return trainer, history


# --------------------------------------------------------------------------- #
# Multi-seed protocol (Sec. 5.2: 5 seeds, mean + shaded band)
# --------------------------------------------------------------------------- #


def run_seeds(
    method: str,
    config: Any,
    seeds: Sequence[int],
    num_iterations: Optional[int] = None,
    max_samples: Optional[float] = None,
    verbose: bool = False,
    logger: Any = None,
    save_checkpoints: bool = False,
    force_surrogate: bool = False,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Train one seed at a time and collect per-seed results.

    Environments are rebuilt per seed so the parallel-simulation RNG state is
    fully re-initialised, matching the paper's protocol of independent runs.
    """
    from sapg.utils.config import SAPGConfig  # lazy

    results: List[Dict[str, Any]] = []
    for seed in seeds:
        run_config = _replace_config(config, seed=int(seed))
        if save_checkpoints:
            from sapg.utils.config import ensure_dir  # lazy

            log_dir = os.path.join(
                str(getattr(run_config, "log_dir", "runs")),
                f"{method}_{getattr(run_config, 'task', 'task')}",
            )
            ensure_dir(log_dir)
            run_config.log_dir = log_dir
            run_config.save_yaml(os.path.join(log_dir, f"config_seed{int(seed)}.yaml"))

        env = make_env_for_config(run_config, force_surrogate=force_surrogate)
        policy = make_policy_for_config(run_config)
        set_seed(int(seed), env=env)

        t0 = time.time()
        trainer, history = train_baseline(
            run_config,
            method=method,
            env=env,
            policy=policy,
            logger=logger,
            num_iterations=num_iterations,
            max_samples=max_samples,
            verbose=verbose,
            **kwargs,
        )
        elapsed = time.time() - t0

        if save_checkpoints:
            try:
                path = os.path.join(
                    str(getattr(run_config, "log_dir", "runs")), f"{method}_seed{int(seed)}.pt"
                )
                trainer.save(path)
            except Exception:  # pragma: no cover - checkpointing is best effort
                pass

        results.append(
            {
                "seed": int(seed),
                "method": method,
                "task": getattr(run_config, "task", None),
                "history": history,
                "elapsed_sec": elapsed,
                "final": _final_metrics(history),
                "trainer": trainer,
            }
        )

        try:  # close any env we created to free GPU memory between seeds
            close = getattr(env, "close", None)
            if callable(close):
                close()
        except Exception:  # pragma: no cover
            pass
    return results


def _replace_config(config: Any, **overrides: Any) -> Any:
    """Copy a config object, overriding fields (dataclass or duck-typed)."""
    import dataclasses

    if dataclasses.is_dataclass(config):
        valid = {k: v for k, v in overrides.items() if hasattr(config, k)}
        return dataclasses.replace(config, **valid)
    clone = copy.deepcopy(config)
    for key, value in overrides.items():
        if hasattr(clone, key):
            setattr(clone, key, value)
    return clone


def _final_metrics(history: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Extract the last finite value of each scalar metric in a history."""
    finals: Dict[str, float] = {}
    if not history:
        return finals
    for entry in history:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            number = _as_float(value)
            if number is not None and math.isfinite(number):
                finals[key] = number
    return finals


def _as_float(value: Any) -> Optional[float]:
    """Coerce a scalar (python float, ndarray, 0-d tensor) to ``float``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    for attr in ("item", "mean"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:
                continue
    try:
        return float(value)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Aggregation: paper shaded band
# --------------------------------------------------------------------------- #


def paper_standard_error(curves: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Mean curve and shaded band exactly as reported in the paper.

    Section 5.2 defines the band as ``(2/sqrt(n)) * sum_i (y(t) - y_i(t))^2``,
    i.e. a variance-like quantity rather than the usual standard error.
    """
    rows = [list(curve or []) for curve in curves]
    rows = [row for row in rows if row]
    if not rows:
        return [], []
    length = min(len(row) for row in rows)
    n = len(rows)
    mean: List[float] = []
    band: List[float] = []
    for t in range(length):
        values = [float(row[t]) for row in rows]
        mu = sum(values) / n
        var = sum((mu - v) ** 2 for v in values)
        mean.append(mu)
        band.append((2.0 / math.sqrt(n)) * var)
    return mean, band


def aggregate_seed_results(results: Sequence[Dict[str, Any]], metric: str = "episode_return") -> Dict[str, Any]:
    """Aggregate per-seed results for a single metric into mean/band curves."""
    curves: List[List[float]] = []
    finals: List[float] = []
    for result in results:
        curve = _metric_curve(result.get("history") or [], metric)
        if curve:
            curves.append(curve)
        final = _as_float((result.get("final") or {}).get(metric))
        if final is None and curve:
            final = curve[-1]
        if final is not None:
            finals.append(final)
    mean, band = paper_standard_error(curves)
    final_mean = sum(finals) / len(finals) if finals else float("nan")
    final_band = (2.0 / math.sqrt(len(finals))) * sum((final_mean - f) ** 2 for f in finals) if finals else float("nan")
    return {
        "metric": metric,
        "num_seeds": len(results),
        "seed_curves": curves,
        "mean": mean,
        "band": band,
        "seed_finals": finals,
        "final_mean": final_mean,
        "final_band": final_band,
        "final_std": _std(finals),
    }


def _metric_curve(history: Sequence[Dict[str, Any]], metric: str) -> List[float]:
    """Pull one metric's series out of a history, trying common aliases."""
    alias_map = {
        "episode_return": ("episode_return", "return", "returns", "rewards", "episode_reward", "mean_return"),
        "successes": ("successes", "success", "episode_successes", "success_count", "avg_successes"),
        "episode_length": ("episode_length", "length", "episode_len"),
    }
    keys = alias_map.get(metric, (metric,))
    for key in keys:
        curve: List[float] = []
        for entry in history:
            if not isinstance(entry, dict):
                break
            value = _as_float(entry.get(key))
            if value is None:
                curve = []
                break
            curve.append(value)
        if curve:
            return curve
    return []


def _std(values: Sequence[float]) -> float:
    """Population standard deviation (0.0 for fewer than two samples)."""
    values = [float(v) for v in values if v is not None]
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def primary_metric_for(task: str, metric: Optional[str] = None) -> str:
    """Paper metric per task: net episode reward for hand tasks, else successes."""
    if metric:
        return metric
    return "episode_return"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def summarise(results: Sequence[Dict[str, Any]], metric: Optional[str] = None) -> Dict[str, Any]:
    """Build the per-seed + aggregated summary for a set of baseline runs."""
    if not results:
        return {"metric": metric, "num_seeds": 0}
    task = results[0].get("task") or "task"
    method = results[0].get("method") or "baseline"
    chosen = primary_metric_for(task, metric)
    aggregated = aggregate_seed_results(results, chosen)
    if not aggregated.get("seed_finals"):
        # Fall back to any available metric with a usable curve.
        for fallback in METRIC_KEYS:
            if fallback == chosen:
                continue
            candidate = aggregate_seed_results(results, fallback)
            if candidate.get("seed_finals"):
                aggregated = candidate
                break

    payload: Dict[str, Any] = {
        "method": method,
        "task": task,
        "metric": aggregated.get("metric", chosen),
        "num_seeds": len(results),
        "final_mean": aggregated.get("final_mean"),
        "final_band": aggregated.get("final_band"),
        "final_std": aggregated.get("final_std"),
        "seed_finals": aggregated.get("seed_finals"),
        "aggregated": {
            "samples": None,
            "mean": aggregated.get("mean"),
            "band": aggregated.get("band"),
        },
        "per_seed": [
            {
                "seed": r.get("seed"),
                "final": r.get("final"),
                "elapsed_sec": r.get("elapsed_sec"),
            }
            for r in results
        ],
    }
    return payload


def paper_reference_for(task: str, entropy_coefficient: Optional[float] = None) -> Optional[Dict[str, float]]:
    """Return the Table 1 reference entry matching the entropy coefficient."""
    key = "sapg_sigma_0.005" if (entropy_coefficient or 0.0) > 0.0 else "sapg_sigma_0"
    entry = PAPER_REFERENCE.get(key, {})
    if task not in entry:
        return None
    return {"sapg": entry[task]}


def print_summary(payload: Dict[str, Any], compare_paper: bool = False) -> None:
    """Print a compact console summary of a baseline run."""
    if not payload or payload.get("num_seeds", 0) == 0:
        print("nothing to summarise")
        return
    method = payload.get("method")
    task = payload.get("task")
    metric = payload.get("metric")
    print("=" * 72)
    print(f"{method.upper()} baseline on {task}  (metric: {metric})")
    print("-" * 72)
    print(f"seeds           : {payload.get('num_seeds')}")
    print(f"final mean      : {_fmt(payload.get('final_mean'))}")
    print(f"shaded band     : {_fmt(payload.get('final_band'))}")
    print(f"std             : {_fmt(payload.get('final_std'))}")
    finals = payload.get("seed_finals") or []
    if finals:
        print("per-seed finals : " + ", ".join(_fmt(f) for f in finals))
    if compare_paper:
        ref = payload.get("paper") or {}
        if ref:
            print(f"paper SAPG ref  : {_fmt(ref.get('sapg'))}")
    print("=" * 72)


def _fmt(value: Any) -> str:
    """Format a number for console output (thousands-aware, NaN-safe)."""
    number = _as_float(value)
    if number is None or not math.isfinite(number):
        return "n/a"
    abs_number = abs(number)
    if abs_number != 0 and (abs_number >= 1e5 or abs_number < 1e-3):
        return f"{number:.3e}"
    return f"{number:.3f}"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, train the requested baseline over seeds and report."""
    parser = build_parser()
    args = parser.parse_args(argv)

    method = _normalise_method(args.method)
    if method == "sapg":  # pragma: no cover - parser rejects, kept for clarity
        print("SAPG is trained by scripts/train.py --method sapg", file=sys.stderr)
        return 2

    if args.surrogate:
        os.environ["SAPG_FORCE_SURROGATE"] = "1"

    config = load_config_from_args(args)
    if args.log_dir is None:
        config.log_dir = os.path.join(
            str(getattr(config, "log_dir", "runs")), f"{method}", str(args.task)
        )

    num_seeds = max(1, int(args.num_seeds))
    seeds = [int(args.seed) + i for i in range(num_seeds)]
    max_samples = args.max_samples
    if max_samples is None and args.iterations is None:
        # Default to the paper budget only when explicitly requested; otherwise
        # a short run keeps smoke tests cheap.
        max_samples = None

    print(
        f"[train_baseline] method={method} task={args.task} "
        f"envs={getattr(config, 'num_envs', None)} seeds={seeds} "
        f"max_samples={max_samples} iterations={args.iterations}"
    )

    logger = None
    try:
        logger = make_logger_for_config(config, verbose=bool(args.verbose))
    except Exception as exc:  # pragma: no cover - logging is optional
        if args.verbose:
            print(f"[train_baseline] logger unavailable ({exc}); continuing without one")

    results = run_seeds(
        method,
        config,
        seeds,
        num_iterations=args.iterations,
        max_samples=max_samples,
        verbose=bool(args.verbose),
        logger=logger,
        save_checkpoints=bool(args.save_checkpoints),
        force_surrogate=bool(args.surrogate),
    )

    payload = summarise(results, metric=primary_metric_for(args.task))
    if args.compare_paper:
        payload["paper"] = paper_reference_for(args.task, getattr(config, "entropy_coefficient", 0.0))

    print_summary(payload, compare_paper=bool(args.compare_paper))

    if args.verbose:
        # Per-iteration history for the first seed only (keeps output readable).
        if results and results[0].get("history"):
            for row in results[0]["history"][-5:]:
                if isinstance(row, dict):
                    print("  " + " ".join(f"{k}={_fmt(v)}" for k, v in sorted(row.items())[:8]))

    # Persist results (drop the live trainer objects, which are not serialisable).
    serialisable = copy.deepcopy({k: v for k, v in payload.items() if k != "trainer"})
    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(serialisable, fh, indent=2, default=str)
        print(f"[train_baseline] wrote {args.output}")
    elif args.log_dir or getattr(config, "log_dir", None):
        out_path = os.path.join(str(config.log_dir), "baseline_summary.json")
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(serialisable, fh, indent=2, default=str)
            print(f"[train_baseline] wrote {out_path}")
        except Exception:  # pragma: no cover
            pass

    if args.json:
        print(json.dumps(serialisable, indent=2, default=str))

    try:
        if logger is not None:
            close = getattr(logger, "close", None)
            if callable(close):
                close()
    except Exception:  # pragma: no cover
        pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
