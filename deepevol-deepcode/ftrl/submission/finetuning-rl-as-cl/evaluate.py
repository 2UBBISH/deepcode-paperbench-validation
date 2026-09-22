"""Evaluation entry point for the fine-tuning-as-forgetting-mitigation reproduction.

This module is *environment agnostic*: :func:`evaluate` resolves an environment
name to the evaluator living in the corresponding ``src.<env>`` package and runs
the paper's evaluation protocol for that benchmark.

Protocols implemented (see the reproduction plan / paper):

* **RoboticSequence (Meta-World)** -- per-stage success rates over
  ``finetune.eval_episodes`` episodes (Figure 7), aggregated over ``>= 20`` seeds
  with 90% confidence intervals, plus optional prefix-task forward transfer
  (Table 6) and expert-action log-likelihood (Figure 8).
* **Montezuma's Revenge** -- mean episode return and **Room 7 success rate**
  over 100 episodes (Figure 3b / 6 / 17-19).
* **NetHack (Human Monk)** -- 1000-episode full evaluation where rollouts stop at
  death, after 150 steps without progress, or at 100k steps (Section 5), plus the
  per-level evaluation (level 4, Sokoban) launched from AutoAscend saves.
* **Toy** -- the two Appendix A sanity checks (two-state MDP scenarios and the
  AppleRetrieval gridworld) as a smoke evaluation.

Heavy dependencies (torch / NLE / Meta-World) are imported lazily so that
``--help`` and the toy path work on a bare CPU-only machine.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Environment / method registries
# --------------------------------------------------------------------------- #

ENVS: Dict[str, str] = {
    "toy": "configs/toy.yaml",
    "robotic_sequence": "configs/robotic_sequence.yaml",
    "metaworld": "configs/robotic_sequence.yaml",
    "montezuma": "configs/montezuma.yaml",
    "nethack": "configs/nethack.yaml",
}

#: Canonical environment aliases -> internal environment keys.
ENV_ALIASES: Dict[str, str] = {
    "robotic": "robotic_sequence",
    "robotic-sequence": "robotic_sequence",
    "roboticsequence": "robotic_sequence",
    "meta-world": "robotic_sequence",
    "metaworld": "robotic_sequence",
    "montezumas-revenge": "montezuma",
    "montezuma_revenge": "montezuma",
    "mz": "montezuma",
    "nethack-challenge": "nethack",
    "human-monk": "nethack",
    "humnmonk": "nethack",
    "two_state": "toy",
    "two-state-mdp": "toy",
    "apple_retrieval": "toy",
    "apple-retrieval": "toy",
}

#: Retention-variant aliases used across configs, trainers and the paper.
METHOD_ALIASES: Dict[str, str] = {
    "vanilla": "none",
    "ft": "none",
    "finetune": "none",
    "fine_tuning": "none",
    "from_scratch": "scratch",
    "from-scratch": "scratch",
    "kickstarting": "ks",
    "kick-start": "ks",
    "behavioral_cloning": "bc",
    "behavioural_cloning": "bc",
    "replay": "bc",
    "episodic_memory": "em",
    "elastic_weight_consolidation": "ewc",
    "l2": "ewc",
}

#: Evaluator implemented per environment (module attribute names).
EVAL_EPISODES: Dict[str, int] = {
    "toy": 200,
    "robotic_sequence": 20,
    "montezuma": 100,
    "nethack": 1000,
}

#: Number of seeds averaged by default (paper uses >= 20 for Meta-World).
NUM_SEEDS: Dict[str, int] = {
    "toy": 5,
    "robotic_sequence": 20,
    "montezuma": 3,
    "nethack": 3,
}

DEFAULT_CONFIDENCE = 0.90
DEFAULT_ENV = "robotic_sequence"

#: Paper reference numbers used for the reproduction sanity checks.
PAPER_REFERENCE: Dict[str, Any] = {
    "nethack": {
        "scratch": 776.0,
        "none": None,
        "ewc": 3976.0,
        "bc": 7610.0,
        "ks": 10588.0,
        "ks_ci": 672.0,
        "pi_star_score": 5000.0,
        "ordering": ("ks", "bc", "ewc", "none"),
    },
    "montezuma": {
        "m1_target_return": 7000.0,
        "ordering": ("bc", "ewc", "none"),
    },
    "robotic_sequence": {
        "pretrained_far_success": 1.0,
        "ordering": ("bc", "em", "ewc", "none"),
        "num_seeds": 20,
        "confidence": 0.90,
    },
}

#: NetHack per-level evaluation targets (Section 5 / Appendix B.1).
NETHACK_PER_LEVEL_TARGETS: Tuple[str, ...] = ("level_4", "sokoban")

MAX_SERIALISED_ITEMS = 500


# --------------------------------------------------------------------------- #
# Small statistics helpers (SciPy free)
# --------------------------------------------------------------------------- #


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile for ``confidence`` (SciPy free)."""

    table = {
        0.50: 0.6745,
        0.68: 0.9945,
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.9600,
        0.98: 2.3263,
        0.99: 2.5758,
    }
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return 1.6449
    if confidence in table:
        return table[confidence]
    # Peter Acklam's rational approximation to the inverse normal CDF.
    p = 1.0 - (1.0 - confidence) / 2.0
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def summarize(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean / std / 90% CI half-width of a list of per-seed values."""

    nums: List[float] = []
    for value in values or []:
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            nums.append(f)
    n = len(nums)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "half_width": float("nan"),
                "n": 0, "confidence": confidence}
    mean = sum(nums) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in nums) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    half = z_for(confidence) * std / math.sqrt(max(n, 1))
    return {"mean": mean, "std": std, "half_width": half, "n": n,
            "confidence": confidence, "lo": mean - half, "hi": mean + half}


# --------------------------------------------------------------------------- #
# Import / config plumbing
# --------------------------------------------------------------------------- #


def _import(module: str, attr: Optional[str] = None) -> Any:
    """Import ``module`` trying a few package layouts, optionally get ``attr``."""

    candidates: List[str] = []
    for name in (module, "src." + module) if not module.startswith("src.") else (module,):
        if name not in candidates:
            candidates.append(name)
        if "." in name:
            short = name.split(".", 1)[1]
            if short not in candidates:
                candidates.append(short)
    last_error: Optional[BaseException] = None
    for name in candidates:
        try:
            mod = importlib.import_module(name)
        except BaseException as exc:  # pragma: no cover - defensive
            last_error = exc
            continue
        if attr is None:
            return mod
        if hasattr(mod, attr):
            return getattr(mod, attr)
        last_error = AttributeError(f"{name} has no attribute {attr!r}")
    if last_error is not None:
        raise last_error
    raise ImportError(module)


def _forward(fn: Callable[..., Any], **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments its signature accepts."""

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(**kwargs)
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return fn(**accepted)


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """Dotted-path lookup tolerant to dict / dataclass / object configs."""

    if cfg is None:
        return default
    node = cfg
    for part in path.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(part, None)
        elif hasattr(node, part):
            node = getattr(node, part)
        else:
            return default
    return default if node is None else node


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_method(method: Optional[str]) -> str:
    """Map a retention-method alias to its canonical short name."""

    if method is None:
        return "none"
    key = str(method).strip().lower().replace(" ", "_")
    return METHOD_ALIASES.get(key, key)


def normalize_env(env: Optional[str], cfg: Any = None) -> str:
    """Resolve the environment key from the argument, environment or config."""

    candidate = env or os.environ.get("FTRL_ENV") or _cfg_get(cfg, "name") or _cfg_get(cfg, "env_name")
    if not candidate:
        return DEFAULT_ENV
    key = str(candidate).strip().lower().replace(" ", "_")
    key = ENV_ALIASES.get(key, key)
    if key in ENV_EVALUATORS:
        return key
    return DEFAULT_ENV


def resolve_config_path(config: Optional[str], env: str) -> Optional[str]:
    """Best-effort resolution of a config path."""

    if config:
        if os.path.isfile(config):
            return config
        for candidate in (
            os.path.join("configs", os.path.basename(config)),
            os.path.join(os.path.dirname(__file__), "configs", os.path.basename(config)),
        ):
            if os.path.isfile(candidate):
                return candidate
        return config
    default = ENVS.get(env)
    if not default:
        return None
    for candidate in (
        os.path.join(os.path.dirname(__file__), default),
        default,
    ):
        if os.path.isfile(candidate):
            return candidate
    return default


def load_cfg(config: Optional[str] = None, env: Optional[str] = None,
             overrides: Optional[Sequence[str]] = None) -> Tuple[Any, str]:
    """Load a config object (or ``None``) plus the resolved environment key."""

    cfg: Any = None
    path = resolve_config_path(config, env or DEFAULT_ENV)
    if path:
        try:
            load_config = _import("common.config", "load_config")
            cfg = load_config(path, overrides)
        except BaseException:
            cfg = None
    env_key = normalize_env(env, cfg)
    return cfg, env_key


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Best-effort conversion of arbitrary evaluation outputs to JSON."""

    if depth > 6:
        return repr(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in list(value.items())[:MAX_SERIALISED_ITEMS]}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in list(value)[:MAX_SERIALISED_ITEMS]]
    for attr in ("as_dict", "to_dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn(), depth + 1)
            except BaseException:
                pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist(), depth + 1)
        except BaseException:
            pass
    return repr(value)


# --------------------------------------------------------------------------- #
# Toy evaluation (Appendix A sanity checks)
# --------------------------------------------------------------------------- #


def evaluate_toy(cfg: Any = None, *, seed: Optional[int] = None, num_episodes: Optional[int] = None,
                 output_dir: Optional[str] = None, verbose: bool = True, **kwargs: Any) -> Dict[str, Any]:
    """Evaluate the two Appendix A toy examples (two-state MDP + AppleRetrieval)."""

    result: Dict[str, Any] = {"env": "toy", "status": "ok", "scenarios": {}, "apple_retrieval": {}}
    ts = None
    try:
        ts = _import("toy.two_state_mdp")
    except BaseException as exc:
        result["two_state_error"] = repr(exc)

    if ts is not None:
        for name in ("coverage_gap", "imperfect_cloning"):
            try:
                summary = ts.run_scenario(name, config=cfg, plot=None)
            except TypeError:
                summary = ts.run_scenario(name)
            except BaseException as exc:  # pragma: no cover - defensive
                result["scenarios"][name] = {"error": repr(exc)}
                continue
            result["scenarios"][name] = _jsonable(summary)

    apple = None
    try:
        apple = _import("toy.apple_retrieval")
    except BaseException as exc:
        result["apple_error"] = repr(exc)

    if apple is not None:
        M = _first(_cfg_get(cfg, "apple_retrieval.M"), 30)
        c = _first(_cfg_get(cfg, "apple_retrieval.c"), 1.0)
        try:
            payload = _forward(
                apple.run_apple_retrieval,
                M=M,
                c=c,
                seed=_first(seed, 0),
                config=cfg,
                record_trace=False,
            )
            result["apple_retrieval"] = _jsonable(payload)
        except BaseException as exc:  # pragma: no cover - defensive
            result["apple_retrieval"] = {"error": repr(exc)}

    reference = {
        "coverage_gap": {"optimum": 0.11, "value": 2.22},
        "imperfect_cloning": {"optimum": 0.08, "value": 9.93},
    }
    result["paper_reference"] = reference
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "toy_eval.json"), "w", encoding="utf-8") as fh:
                json.dump(_jsonable(result), fh, indent=2)
            result["summary_path"] = os.path.join(output_dir, "toy_eval.json")
        except BaseException:  # pragma: no cover - defensive
            pass
    return result


# --------------------------------------------------------------------------- #
# RoboticSequence (Meta-World) evaluation
# --------------------------------------------------------------------------- #


def evaluate_robotic_sequence(cfg: Any = None, *, checkpoint: Optional[str] = None,
                              method: str = "none", num_episodes: Optional[int] = None,
                              seed: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                              num_seeds: Optional[int] = None, stub: Optional[bool] = None,
                              device: Optional[str] = None, levels: Optional[Sequence[str]] = None,
                              output_dir: Optional[str] = None, verbose: bool = True,
                              **kwargs: Any) -> Dict[str, Any]:
    """Per-stage success rates (Figure 7), aggregated over seeds with 90% CIs."""

    from src.robotic_sequence import env as rs_env  # type: ignore

    method = normalize_method(method)
    task_order = _first(_cfg_get(cfg, "env.task_order"), "main")
    tasks = tuple(rs_env.tasks_for(task_order))
    n_episodes = int(_first(num_episodes, _cfg_get(cfg, "finetune.eval_episodes"),
                            _cfg_get(cfg, "eval.episodes"), EVAL_EPISODES["robotic_sequence"]))
    seed_list = _resolve_seeds(seeds, seed, num_seeds, NUM_SEEDS["robotic_sequence"])
    use_stub = bool(_first(stub, _cfg_get(cfg, "env.stub"), False))
    device = _first(device, _cfg_get(cfg, "compute.device"), "cpu")

    env_kwargs = dict(
        task_order=task_order,
        time_limit=int(_first(_cfg_get(cfg, "env.time_limit"), rs_env.TIME_LIMIT)),
        beta=float(_first(_cfg_get(cfg, "env.beta"), rs_env.BETA)),
        stub=use_stub,
        append_timestep=bool(_first(_cfg_get(cfg, "env.append_timestep"), True)),
        append_stage_onehot=bool(_first(_cfg_get(cfg, "env.append_stage_onehot"), False)),
    )

    per_seed: List[Dict[str, Any]] = []
    agent = None
    for s in seed_list:
        entry: Dict[str, Any] = {"seed": int(s)}
        try:
            probe = rs_env.RoboticSequenceEnv(seed=int(s), **env_kwargs)
            try:
                if agent is None:
                    agent = _build_robotic_agent(
                        cfg, probe.observation_dim, probe.action_dim, probe.n_stages,
                        device=device, seed=int(s), stub=use_stub, checkpoint=checkpoint,
                    )
                if checkpoint:
                    _load_robotic_checkpoint(agent, checkpoint)
                if agent is None:
                    raise RuntimeError("no SAC agent available for evaluation")
                success = _per_stage_success(agent, tasks, n_episodes, int(s), env_kwargs)
                entry["success_rates"] = {str(k): float(v) for k, v in success.items()
                                          if isinstance(v, (int, float))}
                far = [v for k, v in entry["success_rates"].items()
                       if k in tuple(rs_env.FAR_TASKS)]
                close = [v for k, v in entry["success_rates"].items()
                         if k in tuple(rs_env.CLOSE_TASKS)]
                entry["far_success"] = sum(far) / len(far) if far else float("nan")
                entry["close_success"] = sum(close) / len(close) if close else float("nan")
                entry["overall_success"] = (
                    sum(entry["success_rates"].values()) / len(entry["success_rates"])
                    if entry["success_rates"] else float("nan")
                )
            finally:
                try:
                    probe.close()
                except BaseException:
                    pass
        except BaseException as exc:  # pragma: no cover - defensive
            entry["error"] = repr(exc)
        per_seed.append(entry)

    aggregates: Dict[str, Any] = {}
    stage_names = list(tasks)
    for stage_name in stage_names:
        values = [e.get("success_rates", {}).get(stage_name) for e in per_seed]
        aggregates[stage_name] = summarize([v for v in values if v is not None])
    for key in ("far_success", "close_success", "overall_success"):
        aggregates[key] = summarize([e.get(key) for e in per_seed if e.get(key) is not None])

    result: Dict[str, Any] = {
        "env": "robotic_sequence",
        "method": method,
        "status": "ok",
        "tasks": list(tasks),
        "task_order": task_order,
        "num_episodes": n_episodes,
        "seeds": [int(s) for s in seed_list],
        "confidence": DEFAULT_CONFIDENCE,
        "per_stage": aggregates,
        "per_seed": per_seed,
        "checkpoint": checkpoint,
        "paper_reference": PAPER_REFERENCE["robotic_sequence"],
        "stub": use_stub,
    }

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "robotic_sequence_eval.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(_jsonable(result), fh, indent=2)
            result["summary_path"] = path
        except BaseException:  # pragma: no cover - defensive
            pass
    return result


def _resolve_seeds(seeds: Optional[Sequence[int]], seed: Optional[int],
                   num_seeds: Optional[int], default: int) -> List[int]:
    """Normalise the different ways a caller can request seeds."""

    if seeds:
        return [int(s) for s in seeds]
    if seed is not None:
        return [int(seed)]
    count = int(num_seeds or default)
    return list(range(max(count, 1)))


def _build_robotic_agent(cfg: Any, obs_dim: int, action_dim: int, n_stages: int, *,
                         device: str = "cpu", seed: int = 0, stub: bool = False,
                         checkpoint: Optional[str] = None) -> Any:
    """Construct a SAC agent for RoboticSequence using the SAC builder."""

    try:
        sac = _import("robotic_sequence.sac")
    except BaseException:
        return None
    sac_cfg = None
    try:
        sac_cfg = sac.SACConfig.from_config(cfg) if cfg is not None else None
    except BaseException:
        sac_cfg = None
    stage_id_in_obs = bool(_first(_cfg_get(cfg, "env.append_stage_onehot"), False))
    kwargs = dict(
        obs_dim=int(obs_dim), action_dim=int(action_dim), n_stages=int(n_stages),
        device=device, retention=None, seed=int(seed), stage_id_in_obs=stage_id_in_obs,
    )
    try:
        if sac_cfg is not None:
            return sac.build_sac_agent(sac_cfg, **kwargs)
    except BaseException:
        pass
    try:
        return sac.build_sac_agent(cfg, **kwargs)
    except BaseException:
        return None


def _load_robotic_checkpoint(agent: Any, checkpoint: str) -> None:
    """Load a SAC checkpoint into an already-constructed agent (best effort)."""

    if agent is None or not checkpoint:
        return
    for name in ("load_pretrained", "load"):
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                fn(checkpoint)
                return
            except BaseException:
                continue


def _per_stage_success(agent: Any, tasks: Sequence[str], num_episodes: int, seed: int,
                       env_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Per-stage success rates using the trainer's evaluator when available."""

    try:
        trainer = _import("robotic_sequence.train_robotic")
        if hasattr(trainer, "evaluate_agent"):
            return _forward(
                trainer.evaluate_agent,
                agent=agent,
                tasks=tuple(tasks),
                num_episodes=num_episodes,
                seed=seed,
                stub=env_kwargs.get("stub", False),
                deterministic=True,
            )
    except BaseException:
        pass
    rs_env = _import("robotic_sequence.env")
    return _forward(
        rs_env.per_stage_success_rate,
        policy=getattr(agent, "policy", agent),
        task_order=env_kwargs.get("task_order", "main"),
        num_episodes=num_episodes,
        seed=seed,
        stub=env_kwargs.get("stub", False),
        deterministic=True,
    )


def evaluate_robotic_forward_transfer(cfg: Any = None, *, results_dir: Optional[str] = None,
                                      methods: Sequence[str] = ("none", "ewc", "bc", "em"),
                                      baseline: str = "scratch",
                                      prefix_lengths: Sequence[int] = (1, 2, 3, 4),
                                      **kwargs: Any) -> Dict[str, Any]:
    """Offline Table-6 forward-transfer table from logged RoboticSequence runs."""

    results_dir = _first(results_dir, _cfg_get(cfg, "output_dir"), "results/robotic_sequence")
    try:
        ft = _import("analysis.forward_transfer")
        table = ft.compute_table(results_dir, methods=tuple(methods), baseline=baseline,
                                 prefix_lengths=tuple(prefix_lengths))
        return {"env": "robotic_sequence", "status": "ok", "results_dir": results_dir,
                "forward_transfer": _jsonable(table)}
    except BaseException as exc:  # pragma: no cover - defensive
        return {"env": "robotic_sequence", "status": "unavailable", "results_dir": results_dir,
                "error": repr(exc)}


# --------------------------------------------------------------------------- #
# Montezuma's Revenge evaluation
# --------------------------------------------------------------------------- #


def evaluate_montezuma(cfg: Any = None, *, checkpoint: Optional[str] = None,
                       method: str = "none", num_episodes: Optional[int] = None,
                       seed: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                       num_seeds: Optional[int] = None, stub: Optional[bool] = None,
                       device: Optional[str] = None, output_dir: Optional[str] = None,
                       verbose: bool = True, **kwargs: Any) -> Dict[str, Any]:
    """Mean episode return and Room-7 success rate (Table 2 / Section 3)."""

    method = normalize_method(method)
    n_episodes = int(_first(num_episodes, _cfg_get(cfg, "eval.episodes"),
                            _cfg_get(cfg, "eval_episodes"), EVAL_EPISODES["montezuma"]))
    seed_list = _resolve_seeds(seeds, seed, num_seeds, NUM_SEEDS["montezuma"])
    use_stub = bool(_first(stub, _cfg_get(cfg, "stub"), False))
    device = _first(device, _cfg_get(cfg, "compute.device"), "cpu")
    far_room = int(_first(_cfg_get(cfg, "env.far_room"), 7))

    per_seed: List[Dict[str, Any]] = []
    for s in seed_list:
        entry: Dict[str, Any] = {"seed": int(s)}
        try:
            agent = _load_montezuma_agent(cfg, checkpoint, stub=use_stub, device=device)
            evaluation = _montezuma_eval(agent, n_episodes, int(s), use_stub, far_room)
            entry.update({k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in evaluation.items() if k != "returns"})
            if "returns" in evaluation:
                entry["returns"] = _jsonable(evaluation["returns"])
        except BaseException as exc:  # pragma: no cover - defensive
            entry["error"] = repr(exc)
        per_seed.append(entry)

    aggregates: Dict[str, Any] = {}
    for key in ("return_mean", "room7_success_rate", "max_room", "mean_length"):
        aggregates[key] = summarize([e.get(key) for e in per_seed if isinstance(e.get(key), (int, float))])

    result: Dict[str, Any] = {
        "env": "montezuma",
        "method": method,
        "status": "ok",
        "num_episodes": n_episodes,
        "seeds": [int(s) for s in seed_list],
        "confidence": DEFAULT_CONFIDENCE,
        "aggregates": aggregates,
        "per_seed": per_seed,
        "checkpoint": checkpoint,
        "paper_reference": PAPER_REFERENCE["montezuma"],
        "stub": use_stub,
    }
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "montezuma_eval.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(_jsonable(result), fh, indent=2)
            result["summary_path"] = path
        except BaseException:  # pragma: no cover - defensive
            pass
    return result


def _load_montezuma_agent(cfg: Any, checkpoint: Optional[str], *, stub: bool = False,
                          device: str = "cpu") -> Any:
    """Load an M1/M2 agent from a checkpoint (or build a fresh stub agent)."""

    path = _first(checkpoint, _cfg_get(cfg, "checkpoint"), _cfg_get(cfg, "m2_checkpoint"),
                  _cfg_get(cfg, "m1_checkpoint"))
    if path:
        try:
            m1 = _import("montezuma.m1_train")
            if hasattr(m1, "load_m1_agent"):
                return _forward(m1.load_m1_agent, path=path, cfg=cfg, stub=stub,
                                device=device, load_optimizer=False)
        except BaseException:
            pass
        try:
            ppo = _import("montezuma.ppo_rnd")
            agent = ppo.PPORNDAgent(config=cfg, obs_shape=None, num_actions=None, device=device)
            agent.load(path)
            return agent
        except BaseException:
            pass
    try:
        ppo = _import("montezuma.ppo_rnd")
        return ppo.PPORNDAgent(config=cfg, device=device)
    except BaseException:
        return None


def _montezuma_eval(agent: Any, num_episodes: int, seed: int, stub: bool, far_room: int) -> Dict[str, Any]:
    """Run the Montezuma evaluation protocol using the env module helpers."""

    mz_env = _import("montezuma.env")
    policy = getattr(agent, "policy", agent)
    evaluation = _forward(
        mz_env.evaluate_policy,
        policy=policy,
        env=None,
        num_episodes=num_episodes,
        seed=seed,
        deterministic=True,
        stub=stub,
        far_room=far_room,
    )
    out = dict(evaluation) if isinstance(evaluation, dict) else {}
    if "room7_success_rate" not in out and hasattr(mz_env, "room7_success_rate"):
        try:
            out["room7_success_rate"] = float(_forward(
                mz_env.room7_success_rate, policy=policy, env=None,
                num_episodes=num_episodes, seed=seed, stub=stub, far_room=far_room,
            ))
        except BaseException:
            pass
    if "return_mean" in out and "episode_return" not in out:
        out["episode_return"] = out["return_mean"]
    return out


# --------------------------------------------------------------------------- #
# NetHack evaluation
# --------------------------------------------------------------------------- #


def evaluate_nethack(cfg: Any = None, *, checkpoint: Optional[str] = None,
                     method: str = "none", num_episodes: Optional[int] = None,
                     seed: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                     num_seeds: Optional[int] = None, stub: Optional[bool] = None,
                     device: Optional[str] = None, per_level: bool = False,
                     levels: Optional[Sequence[str]] = None, save_dir: Optional[str] = None,
                     output_dir: Optional[str] = None, verbose: bool = True,
                     **kwargs: Any) -> Dict[str, Any]:
    """Full NetHack evaluation (1000 episodes) plus optional per-level evaluation."""

    method = normalize_method(method)
    n_episodes = int(_first(num_episodes, _cfg_get(cfg, "eval.episodes"),
                            EVAL_EPISODES["nethack"]))
    seed_list = _resolve_seeds(seeds, seed, num_seeds, NUM_SEEDS["nethack"])
    use_stub = bool(_first(stub, _cfg_get(cfg, "stub"), False))
    device = _first(device, _cfg_get(cfg, "compute.device"), "cpu")
    targets = tuple(levels or _first(_cfg_get(cfg, "eval.per_level_targets"),
                                     NETHACK_PER_LEVEL_TARGETS))

    per_seed: List[Dict[str, Any]] = []
    for s in seed_list:
        entry: Dict[str, Any] = {"seed": int(s)}
        try:
            agent = _load_nethack_agent(cfg, checkpoint, method=method, stub=use_stub, device=device)
            if agent is None:
                raise RuntimeError("no NetHack agent available for evaluation")
            evaluation = _nethack_full_eval(cfg, agent, n_episodes, int(s), use_stub)
            entry.update({k: (float(v) if isinstance(v, (int, float)) else v)
                          for k, v in evaluation.items() if k not in ("returns", "episodes_detail")})
            if per_level:
                entry["per_level"] = _nethack_per_level(
                    cfg, agent, targets, s, use_stub, save_dir=save_dir,
                    num_episodes=n_episodes, output_dir=output_dir,
                )
        except BaseException as exc:  # pragma: no cover - defensive
            entry["error"] = repr(exc)
        per_seed.append(entry)

    aggregates: Dict[str, Any] = {}
    metric_keys = ("score", "mean_score", "turns", "steps", "dlvl", "xplvl", "gold", "eating")
    for key in metric_keys:
        values = [e.get(key) for e in per_seed if isinstance(e.get(key), (int, float))]
        if values:
            aggregates[key] = summarize(values)

    result: Dict[str, Any] = {
        "env": "nethack",
        "method": method,
        "status": "ok",
        "num_episodes": n_episodes,
        "seeds": [int(s) for s in seed_list],
        "confidence": DEFAULT_CONFIDENCE,
        "aggregates": aggregates,
        "per_seed": per_seed,
        "checkpoint": checkpoint,
        "per_level": bool(per_level),
        "per_level_targets": list(targets),
        "paper_reference": PAPER_REFERENCE["nethack"],
        "stub": use_stub,
    }
    result["ordering_check"] = check_ordering({method: aggregates})
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, "nethack_eval.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(_jsonable(result), fh, indent=2)
            result["summary_path"] = path
        except BaseException:  # pragma: no cover - defensive
            pass
    return result


def _load_nethack_agent(cfg: Any, checkpoint: Optional[str], *, method: str = "none",
                        stub: bool = False, device: str = "cpu") -> Any:
    """Load the pi* / fine-tuned NetHack model into an APPO agent."""

    path = _first(checkpoint, _cfg_get(cfg, "checkpoint"),
                  _cfg_get(cfg, "pretrained_checkpoint"))
    try:
        appo = _import("nethack.appo_runner")
    except BaseException:
        return None
    agent = None
    # Preferred: the classmethod that wires model + retention + checkpoint.
    build = getattr(appo.APPOAgent, "build", None)
    if callable(build):
        try:
            agent = _forward(appo.APPOAgent.build, cfg=cfg, model=None, method=method,
                             teacher=None, fisher=None, bc_dataset=None,
                             checkpoint=path, device=device)
        except BaseException:
            agent = None
    if agent is None:
        try:
            model = _forward(appo.build_nethack_model, config=cfg, checkpoint=path,
                             device=device, stub=stub)
            agent = _forward(appo.APPOAgent, model=model, config=cfg, device=device)
        except BaseException:
            agent = None
    return agent


def _nethack_full_eval(cfg: Any, agent: Any, num_episodes: int, seed: int,
                       stub: bool) -> Dict[str, Any]:
    """1000-episode evaluation with the Section 5 stopping rules."""

    try:
        appo = _import("nethack.appo_runner")
        evaluation = _forward(
            appo.evaluate, agent=agent, env=None, num_episodes=num_episodes,
            seed=seed, stub=stub, config=cfg,
        )
        if isinstance(evaluation, dict):
            return evaluation
    except BaseException:
        pass
    # Fallback: build an env and roll the policy out directly.
    try:
        nh_env = _import("nethack.env")
        return _forward(
            nh_env.evaluate_policy, policy=getattr(agent, "model", agent), env=None,
            num_episodes=num_episodes, seed=seed, stub=stub, deterministic=True,
        )
    except BaseException as exc:  # pragma: no cover - defensive
        return {"error": repr(exc)}


def _nethack_per_level(cfg: Any, agent: Any, targets: Sequence[str], seed: int, stub: bool, *,
                       save_dir: Optional[str] = None, num_episodes: Optional[int] = None,
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Per-level (level 4 / Sokoban) evaluation from AutoAscend saves."""

    try:
        ple = _import("nethack.per_level_eval")
    except BaseException as exc:
        return {"status": "unavailable", "error": repr(exc)}
    fn = getattr(ple, "run_per_level_eval", None) or getattr(ple, "evaluate_per_level", None)
    if fn is None:
        return {"status": "unavailable", "error": "no per-level evaluator exported"}
    try:
        payload = _forward(
            fn, agent=agent, step=None, levels=tuple(targets), num_episodes=num_episodes,
            save_dir=_first(save_dir, _cfg_get(cfg, "eval.save_dir")), stub=stub, seed=seed,
            confidence=DEFAULT_CONFIDENCE,
        )
    except BaseException as exc:  # pragma: no cover - defensive
        return {"status": "error", "error": repr(exc)}
    return _jsonable(payload)


# --------------------------------------------------------------------------- #
# Ordering sanity check
# --------------------------------------------------------------------------- #


def check_ordering(aggregates: Dict[str, Any], env: Optional[str] = None) -> Dict[str, Any]:
    """Compare observed means against the ordering claimed in the paper."""

    env = env or "nethack"
    reference = PAPER_REFERENCE.get(env, {})
    expected = reference.get("ordering")
    means: Dict[str, float] = {}
    for method, payload in (aggregates or {}).items():
        value = None
        if isinstance(payload, dict):
            for key in ("score", "final_score", "return_mean", "episode_return"):
                entry = payload.get(key)
                if isinstance(entry, dict):
                    value = entry.get("mean")
                elif isinstance(entry, (int, float)):
                    value = float(entry)
                if value is not None:
                    break
        if value is not None:
            means[normalize_method(method)] = value
    observed = tuple(sorted(means, key=lambda k: means[k], reverse=True))
    check: Dict[str, Any] = {"expected": list(expected) if expected else None,
                             "observed": list(observed), "means": means,
                             "matches": None, "num_methods": len(means)}
    if expected and means:
        filtered = [m for m in observed if m in expected]
        check["matches"] = filtered == [m for m in expected if m in means]
    return check


# --------------------------------------------------------------------------- #
# Registry + top level dispatcher
# --------------------------------------------------------------------------- #

ENV_EVALUATORS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "toy": evaluate_toy,
    "robotic_sequence": evaluate_robotic_sequence,
    "montezuma": evaluate_montezuma,
    "nethack": evaluate_nethack,
}


def evaluate(env: Optional[str] = None, cfg: Any = None, *, method: Optional[str] = None,
             checkpoint: Optional[str] = None, num_episodes: Optional[int] = None,
             seed: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
             num_seeds: Optional[int] = None, stub: Optional[bool] = None,
             device: Optional[str] = None, per_level: bool = False,
             levels: Optional[Sequence[str]] = None, save_dir: Optional[str] = None,
             output_dir: Optional[str] = None, verbose: bool = True, logger: Any = None,
             progress_fn: Optional[Callable[..., Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Evaluate a checkpoint / policy on the requested environment.

    Parameters mirror the flags exposed by :func:`main`; only the arguments the
    selected evaluator accepts are forwarded.
    """

    env_key = normalize_env(env, cfg)
    evaluator = ENV_EVALUATORS.get(env_key)
    started = time.time()
    payload: Dict[str, Any]
    if evaluator is None:
        payload = {"env": env_key, "status": "unavailable",
                   "error": f"no evaluator registered for {env_key!r}"}
    else:
        kwargs = {k: v for k, v in kwargs.items() if k != "config"}
        try:
            payload = _forward(
                evaluator, cfg=cfg, method=normalize_method(method), checkpoint=checkpoint,
                num_episodes=num_episodes, seed=seed, seeds=seeds, num_seeds=num_seeds,
                stub=stub, device=device, levels=levels, save_dir=save_dir,
                output_dir=output_dir, verbose=verbose, logger=logger, progress_fn=progress_fn,
                per_level=per_level, **kwargs,
            )
        except BaseException as exc:  # pragma: no cover - defensive
            payload = {"env": env_key, "status": "error", "error": repr(exc)}
    if not isinstance(payload, dict):
        payload = {"env": env_key, "status": "ok", "result": _jsonable(payload)}
    payload.setdefault("env", env_key)
    payload.setdefault("method", normalize_method(method))
    payload.setdefault("status", "ok")
    payload["elapsed"] = time.time() - started
    if logger is not None:
        try:
            logger.info("evaluation finished for %s: status=%s", env_key, payload.get("status"))
        except BaseException:
            pass
    return payload


def evaluate_checkpoint(checkpoint: str, env: Optional[str] = None, cfg: Any = None,
                        **kwargs: Any) -> Dict[str, Any]:
    """Convenience wrapper: evaluate a checkpoint on ``env``."""

    kwargs.setdefault("checkpoint", checkpoint)
    return evaluate(env=env, cfg=cfg, **kwargs)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Evaluate a policy/checkpoint for one of the paper's environments.",
    )
    parser.add_argument("--env", "-e", default=None,
                        help=f"environment to evaluate (default: {DEFAULT_ENV}); one of {sorted(ENV_EVALUATORS)}")
    parser.add_argument("--config", "-c", default=None, help="path to a YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=None,
                        help="config override key=value (repeatable)")
    parser.add_argument("--method", "-m", default="none",
                        help="retention variant the checkpoint belongs to (none/ewc/bc/ks/em/scratch)")
    parser.add_argument("--checkpoint", "-k", default=None, help="checkpoint to evaluate")
    parser.add_argument("--num-episodes", "-n", type=int, default=None,
                        help="episodes per evaluation (default depends on the environment)")
    parser.add_argument("--seed", type=int, default=None, help="single seed (default: 0)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None, help="explicit list of seeds")
    parser.add_argument("--num-seeds", type=int, default=None, help="number of seeds 0..N-1")
    parser.add_argument("--device", default=None, help="torch device (e.g. cpu or cuda)")
    parser.add_argument("--stub", action="store_true", default=None,
                        help="use the dependency-free stub environments")
    parser.add_argument("--per-level", action="store_true", help="run the NetHack per-level evaluation")
    parser.add_argument("--levels", nargs="*", default=None, help="per-level targets (level_4 sokoban)")
    parser.add_argument("--save-dir", default=None, help="directory with AutoAscend saves")
    parser.add_argument("--output-dir", "-o", default=None, help="where to write the JSON summary")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")
    parser.add_argument("--list", action="store_true", help="list environments and evaluators")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(ENV_EVALUATORS):
            print(f"{name:20s} episodes={EVAL_EPISODES.get(name)} seeds={NUM_SEEDS.get(name)}")
        return 0

    cfg, env_key = load_cfg(args.config, args.env, args.overrides)
    payload = evaluate(
        env=env_key,
        cfg=cfg,
        method=args.method,
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        seed=args.seed,
        seeds=args.seeds,
        num_seeds=args.num_seeds,
        stub=args.stub,
        device=args.device,
        per_level=args.per_level,
        levels=args.levels,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    if args.json:
        print(json.dumps(_jsonable(payload), indent=2))
    else:
        status = payload.get("status")
        print(f"[evaluate] env={env_key} method={payload.get('method')} status={status}")
        aggregates = payload.get("aggregates") or payload.get("per_stage") or {}
        for key, value in list(aggregates.items())[:12]:
            if isinstance(value, dict):
                mean = value.get("mean")
                half = value.get("half_width")
                if isinstance(mean, float) and isinstance(half, float):
                    print(f"  {key:28s} {mean:12.3f} +/- {half:.3f} (n={value.get('n')})")
                else:
                    print(f"  {key:28s} {value}")
            else:
                print(f"  {key:28s} {value}")
        if payload.get("summary_path"):
            print(f"  summary -> {payload['summary_path']}")

    return 0 if payload.get("status") in ("ok", None) else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
