"""Ablation study driver for RICE (ICML 2024, Cheng et al.).

This script is a thin CLI dispatcher over the experiment drivers implemented in
``experiments/``.  It reproduces the *ablations* of the paper:

* ``explanation``  -- Experiment III (Sec. 4.2/4.3, Table 1 right block and
  Appendix C.3 Table 6): fix the refiner to RICE (Algorithm 2 / Ours) and vary
  the Stage-1 explanation in ``{random, statemask, ours, integrated_gradients,
  airs}``.
* ``refining``     -- Experiment II (Table 1 left block): fix the explanation to
  ``ours`` and vary the refining method in ``{ours, ppo_finetune, statemask_r,
  jsrl}`` (plus ``sac/gail/sil`` for Experiment IV / Table 5 when available).
* ``hyperparams``  -- Experiment V (Fig. 7/8/9): sweep the mixed-init
  probability ``p``, the RND coefficient ``lambda`` and the blinding bonus
  ``alpha``.
* ``sil``          -- Table 5: RICE vs. Self-Imitation Learning on the four
  MuJoCo dense environments (optional; requires :mod:`rice.baselines.sil`).
* ``all``          -- run every ablation above for the selected applications.

Every sub-run delegates to the corresponding ``experiments/exp*`` module, so
this file contains no duplicated RL logic.  Results are written to
``<results_dir>/ablation/`` as JSON plus a human readable ``.txt`` summary.

Usage::

    python scripts/run_ablation.py --ablation explanation --envs hopper walker2d
    python scripts/run_ablation.py --ablation hyperparams --envs hopper
    python scripts/run_ablation.py --ablation all --envs hopper --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------------
# make the repository importable when the script is executed directly
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# defensive project imports (the driver must stay importable/introspectable
# even when torch / SB3 / MuJoCo are not installed)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on install
    from rice.utils.io import ensure_dir, get_config, save_json
except Exception:  # pragma: no cover
    import yaml as _yaml

    def ensure_dir(path, *args, **kwargs):
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    def get_config(name="default", config_dir=None):
        cfg_dir = config_dir or os.path.join(_ROOT, "configs")
        base = {}
        default = os.path.join(cfg_dir, "default.yaml")
        if os.path.isfile(default):
            with open(default, "r", encoding="utf-8") as handle:
                base = _yaml.safe_load(handle) or {}
        stem = str(name)
        for cand in (stem, os.path.splitext(stem)[0]):
            path = cand if os.path.isfile(cand) else os.path.join(cfg_dir, f"{cand}.yaml")
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as handle:
                    override = _yaml.safe_load(handle) or {}
                _deep_update(base, override)
                break
        return base

    def _deep_update(base, override):
        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                _deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def save_json(obj, path, indent=2):
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=indent, default=str)
        return path


try:  # pragma: no cover
    from rice.utils.logging import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name="rice", out_dir=None, level=None):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(level or logging.INFO)
        return logger


try:  # pragma: no cover
    from rice.utils.seeding import set_seed
except Exception:  # pragma: no cover

    def set_seed(seed, deterministic=False):
        import random

        import numpy as np

        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass
        return seed


# ---------------------------------------------------------------------------
# experiment drivers (optional -- reported as unavailable when missing)
# ---------------------------------------------------------------------------
try:
    from experiments import exp2_refine_effectiveness as exp2
except Exception:  # pragma: no cover
    try:
        import exp2_refine_effectiveness as exp2  # type: ignore
    except Exception:
        exp2 = None

try:
    from experiments import exp3_explanation_quality as exp3
except Exception:  # pragma: no cover
    try:
        import exp3_explanation_quality as exp3  # type: ignore
    except Exception:
        exp3 = None

try:
    from experiments import exp5_hyperparams as exp5
except Exception:  # pragma: no cover
    try:
        import exp5_hyperparams as exp5  # type: ignore
    except Exception:
        exp5 = None

try:
    from rice.baselines.sil import sil as _sil_baseline
except Exception:  # pragma: no cover
    _sil_baseline = None


# ---------------------------------------------------------------------------
# constants mirroring the paper
# ---------------------------------------------------------------------------
DEFAULT_ENVS: Sequence[str] = (
    "hopper",
    "walker2d",
    "reacher",
    "halfcheetah",
    "selfish_mining",
    "cage2",
    "autodriving",
)
MUJOCO_ENVS: Sequence[str] = ("hopper", "walker2d", "reacher", "halfcheetah")

EXPLANATION_METHODS: Sequence[str] = (
    "random",
    "statemask",
    "ours",
    "integrated_gradients",
    "airs",
)
CORE_EXPLANATION_METHODS: Sequence[str] = ("random", "statemask", "ours")
REFINING_METHODS: Sequence[str] = ("ours", "ppo_finetune", "statemask_r", "jsrl")
EXTRA_REFINING_METHODS: Sequence[str] = ("sac_finetune", "gail", "sil")

# Experiment V sweep grids (paper Fig. 7 = lambda, Fig. 8 = p, Fig. 9 = alpha)
P_VALUES: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDA_VALUES: Sequence[float] = (0.0, 0.1, 0.01, 0.001)
ALPHA_VALUES: Sequence[float] = (0.01, 0.001, 0.0001)

DEFAULT_SEEDS: Sequence[int] = (0, 1, 2)
ABLATIONS: Sequence[str] = ("explanation", "refining", "hyperparams", "sil", "all")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Nested, defence-in-depth lookup into a config mapping."""
    node = cfg
    for key in keys:
        if node is None:
            return default
        if isinstance(node, dict):
            node = node.get(key, None)
        else:  # best effort attribute access
            node = getattr(node, key, None)
    return default if node is None else node


def _results_dir(cfg: Any, out_dir: Optional[str]) -> str:
    if out_dir:
        return ensure_dir(out_dir)
    root = _cfg_get(cfg, "results_dir", default=os.path.join(_ROOT, "results"))
    return ensure_dir(os.path.join(str(root), "ablation"))


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:
            return repr(obj)
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:  # torch tensors
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return repr(obj)


def _mean_std(values: Iterable[float]):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None, None
    import numpy as np

    return float(np.mean(vals)), float(np.std(vals))


def _format_value(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return str(value)


def available_ablation_backends() -> Dict[str, bool]:
    """Report which experiment drivers / baselines could be imported."""
    return {
        "exp2_refining": exp2 is not None,
        "exp3_explanation": exp3 is not None,
        "exp5_hyperparams": exp5 is not None,
        "sil": _sil_baseline is not None,
    }


# ---------------------------------------------------------------------------
# ablation runners
# ---------------------------------------------------------------------------
def run_explanation_ablation(
    env_ids: Sequence[str] = MUJOCO_ENVS,
    methods: Sequence[str] = CORE_EXPLANATION_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cfg: Any = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Experiment III: fix refine = Ours, vary the Stage-1 explanation."""
    logger = logger or get_logger("rice.ablation")
    report: Dict[str, Any] = {
        "ablation": "explanation",
        "envs": list(env_ids),
        "methods": list(methods),
        "seeds": list(seeds),
        "device": device,
        "results": {},
        "trends": {},
        "errors": {},
    }
    if exp3 is None:
        report["errors"]["driver"] = "experiments.exp3_explanation_quality unavailable"
        logger.error("Experiment III driver unavailable; skipping explanation ablation")
        return report

    for env_id in env_ids:
        try:
            logger.info("[ablation:explanation] env=%s methods=%s", env_id, list(methods))
            result = exp3.run_experiment3(
                env_id,
                cfg=cfg,
                explanations=tuple(methods),
                seeds=tuple(seeds),
                refine_timesteps=refine_timesteps,
                mask_timesteps=mask_timesteps,
                pretrain_timesteps=pretrain_timesteps,
                device=device,
                out_dir=out_dir,
                logger=logger,
                progress=progress,
                eval_episodes=eval_episodes,
                **kwargs,
            )
            report["results"][env_id] = _jsonable(result)
            try:
                report["trends"][env_id] = _jsonable(exp3.check_trends(result))
            except Exception as exc:  # pragma: no cover
                report["trends"][env_id] = {"error": str(exc)}
        except Exception as exc:  # pragma: no cover
            logger.error("[ablation:explanation] env=%s failed: %s", env_id, exc)
            report["errors"][env_id] = traceback.format_exc(limit=3)
    return report


def run_refining_ablation(
    env_ids: Sequence[str] = DEFAULT_ENVS,
    methods: Sequence[str] = REFINING_METHODS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cfg: Any = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    explanation: str = "ours",
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Experiment II: fix explanation = ours, vary the refining method."""
    logger = logger or get_logger("rice.ablation")
    report: Dict[str, Any] = {
        "ablation": "refining",
        "envs": list(env_ids),
        "methods": list(methods),
        "explanation": explanation,
        "seeds": list(seeds),
        "device": device,
        "results": {},
        "trends": {},
        "errors": {},
    }
    if exp2 is None:
        report["errors"]["driver"] = "experiments.exp2_refine_effectiveness unavailable"
        logger.error("Experiment II driver unavailable; skipping refining ablation")
        return report

    for env_id in env_ids:
        try:
            logger.info("[ablation:refining] env=%s methods=%s", env_id, list(methods))
            result = exp2.run_experiment2(
                env_id,
                cfg=cfg,
                methods=tuple(methods),
                seeds=tuple(seeds),
                explanation=explanation,
                refine_timesteps=refine_timesteps,
                mask_timesteps=mask_timesteps,
                pretrain_timesteps=pretrain_timesteps,
                device=device,
                out_dir=out_dir,
                logger=logger,
                progress=progress,
                eval_episodes=eval_episodes,
                **kwargs,
            )
            report["results"][env_id] = _jsonable(result)
            try:
                report["trends"][env_id] = _jsonable(exp2.check_trends(result))
            except Exception as exc:  # pragma: no cover
                report["trends"][env_id] = {"error": str(exc)}
        except Exception as exc:  # pragma: no cover
            logger.error("[ablation:refining] env=%s failed: %s", env_id, exc)
            report["errors"][env_id] = traceback.format_exc(limit=3)
    return report


def run_hyperparam_ablation(
    env_ids: Sequence[str] = MUJOCO_ENVS,
    p_values: Sequence[float] = P_VALUES,
    lambda_values: Sequence[float] = LAMBDA_VALUES,
    alpha_values: Sequence[float] = ALPHA_VALUES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    sweeps: Sequence[str] = ("p", "lambda", "alpha"),
    cfg: Any = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Experiment V: sweep p (Fig. 8), lambda (Fig. 7) and alpha (Fig. 9)."""
    logger = logger or get_logger("rice.ablation")
    report: Dict[str, Any] = {
        "ablation": "hyperparams",
        "envs": list(env_ids),
        "sweeps": list(sweeps),
        "p_values": list(p_values),
        "lambda_values": list(lambda_values),
        "alpha_values": list(alpha_values),
        "seeds": list(seeds),
        "device": device,
        "results": {},
        "trends": {},
        "errors": {},
    }
    if exp5 is None:
        report["errors"]["driver"] = "experiments.exp5_hyperparams unavailable"
        logger.error("Experiment V driver unavailable; skipping hyperparameter ablation")
        return report

    for env_id in env_ids:
        try:
            logger.info("[ablation:hyperparams] env=%s sweeps=%s", env_id, list(sweeps))
            result = exp5.run_experiment5(
                env_id,
                cfg=cfg,
                p_values=tuple(p_values),
                lambda_values=tuple(lambda_values),
                alpha_values=tuple(alpha_values),
                seeds=tuple(seeds),
                sweeps=tuple(sweeps),
                refine_timesteps=refine_timesteps,
                mask_timesteps=mask_timesteps,
                pretrain_timesteps=pretrain_timesteps,
                device=device,
                out_dir=out_dir,
                logger=logger,
                progress=progress,
                eval_episodes=eval_episodes,
                **kwargs,
            )
            report["results"][env_id] = _jsonable(result)
            report["trends"][env_id] = _jsonable(result.get("trends", {}))
        except Exception as exc:  # pragma: no cover
            logger.error("[ablation:hyperparams] env=%s failed: %s", env_id, exc)
            report["errors"][env_id] = traceback.format_exc(limit=3)
    return report


def run_sil_ablation(
    env_ids: Sequence[str] = MUJOCO_ENVS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    cfg: Any = None,
    device: str = "cpu",
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    total_timesteps: Optional[int] = None,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Table 5 (optional): RICE vs. Self-Imitation Learning on four MuJoCo games.

    The RICE half of the comparison is produced by Experiment II/III drivers
    (``refining=ours`` / ``explanation=ours``); here we only run the SIL side so
    the two halves can be tabulated together.
    """
    logger = logger or get_logger("rice.ablation")
    report: Dict[str, Any] = {
        "ablation": "sil",
        "envs": list(env_ids),
        "seeds": list(seeds),
        "device": device,
        "results": {},
        "errors": {},
    }
    if _sil_baseline is None:
        report["errors"]["driver"] = "rice.baselines.sil unavailable"
        logger.warning("SIL baseline unavailable; skipping SIL ablation")
        return report

    for env_id in env_ids:
        rewards: List[float] = []
        try:
            for seed in seeds:
                set_seed(seed)
                logger.info("[ablation:sil] env=%s seed=%s", env_id, seed)
                out = _sil_baseline(
                    env=None,
                    env_id=env_id,
                    total_timesteps=total_timesteps,
                    seed=seed,
                    device=device,
                    progress=progress,
                    logger=logger,
                    evaluate=True,
                    eval_episodes=eval_episodes,
                    **kwargs,
                )
                if isinstance(out, dict):
                    value = out.get("final_reward")
                    if value is None:
                        value = _cfg_get(out, "eval", "mean_reward")
                    if value is not None:
                        rewards.append(float(value))
            mean, std = _mean_std(rewards)
            report["results"][env_id] = {
                "method": "sil",
                "final_reward": mean,
                "std": std,
                "rewards": rewards,
                "n_seeds": len(rewards),
            }
        except Exception as exc:  # pragma: no cover
            logger.error("[ablation:sil] env=%s failed: %s", env_id, exc)
            report["errors"][env_id] = traceback.format_exc(limit=3)
    return report


# ---------------------------------------------------------------------------
# report formatting
# ---------------------------------------------------------------------------
def _fidelity_like_table(block: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if not isinstance(block, dict):
        return lines
    results = block.get("results", block)
    if not isinstance(results, dict):
        return lines
    for env_id, entries in results.items():
        if not isinstance(entries, dict):
            continue
        for method, payload in entries.items():
            if not isinstance(payload, dict):
                continue
            value = payload.get("final_reward", payload.get("mean_reward"))
            if value is None and isinstance(payload.get("summary"), dict):
                value = payload["summary"].get("final_reward")
            std = payload.get("std")
            lines.append(
                f"  {env_id:<16} {str(method):<22} "
                f"{_format_value(value)} +- {_format_value(std, 3)}"
            )
    return lines


def format_report(report: Dict[str, Any], decimals: int = 3) -> str:
    """Render a human readable summary of an ablation report."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("RICE ablation report")
    lines.append("=" * 78)
    lines.append(f"ablation     : {report.get('ablation')}")
    lines.append(f"envs         : {report.get('envs')}")
    if report.get("methods"):
        lines.append(f"methods      : {report.get('methods')}")
    lines.append(f"seeds        : {report.get('seeds')}")

    results = report.get("results", {})
    lines.append("-" * 78)
    lines.append(f"{'env':<16} {'method':<22} {'mean +- std':<20}")
    lines.append("-" * 78)

    # exp3-style block: {env: {"results": {method: {...}}}} or {env: {method: {...}}}
    printed = False
    for env_id, entries in (results or {}).items():
        if not isinstance(entries, dict):
            continue
        inner = entries.get("results", entries) if "results" in entries else entries
        if not isinstance(inner, dict):
            continue
        for method, payload in inner.items():
            if not isinstance(payload, dict):
                continue
            value = payload.get("final_reward")
            if value is None and isinstance(payload.get("summary"), dict):
                value = payload["summary"].get("final_reward")
            std = payload.get("std")
            lines.append(
                f"{str(env_id):<16} {str(method):<22} "
                f"{_format_value(value, decimals)} +- {_format_value(std, decimals)}"
            )
            printed = True
    if not printed:
        lines.extend(_fidelity_like_table(report))
    if not printed and not lines[-1].startswith(" "):
        lines.append("  (no numeric results recorded)")

    trends = report.get("trends") or {}
    if trends:
        lines.append("-" * 78)
        lines.append("trend validation")
        for env_id, flags in trends.items():
            lines.append(f"  {env_id}: {flags}")

    errors = report.get("errors") or {}
    if errors:
        lines.append("-" * 78)
        lines.append("errors")
        for key, value in errors.items():
            first = str(value).strip().splitlines()
            lines.append(f"  {key}: {first[0] if first else value}")
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def run_ablation(
    ablation: str = "all",
    env_ids: Optional[Sequence[str]] = None,
    cfg: Any = None,
    device: str = "cpu",
    seeds: Sequence[int] = DEFAULT_SEEDS,
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    sweeps: Sequence[str] = ("p", "lambda", "alpha"),
    refine_timesteps: Optional[int] = None,
    mask_timesteps: Optional[int] = None,
    pretrain_timesteps: Optional[int] = None,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run one (or all) of the RICE ablations and persist the report."""
    logger = logger or get_logger("rice.ablation")
    ablation = str(ablation).lower()
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation '{ablation}'; expected one of {ABLATIONS}")

    if ablation == "explanation":
        chosen = list(env_ids) if env_ids else list(MUJOCO_ENVS)
        envs, methods = chosen, CORE_EXPLANATION_METHODS
    elif ablation == "refining":
        chosen = list(env_ids) if env_ids else list(DEFAULT_ENVS)
        envs, methods = chosen, REFINING_METHODS
    elif ablation == "hyperparams":
        chosen = list(env_ids) if env_ids else list(MUJOCO_ENVS)
        envs, methods = chosen, ("ours",)
    elif ablation == "sil":
        chosen = list(env_ids) if env_ids else list(MUJOCO_ENVS)
        envs, methods = chosen, ("sil",)
    else:  # all
        chosen = list(env_ids) if env_ids else list(MUJOCO_ENVS)
        envs, methods = chosen, CORE_EXPLANATION_METHODS

    root = _results_dir(cfg, out_dir)
    report: Dict[str, Any] = {
        "ablation": ablation,
        "envs": envs,
        "methods": list(methods),
        "seeds": list(seeds),
        "device": device,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backends": available_ablation_backends(),
        "sections": {},
        "errors": {},
    }

    common = dict(
        cfg=cfg,
        device=device,
        out_dir=root,
        logger=logger,
        progress=progress,
        seeds=tuple(seeds),
        refine_timesteps=refine_timesteps,
        mask_timesteps=mask_timesteps,
        pretrain_timesteps=pretrain_timesteps,
        eval_episodes=eval_episodes,
    )

    if ablation in ("explanation", "all"):
        report["sections"]["explanation"] = run_explanation_ablation(
            env_ids=envs, methods=CORE_EXPLANATION_METHODS, **common, **kwargs
        )
    if ablation in ("refining", "all"):
        report["sections"]["refining"] = run_refining_ablation(
            env_ids=envs, methods=REFINING_METHODS, **common, **kwargs
        )
    if ablation in ("hyperparams", "all"):
        report["sections"]["hyperparams"] = run_hyperparam_ablation(
            env_ids=envs, sweeps=tuple(sweeps), **common, **kwargs
        )
    if ablation in ("sil", "all"):
        report["sections"]["sil"] = run_sil_ablation(
            env_ids=envs, total_timesteps=refine_timesteps, **common, **kwargs
        )

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # persist --------------------------------------------------------------
    json_path = os.path.join(root, f"ablation_{ablation}.json")
    txt_path = os.path.join(root, f"ablation_{ablation}.txt")
    try:
        save_json(_jsonable(report), json_path)
        text = "\n\n".join(
            [format_report(sec) for sec in report["sections"].values()]
        ) or format_report(report)
        with open(txt_path, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        report["artifacts"] = {"json": json_path, "txt": txt_path}
        logger.info("ablation report written to %s", json_path)
    except Exception as exc:  # pragma: no cover
        logger.warning("could not persist ablation report: %s", exc)
        report["errors"]["persist"] = str(exc)

    return report


def run_ablation_multi(
    env_ids: Optional[Sequence[str]] = None,
    ablation: str = "all",
    cfg: Any = None,
    out_dir: Optional[str] = None,
    logger: Any = None,
    progress: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Alias kept for symmetry with the other experiment drivers."""
    return run_ablation(
        ablation=ablation,
        env_ids=env_ids,
        cfg=cfg,
        out_dir=out_dir,
        logger=logger,
        progress=progress,
        **kwargs,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ablation",
        description="Run RICE ablations (Experiments III/V, Table 5) for the ICML 2024 paper.",
    )
    parser.add_argument(
        "--ablation",
        default="all",
        choices=list(ABLATIONS),
        help="which ablation to run (default: all)",
    )
    parser.add_argument(
        "--envs",
        nargs="*",
        default=None,
        help=f"applications to run (default depends on the ablation: {list(MUJOCO_ENVS)})",
    )
    parser.add_argument("--config", default=None, help="config name (e.g. hopper) or YAML path")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--sweeps",
        nargs="*",
        default=["p", "lambda", "alpha"],
        help="Experiment V sweeps to run",
    )
    parser.add_argument("--refine-timesteps", type=int, default=None)
    parser.add_argument("--mask-timesteps", type=int, default=None)
    parser.add_argument("--pretrain-timesteps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logger = get_logger("rice.ablation", out_dir=args.out_dir)

    cfg = None
    if args.config:
        try:
            cfg = get_config(args.config)
        except Exception as exc:  # pragma: no cover
            logger.warning("could not load config '%s': %s", args.config, exc)

    report = run_ablation(
        ablation=args.ablation,
        env_ids=args.envs,
        cfg=cfg,
        device=args.device,
        seeds=tuple(args.seeds),
        out_dir=args.out_dir,
        logger=logger,
        progress=args.verbose,
        sweeps=tuple(args.sweeps),
        refine_timesteps=args.refine_timesteps,
        mask_timesteps=args.mask_timesteps,
        pretrain_timesteps=args.pretrain_timesteps,
        eval_episodes=args.eval_episodes,
    )

    for section in report.get("sections", {}).values():
        print(format_report(section))
    print(
        "\nbackends available: "
        f"{json.dumps(report.get('backends', {}), sort_keys=True)}"
    )
    # trend-only reproduction: never fail the CLI on a trend mismatch
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
