"""Command line entry point for Functional Reward Encodings (FRE).

Sub-commands
------------
``train``             Train FRE (encoder/decoder phase + strided RL phase) for one domain.
``train-all``         Train every domain (antmaze, exorl:walker/cheetah, kitchen) in sequence.
``eval``              Zero-shot evaluation from a checkpoint (K=32 reward samples -> z -> rollouts).
``reproduce-tables``  Aggregate saved evaluation results into Table 1 (main results) and
                      Table 4 (prior-family scaling study).
``reproduce``         Evaluate every trained domain, then build the tables.
``info``              Print the resolved configuration for a domain.

The actual algorithms live in :mod:`fre.run_fre` (Algorithm 1 strided schedule: Phase 1 trains the
FRE encoder/decoder only, Phase 2 freezes the encoder and trains the z-conditioned IQL agent) and
in :mod:`fre.evaluation.evaluate` (zero-shot harness, Section 5.2 / Appendix C).  This file only
composes them, resolves the per-domain configuration (``fre/configs/*.yaml`` + CLI overrides) and
formats/serialises the reported tables.  Default hyper-parameters mirror Appendix A Table 3.

Usage
-----
    python -m fre.main train --domain antmaze --eval-after
    python -m fre.main train --domain exorl:walker --prior-preset fre-all
    python -m fre.main eval  --domain antmaze --task-set all --checkpoint runs/antmaze/checkpoint.pt
    python -m fre.main reproduce-tables --results-dir runs --output tables/ --reference
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Paths / domain registry
# --------------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

#: Domain -> bundled config file (relative to this package directory).
DOMAIN_CONFIGS: Dict[str, str] = {
    "antmaze": "configs/antmaze.yaml",
    "exorl": "configs/exorl.yaml",
    "exorl:walker": "configs/exorl.yaml",
    "exorl:cheetah": "configs/exorl.yaml",
    "kitchen": "configs/kitchen.yaml",
}

#: Domains trained by ``train-all`` (ExORL is split per environment, as in the paper).
CANONICAL_DOMAINS: Tuple[str, ...] = ("antmaze", "exorl:walker", "exorl:cheetah", "kitchen")

#: In-house methods (FRE + baselines implemented in this repository).
FRE_METHOD = "fre"
BASELINE_METHODS: Tuple[str, ...] = ("gc-iql", "gc-bc", "opal")
#: External methods (facebookresearch/controllable_agent).
EXTERNAL_METHODS: Tuple[str, ...] = ("fb", "sf")

ALL_METHODS: Tuple[str, ...] = (FRE_METHOD,) + BASELINE_METHODS + EXTERNAL_METHODS

#: Modules implementing the baselines (dispatch targets for `eval --method <m>`).
BASELINE_MODULES: Dict[str, str] = {
    "gc-iql": "fre.baselines.gc_iql",
    "gc-bc": "fre.baselines.gc_bc",
    "opal": "fre.baselines.opal",
    "fb": "fre.baselines.fb_sf_runner",
    "sf": "fre.baselines.fb_sf_runner",
}

#: Task sets reported for each domain in Table 1 (Section 5.2, Appendix C).
TASK_SETS: Dict[str, Tuple[str, ...]] = {
    "antmaze": ("goal-reaching", "directional", "random-simplex", "path", "all"),
    "exorl:walker": ("goals", "velocity", "all"),
    "exorl:cheetah": ("goals", "velocity", "all"),
    "kitchen": ("all",),
}

#: Prior-family presets used for the scaling study (Table 4 / Figure 5).
SCALING_PRESETS: Tuple[str, ...] = (
    "fre-all",
    "fre-goals",
    "fre-lin",
    "fre-mlp",
    "fre-lin-mlp",
    "fre-goal-mlp",
    "fre-goal-lin",
)

#: Reference numbers reported in Table 1 (FRE row, normalised return 0-100, mean of 20 episodes
#: over 5 seeds).  Used only for the deviation column of ``--reference``.
TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    "antmaze": {
        "ant-goal-reaching": 48.8,
        "ant-directional": 55.2,
        "ant-random-simplex": 21.3,
        "ant-path-center": 64.4,
        "ant-path-loop": 67.2,
        "ant-path-edges": 60.0,
        "antmaze-all": 52.8,
    },
    "exorl:walker": {
        "exorl-walker-goals": 94.0,
        "exorl-walker-velocity": 34.0,
        "exorl-all": 51.5,
    },
    "exorl:cheetah": {
        "exorl-cheetah-goals": 58.0,
        "exorl-cheetah-velocity": 20.0,
        "exorl-all": 51.5,
    },
    "kitchen": {"kitchen": 66.0},
    "all": {"all": 57.0},
}


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed the python/numpy/torch RNGs (torch is optional)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy is a hard dependency, stay defensive
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional at CLI level
        pass


def _parse_scalar(text: str) -> Any:
    """Parse a YAML-ish scalar into a python object."""
    text = text.strip()
    if text == "" or text.lower() in ("null", "none", "~"):
        return None
    if text.lower() in ("true", "yes", "on"):
        return True
    if text.lower() in ("false", "no", "off"):
        return False
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        if text[0] == '"':
            try:
                return json.loads(text)
            except Exception:
                return text[1:-1]
        return text[1:-1]
    if text.startswith("[") or text.startswith("{"):
        try:
            return json.loads(text.replace("'", '"'))
        except Exception:
            return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _minimal_yaml_load(text: str) -> Dict[str, Any]:
    """Tiny indentation-based YAML subset parser (flat keys + nested mappings)."""
    root: Dict[str, Any] = {}
    stack: List[Tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.split("  #")[0].rstrip()
        if ":" not in line:
            continue
        key, _, value = line.strip().partition(":")
        key = key.strip().lstrip("-").strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if value.strip() == "":
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file (PyYAML preferred, tiny fallback parser otherwise)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r") as handle:
        text = handle.read()
    try:  # pragma: no cover - depends on environment
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return _minimal_yaml_load(text)


def deep_update(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge used for config composition."""
    out: Dict[str, Any] = dict(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def config_path_for(domain: str, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the config file for ``domain`` (``exorl:walker`` -> ``exorl`` config)."""
    if explicit:
        candidate = explicit if os.path.isabs(explicit) else os.path.join(HERE, explicit)
        return candidate if os.path.exists(candidate) else None
    rel = DOMAIN_CONFIGS.get(domain)
    if rel is None and ":" in domain:
        rel = DOMAIN_CONFIGS.get(domain.split(":")[0])
    if rel is None:
        return None
    candidate = os.path.join(HERE, rel)
    return candidate if os.path.exists(candidate) else None


def domain_base(domain: str) -> str:
    """``exorl:walker`` -> ``exorl``."""
    return domain.split(":")[0]


def run_dir_for(domain: str, run_dir: Optional[str] = None) -> str:
    return run_dir or os.path.join(PROJECT_ROOT, "runs", domain.replace(":", "_"))


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def _split_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [item.strip() for item in str(value).split(",") if item.strip()]


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def cli_train_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect the CLI hyper-parameter overrides that were actually provided."""
    overrides: Dict[str, Any] = {}
    pairs = {
        "encoder_steps": getattr(args, "encoder_steps", None),
        "policy_steps": getattr(args, "policy_steps", None),
        "batch_size": getattr(args, "batch_size", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "beta": getattr(args, "beta", None),
        "num_encoder_pairs": getattr(args, "num_encoder_pairs", None),
        "num_decoder_pairs": getattr(args, "num_decoder_pairs", None),
        "prior_preset": getattr(args, "prior_preset", None),
        "seed": getattr(args, "seed", None),
        "run_dir": getattr(args, "run_dir", None),
        "phase": getattr(args, "phase", None),
    }
    for key, value in pairs.items():
        if value is not None:
            overrides[key] = value
    return overrides


def build_train_config(
    domain: str,
    config_path: Optional[str] = None,
    overrides: Optional[Mapping[str, Any]] = None,
    section: Optional[str] = None,
) -> Any:
    """Compose a :class:`fre.run_fre.FRETrainConfig` from defaults + YAML + CLI overrides.

    The bundled ``configs/*.yaml`` files store the paper hyper-parameters (Appendix A Table 3:
    batch 512, K=32, K'=8, beta 0.01, Adam 1e-4, RL/decoder [512,512,512], 4 attention heads,
    1/3 per reward family) together with per-domain step counts (150k/850k for AntMaze,
    1M/1M for ExORL and Kitchen) and the evaluation task lists.
    """
    from fre.run_fre import FRETrainConfig

    data: Dict[str, Any] = {}
    resolved = config_path_for(domain, config_path)
    if resolved:
        data = deep_update(data, load_yaml(resolved))

    nested_keys = {"encoder", "decoder", "policy", "rl", "iql", "evaluation", "eval", "prior",
                   "domains", "presets"}

    def flatten(cfg: Mapping[str, Any]) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for key, value in cfg.items():
            if key in nested_keys and isinstance(value, Mapping):
                flat.update(flatten(value))
            else:
                flat[key] = value
        return flat

    merged = flatten({k: v for k, v in data.items() if k not in ("antmaze", "exorl", "kitchen",
                                                                 "domains", "walker", "cheetah")})

    # Per-domain sections: top-level `antmaze:`/`exorl:`/`kitchen:` plus `domains:` entries.
    per_domain: Dict[str, Any] = {}
    for key in ("antmaze", "exorl", "kitchen", "walker", "cheetah"):
        if isinstance(data.get(key), Mapping):
            per_domain[key] = flatten(data[key])  # type: ignore[arg-type]
    if isinstance(data.get("domains"), Mapping):
        for key, value in data["domains"].items():  # type: ignore[union-attr]
            if isinstance(value, Mapping):
                per_domain[str(key)] = flatten(value)

    for key in (domain, domain_base(domain), domain.replace(":", "_")):
        if isinstance(per_domain.get(key), Mapping):
            merged = deep_update(merged, per_domain[key])
    if ":" in domain:
        sub = domain.split(":", 1)[1]
        for key in (sub, f"{sub}_env"):
            if isinstance(per_domain.get(key), Mapping):
                merged = deep_update(merged, per_domain[key])

    if section and isinstance(data.get(section), Mapping):
        merged = deep_update(merged, flatten(data[section]))  # type: ignore[arg-type]

    allowed = set(FRETrainConfig.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    merged = {key: value for key, value in merged.items() if key in allowed}
    merged.setdefault("domain", domain)
    if len(domain.split(":")) == 2:
        merged.setdefault("env_name", domain)

    cfg = FRETrainConfig.from_dict(merged)
    if overrides:
        clean = {k: v for k, v in overrides.items() if k in allowed and v is not None}
        if clean:
            cfg = cfg.replace(**clean)
    return cfg


def cmd_train(args: argparse.Namespace) -> int:
    """Train FRE for a single domain using the Algorithm 1 strided schedule."""
    from fre.run_fre import FRERunner

    domain = args.domain
    seed = int(getattr(args, "seed", 0) or 0)
    set_seed(seed)

    cfg = build_train_config(
        domain,
        config_path=getattr(args, "config", None),
        overrides=cli_train_overrides(args),
        section=getattr(args, "config_section", None),
    )

    print(f"[fre] training domain={domain} encoder_steps={cfg.encoder_steps} "
          f"policy_steps={cfg.policy_steps} prior={cfg.prior_preset} seed={seed}")

    runner = FRERunner(config=cfg, device=getattr(args, "device", None))
    runner.setup()
    t0 = time.time()
    runner.train(progress=not getattr(args, "no_progress", False))
    print(f"[fre] training finished in {time.time() - t0:.1f}s")

    run_dir = getattr(args, "run_dir", None)
    if not run_dir and hasattr(cfg, "resolved_run_dir"):
        try:
            run_dir = cfg.resolved_run_dir()
        except Exception:  # pragma: no cover
            run_dir = None
    run_dir = ensure_dir(run_dir or run_dir_for(domain))

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    try:
        runner.save_checkpoint(ckpt_path)
        print(f"[fre] saved checkpoint to {ckpt_path}")
    except Exception as exc:  # pragma: no cover - checkpointing is best effort
        print(f"[fre] warning: could not save checkpoint ({exc})")

    history = getattr(runner, "history", None)
    if history is not None and hasattr(history, "save"):
        try:
            history.save(os.path.join(run_dir, "history.json"))
        except Exception:  # pragma: no cover
            pass

    if getattr(args, "eval_after", False):
        eval_args = argparse.Namespace(
            domain=domain,
            task_set=getattr(args, "task_set", "all"),
            checkpoint=ckpt_path,
            method=FRE_METHOD,
            seeds=getattr(args, "seeds", None),
            num_seeds=None,
            num_episodes=getattr(args, "num_episodes", None),
            num_encoding_samples=getattr(args, "num_encoding_samples", None),
            max_episode_steps=None,
            stochastic=False,
            no_dataset=False,
            device=getattr(args, "device", None),
            output=os.path.join(run_dir, f"{FRE_METHOD}__{domain}__{getattr(args, 'task_set', 'all')}.json"),
            no_progress=getattr(args, "no_progress", False),
        )
        return cmd_eval(eval_args)
    return 0


def cmd_train_all(args: argparse.Namespace) -> int:
    """Train every canonical domain in sequence (optionally evaluating afterwards)."""
    domains = _split_list(getattr(args, "domains", None)) or list(CANONICAL_DOMAINS)
    failures: List[str] = []
    for domain in domains:
        sub = argparse.Namespace(**vars(args))
        sub.domain = domain
        code = cmd_train(sub)
        if code != 0:
            failures.append(domain)
    if failures:
        print(f"[fre] domains that failed: {failures}")
        return 1
    return 0


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def _load_fre_model(checkpoint: str, device: Optional[str] = None) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load a trained FRE model + policy from a checkpoint (returns model, agent, meta)."""
    from fre.run_fre import FRERunner

    runner = FRERunner(device=device)
    meta = runner.load_checkpoint(checkpoint, load_agent=True)
    model = getattr(runner, "model", None)
    agent = getattr(runner, "agent", None)
    if agent is None:
        agent = getattr(runner, "policy", None)
    return model, agent, {"config": getattr(runner, "config", None), "meta": meta}


def _load_dataset_for(domain: str, cfg: Any = None) -> Any:
    """Load the offline dataset for a domain (used to sample encoding states)."""
    from fre.data import load_dataset

    base = domain_base(domain)
    if base == "antmaze":
        return load_dataset("antmaze")
    if base == "kitchen":
        return load_dataset("kitchen")
    if base == "exorl":
        env_name = domain.split(":", 1)[1] if ":" in domain else None
        if env_name is None and cfg is not None:
            env_name = getattr(cfg, "env_name", None) or getattr(cfg, "exorl_env", None)
        env_name = env_name or "walker-run"
        try:
            return load_dataset(f"exorl:{env_name}")
        except Exception:
            return load_dataset(f"exorl:{env_name.split('-')[0]}-run")
    return load_dataset(base)


def cmd_eval(args: argparse.Namespace) -> int:
    """Zero-shot evaluation: encode 32 (s, eta(s)) samples -> z -> rollouts -> normalised return."""
    from fre.evaluation.evaluate import EvalConfig, evaluate_fre, save_results
    from fre.evaluation.metrics import NUM_EVAL_EPISODES, NUM_SEEDS

    domain = args.domain
    method = getattr(args, "method", FRE_METHOD) or FRE_METHOD

    if method != FRE_METHOD:
        return cmd_baseline(args)

    checkpoint = getattr(args, "checkpoint", None)
    if not checkpoint:
        default_ckpt = os.path.join(run_dir_for(domain), "checkpoint.pt")
        if os.path.exists(default_ckpt):
            checkpoint = default_ckpt
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"[fre] error: no checkpoint found for {domain} (looked at {checkpoint})")
        return 2

    model, agent, meta = _load_fre_model(checkpoint, getattr(args, "device", None))
    if model is None:
        print("[fre] error: checkpoint does not contain an FRE encoder/decoder")
        return 2

    if getattr(args, "seeds", None):
        seeds = [int(s) for s in (_split_list(args.seeds) or [])]
    else:
        seeds = list(range(int(getattr(args, "num_seeds", None) or NUM_SEEDS)))

    cfg = EvalConfig(
        domain=domain,
        task_set=getattr(args, "task_set", "all") or "all",
        num_episodes=int(getattr(args, "num_episodes", None) or NUM_EVAL_EPISODES),
        seeds=seeds,
        num_encoding_samples=int(getattr(args, "num_encoding_samples", None) or 32),
        deterministic=not getattr(args, "stochastic", False),
    )
    if getattr(args, "max_episode_steps", None):
        try:
            cfg.max_episode_steps = int(args.max_episode_steps)
        except Exception:  # pragma: no cover
            pass

    dataset = None
    if not getattr(args, "no_dataset", False):
        try:
            dataset = _load_dataset_for(domain, meta.get("config"))
        except Exception as exc:  # pragma: no cover - dataset availability varies
            print(f"[fre] warning: could not load dataset for {domain} ({exc}); "
                  "falling back to rollout-collected encoding states")

    summary = evaluate_fre(
        model,
        agent,
        domain=domain,
        task_set=cfg.task_set,
        dataset=dataset,
        cfg=cfg,
        seeds=seeds,
        device=getattr(args, "device", None),
    )

    output = getattr(args, "output", None)
    if output:
        ensure_dir(os.path.dirname(output))
        payload = {"method": method, "domain": domain, "task_set": cfg.task_set, "summary": summary}
        try:
            save_results(output, summary)
            with open(output, "w") as handle:
                json.dump(payload, handle, indent=2, default=str)
        except Exception as exc:  # pragma: no cover
            print(f"[fre] warning: could not write results to {output} ({exc})")
        else:
            print(f"[fre] wrote results to {output}")

    print(json.dumps(summary, indent=2, default=str))
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Dispatch evaluation/training of a baseline method to its own module CLI."""
    method = getattr(args, "method", "")
    module_name = BASELINE_MODULES.get(method)
    if module_name is None:
        print(f"[fre] error: unknown method '{method}'. Known: {', '.join(ALL_METHODS)}")
        return 2
    try:
        module = __import__(module_name, fromlist=["main"])
    except Exception as exc:  # pragma: no cover - optional module availability
        print(f"[fre] error: could not import {module_name} ({exc})")
        return 2
    if not hasattr(module, "main"):
        print(f"[fre] error: {module_name} exposes no main() entry point")
        return 2
    forwarded: List[str] = ["--domain", str(getattr(args, "domain", ""))]
    for flag, attr in (
        ("--task-set", "task_set"),
        ("--checkpoint", "checkpoint"),
        ("--output", "output"),
        ("--seeds", "seeds"),
        ("--num-episodes", "num_episodes"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            forwarded += [flag, str(value)]
    forwarded += ["--method", method]
    return int(module.main(forwarded))  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Table reproduction
# --------------------------------------------------------------------------------------


def discover_results(paths: Sequence[str], verbose: bool = False) -> List[Dict[str, Any]]:
    """Collect evaluation JSON files from a list of directories/files."""
    files: List[str] = []
    for path in paths:
        if not path:
            continue
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True)))
        elif os.path.exists(path):
            files.append(path)
    results: List[Dict[str, Any]] = []
    for path in files:
        base = os.path.basename(path)
        if "history" in base and "result" not in base:
            continue
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
        except Exception:
            continue
        if not isinstance(data, Mapping):
            continue
        if not any(k in data for k in ("summary", "tasks", "mean", "normalized_mean", "score")):
            continue
        entry = _normalise_result_entry(data, path)
        if entry is not None:
            results.append(entry)
            if verbose:
                print(f"[fre] loaded results: {entry['method']} / {entry['domain']} / "
                      f"{entry['task_set']} -> {entry['score']:.1f}")
    return results


def _normalise_result_entry(data: Mapping[str, Any], path: str) -> Optional[Dict[str, Any]]:
    summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else data
    if not isinstance(summary, Mapping):
        summary = {}

    stem = os.path.splitext(os.path.basename(path))[0]
    method = data.get("method") or data.get("preset") or data.get("agent")
    task_set = data.get("task_set") or data.get("taskSet")
    domain = data.get("domain")

    # Filenames like "<method>__<domain>__<task_set>.json" (default convention in run_all.sh).
    parts = [p for p in stem.split("__") if p]
    if method is None and parts:
        method = parts[0]
    if domain is None and len(parts) >= 2:
        domain = parts[1].replace("_", ":") if parts[1].startswith("exorl_") else parts[1]
    if task_set is None and len(parts) >= 3:
        task_set = parts[2]

    method = str(method or FRE_METHOD).replace("_", "-")
    domain = str(domain or "antmaze")
    task_set = str(task_set or "all")

    score: Optional[float] = None
    for key in ("normalized_mean", "mean", "score", "normalized"):
        value = summary.get(key)
        if value is not None:
            try:
                score = float(value)
                break
            except (TypeError, ValueError):
                continue
    if score is None and isinstance(summary.get("tasks"), Mapping):
        values: List[float] = []
        for task_value in summary["tasks"].values():
            if isinstance(task_value, Mapping):
                for key in ("normalized_mean", "mean", "score"):
                    if task_value.get(key) is not None:
                        try:
                            values.append(float(task_value[key]))
                            break
                        except (TypeError, ValueError):
                            continue
            elif isinstance(task_value, (int, float)):
                values.append(float(task_value))
        if values:
            score = sum(values) / len(values)
    if score is None and isinstance(summary.get("task_sets"), Mapping):
        values = [float(v) for v in summary["task_sets"].values() if isinstance(v, (int, float))]
        if values:
            score = sum(values) / len(values)
    if score is None:
        return None

    return {
        "method": method,
        "domain": domain,
        "domain_base": domain_base(domain),
        "task_set": task_set,
        "score": score,
        "summary": dict(summary),
        "path": path,
    }


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group entries into ``{method: {...task_set/domain/overall means...}}``."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in results:
        method = str(entry["method"])
        bucket = grouped.setdefault(method, {"task_sets": {}, "domains": {}, "scores": []})
        bucket["task_sets"].setdefault(str(entry["task_set"]), []).append(float(entry["score"]))
        bucket["domains"].setdefault(str(entry["domain"]), []).append(float(entry["score"]))
        bucket["scores"].append(float(entry["score"]))
    for bucket in grouped.values():
        bucket["task_set_means"] = {
            key: sum(values) / len(values) for key, values in bucket["task_sets"].items()
        }
        bucket["domain_means"] = {
            key: sum(values) / len(values) for key, values in bucket["domains"].items()
        }
        bucket["overall"] = sum(bucket["scores"]) / len(bucket["scores"]) if bucket["scores"] else 0.0
    return grouped


def build_table1(results: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Method x task-set table of normalised returns (Section 5.2, Table 1)."""
    grouped = aggregate_results(results)
    table: Dict[str, Dict[str, float]] = {}
    for method, bucket in sorted(grouped.items()):
        row: Dict[str, float] = dict(bucket["task_set_means"])
        row["overall"] = float(bucket["overall"])
        table[method] = row
    return table


def build_table4(results: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Prior-family scaling study: relative-normalised scores (best agent per task set = 1.0)."""
    from fre.evaluation.metrics import relative_normalize

    grouped = aggregate_results(results)
    presets = {name: bucket for name, bucket in grouped.items() if name.startswith("fre")}
    if not presets:
        presets = grouped
    per_task_set: Dict[str, Dict[str, float]] = {}
    for name, bucket in presets.items():
        for task_set, value in bucket["task_set_means"].items():
            per_task_set.setdefault(task_set, {})[name] = float(value)
    table: Dict[str, Dict[str, float]] = {}
    for task_set, scores in sorted(per_task_set.items()):
        for name, value in relative_normalize(scores).items():
            table.setdefault(name, {})[task_set] = value
    return table


def format_table_plain(table: Mapping[str, Mapping[str, Any]], title: Optional[str] = None,
                       digits: int = 1) -> str:
    """Render a ``{row: {column: value}}`` mapping as an aligned text table."""
    columns: List[str] = []
    for row in table.values():
        for key in row:
            if key not in columns:
                columns.append(key)
    width = max([len("method")] + [len(str(r)) for r in table])
    lines = [title] if title else []
    lines.append("method".ljust(width) + "".join(col.rjust(max(12, len(col) + 2)) for col in columns))
    lines.append("-" * len(lines[-1]))
    for name in sorted(table):
        cells = []
        for col in columns:
            value = table[name].get(col)
            pad = max(12, len(col) + 2)
            if value is None:
                cells.append("-".rjust(pad))
            elif isinstance(value, (int, float)):
                cells.append(f"{float(value):.{digits}f}".rjust(pad))
            else:
                cells.append(str(value).rjust(pad))
        lines.append(str(name).ljust(width) + "".join(cells))
    return "\n".join(lines)


def _lookup_reference(task: str) -> Optional[float]:
    for values in TABLE1_REFERENCE.values():
        if task in values:
            return values[task]
    for values in TABLE1_REFERENCE.values():
        for key, value in values.items():
            if key in task or task in key:
                return value
    return None


def format_table1_reference(table: Mapping[str, Mapping[str, Any]]) -> str:
    """Table 1 rows annotated with the paper's reported FRE numbers and the deviation."""
    header = "task".ljust(28) + "reproduced".rjust(12) + "paper".rjust(10) + "delta".rjust(10)
    lines = [header, "-" * len(header)]
    for row_name in sorted(table):
        for task, value in sorted(table[row_name].items()):
            if not isinstance(value, (int, float)):
                continue
            reference = _lookup_reference(task)
            ref_text = f"{reference:.1f}" if reference is not None else "-"
            delta = f"{float(value) - reference:+.1f}" if reference is not None else "-"
            lines.append(task.ljust(28) + f"{float(value):.1f}".rjust(12)
                         + ref_text.rjust(10) + delta.rjust(10))
    return "\n".join(lines)


def cmd_reproduce_tables(args: argparse.Namespace) -> int:
    """Aggregate saved results into Table 1 and the Table 4 scaling study."""
    paths = _split_list(getattr(args, "results_dir", None)) or [os.path.join(PROJECT_ROOT, "runs")]
    results = discover_results(paths, verbose=getattr(args, "verbose", False))
    if not results:
        print(f"[fre] no evaluation results found under: {', '.join(paths)}")
        print("[fre] run `python -m fre.main train --domain <d> --eval-after` first, or pass "
              "--results-dir pointing at a folder of evaluation JSON files.")
        return 1

    tables = {t.strip().lower() for t in (_split_list(getattr(args, "tables", None)) or ["1"])}
    output_dir = getattr(args, "output", None)
    if output_dir:
        ensure_dir(output_dir)

    if "1" in tables or "table1" in tables:
        table1 = build_table1(results)
        text = format_table_plain(table1, title="Table 1: normalised return (0-100)")
        print(text)
        if getattr(args, "reference", False):
            print()
            print("FRE reproduced vs. the paper's reported numbers:")
            print(format_table1_reference({FRE_METHOD: table1.get(FRE_METHOD, {})}))
        if output_dir:
            with open(os.path.join(output_dir, "table1.json"), "w") as handle:
                json.dump(table1, handle, indent=2, default=str)
            with open(os.path.join(output_dir, "table1.txt"), "w") as handle:
                handle.write(text + "\n")

    if "4" in tables or "table4" in tables or "scaling" in tables:
        table4 = build_table4(results)
        text = format_table_plain(
            table4, title="Table 4: prior-family scaling (best agent per task set = 1.0)", digits=2
        )
        print(text)
        if output_dir:
            with open(os.path.join(output_dir, "table4.json"), "w") as handle:
                json.dump(table4, handle, indent=2, default=str)
            with open(os.path.join(output_dir, "table4.txt"), "w") as handle:
                handle.write(text + "\n")
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Evaluate FRE for all domains from checkpoints, then build the tables."""
    checkpoints: Dict[str, str] = {}
    for item in _split_list(getattr(args, "checkpoints", None)) or []:
        if "=" in item:
            domain, path = item.split("=", 1)
            checkpoints[domain.strip()] = path.strip()
    domains = _split_list(getattr(args, "domains", None)) or list(CANONICAL_DOMAINS)
    task_sets = _split_list(getattr(args, "task_sets", None)) or ["all"]
    produced: List[str] = []
    for domain in domains:
        ckpt = checkpoints.get(domain) or os.path.join(run_dir_for(domain), "checkpoint.pt")
        if not os.path.exists(ckpt):
            print(f"[fre] skipping {domain}: no checkpoint at {ckpt}")
            continue
        for task_set in task_sets:
            output = os.path.join(
                getattr(args, "results_dir", None) or run_dir_for(domain),
                f"{FRE_METHOD}__{domain}__{task_set}.json",
            )
            eval_args = argparse.Namespace(
                domain=domain,
                task_set=task_set,
                checkpoint=ckpt,
                method=FRE_METHOD,
                seeds=getattr(args, "seeds", None),
                num_seeds=None,
                num_episodes=getattr(args, "num_episodes", None),
                num_encoding_samples=getattr(args, "num_encoding_samples", None),
                max_episode_steps=None,
                stochastic=False,
                no_dataset=False,
                device=getattr(args, "device", None),
                output=output,
                no_progress=getattr(args, "no_progress", False),
            )
            if cmd_eval(eval_args) == 0:
                produced.append(output)
            else:
                print(f"[fre] evaluation failed for {domain}/{task_set}")
    if not produced:
        print("[fre] nothing evaluated; aborting table generation")
        return 1
    args.results_dir = ",".join(sorted({os.path.dirname(p) for p in produced}))
    return cmd_reproduce_tables(args)


def cmd_info(args: argparse.Namespace) -> int:
    """Print the resolved configuration for a domain (sanity check before long runs)."""
    cfg = build_train_config(
        args.domain,
        config_path=getattr(args, "config", None),
        overrides=cli_train_overrides(args),
    )
    payload = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
    print(json.dumps({k: v for k, v in sorted(payload.items())}, indent=2, default=str))
    print(f"# config file: {config_path_for(args.domain, getattr(args, 'config', None))}")
    print(f"# run dir:     {run_dir_for(args.domain, getattr(args, 'run_dir', None))}")
    print(f"# task sets:   {TASK_SETS.get(args.domain, TASK_SETS.get(domain_base(args.domain), ()))}")
    print(f"# scaling:     {', '.join(SCALING_PRESETS)}")
    return 0


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def _add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--domain", type=str, default="antmaze",
                        help="domain to train/evaluate (antmaze, exorl:walker, exorl:cheetah, kitchen)")
    parser.add_argument("--config", type=str, default=None, help="explicit YAML config file")
    parser.add_argument("--config-section", type=str, default=None,
                        help="only apply this top-level YAML section (e.g. fre-hint)")
    parser.add_argument("--encoder-steps", type=int, default=None,
                        help="Phase 1 steps (paper: 150k AntMaze / 1M ExORL+Kitchen)")
    parser.add_argument("--policy-steps", type=int, default=None,
                        help="Phase 2 steps (paper: 850k AntMaze / 1M ExORL+Kitchen)")
    parser.add_argument("--batch-size", type=int, default=None, help="batch size (paper: 512)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Adam lr (paper: 1e-4)")
    parser.add_argument("--beta", type=float, default=None, help="KL weight (paper: 0.01)")
    parser.add_argument("--num-encoder-pairs", type=int, default=None, help="K (paper: 32)")
    parser.add_argument("--num-decoder-pairs", type=int, default=None, help="K' (paper: 8)")
    parser.add_argument("--prior-preset", type=str, default=None,
                        help="reward prior preset (fre-all, fre-goals, fre-hint, ...)")
    parser.add_argument("--phase", type=str, default=None, choices=["encoder", "policy", "both"],
                        help="which part of the strided schedule to run")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--run-dir", type=str, default=None, help="checkpoint/log directory")
    parser.add_argument("--device", type=str, default=None, help="torch device (cuda/cpu)")
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fre",
        description="Functional Reward Encodings: train / evaluate / reproduce the paper's tables.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train -------------------------------------------------------------------------
    p_train = sub.add_parser("train", help="train FRE for one domain (Algorithm 1)")
    _add_common_train_args(p_train)
    p_train.add_argument("--eval-after", action="store_true", help="run zero-shot eval after training")
    p_train.add_argument("--task-set", type=str, default="all", help="task set for --eval-after")
    p_train.add_argument("--seeds", type=str, default=None, help="comma-separated eval seeds")
    p_train.add_argument("--num-episodes", type=int, default=None, help="eval episodes per task")
    p_train.add_argument("--num-encoding-samples", type=int, default=None,
                         help="K reward samples at eval (paper: 32)")
    p_train.set_defaults(func=cmd_train)

    # train-all ---------------------------------------------------------------------
    p_all = sub.add_parser("train-all", help="train all canonical domains sequentially")
    _add_common_train_args(p_all)
    p_all.add_argument("--domains", type=str, default=None, help="comma-separated domain list")
    p_all.add_argument("--eval-after", action="store_true", help="run zero-shot eval after training")
    p_all.add_argument("--task-set", type=str, default="all")
    p_all.add_argument("--seeds", type=str, default=None)
    p_all.add_argument("--num-episodes", type=int, default=None)
    p_all.add_argument("--num-encoding-samples", type=int, default=None)
    p_all.set_defaults(func=cmd_train_all)

    # eval --------------------------------------------------------------------------
    p_eval = sub.add_parser("eval", help="zero-shot evaluation (FRE or a baseline)")
    p_eval.add_argument("--domain", type=str, default="antmaze")
    p_eval.add_argument("--task-set", type=str, default="all",
                        help="all / goal-reaching / directional / random-simplex / path / goals / velocity")
    p_eval.add_argument("--method", type=str, default=FRE_METHOD, choices=list(ALL_METHODS))
    p_eval.add_argument("--checkpoint", type=str, default=None, help="trained FRE checkpoint")
    p_eval.add_argument("--output", type=str, default=None, help="results JSON path")
    p_eval.add_argument("--seeds", type=str, default=None, help="comma-separated seeds (default 0..4)")
    p_eval.add_argument("--num-seeds", type=int, default=None, help="number of seeds")
    p_eval.add_argument("--num-episodes", type=int, default=None, help="episodes per task (default 20)")
    p_eval.add_argument("--num-encoding-samples", type=int, default=None,
                        help="encoder samples (default 32)")
    p_eval.add_argument("--max-episode-steps", type=int, default=None)
    p_eval.add_argument("--stochastic", action="store_true", help="sample actions instead of the mode")
    p_eval.add_argument("--no-dataset", action="store_true",
                        help="collect encoding states by rollout instead of the offline dataset")
    p_eval.add_argument("--device", type=str, default=None)
    p_eval.add_argument("--no-progress", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    # reproduce-tables --------------------------------------------------------------
    p_tab = sub.add_parser("reproduce-tables", help="aggregate results into Table 1 / Table 4")
    p_tab.add_argument("--results-dir", type=str, default=None,
                       help="comma-separated directories containing evaluation JSON files")
    p_tab.add_argument("--tables", type=str, default="1,4", help="which tables to build (1,4)")
    p_tab.add_argument("--output", type=str, default=None, help="directory for table1.json/table1.txt")
    p_tab.add_argument("--reference", action="store_true",
                       help="also print the paper's reported FRE numbers next to the reproduced ones")
    p_tab.add_argument("--verbose", action="store_true")
    p_tab.set_defaults(func=cmd_reproduce_tables)

    # reproduce ---------------------------------------------------------------------
    p_rep = sub.add_parser("reproduce", help="evaluate all trained domains then build the tables")
    p_rep.add_argument("--domains", type=str, default=None, help="comma-separated domain list")
    p_rep.add_argument("--checkpoints", type=str, default=None,
                       help="comma-separated domain=path checkpoint overrides")
    p_rep.add_argument("--task-sets", type=str, default=None, help="comma-separated task sets")
    p_rep.add_argument("--results-dir", type=str, default=None)
    p_rep.add_argument("--tables", type=str, default="1,4")
    p_rep.add_argument("--output", type=str, default=None)
    p_rep.add_argument("--reference", action="store_true")
    p_rep.add_argument("--seeds", type=str, default=None)
    p_rep.add_argument("--num-episodes", type=int, default=None)
    p_rep.add_argument("--num-encoding-samples", type=int, default=None)
    p_rep.add_argument("--device", type=str, default=None)
    p_rep.add_argument("--no-progress", action="store_true")
    p_rep.add_argument("--verbose", action="store_true")
    p_rep.set_defaults(func=cmd_reproduce)

    # info --------------------------------------------------------------------------
    p_info = sub.add_parser("info", help="print the resolved configuration for a domain")
    _add_common_train_args(p_info)
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Allow `python fre/main.py ...` (script mode) in addition to `python -m fre.main`.
    if __package__ in (None, "") and PROJECT_ROOT not in sys.path:  # pragma: no cover
        sys.path.insert(0, PROJECT_ROOT)

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[fre] interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
