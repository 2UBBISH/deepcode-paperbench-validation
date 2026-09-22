#!/usr/bin/env python
"""Training entry point for Simformer (Simulation-based inference with probabilistic diffusion models).

Trains a single amortized transformer-parameterised score network on the *joint*
distribution ``p(theta, x)`` of one SBI task using **masked denoising score matching**
(paper Sec. 3.3, Eq. 1-2) with

* structure-aware attention masks ``M_E`` (Sec. 3.2) — ``dense`` / ``undirected`` /
  ``directed`` variants, where the directed variant is dynamically adapted to the
  per-batch conditioning state via the Webb et al. (2018) graph inversion (Appendix A1.1),
* per-sample condition masks ``M_C`` drawn uniformly from
  {joint, posterior, likelihood, rand(0.3), rand(0.7)} (Sec. 3.1, Appendix A2.1),
* a VESDE / VPSDE forward noising process (Sec. 2.3, Appendix A2.1).

Example
-------
    python scripts/train.py --task two_moons --budget 10000 --max-steps 20000 \
        --mask directed --outdir runs/two_moons_10k_directed

The script is deliberately framework-light (argparse + optional YAML config) so that it
can be driven directly, by ``scripts/run_experiments.py``, or from a Hydra-style config.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Import bootstrap: support both the official layout (repo_root/simformer/... plus
# sibling packages tasks/, eval/, ...) and a flat installed layout.
# --------------------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
_PKG_DIR = _THIS.parents[1]      # <repo>/simformer
_REPO_DIR = _THIS.parents[2]     # <repo>
for _p in (str(_REPO_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _import_first(candidates: Sequence[str], what: str = "module") -> Any:
    """Import the first importable module among ``candidates``."""
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on layout
            last_err = exc
    raise ImportError(f"could not import {what} (tried {list(candidates)}): {last_err}")


def _load_core_modules() -> Dict[str, Any]:
    mods: Dict[str, Any] = {}
    mods["training"] = _import_first(
        ("simformer.simformer.training", "simformer.training"), "training"
    )
    mods["diffusion"] = _import_first(
        ("simformer.simformer.diffusion", "simformer.diffusion"), "diffusion"
    )
    mods["transformer"] = _import_first(
        ("simformer.simformer.transformer", "simformer.transformer"), "transformer"
    )
    mods["attention_masks"] = _import_first(
        ("simformer.simformer.attention_masks", "simformer.attention_masks"),
        "attention_masks",
    )
    mods["tokenizer"] = _import_first(
        ("simformer.simformer.tokenizer", "simformer.tokenizer"), "tokenizer"
    )
    mods["graph_inversion"] = _import_first(
        ("simformer.simformer.graph_inversion", "simformer.graph_inversion"),
        "graph_inversion",
    )
    return mods


# --------------------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------------------

DEFAULT_TRAIN_CONFIG: Dict[str, Any] = {
    "batch_size": 1000,
    "lr": 3e-4,
    "weight_decay": 0.0,
    "gradient_clip": 1.0,
    "max_steps": 100_000,
    "max_epochs": None,
    "warmup_steps": 1000,
    "lr_schedule": "cosine",
    "val_fraction": 0.1,
    "val_every": 500,
    "early_stopping_patience": 20,
    "log_every": 100,
    "device": "cpu",
    "dtype": "float32",
    "normalize": False,
    "loss_space": "score",
}

DEFAULT_TASK_CONFIG: Dict[str, Any] = {
    "task": "two_moons",
    "budget": 10_000,
    "mask": "directed",
    "sde": "vesde",
    "token_dim": 50,
    "n_layers": None,
    "n_heads": 4,
    "attention_size": 10,
    "widening_factor": 3,
    "time_embed_dim": 128,
    "n_sampling_steps": 500,
    "seed": 0,
    "outdir": None,
    "name": None,
}

ALLOWED_MASKS = ("dense", "undirected", "directed", "none")


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Best-effort YAML / JSON config loader (no hard PyYAML dependency)."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    text = p.read_text()
    try:  # preferred
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        pass
    try:
        data = json.loads(text)
        return dict(data) if isinstance(data, dict) else {}
    except Exception:
        pass
    # Extremely small fallback: flat "key: value" pairs.
    cfg: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        try:
            cfg[key] = json.loads(value)
        except Exception:
            cfg[key] = value
    return cfg


def merged(args: argparse.Namespace, cfg: Dict[str, Any], key: str, default: Any) -> Any:
    """CLI value > YAML value > default."""
    value = getattr(args, key, None)
    if value is not None:
        return value
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    return default


# --------------------------------------------------------------------------------------
# Task / data helpers
# --------------------------------------------------------------------------------------


def to_joint(task: Any, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Concatenate parameters and data in the canonical ``[theta | x]`` order."""
    for fn in ("to_joint",):
        if hasattr(task, fn):
            try:
                return np.asarray(getattr(task, fn)(theta, x), dtype=np.float64)
            except Exception:
                pass
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)
    if theta.shape[0] != x.shape[0]:
        theta = np.broadcast_to(theta, (x.shape[0], theta.shape[1]))
    return np.concatenate([theta, x], axis=-1)


def make_joint_dataset(task: Any, n_simulations: int, seed: int, verbose: bool = False) -> np.ndarray:
    """Generate ``n_simulations`` joint samples from the task simulator."""
    rng = np.random.default_rng(int(seed))
    out: Any = None
    for attempt in (
        lambda: task.make_dataset(n_simulations, rng=rng, seed=seed, verbose=verbose),
        lambda: task.make_dataset(n_simulations, rng=rng, seed=seed),
        lambda: task.make_dataset(n_simulations, rng=rng),
        lambda: task.make_dataset(n_simulations, seed=seed),
        lambda: task.make_dataset(n_simulations),
    ):
        try:
            out = attempt()
            break
        except TypeError:
            continue
    if out is None:
        # Manual fall back to prior + simulator.
        theta = np.asarray(task.prior_sample(n_simulations, rng))
        x = np.asarray(task.simulate(theta, rng))
        joint = to_joint(task, theta, x)
    elif isinstance(out, (tuple, list)) and len(out) == 2:
        joint = to_joint(task, np.asarray(out[0]), np.asarray(out[1]))
    else:
        joint = np.asarray(out, dtype=np.float64)
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim == 1:
        joint = joint[None, :]
    return joint


def task_dims(task: Any, joint: np.ndarray) -> Tuple[int, int]:
    n_parameters = getattr(task, "n_parameters", None)
    n_data = getattr(task, "n_data", None)
    if n_parameters is None or n_data is None:
        single = np.asarray(task.simulate(np.asarray(task.prior_sample(1, np.random.default_rng(0))),
                                         np.random.default_rng(0)))
        n_data = int(np.size(single))
        n_parameters = int(np.size(task.prior_sample(1, np.random.default_rng(0))))
    n_parameters = int(n_parameters)
    n_data = int(n_data)
    total = n_parameters + n_data
    if joint.shape[-1] != total and joint.shape[-1] > n_parameters:
        n_data = int(joint.shape[-1] - n_parameters)
    return n_parameters, n_data


# --------------------------------------------------------------------------------------
# Attention masks (Sec. 3.2) including graph inversion for directed masks (Appendix A1.1)
# --------------------------------------------------------------------------------------


def _directed_mask_callable(
    base_mask: np.ndarray,
    n_variables: int,
    input_dim: int,
    tokenizer: Any = None,
    core: Optional[Dict[str, Any]] = None,
) -> Any:
    """Return ``M_C -> adapted attention mask`` using Webb et al. (2018) graph inversion.

    The trainer passes the *variable-level* condition mask used for the batch; we remain
    tolerant to value-level (expanded) masks by collapsing them first.  For tasks whose
    token count differs from the variable count (function-valued parameters) the graph
    inversion is performed on the statistical-variable graph and then expanded to tokens.
    """
    core = core or {}
    gi = core.get("graph_inversion")
    training = core.get("training")
    base = np.asarray(base_mask)
    n_tok = base.shape[0]
    variable_level = (n_tok == n_variables) or (n_tok == input_dim)

    def fn(condition_mask: Any) -> np.ndarray:
        mask = np.asarray(condition_mask)
        if mask.size == 0:
            return base
        mask = (mask > 0.5).astype(np.float64)
        if mask.ndim == 1:
            mask = mask[None, :]
        m = mask.reshape(mask.shape[0], -1)
        # Collapse value-level to variable-level when necessary.
        if m.shape[-1] != n_variables:
            if m.shape[-1] == input_dim:
                if training is not None and hasattr(training, "collapse_value_mask") and tokenizer is not None:
                    try:
                        m = np.asarray(training.collapse_value_mask(m, tokenizer))
                    except Exception:
                        widths = None
                        if training is not None and hasattr(training, "variable_expansion_widths"):
                            try:
                                widths = np.asarray(training.variable_expansion_widths(tokenizer))
                            except Exception:
                                widths = None
                        if widths is not None and int(widths.sum()) == m.shape[-1]:
                            idx = np.repeat(np.arange(len(widths)), widths)
                            m = np.stack([m[b][idx] for b in range(m.shape[0])], axis=0)
                        else:
                            m = m[:, :n_variables]
            else:
                m = m[:, :n_variables]
        if gi is not None and hasattr(gi, "make_condition_aware_mask"):
            adapted = np.asarray(
                gi.make_condition_aware_mask(base, m, directed=True, batched=True)
            )
        elif gi is not None and hasattr(gi, "graph_inversion_batched"):
            edges = np.asarray(gi.graph_inversion_batched(base, m))
            adapted = np.maximum(base[None, :, :], edges)
        else:  # last resort: static directed mask
            adapted = np.broadcast_to(base[None, :, :], (m.shape[0],) + base.shape).copy()
        if adapted.shape[0] == 1 and mask.shape[0] > 1:
            adapted = np.broadcast_to(adapted, (mask.shape[0],) + adapted.shape[1:]).copy()
        return adapted

    if not variable_level:  # function-valued token layouts: keep the static task mask
        def static_fn(condition_mask: Any) -> np.ndarray:  # pragma: no cover - simple
            return base

        return static_fn
    return fn


def build_task_attention_mask(
    task_name: str,
    variant: str,
    n_parameters: int,
    n_data: int,
    tokenizer: Any = None,
    input_dim: Optional[int] = None,
    core: Optional[Dict[str, Any]] = None,
    task: Any = None,
    verbose: bool = True,
) -> Any:
    """Build the attention mask (``M_E``) for one of the Fig. 4 mask variants."""
    variant = (variant or "none").lower()
    if variant in ("none", "null", "off"):
        return None
    core = core or {}
    am = core.get("attention_masks")
    if am is None:
        return None
    kwargs: Dict[str, Any] = {"n_theta": n_parameters, "n_x": n_data}
    n_times = getattr(task, "n_times", None)
    if n_times is not None:
        kwargs["n_times"] = int(n_times)
    mask: Any = None
    if hasattr(am, "mask_variants"):
        try:
            variants = am.mask_variants(task_name, **kwargs)
            mask = variants.get(variant)
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[train] mask_variants failed for {task_name}: {exc}")
            mask = None
    if mask is None and hasattr(am, "build_attention_mask"):
        try:
            mask = am.build_attention_mask(
                task_name, directed=(variant != "undirected"), **kwargs
            )
        except Exception as exc:
            print(f"[train] WARNING: could not build attention mask for '{task_name}': {exc}")
            return None
    if mask is None:
        return None
    mask = np.asarray(mask)
    n_tokens = getattr(tokenizer, "n_tokens", None)
    if n_tokens is not None and mask.shape[0] != int(n_tokens):
        if mask.shape[0] == n_parameters + n_data and variant == "directed":
            # keep going: trainer sanitises sizes; emit a hint
            if verbose:
                print(
                    f"[train] note: mask size {mask.shape[0]} != n_tokens {n_tokens}; "
                    "using static mask"
                )
        return mask if mask.shape[0] == mask.shape[1] else None
    if variant == "directed" and tokenizer is not None:
        n_variables = int(
            getattr(tokenizer, "n_variables", n_parameters + n_data) or (n_parameters + n_data)
        )
        if mask.shape[0] == n_variables:
            return _directed_mask_callable(
                mask,
                n_variables,
                int(input_dim or (n_parameters + n_data)),
                tokenizer=tokenizer,
                core=core,
            )
    return mask


# --------------------------------------------------------------------------------------
# Building the pipeline
# --------------------------------------------------------------------------------------


def build_task_instance(tasks_mod: Any, task_name: str, seed: int) -> Any:
    build = getattr(tasks_mod, "build_task", None)
    if build is None:
        build = getattr(tasks_mod, "get_task")
    for attempt in (
        lambda: build(task_name, seed=seed),
        lambda: build(task_name),
    ):
        try:
            return attempt()
        except TypeError:
            continue
    raise RuntimeError(f"could not instantiate task '{task_name}'")


def build_model(
    task: Any,
    task_name: str,
    token_dim: int,
    n_layers: Optional[int],
    n_heads: int,
    attention_size: int,
    widening_factor: int,
    time_embed_dim: int,
    sde: Any,
    core: Dict[str, Any],
) -> Any:
    transformer = core["transformer"]
    spec = None
    if hasattr(task, "spec"):
        try:
            spec = task.spec()
        except Exception:
            try:
                spec = task.spec(token_dim=token_dim)
            except Exception:
                spec = None
    if spec is None:
        spec = core["tokenizer"].build_benchmark_spec(
            int(getattr(task, "n_parameters", 2)), int(getattr(task, "n_data", 2))
        )
    build_score_network = getattr(transformer, "build_score_network", None)
    if build_score_network is not None:
        try:
            return build_score_network(
                task=task_name,
                spec=spec,
                token_dim=token_dim,
                n_layers=n_layers,
                n_heads=n_heads,
                attention_size=attention_size,
                widening_factor=widening_factor,
                time_embed_dim=time_embed_dim,
            )
        except Exception as exc:
            print(f"[train] build_score_network fallback ({exc})")
    tokenizer = None
    if hasattr(task, "build_tokenizer"):
        try:
            tokenizer = task.build_tokenizer(token_dim=token_dim)
        except Exception:
            tokenizer = None
    if tokenizer is None:
        tokenizer = core["tokenizer"].Tokenizer(spec=spec, token_dim=token_dim)
    net = transformer.build_transformer(
        task=task_name,
        token_dim=token_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        attention_size=attention_size,
        widening_factor=widening_factor,
        time_embed_dim=time_embed_dim,
    )
    if hasattr(transformer, "SimformerScoreNetwork"):
        return transformer.SimformerScoreNetwork(tokenizer, net)
    return net


def build_sde(sde_name: str, core: Dict[str, Any], n_steps: int) -> Any:
    diffusion = core["diffusion"]
    name = (sde_name or "vesde").lower()
    try:
        sde = diffusion.get_sde(name)
    except Exception:
        sde = diffusion.get_sde("vesde")
    try:
        if getattr(sde, "n_steps", None) != n_steps:
            sde.n_steps = int(n_steps)
    except Exception:
        pass
    return sde


# --------------------------------------------------------------------------------------
# Optional end-to-end sampling verification
# --------------------------------------------------------------------------------------


def verify_sampling(model: Any, task: Any, sde: Any, tokenizer: Any, joint: np.ndarray,
                    n_samples: int = 64, n_steps: int = 50, seed: int = 0) -> Dict[str, Any]:
    """Cheap smoke test that reverse-SDE conditional sampling runs end-to-end."""
    info: Dict[str, Any] = {"ran": False}
    try:
        sampling = _import_first(
            ("simformer.simformer.sampling", "simformer.sampling"), "sampling"
        )
    except Exception as exc:  # pragma: no cover
        info["error"] = f"sampling import failed: {exc}"
        return info
    n_parameters = int(getattr(task, "n_parameters", joint.shape[-1] // 2))
    try:
        sampler = sampling.ConditionalSampler(
            model=model, sde=sde, tokenizer=tokenizer, n_parameters=n_parameters,
            n_data=int(joint.shape[-1] - n_parameters), device="cpu",
        )
    except Exception as exc:
        info["error"] = f"ConditionalSampler failed: {exc}"
        return info
    x_obs = np.asarray(joint[0, n_parameters:], dtype=np.float64)
    for call in (
        lambda: sampler.posterior(x_obs, n_samples=n_samples, n_steps=n_steps, seed=seed),
        lambda: sampler.sample_from_condition_mask(
            np.concatenate([np.zeros(n_parameters), np.ones(joint.shape[-1] - n_parameters)]),
            np.asarray(joint[0]),
            n_samples=n_samples,
            n_steps=n_steps,
            seed=seed,
        ),
    ):
        try:
            samples = np.asarray(call())
            if samples.size:
                info.update(
                    ran=True,
                    n_samples=int(samples.shape[0]),
                    mean=np.asarray(samples).reshape(samples.shape[0], -1).mean(axis=0).tolist(),
                    std=np.asarray(samples).reshape(samples.shape[0], -1).std(axis=0).tolist(),
                )
                return info
        except Exception as exc:
            info["error"] = f"{type(exc).__name__}: {exc}"
    return info


# --------------------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train Simformer on a task with masked denoising score matching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None, help="optional YAML/JSON config file")
    p.add_argument("--task", type=str, default=None, help="task name (e.g. two_moons, slcp)")
    p.add_argument("--budget", type=int, default=None, help="number of simulations (1k/10k/100k)")
    p.add_argument("--mask", type=str, default=None, choices=list(ALLOWED_MASKS),
                   help="attention-mask variant (Fig. 4)")
    p.add_argument("--sde", type=str, default=None, choices=["vesde", "vpsde"],
                   help="forward SDE type (Appendix A2.1)")
    p.add_argument("--token-dim", type=int, default=None, dest="token_dim")
    p.add_argument("--n-layers", type=int, default=None, dest="n_layers")
    p.add_argument("--n-heads", type=int, default=None, dest="n_heads")
    p.add_argument("--attention-size", type=int, default=None, dest="attention_size")
    p.add_argument("--widening-factor", type=int, default=None, dest="widening_factor")
    p.add_argument("--time-embed-dim", type=int, default=None, dest="time_embed_dim")
    p.add_argument("--n-sampling-steps", type=int, default=None, dest="n_sampling_steps")
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-steps", type=int, default=None, dest="max_steps")
    p.add_argument("--warmup-steps", type=int, default=None, dest="warmup_steps")
    p.add_argument("--val-fraction", type=float, default=None, dest="val_fraction")
    p.add_argument("--val-every", type=int, default=None, dest="val_every")
    p.add_argument("--patience", type=int, default=None, dest="early_stopping_patience")
    p.add_argument("--log-every", type=int, default=None, dest="log_every")
    p.add_argument("--gradient-clip", type=float, default=None, dest="gradient_clip")
    p.add_argument("--normalize", action="store_true", default=None, dest="normalize")
    p.add_argument("--loss-space", type=str, default=None, choices=["score", "epsilon"],
                   dest="loss_space")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="torch device, e.g. cpu/cuda")
    p.add_argument("--outdir", type=str, default=None, help="directory for outputs")
    p.add_argument("--name", type=str, default=None, help="run name (defaults to task+mask+budget)")
    p.add_argument("--dry-run", action="store_true", help="run 2 training steps only")
    p.add_argument("--verify-sampling", action="store_true",
                   help="run a short reverse-SDE smoke test after training")
    p.add_argument("--no-save-data", action="store_true", help="do not write the dataset .npy")
    p.add_argument("--list-tasks", action="store_true", help="list available tasks and exit")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_yaml_config(args.config)

    tasks_mod = _import_first(("simformer.tasks", "tasks"), "tasks")

    if args.list_tasks:
        try:
            names = tasks_mod.available_tasks()
        except Exception:
            names = list(getattr(tasks_mod, "TASK_MODULES", {}) or {})
        print("available tasks:")
        for n in names:
            print(f"  - {n}")
        return 0

    # ---------------------------------------------------------------- resolve settings
    task_name = str(merged(args, cfg, "task", DEFAULT_TASK_CONFIG["task"]))
    budget = int(merged(args, cfg, "budget", DEFAULT_TASK_CONFIG["budget"]))
    mask_variant = str(merged(args, cfg, "mask", DEFAULT_TASK_CONFIG["mask"])).lower()
    sde_name = str(merged(args, cfg, "sde", DEFAULT_TASK_CONFIG["sde"])).lower()
    token_dim = int(merged(args, cfg, "token_dim", DEFAULT_TASK_CONFIG["token_dim"]))
    n_layers = merged(args, cfg, "n_layers", DEFAULT_TASK_CONFIG["n_layers"])
    n_heads = int(merged(args, cfg, "n_heads", DEFAULT_TASK_CONFIG["n_heads"]))
    attention_size = int(merged(args, cfg, "attention_size", DEFAULT_TASK_CONFIG["attention_size"]))
    widening_factor = int(merged(args, cfg, "widening_factor", DEFAULT_TASK_CONFIG["widening_factor"]))
    time_embed_dim = int(merged(args, cfg, "time_embed_dim", DEFAULT_TASK_CONFIG["time_embed_dim"]))
    n_sampling_steps = int(merged(args, cfg, "n_sampling_steps", DEFAULT_TASK_CONFIG["n_sampling_steps"]))
    seed = int(merged(args, cfg, "seed", DEFAULT_TASK_CONFIG["seed"]))

    train_overrides: Dict[str, Any] = {}
    for key, default in DEFAULT_TRAIN_CONFIG.items():
        train_overrides[key] = merged(args, cfg, key, default)
    if args.dry_run:
        train_overrides["max_steps"] = 2
        train_overrides["warmup_steps"] = 1
        train_overrides["val_every"] = 1
        train_overrides["log_every"] = 1
        train_overrides["early_stopping_patience"] = 2
        budget = min(budget, 200)
    device = str(train_overrides.get("device") or "cpu")

    outdir = args.outdir or cfg.get("outdir") or DEFAULT_TASK_CONFIG["outdir"]
    if outdir is None:
        outdir = os.path.join("runs", f"{task_name}_{mask_variant}_{budget}")
    run_name = args.name or cfg.get("name") or f"{task_name}_{mask_variant}_{budget}"
    outdir = Path(str(outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 78)
        print(f"Simformer training — task={task_name} budget={budget} mask={mask_variant} "
              f"sde={sde_name}")
        print(f"  outdir={outdir}  device={device}  seed={seed}")
        print("=" * 78)

    t_start = time.time()
    core = _load_core_modules()

    # ------------------------------------------------------------------------ task
    task = build_task_instance(tasks_mod, task_name, seed)
    joint = make_joint_dataset(task, budget, seed, verbose=not args.quiet)
    n_parameters, n_data = task_dims(task, joint)
    if not args.quiet:
        print(f"[train] dataset: {joint.shape}  (n_parameters={n_parameters}, n_data={n_data})")

    # ------------------------------------------------------------------- tokenizer
    tokenizer = None
    if hasattr(task, "build_tokenizer"):
        try:
            tokenizer = task.build_tokenizer(token_dim=token_dim)
        except Exception as exc:
            if not args.quiet:
                print(f"[train] task.build_tokenizer failed ({exc}); using default tokenizer")

    # --------------------------------------------------------------- attention mask
    mask: Any = build_task_attention_mask(
        task_name=task_name,
        variant=mask_variant,
        n_parameters=n_parameters,
        n_data=n_data,
        tokenizer=tokenizer,
        input_dim=joint.shape[-1],
        core=core,
        task=task,
        verbose=not args.quiet,
    )
    if not args.quiet:
        if mask is None:
            print("[train] attention mask: fully dense (M_E=None)")
        elif callable(mask):
            print("[train] attention mask: directed + graph inversion (callable of M_C)")
        else:
            m = np.asarray(mask)
            print(f"[train] attention mask: {mask_variant} shape={m.shape} "
                  f"edges={int(m.sum())}")

    # ----------------------------------------------------------------------- model
    model = build_model(
        task=task,
        task_name=task_name,
        token_dim=token_dim,
        n_layers=n_layers if n_layers is None else int(n_layers),
        n_heads=n_heads,
        attention_size=attention_size,
        widening_factor=widening_factor,
        time_embed_dim=time_embed_dim,
        sde=None,
        core=core,
    )
    if tokenizer is None:
        tokenizer = getattr(model, "tokenizer", None)
    n_params = None
    if hasattr(core["transformer"], "count_parameters"):
        try:
            n_params = int(core["transformer"].count_parameters(model))
        except Exception:
            n_params = None
    if not args.quiet:
        print(f"[train] model parameters: {n_params if n_params is not None else 'n/a'}")

    # ------------------------------------------------------------------------- sde
    sde = build_sde(sde_name, core, n_sampling_steps)

    # --------------------------------------------------------------------- trainer
    TrainingConfig = core["training"].TrainingConfig
    trainer_cfg = TrainingConfig(
        batch_size=int(train_overrides["batch_size"]),
        lr=float(train_overrides["lr"]),
        weight_decay=float(train_overrides["weight_decay"]),
        gradient_clip=float(train_overrides["gradient_clip"]),
        max_steps=int(train_overrides["max_steps"]),
        warmup_steps=int(train_overrides["warmup_steps"]),
        lr_schedule=str(train_overrides["lr_schedule"]),
        val_fraction=float(train_overrides["val_fraction"]),
        val_every=int(train_overrides["val_every"]),
        early_stopping_patience=int(train_overrides["early_stopping_patience"]),
        log_every=int(train_overrides["log_every"]),
        seed=seed,
        device=device,
        dtype=str(train_overrides["dtype"]),
        normalize=bool(train_overrides["normalize"]),
        loss_space=str(train_overrides["loss_space"]),
    )
    trainer = core["training"].SimformerTrainer(
        model=model,
        sde=sde,
        attention_mask=mask,
        tokenizer=tokenizer,
        config=trainer_cfg,
    )

    history = trainer.fit(
        x_joint=joint,
        val_fraction=trainer_cfg.val_fraction,
        verbose=not args.quiet,
    )

    # ------------------------------------------------------------------- artefacts
    if not args.no_save_data:
        try:
            np.save(outdir / "dataset.npy", joint.astype(np.float32))
        except Exception:
            pass
    ckpt_path = None
    try:
        ckpt_path = trainer.save(outdir / "model.pt")
    except Exception as exc:
        print(f"[train] WARNING: checkpoint save failed: {exc}")
    hist = history if isinstance(history, dict) else {}
    with open(outdir / "history.json", "w") as fh:
        json.dump(hist, fh, indent=2, default=str)

    meta = {
        "task": task_name,
        "budget": int(budget),
        "mask": mask_variant,
        "sde": sde_name,
        "token_dim": token_dim,
        "n_layers": int(n_layers) if n_layers is not None else None,
        "n_heads": n_heads,
        "attention_size": attention_size,
        "widening_factor": widening_factor,
        "time_embed_dim": time_embed_dim,
        "n_parameters": int(n_parameters),
        "n_data": int(n_data),
        "joint_dim": int(joint.shape[-1]),
        "n_parameters_model": n_params,
        "batch_size": int(trainer_cfg.batch_size),
        "lr": float(trainer_cfg.lr),
        "max_steps": int(trainer_cfg.max_steps),
        "seed": int(seed),
        "device": device,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "history": {k: (v if not isinstance(v, list) or len(v) < 4096 else v[-64:])
                    for k, v in hist.items()},
        "elapsed_sec": time.time() - t_start,
        "run_name": run_name,
    }
    with open(outdir / "train_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    # ------------------------------------------------- optional sampling smoke test
    sampling_info: Dict[str, Any] = {}
    if args.verify_sampling:
        sampling_info = verify_sampling(model, task, sde, tokenizer, joint, seed=seed)
        with open(outdir / "sampling_check.json", "w") as fh:
            json.dump(sampling_info, fh, indent=2, default=str)
        if not args.quiet:
            print(f"[train] sampling smoke test: {sampling_info.get('ran', False)} "
                  f"{sampling_info.get('error', '')}")

    if not args.quiet:
        best = hist.get("best_val_loss")
        print("-" * 78)
        print(f"[train] done in {meta['elapsed_sec']:.1f}s  best_val_loss={best}")
        print(f"[train] checkpoint: {ckpt_path}")
        print(f"[train] artefacts : {outdir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
