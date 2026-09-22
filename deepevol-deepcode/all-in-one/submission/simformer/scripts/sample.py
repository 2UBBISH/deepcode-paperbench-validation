"""Sampling entry point for Simformer.

Loads a trained :class:`SimformerTrainer` checkpoint and draws samples from
arbitrary conditionals of the joint model ``p(theta, x)`` using the reverse SDE
(Simformer Sec. 3.3) or guided reverse SDE (Sec. 3.4 / Algorithm 1).

Modes
-----
``joint``       unconditional joint samples ``p(theta, x)``
``posterior``   ``p(theta | x_obs)``
``likelihood``  ``p(x | theta)``
``conditional`` arbitrary conditional given a user supplied condition mask
``arbitrary``   ``n_targets`` random conditional targets (Sec. 4.1 protocol)
``guided``      posterior (or joint) sampling with interval constraints

Examples
--------
    python -m simformer.scripts.sample --task two_moons --outdir runs/two_moons \
        --mode posterior --n-samples 1000 --n-steps 500

    python -m simformer.scripts.sample --task slcp --outdir runs/slcp \
        --mode guided --constraint-index 0 --constraint-lower -0.3 --constraint-upper 0.3

    python -m simformer.scripts.sample --task gaussian_linear --mode arbitrary --c2st
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Import bootstrap (mirrors scripts/train.py)
# --------------------------------------------------------------------------------------
_REPO_DIR = Path(__file__).resolve().parents[2]
_PKG_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:  # pragma: no cover - script-context import convenience
    from scripts.train import (  # type: ignore
        DEFAULT_TASK_CONFIG,
        DEFAULT_TRAIN_CONFIG,
        _import_first,
        _load_core_modules,
        build_model,
        build_sde,
        build_task_instance,
        load_yaml_config,
        make_joint_dataset,
        merged,
        task_dims,
        to_joint,
    )
except Exception:  # pragma: no cover
    try:
        from .train import (  # type: ignore
            DEFAULT_TASK_CONFIG,
            DEFAULT_TRAIN_CONFIG,
            _import_first,
            _load_core_modules,
            build_model,
            build_sde,
            build_task_instance,
            load_yaml_config,
            make_joint_dataset,
            merged,
            task_dims,
            to_joint,
        )
    except Exception:  # pragma: no cover
        DEFAULT_TASK_CONFIG = {}
        DEFAULT_TRAIN_CONFIG = {}
        _import_first = _load_core_modules = None  # type: ignore
        build_model = build_sde = build_task_instance = None  # type: ignore
        load_yaml_config = make_joint_dataset = merged = task_dims = to_joint = None  # type: ignore


DEFAULT_SAMPLE_CONFIG: Dict[str, Any] = {
    "mode": "posterior",
    "n_samples": 1000,
    "n_steps": 500,
    "batch_size": 256,
    "seed": 0,
    "checkpoint": "model.pt",
    "outdir": "runs/sample",
    "sde": "vesde",
    "task": "two_moons",
    "condition_mask": None,
    "condition_values": None,
    "n_targets": 100,
    "save_samples": True,
    "c2st": False,
    "n_reference": 1000,
    "self_recurrence": 0,
    "guidance_scale": 1.0,
}

ALLOWED_MODES = ("joint", "posterior", "likelihood", "conditional", "arbitrary", "guided")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _core() -> Dict[str, Any]:
    """Import the simformer core modules lazily."""
    if _load_core_modules is not None:
        try:
            return _load_core_modules()
        except Exception:
            pass
    mods: Dict[str, Any] = {}
    for key, candidates in {
        "training": ("simformer.training", "simformer.simformer.training"),
        "diffusion": ("simformer.diffusion", "simformer.simformer.diffusion"),
        "transformer": ("simformer.transformer", "simformer.simformer.transformer"),
        "attention_masks": ("simformer.attention_masks", "simformer.simformer.attention_masks"),
        "tokenizer": ("simformer.tokenizer", "simformer.simformer.tokenizer"),
        "graph_inversion": ("simformer.graph_inversion", "simformer.simformer.graph_inversion"),
        "sampling": ("simformer.sampling", "simformer.simformer.sampling"),
        "guidance": ("simformer.guidance", "simformer.simformer.guidance"),
        "condition_masks": ("simformer.condition_masks", "simformer.simformer.condition_masks"),
    }.items():
        if _import_first is not None:
            try:
                mods[key] = _import_first(candidates, what=key)
                continue
            except Exception:
                pass
        for name in candidates:
            try:
                import importlib

                mods[key] = importlib.import_module(name)
                break
            except Exception:
                continue
    return mods


def _resolve(path: Optional[str], default: str) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(default).expanduser().resolve()


def _as_float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            try:
                return [float(v) for v in json.loads(text)]
            except Exception:
                text = text.strip("[]")
        return [float(v) for v in text.split(",") if v.strip()]
    return [float(value)]


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


# --------------------------------------------------------------------------------------
# Checkpoint / model construction
# --------------------------------------------------------------------------------------
def load_checkpoint_meta(ckpt: Path) -> Dict[str, Any]:
    """Read the ``train_meta.json`` sitting next to a checkpoint (best effort)."""
    meta_path = ckpt.parent / "train_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    return meta


def load_model(
    ckpt: Path,
    task_name: str,
    task: Any,
    core: Dict[str, Any],
    *,
    mask: str = "directed",
    sde_name: str = "vesde",
    n_steps: int = 500,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    """Rebuild the model + trainer from a checkpoint.

    Returns ``(model, sde, tokenizer, meta)``.
    """
    training = core.get("training")
    if training is None:
        raise RuntimeError("simformer.training module unavailable")

    meta = load_checkpoint_meta(ckpt)
    tf_cfg = dict(meta.get("task_config", {}) or {})
    tr_cfg = dict(meta.get("train_config", {}) or {})

    n_parameters, n_data = task_dims(task, None) if task is not None else (None, None)  # type: ignore[misc]
    if n_parameters is None:
        n_parameters = int(getattr(task, "n_parameters", tf_cfg.get("n_parameters", 0)) or 0)
        n_data = int(getattr(task, "n_data", tf_cfg.get("n_data", 0)) or 0)

    model = build_model(  # type: ignore[misc]
        task,
        task_name,
        int(tf_cfg.get("token_dim", DEFAULT_TASK_CONFIG.get("token_dim", 50))),
        meta.get("n_layers", DEFAULT_TASK_CONFIG.get("n_layers")),
        int(tf_cfg.get("n_heads", DEFAULT_TASK_CONFIG.get("n_heads", 4))),
        int(tf_cfg.get("attention_size", DEFAULT_TASK_CONFIG.get("attention_size", 10))),
        int(tf_cfg.get("widening_factor", DEFAULT_TASK_CONFIG.get("widening_factor", 3))),
        int(tf_cfg.get("time_embed_dim", DEFAULT_TASK_CONFIG.get("time_embed_dim", 128))),
        None,
        core,
    )
    sde = build_sde(sde_name or tf_cfg.get("sde", "vesde"), core, n_steps)  # type: ignore[misc]
    tokenizer = getattr(model, "tokenizer", None)

    trainer = training.SimformerTrainer(model=model, sde=sde, tokenizer=tokenizer, device=device)
    if ckpt.exists():
        try:
            trainer.load(str(ckpt))
            if verbose:
                print(f"[sample] loaded checkpoint {ckpt}")
        except Exception as exc:  # pragma: no cover
            print(f"[sample] WARNING: could not load checkpoint ({exc}); using fresh weights")
    elif verbose:
        print(f"[sample] WARNING: checkpoint {ckpt} not found; sampling from untrained model")

    return model, sde, tokenizer, meta


# --------------------------------------------------------------------------------------
# Task observations
# --------------------------------------------------------------------------------------
def make_observation(
    task: Any,
    joint: Optional[np.ndarray],
    rng: np.random.Generator,
    *,
    index: int = 0,
) -> np.ndarray:
    """Pick one joint sample ``[theta | x]`` to act as the ground-truth observation."""
    if joint is not None and len(joint) > 0:
        idx = int(index) % len(joint)
        return np.asarray(joint[idx], dtype=np.float64)
    theta = np.asarray(task.prior_sample(1, rng), dtype=np.float64).reshape(1, -1)
    x = np.asarray(task.simulate(theta, rng), dtype=np.float64).reshape(1, -1)
    return to_joint(task, theta, x)[0]  # type: ignore[misc]


def build_condition_mask(
    mode: str,
    n_parameters: int,
    n_data: int,
    *,
    raw_mask: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Variable-level condition mask ``M_C`` for the requested mode."""
    mask = np.zeros(int(n_parameters) + int(n_data), dtype=np.float64)
    if raw_mask is not None:
        arr = np.asarray(raw_mask, dtype=np.float64).reshape(-1)
        if arr.shape[0] != mask.shape[0]:
            raise ValueError(
                f"condition mask length {arr.shape[0]} != n_variables {mask.shape[0]}"
            )
        return (arr > 0.5).astype(np.float64)
    if mode == "joint":
        return mask
    if mode == "posterior":
        mask[int(n_parameters):] = 1.0
        return mask
    if mode == "likelihood":
        mask[: int(n_parameters)] = 1.0
        return mask
    raise ValueError(f"cannot build condition mask for mode {mode!r}")


# --------------------------------------------------------------------------------------
# Sampling routines
# --------------------------------------------------------------------------------------
def sample_joint(
    model: Any,
    sde: Any,
    tokenizer: Any,
    *,
    n_samples: int,
    n_steps: int,
    seed: int,
    core: Dict[str, Any],
    **kwargs: Any,
) -> np.ndarray:
    sampling = core.get("sampling")
    if sampling is not None and hasattr(sampling, "ConditionalSampler"):
        cs = sampling.ConditionalSampler(model=model, sde=sde, tokenizer=tokenizer)
        return np.asarray(
            cs.joint(n_samples=n_samples, n_steps=n_steps, seed=seed, **kwargs), dtype=np.float64
        )
    if sampling is not None and hasattr(sampling, "sample_joint"):
        return np.asarray(
            sampling.sample_joint(
                model, sde=sde, tokenizer=tokenizer, n_samples=n_samples, n_steps=n_steps, seed=seed
            ),
            dtype=np.float64,
        )
    raise RuntimeError("no joint sampler available in simformer.sampling")


def sample_conditional_mode(
    model: Any,
    sde: Any,
    tokenizer: Any,
    *,
    mode: str,
    observation: np.ndarray,
    n_parameters: int,
    n_data: int,
    n_samples: int,
    n_steps: int,
    seed: int,
    core: Dict[str, Any],
    condition_mask: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Sample one conditional; returns dict with ``samples``/``mask``/``values``."""
    sampling = core.get("sampling")
    if sampling is None:
        raise RuntimeError("simformer.sampling module unavailable")

    if mode == "joint":
        return {
            "samples": sample_joint(
                model, sde, tokenizer, n_samples=n_samples, n_steps=n_steps, seed=seed, core=core
            ),
            "mask": np.zeros(int(n_parameters) + int(n_data)),
            "values": np.zeros(int(n_parameters) + int(n_data)),
            "mode": mode,
        }

    mask = condition_mask
    if mask is None:
        mask = build_condition_mask(mode, n_parameters, n_data)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1)
    values = mask * np.asarray(observation, dtype=np.float64).reshape(-1)

    cs = sampling.ConditionalSampler(
        model=model,
        sde=sde,
        tokenizer=tokenizer,
        n_parameters=n_parameters,
        n_data=n_data,
    )
    if mode == "posterior" and hasattr(cs, "posterior"):
        x_obs = np.asarray(observation, dtype=np.float64).reshape(-1)[int(n_parameters):]
        samples = cs.posterior(x_obs=x_obs, n_samples=n_samples, n_steps=n_steps, seed=seed, **kwargs)
    elif mode == "likelihood" and hasattr(cs, "likelihood"):
        theta = np.asarray(observation, dtype=np.float64).reshape(-1)[: int(n_parameters)]
        samples = cs.likelihood(theta=theta, n_samples=n_samples, n_steps=n_steps, seed=seed, **kwargs)
    else:
        samples = cs.sample_from_condition_mask(
            mask, values, n_samples=n_samples, n_steps=n_steps, seed=seed, **kwargs
        )
    return {
        "samples": np.asarray(samples, dtype=np.float64),
        "mask": mask,
        "values": values,
        "mode": mode,
    }


def sample_arbitrary_conditionals(
    model: Any,
    sde: Any,
    tokenizer: Any,
    *,
    observation: np.ndarray,
    n_parameters: int,
    n_data: int,
    n_targets: int,
    n_samples: int,
    n_steps: int,
    seed: int,
    core: Dict[str, Any],
) -> Tuple[List[np.ndarray], np.ndarray]:
    """``n_targets`` random conditional targets (Sec. 4.1 arbitrary-conditionals protocol)."""
    sampling = core.get("sampling")
    if sampling is not None and hasattr(sampling, "random_conditional_targets"):
        masks = np.asarray(
            sampling.random_conditional_targets(
                int(n_parameters) + int(n_data), n_targets=n_targets, seed=seed
            )
        )
        if masks.ndim == 1:
            masks = masks[None, :]
    else:
        rng = np.random.default_rng(seed)
        masks = (rng.random((int(n_targets), int(n_parameters) + int(n_data))) < 0.5).astype(np.float64)

    cs = None
    if sampling is not None and hasattr(sampling, "ConditionalSampler"):
        cs = sampling.ConditionalSampler(
            model=model,
            sde=sde,
            tokenizer=tokenizer,
            n_parameters=n_parameters,
            n_data=n_data,
        )

    out: List[np.ndarray] = []
    for i, m in enumerate(masks):
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        values = m * np.asarray(observation, dtype=np.float64).reshape(-1)
        if cs is not None:
            s = cs.sample_from_condition_mask(
                m, values, n_samples=n_samples, n_steps=n_steps, seed=seed + i
            )
        else:
            s = sampling.sample_conditional(
                model,
                m,
                values,
                sde=sde,
                tokenizer=tokenizer,
                n_samples=n_samples,
                n_steps=n_steps,
                seed=seed + i,
            )
        out.append(np.asarray(s, dtype=np.float64))
    return out, masks


def sample_guided(
    model: Any,
    sde: Any,
    tokenizer: Any,
    *,
    observation: np.ndarray,
    n_parameters: int,
    n_data: int,
    n_samples: int,
    n_steps: int,
    seed: int,
    core: Dict[str, Any],
    lower: Optional[Sequence[float]],
    upper: Optional[Sequence[float]],
    indices: Optional[Sequence[int]],
    self_recurrence: int,
    guidance_scale: float,
    mode: str = "posterior",
) -> Dict[str, Any]:
    """Interval-guided sampling with the paper's Algorithm 1 (Sec. 3.4)."""
    guidance = core.get("guidance")
    if guidance is None:
        raise RuntimeError("simformer.guidance module unavailable")

    mask = build_condition_mask(mode, n_parameters, n_data)
    values = mask * np.asarray(observation, dtype=np.float64).reshape(-1)

    if hasattr(guidance, "GuidedSampler"):
        gs = guidance.GuidedSampler.from_model(model, sde=sde, tokenizer=tokenizer)  # type: ignore[attr-defined]
        cfg = None
        if hasattr(guidance, "GuidanceConfig"):
            cfg = guidance.GuidanceConfig(
                n_steps=n_steps,
                seed=seed,
                self_recurrence=int(self_recurrence),
                guidance_scale=float(guidance_scale),
            )
            try:
                gs = guidance.GuidedSampler.from_model(
                    model, sde=sde, tokenizer=tokenizer, config=cfg
                )  # type: ignore[attr-defined]
            except TypeError:
                pass
        inter = guidance.interval_constraint(indices=indices, lower=lower, upper=upper)
        result = gs.sample(
            constraint=inter,
            n_samples=n_samples,
            condition_mask=mask,
            condition_values=values,
        )
        samples = getattr(result, "samples", result)
        sat = getattr(result, "constraint_satisfaction", None)
        return {
            "samples": np.asarray(samples, dtype=np.float64),
            "mask": mask,
            "values": values,
            "constraint_satisfaction": sat,
            "mode": "guided",
        }

    # Functional fallback: general_guidance(score_fn, sde, constraint, ...)
    sampling = core.get("sampling")
    score_fn = None
    if sampling is not None and hasattr(sampling, "make_score_fn"):
        score_fn = sampling.make_score_fn(model, sde=sde, tokenizer=tokenizer)
    if score_fn is None:
        raise RuntimeError("unable to build a score function for guided sampling")
    inter = guidance.interval_constraint(indices=indices, lower=lower, upper=upper)
    result = guidance.general_guidance(
        score_fn,
        sde,
        inter,
        n_samples=n_samples,
        n_steps=n_steps,
        seed=seed,
        self_recurrence=int(self_recurrence),
        guidance_scale=float(guidance_scale),
        condition_mask=mask,
        condition_values=values,
    )
    samples = getattr(result, "samples", result)
    return {
        "samples": np.asarray(samples, dtype=np.float64),
        "mask": mask,
        "values": values,
        "constraint_satisfaction": getattr(result, "constraint_satisfaction", None),
        "mode": "guided",
    }


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def reference_samples(
    task: Any,
    mask: np.ndarray,
    values: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    core: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Ground-truth MCMC samples of the same conditional (Appendix A2.2)."""
    reference = core.get("reference")
    if reference is None:
        try:
            modules = _core()
        except Exception:
            modules = {}
        reference = modules.get("reference")
    if reference is None:
        try:
            import importlib

            for name in ("simformer.reference.mcmc", "simformer.reference"):
                try:
                    reference = importlib.import_module(name)
                    break
                except Exception:
                    continue
        except Exception:
            reference = None
    if reference is None or not hasattr(reference, "sample_reference"):
        return None
    try:
        samples = reference.sample_reference(
            task,
            mask,
            values,
            n_samples=n_samples,
            seed=seed,
            return_full=False,
        )
        return np.asarray(samples, dtype=np.float64)
    except Exception as exc:  # pragma: no cover
        print(f"[sample] reference sampling failed: {exc}")
        return None


def c2st_score(
    approx: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int,
    core: Dict[str, Any],
) -> Optional[float]:
    """C2ST accuracy (0.5 == indistinguishable) against reference samples."""
    try:
        from simformer.eval.c2st import evaluate_c2st  # type: ignore
    except Exception:
        try:
            eval_mod = _import_first(("simformer.eval.c2st", "simformer.eval"), what="c2st")
            evaluate_c2st = getattr(eval_mod, "evaluate_c2st")
        except Exception:
            return None
    a = np.asarray(approx, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    # only compare the latent block if shapes mismatch
    if a.shape != b.shape and a.ndim == 2 and b.ndim == 2:
        d = min(a.shape[1], b.shape[1])
        a, b = a[:, :d], b[:, :d]
    try:
        return float(evaluate_c2st(a, b, seed=seed))
    except Exception as exc:  # pragma: no cover
        print(f"[sample] C2ST failed: {exc}")
        return None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simformer-sample",
        description="Sample arbitrary conditionals from a trained Simformer.",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK_CONFIG.get("task", "two_moons"))
    parser.add_argument("--outdir", type=str, default=DEFAULT_SAMPLE_CONFIG["outdir"],
                        help="directory with the training artefacts")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="explicit checkpoint path (default: <outdir>/model.pt)")
    parser.add_argument("--mode", type=str, default="posterior", choices=ALLOWED_MODES)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_SAMPLE_CONFIG["n_samples"])
    parser.add_argument("--n-steps", type=int, default=DEFAULT_SAMPLE_CONFIG["n_steps"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SAMPLE_CONFIG["batch_size"])
    parser.add_argument("--n-targets", type=int, default=DEFAULT_SAMPLE_CONFIG["n_targets"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_CONFIG["seed"])
    parser.add_argument("--index", type=int, default=0, help="which joint sample acts as truth")
    parser.add_argument("--sde", type=str, default=DEFAULT_SAMPLE_CONFIG["sde"])
    parser.add_argument("--mask", type=str, default="directed",
                        choices=("dense", "undirected", "directed", "none"))
    parser.add_argument("--condition-mask", type=str, default=None,
                        help="comma separated 0/1 variable-level condition mask")
    parser.add_argument("--condition-values", type=str, default=None,
                        help="optional explicit conditioning values (comma separated)")
    parser.add_argument("--observation", type=str, default=None,
                        help="path to .npy file with a joint vector [theta | x]")
    parser.add_argument("--dataset", type=str, default=None,
                        help="path to a joint dataset .npy (default: <outdir>/dataset.npy)")
    parser.add_argument("--constraint-index", type=str, default=None,
                        help="comma separated variable indices for interval constraints")
    parser.add_argument("--constraint-lower", type=str, default=None)
    parser.add_argument("--constraint-upper", type=str, default=None)
    parser.add_argument("--self-recurrence", type=int, default=DEFAULT_SAMPLE_CONFIG["self_recurrence"])
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_SAMPLE_CONFIG["guidance_scale"])
    parser.add_argument("--c2st", action="store_true", help="score samples against MCMC references")
    parser.add_argument("--n-reference", type=int, default=DEFAULT_SAMPLE_CONFIG["n_reference"])
    parser.add_argument("--config", type=str, default=None, help="optional YAML/JSON config")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-samples/--no-save-samples", dest="save_samples", default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    verbose = not args.quiet

    cfg: Dict[str, Any] = {}
    if args.config and load_yaml_config is not None:
        try:
            cfg = load_yaml_config(args.config) or {}
        except Exception as exc:
            print(f"[sample] WARNING: could not read config ({exc})")

    def val(key: str, default: Any) -> Any:
        if merged is not None:
            try:
                return merged(args, cfg, key, default)
            except Exception:
                pass
        return getattr(args, key.replace("-", "_"), default)

    task_name = str(val("task", DEFAULT_TASK_CONFIG.get("task", "two_moons")))
    mode = str(val("mode", "posterior"))
    n_samples = int(val("n_samples", 1000))
    n_steps = int(val("n_steps", 500))
    seed = int(val("seed", 0))
    sde_name = str(val("sde", "vesde"))
    mask_variant = str(val("mask", "directed"))

    outdir = _resolve(args.outdir, "runs/sample")
    ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else outdir / "model.pt"
    outdir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    core = _core()
    if verbose:
        print(f"[sample] task={task_name} mode={mode} ckpt={ckpt}")

    # ----- task -------------------------------------------------------------------
    task = None
    if build_task_instance is not None:
        for module_name in ("simformer.tasks", "simformer.tasks.__init__", "tasks"):
            try:
                import importlib

                tasks_mod = importlib.import_module(module_name)
                task = build_task_instance(tasks_mod, task_name, seed)
                break
            except Exception:
                continue

    n_parameters = int(getattr(task, "n_parameters", 0) or 0)
    n_data = int(getattr(task, "n_data", 0) or 0)
    if not n_parameters:
        meta = load_checkpoint_meta(ckpt)
        n_parameters = int(meta.get("n_parameters", 0) or meta.get("train_config", {}).get("n_parameters", 0) or 0)
        n_data = int(meta.get("n_data", 0) or meta.get("train_config", {}).get("n_data", 0) or 0)

    # ----- model ------------------------------------------------------------------
    model, sde, tokenizer, meta = load_model(
        ckpt,
        task_name,
        task,
        core,
        mask=mask_variant,
        sde_name=sde_name,
        n_steps=n_steps,
        device=str(val("device", "cpu")),
        verbose=verbose,
    )

    if not n_parameters:
        n_parameters = int(getattr(model, "n_parameter_variables", 0) or 0)
        n_data = int(getattr(model, "n_data_variables", 0) or 0)

    # ----- observation ------------------------------------------------------------
    rng = np.random.default_rng(seed)
    joint: Optional[np.ndarray] = None
    ds_path = Path(args.dataset).expanduser().resolve() if args.dataset else outdir / "dataset.npy"
    if ds_path.exists():
        try:
            joint = np.load(ds_path)
        except Exception:
            joint = None
    if args.observation:
        try:
            obs = np.load(Path(args.observation).expanduser().resolve())
            observation = np.asarray(obs, dtype=np.float64).reshape(-1)
        except Exception as exc:
            raise SystemExit(f"[sample] could not read observation file: {exc}")
    else:
        if task is None:
            raise SystemExit("[sample] no --observation given and no task available")
        observation = make_observation(task, joint, rng, index=int(val("index", 0)))

    if verbose:
        print(f"[sample] observation dim={observation.shape[0]} (n_theta={n_parameters}, n_x={n_data})")

    raw_mask = _as_float_list(args.condition_mask)
    condition_values = _as_float_list(args.condition_values)

    # ----- sampling ---------------------------------------------------------------
    results: Dict[str, Any] = {}
    payload: Dict[str, Any] = {"mode": mode, "task": task_name, "checkpoint": str(ckpt)}

    if mode in ("joint", "posterior", "likelihood"):
        if mode == "conditional":
            mask = build_condition_mask(mode, n_parameters, n_data)
        else:
            mask = build_condition_mask(mode, n_parameters, n_data, raw_mask=raw_mask)
        res = sample_conditional_mode(
            model,
            sde,
            tokenizer,
            mode="joint" if mode == "joint" else "conditional",
            observation=observation,
            n_parameters=n_parameters,
            n_data=n_data,
            n_samples=n_samples,
            n_steps=n_steps,
            seed=seed,
            core=core,
            condition_mask=mask,
        )
        samples = res["samples"]
        payload["condition_mask"] = np.asarray(res["mask"]).tolist()
        payload["condition_values"] = np.asarray(res["values"]).tolist()

    elif mode == "conditional":
        if raw_mask is None:
            raise SystemExit("[sample] --mode conditional requires --condition-mask")
        if condition_values is None:
            condition_values = observation
        mask = build_condition_mask("conditional", n_parameters, n_data, raw_mask=raw_mask)
        values = mask * np.asarray(condition_values[: mask.shape[0]], dtype=np.float64)
        sampling = core.get("sampling")
        if sampling is None:
            raise SystemExit("[sample] simformer.sampling unavailable")
        if hasattr(sampling, "ConditionalSampler"):
            cs = sampling.ConditionalSampler(
                model=model, sde=sde, tokenizer=tokenizer,
                n_parameters=n_parameters, n_data=n_data,
            )
            samples = cs.sample_from_condition_mask(
                mask, values, n_samples=n_samples, n_steps=n_steps, seed=seed
            )
        else:
            samples = sampling.sample_conditional(
                model, mask, values, sde=sde, tokenizer=tokenizer,
                n_samples=n_samples, n_steps=n_steps, seed=seed,
            )
        samples = np.asarray(samples, dtype=np.float64)
        payload["condition_mask"] = mask.tolist()
        payload["condition_values"] = values.tolist()

    elif mode == "arbitrary":
        samples, masks = sample_arbitrary_conditionals(
            model,
            sde,
            tokenizer,
            observation=observation,
            n_parameters=n_parameters,
            n_data=n_data,
            n_targets=int(val("n_targets", 100)),
            n_samples=n_samples,
            n_steps=n_steps,
            seed=seed,
            core=core,
        )
        payload["n_targets"] = len(samples)
        payload["target_masks"] = np.asarray(masks).tolist()
        if val("save_samples", True):
            stack_path = outdir / "samples_arbitrary.npz"
            np.savez_compressed(
                stack_path,
                **{f"target_{i:04d}": np.asarray(s).astype(np.float32) for i, s in enumerate(samples)},
                masks=np.asarray(masks),
                condition_values=np.asarray(observation, dtype=np.float32),
            )
            payload["samples_file"] = str(stack_path)
            if verbose:
                print(f"[sample] wrote {len(samples)} conditionals to {stack_path}")
        if args.c2st and task is not None:
            accs: List[float] = []
            for i, (s, m) in enumerate(zip(samples, masks)):
                ref = reference_samples(
                    task, np.asarray(m), np.asarray(m) * observation,
                    n_samples=int(val("n_reference", 1000)), seed=seed + i, core=core,
                )
                if ref is None:
                    continue
                score = c2st_score(s, ref, seed=seed + i, core=core)
                if score is not None:
                    accs.append(score)
            if accs:
                payload["c2st"] = {
                    "mean": float(np.mean(accs)),
                    "std": float(np.std(accs)),
                    "median": float(np.median(accs)),
                    "n_targets": len(accs),
                }
                payload["c2st_per_target"] = accs
                if verbose:
                    print(f"[sample] C2ST mean={np.mean(accs):.4f} +/- {np.std(accs):.4f}")
        _save_json(outdir / "sample_arbitrary_summary.json", payload)
        if verbose:
            print(f"[sample] done in {time.time() - start:.1f}s")
        return 0

    else:  # guided
        indices = _as_float_list(args.constraint_index)
        indices = [int(i) for i in indices] if indices is not None else None
        lower = _as_float_list(args.constraint_lower)
        upper = _as_float_list(args.constraint_upper)
        if lower is None and upper is None:
            raise SystemExit("[sample] --mode guided requires --constraint-lower/--constraint-upper")
        res = sample_guided(
            model,
            sde,
            tokenizer,
            observation=observation,
            n_parameters=n_parameters,
            n_data=n_data,
            n_samples=n_samples,
            n_steps=n_steps,
            seed=seed,
            core=core,
            lower=lower,
            upper=upper,
            indices=indices,
            self_recurrence=int(val("self_recurrence", 0)),
            guidance_scale=float(val("guidance_scale", 1.0)),
            mode=str(cfg.get("guidance_mode", "posterior")),
        )
        samples = res["samples"]
        payload["condition_mask"] = np.asarray(res["mask"]).tolist()
        payload["condition_values"] = np.asarray(res["values"]).tolist()
        payload["constraint"] = {"indices": indices, "lower": lower, "upper": upper}
        if res.get("constraint_satisfaction") is not None:
            payload["constraint_satisfaction"] = float(res["constraint_satisfaction"])
            if verbose:
                print(f"[sample] constraint satisfaction={float(res['constraint_satisfaction']):.3f}")

    # ----- landmark statistics ----------------------------------------------------
    payload["n_samples"] = int(np.asarray(samples).shape[0]) if np.asarray(samples).ndim else 1
    payload["n_steps"] = n_steps
    payload["sample_mean"] = np.asarray(samples).mean(axis=0).tolist() if np.asarray(samples).ndim == 2 else None
    payload["sample_std"] = np.asarray(samples).std(axis=0).tolist() if np.asarray(samples).ndim == 2 else None
    payload["elapsed_seconds"] = time.time() - start

    if val("save_samples", True):
        out_npz = outdir / f"samples_{mode}.npz"
        np.savez_compressed(
            out_npz,
            samples=np.asarray(samples, dtype=np.float32),
            condition_mask=np.asarray(payload.get("condition_mask", []), dtype=np.float32),
            condition_values=np.asarray(payload.get("condition_values", []), dtype=np.float32),
            observation=np.asarray(observation, dtype=np.float32),
        )
        payload["samples_file"] = str(out_npz)
        if verbose:
            print(f"[sample] wrote samples -> {out_npz}")

    if args.c2st and task is not None and mode in ("posterior", "likelihood", "conditional", "guided"):
        m = np.asarray(payload.get("condition_mask", []), dtype=np.float64)
        if m.size == 0:
            m = build_condition_mask(
                mode if mode in ("posterior", "likelihood") else "posterior", n_parameters, n_data
            )
        v = m * np.asarray(observation, dtype=np.float64)
        ref = reference_samples(
            task, m, v, n_samples=int(val("n_reference", 1000)), seed=seed, core=core
        )
        if ref is not None:
            score = c2st_score(samples, ref, seed=seed, core=core)
            if score is not None:
                payload["c2st"] = score
                payload["c2st_reference"] = "mcmc"
                if verbose:
                    print(f"[sample] C2ST={score:.4f} (0.5 = indistinguishable)")

    _save_json(outdir / f"sample_{mode}_summary.json", payload)
    if verbose:
        print(f"[sample] done in {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
