"""Full benchmark sweep for Simformer (paper Sec. 4.1, Appendix A2.1/A3.1).

This script orchestrates every experiment that the Simformer paper reports and that
the reproduction plan declares in scope:

* ``benchmark``  - Fig. 4: C2ST accuracy of amortized posteriors against ground-truth
  posteriors on Gaussian Linear / Gaussian Mixture / Two Moons / SLCP at 1k, 10k and
  100k simulations, for the three attention-mask variants ``dense``, ``undirected``
  and ``directed``.
* ``baselines``  - the same C2ST protocol for NPE / NLE / NRE (``sbi``) and NPSE
  (conditional-MLP score network).
* ``arbitrary``  - Sec. 4.1: C2ST of Simformer's arbitrary conditionals against MCMC
  references for 100 random joint/conditional targets on the four benchmark tasks
  plus the Tree and HMM tasks (Appendix A2.2 reference protocols).
* ``reverse_sde`` - Appendix A3.1/Fig. A7: C2ST as a function of the number of
  reverse-SDE evaluation steps (50-500 steps are sufficient).
* ``scientific`` - Sec. 4.2-4.4: Lotka-Volterra / SIRD / Hodgkin-Huxley sanity
  diagnostics (posterior coverage of the true parameters, posterior-predictive
  reconstruction error).

Everything is written to ``--outdir`` as JSON (and a Fig. 4 style PNG when
matplotlib is available).  The script is deliberately tolerant of small API
differences between the sibling modules because it doubles as the paper's
"one command reproduces the tables" entry point.

Usage::

    python scripts/run_experiments.py --experiments benchmark --quick
    python scripts/run_experiments.py --experiments all --outdir runs/full
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------
# import bootstrapping (mirrors scripts/train.py so the script works from any cwd)
# --------------------------------------------------------------------------------------
import importlib
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_REPO_DIR = Path(__file__).resolve().parents[2]
_PKG_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_DIR), str(_PKG_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _import_first(candidates: Sequence[str], what: str = "module") -> Any:
    """Import the first importable module among ``candidates``."""
    last_error: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on environment
            last_error = exc
    raise ImportError(f"could not import any of {list(candidates)} ({what}): {last_error}")


def _script_module(name: str) -> Optional[Any]:
    """Import a sibling script module (``train`` / ``sample``) with fallbacks."""
    for candidate in (f"simformer.scripts.{name}", name):
        try:
            return importlib.import_module(candidate)
        except Exception:
            continue
    return None


TRAIN = _script_module("train")
SAMPLE = _script_module("sample")


# --------------------------------------------------------------------------------------
# core module registry
# --------------------------------------------------------------------------------------
def _core() -> Dict[str, Any]:
    """Resolve the Simformer core modules lazily with dual-name fallbacks."""
    registry: Dict[str, Any] = {}
    paths = {
        "training": ("simformer.simformer.training", "simformer.training"),
        "diffusion": ("simformer.simformer.diffusion", "simformer.diffusion"),
        "transformer": ("simformer.simformer.transformer", "simformer.transformer"),
        "attention_masks": ("simformer.simformer.attention_masks", "simformer.attention_masks"),
        "tokenizer": ("simformer.simformer.tokenizer", "simformer.tokenizer"),
        "graph_inversion": ("simformer.simformer.graph_inversion", "simformer.graph_inversion"),
        "condition_masks": ("simformer.simformer.condition_masks", "simformer.condition_masks"),
        "sampling": ("simformer.simformer.sampling", "simformer.sampling"),
        "guidance": ("simformer.simformer.guidance", "simformer.guidance"),
        "c2st": ("simformer.eval.c2st", "eval.c2st"),
        "coverage": ("simformer.eval.coverage", "eval.coverage"),
        "nll": ("simformer.eval.nll", "eval.nll"),
        "mcmc": ("simformer.reference.mcmc", "reference.mcmc"),
        "tasks": ("simformer.tasks", "tasks"),
        "baselines": ("simformer.baselines", "baselines"),
    }
    for key, candidates in paths.items():
        try:
            registry[key] = _import_first(candidates, key)
        except Exception as exc:  # pragma: no cover
            registry[key] = None
            registry.setdefault("_errors", {})[key] = repr(exc)
    return registry


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
BENCHMARK_TASKS: Tuple[str, ...] = ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp")
ARBITRARY_TASKS: Tuple[str, ...] = BENCHMARK_TASKS + ("tree", "hmm")
SCIENTIFIC_TASKS: Tuple[str, ...] = ("lotka_volterra", "sird", "hodgkin_huxley")
DEFAULT_BUDGETS: Tuple[int, ...] = (1000, 10000, 100000)
MASK_VARIANTS: Tuple[str, ...] = ("dense", "undirected", "directed")
BASELINES: Tuple[str, ...] = ("npe", "nle", "nre", "npse")

#: Training steps used per simulation budget.  The paper is silent on the exact number
#: (it only specifies Adam + early stopping on the validation loss), so we scale the
#: optimisation budget with the number of available simulations, which is the behaviour
#: of the official code.
DEFAULT_STEPS_FOR_BUDGET: Dict[int, int] = {1000: 20000, 10000: 50000, 100000: 100000}

#: Number of posterior targets used for the C2ST estimates (Sec. 4.1 draws several
#: observations and averages the classifier accuracy).
DEFAULT_N_TARGETS = 10
DEFAULT_N_REFERENCE = 1000
DEFAULT_N_EVAL_SAMPLES = 1000

EXPERIMENTS = ("benchmark", "baselines", "arbitrary", "reverse_sde", "scientific")


def _as_array(x: Any):
    import numpy as np

    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):  # torch tensor
        try:
            return x.detach().cpu().numpy()
        except Exception:
            pass
    try:
        return np.asarray(x)
    except Exception:
        return x


def _call_with_fallbacks(fn: Callable, attempts: Sequence[Callable], what: str = "call") -> Any:
    """Try a sequence of thunks, ignoring ``TypeError`` signature mismatches."""
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
            continue
    raise TypeError(f"no compatible signature for {what}: {last_error}")


# --------------------------------------------------------------------------------------
# dataset / task plumbing
# --------------------------------------------------------------------------------------
def _task_module() -> Any:
    core = _core()
    mod = core.get("tasks")
    if mod is None:
        raise ImportError("simformer.tasks is not importable")
    return mod


def build_task_instance(task_name: str, seed: int = 0) -> Any:
    """Instantiate a task simulator (delegates to ``scripts/train.py`` when possible)."""
    if TRAIN is not None and hasattr(TRAIN, "build_task_instance"):
        try:
            return TRAIN.build_task_instance(_task_module(), task_name, seed)
        except Exception:
            pass
    mod = _task_module()
    if hasattr(mod, "build_task"):
        try:
            return mod.build_task(task_name, seed=seed)
        except TypeError:
            return mod.build_task(task_name)
    raise RuntimeError(f"cannot build task {task_name!r}")


def make_joint_dataset(task: Any, n_simulations: int, seed: int = 0, verbose: bool = False):
    """Generate ``(theta, x)`` at a given simulation budget."""
    if TRAIN is not None and hasattr(TRAIN, "make_joint_dataset"):
        try:
            return TRAIN.make_joint_dataset(task, n_simulations, seed, verbose=verbose)
        except TypeError:
            try:
                return TRAIN.make_joint_dataset(task, n_simulations, seed)
            except Exception:
                pass
        except Exception:
            pass
    rng = _default_rng(seed)
    return task.make_dataset(n_simulations, rng=rng, verbose=verbose)


def _default_rng(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def task_dims(task: Any, joint: Any) -> Tuple[int, int]:
    """Return ``(n_parameters, n_data)`` for a task."""
    if TRAIN is not None and hasattr(TRAIN, "task_dims"):
        try:
            return tuple(TRAIN.task_dims(task, joint))  # type: ignore[return-value]
        except Exception:
            pass
    n_parameters = int(getattr(task, "n_parameters", 0) or 0)
    n_data = int(getattr(task, "n_data", 0) or 0)
    if (not n_parameters or not n_data) and joint is not None:
        import numpy as np

        total = int(np.asarray(joint).shape[-1])
        if not n_parameters:
            n_parameters = total - n_data
        if not n_data:
            n_data = total - n_parameters
    return n_parameters, n_data


def to_joint(task: Any, theta: Any, x: Any):
    import numpy as np

    if hasattr(task, "to_joint"):
        try:
            return np.asarray(task.to_joint(theta, x))
        except Exception:
            pass
    return np.concatenate([np.atleast_2d(theta), np.atleast_2d(x)], axis=-1)


def split_joint(task: Any, joint: Any, n_parameters: int) -> Tuple[Any, Any]:
    import numpy as np

    joint = np.atleast_2d(np.asarray(joint))
    return joint[:, :n_parameters], joint[:, n_parameters:]


# --------------------------------------------------------------------------------------
# training a single Simformer
# --------------------------------------------------------------------------------------
def train_simformer(
    task_name: str,
    budget: int,
    *,
    variant: str = "directed",
    seed: int = 0,
    max_steps: Optional[int] = None,
    batch_size: int = 1000,
    lr: float = 3e-4,
    sde: str = "vesde",
    n_sampling_steps: int = 500,
    device: str = "cpu",
    normalize: bool = False,
    task: Any = None,
    joint: Any = None,
    val_fraction: float = 0.1,
    verbose: bool = False,
    core: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train one Simformer configuration and return a bundle with the trained artefacts.

    Returns a dict with keys ``model``, ``sde``, ``tokenizer``, ``trainer``,
    ``attention_mask``, ``n_parameters``, ``n_data``, ``token_spec``, ``history``.
    """
    import numpy as np

    core = core or _core()
    tasks_mod = core.get("tasks")

    if task is None:
        task = build_task_instance(task_name, seed)
    if joint is None:
        joint = make_joint_dataset(task, budget, seed=seed, verbose=verbose)
    joint = np.atleast_2d(np.asarray(joint, dtype="float32"))
    n_parameters, n_data = task_dims(task, joint)

    # ---- model -----------------------------------------------------------------
    if TRAIN is not None and hasattr(TRAIN, "build_model"):
        sde_obj = TRAIN.build_sde(sde, core, n_sampling_steps)
        model = TRAIN.build_model(
            task=task,
            task_name=task_name,
            token_dim=getattr(TRAIN.DEFAULT_TASK_CONFIG, "get", lambda k, d=None: d)("token_dim", 50)
            if isinstance(TRAIN.DEFAULT_TASK_CONFIG, dict)
            else 50,
            n_layers=None,
            n_heads=4,
            attention_size=10,
            widening_factor=3,
            time_embed_dim=128,
            sde=sde_obj,
            core=core,
        )
    else:  # pragma: no cover - fallback path
        tokenizer_mod = core["tokenizer"]
        transformer_mod = core["transformer"]
        diffusion_mod = core["diffusion"]
        spec = task.token_spec() if hasattr(task, "token_spec") else tokenizer_mod.build_benchmark_spec(
            n_parameters, n_data
        )
        model = transformer_mod.build_score_network(task=task_name, spec=spec, token_dim=50)
        sde_obj = diffusion_mod.get_sde(sde, n_steps=n_sampling_steps)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None and task is not None and hasattr(task, "build_tokenizer"):
        try:
            tokenizer = task.build_tokenizer()
        except Exception:
            tokenizer = None
    token_spec = getattr(tokenizer, "spec", None)

    # ---- attention mask ---------------------------------------------------------
    attention_mask: Any = None
    if variant not in ("dense", "none", None) and TRAIN is not None:
        try:
            attention_mask = TRAIN.build_task_attention_mask(
                task_name,
                variant,
                n_parameters,
                n_data,
                tokenizer=tokenizer,
                input_dim=int(joint.shape[-1]),
                core=core,
                task=task,
                verbose=verbose,
            )
        except Exception as exc:  # pragma: no cover
            if verbose:
                print(f"[warn] attention mask {variant!r} failed ({exc}); using dense")
            attention_mask = None

    # ---- trainer ----------------------------------------------------------------
    training_mod = core["training"]
    cfg_cls = getattr(training_mod, "TrainingConfig", None)
    steps = max_steps if max_steps is not None else DEFAULT_STEPS_FOR_BUDGET.get(
        budget, max(5000, budget // 2)
    )
    config_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "lr": lr,
        "max_steps": int(steps),
        "device": device,
        "seed": seed,
        "normalize": bool(normalize),
    }
    config = None
    if cfg_cls is not None:
        try:
            config = cfg_cls(**config_kwargs)
        except TypeError:
            config = cfg_cls()

    trainer_cls = getattr(training_mod, "SimformerTrainer", None)
    trainer = trainer_cls(
        model,
        sde=sde_obj,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        config=config,
    )

    # validation split
    n_val = int(max(1, round(val_fraction * joint.shape[0]))) if val_fraction else 0
    val_joint = joint[:n_val] if n_val else None
    train_joint = joint[n_val:] if n_val else joint

    history = trainer.fit(
        x_joint=train_joint,
        val_joint=val_joint,
        verbose=verbose,
    )

    return {
        "task": task,
        "task_name": task_name,
        "model": model,
        "sde": sde_obj,
        "tokenizer": tokenizer,
        "trainer": trainer,
        "attention_mask": attention_mask,
        "attention_mask_variant": variant,
        "n_parameters": n_parameters,
        "n_data": n_data,
        "token_spec": token_spec,
        "budget": int(budget),
        "history": history if isinstance(history, dict) else {},
        "max_steps": int(steps),
    }


# --------------------------------------------------------------------------------------
# conditional sampling helpers
# --------------------------------------------------------------------------------------
def make_sampler(bundle: Dict[str, Any], *, n_sampling_steps: int = 500, seed: int = 0) -> Any:
    """Build a :class:`ConditionalSampler` for a trained bundle."""
    core = _core()
    sampling_mod = core["sampling"]
    sampler_cls = getattr(sampling_mod, "ConditionalSampler")
    mask = bundle.get("attention_mask")
    mask_fn = mask if callable(mask) else None
    static_mask = None if callable(mask) else mask
    kwargs: Dict[str, Any] = dict(
        model=bundle["model"],
        sde=bundle["sde"],
        tokenizer=bundle.get("tokenizer"),
        n_parameters=bundle.get("n_parameters"),
        n_data=bundle.get("n_data"),
    )
    if mask_fn is not None:
        kwargs["attention_mask_fn"] = mask_fn
    else:
        kwargs["attention_mask"] = static_mask
    try:
        return sampler_cls(**kwargs)
    except TypeError:
        kwargs.pop("attention_mask_fn", None)
        return sampler_cls(**kwargs)


def conditional_samples(
    sampler: Any,
    condition_mask: Any,
    condition_values: Any,
    *,
    n_samples: int = 1000,
    n_steps: int = 500,
    seed: int = 0,
) -> Any:
    """Draw samples for one conditioning assignment, tolerating API differences."""
    import numpy as np

    mask = np.asarray(condition_mask)
    values = np.asarray(condition_values, dtype="float32")

    fn = getattr(sampler, "sample_from_condition_mask", None)
    if not callable(fn):
        fn = getattr(sampler, "sample", None)

    def _normalize(out: Any) -> Any:
        out = _as_array(out)
        if hasattr(out, "samples"):
            out = out.samples
        return np.atleast_2d(np.asarray(out, dtype="float32"))

    if callable(fn):
        attempts = [
            lambda: fn(mask, values, n_samples=n_samples, n_steps=n_steps, seed=seed),
            lambda: fn(mask, values, n_samples=n_samples, n_steps=n_steps),
            lambda: fn(mask, values, n_samples=n_samples, seed=seed),
            lambda: fn(mask, values, n_samples=n_samples),
            lambda: fn(mask, values),
        ]
        return _normalize(_call_with_fallbacks(fn, attempts, "conditional sampling"))

    # fall back to the functional entry point
    sampling_mod = _core()["sampling"]
    sample_conditional = getattr(sampling_mod, "sample_conditional")
    return _normalize(
        sample_conditional(
            sampler.model if hasattr(sampler, "model") else None,
            mask,
            values,
            n_samples=n_samples,
            n_steps=n_steps,
            seed=seed,
        )
    )


def posterior_condition_mask(n_parameters: int, n_data: int):
    import numpy as np

    return np.concatenate(
        [np.zeros(n_parameters, dtype="float64"), np.ones(n_data, dtype="float64")]
    )


def likelihood_condition_mask(n_parameters: int, n_data: int):
    import numpy as np

    return np.concatenate(
        [np.ones(n_parameters, dtype="float64"), np.zeros(n_data, dtype="float64")]
    )


def joint_condition_mask(n_parameters: int, n_data: int):
    import numpy as np

    return np.zeros(n_parameters + n_data, dtype="float64")


# --------------------------------------------------------------------------------------
# reference (ground-truth) conditionals
# --------------------------------------------------------------------------------------
def reference_posterior_samples(
    task: Any,
    x_obs: Any,
    *,
    n_samples: int = 1000,
    seed: int = 0,
    n_parameters: Optional[int] = None,
    n_data: Optional[int] = None,
    use_mcmc: bool = True,
) -> Any:
    """Ground-truth posterior samples ``theta ~ p(theta | x_obs)`` for one observation.

    Preference order: task-provided exact/reference sampler, then the MCMC reference
    protocol from :mod:`simformer.reference.mcmc`.
    """
    import numpy as np

    rng = _default_rng(seed)
    x_obs = np.asarray(x_obs, dtype="float64")
    if x_obs.ndim > 1:
        x_obs = x_obs.reshape(-1)

    for name in ("reference_posterior_sample", "sample_reference_posterior", "ground_truth_posterior"):
        fn = getattr(task, name, None)
        if callable(fn):
            attempts = [
                lambda: fn(x_obs, n_samples=n_samples, rng=rng),
                lambda: fn(x_obs, n_samples=n_samples),
                lambda: fn(x_obs, n_samples, rng),
                lambda: fn(x_obs, n_samples),
                lambda: fn(x_obs),
            ]
            try:
                out = _call_with_fallbacks(fn, attempts, f"{name}")
                out = _as_array(out)
                if out is not None and np.size(out):
                    out = np.atleast_2d(np.asarray(out, dtype="float32"))
                    if out.shape[0] > n_samples:
                        out = out[:n_samples]
                    return out
            except Exception:
                continue

    map_fn = getattr(task, "map_estimate", None)
    log_prob_fn = getattr(task, "posterior_log_prob", None)
    if callable(log_prob_fn) and not use_mcmc:
        # crude importance-free fallback: MAP plus jitter (only used when MCMC is off)
        center = None
        if callable(map_fn):
            try:
                center = _as_array(map_fn(x_obs))
            except Exception:
                center = None
        if center is None:
            center = np.zeros(n_parameters or 1, dtype="float64")
        return np.asarray(
            center + 0.1 * rng.standard_normal((n_samples, np.size(center))), dtype="float32"
        )

    if use_mcmc:
        try:
            mcmc = _core()["mcmc"]
            mask = posterior_condition_mask(n_parameters or 0, n_data or 0)
            joint_dim = (n_parameters or 0) + (n_data or 0)
            values = np.zeros(joint_dim, dtype="float64")
            values[(n_parameters or 0):] = x_obs
            out = mcmc.sample_reference(
                task,
                mask,
                values,
                n_samples=n_samples,
                seed=seed,
                return_full=False,
            )
            out = _as_array(out)
            if out is not None and np.size(out):
                return np.atleast_2d(np.asarray(out, dtype="float32"))
        except Exception:
            pass

    raise RuntimeError("no reference posterior sampler available for this task")


# --------------------------------------------------------------------------------------
# C2ST evaluation
# --------------------------------------------------------------------------------------
def c2st(approx: Any, reference: Any, *, seed: int = 0, return_result: bool = False) -> Any:
    import numpy as np

    c2st_mod = _core()["c2st"]
    approx = np.atleast_2d(np.asarray(_as_array(approx), dtype="float32"))
    reference = np.atleast_2d(np.asarray(_as_array(reference), dtype="float32"))
    n = min(approx.shape[0], reference.shape[0])
    approx, reference = approx[:n], reference[:n]
    for name in ("evaluate_c2st", "c2st_accuracy", "c2st"):
        fn = getattr(c2st_mod, name, None)
        if callable(fn):
            try:
                return fn(approx, reference, seed=seed, n_trees=100, return_result=return_result)
            except TypeError:
                try:
                    return fn(approx, reference, seed=seed)
                except TypeError:
                    continue
    raise RuntimeError("no C2ST entry point available")


def evaluate_posterior_c2st(
    bundle: Dict[str, Any],
    *,
    n_targets: int = DEFAULT_N_TARGETS,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    n_steps: int = 500,
    seed: int = 0,
    sampler: Any = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """C2ST of Simformer's posterior against the ground-truth posterior (Sec. 4.1)."""
    import numpy as np

    task = bundle["task"]
    n_parameters = int(bundle["n_parameters"])
    n_data = int(bundle["n_data"])
    rng = _default_rng(seed)
    sampler = sampler or make_sampler(bundle, n_sampling_steps=n_steps, seed=seed)

    joint = make_joint_dataset(task, max(n_targets, 1), seed=seed + 12345)
    joint = np.atleast_2d(np.asarray(joint, dtype="float32"))[:n_targets]

    mask = posterior_condition_mask(n_parameters, n_data)
    accuracies: List[float] = []
    for i in range(joint.shape[0]):
        row = joint[i]
        theta_true, x_obs = row[:n_parameters], row[n_parameters:]
        try:
            approx = conditional_samples(
                sampler, mask, row, n_samples=n_samples, n_steps=n_steps, seed=seed + i
            )
            approx = approx[:, :n_parameters]
        except Exception as exc:
            if verbose:
                print(f"[warn] posterior sampling failed ({exc})")
            continue
        try:
            ref = reference_posterior_samples(
                task,
                x_obs,
                n_samples=n_reference,
                seed=seed + i,
                n_parameters=n_parameters,
                n_data=n_data,
            )
        except Exception as exc:
            if verbose:
                print(f"[warn] reference posterior unavailable ({exc})")
            continue
        try:
            acc = c2st(approx, ref, seed=seed + i)
        except Exception as exc:
            if verbose:
                print(f"[warn] c2st failed ({exc})")
            continue
        acc = float(_as_array(acc) if not isinstance(acc, float) else acc)
        accuracies.append(acc)

    if not accuracies:
        return {"c2st_mean": None, "c2st_std": None, "n_targets": 0, "per_target": []}
    arr = np.asarray(accuracies, dtype="float64")
    return {
        "c2st_mean": float(arr.mean()),
        "c2st_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "c2st_median": float(np.median(arr)),
        "n_targets": int(arr.size),
        "per_target": [float(v) for v in arr],
    }


# --------------------------------------------------------------------------------------
# experiment 1: Fig. 4 benchmark (Simformer, mask variants)
# --------------------------------------------------------------------------------------
def run_benchmark(
    *,
    tasks: Sequence[str] = BENCHMARK_TASKS,
    budgets: Sequence[int] = DEFAULT_BUDGETS,
    variants: Sequence[str] = MASK_VARIANTS,
    seed: int = 0,
    n_targets: int = DEFAULT_N_TARGETS,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    n_steps: int = 500,
    max_steps: Optional[int] = None,
    device: str = "cpu",
    outdir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    core = _core()
    results: Dict[str, Any] = {}
    for task_name in tasks:
        results[task_name] = {}
        task = None
        for budget in budgets:
            joint = None
            for variant in variants:
                key = str(int(budget))
                results[task_name].setdefault(key, {})
                t0 = time.time()
                if verbose:
                    print(f"[benchmark] task={task_name} budget={budget} variant={variant}")
                try:
                    bundle = train_simformer(
                        task_name,
                        budget,
                        variant=variant,
                        seed=seed,
                        max_steps=max_steps,
                        device=device,
                        task=task,
                        joint=joint,
                        verbose=False,
                        core=core,
                    )
                    task = bundle["task"]
                    joint = make_joint_dataset(task, budget, seed=seed)
                    stats = evaluate_posterior_c2st(
                        bundle,
                        n_targets=n_targets,
                        n_samples=n_samples,
                        n_reference=n_reference,
                        n_steps=n_steps,
                        seed=seed,
                        verbose=False,
                    )
                    stats["budget"] = int(budget)
                    stats["variant"] = variant
                    stats["elapsed_seconds"] = round(time.time() - t0, 2)
                    stats["chance_level"] = 0.5
                    stats["n_parameters"] = bundle["n_parameters"]
                    stats["n_data"] = bundle["n_data"]
                    results[task_name][key][variant] = stats
                    if verbose:
                        print(
                            f"    -> C2ST {stats['c2st_mean']} "
                            f"(+- {stats['c2st_std']}) in {stats['elapsed_seconds']}s"
                        )
                    if outdir is not None:
                        _save_checkpoint(bundle, Path(outdir), task_name, budget, variant)
                except Exception as exc:
                    results[task_name][key][variant] = {"error": repr(exc)}
                    if verbose:
                        print(f"    -> FAILED: {exc}")
    return results


def _save_checkpoint(bundle: Dict[str, Any], outdir: Path, task_name: str, budget: int, variant: str) -> None:
    ckpt_dir = outdir / "checkpoints" / f"{task_name}_{budget}_{variant}"
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer = bundle.get("trainer")
        if trainer is not None and hasattr(trainer, "save"):
            trainer.save(ckpt_dir / "model.pt")
        meta = {
            "task": task_name,
            "budget": int(budget),
            "mask_variant": variant,
            "n_parameters": bundle.get("n_parameters"),
            "n_data": bundle.get("n_data"),
            "max_steps": bundle.get("max_steps"),
            "history": _truncate_history(bundle.get("history", {})),
        }
        (ckpt_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass


def _truncate_history(history: Any) -> Any:
    if not isinstance(history, dict):
        return history
    out = {}
    for key, value in history.items():
        if isinstance(value, list) and len(value) > 64:
            out[key] = value[-64:]
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------------------
# experiment 2: baselines (NPE / NLE / NRE / NPSE)
# --------------------------------------------------------------------------------------
def run_baselines(
    *,
    tasks: Sequence[str] = BENCHMARK_TASKS,
    budget: int = 10000,
    methods: Sequence[str] = BASELINES,
    seed: int = 0,
    n_targets: int = DEFAULT_N_TARGETS,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run NPE/NLE/NRE (sbi) and NPSE for the C2ST comparison of Fig. 4."""
    core = _core()
    baselines_mod = core.get("baselines")
    results: Dict[str, Any] = {}
    if baselines_mod is None:
        return {"error": "simformer.baselines is not importable"}

    for task_name in tasks:
        results[task_name] = {}
        task = build_task_instance(task_name, seed)
        joint = make_joint_dataset(task, budget, seed=seed)
        n_parameters, n_data = task_dims(task, joint)
        for method in methods:
            t0 = time.time()
            if verbose:
                print(f"[baselines] task={task_name} method={method} budget={budget}")
            try:
                baseline = build_baseline(
                    method, task, n_simulations=budget, seed=seed, verbose=False, core=core
                )
                stats = evaluate_baseline_c2st(
                    baseline,
                    task,
                    n_parameters=n_parameters,
                    n_data=n_data,
                    n_targets=n_targets,
                    n_samples=n_samples,
                    n_reference=n_reference,
                    seed=seed,
                    verbose=False,
                )
                stats["elapsed_seconds"] = round(time.time() - t0, 2)
                stats["method"] = method
                stats["budget"] = int(budget)
                stats["chance_level"] = 0.5
                results[task_name][method] = stats
                if verbose:
                    print(f"    -> C2ST {stats['c2st_mean']}")
            except Exception as exc:
                results[task_name][method] = {"error": repr(exc)}
                if verbose:
                    print(f"    -> FAILED: {exc}")
    return results


def build_baseline(method: str, task: Any, *, n_simulations: int, seed: int = 0, verbose: bool = False, core: Optional[Dict[str, Any]] = None) -> Any:
    """Instantiate a baseline inference method through ``simformer.baselines``."""
    core = core or _core()
    mod = core.get("baselines")
    if mod is None:
        raise ImportError("simformer.baselines is not importable")
    if hasattr(mod, "build_baseline"):
        attempts = [
            lambda: mod.build_baseline(method, task, n_simulations=n_simulations, seed=seed),
            lambda: mod.build_baseline(method, task, n_simulations=n_simulations),
            lambda: mod.build_baseline(method, task, budget=n_simulations, seed=seed),
            lambda: mod.build_baseline(method, task),
        ]
        return _call_with_fallbacks(mod.build_baseline, attempts, f"build_baseline({method})")
    raise RuntimeError("simformer.baselines does not expose build_baseline")


def evaluate_baseline_c2st(
    baseline: Any,
    task: Any,
    *,
    n_parameters: int,
    n_data: int,
    n_targets: int = DEFAULT_N_TARGETS,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """C2ST of a baseline's posterior against the ground-truth posterior."""
    import numpy as np

    rng = _default_rng(seed)
    joint = make_joint_dataset(task, max(n_targets, 1), seed=seed + 999)
    joint = np.atleast_2d(np.asarray(joint, dtype="float32"))[:n_targets]
    accuracies: List[float] = []
    for i in range(joint.shape[0]):
        row = joint[i]
        x_obs = row[n_parameters:]
        try:
            approx = _call_with_fallbacks(
                _baseline_posterior_fn(baseline),
                [
                    lambda: _baseline_posterior_fn(baseline)(
                        x_obs, n_samples=n_samples, seed=seed + i
                    ),
                    lambda: _baseline_posterior_fn(baseline)(x_obs, n_samples=n_samples),
                    lambda: _baseline_posterior_fn(baseline)(x_obs),
                ],
                "baseline posterior sampling",
            )
            approx = np.atleast_2d(np.asarray(_as_array(approx), dtype="float32"))
            ref = reference_posterior_samples(
                task,
                x_obs,
                n_samples=n_reference,
                seed=seed + i,
                n_parameters=n_parameters,
                n_data=n_data,
            )
            accuracies.append(float(c2st(approx, ref, seed=seed + i)))
        except Exception as exc:
            if verbose:
                print(f"[warn] baseline eval failed: {exc}")
            continue
    if not accuracies:
        return {"c2st_mean": None, "c2st_std": None, "n_targets": 0, "per_target": []}
    arr = np.asarray(accuracies, dtype="float64")
    return {
        "c2st_mean": float(arr.mean()),
        "c2st_std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "n_targets": int(arr.size),
        "per_target": [float(v) for v in arr],
    }


def _baseline_posterior_fn(baseline: Any) -> Callable:
    for name in ("sample_posterior", "posterior", "sample", "sample_posterior_samples"):
        fn = getattr(baseline, name, None)
        if callable(fn):
            return fn
    if callable(baseline):
        return baseline
    raise TypeError("baseline exposes no posterior sampling method")


# --------------------------------------------------------------------------------------
# experiment 3: arbitrary conditionals vs MCMC (Sec. 4.1)
# --------------------------------------------------------------------------------------
def run_arbitrary_conditionals(
    *,
    tasks: Sequence[str] = ARBITRARY_TASKS,
    budget: int = 10000,
    variant: str = "directed",
    n_targets: int = 100,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    n_steps: int = 500,
    max_steps: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """C2ST of arbitrary conditionals ``p(z_latent | z_observed)`` vs MCMC references."""
    import numpy as np

    core = _core()
    sampling_mod = core["sampling"]
    results: Dict[str, Any] = {}

    for task_name in tasks:
        if verbose:
            print(f"[arbitrary] task={task_name}")
        entry: Dict[str, Any] = {"per_target": [], "masks": []}
        try:
            task = build_task_instance(task_name, seed)
            joint = make_joint_dataset(task, budget, seed=seed)
            n_parameters, n_data = task_dims(task, joint)
            joint_dim = n_parameters + n_data
            bundle = train_simformer(
                task_name,
                budget,
                variant=variant,
                seed=seed,
                max_steps=max_steps,
                device=device,
                task=task,
                joint=joint,
                core=core,
            )
            sampler = make_sampler(bundle, n_sampling_steps=n_steps, seed=seed)

            # random conditional targets (100 in the paper)
            masks = _random_condition_masks(joint_dim, n_targets, n_parameters, seed=seed)
            obs_joint = np.atleast_2d(np.asarray(joint, dtype="float32"))
            mcmc = core.get("mcmc")

            for k, mask in enumerate(masks):
                row = obs_joint[k % obs_joint.shape[0]]
                latent_idx = np.flatnonzero(np.asarray(mask, dtype="float64") < 0.5)
                try:
                    approx = conditional_samples(
                        sampler, mask, row, n_samples=n_samples, n_steps=n_steps, seed=seed + k
                    )
                    approx = approx[:, latent_idx] if latent_idx.size else approx
                except Exception as exc:
                    if verbose:
                        print(f"  [warn] sampling failed for target {k}: {exc}")
                    continue
                try:
                    ref = mcmc.sample_reference(
                        task,
                        np.asarray(mask, dtype="float64"),
                        np.asarray(row, dtype="float64"),
                        n_samples=n_reference,
                        seed=seed + k,
                        return_full=False,
                    )
                    ref = np.atleast_2d(np.asarray(_as_array(ref), dtype="float32"))
                except Exception as exc:
                    if verbose:
                        print(f"  [warn] MCMC reference failed for target {k}: {exc}")
                    continue
                try:
                    acc = float(c2st(approx, ref, seed=seed + k))
                except Exception:
                    continue
                entry["per_target"].append(acc)
                entry["masks"].append([float(v) for v in np.asarray(mask).ravel()])

            per_target = np.asarray(entry["per_target"], dtype="float64")
            if per_target.size:
                entry.update(
                    {
                        "c2st_mean": float(per_target.mean()),
                        "c2st_std": float(per_target.std(ddof=1)) if per_target.size > 1 else 0.0,
                        "c2st_median": float(np.median(per_target)),
                        "n_targets_evaluated": int(per_target.size),
                        "n_parameters": n_parameters,
                        "n_data": n_data,
                        "budget": int(budget),
                        "variant": variant,
                    }
                )
            else:
                entry["c2st_mean"] = None
        except Exception as exc:
            entry = {"error": repr(exc)}
        results[task_name] = entry
        if verbose:
            print(f"    -> C2ST {entry.get('c2st_mean')}")
    return results


def _random_condition_masks(joint_dim: int, n_targets: int, n_parameters: int, seed: int = 0):
    """Random conditional targets, matching the Sec. 4.1 arbitrary-conditional protocol."""
    import numpy as np

    sampling_mod = _core()["sampling"]
    rng = _default_rng(seed)
    fn = getattr(sampling_mod, "random_conditional_targets", None)
    if callable(fn):
        for kwargs in (
            dict(seed=seed, n_targets=n_targets),
            dict(n_targets=n_targets),
        ):
            try:
                out = fn(joint_dim, **kwargs)
                out = np.atleast_2d(np.asarray(out, dtype="float64"))
                if out.shape[0] >= n_targets:
                    return out[:n_targets]
            except Exception:
                continue
    return (rng.random((n_targets, joint_dim)) < rng.choice([0.3, 0.7])).astype("float64")


# --------------------------------------------------------------------------------------
# experiment 4: reverse-SDE step ablation (Fig. A7)
# --------------------------------------------------------------------------------------
def run_reverse_sde_ablation(
    *,
    task_name: str = "two_moons",
    budget: int = 10000,
    variant: str = "directed",
    step_counts: Sequence[int] = (1, 2, 5, 10, 20, 50, 100, 200, 500),
    n_targets: int = DEFAULT_N_TARGETS,
    n_samples: int = DEFAULT_N_EVAL_SAMPLES,
    n_reference: int = DEFAULT_N_REFERENCE,
    seed: int = 0,
    max_steps: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """C2ST as a function of the number of reverse-SDE evaluation steps (Appendix A3.1)."""
    core = _core()
    if verbose:
        print(f"[reverse_sde] task={task_name}")
    task = build_task_instance(task_name, seed)
    joint = make_joint_dataset(task, budget, seed=seed)
    bundle = train_simformer(
        task_name,
        budget,
        variant=variant,
        seed=seed,
        max_steps=max_steps,
        device=device,
        task=task,
        joint=joint,
        core=core,
    )
    results: Dict[str, Any] = {"task": task_name, "budget": int(budget), "variant": variant, "steps": {}}
    for n_steps in step_counts:
        sampler = make_sampler(bundle, n_sampling_steps=int(n_steps), seed=seed)
        try:
            stats = evaluate_posterior_c2st(
                bundle,
                n_targets=n_targets,
                n_samples=n_samples,
                n_reference=n_reference,
                n_steps=int(n_steps),
                seed=seed,
                sampler=sampler,
                verbose=False,
            )
        except Exception as exc:
            stats = {"error": repr(exc)}
        results["steps"][str(int(n_steps))] = stats
        if verbose:
            print(f"  steps={n_steps:>4} -> C2ST {stats.get('c2st_mean')}")
    return results


# --------------------------------------------------------------------------------------
# experiment 5: scientific simulators (Sec. 4.2-4.4)
# --------------------------------------------------------------------------------------
def run_scientific(
    *,
    tasks: Sequence[str] = SCIENTIFIC_TASKS,
    budget: int = 10000,
    variant: str = "directed",
    n_samples: int = 200,
    n_steps: int = 100,
    max_steps: Optional[int] = None,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Light-weight sanity diagnostics for the scientific simulators.

    For each task we train a Simformer, draw the posterior for one ground-truth
    observation and report:

    * ``coverage``: fraction of the ground-truth parameter vector's coordinates inside
      the central 90% posterior interval (the paper inspects posterior coverage
      qualitatively in Sec. 4.2-4.4);
    * ``predictive_rmse``: error of the mean posterior-predictive data reconstruction
      against the observed data, i.e. whether the posterior predictive stays realistic.
    """
    import numpy as np

    core = _core()
    results: Dict[str, Any] = {}
    for task_name in tasks:
        if verbose:
            print(f"[scientific] task={task_name}")
        entry: Dict[str, Any] = {}
        try:
            task = build_task_instance(task_name, seed)
            joint = make_joint_dataset(task, budget, seed=seed)
            joint = np.atleast_2d(np.asarray(joint, dtype="float32"))
            n_parameters, n_data = task_dims(task, joint)
            bundle = train_simformer(
                task_name,
                budget,
                variant=variant,
                seed=seed,
                max_steps=max_steps,
                device=device,
                task=task,
                joint=joint,
                core=core,
            )
            sampler = make_sampler(bundle, n_sampling_steps=n_steps, seed=seed)
            row = joint[0]
            theta_true = row[:n_parameters]
            mask = posterior_condition_mask(n_parameters, n_data)
            try:
                samples = conditional_samples(
                    sampler, mask, row, n_samples=n_samples, n_steps=n_steps, seed=seed
                )
                theta_samples = samples[:, :n_parameters]
                lo = np.percentile(theta_samples, 5, axis=0)
                hi = np.percentile(theta_samples, 95, axis=0)
                inside = (theta_true >= lo) & (theta_true <= hi)
                entry["coverage"] = float(np.mean(inside))
                entry["posterior_mean_error"] = float(
                    np.mean(np.abs(theta_samples.mean(axis=0) - theta_true))
                )
                # posterior predictive: push posterior samples through the simulator
                try:
                    theta_rep = np.repeat(np.asarray(theta_samples[: min(32, len(theta_samples))], dtype="float64"), 1, axis=0)
                    x_pred = task.simulate(theta_rep, rng=_default_rng(seed + 7))
                    x_pred = np.atleast_2d(np.asarray(_as_array(x_pred), dtype="float64"))
                    x_obs = np.atleast_1d(np.asarray(row[n_parameters:], dtype="float64"))
                    if x_pred.shape[-1] == x_obs.size:
                        entry["predictive_rmse"] = float(
                            np.sqrt(np.mean((x_pred.mean(axis=0) - x_obs) ** 2))
                        )
                    else:  # summary-statistics tasks (e.g. Hodgkin-Huxley)
                        entry["predictive_rmse"] = None
                        entry["predictive_dim_mismatch"] = True
                except Exception as exc:
                    entry["predictive_error"] = repr(exc)
            except Exception as exc:
                entry["error"] = repr(exc)
            entry["n_parameters"] = n_parameters
            entry["n_data"] = n_data
            entry["budget"] = int(budget)
        except Exception as exc:
            entry = {"error": repr(exc)}
        results[task_name] = entry
    return results


# --------------------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------------------
def summarize_benchmark(benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the nested benchmark results into a table keyed by ``task/budget/variant``."""
    table: Dict[str, Dict[str, Any]] = {}
    for task_name, budgets in (benchmark or {}).items():
        if not isinstance(budgets, dict):
            continue
        for budget, variants in budgets.items():
            if not isinstance(variants, dict):
                continue
            for variant, stats in variants.items():
                if not isinstance(stats, dict):
                    continue
                table[f"{task_name}/{budget}/{variant}"] = {
                    "c2st_mean": stats.get("c2st_mean"),
                    "c2st_std": stats.get("c2st_std"),
                    "n_targets": stats.get("n_targets"),
                    "error": stats.get("error"),
                }
    return table


def write_figure(benchmark: Dict[str, Any], outdir: Path) -> Optional[str]:
    """Render a Fig. 4-style C2ST-vs-budget plot when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    try:
        tasks = [t for t, v in (benchmark or {}).items() if isinstance(v, dict) and v]
        if not tasks:
            return None
        fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.4), squeeze=False)
        for ax, task_name in zip(axes[0], tasks):
            budgets = sorted(int(b) for b in benchmark[task_name].keys())
            for variant in MASK_VARIANTS:
                xs, ys, es = [], [], []
                for b in budgets:
                    stats = benchmark[task_name].get(str(b), {}).get(variant)
                    if isinstance(stats, dict) and stats.get("c2st_mean") is not None:
                        xs.append(b)
                        ys.append(stats["c2st_mean"])
                        es.append(stats.get("c2st_std") or 0.0)
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=variant)
            ax.axhline(0.5, ls="--", lw=1, color="grey")
            ax.set_xscale("log")
            ax.set_title(task_name)
            ax.set_xlabel("simulations")
            ax.set_ylabel("C2ST accuracy")
            ax.set_ylim(0.45, 1.02)
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = outdir / "fig4_c2st.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return str(path)
    except Exception:
        return None


def print_table(table: Dict[str, Dict[str, Any]]) -> None:
    if not table:
        return
    width = max(len(k) for k in table)
    print("\n" + "=" * (width + 26))
    print(f"{'configuration'.ljust(width)}  C2ST (mean +- std)")
    print("-" * (width + 26))
    for key, stats in table.items():
        mean = stats.get("c2st_mean")
        std = stats.get("c2st_std") or 0.0
        if mean is None:
            value = f"n/a ({stats.get('error', 'failed')})"
        else:
            value = f"{mean:.4f} +- {std:.4f}"
        print(f"{key.ljust(width)}  {value}")
    print("=" * (width + 26) + "\n")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the Simformer paper experiments (Sec. 4.1-4.4)."
    )
    parser.add_argument(
        "--experiments",
        default="benchmark",
        help=(
            "comma separated subset of {all, benchmark, baselines, arbitrary, reverse_sde, "
            "scientific}"
        ),
    )
    parser.add_argument("--tasks", default="", help="comma separated task names (empty = defaults)")
    parser.add_argument("--budgets", default="", help="comma separated simulation budgets")
    parser.add_argument("--variants", default="dense,undirected,directed", help="attention-mask variants")
    parser.add_argument("--baselines", default=",".join(BASELINES), help="comma separated baselines")
    parser.add_argument("--budget", type=int, default=10000, help="single budget for the non-benchmark experiments")
    parser.add_argument("--max-steps", type=int, default=None, help="override training steps")
    parser.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS, help="posterior targets for C2ST")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_EVAL_SAMPLES, help="samples per conditional")
    parser.add_argument("--n-reference", type=int, default=DEFAULT_N_REFERENCE, help="MCMC reference samples")
    parser.add_argument("--n-steps", type=int, default=500, help="reverse-SDE evaluation steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outdir", default="runs/experiments")
    parser.add_argument("--no-figure", action="store_true", help="skip the Fig. 4 matplotlib rendering")
    parser.add_argument("--quick", action="store_true", help="tiny smoke-test configuration")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _csv(value: str) -> List[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_experiments:
        print("experiments:", ", ".join(EXPERIMENTS))
        print("benchmark tasks:", ", ".join(BENCHMARK_TASKS))
        print("arbitrary-conditional tasks:", ", ".join(ARBITRARY_TASKS))
        print("scientific tasks:", ", ".join(SCIENTIFIC_TASKS))
        print("mask variants:", ", ".join(MASK_VARIANTS))
        print("baselines:", ", ".join(BASELINES))
        return 0

    requested = _csv(args.experiments)
    if "all" in requested or not requested:
        requested = list(EXPERIMENTS)
    unknown = [e for e in requested if e not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiments {unknown}; choose from {EXPERIMENTS}")

    variants = _csv(args.variants) or list(MASK_VARIANTS)
    budgets = [int(b) for b in _csv(args.budgets)] or list(DEFAULT_BUDGETS)
    verbose = not args.quiet

    n_targets = args.n_targets
    n_samples = args.n_samples
    n_reference = args.n_reference
    max_steps = args.max_steps
    budgets_used = budgets
    arb_targets = max(n_targets, 100)
    reverse_steps = (1, 2, 5, 10, 20, 50, 100, 200, 500)
    if args.quick:
        budgets_used = [min(budgets)]
        n_targets = min(n_targets, 3)
        n_samples = min(n_samples, 50)
        n_reference = min(n_reference, 50)
        max_steps = max_steps or 30
        arb_targets = 3
        reverse_steps = (1, 5, 50)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    core = _core()
    if core.get("_errors") and verbose:
        print(f"[info] optional modules unavailable: {core['_errors']}")

    results: Dict[str, Any] = {
        "meta": {
            "experiments": requested,
            "tasks_benchmark": BENCHMARK_TASKS,
            "budgets": budgets_used,
            "variants": variants,
            "seed": args.seed,
            "quick": bool(args.quick),
            "max_steps": max_steps,
            "n_targets": n_targets,
            "n_samples": n_samples,
            "n_reference": n_reference,
            "n_steps": args.n_steps,
            "n_parameters": None,
        },
        "benchmark": {},
        "baselines": {},
        "arbitrary": {},
        "reverse_sde": {},
        "scientific": {},
    }
    started = time.time()

    if "benchmark" in requested:
        tasks_bench = _csv(args.tasks) or list(BENCHMARK_TASKS)
        results["benchmark"] = run_benchmark(
            tasks=tasks_bench,
            budgets=budgets_used,
            variants=variants,
            seed=args.seed,
            n_targets=n_targets,
            n_samples=n_samples,
            n_reference=n_reference,
            n_steps=args.n_steps,
            max_steps=max_steps,
            device=args.device,
            outdir=outdir,
            verbose=verbose,
        )
        table = summarize_benchmark(results["benchmark"])
        results["benchmark_table"] = table
        print_table(table)
        if not args.no_figure:
            fig = write_figure(results["benchmark"], outdir)
            if fig:
                results["figure"] = fig
                if verbose:
                    print(f"[info] wrote figure {fig}")

    if "baselines" in requested:
        tasks_base = _csv(args.tasks) or list(BENCHMARK_TASKS)
        results["baselines"] = run_baselines(
            tasks=tasks_base,
            budget=budgets_used[0],
            methods=_csv(args.baselines) or list(BASELINES),
            seed=args.seed,
            n_targets=n_targets,
            n_samples=n_samples,
            n_reference=n_reference,
            verbose=verbose,
        )

    if "arbitrary" in requested:
        tasks_arb = _csv(args.tasks) or list(ARBITRARY_TASKS)
        results["arbitrary"] = run_arbitrary_conditionals(
            tasks=tasks_arb,
            budget=budgets_used[0],
            variant=variants[-1],
            n_targets=arb_targets,
            n_samples=n_samples,
            n_reference=n_reference,
            n_steps=args.n_steps,
            max_steps=max_steps,
            seed=args.seed,
            device=args.device,
            verbose=verbose,
        )

    if "reverse_sde" in requested:
        results["reverse_sde"] = run_reverse_sde_ablation(
            task_name=(_csv(args.tasks) or ["two_moons"])[0],
            budget=budgets_used[0],
            variant=variants[-1],
            step_counts=reverse_steps,
            n_targets=n_targets,
            n_samples=n_samples,
            n_reference=n_reference,
            seed=args.seed,
            max_steps=max_steps,
            device=args.device,
            verbose=verbose,
        )

    if "scientific" in requested:
        tasks_sci = _csv(args.tasks) or list(SCIENTIFIC_TASKS)
        results["scientific"] = run_scientific(
            tasks=tasks_sci,
            budget=budgets_used[0],
            variant=variants[-1],
            n_samples=min(n_samples, 200),
            n_steps=min(args.n_steps, 100),
            max_steps=max_steps,
            seed=args.seed,
            device=args.device,
            verbose=verbose,
        )

    results["meta"]["elapsed_seconds"] = round(time.time() - started, 2)
    results["meta"]["core_errors"] = core.get("_errors", {})

    out_path = outdir / "results.json"
    try:
        out_path.write_text(json.dumps(results, indent=2, default=str))
        if verbose:
            print(f"[info] wrote {out_path}")
    except Exception as exc:
        print(f"[warn] could not write results.json: {exc}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
