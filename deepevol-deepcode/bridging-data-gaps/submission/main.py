#!/usr/bin/env python
"""Central entry point for the DPMs-ANT reproduction.

This module is a thin, dependency-light *dispatcher* on top of the building
blocks of the code base.  It resolves the configuration (``configs/default.yaml``
+ ``configs/per_task.yaml`` + ``configs/classifier.yaml``), applies CLI
overrides, then forwards to the appropriate driver:

===========================  ==================================================
command                      delegate
===========================  ==================================================
``train_classifier``         ``scripts/train_classifier.py``
``train_ant``                ``scripts/train_ant.py``
``sample``                   ``scripts/sample.py``
``evaluate``                 ``dpm_ant.evaluation.evaluate``
``prepare_data``             ``dpm_ant.data.prepare_data``
``toy``                      ``dpm_ant.toy.toy_2d``
``ablation``                 ``scripts/run_ablation.py``
``all``                      classifier -> ANT -> sample -> evaluate (built-in)
``list``                     print the task registry
===========================  ==================================================

Paper defaults (Section 5.2 / Algorithm 1 / Appendix A.2 / addendum) wired here:

* diffusion: ``T = 1000``, linear beta schedule, DDIM ``eta = 0`` for evaluation
* adaptor: ``c = 4, d = 8`` (DDPM) and ``c = 2, d = 8`` (LDM), zero-initialised,
  inserted into the U-Net *shift* module only; the per-task ``C`` overrides of
  the addendum (``C = 8`` / ``C = 16``) are read from ``configs/per_task.yaml``
* ANT: ``gamma`` per task (3 - 15), ``omega = 0.02``, ``J = 10``, batch size
  ``40``, Adam with lr ``5e-5`` (DDPM) / ``1e-5`` - ``2e-5`` (LDM),
  160 - 500 training iterations depending on the task
* classifier: binary source/target head, Adam, lr ``1e-4``, batch ``64``,
  ``300`` iterations on noised source+target images
* metrics: Intra-LPIPS (1,000 generated images, higher better) and FID against
  the larger target sets (Babies 2.7k, Sunglasses 2.5k, lower better)

Usage examples::

    python main.py list
    python main.py --print-config --task ffhq_sunglasses
    python main.py train_classifier --task ffhq_sunglasses --backbone ddpm
    python main.py train_ant --task ffhq_sunglasses --backbone ddpm --dry-run
    python main.py sample --task ffhq_sunglasses --num-samples 1000 --num-steps 100
    python main.py evaluate --generated-dir outputs/samples/ffhq_sunglasses_ddpm \
        --reference-dir data/targets/sunglasses
    python main.py toy
    python main.py ablation --ablation-run everything
    python main.py all --tasks ffhq_sunglasses --backbones ddpm --smoke
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_CONFIG = os.path.join(_ROOT, "configs", "default.yaml")
PER_TASK_CONFIG = os.path.join(_ROOT, "configs", "per_task.yaml")
CLASSIFIER_CONFIG = os.path.join(_ROOT, "configs", "classifier.yaml")

LOGGER = logging.getLogger("dpm_ant.main")

COMMANDS = [
    "train_classifier",
    "train_ant",
    "sample",
    "evaluate",
    "prepare_data",
    "toy",
    "ablation",
    "all",
    "list",
]

#: default task registry (source -> target) with the backbone used in the paper
DEFAULT_TASKS: Dict[str, Dict[str, str]] = {
    "ffhq_babies": {"source": "ffhq", "target": "babies", "backbone": "ddpm"},
    "ffhq_sunglasses": {"source": "ffhq", "target": "sunglasses", "backbone": "ddpm"},
    "ffhq_raphael": {"source": "ffhq", "target": "raphael", "backbone": "ddpm"},
    "church_haunted_houses": {
        "source": "lsun_church",
        "target": "haunted_houses",
        "backbone": "ddpm",
    },
    "church_landscape_drawings": {
        "source": "lsun_church",
        "target": "landscape_drawings",
        "backbone": "ddpm",
    },
    "ffhq_babies_ldm": {"source": "ffhq", "target": "babies", "backbone": "ldm"},
    "ffhq_sunglasses_ldm": {"source": "ffhq", "target": "sunglasses", "backbone": "ldm"},
    "ffhq_raphael_ldm": {"source": "ffhq", "target": "raphael", "backbone": "ldm"},
    "church_haunted_houses_ldm": {
        "source": "lsun_church",
        "target": "haunted_houses",
        "backbone": "ldm",
    },
    "church_landscape_drawings_ldm": {
        "source": "lsun_church",
        "target": "landscape_drawings",
        "backbone": "ldm",
    },
}

#: addendum ("Hyperparameters for Table 3") - per-task override table.
#: Used when ``configs/per_task.yaml`` does not provide the entry.
ADDENDUM_TASK_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "ffhq_babies": {"lr": 5e-6, "C": 8, "omega": 0.02, "J": 10, "gamma": 3, "iterations": 160},
    "ffhq_sunglasses": {"lr": 5e-5, "C": 8, "omega": 0.02, "J": 10, "gamma": 15, "iterations": 200},
    "ffhq_raphael": {"lr": 5e-5, "C": 8, "omega": 0.02, "J": 10, "gamma": 10, "iterations": 500},
    "church_haunted_houses": {
        "lr": 5e-5,
        "C": 8,
        "omega": 0.02,
        "J": 10,
        "gamma": 10,
        "iterations": 320,
    },
    "church_landscape_drawings": {
        "lr": 5e-5,
        "C": 16,
        "omega": 0.02,
        "J": 10,
        "gamma": 10,
        "iterations": 500,
    },
    "ffhq_babies_ldm": {"lr": 5e-6, "C": 16, "omega": 0.02, "J": 10, "gamma": 5, "iterations": 320},
    "ffhq_sunglasses_ldm": {"lr": 1e-5, "C": 8, "omega": 0.02, "J": 10, "gamma": 5, "iterations": 280},
    "ffhq_raphael_ldm": {"lr": 1e-5, "C": 8, "omega": 0.02, "J": 10, "gamma": 5, "iterations": 320},
    "church_haunted_houses_ldm": {
        "lr": 2e-5,
        "C": 8,
        "omega": 0.02,
        "J": 10,
        "gamma": 5,
        "iterations": 500,
    },
    "church_landscape_drawings_ldm": {
        "lr": 2e-5,
        "C": 8,
        "omega": 0.02,
        "J": 10,
        "gamma": 5,
        "iterations": 500,
    },
}


# --------------------------------------------------------------------------- #
# Small yaml / dict helpers (avoid a hard dependency on PyYAML at import time)
# --------------------------------------------------------------------------- #
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Safely read a YAML config file; return ``{}`` when unavailable."""
    if not path:
        return {}
    if not os.path.exists(path):
        LOGGER.debug("config not found: %s", path)
        return {}
    try:
        import yaml  # noqa: WPS433 (lazy import)
    except Exception:  # pragma: no cover - pyyaml is in requirements
        LOGGER.warning("PyYAML is not installed; ignoring %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    config_path: Optional[str] = None,
    per_task_path: Optional[str] = None,
    classifier_path: Optional[str] = None,
    task: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    flatten_defaults: bool = True,
) -> Dict[str, Any]:
    """Merge the global / per-task / classifier configs plus explicit overrides.

    The merged dictionary contains (when the files exist):

    * every key of ``configs/default.yaml``
    * the ``defaults`` block of ``configs/per_task.yaml`` flattened into
      ``ant`` (training hyper-parameters) and ``adaptor`` (adaptor shape)
    * the selected entry of ``tasks[task]`` as ``cfg["task_cfg"]``
    * the ``classifier`` block of ``configs/classifier.yaml``
    """
    cfg: Dict[str, Any] = _load_yaml(config_path or DEFAULT_CONFIG)
    per_task = _load_yaml(per_task_path or PER_TASK_CONFIG)
    classifier_cfg = _load_yaml(classifier_path or CLASSIFIER_CONFIG)

    defaults = per_task.get("defaults") or {}
    if flatten_defaults and defaults:
        ant_block = dict(cfg.get("ant") or {})
        adaptor_block = dict(cfg.get("adaptor") or {})
        for key, value in defaults.items():
            if key in ("adaptor", "diffusion", "sampling"):
                continue
            ant_block.setdefault(key, value)
        _deep_update(cfg.setdefault("adaptor", {}), adaptor_block)
        for key, value in (defaults.get("adaptor") or {}).items():
            cfg["adaptor"].setdefault(key, value)
        cfg["ant"] = ant_block

    # keep the task / sensitivity / ablation registries reachable
    for key in ("tasks", "sensitivity", "classifier_ablation", "ablation_variants"):
        if key in per_task:
            cfg.setdefault(key, per_task[key])

    # make sure every task advertised by the addendum exists
    tasks = cfg.setdefault("tasks", {})
    for name, values in DEFAULT_TASKS.items():
        entry = tasks.setdefault(name, {})
        for key, value in values.items():
            entry.setdefault(key, value)
    for name, values in ADDENDUM_TASK_HYPERPARAMS.items():
        tasks.setdefault(name, {})
        for key, value in values.items():
            tasks[name].setdefault(key, value)

    if task:
        task_cfg = dict(tasks.get(task) or DEFAULT_TASKS.get(task) or {})
        if task_cfg:
            cfg["task_cfg"] = task_cfg
            # per-task values win over the flattened defaults (addendum Table 3)
            ant_block = dict(cfg.get("ant") or {})
            for key in ("gamma", "omega", "J", "iterations", "lr", "batch_size", "norm"):
                if task_cfg.get(key) is not None:
                    ant_block[key] = task_cfg[key]
            cfg["ant"] = ant_block
            for key in ("bottleneck_c", "bottleneck_d", "hidden_dims", "C", "composition"):
                if task_cfg.get(key) is not None:
                    cfg.setdefault("adaptor", {})[key] = task_cfg[key]

    if classifier_cfg:
        merged = dict(cfg.get("classifier") or {})
        _deep_update(merged, classifier_cfg.get("classifier") or {})
        cfg["classifier"] = merged
        _deep_update(cfg.setdefault("models", {}), classifier_cfg.get("models") or {})
        for key in ("validation", "classifier_ablation"):
            if key in classifier_cfg:
                cfg.setdefault(key, classifier_cfg[key])

    if overrides:
        _deep_update(cfg, {k: v for k, v in overrides.items() if v is not None})
    return cfg


def list_tasks(cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the task names declared in the config (falls back to defaults)."""
    if cfg:
        tasks = cfg.get("tasks") or {}
        if tasks:
            return list(tasks.keys())
    return list(DEFAULT_TASKS.keys())


def resolve_task(cfg: Dict[str, Any], task: Optional[str]) -> Dict[str, Any]:
    """Return the per-task dictionary for ``task``."""
    entry: Dict[str, Any] = {}
    if task:
        entry = dict((cfg.get("tasks") or {}).get(task) or {})
        if not entry:
            entry = dict(DEFAULT_TASKS.get(task) or {})
            entry.update(ADDENDUM_TASK_HYPERPARAMS.get(task) or {})
    if cfg.get("task_cfg"):
        entry = {**entry, **cfg["task_cfg"]}
    return entry


def resolve_backbone(
    cfg: Optional[Dict[str, Any]],
    task: Optional[str] = None,
    backbone: Optional[str] = None,
) -> str:
    """Resolve ``"ddpm"`` or ``"ldm"`` for a task (CLI wins over config)."""
    if backbone:
        return str(backbone).lower()
    entry = resolve_task(cfg or {}, task)
    if entry.get("backbone"):
        return str(entry["backbone"]).lower()
    if task and "ldm" in str(task).lower():
        return "ldm"
    return "ddpm"


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (lazily) for reproducibility."""
    import random

    random.seed(seed)
    try:  # pragma: no cover - optional dependency
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


def resolve_device(device: Optional[str] = None) -> str:
    """Return ``"cuda"`` when available (unless explicitly requested otherwise)."""
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- #
# Module loading helpers
# --------------------------------------------------------------------------- #
def _load_module(module_name: str, file_path: str):
    """Import ``module_name`` normally, falling back to a file-based import."""
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - namespace package fallback
        LOGGER.debug("importing %s failed (%s); loading from %s", module_name, exc, file_path)
    if not os.path.exists(file_path):
        raise ImportError(f"cannot import {module_name} (missing {file_path})")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _script_main(script: str):
    """Return the ``main`` callable of ``scripts/<script>.py``."""
    path = os.path.join(_ROOT, "scripts", f"{script}.py")
    module = _load_module(f"scripts.{script}", path)
    fn = getattr(module, "main", None)
    if fn is None:
        raise AttributeError(f"scripts/{script}.py does not define main(argv)")
    return fn


# --------------------------------------------------------------------------- #
# argv translation
# --------------------------------------------------------------------------- #
#: global flags that are meaningful to every delegate
_PASSTHROUGH_FLAGS: List[Tuple[str, str]] = [
    ("task", "--task"),
    ("backbone", "--backbone"),
    ("config", "--config"),
    ("per_task_config", "--per-task-config"),
    ("classifier_config", "--classifier-config"),
    ("device", "--device"),
    ("seed", "--seed"),
    ("out", "--out"),
    ("gamma", "--gamma"),
    ("omega", "--omega"),
    ("J", "--J"),
    ("iterations", "--iterations"),
    ("batch_size", "--batch-size"),
    ("num_samples", "--num-samples"),
    ("num_steps", "--num-steps"),
    ("eta", "--eta"),
    ("method", "--method"),
    ("adaptor_checkpoint", "--adaptor-checkpoint"),
    ("classifier_checkpoint", "--classifier-checkpoint"),
    ("diffusion_checkpoint", "--diffusion-checkpoint"),
    ("target_dir", "--target-dir"),
    ("source_dir", "--source-dir"),
    ("generated_dir", "--generated-dir"),
    ("reference_dir", "--reference-dir"),
    ("report", "--report"),
    ("fid_backend", "--fid-backend"),
    ("variant", "--variant"),
    ("shots", "--shots"),
    ("log_interval", "--log-interval"),
]


def _translate_argv(args: argparse.Namespace, unknown: Sequence[str]) -> List[str]:
    """Convert parsed global flags into the delegate script's CLI form."""
    argv: List[str] = []
    for attr, flag in _PASSTHROUGH_FLAGS:
        value = getattr(args, attr, None)
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if flag in ("--config", "--per-task-config", "--classifier-config"):
            continue  # handled inside the delegates via their own defaults
        argv.extend([flag, str(value)])
    if getattr(args, "split", None):
        argv.extend(["--split", str(args.split)])
    if getattr(args, "dry_run", False):
        argv.append("--dry-run")
    if getattr(args, "verbose", False):
        argv.append("--verbose")
    argv.extend([str(item) for item in unknown])
    return argv


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list(args: argparse.Namespace) -> int:
    """Print the task registry (paper source -> target pairs)."""
    cfg = load_config(args.config, args.per_task_config, args.classifier_config)
    tasks = cfg.get("tasks") or DEFAULT_TASKS
    print(f"{'task':34s} {'backbone':9s} source -> target")
    print("-" * 78)
    for name in tasks:
        entry = resolve_task(cfg, name) or DEFAULT_TASKS.get(name, {})
        source = entry.get("source", "?")
        target = entry.get("target", "?")
        backbone = resolve_backbone(cfg, name, None)
        marker = " *" if args.task and args.task == name else ""
        print(f"{name:34s} {backbone:9s} {source} -> {target}{marker}")
    print("-" * 78)
    print("tasks marked * are selected by --task")

    variants = cfg.get("ablation_variants") or {}
    if variants:
        print("\nAblation variants (Section 5.4, Figure 4):")
        for key in variants:
            print(f"  {key}")

    print("\nPaper reference FID (lower is better):")
    ref = cfg.get("paper_reference") or {}
    for key, value in (ref.get("fid") or {}).items():
        print(f"  {key}: {value}")
    return 0


def cmd_print_config(args: argparse.Namespace) -> int:
    """Dump the fully merged configuration as JSON."""
    cfg = load_config(args.config, args.per_task_config, args.classifier_config, task=args.task)
    if args.backbone:
        cfg["backbone"] = args.backbone
    cfg["resolved_backbone"] = resolve_backbone(cfg, args.task, args.backbone)
    cfg["resolved_device"] = resolve_device(args.device)
    if args.task:
        cfg["task_cfg"] = resolve_task(cfg, args.task)
    print(json.dumps(cfg, indent=2, default=str))
    return 0


def _forward(script: str, args: argparse.Namespace, argv: Sequence[str]) -> int:
    """Delegate to ``scripts/<script>.py`` ``main()``."""
    LOGGER.info("dispatching to scripts/%s.py with argv=%s", script, list(argv))
    fn = _script_main(script)
    return int(fn(list(argv)) or 0)


def cmd_train_classifier(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Fine-tune the binary source/target classifier p_phi (Section 4.1/5.2)."""
    argv = _translate_argv(args, unknown)
    if args.shots is not None:
        argv.extend(["--target-pool", str(args.shots)])
    return _forward("train_classifier", args, argv)


def cmd_train_ant(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Adapt the frozen backbone with zero-init adaptors (Algorithm 1, Eq. 8)."""
    return _forward("train_ant", args, _translate_argv(args, unknown))


def cmd_sample(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Generate images with the adapted reverse process (Eq. 2/3)."""
    return _forward("sample", args, _translate_argv(args, unknown))


def cmd_evaluate(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Compute Intra-LPIPS / FID / efficiency metrics for generated images."""
    path = os.path.join(_ROOT, "dpm_ant", "evaluation", "evaluate.py")
    module = _load_module("dpm_ant.evaluation.evaluate", path)
    argv = _translate_argv(args, unknown)
    if args.task:
        argv.extend(["--split", args.task])
    LOGGER.info("dispatching to dpm_ant.evaluation.evaluate with argv=%s", argv)
    return int(module.main(argv) or 0)


def cmd_prepare_data(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Download / resize / crop the source and target datasets (Section 5.2)."""
    path = os.path.join(_ROOT, "dpm_ant", "data", "prepare_data.py")
    module = _load_module("dpm_ant.data.prepare_data", path)
    return int(module.main(_translate_argv(args, unknown)) or 0)


def cmd_toy(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Run the 2-D Gaussian toy experiment (Section 5.1, Figure 2)."""
    path = os.path.join(_ROOT, "dpm_ant", "toy", "toy_2d.py")
    if not os.path.exists(path):
        LOGGER.error("toy experiment not found at %s", path)
        return 2
    module = _load_module("dpm_ant.toy.toy_2d", path)
    fn = getattr(module, "main", None)
    if fn is None:
        LOGGER.error("dpm_ant/toy/toy_2d.py must define main(argv)")
        return 2
    return int(fn(_translate_argv(args, unknown)) or 0)


def cmd_ablation(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Run the ablation / sensitivity study (Section 5.4/5.5, Tables 5-7)."""
    path = os.path.join(_ROOT, "scripts", "run_ablation.py")
    if not os.path.exists(path):
        LOGGER.error("ablation runner not found at %s", path)
        return 2
    module = _load_module("scripts.run_ablation", path)
    fn = getattr(module, "main", None)
    if fn is None:
        LOGGER.error("scripts/run_ablation.py must define main(argv)")
        return 2
    return int(fn(_translate_argv(args, unknown)) or 0)


# --------------------------------------------------------------------------- #
# `all` pipeline
# --------------------------------------------------------------------------- #
def _pipeline_for_task(
    cfg: Dict[str, Any],
    task: str,
    backbone: str,
    out_root: str,
    *,
    args: argparse.Namespace,
    strict: bool = False,
) -> Dict[str, Any]:
    """Run classifier -> ANT -> sample -> evaluate for one task."""
    record: Dict[str, Any] = {"task": task, "backbone": backbone, "steps": {}}
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0) or 0)
    device = resolve_device(args.device)

    ckpt_dir = os.path.join(out_root, "checkpoints")
    classifier_dir = os.path.join(out_root, "classifier")
    sample_dir = os.path.join(out_root, "samples", f"{task}_{backbone}")
    report_path = os.path.join(out_root, "reports", f"{task}_{backbone}.json")
    for path in (ckpt_dir, classifier_dir, sample_dir, os.path.dirname(report_path)):
        os.makedirs(path, exist_ok=True)

    classifier_ckpt = args.classifier_checkpoint or os.path.join(
        classifier_dir, f"classifier_{task}_{backbone}.pt"
    )
    adaptor_ckpt = args.adaptor_checkpoint or os.path.join(ckpt_dir, f"{task}_{backbone}_ant.pt")

    iterations = args.iterations
    if args.smoke and iterations is None:
        iterations = 2

    common = ["--task", task, "--backbone", backbone, "--device", device, "--seed", str(seed)]
    steps: List[Tuple[str, List[str]]] = []

    if not args.skip_classifier:
        if os.path.exists(classifier_ckpt) and not args.force:
            LOGGER.info("[%s/%s] classifier checkpoint exists, skipping", task, backbone)
            record["steps"]["train_classifier"] = "skipped"
        else:
            cli = list(common) + ["--out", classifier_ckpt]
            if iterations is not None:
                cli += ["--iterations", str(iterations)]
            if args.shots is not None:
                cli += ["--target-pool", str(args.shots)]
            steps.append(("train_classifier", cli))

    if not args.skip_train:
        if os.path.exists(adaptor_ckpt) and not args.force:
            LOGGER.info("[%s/%s] adaptor checkpoint exists, skipping", task, backbone)
            record["steps"]["train_ant"] = "skipped"
        else:
            cli = list(common) + [
                "--classifier-checkpoint",
                classifier_ckpt,
                "--out",
                adaptor_ckpt,
            ]
            if iterations is not None:
                cli += ["--iterations", str(iterations)]
            if args.variant:
                cli += ["--variant", args.variant]
            if backbone == "ldm":
                cli.append("--adaptor-only")
            steps.append(("train_ant", cli))

    if not args.skip_sample:
        num_samples = args.num_samples
        num_steps = args.num_steps
        if args.smoke and num_samples is None:
            num_samples = 4
        if args.smoke and num_steps is None:
            num_steps = 5
        cli = list(common) + [
            "--out",
            sample_dir,
            "--adaptor-checkpoint",
            adaptor_ckpt,
            "--classifier-checkpoint",
            classifier_ckpt,
        ]
        if num_samples is not None:
            cli += ["--num-samples", str(num_samples)]
        if num_steps is not None:
            cli += ["--num-steps", str(num_steps)]
        steps.append(("sample", cli))

    if not args.skip_eval:
        entry = resolve_task(cfg, task)
        reference = args.reference_dir
        if reference is None:
            target = entry.get("target")
            reference = ((cfg.get("data") or {}).get("target_dirs") or {}).get(target)
        cli = [
            "--generated-dir",
            sample_dir,
            "--split",
            task,
            "--report",
            report_path,
            "--device",
            device,
        ]
        if reference:
            cli += ["--reference-dir", str(reference)]
        steps.append(("evaluate", cli))

    for name, cli in steps:
        try:
            if name == "evaluate":
                module = _load_module(
                    "dpm_ant.evaluation.evaluate",
                    os.path.join(_ROOT, "dpm_ant", "evaluation", "evaluate.py"),
                )
                rc = int(module.main(cli) or 0)
            else:
                rc = _forward(name, args, cli)
            record["steps"][name] = "ok" if rc == 0 else f"rc={rc}"
            if rc != 0 and strict:
                break
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.error("[%s/%s] %s failed: %s", task, backbone, name, exc)
            record["steps"][name] = f"error: {exc}"
            if strict:
                break

    return record


def cmd_all(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    """Run the complete pipeline for the requested tasks / backbones."""
    cfg = load_config(args.config, args.per_task_config, args.classifier_config)
    out_root = args.out or (cfg.get("logging") or {}).get("log_dir") or "outputs"
    os.makedirs(out_root, exist_ok=True)

    tasks = args.tasks.split() if args.tasks else (["ffhq_sunglasses"] if args.smoke else list(DEFAULT_TASKS)[:5])
    backbones = args.backbones.split() if args.backbones else [""]
    if args.smoke:
        LOGGER.warning("smoke mode: tiny iteration/sample counts; results are NOT paper numbers")

    records: List[Dict[str, Any]] = []
    started = time.time()
    for task in tasks:
        for backbone in backbones:
            bb = resolve_backbone(cfg, task, backbone or None)
            records.append(
                _pipeline_for_task(cfg, task, bb, out_root, args=args, strict=bool(args.strict))
            )

    summary = {
        "out_root": out_root,
        "tasks": tasks,
        "records": records,
        "elapsed_hours": (time.time() - started) / 3600.0,
    }
    summary_path = os.path.join(out_root, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    LOGGER.info("pipeline summary written to %s", summary_path)
    print(json.dumps(summary, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="DPMs-ANT reproduction entry point "
        "(zero-init adaptors + adversarial-noise similarity-guided few-shot adaptation).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="list",
        choices=COMMANDS,
        help="which driver to run",
    )

    # config
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="global config yaml")
    parser.add_argument("--per-task-config", dest="per_task_config", default=PER_TASK_CONFIG)
    parser.add_argument("--classifier-config", dest="classifier_config", default=CLASSIFIER_CONFIG)

    # task / backbone
    parser.add_argument("--task", default=None, help="task key, e.g. ffhq_sunglasses")
    parser.add_argument("--tasks", default=None, help="space separated task keys (command=all)")
    parser.add_argument("--backbone", default=None, choices=["ddpm", "ldm"])
    parser.add_argument("--backbones", default=None, help="space separated backbones (command=all)")
    parser.add_argument("--variant", default=None, help="ablation variant key")

    # runtime
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default=None, help="output root directory")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help="print the merged configuration and exit",
    )

    # hyper-parameters (paper defaults live in configs/*.yaml and the addendum)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--J", type=int, default=None)
    parser.add_argument("--shots", type=int, default=None, help="target shots (10 or 100)")

    # sampling / evaluation
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=None)
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--method", default=None, choices=["ddim", "ddpm"])
    parser.add_argument("--adaptor-checkpoint", dest="adaptor_checkpoint", default=None)
    parser.add_argument("--classifier-checkpoint", dest="classifier_checkpoint", default=None)
    parser.add_argument("--diffusion-checkpoint", dest="diffusion_checkpoint", default=None)
    parser.add_argument("--source-dir", dest="source_dir", default=None)
    parser.add_argument("--target-dir", dest="target_dir", default=None)
    parser.add_argument("--generated-dir", dest="generated_dir", default=None)
    parser.add_argument("--reference-dir", dest="reference_dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--fid-backend", dest="fid_backend", default=None)
    parser.add_argument("--log-interval", dest="log_interval", type=int, default=None)

    # `all` pipeline controls
    parser.add_argument("--skip-classifier", dest="skip_classifier", action="store_true")
    parser.add_argument("--skip-train", dest="skip_train", action="store_true")
    parser.add_argument("--skip-sample", dest="skip_sample", action="store_true")
    parser.add_argument("--skip-eval", dest="skip_eval", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run even if checkpoints exist")
    parser.add_argument("--strict", action="store_true", help="abort on first failing step")
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end smoke test")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of ``main.py``."""
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
    )

    if args.seed is not None:
        set_seed(int(args.seed))
    else:
        cfg_seed = load_config(args.config, args.per_task_config).get("seed") or 0
        set_seed(int(cfg_seed))

    if args.print_config:
        return cmd_print_config(args)

    command = args.command
    dispatch = {
        "list": lambda: cmd_list(args),
        "train_classifier": lambda: cmd_train_classifier(args, unknown),
        "train_ant": lambda: cmd_train_ant(args, unknown),
        "sample": lambda: cmd_sample(args, unknown),
        "evaluate": lambda: cmd_evaluate(args, unknown),
        "prepare_data": lambda: cmd_prepare_data(args, unknown),
        "toy": lambda: cmd_toy(args, unknown),
        "ablation": lambda: cmd_ablation(args, unknown),
        "all": lambda: cmd_all(args, unknown),
    }
    handler = dispatch.get(command)
    if handler is None:  # pragma: no cover - argparse restricts choices
        parser.error(f"unknown command {command!r}")
    return int(handler() or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
