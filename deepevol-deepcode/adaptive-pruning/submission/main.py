#!/usr/bin/env python
"""Thin command-line entry point for the APT reproduction codebase.

This module is intentionally a *dispatcher*: it contains no algorithms of its
own.  Every sub-command forwards to the implementation that lives in the
``apt`` package (library API) or, for the table/figure reproductions, to the
scripts under ``scripts/``.

Examples
--------
APT training (RoBERTa-base / SST-2, 60% sparsity)::

    python main.py train-apt --config apt/configs/roberta_sst2.yaml

Baseline training (FT / LoRA / LoRA+Prune / Prune+Distill / LoRA+Prune+Distill)::

    python main.py train-baseline --method ft --config apt/configs/roberta_sst2.yaml

Evaluation of a saved checkpoint::

    python main.py eval --model-dir outputs/apt_roberta_sst2 --task sst2

Paper reproductions::

    python main.py table2  --model roberta-base
    python main.py table4  --model roberta-base
    python main.py table7  --model bert-base-uncased
    python main.py table8  --model roberta-base
    python main.py figure3 --model roberta-base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

CONFIG_BASE = "apt/configs"

CONFIG_ALIASES: Dict[str, str] = {
    # model + task -> config file
    "roberta-sst2": "roberta_sst2.yaml",
    "roberta_sst_2": "roberta_sst2.yaml",
    "roberta-mnli": "roberta_mnli.yaml",
    "roberta-squad": "squad.yaml",
    "roberta-squad_v2": "squad.yaml",
    "roberta-squadv2": "squad.yaml",
    "t5-cnndm": "t5_cnndm.yaml",
    "t5-cnn_dailymail": "t5_cnndm.yaml",
    "t5-cnndailymail": "t5_cnndm.yaml",
    "default": "default.yaml",
}

BASELINE_METHODS: List[str] = [
    "ft",
    "lora",
    "mask_tuning",
    "cofi",
    "lora_prune_distill",
]

METHOD_ALIASES: Dict[str, str] = {
    "finetune": "ft",
    "fine_tune": "ft",
    "full": "ft",
    "full_finetuning": "ft",
    "lora": "lora",
    "mask_tuning": "mask_tuning",
    "mask-tuning": "mask_tuning",
    "lora+prune": "mask_tuning",
    "lora_prune": "mask_tuning",
    "retraining_free": "mask_tuning",
    "cofi": "cofi",
    "prune+distill": "cofi",
    "prune_distill": "cofi",
    "lora_prune_distill": "lora_prune_distill",
    "lora+prune+distill": "lora_prune_distill",
    "lora_cofi": "lora_prune_distill",
}

TASK_ALIASES: Dict[str, str] = {
    "sst-2": "sst2",
    "sst_2": "sst2",
    "mnli-mm": "mnli",
    "mnli_matched": "mnli",
    "mnli_mismatched": "mnli",
    "squad2": "squad_v2",
    "squad-v2": "squad_v2",
    "squadv2": "squad_v2",
    "cnn_dailymail": "cnndm",
    "cnn-dailymail": "cnndm",
    "cnn/dm": "cnndm",
    "sts-b": "stsb",
    "stsb": "stsb",
}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _log(msg: str) -> None:
    print(msg, flush=True)


def _version() -> str:
    try:
        from apt import __version__  # noqa: WPS433 (lazy import on purpose)

        return str(__version__)
    except Exception:  # pragma: no cover - package not importable
        return "unknown"


def resolve_config(path: Optional[str], model: Optional[str] = None,
                   task: Optional[str] = None) -> Optional[str]:
    """Resolve a ``--config`` argument into an existing YAML path.

    Accepts either a direct path, a bare filename, a short alias such as
    ``roberta-sst2``, or ``(model, task)`` which are combined into an alias.
    Returns ``None`` when nothing sensible can be found.
    """
    if path:
        candidates = [path, os.path.join(CONFIG_BASE, path)]
        if not path.endswith((".yaml", ".yml")):
            candidates += [
                os.path.join(CONFIG_BASE, path + ".yaml"),
                path + ".yaml",
            ]
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
        alias = path.lower()
        if alias in CONFIG_ALIASES:
            cand = os.path.join(CONFIG_BASE, CONFIG_ALIASES[alias])
            if os.path.isfile(cand):
                return cand
        # fall through: return the raw string so downstream loaders can error
        return path

    if model and task:
        key = "{}-{}".format(model.lower(), task.lower())
        if key in CONFIG_ALIASES:
            cand = os.path.join(CONFIG_BASE, CONFIG_ALIASES[key])
            if os.path.isfile(cand):
                return cand
    return None


def normalize_method(method: str) -> str:
    key = method.strip().lower().replace(" ", "_")
    return METHOD_ALIASES.get(key, key)


def normalize_task(task: str) -> str:
    key = task.strip().lower().replace(" ", "")
    return TASK_ALIASES.get(key, key)


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file, returning ``{}`` when unavailable."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyyaml is required to load config files: %s" % exc)

    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config %s did not contain a mapping" % path)
    return data


def _dump(obj: Any) -> None:
    """Print a result dictionary as JSON when possible."""
    try:
        print(json.dumps(obj, indent=2, default=str, sort_keys=True), flush=True)
    except Exception:  # pragma: no cover
        print(repr(obj), flush=True)


def _require_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for this command. Install with "
            "'pip install -r requirements.txt' (%s)" % exc
        )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_train_apt(args: argparse.Namespace) -> int:
    """Stage 1+2 APT training (Algorithm 1)."""
    config_path = resolve_config(args.config, args.model, args.task)
    merged = load_yaml_config(config_path)
    if args.task:
        merged["task"] = normalize_task(args.task)
    if args.model:
        merged["model_name_or_path"] = args.model
    if args.sparsity is not None:
        merged["target_sparsity"] = float(args.sparsity)
    if args.epochs is not None:
        merged["epochs"] = int(args.epochs)
    if args.distill_epochs is not None:
        merged["distill_epochs"] = int(args.distill_epochs)
    if args.output_dir:
        merged["output_dir"] = args.output_dir
    if args.seed is not None:
        merged["seed"] = int(args.seed)

    _log("[main] APT training  config=%s" % (config_path or "<inline>"))
    if merged:
        _log("[main] hyperparameters: " + json.dumps(merged, sort_keys=True,
                                                     default=str))

    _require_torch()

    # Prefer the script if present (it owns data/model construction), else the
    # library entry point.
    try:
        from scripts import train_apt as train_apt_script  # type: ignore

        summary = train_apt_script.main(merged, argv=None)
    except Exception:
        from apt.training import TrainConfig, train_apt

        config = TrainConfig.from_dict(merged)
        summary = train_apt(config)
    _dump(_summarize(summary))
    return 0


def cmd_train_baseline(args: argparse.Namespace) -> int:
    """Train one of the paper's comparison baselines."""
    method = normalize_method(args.method)
    if method not in BASELINE_METHODS:
        _log("[main] unknown baseline method %r; expected one of %s"
             % (args.method, ", ".join(BASELINE_METHODS)))
        return 2

    config_path = resolve_config(args.config, args.model, args.task)
    merged = load_yaml_config(config_path)
    merged["method"] = method
    if args.task:
        merged["task"] = normalize_task(args.task)
    if args.model:
        merged["model_name_or_path"] = args.model
    if args.sparsity is not None:
        merged["target_sparsity"] = float(args.sparsity)
    if args.epochs is not None:
        merged["epochs"] = int(args.epochs)
    if args.output_dir:
        merged["output_dir"] = args.output_dir
    if args.seed is not None:
        merged["seed"] = int(args.seed)

    _log("[main] baseline training  method=%s  config=%s"
         % (method, config_path or "<inline>"))
    _require_torch()

    summary = _run_baseline(method, merged)
    _dump(_summarize(summary))
    return 0


def _run_baseline(method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    from apt.baselines import get_method

    try:
        trainer_cls = get_method(method)
    except Exception as exc:
        raise RuntimeError(
            "baseline %r is unavailable (%s). In-repo methods are 'ft' and "
            "'lora'; 'mask_tuning'/'cofi' need the external repos (see README)."
            % (method, exc)
        )

    # Every baseline module exposes a train_* convenience function; use it when
    # the class alone is insufficient.
    entry_points = {
        "ft": ("apt.baselines.ft", "train_ft"),
        "lora": ("apt.baselines.lora", "train_lora"),
        "mask_tuning": ("apt.baselines.mask_tuning", "train_mask_tuning"),
        "cofi": ("apt.baselines.cofi", "train_cofi"),
        "lora_prune_distill": (
            "apt.baselines.lora_prune_distill",
            "train_lora_prune_distill",
        ),
    }
    module_name, func_name = entry_points[method]
    try:
        module = __import__(module_name, fromlist=[func_name])
        entry = getattr(module, func_name)
    except Exception:
        entry = None

    config_cls = getattr(trainer_cls, "__init__", None)
    del config_cls  # only used for the availability probe above

    if entry is not None:
        return entry(config=config)

    # Fallback: instantiate the trainer with a config dataclass if it exists.
    cfg_name = {
        "ft": "FTConfig",
        "lora": "LoRAConfig",
        "mask_tuning": "MaskTuningConfig",
        "cofi": "CoFiConfig",
        "lora_prune_distill": "LoRAPruneDistillConfig",
    }[method]
    config_cls = getattr(module, cfg_name, None)
    if config_cls is None:
        raise RuntimeError("could not find %s.%s" % (module_name, cfg_name))
    cfg = config_cls.from_dict(config)
    trainer = trainer_cls(config=cfg)
    return trainer.fit() if hasattr(trainer, "fit") else {}


def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a (possibly merged/pruned) APT or baseline checkpoint."""
    _require_torch()
    from apt.eval.run_eval import EvalConfig, evaluate_model

    merged = load_yaml_config(args.config)
    if args.task:
        merged["task"] = normalize_task(args.task)
    if args.model_dir:
        merged["model_name_or_path"] = args.model_dir
    elif args.model:
        merged["model_name_or_path"] = args.model
    if args.method:
        merged["method"] = normalize_method(args.method)
    if args.sparsity is not None:
        merged["sparsity"] = float(args.sparsity)
    if args.output_dir:
        merged["output_dir"] = args.output_dir

    config = EvalConfig.from_dict(merged)
    result = evaluate_model(None, task=config.task, model_name=config.model_name_or_path)
    _dump(result.as_dict() if hasattr(result, "as_dict") else result)
    return 0


def _dispatch_script(script_name: str, args: argparse.Namespace,
                     extra: Optional[List[str]] = None) -> int:
    """Run one of the reproduction scripts in-process or as a subprocess."""
    module = "scripts.%s" % script_name
    argv = list(extra or [])
    try:
        script = __import__(module, fromlist=["main"])
    except Exception:
        # Fall back to executing the file directly.
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "scripts", script_name + ".py")
        if not os.path.isfile(path):
            _log("[main] cannot find %s (%s)" % (script_name, path))
            return 2
        import runpy

        sys.argv = [path] + argv
        runpy.run_path(path, run_name="__main__")
        return 0

    main_fn = getattr(script, "main", None)
    if main_fn is None:
        import runpy

        path = getattr(script, "__file__", None)
        sys.argv = [str(path)] + argv
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return 0

    result = _call_main(main_fn, argv)
    return 0 if result is None else int(result or 0)


def _call_main(main_fn: Any, argv: List[str]) -> Any:
    """Call a script ``main`` robustly across its possible signatures."""
    import inspect

    try:
        sig = inspect.signature(main_fn)
        params = list(sig.parameters)
    except (TypeError, ValueError):  # pragma: no cover
        params = []

    if not params:
        return main_fn()
    if len(params) == 1:
        return main_fn(argv)
    return main_fn(argv, None)


def cmd_table2(args: argparse.Namespace) -> int:
    return _dispatch_script("run_table2", args, _script_args(args))


def cmd_table4(args: argparse.Namespace) -> int:
    return _dispatch_script("run_table4_ablation", args, _script_args(args))


def cmd_table7(args: argparse.Namespace) -> int:
    return _dispatch_script("run_table7_bert", args, _script_args(args))


def cmd_table8(args: argparse.Namespace) -> int:
    return _dispatch_script("run_table8_glue", args, _script_args(args))


def cmd_figure3(args: argparse.Namespace) -> int:
    return _dispatch_script("run_figure3_sparsity", args, _script_args(args))


def _script_args(args: argparse.Namespace) -> List[str]:
    argv: List[str] = []
    if args.model:
        argv += ["--model", args.model]
    if args.task:
        argv += ["--task", args.task]
    if args.config:
        argv += ["--config", args.config]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if getattr(args, "sparsity", None) is not None:
        argv += ["--sparsity", str(args.sparsity)]
    return argv


def cmd_info(args: argparse.Namespace) -> int:
    """Report which optional pieces of the codebase are importable."""
    info: Dict[str, Any] = {"apt_version": _version()}

    try:
        from apt.data import available as data_available

        info["data"] = data_available()
    except Exception as exc:  # pragma: no cover
        info["data"] = "unavailable: %s" % exc

    try:
        from apt.eval import available as eval_available

        info["eval"] = eval_available()
    except Exception as exc:  # pragma: no cover
        info["eval"] = "unavailable: %s" % exc

    try:
        from apt.baselines import available as baseline_available

        info["baselines"] = baseline_available()
    except Exception as exc:  # pragma: no cover
        info["baselines"] = "unavailable: %s" % exc

    try:
        import torch  # type: ignore

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None

    try:
        import transformers  # type: ignore

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None

    info["configs"] = sorted(
        f for f in os.listdir(CONFIG_BASE)
        if f.endswith((".yaml", ".yml"))
    ) if os.path.isdir(CONFIG_BASE) else []

    _dump(info)
    return 0


def _summarize(summary: Any) -> Dict[str, Any]:
    """Keep only JSON-friendly, paper-relevant keys from a training summary."""
    if summary is None:
        return {}
    if isinstance(summary, dict):
        keep = (
            "method",
            "model",
            "task",
            "sparsity",
            "metrics",
            "primary",
            "train_time_s",
            "train_peak_mem_mb",
            "tta_seconds",
            "inf_time_ms",
            "inf_mem_mb",
            "relative",
            "num_parameters",
            "output_dir",
        )
        out = {k: summary[k] for k in keep if k in summary}
        if not out:
            out = {k: v for k, v in summary.items() if k != "trainer"}
        return out
    return {"summary": str(summary)}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="APT: Adaptive Pruning and Tuning pretrained LMs "
                    "(reproduction entry point)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default=None,
                       help="YAML config path, filename or short alias")
        p.add_argument("--model", default=None,
                       help="HF model name or local checkpoint dir")
        p.add_argument("--task", default=None, help="task name (sst2, mnli, ...)")
        p.add_argument("--sparsity", type=float, default=None,
                       help="target sparsity (paper uses 0.60 for RoBERTa/T5)")
        p.add_argument("--epochs", type=int, default=None)
        p.add_argument("--distill-epochs", dest="distill_epochs",
                       type=int, default=None)
        p.add_argument("--output-dir", dest="output_dir", default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--json", action="store_true", help="print JSON output")

    p_train = sub.add_parser("train-apt", help="train APT (Algorithm 1)")
    add_common(p_train)
    p_train.set_defaults(func=cmd_train_apt)

    p_base = sub.add_parser("train-baseline", help="train an FT/LoRA/CoFi baseline")
    add_common(p_base)
    p_base.add_argument("--method", default="ft",
                        help="one of: " + ", ".join(BASELINE_METHODS))
    p_base.set_defaults(func=cmd_train_baseline)

    p_eval = sub.add_parser("eval", help="evaluate a trained checkpoint")
    add_common(p_eval)
    p_eval.add_argument("--model-dir", dest="model_dir", default=None)
    p_eval.add_argument("--method", default=None)
    p_eval.set_defaults(func=cmd_eval)

    p_t2 = sub.add_parser("table2", help="reproduce Table 2 (RoBERTa/T5, 60%)")
    add_common(p_t2)
    p_t2.set_defaults(func=cmd_table2)

    p_t4 = sub.add_parser("table4", help="reproduce Table 4 (RoBERTa ablations)")
    add_common(p_t4)
    p_t4.set_defaults(func=cmd_table4)

    p_t7 = sub.add_parser("table7", help="reproduce Table 7 (BERT vs PST/LRP)")
    add_common(p_t7)
    p_t7.set_defaults(func=cmd_table7)

    p_t8 = sub.add_parser("table8", help="reproduce Table 8 (GLUE vs LoRA+Distill)")
    add_common(p_t8)
    p_t8.set_defaults(func=cmd_table8)

    p_f3 = sub.add_parser("figure3", help="reproduce Figure 3 (sparsity sweep)")
    add_common(p_f3)
    p_f3.set_defaults(func=cmd_figure3)

    p_info = sub.add_parser("info", help="report environment / availability")
    p_info.set_defaults(func=cmd_info)

    # also expose the baseline list for convenience
    parser.add_argument("--version", action="version",
                        version="APT reproduction %s" % _version())
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
