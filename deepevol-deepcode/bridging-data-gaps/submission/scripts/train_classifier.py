#!/usr/bin/env python
"""Fine-tune the binary source/target classifier ``p_phi``.

Reproduces the classifier fine-tuning step of DPMs-ANT (Section 4.1 / 5.2 and
the addendum "Classifier Training (Section 5.2)"):

* load a pretrained (ImageNet) guided-diffusion classifier,
* replace its final layer with a 2-way (source=0 / target=1) head,
* fine-tune on noised source and target images with ``t ~ Uniform({1..T})``
  using Adam, lr=1e-4, batch size 64 and 300 iterations,
* save the fine-tuned classifier which is later frozen while training the
  adaptors (Section 4.3).

Example
-------
    python scripts/train_classifier.py --config configs/default.yaml \
        --classifier-config configs/classifier.yaml \
        --backbone ddpm --task ffhq_sunglasses \
        --source-dir data/source/ffhq --target-dir data/target/sunglasses \
        --out runs/classifier_ffhq_sunglasses.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# make the repository root importable when executed as a script
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

LOGGER = logging.getLogger("dpm_ant.train_classifier")


# ---------------------------------------------------------------------------
# config handling
# ---------------------------------------------------------------------------
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover
        LOGGER.warning("pyyaml not installed; cannot load %s", path)
        return {}
    if not os.path.isfile(path):
        LOGGER.warning("config file %s does not exist; ignoring", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a mapping at the top level")
    return data


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: Optional[str] = None,
    classifier_config_path: Optional[str] = None,
    per_task_path: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge default + per-task + classifier configs into one dict."""
    cfg = _load_yaml(config_path)
    if per_task_path:
        per_task = _load_yaml(per_task_path)
        cfg = _deep_update(cfg, per_task)
    if classifier_config_path:
        clf_specific = _load_yaml(classifier_config_path)
        # ``classifier.yaml`` may nest its values under ``classifier`` already;
        # merge whatever it provides.
        cfg = _deep_update(cfg, clf_specific)
    return cfg


def _ensure_classifier_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee ``cfg['classifier']`` exists with the paper defaults."""
    clf = dict(cfg.get("classifier", {}) or {})
    defaults = {
        "num_classes": 2,
        "target_index": 1,
        "lr": 1e-4,
        "batch_size": 64,
        "iterations": 300,
        "optimizer": "adam",
        "adam_betas": [0.9, 0.999],
        "t_sampling": "uniform_1_T",
        "noise_frac_source": 0.5,
        "source_pool": 1000,
        "target_pool": 10,
        "replace_head": True,
        "freeze_during_ant": True,
        "input_mode": "pixel",
    }
    for key, value in defaults.items():
        clf.setdefault(key, value)
    cfg["classifier"] = clf
    return clf


def _task_entry(cfg: Dict[str, Any], task: Optional[str]) -> Dict[str, Any]:
    if not task:
        return {}
    tasks = cfg.get("tasks", {}) or {}
    entry = tasks.get(task)
    if entry is None:
        LOGGER.warning("task '%s' not present in config; ignoring per-task settings", task)
        return {}
    return entry


def _resolve_dir(cli_value: Optional[str], cfg: Dict[str, Any], candidates) -> Optional[str]:
    if cli_value:
        return cli_value
    for candidate in candidates:
        node: Any = cfg
        ok = True
        for part in candidate:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if ok and isinstance(node, (str, os.PathLike)):
            return str(node)
    return None


# ---------------------------------------------------------------------------
# main routine
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune the 2-way source/target classifier (DPMs-ANT, Section 5.2)."
    )
    p.add_argument("--config", default="configs/default.yaml", help="global config yaml")
    p.add_argument("--per-task-config", default="configs/per_task.yaml", help="per-task overrides yaml")
    p.add_argument("--classifier-config", default="configs/classifier.yaml", help="classifier yaml")
    p.add_argument("--task", default=None, help="task key, e.g. ffhq_sunglasses")
    p.add_argument("--backbone", default=None, choices=[None, "ddpm", "ldm"], help="diffusion backbone")
    p.add_argument("--source", default=None, help="source dataset name (ffhq / lsun_church)")
    p.add_argument("--target", default=None, help="target dataset name (babies / sunglasses / ...)")
    p.add_argument("--source-dir", default=None, help="explicit source image directory")
    p.add_argument("--target-dir", default=None, help="explicit target image directory")
    p.add_argument("--target-pool", type=int, default=None,
                   help="number of target images used for fine-tuning (10, or 100 for the ablation)")
    p.add_argument("--checkpoint", default=None,
                   help="pretrained classifier checkpoint to initialise from")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-timesteps", type=int, default=None)
    p.add_argument("--schedule", default=None, help="beta schedule (linear / cosine)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="output checkpoint path (.pt)")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="build everything but skip optimisation")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(
        config_path=args.config,
        classifier_config_path=args.classifier_config,
        per_task_path=args.per_task_config,
        task=args.task,
    )
    clf_cfg = _ensure_classifier_block(cfg)
    task_entry = _task_entry(cfg, args.task)

    # ---- resolve runtime settings ----------------------------------------
    backbone = args.backbone or task_entry.get("backbone") or cfg.get("backbone") or "ddpm"
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    device = args.device or cfg.get("device", None)

    num_timesteps = args.num_timesteps or int(
        (cfg.get("diffusion", {}) or {}).get("num_timesteps", 1000)
    )
    schedule_name = args.schedule or (cfg.get("diffusion", {}) or {}).get("schedule", "linear")

    iterations = args.iterations if args.iterations is not None else int(clf_cfg.get("iterations", 300))
    batch_size = args.batch_size if args.batch_size is not None else int(clf_cfg.get("batch_size", 64))
    lr = args.lr if args.lr is not None else float(clf_cfg.get("lr", 1e-4))
    target_pool = (
        args.target_pool
        if args.target_pool is not None
        else int(clf_cfg.get("target_pool", 10))
    )

    source_name = args.source or task_entry.get("source") or "ffhq"
    target_name = args.target or task_entry.get("target") or "sunglasses"

    source_dir = _resolve_dir(
        args.source_dir, cfg, [("data", "source_dirs", source_name), ("data", "source_dirs", "ffhq")]
    )
    target_dir = _resolve_dir(
        args.target_dir, cfg, [("data", "target_dirs", target_name), ("data", "dirs", target_name)]
    )

    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    # ---- lazy imports (heavy) --------------------------------------------
    import torch

    from dpm_ant.diffusion.schedule import build_schedule
    from dpm_ant.models.classifier import build_classifier
    from dpm_ant.training.classifier_train import train_classifier

    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    LOGGER.info("classifier fine-tuning | backbone=%s device=%s seed=%d", backbone, device, seed)
    LOGGER.info("task=%s source=%s target=%s target_pool=%d", args.task, source_name, target_name, target_pool)
    LOGGER.info("iterations=%d batch_size=%d lr=%g T=%d schedule=%s", iterations, batch_size, lr, num_timesteps, schedule_name)

    schedule = build_schedule(
        {
            "diffusion": {
                "num_timesteps": num_timesteps,
                "schedule": schedule_name,
                **(cfg.get("diffusion", {}) or {}),
            }
        }
    )

    # ---- classifier -------------------------------------------------------
    classifier = build_classifier(
        cfg=cfg,
        backbone=backbone,
        num_classes=int(clf_cfg.get("num_classes", 2)),
        target_index=int(clf_cfg.get("target_index", 1)),
        checkpoint=args.checkpoint,
        device=device,
    )

    if args.dry_run:
        LOGGER.info("dry-run: classifier built (%s), skipping fine-tuning", type(classifier).__name__)
        return 0

    # ---- data -------------------------------------------------------------
    if source_dir is None and target_dir is None:
        LOGGER.warning(
            "no source/target directories resolved from config; classifier fine-tuning "
            "will use synthetic noise-only batches so the pipeline remains runnable."
        )

    trained_classifier, trainer = train_classifier(
        classifier=classifier,
        cfg=cfg,
        backbone=backbone,
        task=args.task,
        source_dir=source_dir,
        target_dir=target_dir,
        device=device,
        iterations=iterations,
        checkpoint_path=args.out,
        verbose=not args.dry_run,
        lr=lr,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        schedule=schedule_name,
        target_pool=target_pool,
        log_interval=args.log_interval,
        seed=seed,
    )

    history = list(getattr(trainer, "history", []) or [])
    if history:
        first, last = history[0], history[-1]
        LOGGER.info(
            "classifier loss %.4f -> %.4f over %d iterations",
            float(first.get("loss", float("nan"))),
            float(last.get("loss", float("nan"))),
            len(history),
        )

    out_path = args.out or os.path.join(
        (cfg.get("logging", {}) or {}).get("ckpt_dir", "runs/checkpoints"),
        f"classifier_{backbone}_{target_name}.pt",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    state = {
        "model": trained_classifier.state_dict()
        if hasattr(trained_classifier, "state_dict")
        else None,
        "config": {
            "backbone": backbone,
            "task": args.task,
            "source": source_name,
            "target": target_name,
            "target_pool": target_pool,
            "num_classes": int(clf_cfg.get("num_classes", 2)),
            "target_index": int(clf_cfg.get("target_index", 1)),
            "iterations": iterations,
            "batch_size": batch_size,
            "lr": lr,
            "num_timesteps": num_timesteps,
            "schedule": schedule_name,
            "seed": seed,
        },
        "history": history,
    }
    torch.save(state, out_path)

    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(
            {k: v for k, v in state["config"].items()},
            fh,
            indent=2,
            sort_keys=True,
        )

    LOGGER.info("saved fine-tuned classifier -> %s", out_path)
    LOGGER.info("saved metadata            -> %s", meta_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
