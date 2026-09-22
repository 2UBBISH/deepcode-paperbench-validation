#!/usr/bin/env python
"""Entry point for DPMs-ANT adaptor training (Algorithm 1 / Eq. (8)).

Fine-tunes *only* zero-initialized adaptor parameters ``psi`` inserted into the
frozen pretrained diffusion backbone ``theta`` (guided-diffusion DDPM 256x256 or
CompVis LDM 64x64), using adversarial noise selection (Eq. 7) and the
similarity-guided objective (Eq. 5/8) driven by a frozen 2-way classifier
``p_phi(y=T | x_t)``.

Paper mapping
-------------
* Eq. (5)   similarity-guided loss  -> ``dpm_ant.training.sg_loss``
* Eq. (7)   adversarial noise       -> ``dpm_ant.training.adv_noise``
* Eq. (8)/Alg. 1 adaptor update     -> ``dpm_ant.training.ant_trainer``
* Section 5.2 / addendum hyperparameters (gamma=5, omega=0.02, J=10, batch 40,
  adam, lr 5e-5 DDPM / 1e-5 LDM, ~300 outer iterations) are read from
  ``configs/default.yaml`` + ``configs/per_task.yaml``.

Examples
--------
DDPM, FFHQ -> Sunglasses 10-shot::

    python scripts/train_ant.py --task ffhq_sunglasses \
        --diffusion-checkpoint /ckpt/256x256_diffusion.pt \
        --classifier-checkpoint /ckpt/256x256_sunglasses_classifier.pt \
        --target-dir data/targets/sunglasses --out runs/sunglasses_ant

LDM synonym::

    python scripts/train_ant.py --task ffhq_sunglasses_ldm \
        --diffusion-checkpoint /ckpt/ldm_ffhq.ckpt --out runs/sunglasses_ldm_ant

Ablation ("DPMs-ANT w/o AN")::

    python scripts/train_ant.py --task ffhq_sunglasses --variant ant_wo_an ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("dpm_ant.train_ant")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_CONFIG = os.path.join(_ROOT, "configs", "default.yaml")
PER_TASK_CONFIG = os.path.join(_ROOT, "configs", "per_task.yaml")

# ---------------------------------------------------------------------------
# Paper-specified global defaults (Section 5.2 Configurations)
# ---------------------------------------------------------------------------
ANT_DEFAULTS: Dict[str, Any] = {
    "gamma": 5.0,
    "omega": 0.02,
    "J": 10,
    "iterations": 300,
    "batch_size": 40,
    "lr_ddpm": 5e-5,
    "lr_ldm": 1e-5,
    "optimizer": "adam",
    "adam_betas": [0.9, 0.999],
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "norm": "per_sample",
    "use_adv_noise": True,
    "freeze_backbone": True,
    "only_adaptor": True,
    "detach_classifier_grad": True,
    "classifier_target_index": 1,
    "log_interval": 10,
    "save_interval": 100,
}

#: Ablation variants from Section 5.4 / Figure 4 (FID on 10-shot Sunglasses).
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full_finetune": {"only_adaptor": False, "freeze_backbone": False,
                      "use_adv_noise": False, "gamma": 0.0},
    "adaptor_only": {"only_adaptor": True, "freeze_backbone": True,
                     "use_adv_noise": False},
    "ant_wo_an": {"only_adaptor": True, "freeze_backbone": True,
                  "use_adv_noise": False},
    "full_ant": {"only_adaptor": True, "freeze_backbone": True,
                 "use_adv_noise": True},
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML file, returning ``{}`` when unavailable."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - pyyaml is a hard requirement
        LOGGER.warning("pyyaml unavailable; cannot read %s", path)
        return {}
    with open(path, "r") as fh:
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
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge ``configs/default.yaml`` with ``configs/per_task.yaml``."""
    cfg = _load_yaml(config_path or DEFAULT_CONFIG)
    _deep_update(cfg, _load_yaml(per_task_path or PER_TASK_CONFIG))
    # Flatten per_task.yaml ``defaults`` into the top-level namespaces so both
    # nested (default.yaml) and flat (per_task.yaml) layouts work.
    flat = cfg.get("defaults")
    if isinstance(flat, dict):
        ant = cfg.setdefault("ant", {})
        for key, value in flat.items():
            if key == "adaptor" and isinstance(value, dict):
                _deep_update(cfg.setdefault("adaptor", {}), value)
            else:
                ant.setdefault(key, value)
    if task:
        cfg["task"] = task
    return cfg


def _ensure_ant_block(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in Section 5.2 ANT defaults missing from the loaded config."""
    ant = cfg.setdefault("ant", {})
    for key, value in ANT_DEFAULTS.items():
        ant.setdefault(key, value)
    diffusion = cfg.setdefault("diffusion", {})
    diffusion.setdefault("num_timesteps", 1000)
    diffusion.setdefault("schedule", "linear")
    diffusion.setdefault("beta_start", 1e-4)
    diffusion.setdefault("beta_end", 0.02)
    diffusion.setdefault("eta", 0.0)
    cfg.setdefault("seed", 0)
    cfg.setdefault("device", "cuda")
    return cfg


def _task_entry(cfg: Dict[str, Any], task: Optional[str]) -> Dict[str, Any]:
    """Return the per-task hyperparameter entry (addendum Table 3)."""
    if not task:
        return {}
    tasks = cfg.get("tasks") or {}
    entry = tasks.get(task)
    if entry is None:
        LOGGER.warning("task '%s' not found in config; using global defaults", task)
        return {}
    return dict(entry)


def _resolve_dir(cli_value: Optional[str], cfg: Dict[str, Any],
                 candidates: List[str]) -> Optional[str]:
    """Resolve a directory from CLI value then nested config keys."""
    if cli_value:
        return cli_value
    node: Any = cfg
    for key in candidates:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            node = None
            break
    if isinstance(node, str):
        return node
    return None


def _resolve_backbone(cfg: Dict[str, Any], backbone: Optional[str],
                      task: Optional[str]) -> str:
    """Resolve the backbone (``ddpm`` / ``ldm``) from CLI, task entry, config."""
    if backbone:
        return backbone.lower()
    entry = _task_entry(cfg, task)
    value = entry.get("backbone") or cfg.get("backbone") or "ddpm"
    return str(value).lower()


def _checkpoint_path(cfg: Dict[str, Any], backbone: str,
                     key: str) -> Optional[str]:
    """Look up a checkpoint path under ``models.<backbone>.<key>``."""
    models = cfg.get("models") or {}
    block = models.get(backbone) or {}
    value = block.get(key)
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DPMs-ANT adaptor training (Algorithm 1 / Eq. 8)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="path to configs/default.yaml")
    parser.add_argument("--per-task-config", default=PER_TASK_CONFIG,
                        help="path to configs/per_task.yaml")
    parser.add_argument("--task", default=None,
                        help="task key, e.g. ffhq_sunglasses / ffhq_sunglasses_ldm")
    parser.add_argument("--backbone", default=None, choices=["ddpm", "ldm"],
                        help="override the diffusion backbone")
    parser.add_argument("--target", default=None, help="target dataset name")
    parser.add_argument("--target-dir", default=None,
                        help="directory containing the 10-shot target images")
    parser.add_argument("--source-dir", default=None,
                        help="directory of source (e.g. FFHQ) images")
    parser.add_argument("--diffusion-checkpoint", default=None,
                        help="pretrained diffusion backbone checkpoint")
    parser.add_argument("--autoencoder-checkpoint", default=None,
                        help="LDM autoencoder checkpoint (f=4)")
    parser.add_argument("--classifier-checkpoint", default=None,
                        help="fine-tuned 2-way classifier checkpoint")
    parser.add_argument("--adaptor-init", default=None,
                        help="optional adaptor state dict to start from")
    parser.add_argument("--variant", default=None, choices=sorted(ABLATION_VARIANTS),
                        help="ablation variant from Section 5.4 / Figure 4")
    parser.add_argument("--iterations", type=int, default=None,
                        help="outer adaptor iterations (default 300)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="target batch size (default 40)")
    parser.add_argument("--lr", type=float, default=None, help="adaptor learning rate")
    parser.add_argument("--gamma", type=float, default=None,
                        help="similarity-guided weight gamma (Eq. 5/8)")
    parser.add_argument("--omega", type=float, default=None,
                        help="adversarial noise step size omega (Eq. 7)")
    parser.add_argument("--J", type=int, default=None,
                        help="inner ascent steps J (Eq. 7 / Algorithm 1)")
    parser.add_argument("--norm", default=None, choices=["per_sample", "per_channel"],
                        help="Norm(.) mode used in Eq. (7)")
    parser.add_argument("--no-adv-noise", action="store_true",
                        help="disable adversarial noise (DPMs-ANT w/o AN)")
    parser.add_argument("--full-finetune", action="store_true",
                        help="update all model parameters (direct fine-tuning)")
    parser.add_argument("--shots", type=int, default=None,
                        help="number of target shots (default 10)")
    parser.add_argument("--repeat", type=int, default=None,
                        help="repeat factor for the few-shot dataset")
    parser.add_argument("--num-timesteps", type=int, default=None,
                        help="diffusion timesteps T (default 1000)")
    parser.add_argument("--schedule", default=None, choices=["linear", "cosine"],
                        help="beta schedule (default linear)")
    parser.add_argument("--device", default=None, help="cuda / cpu")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--out", default=None,
                        help="output directory (or .pt path) for adaptor weights")
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve configuration and exit without training")
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


# ---------------------------------------------------------------------------
# Model / data construction
# ---------------------------------------------------------------------------
def build_schedule_from_cfg(cfg: Dict[str, Any]):
    """Build the noise schedule from the ``diffusion`` config block."""
    from dpm_ant.diffusion.schedule import build_schedule

    diff = cfg.get("diffusion", {})
    return build_schedule(diff)


def build_backbone(cfg: Dict[str, Any], backbone: str, task: Optional[str],
                   device: str, checkpoint: Optional[str] = None,
                   autoencoder_checkpoint: Optional[str] = None,
                   verbose: bool = True):
    """Load the frozen backbone, insert zero-init adaptors into shift modules.

    Returns
    -------
    (model, autoencoder)
        ``model`` exposes ``epsilon_theta(x_t, t)`` (and, after freeze, only the
        inserted adaptor parameters require grad).  ``autoencoder`` is a frozen
        ``FrozenLDMAutoencoder`` for LDM and ``None`` for DDPM.
    """
    from dpm_ant.models.adaptor import build_adaptor_factory

    adaptor_cfg = cfg.get("adaptor", {})
    factory = build_adaptor_factory(cfg=cfg, backbone=backbone, task=task,
                                    **{k: v for k, v in adaptor_cfg.items()
                                       if k in ("bottleneck_c", "bottleneck_c_ldm",
                                                "bottleneck_d", "hidden_dims",
                                                "composition", "num_heads",
                                                "up_sample_factor")})

    if backbone == "ldm":
        from dpm_ant.models.ldm_loader import (
            FrozenLDMAutoencoder,
            FrozenLDMUNet,
            insert_adaptors_ldm,
            load_autoencoder,
            load_ldm_unet,
        )

        unet = load_ldm_unet(checkpoint=checkpoint, cfg=cfg, device=device,
                             verbose=verbose)
        insert_adaptors_ldm(unet, factory, shift_only=True, verbose=verbose)
        model = FrozenLDMUNet(unet, freeze=True)
        autoencoder = FrozenLDMAutoencoder(
            load_autoencoder(checkpoint=autoencoder_checkpoint, cfg=cfg,
                             device=device, verbose=verbose))
        autoencoder.freeze()
        return model, autoencoder

    from dpm_ant.models.unet_loader import (
        FrozenDDPMUNet,
        insert_adaptors,
        load_guided_diffusion_unet,
    )

    unet = load_guided_diffusion_unet(checkpoint=checkpoint, cfg=cfg,
                                      device=device, strict=False,
                                      verbose=verbose)
    learn_sigma = (cfg.get("models", {}).get("ddpm", {}) or {}).get("learn_sigma", True)
    insert_adaptors(unet, factory, shift_only=True, verbose=verbose)
    model = FrozenDDPMUNet(unet, learn_sigma=bool(learn_sigma), freeze=True)
    return model, None


def build_classifier_for_task(cfg: Dict[str, Any], backbone: str,
                              checkpoint: Optional[str], device: str,
                              verbose: bool = True):
    """Load the frozen 2-way source/target classifier ``p_phi`` (may be None)."""
    if not checkpoint:
        LOGGER.warning("no classifier checkpoint provided: similarity-guided term "
                       "will be inactive (gamma effectively 0)")
        return None
    try:
        from dpm_ant.models.classifier import build_classifier

        classifier = build_classifier(cfg=cfg, backbone=backbone,
                                      checkpoint=checkpoint, device=device)
        if hasattr(classifier, "freeze"):
            classifier.freeze()
        return classifier
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("failed to build classifier (%s); continuing without it", exc)
        return None


def build_target_data(cfg: Dict[str, Any], backbone: str, target: Optional[str],
                      target_dir: Optional[str], shots: int, repeat: int,
                      size: int, autoencoder=None, device: str = "cpu",
                      num_workers: int = 0, seed: int = 0):
    """Build the 10-shot target dataset (latents for LDM, pixels for DDPM)."""
    from dpm_ant.data.datasets import (
        TensorDataset,
        build_dataset,
        build_target_dataset,
        load_image,
        list_images,
        to_latents,
    )

    if target_dir:
        dataset = build_dataset(target_dir, size=size, shots=shots, repeat=repeat,
                                seed=seed)
    else:
        dataset = build_target_dataset(cfg=cfg, target=target, shots=shots,
                                       size=size, backbone=backbone,
                                       repeat=repeat, seed=seed)

    if backbone != "ldm":
        return dataset

    # LDM adaptation happens in the 64x64 latent space: encode the few-shot
    # images with the frozen autoencoder once (Section 5.2 Configurations).
    try:
        total = len(dataset)
        images = []
        for index in range(total):
            item = dataset[index]
            images.append(item[0] if isinstance(item, (tuple, list)) else item)
        import torch

        batch = torch.stack(images).float()
        latents = to_latents(batch, autoencoder=autoencoder)
        LOGGER.info("encoded %d target images to LDM latents %s", total,
                    tuple(latents.shape))
        return TensorDataset(latents, repeat=1)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("LDM latent encoding failed (%s); falling back to the "
                       "raw target dataset (64x64 crops)", exc)
        return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    import torch

    cfg = load_config(args.config, args.per_task_config, args.task)
    _ensure_ant_block(cfg)

    backbone = _resolve_backbone(cfg, args.backbone, args.task)
    task_entry = _task_entry(cfg, args.task)
    ant_cfg = cfg.setdefault("ant", {})

    # --- resolve hyperparameters (CLI > task entry > config > paper default) ---
    def pick(name: str, cli_value: Any, default: Any) -> Any:
        if cli_value is not None:
            return cli_value
        if name in task_entry:
            return task_entry[name]
        return ant_cfg.get(name, default)

    iterations = int(pick("iterations", args.iterations, ANT_DEFAULTS["iterations"]))
    batch_size = int(pick("batch_size", args.batch_size, ANT_DEFAULTS["batch_size"]))
    lr = args.lr
    if lr is None:
        lr = task_entry.get("lr")
    if lr is None:
        lr = ant_cfg.get("lr_ddpm" if backbone == "ddpm" else "lr_ldm",
                         ANT_DEFAULTS["lr_ddpm" if backbone == "ddpm" else "lr_ldm"])
    gamma = float(pick("gamma", args.gamma, ANT_DEFAULTS["gamma"]))
    omega = float(pick("omega", args.omega, ANT_DEFAULTS["omega"]))
    J = int(pick("J", args.J, ANT_DEFAULTS["J"]))
    norm = args.norm or ant_cfg.get("norm", ANT_DEFAULTS["norm"])
    use_adv_noise = bool(ant_cfg.get("use_adv_noise", ANT_DEFAULTS["use_adv_noise"]))
    only_adaptor = bool(ant_cfg.get("only_adaptor", ANT_DEFAULTS["only_adaptor"]))
    freeze_backbone = bool(ant_cfg.get("freeze_backbone",
                                       ANT_DEFAULTS["freeze_backbone"]))
    if args.no_adv_noise:
        use_adv_noise = False
    if args.full_finetune:
        only_adaptor, freeze_backbone = False, False

    variants = cfg.get("ablation_variants") or {}
    if args.variant:
        overrides = ABLATION_VARIANTS.get(args.variant, {})
        overrides = _deep_update(dict(overrides), variants.get(args.variant, {}))
        use_adv_noise = bool(overrides.get("use_adv_noise", use_adv_noise))
        only_adaptor = bool(overrides.get("only_adaptor", only_adaptor))
        freeze_backbone = bool(overrides.get("freeze_backbone", freeze_backbone))
        gamma = float(overrides.get("gamma", gamma))

    num_timesteps = int(args.num_timesteps or cfg.get("diffusion", {}).get(
        "num_timesteps", 1000))
    schedule_name = args.schedule or cfg.get("diffusion", {}).get("schedule", "linear")
    cfg.setdefault("diffusion", {})
    cfg["diffusion"]["num_timesteps"] = num_timesteps
    cfg["diffusion"]["schedule"] = schedule_name

    shots = int(args.shots if args.shots is not None
                else (task_entry.get("shots") or cfg.get("data", {}).get("shots", 10)
                      or 10))
    repeat = int(args.repeat if args.repeat is not None
                 else task_entry.get("repeat", 1))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    device = args.device or cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA unavailable; falling back to CPU")
        device = "cpu"
    torch.manual_seed(seed)

    target = args.target or task_entry.get("target")
    target_dir = _resolve_dir(
        args.target_dir, cfg,
        ["data", "target_dirs", target] if target else ["data", "target_dirs"])
    diffusion_ckpt = args.diffusion_checkpoint or _checkpoint_path(
        cfg, backbone, "diffusion_ckpt" if backbone == "ddpm" else "unet_ckpt")
    classifier_ckpt = args.classifier_checkpoint or _checkpoint_path(
        cfg, backbone, "classifier_ckpt")
    ae_ckpt = args.autoencoder_checkpoint or _checkpoint_path(cfg, "ldm",
                                                              "autoencoder_ckpt")
    out_dir = args.out

    LOGGER.info("task=%s backbone=%s device=%s", args.task, backbone, device)
    LOGGER.info("iterations=%d batch_size=%d lr=%s gamma=%s omega=%s J=%d",
                iterations, batch_size, lr, gamma, omega, J)
    LOGGER.info("use_adv_noise=%s only_adaptor=%s freeze_backbone=%s shots=%d",
                use_adv_noise, only_adaptor, freeze_backbone, shots)
    LOGGER.info("target=%s target_dir=%s", target, target_dir)

    if args.dry_run:
        LOGGER.info("dry run: configuration resolved, exiting")
        print(json.dumps({
            "task": args.task, "backbone": backbone, "iterations": iterations,
            "batch_size": batch_size, "lr": lr, "gamma": gamma, "omega": omega,
            "J": J, "norm": norm, "use_adv_noise": use_adv_noise,
            "only_adaptor": only_adaptor, "freeze_backbone": freeze_backbone,
            "shots": shots, "target_dir": target_dir,
            "diffusion_checkpoint": diffusion_ckpt,
        }, indent=2, default=str))
        return 0

    # --- build schedule, model, classifier, data -----------------------------
    schedule = build_schedule_from_cfg(cfg)
    tag = f" [{args.task}]" if args.task else ""
    schedule.to(device)
    LOGGER.info("schedule%s: T=%d betas=%s", tag,
                getattr(schedule, "num_timesteps", num_timesteps),
                getattr(schedule, "schedule_name", schedule_name))

    model, autoencoder = build_backbone(cfg, backbone, args.task, device,
                                        checkpoint=diffusion_ckpt,
                                        autoencoder_checkpoint=ae_ckpt,
                                        verbose=not args.verbose is False)
    if args.adaptor_init:
        state = torch.load(args.adaptor_init, map_location=device)
        state = state.get("adaptor", state)
        for module in model.modules():
            if hasattr(module, "load_adaptor_state_dict"):
                module.load_adaptor_state_dict(state)
                break
        LOGGER.info("initialised adaptors from %s", args.adaptor_init)

    classifier = build_classifier_for_task(cfg, backbone, classifier_ckpt, device,
                                           verbose=True)
    if classifier is not None and hasattr(classifier, "to"):
        classifier.to(device)
    if classifier is None:
        gamma = 0.0

    image_size = int(cfg.get("data", {}).get("image_size", 256))
    size = (int(cfg.get("models", {}).get("ldm", {}).get("latent_size", 64))
            if backbone == "ldm" else image_size)
    target_data = build_target_data(cfg, backbone, target, target_dir, shots,
                                    repeat, size, autoencoder=autoencoder,
                                    device=device, seed=seed)
    LOGGER.info("target shots=%d (dataset length=%d)", shots, len(target_data))

    # --- train ---------------------------------------------------------------
    from dpm_ant.training.ant_trainer import train_ant

    overrides: Dict[str, Any] = {
        "gamma": gamma,
        "omega": omega,
        "J": J,
        "norm": norm,
        "batch_size": batch_size,
        "lr": lr,
        "use_adv_noise": use_adv_noise,
        "only_adaptor": only_adaptor,
        "freeze_backbone": freeze_backbone,
        "num_timesteps": num_timesteps,
        "schedule": schedule_name,
        "detach_classifier_grad": bool(ant_cfg.get("detach_classifier_grad", True)),
        "classifier_target_index": int(ant_cfg.get("classifier_target_index", 1)),
        "seed": seed,
        "device": device,
    }
    if args.log_interval is not None:
        overrides["log_interval"] = int(args.log_interval)
    if args.save_interval is not None:
        overrides["save_interval"] = int(args.save_interval)
    if args.num_workers is not None:
        overrides["num_workers"] = int(args.num_workers)
    overrides = _deep_update({k: v for k, v in ant_cfg.items()}, overrides)

    start = time.time()
    trainer, history = train_ant(
        model,
        target_data,
        classifier=classifier,
        cfg=cfg,
        backbone=backbone,
        task=args.task,
        iterations=iterations,
        device=device,
        verbose=True,
        **overrides,
    )
    elapsed = time.time() - start

    param_rate = None
    if hasattr(trainer, "parameter_rate"):
        try:
            param_rate = float(trainer.parameter_rate())
        except Exception:  # pragma: no cover - defensive
            param_rate = None
    LOGGER.info("training finished in %.3f GPU hours (%.1f s)",
                elapsed / 3600.0, elapsed)
    if param_rate is not None:
        LOGGER.info("fine-tuned parameter rate = %.4f (%.2f%%)",
                    param_rate, 100.0 * param_rate)
    if history:
        LOGGER.info("final loss: %s", history[-1])

    # --- save ---------------------------------------------------------------
    if out_dir:
        ckpt_path = out_dir
        if not ckpt_path.endswith(".pt") and not ckpt_path.endswith(".pth"):
            os.makedirs(out_dir, exist_ok=True)
            ckpt_path = os.path.join(out_dir, "adaptor.pt")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)) or ".", exist_ok=True)
        if hasattr(trainer, "save"):
            trainer.save(ckpt_path)
        else:  # pragma: no cover - fallback
            torch.save({"adaptor": getattr(trainer, "adaptor_state_dict", lambda: {})()},
                       ckpt_path)
        LOGGER.info("saved adaptor checkpoint to %s", ckpt_path)

        meta = {
            "task": args.task,
            "backbone": backbone,
            "variant": args.variant,
            "shots": shots,
            "iterations": iterations,
            "batch_size": batch_size,
            "lr": lr,
            "gamma": gamma,
            "omega": omega,
            "J": J,
            "norm": norm,
            "use_adv_noise": use_adv_noise,
            "only_adaptor": only_adaptor,
            "freeze_backbone": freeze_backbone,
            "num_timesteps": num_timesteps,
            "schedule": schedule_name,
            "seed": seed,
            "elapsed_seconds": elapsed,
            "elapsed_gpu_hours": elapsed / 3600.0,
            "parameter_rate": param_rate,
            "target_dir": target_dir,
            "diffusion_checkpoint": diffusion_ckpt,
            "classifier_checkpoint": classifier_ckpt,
        }
        meta_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                 "train_ant_meta.json")
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        if history:
            hist_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)),
                                     "train_ant_history.json")
            with open(hist_path, "w") as fh:
                json.dump(history, fh, indent=2, default=str)
        LOGGER.info("wrote metadata to %s", meta_path)
    else:
        LOGGER.info("no --out given; adaptor weights were not persisted")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
