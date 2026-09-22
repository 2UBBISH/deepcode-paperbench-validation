#!/usr/bin/env python
"""Ablation, sensitivity and classifier-pool experiments for DPMs-ANT.

This runner drives the three evaluation studies reported in the paper:

* **Ablation study** (Section 5.4, Figure 4) - on FFHQ -> Sunglasses with 10
  target images and 300 training iterations, comparing

      1. ``full_finetune``    - direct full-model fine-tuning       (FID 41.88)
      2. ``adaptor_only``     - zero-init adaptors, vanilla DDPM    (FID 38.65)
      3. ``ant_wo_an``        - similarity-guided only (no AN)      (FID 26.41)
      4. ``full_ant``         - full DPMs-ANT (AN + SG)             (FID 20.66)

* **Sensitivity analysis** (Section 5.5 / Appendix B.3, Tables 5-7) - sweeps of
  ``gamma`` in {1, 3, 5, 7, 9}, ``omega`` in {0.01 ... 0.05} and the number of
  training iterations in {0, 100, 200, 300, 400}.  The optimum reported in the
  paper is ``gamma = 5``, ``omega = 0.02`` and ``300`` iterations.

* **Classifier ablation** (Section 5.5, Table 3) - fine-tuning the source/target
  classifier ``p_phi`` on 10 versus 100 target images.  Expected (FFHQ ->
  Sunglasses): Intra-LPIPS 0.613 +- 0.023 / FID 20.06 (10 shots) and
  Intra-LPIPS 0.637 +- 0.013 / FID 22.84 (100 shots).

Every configuration is trained through :func:`dpm_ant.training.ant_trainer.train_ant`
(Algorithm 1 / Eq. 8), sampled through :mod:`dpm_ant.sampling.sampler` and scored
with Intra-LPIPS (:mod:`dpm_ant.evaluation.intra_lpips`) and FID
(:mod:`dpm_ant.evaluation.fid`).  The script is deliberately defensive: missing
checkpoints, missing metric backends or missing datasets degrade to warnings and
partial reports instead of aborting the sweep.

Usage
-----
::

    # Full ablation on FFHQ -> Sunglasses (DDPM)
    python scripts/run_ablation.py --mode ablation --task ffhq_sunglasses

    # Sensitivity sweeps (gamma / omega / iterations)
    python scripts/run_ablation.py --mode sensitivity --task ffhq_sunglasses

    # Classifier pool ablation (10 vs 100 target images)
    python scripts/run_ablation.py --mode classifier --task ffhq_sunglasses

    # Everything at once, smoke size (tiny sample counts / iterations)
    python scripts/run_ablation.py --mode all --smoke --out outputs/ablation_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("dpm_ant.run_ablation")

# --------------------------------------------------------------------------- #
# Path / config plumbing
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_CONFIG = os.path.join(_ROOT, "configs", "default.yaml")
PER_TASK_CONFIG = os.path.join(_ROOT, "configs", "per_task.yaml")
CLASSIFIER_CONFIG = os.path.join(_ROOT, "configs", "classifier.yaml")

# --------------------------------------------------------------------------- #
# Paper reference numbers (also mirrored in dpm_ant/evaluation/*.py)
# --------------------------------------------------------------------------- #
#: Section 5.4 / Figure 4 - FID on 10-shot Sunglasses after 300 iterations.
ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    "full_finetune": {
        "only_adaptor": False,
        "freeze_backbone": False,
        "use_adv_noise": True,
        "gamma": 5.0,
        "paper_fid": 41.88,
        "description": "direct full-model fine-tuning",
    },
    "adaptor_only": {
        "only_adaptor": True,
        "freeze_backbone": True,
        "use_adv_noise": False,
        "gamma": 0.0,
        "paper_fid": 38.65,
        "description": "adaptor-only training with the vanilla DDPM loss",
    },
    "ant_wo_an": {
        "only_adaptor": True,
        "freeze_backbone": True,
        "use_adv_noise": False,
        "gamma": 5.0,
        "paper_fid": 26.41,
        "description": "DPMs-ANT without adversarial noise (similarity-guided only)",
    },
    "full_ant": {
        "only_adaptor": True,
        "freeze_backbone": True,
        "use_adv_noise": True,
        "gamma": 5.0,
        "paper_fid": 20.66,
        "description": "full DPMs-ANT (adversarial noise + similarity guidance)",
    },
}
ABLATION_ORDER: Tuple[str, ...] = ("full_finetune", "adaptor_only", "ant_wo_an", "full_ant")

#: Appendix B.3 Tables 5-7 - sweep grids and the reported optimum.
GAMMA_GRID: Tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)
OMEGA_GRID: Tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)
ITERATION_GRID: Tuple[int, ...] = (0, 100, 200, 300, 400)
SENSITIVITY_BEST: Dict[str, Any] = {
    "gamma": 5.0,
    "omega": 0.02,
    "iterations": 300,
    "fid": 18.13,
}

#: Section 5.5 Table 3 - classifier trained on 10 vs 100 target images.
CLASSIFIER_POOL_GRID: Tuple[int, ...] = (10, 100)
PAPER_CLASSIFIER_ABLATION: Dict[int, Dict[str, float]] = {
    10: {"intra_lpips": 0.613, "intra_lpips_std": 0.023, "fid": 20.06},
    100: {"intra_lpips": 0.637, "intra_lpips_std": 0.013, "fid": 22.84},
}

#: Section 5.2 hyper-parameters reused as the sweep baseline.
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
    "num_timesteps": 1000,
    "schedule": "linear",
    "eta": 0.0,
    "shots": 10,
    "num_samples": 1000,
    "num_steps": 100,
    "method": "ddim",
    "sampling_eta": 0.0,
    "lpips_net": "alex",
    "fid_backend": "clean-fid",
}

# --------------------------------------------------------------------------- #
# Optional YAML
# --------------------------------------------------------------------------- #
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Best-effort YAML loader; returns ``{}`` when the file/dep is missing."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - pyyaml is a normal dependency
        LOGGER.warning("PyYAML is not installed; ignoring %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - malformed config
        LOGGER.warning("Failed to read config %s: %s", path, exc)
        return {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (``override`` wins)."""
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: Optional[str] = None,
    per_task_path: Optional[str] = None,
    classifier_path: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge ``default.yaml`` + ``per_task.yaml`` (+ ``classifier.yaml``).

    The per-task ``defaults`` block is flattened into the ``ant`` / ``adaptor``
    blocks, mirroring :mod:`scripts.train_ant` so hyper-parameter precedence is
    identical between the training script and this runner.
    """
    cfg = _load_yaml(config_path or DEFAULT_CONFIG)
    per_task = _load_yaml(per_task_path or PER_TASK_CONFIG)
    cls_cfg = _load_yaml(classifier_path or CLASSIFIER_CONFIG)

    defaults = per_task.get("defaults") or {}
    cfg = _deep_update(cfg, {k: v for k, v in per_task.items() if k != "defaults"})

    ant_block = _deep_update(cfg.get("ant", {}) or {}, _as_dict(defaults))
    adaptor_block = _deep_update(
        cfg.get("adaptor", {}) or {}, _as_dict(defaults.get("adaptor"))
    )
    cfg["ant"] = ant_block
    cfg["adaptor"] = adaptor_block
    if cls_cfg:
        cfg = _deep_update(cfg, cls_cfg)

    if task:
        cfg["task_cfg"] = _task_entry(cfg, task)
        cfg["task"] = task
    return cfg


def _as_dict(obj: Any) -> Dict[str, Any]:
    return dict(obj) if isinstance(obj, dict) else {}


def _task_entry(cfg: Dict[str, Any], task: Optional[str]) -> Dict[str, Any]:
    if not task:
        return {}
    tasks = cfg.get("tasks") or {}
    entry = tasks.get(task)
    return dict(entry) if isinstance(entry, dict) else {}


def resolve_backbone(cfg: Dict[str, Any], task: Optional[str], cli_backbone: Optional[str]) -> str:
    """Resolve ``ddpm`` / ``ldm`` (CLI > per-task entry > name heuristic)."""
    if cli_backbone:
        return str(cli_backbone).lower()
    entry = _task_entry(cfg, task)
    if entry.get("backbone"):
        return str(entry["backbone"]).lower()
    if task and str(task).lower().endswith("_ldm"):
        return "ldm"
    return "ddpm"


def resolve_target(cfg: Dict[str, Any], task: Optional[str], cli_target: Optional[str]) -> Optional[str]:
    if cli_target:
        return cli_target
    entry = _task_entry(cfg, task)
    return entry.get("target")


def resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch  # noqa: WPS433 (lazy import)

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def set_seed(seed: int) -> None:
    try:
        import random

        import numpy as np
        import torch
    except Exception:  # pragma: no cover
        return
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Training / sampling / evaluation helpers
# --------------------------------------------------------------------------- #
def build_ant_overrides(
    variant: Optional[str] = None,
    gamma: Optional[float] = None,
    omega: Optional[float] = None,
    J: Optional[int] = None,
    iterations: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose the ANT override dict for one sweep point / ablation variant."""
    overrides: Dict[str, Any] = dict(ANT_DEFAULTS)
    if variant:
        spec = ABLATION_VARIANTS.get(variant)
        if spec is None:
            raise KeyError(
                "unknown ablation variant %r (choose from %s)"
                % (variant, ", ".join(ABLATION_ORDER))
            )
        overrides.update({k: v for k, v in spec.items() if k not in ("paper_fid", "description")})
    if gamma is not None:
        overrides["gamma"] = float(gamma)
    if omega is not None:
        overrides["omega"] = float(omega)
    if J is not None:
        overrides["J"] = int(J)
    if iterations is not None:
        overrides["iterations"] = int(iterations)
    overrides.pop("shots", None)
    overrides.pop("num_samples", None)
    overrides.pop("num_steps", None)
    overrides.pop("method", None)
    overrides.pop("sampling_eta", None)
    overrides.pop("lpips_net", None)
    overrides.pop("fid_backend", None)
    if extra:
        overrides.update(extra)
    return overrides


def _import_train_ant_helpers():
    """Import the shared builders from :mod:`scripts.train_ant`."""
    import importlib

    for name in ("scripts.train_ant", "train_ant"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    # File-based fallback (project not installed as a package).
    import importlib.util

    path = os.path.join(_HERE, "train_ant.py")
    if os.path.isfile(path):
        spec = importlib.util.spec_from_file_location("_dpm_ant_train_ant", path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module
    raise ImportError("could not import scripts/train_ant.py")


def _train_one(
    cfg: Dict[str, Any],
    task: str,
    backbone: str,
    overrides: Dict[str, Any],
    device: str,
    classifier_path: Optional[str] = None,
    target_dir: Optional[str] = None,
    shots: int = 10,
    diffusion_checkpoint: Optional[str] = None,
    seed: int = 0,
    adaptor_init: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train one adaptor configuration and return the adapted model + context."""
    import torch  # noqa: WPS433

    helpers = _import_train_ant_helpers()
    model, autoencoder = helpers.build_backbone(
        cfg,
        backbone,
        task,
        device,
        checkpoint=diffusion_checkpoint,
        verbose=verbose,
    )
    classifier = None
    if str(overrides.get("gamma", 5.0)) not in ("0", "0.0", 0, 0.0):
        classifier = helpers.build_classifier_for_task(
            cfg, backbone, classifier_path, device, verbose=verbose
        )
    if classifier is None and overrides.get("gamma"):
        LOGGER.warning("No classifier available; disabling similarity guidance (gamma=0).")
        overrides = dict(overrides)
        overrides["gamma"] = 0.0

    target_data = helpers.build_target_data(
        cfg,
        backbone,
        resolve_target(cfg, task, target_dir),
        target_dir,
        shots,
        1,
        None,
        autoencoder=autoencoder,
        device=device,
        seed=seed,
    )

    if adaptor_init and os.path.isfile(adaptor_init):
        try:  # pragma: no cover - depends on trained checkpoints
            state = torch.load(adaptor_init, map_location=device)
            state = state.get("adaptor", state) if isinstance(state, dict) else state
            model.load_state_dict(state, strict=False)
            LOGGER.info("Initialised adaptors from %s", adaptor_init)
        except Exception as exc:
            LOGGER.warning("Could not load adaptor init %s: %s", adaptor_init, exc)

    from dpm_ant.training.ant_trainer import train_ant

    started = time.time()
    trainer, history = train_ant(
        model,
        target_data,
        classifier=classifier,
        cfg=cfg,
        backbone=backbone,
        task=task,
        iterations=int(overrides.get("iterations", ANT_DEFAULTS["iterations"])),
        device=device,
        verbose=verbose,
        **overrides,
    )
    elapsed = time.time() - started

    rate = None
    try:
        rate = float(trainer.parameter_rate())
    except Exception:
        try:
            from dpm_ant.evaluation.metrics import adaptor_parameter_rate

            rate = adaptor_parameter_rate(model, backbone=backbone).get("parameter_rate")
        except Exception:
            rate = None

    return {
        "model": model,
        "autoencoder": autoencoder,
        "classifier": classifier,
        "trainer": trainer,
        "history": history,
        "train_seconds": elapsed,
        "parameter_rate": rate,
        "final_loss": (history[-1].get("loss") if history else None),
    }


def _generate(
    model,
    autoencoder,
    classifier,
    cfg: Dict[str, Any],
    backbone: str,
    task: str,
    device: str,
    num_samples: int,
    num_steps: int,
    method: str,
    eta: float,
    seed: int,
    batch_size: int = 16,
    guidance_scale: Optional[float] = None,
) -> "Any":
    """Sample ``num_samples`` images with the adapted reverse process."""
    from dpm_ant.sampling.sampler import build_sampler

    sampler = build_sampler(
        cfg=cfg,
        model=model,
        classifier=classifier,
        autoencoder=autoencoder,
        backbone=backbone,
        task=task,
        device=device,
        method=method,
        num_steps=num_steps,
        eta=eta,
        batch_size=batch_size,
        use_classifier_guidance=guidance_scale is not None,
        guidance_scale=guidance_scale if guidance_scale is not None else 5.0,
    )
    images = sampler.sample(num_samples=num_samples, num_steps=num_steps, seed=seed, verbose=False)
    if isinstance(images, tuple):  # (latents, images) style return
        images = images[-1]
    return images


def _reference_tensors(
    cfg: Dict[str, Any],
    backbone: str,
    target: Optional[str],
    shots: int,
    want_fid: bool,
) -> Tuple[Optional["Any"], Optional["Any"], Optional[str]]:
    """Return ``(train_images, fid_reference, target_dir)`` for evaluation."""
    try:
        from dpm_ant.data.datasets import build_fid_dataset, build_target_dataset
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("data pipeline unavailable: %s", exc)
        return None, None, None

    train_images, fid_reference, target_dir = None, None, None
    try:
        ds = build_target_dataset(cfg=cfg, target=target, shots=shots, backbone=backbone)
        train_images = _dataset_tensor(ds)
        if isinstance(train_images, list) and train_images:
            import torch

            train_images = torch.stack([t.reshape(3, -1).shape and t for t in train_images])  # type: ignore
    except Exception as exc:
        LOGGER.warning("Could not build %d-shot reference set: %s", shots, exc)

    if want_fid:
        try:
            ds = build_fid_dataset(cfg=cfg, target=target, backbone=backbone)
            fid_reference = _dataset_tensor(ds)
        except Exception as exc:
            LOGGER.warning("Could not build FID reference set for %s: %s", target, exc)
    return train_images, fid_reference, target_dir


def _dataset_tensor(dataset) -> Optional["Any"]:
    """Materialise a dataset of 10-100 items into a tensor (None on failure)."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return None
    if dataset is None:
        return None
    try:
        items = []
        limit = min(len(dataset), 5000)
        for i in range(limit):
            item = dataset[i]
            if isinstance(item, (tuple, list)):
                item = item[0]
            items.append(item)
        if not items:
            return None
        return torch.stack([t.detach() for t in items])
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Could not materialise dataset: %s", exc)
        return None


def _evaluate(
    images,
    cfg: Dict[str, Any],
    backbone: str,
    target: Optional[str],
    train_images,
    fid_reference,
    device: str,
    lpips_net: str,
    fid_backend: str,
    compute_intra_lpips: bool,
    compute_fid: bool,
) -> Dict[str, Any]:
    """Score generated images with Intra-LPIPS and FID (both best-effort)."""
    out: Dict[str, Any] = {"num_generated": int(len(images)) if images is not None else 0}
    if images is None:
        return out

    if compute_intra_lpips and train_images is not None:
        try:
            from dpm_ant.evaluation.intra_lpips import compute_intra_lpips as _il

            res = _il(images, train_images=train_images, net=lpips_net, device=device)
            out["intra_lpips"] = float(res.get("intra_lpips", float("nan")))
            out["intra_lpips_details"] = {
                k: res[k]
                for k in ("num_train", "num_nonempty_clusters", "mean_nearest_distance")
                if k in res
            }
        except Exception as exc:
            LOGGER.warning("Intra-LPIPS failed: %s", exc)
            out["intra_lpips_error"] = str(exc)

    if compute_fid and fid_reference is not None:
        try:
            from dpm_ant.evaluation.fid import compute_fid as _fid

            res = _fid(
                images,
                reference=fid_reference,
                backend=fid_backend,
                device=device,
                target=target,
            )
            out["fid"] = float(res.get("fid", float("nan"))) if isinstance(res, dict) else float(res)
        except Exception as exc:
            LOGGER.warning("FID failed: %s", exc)
            out["fid_error"] = str(exc)
    return out


# --------------------------------------------------------------------------- #
# Study runners
# --------------------------------------------------------------------------- #
def run_configuration(
    cfg: Dict[str, Any],
    task: str,
    backbone: str,
    device: str,
    overrides: Dict[str, Any],
    label: str,
    num_samples: int = 1000,
    num_steps: int = 100,
    method: str = "ddim",
    eta: float = 0.0,
    shots: int = 10,
    seed: int = 0,
    lpips_net: str = "alex",
    fid_backend: str = "clean-fid",
    compute_intra_lpips: bool = True,
    compute_fid: bool = True,
    classifier_path: Optional[str] = None,
    target_dir: Optional[str] = None,
    diffusion_checkpoint: Optional[str] = None,
    adaptor_init: Optional[str] = None,
    save_samples: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train, sample and score a single configuration; never raises."""
    record: Dict[str, Any] = {
        "label": label,
        "task": task,
        "backbone": backbone,
        "overrides": {k: v for k, v in overrides.items()},
        "seed": seed,
        "num_samples": num_samples,
        "num_steps": num_steps,
        "method": method,
        "eta": eta,
    }
    try:
        set_seed(seed)
        trained = _train_one(
            cfg,
            task,
            backbone,
            overrides,
            device,
            classifier_path=classifier_path,
            target_dir=target_dir,
            shots=shots,
            diffusion_checkpoint=diffusion_checkpoint,
            seed=seed,
            adaptor_init=adaptor_init,
            verbose=verbose,
        )
        record["train_seconds"] = trained["train_seconds"]
        record["parameter_rate"] = trained["parameter_rate"]
        record["final_loss"] = trained["final_loss"]

        target = resolve_target(cfg, task, target_dir)
        train_images, fid_reference, _ = _reference_tensors(cfg, backbone, target, shots, compute_fid)

        images = _generate(
            trained["model"],
            trained["autoencoder"],
            trained["classifier"],
            cfg,
            backbone,
            task,
            device,
            num_samples=num_samples,
            num_steps=num_steps,
            method=method,
            eta=eta,
            seed=seed,
        )
        if save_samples:
            try:
                from dpm_ant.evaluation.fid import save_images

                os.makedirs(save_samples, exist_ok=True)
                save_images(images, save_samples, prefix=label)
                record["sample_dir"] = save_samples
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Could not save samples: %s", exc)

        metrics = _evaluate(
            images,
            cfg,
            backbone,
            target,
            train_images,
            fid_reference,
            device,
            lpips_net,
            fid_backend,
            compute_intra_lpips,
            compute_fid,
        )
        record.update(metrics)
    except Exception as exc:  # pragma: no cover - robustness for long sweeps
        LOGGER.error("Configuration %s failed: %s", label, exc)
        record["error"] = str(exc)
    return record


def run_ablation(
    cfg: Dict[str, Any],
    task: str = "ffhq_sunglasses",
    backbone: str = "ddpm",
    device: str = "cpu",
    variants: Sequence[str] = ABLATION_ORDER,
    iterations: int = 300,
    seeds: Sequence[int] = (0,),
    verbose: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Section 5.4 / Figure 4 - ablation over variants on 10-shot Sunglasses."""
    results: List[Dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            overrides = build_ant_overrides(variant=variant, iterations=iterations)
            label = f"{variant}_iter{iterations}_seed{seed}"
            LOGGER.info("[ablation] %s - %s", task, label)
            results.append(
                run_configuration(
                    cfg,
                    task,
                    backbone,
                    device,
                    overrides,
                    label=label,
                    seed=seed,
                    verbose=verbose,
                    **kwargs,
                )
            )
    results.sort(key=lambda r: (str(r.get("label", "")).split("_iter")[0], r.get("seed", 0)))
    _log_comparison(results, "paper_fid" if False else ABLATION_VARIANTS, key="label")
    return results


def run_sensitivity(
    cfg: Dict[str, Any],
    task: str = "ffhq_sunglasses",
    backbone: str = "ddpm",
    device: str = "cpu",
    gammas: Sequence[float] = GAMMA_GRID,
    omegas: Sequence[float] = OMEGA_GRID,
    iteration_grid: Sequence[int] = ITERATION_GRID,
    iterations: int = 300,
    seeds: Sequence[int] = (0,),
    verbose: bool = True,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Section 5.5 / Appendix B.3 - gamma, omega and iteration sweeps."""
    results: List[Dict[str, Any]] = []
    for gamma in gammas:
        for seed in seeds:
            for omega in [None]:
                overrides = build_ant_overrides(gamma=gamma)
                label = f"gamma_{gamma:g}_seed{seed}"
                LOGGER.info("[sensitivity] %s - %s", task, label)
                results.append(
                    run_configuration(
                        cfg, task, backbone, device, overrides, label=label,
                        seed=seed, verbose=verbose, **kwargs,
                    )
                )
    for omega in omegas:
        for seed in seeds:
            overrides = build_ant_overrides(omega=omega)
            label = f"omega_{omega:g}_seed{seed}"
            LOGGER.info("[sensitivity] %s - %s", task, label)
            results.append(
                run_configuration(
                    cfg, task, backbone, device, overrides, label=label,
                    seed=seed, verbose=verbose, **kwargs,
                )
            )
    for iters in iteration_grid:
        for seed in seeds:
            overrides = build_ant_overrides(iterations=iters)
            label = f"iterations_{iters}_seed{seed}"
            LOGGER.info("[sensitivity] %s - %s", task, label)
            results.append(
                run_configuration(
                    cfg, task, backbone, device, overrides, label=label,
                    seed=seed, verbose=verbose, **kwargs,
                )
            )
    summarize_sensitivity(results)
    return results


def run_classifier_ablation(
    cfg: Dict[str, Any],
    task: str = "ffhq_sunglasses",
    backbone: str = "ddpm",
    device: str = "cpu",
    pools: Sequence[int] = CLASSIFIER_POOL_GRID,
    iterations: int = 300,
    seeds: Sequence[int] = (0,),
    verbose: bool = True,
    classifier_checkpoint_pattern: str = "outputs/classifier/classifier_{task}_{backbone}_pool{pool}.pt",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Section 5.5 / Table 3 - classifier trained on ``pool`` target images.

    When the corresponding fine-tuned classifier checkpoint exists it is used;
    otherwise the study records the paper reference numbers together with the
    ANT result obtained from the currently available classifier.
    """
    results: List[Dict[str, Any]] = []
    for pool in pools:
        for seed in seeds:
            ckpt = classifier_checkpoint_pattern.format(task=task, backbone=backbone, pool=pool)
            if not os.path.isfile(ckpt):
                LOGGER.warning(
                    "Classifier checkpoint for pool=%s not found at %s; run "
                    "scripts/train_classifier.py --target-pool %s first. Using default "
                    "classifier checkpoint.",
                    pool,
                    ckpt,
                    pool,
                )
                ckpt = None
            overrides = build_ant_overrides(variant="full_ant", iterations=iterations)
            label = f"classifier_pool{pool}_seed{seed}"
            LOGGER.info("[classifier] %s - %s", task, label)
            record = run_configuration(
                cfg,
                task,
                backbone,
                device,
                overrides,
                label=label,
                shots=int(pool),
                seed=seed,
                classifier_path=ckpt,
                verbose=verbose,
                **kwargs,
            )
            ref = PAPER_CLASSIFIER_ABLATION.get(int(pool), {})
            record["classifier_pool"] = int(pool)
            record["paper_intra_lpips"] = ref.get("intra_lpips")
            record["paper_intra_lpips_std"] = ref.get("intra_lpips_std")
            record["paper_fid"] = ref.get("fid")
            results.append(record)
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _mean(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _std(values: Iterable[Any]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and v == v]
    if len(nums) < 2:
        return 0.0 if nums else None
    mean = sum(nums) / len(nums)
    var = sum((x - mean) ** 2 for x in nums) / (len(nums) - 1)
    return float(var ** 0.5)


def aggregate_results(results: Sequence[Dict[str, Any]], group_key: str = "label") -> List[Dict[str, Any]]:
    """Mean +- std aggregation over seeds for each group."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in results:
        base = str(rec.get(group_key, ""))
        base = base.split("_seed")[0] if "_seed" in base else base
        groups.setdefault(base, []).append(rec)

    rows: List[Dict[str, Any]] = []
    for base, records in groups.items():
        row: Dict[str, Any] = {"group": base, "n": len(records)}
        for metric in ("intra_lpips", "fid", "parameter_rate", "train_seconds", "final_loss"):
            vals = [r.get(metric) for r in records]
            row[f"{metric}_mean"] = _mean(vals)
            row[f"{metric}_std"] = _std(vals)
        row["errors"] = [r.get("error") for r in records if r.get("error")]
        rows.append(row)
    return rows


def summarize_sensitivity(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Locate the empirical optima of the ``gamma`` / ``omega`` / iteration sweeps."""
    summary: Dict[str, Any] = {"best": {}, "paper_best": dict(SENSITIVITY_BEST)}
    for axis, prefix in (("gamma", "gamma_"), ("omega", "omega_"), ("iterations", "iterations_")):
        pairs: List[Tuple[float, float]] = []
        for rec in results:
            label = str(rec.get("label", ""))
            fid = rec.get("fid")
            metric = rec.get("intra_lpips")
            if not label.startswith(prefix) or fid is None:
                continue
            try:
                value = float(label[len(prefix):].split("_")[0])
            except ValueError:
                continue
            pairs.append((value, float(fid), float(metric) if metric is not None else float("nan")))
        if pairs:
            best = min(pairs, key=lambda p: p[1])
            summary[axis] = [
                {"value": v, "fid": f, "intra_lpips": m} for v, f, m in sorted(pairs)
            ]
            summary["best"][axis] = {"value": best[0], "fid": best[1], "intra_lpips": best[2]}
            summary["best"][axis]["paper_value"] = SENSITIVITY_BEST.get(axis)
    return summary


def _log_comparison(results: Sequence[Dict[str, Any]], reference: Dict[str, Any], key: str = "label") -> None:
    """Log measured vs. paper FID for the ablation variants."""
    for rec in results:
        label = str(rec.get(key, ""))
        variant = label.split("_iter")[0]
        spec = reference.get(variant) if isinstance(reference, dict) else None
        paper = spec.get("paper_fid") if isinstance(spec, dict) else None
        fid = rec.get("fid")
        if paper is not None and fid is not None:
            LOGGER.info(
                "%-28s FID %.2f (paper %.2f, delta %+0.2f)",
                variant, float(fid), float(paper), float(fid) - float(paper),
            )


def build_report(
    results: Sequence[Dict[str, Any]],
    mode: str,
    task: str,
    backbone: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "mode": mode,
        "task": task,
        "backbone": backbone,
        "num_configurations": len(results),
        "results": list(results),
        "aggregate": aggregate_results(results),
        "paper_reference": {
            "ablation_variants": {
                k: {"paper_fid": v["paper_fid"], "description": v["description"]}
                for k, v in ABLATION_VARIANTS.items()
            },
            "sensitivity_best": dict(SENSITIVITY_BEST),
            "classifier_ablation": PAPER_CLASSIFIER_ABLATION,
        },
    }
    if mode in ("sensitivity", "all"):
        report["sensitivity"] = summarize_sensitivity(results)
    if extra:
        report.update(extra)
    return report


def format_report(report: Dict[str, Any]) -> str:
    """Human readable rendering of the aggregated results."""
    lines = [
        "=" * 78,
        f"DPMs-ANT {report.get('mode', '')} report | task={report.get('task')} "
        f"backbone={report.get('backbone')} | {report.get('num_configurations', 0)} runs",
        "=" * 78,
        f"{'configuration':<34}{'Intra-LPIPS':>14}{'FID':>12}{'param rate':>12}",
        "-" * 78,
    ]
    for row in report.get("aggregate", []):
        il = row.get("intra_lpips_mean")
        il_std = row.get("intra_lpips_std") or 0.0
        fid = row.get("fid_mean")
        rate = row.get("parameter_rate_mean")
        il_s = f"{il:.3f}+-{il_std:.3f}" if il is not None else "n/a"
        fid_s = f"{fid:.2f}" if fid is not None else "n/a"
        rate_s = f"{rate * 100:.2f}%" if rate is not None else "n/a"
        lines.append(f"{str(row.get('group'))[:33]:<34}{il_s:>14}{fid_s:>12}{rate_s:>12}")
    lines.append("-" * 78)

    sens = report.get("sensitivity")
    if sens:
        for axis in ("gamma", "omega", "iterations"):
            best = (sens.get("best") or {}).get(axis)
            if best:
                lines.append(
                    f"sensitivity {axis:<11} best={best.get('value')} FID={best.get('fid'):.2f} "
                    f"(paper {best.get('paper_value')}, FID {SENSITIVITY_BEST['fid']:.2f})"
                )
    for row in report.get("aggregate", []):
        if row.get("errors"):
            lines.append(f"warning: {row['group']} had {len(row['errors'])} failed run(s)")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], out_dir: str, name: str = "ablation_report.json") -> str:
    """Persist JSON report + CSV table; returns the JSON path."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, name)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    csv_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".csv")
    rows = report.get("aggregate", [])
    if rows:
        keys = sorted({k for row in rows for k in row})
        try:
            with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k) for k in keys})
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not write CSV: %s", exc)
    txt_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".txt")
    try:
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(format_report(report) + "\n")
    except Exception:  # pragma: no cover
        pass
    return json_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ablation.py",
        description="DPMs-ANT ablation / sensitivity / classifier-pool experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", default="all",
                        choices=["ablation", "sensitivity", "classifier", "all"])
    parser.add_argument("--task", default="ffhq_sunglasses",
                        help="task key from configs/per_task.yaml")
    parser.add_argument("--backbone", default=None, choices=["ddpm", "ldm", None])
    parser.add_argument("--target", default=None, help="override target dataset name")
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--shots", type=int, default=10, help="target images for ANT")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--variants", nargs="*", default=list(ABLATION_ORDER),
                        choices=list(ABLATION_ORDER))
    parser.add_argument("--gammas", nargs="*", type=float, default=list(GAMMA_GRID))
    parser.add_argument("--omegas", nargs="*", type=float, default=list(OMEGA_GRID))
    parser.add_argument("--iteration-grid", nargs="*", type=int, default=list(ITERATION_GRID))
    parser.add_argument("--classifier-pools", nargs="*", type=int, default=list(CLASSIFIER_POOL_GRID))
    parser.add_argument("--classifier-checkpoint", default=None)
    parser.add_argument("--classifier-checkpoint-pattern",
                        default="outputs/classifier/classifier_{task}_{backbone}_pool{pool}.pt")
    parser.add_argument("--diffusion-checkpoint", default=None)
    parser.add_argument("--adaptor-init", default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--method", default="ddim", choices=["ddim", "ddpm"])
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--fid-backend", default="clean-fid")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="enable classifier-guided sampling with this scale")
    parser.add_argument("--no-intra-lpips", action="store_true")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--save-samples", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--per-task-config", default=PER_TASK_CONFIG)
    parser.add_argument("--classifier-config", default=CLASSIFIER_CONFIG)
    parser.add_argument("--out", default=os.path.join("outputs", "ablation"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run: 8 samples, 4 iterations, 8 steps")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config, args.per_task_config, args.classifier_config, args.task)
    backbone = resolve_backbone(cfg, args.task, args.backbone)
    device = resolve_device(args.device)
    set_seed(args.seed)

    if args.smoke:
        args.num_samples = min(args.num_samples, 8)
        args.num_steps = min(args.num_steps, 8)
        args.iterations = min(args.iterations, 4)
        args.iteration_grid = [0, 2, 4]
        args.gammas = [1.0, 5.0]
        args.omegas = [0.01, 0.02]
        args.seeds = [args.seed]

    run_kwargs: Dict[str, Any] = dict(
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        method=args.method,
        eta=args.eta,
        shots=args.shots,
        lpips_net=args.lpips_net,
        fid_backend=args.fid_backend,
        compute_intra_lpips=not args.no_intra_lpips,
        compute_fid=not args.no_fid,
        diffusion_checkpoint=args.diffusion_checkpoint,
        adaptor_init=args.adaptor_init,
        target_dir=args.target_dir,
        verbose=args.verbose,
    )

    if args.dry_run:
        print(json.dumps(
            {
                "mode": args.mode,
                "task": args.task,
                "backbone": backbone,
                "device": device,
                "variants": args.variants,
                "gammas": args.gammas,
                "omegas": args.omegas,
                "iteration_grid": args.iteration_grid,
                "classifier_pools": args.classifier_pools,
                "run_kwargs": {k: v for k, v in run_kwargs.items() if k != "verbose"},
                "out": args.out,
            },
            indent=2,
            default=str,
        ))
        return 0

    results: List[Dict[str, Any]] = []
    if args.mode in ("ablation", "all"):
        results += run_ablation(
            cfg,
            task=args.task,
            backbone=backbone,
            device=device,
            variants=args.variants,
            iterations=args.iterations,
            seeds=args.seeds,
            **run_kwargs,
        )
    if args.mode in ("sensitivity", "all"):
        results += run_sensitivity(
            cfg,
            task=args.task,
            backbone=backbone,
            device=device,
            gammas=args.gammas,
            omegas=args.omegas,
            iteration_grid=args.iteration_grid,
            seeds=args.seeds,
            **run_kwargs,
        )
    if args.mode in ("classifier", "all"):
        results += run_classifier_ablation(
            cfg,
            task=args.task,
            backbone=backbone,
            device=device,
            pools=args.classifier_pools,
            iterations=args.iterations,
            seeds=args.seeds,
            classifier_checkpoint_pattern=args.classifier_checkpoint_pattern,
            **run_kwargs,
        )

    report = build_report(results, mode=args.mode, task=args.task, backbone=backbone)
    path = save_report(report, args.out, name=f"ablation_report_{args.mode}.json")
    print(format_report(report))
    print(f"report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
