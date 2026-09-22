"""Classifier fine-tuning for DPMs-ANT (Section 4.1 / 5.2).

The similarity-guided objective of Eq. (5) requires a binary classifier
``p_phi(y | x_t)`` that distinguishes *source*-domain from *target*-domain
images at every diffusion timestep ``t``.  Following Section 5.2 (and the
addendum "Classifier Training (Section 5.2)") we

  1. load the released ImageNet diffusion classifier checkpoint
     (``256x256_classifier.pt`` for DDPM, ``64x64_classifier.pt`` for LDM),
  2. replace its final pooling/output layer with a 2-way head,
  3. and fine-tune it on *noised* source and target images drawn with
     ``t ~ Uniform({1, ..., T})``,

using Adam with ``lr = 1e-4``, ``batch size = 64`` and ``300`` iterations.
The classifier is then frozen (``phi`` fixed) during ANT adaptation.

Convention throughout this module (matches the rest of the package):
``y == 0`` means *source*, ``y == 1`` means *target* (``target_index``).
"""

from __future__ import annotations

import copy
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion.schedule import NoiseSchedule, build_schedule

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ClassifierTrainConfig",
    "ClassifierTrainer",
    "noised_source_target_batch",
    "fine_tune_classifier",
    "train_classifier",
    "classifier_accuracy",
    "build_classifier_trainer",
]

SOURCE_LABEL = 0
TARGET_LABEL = 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ClassifierTrainConfig:
    """Hyper-parameters for classifier fine-tuning (Section 5.2)."""

    # optimisation
    lr: float = 1e-4
    batch_size: int = 64
    iterations: int = 300
    optimizer: str = "adam"
    adam_betas: Sequence[float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    momentum: float = 0.9

    # noise / data
    num_timesteps: int = 1000
    schedule: str = "linear"
    eta: float = 0.0
    t_sampling: str = "uniform_1_T"
    noise_frac_source: float = 0.5
    source_pool: Optional[int] = None
    target_pool: int = 10

    # model
    num_classes: int = 2
    target_index: int = TARGET_LABEL
    replace_head: bool = True
    input_mode: str = "pixel"

    # runtime
    device: Optional[str] = None
    seed: int = 0
    log_interval: int = 50
    save_interval: int = 0
    freeze_after_training: bool = True
    verbose: bool = True

    # extras
    backbone: str = "ddpm"

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["adam_betas"] = list(self.adam_betas)
        return out

    def replace(self, **overrides) -> "ClassifierTrainConfig":
        data = dict(self.__dict__)
        data.update({k: v for k, v in overrides.items() if v is not None})
        return ClassifierTrainConfig(**data)

    @classmethod
    def from_dict(
        cls,
        cfg: Optional[Dict[str, Any]] = None,
        backbone: str = "ddpm",
        task: Optional[str] = None,
        **overrides,
    ) -> "ClassifierTrainConfig":
        """Build from a (possibly nested) YAML config dictionary."""
        cfg = dict(cfg or {})
        flat: Dict[str, Any] = {}

        # nested blocks used by configs/default.yaml + configs/per_task.yaml
        for block in ("classifier", "defaults", "ant"):
            node = cfg.get(block)
            if isinstance(node, dict):
                flat.update(node)
        if isinstance(cfg.get("diffusion"), dict):
            diff = cfg["diffusion"]
            flat.setdefault("num_timesteps", diff.get("num_timesteps", diff.get("T")))
            flat.setdefault("schedule", diff.get("schedule"))
            flat.setdefault("eta", diff.get("eta"))
        tasks = cfg.get("tasks")
        if isinstance(tasks, dict) and task and isinstance(tasks.get(task), dict):
            flat.update(tasks[task])
        # top level keys that are not nested containers
        for key, value in cfg.items():
            if not isinstance(value, dict):
                flat.setdefault(key, value)

        def pick(*names, default=None):
            for name in names:
                if name in flat and flat[name] is not None:
                    return flat[name]
            return default

        params: Dict[str, Any] = {}
        params["lr"] = float(pick("lr", default=cls.lr))
        params["batch_size"] = int(pick("batch_size", default=cls.batch_size))
        params["iterations"] = int(pick("iterations", default=cls.iterations))
        params["optimizer"] = str(pick("optimizer", default=cls.optimizer))
        betas = pick("adam_betas", "betas")
        if betas is not None:
            params["adam_betas"] = tuple(float(b) for b in betas)
        params["weight_decay"] = float(pick("weight_decay", default=cls.weight_decay))
        params["grad_clip"] = pick("grad_clip", default=cls.grad_clip)
        params["num_timesteps"] = int(pick("num_timesteps", "T", default=cls.num_timesteps))
        params["schedule"] = str(pick("schedule", "beta_schedule", default=cls.schedule))
        params["eta"] = float(pick("eta", default=cls.eta))
        params["t_sampling"] = str(pick("t_sampling", default=cls.t_sampling))
        params["noise_frac_source"] = float(
            pick("noise_frac_source", default=cls.noise_frac_source)
        )
        params["source_pool"] = pick("source_pool", default=cls.source_pool)
        params["target_pool"] = int(
            pick("target_pool", "target_pool_ablation", default=cls.target_pool) or cls.target_pool
        )
        params["num_classes"] = int(pick("num_classes", default=cls.num_classes))
        params["target_index"] = int(pick("target_index", default=cls.target_index))
        params["device"] = pick("device", default=cls.device)
        params["seed"] = int(pick("seed", default=cls.seed))
        params["log_interval"] = int(pick("log_interval", default=cls.log_interval))
        params["save_interval"] = int(pick("save_interval", default=cls.save_interval))
        params["backbone"] = backbone

        params.update({k: v for k, v in overrides.items() if v is not None})
        valid = {k: v for k, v in params.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ---------------------------------------------------------------------------
# Noise construction helpers
# ---------------------------------------------------------------------------
def _extract(arr: torch.Tensor, t: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    """Gather per-sample values from ``arr`` (length >= T+1) and broadcast."""
    arr = arr.to(t.device)
    while t.dim() < 1:
        t = t.unsqueeze(0)
    out = arr[t.long()]
    while out.dim() < ndim:
        out = out.unsqueeze(-1)
    return out


def noised_source_target_batch(
    source_images: torch.Tensor,
    target_images: torch.Tensor,
    schedule: NoiseSchedule,
    num_timesteps: Optional[int] = None,
    noise_frac_source: float = 0.5,
    generator: Optional[torch.Generator] = None,
    t_min: int = 1,
    return_t: bool = False,
):
    """Build a batch of noised source/target images with pseudo labels.

    For every image we draw ``t ~ Uniform({1, ..., T})`` and form

        x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps

    The batch concatenates (a fraction of) source images labelled 0 with
    target images labelled 1.  ``noise_frac_source`` controls how many of the
    images come from the source distribution (0.5 => balanced batch).
    """
    device = source_images.device
    B = source_images.shape[0]

    if noise_frac_source <= 0.0:
        images = target_images
        labels = torch.full((images.shape[0],), TARGET_LABEL, device=device, dtype=torch.long)
    elif noise_frac_source >= 1.0:
        images = source_images
        labels = torch.full((images.shape[0],), SOURCE_LABEL, device=device, dtype=torch.long)
    else:
        n_src = max(1, int(round(B * noise_frac_source)))
        n_src = min(n_src, source_images.shape[0])
        n_tgt = B - n_src
        if n_tgt > target_images.shape[0]:
            # repeat the few-shot target set if needed
            reps = int(math.ceil(n_tgt / max(1, target_images.shape[0])))
            target_images = target_images.repeat(reps, 1, 1, 1)
        if n_src > source_images.shape[0]:
            reps = int(math.ceil(n_src / max(1, source_images.shape[0])))
            source_images = source_images.repeat(reps, 1, 1, 1)
        images = torch.cat([source_images[:n_src], target_images[:n_tgt]], dim=0)
        labels = torch.cat(
            [
                torch.full((n_src,), SOURCE_LABEL, device=device, dtype=torch.long),
                torch.full((n_tgt,), TARGET_LABEL, device=device, dtype=torch.long),
            ],
            dim=0,
        )

    T = int(num_timesteps or schedule.num_timesteps)
    t = torch.randint(
        low=int(t_min),
        high=T + 1,
        size=(images.shape[0],),
        device=device,
        generator=generator,
    )
    noise = torch.randn(
        images.shape, device=device, dtype=images.dtype, generator=generator
    )
    x_t = schedule.q_sample(images, t, noise)

    if return_t:
        return x_t, t, labels, noise
    return x_t, labels


def noise_at_t(x0: torch.Tensor, t: torch.Tensor, schedule: NoiseSchedule,
               noise: Optional[torch.Tensor] = None):
    """Deterministic-ish forward noising helper used for evaluation."""
    if noise is None:
        noise = torch.randn_like(x0)
    return schedule.q_sample(x0, t, noise), noise


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class ClassifierTrainer:
    """Fine-tunes a 2-way source/target classifier on noised images."""

    def __init__(
        self,
        classifier: nn.Module,
        config: Optional[ClassifierTrainConfig] = None,
        schedule: Optional[NoiseSchedule] = None,
        source_data: Optional[torch.Tensor] = None,
        target_data: Optional[torch.Tensor] = None,
        source_loader: Optional[Iterable] = None,
        target_loader: Optional[Iterable] = None,
        device: Optional[str] = None,
        **overrides,
    ) -> None:
        self.config = (
            config
            if config is not None
            else ClassifierTrainConfig(**{k: v for k, v in overrides.items()
                                         if k in ClassifierTrainConfig.__dataclass_fields__})
        )
        if overrides and config is not None:
            self.config = config.replace(**overrides)

        self.device = torch.device(
            device or self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.classifier = classifier.to(self.device)
        self.schedule = schedule or build_schedule(
            {
                "num_timesteps": self.config.num_timesteps,
                "schedule": self.config.schedule,
                "eta": self.config.eta,
            }
        )
        self.schedule.to(self.device)

        self.source_data = source_data
        self.target_data = target_data
        self.source_loader = source_loader
        self.target_loader = target_loader

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.history: List[Dict[str, float]] = []
        self.step = 0

    # -- data ------------------------------------------------------------
    def _batch_from_tensor(self, data: torch.Tensor, n: int,
                           generator: Optional[torch.Generator] = None) -> torch.Tensor:
        idx = torch.randint(0, data.shape[0], (n,), device=data.device, generator=generator)
        return data[idx]

    def _next(self, loader: Optional[Iterable], fallback: Optional[torch.Tensor],
              n: int, generator: Optional[torch.Generator]) -> Optional[torch.Tensor]:
        if loader is not None:
            try:
                batch = next(iter(loader)) if not hasattr(loader, "__next__") else next(loader)
            except StopIteration:
                return None
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            return batch.to(self.device)
        if fallback is not None:
            return self._batch_from_tensor(fallback, n, generator)
        return None

    def sample_batch(
        self,
        batch_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(batch_size or self.config.batch_size)
        n_src = max(1, int(round(B * self.config.noise_frac_source))) \
            if 0.0 < self.config.noise_frac_source < 1.0 else B
        src = self._next(self.source_loader, self.source_data, n_src, generator)
        tgt = self._next(self.target_loader, self.target_data, B, generator)

        if src is None and tgt is None:
            raise RuntimeError(
                "ClassifierTrainer needs source and/or target data "
                "(tensors or loaders)."
            )
        if tgt is None:
            tgt = src
        if src is None:
            src = tgt

        x_t, labels = noised_source_target_batch(
            src,
            tgt,
            self.schedule,
            num_timesteps=self.config.num_timesteps,
            noise_frac_source=self.config.noise_frac_source,
            generator=generator,
        )
        return x_t, labels

    # -- optimisation ----------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.classifier.parameters() if p.requires_grad]
        name = (self.config.optimizer or "adam").lower()
        if name == "adamw":
            return torch.optim.AdamW(
                params, lr=self.config.lr, betas=tuple(self.config.adam_betas),
                weight_decay=self.config.weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(
                params, lr=self.config.lr, momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        return torch.optim.Adam(
            params, lr=self.config.lr, betas=tuple(self.config.adam_betas),
            weight_decay=self.config.weight_decay,
        )

    def _forward_logits(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Call the classifier under a uniform interface."""
        if hasattr(self.classifier, "forward_logits"):
            out = self.classifier.forward_logits(x_t, t)
        else:
            try:
                out = self.classifier(x_t, t)
            except TypeError:
                out = self.classifier(x_t, t.float() / self.config.num_timesteps)
        if isinstance(out, dict):
            for key in ("logits", "out", "logit"):
                if key in out:
                    out = out[key]
                    break
        elif isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:  # attention pooling variants may keep spatial dims
            out = out.mean(dim=(-1, -2)) if out.shape[-1] == out.shape[-2] else out.flatten(1)
        if out.shape[-1] != self.config.num_classes:
            out = out[..., : self.config.num_classes]
        return out

    def train_step(
        self,
        batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, float]:
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        if batch is None:
            batch = self.sample_batch(generator=generator)
        x_t, labels = batch
        x_t = x_t.to(self.device)
        labels = labels.to(self.device)

        self.classifier.train()
        logits = self._forward_logits(x_t, torch.full(
            (x_t.shape[0],), self.config.num_timesteps // 2, device=x_t.device,
            dtype=torch.long,
        ) if False else None) if False else None  # placeholder (never executed)

        # timesteps are reconstructed from the noise level of x_t itself;
        # for API uniformity we pass t explicitly when available.
        t = getattr(self, "_last_t", None)
        if t is None:
            t = torch.randint(
                1, self.config.num_timesteps + 1, (x_t.shape[0],),
                device=x_t.device, generator=generator,
            )
        logits = self._forward_logits(x_t, t)

        loss = F.cross_entropy(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.grad_clip:
            nn.utils.clip_grad_norm_(self.classifier.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.step += 1

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        return {"loss": float(loss.item()), "acc": acc, "step": float(self.step)}

    # -- public API ------------------------------------------------------
    def train(
        self,
        source_data: Optional[torch.Tensor] = None,
        target_data: Optional[torch.Tensor] = None,
        iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        callback=None,
        verbose: Optional[bool] = None,
    ) -> List[Dict[str, float]]:
        if source_data is not None:
            self.source_data = source_data
        if target_data is not None:
            self.target_data = target_data
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()

        n_iter = int(iterations or self.config.iterations)
        B = int(batch_size or self.config.batch_size)
        verbose = self.config.verbose if verbose is None else verbose
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.config.seed))

        t0 = time.time()
        for it in range(n_iter):
            x_t, labels = self.sample_batch(batch_size=B, generator=generator)
            # keep t consistent between sampling and forward
            T = int(self.config.num_timesteps)
            t = torch.randint(1, T + 1, (x_t.shape[0],), device=self.device, generator=generator)
            x0 = self._last_x0 if getattr(self, "_last_x0", None) is not None else None
            logits = self._forward_logits(x_t, t)
            labels = labels.to(self.device)
            loss = F.cross_entropy(logits, labels)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip:
                nn.utils.clip_grad_norm_(self.classifier.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.step += 1

            with torch.no_grad():
                acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            record = {"loss": float(loss.item()), "acc": acc, "step": float(self.step)}
            self.history.append(record)

            if verbose and (it % max(1, self.config.log_interval) == 0 or it == n_iter - 1):
                LOGGER.info(
                    "[classifier] iter %d/%d loss=%.4f acc=%.3f",
                    it + 1, n_iter, record["loss"], record["acc"],
                )
            if self.config.save_interval and (it + 1) % self.config.save_interval == 0:
                try:
                    self.save(os.path.join("checkpoints", f"classifier_step{self.step}.pt"))
                except Exception:  # pragma: no cover - best effort
                    pass
            if callback is not None:
                callback(it, record)

        LOGGER.info("[classifier] fine-tuning finished in %.1fs", time.time() - t0)
        if self.config.freeze_after_training:
            self.freeze()
        return self.history

    def freeze(self) -> None:
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.classifier.eval()

    def unfreeze(self) -> None:
        for p in self.classifier.parameters():
            p.requires_grad_(True)

    def accuracy(
        self,
        x_t: torch.Tensor,
        labels: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> float:
        self.classifier.eval()
        with torch.no_grad():
            if t is None:
                t = torch.full(
                    (x_t.shape[0],), self.config.num_timesteps // 2,
                    device=x_t.device, dtype=torch.long,
                )
            logits = self._forward_logits(x_t.to(self.device), t.to(self.device))
            return float((logits.argmax(-1) == labels.to(self.device)).float().mean())

    # -- persistence -----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "classifier": self.classifier.state_dict(),
            "config": self.config.to_dict(),
            "step": self.step,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        LOGGER.info("[classifier] saved -> %s", path)
        return path

    def load(self, path: str, device: Optional[str] = None) -> "ClassifierTrainer":
        state = torch.load(path, map_location=device or self.device)
        if isinstance(state, dict) and "classifier" in state:
            self.classifier.load_state_dict(state["classifier"], strict=False)
            self.step = int(state.get("step", 0))
        else:
            self.classifier.load_state_dict(state, strict=False)
        return self


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------
def classifier_accuracy(
    classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    schedule: Optional[NoiseSchedule] = None,
    num_timesteps: int = 1000,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> float:
    """Sanity-check accuracy of a classifier across random timesteps."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    schedule = schedule or build_schedule({"num_timesteps": num_timesteps})
    schedule.to(device)
    n = min(batch_size // 2 if batch_size > 1 else 1, source_data.shape[0],
            target_data.shape[0])
    n = max(1, n)
    src = source_data[:n].to(device)
    tgt = target_data[:n].to(device)
    x_t, labels = noised_source_target_batch(
        src, tgt, schedule, num_timesteps=num_timesteps, noise_frac_source=0.5
    )
    classifier = classifier.to(device).eval()
    with torch.no_grad():
        try:
            logits = classifier(x_t, torch.full(
                (x_t.shape[0],), num_timesteps // 2, device=device, dtype=torch.long
            ))
        except TypeError:
            logits = classifier(x_t)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.dim() > 2:
            logits = logits.mean(dim=(-1, -2))
        preds = logits.argmax(-1)
        return float((preds == labels).float().mean())


def fine_tune_classifier(
    classifier: nn.Module,
    source_data: torch.Tensor,
    target_data: torch.Tensor,
    config: Optional[ClassifierTrainConfig] = None,
    cfg: Optional[Dict[str, Any]] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    device: Optional[str] = None,
    iterations: Optional[int] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
    **overrides,
) -> Tuple[ClassifierTrainer, List[Dict[str, float]]]:
    """Fine-tune ``classifier`` on noised source/target images.

    Mirrors the addendum recipe: Adam, ``lr=1e-4``, ``batch_size=64``,
    ``300`` iterations, ``t ~ Uniform({1, ..., T})``.
    """
    if config is None:
        config = ClassifierTrainConfig.from_dict(cfg, backbone=backbone, task=task)
    if iterations is not None:
        config = config.replace(iterations=iterations)
    if lr is not None:
        config = config.replace(lr=lr)
    if batch_size is not None:
        config = config.replace(batch_size=batch_size)
    if overrides:
        config = config.replace(**overrides)

    trainer = ClassifierTrainer(
        classifier=classifier,
        config=config,
        source_data=source_data,
        target_data=target_data,
        device=device,
    )
    history = trainer.train(verbose=verbose)
    if checkpoint_path:
        trainer.save(checkpoint_path)
    return trainer, history


def train_classifier(
    classifier: Optional[nn.Module] = None,
    cfg: Optional[Dict[str, Any]] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    source_data: Optional[torch.Tensor] = None,
    target_data: Optional[torch.Tensor] = None,
    source_dir: Optional[str] = None,
    target_dir: Optional[str] = None,
    device: Optional[str] = None,
    iterations: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
    autoencoder: Optional[nn.Module] = None,
    verbose: bool = True,
    **overrides,
) -> Tuple[nn.Module, ClassifierTrainer]:
    """High-level classifier fine-tuning entry point.

    Loads a pretrained classifier when ``classifier`` is not given, loads
    source/target images from directories when tensors are not given, runs
    the fine-tuning loop, and freezes ``phi`` for the subsequent ANT stage.
    """
    config = ClassifierTrainConfig.from_dict(cfg, backbone=backbone, task=task)

    # ---- data -------------------------------------------------------
    if source_data is None and source_dir is not None:
        from ..data.datasets import load_image, list_images
        paths = list_images(source_dir, recursive=True)
        if config.source_pool:
            paths = paths[: int(config.source_pool)]
        imgs = [load_image(p, size=256 if backbone != "ldm" else 256) for p in paths]
        if imgs:
            source_data = torch.stack(imgs, dim=0)
    if target_data is None and target_dir is not None:
        from ..data.datasets import load_image, list_images
        paths = list_images(target_dir, recursive=True)[: int(config.target_pool)]
        imgs = [load_image(p, size=256 if backbone != "ldm" else 256) for p in paths]
        if imgs:
            target_data = torch.stack(imgs, dim=0)

    if source_data is None or target_data is None:
        raise ValueError(
            "train_classifier requires source_data/target_data tensors or "
            "source_dir/target_dir paths."
        )
    if backbone == "ldm" and autoencoder is not None:
        from ..data.datasets import to_latents
        source_data = to_latents(source_data, autoencoder)
        target_data = to_latents(target_data, autoencoder)

    # ---- model ------------------------------------------------------
    if classifier is None:
        from ..models.classifier import build_classifier, load_pretrained_classifier
        ckpt = None
        if cfg:
            models = cfg.get("models", {})
            sub = models.get(backbone, {}) if isinstance(models, dict) else {}
            ckpt = sub.get("classifier_ckpt") or cfg.get("classifier_ckpt")
        classifier = load_pretrained_classifier(
            checkpoint=ckpt,
            cfg=cfg,
            backbone=backbone,
            num_classes=config.num_classes,
            target_index=config.target_index,
            device=device or config.device or "cpu",
            replace_head=True,
        )

    trainer, history = fine_tune_classifier(
        classifier,
        source_data,
        target_data,
        config=config,
        device=device,
        iterations=iterations,
        checkpoint_path=checkpoint_path,
        verbose=verbose,
        **overrides,
    )
    return trainer.classifier, trainer


def build_classifier_trainer(
    cfg: Optional[Dict[str, Any]] = None,
    classifier: Optional[nn.Module] = None,
    source_data: Optional[torch.Tensor] = None,
    target_data: Optional[torch.Tensor] = None,
    backbone: str = "ddpm",
    task: Optional[str] = None,
    device: Optional[str] = None,
    **overrides,
) -> ClassifierTrainer:
    """Config-driven factory mirroring ``build_ant_trainer``."""
    config = ClassifierTrainConfig.from_dict(cfg, backbone=backbone, task=task, **overrides)
    if classifier is None:
        from ..models.classifier import load_pretrained_classifier
        classifier = load_pretrained_classifier(
            cfg=cfg, backbone=backbone, num_classes=config.num_classes,
            target_index=config.target_index, device=device or config.device or "cpu",
        )
    return ClassifierTrainer(
        classifier=classifier,
        config=config,
        source_data=source_data,
        target_data=target_data,
        device=device,
    )
