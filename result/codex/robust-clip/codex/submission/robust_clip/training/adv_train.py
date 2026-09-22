"""Adversarial fine-tuning of CLIP vision encoders -- FARE and TeCoA.

All hyper-parameters are the ones of App. B.1/B.3 of the paper:

======================  =====================================================
dataset                 ImageNet, resolution 224x224
epochs                  2
inner attack            10 PGD steps, step size 1/255, eps 2/255 or 4/255
optimizer               AdamW, beta1 = 0.9, beta2 = 0.95
learning rate           cosine decay with linear warmup to the peak of 1e-5
                        attained at 7% of the total steps
weight decay            1e-4
effective batch size    128
trainable parameters    the vision encoder only (the text tower stays frozen)
======================  =====================================================

The FARE loss (Eq. (3)) is evaluated on the class token only, as motivated in
App. B.1: *"using only the class-token in the fine-tuning loss is sufficient to
attain good results with down-stream LVLMs"*.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch

import open_clip

from ..attacks.pgd import pgd_attack
from ..data.registry import imagenet_classnames
from ..models.clip_encoder import _strip_prefixes
from ..utils.misc import AverageMeter, set_seed
from ..utils.precision import eps_to_int
from ..utils.transforms import normalize_images
from .imagenet import imagenet_class_texts
from .losses import embedding_distance, fare_loss, tecoa_loss

LOGGER = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Configuration of a single robust fine-tuning run."""

    method: str = "fare"  # 'fare' or 'tecoa'
    arch: str = "ViT-L-14"
    pretrained: str = "openai"
    eps: str = "2/255"
    pgd_steps: int = 10
    pgd_alpha: str = "1/255"
    epochs: int = 2
    batch_size: int = 128  # effective batch size
    micro_batch_size: int = 32
    lr: float = 1e-5
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_frac: float = 0.07
    lr_schedule: str = "cosine"
    clean_weight: float = 1.0  # lambda of Eq. (3)
    norm: str = "l2_squared"  # 'l1' for the ablation of App. B.4
    resolution: int = 224
    attack_dtype: str = "fp32"  # 'fp16' -> half precision inner attack
    amp: bool = False
    seed: int = 0
    num_workers: int = 8
    output_dir: str = "checkpoints"
    run_name: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_grad_norm: Optional[float] = None
    log_every: int = 20
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    data_dir: Optional[str] = None
    imagenet_name: str = "imagenet-1k"
    hf_token: Optional[str] = None
    device: str = "cuda"
    wandb: bool = False

    def default_name(self) -> str:
        eps_int = eps_to_int(self.eps)
        return f"{self.method.upper()}{eps_int}-{self.arch}-{self.pretrained}".replace("/", "-")


def _lr_lambda(step: int, total_steps: int, warmup_frac: float, schedule: str = "cosine") -> float:
    """Linear warmup to the peak LR (attained at ``warmup_frac`` of the steps) + cosine decay."""
    warmup = max(1, int(warmup_frac * total_steps))
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(1, total_steps - warmup))
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    if schedule == "constant":
        return 1.0
    raise ValueError(f"unknown schedule {schedule!r}")


class AdversarialFineTuner:
    """Runs one FARE / TeCoA fine-tuning of a CLIP vision encoder."""

    def __init__(self, config: FineTuneConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or LOGGER
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        set_seed(config.seed)

        if config.method not in {"fare", "tecoa"}:
            raise ValueError(f"unknown method {config.method!r}")

        # trainable model (vision encoder) and frozen reference model
        self.model, _, _ = open_clip.create_model_and_transforms(config.arch, pretrained=config.pretrained)
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            param.requires_grad_(name.startswith("visual."))

        self.ref_model = None
        if config.method == "fare":
            self.ref_model, _, _ = open_clip.create_model_and_transforms(config.arch, pretrained=config.pretrained)
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad_(False)

        self.tokenizer = open_clip.get_tokenizer(config.arch)
        self.attack_dtype = {"fp32": torch.float32, "fp16": torch.float16}[config.attack_dtype]
        self.eps_int = eps_to_int(config.eps)
        self.mean = tuple(self.model.visual.image_mean)
        self.std = tuple(self.model.visual.image_std)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp and torch.cuda.is_available())

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )
        self.total_steps = None  # set in `fit`
        self.global_step = 0

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize the ``[0, 1]`` images; the attack acts on the raw pixels."""
        return normalize_images(images, self.mean, self.std)

    def _forward_image(self, x_adv: torch.Tensor, model, half: Optional[bool] = None) -> torch.Tensor:
        """Differentiable image embedding at the (possibly half precision) input."""
        if half is None:
            half = self.attack_dtype != torch.float32
        images = self._preprocess(x_adv)
        if half and images.device.type == "cuda":
            # half precision attack: run the forward pass in fp16 (autocast keeps
            # the numerically sensitive operations, e.g. layer norms, in fp32)
            with torch.autocast(device_type="cuda", dtype=self.attack_dtype):
                features = model.encode_image(images)
        else:
            features = model.encode_image(images)
        return features.float() if features.dtype != torch.float32 else features

    def _inner_attack(self, images: torch.Tensor, labels: torch.Tensor):
        """10-step PGD that produces the adversarial point of Eq. (3) / TeCoA."""
        cfg = self.config

        if cfg.method == "fare":

            def forward_fn(x_adv):
                return self._forward_image(x_adv, self.model)

            with torch.no_grad():
                phi_org = self._forward_image(images, self.ref_model)

            def loss_fn(phi_adv, x_adv):
                return embedding_distance(phi_adv, phi_org, norm=cfg.norm)

        else:  # tecoa
            texts = imagenet_class_texts(labels.tolist())
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
            logit_scale = self.model.logit_scale.exp()

            def forward_fn(x_adv):
                return self._forward_image(x_adv, self.model)

            def loss_fn(image_features_adv, x_adv):
                return tecoa_loss(image_features_adv, text_features, logit_scale, reduction="none")

        x_adv, _ = pgd_attack(
            images,
            forward_fn,
            loss_fn,
            eps=cfg.eps,
            alpha=cfg.pgd_alpha,
            steps=cfg.pgd_steps,
            dtype=self.attack_dtype,
            momentum=0.9,
            random_start=True,
            maximize=True,
            return_best=False,
        )
        return x_adv

    def compute_loss(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the training loss and its components for one (micro) batch."""
        x_adv = self._inner_attack(images, labels)
        phi_ft_adv = self._forward_image(x_adv, self.model, half=False)

        if self.config.method == "fare":
            with torch.no_grad():
                phi_org = self._forward_image(images, self.ref_model, half=False)
            phi_ft_clean = self._forward_image(images, self.model, half=False)
            loss = fare_loss(
                phi_ft_adv,
                phi_ft_clean,
                phi_org,
                clean_weight=self.config.clean_weight,
                norm=self.config.norm,
            )
            with torch.no_grad():
                info = {
                    "adv_embedding_loss": embedding_distance(phi_ft_adv, phi_org).mean(),
                    "clean_embedding_loss": embedding_distance(phi_ft_clean, phi_org).mean(),
                }
        else:
            texts = imagenet_class_texts(labels.tolist())
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
            logit_scale = self.model.logit_scale.exp()
            loss = tecoa_loss(phi_ft_adv, text_features, logit_scale)
            with torch.no_grad():
                logits = logit_scale * phi_ft_adv @ text_features.t()
                info = {"adv_accuracy": (logits.argmax(dim=1) == torch.arange(logits.shape[0], device=logits.device)).float().mean()}

        info["loss"] = loss
        return info

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def build_scheduler(self, steps_per_epoch: int) -> None:
        cfg = self.config
        self.total_steps = max(1, steps_per_epoch * cfg.epochs)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _lr_lambda(step, self.total_steps, cfg.warmup_frac, cfg.lr_schedule),
        )

    def fit(self, train_loader, val_loader=None, eval_fn=None) -> Dict[str, float]:
        cfg = self.config
        accum_steps = max(1, cfg.batch_size // max(1, cfg.micro_batch_size))
        steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
        self.build_scheduler(steps_per_epoch)

        self.model.train()
        history = {"loss": []}
        for epoch in range(cfg.epochs):
            meters = {k: AverageMeter(k) for k in ["loss", "adv_embedding_loss", "clean_embedding_loss"]}
            start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=cfg.amp and self.device.type == "cuda"):
                    info = self.compute_loss(images, labels)
                    loss = info["loss"] / accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    if cfg.max_grad_norm is not None:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad], cfg.max_grad_norm
                        )
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                for key in meters:
                    if key in info:
                        meters[key].update(float(info[key]), n=images.shape[0])

                if (step + 1) % cfg.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.logger.info(
                        "epoch %d/%d | step %d/%d | loss %.4f | lr %.3e | %.1fs",
                        epoch + 1,
                        cfg.epochs,
                        step + 1,
                        len(train_loader),
                        meters["loss"].avg,
                        lr,
                        time.time() - start,
                    )

            history["loss"].append(meters["loss"].avg)
            self.logger.info("epoch %d finished: %s", epoch + 1, {k: round(m.avg, 4) for k, m in meters.items()})

            if (epoch + 1) % cfg.save_every_epochs == 0:
                self.save_checkpoint(epoch)

            if val_loader is not None and eval_fn is not None and (epoch + 1) % cfg.eval_every_epochs == 0:
                metrics = eval_fn(self.model, val_loader, self.device)
                self.logger.info("epoch %d evaluation: %s", epoch + 1, metrics)
                history[f"eval_epoch{epoch + 1}"] = metrics
            self.model.train()

        return history

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    @property
    def checkpoint_path(self) -> str:
        name = self.config.run_name or self.config.default_name()
        os.makedirs(self.config.output_dir, exist_ok=True)
        return os.path.join(self.config.output_dir, f"{name}.pt")

    def save_checkpoint(self, epoch: Optional[int] = None) -> str:
        path = self.checkpoint_path
        state = {
            "model": self.model.state_dict(),
            "config": asdict(self.config),
            "epoch": epoch,
            "arch": self.config.arch,
            "pretrained": self.config.pretrained,
        }
        torch.save(state, path)
        with open(os.path.splitext(path)[0] + ".json", "w") as handle:
            json.dump(asdict(self.config), handle, indent=2)
        self.logger.info("saved checkpoint to %s", path)
        return path

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(_strip_prefixes(state), strict=False)

    def state_dict(self):
        return self.model.state_dict()


@torch.no_grad()
def evaluate_imagenet(
    model,
    loader,
    device,
    radii=("2/255", "4/255"),
    n_samples: Optional[int] = None,
    n_iter: int = 100,
    mean=None,
    std=None,
) -> Dict[str, float]:
    """Clean and adversarial (AutoAttack style) ImageNet accuracy of a vision encoder.

    Used for the ablation tables of App. B.3/B.5 (ImageNet clean accuracy and
    robust accuracy at 2/255 and 4/255), with the same APGD attacks as the
    zero-shot evaluation of Sec. 4.3.
    """
    from ..attacks.apgd import APGDAttack
    from ..utils.transforms import normalize_images

    mean = mean or tuple(model.visual.image_mean)
    std = std or tuple(model.visual.image_std)
    model.eval()
    clean_correct = 0
    robust_correct = {str(eps): 0 for eps in radii}
    seen = 0

    def logits(images):
        features = model.encode_image(normalize_images(images, mean, std))
        return model.logit_scale.exp() * features / features.norm(dim=-1, keepdim=True)

    for images, labels in loader:
        if n_samples is not None and seen >= n_samples:
            break
        if n_samples is not None:
            images, labels = images[: n_samples - seen], labels[: n_samples - seen]
        images, labels = images.to(device), labels.to(device)

        # zero-shot style evaluation against the class texts is not needed here:
        # the ablation tables use the ImageNet classifier of CLIP, so we simply
        # compare the image features to the class text features of the dataset.
        text_features = imagenet_classifier_text_features(model, device)
        with torch.no_grad():
            predictions = (logits(images) @ text_features.t()).argmax(dim=1).cpu()
        clean_correct += int((predictions == labels.cpu()).sum())

        for eps in radii:
            attacker = APGDAttack(eps=eps, n_iter=n_iter, loss="ce", dtype=torch.float32)
            x_adv = attacker.attack(images, lambda x: logits(x) @ text_features.t(), labels=labels)
            with torch.no_grad():
                predictions = (logits(x_adv) @ text_features.t()).argmax(dim=1).cpu()
            robust_correct[str(eps)] += int((predictions == labels.cpu()).sum())
        seen += images.shape[0]

    total = max(seen, 1)
    results = {"clean": 100.0 * clean_correct / total}
    for eps in radii:
        results[f"robust_{eps}"] = 100.0 * robust_correct[str(eps)] / total
    return results


_IMAGENET_TEXT_FEATURES = {}


def imagenet_classifier_text_features(model, device, templates=("a photo of a {}.",)):
    """Cached text features of the 1000 ImageNet class prompts."""
    key = (id(model), tuple(templates))
    if key not in _IMAGENET_TEXT_FEATURES:
        classnames = imagenet_classnames()
        prompts = [template.format(name) for name in classnames for template in templates]
        import open_clip

        tokenizer = open_clip.get_tokenizer(_arch_of(model))
        with torch.no_grad():
            features = model.encode_text(tokenizer(prompts).to(device))
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.reshape(len(classnames), len(templates), -1).mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
        _IMAGENET_TEXT_FEATURES[key] = features
    return _IMAGENET_TEXT_FEATURES[key]


def _arch_of(model) -> str:
    """Infer the open_clip architecture name from the model itself."""
    layers = len(model.visual.transformer.resblocks)
    width = model.visual.width
    patch = model.visual.patch_size
    patch = patch[0] if isinstance(patch, (tuple, list)) else patch
    if width == 768 and patch == 32:
        return "ViT-B-32"
    if width == 1024 and patch == 14:
        return "ViT-L-14"
    raise ValueError(f"cannot infer the architecture (width={width}, patch={patch}, layers={layers})")
