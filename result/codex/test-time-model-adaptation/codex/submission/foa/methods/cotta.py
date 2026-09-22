"""CoTTA (Wang et al., 2022) - continual test-time adaptation with teacher-student
consistency maximisation, augmentation averaging and stochastic weight restoration.

Official implementation: https://github.com/qinenergy/cotta
Appendix B.2: SGD, momentum 0.9, batch size 64, learning rate 0.05, augmentation
threshold ``p_th = 0.1``, up to 32 augmentations for low-confidence samples,
restoration probability 0.01, teacher EMA factor 0.999, trainable parameters = all
parameters of ViT-Base.
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .base import TTAMethod


def build_cotta_augmentation() -> T.Compose:
    """The augmentation set listed in Appendix B.2."""
    return T.Compose(
        [
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            T.RandomHorizontalFlip(p=0.5),
        ]
    )


class CoTTA(TTAMethod):
    name = "CoTTA"

    def __init__(
        self,
        model,
        lr: float = 0.05,
        momentum: float = 0.9,
        augmentation_threshold: float = 0.1,
        num_augmentations: int = 32,
        restore_prob: float = 0.01,
        ema_alpha: float = 0.999,
        device: Optional[torch.device] = None,
        noise_std: float = 0.005,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.last_extra = {}
        self.aug_threshold = augmentation_threshold
        self.num_aug = num_augmentations
        self.restore_prob = restore_prob
        self.ema_alpha = ema_alpha
        self.noise_std = noise_std
        self.lr = lr
        self.momentum = momentum
        self.augment = build_cotta_augmentation()
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)
        self.reset()

    # ----------------------------------------------------------------------------------
    def reset(self) -> None:
        # the student is the (trainable) model, the teacher is a frozen EMA copy
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.teacher = copy.deepcopy(self.model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.model_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.lr, momentum=self.momentum
        )

    # ----------------------------------------------------------------------------------
    def _augment(self, images: torch.Tensor) -> torch.Tensor:
        out = self.augment(images)
        if self.noise_std:
            out = out + torch.randn(out.shape, generator=self._rng) * self.noise_std
        return out.clamp(0, 1) if out.min() >= 0 else out

    @torch.no_grad()
    def _teacher_logits(self, images: torch.Tensor) -> torch.Tensor:
        first = self.teacher(images)
        # augmentation averaging is only needed for the uncertain samples of the batch
        probs = F.softmax(first, dim=1)
        uncertain = probs.max(dim=1)[0] < self.aug_threshold
        if uncertain.any():
            x_un = images[uncertain]
            aug_logits = [first[uncertain]]
            for _ in range(self.num_aug):
                aug_logits.append(self.teacher(self._augment(x_un)))
                conf = F.softmax(torch.stack(aug_logits, 0).mean(0), dim=1).max(dim=1)[0]
                if bool((conf >= self.aug_threshold).all()):
                    break  # early stop once the pseudo-labels are confident enough
            first = first.clone()
            first[uncertain] = torch.stack(aug_logits, 0).mean(0)
        return first

    def step(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        with torch.no_grad():
            teacher_logits = self._teacher_logits(images)
            teacher_probs = F.softmax(teacher_logits, dim=1)
            mask = teacher_probs.max(dim=1)[0] > self.aug_threshold

        # the student is trained on the clean input against the augmentation-averaged
        # teacher pseudo-labels (CoTTA's consistency maximisation)
        self.model.train()
        self.optimizer.zero_grad()
        student_logits = self.model(images)
        if mask.any():
            loss = -(teacher_probs[mask] * F.log_softmax(student_logits[mask], dim=1)).sum(1).mean()
        else:
            loss = student_logits.sum() * 0.0
        loss.backward()
        self.optimizer.step()

        # stochastic restoration towards the source model
        if self.restore_prob > 0:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        continue
                    restore = (
                        torch.rand(param.shape, generator=self._rng) < self.restore_prob
                    ).to(param.device)
                    if restore.any():
                        param.data[restore] = self.model_state[name][restore].to(param.device)

        # EMA update of the teacher
        with torch.no_grad():
            for tp, sp in zip(self.teacher.parameters(), self.model.parameters()):
                tp.mul_(self.ema_alpha).add_(sp.detach(), alpha=1 - self.ema_alpha)
            for tb, sb in zip(self.teacher.buffers(), self.model.buffers()):
                if tb.dtype.is_floating_point:
                    tb.mul_(self.ema_alpha).add_(sb.detach(), alpha=1 - self.ema_alpha)
                else:
                    tb.copy_(sb)

        self.last_extra = {"loss": float(loss.item())}
        return teacher_logits.detach()
