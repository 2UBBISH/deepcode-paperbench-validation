"""Distillation-based knowledge retention: behavioral cloning and kickstarting.

Behavioral cloning (Appendix C.2)::

    L_BC(theta) = E_{s ~ B} [ D_KL( pi_*  ||  pi_theta ) ]        (forward KL)

where ``B`` is a buffer of states gathered from the pre-training distribution.
Minimising the forward KL is equivalent (up to the constant teacher entropy) to
the cross-entropy supervised objective used to pre-train the NetHack policy.

Kickstarting::

    L_KS(theta) = E_{s ~ B_theta} [ D_KL( pi_* || pi_theta ) ]

where the states are sampled by the *online* policy ``pi_theta``.  In NetHack the
loss is scaled by ``0.5`` and decayed with ``0.99998`` every training step.

``kl_divergence`` supports both directions because the main text and the
appendix phrase the KL in opposite orders; the default ``forward`` direction
matches the standard supervised behavioral-cloning objective.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .base import RetentionConfig, RetentionMethod
from .schedules import make_schedule

_EPS = 1e-8


def kl_divergence(
    teacher_logits: Tensor,
    student_logits: Tensor,
    direction: str = "forward",
    eps: float = _EPS,
) -> Tensor:
    """Batched KL divergence between two categorical distributions.

    Parameters
    ----------
    teacher_logits, student_logits: ``(..., num_actions)`` logits.
    direction:
        * ``"forward"`` -- ``D_KL(teacher || student)`` (cross-entropy up to a
          constant); this is behavioral cloning in the usual supervised sense.
        * ``"reverse"`` -- ``D_KL(student || teacher)``.
    """

    log_t = F.log_softmax(teacher_logits, dim=-1)
    log_s = F.log_softmax(student_logits, dim=-1)
    if direction == "forward":
        # D_KL(teacher || student) = E_teacher[log teacher - log student]
        return (log_t.exp() * (log_t - log_s)).sum(dim=-1)
    if direction == "reverse":
        return (log_s.exp() * (log_s - log_t)).sum(dim=-1)
    raise ValueError(f"Unknown KL direction {direction!r}")


def _mean_kl(kl: Tensor) -> Tensor:
    return kl.sum() / max(kl.numel(), 1)


class _DistillationBase(RetentionMethod):
    def __init__(self, config: Optional[RetentionConfig] = None) -> None:
        super().__init__(config)
        self.schedule = make_schedule(self.config.coefficient, self.config.decay)
        self.direction = str(self.config.extra.get("kl_direction", "forward"))
        self.num_actions: Optional[int] = self.config.extra.get("num_actions")

    @property
    def coefficient(self) -> float:  # schedule aware
        return self.schedule.value(self._step)

    def kl_from_logits(self, teacher_logits: Tensor, student_logits: Tensor) -> Tensor:
        return _mean_kl(kl_divergence(teacher_logits, student_logits, self.direction))


class BehavioralCloning(_DistillationBase):
    """Behavioral cloning / teacher distillation on a buffer of pre-training states."""

    def aux_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        """Return ``E_{s~B}[ D_KL(pi_* || pi_theta) ]``.

        ``teacher_logits`` may be omitted when the teacher is queried inside the
        training loop (for instance with the frozen pre-trained network); in
        that case the callers pass the already-computed teacher logits.
        """

        if teacher_logits is None:
            raise ValueError("BehavioralCloning.aux_loss requires teacher_logits.")
        return self.kl_from_logits(teacher_logits, student_logits)


class Kickstarting(_DistillationBase):
    """Kickstarting: match the frozen teacher on states visited by the student."""

    def aux_loss(
        self,
        student_logits: Tensor,
        teacher_logits: Optional[Tensor] = None,
        **_: object,
    ) -> Tensor:
        if teacher_logits is None:
            raise ValueError("Kickstarting.aux_loss requires teacher_logits.")
        return self.kl_from_logits(teacher_logits, student_logits)
