"""Efficient self-knowledge distillation (Section 4.4 of the paper).

Objectives (Eq. 7)::

    L          = mu * L_distill + (1 - mu) * L_ft
    L_layer    = sum_{i in T} MSE(Tr(H_s^{phi(i)}), H_t^i)

and, following the addendum shipped with this reproduction task::

    L_distill  = L_pred + 0.9 * L_layer          (GLUE / classification)
    L_distill  = 0.1 * L_pred + 0.9 * L_layer    (SQuAD v2 and CNN/DM)

Key efficiency trick: *no separate teacher checkpoint* is needed.  The teacher
is the very same network evaluated **without the pruning masks**: the frozen
backbone is shared between teacher and student, only the (few) tuning layers are
duplicated.  This removes the "teacher occupies the GPU next to the student"
cost that makes CoFi-style distillation expensive.

Besides that:

* ``T`` is a set of *block-wise randomly sampled* teacher layers (RAIL-KD,
  Haidar et al. 2022).
* ``phi(.)`` is the teacher->student layer-mapping function; it is *re-computed
  every training step* (addendum) and matches each teacher layer to its closest
  non-pruned student layer.
* ``Tr`` is a trainable identity-initialised low-rank transform
  (``Tr(H) = H + B A H`` with ``B = 0``).
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoRA(nn.Module):
    """``Tr``: a tunable layer transformation initialised to the identity.

    ``Tr(H) = H + B A H`` with ``B = 0`` at initialisation, so
    ``Tr`` starts as the identity matrix ``I`` exactly as the paper states.
    """

    def __init__(self, dim: int, rank: int = 8, scaling: float = 1.0) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.scaling = scaling
        self.lora_A = nn.Parameter(torch.zeros(rank, dim))
        self.lora_B = nn.Parameter(torch.zeros(dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.scaling * F.linear(F.linear(h, self.lora_A), self.lora_B)


@contextlib.contextmanager
def teacher_masks(topo, enabled: bool = True):
    """Run the shared backbone *without* pruning masks (teacher forward).

    The buffers are *swapped* (rather than mutated in place) and restored
    afterwards, so no autograd graph that captured the student masks is
    invalidated.
    """
    if not enabled:
        yield
        return
    saved = []
    for lin in topo.linears.values():
        saved.append((lin, lin.mask_in, lin.mask_out))
        lin.mask_in = torch.ones_like(lin.mask_in)
        lin.mask_out = torch.ones_like(lin.mask_out)
    try:
        yield
    finally:
        for lin, mi, mo in saved:
            lin.mask_in = mi
            lin.mask_out = mo


@contextlib.contextmanager
def track_off(topo):
    """Disable salience statistics collection (used for the teacher forward)."""
    saved = [(lin, lin.track) for lin in topo.linears.values()]
    for lin, _ in saved:
        lin.track = False
    try:
        yield
    finally:
        for lin, was in saved:
            lin.track = was


class TeacherCache:
    """Holds the duplicated tuning layers (the teacher) and the ``Tr`` modules."""

    def __init__(self, topo, hidden_size: int, tr_rank: int = 8, device=None) -> None:
        self.topo = topo
        self.hidden_size = hidden_size
        self.tr_rank = tr_rank
        self.device = device
        #: one ``Tr`` per hidden-state stack: RoBERTa has a single stack while an
        #: encoder-decoder LM has one for the encoder and one for the decoder.
        self.trs: List[IdentityLoRA] = []
        self.tr = self.tr_for(0)
        # duplicated tuning weights; refreshed from the student each step
        self.teacher_lora: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def tr_for(self, index: int) -> IdentityLoRA:
        while len(self.trs) <= index:
            tr = IdentityLoRA(self.hidden_size, rank=self.tr_rank)
            if self.device is not None:
                tr = tr.to(self.device)
            self.trs.append(tr)
        return self.trs[index]

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for tr in self.trs:
            params += [tr.lora_A, tr.lora_B]
        return params

    @torch.no_grad()
    def sync_from_student(self) -> None:
        for name, lin in self.topo.linears.items():
            if lin.use_lora:
                self.teacher_lora[name] = (lin.lora_A.detach().clone(), lin.lora_B.detach().clone())

    @contextlib.contextmanager
    def load_teacher_weights(self):
        saved = {}
        for name, lin in self.topo.linears.items():
            if not lin.use_lora or name not in self.teacher_lora:
                continue
            saved[name] = (lin.lora_A.data.clone(), lin.lora_B.data.clone())
            ta, tb = self.teacher_lora[name]
            lin.lora_A.data.copy_(ta)
            lin.lora_B.data.copy_(tb)
        try:
            yield
        finally:
            for name, (a, b) in saved.items():
                lin = self.topo.linears[name]
                lin.lora_A.data.copy_(a)
                lin.lora_B.data.copy_(b)


# --------------------------------------------------------------------------- #
# Layer mapping
# --------------------------------------------------------------------------- #
def sample_teacher_layers(
    n_layers: int, n_sample: int, generator: Optional[torch.Generator] = None
) -> List[int]:
    """Block-wise random sampling of teacher layers (RAIL-KD)."""
    n_sample = max(1, min(n_sample, n_layers))
    perm = torch.randperm(n_layers, generator=generator)
    return sorted(perm[:n_sample].tolist())


def layer_mapping(
    teacher_layers: Sequence[int], student_layers: Sequence[int]
) -> Dict[int, int]:
    """``phi``: match each teacher layer to its closest *non-pruned* student layer.

    Re-computed at every training step (addendum).  For the width-only pruning
    of APT all student layers survive, so the mapping is proportional;
    ``student_layers`` exists to support layer-dropping variants.
    """
    mapping: Dict[int, int] = {}
    if not student_layers:
        return mapping
    n_t = max(teacher_layers) if teacher_layers else 0
    n_s = max(student_layers) + 1
    for t in teacher_layers:
        frac = t / max(1, n_t) if n_t else 0.0
        target = int(round(frac * (n_s - 1)))
        mapping[t] = min(student_layers, key=lambda s: abs(s - target))
    return mapping


def layerwise_distillation_loss(
    tr: nn.Module,
    student_hidden: Sequence[torch.Tensor],
    teacher_hidden: Sequence[torch.Tensor],
    mapping: Dict[int, int],
    only_last_token: bool = False,
) -> torch.Tensor:
    """``L_layer = sum_i MSE(Tr(H_s^{phi(i)}), H_t^i)``."""
    if not mapping:
        return torch.zeros((), device=_device_of(student_hidden))
    total = None
    for t_idx, s_idx in mapping.items():
        if t_idx >= len(teacher_hidden) or s_idx >= len(student_hidden):
            continue
        s = student_hidden[s_idx]
        t = teacher_hidden[t_idx]
        if only_last_token:
            s, t = s[:, -1], t[:, -1]
        term = F.mse_loss(tr(s), t)
        total = term if total is None else total + term
    if total is None:
        return torch.zeros((), device=_device_of(student_hidden))
    return total / len(mapping)


def _device_of(seq: Sequence[torch.Tensor]) -> torch.device:
    for x in seq:
        if torch.is_tensor(x):
            return x.device
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Combined objective
# --------------------------------------------------------------------------- #
@dataclass
class DistillWeights:
    """``L_distill = pred_weight * L_pred + layer_weight * L_layer``."""

    pred_weight: float = 1.0
    layer_weight: float = 0.9

    @staticmethod
    def for_task(task: str) -> "DistillWeights":
        """GLUE uses ``1 / 0.9``; SQuAD and CNN/DM use ``0.1 / 0.9`` (addendum)."""
        if task.lower() in {"squad", "squad_v2", "cnn_dm", "cnndm", "cnn_dailymail"}:
            return DistillWeights(pred_weight=0.1, layer_weight=0.9)
        return DistillWeights(pred_weight=1.0, layer_weight=0.9)


def distill_loss(
    pred_loss: torch.Tensor,
    layer_loss: torch.Tensor,
    weights: DistillWeights,
) -> torch.Tensor:
    return weights.pred_weight * pred_loss + weights.layer_weight * layer_loss


def total_loss(
    distill: torch.Tensor,
    finetune: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """``L = mu L_distill + (1 - mu) L_ft`` (Eq. 7)."""
    return mu * distill + (1.0 - mu) * finetune


__all__ = [
    "IdentityLoRA",
    "TeacherCache",
    "teacher_masks",
    "track_off",
    "sample_teacher_layers",
    "layer_mapping",
    "layerwise_distillation_loss",
    "DistillWeights",
    "distill_loss",
    "total_loss",
]
